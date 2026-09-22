# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Committed CPU logprobs carried by the PD completion control message.

This payload never occupies registered GPU KV storage. It is published before
Prefill releases completion, restored before Decode exposes the bootstrap token,
and owned by one bootstrap room until the receiver consumes it.
"""

from __future__ import annotations

import hashlib
import math

import msgspec


class PrefillLogprobs(msgspec.Struct, frozen=True):
    version: int
    prompt_length: int
    prompt_digest: bytes
    start: int
    topk: int
    input_values: tuple[float | None, ...]
    input_top_values: tuple[tuple[float, ...] | None, ...]
    input_top_ids: tuple[tuple[int, ...] | None, ...]
    token: int
    value: float
    top_values: tuple[float, ...]
    top_ids: tuple[int, ...]

    def validate(self) -> None:
        if self.version != 1 or self.prompt_length < 1 or len(self.prompt_digest) != 32:
            raise ValueError("Invalid PD logprob payload version or prompt identity")
        if not -1 <= self.start < self.prompt_length or self.topk < 0 or self.token < 0:
            raise ValueError("Invalid PD logprob request geometry")
        expected = self.prompt_length - self.start if self.start >= 0 else 0
        if len(self.input_values) != expected:
            raise ValueError("Incomplete PD input logprobs")
        for i, value in enumerate(self.input_values):
            if self.start == 0 and i == 0:
                if value is not None:
                    raise ValueError("First prompt logprob must be null")
            else:
                _score(value)
        expected_top = expected if self.topk else 0
        if (
            len(self.input_top_values) != expected_top
            or len(self.input_top_ids) != expected_top
        ):
            raise ValueError("Incomplete PD input top logprobs")
        for i, (values, ids) in enumerate(
            zip(self.input_top_values, self.input_top_ids, strict=True)
        ):
            if self.start == 0 and i == 0:
                if values is not None or ids is not None:
                    raise ValueError("First prompt top logprobs must be null")
            else:
                _top(values, ids, self.topk)
        _score(self.value)
        _top(self.top_values, self.top_ids, self.topk)


def _score(value) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value > 1e-6:
        raise ValueError("Invalid PD log probability")


def _top(values, ids, width: int) -> None:
    if values is None or ids is None or len(values) != width or len(ids) != width:
        raise ValueError("PD top-logprob width mismatch")
    if any(type(token) is not int or token < 0 for token in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("Invalid PD top-logprob token IDs")
    for value in values:
        _score(value)
    if any(a < b for a, b in zip(values, values[1:])):
        raise ValueError("PD top logprobs must be ordered")


def _digest(ids) -> bytes:
    return hashlib.sha256(msgspec.msgpack.encode(tuple(ids))).digest()


def snapshot_prefill_logprobs(state) -> bytes | None:
    """Freeze a completed Prefill request's scores without retaining request state."""
    if (
        not state.return_logprob
        or not state.prefill_finished
        or not state.output_ids
        or state.to_abort
    ):
        return None
    if (
        len(state.output_ids) != 1
        or state.output_token_logprobs_val is None
        or len(state.output_token_logprobs_val) != 1
    ):
        raise ValueError("PD Prefill must produce one scored bootstrap token")
    topk = state.top_logprobs_num
    want_input = state.logprob_start_len >= 0
    if topk and (
        len(state.output_top_logprobs_val) != 1
        or len(state.output_top_logprobs_idx) != 1
    ):
        raise ValueError("Missing PD bootstrap top logprobs")
    payload = PrefillLogprobs(
        version=1,
        prompt_length=state.input_length,
        prompt_digest=_digest(state.prompt_input_ids),
        start=state.logprob_start_len,
        topk=topk,
        input_values=tuple(state.input_token_logprobs_val) if want_input else (),
        input_top_values=(
            tuple(
                None if row is None else tuple(row)
                for row in state.input_top_logprobs_val
            )
            if want_input and topk
            else ()
        ),
        input_top_ids=(
            tuple(
                None if row is None else tuple(row)
                for row in state.input_top_logprobs_idx
            )
            if want_input and topk
            else ()
        ),
        token=state.output_ids[0],
        value=state.output_token_logprobs_val[0],
        top_values=tuple(state.output_top_logprobs_val[0]) if topk else (),
        top_ids=tuple(state.output_top_logprobs_idx[0]) if topk else (),
    )
    payload.validate()
    return msgspec.msgpack.encode(payload)


def restore_prefill_logprobs(state, bootstrap_token: int, wire: bytes | None) -> None:
    """Restore input scores and output position zero before Decode can emit tokens."""
    if not state.return_logprob:
        return
    if not wire:
        raise ValueError(
            "PD peer did not supply requested prefill/bootstrap logprobs; update both peers"
        )
    try:
        payload = msgspec.msgpack.decode(wire, type=PrefillLogprobs)
    except msgspec.DecodeError as exc:
        raise ValueError(f"Malformed PD logprob payload: {exc}") from exc
    payload.validate()
    if (
        payload.prompt_length != state.input_length
        or payload.prompt_digest != _digest(state.prompt_input_ids)
        or payload.start != state.logprob_start_len
        or payload.topk != state.top_logprobs_num
        or payload.token != bootstrap_token
    ):
        raise ValueError(
            "PD logprobs disagree with the Decode request or bootstrap token"
        )
    # Complete validation precedes mutation, so a rejected payload cannot leave
    # partially installed scores alongside a token that Decode might publish.
    state.input_token_logprobs_val = list(payload.input_values)
    state.input_token_logprobs_idx = (
        list(state.prompt_input_ids[payload.start :]) if payload.start >= 0 else []
    )
    state.input_top_logprobs_val = [
        None if row is None else list(row) for row in payload.input_top_values
    ]
    state.input_top_logprobs_idx = [
        None if row is None else list(row) for row in payload.input_top_ids
    ]
    state.output_token_logprobs_val = [payload.value]
    state.output_token_logprobs_idx = [bootstrap_token]
    state.output_top_logprobs_val = [list(payload.top_values)] if payload.topk else []
    state.output_top_logprobs_idx = [list(payload.top_ids)] if payload.topk else []
