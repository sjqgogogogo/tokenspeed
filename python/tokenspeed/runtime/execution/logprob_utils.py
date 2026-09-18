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

"""Opt-in, forward-local raw input/Top-K collection; no sampler mutation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.types import LogprobRequestConfig

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput


class RawLogitsSnapshot:
    """Copy pre-sampling decode logits into externally owned, stable storage.

    Args:
        buffer: A persistent float32 [captured_batch, vocabulary] GPU view.
            The runner allocates its backing storage before graph capture.
    """

    def __init__(self, buffer: torch.Tensor) -> None:
        if buffer.ndim != 2 or buffer.dtype != torch.float32:
            raise ValueError("Raw graph logits require a float32 matrix")
        self.buffer = buffer

    def capture(self, logits_output: LogitsProcessorOutput) -> None:
        """Record a copy after audit but before any sampling transformation."""
        logits = logits_output.next_token_logits
        if (
            logits_output.logits_layout_plan is not None
            or tuple(logits.shape) != tuple(self.buffer.shape)
            or logits.device != self.buffer.device
            or not logits.is_floating_point()
        ):
            raise RuntimeError("Raw graph logits must be full-vocabulary decode rows")
        self.buffer.copy_(logits)


class TopLogprobCapture:
    """Own one forward's diagnostic tensors until the existing D2H fence.

    Input rows are source positions in each complete prefill chunk, including
    its last row. The output processor aligns source t with input token t+1,
    discarding the prompt's last source row (the first output prediction).
    Output Top-K is raw model log-softmax, before sampling transforms.
    """

    def __init__(
        self,
        configs: tuple[LogprobRequestConfig, ...],
        num_extends: int,
        input_lengths: tuple[int, ...],
        vocab_size: int,
    ) -> None:
        if not configs or len(configs) != len(input_lengths):
            raise ValueError("Logprob controls must match the forward request count")
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("Top-K vocabulary size must be a positive integer")
        if type(num_extends) is not int or not 0 <= num_extends <= len(configs):
            raise ValueError("Top-K num_extends is outside the request count")
        if any(type(length) is not int or length <= 0 for length in input_lengths):
            raise ValueError("Top-K diagnostics require positive input lengths")
        if any(length != 1 for length in input_lengths[num_extends:]):
            raise ValueError("Top-K diagnostics support one decode token per request")
        for config in configs:
            if (
                type(config.top_logprobs_num) is not int
                or not 0 <= config.top_logprobs_num <= vocab_size
                or type(config.logprob_start_len) is not int
                or config.logprob_start_len < -1
            ):
                raise ValueError("Invalid Top-K count or input logprob start position")
        self.configs = configs
        self.vocab_size = vocab_size
        self.seq_lens = list(input_lengths)
        self.topk_nums = [
            c.top_logprobs_num if c.return_logprob else 0 for c in configs
        ]
        self.pruned_lens = [
            (
                length
                if i < num_extends and c.return_logprob and c.logprob_start_len >= 0
                else 0
            )
            for i, (c, length) in enumerate(zip(configs, input_lengths))
        ]
        if not any(self.topk_nums) and not any(self.pruned_lens):
            raise ValueError(
                "Logprob capture requires requested input rows or output Top-K"
            )
        for config, length in zip(configs, self.pruned_lens):
            if len(config.input_token_ids) != length or any(
                type(token) is not int or not 0 <= token < vocab_size
                for token in config.input_token_ids
            ):
                raise ValueError(
                    "Input logprob target ids must match source rows and vocabulary"
                )
        self.input_token_ids = tuple(
            token for c in configs for token in c.input_token_ids
        )
        self.start_lens = [n - p for n, p in zip(self.seq_lens, self.pruned_lens)]
        self.input_token_values = None
        self.input_values = None
        self.input_indices = None
        self.output_values = None
        self.output_indices = None
        self._captured = False

    def capture(self, logits_output: LogitsProcessorOutput) -> None:
        """Collect next-token distributions without changing sampled logits."""
        logits = logits_output.next_token_logits
        self._validate_next_logits(logits, logits_output.logits_layout_plan)
        self._capture_prompt(logits_output, logits)
        self._capture_output(logits)

    def _validate_next_logits(self, logits: torch.Tensor, layout) -> None:
        if self._captured:
            raise RuntimeError(
                "Top-K capture is forward-local and may execute only once"
            )
        if logits.ndim != 2 or logits.shape != (len(self.configs), self.vocab_size):
            raise RuntimeError(
                "Top-K diagnostics require one full-vocabulary row per request"
            )
        if layout is not None:
            raise RuntimeError("Top-K diagnostics do not support batch-sharded logits")

    def _capture_prompt(
        self, logits_output: LogitsProcessorOutput, logits: torch.Tensor
    ) -> None:
        if any(self.pruned_lens):
            token_values = logits_output.input_token_logprobs
            if (
                not isinstance(token_values, torch.Tensor)
                or tuple(token_values.shape) != (sum(self.pruned_lens),)
                or not token_values.is_floating_point()
                or token_values.device != logits.device
            ):
                raise RuntimeError(
                    "Prompt token logprobs must match requested source rows"
                )
            self.input_token_values = []
            offset = 0
            for length in self.pruned_lens:
                self.input_token_values.append(
                    token_values[offset : offset + length] if length else None
                )
                offset += length
        if any(length and k for length, k in zip(self.pruned_lens, self.topk_nums)):
            input_values = logits_output.input_top_logprobs_val
            input_indices = logits_output.input_top_logprobs_idx
            if input_values is None or input_indices is None:
                raise RuntimeError("Model did not return requested prompt Top-K rows")
            if len(input_values) != len(self.configs) or len(input_indices) != len(
                self.configs
            ):
                raise RuntimeError(
                    "Prompt Top-K payload must contain one entry per request"
                )
            for values, indices, length, k in zip(
                input_values, input_indices, self.pruned_lens, self.topk_nums
            ):
                if length == 0 or k == 0:
                    if values is not None or indices is not None:
                        raise RuntimeError("Disabled prompt Top-K entries must be None")
                    continue
                if (
                    not isinstance(values, torch.Tensor)
                    or not isinstance(indices, torch.Tensor)
                    or tuple(values.shape) != (length, k)
                    or tuple(indices.shape) != (length, k)
                    or not values.is_floating_point()
                    or indices.dtype != torch.int64
                    or values.device != logits.device
                    or indices.device != logits.device
                ):
                    raise RuntimeError(
                        "Prompt Top-K payload has invalid shape, dtype or device"
                    )
            self.input_values = input_values
            self.input_indices = input_indices

    def capture_replayed_logits(self, logits: torch.Tensor) -> None:
        """Collect live decode rows after actual replay on the same stream."""
        self._validate_next_logits(logits, None)
        if any(self.pruned_lens):
            raise RuntimeError("Decode graph cannot supply prompt logprob rows")
        self._capture_output(logits)

    def _capture_output(self, logits: torch.Tensor) -> None:
        if any(self.topk_nums):
            values, indices = torch.topk(
                torch.log_softmax(logits.float(), dim=-1), max(self.topk_nums), dim=-1
            )
            self.output_values = values.unsqueeze(1)
            self.output_indices = indices.unsqueeze(1)
        self._captured = True

    def copy_to_cpu(self) -> dict:
        """Enqueue owned D2H tensors, consumed only after result.copy_event."""
        if not self._captured:
            raise RuntimeError("Requested Top-K forward did not execute its capture")
        return {
            "input_token_logprobs": (
                None
                if self.input_token_values is None
                else [
                    None if v is None else v.to("cpu", non_blocking=True)
                    for v in self.input_token_values
                ]
            ),
            "input_top_logprobs_val": (
                None
                if self.input_values is None
                else [
                    None if v is None else v.to("cpu", non_blocking=True)
                    for v in self.input_values
                ]
            ),
            "input_top_logprobs_idx": (
                None
                if self.input_indices is None
                else [
                    None if v is None else v.to("cpu", non_blocking=True)
                    for v in self.input_indices
                ]
            ),
            "output_top_logprobs_val": (
                None
                if self.output_values is None
                else self.output_values.to("cpu", non_blocking=True)
            ),
            "output_top_logprobs_idx": (
                None
                if self.output_indices is None
                else self.output_indices.to("cpu", non_blocking=True)
            ),
        }


def split_prompt_topk(
    all_logprobs: torch.Tensor,
    topk_nums: list[int],
    pruned_lens: list[int],
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    """Split GPU Top-K by request without synchronizing into Python lists."""
    if (
        all_logprobs.ndim != 2
        or not topk_nums
        or len(topk_nums) != len(pruned_lens)
        or any(
            type(k) is not int or not 0 <= k <= all_logprobs.shape[1] for k in topk_nums
        )
        or any(type(length) is not int or length < 0 for length in pruned_lens)
        or sum(pruned_lens) != all_logprobs.shape[0]
    ):
        raise ValueError("Prompt Top-K metadata does not match logits rows")
    top = all_logprobs.topk(max(topk_nums), dim=-1)
    values, indices = [], []
    offset = 0
    for k, length in zip(topk_nums, pruned_lens):
        values.append(
            top.values[offset : offset + length, :k] if length and k else None
        )
        indices.append(
            top.indices[offset : offset + length, :k] if length and k else None
        )
        offset += length
    return values, indices
