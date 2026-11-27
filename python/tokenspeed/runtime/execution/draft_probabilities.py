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


"""Allocation of dense, vocabulary-sharded and explicit-candidate draft q."""

import math
import struct
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DraftProbabilities:
    probabilities: torch.Tensor
    token_ids: torch.Tensor | None
    vocab_start: int
    sentinel: float


def allocate_draft_probabilities(
    *,
    pool_size: int,
    spec_num_tokens: int,
    vocab_size: int,
    reject_threshold: float,
    candidate_cap: int | None,
    shard: tuple[int, int] | None,
    device: torch.device | str,
) -> DraftProbabilities:
    """Allocate proposal q with an invalid sentinel and a zero bonus column.

    Dense rows address token IDs implicitly. Sharded rows start at vocab_start.
    Compact rows pair every probability with an explicit, initially invalid ID.
    Compact and vocabulary-sharded layouts cannot be combined.
    """
    if min(pool_size, spec_num_tokens, vocab_size) < 1:
        raise ValueError("Draft probability dimensions must be positive")
    if not math.isfinite(reject_threshold) or reject_threshold < 1:
        raise ValueError("Draft rejection threshold must be finite and at least one")
    if candidate_cap is not None and shard is not None:
        raise ValueError("Compact candidate IDs cannot also be vocabulary sharded")
    if candidate_cap is not None and not 1 <= candidate_cap <= vocab_size:
        raise ValueError("Draft candidate cap must fit the vocabulary")
    first, columns = shard or (0, candidate_cap or vocab_size)
    if first < 0 or columns < 1 or first + columns > vocab_size:
        raise ValueError("Draft vocabulary shard falls outside the vocabulary")
    try:
        sentinel = struct.unpack("f", struct.pack("f", float(reject_threshold) + 1.0))[
            0
        ]
    except OverflowError as exc:
        raise ValueError("Draft rejection sentinel must fit FP32") from exc
    if not math.isfinite(sentinel) or sentinel <= reject_threshold:
        raise ValueError("Draft rejection sentinel must exceed its threshold in FP32")
    probabilities = torch.full(
        (pool_size, spec_num_tokens, columns),
        sentinel,
        dtype=torch.float32,
        device=device,
    )
    probabilities[:, -1] = 0
    ids = (
        None
        if candidate_cap is None
        else torch.full(probabilities.shape, -1, dtype=torch.int32, device=device)
    )
    return DraftProbabilities(probabilities, ids, first, sentinel)
