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


import pytest
import torch

from tokenspeed.runtime.execution.draft_probabilities import (
    allocate_draft_probabilities,
)


@pytest.mark.parametrize(
    "cap,shard,columns,first",
    [(None, None, 17, 0), (None, (8, 9), 9, 8), (5, None, 5, 0)],
)
def test_proposals_start_invalid_and_bonus_is_not_a_distribution(
    cap, shard, columns, first
):
    state = allocate_draft_probabilities(
        pool_size=4,
        spec_num_tokens=6,
        vocab_size=17,
        reject_threshold=2.0,
        candidate_cap=cap,
        shard=shard,
        device="cpu",
    )
    assert torch.equal(state.probabilities[:, :-1], torch.full((4, 5, columns), 3.0))
    assert torch.count_nonzero(state.probabilities[:, -1]) == 0
    assert state.vocab_start == first
    if cap is not None:
        assert torch.equal(
            state.token_ids, torch.full((4, 6, columns), -1, dtype=torch.int32)
        )
        # Candidate order is explicit and need not be vocabulary order.
        state.token_ids[0, 0] = torch.tensor([16, 0, 9, 2, 1], dtype=torch.int32)
        state.probabilities[0, 0] = torch.tensor([0.1, 0.2, 0.3, 0.1, 0.3])
        restored = torch.zeros(17).scatter(
            0, state.token_ids[0, 0].long(), state.probabilities[0, 0]
        )
        assert restored[9] == 0.3 and restored[0] == 0.2
    else:
        assert state.token_ids is None


@pytest.mark.parametrize(
    "cap,shard", [(0, None), (18, None), (4, (0, 8)), (None, (-1, 8)), (None, (12, 8))]
)
def test_invalid_proposal_layouts_fail_before_allocation(cap, shard):
    with pytest.raises(ValueError):
        allocate_draft_probabilities(
            pool_size=4,
            spec_num_tokens=6,
            vocab_size=17,
            reject_threshold=2.0,
            candidate_cap=cap,
            shard=shard,
            device="cpu",
        )


@pytest.mark.parametrize("threshold", [float("inf"), -1.0, float(2**24), 1.0e100])
def test_invalid_sentinel_cannot_be_mistaken_for_a_real_proposal(threshold):
    with pytest.raises(ValueError):
        allocate_draft_probabilities(
            pool_size=4,
            spec_num_tokens=6,
            vocab_size=17,
            reject_threshold=threshold,
            candidate_cap=5,
            shard=None,
            device="cpu",
        )
