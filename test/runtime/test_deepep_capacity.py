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

"""CPU checks for the token-sliced DeepEP send-capacity contract."""

import pytest

from tokenspeed.runtime.moe.capacity import resolve_deepep_token_capacity


@pytest.mark.parametrize(
    "max_num_seqs,dp,tp,width,expected",
    [(128, 4, 8, 8, 32), (80, 4, 8, 8, 20), (12, 4, 8, 7, 3), (4, 4, 8, 1, 1)],
)
def test_capacity_covers_every_disjoint_tp_slice(
    max_num_seqs: int, dp: int, tp: int, width: int, expected: int
) -> None:
    capacity = resolve_deepep_token_capacity(
        requested_capacity=None,
        max_num_seqs=max_num_seqs,
        attn_dp_size=dp,
        attn_tp_size=tp,
        tokens_per_request=width,
        token_sliced=True,
        low_latency_enabled=True,
        max_extend_tokens=0,
    )
    assert capacity == expected
    for requests in range(max_num_seqs // dp + 1):
        tokens = requests * width
        slices = [tokens * (rank + 1) // tp - tokens * rank // tp for rank in range(tp)]
        assert sum(slices) == tokens
        assert max(slices) <= capacity


def test_small_explicit_capacity_fails_before_a_collective() -> None:
    with pytest.raises(ValueError, match="need at least 32 rows"):
        resolve_deepep_token_capacity(
            requested_capacity=31,
            max_num_seqs=128,
            attn_dp_size=4,
            attn_tp_size=8,
            tokens_per_request=8,
            token_sliced=True,
            low_latency_enabled=True,
            max_extend_tokens=0,
        )


def test_pinned_low_latency_also_covers_recovery_prefill() -> None:
    arguments = dict(
        max_num_seqs=128,
        attn_dp_size=4,
        attn_tp_size=8,
        tokens_per_request=8,
        token_sliced=True,
        low_latency_enabled=True,
        max_extend_tokens=8192,
    )
    assert resolve_deepep_token_capacity(requested_capacity=None, **arguments) == 1056
    with pytest.raises(ValueError, match="use --deepep-mode auto"):
        resolve_deepep_token_capacity(requested_capacity=32, **arguments)


def test_mixed_recovery_budgets_decode_rows_beside_prefill() -> None:
    assert (
        resolve_deepep_token_capacity(
            requested_capacity=None,
            max_num_seqs=128,
            attn_dp_size=4,
            attn_tp_size=8,
            tokens_per_request=8,
            token_sliced=True,
            low_latency_enabled=True,
            max_extend_tokens=128,
        )
        == 48
    )


def test_small_ep_groups_respect_deepep_receive_alignment() -> None:
    arguments = dict(
        max_num_seqs=1,
        attn_dp_size=1,
        attn_tp_size=2,
        tokens_per_request=1,
        token_sliced=True,
        low_latency_enabled=True,
        max_extend_tokens=0,
    )
    assert resolve_deepep_token_capacity(requested_capacity=None, **arguments) == 2
    with pytest.raises(ValueError, match="multiple of 2"):
        resolve_deepep_token_capacity(requested_capacity=3, **arguments)


@pytest.mark.parametrize("requested", [32, 64])
def test_explicit_capacity_is_preserved(requested: int) -> None:
    assert (
        resolve_deepep_token_capacity(
            requested_capacity=requested,
            max_num_seqs=128,
            attn_dp_size=4,
            attn_tp_size=8,
            tokens_per_request=8,
            token_sliced=True,
            low_latency_enabled=True,
            max_extend_tokens=0,
        )
        == requested
    )


@pytest.mark.parametrize("token_sliced,low_latency", [(False, True), (True, False)])
def test_other_plans_keep_their_existing_capacity(
    token_sliced: bool, low_latency: bool
) -> None:
    for requested, expected in [(None, 256), (64, 64)]:
        assert (
            resolve_deepep_token_capacity(
                requested_capacity=requested,
                max_num_seqs=1024,
                attn_dp_size=1,
                attn_tp_size=1,
                tokens_per_request=8,
                token_sliced=token_sliced,
                low_latency_enabled=low_latency,
                max_extend_tokens=0,
            )
            == expected
        )


@pytest.mark.parametrize("requested", [0, -1])
def test_nonpositive_capacity_is_rejected(requested: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        resolve_deepep_token_capacity(
            requested_capacity=requested,
            max_num_seqs=128,
            attn_dp_size=4,
            attn_tp_size=8,
            tokens_per_request=8,
            token_sliced=True,
            low_latency_enabled=True,
            max_extend_tokens=0,
        )
