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

"""Compact sampling separates prefill draws from decode proposal verification."""

import pytest

from tokenspeed.runtime.utils.server_args import prepare_server_args

BASE = [
    "--model",
    "x",
    "--device",
    "cuda",
    "--sampling-backend",
    "compact",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-num-steps",
    "5",
    "--speculative-num-draft-tokens",
    "6",
    "--sampling-stream",
    "batch",
    "--sampling-force-topk",
    "1024",
    "--spec-reject-draft-prob-penalty",
    "0.00005",
    "--spec-draft-confidence-temperature",
    "1.0",
]


def test_prefill_compact_needs_no_draft_distribution():
    args = prepare_server_args([*BASE, "--disaggregation-mode", "prefill"])
    assert args.sampling_backend == "compact"
    assert not args.enable_speculative_sampling


@pytest.mark.parametrize("role", ["null", "decode"])
def test_compact_verification_requires_recorded_proposals(role):
    with pytest.raises(ValueError, match="probability-aware verification"):
        prepare_server_args([*BASE, "--disaggregation-mode", role])
    args = prepare_server_args(
        [*BASE, "--disaggregation-mode", role, "--enable-speculative-sampling"]
    )
    assert args.enable_speculative_sampling


def test_prefill_rejects_unused_draft_probability_allocation():
    with pytest.raises(ValueError, match="no effect on the prefill role"):
        prepare_server_args(
            [*BASE, "--disaggregation-mode", "prefill", "--enable-speculative-sampling"]
        )
