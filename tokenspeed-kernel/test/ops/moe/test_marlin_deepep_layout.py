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

"""GPU numerical and replay checks for compact DeepEP-to-Marlin layouts."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_compact_marlin_masked_rows_and_graph_replay():
    from kimi3_reference import mxfp4_moe_reference
    from tokenspeed_kernel.ops.moe.marlin.mxfp4 import (
        marlin_mxfp4_masked_moe_apply,
        marlin_mxfp4_moe_weights,
    )
    from tokenspeed_kernel.platform import current_platform
    from utils import make_mxfp4_moe_weights

    platform = current_platform()
    if not platform.is_nvidia or platform.arch_version.major < 9:
        pytest.skip("Marlin requires SM90+")
    experts, recv_m, hidden, intermediate = 4, 32, 256, 128
    generator = torch.Generator(device="cuda").manual_seed(901)
    raw = make_mxfp4_moe_weights(
        experts, hidden, intermediate, generator, device="cuda"
    )
    weights = torch.nn.Module()
    weights.w13_weight = torch.nn.Parameter(raw["w13_weight"].clone(), False)
    weights.w13_weight_scale = torch.nn.Parameter(raw["w13_scale"].clone(), False)
    weights.w2_weight = torch.nn.Parameter(raw["w2_weight"].clone(), False)
    weights.w2_weight_scale = torch.nn.Parameter(raw["w2_scale"].clone(), False)
    weights.activation = "situ"
    weights.activation_situ_beta = 4.0
    weights.activation_situ_linear_beta = 25.0
    weights.num_local_experts = experts
    plan = {"activation": "situ"}
    marlin_mxfp4_moe_weights(plan, weights)
    source = (
        torch.randn((experts, recv_m, hidden), generator=generator, device="cuda") * 0.2
    ).bfloat16()
    routing = (
        torch.arange(experts, device="cuda")
        .view(-1, 1)
        .expand(-1, recv_m)
        .reshape(-1, 1)
    )
    expected = mxfp4_moe_reference(
        source.flatten(0, 1),
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        routing,
        torch.ones((experts * recv_m, 1), device="cuda"),
        activation_dtype=torch.bfloat16,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    ).view_as(source)
    recv = source.clone()
    counts = torch.zeros(experts, dtype=torch.int32, device="cuda")

    def apply():
        return marlin_mxfp4_masked_moe_apply(plan, recv, weights, counts, 16, 2)

    # Capture with all counts zero, then replay real and skewed layouts. The
    # count-driven loop must not freeze the capture's empty expert schedule.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            apply()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = apply()
    assert actual.data_ptr() == recv.data_ptr()

    for live_counts in ([2, 0, 3, 1], [7, 0, 0, 0], [0, 0, 0, 0], [1, 1, 6, 2]):
        recv.copy_(source)
        counts.copy_(torch.tensor(live_counts, dtype=torch.int32, device="cuda"))
        graph.replay()
        for expert, count in enumerate(live_counts):
            torch.testing.assert_close(
                actual[expert, :count].float(),
                expected[expert, :count].float(),
                atol=5e-2,
                rtol=5e-2,
            )
            # Neither SiTU nor unpack may write padding or stale rows from a
            # previous replay when the expert's live count decreases.
            torch.testing.assert_close(actual[expert, count:], source[expert, count:])
