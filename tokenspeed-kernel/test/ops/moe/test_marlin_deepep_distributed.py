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

"""NVLink smoke test for the Marlin MXFP4 DeepEP MoE (SM90).

Each rank holds a distinct token batch and one shard of the experts; the
DeepEP apply must return, per rank, the same result as the bf16 dequant
reference computed over ALL experts for that rank's tokens. Single-node
DeepEP traffic runs the NVLink P2P path in both modes (no IBGDA / RDMA).

The DeepEP buffer is a process-wide singleton pinned to its first mode, so
one torchrun invocation exercises one mode. Normal one-GPU pytest runs skip
this file. Exercise it with:

``torchrun --standalone --nproc-per-node=2 -m pytest -q <this file>``
``TEST_DEEPEP_MODE=low_latency torchrun --standalone --nproc-per-node=2 \
  -m pytest -q <this file>``

The same test supports a four-node torchrun world of 32 ranks. Set
``TEST_K3_MOE_GEOMETRY=1`` for 896 experts/top-16/latent-3584, and optionally
``TEST_MOE_INTERMEDIATE_SIZE=3072`` for full K3 expert GEMM dimensions.
The low-latency invocation also replays CUDA graphs with changed routing.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from kimi3_reference import mxfp4_moe_reference
from utils import make_mxfp4_moe_weights

deep_ep = pytest.importorskip("deep_ep", reason="deep_ep is an optional dependency")


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _deepep_mode() -> str:
    return os.environ.get("TEST_DEEPEP_MODE", "normal")


@pytest.mark.skipif(
    _world_size() not in {2, 4, 8, 32},
    reason="launch with torchrun world size 2, 4, 8, or 32",
)
def test_marlin_deepep_matches_replicated_reference() -> None:
    import tokenspeed_kernel

    world_size = _world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()

    # All ranks share the same weights (same seed); each holds its own tokens.
    generator = torch.Generator(device="cuda").manual_seed(20260822)
    k3_geometry = os.environ.get("TEST_K3_MOE_GEOMETRY", "0") == "1"
    num_experts = 896 if k3_geometry else 4 * world_size
    num_local = num_experts // world_size
    top_k = 16 if k3_geometry else 4
    hidden_size = 3584 if k3_geometry else 2048
    intermediate_size = int(os.environ.get("TEST_MOE_INTERMEDIATE_SIZE", "256"))
    capacity = 16
    # Rank zero is idle while peers send uneven batches; every rank still
    # receives and computes its experts. Keep capacity uniform across ranks.
    source_counts = [0] + [1 + peer % capacity for peer in range(1, world_size)]
    num_tokens = source_counts[rank]
    beta, linear_beta = 4.0, 25.0

    raw = make_mxfp4_moe_weights(num_experts, hidden_size, intermediate_size, generator)
    # Distinct per-rank tokens/routing from a rank-seeded generator.
    rank_gen = torch.Generator(device="cuda").manual_seed(1000 + rank)
    x = (
        torch.randn(
            num_tokens,
            hidden_size,
            generator=rank_gen,
            device="cuda",
        )
        * 0.2
    ).to(torch.bfloat16)
    topk_ids = (
        torch.rand((num_tokens, num_experts), generator=rank_gen, device="cuda")
        .argsort(dim=1)[:, :top_k]
        .to(torch.int32)
        .contiguous()
    )
    topk_weights = torch.rand(
        num_tokens, top_k, generator=rank_gen, device="cuda", dtype=torch.float32
    )
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    if num_tokens and rank % 2:
        topk_ids[-1].fill_(-1)
        topk_weights[-1].zero_()

    expected = mxfp4_moe_reference(
        x,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
        activation_dtype=torch.bfloat16,
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )

    lo, hi = rank * num_local, (rank + 1) * num_local
    module = torch.nn.Module()
    module.w13_weight = torch.nn.Parameter(raw["w13_weight"][lo:hi].clone(), False)
    module.w13_weight_scale = torch.nn.Parameter(raw["w13_scale"][lo:hi].clone(), False)
    module.w2_weight = torch.nn.Parameter(raw["w2_weight"][lo:hi].clone(), False)
    module.w2_weight_scale = torch.nn.Parameter(raw["w2_scale"][lo:hi].clone(), False)
    module.top_k = top_k
    module.num_experts = num_experts
    module.num_local_experts = num_local
    module.ep_rank = rank
    module.ep_size = world_size
    module.activation = "situ"
    module.activation_situ_beta = beta
    module.activation_situ_linear_beta = linear_beta

    mode = _deepep_mode()
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        a2a_backend="deepep",
        ep_size=world_size,
        ispp=intermediate_size,
        internal_activation_dtype="input",
        deepep_group=dist.group.WORLD,
        deepep_mode=mode,
        deepep_low_latency_max_num_tokens_per_gpu=(
            capacity if mode == "low_latency" else None
        ),
        solution="marlin",
    )
    assert plan["apply_kernel_name"] == "marlin_mxfp4_deepep_moe_apply"
    tokenspeed_kernel.moe_process_weights(plan, module)

    actual = tokenspeed_kernel.moe_apply(
        plan,
        x,
        module,
        torch.zeros((num_tokens, num_experts), dtype=torch.float32, device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_tokens_global=sum(source_counts),
        low_latency=(mode == "low_latency"),
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)
    dist.barrier()
    if mode == "low_latency":
        logits = torch.zeros(
            (num_tokens, num_experts), dtype=torch.float32, device="cuda"
        )

        def apply():
            return tokenspeed_kernel.moe_apply(
                plan,
                x,
                module,
                logits,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                num_tokens_global=sum(source_counts),
                low_latency=True,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                apply()
        torch.cuda.current_stream().wait_stream(stream)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = apply()
        for shift in (1, 7):
            topk_ids.copy_(
                torch.where(topk_ids < 0, topk_ids, (topk_ids + shift) % num_experts)
            )
            expected = mxfp4_moe_reference(
                x,
                raw["w13_weight"],
                raw["w13_scale"],
                raw["w2_weight"],
                raw["w2_scale"],
                topk_ids,
                topk_weights,
                activation_dtype=torch.bfloat16,
                situ_beta=beta,
                situ_linear_beta=linear_beta,
            )
            graph.replay()
            torch.testing.assert_close(
                captured.float(), expected.float(), atol=5e-2, rtol=5e-2
            )
        dist.barrier()
