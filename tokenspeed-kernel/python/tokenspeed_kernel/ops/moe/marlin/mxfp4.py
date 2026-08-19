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

"""Marlin W4A16 MXFP4 MoE apply for SM90 (Hopper).

Weight-only 4-bit experts: MXFP4 (E2M1 packed, E8M0 group-32 scales) weights
are dequantized inside the Marlin grouped GEMM; activations stay bf16, so no
FP4 tensor cores are required. This is the Hopper path for Kimi-K3, whose
routed experts use the SiTU gated activation (applied as a fused Triton
epilogue between the two GEMMs). Routing is precomputed upstream (the K3 gate
the fused kernels cannot reproduce).

The kernel and its weight repack use the vendored Marlin MoE; the grouped GEMM
lives in ``thirdparty/cuda/csrc/marlin_moe`` and is pre-compiled.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.activation.triton import situ_and_mul, situ_and_mul_masked
from tokenspeed_kernel.ops.communication.deep_ep import DeepEPDispatcher, DeepEPMode
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.marlin import gptq_marlin_repack
from tokenspeed_kernel.thirdparty.cuda.marlin_moe import (
    marlin_make_workspace,
    marlin_permute_scales,
    moe_align_block_size,
    moe_wna16_marlin_gemm,
    mxfp4_marlin_process_scales,
)

MXFP4_BLOCK = 32
MARLIN_M_BLOCK = 8


@triton.jit
def _masked_marlin_schedule_kernel(
    masked_m_ptr,
    block_offsets_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    capacity: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Build Marlin's padded route schedule directly from DeepEP counts."""
    expert = tl.program_id(0)
    valid_rows = tl.load(masked_m_ptr + expert)
    block_offset = tl.load(block_offsets_ptr + expert)
    num_blocks = (valid_rows + BLOCK_M - 1) // BLOCK_M
    cols = tl.arange(0, BLOCK_M)
    block = 0
    while block < num_blocks:
        row = block * BLOCK_M + cols
        output_block = block_offset + block
        route = expert * capacity + row
        route = tl.where(row < valid_rows, route, tl.num_programs(0) * capacity)
        tl.store(sorted_ids_ptr + output_block * BLOCK_M + cols, route)
        tl.store(expert_ids_ptr + output_block, expert)
        block += 1


def _masked_marlin_schedule(
    masked_m: torch.Tensor,
    capacity: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a packed, count-driven schedule without materializing route ids."""
    num_experts = masked_m.shape[0]
    block_counts = torch.div(
        masked_m + block_m - 1,
        block_m,
        rounding_mode="floor",
    )
    block_ends = torch.cumsum(block_counts, dim=0, dtype=torch.int32)
    block_offsets = block_ends - block_counts
    max_blocks = num_experts * ((capacity + block_m - 1) // block_m)
    sorted_ids = torch.empty(
        max_blocks * block_m, dtype=torch.int32, device=masked_m.device
    )
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=masked_m.device)
    num_tokens_post_padded = (
        block_ends[-1:].mul(block_m)
        if num_experts
        else torch.zeros(1, dtype=torch.int32, device=masked_m.device)
    )
    if num_experts:
        _masked_marlin_schedule_kernel[(num_experts,)](
            masked_m,
            block_offsets,
            sorted_ids,
            expert_ids,
            capacity=capacity,
            BLOCK_M=block_m,
            num_warps=1,
        )
    return sorted_ids, expert_ids, num_tokens_post_padded


def _block_size_m(num_tokens: int, top_k: int, num_experts: int) -> int:
    """Pick the token block granularity Marlin schedules over.

    Grow the block until each expert holds under ~0.9 blocks of routed tokens,
    keeping small batches on tight blocks and large batches on wide ones.
    """
    block = 8
    for candidate in (8, 16, 32, 48, 64):
        block = candidate
        if num_tokens * top_k / num_experts / candidate < 0.9:
            break
    return block


def marlin_mxfp4_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Repack loader-format MXFP4 experts into the Marlin layout, once.

    Input (from the mxfp4 weight loader): ``w13_weight``/``w2_weight`` packed
    E2M1 as uint8 ``[E, N, K//2]`` and ``w13_weight_scale``/``w2_weight_scale``
    raw E8M0 as uint8 ``[E, N, K//32]``. Output: Marlin-repacked int32 weights
    and permuted float8_e8m0fnu scales, written back onto the module. K3's
    shapes are already aligned (hidden 7168%256, ispp 3072%128), so no padding.
    """
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    if any(not hasattr(w, name) for name in names):
        raise ValueError("MXFP4 MoE weights are incomplete for Marlin repack")
    if getattr(w, "_marlin_repacked", False):
        return

    activation = plan.get("activation") or getattr(w, "activation", "silu")
    if activation not in {"silu", "situ", "swiglu"}:
        raise ValueError(f"Marlin MXFP4 MoE does not support activation {activation!r}")

    w13 = w.w13_weight.data
    w2 = w.w2_weight.data
    w13_scale = w.w13_weight_scale.data
    w2_scale = w.w2_weight_scale.data
    num_experts = w13.shape[0]
    # w13 is [E, 2*ispp, hidden//2] packed; w2 is [E, hidden, ispp//2] packed.
    two_ispp = w13.shape[1]
    hidden = w13.shape[2] * 2
    ispp = w2.shape[2] * 2
    if two_ispp != 2 * ispp:
        raise ValueError(
            f"w13/w2 intermediate mismatch: w13 {two_ispp} vs 2*ispp {2 * ispp}"
        )
    if hidden % 256 != 0 or ispp % MXFP4_BLOCK != 0:
        raise ValueError(
            f"Marlin MXFP4 needs hidden%256==0 and ispp%32==0, got "
            f"hidden={hidden}, ispp={ispp}"
        )

    device = w13.device
    perm = torch.empty(0, dtype=torch.int, device=device)

    def _repack(weight: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
        # gptq_marlin_repack wants int32 [size_k/pack, size_n]; the packed
        # uint8 [E, size_n, size_k/2] view transposes to that per expert.
        out = []
        for e in range(num_experts):
            qw = weight[e].view(torch.int32).T.contiguous()
            out.append(
                gptq_marlin_repack(qw, perm, size_k=size_k, size_n=size_n, num_bits=4)
            )
        return torch.stack(out)

    def _permute(scale: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
        out = []
        for e in range(num_experts):
            s = scale[e].T.contiguous()
            permuted = marlin_permute_scales(
                s, size_k=size_k, size_n=size_n, group_size=MXFP4_BLOCK
            )
            out.append(mxfp4_marlin_process_scales(permuted))
        return torch.stack(out)

    w13_marlin = _repack(w13, size_n=2 * ispp, size_k=hidden)
    w2_marlin = _repack(w2, size_n=hidden, size_k=ispp)
    w13_scale_marlin = _permute(w13_scale, size_n=2 * ispp, size_k=hidden)
    w2_scale_marlin = _permute(w2_scale, size_n=hidden, size_k=ispp)

    w.w13_weight = torch.nn.Parameter(w13_marlin, requires_grad=False)
    w.w2_weight = torch.nn.Parameter(w2_marlin, requires_grad=False)
    w.w13_weight_scale = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)
    w.w2_weight_scale = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)
    w._marlin_hidden_size = hidden
    w._marlin_ispp = ispp
    w._marlin_repacked = True


def _marlin_mxfp4_local_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    local_topk_ids: torch.Tensor | None,
    topk_weights: torch.Tensor,
    *,
    is_ep: bool = True,
    masked_m: torch.Tensor | None = None,
    expected_m: int | None = None,
) -> torch.Tensor:
    """Run local Marlin experts for already-local route ids.

    Normal DeepEP supplies compact ``[tokens, top_k]`` routing metadata. Its
    low-latency leg supplies an expert-major ``[E, capacity, H]`` buffer and
    device-side valid-row counts; in that case ``local_topk_ids`` has one route
    per capacity row and SiTU only touches rows below ``masked_m``.
    """
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Marlin MXFP4 MoE requires bf16 activations, got {x.dtype}")
    activation = plan.get("activation") or getattr(w, "activation", "silu")
    hidden = int(getattr(w, "_marlin_hidden_size", x.shape[-1]))
    ispp = int(getattr(w, "_marlin_ispp", w.w2_weight.shape[1] * 16))
    num_local_experts = int(getattr(w, "num_local_experts", w.w13_weight.shape[0]))
    flat_x = x.reshape(-1, hidden)
    if masked_m is not None:
        if x.ndim != 3:
            raise ValueError("masked Marlin input must be [experts, capacity, hidden]")
        num_tokens, top_k = flat_x.shape[0], 1
    else:
        if local_topk_ids is None or local_topk_ids.ndim != 2:
            raise ValueError("Marlin local top-k ids must be a 2D tensor")
        num_tokens, top_k = local_topk_ids.shape
    if flat_x.shape[0] != num_tokens:
        raise ValueError(
            f"Marlin routes have {num_tokens} rows but activations have "
            f"{flat_x.shape[0]}"
        )

    topk_weights = topk_weights.to(torch.float32)
    if masked_m is not None:
        block_m = MARLIN_M_BLOCK
        sorted_ids, expert_ids, num_tokens_post_padded = _masked_marlin_schedule(
            masked_m, x.shape[1], block_m
        )
    else:
        local_topk_ids = local_topk_ids.to(torch.int32)
        block_m = _block_size_m(num_tokens, top_k, num_local_experts)
        sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            local_topk_ids, block_m, num_local_experts
        )
    workspace = marlin_make_workspace(flat_x.device)

    # Low latency deliberately leaves skipped capacity rows uninitialized.
    # Marlin schedules only ids >= 0 and DeepEP combine consumes only masked_m.
    gateup_out = (
        torch.empty(
            (num_tokens * top_k, 2 * ispp), dtype=flat_x.dtype, device=flat_x.device
        )
        if masked_m is not None
        else None
    )
    gateup = moe_wna16_marlin_gemm(
        flat_x,
        gateup_out,
        w.w13_weight,
        w.w13_weight_scale,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_m,
        top_k=top_k,
        mul_topk_weights=False,
        is_ep=is_ep,
        size_m=num_tokens,
        size_n=2 * ispp,
        size_k=hidden,
    )

    beta = float(getattr(w, "activation_situ_beta", 1.0))
    linear_beta = getattr(w, "activation_situ_linear_beta", None)
    linear_beta = None if linear_beta is None else float(linear_beta)
    if masked_m is not None:
        num_experts, capacity, _ = x.shape
        gateup = gateup.view(num_experts, capacity, 2 * ispp)
        if activation == "situ":
            down_in = situ_and_mul_masked(
                gateup,
                masked_m,
                beta=beta,
                linear_beta=linear_beta,
                expected_m=expected_m,
            ).view(-1, ispp)
        else:
            raise ValueError(
                f"masked Marlin DeepEP only supports SiTU, got {activation!r}"
            )
    elif activation == "situ":
        down_in = situ_and_mul(gateup, beta=beta, linear_beta=linear_beta)
    else:
        from tokenspeed_kernel.ops.activation.triton import silu_and_mul

        down_in = silu_and_mul(gateup)

    # Normal combine has no routing weights of its own, so W2 applies them and
    # invalid top-k slots must be zero. Low-latency combine applies the original
    # weights and ignores padded rows, allowing an uninitialized capacity tail.
    low_latency = masked_m is not None
    expert_out = moe_wna16_marlin_gemm(
        down_in,
        (
            torch.empty(
                (num_tokens * top_k, hidden),
                dtype=flat_x.dtype,
                device=flat_x.device,
            )
            if low_latency
            else torch.zeros(
                (num_tokens * top_k, hidden),
                dtype=flat_x.dtype,
                device=flat_x.device,
            )
        ),
        w.w2_weight,
        w.w2_weight_scale,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_m,
        top_k=1,
        mul_topk_weights=not low_latency,
        is_ep=is_ep,
        size_m=num_tokens * top_k,
        size_n=hidden,
        size_k=ispp,
    )
    if low_latency:
        return expert_out.view_as(x)
    return expert_out.view(num_tokens, top_k, hidden).sum(dim=1)


def _get_deepep_dispatcher(
    plan: dict,
    w: torch.nn.Module,
    x: torch.Tensor,
) -> DeepEPDispatcher:
    dispatcher = plan.get("_deepep_dispatcher")
    if dispatcher is not None:
        return dispatcher
    group = plan.get("deepep_group")
    if group is None:
        raise ValueError("DeepEP Marlin plan is missing deepep_group")
    deepep_mode = DeepEPMode(plan.get("deepep_mode") or DeepEPMode.auto.value)
    capacity = plan.get("deepep_low_latency_max_num_tokens_per_gpu")
    if deepep_mode.enable_low_latency() and not capacity:
        raise ValueError(
            f"DeepEP plan with mode {deepep_mode.value} is missing "
            "deepep_low_latency_max_num_tokens_per_gpu"
        )
    config = SimpleNamespace(
        top_k=getattr(w, "top_k"),
        num_experts=getattr(w, "num_experts"),
        low_latency_max_num_tokens_per_gpu=capacity,
        hidden_size=x.shape[-1],
        world_size=getattr(w, "ep_size", group.size()),
        group=group,
        params_dtype=torch.bfloat16,
    )
    dispatcher = DeepEPDispatcher(
        config,
        deepep_mode=deepep_mode,
        async_finish=True,
        return_recv_hook=True,
        use_fp8=False,
        normal_expert_alignment=MARLIN_M_BLOCK,
    )
    plan["_deepep_dispatcher"] = dispatcher
    return dispatcher


def _deepep_expected_m(
    x: torch.Tensor,
    w: torch.nn.Module,
    top_k: int,
    num_tokens_global: int | None,
    recv_m: int,
) -> int:
    num_experts = int(getattr(w, "num_experts", 0) or 0)
    if num_experts <= 0:
        return recv_m
    ep_size = int(getattr(w, "ep_size", 1) or 1)
    total_tokens = int(num_tokens_global or x.shape[0] * ep_size)
    expected = (total_tokens * top_k + num_experts - 1) // num_experts
    return max(1, min(expected, recv_m))


@register_kernel(
    "moe",
    "apply",
    name="marlin_mxfp4_precomputed_moe_apply",
    solution="marlin",
    weight_preprocessor=marlin_mxfp4_moe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"silu", "situ", "swiglu"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({MXFP4_BLOCK}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def marlin_mxfp4_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply a Marlin W4A16 MXFP4 MoE with precomputed routing.

    Args:
        plan: MoE plan; ``activation`` selects the GEMM1 epilogue.
        x: bf16 hidden states ``[tokens, hidden]``.
        w: Module holding Marlin-repacked ``w13_weight``/``w2_weight`` (int32)
            and permuted ``w13_weight_scale``/``w2_weight_scale``
            (float8_e8m0fnu), plus ``num_local_experts``/``ep_rank`` for EP.
        router_logits: Unused; routing is precomputed.
        topk_weights: Route weights ``[tokens, top_k]``.
        topk_ids: Global expert ids ``[tokens, top_k]``. In EP they are remapped
            to local ids; non-local ids become -1 and contribute zero.
        num_tokens_global: Unused; distributed EP dispatch is not owned here.
        max_num_tokens_per_gpu: Unused capacity hint.
        do_finalize: Must be true (no deferred finalize).
        enable_pdl: Unused launch hint.

    Returns:
        Finalized hidden states ``[tokens, hidden]`` in the dtype of ``x``.
    """
    del router_logits, num_tokens_global, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("Marlin MXFP4 MoE cannot defer finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Marlin MXFP4 MoE requires precomputed top-k")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Marlin MXFP4 MoE requires bf16 activations, got {x.dtype}")

    num_local_experts = int(getattr(w, "num_local_experts", w.w13_weight.shape[0]))
    ep_size = int(getattr(w, "ep_size", 1))
    ep_rank = int(getattr(w, "ep_rank", 0))
    global_num_experts = num_local_experts * ep_size

    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)
    is_ep = ep_size > 1
    if is_ep:
        # Global -> local id remap: [expert_start, +num_local) -> [0, num_local),
        # everything else -> -1 (the align kernel parks these in the extra lane
        # and the EP GEMM skips those blocks).
        expert_start = ep_rank * num_local_experts
        mapping = torch.full(
            (global_num_experts,), -1, dtype=torch.int32, device=topk_ids.device
        )
        mapping[expert_start : expert_start + num_local_experts] = torch.arange(
            num_local_experts, dtype=torch.int32, device=topk_ids.device
        )
        local_topk_ids = mapping[topk_ids.long()]
    else:
        local_topk_ids = topk_ids
    return _marlin_mxfp4_local_apply(
        plan,
        x,
        w,
        local_topk_ids,
        topk_weights,
        is_ep=is_ep,
    )


@register_kernel(
    "moe",
    "apply",
    name="marlin_mxfp4_deepep_precomputed_moe_apply",
    solution="marlin",
    weight_preprocessor=marlin_mxfp4_moe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"mxfp4"}),
        "activation": frozenset({"situ"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({True}),
        "deepep_modes": frozenset({"normal", "low_latency"}),
        "ispp_alignment": frozenset({MXFP4_BLOCK}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PERFORMANT,
)
def marlin_mxfp4_deepep_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
    low_latency: bool | None = None,
    overlap_fn: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Run K3's precomputed TopK through DeepEP and local Marlin experts."""
    del router_logits, max_num_tokens_per_gpu, enable_pdl
    if not do_finalize:
        raise ValueError("DeepEP Marlin MXFP4 MoE cannot defer finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("DeepEP Marlin MXFP4 MoE requires precomputed top-k")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Marlin MXFP4 MoE requires bf16 activations, got {x.dtype}")

    topk_ids = topk_ids.to(dtype=torch.int64, memory_format=torch.contiguous_format)
    topk_weights = topk_weights.to(
        dtype=torch.float32, memory_format=torch.contiguous_format
    )
    dispatcher = _get_deepep_dispatcher(plan, w, x)
    resolved = dispatcher.deepep_mode.resolve(low_latency)
    dispatcher.dispatch_a(
        x,
        topk_ids,
        topk_weights,
        low_latency=resolved == DeepEPMode.low_latency,
    )
    if overlap_fn is not None:
        overlap_fn()
    (
        recv_x,
        recv_topk_ids,
        recv_topk_weights,
        _,
        _,
        _,
        masked_m,
    ) = dispatcher.dispatch_b()

    if resolved == DeepEPMode.normal:
        if recv_x.shape[0] == 0:
            combine_in = recv_x.new_empty((0, x.shape[1]))
        else:
            combine_in = _marlin_mxfp4_local_apply(
                plan,
                recv_x,
                w,
                recv_topk_ids,
                recv_topk_weights,
            )
        dispatcher.combine_a(
            combine_in,
            recv_topk_ids,
            recv_topk_weights,
            low_latency=False,
        )
        return dispatcher.combine_b()

    num_local_experts, capacity, _ = recv_x.shape
    # Low-latency DeepEP applies the original route weights during combine; the
    # Marlin calls only need a correctly typed placeholder with top_k=1.
    local_weights = torch.empty(
        (num_local_experts * capacity, 1), dtype=torch.float32, device=x.device
    )
    expected_m = _deepep_expected_m(
        x, w, topk_ids.shape[1], num_tokens_global, capacity
    )
    combine_in = _marlin_mxfp4_local_apply(
        plan,
        recv_x,
        w,
        None,
        local_weights,
        is_ep=False,
        masked_m=masked_m,
        expected_m=expected_m,
    )
    dispatcher.combine_a(
        combine_in,
        topk_ids,
        topk_weights,
        low_latency=True,
    )
    return dispatcher.combine_b()
