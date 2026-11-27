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


"""Standalone tuning must not invoke batch-owned collective kernels without a batch."""

import torch
from tokenspeed_kernel.ops.moe import moe_plan
from tokenspeed_kernel.registry import KernelRegistry, register_kernel
from tokenspeed_kernel.signature import format_signatures

from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.layers.moe.expert import MoELayer


def test_collective_draft_plan_bootstraps_without_scheduler_capacity():
    registry = KernelRegistry.get()
    snapshot = registry.snapshot()
    try:

        @register_kernel(
            "moe",
            "apply",
            name="test_batch_owned_collective",
            solution="test_batch_owned",
            signatures=format_signatures("x", "dense", {torch.float16}),
            traits={
                "weight_dtype": frozenset({"unquant"}),
                "supports_all_to_all_ep": frozenset({True}),
                "supports_ep": frozenset({True}),
                "a2a_backend": frozenset({"test_batch_owned"}),
                "routing_mode": frozenset({"precomputed_topk"}),
            },
        )
        def apply(
            *,
            plan,
            x,
            w,
            router_logits,
            topk_weights,
            topk_ids,
            num_tokens_global,
            max_num_tokens_per_gpu,
            do_finalize,
            enable_pdl,
            low_latency,
            overlap_fn
        ):
            # This stands in for the unavailable cross-rank transport. Like
            # real collective providers, it refuses an unscheduled invocation.
            if max_num_tokens_per_gpu is None:
                raise ValueError("collective apply requires scheduler capacity")
            return x

        plan = moe_plan(
            "unquant",
            input_dtype=torch.float16,
            activation="swiglu",
            routing_mode="precomputed_topk",
            a2a_backend="test_batch_owned",
            ep_size=2,
            ispp=32,
            hidden=64,
            swiglu_form="standard",
            activation_clamped=False,
            expert_id_repeats=False,
            fast_math=False,
            combine_order="rank",
            features=frozenset(),
            solution="test_batch_owned",
        )
        # Omit checkpoint allocation only. Runtime owners, plan selection and
        # callable dispatch are production classes, with no patched apply path.
        layer = MoELayer.__new__(MoELayer)
        torch.nn.Module.__init__(layer)
        layer.plan = plan
        layer.prefix = "draft.collective"
        layer.hidden_size, layer.intermediate_size = 64, 32
        layer.num_experts, layer.top_k = 8, 3
        layer.input_dtype = torch.float16
        model = torch.nn.Module()
        model.add_module("experts", layer)
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = model
        executor = ModelExecutor.__new__(ModelExecutor)
        executor.device = torch.device("cpu")
        executor.drafter = BaseDrafter(
            spec_num_tokens=6, spec_num_steps=5, draft_model_runner=runner
        )
        executor._autotune_draft_experts(16)
    finally:
        registry.restore(snapshot)
