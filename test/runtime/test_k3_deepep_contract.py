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

"""CPU execution checks for K3's token-home and Marlin bridge contracts.

Only the tested production functions are loaded from their AST, so these
checks do not import CUDA/Triton packages on developer machines. They execute
real tensor math and the real model orchestration with simulated collectives;
GPU numerical and graph tests live under tokenspeed-kernel/test/ops/moe.
"""

from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "python/tokenspeed/runtime/models/kimi_k3.py"
COMM = ROOT / "python/tokenspeed/runtime/models/kimi_k3_comm.py"
BRIDGE = (
    ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/marlin/deepep_mxfp4.py"
)
LAYOUT = (
    ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/marlin/deepep_layout.py"
)


def load_function(path, name, namespace, owner):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if owner is not None:
        nodes = next(
            node
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == owner
        ).body
    fn = next(
        node
        for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def scatter_count(tokens, tp):
    base, remainder = divmod(tokens, tp)
    return [base + (rank < remainder) for rank in range(tp)]


@pytest.mark.parametrize("ep_rank", [0, 1, 7, 31])
def test_normal_dispatch_restores_global_expert_ids(ep_rank):
    recv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ids = torch.tensor([[0, 27], [-1, 3]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.0, 1.0]])
    calls = []
    w = SimpleNamespace(ep_rank=ep_rank, num_local_experts=28)

    class Dispatcher:
        def dispatch_a(self, x, topk_ids, topk_weights, *, low_latency):
            assert not low_latency

        def dispatch_b(self):
            return recv, ids, weights, None, None, None, None

        def combine_a(self, output, topk_ids, topk_weights, *, low_latency):
            assert not low_latency
            torch.testing.assert_close(topk_ids, ids)
            self.output = output

        def combine_b(self):
            return self.output

    def apply(plan, x, module, logits, *, topk_weights, topk_ids, enable_pdl):
        expected_ids = torch.where(ids < 0, ids, ids + ep_rank * 28)
        torch.testing.assert_close(topk_ids, expected_ids)
        coefficients = torch.where(topk_ids < 0, 0, topk_ids + 1)
        return x * (coefficients * topk_weights).sum(1, keepdim=True)

    namespace = dict(torch=torch, marlin_mxfp4_precomputed_moe_apply=apply)
    normal = load_function(BRIDGE, "_apply_normal", namespace, None)
    actual = normal(
        Dispatcher(), {}, recv, w, weights, ids, False, lambda: calls.append("shared")
    )
    expected_ids = torch.where(ids < 0, ids, ids + ep_rank * 28)
    expected = recv * (torch.where(ids < 0, 0, expected_ids + 1) * weights).sum(
        1, keepdim=True
    )
    torch.testing.assert_close(actual, expected)
    assert calls == ["shared"]


def test_weight_preprocessing_reserves_deepep_before_forward():
    calls = []
    w = SimpleNamespace(w13_weight=torch.empty(1))

    def repack(plan, weights):
        calls.append("repack")
        weights._marlin_hidden_size = 3584

    def dispatcher(plan, weights, x):
        assert x.shape == (0, 3584)
        return SimpleNamespace(prepare=lambda: calls.append("reserve"))

    fn = load_function(
        BRIDGE,
        "marlin_mxfp4_deepep_moe_weights",
        dict(torch=torch, marlin_mxfp4_moe_weights=repack, _get_dispatcher=dispatcher),
        None,
    )
    fn({}, w)
    assert calls == ["repack", "reserve"]


@pytest.mark.parametrize("experts", [28, 112])
@pytest.mark.parametrize("global_tokens", [0, 1, 7, 64, 1024])
def test_compact_capacity_covers_skewed_routes(experts, global_tokens):
    capacity = load_function(LAYOUT, "compact_row_capacity", {}, None)
    recv_m = max(global_tokens, 1)
    rows = capacity(global_tokens, 16, experts, recv_m, 16)
    # A maximally skewed batch can select the same 16 experts for every token.
    counts = [global_tokens if expert < 16 else 0 for expert in range(experts)]
    required = sum(((count + 15) // 16) * 16 for count in counts)
    assert rows >= required
    assert rows % 16 == 0
    assert rows <= global_tokens * 16 + experts * 15 + 16


def test_compact_capacity_does_not_scale_intermediates_with_wire_capacity():
    capacity = load_function(LAYOUT, "compact_row_capacity", {}, None)
    # E896/EP32, C256 has 229376 padded receive rows. A 64-token bucket
    # schedules at most 1024 actual routes on any rank, including worst skew.
    rows = capacity(64, 16, 28, 256 * 32, 16)
    assert rows == 1456
    assert rows < (896 * 256) // 100


@pytest.mark.parametrize("tokens", [0, 1, 5, 9, 17])
@pytest.mark.parametrize("dp_rank", [0, 3])
def test_model_slices_once_and_tail_restores_only_its_tp_group(tokens, dp_rank):
    tp_size, width, latent = 8, 16, 4
    hidden = (
        torch.arange(tokens * width, dtype=torch.float32).reshape(tokens, width) / 31
    )
    prefix = torch.ones_like(hidden) * 7
    projection = (
        torch.arange(latent * width, dtype=torch.float32).reshape(latent, width) / 127
    )
    slots = torch.ones(tokens, dtype=torch.int64)
    if tokens > 1:
        slots[-1] = 0
    routed = hidden[:, :latent] * 2 * (slots > 0)[:, None]
    norm = lambda x: x / torch.sqrt(x.square().mean(1, keepdim=True) + 1e-6)
    up = norm(routed) @ projection
    expected = prefix + hidden + up
    counts = scatter_count(tokens, tp_size)
    sent = []

    for rank in range(tp_size):
        start, count = sum(counts[:rank]), counts[rank]
        group = tuple(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        tp = SimpleNamespace(tp_size=tp_size, tp_rank=rank, tp_group=group, dp_size=4)
        shared = hidden * ((rank + 1) / 36)
        columns = slice(rank * 2, (rank + 1) * 2)

        def gather(value, *, group, scattered_num_tokens):
            assert group == tp.tp_group
            assert scattered_num_tokens == counts
            torch.testing.assert_close(value, routed[start : start + count])
            return routed.clone()

        def reduce(value, group):
            assert group == tp.tp_group
            partial = shared.clone()
            partial[:, columns] += up[:, columns]
            torch.testing.assert_close(value, partial)
            return hidden + up

        globals_ = dict(
            torch=torch,
            CommManager=SimpleNamespace(_scatter_count=scatter_count),
            token_all_gather=gather,
            all_reduce=reduce,
        )
        tail = load_function(COMM, "_tail_combined_latent", globals_, "K3MoeTailComm")
        tail_self = SimpleNamespace(
            mapping=SimpleNamespace(attn=tp),
            routed_norm=norm,
            _shard_up_projection=True,
            up_proj=SimpleNamespace(
                project_shard=lambda x: (x @ projection)[:, columns],
                shard_slice=(rank * 2, 2),
            ),
        )
        plan = SimpleNamespace(
            symm_outputs=None,
            lane=None,
            defer_finalize=False,
            split_shared_rs=False,
            routed_in_fork=False,
        )
        comm = SimpleNamespace(
            plan=lambda *args, **kwargs: plan,
            run=lambda plan, routed, shared, prefix, nt, hs, prepared: tail(
                tail_self, routed, shared, prefix, nt
            ),
        )

        class TopK:
            def __call__(self, x, logits, *, output_format):
                return SimpleNamespace(
                    topk_ids=torch.zeros((x.shape[0], 1), dtype=torch.int64),
                    topk_weights=torch.ones((x.shape[0], 1)),
                )

            def empty_topk_output(self, device, *, hidden_states, router_logits):
                return SimpleNamespace(
                    topk_ids=torch.empty((0, 1), dtype=torch.int64),
                    topk_weights=torch.empty((0, 1)),
                )

        class Fork:
            _active = False

            def scope(self, *, enable, overlap):
                assert not enable
                return nullcontext(self)

            def branch(self):
                return nullcontext()

        def experts(
            x, topk, global_tokens, max_tokens, *, do_finalize, low_latency, overlap_fn
        ):
            assert do_finalize and low_latency
            assert global_tokens == tokens * 4
            sent.append((start, count))
            torch.testing.assert_close(x, hidden[start : start + count, :latent])
            live = slots[start : start + count] > 0
            torch.testing.assert_close(topk.topk_ids[:, 0], torch.where(live, 0, -1))
            torch.testing.assert_close(topk.topk_weights[:, 0], live.float())
            overlap_fn()
            return x * 2 * live[:, None]

        format_ = SimpleNamespace(is_standard=lambda: True)
        model = SimpleNamespace(
            _use_deepep=True,
            _gather_dp_tokens_for_moe=False,
            native_latent_moe=None,
            mapping=SimpleNamespace(attn=tp),
            comm=comm,
            experts=SimpleNamespace(),
            routed_hidden=latent,
            num_experts=896,
            _routing_output_format=lambda ctx: format_,
            gate=lambda x: torch.zeros((x.shape[0], 1)),
            topk=TopK(),
            stream_fork=Fork(),
            _topk_ready=None,
            shared_experts=lambda x, down_out: shared.clone(),
            routed_expert_down_proj=lambda x: (x[:, :latent], None),
            _routed_experts=experts,
        )
        mode = SimpleNamespace(is_decode=lambda: True, is_decode_or_idle=lambda: True)
        ctx = SimpleNamespace(
            forward_mode=mode,
            all_decode_or_idle=True,
            attn_backend=SimpleNamespace(decode_window_locations=lambda: slots),
        )
        globals_.update(
            get_is_cuda_graph_phase=lambda: True,
            get_is_capture_mode=lambda: True,
            use_deepep_low_latency=lambda ctx, dp: ctx.all_decode_or_idle,
        )
        forward = load_function(MODEL, "forward", globals_, "KimiLinearMoE")
        actual = forward(model, hidden, prefix, tokens * 4, max(counts), ctx)
        torch.testing.assert_close(actual, expected)
    assert len(sent) == 8  # empty source ranks also enter EP
    assert sum(count for _, count in sent) == tokens
    assert [
        row for start, count in sent for row in range(start, start + count)
    ] == list(range(tokens))


def test_low_latency_bridge_preserves_current_combine_api_and_compact_bound():
    recv = torch.randn(4, 32, 8)
    counts = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    origin = torch.randn(2, 8)
    ids = torch.zeros((2, 16), dtype=torch.int64)
    weights = torch.ones((2, 16)) / 16
    calls = []

    class Dispatcher:
        def dispatch_a(self, x, topk_ids, topk_weights, *, low_latency):
            assert low_latency
            calls.append("dispatch")

        def dispatch_b(self):
            return recv, None, None, None, None, None, counts

        def combine_a(self, x, topk_ids, topk_weights, *, low_latency):
            # Deliberately accepts the current Legacy API: no x_ori or
            # identity-expert argument exists for K3's independent shared MLP.
            assert low_latency
            assert x.data_ptr() == recv.data_ptr()
            assert topk_ids is ids and topk_weights is weights
            calls.append("combine")

        def combine_b(self):
            return origin

    def apply(plan, received, module, actual_counts, global_bound, top_k):
        assert received is recv and actual_counts is counts
        assert global_bound == 12 and top_k == 16
        calls.append("marlin")
        return received

    fn = load_function(
        BRIDGE,
        "_apply_low_latency",
        dict(torch=torch, marlin_mxfp4_masked_moe_apply=apply),
        None,
    )
    actual = fn(
        Dispatcher(),
        {},
        origin,
        object(),
        weights,
        ids,
        12,
        lambda: calls.append("shared"),
    )
    assert actual is origin
    assert calls == ["dispatch", "shared", "marlin", "combine"]
