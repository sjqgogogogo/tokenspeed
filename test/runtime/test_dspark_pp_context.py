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

"""Numerical and lifetime contracts for pipeline DSpark context production."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tokenspeed.runtime.execution.context_producer import PrefillTargetContextProducer
from tokenspeed.runtime.models.context_projection import (
    context_tap_owner_layer,
    project_context_tap,
)


class ReferenceNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.6, 1.3, width))
        self.eps = eps

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        values = rows.float()
        return (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
            * self.weight.float()
        ).to(rows.dtype)


class ContextWeights:
    def __init__(
        self, weights, norms, final_norm, stage: int, owned: tuple[int, ...]
    ) -> None:
        self.weights = weights
        self.norms = norms
        self.final_norm = final_norm
        self.hidden_size = weights[0].shape[0]
        self.mapping = SimpleNamespace(
            is_first_pp_rank=stage == 0, is_last_pp_rank=stage == 3
        )
        self.owned = owned
        self.writes = []

    def project_target_tap(
        self, capture_idx: int, hidden: torch.Tensor
    ) -> torch.Tensor:
        assert capture_idx in self.owned
        return project_context_tap(
            hidden, self.weights[capture_idx], self.norms[capture_idx]
        )

    def finalize_target_projection(self, projected: torch.Tensor) -> torch.Tensor:
        return self.final_norm(projected.to(self.weights[0].dtype))

    def write_context_kv(self, hidden, positions, cache_locs, pool) -> None:
        self.writes.append(
            (hidden.clone(), positions.clone(), cache_locs.clone(), pool)
        )


@pytest.mark.parametrize("normalize_taps", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 11])
def test_pipeline_sum_matches_concatenated_projection(
    normalize_taps: bool, tokens: int
) -> None:
    generator = torch.Generator().manual_seed(731)
    taps = [torch.randn(tokens, 8, generator=generator) for _ in range(5)]
    weights = [torch.randn(6, 8, generator=generator) for _ in range(5)]
    norms = [ReferenceNorm(8, 1e-5) if normalize_taps else None for _ in taps]
    final_norm = ReferenceNorm(6, 1e-5)
    # The published prefix taps [2, 23, 47, 71, 89] leave stage 1 with no tap.
    ownership = ((0, 1), (), (2,), (3, 4))
    accumulator = None
    pool = object()
    for stage, owned in enumerate(ownership):
        model = ContextWeights(weights, norms, final_norm, stage, owned)
        producer = PrefillTargetContextProducer(model, pool if stage == 3 else None)
        accumulator = producer.begin_stage(taps[0], accumulator)
        for capture_idx in owned:
            producer.add_capture(accumulator, capture_idx, taps[capture_idx])
    normalized = [
        norm(tap) if norm is not None else tap
        for norm, tap in zip(norms, taps, strict=True)
    ]
    expected = final_norm(
        F.linear(torch.cat(normalized, dim=-1), torch.cat(weights, dim=-1))
    )
    positions = torch.arange(tokens, dtype=torch.int64) + 128
    locations = torch.arange(tokens, dtype=torch.int64) + 512
    producer.write_context(accumulator, positions, locations)
    actual, actual_positions, actual_locations, actual_pool = model.writes[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_positions, positions)
    torch.testing.assert_close(actual_locations, locations)
    assert actual_pool is pool


@pytest.mark.parametrize("normalize_taps", [False, True])
def test_bfloat16_projection_split_has_bounded_roundoff(normalize_taps: bool) -> None:
    generator = torch.Generator().manual_seed(733)
    taps = [torch.randn(37, 64, generator=generator).bfloat16() for _ in range(5)]
    weights = [torch.randn(48, 64, generator=generator).bfloat16() for _ in range(5)]
    norms = [
        ReferenceNorm(64, 1e-5).bfloat16() if normalize_taps else None for _ in taps
    ]
    final_norm = ReferenceNorm(48, 1e-5).bfloat16()
    accumulator = None
    for stage, owned in enumerate(((0, 1), (), (2,), (3, 4))):
        model = ContextWeights(weights, norms, final_norm, stage, owned)
        producer = PrefillTargetContextProducer(model, object() if stage == 3 else None)
        accumulator = producer.begin_stage(taps[0], accumulator)
        for capture_idx in owned:
            producer.add_capture(accumulator, capture_idx, taps[capture_idx])
    normalized = [
        norm(tap) if norm is not None else tap
        for norm, tap in zip(norms, taps, strict=True)
    ]
    expected = final_norm(
        F.linear(torch.cat(normalized, dim=-1), torch.cat(weights, dim=-1))
    )
    actual = model.finalize_target_projection(accumulator)
    relative_error = torch.linalg.vector_norm(
        actual.float() - expected.float()
    ) / torch.linalg.vector_norm(expected.float())
    assert relative_error < 0.007
    assert (actual.float() - expected.float()).abs().max() < 0.04


def test_queued_chunks_keep_independent_accumulators() -> None:
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    model = ContextWeights([weight], [None], nn.Identity(), 0, (0,))
    producer = PrefillTargetContextProducer(model, None)
    first_hidden = torch.ones(2, 4)
    second_hidden = torch.full((2, 4), 3.0)
    first = producer.begin_stage(first_hidden, None)
    producer.add_capture(first, 0, first_hidden)
    saved = first.clone()
    second = producer.begin_stage(second_hidden, None)
    producer.add_capture(second, 0, second_hidden)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, saved)
    torch.testing.assert_close(second, saved * 3)


@pytest.mark.parametrize("stream, expected", [("prefix", 23), ("attn_res", 24)])
def test_boundary_tap_belongs_to_stage_with_required_weights(
    stream: str, expected: int
) -> None:
    assert context_tap_owner_layer(23, 93, stream) == expected
    assert context_tap_owner_layer(92, 93, stream) == 92


def test_downstream_requires_full_precision_matching_accumulator() -> None:
    model = ContextWeights([torch.ones(3, 4)], [None], nn.Identity(), 1, ())
    producer = PrefillTargetContextProducer(model, None)
    hidden = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="missing"):
        producer.begin_stage(hidden, None)
    with pytest.raises(ValueError, match="float32"):
        producer.begin_stage(hidden, torch.zeros(2, 3, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="float32"):
        producer.begin_stage(hidden, torch.zeros(1, 3))


def test_only_last_stage_may_have_cache_view() -> None:
    model = ContextWeights([torch.ones(3, 4)], [None], nn.Identity(), 0, (0,))
    with pytest.raises(ValueError, match="final pipeline stage"):
        PrefillTargetContextProducer(model, object())
    model.mapping.is_last_pp_rank = True
    with pytest.raises(ValueError, match="final pipeline stage"):
        PrefillTargetContextProducer(model, None)


def _pipeline_forward_methods():
    # Exercise the production pipeline layer loop without importing CUDA
    # backends on CPU hosts. Only model kernels are supplied as test doubles;
    # tap placement, PP handoff and writer ordering execute the real methods.
    path = (
        Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/models/kimi_k3.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KimiLinearModel"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("forward", "_dspark_capture_stream")
    ]

    def apply_attn_res(*args, **kwargs):
        return args[0] * args[2]

    namespace = {
        "torch": torch,
        "PPStageState": SimpleNamespace,
        "ceil_div": lambda size, divisor: (size + divisor - 1) // divisor,
        "_apply_attn_res": apply_attn_res,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return {name: namespace[name] for name in ("forward", "_dspark_capture_stream")}


@pytest.mark.parametrize("stream", ["prefix", "attn_res"])
def test_production_pp_loop_captures_boundary_taps_once_on_the_owner(
    stream: str,
) -> None:
    model_type = type("PipelineModel", (), _pipeline_forward_methods())
    target_layers = 12
    tap_layers = [2, 3, 5, 8, 11]
    weights = [torch.full((3, 4), float(idx + 1)) for idx in range(len(tap_layers))]
    final_norm = ReferenceNorm(3, 1e-5)
    pool = object()
    state = None
    positions = torch.arange(3, dtype=torch.int64) + 128
    locations = positions + 256

    class Layer:
        def __init__(self, idx: int) -> None:
            self.self_attention_res_proj = float(idx + 2)
            self.self_attention_res_norm = nn.Identity()
            self.prev_valid_blocks = idx // 3

        def __call__(self, positions, prefix_sum, ctx, blocks):
            prefix_sum.add_(1.0)
            return prefix_sum, blocks

    for stage in range(4):
        start, end = stage * 3, (stage + 1) * 3
        owned = tuple(
            idx
            for idx, tap in enumerate(tap_layers)
            if start <= context_tap_owner_layer(tap, target_layers, stream) < end
        )
        context_model = ContextWeights(
            weights, [None] * len(weights), final_norm, stage, owned
        )
        producer = PrefillTargetContextProducer(
            context_model, pool if stage == 3 else None
        )
        model = model_type()
        model.config = SimpleNamespace(
            num_hidden_layers=target_layers, attn_res_block_size=3
        )
        model.mapping = context_model.mapping
        model.pp_start_layer, model.pp_end_layer = start, end
        # Missing layers intentionally expose no mixing weights. A boundary
        # tap computed on its producer stage fails rather than being hidden.
        model.layers = [
            Layer(idx) if start <= idx < end else object()
            for idx in range(target_layers)
        ]
        model.layers_to_capture = tap_layers
        model.eagle3_layers_to_capture = ()
        model.dflash_aux_stream = stream
        model._dflash_capture_idx_map = {tap: idx for idx, tap in enumerate(tap_layers)}
        model.embed_tokens = lambda ids: torch.zeros(len(ids), 4)
        model.output_attn_res_proj = 99.0
        model.output_attn_res_norm = nn.Identity()
        model.norm = nn.Identity()
        ctx = SimpleNamespace(
            target_context_producer=producer,
            target_capture_sink=None,
            num_extends=1,
            bs=1,
            input_num_tokens=3,
            attn_backend=SimpleNamespace(extend_span_locations=lambda: locations),
        )
        output, auxiliary = model.forward(
            torch.zeros(3, dtype=torch.int64),
            positions,
            ctx,
            input_embeds=None,
            pp_inbound=state,
        )
        assert auxiliary is None
        if stage < 3:
            state = output
            assert state.projected_context.shape == (3, 3)
    expected_taps = []
    for tap in tap_layers:
        scale = (
            1.0
            if stream == "prefix"
            else (float(tap + 3) if tap + 1 < target_layers else 99.0)
        )
        expected_taps.append(torch.full((3, 4), (tap + 1) * scale))
    expected = final_norm(
        F.linear(torch.cat(expected_taps, dim=-1), torch.cat(weights, dim=-1))
    )
    assert len(context_model.writes) == 1
    torch.testing.assert_close(context_model.writes[0][0], expected)


def _context_loader_type():
    path = (
        Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/models/kimi_k3_dspark.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "K3DSparkContextModel"
    )
    names = ("load_weights", "checkpoint_weight_name_filter")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]

    def load_parameter(param, value):
        with torch.no_grad():
            param.copy_(value)

    namespace = {"torch": torch, "default_weight_loader": load_parameter}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return type(
        "ContextLoader", (nn.Module,), {name: namespace[name] for name in names}
    )


def test_checkpoint_projection_columns_and_fc_norm_follow_global_tap_indices() -> None:
    model = _context_loader_type()()
    model.config = SimpleNamespace(target_hidden_size=4)
    model.hidden_size = 3
    model.num_context_features = 5
    model.capture_indices = (2, 4)
    model.mapping = SimpleNamespace(is_last_pp_rank=False)
    model.tap_weights = nn.ParameterDict(
        {str(i): nn.Parameter(torch.zeros(3, 4)) for i in model.capture_indices}
    )
    model.tap_norms = nn.ModuleDict(
        {str(i): ReferenceNorm(4, 1e-5) for i in model.capture_indices}
    )
    projection = torch.arange(60, dtype=torch.float32).reshape(3, 20)
    model.load_weights(
        iter(
            [
                ("model.context_proj.weight", projection),
                ("fc_norm.4.weight", torch.full((4,), 4.0)),
                ("fc_norm.2.weight", torch.full((4,), 2.0)),
                ("fc_norm.0.weight", torch.zeros(4)),
                ("layers.0.mlp.gate_proj.weight", torch.empty(0)),
            ]
        )
    )
    torch.testing.assert_close(model.tap_weights["2"], projection[:, 8:12])
    torch.testing.assert_close(model.tap_weights["4"], projection[:, 16:20])
    torch.testing.assert_close(model.tap_norms["2"].weight, torch.full((4,), 2.0))
    torch.testing.assert_close(model.tap_norms["4"].weight, torch.full((4,), 4.0))
    assert not model.checkpoint_weight_name_filter("model.fc_norm.0.weight")
    assert not model.checkpoint_weight_name_filter(
        "model.layers.0.mlp.gate_proj.weight"
    )
    with pytest.raises(ValueError, match="missing weights"):
        model.load_weights(iter([("context_proj.weight", projection)]))


def test_empty_tap_stage_never_consumes_the_lazy_checkpoint_iterator() -> None:
    model = _context_loader_type()()

    def unexpected_weights():
        raise AssertionError("An empty tap stage must not read checkpoint shards")
        yield

    model.load_weights(unexpected_weights())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA context kernels required"
)
def test_minimal_context_models_load_only_owned_weights_and_write_mla() -> None:
    from tokenspeed.runtime.configs.kimi_k3_dspark_config import KimiK3DSparkConfig
    from tokenspeed.runtime.models.kimi_k3_dspark import K3DSparkContextModel

    config = KimiK3DSparkConfig(
        hidden_size=16,
        target_hidden_size=16,
        intermediate_size=32,
        num_attention_heads=8,
        num_key_value_heads=8,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        vocab_size=32,
        mask_token_id=31,
        markov_rank=4,
        target_layer_ids=[2, 23, 47, 71, 89],
        fc_norm=True,
        max_position_embeddings=1024,
    )
    generator = torch.Generator().manual_seed(97)
    checkpoint = {"context_proj.weight": torch.randn(16, 80, generator=generator)}
    checkpoint["context_norm.weight"] = torch.ones(16)
    for idx in range(5):
        checkpoint[f"fc_norm.{idx}.weight"] = torch.ones(16)
        checkpoint[f"layers.{idx}.self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(
            24, 16, generator=generator
        )
        checkpoint[f"layers.{idx}.self_attn.kv_a_layernorm.weight"] = torch.ones(16)
        # These large branches have no destination in any prefill model.
        checkpoint[f"layers.{idx}.mlp.gate_proj.weight"] = torch.empty(0)
    models = []
    for stage in range(4):
        mapping = SimpleNamespace(
            pp_size=4,
            pp_rank=stage,
            pp_layer_partition=None,
            is_first_pp_rank=stage == 0,
            is_last_pp_rank=stage == 3,
            attn=SimpleNamespace(
                tp_size=8, tp_group=tuple(range(stage * 8, (stage + 1) * 8))
            ),
        )
        with torch.device("cuda"):
            model = K3DSparkContextModel(config, mapping, None).to(torch.bfloat16)
        model.load_weights(iter(checkpoint.items()))
        assert len(model.layers) == (5 if stage == 3 else 0)
        assert all(
            "mlp" not in name and "markov" not in name
            for name, _ in model.named_parameters()
        )
        assert tuple(model.tap_weights) == tuple(
            str(i) for i in ((0, 1), (), (2,), (3, 4))[stage]
        )
        models.append(model)

    class Cache:
        def __init__(self) -> None:
            self.rows = {}

        def set_mla_kv_buffer(self, layer, loc, latent, rope) -> None:
            self.rows[layer.layer_id] = (loc.clone(), latent.clone(), rope.clone())

    cache = Cache()
    taps = [torch.randn(7, 16, generator=generator).cuda().bfloat16() for _ in range(5)]
    accumulator = None
    with torch.inference_mode():
        for model in models:
            producer = PrefillTargetContextProducer(
                model, cache if model.mapping.is_last_pp_rank else None
            )
            accumulator = producer.begin_stage(taps[0], accumulator)
            for idx in model.capture_indices:
                producer.add_capture(accumulator, idx, taps[idx])
        positions = torch.arange(7, device="cuda", dtype=torch.int64)
        locations = positions + 256
        producer.write_context(accumulator, positions, locations)
    assert set(cache.rows) == set(range(5))
    for layer_id, (loc, latent, rope) in cache.rows.items():
        torch.testing.assert_close(loc, locations)
        assert latent.shape == (7, 16)
        assert rope.shape == (7, 8)
        assert torch.isfinite(latent).all() and torch.isfinite(rope).all()
        assert latent.is_contiguous() and rope.is_contiguous()
