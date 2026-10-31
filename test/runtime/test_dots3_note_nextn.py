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


"""MTP checkpoint ownership and accepted-prefix narrowing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tokenspeed.runtime.models.dots3_note_nextn import Dots3NoteForCausalLMNextN


def test_draft_loader_preserves_own_embedding_and_shares_only_head(monkeypatch):
    from tokenspeed.runtime.execution.factory import _share_target_embed_and_head

    draft = Dots3NoteForCausalLMNextN.__new__(Dots3NoteForCausalLMNextN)
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(mtp_layer_offset=46)
    draft.model = nn.Module()
    draft.model.embed_tokens = nn.Embedding(8, 4)
    draft.model.eh_proj = nn.Linear(8, 4, bias=False)
    draft.model.decoder = nn.Module()
    draft.model.decoder.input_layernorm = nn.LayerNorm(4, bias=False)
    draft.lm_head = nn.Linear(4, 8, bias=False)
    draft.post_load_weights = Mock()
    own_embedding = draft.model.embed_tokens.weight
    loaded = draft.load_weights(
        [
            ("model.embed_tokens.weight", torch.full((8, 4), 99.0)),
            ("model.mtp.embed_tokens.weight", torch.full((8, 4), 2.0)),
            ("model.layers.46.eh_proj.weight", torch.full((4, 8), 3.0)),
            ("model.layers.46.input_layernorm.weight", torch.full((4,), 4.0)),
            ("model.layers.45.input_layernorm.weight", torch.full((4,), 99.0)),
        ]
    )
    assert loaded == {
        "model.embed_tokens.weight",
        "model.eh_proj.weight",
        "model.decoder.input_layernorm.weight",
    }
    assert torch.all(own_embedding == 2)
    assert torch.all(draft.model.eh_proj.weight == 3)
    assert torch.all(draft.model.decoder.input_layernorm.weight == 4)
    main_embedding = nn.Parameter(torch.ones(8, 4))
    main_head = nn.Parameter(torch.zeros(8, 4))
    main_lm_head = nn.Linear(4, 8, bias=False)
    main_lm_head.weight = main_head
    layout = SimpleNamespace(tp_group=None, tp_size=1, dp_lm_head_tp=False)
    draft.logits_processor = layout
    target = SimpleNamespace(
        logits_processor=layout,
        get_embed_and_head=lambda: (main_embedding, main_head),
        lm_head=main_lm_head,
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    runner = SimpleNamespace(
        model=target, mapping=SimpleNamespace(has_pp=False, is_last_pp_rank=True)
    )
    _share_target_embed_and_head(runner, draft)
    assert draft.model.embed_tokens.weight is own_embedding
    assert draft.lm_head.weight is main_head
    with pytest.raises(KeyError):
        draft.load_weights([("model.layers.46.unknown.weight", torch.ones(1))])


@pytest.mark.parametrize("quantized", [False, True])
@torch.inference_mode()
def test_factory_shares_complete_head_and_keeps_mtp_embedding(quantized, monkeypatch):
    from tokenspeed.runtime.configs.dots3_note_config import Dots3NoteConfig
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.execution.factory import _share_target_embed_and_head
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
    from tokenspeed.runtime.layers.linear import ReplicatedLinear
    from tokenspeed.runtime.layers.logits_processor import (
        LogitsMetadata,
        LogitsProcessor,
    )
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    monkeypatch.setitem(global_server_args_dict, "logprob_order", "torch")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    hidden_size, vocab_size = 128, 256
    config = Dots3NoteConfig(hidden_size=hidden_size, vocab_size=vocab_size)
    quant = (
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        if quantized
        else None
    )
    target_head = ReplicatedLinear(
        hidden_size,
        vocab_size,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant,
        prefix="lm_head",
    )
    target_head.weight.fill_(0.25)
    if quantized:
        target_head.weight_scale_inv.fill_(1.25)

    def cpu_fp8_apply(method, layer, value, bias=None):
        # Replace the GPU FP8 GEMM only; consume the actual shared module's
        # weights and block scales, so a weight-only alias cannot pass.
        scale = layer.weight_scale_inv.repeat_interleave(128, 0).repeat_interleave(
            128, 1
        )
        weight = layer.weight.float() * scale
        return torch.nn.functional.linear(value.float(), weight, bias).to(value.dtype)

    monkeypatch.setattr(Fp8LinearMethod, "apply", cpu_fp8_apply)

    class Target(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = target_head
            self.logits_processor = LogitsProcessor(config, dp_lm_head_tp=False)

        def get_embed_and_head(self):
            return self.embed_tokens.weight, self.lm_head.weight

    target = Target()
    draft = Dots3NoteForCausalLMNextN.__new__(Dots3NoteForCausalLMNextN)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    draft.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
    draft.model.embed_tokens.weight.fill_(2.0)
    own_embedding = draft.model.embed_tokens.weight
    draft.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    draft.logits_processor = LogitsProcessor(config, dp_lm_head_tp=False)
    # ModelRunner's checkpoint/process-group initialization needs GPUs; only
    # the real wiring function's two already-prepared fields are needed here.
    from tokenspeed.runtime.execution.model_runner import ModelRunner

    runner = ModelRunner.__new__(ModelRunner)
    runner.model = target
    runner.mapping = Mapping(rank=0, world_size=8, attn_tp_size=1, attn_dp_size=8)
    _share_target_embed_and_head(runner, draft)
    assert draft.model.embed_tokens.weight is own_embedding
    torch.testing.assert_close(own_embedding, torch.full_like(own_embedding, 2.0))
    assert draft.lm_head is target.lm_head
    assert draft.lm_head.quant_method is target_head.quant_method
    if quantized:
        assert draft.lm_head.weight_scale_inv is target_head.weight_scale_inv
    hidden = torch.ones(2, hidden_size, dtype=torch.bfloat16)
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    logits = draft.logits_processor._get_logits(
        hidden, draft.lm_head, metadata, require_full_vocab=False
    )
    expected = 40.0 if quantized else 32.0
    torch.testing.assert_close(
        logits, torch.full_like(logits, expected), atol=0, rtol=0
    )
    if quantized:
        # Later target scale updates must remain visible without rewiring.
        target_head.weight_scale_inv.fill_(2.5)
        logits = draft.logits_processor._get_logits(
            hidden, draft.lm_head, metadata, require_full_vocab=False
        )
        torch.testing.assert_close(
            logits, torch.full_like(logits, 80.0), atol=0, rtol=0
        )


@pytest.mark.parametrize("draft_model", [False, True])
@pytest.mark.parametrize("attention_dp", [False, True])
@torch.inference_mode()
def test_native_bf16_checkpoint_head_survives_dp_and_tp_layouts(
    draft_model, attention_dp, monkeypatch
):
    from tokenspeed.runtime.configs.dots3_note_config import Dots3NoteConfig
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.logits_processor import (
        LogitsMetadata,
        LogitsProcessor,
    )
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
    from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype
    from tokenspeed.runtime.models.dots3_note import Dots3NoteForCausalLM
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    monkeypatch.setitem(global_server_args_dict, "logprob_order", "torch")
    hidden_size, vocab_size, rank = 128, 256, 3
    config = Dots3NoteConfig(
        hidden_size=hidden_size, vocab_size=vocab_size, tie_word_embeddings=False
    )
    quant = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    model_class = Dots3NoteForCausalLMNextN if draft_model else Dots3NoteForCausalLM
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.mapping = Mapping(
        rank=rank,
        world_size=8,
        attn_tp_size=1 if attention_dp else 8,
        attn_dp_size=8 if attention_dp else 1,
    )
    # Construct the real head with the same BF16 allocation scope as the
    # checkpoint loader. The full GPU transformer need not be allocated.
    with set_default_torch_dtype(torch.bfloat16):
        head = model.resolve_lm_head(config, quant, "")
    checkpoint_weight = (
        torch.arange(vocab_size, dtype=torch.float32)[:, None]
        .expand(vocab_size, hidden_size)
        .div(1024)
        .to(torch.bfloat16)
        .contiguous()
    )
    head.weight_loader(head.weight, checkpoint_weight)
    assert head.weight.dtype == torch.bfloat16
    assert not hasattr(head, "weight_scale_inv")
    # Ordinary TP keeps the original ParallelLMHead quantization metadata;
    # only the DP-replicated head excludes the global serialized-FP8 config.
    assert head.quant_config is (None if attention_dp else quant)
    width = vocab_size if attention_dp else vocab_size // 8
    start = 0 if attention_dp else rank * width
    expected_weight = checkpoint_weight[start : start + width]
    torch.testing.assert_close(head.weight, expected_weight, atol=0, rtol=0)
    processor = LogitsProcessor(
        config,
        skip_all_gather=True,
        tp_rank=model.mapping.lm_head.tp_rank,
        tp_size=model.mapping.lm_head.tp_size,
        tp_group=model.mapping.lm_head.tp_group,
        dp_lm_head_tp=False,
    )
    hidden = torch.ones(2, hidden_size, dtype=torch.bfloat16)
    logits = processor._get_logits(
        hidden,
        head,
        LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None),
        require_full_vocab=False,
    )
    torch.testing.assert_close(
        logits, torch.nn.functional.linear(hidden, expected_weight), atol=0, rtol=0
    )


@pytest.mark.parametrize("mode_name", ["MIXED", "DECODE"])
def test_first_draft_writes_all_rows_but_attends_only_live_rows(monkeypatch, mode_name):
    import contextlib

    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.models import dots3_note_nextn as impl

    mode = ForwardMode[mode_name]
    attn = impl.Dots3NoteDraftAttention.__new__(impl.Dots3NoteDraftAttention)
    nn.Module.__init__(attn)
    attn.num_local_heads = 1
    attn.head_start = 0
    attn.v_head_dim = attn.kv_lora_rank = 2
    attn.w_vc = None
    attn.q_a_proj = attn.kv_a_proj_with_mqa = attn.q_b_proj = lambda x: (x, None)
    attn.normalize_latents = lambda q, kv: (q, kv)
    attn.g_proj = lambda x: (torch.zeros(x.shape[0], 1), None)
    attn.o_proj = lambda x: (x, None)
    rows = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    positions = torch.arange(5)
    live = torch.tensor([1, 4])
    events = []

    def absorb(q, latent, pos, ctx, slots):
        events.append("write")
        assert ctx.forward_mode == ForwardMode.DECODE
        torch.testing.assert_close(q, rows)
        torch.testing.assert_close(pos, positions)
        torch.testing.assert_close(slots, positions)
        return q

    def core(q, k, v, *, positions, ctx, record_kv_cache):
        events.append("attend")
        torch.testing.assert_close(q, rows[live])
        assert record_kv_cache == (mode_name == "MIXED")
        return q

    attn._absorb = absorb
    attn.attn_mqa = core
    backend = SimpleNamespace(
        forward_write_locations=lambda layer, mode: positions,
        override_num_extends=lambda n: contextlib.nullcontext(),
    )
    ctx = ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=None,
        bs=2,
        num_extends=1 if mode_name == "MIXED" else 0,
        input_num_tokens=5,
        forward_mode=mode,
        output_layout=None,
        gather_ids=live,
        draft_narrowing=SimpleNamespace(
            publish_accepted_prefix=lambda: events.append("publish")
        ),
    )
    monkeypatch.setattr(
        impl, "mla_project_value", lambda x, w, *, out: out.copy_(x.flatten(1))
    )
    result = attn(positions, rows, ctx, SimpleNamespace(pre_attn_comm=lambda x, c: x))
    assert events == ["publish", "write", "attend"]
    torch.testing.assert_close(result, rows[live] / 2)
    torch.testing.assert_close(
        impl.Dots3NoteDraftDecoderLayer.narrow_residual(None, rows, ctx), rows[live]
    )


def test_target_prefill_marks_pd_ready_after_all_cache_writes():
    from tokenspeed.runtime.distributed import Mapping
    from tokenspeed.runtime.distributed.comm_manager import CommManager
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.models.dots3_note import Dots3NoteAttention

    events = []
    backend = SimpleNamespace(
        step_counter=SimpleNamespace(record_cache=lambda: events.append("ready"))
    )
    backend.record_pd_cache_step = (
        lambda *a, **kw: AttentionBackend.record_pd_cache_step(backend, *a, **kw)
    )
    attn = Dots3NoteAttention.__new__(Dots3NoteAttention)
    nn.Module.__init__(attn)
    rows = torch.ones(2, 4)

    def compute(*args):
        events.append("write_latent_and_index")
        return rows

    attn._forward = compute
    comm = CommManager(
        Mapping(rank=0, world_size=1),
        layer_id=0,
        is_moe=False,
        prev_is_moe=False,
        dense_batch_invariant=False,
        query_sharded=False,
    )
    ctx = SimpleNamespace(attn_backend=backend, forward_mode=ForwardMode.EXTEND)
    assert attn(torch.arange(2), rows, ctx, comm) is rows
    assert events == ["write_latent_and_index", "ready"]
    events.clear()
    ctx.forward_mode = ForwardMode.DECODE
    attn(torch.arange(2), rows, ctx, comm)
    assert events == ["write_latent_and_index"]


@torch.inference_mode()
def test_idle_draft_skips_local_projection_but_still_reaches_decoder():
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.layernorm import RMSNorm
    from tokenspeed.runtime.models.dots3_note_nextn import Dots3NoteModelNextN

    class UnexpectedProjection(nn.Module):
        def forward(self, value):
            raise AssertionError("idle draft must not launch the GPU projection")

    class CPUDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.received = None

        def forward(self, positions, hidden_states, ctx, residual):
            self.received = hidden_states
            return hidden_states, residual

    # The checkpoint/GPU allocations are unavailable in this CPU test. Run
    # the actual model forward with real lightweight torch modules instead.
    model = Dots3NoteModelNextN.__new__(Dots3NoteModelNextN)
    nn.Module.__init__(model)
    model.hidden_size = 4
    model.embed_tokens = nn.Embedding(8, 4)
    model.enorm = RMSNorm(4)
    model.hnorm = RMSNorm(4)
    model.eh_proj = UnexpectedProjection()
    model.decoder = CPUDecoder()
    model.shared_head = nn.ModuleDict({"norm": RMSNorm(4)})
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=0,
        num_extends=0,
        input_num_tokens=0,
        forward_mode=ForwardMode.IDLE,
        output_layout=ForwardOutputLayout(0, 0, 0, 1),
    )
    hidden, auxiliary = model(
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.int64),
        ctx,
    )
    assert model.decoder.received is hidden
    assert hidden.shape == (0, 4)
    assert hidden.dtype == model.enorm.weight.dtype
    assert auxiliary is None
