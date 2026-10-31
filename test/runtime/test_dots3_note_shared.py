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

"""CPU numerical and checkpoint-loader tests for Dots3 shared row ownership."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.configs.dots3_note_config import Dots3NoteConfig
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.dots3_note import (
    Dots3NoteAttention,
    Dots3NoteOwnedRowsMoE,
    Dots3NoteReplicatedSharedMLP,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.fixture(autouse=True)
def execution_options(monkeypatch):
    for key, value in {
        "numerics": "auto",
        "moe_combine_order": "rank",
        "all2all_backend": "none",
        "layer_boundary_norm": "fused",
    }.items():
        monkeypatch.setitem(global_server_args_dict, key, value)


def shared_mlp(rank, *, hidden=4, intermediate=7, quant_config=None):
    return Dots3NoteReplicatedSharedMLP(
        hidden,
        intermediate,
        "silu",
        quant_config,
        "model.layers.1.mlp.shared_experts",
        owner_rank=rank,
        owner_size=8,
    )


@pytest.mark.parametrize("rows", [0, 1, 3, 7, 8, 11, 25])
@torch.inference_mode()
def test_owned_shared_rows_reproduce_one_complete_shared_contribution(
    rows, monkeypatch
):
    from tokenspeed.runtime.layers.dense import unquant

    # Only replace the unavailable GPU GEMM boundary. The real linear layers,
    # weights, activation, row selection and in-place contribution all run.
    monkeypatch.setattr(
        unquant, "kernel_mm", lambda x, weight, *, bias=None: F.linear(x, weight, bias)
    )
    torch.manual_seed(28)
    hidden, intermediate = 4, 7
    x = torch.randn(rows, hidden)
    gate, up = torch.randn(2, intermediate, hidden)
    down = torch.randn(hidden, intermediate)
    routed = torch.randn(8, rows, hidden)
    expected_shared = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
    partials, owned_counts = [], []
    for rank in range(8):
        module = shared_mlp(rank)
        module.gate_up_proj.weight.copy_(torch.cat((gate, up)))
        module.down_proj.weight.copy_(down)
        contribution = module(x)
        partial = routed[rank].clone()
        if contribution is not None:
            assert module.add_to_routed(partial, contribution) is partial
            owned_counts.append(contribution.shape[0])
        else:
            owned_counts.append(0)
        partials.append(partial)
    assert sum(owned_counts) == rows
    assert max(owned_counts) - min(owned_counts) <= 1
    torch.testing.assert_close(
        torch.stack(partials).sum(0),
        routed.sum(0) + expected_shared,
        atol=3e-6,
        rtol=3e-6,
    )
    # A reduce-scatter returns these same sums on the original token owners,
    # which need not be the shared compute owners.
    reference = routed.sum(0) + expected_shared
    for actual, wanted in zip(
        torch.tensor_split(torch.stack(partials).sum(0), 8),
        torch.tensor_split(reference, 8),
        strict=True,
    ):
        torch.testing.assert_close(actual, wanted, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("rank", [0, 7])
@torch.inference_mode()
def test_replica_loads_complete_fp8_weights_and_block_scales(rank):
    hidden, intermediate = 256, 1536
    module = shared_mlp(
        rank,
        hidden=hidden,
        intermediate=intermediate,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
        ),
    )
    gate = torch.full((intermediate, hidden), 0.25, dtype=torch.float8_e4m3fn)
    up = torch.full_like(gate, 0.5)
    down = torch.full((hidden, intermediate), 0.75, dtype=torch.float8_e4m3fn)
    gate_scale = torch.full((12, 2), 1.25)
    up_scale = torch.full((12, 2), 1.5)
    down_scale = torch.full((2, 12), 2.0)
    merged = module.gate_up_proj
    for param, value, shard in (
        (merged.weight, gate, 0),
        (merged.weight, up, 1),
        (merged.weight_scale_inv, gate_scale, 0),
        (merged.weight_scale_inv, up_scale, 1),
    ):
        param.weight_loader(param, value, shard)
    for param, value in (
        (module.down_proj.weight, down),
        (module.down_proj.weight_scale_inv, down_scale),
    ):
        param.weight_loader(param, value)
    torch.testing.assert_close(
        merged.weight.float(), torch.cat((gate, up)).float(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        merged.weight_scale_inv, torch.cat((gate_scale, up_scale)), atol=0, rtol=0
    )
    torch.testing.assert_close(
        module.down_proj.weight.float(), down.float(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        module.down_proj.weight_scale_inv, down_scale, atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("moe_combine_order", "slot", "rank combine"),
        ("all2all_backend", "deepep", "non-A2A"),
        ("numerics", "rl-bitwise", "numerics=auto"),
    ],
)
def test_owned_shared_rejects_unsupported_execution(key, value, message, monkeypatch):
    monkeypatch.setitem(global_server_args_dict, key, value)
    config = Dots3NoteConfig(shared_experts_strategy="replicated_owned_rows")
    with pytest.raises(ValueError, match=message):
        Dots3NoteOwnedRowsMoE(config, Mapping(rank=0, world_size=8), None, 1, "", None)


@pytest.mark.parametrize("gathered_rows", [0, 1])
@torch.inference_mode()
def test_empty_attention_shard_joins_required_gather_before_return(
    gathered_rows, monkeypatch
):
    from tokenspeed.runtime.distributed import comm_manager as comm_module

    class CPUAttention(Dots3NoteAttention):
        def __init__(self):
            nn.Module.__init__(self)

        def _forward(self, positions, hidden_states, ctx, comm_manager, block_scale):
            # Stand in for the GPU attention kernels after communication.
            return hidden_states * 2

    recovered = torch.arange(gathered_rows * 4, dtype=torch.float32).reshape(-1, 4)

    def cpu_gather(value, *, group, scattered_num_tokens):
        assert value.shape == (0, 4)
        return recovered

    monkeypatch.setattr(comm_module, "token_all_gather", cpu_gather)
    mapping = Mapping(
        rank=7,
        world_size=8,
        attn_tp_size=8,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
    )
    comm = CommManager(
        mapping,
        layer_id=1,
        is_moe=True,
        prev_is_moe=False,
        dense_batch_invariant=False,
        query_sharded=False,
    )
    backend = Mock(spec=AttentionBackend)
    backend.record_pd_cache_step.return_value = nullcontext()
    ctx = ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=None,
        bs=1,
        num_extends=0,
        input_num_tokens=gathered_rows,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 1, 1),
    )
    result = CPUAttention()(
        torch.empty(0, dtype=torch.int64), torch.empty(0, 4), ctx, comm
    )
    torch.testing.assert_close(result, recovered * 2, atol=0, rtol=0)


def test_shared_strategy_is_an_explicit_checkpoint_override():
    assert Dots3NoteConfig().shared_experts_strategy is None
    config = Dots3NoteConfig(shared_experts_strategy="replicated_owned_rows")
    assert (
        Dots3NoteConfig(**config.to_dict()).shared_experts_strategy
        == "replicated_owned_rows"
    )
    with pytest.raises(ValueError, match="shared_experts_strategy"):
        Dots3NoteConfig(shared_experts_strategy="unknown")
    config.update({"shared_experts_strategy": "unknown"})
    with pytest.raises(ValueError, match="shared_experts_strategy"):
        config.resolve_shared_experts_strategy()
