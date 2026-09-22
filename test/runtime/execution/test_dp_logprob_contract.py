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
"""CPU contract checks of production DP metadata and V4.1 row ownership.

These execute real methods with collective transport replaced; GPU/NCCL
serving remains a separate acceptance gate.
"""

from contextlib import contextmanager
from enum import IntEnum
from types import SimpleNamespace

import pytest
import test_top_logprob_capture as common
import torch


class Mode(IntEnum):
    IDLE = 0
    DECODE = 1
    EXTEND = 2
    MIXED = 3

    @classmethod
    def from_num_extends(cls, extends, bs):
        return cls.EXTEND if extends == bs else cls.MIXED if extends else cls.DECODE


def production_method(file, cls, method, namespace):
    return common.exec_nodes(
        [common.method(common.RUNTIME / file, cls, method)],
        f"_dp_contract_{method}",
        namespace,
    ).__dict__[method]


@pytest.mark.parametrize("full_prompt", [False, True])
@pytest.mark.parametrize(
    "counts,prefixes,prompts,expected",
    [
        ([8192], [0], [8192], 128),
        ([8192], [0], [16384], 1),
        ([20], [0], [20], 20),
        ([8192, 256, 1], [0, 0], [16384, 256], 130),
        ([1, 1], [], [], 2),
    ],
)
def test_dp_metadata_reports_decoder_counts(
    counts, prefixes, prompts, expected, full_prompt
):
    def gather(output, local, group):
        output.copy_(local.expand_as(output))

    method = production_method(
        "engine/event_loop.py",
        "EventLoop",
        "_dp_sync_and_check",
        {
            "ForwardMode": Mode,
            "DpForwardMetadata": common.TYPES.DpForwardMetadata,
            "dist": SimpleNamespace(all_gather_single=gather),
        },
    )
    loop = SimpleNamespace(
        _dp_local_info=torch.zeros(1, 4, dtype=torch.int32),
        _dp_global_info=torch.zeros(4, 4, dtype=torch.int32),
        _prefill_decoder_window=128,
        world_cpu_group=None,
        output_processor=SimpleNamespace(
            rid_to_state={
                i: SimpleNamespace(
                    return_logprob=full_prompt,
                    logprob_start_len=0,
                )
                for i in range(len(counts))
            }
        ),
    )
    op = SimpleNamespace(
        request_ids=list(range(len(counts))),
        input_lengths=counts,
        extend_prefix_lens=prefixes,
        prefill_lengths=prompts,
        num_extends=lambda: len(prefixes),
    )
    metadata = method(loop, op)
    assert metadata.global_num_tokens == [sum(counts)] * 4
    assert (
        metadata.global_decoder_num_tokens
        == [sum(counts) if full_prompt else expected] * 4
    )
    assert metadata.all_decode_or_idle == (len(prefixes) == 0)


def test_idle_dp_metadata_preserves_peer_decoder_counts():
    gathered = torch.tensor(
        [[8192, 1, Mode.EXTEND, 128], [0, 0, Mode.IDLE, 0]], dtype=torch.int32
    )

    def gather(output, local, group):
        assert local.tolist() == [[0, 0, Mode.IDLE, 0]]
        output.copy_(gathered)

    method = production_method(
        "engine/event_loop.py",
        "EventLoop",
        "_dp_sync_and_check",
        {
            "ForwardMode": Mode,
            "DpForwardMetadata": common.TYPES.DpForwardMetadata,
            "dist": SimpleNamespace(all_gather_single=gather),
        },
    )
    loop = SimpleNamespace(
        _dp_local_info=torch.zeros(1, 4, dtype=torch.int32),
        _dp_global_info=torch.zeros(2, 4, dtype=torch.int32),
        _prefill_decoder_window=128,
        world_cpu_group=None,
    )
    result = method(loop, None)
    assert result.need_idle_forward and not result.all_decode_or_idle
    assert result.global_decoder_num_tokens == [128, 0]


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_v41_moe_scatters_and_restores_attention_tp_rows(tp_rank):
    # Two TP peers own [2, 1] rows of this DP replica, before joining global EP.
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    local = x[:2] if tp_rank == 0 else x[2:]

    def pre(value, ctx):
        torch.testing.assert_close(value, local)
        return x

    def post(value, residual, ctx):
        torch.testing.assert_close(value, x + 1)
        return local + 1, None

    def gather(value, *, group, scattered_num_tokens):
        torch.testing.assert_close(value, local + 1)
        assert scattered_num_tokens == [2, 1] and group == (0, 1)
        return x + 1

    method = production_method(
        "models/deepseek_v41.py",
        "DeepseekV41DecoderLayer",
        "_forward_ffn",
        {
            "token_all_gather": gather,
        },
    )

    class Experts:
        use_mega_moe = False

        def __call__(self, value, mask, total, maximum, *, ctx, comm_manager):
            assert (total, maximum) == (3, 2)
            return value + 1

    layer = SimpleNamespace(
        ffn=Experts(),
        comm_manager=SimpleNamespace(
            mapping=SimpleNamespace(
                attn=SimpleNamespace(
                    tp_size=2, tp_rank=tp_rank, tp_group=(0, 1), has_dp=True
                )
            ),
            use_all_reduce=lambda **kwargs: False,
            attn_tp_group_scattered_num_tokens=lambda ctx: [2, 1],
            pre_mlp_comm=pre,
            post_mlp_comm=post,
            get_num_tokens=lambda ctx: (3, 2),
        ),
    )
    torch.testing.assert_close(method(layer, x, None, object()), x + 1)


def test_v41_idle_rank_runs_every_moe_with_stage_counts():
    events = []

    @contextmanager
    def sizing(ctx, local, counts):
        assert local == 0
        events.append(tuple(counts))
        yield

    method = production_method(
        "models/deepseek_v41.py",
        "DeepseekV41Model",
        "forward",
        {
            "report_collective_sizing": sizing,
        },
    )

    def expert(value, mask, ctx):
        assert value.shape == (0, 4)
        events.append("moe")

    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4),
        embed_tokens=SimpleNamespace(weight=torch.empty(8, 4)),
        ced_decoder_start=2,
        layers=[SimpleNamespace(layer_id=i, _forward_ffn=expert) for i in range(4)],
    )
    ctx = SimpleNamespace(
        global_num_tokens=[8192, 0], global_decoder_num_tokens=[128, 0]
    )
    hidden, capture = method(
        model,
        torch.empty(0, dtype=torch.int32),
        None,
        ctx,
        None,
        None,
        engram_previous_tokens=None,
        engram_token_mask=None,
        image_mask=None,
    )
    assert hidden.shape == (0, 4) and capture is None
    assert events == [
        (8192, 0),
        "moe",
        (8192, 0),
        "moe",
        (128, 0),
        "moe",
        (128, 0),
        "moe",
    ]


def test_prompt_scoring_keeps_all_decoder_rows_and_original_spans():
    method = production_method(
        "layers/attention/backends/specific/deepseek_v41.py",
        "DeepseekV41AttentionBackend",
        "_build_decoder_view",
        {
            "V41DecoderView": lambda *args: args,
        },
    )
    spans = [SimpleNamespace(count=8192), SimpleNamespace(count=256)]
    prefill = SimpleNamespace(positions=torch.empty(8448))
    backend = SimpleNamespace(
        _prefill_spans=spans,
        forward_prefill_metadata=prefill,
        query_metadata=lambda mode: "full",
    )
    result = method(backend, prefill, [False, True], 128, None, full_prompt_logits=True)
    assert result == ("full", prefill, spans, None, None)
