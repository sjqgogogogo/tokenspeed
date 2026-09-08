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

"""Exercise the real rank mapping without importing GPU collective backends."""

import importlib.util
from pathlib import Path


def _mapping_type():
    path = (
        Path(__file__).parents[2] / "python/tokenspeed/runtime/distributed/mapping.py"
    )
    spec = importlib.util.spec_from_file_location("_k3_pd_mapping", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Mapping


Mapping = _mapping_type()


def test_prefill_pp4_tp8_ep8_stays_inside_each_h200_node() -> None:
    attention_groups = set()
    for rank in range(32):
        mapping = Mapping(
            world_size=32,
            pp_size=4,
            attn_tp_size=8,
            attn_cp_size=1,
            attn_dp_size=1,
            dense_tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            nprocs_per_node=8,
            nnodes=4,
        )
        mapping.rank = rank
        node = rank // 8
        node_ranks = tuple(range(node * 8, (node + 1) * 8))
        assert mapping.pp_rank == node
        assert mapping.attn.tp_group == node_ranks
        assert mapping.dense.tp_group == node_ranks
        assert mapping.moe.ep_group == node_ranks
        assert mapping.moe.tp_size == 1
        assert mapping.pp_group == tuple(range(rank % 8, 32, 8))
        attention_groups.add(mapping.attn.tp_group)
    assert len(attention_groups) == 4


def test_decode_dp4_tp8_ep32_shares_experts_across_attention_replicas() -> None:
    attention_groups = set()
    for rank in range(32):
        mapping = Mapping(
            world_size=32,
            pp_size=1,
            attn_tp_size=8,
            attn_cp_size=1,
            attn_dp_size=4,
            dense_tp_size=8,
            moe_tp_size=1,
            moe_ep_size=32,
            nprocs_per_node=8,
            nnodes=4,
        )
        mapping.rank = rank
        dp_rank = rank // 8
        assert mapping.attn.dp_rank == dp_rank
        assert mapping.attn.tp_group == tuple(range(dp_rank * 8, (dp_rank + 1) * 8))
        assert mapping.dense.tp_group == mapping.attn.tp_group
        assert mapping.moe.ep_group == tuple(range(32))
        assert mapping.moe.tp_size == 1
        assert mapping.moe.dp_size == 1
        attention_groups.add(mapping.attn.tp_group)
    assert len(attention_groups) == 4
