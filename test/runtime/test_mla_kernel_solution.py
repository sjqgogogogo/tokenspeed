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

"""MLA plugin solution binding with real shared CPU metadata buffers."""

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig


class PluginMLABackend(MLAAttnBackend):
    @classmethod
    def resolve_kernel_solution(cls, backend_name: str) -> str:
        assert backend_name == "plugin_mla"
        return "plugin_solution"


@pytest.mark.parametrize(
    "backend_cls,name,solution",
    [
        (MLAAttnBackend, "mla", None),
        (MLAAttnBackend, "gluon", "gluon"),
        (PluginMLABackend, "plugin_mla", "plugin_solution"),
    ],
)
def test_solution_binding_preserves_backend_identity_and_decode_refresh(
    backend_cls,
    name,
    solution,
):
    spec = MLAConfig(
        backend_name=name,
        num_attention_heads=32,
        num_kv_heads=1,
        head_dim=192,
        attn_tp_size=1,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=576,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=128,
        context_len=512,
        max_bs=2,
        components=(spec,),
    )
    backend = backend_cls(config, spec, kernel_page_size=128)
    assert backend.kernel_solution == solution
    assert config.component(MLAConfig) is spec
    assert spec.backend_name == name
    backend.init_cuda_graph_state(2)
    table = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
    lengths = torch.tensor([7, 129], dtype=torch.int32)
    backend.refresh_decode_metadata(2, 2, lengths, table)
    metadata = backend.forward_decode_metadata
    address = metadata.seq_lens.data_ptr()
    assert torch.equal(metadata.seq_lens, lengths)
    assert torch.equal(metadata.page_table, table)
    backend.refresh_decode_metadata(2, 2, lengths + 1, table.flip(1))
    assert backend.forward_decode_metadata.seq_lens.data_ptr() == address
    assert torch.equal(backend.forward_decode_metadata.seq_lens, lengths + 1)
    assert torch.equal(backend.forward_decode_metadata.page_table, table.flip(1))
