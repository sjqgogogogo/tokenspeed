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

"""CPU protocol tests; GPU JIT/PDL execution remains a target-host check."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = (
    ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/_flashinfer"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"test_{name}", ADAPTERS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module(monkeypatch, name, **values):
    module = ModuleType(name)
    vars(module).update(values)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def adapter(monkeypatch):
    def legacy(*, q):
        return q

    _module(monkeypatch, "torch")
    _module(monkeypatch, "flashinfer")
    _module(
        monkeypatch,
        "flashinfer.gdn_decode",
        gated_delta_rule_decode_pretranspose=legacy,
    )
    _module(monkeypatch, "flashinfer.gdn_prefill", chunk_gated_delta_rule=legacy)
    _module(monkeypatch, "flashinfer.gdn_kernels")
    _module(
        monkeypatch,
        "flashinfer.gdn_kernels.gdn_decode_mtp",
        get_tile_v_mtp=None,
        get_vec_size_mtp=None,
    )
    return _load("adapter")


def test_legacy_decode_preserves_explicit_flashinfer_selection(adapter):
    assert (
        adapter.gated_delta_rule_decode_pretranspose(
            q=7, backend="flashinfer", enable_pdl=False
        )
        == 7
    )
    assert (
        adapter.chunk_gated_delta_rule(q=9, backend="flashinfer", enable_pdl=False) == 9
    )
    with pytest.raises(ValueError, match="only supports"):
        adapter.gated_delta_rule_decode_pretranspose(
            q=7, backend="cake_gdn", enable_pdl=False
        )


def test_modern_backend_keyword_is_preserved(adapter):
    def modern(*, q, backend):
        return q, backend

    supplied = {"q": 7, "backend": "flashinfer"}
    assert adapter._backend_kwargs(modern, supplied) is supplied
    assert modern(**adapter._backend_kwargs(modern, supplied)) == (7, "flashinfer")


@pytest.mark.parametrize("persistent", [False, True])
def test_pdl_namespace_supports_memory_only_and_persistent_caches(
    monkeypatch, persistent
):
    cutlass = _module(monkeypatch, "cutlass", Constexpr=object())
    cute = _module(monkeypatch, "cutlass.cute", kernel=lambda f: f, jit=lambda f: f)
    cutlass.cute = cute
    pdl = _load("pdl")
    upstream = ModuleType("upstream")
    calls = []

    def builder(module, kernel, compile_fn, *, extra_key_files):
        calls.append((module, extra_key_files))
        return compile_fn()

    upstream.cache = {}
    if persistent:
        upstream.build_and_load_cute_dsl_kernel = builder
        source = "def run():\n    return build_and_load_cute_dsl_kernel('gdn', 'kernel', lambda: 42, extra_key_files=())\n"
    else:
        source = "def run():\n    cache['value'] = 42\n    return cache['value']\n"
    exec(source, vars(upstream))
    namespace = pdl._adapt_module(
        upstream,
        kernels=(),
        launchers=(),
        entrypoints=("run",),
        caches=("cache",),
        overrides={},
    )
    assert namespace["run"]() == 42
    assert upstream.cache == {}
    if persistent:
        assert calls[0][0] == "tokenspeed_pdl_gdn"
        assert any(path.endswith("adapter.py") for path in calls[0][1])
        assert upstream.build_and_load_cute_dsl_kernel is builder
    else:
        assert "build_and_load_cute_dsl_kernel" not in namespace
        assert namespace["cache"] == {"value": 42}
