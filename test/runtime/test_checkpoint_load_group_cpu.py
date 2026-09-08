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

"""Execute loader boundaries with fake I/O and process groups, without CUDA.

The production methods are loaded independently of the runtime's kernel
imports. Only transport and file I/O are replaced; group selection and
iterator forwarding execute the implementation used by GPU workers.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_function(filename: str, parent: str | None, name: str, namespace: dict):
    path = (
        Path(__file__).parents[2] / "python/tokenspeed/runtime/model_loader" / filename
    )
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if parent is not None:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == parent
        ).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("stage_ranks", [None, tuple(range(16, 24))])
def test_default_loader_respects_the_checkpoint_consumer_group(
    monkeypatch, stage_ranks
) -> None:
    world, stage = object(), object()
    selections = []

    def get_group(backend, ranks):
        selections.append((backend, ranks))
        return stage

    monkeypatch.setitem(
        sys.modules,
        "tokenspeed.runtime.distributed.process_group_manager",
        SimpleNamespace(
            process_group_manager=SimpleNamespace(get_process_group=get_group)
        ),
    )
    distributed = SimpleNamespace(
        is_initialized=lambda: True,
        get_world_size=lambda: 32,
        group=SimpleNamespace(WORLD=world),
    )
    calls = []

    def iterator(files, *, process_group):
        calls.append((files, process_group))
        return iter([("context_proj.weight", 7)])

    formats = SimpleNamespace(
        INSTANTTENSOR="instanttensor", NPCACHE="npcache", MISTRAL="mistral"
    )
    function = _load_function(
        "loader.py",
        "DefaultModelLoader",
        "_get_weights_iterator",
        {
            "torch": SimpleNamespace(distributed=distributed),
            "LoadFormat": formats,
            "instanttensor_weights_iterator": iterator,
        },
    )
    loader = SimpleNamespace(
        load_config=SimpleNamespace(load_format="instanttensor"),
        _prepare_weights=lambda model, revision, fallback: (
            "/weights",
            ["stage.safetensors"],
            True,
        ),
    )
    source = SimpleNamespace(
        model_or_path="model", revision=None, fall_back_to_pt=False, prefix="draft."
    )
    assert list(function(loader, source, None, stage_ranks)) == [
        ("draft.context_proj.weight", 7)
    ]
    assert calls == [(["stage.safetensors"], world if stage_ranks is None else stage)]
    assert selections == ([] if stage_ranks is None else [("nccl", stage_ranks)])


@pytest.mark.parametrize("distributed_loading", [False, True])
def test_instanttensor_does_not_replace_an_explicit_group_with_world(
    distributed_loading: bool,
) -> None:
    group = object() if distributed_loading else None
    calls = []

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def keys(self):
            return ["weight"]

        def tensors(self):
            return iter([("weight", 11)])

    def safe_open(files, **kwargs):
        calls.append((files, kwargs))
        return Reader()

    function = _load_function(
        "weight_utils.py",
        None,
        "_instanttensor_tensors",
        {
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(current_device=lambda: 2),
                distributed=SimpleNamespace(
                    is_initialized=lambda: True, get_rank=lambda: 18
                ),
            ),
            "tqdm": lambda values, **kwargs: values,
            "_BAR_FORMAT": "",
        },
    )
    assert list(function(SimpleNamespace(safe_open=safe_open), ["weights"], group)) == [
        ("weight", 11)
    ]
    assert calls[0][1]["process_group"] is group
    assert calls[0][1]["device"] == 2
