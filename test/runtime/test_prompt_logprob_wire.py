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

"""Real msgspec wire regression tests, not mock dataclass serialization.

Load only the production IPC classes by AST to avoid GPU imports. Their
msgspec.Struct bases, ordered fields, types and codec are all real. The
producer methods come from the pipeline tests' actual source loader.
"""

from __future__ import annotations

import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import msgspec
import pytest

TEST_DIR = Path(__file__).resolve().parent
ROOT = TEST_DIR.parents[1]
CURRENT = ROOT / "python/tokenspeed/runtime/engine/io_struct.py"
PRODUCER = runpy.run_path(str(TEST_DIR / "test_prompt_top_logprobs_pipeline.py"))


def _load_wire(path, module_name, legacy):
    module = ModuleType(module_name)
    sys.modules[module_name] = module
    module.__dict__.update(msgspec=msgspec, Any=Any, FinishReasonDict=dict)
    names = {"BaseBatchReq", "BatchTokenIDOut", "BatchStrOut"}
    tree = ast.parse(path.read_text())
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    nodes.extend(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names
    )
    if legacy:
        # Model the two legacy flat input annotations explicitly, while the
        # independent field/type contract below pins the remaining IPC schema.
        for cls in nodes:
            if isinstance(cls, ast.ClassDef):
                for field in cls.body:
                    if isinstance(field, ast.AnnAssign):
                        old_type = {
                            "input_token_logprobs_val": "list[float]",
                            "input_token_logprobs_idx": "list[int]",
                        }.get(field.target.id)
                        if old_type is not None:
                            field.annotation = ast.parse(old_type, mode="eval").body
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(path),
            "exec",
        ),
        module.__dict__,
    )
    return module


OLD = _load_wire(CURRENT, "_prompt_wire_legacy_contract", True)
NEW = _load_wire(CURRENT, "_prompt_wire_current", False)


def _payload(cls, overrides):
    values = {field.name: [] for field in msgspec.structs.fields(cls)}
    values["generated_time"] = 0.0
    values.update(overrides)
    return cls(**values)


def _roundtrip(frame):
    data = msgspec.msgpack.encode(frame)
    decoder = msgspec.msgpack.Decoder(NEW.BatchTokenIDOut | NEW.BatchStrOut)
    return decoder.decode(data)


def test_reproduce_original_warmup_failure_at_wire_field_15():
    frame = _payload(
        OLD.BatchTokenIDOut, {"rids": ["warmup"], "input_token_logprobs_val": [[]]}
    )
    wire = msgspec.msgpack.encode(frame)
    raw = msgspec.msgpack.decode(wire)
    assert raw[15] == [[]]
    with pytest.raises(
        msgspec.ValidationError, match=r"Expected `float`, got `array`.*\$\[15\]\[0\]"
    ):
        msgspec.msgpack.Decoder(OLD.BatchTokenIDOut).decode(wire)


@pytest.mark.parametrize("class_name", ["BatchTokenIDOut", "BatchStrOut"])
def test_actual_input_values_and_first_none_roundtrip_without_flattening(class_name):
    cls = getattr(NEW, class_name)
    frame = _payload(
        cls,
        {
            "rids": ["ordinary", "input", "partial"],
            "input_token_logprobs_val": [[], [None, -0.25, -0.5], [-1.5]],
            "input_token_logprobs_idx": [[], [11, 12, 13], [21]],
            "input_top_logprobs_val": [[], [None, [-0.25, -1.0], [-0.5, -0.7]], []],
            "input_top_logprobs_idx": [[], [None, [12, 4], [13, 5]], []],
        },
    )
    out = _roundtrip(frame)
    assert out == frame
    assert out.input_token_logprobs_val[1] == [None, -0.25, -0.5]
    assert out.input_token_logprobs_idx[2] == [21]


@pytest.mark.parametrize("logprob", [False, True])
def test_real_ordinary_producer_wire_retains_original_empty_input_payload(logprob):
    state = PRODUCER["_state"](3, -1, 0, False, 1)
    state.return_logprob = logprob
    if logprob:
        state.output_token_logprobs_val = []
        state.output_token_logprobs_idx = []
    processor, sender = PRODUCER["_processor"](state)
    old_type = PRODUCER["CLASSES"]["BatchTokenIDOut"]
    PRODUCER["CLASSES"]["BatchTokenIDOut"] = NEW.BatchTokenIDOut
    try:
        PRODUCER["_step"](processor, state, 0, 3, 7, False)
    finally:
        PRODUCER["CLASSES"]["BatchTokenIDOut"] = old_type
    frame = sender.items[0]
    assert isinstance(frame, NEW.BatchTokenIDOut)
    assert frame.input_token_logprobs_val == frame.input_token_logprobs_idx == []
    assert frame.input_top_logprobs_val == frame.output_top_logprobs_val == []
    decoded = _roundtrip(frame)
    legacy = msgspec.msgpack.Decoder(OLD.BatchTokenIDOut).decode(
        msgspec.msgpack.encode(frame)
    )
    assert decoded.output_ids == legacy.output_ids == [[7]]
    assert decoded.output_token_logprobs_val == ([[-0.5]] if logprob else [[]])


@pytest.mark.parametrize("stream", [False, True])
def test_real_chunked_diagnostic_producer_wire_keeps_all_four_fields(stream):
    state = PRODUCER["_state"](5, 0, 2, stream, 2)
    processor, sender = PRODUCER["_processor"](state)
    old_type = PRODUCER["CLASSES"]["BatchTokenIDOut"]
    PRODUCER["CLASSES"]["BatchTokenIDOut"] = NEW.BatchTokenIDOut
    try:
        PRODUCER["_step"](processor, state, 0, 2, 99, False)
        PRODUCER["_step"](processor, state, 2, 3, 7, False)
        PRODUCER["_step"](processor, state, 5, 1, 8, True)
    finally:
        PRODUCER["CLASSES"]["BatchTokenIDOut"] = old_type
    first, last = [_roundtrip(item) for item in sender.items]
    assert first.input_token_logprobs_val == [[None, 0.0, -1.0, -2.0, -3.0]]
    assert first.input_token_logprobs_idx == [[0, 1, 2, 3, 4]]
    assert len(first.input_top_logprobs_val[0]) == 5
    assert len(first.output_top_logprobs_val[0]) == 1
    assert first.output_token_logprobs_val == [[-0.5]]
    assert last.input_token_logprobs_val == [[]]
    assert last.output_token_logprobs_idx == [[8]]


EXPECTED_LEGACY_ANNOTATIONS = {
    "BaseBatchReq": {"rids": "list[str] | None"},
    "BatchTokenIDOut": {
        "finished_reasons": "list[FinishReasonDict | None]",
        "decoded_texts": "list[str]",
        "decode_ids": "list[list[int]]",
        "read_offsets": "list[int]",
        "output_ids": "list[list[int]] | None",
        "output_multi_ids": "list[list[int]] | None",
        "skip_special_tokens": "list[bool]",
        "spaces_between_special_tokens": "list[bool]",
        "no_stop_trim": "list[bool]",
        "prompt_tokens": "list[int]",
        "completion_tokens": "list[int]",
        "cached_tokens": "list[int]",
        "spec_verify_ct": "list[int]",
        "input_token_logprobs_val": "list[float]",
        "input_token_logprobs_idx": "list[int]",
        "output_token_logprobs_val": "list[list[float]]",
        "output_token_logprobs_idx": "list[list[int]]",
        "input_top_logprobs_val": "list[list]",
        "input_top_logprobs_idx": "list[list]",
        "output_top_logprobs_val": "list[list]",
        "output_top_logprobs_idx": "list[list]",
        "input_token_ids_logprobs_val": "list[list]",
        "input_token_ids_logprobs_idx": "list[list]",
        "output_token_ids_logprobs_val": "list[list]",
        "output_token_ids_logprobs_idx": "list[list]",
        "output_hidden_states": "list[list[float]]",
        "batch_accept_draft_tokens": "list[float | None]",
        "output_extra_infos": "list[dict[str, Any]]",
        "generated_time": "float",
    },
    "BatchStrOut": {
        "finished_reasons": "list[FinishReasonDict | None]",
        "output_strs": "list[str]",
        "output_ids": "list[int] | None",
        "prompt_tokens": "list[int]",
        "completion_tokens": "list[int]",
        "cached_tokens": "list[int]",
        "spec_verify_ct": "list[int]",
        "input_token_logprobs_val": "list[float]",
        "input_token_logprobs_idx": "list[int]",
        "output_token_logprobs_val": "list[float]",
        "output_token_logprobs_idx": "list[int]",
        "input_top_logprobs_val": "list[list]",
        "input_top_logprobs_idx": "list[list]",
        "output_top_logprobs_val": "list[list]",
        "output_top_logprobs_idx": "list[list]",
        "input_token_ids_logprobs_val": "list[list]",
        "input_token_ids_logprobs_idx": "list[list]",
        "output_token_ids_logprobs_val": "list[list]",
        "output_token_ids_logprobs_idx": "list[list]",
        "output_hidden_states": "list[list[float]]",
        "batch_accept_draft_tokens": "list[float | None]",
        "output_extra_infos": "list[dict[str, Any]]",
        "generated_time": "float",
    },
}


def test_wire_field_order_and_non_input_annotations_are_unchanged():
    tree = ast.parse(CURRENT.read_text())
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in EXPECTED_LEGACY_ANNOTATIONS:
            actual = {
                field.target.id: ast.unparse(field.annotation)
                for field in cls.body
                if isinstance(field, ast.AnnAssign)
            }
            expected = EXPECTED_LEGACY_ANNOTATIONS[cls.name]
            assert list(actual) == list(expected)
            for name, annotation in actual.items():
                if name not in ("input_token_logprobs_val", "input_token_logprobs_idx"):
                    assert annotation == expected[name]
    for class_name in ("BatchTokenIDOut", "BatchStrOut"):
        old = msgspec.structs.fields(getattr(OLD, class_name))
        new = msgspec.structs.fields(getattr(NEW, class_name))
        assert [f.name for f in new] == [f.name for f in old]
        for previous, current in zip(old, new):
            if current.name not in (
                "input_token_logprobs_val",
                "input_token_logprobs_idx",
            ):
                assert previous.type == current.type
