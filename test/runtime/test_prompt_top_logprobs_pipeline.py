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

"""CPU checks of the actual opt-in request/commit/render methods.

AST loading avoids importing GPU-only kernel packages on CPU development
hosts. Method bodies come directly from the production files, not copies.
Only peripheral transports, finish records and the disabled stats sink are
stubbed; all Top-K indexing, buffering, emitting and merging code is real.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import logging
import runpy
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ENGINE = Path(__file__).resolve().parents[2] / "python/tokenspeed/runtime/engine"


class _Stats:
    def mark_prefill_done(self, timestamp):
        pass

    def record_decode_step(self, duration, prefilling_others):
        pass

    def mark_first_token(self, timestamp):
        pass


class _Finish:
    def __init__(self, **values):
        self.values = values

    def to_json(self):
        return self.values


class _Params(SimpleNamespace):
    def resolve_seed(self, rid):
        pass

    def normalize(self, tokenizer):
        pass

    def verify(self, vocab_size):
        pass

    @property
    def logprobs(self):
        return self.__dict__.get("_logprobs")


class _Input(SimpleNamespace):
    pass


def _load_classes(filename, names, namespace):
    tree = ast.parse((ENGINE / filename).read_text())
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    body.extend(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    )
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(ENGINE / filename), "exec"), namespace)
    return namespace


NOOP = _Stats()
CLASSES = _load_classes(
    "generation_output_processor.py",
    {"RequestState", "OutputProcesser"},
    {
        "NOOP_STATS": NOOP,
        "time": time,
        "logger": logging.getLogger(__name__),
        "INIT_INCREMENTAL_DETOKENIZATION_OFFSET": 5,
        "DEFAULT_FORCE_STREAM_INTERVAL": 50,
        "FINISH_LENGTH": _Finish,
        "FINISH_MATCHED_TOKEN": _Finish,
        "FINISH_MATCHED_STR": _Finish,
        "FINISH_ABORT": _Finish,
        "ABORT_CODE": SimpleNamespace(NumericalError="numerical"),
        "BatchTokenIDOut": SimpleNamespace,
        "make_extend_result_event": lambda *args: ("extend", args),
        "make_finish_event": lambda *args: ("finish", args),
        "make_abort_event": lambda *args: ("abort", args),
        "make_update_reserve_tokens_event": lambda *args: ("reserve", args),
    },
)
RequestState = CLASSES["RequestState"]
OutputProcesser = CLASSES["OutputProcesser"]
InputProcessor = _load_classes(
    "input_processor.py",
    {"InputProcessor"},
    {
        "time": time,
        "GenerateReqInput": _Input,
        "TokenizedGenerateReqInput": SimpleNamespace,
        "SamplingParams": _Params,
    },
)["InputProcessor"]
LogprobsProcessor = runpy.run_path(str(ENGINE / "logprobs.py"))["LogprobsProcessor"]
RequestOutputCollector = runpy.run_path(str(ENGINE / "collector.py"))[
    "RequestOutputCollector"
]


def _engine():
    return SimpleNamespace(
        server_args=SimpleNamespace(
            enable_output_logprobs=True,
            enable_input_logprobs=True,
            enable_logprob_graph=False,
            device="cuda",
            disable_prefill_graph=False,
            enforce_eager=True,
            disable_overlap_schedule=True,
            enable_prefix_caching=False,
            disable_kvstore=True,
            speculative_algorithm=None,
            pipeline_parallel_size=1,
            mapping=SimpleNamespace(
                attn=SimpleNamespace(dp_size=1, cp_size=1),
                dense=SimpleNamespace(has_dp=False),
            ),
            dp_sampling=False,
            disaggregation_mode="null",
            reasoning_parser=None,
        ),
        model_config=SimpleNamespace(vocab_size=32, is_multimodal=False),
        tokenizer=None,
        is_generation=True,
        max_req_input_len=100,
        context_len=128,
        logger=logging.getLogger(__name__),
    )


def _request(start, count):
    return _Input(
        return_logprob=True,
        logprob_start_len=start,
        top_logprobs_num=count,
        token_ids_logprob=None,
        logprob_format=None,
        input_ids=[7, 8, 9, 10, 11],
        text=None,
        input_embeds=None,
        precomputed_multimodal_inputs=None,
        session_params=None,
        sampling_params={"max_new_tokens": 2},
        rid="r",
        stream=False,
        bootstrap_host=None,
        bootstrap_port=None,
        bootstrap_room=None,
        data_parallel_rank=None,
        custom_logit_processor=None,
        return_hidden_states=False,
        input_multi_ids=None,
        input_extra_infos=None,
    )


def _state(length, start, count, stream, maximum):
    params = SimpleNamespace(
        max_new_tokens=maximum,
        ignore_eos=True,
        stop_token_ids=set(),
        stop_strs=[],
        stop_str_max_len=0,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        no_stop_trim=False,
        stream_interval=1,
    )
    return RequestState.from_recv_req(
        SimpleNamespace(
            input_ids=list(range(length)),
            sampling_params=params,
            stream=stream,
            return_logprob=count > 0 or start >= 0,
            top_logprobs_num=count,
            logprob_start_len=start,
        ),
        tokenizer=SimpleNamespace(eos_token_id=2, additional_stop_token_ids=None),
        eos_token_ids=[2],
    )


def _chunk(start, length, count):
    ids = torch.arange(start, start + length, dtype=torch.int64)[:, None] * 10
    ids = ids + torch.arange(count, dtype=torch.int64)[None, :]
    return -ids.to(torch.float32) / 10, ids


def _processor(state):
    sender = SimpleNamespace(items=[])
    sender.send_pyobj = sender.items.append
    processor = OutputProcesser(sender, metrics=SimpleNamespace(enabled=False))
    processor.rid_to_state["r"] = state
    return processor, sender


def _step(processor, state, prefix, count, token, decode):
    values, indices = _chunk(prefix, count, 3)
    forward = SimpleNamespace(
        request_ids=["r"],
        input_lengths=[count],
        extend_prefix_lens=[] if decode else [prefix],
        extend_replay_lens=[] if decode else [0],
        prefill_lengths=[] if decode else [state.input_length],
        num_extends=lambda: 0 if decode else 1,
    )
    result = SimpleNamespace(
        grammar_completion=None,
        output_logprobs=torch.tensor([-0.5]),
        output_nan_flags=None,
        output_lengths=torch.tensor([1], dtype=torch.int32),
        output_tokens=torch.tensor([token], dtype=torch.int32),
        next_input_ids=None,
        input_token_logprobs=[
            None if decode else torch.arange(prefix, prefix + count).float().neg()
        ],
        input_top_logprobs_val=[None if decode else values],
        input_top_logprobs_idx=[None if decode else indices],
        output_top_logprobs_val=torch.tensor([[[-0.1, -0.2, -0.3]]]),
        output_top_logprobs_idx=torch.tensor([[[20, 21, 22]]]),
    )
    return processor.post_process_forward_op(forward, result, is_prefill_instance=False)


def test_input_zero_survives_real_tokenization_and_request_state():
    req = asyncio.run(InputProcessor(_engine()).tokenize_one_request(_request(0, 3)))
    assert (req.logprob_start_len, req.top_logprobs_num) == (0, 3)
    state = RequestState.from_recv_req(req, tokenizer=None, eos_token_ids=[])
    assert (state.logprob_start_len, state.top_logprobs_num) == (0, 3)


def test_input_logprob_flag_defaults_off_and_cli_enables_it():
    source = ENGINE.parent / "utils/server_args.py"
    tree = ast.parse(source.read_text())
    fields = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "enable_input_logprobs"
    ]
    assert len(fields) == 1 and ast.literal_eval(fields[0].value) is False
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--enable-input-logprobs"
    ]
    assert len(calls) == 1
    parser = argparse.ArgumentParser()
    module = ast.fix_missing_locations(
        ast.Module(body=[ast.Expr(value=calls[0])], type_ignores=[])
    )
    exec(
        compile(module, str(source), "exec"),
        {"parser": parser, "ServerArgs": SimpleNamespace(enable_input_logprobs=False)},
    )
    assert parser.parse_args([]).enable_input_logprobs is False
    assert parser.parse_args(["--enable-input-logprobs"]).enable_input_logprobs is True


def test_input_gate_does_not_disable_output_only_topk():
    engine = _engine()
    engine.server_args.enable_input_logprobs = False
    with pytest.raises(ValueError, match="--enable-input-logprobs"):
        asyncio.run(InputProcessor(engine).tokenize_one_request(_request(0, 2)))
    output_only = asyncio.run(
        InputProcessor(engine).tokenize_one_request(_request(-1, 2))
    )
    assert output_only.return_logprob
    assert (output_only.logprob_start_len, output_only.top_logprobs_num) == (-1, 2)


@pytest.mark.parametrize(
    "start,count",
    [(-2, 3), (5, 3), (True, 3), (1.5, 3), (0, -1), (0, 33), (0, True), (0, 1.5)],
)
def test_invalid_topk_request_fails(start, count):
    with pytest.raises(ValueError):
        InputProcessor(_engine())._validate_top_logprobs_request(
            _request(start, count), 5
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("speculative_algorithm", "MTP"),
        ("pipeline_parallel_size", 2),
        ("dp_sampling", True),
    ],
)
def test_diagnostic_configuration_is_explicit(field, value):
    engine = _engine()
    setattr(engine.server_args, field, value)
    with pytest.raises(
        ValueError,
        match="Top-K diagnostics currently require|does not support speculative",
    ) as error:
        InputProcessor(engine)._validate_top_logprobs_request(_request(0, 2), 5)
    if field == "speculative_algorithm":
        with pytest.raises(ValueError, match="speculative"):
            InputProcessor(engine)._validate_top_logprobs_request(_request(-1, 0), 5)
    else:
        assert InputProcessor(engine)._validate_top_logprobs_request(
            _request(-1, 0), 5
        ) == (-1, 0)


@pytest.mark.parametrize("which", ["attn_dp", "attn_cp", "dense_dp"])
def test_distributed_diagnostic_gates(which):
    engine = _engine()
    if which == "dense_dp":
        engine.server_args.mapping.dense.has_dp = True
    else:
        setattr(
            engine.server_args.mapping.attn,
            "dp_size" if which == "attn_dp" else "cp_size",
            2,
        )
    if which == "attn_cp":
        with pytest.raises(ValueError):
            InputProcessor(engine)._validate_top_logprobs_request(_request(0, 2), 5)
    else:
        assert InputProcessor(engine)._validate_top_logprobs_request(
            _request(0, 2), 5
        ) == (0, 2)


def test_enable_output_logprobs_gate_and_ordinary_requests_unchanged():
    engine = _engine()
    engine.server_args.enable_output_logprobs = False
    with pytest.raises(ValueError, match="enable_output_logprobs"):
        asyncio.run(InputProcessor(engine).tokenize_one_request(_request(0, 2)))
    ordinary = _request(None, None)
    ordinary.return_logprob = False
    engine.server_args.enforce_eager = False
    req = asyncio.run(InputProcessor(engine).tokenize_one_request(ordinary))
    assert not req.return_logprob
    assert (req.top_logprobs_num, req.logprob_start_len) == (0, -1)


def test_sampled_output_only_keeps_existing_non_eager_path():
    engine = _engine()
    engine.server_args.enforce_eager = False
    engine.server_args.disable_overlap_schedule = False
    engine.server_args.enable_prefix_caching = True
    engine.server_args.disable_kvstore = False
    req = asyncio.run(InputProcessor(engine).tokenize_one_request(_request(-1, 0)))
    assert req.return_logprob
    assert (req.top_logprobs_num, req.logprob_start_len) == (0, -1)


@pytest.mark.parametrize("start,count", [(0, 0), (0, 2), (-1, 2)])
@pytest.mark.parametrize("graph", [False, True])
def test_new_diagnostic_stream_is_rejected_at_tokenization(start, count, graph):
    engine = _engine()
    if graph:
        engine.server_args.enable_logprob_graph = True
        engine.server_args.enforce_eager = False
        engine.server_args.disable_prefill_graph = True
    obj = _request(start, count)
    obj.stream = True
    with pytest.raises(ValueError, match="require stream=False"):
        asyncio.run(InputProcessor(engine).tokenize_one_request(obj))


@pytest.mark.parametrize("sampled", [False, True])
def test_existing_ordinary_and_sampled_output_stream_are_not_gated(sampled):
    engine = _engine()
    engine.server_args.enforce_eager = False
    engine.server_args.disable_overlap_schedule = False
    obj = _request(-1, 0)
    obj.return_logprob = sampled
    obj.stream = True
    req = asyncio.run(InputProcessor(engine).tokenize_one_request(obj))
    assert req.stream is True
    assert req.return_logprob is sampled
    assert (req.logprob_start_len, req.top_logprobs_num) == (-1, 0)


@pytest.mark.parametrize(
    "fmt,legacy", [("vllm", False), ("invalid", False), (None, True)]
)
def test_render_dialect_cannot_silently_discard_topk(fmt, legacy):
    obj = _request(0, 2)
    obj.logprob_format = fmt
    if legacy:
        obj.sampling_params["logprobs"] = 0
    with pytest.raises(ValueError, match="logprob_format"):
        InputProcessor(_engine())._validate_top_logprobs_request(obj, 5)


@pytest.mark.parametrize(
    "field",
    [
        "input_embeds",
        "precomputed_multimodal_inputs",
        "image_data",
        "video_data",
        "audio_data",
    ],
)
def test_diagnostic_rejects_unhandled_embedding_positions(field):
    obj = _request(0, 2)
    engine = _engine()
    if field == "is_multimodal":
        engine.model_config.is_multimodal = True
    else:
        setattr(obj, field, [1])
    with pytest.raises(ValueError, match="text/token-ID"):
        InputProcessor(engine)._validate_top_logprobs_request(obj, 5)


@pytest.mark.parametrize("start", [0, 1, 2, 3, 4])
def test_chunk_source_rows_shift_once_and_last_prompt_row_is_output(start):
    state = _state(5, start, 2, False, 2)
    for prefix, count in [(0, 2), (2, 2), (4, 1)]:
        values, indices = _chunk(prefix, count, 3)
        state.append_prompt_top_logprobs(values, indices, prefix, count)
    expected = [None] + [[10 * source, 10 * source + 1] for source in range(4)]
    assert state.input_top_logprobs_idx == expected[start:]
    assert len(state.input_top_logprobs_val) == 5 - start


def test_retract_duplicate_sources_do_not_duplicate_prompt_positions():
    state = _state(5, 0, 2, False, 2)
    first = _chunk(0, 2, 2)
    state.append_prompt_top_logprobs(*first, 0, 2)
    state.append_prompt_top_logprobs(*first, 0, 2)
    state.append_prompt_top_logprobs(*_chunk(2, 3, 2), 2, 3)
    assert state.input_top_logprobs_idx == [None, [0, 1], [10, 11], [20, 21], [30, 31]]


def test_prompt_gap_and_missing_tensor_fail_loudly():
    state = _state(5, 0, 2, False, 2)
    with pytest.raises(RuntimeError, match="missing"):
        state.append_prompt_top_logprobs(None, None, 0, 2)
    with pytest.raises(RuntimeError, match="position gap"):
        state.append_prompt_top_logprobs(*_chunk(2, 2, 2), 2, 2)


@pytest.mark.parametrize(
    "kind",
    ["wrong_length", "wrong_rank", "different_shapes", "too_few_candidates", "device"],
)
def test_prompt_tensor_boundary_is_validated_before_mutation(kind):
    state = _state(5, 0, 2, False, 2)
    values, indices = _chunk(0, 2, 2)
    if kind == "wrong_length":
        values, indices = values[:1], indices[:1]
    elif kind == "wrong_rank":
        values, indices = values.flatten(), indices.flatten()
    elif kind == "different_shapes":
        indices = indices[:1]
    elif kind == "too_few_candidates":
        values, indices = values[:, :1], indices[:, :1]
    else:
        values = torch.empty((2, 2), device="meta")
    with pytest.raises(RuntimeError):
        state.append_prompt_top_logprobs(values, indices, 0, 2)
    assert not state.input_top_logprobs_val


@pytest.mark.parametrize("stream", [False])
def test_commit_captures_midchunk_then_emits_prompt_once_and_output_deltas(stream):
    state = _state(5, 0, 2, stream, 2)
    processor, sender = _processor(state)
    _step(processor, state, 0, 2, 99, False)
    _step(processor, state, 2, 2, 99, False)
    assert not sender.items and not state.output_ids
    _step(processor, state, 4, 1, 77, False)
    assert sender.items[0].input_top_logprobs_idx[0] == [
        None,
        [0, 1],
        [10, 11],
        [20, 21],
        [30, 31],
    ]
    assert sender.items[0].output_top_logprobs_idx == [[[20, 21]]]
    assert sender.items[0].output_ids == [[77]]
    _step(processor, state, 5, 1, 78, True)
    assert sender.items[1].input_top_logprobs_idx == [[]]
    assert sender.items[1].output_top_logprobs_idx == [[[20, 21]]]
    assert sender.items[1].output_ids == [[78]]
    assert state.output_ids == [77, 78]
    assert "r" not in processor.rid_to_state
    _step(processor, state, 6, 1, 79, True)
    processor.stream_output(["r"], [state])
    assert len(sender.items) == 2


def test_one_token_prompt_has_single_none_and_eos_distribution_is_kept():
    state = _state(1, 0, 1, True, 8)
    state.sampling_params.ignore_eos = False
    processor, sender = _processor(state)
    _step(processor, state, 0, 1, 2, False)
    assert state.finished
    assert sender.items[0].input_top_logprobs_idx == [[None]]
    assert sender.items[0].output_ids == [[2]]
    assert sender.items[0].output_top_logprobs_idx == [[[20]]]


def test_mixed_request_flags_and_candidate_counts_keep_wire_rows_aligned():
    states = [
        _state(2, -1, 0, True, 1),
        _state(3, 1, 2, True, 1),
        _state(1, -1, 1, True, 1),
    ]
    processor, sender = _processor(states[0])
    processor.rid_to_state = dict(zip(["a", "b", "c"], states))
    values, indices = _chunk(0, 3, 3)
    forward = SimpleNamespace(
        request_ids=["a", "b", "c"],
        input_lengths=[2, 3, 1],
        extend_prefix_lens=[0, 0, 0],
        extend_replay_lens=[0, 0, 0],
        prefill_lengths=[2, 3, 1],
        num_extends=lambda: 3,
    )
    result = SimpleNamespace(
        grammar_completion=None,
        output_logprobs=torch.tensor([-0.5, -0.6, -0.7]),
        output_nan_flags=None,
        next_input_ids=None,
        output_lengths=torch.tensor([1, 1, 1]),
        output_tokens=torch.tensor([81, 82, 83]),
        input_token_logprobs=[None, torch.tensor([-1.0, -2.0, -3.0]), None],
        input_top_logprobs_val=[None, values, None],
        input_top_logprobs_idx=[None, indices, None],
        output_top_logprobs_val=torch.tensor([[[-0.1, -0.2, -0.3]]] * 3),
        output_top_logprobs_idx=torch.tensor([[[20, 21, 22]]] * 3),
    )
    processor.post_process_forward_op(forward, result, is_prefill_instance=False)
    frame = sender.items[0]
    assert frame.rids == ["a", "b", "c"]
    assert frame.input_top_logprobs_idx == [[], [[0, 1], [10, 11]], []]
    assert frame.output_top_logprobs_idx == [[], [[20, 21]], [[20]]]
    assert frame.output_token_logprobs_idx == [[], [82], [83]]
    assert not processor.rid_to_state


def test_abort_before_prefill_returns_no_fabricated_prompt_positions():
    state = _state(5, 0, 2, True, 2)
    processor, sender = _processor(state)
    state.set_finish_with_abort("cancelled", notify_client=True)
    processor.stream_output(["r"], [state])
    assert sender.items[0].input_top_logprobs_idx == [[]]
    assert sender.items[0].output_top_logprobs_idx == [[]]


def test_missing_output_topk_is_not_silently_returned():
    state = _state(1, -1, 1, False, 2)
    processor, sender = _processor(state)
    state.output_ids.append(5)
    with pytest.raises(RuntimeError, match="positions"):
        processor.stream_output(["r"], [state])
    assert not sender.items


def test_incomplete_prompt_does_not_emit_successful_topk_frame():
    state = _state(3, 0, 2, True, 1)
    processor, sender = _processor(state)
    state.append_prompt_top_logprobs(*_chunk(0, 1, 2), 0, 1)
    state.computed_length = 3
    state.output_ids.append(77)
    state.output_top_logprobs_val.append([-0.1, -0.2])
    state.output_top_logprobs_idx.append([20, 21])
    with pytest.raises(RuntimeError, match="incomplete"):
        processor.stream_output(["r"], [state])
    assert not sender.items


@pytest.mark.parametrize("stream", [False])
def test_real_renderer_and_collector_keep_repeated_topk_and_prompt_none(stream):
    state = _state(2, 0, 2, stream, 2)
    processor, sender = _processor(state)
    _step(processor, state, 0, 2, 77, False)
    _step(processor, state, 2, 1, 78, True)
    renderer = LogprobsProcessor(SimpleNamespace(tokenizer=None))
    collector = RequestOutputCollector()
    accumulated = {}
    for frame in sender.items:
        info = {} if stream else accumulated
        renderer.convert_logprob_style(info, "sglang", 2, None, False, frame, 0)
        collector.put(
            {"output_ids": frame.output_ids[0], "meta_info": info}, stream=stream
        )
    result = collector.take()["meta_info"]
    assert len(result["input_top_logprobs"]) == 2
    assert result["input_top_logprobs"][0] is None
    assert len(result["output_top_logprobs"]) == 2
    assert result["output_top_logprobs"][0] == result["output_top_logprobs"][1]
    assert [x[1] for x in result["output_token_logprobs"]] == [77, 78]


def test_nonstream_collector_uses_latest_complete_response_without_dedup():
    collector = RequestOutputCollector()
    first = {
        "output_ids": [77],
        "meta_info": {"output_top_logprobs": [[(-0.5, 7, None)]]},
    }
    second = {
        "output_ids": [77, 77],
        "meta_info": {"output_top_logprobs": [[(-0.5, 7, None)], [(-0.5, 7, None)]]},
    }
    collector.put(first, stream=False)
    collector.put(second, stream=False)
    result = collector.take()
    assert result is second and result["output_ids"] == [77, 77]
    assert len(result["meta_info"]["output_top_logprobs"]) == 2
    assert len(first["meta_info"]["output_top_logprobs"]) == 1
    assert len(second["meta_info"]["output_top_logprobs"]) == 2


def test_actual_nonstream_frontend_keeps_repeated_token_ids_and_all_scores():
    output_processor = _load_classes(
        "output_processor.py",
        {"OutputProcessor"},
        {
            "logger": logging.getLogger(__name__),
            "time": time,
            "LogprobsProcessor": LogprobsProcessor,
            "BatchTokenIDOut": SimpleNamespace,
            "BatchStrOut": type("UnusedStringFrame", (), {}),
            "BatchEmbeddingOut": type("UnusedEmbeddingFrame", (), {}),
        },
    )["OutputProcessor"]
    state = _state(2, 0, 2, False, 2)
    processor, sender = _processor(state)
    _step(processor, state, 0, 2, 77, False)
    _step(processor, state, 2, 1, 77, True)
    frontend_state = SimpleNamespace(
        obj=_request(0, 2),
        collector=RequestOutputCollector(),
        logprobs_info={},
        output_ids=[],
        text="",
        last_output_offset=0,
        finished=False,
        event=SimpleNamespace(set=lambda: None),
        created_time=time.time(),
    )
    frontend = output_processor(
        SimpleNamespace(
            tokenizer=None,
            rid_to_state={"r": frontend_state},
            server_args=SimpleNamespace(
                weight_version=0,
                enable_inline_detokenizer=False,
                stream_output=True,
                speculative_algorithm=None,
            ),
            enable_metrics=False,
            dump_requests_folder=None,
        )
    )
    for frame in sender.items:
        frontend.handle_batch_output(frame)
    result = frontend_state.collector.take()
    assert result["output_ids"] == [77, 77]
    assert [row[1] for row in result["meta_info"]["output_token_logprobs"]] == [77, 77]
    assert len(result["meta_info"]["output_top_logprobs"]) == 2
    assert result["meta_info"]["input_token_logprobs"] == [
        (None, 0, None),
        (0.0, 1, None),
    ]
    assert len(result["meta_info"]["input_top_logprobs"]) == 2


@pytest.mark.parametrize("start", [0, 1, 2, 3, 4])
def test_actual_prompt_token_scores_align_across_chunks(start):
    state = _state(5, start, 0, False, 1)
    for prefix, length in [(0, 2), (2, 2), (4, 1)]:
        values = -torch.arange(prefix + 1, prefix + length + 1).float()
        state.append_prompt_token_logprobs(values, prefix, length)
        state.append_prompt_token_logprobs(values, prefix, length)
    assert state.input_token_logprobs_val == [None, -1.0, -2.0, -3.0, -4.0][start:]
    assert state.input_token_logprobs_idx == list(range(5))[start:]


def test_sampled_input_without_topk_is_supported_and_gated():
    engine = _engine()
    req = asyncio.run(InputProcessor(engine).tokenize_one_request(_request(0, 0)))
    assert (req.logprob_start_len, req.top_logprobs_num) == (0, 0)
    engine.server_args.enforce_eager = False
    req = asyncio.run(InputProcessor(engine).tokenize_one_request(_request(0, 0)))
    assert (req.logprob_start_len, req.top_logprobs_num) == (0, 0)


@pytest.mark.parametrize("stream", [False])
def test_all_four_fields_render_real_input_scores_and_output_deltas(stream):
    state = _state(5, 0, 2, stream, 2)
    processor, sender = _processor(state)
    _step(processor, state, 0, 2, 99, False)
    _step(processor, state, 2, 3, 77, False)
    _step(processor, state, 5, 1, 78, True)
    assert sender.items[0].input_token_logprobs_val == [[None, 0.0, -1.0, -2.0, -3.0]]
    assert sender.items[0].input_token_logprobs_idx == [[0, 1, 2, 3, 4]]
    assert sender.items[1].input_token_logprobs_val == [[]]
    renderer = LogprobsProcessor(SimpleNamespace(tokenizer=None))
    collector = RequestOutputCollector()
    accumulated = {}
    for frame in sender.items:
        info = {} if stream else accumulated
        renderer.convert_logprob_style(info, "sglang", 2, None, False, frame, 0)
        collector.put(
            {"output_ids": frame.output_ids[0], "meta_info": info}, stream=stream
        )
    meta = collector.take()["meta_info"]
    assert meta["input_token_logprobs"] == [
        (None, 0, None),
        (0.0, 1, None),
        (-1.0, 2, None),
        (-2.0, 3, None),
        (-3.0, 4, None),
    ]
    assert len(meta["input_top_logprobs"]) == 5
    assert len(meta["output_top_logprobs"]) == len(meta["output_token_logprobs"]) == 2


@pytest.mark.parametrize(
    "values", [None, torch.ones(2, 1), torch.ones(1), torch.empty(2, device="meta")]
)
def test_input_sampled_boundary_rejects_missing_shape_and_device(values):
    state = _state(3, 0, 0, False, 1)
    with pytest.raises(RuntimeError):
        state.append_prompt_token_logprobs(values, 0, 2)
    assert not state.input_token_logprobs_val


def test_k0_sampled_prompt_emits_without_fake_topk():
    state = _state(3, 1, 0, False, 1)
    processor, sender = _processor(state)
    _step(processor, state, 0, 3, 77, False)
    frame = sender.items[0]
    assert frame.input_token_logprobs_idx == [[1, 2]]
    assert frame.input_token_logprobs_val == [[0.0, -1.0]]
    assert frame.input_top_logprobs_val == frame.output_top_logprobs_val == [[]]


def test_logprob_graph_flag_defaults_off_and_cli_enables_it():
    source = ENGINE.parent / "utils/server_args.py"
    tree = ast.parse(source.read_text())
    fields = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "enable_logprob_graph"
    ]
    assert len(fields) == 1 and ast.literal_eval(fields[0].value) is False
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "add_argument"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "--enable-logprob-graph"
    ]
    assert len(calls) == 1
    parser = argparse.ArgumentParser()
    module = ast.fix_missing_locations(
        ast.Module(body=[ast.Expr(value=calls[0])], type_ignores=[])
    )
    exec(
        compile(module, str(source), "exec"),
        {"parser": parser, "ServerArgs": SimpleNamespace(enable_logprob_graph=False)},
    )
    assert parser.parse_args([]).enable_logprob_graph is False
    assert parser.parse_args(["--enable-logprob-graph"]).enable_logprob_graph is True


@pytest.mark.parametrize("start,k", [(0, 2), (0, 0), (-1, 2)])
def test_explicit_cuda_decode_graph_mode_permits_diagnostics(start, k):
    engine = _engine()
    engine.server_args.enable_logprob_graph = True
    engine.server_args.enforce_eager = False
    engine.server_args.disable_prefill_graph = True
    req = asyncio.run(InputProcessor(engine).tokenize_one_request(_request(start, k)))
    assert (req.logprob_start_len, req.top_logprobs_num) == (start, k)


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "npu"),
        ("device", "cpu"),
        ("dp_sampling", True),
    ],
)
def test_graph_opt_in_preserves_other_gates_and_cannot_silently_fall_back(field, value):
    engine = _engine()
    engine.server_args.enable_logprob_graph = True
    engine.server_args.enforce_eager = False
    engine.server_args.disable_prefill_graph = True
    setattr(engine.server_args, field, value)
    with pytest.raises(ValueError) as error:
        asyncio.run(InputProcessor(engine).tokenize_one_request(_request(0, 2)))
    ordinary = _request(-1, 0)
    ordinary.return_logprob = False
    assert not asyncio.run(
        InputProcessor(engine).tokenize_one_request(ordinary)
    ).return_logprob


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("start,k", [(0, 0), (0, 3), (2, 3), (-1, 3)])
def test_diagnostics_keep_overlap_prefill_graph_and_both_cache_tiers(eager, start, k):
    engine = _engine()
    engine.server_args.enforce_eager = eager
    engine.server_args.enable_logprob_graph = False
    engine.server_args.disable_prefill_graph = False
    engine.server_args.disable_overlap_schedule = False
    engine.server_args.enable_prefix_caching = True
    engine.server_args.disable_kvstore = False
    req = asyncio.run(InputProcessor(engine).tokenize_one_request(_request(start, k)))
    assert (req.logprob_start_len, req.top_logprobs_num) == (start, k)
    assert engine.server_args.enable_prefix_caching
    assert not engine.server_args.disable_overlap_schedule


@pytest.mark.parametrize(
    "return_logprob,start,reuse",
    [(False, -1, True), (True, -1, True), (True, 0, False), (True, 2, False)],
)
def test_request_admission_bypasses_cache_only_for_input_scores(
    return_logprob, start, reuse
):
    path = ENGINE / "request_handler.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RequestHandler"
    )
    method = next(
        n for n in cls.body if getattr(n, "name", None) == "handle_generate_request"
    )
    method.decorator_list = []
    state = SimpleNamespace(
        return_logprob=return_logprob,
        logprob_start_len=start,
        sampling_params=SimpleNamespace(max_new_tokens=4),
        prompt_input_ids=[1, 2, 3],
    )
    ns = {
        "make_spec": lambda **kw: SimpleNamespace(**kw),
        "RequestState": SimpleNamespace(from_recv_req=lambda *a, **kw: state),
        "BootstrapInfo": lambda *a: a,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    handler = SimpleNamespace(
        server_args=SimpleNamespace(disaggregation_bootstrap_port=0),
        tokenizer=None,
        hf_eos_token_id=None,
        max_req_len=100,
    )
    req = SimpleNamespace(
        rid="r",
        input_ids=[1, 2, 3],
        bootstrap_port=None,
        bootstrap_host=None,
        bootstrap_room=None,
        session_params=None,
    )
    spec, _, _ = ns["handle_generate_request"](handler, req)
    assert spec.reuse_prefix_cache is reuse


@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        "spec",
        "pd",
        "pp",
        "attn_dp",
        "attn_cp",
        "dense_dp",
        "dp_sampling",
        "multimodal",
    ],
)
def test_graph_capture_is_automatic_only_for_supported_logprob_execution(unsupported):
    args = _engine().server_args
    args.mapping.pp_size = 1
    model = SimpleNamespace(is_multimodal=False)
    if unsupported == "spec":
        args.speculative_algorithm = "MTP"
    elif unsupported == "pd":
        args.disaggregation_mode = "decode"
    elif unsupported == "pp":
        args.mapping.pp_size = 2
    elif unsupported in ("attn_dp", "attn_cp"):
        setattr(
            args.mapping.attn, "dp_size" if unsupported == "attn_dp" else "cp_size", 2
        )
    elif unsupported == "dense_dp":
        args.mapping.dense.has_dp = True
    elif unsupported == "dp_sampling":
        args.dp_sampling = True
    elif unsupported == "multimodal":
        model.is_multimodal = True
    path = ENGINE.parent / "execution/model_executor.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelExecutorConfig"
    )
    factory = next(
        n for n in cls.body if getattr(n, "name", None) == "from_server_args"
    )
    call = next(
        n
        for n in ast.walk(factory)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "ModelExecutorConfig"
    )
    value = next(k.value for k in call.keywords if k.arg == "enable_logprob_graph")
    enabled = eval(
        compile(ast.Expression(value), str(path), "eval"),
        {"server_args": args, "model_config": model},
    )
    assert enabled is (unsupported in (None, "pd", "attn_dp", "dense_dp", "multimodal"))


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_non_speculative_pd_logprob_requests_are_supported(role):
    engine = _engine()
    engine.server_args.disaggregation_mode = role
    assert InputProcessor(engine)._validate_top_logprobs_request(_request(0, 2), 5) == (
        0,
        2,
    )
