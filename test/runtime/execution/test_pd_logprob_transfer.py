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
"""CPU tests of PD wire, rendezvous and real output-assembly methods."""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import threading
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "python/tokenspeed/runtime"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAYLOAD = load("_pd_logprob_payload", RUNTIME / "pd/logprobs.py")
PIPE = load(
    "_pd_logprob_pipeline", ROOT / "test/runtime/test_prompt_top_logprobs_pipeline.py"
)
STATUS = load("_pd_logprob_status", RUNTIME / "pd/base/status.py").TransferPoll


def methods(path, cls, names):
    tree = ast.parse((RUNTIME / path).read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and cls is None and node.name in names:
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == cls:
            nodes.extend(
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name in names
            )
    assert len(nodes) == len(names)
    ns = {"np": np, "TransferPoll": STATUS, "logger": logging.getLogger(__name__)}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        *nodes,
                    ],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        ns,
    )
    return ns


@pytest.fixture(autouse=True)
def payload_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.pd.logprobs", PAYLOAD)


def state(start, k, maximum):
    result = PIPE._state(3, start, k, False, maximum)
    result.return_logprob = True
    return result


def prefill(start, k):
    p = state(start, k, 1)
    p.computed_length = 3
    p.output_ids = [21]
    p.output_token_logprobs_idx = [21]
    p.output_token_logprobs_val = [0.0]
    p.output_top_logprobs_idx = [[21, 20][:k]] if k else []
    p.output_top_logprobs_val = [[0.0, -100.0][:k]] if k else []
    if start >= 0:
        p.input_token_logprobs_val = [None, -0.2, 0.0][start:]
        p.input_token_logprobs_idx = list(range(3))[start:]
        p.input_top_logprobs_val = (
            [None, [-0.2, -2.0][:k], [0.0, -100.0][:k]][start:] if k else []
        )
        p.input_top_logprobs_idx = [None, [1, 7][:k], [2, 7][:k]][start:] if k else []
    return p


@pytest.mark.parametrize("start", [-1, 0, 1])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_payload_roundtrip_is_exact_and_source_independent(start, k):
    p, d = prefill(start, k), state(start, k, 8)
    wire = PAYLOAD.snapshot_prefill_logprobs(p)
    p.output_token_logprobs_val[0] = -99
    PAYLOAD.restore_prefill_logprobs(d, 21, wire)
    assert d.output_token_logprobs_val == [0.0]
    assert d.output_token_logprobs_idx == [21]
    assert d.input_token_logprobs_val == (
        [None, -0.2, 0.0][start:] if start >= 0 else []
    )
    assert (
        len(d.output_top_logprobs_val[0]) == k if k else not d.output_top_logprobs_val
    )
    if start == 0:
        assert d.input_token_logprobs_val[0] is None
        if k:
            assert d.input_top_logprobs_val[0] is None


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "malformed",
        "version",
        "prompt",
        "start",
        "k",
        "bootstrap",
        "input_length",
        "nonfinite",
    ],
)
def test_invalid_handoff_does_not_partially_mutate_decode(corruption):
    wire = PAYLOAD.snapshot_prefill_logprobs(prefill(0, 2))
    d = state(0, 2, 8)
    token = 21
    obj = msgspec.msgpack.decode(wire)
    if corruption == "missing":
        wire = None
    elif corruption == "malformed":
        wire = b"\xc1"
    elif corruption == "version":
        obj["version"] = 7
    elif corruption == "prompt":
        d.prompt_input_ids[1] = 9
    elif corruption == "start":
        obj["start"] = 1
    elif corruption == "k":
        obj["topk"] = 1
    elif corruption == "bootstrap":
        token = 99
    elif corruption == "input_length":
        obj["input_values"].pop()
    elif corruption == "nonfinite":
        obj["value"] = float("nan")
    if corruption not in ("missing", "malformed"):
        wire = msgspec.msgpack.encode(obj)
    with pytest.raises(ValueError):
        PAYLOAD.restore_prefill_logprobs(d, token, wire)
    assert d.output_token_logprobs_val == []
    assert d.input_token_logprobs_val == []


def test_partial_prefill_does_not_publish_scores():
    p = prefill(0, 2)
    p.computed_length = 2
    assert PAYLOAD.snapshot_prefill_logprobs(p) is None
    assert (
        PAYLOAD.snapshot_prefill_logprobs(SimpleNamespace(return_logprob=False)) is None
    )


def test_bootstrap_scores_precede_one_token_finish_and_are_not_duplicated():
    d = state(0, 2, 1)
    proc, sender = PIPE._processor(d)
    wire = PAYLOAD.snapshot_prefill_logprobs(prefill(0, 2))
    proc.on_remote_prefill_done("r", 21, 0, logprobs=wire)
    assert d.finished and d.output_ids == [21]
    proc.on_remote_prefill_done("r", 21, 0, logprobs=wire)
    assert d.output_ids == [21] and d.output_token_logprobs_val == [0.0]
    assert proc.finish_remote_prefill_only_request("r") == [("finish", ("r",))]
    assert len(sender.items) == 1
    packet = sender.items[0]
    assert packet.output_token_logprobs_idx == [[21]]
    assert packet.output_token_logprobs_val == [[0.0]]
    assert packet.input_token_logprobs_idx == [[0, 1, 2]]


def test_first_decode_appends_after_bootstrap_not_over_it():
    d = state(0, 2, 8)
    proc, _ = PIPE._processor(d)
    proc.on_remote_prefill_done(
        "r", 21, 0, logprobs=PAYLOAD.snapshot_prefill_logprobs(prefill(0, 2))
    )
    PIPE._step(proc, d, 3, 1, 20, True)
    assert d.output_ids == [21, 20]
    assert d.output_token_logprobs_idx == [21, 20]
    assert d.output_token_logprobs_val == [0.0, -0.5]
    assert len(d.output_top_logprobs_val) == 2
    assert d.input_token_logprobs_val == [None, -0.2, 0.0]


def test_old_peer_without_scores_aborts_the_request_with_a_client_response():
    d = state(0, 2, 8)
    proc, sender = PIPE._processor(d)
    proc.on_remote_prefill_done("r", 21, 0)
    assert d.to_abort and d.abort_notify_client
    assert "update both peers" in d.to_abort_message
    assert proc.finish_remote_prefill_only_request("r") == [("abort", ("r",))]
    assert len(sender.items) == 1


def wire_message(wire):
    funcs = methods(
        "pd/mooncake/prefill.py",
        "MooncakeKVManagerPrefill",
        {"sync_status_to_decode_endpoint"},
    )
    sent = []
    p = SimpleNamespace(
        bootstrap_token_cond=threading.Condition(),
        cached_tokens={9: 0},
        prefill_logprobs={9: wire} if wire else {},
        _connect=lambda endpoint: (
            SimpleNamespace(send_multipart=sent.append),
            nullcontext(),
        ),
    )
    funcs["sync_status_to_decode_endpoint"](
        p, "127.0.0.1", 1234, 9, STATUS.Success, 0, 21, None
    )
    return sent[0]


def test_status_wire_adds_optional_cpu_frame_and_preserves_legacy_messages():
    parse = methods("pd/mooncake/decode.py", None, {"parse_prefill_status_message"})[
        "parse_prefill_status_message"
    ]
    wire = PAYLOAD.snapshot_prefill_logprobs(prefill(0, 2))
    parts = wire_message(wire)
    assert len(parts) == 7 and parts[-1] == wire
    assert parse(parts) == (9, STATUS.Success, 0, 21, None, 0, wire)
    legacy = wire_message(None)
    assert len(legacy) == 6 and parse(legacy)[-1] is None


@pytest.mark.parametrize(
    "expected", [frozenset({0}), frozenset({0, 2}), frozenset(range(4))]
)
def test_heterogeneous_tp_completion_waits_for_all_routes_then_publishes_scores(
    expected,
):
    funcs = methods(
        "pd/mooncake/decode.py",
        "MooncakeKVManagerDecode",
        {"_handle_prefill_status", "pop_logprobs"},
    )
    d = SimpleNamespace(
        request_status={9: STATUS.WaitingForInput},
        expected_prefill_ranks_table={9: expected},
        prefill_response_tracker=defaultdict(set),
        cached_tokens_table={},
        bootstrap_token_table={},
        spec_candidate_ids_table={},
        _pending_bootstrap_token_table={},
        _pending_spec_candidate_ids_table={},
        logprobs_table={},
        _pending_logprobs_table={},
        record_failure=lambda *args: pytest.fail(str(args)),
    )
    d.update_status = lambda room, status: d.request_status.__setitem__(room, status)
    wire = PAYLOAD.snapshot_prefill_logprobs(prefill(0, 2))
    ordered = sorted(expected)
    for index, rank in enumerate(ordered):
        funcs["_handle_prefill_status"](
            d,
            9,
            STATUS.Success,
            rank,
            21 if index == 0 else -1,
            None,
            0,
            wire if index == 0 else None,
        )
        if index + 1 < len(expected):
            assert (
                d.request_status[9] == STATUS.WaitingForInput and d.logprobs_table == {}
            )
    assert d.request_status[9] == STATUS.Success
    assert funcs["pop_logprobs"](d, 9) == wire
    assert funcs["pop_logprobs"](d, 9) is None
    assert d._pending_logprobs_table == {}


def test_bootstrap_eos_finishes_with_its_score():
    d = state(0, 2, 8)
    d.sampling_params.ignore_eos = False
    d.sampling_params.stop_token_ids = {21}
    proc, sender = PIPE._processor(d)
    proc.on_remote_prefill_done(
        "r", 21, 0, logprobs=PAYLOAD.snapshot_prefill_logprobs(prefill(0, 2))
    )
    assert d.finished and d.output_ids == [21]
    proc.finish_remote_prefill_only_request("r")
    assert sender.items[0].output_token_logprobs_val == [[0.0]]


def test_prefill_room_cleanup_and_reuse_drop_scores():
    funcs = methods(
        "pd/mooncake/prefill.py",
        "MooncakeKVManagerPrefill",
        {"record_logprobs", "begin_room", "discard_room"},
    )
    p = SimpleNamespace(
        bootstrap_token_cond=threading.Condition(),
        request_status={9: STATUS.WaitingForInput},
        prefill_metadata={},
        prefill_logprobs={},
        cached_tokens={},
        transfer_infos={},
    )
    p.update_status = lambda room, status: p.request_status.__setitem__(room, status)
    funcs["record_logprobs"](p, 9, b"old")
    funcs["begin_room"](p, 9)
    assert p.prefill_logprobs == {}
    funcs["record_logprobs"](p, 9, b"new")
    funcs["discard_room"](p, 9)
    funcs["record_logprobs"](p, 9, b"late")
    assert p.prefill_logprobs == {} and p.request_status == {}


@pytest.mark.parametrize("pending", [False, True])
def test_decode_receiver_cleanup_discards_scores_on_success_or_abort(pending):
    clear = methods("pd/mooncake/receiver.py", "MooncakeKVReceiver", {"clear"})["clear"]
    fields = (
        "request_status",
        "prefill_response_tracker",
        "expected_prefill_ranks_table",
        "bootstrap_token_table",
        "spec_candidate_ids_table",
        "cached_tokens_table",
        "logprobs_table",
        "_pending_logprobs_table",
        "_pending_bootstrap_token_table",
        "_pending_spec_candidate_ids_table",
        "failure_records",
    )
    manager = SimpleNamespace(
        **{key: {} for key in fields},
        failure_lock=threading.Lock(),
        connection_lock=threading.Lock(),
        addr_to_rooms_tracker={"peer": {9}},
    )
    manager.request_status[9] = STATUS.Failed if pending else STATUS.Success
    (manager._pending_logprobs_table if pending else manager.logprobs_table)[
        9
    ] = b"scores"
    receiver = SimpleNamespace(kv_mgr=manager, bootstrap_room=9, bootstrap_addr="peer")
    clear(receiver)
    assert all(not vars(manager)[field] for field in fields)
    assert manager.addr_to_rooms_tracker["peer"] == set()


def test_real_zmq_transfers_variable_length_cpu_payload_without_registration():
    zmq = pytest.importorskip("zmq")
    context = zmq.Context()
    sender, receiver = context.socket(zmq.PAIR), context.socket(zmq.PAIR)
    sender.setsockopt(zmq.SNDTIMEO, 1000)
    receiver.setsockopt(zmq.RCVTIMEO, 1000)
    try:
        sender.bind("inproc://pd-logprobs")
        receiver.connect("inproc://pd-logprobs")
        for k in (0, 2):
            wire = PAYLOAD.snapshot_prefill_logprobs(prefill(0, k))
            sender.send_multipart(wire_message(wire))
            received = receiver.recv_multipart()
            d = state(0, k, 8)
            PAYLOAD.restore_prefill_logprobs(d, int(received[3]), received[6])
            assert d.input_token_logprobs_val == [None, -0.2, 0.0]
    finally:
        sender.close(linger=0)
        receiver.close(linger=0)
        context.term()


@pytest.mark.parametrize("terminal", [STATUS.Success, STATUS.Failed])
def test_late_completion_cannot_recreate_consumed_or_aborted_payload(terminal):
    handle = methods(
        "pd/mooncake/decode.py", "MooncakeKVManagerDecode", {"_handle_prefill_status"}
    )["_handle_prefill_status"]
    d = SimpleNamespace(
        request_status={9: terminal}, logprobs_table={}, _pending_logprobs_table={}
    )
    handle(d, 9, STATUS.Success, 0, 21, None, 0, b"late")
    assert d.logprobs_table == {} and d._pending_logprobs_table == {}
    assert d.request_status[9] == terminal


def test_executor_takes_scores_before_receiver_clear_and_hook_finishes_request():
    class RemoteDone:
        def __init__(self, rid, token):
            self.request_id, self.bootstrap_token = rid, token

    class OtherEvent:
        pass

    class Decode(SimpleNamespace):
        pass

    pd = SimpleNamespace(
        RemotePrefillDoneEvent=RemoteDone,
        SucceededEvent=OtherEvent,
        FailedEvent=OtherEvent,
    )
    wire = PAYLOAD.snapshot_prefill_logprobs(prefill(0, 2))
    manager = SimpleNamespace(payload=wire)
    manager.pop_prefill_metadata = lambda room: (21, None, 0)

    def pop(room):
        result, manager.payload = manager.payload, None
        return result

    manager.pop_logprobs = pop
    cleared = []

    def clear():
        assert manager.payload is None
        cleared.append(True)

    decode = Decode(
        receivers={"r": SimpleNamespace(bootstrap_room=9, clear=clear)},
        kv_manager=manager,
        gloo_group=None,
        _local_states={"r": STATUS.Bootstrapped},
        _admissions={"r": (3, 0)},
        _remote_logprobs={},
        _remote_cached_tokens={},
        _remote_cache_slots={},
        _remote_spec_candidate_ids={},
    )
    names = {
        "generate_events",
        "pop_remote_logprobs",
        "pop_remote_cached_tokens",
        "pop_remote_cache_slot",
        "pop_remote_spec_candidate_ids",
    }
    funcs = methods("pd/decode_executor.py", "DisaggDecodeExecutor", names)
    funcs.update(PD=pd, poll_and_all_reduce=lambda receivers, group: [STATUS.Success])
    for name in names:
        vars(decode)[name] = funcs[name].__get__(decode)
    d = state(0, 2, 1)
    proc, sender = PIPE._processor(d)
    hook_funcs = methods(
        "pd/transfer_hooks.py", "PdTransferHooks", {"poll_transfer_events"}
    )
    hook_funcs.update(
        PD=pd,
        DisaggPrefillExecutor=type("Prefill", (), {}),
        DisaggDecodeExecutor=Decode,
    )
    landed = []
    hook = SimpleNamespace(
        _loop=SimpleNamespace(kv_transfer=decode, output_processor=proc),
        _device=SimpleNamespace(
            run_remote_prefill_landing=lambda *args: landed.append(args)
        ),
    )
    events = hook_funcs["poll_transfer_events"](hook)
    assert isinstance(events[0], RemoteDone) and events[1] == ("finish", ("r",))
    assert cleared == [True] and landed == [(None, None)]
    assert decode._remote_logprobs == {} and decode.receivers == {}
    assert sender.items[0].output_token_logprobs_val == [[0.0]]
    assert sender.items[0].input_token_logprobs_idx == [[0, 1, 2]]


def test_prefill_hook_freezes_only_final_committed_chunk():
    class Prefill(SimpleNamespace):
        pass

    funcs = methods("pd/transfer_hooks.py", "PdTransferHooks", {"record_prefill_usage"})
    funcs.update(
        DisaggPrefillExecutor=Prefill,
        snapshot_prefill_logprobs=PAYLOAD.snapshot_prefill_logprobs,
    )
    recorded = []
    executor = Prefill(
        record_cached_tokens=lambda *args: None,
        record_logprobs=lambda *args: recorded.append(args),
    )
    p = prefill(0, 2)
    p.computed_length = 2
    hook = SimpleNamespace(
        _loop=SimpleNamespace(
            kv_transfer=executor,
            output_processor=SimpleNamespace(rid_to_state={"r": p}),
        )
    )
    funcs["record_prefill_usage"](hook, ["r", "retired"])
    assert recorded == []
    p.computed_length = 3
    funcs["record_prefill_usage"](hook, ["r"])
    assert len(recorded) == 1 and recorded[0][0] == "r"
    p.input_token_logprobs_val[1] = -99
    d = state(0, 2, 8)
    PAYLOAD.restore_prefill_logprobs(d, 21, recorded[0][1])
    assert d.input_token_logprobs_val == [None, -0.2, 0.0]
