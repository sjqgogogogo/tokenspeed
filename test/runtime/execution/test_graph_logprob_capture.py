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
"""Actual runner-method CPU tests; mocked replay is not CUDA validation."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

PATH = Path(__file__).with_name("test_top_logprob_capture.py")
SPEC = importlib.util.spec_from_file_location("_graph_logprob_cpu_common", PATH)
COMMON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMMON
SPEC.loader.exec_module(COMMON)
HELPER = COMMON.HELPER
Config = COMMON.Config
RUNTIME = COMMON.RUNTIME


def mode(decode):
    return SimpleNamespace(
        is_decode=lambda: decode,
        is_idle=lambda: False,
        is_mixed=lambda: False,
        name="DECODE" if decode else "EXTEND",
    )


def context(bs, capture, diagnostic, decode):
    return SimpleNamespace(
        bs=bs,
        top_logprob_capture=capture,
        logprob_diagnostic=diagnostic,
        num_extends=0 if decode else bs,
        forward_mode=mode(decode),
        global_num_tokens=None,
        all_decode_or_idle=True,
        capture_hidden_mode=None,
        input_num_tokens=bs,
    )


def capture_for(ks):
    return HELPER.TopLogprobCapture(
        tuple(Config(True, -1, k, ()) for k in ks), 0, (1,) * len(ks), 7
    )


def make_runner(use_graph):
    events = []
    logs = []
    node = COMMON.method(
        RUNTIME / "execution/forward_step.py", "ForwardStepRunner", "__call__"
    )
    replay_helpers = [
        item
        for item in ast.parse((RUNTIME / "execution/forward_step.py").read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "replay_graph_then_update"
    ]
    call = COMMON.exec_nodes(
        [*replay_helpers, node],
        "_graph_logprob_cpu_runner",
        {
            "torch": torch,
            "nvtx_range": lambda *a, **k: nullcontext(),
            "logger": SimpleNamespace(info=lambda *args: logs.append(args)),
        },
    ).__call__
    storage = torch.empty(4, 7)
    original = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    tokens = torch.full((4, 1), 6, dtype=torch.int32)
    lengths = torch.ones(4, dtype=torch.int32)
    token_lp = torch.log_softmax(original, -1)[:, -1:]

    def replay(diagnostic):
        events.append("diagnostic_replay" if diagnostic else "ordinary_replay")
        if diagnostic:
            HELPER.RawLogitsSnapshot(storage).capture(
                COMMON.output(original, None, None, None, None)
            )
        # A sampler may overwrite its model logits after the snapshot copy.
        original.fill_(-1000)

    def eager(**kwargs):
        events.append("eager")
        if kwargs["ctx"].top_logprob_capture is not None:
            kwargs["ctx"].top_logprob_capture.capture(
                COMMON.output(original[: kwargs["bs"]], None, None, None, None)
            )
        return tokens[: kwargs["bs"]], lengths[: kwargs["bs"]], token_lp[: kwargs["bs"]]

    runner = SimpleNamespace(
        config=SimpleNamespace(enable_logprob_graph=True),
        _can_use_graph=lambda bs, ctx: use_graph and ctx.forward_mode.is_decode(),
        _padded_bs=lambda bs, ctx: 4,
        input_buffers=SimpleNamespace(
            req_pool_indices_buf=torch.arange(4, dtype=torch.int32),
            seq_lens_buf=torch.ones(4, dtype=torch.int32),
        ),
        _pad_graph_req_pool_indices=lambda values, padded: torch.cat(
            (values, torch.zeros(padded - len(values), dtype=torch.int32))
        ),
        _set_graph_state_write_indices=lambda *args: events.append("state_write"),
        _prepare_request_token_history_graph_inputs=lambda **kwargs: events.append(
            "history"
        ),
        token_to_kv_pool=SimpleNamespace(arena=SimpleNamespace(cache_group_specs=())),
        _prepare_decode_metadata=lambda *args, **kwargs: events.append(
            ("refresh", args[0], args[1], kwargs["use_graph"])
        ),
        _init_forward_metadata=lambda *args, **kwargs: events.append("extend_metadata"),
        deepep_adapter=SimpleNamespace(replay=lambda: events.append("deepep")),
        _cuda_graph_key=lambda bs: ("default", bs),
        graphs={("default", 4): SimpleNamespace(replay=lambda: replay(False))},
        diagnostic_graphs={
            ("default", 4): SimpleNamespace(replay=lambda: replay(True))
        },
        output_buffers={("default", 4): (tokens, lengths, token_lp)},
        diagnostic_output_buffers={("default", 4): (tokens, lengths, token_lp)},
        _graph_debug=True,
        device="cuda",
        _diagnostic_logits=storage,
        _verify_graph_metadata=lambda key: events.append(("pointer_guard", key)),
        diagnostic_graph_replays=0,
        global_rank=2,
        overlap_schedule_depth=1,
        max_tokens_per_req=1,
        _forward_func=eager,
        drafter=None,
    )
    return call, runner, events, logs, original


def run(call, runner, ctx):
    empty = torch.empty(0, dtype=torch.int32)
    return call(
        runner,
        ctx.bs,
        ctx,
        None,
        extend_with_prefix=False,
        extend_prefix_lens=empty,
        extend_prefix_lens_cpu=empty,
        extend_seq_lens=empty,
        extend_seq_lens_cpu=empty,
        extend_replay_lens_cpu=empty,
        extend_prompt_lens_cpu=empty,
        positions=None,
        block_tables={},
    )


class GraphLogprobTests(unittest.TestCase):
    def test_non_debug_diagnostic_counts_replay_without_info_logs(self):
        call, runner, events, logs, original = make_runner(True)
        runner._graph_debug = False
        run(call, runner, context(1, capture_for((2,)), True, True))
        self.assertEqual(runner.diagnostic_graph_replays, 1)
        self.assertIn("diagnostic_replay", events)
        self.assertEqual(logs, [])

    def test_actual_runner_padded_replay_then_dynamic_topk(self):
        call, runner, events, logs, original = make_runner(True)
        expected = torch.log_softmax(original[:3].clone(), -1).topk(3, -1)
        capture = capture_for((1, 3, 2))
        ctx = context(3, capture, True, True)
        returned = run(call, runner, ctx)
        self.assertEqual(ctx.bs, 3)
        self.assertEqual(len(returned), 3)
        self.assertEqual([tuple(t.shape) for t in returned], [(3, 1), (3,), (3, 1)])
        self.assertIn(("refresh", 4, 3, True), events)
        self.assertIn("diagnostic_replay", events)
        self.assertNotIn("eager", events)
        self.assertTrue(torch.equal(capture.output_values[:, 0], expected.values))
        self.assertTrue(torch.equal(capture.output_indices[:, 0], expected.indices))
        self.assertEqual(runner.diagnostic_graph_replays, 1)
        self.assertEqual(
            logs[0][0],
            "LOGPROB_GRAPH_REPLAY rank=2 count=1 live_bs=3 padded_bs=4 variant=default snapshot=True overlap_depth=1",
        )

    def test_live_k_batch_switch_and_normal_request_do_not_reuse_results(self):
        call, runner, events, logs, original = make_runner(True)
        ptr = runner._diagnostic_logits.data_ptr()
        captured = []
        for bs, k, offset in ((3, 2, 0), (1, 6, 30), (4, 1, -7)):
            fresh = (
                torch.arange(28, dtype=torch.float32).reshape(4, 7).roll(offset % 7, 1)
            )
            original.copy_(fresh)
            capture = capture_for((k,) * bs)
            run(call, runner, context(bs, capture, True, True))
            expected = torch.log_softmax(fresh[:bs], -1).topk(k, -1)
            self.assertTrue(torch.equal(capture.output_values[:, 0], expected.values))
            self.assertTrue(torch.equal(capture.output_indices[:, 0], expected.indices))
            captured.append(capture.copy_to_cpu()["output_top_logprobs_val"].clone())
            self.assertEqual(runner._diagnostic_logits.data_ptr(), ptr)
            saved = runner._diagnostic_logits.clone()
            run(call, runner, context(2, None, False, True))
            self.assertTrue(torch.equal(runner._diagnostic_logits, saved))
        self.assertEqual(runner.diagnostic_graph_replays, 3)
        self.assertEqual(len(logs), 3)
        self.assertEqual(events.count("diagnostic_replay"), 3)
        self.assertEqual(events.count("ordinary_replay"), 3)
        self.assertEqual(
            [tuple(x.shape) for x in captured], [(3, 1, 2), (1, 1, 6), (4, 1, 1)]
        )

    def test_input_only_k_zero_replays_original_graph_but_counts_diagnostic(self):
        call, runner, events, logs, original = make_runner(True)
        runner._diagnostic_logits.fill_(123)
        run(call, runner, context(1, None, True, True))
        self.assertIn("ordinary_replay", events)
        self.assertEqual(runner.diagnostic_graph_replays, 1)
        self.assertTrue(torch.all(runner._diagnostic_logits == 123))
        self.assertIn("snapshot=False overlap_depth=1", logs[0][0])

    def test_missing_graph_or_disabled_flag_cannot_silently_eager_fallback(self):
        for available, enabled, captured in (
            (True, False, True),
            (True, True, False),
        ):
            call, runner, events, logs, original = make_runner(available)
            runner.config.enable_logprob_graph = enabled
            if not captured:
                runner.diagnostic_graphs.clear()
            with self.subTest(
                available=available, enabled=enabled, captured=captured
            ), self.assertRaises(RuntimeError):
                run(call, runner, context(1, capture_for((2,)), True, True))
            self.assertNotIn("eager", events)
            self.assertNotIn("diagnostic_replay", events)

    def test_diagnostic_above_ladder_keeps_same_eager_route(self):
        call, runner, events, logs, original = make_runner(False)
        capture = capture_for((2, 1))
        expected = torch.log_softmax(original[:2].clone(), -1).topk(2, -1)
        run(call, runner, context(2, capture, True, True))
        self.assertIn(("refresh", 2, 2, False), events)
        self.assertIn("eager", events)
        self.assertTrue(torch.equal(capture.output_values[:, 0], expected.values))
        self.assertEqual(runner.diagnostic_graph_replays, 0)

    def test_queued_results_survive_later_replays_before_cpu_consumption(self):
        call, runner, events, logs, original = make_runner(True)
        pending = []
        expected = []
        for shift in (0, 3, 1):
            logits = torch.arange(28, dtype=torch.float32).reshape(4, 7).roll(shift, -1)
            original.copy_(logits)
            capture = capture_for((3, 1, 2))
            run(call, runner, context(3, capture, True, True))
            pending.append(capture.copy_to_cpu())
            expected.append(torch.log_softmax(logits[:3], -1).topk(3, -1))
        # The control plane consumes these only after all three forwards have
        # reused the shared graph snapshot. Each result must own its data.
        for result, ref in zip(pending, expected):
            self.assertTrue(
                torch.equal(result["output_top_logprobs_val"][:, 0], ref.values)
            )
            self.assertTrue(
                torch.equal(result["output_top_logprobs_idx"][:, 0], ref.indices)
            )
        self.assertEqual(events.count("diagnostic_replay"), 3)
        self.assertNotIn("eager", events)

    def test_ordinary_above_ladder_keeps_existing_eager_route(self):
        call, runner, events, logs, original = make_runner(False)
        run(call, runner, context(2, None, False, True))
        self.assertIn(("refresh", 2, 2, False), events)
        self.assertIn("eager", events)
        self.assertEqual(logs, [])

    def test_output_only_prefill_still_uses_same_eager_route(self):
        call, runner, events, logs, original = make_runner(True)
        capture = capture_for((2, 1))
        run(call, runner, context(2, capture, True, False))
        self.assertIn("extend_metadata", events)
        self.assertIn("eager", events)
        self.assertEqual(logs, [])
        self.assertEqual(tuple(capture.output_values.shape), (2, 1, 2))

    def test_snapshot_guard_and_replayed_prompt_rejection(self):
        for buffer in (torch.ones(3), torch.ones(2, 7, dtype=torch.float16)):
            with self.assertRaises(ValueError):
                HELPER.RawLogitsSnapshot(buffer)
        snapshot = HELPER.RawLogitsSnapshot(torch.zeros(2, 7))
        for logits, layout in (
            (torch.ones(1, 7), None),
            (torch.ones(2, 7), object()),
            (torch.ones(2, 7, dtype=torch.int64), None),
        ):
            with self.assertRaises(RuntimeError):
                snapshot.capture(COMMON.output(logits, None, None, layout, None))
        prompt = HELPER.TopLogprobCapture((Config(True, 0, 2, (1,)),), 1, (1,), 7)
        with self.assertRaisesRegex(RuntimeError, "prompt"):
            prompt.capture_replayed_logits(torch.ones(1, 7))
        decode = capture_for((2,))
        decode.capture_replayed_logits(torch.ones(1, 7))
        with self.assertRaisesRegex(RuntimeError, "only once"):
            decode.capture_replayed_logits(torch.ones(1, 7))

    def test_snapshot_actual_forward_precedes_sampling_mutation(self):
        node = COMMON.method(
            RUNTIME / "execution/model_executor.py", "ModelExecutor", "_forward_step"
        )
        forward = COMMON.exec_nodes(
            [node], "_graph_logprob_cpu_forward", {"torch": torch}
        )._forward_step
        logits = torch.arange(7, dtype=torch.float32).reshape(1, 7)
        snapshot = HELPER.RawLogitsSnapshot(torch.empty_like(logits))
        value = COMMON.output(logits, None, None, None, None)

        def sampler(value, *args):
            value.next_token_logits.fill_(-99)
            return torch.ones(1, 1, dtype=torch.int32), torch.ones(1, dtype=torch.int32)

        executor = SimpleNamespace(
            capturable_grammar=None,
            dspark_context_producer=None,
            drafter=None,
            config=SimpleNamespace(pp_size=1),
            _run_target_forward=lambda ctx: value,
            _decode_candidates=lambda ctx: None,
            nan_guard=SimpleNamespace(
                audit_logits=lambda value, ctx: value.next_token_logits.add_(1),
                merge_oov=lambda *args: None,
            ),
            _run_sampling=sampler,
            runtime_states=SimpleNamespace(vocab_size=7),
        )
        ctx = SimpleNamespace(raw_logit_snapshot=snapshot, top_logprob_capture=None)
        self.assertEqual(len(forward(executor, 1, ctx, None)), 3)
        self.assertTrue(
            torch.equal(
                snapshot.buffer, torch.arange(1, 8, dtype=torch.float32).reshape(1, 7)
            )
        )
        self.assertTrue(torch.all(logits == -99))

    def test_prefill_graph_replay_keeps_prompt_scores_in_the_logits_tail(self):
        nodes = [
            COMMON.method(RUNTIME / "execution/prefill_graph.py", "PrefillGraph", name)
            for name in ("replay", "_padded_to")
        ]
        module = COMMON.exec_nodes(
            nodes,
            "_prefill_logprob_cpu_replay",
            {
                "torch": torch,
                "contextmanager": contextmanager,
                "active_forward": lambda ctx: nullcontext(),
                "LogitsMetadata": COMMON.LOGITS.LogitsMetadata,
            },
        )
        hidden = torch.arange(28, dtype=torch.float32).reshape(4, 7)
        capture = HELPER.TopLogprobCapture((Config(True, 0, 2, (1, 2, 0)),), 1, (3,), 7)
        ctx = SimpleNamespace(
            input_num_tokens=3,
            global_num_tokens=None,
            global_bs=1,
            top_logprob_capture=capture,
            forward_mode=SimpleNamespace(is_extend_or_mixed=lambda: True),
            gather_ids=torch.tensor([2]),
            capture_hidden_mode=COMMON.CaptureMode.NULL,
        )
        events = []
        processor = COMMON.make_processor(COMMON.LOGITS)

        def logits_tail(ids, states, head, metadata, aux):
            events.append("logits")
            self.assertEqual(ctx.input_num_tokens, 3)
            self.assertEqual(tuple(states.shape), (3, 7))
            return processor.forward(ids, states, head, metadata, aux)

        runner = SimpleNamespace(
            _replay_bucket=lambda ctx: 4,
            _log_engaged_once=lambda *args: None,
            _land_input_embeds=lambda *args: events.append("embed"),
            _embed_tokens=lambda ids: ids,
            input_buffers=SimpleNamespace(
                input_ids_buf=torch.ones(4, dtype=torch.int32),
                positions_buf=torch.arange(4),
            ),
            config=SimpleNamespace(model_is_mrope=False, world_size=1),
            dp_size=1,
            _narrowing=None,
            attn_backend=SimpleNamespace(step_counter=object()),
            _captures={
                (4, None): (
                    SimpleNamespace(
                        replay=lambda **kw: events.append(
                            ("graph", ctx.input_num_tokens, kw["valid_rows"])
                        )
                    ),
                    SimpleNamespace(sliced=lambda n: (hidden[:n], None)),
                )
            },
            text_model=SimpleNamespace(
                logits_processor=logits_tail, lm_head=torch.eye(7)
            ),
        )
        runner._padded_to = lambda ctx, bucket: contextmanager(module._padded_to)(
            runner, ctx, bucket
        )
        result = module.replay(runner, ctx, torch.tensor([0, 1, 2]), None)
        capture.capture(result)
        self.assertEqual(events, ["embed", ("graph", 4, 3), "logits"])
        expected = torch.log_softmax(hidden[:3], -1)
        self.assertTrue(
            torch.equal(
                capture.input_token_values[0],
                expected[torch.arange(3), torch.tensor([1, 2, 0])],
            )
        )
        self.assertTrue(
            torch.equal(capture.input_values[0], expected.topk(2, -1).values)
        )

    def test_capture_assets_are_separate_and_config_flag_is_explicit(self):
        method = COMMON.method(
            RUNTIME / "execution/forward_step.py", "ForwardStepRunner", "capture"
        )
        calls = [
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_capture_one"
        ]
        self.assertEqual(len(calls), 2)
        values = [
            next(k.value for k in c.keywords if k.arg == "logprob_snapshot")
            for c in calls
        ]
        self.assertTrue(
            any(isinstance(v, ast.Constant) and v.value is None for v in values)
        )
        self.assertTrue(
            any(
                isinstance(v, ast.Call) and v.func.id == "RawLogitsSnapshot"
                for v in values
            )
        )
        method = COMMON.method(
            RUNTIME / "execution/model_executor.py",
            "ModelExecutorConfig",
            "from_server_args",
        )
        call = next(
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "ModelExecutorConfig"
        )
        self.assertIn("enable_logprob_graph", [k.arg for k in call.keywords])

    def test_actual_capture_one_keeps_tuple3_and_only_diagnostic_context_has_snapshot(
        self,
    ):
        node = COMMON.method(
            RUNTIME / "execution/forward_step.py", "ForwardStepRunner", "_capture_one"
        )
        # Replace only the optional grammar import; execute the actual method.
        node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
        forward_mode = SimpleNamespace(DECODE=mode(True))
        module = COMMON.exec_nodes(
            [node],
            "_graph_logprob_cpu_capture",
            {
                "ForwardMode": forward_mode,
                "ForwardContext": SimpleNamespace,
                "CaptureHiddenMode": SimpleNamespace(NULL=None, FULL="full"),
                "SamplingBatchInfo": SimpleNamespace,
                "dist": SimpleNamespace(barrier=lambda: None),
                "bind_grammar_mask_buf": lambda *args, **kwargs: None,
                "_is_cuda_graph_phase": False,
                "_is_capture_mode": False,
                "global_graph_memory_pool": None,
                "snapshot_graph_metadata": lambda backend: {"pointer": 123},
            },
        )
        capture = module._capture_one
        snapshots_seen = []

        def forward(bs, ctx, sampling_info):
            snapshots_seen.append(ctx.raw_logit_snapshot)
            if ctx.raw_logit_snapshot is not None:
                ctx.raw_logit_snapshot.capture(
                    COMMON.output(torch.full((bs, 7), 9.0), None, None, None, None)
                )
            return torch.zeros(bs, 1), torch.ones(bs), None

        runner = SimpleNamespace(
            device="cuda",
            device_module=SimpleNamespace(
                CUDAGraph=lambda: SimpleNamespace(pool=lambda: "shared"),
                stream=lambda stream: nullcontext(),
                synchronize=lambda: None,
                graph=lambda *args, **kwargs: nullcontext(),
            ),
            _prepare_request_token_history_graph_inputs=lambda **kwargs: None,
            attn_backend=object(),
            token_to_kv_pool=object(),
            max_tokens_per_req=1,
            drafter=None,
            _request_token_history_view=lambda bs: None,
            dp_size=1,
            input_buffers=SimpleNamespace(
                req_pool_indices_buf=torch.zeros(4, dtype=torch.int32),
                seq_lens_buf=torch.ones(4, dtype=torch.int32),
            ),
            runtime_states=None,
            vocab_size=7,
            eager_grammar_buffers=None,
            capturable_grammar=None,
            dspark_context_producer=None,
            grammar_backend="none",
            _forward_func=forward,
            stream=object(),
            _prepare_sampling_capture=lambda **kwargs: None,
            _init_capture_metadata=lambda bs: None,
            sampling_backend=None,
            deepep_adapter=SimpleNamespace(capture=lambda: None),
            _graph_debug=True,
            draft_attn_backend=None,
            _metadata_snapshots={},
        )
        storage = torch.zeros(4, 7)
        sink = HELPER.RawLogitsSnapshot(storage[:3])
        for requested in (None, sink):
            graph, result = capture(runner, 3, "default", requested)
            self.assertEqual(len(result), 3)
            self.assertTrue(all(item is requested for item in snapshots_seen[-5:]))
        self.assertEqual(len(snapshots_seen), 10)
        self.assertTrue(torch.all(storage[:3] == 9))
        self.assertTrue(torch.all(storage[3] == 0))
        self.assertEqual(
            set(runner._metadata_snapshots),
            {("default", 3, False), ("default", 3, True)},
        )
        self.assertFalse(module._is_capture_mode)
        self.assertFalse(module._is_cuda_graph_phase)


if __name__ == "__main__":
    unittest.main()
