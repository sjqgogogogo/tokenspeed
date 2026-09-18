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
"""CPU torch and real-method AST tests; no CUDA, scheduler extension or server."""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib.util
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "python/tokenspeed/runtime/execution/logprob_utils.py").is_file()
)
RUNTIME = ROOT / "python/tokenspeed/runtime"


def load_types():
    path = RUNTIME / "execution/types.py"
    spec = importlib.util.spec_from_file_location("_top_logprob_cpu_types", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TYPES = load_types()
Config = TYPES.LogprobRequestConfig


def exec_nodes(nodes, name, namespace):
    module = ModuleType(name)
    module.__dict__.update(namespace)
    sys.modules[name] = module
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), name, "exec"), module.__dict__)
    return module


def load_helper():
    tree = ast.parse((RUNTIME / "execution/logprob_utils.py").read_text())
    nodes = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module in {"__future__", "tokenspeed.runtime.execution.types"}
        )
    ]
    return exec_nodes(
        nodes, "_top_logprob_cpu_helper", {"LogprobRequestConfig": Config}
    )


HELPER = load_helper()


class CaptureMode(Enum):
    NULL = 0

    def need_capture(self):
        return False


def load_logits_classes(path, name):
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name in {"LogitsProcessorOutput", "LogitsMetadata"}:
            nodes.append(node)
        elif node.name == "LogitsProcessor":
            node.bases = []
            node.body = [
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef)
                and method.name
                in {
                    "forward",
                    "get_top_logprobs",
                    "compute_temp_top_p_normalized_logprobs",
                }
            ]
            nodes.append(node)
    return exec_nodes(
        nodes,
        name,
        {
            "dataclasses": dataclasses,
            "torch": torch,
            "CaptureHiddenMode": CaptureMode,
            "is_pin_memory_available": lambda: False,
            "split_prompt_topk": HELPER.split_prompt_topk,
        },
    )


LOGITS = load_logits_classes(
    RUNTIME / "layers/logits_processor.py", "_top_logprob_cpu_logits"
)


def make_processor(module):
    processor = module.LogitsProcessor()
    processor.do_argmax = False
    processor._resolve_logits_layout_plan = lambda states, metadata: None
    processor._get_logits = lambda states, head, metadata, *, plan: states @ head
    return processor


def output(logits, values, indices, layout, token_values):
    return LOGITS.LogitsProcessorOutput(
        next_token_logits=logits,
        input_top_logprobs_val=values,
        input_top_logprobs_idx=indices,
        logits_layout_plan=layout,
        input_token_logprobs=token_values,
    )


def context(capture, lengths, extends):
    return SimpleNamespace(
        top_logprob_capture=capture,
        capture_hidden_mode=CaptureMode.NULL,
        gather_ids=(
            torch.tensor(lengths, dtype=torch.int64).cumsum(0) - 1 if extends else None
        ),
        forward_mode=SimpleNamespace(is_extend_or_mixed=lambda: bool(extends)),
    )


def method(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    node = copy.deepcopy(
        next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
    )
    node.decorator_list = []
    return node


class CaptureTests(unittest.TestCase):
    def test_mixed_lengths_and_raw_output_log_softmax(self):
        configs = (
            Config(True, 0, 2, (1, 4, 0)),
            Config(False, 0, 3, ()),
            Config(True, -1, 3, ()),
            Config(True, 0, 1, ()),
        )
        capture = HELPER.TopLogprobCapture(configs, 2, (3, 2, 1, 1), 5)
        self.assertEqual(capture.pruned_lens, [3, 0, 0, 0])
        self.assertEqual(capture.start_lens, [0, 2, 1, 1])
        hidden = torch.arange(28, dtype=torch.float32).reshape(7, 4) / 8
        head = torch.tensor(
            [[1, 2, -1, 3, 0], [0, 1, 4, -2, 3], [2, 0, -3, 1, 4], [-2, 3, 1, 0, 5]],
            dtype=torch.float32,
        )
        logits = hidden @ head
        metadata = LOGITS.LogitsMetadata.from_forward_context(
            context(capture, (3, 2, 1, 1), 2)
        )
        result = make_processor(LOGITS).forward(None, hidden, head, metadata, None)
        torch.testing.assert_close(
            result.next_token_logits, logits[[2, 4, 5, 6]], rtol=0, atol=0
        )
        expected = torch.log_softmax(logits[:3], -1).topk(2, dim=-1)
        torch.testing.assert_close(
            result.input_top_logprobs_val[0], expected.values, rtol=0, atol=0
        )
        self.assertTrue(torch.equal(result.input_top_logprobs_idx[0], expected.indices))
        self.assertEqual(result.input_top_logprobs_val[1:], [None, None, None])
        expected_tokens = torch.log_softmax(logits[:3], -1)[
            torch.arange(3), torch.tensor([1, 4, 0])
        ]
        torch.testing.assert_close(
            result.input_token_logprobs, expected_tokens, rtol=0, atol=0
        )
        capture.capture(result)
        raw = torch.log_softmax(logits[[2, 4, 5, 6]], -1).topk(3, dim=-1)
        torch.testing.assert_close(
            capture.output_values[:, 0], raw.values, rtol=0, atol=0
        )
        self.assertTrue(torch.equal(capture.output_indices[:, 0], raw.indices))
        result.next_token_logits.fill_(-1000)  # Sampler may mutate after capture.
        copied = capture.copy_to_cpu()
        self.assertEqual(tuple(copied["output_top_logprobs_val"].shape), (4, 1, 3))
        torch.testing.assert_close(
            copied["output_top_logprobs_val"][:, 0], raw.values, rtol=0, atol=0
        )
        self.assertEqual(tuple(copied["input_top_logprobs_val"][0].shape), (3, 2))
        self.assertEqual(copied["input_top_logprobs_val"][0].device.type, "cpu")
        self.assertTrue(torch.equal(copied["input_token_logprobs"][0], expected_tokens))
        self.assertEqual(copied["input_token_logprobs"][1:], [None, None, None])

    def test_heterogeneous_k_all_prompt_source_rows_including_chunk_last(self):
        configs = (Config(True, 100, 1, (2, 4)), Config(True, 0, 3, (1, 3, 0)))
        capture = HELPER.TopLogprobCapture(configs, 2, (2, 3), 5)
        self.assertEqual(
            capture.pruned_lens, [2, 3]
        )  # Global start alignment is commit-side.
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 4
        head = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 8
        metadata = LOGITS.LogitsMetadata.from_forward_context(
            context(capture, (2, 3), 2)
        )
        result = make_processor(LOGITS).forward(None, hidden, head, metadata, None)
        expected = torch.log_softmax(hidden @ head, -1)
        for index, rows, k in ((0, slice(0, 2), 1), (1, slice(2, 5), 3)):
            top = expected[rows].topk(k, dim=-1)
            torch.testing.assert_close(
                result.input_top_logprobs_val[index], top.values, rtol=0, atol=0
            )
            self.assertTrue(
                torch.equal(result.input_top_logprobs_idx[index], top.indices)
            )
        capture.capture(result)

    def test_decode_and_output_only_extend_do_not_request_prompt_logits(self):
        for extends, lengths in ((0, (1, 1)), (2, (3, 2))):
            capture = HELPER.TopLogprobCapture(
                (Config(True, -1, 2, ()), Config(True, -1, 1, ())), extends, lengths, 5
            )
            metadata = LOGITS.LogitsMetadata.from_forward_context(
                context(capture, lengths, extends)
            )
            self.assertFalse(metadata.extend_return_logprob)
            logits = torch.arange(10, dtype=torch.float32).reshape(2, 5)
            capture.capture(output(logits, None, None, None, None))
            self.assertIsNone(capture.copy_to_cpu()["input_top_logprobs_val"])

    def test_normal_processor_matches_matmul_rows_for_extend_mixed_decode(self):
        for extends, lengths in ((0, (1, 1)), (2, (3, 2)), (1, (3, 1))):
            ctx = context(None, lengths, extends)
            new_metadata = LOGITS.LogitsMetadata.from_forward_context(ctx)
            self.assertFalse(new_metadata.keep_topk_on_device)
            self.assertFalse(new_metadata.extend_return_logprob)
            hidden = (
                torch.arange(sum(lengths) * 4, dtype=torch.float32).reshape(-1, 4) / 4
            )
            head = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 8
            new = make_processor(LOGITS).forward(None, hidden, head, new_metadata, None)
            # Independent oracle: ordinary prefill/mixed selects each request's
            # final source row; decode already supplies one row per request.
            expected = hidden @ head
            if extends:
                rows = torch.tensor(lengths, dtype=torch.int64).cumsum(0) - 1
                expected = expected[rows]
            self.assertTrue(torch.equal(new.next_token_logits, expected))
            self.assertIsNone(new.input_top_logprobs_val)

    def test_invalid_configuration_rejected_before_tensor_work(self):
        valid = ((Config(True, 0, 2, (1, 2)),), 1, (2,), 5)
        bad = (
            ((), 0, (), 5),
            ((Config(True, 0, 2, (1, 2)),), 1, (), 5),
            (valid[0], -1, (2,), 5),
            (valid[0], 2, (2,), 5),
            (valid[0], True, (2,), 5),
            (valid[0], 1, (0,), 5),
            (valid[0], 1, (-1,), 5),
            (valid[0], 0, (2,), 5),
            (valid[0], 1, (2,), 0),
            ((Config(True, -2, 2, (1, 2)),), 1, (2,), 5),
            ((Config(True, 0, -1, (1, 2)),), 1, (2,), 5),
            ((Config(True, 0, 6, (1, 2)),), 1, (2,), 5),
            ((Config(True, 0, True, (1, 2)),), 1, (2,), 5),
            ((Config(False, 0, 2, ()),), 1, (2,), 5),
            ((Config(True, 0, 0, ()),), 1, (2,), 5),
            ((Config(True, 0, 0, (1, 5)),), 1, (2,), 5),
            ((Config(True, 0, 0, (-1, 2)),), 1, (2,), 5),
        )
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                HELPER.TopLogprobCapture(*args)

    def test_bad_next_logits_missing_capture_and_layout_fail_closed(self):
        for logits, layout in (
            (torch.ones(5), None),
            (torch.ones(2, 5), None),
            (torch.ones(1, 4), None),
            (torch.ones(1, 5), object()),
        ):
            capture = HELPER.TopLogprobCapture((Config(True, -1, 2, ()),), 0, (1,), 5)
            with self.assertRaises(RuntimeError):
                capture.capture(output(logits, None, None, layout, None))
        with self.assertRaises(RuntimeError):
            capture.copy_to_cpu()

    def test_prompt_payload_requires_one_typed_pair_per_expected_request(self):
        good_values, good_indices = torch.ones(2, 2), torch.ones(
            2, 2, dtype=torch.int64
        )
        cases = (
            (None, None),
            ([good_values], [good_indices]),
            ([good_values, None], [torch.ones(1, 2, dtype=torch.int64), None]),
            ([good_values, None], [good_indices.float(), None]),
            ([good_values.to(torch.int32), None], [good_indices, None]),
            ([good_values, good_values], [good_indices, good_indices]),
        )
        for values, indices in cases:
            capture = HELPER.TopLogprobCapture(
                (Config(True, 0, 2, (1, 2)), Config(True, -1, 1, ())), 1, (2, 1), 5
            )
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                capture.capture(
                    output(torch.ones(2, 5), values, indices, None, torch.ones(2))
                )
        capture.capture(
            output(
                torch.ones(2, 5),
                [good_values, None],
                [good_indices, None],
                None,
                torch.ones(2),
            )
        )
        with self.assertRaises(RuntimeError):
            capture.capture(
                output(
                    torch.ones(2, 5),
                    [good_values, None],
                    [good_indices, None],
                    None,
                    torch.ones(2),
                )
            )

    def test_split_preserves_tensor_rows_dtype_device_and_none_slots(self):
        logprobs = torch.log_softmax(
            torch.arange(20, dtype=torch.float32).reshape(4, 5), -1
        )
        values, indices = HELPER.split_prompt_topk(logprobs, [2, 0, 3], [1, 0, 3])
        self.assertEqual(
            [None if v is None else tuple(v.shape) for v in values],
            [(1, 2), None, (3, 3)],
        )
        self.assertIsNone(indices[1])
        for index, rows, k in ((0, slice(0, 1), 2), (2, slice(1, 4), 3)):
            expected = logprobs[rows].topk(k, dim=-1)
            self.assertTrue(torch.equal(values[index], expected.values))
            self.assertTrue(torch.equal(indices[index], expected.indices))
            self.assertEqual(indices[index].dtype, torch.int64)

    def test_split_rejects_malformed_and_negative_metadata(self):
        cases = (
            (torch.ones(2), [1], [2]),
            (torch.ones(2, 5), [], []),
            (torch.ones(2, 5), [2], [1]),
            (torch.ones(2, 5), [1, 2], [-1, 3]),
            (torch.ones(2, 5), [6], [2]),
            (torch.ones(2, 5), [-1], [2]),
            (torch.ones(2, 5), [True], [2]),
            (torch.ones(2, 5), [1], [2.0]),
        )
        for args in cases:
            with self.subTest(args=args[1:]), self.assertRaises(ValueError):
                HELPER.split_prompt_topk(*args)

    def test_input_sampled_with_zero_k_and_mixed_output_topk(self):
        configs = (Config(True, 0, 0, (4, 1, 0)), Config(True, -1, 2, ()))
        capture = HELPER.TopLogprobCapture(configs, 1, (3, 1), 5)
        metadata = LOGITS.LogitsMetadata.from_forward_context(
            context(capture, (3, 1), 1)
        )
        self.assertTrue(metadata.extend_return_logprob)
        self.assertFalse(metadata.extend_return_top_logprob)
        hidden = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 8
        head = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 8
        result = make_processor(LOGITS).forward(None, hidden, head, metadata, None)
        expected = torch.log_softmax(hidden[:3] @ head, -1)[
            torch.arange(3), torch.tensor([4, 1, 0])
        ]
        self.assertTrue(torch.equal(result.input_token_logprobs, expected))
        capture.capture(result)
        copied = capture.copy_to_cpu()
        self.assertTrue(torch.equal(copied["input_token_logprobs"][0], expected))
        self.assertIsNone(copied["input_token_logprobs"][1])
        self.assertIsNone(copied["input_top_logprobs_val"])
        self.assertEqual(tuple(copied["output_top_logprobs_val"].shape), (2, 1, 2))

        capture = HELPER.TopLogprobCapture((Config(True, 0, 0, (1, 0)),), 1, (2,), 5)
        capture.capture(
            output(torch.ones(1, 5), None, None, None, torch.tensor([-2.0, -3.0]))
        )
        copied = capture.copy_to_cpu()
        self.assertIsNone(copied["output_top_logprobs_val"])
        self.assertIsNone(copied["output_top_logprobs_idx"])
        self.assertEqual(copied["input_token_logprobs"][0].tolist(), [-2.0, -3.0])

    def test_input_sampled_payload_shape_and_dtype_fail_closed(self):
        for token_values in (
            None,
            torch.ones(1),
            torch.ones(2, 1),
            torch.ones(2, dtype=torch.int64),
        ):
            capture = HELPER.TopLogprobCapture(
                (Config(True, 0, 0, (1, 0)),), 1, (2,), 5
            )
            with self.subTest(token_values=token_values), self.assertRaises(
                RuntimeError
            ):
                capture.capture(
                    output(torch.ones(1, 5), None, None, None, token_values)
                )


class WiringTests(unittest.TestCase):
    def test_real_forward_step_captures_after_audit_before_sampling_and_keeps_tuple3(
        self,
    ):
        node = method(
            RUNTIME / "execution/model_executor.py", "ModelExecutor", "_forward_step"
        )
        forward = exec_nodes(
            [node], "_top_logprob_cpu_forward", {"torch": torch}
        )._forward_step
        for enabled in (False, True):
            events = []
            logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
            result = output(logits, None, None, None, None)
            actual_capture = HELPER.TopLogprobCapture(
                (Config(True, -1, 3, ()),), 0, (1,), 5
            )

            def capture(value):
                events.append("capture")
                actual_capture.capture(value)

            def audit(value, ctx):
                events.append("audit")
                value.next_token_logits.add_(1)

            def sample(value, info, ctx, candidates):
                events.append("sample")
                value.next_token_logprobs = torch.log_softmax(
                    value.next_token_logits, -1
                )[:, -1:]
                value.next_token_logits.fill_(-1)
                return torch.tensor([[4]], dtype=torch.int32), torch.tensor(
                    [1], dtype=torch.int32
                )

            executor = SimpleNamespace(
                capturable_grammar=None,
                drafter=None,
                config=SimpleNamespace(pp_size=1),
                _run_target_forward=lambda ctx: (events.append("target"), result)[1],
                nan_guard=SimpleNamespace(
                    audit_logits=audit, merge_oov=lambda *args: events.append("oov")
                ),
                _decode_candidates=lambda ctx: (events.append("candidates"), None)[1],
                _run_sampling=sample,
                runtime_states=SimpleNamespace(vocab_size=5),
            )
            ctx = SimpleNamespace(
                top_logprob_capture=(
                    SimpleNamespace(capture=capture) if enabled else None
                ),
                raw_logit_snapshot=None,
            )
            returned = forward(executor, 1, ctx, None)
            self.assertEqual(len(returned), 3)
            self.assertEqual(
                events,
                [
                    "target",
                    "audit",
                    *(["capture"] if enabled else []),
                    "candidates",
                    "sample",
                    "oov",
                ],
            )
            self.assertTrue(
                torch.equal(returned[0], torch.tensor([[4]], dtype=torch.int32))
            )
            self.assertIs(returned[2], result.next_token_logprobs)
            if enabled:
                expected = torch.log_softmax(
                    torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]), -1
                ).topk(3, -1)
                torch.testing.assert_close(
                    actual_capture.output_values[:, 0], expected.values, rtol=0, atol=0
                )

    def test_event_loop_snapshot_is_immutable_and_request_ordered(self):
        node = method(
            RUNTIME / "engine/event_loop.py", "EventLoop", "_gather_logprob_configs"
        )
        gather = exec_nodes(
            [node], "_top_logprob_cpu_gather", {"LogprobRequestConfig": Config}
        )._gather_logprob_configs
        first = SimpleNamespace(
            return_logprob=True,
            logprob_start_len=2,
            top_logprobs_num=3,
            prompt_input_ids=[0, 1, 2, 3],
        )
        second = SimpleNamespace(
            return_logprob=False, logprob_start_len=-1, top_logprobs_num=0
        )
        loop = SimpleNamespace(
            output_processor=SimpleNamespace(rid_to_state={"a": first, "b": second})
        )
        gathered = gather(
            loop,
            SimpleNamespace(
                request_ids=["b", "a"],
                num_extends=lambda: 2,
                extend_prefix_lens=[0, 1],
                input_lengths=[2, 2],
            ),
        )
        first.top_logprobs_num = 9
        first.prompt_input_ids[2] = 99
        self.assertEqual(
            gathered, (Config(False, -1, 0, ()), Config(True, 2, 3, (2, 3)))
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            gathered[1].top_logprobs_num = 5

    def test_real_chunk_target_alignment_including_cross_chunk_and_final_dummy(self):
        node = method(
            RUNTIME / "engine/event_loop.py", "EventLoop", "_gather_logprob_configs"
        )
        gather = exec_nodes(
            [node], "_top_logprob_cpu_chunks", {"LogprobRequestConfig": Config}
        )._gather_logprob_configs
        prompt = [0, 4, 1, 3, 2]
        state = SimpleNamespace(
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=0,
            prompt_input_ids=prompt,
        )
        loop = SimpleNamespace(
            output_processor=SimpleNamespace(rid_to_state={"r": state})
        )
        all_hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 4
        head = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 8
        all_logprobs = torch.log_softmax(all_hidden @ head, -1)
        actual = []
        targets_seen = []
        for prefix, length in ((0, 2), (2, 2), (4, 1)):
            op = SimpleNamespace(
                request_ids=["r"],
                num_extends=lambda: 1,
                extend_prefix_lens=[prefix],
                input_lengths=[length],
            )
            configs = gather(loop, op)
            targets_seen.append(configs[0].input_token_ids)
            capture = HELPER.TopLogprobCapture(configs, 1, (length,), 5)
            metadata = LOGITS.LogitsMetadata.from_forward_context(
                context(capture, (length,), 1)
            )
            result = make_processor(LOGITS).forward(
                None, all_hidden[prefix : prefix + length], head, metadata, None
            )
            capture.capture(result)
            copied = capture.copy_to_cpu()["input_token_logprobs"][0]
            for row in range(length):
                if prefix + row + 1 < len(prompt):
                    actual.append(copied[row])
        self.assertEqual(targets_seen, [(4, 1), (3, 2), (0,)])
        expected = all_logprobs[torch.arange(4), torch.tensor(prompt[1:])]
        self.assertTrue(torch.equal(torch.stack(actual), expected))
        # Retraction replay beyond prompt: no invented prompt target is exposed.
        replay = SimpleNamespace(
            request_ids=["r"],
            num_extends=lambda: 1,
            extend_prefix_lens=[5],
            input_lengths=[2],
        )
        self.assertEqual(gather(loop, replay)[0].input_token_ids, (0, 0))

    def test_input_only_k_zero_does_not_arm_capture_on_decode(self):
        node = method(
            RUNTIME / "execution/model_executor.py",
            "ModelExecutor",
            "execute_forward_op",
        )
        arming = next(
            value
            for value in node.body
            if isinstance(value, ast.If)
            and any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "TopLogprobCapture"
                for inner in ast.walk(value)
            )
        )
        expression = compile(
            ast.Expression(body=arming.test), "<real-capture-gate>", "eval"
        )
        configs = (Config(True, 0, 0, ()),)
        self.assertFalse(
            eval(expression, {"logprob_configs": configs, "num_extends": 0})
        )
        self.assertTrue(
            eval(expression, {"logprob_configs": configs, "num_extends": 1})
        )
        self.assertTrue(
            eval(
                expression,
                {"logprob_configs": (Config(True, -1, 3, ()),), "num_extends": 0},
            )
        )

    def test_required_controls_reach_both_production_callsites(self):
        seen = []
        for path in (RUNTIME / "engine/event_loop.py", RUNTIME / "execution/device.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (
                        node.func.attr if isinstance(node.func, ast.Attribute) else None
                    )
                )
                if name in {"PlannedForward", "execute_forward_op"}:
                    self.assertIn(
                        "logprob_configs", [keyword.arg for keyword in node.keywords]
                    )
                    seen.append(name)
        self.assertEqual(sorted(seen), ["PlannedForward", "execute_forward_op"])
        signature = method(
            RUNTIME / "execution/model_executor.py",
            "ModelExecutor",
            "execute_forward_op",
        )
        parameters = [argument.arg for argument in signature.args.args]
        required = parameters[: len(parameters) - len(signature.args.defaults)]
        self.assertIn("logprob_configs", required)
        with self.assertRaises(TypeError):
            TYPES.PlannedForward(
                forward_op=None,
                sampling_params_list=[],
                dp_metadata=None,
                grammar_inputs=None,
                multimodal_context=None,
            )

    def test_capture_cpu_copies_precede_existing_event_and_result_publication(self):
        node = method(
            RUNTIME / "execution/model_executor.py",
            "ModelExecutor",
            "execute_forward_op",
        )
        calls = [value for value in ast.walk(node) if isinstance(value, ast.Call)]
        copied = next(
            value.lineno
            for value in calls
            if isinstance(value.func, ast.Attribute)
            and value.func.attr == "copy_to_cpu"
        )
        recorded = next(
            value.lineno
            for value in calls
            if isinstance(value.func, ast.Attribute)
            and value.func.attr == "record"
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "copy_event"
        )
        result = next(
            value
            for value in calls
            if isinstance(value.func, ast.Name)
            and value.func.id == "ModelExecutionResult"
        )
        self.assertLess(copied, recorded)
        self.assertLess(recorded, result.lineno)
        self.assertTrue(
            any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "topk_result"
                for keyword in result.keywords
            )
        )
        forward = method(
            RUNTIME / "execution/model_executor.py", "ModelExecutor", "_forward_step"
        )
        for returned in (
            value for value in ast.walk(forward) if isinstance(value, ast.Return)
        ):
            self.assertIsInstance(returned.value, ast.Tuple)
            self.assertEqual(len(returned.value.elts), 3)


if __name__ == "__main__":
    unittest.main()
