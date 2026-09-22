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

"""CPU coverage of the direct Engine API, not GPU score computation.

Execute production Engine methods and the real synchronous LLM bridge without
importing GPU modules. Only request construction and AsyncLLM are stubbed.
"""

from __future__ import annotations

import ast
import asyncio
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "python/tokenspeed/runtime"


def _load_class(path, class_name, methods, namespace):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    cls.bases = []
    if methods is not None:
        cls.body = [n for n in cls.body if getattr(n, "name", None) in methods]
        assert {n.name for n in cls.body} == methods
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        cls,
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[class_name]


Engine = _load_class(
    RUNTIME / "entrypoints/engine.py",
    "Engine",
    {"generate", "async_generate"},
    {"GenerateReqInput": SimpleNamespace, "asyncio": asyncio},
)
LLM = _load_class(
    RUNTIME / "engine/llm.py",
    "LLM",
    None,
    {
        "asyncio": asyncio,
        "threading": threading,
        "queue": queue,
        "_STREAM_END": object(),
    },
)


class _AsyncLLM:
    def __init__(self, result, error):
        self.result = result
        self.error = error
        self.requests = []

    async def generate_request(self, obj):
        self.requests.append(obj)
        if self.error is not None:
            raise self.error
        yield self.result


def _result():
    return {
        "output_ids": [4, 2],
        "meta_info": {
            "input_token_logprobs": [[None, 1, None], [-0.4, 3, None]],
            "input_top_logprobs": [None, [[-0.4, 3, None], [-1.3, 5, None]]],
            "output_token_logprobs": [[-0.2, 4, None], [-0.3, 2, None]],
            "output_top_logprobs": [
                [[-0.2, 4, None], [-2.0, 6, None]],
                [[-0.3, 2, None], [-1.9, 7, None]],
            ],
        },
    }


def _arguments(start, count):
    return dict(
        prompt=None,
        input_ids=[1, 3],
        sampling_params={
            "max_new_tokens": 2,
            "temperature": 0,
            "no_stop_trim": True,
            "skip_special_tokens": False,
        },
        return_logprob=True,
        logprob_start_len=start,
        top_logprobs_num=count,
        token_ids_logprob=None,
        return_text_in_logprobs=False,
        logprob_format="sglang",
        custom_logit_processor=None,
        return_hidden_states=False,
        stream=False,
        bootstrap_host=None,
        bootstrap_port=None,
        bootstrap_room=None,
        data_parallel_rank=None,
    )


@pytest.fixture
def engine():
    instance = Engine()
    instance.tokenizer_manager = _AsyncLLM(_result(), None)
    instance.llm = LLM(instance.tokenizer_manager)
    try:
        yield instance
    finally:
        instance.llm.run(instance.llm._loop.shutdown_asyncgens())
        instance.llm.shutdown()
        instance.llm._loop.close()


@pytest.mark.parametrize("use_async", [False, True])
def test_four_fields_and_token_ids_survive_direct_engine_api(engine, use_async):
    args = _arguments(0, 2)
    if use_async:
        args.update(
            input_embeds=None,
            input_multi_ids=None,
            input_extra_infos=None,
            user_rid=None,
        )
        result = asyncio.run(engine.async_generate(**args))
    else:
        result = engine.generate(**args)
    assert result == _result()
    assert result is engine.tokenizer_manager.result
    request = engine.tokenizer_manager.requests[0]
    for name in (
        "input_ids",
        "sampling_params",
        "return_logprob",
        "logprob_start_len",
        "top_logprobs_num",
        "logprob_format",
        "stream",
    ):
        assert getattr(request, name) == args[name]
    assert request.return_text_in_logprobs is False


@pytest.mark.parametrize("start,count", [(0, 0), (1, 10), (-1, 10)])
def test_start_and_candidate_count_are_not_rewritten(engine, start, count):
    engine.generate(**_arguments(start, count))
    request = engine.tokenizer_manager.requests[0]
    assert (request.logprob_start_len, request.top_logprobs_num) == (start, count)


@pytest.mark.parametrize("use_async", [False, True])
def test_nonstream_engine_errors_are_not_swallowed(engine, use_async):
    error = ValueError("prompt logprobs are disabled")
    engine.tokenizer_manager.error = error
    args = _arguments(0, 2)
    with pytest.raises(ValueError, match="prompt logprobs are disabled") as caught:
        if use_async:
            args.update(
                input_embeds=None,
                input_multi_ids=None,
                input_extra_infos=None,
                user_rid=None,
            )
            asyncio.run(engine.async_generate(**args))
        else:
            engine.generate(**args)
    assert caught.value is error
