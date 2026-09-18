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
"""Real CUDA graph/stream lifetime regression, without model downloads.

Run on CUDA with pytest. This checks diagnostic buffer ownership; a real-model
HTTP/CI run is still needed to validate attention and the complete scheduler.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

COMMON = runpy.run_path(str(Path(__file__).with_name("test_top_logprob_capture.py")))
HELPER = COMMON["HELPER"]
Config = COMMON["Config"]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_graph_results_survive_multiple_in_flight_replays():
    stream = torch.cuda.Stream()
    weights_cpu = torch.arange(28, dtype=torch.float32).reshape(4, 7).sin() * 0.5
    weights = weights_cpu.cuda()
    inputs = torch.eye(4, device="cuda")
    storage = torch.empty(4, 7, dtype=torch.float32, device="cuda")
    snapshot = HELPER.RawLogitsSnapshot(storage)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            logits = inputs @ weights
            snapshot.capture(
                SimpleNamespace(next_token_logits=logits, logits_layout_plan=None)
            )
            logits.zero_()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        logits = inputs @ weights
        snapshot.capture(
            SimpleNamespace(next_token_logits=logits, logits_layout_plan=None)
        )
        logits.zero_()  # A sampler is permitted to mutate the original logits.

    pending = []
    with torch.cuda.stream(stream):
        for shift, bs in ((0, 3), (1, 1), (2, 4)):
            source = torch.eye(4).roll(shift, 0)
            inputs.copy_(source.to("cuda"))
            graph.replay()
            capture = HELPER.TopLogprobCapture(
                tuple(Config(True, -1, 3, ()) for _ in range(bs)), 0, (1,) * bs, 7
            )
            capture.capture_replayed_logits(storage[:bs])
            result = capture.copy_to_cpu()
            done = torch.cuda.Event()
            done.record(stream)
            expected = torch.log_softmax((source @ weights_cpu)[:bs], -1).topk(3, -1)
            pending.append((done, result, expected))
    # No per-request drain: the CPU consumes all results after later graphs
    # have been queued against the very same snapshot storage.
    for done, result, expected in pending:
        done.synchronize()
        torch.testing.assert_close(
            result["output_top_logprobs_val"][:, 0],
            expected.values,
            rtol=1e-5,
            atol=1e-6,
        )
        assert torch.equal(result["output_top_logprobs_idx"][:, 0], expected.indices)
