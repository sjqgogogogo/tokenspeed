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

"""Probe Kineto/CUPTI behavior across a custom long-lived CUDA worker thread.

This intentionally has no TokenSpeed runtime/model/distributed dependency. It
models the post-0da906d0 execution topology: the main/control thread captures a
CUDA graph and handles runtime profiler control, while a custom worker thread
launches eager CUDA work or replays that pre-captured graph. ``legacy`` runs
the old completed CPU-only warmup before capture; ``prepared`` calls
``prepare_trace`` before capture and starts that same session afterward.

Run every case in a fresh process so one Kineto session cannot contaminate the
next::

    CUDA_VISIBLE_DEVICES=0 python test/runtime/profiler_thread_probe.py

The output is one JSON object per (profiler placement, workload) case. A usable
GPU trace has ``kernel > 0`` and ``cuda_runtime > 0``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import queue
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path

import torch


CASES = (
    "same",
    "control",
    "worker",
    "global",
    "child_attach",
    "legacy",
    "prepared",
)
WORKLOADS = ("eager", "graph")

_ENABLE_CHILD_SYMBOL = "_ZN5torch8autograd8profiler27enableProfilerInChildThreadEv"
_DISABLE_CHILD_SYMBOL = "_ZN5torch8autograd8profiler28disableProfilerInChildThreadEv"


class Worker:
    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        torch.cuda.set_device(self._device)
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, future = item
            try:
                future.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - relayed to main
                future.set_exception(exc)

    def run(self, fn):
        future = Future()
        self._queue.put((fn, future))
        return future.result()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()


def _child_profiler_hook(symbol_name: str):
    process = ctypes.CDLL(None)
    try:
        hook = getattr(process, symbol_name)
    except AttributeError:
        lib_dir = Path(torch.__file__).parent / "lib"
        matches = sorted(lib_dir.glob("libtorch_cpu.*"))
        if not matches:
            raise RuntimeError(f"libtorch_cpu not found below {lib_dir}") from None
        library = ctypes.CDLL(str(matches[0]), mode=ctypes.RTLD_GLOBAL)
        hook = getattr(library, symbol_name)
    hook.argtypes = []
    hook.restype = None
    return hook


def _make_workload(
    kind: str,
    iterations: int,
    *,
    legacy_init: bool = False,
    before_capture: Callable[[], None] | None = None,
):
    device = torch.device("cuda", 0)
    lhs = torch.randn((1024, 1024), device=device)
    rhs = torch.randn((1024, 1024), device=device)
    out = torch.empty_like(lhs)

    if legacy_init:
        # The CUDA < 12 workaround used by TokenSpeed before this probe split
        # the profiler lifecycle cases. On torch 2.13+cu129 it leaves the next
        # Kineto session without CUDA runtime/kernel activities.
        from torch.profiler._utils import _init_for_cuda_graphs

        _init_for_cuda_graphs()
    torch.mm(lhs, rhs, out=out)
    torch.cuda.synchronize()
    if before_capture is not None:
        before_capture()

    if kind == "eager":

        def run():
            for _ in range(iterations):
                torch.mm(lhs, rhs, out=out)
            torch.cuda.synchronize()

        return run

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.mm(lhs, rhs, out=out)

    def replay():
        for _ in range(iterations):
            graph.replay()
        torch.cuda.synchronize()

    return replay


def _start_profiler(*, all_threads: bool = False):
    kwargs = {}
    if all_threads:
        kwargs["experimental_config"] = torch.profiler._ExperimentalConfig(
            profile_all_threads=True
        )
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=False,
        record_shapes=False,
        **kwargs,
    )
    profiler.start()
    return profiler


def _prepare_profiler():
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=False,
        record_shapes=False,
        experimental_config=torch.profiler._ExperimentalConfig(
            profile_all_threads=True
        ),
    )
    profiler.prepare_trace()
    return profiler


def _trace_counts(path: Path) -> dict[str, int]:
    trace = json.loads(path.read_text())
    events = trace.get("traceEvents", [])
    categories = {}
    for event in events:
        category = str(event.get("cat", ""))
        categories[category] = categories.get(category, 0) + 1
    names = [str(event.get("name", "")) for event in events]
    return {
        "events": len(events),
        "kernel": categories.get("kernel", 0),
        "cuda_runtime": categories.get("cuda_runtime", 0),
        "gpu_memcpy": categories.get("gpu_memcpy", 0),
        "cuda_graph_launch": sum(name == "cudaGraphLaunch" for name in names),
        "aten_mm": sum(name == "aten::mm" for name in names),
    }


def run_case(case: str, workload_kind: str, iterations: int, output_dir: Path) -> dict:
    torch.cuda.set_device(0)
    prepared_profilers = []

    def prepare_profiler():
        prepared_profilers.append(_prepare_profiler())

    workload = _make_workload(
        workload_kind,
        iterations,
        legacy_init=case == "legacy",
        before_capture=prepare_profiler if case == "prepared" else None,
    )
    profiler = prepared_profilers[0] if prepared_profilers else None
    worker = Worker(torch.device("cuda", 0))
    trace_path = output_dir / f"{case}-{workload_kind}.json"
    try:
        if case == "same":
            profiler = _start_profiler()
            workload()
            profiler.stop()
        elif case == "control":
            profiler = _start_profiler()
            worker.run(workload)
            profiler.stop()
        elif case == "worker":
            profiler = worker.run(_start_profiler)
            worker.run(workload)
            worker.run(profiler.stop)
        elif case == "global":
            profiler = _start_profiler(all_threads=True)
            worker.run(workload)
            profiler.stop()
        elif case == "child_attach":
            enable = _child_profiler_hook(_ENABLE_CHILD_SYMBOL)
            disable = _child_profiler_hook(_DISABLE_CHILD_SYMBOL)
            profiler = _start_profiler()
            worker.run(enable)
            worker.run(workload)
            worker.run(disable)
            profiler.stop()
        elif case == "legacy":
            profiler = _start_profiler()
            workload()
            profiler.stop()
        elif case == "prepared":
            profiler.start_trace()
            worker.run(workload)
            profiler.stop_trace()
        else:
            raise ValueError(case)
        profiler.export_chrome_trace(str(trace_path))
        return {
            "case": case,
            "workload": workload_kind,
            "status": "ok",
            "torch": torch.__version__,
            "supported": sorted(
                activity.name for activity in torch.profiler.supported_activities()
            ),
            **_trace_counts(trace_path),
            "trace": str(trace_path),
        }
    finally:
        worker.close()


def run_matrix(args) -> int:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="ts-profiler-probe-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for case in CASES:
        for workload in WORKLOADS:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case",
                case,
                "--workload",
                workload,
                "--iterations",
                str(args.iterations),
                "--output-dir",
                str(output_dir),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip(), flush=True)
            else:
                failed = True
                print(
                    json.dumps(
                        {
                            "case": case,
                            "workload": workload,
                            "status": "failed",
                            "returncode": result.returncode,
                            "stdout": result.stdout[-2000:],
                            "stderr": result.stderr[-4000:],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    print(json.dumps({"output_dir": str(output_dir)}), flush=True)
    return int(failed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--workload", choices=WORKLOADS)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.case is None:
        return run_matrix(args)
    if args.workload is None:
        raise SystemExit("--workload is required with --case")
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="ts-profiler-probe-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            run_case(args.case, args.workload, args.iterations, output_dir),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
