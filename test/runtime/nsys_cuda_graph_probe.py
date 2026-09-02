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

"""Isolate Nsight/CUDA Graph failures in the post-0da906d0 execution topology.

Each case must run under a separate ``nsys profile`` process because
``--capture-range-end=stop`` consumes one CUDA Profiler API capture range::

    export CUDA_INJECTION_SHM_ALLOWED=TRUE

    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --sample=none --cpuctxsw=none --force-overwrite=true \
      -o /tmp/nsys-thread-control \
      python test/runtime/nsys_cuda_graph_probe.py --case thread-control

    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --sample=none --cpuctxsw=none --force-overwrite=true \
      -o /tmp/nsys-thread-worker \
      python test/runtime/nsys_cuda_graph_probe.py --case thread-worker

    CUDA_VISIBLE_DEVICES=0,1 nsys profile \
      --trace=cuda,nvtx --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --sample=none --cpuctxsw=none --force-overwrite=true \
      -o /tmp/nsys-nccl-worker \
      python test/runtime/nsys_cuda_graph_probe.py --case nccl-worker

Cases:

* ``thread-control`` captures a graph on the control thread, starts/stops the
  CUDA profiler there, and replays the graph on a persistent worker thread.
* ``thread-worker`` captures on the control thread, but profiler control and
  graph replay all execute on the persistent worker thread.
* ``nccl-worker`` repeats the worker-controlled lifecycle in two spawned
  processes with one captured NCCL all-reduce graph per GPU.

The probes use little memory, synchronize after every replay, and emit one
JSON stage record before and after each operation so the last successful stage
is visible even if a CUDA context fails.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CASES = ("thread-control", "thread-worker", "nccl-worker")
NCCL_WORLD_SIZE = 2


def _stage(case: str, stage: str, *, rank: int | None = None, **extra: Any) -> None:
    payload = {
        "case": case,
        "pid": os.getpid(),
        "rank": rank,
        "stage": stage,
        "thread": threading.current_thread().name,
        **extra,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


class CudaWorker:
    """One persistent CUDA worker matching TokenSpeed's ForwardThread."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._loop,
            name="nsys-probe-worker",
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        torch.cuda.set_device(self._device)
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - relayed to caller
                future.set_exception(exc)

    def run(self, fn: Callable[[], Any], *, timeout: float | None = None) -> Any:
        future: Future = Future()
        self._queue.put((fn, future))
        return future.result(timeout=timeout)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise TimeoutError("CUDA worker did not stop within 30 seconds")


def _capture_mm_graph(case: str, device: torch.device):
    torch.cuda.set_device(device)
    torch.manual_seed(1)
    lhs = torch.randn((1024, 1024), device=device)
    rhs = torch.randn((1024, 1024), device=device)
    out = torch.empty_like(lhs)

    torch.mm(lhs, rhs, out=out)
    torch.cuda.synchronize(device)
    _stage(case, "warmup_complete")

    graph = torch.cuda.CUDAGraph()
    _stage(case, "graph_capture_begin")
    with torch.cuda.graph(graph):
        torch.mm(lhs, rhs, out=out)
    torch.cuda.synchronize(device)
    _stage(case, "graph_capture_complete")
    return graph, lhs, rhs, out


def _profiler_start(case: str, device: torch.device, *, rank: int | None = None):
    _stage(case, "profiler_start_begin", rank=rank)
    torch.cuda.synchronize(device)
    status = torch.cuda.cudart().cudaProfilerStart()
    _stage(case, "profiler_start_complete", rank=rank, status=str(status))


def _profiler_stop(case: str, device: torch.device, *, rank: int | None = None):
    _stage(case, "profiler_stop_begin", rank=rank)
    torch.cuda.synchronize(device)
    status = torch.cuda.cudart().cudaProfilerStop()
    _stage(case, "profiler_stop_complete", rank=rank, status=str(status))


def _raw_profiler_stop(case: str, *, rank: int | None = None) -> None:
    """Best-effort capture finalization after a replay error."""
    try:
        status = torch.cuda.cudart().cudaProfilerStop()
        _stage(case, "profiler_cleanup_stop", rank=rank, status=str(status))
    except BaseException as exc:  # noqa: BLE001 - CUDA context may be poisoned
        _stage(
            case,
            "profiler_cleanup_stop_failed",
            rank=rank,
            error=f"{type(exc).__name__}: {exc}",
        )


def _replay_mm_graph(
    case: str,
    graph: torch.cuda.CUDAGraph,
    device: torch.device,
    iterations: int,
) -> None:
    for iteration in range(iterations):
        _stage(case, "graph_replay_begin", iteration=iteration)
        graph.replay()
        torch.cuda.synchronize(device)
        _stage(case, "graph_replay_complete", iteration=iteration)


def _run_thread_case(case: str, iterations: int, timeout_seconds: float) -> None:
    device = torch.device("cuda", 0)
    graph, lhs, rhs, out = _capture_mm_graph(case, device)
    worker = CudaWorker(device)
    profiler_started = False
    run_profiler_op = (
        (lambda fn: fn())
        if case == "thread-control"
        else lambda fn: worker.run(fn, timeout=timeout_seconds)
    )
    try:
        run_profiler_op(lambda: _profiler_start(case, device))
        profiler_started = True
        worker.run(
            lambda: _replay_mm_graph(case, graph, device, iterations),
            timeout=timeout_seconds,
        )
        run_profiler_op(lambda: _profiler_stop(case, device))
        profiler_started = False
        _stage(
            case,
            "case_complete",
            iterations=iterations,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            checksum=float(out[0, 0].item()),
        )
    finally:
        if profiler_started:
            try:
                if case == "thread-control":
                    _raw_profiler_stop(case)
                else:
                    worker.run(
                        lambda: _raw_profiler_stop(case),
                        timeout=min(timeout_seconds, 5),
                    )
            except BaseException as exc:  # noqa: BLE001 - cleanup only
                _stage(
                    case,
                    "profiler_cleanup_dispatch_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
        worker.close()
        del graph, lhs, rhs, out


def _nccl_child(
    rank: int,
    init_method: str,
    iterations: int,
    timeout_seconds: float,
) -> None:
    case = "nccl-worker"
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=NCCL_WORLD_SIZE,
        timeout=timedelta(seconds=timeout_seconds),
        device_id=device,
    )
    _stage(case, "process_group_ready", rank=rank)

    value = torch.zeros(1, dtype=torch.float32, device=device)
    value.fill_(rank)
    dist.all_reduce(value)
    torch.cuda.synchronize(device)
    _stage(case, "communicator_warm", rank=rank)

    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    _stage(case, "graph_capture_begin", rank=rank)
    with torch.cuda.graph(graph):
        dist.all_reduce(value)
    torch.cuda.synchronize(device)
    _stage(case, "graph_capture_complete", rank=rank)
    dist.barrier()

    worker = CudaWorker(device)
    profiler_started = False
    try:
        worker.run(
            lambda: _profiler_start(case, device, rank=rank),
            timeout=timeout_seconds,
        )
        profiler_started = True
        dist.barrier()

        def replay_collectives() -> None:
            for iteration in range(iterations):
                active_rank = iteration % NCCL_WORLD_SIZE
                _stage(
                    case,
                    "graph_replay_begin",
                    rank=rank,
                    iteration=iteration,
                    active_rank=active_rank,
                )
                value.fill_(1.0 if rank == active_rank else 0.0)
                graph.replay()
                torch.cuda.synchronize(device)
                observed = float(value.item())
                if observed != 1.0:
                    raise AssertionError(
                        f"rank {rank} iteration {iteration}: observed {observed}"
                    )
                _stage(
                    case,
                    "graph_replay_complete",
                    rank=rank,
                    iteration=iteration,
                    observed=observed,
                )

        worker.run(replay_collectives, timeout=timeout_seconds)
        dist.barrier()
        worker.run(
            lambda: _profiler_stop(case, device, rank=rank),
            timeout=timeout_seconds,
        )
        profiler_started = False
        _stage(
            case,
            "rank_complete",
            rank=rank,
            iterations=iterations,
            torch=torch.__version__,
            cuda=torch.version.cuda,
        )
    finally:
        if profiler_started:
            try:
                worker.run(
                    lambda: _raw_profiler_stop(case, rank=rank),
                    timeout=min(timeout_seconds, 5),
                )
            except BaseException as exc:  # noqa: BLE001 - cleanup only
                _stage(
                    case,
                    "profiler_cleanup_dispatch_failed",
                    rank=rank,
                    error=f"{type(exc).__name__}: {exc}",
                )
        worker.close()
        del graph, value
        try:
            torch.cuda.synchronize(device)
        finally:
            dist.destroy_process_group()


def _run_nccl_case(iterations: int, timeout_seconds: float) -> None:
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    with tempfile.TemporaryDirectory(prefix="nsys-cuda-graph-probe-") as directory:
        rendezvous = (Path(directory) / "nccl.rdzv").resolve()
        context = mp.spawn(
            _nccl_child,
            args=(rendezvous.as_uri(), iterations, timeout_seconds),
            nprocs=NCCL_WORLD_SIZE,
            join=False,
        )
        deadline = time.monotonic() + timeout_seconds
        try:
            while time.monotonic() < deadline:
                if context.join(timeout=1.0):
                    _stage(
                        "nccl-worker",
                        "case_complete",
                        iterations=iterations,
                        world_size=NCCL_WORLD_SIZE,
                    )
                    return
            raise TimeoutError(
                f"nccl-worker exceeded the {timeout_seconds:.0f}-second timeout"
            )
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = _parse_args()
    required_gpus = NCCL_WORLD_SIZE if args.case == "nccl-worker" else 1
    if torch.cuda.device_count() < required_gpus:
        raise SystemExit(f"{args.case} requires {required_gpus} visible CUDA GPU(s)")

    if args.case in {"thread-control", "thread-worker"}:
        _run_thread_case(args.case, args.iterations, args.timeout)
    else:
        _run_nccl_case(args.iterations, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
