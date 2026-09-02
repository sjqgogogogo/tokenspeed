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

"""Probe prepared Kineto sessions across DP-style CUDA graph collectives.

Two ranks capture the same fixed-shape NCCL all-reduce graph. Runtime rounds
alternate which rank contributes the nonzero "active" value, approximating a
DP active/idle role swap without TokenSpeed model or scheduler dependencies.

Cases:

* ``baseline``: capture and replay with no profiler interaction.
* ``prepared``: call ``prepare_trace`` before capture but never start it.
* ``prepared_profile``: prepare before capture, then start/stop that same trace
  around the alternating replays.

Run on a host with at least two visible GPUs::

    CUDA_VISIBLE_DEVICES=0,1 python test/runtime/profiler_dp_graph_probe.py
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


CASES = ("baseline", "prepared", "prepared_profile")
WORLD_SIZE = 2


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
    events = json.loads(path.read_text()).get("traceEvents", [])
    categories: dict[str, int] = {}
    for event in events:
        category = str(event.get("cat", ""))
        categories[category] = categories.get(category, 0) + 1
    names = [str(event.get("name", "")) for event in events]
    return {
        "events": len(events),
        "kernel": categories.get("kernel", 0),
        "cuda_runtime": categories.get("cuda_runtime", 0),
        "cuda_graph_launch": sum(name == "cudaGraphLaunch" for name in names),
    }


def _worker(
    rank: int,
    init_method: str,
    case: str,
    rounds: int,
    output_dir: str,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=WORLD_SIZE,
    )
    device = torch.device("cuda", rank)
    value = torch.zeros(1, dtype=torch.float32, device=device)

    # Initialize the communicator before either profiling or graph capture.
    for _ in range(3):
        value.fill_(rank)
        dist.all_reduce(value)
    torch.cuda.synchronize(device)
    dist.barrier()

    profiler = _prepare_profiler() if case != "baseline" else None

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph, stream=capture_stream):
        dist.all_reduce(value)
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    torch.cuda.synchronize(device)
    dist.barrier()

    profiling = case == "prepared_profile"
    if profiling:
        profiler.start_trace()

    for round_idx in range(rounds):
        active_rank = round_idx % WORLD_SIZE
        value.fill_(1.0 if rank == active_rank else 0.0)
        graph.replay()
        torch.cuda.synchronize(device)
        observed = float(value.item())
        if observed != 1.0:
            raise AssertionError(
                f"rank {rank} round {round_idx}: all-reduce produced {observed}"
            )
        dist.barrier()

    summary = {
        "case": case,
        "rank": rank,
        "rounds": rounds,
        "status": "ok",
        "torch": torch.__version__,
    }
    if profiling:
        profiler.stop_trace()
        trace_path = Path(output_dir) / f"{case}-rank{rank}.json"
        profiler.export_chrome_trace(str(trace_path))
        summary.update(_trace_counts(trace_path))
        summary["trace"] = str(trace_path)

    (Path(output_dir) / f"{case}-rank{rank}-summary.json").write_text(
        json.dumps(summary, sort_keys=True)
    )
    dist.destroy_process_group()


def run_case(case: str, rounds: int, output_dir: Path, timeout: float) -> list[dict]:
    rendezvous = output_dir / f"{case}.rdzv"
    summaries = [output_dir / f"{case}-rank{rank}-summary.json" for rank in range(2)]
    for path in (rendezvous, *summaries):
        path.unlink(missing_ok=True)

    context = mp.spawn(
        _worker,
        args=(rendezvous.as_uri(), case, rounds, str(output_dir)),
        nprocs=WORLD_SIZE,
        join=False,
    )
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if context.join(timeout=1.0):
                return [json.loads(path.read_text()) for path in summaries]
        raise TimeoutError(f"{case} exceeded {timeout:.0f}s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if torch.cuda.device_count() < WORLD_SIZE:
        raise SystemExit("at least two visible CUDA devices are required")

    output_dir = Path(
        args.output_dir or tempfile.mkdtemp(prefix="ts-profiler-dp-graph-")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = (args.case,) if args.case is not None else CASES
    failed = False
    for case in cases:
        try:
            summaries = run_case(case, args.rounds, output_dir, args.timeout)
            print(json.dumps({"case": case, "ranks": summaries}, sort_keys=True))
        except BaseException as exc:  # noqa: BLE001 - standalone process boundary
            failed = True
            print(
                json.dumps(
                    {
                        "case": case,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    sort_keys=True,
                )
            )
    print(json.dumps({"output_dir": str(output_dir)}, sort_keys=True))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
