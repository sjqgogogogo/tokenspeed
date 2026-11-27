# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
#
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

"""Per-round destination load statistics, independent of expert migration."""

from __future__ import annotations

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.moe import bind_moe_local_expert_counts


class MoELoadStatistics:
    """Collect one counter matrix per model on the execution stream.

    Counters are int32 [layers, local physical experts]. Every rank advances
    once per global forward, including idle forwards. Startup graph work is
    discarded by the first ``begin_round``. ``step`` is outside captured graphs;
    its periodic gather and host copies use the caller's stream without a wait.
    """

    def __init__(
        self,
        counters: tuple[torch.Tensor, ...],
        *,
        interval: int,
        group: dist.ProcessGroup,
    ) -> None:
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError("Load statistics interval must be a positive integer")
        if not counters or any(
            c.ndim != 2
            or min(c.shape) <= 0
            or c.dtype != torch.int32
            or c.device != counters[0].device
            for c in counters
        ):
            raise ValueError(
                "Load counters must be nonempty int32 matrices on one device"
            )
        self.counters = counters
        self.interval = interval
        self.group = group
        self.world_size = dist.get_world_size(group)
        if self.world_size <= 1:
            raise ValueError("Load statistics require multiple expert ranks")
        self.index = 0
        self.rounds = 0
        self.flushes = 0
        self.started = False
        self.history = tuple(
            torch.zeros((interval, c.shape[0]), dtype=torch.int64, device=c.device)
            for c in counters
        )
        self.load_balance_ratios = tuple(
            torch.empty(
                2, dtype=torch.float32, device="cpu", pin_memory=c.device.type != "cpu"
            )
            for c in counters
        )

    def begin_round(self) -> None:
        if not self.started:
            for counter in self.counters:
                counter.zero_()
            self.started = True

    def step(self) -> bool:
        """Accumulate this round, returning whether the periodic gather ran.

        Host ratio tensors contain [sum(mean rank load), sum(max rank load)]
        across steps and layers after the caller's stream completes.
        """
        if not self.started:
            raise RuntimeError("begin_round must precede the model forward")
        flush = self.index + 1 == self.interval
        for counts, history, ratio in zip(
            self.counters, self.history, self.load_balance_ratios, strict=True
        ):
            torch.sum(counts, dim=-1, dtype=torch.int64, out=history[self.index])
            if flush:
                gathered = torch.empty(
                    (self.world_size * self.interval, counts.shape[0]),
                    dtype=torch.int64,
                    device=counts.device,
                )
                dist.all_gather_into_tensor(gathered, history, group=self.group)
                loads = gathered.view(
                    self.world_size, self.interval, counts.shape[0]
                ).float()
                ratio[0].copy_(loads.mean(dim=0).sum(), non_blocking=True)
                ratio[1].copy_(loads.max(dim=0).values.sum(), non_blocking=True)
            counts.zero_()
        self.rounds += 1
        self.index = 0 if flush else self.index + 1
        self.flushes += int(flush)
        return flush


def build_moe_load_statistics(
    models, *, interval: int, group, device
) -> MoELoadStatistics:
    """Bind target/draft counters through each model's selected MoE providers."""
    from tokenspeed.runtime.layers.moe.expert import MoELayer

    counters = []
    for model in models:
        layers = [layer for layer in model.modules() if isinstance(layer, MoELayer)]
        if not layers or len({layer.num_local_experts for layer in layers}) != 1:
            raise ValueError(
                "Load statistics require a uniform local expert count per model"
            )
        counter = torch.zeros(
            (len(layers), layers[0].num_local_experts), dtype=torch.int32, device=device
        )
        for layer, row in zip(layers, counter, strict=True):
            bind_moe_local_expert_counts(layer.plan, row)
        counters.append(counter)
    return MoELoadStatistics(tuple(counters), interval=interval, group=group)
