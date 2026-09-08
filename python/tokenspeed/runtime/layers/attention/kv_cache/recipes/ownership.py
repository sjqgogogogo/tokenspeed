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

"""Layer ownership shared by PP construction and cache transfer planning.

Target layers are partitioned across stages. Continuation layers belong to
one producer on the last stage, so adding a draft never moves a target cut.
This module is device-independent; model construction and PD use the same
windows without importing execution or communication code.
"""

from __future__ import annotations

from dataclasses import dataclass


def target_stage_windows(
    num_layers: int,
    pp_size: int,
    partition: tuple[int, ...] | None,
) -> list[tuple[int, int]]:
    """Return target layer windows for every stage.

    Args:
        num_layers: Positive target layer count, excluding draft layers.
        pp_size: Number of nonempty pipeline stages.
        partition: Explicit target layers per stage, or None for even cuts.

    Returns:
        Ordered half-open windows covering exactly the target layers.
    """
    if (
        isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
        or num_layers < 1
    ):
        raise ValueError("target layer count must be a positive integer")
    if (
        isinstance(pp_size, bool)
        or not isinstance(pp_size, int)
        or not 1 <= pp_size <= num_layers
    ):
        raise ValueError("pipeline stages must each own at least one target layer")
    if partition is not None:
        if len(partition) != pp_size:
            raise ValueError(
                f"pp layer partition {partition} has {len(partition)} entries "
                f"for {pp_size} pipeline stages"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in partition
        ):
            raise ValueError(
                f"pp layer partition {partition} must give every stage at least one layer"
            )
        if sum(partition) != num_layers:
            raise ValueError(
                f"pp layer partition {partition} sums to {sum(partition)} "
                f"but the model has {num_layers} layers"
            )
        counts = partition
    else:
        base, remainder = divmod(num_layers, pp_size)
        counts = tuple(base + (stage < remainder) for stage in range(pp_size))
    windows = []
    start = 0
    for length in counts:
        windows.append((start, start + length))
        start += length
    return windows


@dataclass(frozen=True)
class CacheLayerOwnership:
    """One stage's target producers and optional trailing draft producer.

    ``target_window`` counts target execution steps. ``resident_window``
    additionally includes every continuation layer on the last stage; the
    draft fields become ready together at one final producer barrier.
    """

    num_target_layers: int
    num_draft_layers: int
    target_window: tuple[int, int]

    def __post_init__(self) -> None:
        for name, minimum in (("num_target_layers", 1), ("num_draft_layers", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        start, end = self.target_window
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (start, end)
            )
            or not 0 <= start < end <= self.num_target_layers
        ):
            raise ValueError("PP layer window is outside the target layer range")

    @property
    def owns_draft(self) -> bool:
        return (
            self.num_draft_layers > 0
            and self.target_window[1] == self.num_target_layers
        )

    @property
    def resident_window(self) -> tuple[int, int]:
        start, end = self.target_window
        return start, end + (self.num_draft_layers if self.owns_draft else 0)

    @property
    def producer_layers(self) -> tuple[tuple[int, ...], ...]:
        start, end = self.target_window
        steps = tuple((layer,) for layer in range(start, end))
        if self.owns_draft:
            steps += (tuple(range(end, end + self.num_draft_layers)),)
        return steps


def pipeline_cache_ownership(
    num_target_layers: int,
    num_draft_layers: int,
    pp_size: int,
    partition: tuple[int, ...] | None,
) -> tuple[CacheLayerOwnership, ...]:
    """Return one cache owner per stage, with draft fields only on the last.

    Args:
        num_target_layers: Number of target layers before continuation layers.
        num_draft_layers: Number of trailing draft cache layers.
        pp_size: Pipeline stage count, including one for an unpartitioned model.
        partition: Target layers per stage, or None for an even target split.

    Returns:
        Owners in stage order, sharing one logical target/draft layer count.
    """
    return tuple(
        CacheLayerOwnership(num_target_layers, num_draft_layers, window)
        for window in target_stage_windows(num_target_layers, pp_size, partition)
    )
