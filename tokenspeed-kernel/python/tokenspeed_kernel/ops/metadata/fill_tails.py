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

"""Prepared, grouped padding fills with runtime live lengths."""

import operator

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)


class FillTailPlan:
    """Bind independent INT32/INT64 buffer tails once, then refresh live counts.

    Args:
        buffers: Nonoverlapping contiguous one-dimensional integer tensors on
            one device. Their storage and geometry must stay unchanged.
        live_groups: Per-buffer index into the live-count sequence passed to
            run(). Groups must cover 0..N-1 with no holes, for 1 <= N <= 8.
        values: Per-buffer integer padding value, representable in its dtype.

    The plan retains the tensors. It never modifies their live prefixes and
    writes every tail on every call, including after the batch shrinks.
    """

    def __init__(self, buffers, *, live_groups, values):
        self._buffers = tuple(buffers)
        self._groups = tuple(operator.index(g) for g in live_groups)
        self._values = tuple(operator.index(v) for v in values)
        if (
            not self._buffers
            or len(self._buffers) != len(self._groups)
            or len(self._buffers) != len(self._values)
        ):
            raise ValueError(
                "Padding buffer, group and value counts must match and be nonempty"
            )
        groups = sorted(set(self._groups))
        if groups != list(range(len(groups))) or len(groups) > 8:
            raise ValueError("Padding live groups must cover 0..N-1, with N <= 8")
        device = self._buffers[0].device
        limits = [None] * len(groups)
        for buf, group, value in zip(self._buffers, self._groups, self._values):
            if buf.ndim != 1 or buf.stride(0) != 1:
                raise ValueError("Padding buffers must be contiguous vectors")
            if buf.dtype not in (torch.int32, torch.int64) or buf.device != device:
                raise ValueError("Padding buffers must be colocated INT32/INT64")
            info = torch.iinfo(buf.dtype)
            if not info.min <= value <= info.max:
                raise ValueError("Padding value exceeds buffer dtype")
            limits[group] = (
                buf.numel()
                if limits[group] is None
                else min(limits[group], buf.numel())
            )
        spans = sorted(
            (buf.data_ptr(), buf.data_ptr() + buf.numel() * buf.element_size())
            for buf in self._buffers
            if buf.numel()
        )
        if any(end > begin for (_, end), (begin, _) in zip(spans, spans[1:])):
            raise ValueError("Padding buffers must not overlap")
        self._limits = tuple(limits)
        if device.type == "cpu":
            self._runner = _TorchFillTails(self._buffers, self._groups, self._values)
        else:
            factory = select_kernel(
                "metadata",
                "prepare_fill_tails",
                format_signature(metadata=dense_tensor_format(torch.int64)),
            )
            self._runner = factory(self._buffers, self._groups, self._values)

    def run(self, live_counts) -> None:
        """Write tails using all live-group lengths; leave live prefixes intact."""
        counts = tuple(operator.index(x) for x in live_counts)
        if len(counts) != len(self._limits) or any(
            not 0 <= n <= limit for n, limit in zip(counts, self._limits)
        ):
            raise ValueError("Padding live counts exceed prepared capacities")
        self._runner.run(counts)


class _TorchFillTails:
    def __init__(self, buffers, groups, values):
        self._entries = tuple(zip(buffers, groups, values))

    def run(self, counts):
        for buf, group, value in self._entries:
            buf[counts[group] :].fill_(value)


class _TritonFillTails:
    def __init__(self, buffers, groups, values):
        from tokenspeed_kernel.ops.metadata.prep_tape import PrepTape, Reg

        self._keys = tuple(Reg(i + 8) for i in range(max(groups) + 1))
        self._tape = PrepTape(buffers[0].device)
        for buf, group, value in zip(buffers, groups, values):
            self._tape.filltail(buf, self._keys[group], buf.numel(), value)
        self._tape.finalize()

    def run(self, counts):
        self._tape.run(dict(zip(self._keys, counts)))


@register_kernel(
    "metadata",
    "prepare_fill_tails",
    name="triton_prepare_fill_tails",
    solution="triton",
    signatures=format_signatures("metadata", "dense", {torch.int64}),
    priority=Priority.PORTABLE,
)
def triton_prepare_fill_tails(buffers, groups, values):
    return _TritonFillTails(buffers, groups, values)
