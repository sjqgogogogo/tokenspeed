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

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PreparedTorchProfiler:
    """A CUDA profiler prepared before graph capture but not started yet."""

    profiler: Any
    activities: frozenset[str]
    with_stack: bool
    record_shapes: bool

    def configuration_error(
        self,
        activities: list[str],
        with_stack: bool | None,
        record_shapes: bool | None,
    ) -> str | None:
        torch_activities = frozenset(
            activity for activity in activities if activity in {"CPU", "GPU"}
        )
        requested_with_stack = with_stack if with_stack is not None else True
        requested_record_shapes = record_shapes if record_shapes is not None else False
        if torch_activities != self.activities:
            return (
                "the first graph-mode profile must use activities "
                f"{sorted(self.activities)}"
            )
        if requested_with_stack != self.with_stack:
            return f"the first graph-mode profile must use with_stack={self.with_stack}"
        if requested_record_shapes != self.record_shapes:
            return (
                "the first graph-mode profile must use "
                f"record_shapes={self.record_shapes}"
            )
        return None


def prepare_torch_cuda_profiler() -> PreparedTorchProfiler | None:
    """Prepare a first runtime profiler before serving graph capture.

    ``prepare_trace`` initializes Kineto/CUPTI but deliberately does not start
    or stop a session. The same profiler is started by the first runtime
    ``/start_profile`` request, avoiding both post-capture attachment and a
    pre-serving CUPTI teardown.
    """
    cuda_activity = torch.profiler.ProfilerActivity.CUDA
    if (
        torch.version.cuda is None
        or cuda_activity not in torch.profiler.supported_activities()
    ):
        return None

    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, cuda_activity],
        with_stack=False,
        record_shapes=False,
        experimental_config=torch.profiler._ExperimentalConfig(
            profile_all_threads=True
        ),
    )
    profiler.prepare_trace()
    return PreparedTorchProfiler(
        profiler=profiler,
        activities=frozenset({"CPU", "GPU"}),
        with_stack=False,
        record_shapes=False,
    )
