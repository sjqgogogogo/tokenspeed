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

import torch


def prime_torch_cuda_profiler() -> None:
    """Initialize CUDA profiling before serving CUDA graphs are captured.

    The legacy ``torch.profiler._utils._init_for_cuda_graphs`` helper opens a
    CPU-only profiler session. On modern PyTorch/CUDA builds that session can
    prevent every later Kineto session from collecting CUDA activities. An
    explicit CUDA-aware session initializes CUPTI without poisoning subsequent
    runtime profiles.
    """
    cuda_activity = torch.profiler.ProfilerActivity.CUDA
    if (
        torch.version.cuda is None
        or cuda_activity not in torch.profiler.supported_activities()
    ):
        return

    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, cuda_activity],
        with_stack=False,
        record_shapes=False,
    )
    profiler.start()
    profiler.stop()
