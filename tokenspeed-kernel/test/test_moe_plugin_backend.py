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

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_plan
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.signature import format_signatures

pytestmark = pytest.mark.usefixtures("fresh_registry")


def _plan(backend):
    return moe_plan(
        "unquant", input_dtype=torch.float32, activation="swiglu",
        routing_mode="precomputed_topk", a2a_backend=backend,
        ep_size=2, ispp=16, hidden=8, swiglu_form="standard",
        activation_clamped=False, expert_id_repeats=False,
        fast_math=False, combine_order="rank", features=frozenset(),
        solution="fixture_collective",
    )


@pytest.mark.parametrize("collective", [False, True])
def test_registered_collective_backend_reaches_its_apply(collective):
    @register_kernel(
        "moe", "apply", name="fixture_collective_apply",
        solution="fixture_collective",
        signatures=format_signatures("x", "dense", {torch.float32}),
        traits={
            "weight_dtype": frozenset({"unquant"}),
            "supports_all_to_all_ep": frozenset({collective}),
            "a2a_backend": frozenset({"fixture_transport"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_ep": frozenset({True}),
        },
    )
    def apply(x):
        return -x

    if not collective:
        with pytest.raises(NotImplementedError, match="unsupported: fixture_transport"):
            _plan("fixture_transport")
        return
    plan = _plan("fixture_transport")
    x = torch.tensor([[1.0, -2.0]])
    torch.testing.assert_close(plan["apply_kernel"](x), -x)


def test_unregistered_collective_backend_is_rejected():
    with pytest.raises(NotImplementedError, match="unsupported: unregistered_transport"):
        _plan("unregistered_transport")
