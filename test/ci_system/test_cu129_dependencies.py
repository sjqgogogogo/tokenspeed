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

"""CPU checks of cu129 dependency metadata and installer agreement."""

import importlib.util
import runpy
from pathlib import Path

import setuptools
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "tokenspeed-kernel/python/setup.py"


def _metadata(monkeypatch, variant):
    captured = {}
    monkeypatch.setenv("TOKENSPEED_KERNEL_BACKEND", "cuda")
    if variant is None:
        monkeypatch.delenv("TOKENSPEED_KERNEL_CUDA_VARIANT", raising=False)
    else:
        monkeypatch.setenv("TOKENSPEED_KERNEL_CUDA_VARIANT", variant)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(SETUP))
    return {req.name: req for req in map(Requirement, captured["install_requires"])}


def test_cu129_metadata_uses_compatible_pair_and_keeps_other_pins(monkeypatch):
    cuda = _metadata(monkeypatch, None)
    cu129 = _metadata(monkeypatch, "cu129")
    assert str(cu129["flashinfer-python"].specifier) == "==0.6.18"
    assert str(cu129["nvidia-cudnn-frontend"].specifier) == "==1.28.0"
    assert not cu129["nvidia-cutlass-dsl"].extras
    assert "nvidia-cutlass-dsl-libs-cu12" in cu129
    assert "nvidia-cutlass-dsl-libs-cu13" not in cu129
    assert "tokenspeed-cutedsl-kda" not in cu129
    changed = {
        "flashinfer-python",
        "nvidia-cudnn-frontend",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-cu13",
        "tokenspeed-cutedsl-kda",
    }
    for name, requirement in cuda.items():
        if name not in changed:
            assert str(cu129[name]) == str(requirement)


def test_default_cuda_metadata_still_uses_07_and_cu13(monkeypatch):
    cuda = _metadata(monkeypatch, None)
    assert str(cuda["flashinfer-python"].specifier) == "==0.7.0"
    assert str(cuda["nvidia-cudnn-frontend"].specifier) == "==1.29.0"
    assert cuda["nvidia-cutlass-dsl"].extras == {"cu13"}


def test_cu129_constraints_match_metadata_and_cubin(monkeypatch):
    requirements = _metadata(monkeypatch, "cu129")
    constraints = {
        req.name: req
        for raw_line in (ROOT / "requirements/nvidia-cu129-constraints.txt")
        .read_text()
        .splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
        for req in [Requirement(line)]
    }
    for name in ("flashinfer-python", "nvidia-cudnn-frontend"):
        version = next(iter(requirements[name].specifier)).version
        assert version in constraints[name].specifier
    version = next(iter(requirements["flashinfer-python"].specifier)).version
    assert version in constraints["flashinfer-cubin"].specifier
    # Removing the CUDA-13 exclusion would mask the original bug.
    assert str(constraints["nvidia-cutlass-dsl-libs-cu13"].specifier) == "<0"


def test_installer_uses_adjusted_requirements_and_matching_cubin(monkeypatch):
    path = ROOT / "test/ci_system/install_deps_cu129.py"
    spec = importlib.util.spec_from_file_location("cu129_installer", path)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    monkeypatch.setenv("WORKSPACE", str(ROOT))
    monkeypatch.setattr(installer, "validate_environment", lambda *args: None)
    # main mutates os.environ for subprocesses; restore the test environment.
    monkeypatch.setattr(installer.os, "environ", dict(installer.os.environ))
    calls = []
    monkeypatch.setattr(
        installer, "pip_install", lambda args, index: calls.append((args, index))
    )
    original_run = installer.subprocess.run

    def run(command, **kwargs):
        if command[:3] == [installer.sys.executable, "-m", "pip"]:
            return None
        return original_run(command, **kwargs)

    monkeypatch.setattr(installer.subprocess, "run", run)
    installer.main()
    resolved = next(args for args, _ in calls if "flashinfer-python==0.6.18" in args)
    assert "nvidia-cudnn-frontend==1.28.0" in resolved
    assert "nvidia-cutlass-dsl==4.8.0" in resolved
    assert "nvidia-cutlass-dsl-libs-cu12==4.8.0" in resolved
    assert not any("[cu13]" in arg for arg in resolved)
    assert any("/v0.6.18/flashinfer_cubin-0.6.18-" in arg for arg in resolved)
