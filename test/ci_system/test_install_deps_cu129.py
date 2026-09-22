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

"""Offline regressions for large-wheel downloads in the cu129 installer."""

import importlib.metadata
import os

import install_deps_cu129 as installer
import pytest


@pytest.mark.parametrize("installed", [None, "0.6.17", "0.6.18+local", "0.6.19"])
def test_missing_or_different_cubin_uses_pinned_release(monkeypatch, installed):
    def version(name):
        assert name == "flashinfer-cubin"
        if installed is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(installer.importlib.metadata, "version", version)
    assert installer.flashinfer_cubin_requirement("0.6.18") == (
        "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18/"
        "flashinfer_cubin-0.6.18-py3-none-any.whl"
    )


def test_matching_cubin_uses_named_requirement_instead_of_downloading(monkeypatch):
    monkeypatch.setattr(installer.importlib.metadata, "version", lambda name: "0.6.18")
    assert (
        installer.flashinfer_cubin_requirement("0.6.18") == "flashinfer-cubin==0.6.18"
    )


def test_download_defaults_and_explicit_overrides(monkeypatch):
    monkeypatch.setattr(installer.os, "environ", {})
    installer.configure_pip_downloads()
    assert os.environ == {
        "PIP_DEFAULT_TIMEOUT": "120",
        "PIP_RETRIES": "10",
        "PIP_RESUME_RETRIES": "20",
    }
    os.environ.update(
        PIP_DEFAULT_TIMEOUT="300", PIP_RETRIES="3", PIP_RESUME_RETRIES="50"
    )
    installer.configure_pip_downloads()
    assert os.environ == {
        "PIP_DEFAULT_TIMEOUT": "300",
        "PIP_RETRIES": "3",
        "PIP_RESUME_RETRIES": "50",
    }


def test_timeout_alias_is_not_shadowed(monkeypatch):
    monkeypatch.setattr(installer.os, "environ", {"PIP_TIMEOUT": "60"})
    installer.configure_pip_downloads()
    assert os.environ["PIP_TIMEOUT"] == "60"
    assert "PIP_DEFAULT_TIMEOUT" not in os.environ


def test_main_reuses_matching_cubin_and_keeps_final_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        installer.os,
        "environ",
        {"WORKSPACE": str(tmp_path), "CUDA_HOME": str(tmp_path / "cuda")},
    )
    (tmp_path / "python").mkdir()
    (tmp_path / "python/pyproject.toml").write_text(
        '[project]\ndependencies = ["tokenspeed-mooncake>=0.3.13"]\n'
    )
    monkeypatch.setattr(installer, "validate_environment", lambda *args: None)
    monkeypatch.setattr(
        installer, "kernel_requirements", lambda root: ["flashinfer-python==0.6.18"]
    )
    monkeypatch.setattr(installer.importlib.metadata, "version", lambda name: "0.6.18")
    calls = []

    def pip_install(arguments, index):
        assert os.environ["PIP_RESUME_RETRIES"] == "20"
        calls.append((arguments, index))

    monkeypatch.setattr(installer, "pip_install", pip_install)
    finished = []
    monkeypatch.setattr(
        installer, "configure_triton_ptxas", lambda home: finished.append("configure")
    )
    monkeypatch.setattr(
        installer, "verify_triton_ptxas", lambda: finished.append("verify")
    )
    monkeypatch.setattr(
        installer.subprocess, "run", lambda cmd, **kw: finished.append(cmd)
    )
    installer.main()
    cubin_install = [
        args for args, _ in calls if any("flashinfer-cubin" in arg for arg in args)
    ]
    assert cubin_install == [
        [
            "flashinfer-python==0.6.18",
            "tokenspeed-mooncake>=0.3.13",
            "flashinfer-cubin==0.6.18",
        ]
    ]
    assert not any("flashinfer_cubin-" in arg for args, _ in calls for arg in args)
    assert finished == [
        "configure",
        "verify",
        [installer.sys.executable, "-m", "pip", "check"],
    ]
