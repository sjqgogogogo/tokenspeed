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

import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "ptxas_installer", Path(__file__).with_name("install_deps_cu129.py")
)
installer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(installer)


@pytest.mark.parametrize("version", ["12.6", "12.9", "13.4", "unknown"])
def test_toolkit_version_validation(monkeypatch, version):
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda args, *, check, capture_output, text: SimpleNamespace(
            stdout=f"Cuda compilation tools, release {version}, V{version}.0"
        ),
    )
    tool = Path("/cuda/bin/ptxas")
    if version == "12.9":
        installer.validate_cuda129_tool(tool)
    else:
        with pytest.raises(RuntimeError, match="Expected CUDA 12.9"):
            installer.validate_cuda129_tool(tool)


@pytest.mark.parametrize("layout", ["regular", "symlink", "outside", "missing"])
def test_bundled_assemblers_are_replaced_only_inside_venv(
    monkeypatch, tmp_path, layout
):
    venv = tmp_path / "venv"
    site = venv / "site-packages"
    tools = site / "tokenspeed_triton/backends/nvidia/bin"
    tools.mkdir(parents=True)
    cuda_home = tmp_path / "cuda"
    source = cuda_home / "bin/ptxas"
    source.parent.mkdir(parents=True)
    source.write_text("CUDA 12.9 assembler")
    source.chmod(0o755)
    external = tmp_path / "external-ptxas"
    external.write_text("external assembler")
    relatives = [
        Path("tokenspeed_triton/backends/nvidia/bin") / name
        for name in ("ptxas", "ptxas-blackwell")
    ]
    for relative in relatives:
        target = site / relative
        if layout == "symlink":
            target.symlink_to(external)
        else:
            target.write_text("CUDA 13 assembler")
    monkeypatch.setattr(installer.sys, "prefix", str(venv))
    distribution = SimpleNamespace(
        files=relatives[:1] if layout == "missing" else relatives,
        locate_file=lambda relative: (
            tmp_path / "outside" / relative if layout == "outside" else site / relative
        ),
    )
    monkeypatch.setattr(
        installer.importlib.metadata, "distribution", lambda name: distribution
    )
    monkeypatch.setattr(installer, "validate_cuda129_tool", lambda tool: None)
    if layout in {"outside", "missing"}:
        with pytest.raises(RuntimeError, match="outside this venv|wheel is missing"):
            installer.configure_triton_ptxas(cuda_home)
        for relative in relatives:
            assert (site / relative).read_text() == "CUDA 13 assembler"
    else:
        installer.configure_triton_ptxas(cuda_home)
        for relative in relatives:
            target = site / relative
            assert not target.is_symlink()
            assert target.read_bytes() == source.read_bytes()
            assert os.access(target, os.X_OK)
        assert external.read_text() == "external assembler"


@pytest.mark.parametrize("version", ["12.9", "13.4"])
def test_sm90_assembler_verification_in_fresh_process(monkeypatch, tmp_path, version):
    package = tmp_path / "tokenspeed_triton/backends/nvidia"
    package.mkdir(parents=True)
    (package / "compiler.py").write_text(
        "from types import SimpleNamespace\n"
        "def get_ptxas_for_arch(arch):\n"
        "    assert arch == 90\n"
        f"    return SimpleNamespace(path='/fake/ptxas-blackwell', version={version!r})\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    if version == "12.9":
        installer.verify_triton_ptxas()
    else:
        with pytest.raises(subprocess.CalledProcessError):
            installer.verify_triton_ptxas()


@pytest.mark.parametrize(
    "override", ["TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH"]
)
def test_invalid_override_fails_before_install(monkeypatch, override):
    monkeypatch.setenv(override, "/cuda13/bin/ptxas")
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        installer,
        "sys",
        SimpleNamespace(version_info=(3, 11), prefix="/venv", base_prefix="/base"),
    )
    monkeypatch.setattr(installer.importlib.metadata, "distributions", lambda: [])

    def validate(tool):
        if tool == Path("/cuda13/bin/ptxas"):
            raise RuntimeError("Expected CUDA 12.9")

    monkeypatch.setattr(installer, "validate_cuda129_tool", validate)
    monkeypatch.setattr(
        installer, "pip_install", lambda *args: pytest.fail("pip must not run")
    )
    with pytest.raises(RuntimeError, match="Expected CUDA 12.9"):
        installer.main(configure_triton_only=False)


def test_toolchain_only_repair_does_not_reinstall_packages(monkeypatch):
    events = []
    monkeypatch.setenv("WORKSPACE", str(ROOT))
    monkeypatch.setenv("CUDA_HOME", "/cuda12.9")
    monkeypatch.setattr(
        installer,
        "validate_environment",
        lambda workspace, cuda: events.append("validate"),
    )
    monkeypatch.setattr(
        installer,
        "configure_triton_ptxas",
        lambda cuda: events.append(("configure", cuda)),
    )
    monkeypatch.setattr(
        installer, "verify_triton_ptxas", lambda: events.append("verify")
    )
    monkeypatch.setattr(
        installer,
        "pip_install",
        lambda *args: pytest.fail("repair must not install packages"),
    )
    monkeypatch.setattr(
        installer,
        "kernel_requirements",
        lambda *args: pytest.fail("repair must not build metadata"),
    )
    installer.main(configure_triton_only=True)
    assert events == ["validate", ("configure", Path("/cuda12.9").resolve()), "verify"]
