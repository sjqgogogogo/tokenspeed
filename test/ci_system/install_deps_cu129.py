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

"""Experimental H100/H200 + CUDA 12.9 source install into an active Python 3.11 venv."""

import importlib.metadata
import os
import platform
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PYPI = "https://pypi.org/simple"
PYTORCH = "https://download.pytorch.org/whl/cu126"
WHEELS = "https://lightseek.org/whl/cu129/"
NATIVE_PACKAGES = {
    "tokenspeed-deepep",
    "tokenspeed-deepgemm",
    "tokenspeed-flashmla",
    "tokenspeed-fast-hadamard-transform",
    "tokenspeed-fa3",
    "tokenspeed-trtllm-kernel",
    "tokenspeed-flashkda",
}


CUDA13_UNSUFFIXED_PACKAGES = {
    "tokenspeed-cutedsl-kda",  # Current AOT distribution is CUDA-13-only.
    "nvidia-cublas",
    "nvidia-cuda-runtime",
    "nvidia-cuda-nvrtc",
}
CUDA_VERSIONED_PACKAGES = {"cuda-toolkit", "cuda-python", "cuda-bindings"}


def validate_cuda129_tool(tool: Path) -> None:
    """Require an executable CUDA 12.9 tool before changing the environment."""
    try:
        result = subprocess.run(
            [str(tool), "--version"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Expected a working CUDA 12.9 tool: {tool}") from exc
    match = re.search(r"\brelease\s+(\d+)\.(\d+)", result.stdout)
    if match is None or tuple(map(int, match.groups())) != (12, 9):
        raise RuntimeError(
            f"Expected CUDA 12.9 from {tool}, got {result.stdout.strip()!r}"
        )


def validate_environment(workspace: Path, cuda_home: Path) -> None:
    """Reject an unsupported or mixed CUDA environment before installing packages.

    Args:
        workspace: Checkout whose packages and constraints will be installed.
        cuda_home: Host-supplied CUDA Toolkit directory.

    Returns:
        None when Linux, Python 3.11, an active venv and CUDA 12.9 are present.
    """
    if platform.system() != "Linux":
        raise RuntimeError(
            "The Hopper/cu129 installer requires Linux; no packages were changed"
        )
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "Activate a Python 3.11 venv before running the Hopper/cu129 installer"
        )
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "Activate a Python venv before running the Hopper/cu129 installer"
        )
    for relative in (
        "requirements/nvidia-cu129-constraints.txt",
        "tokenspeed-kernel/python/setup.py",
        "tokenspeed-scheduler/pyproject.toml",
        "python/pyproject.toml",
    ):
        if not (workspace / relative).is_file():
            raise RuntimeError(
                f"Incomplete TokenSpeed checkout: missing {workspace / relative}"
            )

    conflicts = []
    for distribution in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata.get("Name", "")).lower()
        version = distribution.version
        if (
            (name.startswith("nvidia-") and name.endswith("-cu13"))
            or name in CUDA13_UNSUFFIXED_PACKAGES
            or (name in CUDA_VERSIONED_PACKAGES and re.match(r"^13(?:\.|$)", version))
            or (name in {"torch", "torchvision"} and re.search(r"\+cu13[0-9]", version))
        ):
            conflicts.append(f"{name}=={version}")
    if conflicts:
        raise RuntimeError(
            "CUDA 13 packages are already installed: "
            + ", ".join(sorted(conflicts))
            + ". Use a fresh Python 3.11 venv for cu129; this installer will not "
            "uninstall or replace an existing CUDA 13 environment."
        )
    for name in ("nvcc", "ptxas"):
        validate_cuda129_tool(cuda_home / "bin" / name)
    # Preserve explicit overrides, but reject ones that defeat this recipe.
    for name in ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH"):
        if name in os.environ:
            validate_cuda129_tool(Path(os.environ[name]))


def configure_triton_ptxas(cuda_home: Path) -> None:
    """Persist CUDA 12.9 assemblers in this venv's tokenspeed-triton wheel.

    Newer Triton selects ptxas-blackwell for sm90 as well as Blackwell.
    CUDA_HOME does not override that bundled tool, and installer environment
    variables do not survive into subsequent serving processes. Replace both
    bundled entry points after all pip operations so upgrades cannot undo it.
    Only the active venv's wheel files may be changed; editable/shared installs
    outside it are rejected.
    """
    source = cuda_home / "bin" / "ptxas"
    validate_cuda129_tool(source)
    distribution = importlib.metadata.distribution("tokenspeed-triton")
    files = {str(path) for path in distribution.files or ()}
    destinations = []
    for name in ("ptxas", "ptxas-blackwell"):
        relative = f"tokenspeed_triton/backends/nvidia/bin/{name}"
        if relative not in files:
            raise RuntimeError(f"tokenspeed-triton wheel is missing {relative}")
        path = Path(distribution.locate_file(relative))
        # Resolve the parent, not the file: replace a previous symlink without
        # modifying its target, which may be outside the venv.
        destination = path.parent.resolve() / path.name
        if not destination.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"Triton assembler is outside this venv: {destination}")
        destinations.append(destination)
    for destination in destinations:
        with tempfile.TemporaryDirectory(
            prefix=".cu129-ptxas-", dir=destination.parent
        ) as staging:
            staged = Path(staging) / destination.name
            shutil.copy2(source, staged)
            os.replace(staged, destination)
        print(f"Triton CUDA 12.9 assembler: {destination}", flush=True)


def verify_triton_ptxas() -> None:
    """Check the actual sm90 compiler selection in a fresh serving-like process."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tokenspeed_triton.backends.nvidia.compiler "
            "import get_ptxas_for_arch\n"
            "tool = get_ptxas_for_arch(90)\n"
            "print(f'Hopper Triton ptxas: {tool.path}, CUDA {tool.version}', flush=True)\n"
            "if tool.version != '12.9':\n"
            "    raise RuntimeError('Hopper/cu129 requires Triton ptxas 12.9')\n",
        ],
        check=True,
    )


def kernel_requirements(workspace: Path) -> list[str]:
    # Evaluate metadata from the checkout, without running any build command.
    import setuptools

    captured = {}
    original = setuptools.setup
    try:
        setuptools.setup = lambda **kwargs: captured.update(kwargs)
        runpy.run_path(str(workspace / "tokenspeed-kernel" / "python" / "setup.py"))
    finally:
        setuptools.setup = original
    return captured["install_requires"]


def pip_install(arguments: list[str], index: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--index-url", index, *arguments],
        check=True,
    )


def configure_pip_downloads() -> None:
    """Allow large wheels more time/retries, preserving explicit pip overrides."""
    if not any(name in os.environ for name in ("PIP_TIMEOUT", "PIP_DEFAULT_TIMEOUT")):
        os.environ["PIP_DEFAULT_TIMEOUT"] = "120"
    os.environ.setdefault("PIP_RETRIES", "10")
    os.environ.setdefault("PIP_RESUME_RETRIES", "20")


def flashinfer_cubin_requirement(version: str) -> str:
    """Reuse an exact installed cubin version; otherwise select its release wheel.

    Passing the direct URL unconditionally makes pip fetch the large wheel to
    inspect its metadata even when that version is already installed. A named
    exact requirement lets pip keep the installed distribution while still
    checking its dependencies.
    """
    try:
        installed = importlib.metadata.version("flashinfer-cubin")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed == version:
        print(f"Reusing installed flashinfer-cubin=={version}", flush=True)
        return f"flashinfer-cubin=={version}"
    return (
        "https://github.com/flashinfer-ai/flashinfer/releases/download/"
        f"v{version}/flashinfer_cubin-{version}-py3-none-any.whl"
    )


def main() -> None:
    workspace = Path(os.environ.get("WORKSPACE", Path(__file__).resolve().parents[2]))
    workspace = workspace.resolve()
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    validate_environment(workspace, cuda_home)
    os.environ.update(
        CUDA_HOME=str(cuda_home),
        FLASHINFER_NVCC=str(cuda_home / "bin" / "nvcc"),
        TOKENSPEED_KERNEL_BACKEND="cuda",
        TOKENSPEED_KERNEL_CUDA_VARIANT="cu129",
        FLASHINFER_CUDA_ARCH_LIST="9.0a",
        MAX_JOBS=os.environ.get("BUILD_AND_DOWNLOAD_PARALLEL", "16"),
        PIP_CONSTRAINT=str(workspace / "requirements" / "nvidia-cu129-constraints.txt"),
        PIP_CONFIG_FILE=os.devnull,
        PIP_INDEX_URL=PYPI,
    )
    # Identical native package versions exist on PyPI with a different CUDA
    # ABI. Each install below uses one source, including in configured hosts.
    for name in (
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        # A requirements file can itself add indexes or wheel directories.
        "PIP_REQUIREMENT",
        "PIP_NO_INDEX",
        "PIP_NO_DEPS",
        "PIP_FORCE_REINSTALL",
        "PIP_IGNORE_INSTALLED",
        "PIP_UPGRADE",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
    ):
        os.environ.pop(name, None)
    print(
        f"Experimental Hopper/cu129 install: {workspace}, Python: {sys.executable}",
        flush=True,
    )
    configure_pip_downloads()

    pip_install(["--upgrade", "pip", "setuptools==83.0.0", "wheel", "packaging"], PYPI)
    requirements = kernel_requirements(workspace)
    pip_install(["torch", "torchvision"], PYTORCH)

    native = [req for req in requirements if req.split("==")[0] in NATIVE_PACKAGES]
    runtime = tomllib.loads((workspace / "python" / "pyproject.toml").read_text())
    mooncake = [
        req
        for req in runtime["project"]["dependencies"]
        if req.startswith("tokenspeed-mooncake>=")
    ]
    pip_install(["--force-reinstall", "--no-deps", *native, *mooncake], WHEELS)

    flashinfer = next(
        req.split("==")[1]
        for req in requirements
        if req.startswith("flashinfer-python==")
    )
    cubin = flashinfer_cubin_requirement(flashinfer)
    # Resolve native wheels' transitive dependencies after the right binaries
    # are installed. Matching public versions (including local suffixes) stay.
    pip_install([*requirements, *mooncake, cubin], PYPI)
    pip_install(
        [
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(workspace / "tokenspeed-kernel" / "python"),
        ],
        PYPI,
    )
    with tempfile.TemporaryDirectory(prefix="tokenspeed-cu129-scheduler-") as build_dir:
        pip_install(
            [
                str(workspace / "tokenspeed-scheduler"),
                f"--config-settings=build-dir={build_dir}",
            ],
            PYPI,
        )
    pip_install(["--no-build-isolation", "-e", str(workspace / "python")], PYPI)
    configure_triton_ptxas(cuda_home)
    verify_triton_ptxas()
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
