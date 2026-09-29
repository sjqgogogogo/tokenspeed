# Experimental Hopper / CUDA 12.9 installation

This source-install recipe targets H100/H200, CUDA Toolkit 12.9, and Python
3.11 on Linux. It installs the kernel package, scheduler, and Python runtime
from this checkout. GPU build and serving validation must be performed on
the target host.

With the virtual environment activated and CUDA Toolkit 12.9 available,
run this single installation command from the repository root:

```bash
CUDA_VARIANT=cu129 bash test/ci_system/install_deps.sh
```

For the first installation, prepare a fresh environment to avoid carrying
over CUDA 13 packages from a previous installation:

```bash
uv venv .venv-cu129 --python 3.11 --seed
source .venv-cu129/bin/activate
export CUDA_HOME=/usr/local/cuda
CUDA_VARIANT=cu129 bash test/ci_system/install_deps.sh
```

Set `CUDA_HOME` to the CUDA 12.9 toolkit location when it is installed
elsewhere. Before installing packages, the helper checks Linux, Python 3.11,
an active virtual environment, and the version reported by
`$CUDA_HOME/bin/nvcc`. That compiler is also selected explicitly for the
in-tree kernel build. The host supplies its C++ compiler, OpenSSL development
files, and required runtime libraries. The recipe does not run apt or sudo.
Each pip call uses the active Python interpreter; proxy environment variables
are inherited.

The helper rejects installed CUDA 13 runtime packages and Torch wheels marked
cu13 before changing packages. It reports the conflicting distributions and
leaves them installed. Use a fresh cu129 venv instead of mixing the two stacks;
constraints cannot remove unrelated CUDA 13 packages left in an old environment.

DeepEP follows the same pinned installation path as the other native packages.
The dependency remains `tokenspeed-deepep==2.1.0.post20260810`; pending upstream
DeepEP changes are maintained separately from TokenSpeed. If a temporary build
is needed, install its cu129/Python 3.11 wheel manually **after** the complete
environment installation, with `DEEPEP_WHEEL` set to that wheel's path:

```bash
CUDA_VARIANT=cu129 bash test/ci_system/install_deps.sh
python -m pip install --force-reinstall --no-deps "$DEEPEP_WHEEL"
```

Rerunning the environment installer restores the declared native package pins,
so apply the manual wheel again afterwards when testing pending upstream work.
The installer does not patch DeepEP or add development-version exceptions.
Update the official dependency pin after the upstream changes are released.

The explicit `CUDA_VARIANT=cu129` value dispatches to
`test/ci_system/install_deps_cu129.py`. The recipe uses Torch `2.14.0+cu126` and
torchvision `0.29.0+cu126` from the PyTorch cu126 index because this Torch release
has no cu129 wheels. Its Python CUDA runtime packages use CUDA 12.6.3, while
`CUDA_HOME` and the native kernel compiler remain on CUDA Toolkit 12.9.
Native TokenSpeed wheels still come from the LightSeek cu129 index.
It switches the CuTe runtime to cu12 and omits the CUDA-13-only CuteDSL KDA
AOT package. The cu129 metadata selects FlashInfer Python/cubin `0.6.18`
and cuDNN Frontend `1.28.0`: Frontend `1.29.0` unconditionally requires
`nvidia-cutlass-dsl[cu13]`, and FlashInfer `0.7.0` requires Frontend >=1.29.
Changing only the direct CuTe dependency cannot resolve that transitive
conflict. Keep the cu13 exclusions; do not bypass resolution with `--no-deps`.
The kernel package records the same cu129 pins, so the final runtime install
does not restore the incompatible pair. Other checkout pins stay unchanged;
the default CUDA 13 installation retains FlashInfer `0.7.0` / Frontend `1.29.0`.
`requirements/nvidia-cu129-constraints.txt` contains the additional version
constraints. Each installation step selects its package index explicitly;
global pip configuration, environment-supplied requirements files, and extra package indexes are excluded so native
wheels with identical public versions cannot be chosen from a cu130 source.
Environment flags that force reinstallation, ignore installed packages, or
redirect pip outside the active venv are cleared for the installer process.
The normal checkout pins remain authoritative, including the updated
`tokenspeed-mla` and scheduler versions from main.

The kernel build uses that adjusted dependency metadata, skips its default
build-time pip install, and rebuilds the in-tree CUDA kernels for `sm90a`.
Each kernel group records its successful build's CUDA toolkit, compiler paths,
architecture/compile flags and library search paths. Returning to a CUDA 13
build invalidates CUDA 12 artifacts even when their timestamps are newer than
the sources; missing or invalid identities also trigger a rebuild. The old
identity is removed before rebuilding and a new one is published only after
a successful link. This prevents a failed partial rebuild from advertising
mixed binaries as reusable.

The source checkout's native output directory is shared by its editable
installs. Use separate checkouts as well as separate venvs when maintaining
CUDA 12 and CUDA 13 installations concurrently.

The scheduler builds in a temporary directory. After updating this checkout,
rerun the same installation command in the activated environment to rebuild
the kernel and scheduler and refresh the editable runtime installation.
Without the cu129 opt-in, the existing NVIDIA and ROCm installers and package
metadata are unchanged.

The installer finishes with `python -m pip check`. On the target host, also
verify imports and the CLI before running model-specific kernel and serving
tests:

```bash
tokenspeed env
python -c 'import torch; import tokenspeed_kernel; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
tokenspeed serve --help
```

Dependency checks alone do not validate binary ABI compatibility or model
serving correctness.

The FlashInfer GDN adapter also supports the cu129 `0.6.18` API: explicit
`backend="flashinfer"` selects its sole backend, and PDL isolates its
in-memory caches without requiring the persistent-cache builder introduced
in `0.7`. Other backend requests are rejected on the older API. GPU kernel
and serving tests are still required for this dependency combination.

CPU metadata and adapter-protocol regression tests:

```bash
python -m pytest test/ci_system/test_cu129_dependencies.py \
    test/ci_system/test_flashinfer_legacy_adapter.py \
    test/ci_system/test_cuda_install_validation.py
```
