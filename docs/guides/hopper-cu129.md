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
`$CUDA_HOME/bin/nvcc` and `$CUDA_HOME/bin/ptxas`. Both must report CUDA 12.9.
The nvcc compiler is also selected explicitly for the
in-tree kernel build. The host supplies its C++ compiler, OpenSSL development
files, and required runtime libraries. The recipe does not run apt or sudo.
Each pip call uses the active Python interpreter; proxy environment variables
are inherited.

The installer reuses `flashinfer-cubin` when its installed version exactly
matches the checkout's FlashInfer pin. It passes a named requirement to pip
instead of the release URL, avoiding another metadata download of the large
wheel. Missing or different versions still come from the pinned GitHub release.
Downloads default to a 120-second socket timeout, 10 connection retries and
20 incomplete-download resume/restart attempts. Explicit `PIP_TIMEOUT` /
`PIP_DEFAULT_TIMEOUT`, `PIP_RETRIES` and `PIP_RESUME_RETRIES` values are preserved.
These are pip's own retry controls, not a guarantee that an interrupted partial
download survives a separate installer invocation or that a proxy supports
resuming it. A slow or unreliable connection can use, for example:

```bash
PIP_DEFAULT_TIMEOUT=300 PIP_RESUME_RETRIES=50 \
  CUDA_VARIANT=cu129 bash test/ci_system/install_deps.sh
```

An `incomplete-download` error is a failed wheel transfer, not a dependency
version conflict. Retry after fixing connectivity. Alternatively, download the
official pinned wheel through a working connection, install that local file
with `python -m pip install --no-deps /path/to/flashinfer_cubin-<version>-py3-none-any.whl`,
and rerun this installer; the matching installed version will then be reused.
The installer still finishes with `pip check` and verifies Triton's assembler.

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
It uses the existing checkout's kernel versions, switches the CuTe runtime
to cu12, and omits the CUDA-13-only CuteDSL KDA AOT package.
`requirements/nvidia-cu129-constraints.txt` contains the additional version
constraints. Each installation step selects its package index explicitly;
global pip configuration, environment-supplied requirements files, and extra package indexes are excluded so native
wheels with identical public versions cannot be chosen from a cu130 source.
Environment flags that force reinstallation, ignore installed packages, or
redirect pip outside the active venv are cleared for the installer process.
The normal checkout pins remain authoritative, including the updated
`tokenspeed-mla` and scheduler versions from main.

After all package installations, the helper copies the validated
`$CUDA_HOME/bin/ptxas` into both `tokenspeed-triton` bundled assembler paths
(`ptxas` and `ptxas-blackwell`) inside the active venv. This is a deliberate
local modification of the installed wheel. Triton `3.8.10.post20260920`
selects `ptxas-blackwell` for Hopper too, and ships CUDA 13.4 at that path;
setting `CUDA_HOME` alone does not change Triton's choice. The helper then
checks the actual sm90 assembler selection in a fresh Python process and
fails unless it reports CUDA 12.9. It does not need a GPU for this check.

The copied tools persist across new shells and worker processes; no additional
exports or venv reactivation are required. Every run of the installer reapplies
them after pip, including after Triton upgrades. If Triton is reinstalled
separately, rerun this installer before serving. Explicit
`TRITON_PTXAS_PATH` and `TRITON_PTXAS_BLACKWELL_PATH` overrides are preserved,
but installation rejects them unless they point to working CUDA 12.9 tools.
Overrides set later can still bypass the installed tools. Shared or editable
Triton installations outside the active venv are rejected.

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
python -c 'from tokenspeed_triton.backends.nvidia.compiler import get_ptxas_for_arch; tool = get_ptxas_for_arch(90); print(tool.path, tool.version)'
```

Dependency checks alone do not validate binary ABI compatibility or model
serving correctness.

For DeepSeek V4.1, check the compressor metadata kernel before loading the
model, then validate serving on the target GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python -X faulthandler -m pytest -q \
  tokenspeed-kernel/test/ops/test_attention_dsv41.py \
  -k 'compressor_metadata_consecutive_requests_and_refresh and cuda'
```

Use `--chunked-prefill-size 8192` for initial serving validation. This controls
the per-iteration token budget and startup tuning batch, independently of the
model's maximum context length. A successful assembler check does not by itself
establish that a CUDA driver loading fault or a model serving failure is fixed.
