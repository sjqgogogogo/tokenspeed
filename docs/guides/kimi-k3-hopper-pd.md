# Kimi K3 on Hopper with PD, DeepEP and DSpark

This layout uses 64 GPUs: four eight-GPU H200 nodes for each engine.
The runtime supports the configuration below; the GPU validation steps at
the end are required before treating a deployment as validated.

| Engine | Attention | Routed experts | Shared experts | Execution |
| --- | --- | --- | --- | --- |
| P | PP4, TP8, DP1 | EP8 within each stage, MoE TP1 | Stage TP8 | Eager, DeepEP normal |
| D | PP1, TP8, DP4 | EP32 across four nodes, MoE TP1 | Each replica's TP8 | Decode graph, DeepEP auto (low latency for decode) |

EP uses the same GPUs as attention. P and D both use attention TP8, so
CachePD does not change the target cache's TP shard width at handoff.

## Dependencies and model contract

Use the MXFP4 K3 target and a compatible `k3_dspark` checkpoint. Marlin
executes packed W4A16 routed experts with BF16 activations. The target's
TokenSpeed MLA cache is FP8; the draft cache retains BF16 through the cache
recipe. Build the kernel package with its Marlin CUDA extension on the GPU
hosts.

DeepEP must provide the legacy `Buffer` API with BF16 low-latency support
for the checkpoint's latent width and top-k. The standard K3 geometry is
3584 latent dimensions, top-k 16, and 896 experts: 112 per P rank and 28 per
D rank. The normal interface returns local expert IDs; the bridge converts
them explicitly at the Marlin boundary. K3 does not require an
identity-expert `x_ori` argument on low-latency combine.

EP32 low latency requires the installed DeepEP build's inter-node transport
prerequisites. Use consistent dependency builds on all ranks. For initial
validation, `--load-format auto` keeps checkpoint loading independent of
InstantTensor; optional InstantTensor context loading uses each P stage's
TP group because stages consume different weight subsets.

## Engine arguments

Add these argument sets to the existing engine launcher. Supply target and
draft paths, node rank, rendezvous address, gateway transport, context
length and memory utilization separately. Run one launcher per node with
eight GPU workers and use separate P/D rendezvous addresses. These argument
sets do not configure or start an SMG gateway.

Common arguments:

```text
--model <K3_MODEL_PATH>
--speculative-draft-model-path <K3_DSPARK_MODEL_PATH>
--speculative-algorithm DSPARK
--speculative-num-draft-tokens 8
--dtype bfloat16
--kv-cache-dtype fp8
--attention-backend tokenspeed_mla
--drafter-attention-backend mla
--moe-backend marlin
--all2all-backend deepep
--world-size 32
--nnodes 4
--nprocs-per-node 8
--attn-tp-size 8
--dense-tp-size 8
--moe-tp-size 1
--disable-kvstore
```

Verify width includes the anchor: eight target rows use seven DSpark
proposal queries. Select a width supported by the draft checkpoint.

P arguments:

```text
--disaggregation-mode prefill
--pipeline-parallel-size 4
--data-parallel-size 1
--expert-parallel-size 8
--deepep-mode normal
--mm-encoder-tp-mode data
--max-num-seqs 32
--chunked-prefill-size 8192
```

P is eager by role. Its layer windows use the target layer count, excluding
draft layers. The automatic split of 93 layers is `24,23,23,23`.
`--pp-layer-partition 24,24,24,21` is another candidate: its first three
boundaries align with the standard AttnRes blocks and the last stage owns
fewer target layers alongside context writing. Measure stage time before
choosing a partition for throughput. KDA geometry can reduce the effective
prefill chunk size to align state checkpoints; startup logs report it.

D arguments:

```text
--disaggregation-mode decode
--pipeline-parallel-size 1
--data-parallel-size 4
--expert-parallel-size 32
--deepep-mode auto
--max-num-seqs 128
--max-cudagraph-capture-size 32
```

`auto` keeps ordinary decode on low latency and CUDA graph. If cache
pressure retracts a request and the scheduler performs a recovery prefill,
that extend-shaped work uses normal dispatch and eager execution. Disabling
KVStore does not eliminate recovery prefills.

`max-num-seqs` is global across DP: this example allows 32 requests per D
replica and captures through that local batch size. Keep graph padding
enabled and omit `--enforce-eager`. An explicit capture-size list must
cover the intended workload too. These settings do not establish a tested
maximum context capacity or memory utilization.

## Token layout and capacity

Each TP rank takes a disjoint slice of its attention batch before routing
and latent projection. DeepEP carries routed latent work and returns the
combined result to the source slice. A TP8 token gather restores that
replica's latent batch before normalization and the shared/up-projection
TP reduction. No full hidden-state or residual gather replicates the four
attention-DP batches across EP32.

Automatic low-latency capacity for token-sliced K3 in `auto` mode is:

```text
ceil((max_num_seqs // attention_DP) * verify_width / attention_TP)
```

The D example allocates 32 source rows per EP rank. A smaller graph ladder
does not reduce this capacity: eager batches above the ladder remain valid.
An explicit `--low-latency-max-num-tokens-per-gpu` below the required bound
fails at startup; a larger explicit value is respected. Other model paths
retain their previous automatic value of 256. If `low_latency` is pinned,
the capacity also covers the configured prefill/recovery chunk beside the
decode batch before TP slicing, because no normal buffers exist. For the
D example and an 8192-token chunk this requires 1056 rows per source rank
and substantially more memory;
an explicit decode-only capacity such as 32 is rejected. Prefer `auto` for
the D engine.

DeepEP's own receive buffers still reserve expert capacity. The Marlin
bridge uses device counts to construct aligned work and bound intermediate
storage by source routes, avoiding SiTU work over the entire
`experts * capacity` padding extent. Communication buffers are prepared
with weights before KV memory profiling. Measure persistent memory and
capture/runtime peaks separately.

Empty slices and idle DP ranks participate in both EP legs. The target's
persistent cache write slots identify graph padding on the GPU; those
routes use ID -1 and weight zero. Python counts recorded during capture
are not used as the live-token mask.

## Context production and handoff

P stages project DSpark taps as they appear and pass one FP32
`[tokens, draft_hidden]` accumulator alongside ordinary PP state. Per-tap
`fc_norm` precedes projection; `context_norm` follows the complete sum.
An AttnRes tap requiring the next stage's first layer is evaluated at that
stage's entrance using the incoming PP state.

Each stage loads only its tap projection parameters. The last also loads
context-to-KV weights, KV normalization and rotary components, and writes
all draft layers' prompt KV. P does not load the complete proposal model
or execute draft blocks.

All P stages share one logical cache geometry, with target physical fields
assigned by stage and draft fields owned only by the last stage. The same
ownership drives producer readiness, sender filtering and receiver routes.
A final producer barrier covers draft context writes. Bootstrap metadata
explicitly carries the target layer count, so draft layers cannot move PP
boundaries.

P sends its bootstrap token without speculative candidates. D reuses the
existing candidate-free bootstrap: the ordinary verify graph accepts one
token in the first round, then generates DSpark proposals for subsequent
rounds. MLA, KDA recurrent/conv state and draft KV must all arrive before
that round. There is no separate bootstrap execution path.

## Validation

CPU tests cover rank topology, capacity, stage ownership, transfer-field
coverage, context arithmetic and layout construction. They do not validate
compiled GPU kernels, transport progress, graph replay or checkpoint-level
numerics. On the GPU hosts:

1. Compare Marlin with the dequantized reference, then test DeepEP normal
   EP8 and low latency EP32 at actual K3 geometry. Include nonzero EP ranks,
   empty/unequal source batches and skewed expert routing.
2. Compare eager and repeated graph MoE results while changing routes,
   active DP ranks and live counts inside one capture bucket. Include
   batches below TP8 and padded verify rows.
3. Compare concatenated context projection with PP accumulation, including
   `fc_norm`, both configured tap streams and PP boundaries. Floating-point
   addition order differs; compare errors and finite values rather than
   requiring BF16 bitwise equality.
4. Validate P-to-D target/cache behavior without speculation, then enable
   DSpark and compare initial logits, generation and acceptance. Exercise
   the first D round without candidates explicitly.
5. Exercise chunked prompts, prefix hits, page boundaries, concurrency,
   cancellation and repeated slot reuse. Keep graph enabled with unequal
   DP batches. `TOKENSPEED_GRAPH_DEBUG=1` checks metadata pointer stability
   during correctness runs.
6. Measure realistic prompts across concurrency levels. Record P computed
   input tokens/s, D output tokens/s, TTFT/TPOT, acceptance, per-rank memory
   peaks and errors, with exact arguments and dependency versions.

Disabling graph is a diagnostic comparison, not completion of the decode
graph requirement. No GPU performance or stability result is implied by
the presence of this configuration.

### Focused test commands

From the repository root, the CPU contract suite can run without loading
GPU kernels:

```bash
PYTHONPATH=python:tokenspeed-kernel/python:tokenspeed-scheduler/python:test \
python -m pytest -q \
  test/runtime/test_deepep_capacity.py \
  test/runtime/test_k3_pd_topology_cpu.py \
  test/runtime/test_checkpoint_load_group_cpu.py \
  test/runtime/test_dspark_pp_context.py \
  test/runtime/test_k3_deepep_contract.py \
  test/runtime/test_pp_cache_ownership.py \
  test/runtime/distributed/test_pd_transfer_plan.py \
  test/runtime/distributed/test_cache_pd_manifest.py \
  -k 'not receiver_calc and not derives_ordinary'
```

The excluded receiver/model-config tests need the complete runtime import
environment. Run them too on the GPU hosts, together with
`test/runtime/test_cli_config_compat.py`,
`test/runtime/test_kimi_k3_cache_spec.py`,
`test/runtime/test_model_executor_cache_state.py`, and the existing K3
DSpark/capture tests. Keep runtime and kernel test roots in separate pytest
invocations to avoid their conflicting `conftest` module names.

For single-GPU compact Marlin numerics and graph replay:

```bash
PYTHONPATH=python:tokenspeed-kernel/python:tokenspeed-kernel/test \
python -m pytest -q \
  tokenspeed-kernel/test/ops/moe/test_marlin_deepep_layout.py
```

For one-node EP8 normal dispatch, with actual K3 routing geometry and
smaller expert intermediate tensors:

```bash
TEST_DEEPEP_MODE=normal TEST_K3_MOE_GEOMETRY=1 \
PYTHONPATH=python:tokenspeed-kernel/python:tokenspeed-kernel/test \
torchrun --standalone --nproc-per-node=8 -m pytest -q \
  tokenspeed-kernel/test/ops/moe/test_marlin_deepep_distributed.py
```

For EP32 low-latency graph replay, run the following on each of four GPU
nodes, setting `NODE_RANK` to 0..3 and a common test rendezvous address/port:

```bash
TEST_DEEPEP_MODE=low_latency TEST_K3_MOE_GEOMETRY=1 \
PYTHONPATH=python:tokenspeed-kernel/python:tokenspeed-kernel/test \
torchrun --nnodes=4 --nproc-per-node=8 \
  --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" -m pytest -q \
  tokenspeed-kernel/test/ops/moe/test_marlin_deepep_distributed.py
```

Set `TEST_MOE_INTERMEDIATE_SIZE=3072` to exercise full K3 expert matrix
dimensions after the smaller reference test passes. The test constructs a
replicated reference over all experts, so its memory needs exceed the
sharded serving weights. Run normal and low-latency tests in separate
process invocations; the DeepEP buffer is process-scoped.
