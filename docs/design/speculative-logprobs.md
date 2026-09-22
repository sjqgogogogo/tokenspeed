# Speculative logprobs: follow-up feasibility

Status: investigation only. All logprob requests reject speculative execution, including output-only
chosen-token requests. This document does not enable MTP, tree verification, or PD serving.

## Score the committed token positions

Acceptance rate is a throughput statistic, not a multiplier for log probabilities.
For each request, select target-model distributions for the token positions
actually emitted by verify. Discard rejected draft suffixes; include the target
replacement/bonus token. Trim scores together with IDs at EOS, stop, length, and
grammar termination. Target raw-model scores remain distinct from draft scores
and from any temperature/penalty/rejection-adjusted sampling distribution.

For the current chain verification kernels, draft position i is checked against
target position i-1. If two draft tokens are accepted, three outputs may be
emitted (two accepted tokens plus the target token). Backend `accept_length`
already includes that extra token after its `+= 1`; consumers must not add one
again. A zero-accepted-draft round still emits a target token.

## Existing building blocks

- Greedy, Triton, and FlashInfer `sampling/backends/*:verify` already produce
  selected target token logprobs when output scoring is enabled.
- The verify kernels populate accepted indices/counts; samplers synchronize the
  prediction, index, and length triple across attention-TP ranks.
- `OutputProcesser.post_process_forward_op` uses `output_lengths` to
  slice valid results while advancing across the fixed speculative slot width.
  It subsequently clips scores with the actual emitted tokens.
- Current diagnostic `TopLogprobCapture` deliberately requires one decode row
  per request and produces `[batch, 1, K]`. Graph snapshot allocation uses
  `[captured_batch, vocab]`. Those are the main shape assumptions to generalize.

## Proposed extension

For chain MTP, capture target distributions as `[batch, verify_width, ...]` before
sampling transforms. Keep static maximum-width buffers for graph capture, then
use the device-side accepted lengths/indices to select valid rows. Copy only
needed data or carry valid lengths through the existing result/event boundary.
Do not read acceptance counts with `.tolist()` on the forward path: it would
introduce a CPU synchronization that defeats overlap. Per-request K, mixed
prefill/verify batches, padding, and delayed completions must remain aligned.

The current accepted-chain prefix can be selected using lengths, but future tree
verification needs the actual accepted-path indices. Preserve an explicit mapping
contract rather than assuming every backend emits consecutive target rows.
Sampling-algorithm-specific replacement tokens must be scored under the agreed
target-model distribution, not accidentally under the residual sampling law.

A bounded chain-only extension looks feasible using existing machinery. The
main work is row ownership/mapping and memory cost (`batch * verify_width * vocab`
for full raw snapshots), rather than scheduler redesign. It is a separate change
with GPU validation: accept none/some/all, bonus token, EOS/length truncation,
heterogeneous batches, graph padding, TP agreement, and multiple in-flight rounds.
No claim of current MTP top-K correctness follows from existing output-only support.
