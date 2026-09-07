# MTP Acceptance Checklist

This checklist covers usage-layer validation for Qwen3.5/Qwen3.6/Qwen3.8 MTP
and Gemma4 assistant-drafter support. Real-checkpoint tests are ignored by
default because they require local model snapshots and an MLX runtime.

MTP capability acceptance requires output-equivalent greedy generation plus
non-zero draft activity. It does not imply a speedup for every checkpoint,
prompt, context length, or draft depth; performance claims require a balanced
fixed-condition comparison against the same base model without MTP.

## Environment Variables

| Variable | Checkpoint |
| --- | --- |
| `MLX_DIR` | Local MLX C++ runtime directory, usually `$HOME/.local/mlx`. |
| `QWEN35_MODEL` | `mlx-community/Qwen3.5-4B-MLX-4bit` snapshot. |
| `QWEN35_MTP_MODEL` | `mlx-community/Qwen3.5-4B-MTP-4bit` snapshot. |
| `QWEN36_DENSE_MODEL` | `mlx-community/Qwen3.6-27B-4bit` snapshot. |
| `QWEN36_DENSE_MTP_MODEL` | `mlx-community/Qwen3.6-27B-MTP-4bit` snapshot. |
| `QWEN38_DENSE_MODEL` | Matching `mlx-community/Qwen3.8-27B-4bit` or `Qwen3.8-27B-8bit` snapshot. |
| `QWEN38_DENSE_MTP_MODEL` | Matching `mlx-community/Qwen3.8-27B-MTP-4bit` or `Qwen3.8-27B-MTP-8bit` snapshot. |
| `QWEN36_MOE_MODEL` | `mlx-community/Qwen3.6-35B-A3B-4bit` snapshot. |
| `QWEN36_MOE_MTP_MODEL` | `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` snapshot. |
| `GEMMA4_LONG_CONTEXT_MODEL` | Matching `gemma4` or `gemma4_unified` base checkpoint. |
| `GEMMA4_LONG_CONTEXT_DRAFTER` | Matching Gemma4 assistant checkpoint. |

For every Qwen3.8 command below, the 8-bit acceptance variant substitutes the
matching `Qwen3.8-27B-8bit` base and `Qwen3.8-27B-MTP-8bit` snapshots. Do not
mix base and MTP precisions in the acceptance matrix.

## Qwen3.8 affine8 B2/B4 Performance Record

The retained Apple M5 Max non-DFlash2 B2/B4 measurements are archived in
[Qwen3.8-27B affine8 MTP B2/B4 Performance Archive](benchmarks/qwen38-affine8-mtp/2026-08-26/summary.md).
The record includes Dense short/long paired rows, B4 Paged/Turbo3/Turbo4/K3V4
profiles, and B1/B2 cross-commit regression data. It does not qualify B8 MTP
or DFlash2.

The current affine8 DFlash2 performance record is archived separately in
[Qwen3.8-27B affine8 DFlash2 性能归档](benchmarks/qwen38-affine8-dflash2/2026-08-29/summary.md).
The latest Greedy summary is bound to `dd37fde67af113501f40ce893b55e7a5609907e1`;
B1/B2/B4 at both 64 and 256 output tokens are positive-throughput cases. This
record supersedes the earlier B4/256 negative-throughput screening result and
does not by itself qualify Sampled or KV-profile performance.

The matching affine4 DFlash2 regression on the same commit is archived in
[Qwen3.8-27B affine4 DFlash2 回归归档](benchmarks/qwen38-affine4-dflash2/2026-08-29/summary.md).
Its Q4 Greedy B1/B2/B4 results are 56.108/60.254/58.139 TPS, respectively,
with output hashes and the 64-token Greedy and 256-token Sampled B4-versus-B1
checks passing.

## Fast Validation

```sh
MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx \
  cli::generate::tests::mtp_support_policy_allows_qwen_text_and_vl_and_rejects_other_architectures

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib actor_mtp_mode -- --nocapture

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --lib health_collector_mtp -- --nocapture

MLX_DIR=$HOME/.local/mlx cargo test -p ironmlx --test cli_generate_mtp_e2e -- --list
```

## CLI Real-Checkpoint Smoke Tests

Text-only model-path coverage:

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit/snapshots/<sha> \
QWEN35_MTP_MODEL=/path/to/Qwen3.5-4B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen35_text_generate_with_mtp_accepts_request \
  -- --ignored --test-threads=1 --nocapture
```

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN36_DENSE_MODEL=/path/to/Qwen3.6-27B-4bit/snapshots/<sha> \
QWEN36_DENSE_MTP_MODEL=/path/to/Qwen3.6-27B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen36_dense_text_generate_with_mtp_accepts_request \
  -- --ignored --test-threads=1 --nocapture
```

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN38_DENSE_MODEL=/path/to/Qwen3.8-27B-4bit/snapshots/<sha> \
QWEN38_DENSE_MTP_MODEL=/path/to/Qwen3.8-27B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen38_dense_text_generate_with_mtp_accepts_request \
  -- --ignored --test-threads=1 --nocapture
```

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN36_MOE_MODEL=/path/to/Qwen3.6-35B-A3B-4bit/snapshots/<sha> \
QWEN36_MOE_MTP_MODEL=/path/to/Qwen3.6-35B-A3B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen36_moe_text_generate_with_mtp_accepts_request \
  -- --ignored --test-threads=1 --nocapture
```

VL model-path coverage:

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit/snapshots/<sha> \
QWEN35_MTP_MODEL=/path/to/Qwen3.5-4B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen35_vl_generate_with_mtp_accepts_image_request \
  -- --ignored --test-threads=1 --nocapture
```

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN36_MOE_MODEL=/path/to/Qwen3.6-35B-A3B-4bit/snapshots/<sha> \
QWEN36_MOE_MTP_MODEL=/path/to/Qwen3.6-35B-A3B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test cli_generate_mtp_e2e \
  qwen36_moe_vl_generate_with_mtp_accepts_image_request \
  -- --ignored --test-threads=1 --nocapture
```

## Server Real-Checkpoint Smoke Tests

Strict Qwen exact-verify matrix (the shared Qwen3.5 execution test also accepts
Qwen3.8 checkpoints):

```sh
MLX_DIR=$HOME/.local/mlx \
PROMPT_LOOKUP_VERIFY_QWEN35_MODEL=/path/to/Qwen3.8-27B-4bit/snapshots/<sha> \
PROMPT_LOOKUP_VERIFY_REQUIRE_ZERO_DIFF=1 \
PROMPT_LOOKUP_VERIFY_BATCHES=1,2,4,8 \
PROMPT_LOOKUP_VERIFY_PREFIX_LENS=1024,1025,4096,4097,8192,32768,65536 \
PROMPT_LOOKUP_VERIFY_WIDTHS=2,3,4,5,6,8 \
PROMPT_LOOKUP_VERIFY_MAX_WIDTH=8 \
cargo test --release -p ironmlx --test prompt_lookup_verify_qualification \
  qwen35_dense_qgt1_matches_sequential_verify \
  -- --ignored --test-threads=1 --nocapture
```

Qwen3.6 Affine5 long-context exact-state coverage:

```sh
MLX_DIR=$HOME/.local/mlx \
PROMPT_LOOKUP_VERIFY_QWEN36_DENSE_MODEL=/path/to/Qwen3.6-27B-5bit/snapshots/<sha> \
PROMPT_LOOKUP_VERIFY_REQUIRE_ZERO_DIFF=1 \
PROMPT_LOOKUP_VERIFY_BATCHES=1 \
PROMPT_LOOKUP_VERIFY_PREFIX_LENS=8192,32768 \
PROMPT_LOOKUP_VERIFY_WIDTHS=2,3 \
PROMPT_LOOKUP_VERIFY_MAX_WIDTH=3 \
cargo test --release -p ironmlx --test prompt_lookup_verify_qualification \
  qwen36_dense_long_context_qgt1_matches_sequential_verify \
  -- --ignored --test-threads=1 --nocapture

MLX_DIR=$HOME/.local/mlx \
PROMPT_LOOKUP_VERIFY_QWEN36_MOE_MODEL=/path/to/Qwen3.6-35B-A3B-5bit/snapshots/<sha> \
PROMPT_LOOKUP_VERIFY_REQUIRE_ZERO_DIFF=1 \
PROMPT_LOOKUP_VERIFY_BATCHES=1 \
PROMPT_LOOKUP_VERIFY_PREFIX_LENS=8192,32768 \
PROMPT_LOOKUP_VERIFY_WIDTHS=2,3 \
PROMPT_LOOKUP_VERIFY_MAX_WIDTH=3 \
cargo test --release -p ironmlx --test prompt_lookup_verify_qualification \
  qwen36_moe_long_context_qgt1_matches_sequential_verify \
  -- --ignored --test-threads=1 --nocapture
```

These focused B1 gates cover the Dense and MoE hybrid Full-KV/GatedDelta state
after Q>1 verification; hidden states, logits, greedy tokens, and the subsequent
Q1 tail must match the ordinary sequential reference exactly. Run the existing
`qwen36_moe_qgt1_matches_sequential_verify` test separately for Paged,
TurboQuant, and ragged-batch token-equivalence coverage.

Qwen3.5-4B Paged KV long-context multi-token coverage runs the 8K, 32K, and
64K matrix. Each context length starts fresh `draft=1` and `draft=2` servers:

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit/snapshots/<sha> \
QWEN35_MTP_MODEL=/path/to/Qwen3.5-4B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test paged_prefix_matrix_e2e \
  qwen35_dense_paged_kv_long_context_multi_token_mtp_matches_single_token \
  -- --ignored --test-threads=1 --nocapture
```

At every context length, the generated message, finish reason, and completion
length must match exactly. The `draft=2` server must report requested and
effective draft width 2, increase `prefill_count`, `step_count`,
`drafted_tokens`, `accepted_draft_tokens`, and `multi_token_windows`, and leave
`fallback_prefill_count` unchanged. `multi_token_windows` directly counts
windows that attempted a second draft position, so later single-token or
zero-draft control windows cannot hide the multi-token execution.

Paged KV multi-token MTP must match the same checkpoint at `draft=1` exactly:

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN38_DENSE_MODEL=/path/to/Qwen3.8-27B-4bit/snapshots/<sha> \
QWEN38_DENSE_MTP_MODEL=/path/to/Qwen3.8-27B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test paged_prefix_matrix_e2e \
  qwen38_dense_paged_kv_multi_token_matches_single_token_mtp \
  -- --ignored --test-threads=1 --nocapture
```

The generated message, finish reason, and completion length must match exactly.
For the `draft=2` run, `/healthz` must report both requested and effective draft
width as 2, with a positive `multi_token_windows` delta proving a multi-token
window ran.

Run the same Paged KV check for the independent Gemma4 drafter path:

```sh
MLX_DIR=$HOME/.local/mlx \
GEMMA4_LONG_CONTEXT_MODEL=/path/to/gemma4-base/snapshots/<sha> \
GEMMA4_LONG_CONTEXT_DRAFTER=/path/to/gemma4-assistant/snapshots/<sha> \
cargo test --release -p ironmlx --test paged_prefix_matrix_e2e \
  gemma4_unified_paged_kv_multi_token_matches_single_token_drafter \
  -- --ignored --test-threads=1 --nocapture
```

Gemma4 and Gemma4 Unified long-context assistant-drafter parity (defaults to
8K, 32K, and 64K contexts with 64 generated tokens):

```sh
MLX_DIR=$HOME/.local/mlx \
GEMMA4_LONG_CONTEXT_MODEL=/path/to/gemma4-base/snapshots/<sha> \
GEMMA4_LONG_CONTEXT_DRAFTER=/path/to/gemma4-assistant/snapshots/<sha> \
cargo test --release -p ironmlx --test gemma4_long_context_parity \
  gemma4_drafter_long_context_tokens_match_ordinary_q1_exactly \
  -- --ignored --test-threads=1 --nocapture
```

The test requires exact token equality with ordinary greedy Q1 generation and
non-zero verify windows and drafted tokens at every context length. Performance
must be measured separately against the same base checkpoint without a drafter;
passing parity does not imply a decode or end-to-end speedup.

For fixed-work performance runs, `ironmlx-core-bench` accepts
`--prompt-target-tokens 8192|32768|65536`, `--ignore-eos`, and
`--scheduler-baseline-out <path>`. Repeat `--prompt-file` to create B2; paired
baseline/drafter runs clear the MLX allocator cache between sides. Scheduler
benchmarks enable the production process-memory governor, and Gemma4 runs
reject an assistant whose model type, backbone hidden size, or vocabulary does
not match the base checkpoint.

The 2026-08-20 Dense/MoE B1/B2 performance matrix and its evidence boundaries
are recorded in
[`docs/benchmarks/gemma4-drafter-performance/2026-08-20/summary.md`](benchmarks/gemma4-drafter-performance/2026-08-20/summary.md).

The 2026-08-21 Qwen/Gemma4 policy-split 32K cross-commit regression gate is
recorded in
[`docs/benchmarks/mtp-policy-split/2026-08-21/summary.md`](benchmarks/mtp-policy-split/2026-08-21/summary.md).

Gemma4 PromptLookup exact verify has no separate 1024-token context cap. Verify
the production qualification at boundary and long-context lengths:

```sh
MLX_DIR=$HOME/.local/mlx \
PROMPT_LOOKUP_VERIFY_GEMMA4_MODEL=/path/to/gemma4-base/snapshots/<sha> \
PROMPT_LOOKUP_VERIFY_BATCHES=1,2,4,8 \
PROMPT_LOOKUP_VERIFY_PREFIX_LENS=1024,1025,8192,32768,65536 \
PROMPT_LOOKUP_VERIFY_WIDTHS=2,3,4,5 \
cargo test --release -p ironmlx --test prompt_lookup_verify_qualification \
  gemma4_qgt1_matches_sequential_verify \
  -- --ignored --test-threads=1 --nocapture
```

Affine4 Gemma4 checkpoints use sequential Q1 PromptLookup verification with
TurboQuant KV because K3V4/K4V4 Q>1 is not token exact. This does not affect the
separate assistant-drafter K3V4 path below.

For assistant-drafter K3V4, long-context Q>1 verify uses stable attention.
Quantized profiles outside the exact batched qualification evaluate the
complete target as sequential `[B,1]` positions for both B1 and multi-row
batches. This test covers both one and two active requests under `b_max=4`,
requires exact Q1 token parity, and checks that the second draft position is
attempted:

```sh
MLX_DIR=$HOME/.local/mlx \
GEMMA4_LONG_CONTEXT_MODEL=/path/to/gemma4-base/snapshots/<sha> \
GEMMA4_LONG_CONTEXT_DRAFTER=/path/to/gemma4-assistant/snapshots/<sha> \
GEMMA4_K3V4_CONTEXT_TOKENS=8192 \
GEMMA4_K3V4_ACTIVE_REQUESTS=1,2 \
cargo test --release -p ironmlx --test gemma4_long_context_parity \
  gemma4_k3v4_long_context_scheduler_uses_multi_token_verify_exactly \
  -- --ignored --test-threads=1 --nocapture
```

Use `GEMMA4_K3V4_ACTIVE_REQUESTS=1` when validating 64K on hardware where the
B2 ordinary-Q1 prefill exceeds the Metal single-buffer or memory budget.

Qwen3.6-27B MTP Active KV swap-out/swap-in coverage:

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN36_DENSE_MODEL=/path/to/Qwen3.6-27B-4bit/snapshots/<sha> \
QWEN36_DENSE_MTP_MODEL=/path/to/Qwen3.6-27B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test paged_prefix_matrix_e2e \
  qwen36_dense_mtp_active_kv_offload_restores_speculative_side_cache \
  -- --ignored --test-threads=1 --nocapture
```

Qwen3.8-27B MTP Active KV swap-out/swap-in coverage:

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN38_DENSE_MODEL=/path/to/Qwen3.8-27B-4bit/snapshots/<sha> \
QWEN38_DENSE_MTP_MODEL=/path/to/Qwen3.8-27B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test paged_prefix_matrix_e2e \
  qwen38_dense_mtp_active_kv_offload_restores_speculative_side_cache \
  -- --ignored --test-threads=1 --nocapture
```

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN35_MODEL=/path/to/Qwen3.5-4B-MLX-4bit/snapshots/<sha> \
QWEN35_MTP_MODEL=/path/to/Qwen3.5-4B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test vl_mtp_paged_prefix_e2e \
  qwen35_vl_mtp_paged_prefix_cache_exact_hit_batch_and_image_miss \
  -- --ignored --test-threads=1 --nocapture
```

```sh
MLX_DIR=$HOME/.local/mlx \
QWEN36_MOE_MODEL=/path/to/Qwen3.6-35B-A3B-4bit/snapshots/<sha> \
QWEN36_MOE_MTP_MODEL=/path/to/Qwen3.6-35B-A3B-MTP-4bit/snapshots/<sha> \
cargo test --release -p ironmlx --test vl_mtp_paged_prefix_e2e \
  qwen36_moe_vl_mtp_paged_prefix_cache_exact_hit_batch_and_image_miss \
  -- --ignored --test-threads=1 --nocapture
```

## `/healthz` Acceptance

When the server starts with `--mtp-model-dir`, `/healthz.mtp.enabled` must be
`true` and `draft_tokens` must match the configured or model-aware default draft
depth.

For greedy eligible requests:

- `prefill_count` increases when the actor calls the scheduler MTP prefill path.
- `step_count` increases when the actor calls the scheduler MTP decode path.
- `drafted_tokens` and `accepted_draft_tokens` reflect the latest cumulative
  scheduler MTP stats, with `accepted_draft_tokens <= drafted_tokens`.

For non-greedy or otherwise ineligible requests:

- `fallback_prefill_count` increases.
- `prefill_count` and `step_count` do not increase because that request uses the
  ordinary scheduler path.
