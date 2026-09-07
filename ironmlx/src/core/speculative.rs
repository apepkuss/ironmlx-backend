//! Speculative decoding helpers shared by MTP generation paths.

use std::cell::Cell;
use std::collections::VecDeque;
use std::time::Instant;

use anyhow::anyhow;
use mlx::{random, Array, Dtype, StreamOrDevice};
use serde::{Deserialize, Serialize};

use crate::core::cache::{MtpCache, MtpCacheSnapshot};
use crate::core::constrained::{apply_speculative_token_masks, ConstraintSession};
use crate::core::generate::{build_position_ids, GenerateEvent, GenerateRequest};
#[cfg(test)]
use crate::core::sampler::SamplingDistribution;
use crate::core::sampler::{draw_uniforms, sample_target_tokens_with_uniforms_batch};
use crate::core::tokenizer::{DecodeStream, Tokenizer};
use crate::core::{Loader, Model, Sampler};
use crate::models::{Qwen35Model, Qwen35MoeModel, Qwen35MoeMtp, Qwen36MoeModel};
use crate::nn::{enable_turboquant_kv_caches, LayerCache, LayerCacheSnapshot, Mtp, MtpStepOutput};
use crate::Result;

/// Runtime limits for a single-request MTP speculative generation stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpSpeculativeConfig {
    pub max_draft_tokens: usize,
}

thread_local! {
    static QWEN_FIXED_DRAFT_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Benchmark-only scope that freezes Qwen's adaptive MTP draft policy at the
/// configured maximum depth. Production callers must retain the adaptive
/// policy so an unprofitable drafter can fail closed to ordinary decode.
#[doc(hidden)]
pub struct QwenFixedMtpDraftDepthScope;

impl Drop for QwenFixedMtpDraftDepthScope {
    fn drop(&mut self) {
        QWEN_FIXED_DRAFT_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

#[doc(hidden)]
pub fn qwen_fixed_mtp_draft_depth_scope() -> QwenFixedMtpDraftDepthScope {
    QWEN_FIXED_DRAFT_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
    QwenFixedMtpDraftDepthScope
}

fn qwen_fixed_mtp_draft_depth_is_armed() -> bool {
    QWEN_FIXED_DRAFT_DEPTH.with(|depth| depth.get() > 0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpDraftTokensArg {
    Explicit(usize),
    Omitted,
}

pub fn resolve_mtp_draft_tokens(raw_config: &serde_json::Value, arg: MtpDraftTokensArg) -> usize {
    match arg {
        MtpDraftTokensArg::Explicit(value) => value,
        MtpDraftTokensArg::Omitted => default_mtp_draft_tokens_for_config(raw_config),
    }
}

pub fn default_mtp_draft_tokens_for_config(raw_config: &serde_json::Value) -> usize {
    let model_type = raw_config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let text = raw_config
        .get("text_config")
        .and_then(serde_json::Value::as_object);
    let hidden_size = text
        .and_then(|value| value.get("hidden_size"))
        .and_then(serde_json::Value::as_i64);
    let layers = text
        .and_then(|value| value.get("num_hidden_layers"))
        .and_then(serde_json::Value::as_i64);
    let experts = text
        .and_then(|value| value.get("num_experts"))
        .and_then(serde_json::Value::as_i64);
    let experts_per_tok = text
        .and_then(|value| value.get("num_experts_per_tok"))
        .and_then(serde_json::Value::as_i64);

    match (model_type, hidden_size, layers, experts, experts_per_tok) {
        // Qwen3.6-27B Dense and Qwen3.8-27B Dense share this text
        // architecture and retain their pre-Gemma d=2 default.
        ("qwen3_5", Some(5120), Some(64), None, None) => 2,
        ("qwen3_5_moe", Some(2048), Some(40), Some(256), Some(8)) => 2,
        // Qwen3.5-4B and Gemma4 keep the conservative d=1 default.
        _ => 1,
    }
}

impl MtpSpeculativeConfig {
    pub fn new(max_draft_tokens: usize, sampler: Sampler) -> Result<Self> {
        if max_draft_tokens == 0 {
            return Err(anyhow!(
                "MtpSpeculativeConfig::new: max_draft_tokens must be > 0"
            ));
        }
        anyhow::ensure!(
            sampler.temperature.is_finite(),
            "sampling temperature must be finite"
        );
        Ok(Self { max_draft_tokens })
    }
}

const MAX_DRAFT_CAP_OBSERVATION_REGIMES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtpDraftCapContextBucket {
    UpTo2k,
    UpTo8k,
    UpTo32k,
    UpTo128k,
    Above128k,
}

impl MtpDraftCapContextBucket {
    pub fn for_tokens(tokens: usize) -> Self {
        match tokens {
            0..=2_048 => Self::UpTo2k,
            2_049..=8_192 => Self::UpTo8k,
            8_193..=32_768 => Self::UpTo32k,
            32_769..=131_072 => Self::UpTo128k,
            _ => Self::Above128k,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MtpDraftCapObservation {
    pub configured_max_draft_tokens: usize,
    pub min_draft_tokens: usize,
    pub max_draft_tokens: usize,
    pub batch_width: usize,
    pub context_bucket: MtpDraftCapContextBucket,
    pub mixed_context_buckets: bool,
    pub windows: usize,
    pub drafted_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub committed_tokens: usize,
    pub rollback_count: usize,
    pub total_us: u64,
    pub draft_forward_us: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub sampling_us: u64,
    pub main_rollback_us: u64,
    pub decode_cache_commit_us: u64,
    pub cache_restore_us: u64,
}

impl MtpDraftCapObservation {
    fn same_regime(&self, other: &Self) -> bool {
        self.configured_max_draft_tokens == other.configured_max_draft_tokens
            && self.min_draft_tokens == other.min_draft_tokens
            && self.max_draft_tokens == other.max_draft_tokens
            && self.batch_width == other.batch_width
            && self.context_bucket == other.context_bucket
            && self.mixed_context_buckets == other.mixed_context_buckets
    }

    fn add_assign(&mut self, other: &Self) {
        debug_assert!(self.same_regime(other));
        self.windows = self.windows.saturating_add(other.windows);
        self.drafted_tokens = self.drafted_tokens.saturating_add(other.drafted_tokens);
        self.accepted_draft_tokens = self
            .accepted_draft_tokens
            .saturating_add(other.accepted_draft_tokens);
        self.committed_tokens = self.committed_tokens.saturating_add(other.committed_tokens);
        self.rollback_count = self.rollback_count.saturating_add(other.rollback_count);
        self.total_us = self.total_us.saturating_add(other.total_us);
        self.draft_forward_us = self.draft_forward_us.saturating_add(other.draft_forward_us);
        self.verify_forward_us = self
            .verify_forward_us
            .saturating_add(other.verify_forward_us);
        self.projection_us = self.projection_us.saturating_add(other.projection_us);
        self.sampling_us = self.sampling_us.saturating_add(other.sampling_us);
        self.main_rollback_us = self.main_rollback_us.saturating_add(other.main_rollback_us);
        self.decode_cache_commit_us = self
            .decode_cache_commit_us
            .saturating_add(other.decode_cache_commit_us);
        self.cache_restore_us = self.cache_restore_us.saturating_add(other.cache_restore_us);
    }

    fn saturating_delta_since(&self, before: Option<&Self>) -> Self {
        let before = before.filter(|value| self.same_regime(value));
        Self {
            configured_max_draft_tokens: self.configured_max_draft_tokens,
            min_draft_tokens: self.min_draft_tokens,
            max_draft_tokens: self.max_draft_tokens,
            batch_width: self.batch_width,
            context_bucket: self.context_bucket,
            mixed_context_buckets: self.mixed_context_buckets,
            windows: self
                .windows
                .saturating_sub(before.map_or(0, |value| value.windows)),
            drafted_tokens: self
                .drafted_tokens
                .saturating_sub(before.map_or(0, |value| value.drafted_tokens)),
            accepted_draft_tokens: self
                .accepted_draft_tokens
                .saturating_sub(before.map_or(0, |value| value.accepted_draft_tokens)),
            committed_tokens: self
                .committed_tokens
                .saturating_sub(before.map_or(0, |value| value.committed_tokens)),
            rollback_count: self
                .rollback_count
                .saturating_sub(before.map_or(0, |value| value.rollback_count)),
            total_us: self
                .total_us
                .saturating_sub(before.map_or(0, |value| value.total_us)),
            draft_forward_us: self
                .draft_forward_us
                .saturating_sub(before.map_or(0, |value| value.draft_forward_us)),
            verify_forward_us: self
                .verify_forward_us
                .saturating_sub(before.map_or(0, |value| value.verify_forward_us)),
            projection_us: self
                .projection_us
                .saturating_sub(before.map_or(0, |value| value.projection_us)),
            sampling_us: self
                .sampling_us
                .saturating_sub(before.map_or(0, |value| value.sampling_us)),
            main_rollback_us: self
                .main_rollback_us
                .saturating_sub(before.map_or(0, |value| value.main_rollback_us)),
            decode_cache_commit_us: self
                .decode_cache_commit_us
                .saturating_sub(before.map_or(0, |value| value.decode_cache_commit_us)),
            cache_restore_us: self
                .cache_restore_us
                .saturating_sub(before.map_or(0, |value| value.cache_restore_us)),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MtpDraftCapTiming {
    draft_forward_us: u64,
    verify_forward_us: u64,
    projection_us: u64,
    sampling_us: u64,
    main_rollback_us: u64,
    decode_cache_commit_us: u64,
    cache_restore_us: u64,
}

/// Runtime counters collected by [`MtpTextGenerationStream`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MtpSpeculativeStats {
    /// Speculative windows verified by the main model.
    pub windows: usize,
    /// Draft tokens proposed by the MTP head.
    pub drafted_tokens: usize,
    /// Draft tokens accepted before mismatch.
    pub accepted_draft_tokens: usize,
    /// Non-greedy windows resolved with exact speculative sampling.
    pub exact_sampling_windows: usize,
    /// Acceptance Bernoulli draws consumed by exact sampling.
    pub exact_acceptance_draws: usize,
    /// Rejections corrected from the normalized positive residual `(p - q)+`.
    pub exact_residual_corrections: usize,
    /// Target-distribution bonus samples emitted after full draft acceptance.
    pub exact_bonus_samples: usize,
    /// Draft windows that attempted each zero-based draft position.
    pub draft_attempts_by_position: Vec<usize>,
    /// Draft windows that accepted each zero-based draft position.
    pub draft_accepts_by_position: Vec<usize>,
    /// Windows that required committing only an accepted main-cache prefix.
    pub rollback_count: usize,
    /// Windows that reused the temporary draft MTP cache after full acceptance.
    pub mtp_cache_reuse_count: usize,
    /// MTP cache token positions kept from the temporary draft cache.
    pub mtp_cache_reused_tokens: usize,
    /// Number of times adaptive draft budget decreased after a low-acceptance window.
    pub draft_budget_reductions: usize,
    /// Number of times adaptive draft budget increased after a full-acceptance window.
    pub draft_budget_increases: usize,
    /// Microseconds spent in MTP draft hidden forward passes.
    pub draft_forward_us: u64,
    /// Microseconds spent in main-model verify and fallback replay hidden forwards.
    pub verify_forward_us: u64,
    /// Microseconds spent projecting hidden states to logits.
    pub projection_us: u64,
    /// Microseconds spent sampling logits.
    pub sampling_us: u64,
    /// Host synchronizations performed while constructing neural draft chains.
    pub draft_host_sync_count: usize,
    /// Microseconds blocked on host synchronization while constructing draft chains.
    pub draft_host_sync_us: u64,
    /// Host synchronizations performed to resolve a verified speculative window.
    pub verify_accept_host_sync_count: usize,
    /// Microseconds blocked on the compact verify-acceptance result.
    pub verify_accept_host_sync_us: u64,
    /// Microseconds spent trimming, restoring, or replaying main KV after mismatch.
    pub main_rollback_us: u64,
    /// Microseconds spent committing accepted tokens into the MTP KV cache.
    pub mtp_cache_commit_us: u64,
    /// Microseconds spent building MTP KV cache entries during prompt prefill.
    pub mtp_prefill_cache_commit_us: u64,
    /// Microseconds spent committing accepted decode tokens into the MTP KV cache.
    pub mtp_decode_cache_commit_us: u64,
    /// Microseconds spent restoring the MTP KV cache after temporary draft.
    pub mtp_cache_restore_us: u64,
    /// Bounded, regime-level observations used only by offline draft-cap calibration.
    pub draft_cap_observations: Vec<MtpDraftCapObservation>,
    /// Windows omitted after the bounded observation table reached capacity.
    pub draft_cap_observation_dropped_windows: usize,
}

impl MtpSpeculativeStats {
    /// Speculative windows that attempted at least two draft tokens.
    pub fn multi_token_windows(&self) -> usize {
        self.draft_attempts_by_position
            .get(1)
            .copied()
            .unwrap_or_default()
    }

    pub(crate) fn draft_cap_timing(&self) -> MtpDraftCapTiming {
        MtpDraftCapTiming {
            draft_forward_us: self.draft_forward_us,
            verify_forward_us: self.verify_forward_us,
            projection_us: self.projection_us,
            sampling_us: self.sampling_us,
            main_rollback_us: self.main_rollback_us,
            decode_cache_commit_us: self.mtp_decode_cache_commit_us,
            cache_restore_us: self.mtp_cache_restore_us,
        }
    }

    pub fn saturating_delta_since(&self, before: &Self) -> Self {
        fn vec_delta(current: &[usize], before: &[usize]) -> Vec<usize> {
            let len = current.len().max(before.len());
            (0..len)
                .map(|idx| {
                    current
                        .get(idx)
                        .copied()
                        .unwrap_or_default()
                        .saturating_sub(before.get(idx).copied().unwrap_or_default())
                })
                .collect()
        }

        let draft_cap_observations = self
            .draft_cap_observations
            .iter()
            .map(|current| {
                let before = before
                    .draft_cap_observations
                    .iter()
                    .find(|value| current.same_regime(value));
                current.saturating_delta_since(before)
            })
            .filter(|value| value.windows > 0)
            .collect();

        Self {
            windows: self.windows.saturating_sub(before.windows),
            drafted_tokens: self.drafted_tokens.saturating_sub(before.drafted_tokens),
            accepted_draft_tokens: self
                .accepted_draft_tokens
                .saturating_sub(before.accepted_draft_tokens),
            exact_sampling_windows: self
                .exact_sampling_windows
                .saturating_sub(before.exact_sampling_windows),
            exact_acceptance_draws: self
                .exact_acceptance_draws
                .saturating_sub(before.exact_acceptance_draws),
            exact_residual_corrections: self
                .exact_residual_corrections
                .saturating_sub(before.exact_residual_corrections),
            exact_bonus_samples: self
                .exact_bonus_samples
                .saturating_sub(before.exact_bonus_samples),
            draft_attempts_by_position: vec_delta(
                &self.draft_attempts_by_position,
                &before.draft_attempts_by_position,
            ),
            draft_accepts_by_position: vec_delta(
                &self.draft_accepts_by_position,
                &before.draft_accepts_by_position,
            ),
            rollback_count: self.rollback_count.saturating_sub(before.rollback_count),
            mtp_cache_reuse_count: self
                .mtp_cache_reuse_count
                .saturating_sub(before.mtp_cache_reuse_count),
            mtp_cache_reused_tokens: self
                .mtp_cache_reused_tokens
                .saturating_sub(before.mtp_cache_reused_tokens),
            draft_budget_reductions: self
                .draft_budget_reductions
                .saturating_sub(before.draft_budget_reductions),
            draft_budget_increases: self
                .draft_budget_increases
                .saturating_sub(before.draft_budget_increases),
            draft_forward_us: self
                .draft_forward_us
                .saturating_sub(before.draft_forward_us),
            verify_forward_us: self
                .verify_forward_us
                .saturating_sub(before.verify_forward_us),
            projection_us: self.projection_us.saturating_sub(before.projection_us),
            sampling_us: self.sampling_us.saturating_sub(before.sampling_us),
            draft_host_sync_count: self
                .draft_host_sync_count
                .saturating_sub(before.draft_host_sync_count),
            draft_host_sync_us: self
                .draft_host_sync_us
                .saturating_sub(before.draft_host_sync_us),
            verify_accept_host_sync_count: self
                .verify_accept_host_sync_count
                .saturating_sub(before.verify_accept_host_sync_count),
            verify_accept_host_sync_us: self
                .verify_accept_host_sync_us
                .saturating_sub(before.verify_accept_host_sync_us),
            main_rollback_us: self
                .main_rollback_us
                .saturating_sub(before.main_rollback_us),
            mtp_cache_commit_us: self
                .mtp_cache_commit_us
                .saturating_sub(before.mtp_cache_commit_us),
            mtp_prefill_cache_commit_us: self
                .mtp_prefill_cache_commit_us
                .saturating_sub(before.mtp_prefill_cache_commit_us),
            mtp_decode_cache_commit_us: self
                .mtp_decode_cache_commit_us
                .saturating_sub(before.mtp_decode_cache_commit_us),
            mtp_cache_restore_us: self
                .mtp_cache_restore_us
                .saturating_sub(before.mtp_cache_restore_us),
            draft_cap_observations,
            draft_cap_observation_dropped_windows: self
                .draft_cap_observation_dropped_windows
                .saturating_sub(before.draft_cap_observation_dropped_windows),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_draft_cap_observation(
        &mut self,
        configured_max_draft_tokens: usize,
        draft_tokens_by_row: &[usize],
        context_tokens_by_row: &[usize],
        accepted_draft_tokens: usize,
        committed_tokens: usize,
        rollback_count: usize,
        total_us: u64,
        timing_delta: MtpDraftCapTiming,
    ) {
        if draft_tokens_by_row.is_empty()
            || draft_tokens_by_row.len() != context_tokens_by_row.len()
        {
            return;
        }
        let min_draft_tokens = draft_tokens_by_row.iter().copied().min().unwrap_or(0);
        let max_draft_tokens = draft_tokens_by_row.iter().copied().max().unwrap_or(0);
        if min_draft_tokens == 0 {
            return;
        }
        let first_context_bucket = MtpDraftCapContextBucket::for_tokens(context_tokens_by_row[0]);
        let mixed_context_buckets = context_tokens_by_row
            .iter()
            .copied()
            .map(MtpDraftCapContextBucket::for_tokens)
            .any(|bucket| bucket != first_context_bucket);
        let observation = MtpDraftCapObservation {
            configured_max_draft_tokens,
            min_draft_tokens,
            max_draft_tokens,
            batch_width: draft_tokens_by_row.len(),
            context_bucket: context_tokens_by_row
                .iter()
                .copied()
                .map(MtpDraftCapContextBucket::for_tokens)
                .max()
                .unwrap_or(first_context_bucket),
            mixed_context_buckets,
            windows: draft_tokens_by_row.len(),
            drafted_tokens: draft_tokens_by_row.iter().copied().sum(),
            accepted_draft_tokens,
            committed_tokens,
            rollback_count,
            total_us,
            draft_forward_us: timing_delta.draft_forward_us,
            verify_forward_us: timing_delta.verify_forward_us,
            projection_us: timing_delta.projection_us,
            sampling_us: timing_delta.sampling_us,
            main_rollback_us: timing_delta.main_rollback_us,
            decode_cache_commit_us: timing_delta.decode_cache_commit_us,
            cache_restore_us: timing_delta.cache_restore_us,
        };
        if let Some(current) = self
            .draft_cap_observations
            .iter_mut()
            .find(|value| value.same_regime(&observation))
        {
            current.add_assign(&observation);
        } else if self.draft_cap_observations.len() < MAX_DRAFT_CAP_OBSERVATION_REGIMES {
            self.draft_cap_observations.push(observation);
        } else {
            self.draft_cap_observation_dropped_windows = self
                .draft_cap_observation_dropped_windows
                .saturating_add(observation.windows);
        }
    }

    pub(crate) fn merge_from(&mut self, other: Self) {
        self.windows = self.windows.saturating_add(other.windows);
        self.drafted_tokens = self.drafted_tokens.saturating_add(other.drafted_tokens);
        self.accepted_draft_tokens = self
            .accepted_draft_tokens
            .saturating_add(other.accepted_draft_tokens);
        self.exact_sampling_windows = self
            .exact_sampling_windows
            .saturating_add(other.exact_sampling_windows);
        self.exact_acceptance_draws = self
            .exact_acceptance_draws
            .saturating_add(other.exact_acceptance_draws);
        self.exact_residual_corrections = self
            .exact_residual_corrections
            .saturating_add(other.exact_residual_corrections);
        self.exact_bonus_samples = self
            .exact_bonus_samples
            .saturating_add(other.exact_bonus_samples);
        merge_counter_vec(
            &mut self.draft_attempts_by_position,
            other.draft_attempts_by_position,
        );
        merge_counter_vec(
            &mut self.draft_accepts_by_position,
            other.draft_accepts_by_position,
        );
        self.rollback_count = self.rollback_count.saturating_add(other.rollback_count);
        self.mtp_cache_reuse_count = self
            .mtp_cache_reuse_count
            .saturating_add(other.mtp_cache_reuse_count);
        self.mtp_cache_reused_tokens = self
            .mtp_cache_reused_tokens
            .saturating_add(other.mtp_cache_reused_tokens);
        self.draft_budget_reductions = self
            .draft_budget_reductions
            .saturating_add(other.draft_budget_reductions);
        self.draft_budget_increases = self
            .draft_budget_increases
            .saturating_add(other.draft_budget_increases);
        self.draft_forward_us = self.draft_forward_us.saturating_add(other.draft_forward_us);
        self.verify_forward_us = self
            .verify_forward_us
            .saturating_add(other.verify_forward_us);
        self.projection_us = self.projection_us.saturating_add(other.projection_us);
        self.sampling_us = self.sampling_us.saturating_add(other.sampling_us);
        self.draft_host_sync_count = self
            .draft_host_sync_count
            .saturating_add(other.draft_host_sync_count);
        self.draft_host_sync_us = self
            .draft_host_sync_us
            .saturating_add(other.draft_host_sync_us);
        self.verify_accept_host_sync_count = self
            .verify_accept_host_sync_count
            .saturating_add(other.verify_accept_host_sync_count);
        self.verify_accept_host_sync_us = self
            .verify_accept_host_sync_us
            .saturating_add(other.verify_accept_host_sync_us);
        self.main_rollback_us = self.main_rollback_us.saturating_add(other.main_rollback_us);
        self.mtp_cache_commit_us = self
            .mtp_cache_commit_us
            .saturating_add(other.mtp_cache_commit_us);
        self.mtp_prefill_cache_commit_us = self
            .mtp_prefill_cache_commit_us
            .saturating_add(other.mtp_prefill_cache_commit_us);
        self.mtp_decode_cache_commit_us = self
            .mtp_decode_cache_commit_us
            .saturating_add(other.mtp_decode_cache_commit_us);
        self.mtp_cache_restore_us = self
            .mtp_cache_restore_us
            .saturating_add(other.mtp_cache_restore_us);
        for observation in other.draft_cap_observations {
            if let Some(current) = self
                .draft_cap_observations
                .iter_mut()
                .find(|value| value.same_regime(&observation))
            {
                current.add_assign(&observation);
            } else if self.draft_cap_observations.len() < MAX_DRAFT_CAP_OBSERVATION_REGIMES {
                self.draft_cap_observations.push(observation);
            } else {
                self.draft_cap_observation_dropped_windows = self
                    .draft_cap_observation_dropped_windows
                    .saturating_add(observation.windows);
            }
        }
        self.draft_cap_observation_dropped_windows = self
            .draft_cap_observation_dropped_windows
            .saturating_add(other.draft_cap_observation_dropped_windows);
    }

    pub fn record_window_acceptance(
        &mut self,
        attempted_draft_tokens: usize,
        accepted_draft_tokens: usize,
    ) {
        if attempted_draft_tokens == 0 {
            return;
        }
        let accepted = accepted_draft_tokens.min(attempted_draft_tokens);
        if self.draft_attempts_by_position.len() < attempted_draft_tokens {
            self.draft_attempts_by_position
                .resize(attempted_draft_tokens, 0);
            self.draft_accepts_by_position
                .resize(attempted_draft_tokens, 0);
        }
        for idx in 0..attempted_draft_tokens {
            self.draft_attempts_by_position[idx] =
                self.draft_attempts_by_position[idx].saturating_add(1);
            if idx < accepted {
                self.draft_accepts_by_position[idx] =
                    self.draft_accepts_by_position[idx].saturating_add(1);
            }
        }
    }

    pub(crate) fn record_exact_sampling(&mut self, counters: ExactSamplingCounters) {
        self.exact_sampling_windows = self.exact_sampling_windows.saturating_add(counters.windows);
        self.exact_acceptance_draws = self
            .exact_acceptance_draws
            .saturating_add(counters.acceptance_draws);
        self.exact_residual_corrections = self
            .exact_residual_corrections
            .saturating_add(counters.residual_corrections);
        self.exact_bonus_samples = self
            .exact_bonus_samples
            .saturating_add(counters.bonus_samples);
    }
}

impl MtpDraftCapTiming {
    pub(crate) fn saturating_delta_since(self, before: Self) -> Self {
        Self {
            draft_forward_us: self
                .draft_forward_us
                .saturating_sub(before.draft_forward_us),
            verify_forward_us: self
                .verify_forward_us
                .saturating_sub(before.verify_forward_us),
            projection_us: self.projection_us.saturating_sub(before.projection_us),
            sampling_us: self.sampling_us.saturating_sub(before.sampling_us),
            main_rollback_us: self
                .main_rollback_us
                .saturating_sub(before.main_rollback_us),
            decode_cache_commit_us: self
                .decode_cache_commit_us
                .saturating_sub(before.decode_cache_commit_us),
            cache_restore_us: self
                .cache_restore_us
                .saturating_sub(before.cache_restore_us),
        }
    }
}

fn merge_counter_vec(dst: &mut Vec<usize>, src: Vec<usize>) {
    if dst.len() < src.len() {
        dst.resize(src.len(), 0);
    }
    for (idx, value) in src.into_iter().enumerate() {
        dst[idx] = dst[idx].saturating_add(value);
    }
}

/// Narrow model capability required by single-request MTP speculative decoding.
pub trait MtpSpeculativeModel: Model {
    type MtpHead;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead>;

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache>;

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32;

    fn mtp_hidden_dtype(&self, mtp: &Self::MtpHead) -> Dtype;

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Model::project_hidden_on(self, hidden, target.into())
    }

    fn supports_mtp_accepted_prefix_restore(&self) -> bool {
        false
    }

    fn supports_affine8_b4_mtp_exact_hot_path(
        &self,
        _batch_width: usize,
        _verify_width: usize,
    ) -> bool {
        false
    }

    fn begin_mtp_accepted_prefix_capture(&self, _cache: &mut [LayerCache]) -> Result<()> {
        Err(anyhow!(
            "{} does not support MTP accepted-prefix capture",
            std::any::type_name::<Self>()
        ))
    }

    fn restore_mtp_accepted_prefix_rows_on(
        &self,
        _cache: &mut [LayerCache],
        _snapshots: &[LayerCacheSnapshot],
        _accepted_lens: &[usize],
        _target: StreamOrDevice,
    ) -> Result<()> {
        Err(anyhow!(
            "{} does not support MTP accepted-prefix restore",
            std::any::type_name::<Self>()
        ))
    }

    fn discard_mtp_accepted_prefix_capture(&self, cache: &mut [LayerCache]) {
        for layer in cache {
            layer.discard_speculative_prefix_capture();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array>;

    #[allow(clippy::too_many_arguments)]
    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput>;
}

impl MtpSpeculativeModel for Qwen35Model {
    type MtpHead = Mtp;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead> {
        Qwen35Model::load_mtp_head(self, loader)
    }

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache> {
        let layer_cfg = mtp.config().layer;
        MtpCache::new_with_cap(
            mtp.num_layers(),
            batch,
            layer_cfg.num_kv_heads,
            layer_cfg.head_dim,
            layer_cfg.head_dim,
            dtype,
            cap,
        )
    }

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35Model::project_mtp_verify_hidden_on(self, hidden, target)
    }

    fn supports_mtp_accepted_prefix_restore(&self) -> bool {
        true
    }

    fn supports_affine8_b4_mtp_exact_hot_path(
        &self,
        batch_width: usize,
        verify_width: usize,
    ) -> bool {
        Qwen35Model::supports_affine8_b4_mtp_exact_hot_path(self, batch_width, verify_width)
    }

    fn begin_mtp_accepted_prefix_capture(&self, cache: &mut [LayerCache]) -> Result<()> {
        for layer in cache {
            layer.begin_speculative_prefix_capture()?;
        }
        Ok(())
    }

    fn restore_mtp_accepted_prefix_rows_on(
        &self,
        cache: &mut [LayerCache],
        snapshots: &[LayerCacheSnapshot],
        accepted_lens: &[usize],
        target: StreamOrDevice,
    ) -> Result<()> {
        self.text().restore_dflash2_speculative_prefix_rows_on(
            cache,
            snapshots,
            accepted_lens,
            target,
        )
    }

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32 {
        mtp.config().hidden_size
    }

    fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
        self.hidden_dtype()
    }

    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput> {
        Qwen35Model::mtp_forward_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }

    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35Model::mtp_forward_hidden_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }
}

impl MtpSpeculativeModel for Qwen35MoeModel {
    type MtpHead = Qwen35MoeMtp;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead> {
        Qwen35MoeModel::load_mtp_head(self, loader)
    }

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache> {
        let layer_cfg = mtp.config().layer;
        MtpCache::new_with_cap(
            mtp.num_layers(),
            batch,
            layer_cfg.num_kv_heads,
            layer_cfg.head_dim,
            layer_cfg.head_dim,
            dtype,
            cap,
        )
    }

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35MoeModel::project_mtp_verify_hidden_on(self, hidden, target)
    }

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32 {
        mtp.config().hidden_size
    }

    fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
        self.hidden_dtype()
    }

    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput> {
        Qwen35MoeModel::mtp_forward_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }

    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen35MoeModel::mtp_forward_hidden_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }
}

impl MtpSpeculativeModel for Qwen36MoeModel {
    type MtpHead = Qwen35MoeMtp;

    fn load_mtp_head(&self, loader: &Loader) -> Result<Self::MtpHead> {
        Qwen36MoeModel::load_mtp_head(self, loader)
    }

    fn make_mtp_cache(
        &self,
        mtp: &Self::MtpHead,
        batch: i32,
        cap: i32,
        dtype: Dtype,
    ) -> Result<MtpCache> {
        let layer_cfg = mtp.config().layer;
        MtpCache::new_with_cap(
            mtp.num_layers(),
            batch,
            layer_cfg.num_kv_heads,
            layer_cfg.head_dim,
            layer_cfg.head_dim,
            dtype,
            cap,
        )
    }

    fn project_mtp_verify_hidden_on(
        &self,
        hidden: &Array,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen36MoeModel::project_mtp_verify_hidden_on(self, hidden, target)
    }

    fn mtp_hidden_size(&self, mtp: &Self::MtpHead) -> i32 {
        mtp.config().hidden_size
    }

    fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
        self.hidden_dtype()
    }

    fn mtp_forward_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<MtpStepOutput> {
        Qwen36MoeModel::mtp_forward_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }

    fn mtp_forward_hidden_on(
        &self,
        mtp: &Self::MtpHead,
        hidden_states: &Array,
        next_token_ids: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        mtp_cache: Option<&mut MtpCache>,
        target: impl Into<StreamOrDevice>,
    ) -> Result<Array> {
        Qwen36MoeModel::mtp_forward_hidden_on(
            self,
            mtp,
            hidden_states,
            next_token_ids,
            position_ids,
            mask,
            mtp_cache,
            target,
        )
    }
}

pub(crate) fn elapsed_us_since(start: Instant) -> u64 {
    start.elapsed().as_micros().min(u128::from(u64::MAX)) as u64
}

pub(crate) fn add_elapsed_us(counter: &mut u64, start: Instant) {
    *counter = counter.saturating_add(elapsed_us_since(start));
}

pub(crate) fn add_mtp_prefill_cache_commit_us(stats: &mut MtpSpeculativeStats, start: Instant) {
    let elapsed = elapsed_us_since(start);
    stats.mtp_cache_commit_us = stats.mtp_cache_commit_us.saturating_add(elapsed);
    stats.mtp_prefill_cache_commit_us = stats.mtp_prefill_cache_commit_us.saturating_add(elapsed);
}

pub(crate) fn add_mtp_decode_cache_commit_us(stats: &mut MtpSpeculativeStats, start: Instant) {
    let elapsed = elapsed_us_since(start);
    stats.mtp_cache_commit_us = stats.mtp_cache_commit_us.saturating_add(elapsed);
    stats.mtp_decode_cache_commit_us = stats.mtp_decode_cache_commit_us.saturating_add(elapsed);
}

/// Outcome of comparing MTP draft tokens with the main model's verified tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeResolution {
    /// Number of MTP draft tokens accepted before the first mismatch.
    pub accepted_draft_len: usize,
    /// Tokens that should be appended to generation history:
    /// accepted draft tokens plus either the corrected token or the bonus token.
    pub tokens_to_append: Vec<u32>,
    /// Number of verify input tokens that must remain in the main KV cache.
    ///
    /// The verify input is `[current_token] + draft_tokens`; keeping
    /// `accepted_draft_len + 1` positions preserves the current token and the
    /// accepted draft prefix.
    pub accepted_verify_input_len: usize,
    /// Whether the caller must rollback the main KV cache after a full-window
    /// verify pass.
    pub needs_rollback: bool,
    pub(crate) exact_sampling: ExactSamplingCounters,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ExactSamplingCounters {
    pub windows: usize,
    pub acceptance_draws: usize,
    pub residual_corrections: usize,
    pub bonus_samples: usize,
}

#[derive(Debug, Clone)]
pub(crate) enum DraftTokenDistribution {
    Deterministic,
    #[cfg(test)]
    Sampled(SamplingDistribution),
}

pub(crate) fn split_speculative_draft_prng(prng_state: &mut Array) -> Result<Array> {
    anyhow::ensure!(
        prng_state.size() == 2,
        "speculative PRNG state must contain one two-word key, got shape {:?}",
        prng_state.shape().as_slice()
    );
    let original_shape = prng_state.shape().as_slice().to_vec();
    let flat = prng_state.reshape(&[2_i32][..])?;
    let (next_decision_key, draft_key) = random::split(&flat)?;
    *prng_state = next_decision_key.reshape(original_shape.as_slice())?;
    Ok(draft_key)
}

pub(crate) fn sample_draft_logits_position(
    logits: &Array,
    _sampler: Sampler,
    _history: &[u32],
    _draft_prng: Option<&mut Array>,
) -> Result<(u32, DraftTokenDistribution)> {
    let dims = logits.shape();
    let dims = dims.as_slice();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] == 1,
        "draft logits must be [1, 1, V], got {dims:?}"
    );
    let vocab = dims[2];
    let row = logits.reshape((vocab,))?;
    let token = mlx::ops::reduction::argmax(&row, -1, false)?.item::<u32>()?;
    Ok((token, DraftTokenDistribution::Deterministic))
}

pub(crate) fn sample_draft_logits_position_with_uniform(
    logits: &Array,
    _sampler: Sampler,
    _history: &[u32],
    _uniform: f32,
) -> Result<(u32, DraftTokenDistribution)> {
    let dims = logits.shape();
    let dims = dims.as_slice();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] == 1,
        "draft logits must be [1, 1, V], got {dims:?}"
    );
    let vocab = dims[2];
    let row = logits.reshape((vocab,))?;
    let token = mlx::ops::reduction::argmax(&row, -1, false)?.item::<u32>()?;
    Ok((token, DraftTokenDistribution::Deterministic))
}

#[cfg(test)]
impl DraftTokenDistribution {
    fn acceptance_probability(&self, target: &SamplingDistribution, token: u32) -> Result<f32> {
        match self {
            Self::Deterministic => Ok(target.probability(token)),
            #[cfg(test)]
            Self::Sampled(draft) => target.acceptance_probability(draft, token),
        }
    }

    fn residual(&self, target: &SamplingDistribution, token: u32) -> Result<SamplingDistribution> {
        match self {
            Self::Deterministic => target.residual_point_mass(token),
            #[cfg(test)]
            Self::Sampled(draft) => target.residual(draft),
        }
    }
}

pub fn resolve_speculative_tokens(
    draft_tokens: &[u32],
    verified_tokens: &[u32],
) -> Result<SpeculativeResolution> {
    if verified_tokens.len() != draft_tokens.len() + 1 {
        return Err(anyhow!(
            "resolve_speculative_tokens: verified tokens len {} != draft len {} + 1",
            verified_tokens.len(),
            draft_tokens.len()
        ));
    }

    let accepted_draft_len = draft_tokens
        .iter()
        .zip(verified_tokens.iter())
        .take_while(|(draft, verified)| draft == verified)
        .count();
    let mut tokens_to_append = Vec::with_capacity(accepted_draft_len + 1);
    tokens_to_append.extend_from_slice(&draft_tokens[..accepted_draft_len]);
    tokens_to_append.push(verified_tokens[accepted_draft_len]);
    let accepted_verify_input_len = accepted_draft_len + 1;
    let needs_rollback = accepted_draft_len < draft_tokens.len();

    Ok(SpeculativeResolution {
        accepted_draft_len,
        tokens_to_append,
        accepted_verify_input_len,
        needs_rollback,
        exact_sampling: ExactSamplingCounters::default(),
    })
}

#[cfg(test)]
pub(crate) fn resolve_exact_speculative_logits(
    draft_tokens: &[u32],
    draft_distributions: &[DraftTokenDistribution],
    target_logits: &Array,
    sampler: Sampler,
    history: &[u32],
    prng_state: &mut Array,
) -> Result<SpeculativeResolution> {
    anyhow::ensure!(
        sampler.temperature > 0.0,
        "exact speculative sampling requires temperature > 0"
    );
    anyhow::ensure!(
        draft_tokens.len() == draft_distributions.len(),
        "exact speculative draft token count {} != distribution count {}",
        draft_tokens.len(),
        draft_distributions.len()
    );
    let shape = target_logits.shape();
    let dims = shape.as_slice();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1,
        "exact speculative target logits must be [1, S, V], got {dims:?}"
    );
    anyhow::ensure!(
        dims[1] as usize == draft_tokens.len() + 1,
        "exact speculative target positions {} != draft count {} + 1",
        dims[1],
        draft_tokens.len()
    );

    let target_distributions =
        speculative_target_distributions(target_logits, sampler, history, draft_tokens)?;
    let uniforms = draw_uniforms(prng_state, draft_tokens.len() + 1)?;
    let correction_uniform = uniforms[draft_tokens.len()];
    let mut tokens_to_append = Vec::with_capacity(draft_tokens.len() + 1);
    let mut exact_sampling = ExactSamplingCounters {
        windows: 1,
        ..ExactSamplingCounters::default()
    };
    for (position, (&draft_token, draft_distribution)) in
        draft_tokens.iter().zip(draft_distributions).enumerate()
    {
        let target = &target_distributions[position];
        let accept_probability = draft_distribution.acceptance_probability(&target, draft_token)?;
        exact_sampling.acceptance_draws = exact_sampling.acceptance_draws.saturating_add(1);
        if uniforms[position] < accept_probability {
            tokens_to_append.push(draft_token);
            continue;
        }

        let corrected = draft_distribution
            .residual(target, draft_token)?
            .sample_with_uniform(correction_uniform)?;
        exact_sampling.residual_corrections = exact_sampling.residual_corrections.saturating_add(1);
        tokens_to_append.push(corrected);
        return Ok(SpeculativeResolution {
            accepted_draft_len: position,
            tokens_to_append,
            accepted_verify_input_len: position + 1,
            needs_rollback: true,
            exact_sampling,
        });
    }

    tokens_to_append
        .push(target_distributions[draft_tokens.len()].sample_with_uniform(correction_uniform)?);
    exact_sampling.bonus_samples = exact_sampling.bonus_samples.saturating_add(1);
    Ok(SpeculativeResolution {
        accepted_draft_len: draft_tokens.len(),
        tokens_to_append,
        accepted_verify_input_len: draft_tokens.len() + 1,
        needs_rollback: false,
        exact_sampling,
    })
}

#[cfg(test)]
pub(crate) fn resolve_exact_deterministic_target_distributions(
    draft_tokens: &[u32],
    target_distributions: &[SamplingDistribution],
    prng_state: &mut Array,
) -> Result<SpeculativeResolution> {
    anyhow::ensure!(
        target_distributions.len() == draft_tokens.len() + 1,
        "exact deterministic target distribution count {} != draft count {} + 1",
        target_distributions.len(),
        draft_tokens.len()
    );
    let uniforms = draw_uniforms(prng_state, target_distributions.len())?;
    let mut tokens_to_append = Vec::with_capacity(target_distributions.len());
    let mut exact_sampling = ExactSamplingCounters {
        windows: 1,
        ..ExactSamplingCounters::default()
    };

    for (position, &draft_token) in draft_tokens.iter().enumerate() {
        let target_token =
            target_distributions[position].sample_with_uniform(uniforms[position])?;
        exact_sampling.acceptance_draws = exact_sampling.acceptance_draws.saturating_add(1);
        if target_token == draft_token {
            tokens_to_append.push(draft_token);
            continue;
        }

        tokens_to_append.push(target_token);
        exact_sampling.residual_corrections = exact_sampling.residual_corrections.saturating_add(1);
        return Ok(SpeculativeResolution {
            accepted_draft_len: position,
            tokens_to_append,
            accepted_verify_input_len: position + 1,
            needs_rollback: true,
            exact_sampling,
        });
    }

    tokens_to_append.push(
        target_distributions[draft_tokens.len()]
            .sample_with_uniform(uniforms[draft_tokens.len()])?,
    );
    exact_sampling.bonus_samples = exact_sampling.bonus_samples.saturating_add(1);
    Ok(SpeculativeResolution {
        accepted_draft_len: draft_tokens.len(),
        tokens_to_append,
        accepted_verify_input_len: draft_tokens.len() + 1,
        needs_rollback: false,
        exact_sampling,
    })
}

pub(crate) fn resolve_exact_deterministic_target_tokens(
    draft_tokens: &[u32],
    target_tokens: &[u32],
) -> Result<SpeculativeResolution> {
    anyhow::ensure!(
        target_tokens.len() == draft_tokens.len() + 1,
        "exact deterministic target token count {} != draft count {} + 1",
        target_tokens.len(),
        draft_tokens.len()
    );
    let accepted_draft_len = draft_tokens
        .iter()
        .zip(target_tokens)
        .take_while(|(draft, target)| draft == target)
        .count();
    let mut tokens_to_append = Vec::with_capacity(accepted_draft_len + 1);
    tokens_to_append.extend_from_slice(&draft_tokens[..accepted_draft_len]);
    tokens_to_append.push(target_tokens[accepted_draft_len]);
    let mismatch = accepted_draft_len < draft_tokens.len();

    Ok(SpeculativeResolution {
        accepted_draft_len,
        tokens_to_append,
        accepted_verify_input_len: accepted_draft_len + 1,
        needs_rollback: mismatch,
        exact_sampling: ExactSamplingCounters {
            windows: 1,
            acceptance_draws: if mismatch {
                accepted_draft_len + 1
            } else {
                draft_tokens.len()
            },
            residual_corrections: usize::from(mismatch),
            bonus_samples: usize::from(!mismatch),
        },
    })
}

/// Exact coupling for a deterministic draft distribution `q = delta(draft)`.
///
/// Sampling once from each target distribution `p` is equivalent to the
/// standard accept/reject algorithm: a sampled draft token is accepted when
/// the target sample matches it; otherwise that same target sample has the
/// conditional residual distribution over all non-draft tokens.
pub(crate) fn resolve_exact_deterministic_target_logits(
    draft_tokens: &[u32],
    target_logits: &Array,
    sampler: Sampler,
    history: &[u32],
    prng_state: &mut Array,
) -> Result<SpeculativeResolution> {
    anyhow::ensure!(
        sampler.temperature > 0.0,
        "exact deterministic target sampling requires temperature > 0"
    );
    let shape = target_logits.shape();
    let dims = shape.as_slice();
    let positions = draft_tokens.len() + 1;
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] as usize == positions,
        "exact deterministic target logits must be [1, {positions}, V], got {dims:?}"
    );
    let rows = target_logits.reshape(&[i32::try_from(positions)?, dims[2]][..])?;
    let histories = (0..positions)
        .map(|position| {
            let mut position_history = Vec::with_capacity(history.len() + position);
            position_history.extend_from_slice(history);
            position_history.extend_from_slice(&draft_tokens[..position]);
            position_history
        })
        .collect::<Vec<_>>();
    let history_refs = histories.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let sampler_refs = vec![&sampler; positions];
    let uniforms = draw_uniforms(prng_state, positions)?;
    let target_tokens =
        sample_target_tokens_with_uniforms_batch(&sampler_refs, &rows, &history_refs, &uniforms)?;
    resolve_exact_deterministic_target_tokens(draft_tokens, &target_tokens)
}

#[cfg(test)]
fn speculative_target_distributions(
    logits: &Array,
    sampler: Sampler,
    history: &[u32],
    draft_tokens: &[u32],
) -> Result<Vec<SamplingDistribution>> {
    let dims = logits.shape();
    let dims = dims.as_slice();
    let positions = draft_tokens.len() + 1;
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] as usize == positions,
        "speculative target distributions require [1, {positions}, V], got {dims:?}"
    );
    let rows = logits.reshape(&[i32::try_from(positions)?, dims[2]][..])?;
    let histories = (0..positions)
        .map(|position| {
            let mut position_history = Vec::with_capacity(history.len() + position);
            position_history.extend_from_slice(history);
            position_history.extend_from_slice(&draft_tokens[..position]);
            position_history
        })
        .collect::<Vec<_>>();
    let history_refs = histories.iter().map(Vec::as_slice).collect::<Vec<_>>();
    sampler.distributions(&rows, &history_refs)
}

#[derive(Debug)]
pub(crate) struct MtpDraftResult {
    pub tokens: Vec<u32>,
    pub distributions: Vec<DraftTokenDistribution>,
    pub cache_snapshot: MtpCacheSnapshot,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MtpDraftPolicyWindow {
    pub attempted_draft_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub committed_tokens: usize,
    pub total_us: u64,
    pub context_tokens: usize,
    pub batch_width: usize,
    pub kv_state: MtpDraftPolicyKvState,
    pub draft_forward_us: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub sampling_us: u64,
    pub verify_accept_host_sync_us: u64,
    pub main_rollback_us: u64,
    pub mtp_cache_commit_us: u64,
    pub mtp_prefill_cache_commit_us: u64,
    pub mtp_decode_cache_commit_us: u64,
    pub mtp_cache_restore_us: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(crate) enum MtpDraftPolicyKvState {
    #[default]
    Contiguous,
    Paged,
    PagedActiveKv,
}

impl MtpDraftPolicyKvState {
    pub(crate) fn from_runtime(paged: bool, active_kv: bool) -> Self {
        if active_kv {
            Self::PagedActiveKv
        } else if paged {
            Self::Paged
        } else {
            Self::Contiguous
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MtpDraftPolicyRegime {
    context_bucket: MtpDraftCapContextBucket,
    batch_width: usize,
    kv_state: MtpDraftPolicyKvState,
}

impl MtpDraftPolicyWindow {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_stats_delta(
        attempted_draft_tokens: usize,
        accepted_draft_tokens: usize,
        committed_tokens: usize,
        total_us: u64,
        context_tokens: usize,
        batch_width: usize,
        kv_state: MtpDraftPolicyKvState,
        delta: &MtpSpeculativeStats,
    ) -> Self {
        Self {
            attempted_draft_tokens,
            accepted_draft_tokens,
            committed_tokens,
            total_us,
            context_tokens,
            batch_width,
            kv_state,
            draft_forward_us: delta.draft_forward_us,
            verify_forward_us: delta.verify_forward_us,
            projection_us: delta.projection_us,
            sampling_us: delta.sampling_us,
            verify_accept_host_sync_us: delta.verify_accept_host_sync_us,
            main_rollback_us: delta.main_rollback_us,
            mtp_cache_commit_us: delta.mtp_cache_commit_us,
            mtp_prefill_cache_commit_us: delta.mtp_prefill_cache_commit_us,
            mtp_decode_cache_commit_us: delta.mtp_decode_cache_commit_us,
            mtp_cache_restore_us: delta.mtp_cache_restore_us,
        }
    }

    fn measured_components_us(self) -> u64 {
        self.draft_forward_us
            .saturating_add(self.verify_forward_us)
            .saturating_add(self.projection_us)
            .saturating_add(self.sampling_us)
            .saturating_add(self.verify_accept_host_sync_us)
            .saturating_add(self.main_rollback_us)
            .saturating_add(self.mtp_decode_cache_commit_us)
            .saturating_add(self.mtp_cache_restore_us)
    }

    fn gemma4_cost_per_committed_token_us(self) -> f64 {
        let measured_components_us = self.measured_components_us();
        let comparable_us = if self.attempted_draft_tokens == 0 && measured_components_us > 0 {
            // A zero-draft control window still runs inside speculative
            // bookkeeping so the drafter cache can be resumed if it wins.
            // Snapshot/resolve/state-maintenance overhead disappears after a
            // permanent switch to the ordinary scheduler and must not make
            // that control path look artificially expensive.
            measured_components_us
        } else {
            self.total_us.max(measured_components_us)
        };
        comparable_us as f64 / self.committed_tokens.max(1) as f64
    }

    fn qwen_cost_per_committed_token_us(self) -> f64 {
        self.total_us.max(self.measured_components_us()) as f64
            / self.committed_tokens.max(1) as f64
    }

    fn regime(self) -> MtpDraftPolicyRegime {
        MtpDraftPolicyRegime {
            context_bucket: MtpDraftCapContextBucket::for_tokens(self.context_tokens),
            batch_width: self.batch_width.max(1),
            kv_state: self.kv_state,
        }
    }

    fn acceptance_rate(self) -> f64 {
        if self.attempted_draft_tokens == 0 {
            1.0
        } else {
            self.accepted_draft_tokens.min(self.attempted_draft_tokens) as f64
                / self.attempted_draft_tokens as f64
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MtpDraftBudgetChange {
    pub reduced: bool,
    pub increased: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Gemma4DrafterPolicyState {
    max_draft_tokens: usize,
    current_budget: usize,
    acceptance_ewma: Option<f64>,
    active_regime: Option<MtpDraftPolicyRegime>,
    cost_estimates: Vec<MtpDraftCostEstimate>,
    probe_budget: Option<usize>,
    probe_origin_budget: Option<usize>,
    probe_windows_remaining: usize,
    cooldown_windows: usize,
    next_probe_cooldown_windows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct Gemma4DrafterPolicySnapshot {
    max_draft_tokens: usize,
    current_budget: usize,
    acceptance_ewma_bits: Option<u64>,
    active_regime: Option<MtpDraftPolicyRegime>,
    cost_estimates: Vec<MtpDraftCostEstimateSnapshot>,
    probe_budget: Option<usize>,
    probe_origin_budget: Option<usize>,
    probe_windows_remaining: usize,
    cooldown_windows: usize,
    next_probe_cooldown_windows: usize,
}

#[derive(Debug, Clone)]
struct MtpDraftCostEstimate {
    regime: MtpDraftPolicyRegime,
    draft_tokens: usize,
    cost_ewma: f64,
    acceptance_ewma: f64,
    samples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MtpDraftCostEstimateSnapshot {
    regime: MtpDraftPolicyRegime,
    draft_tokens: usize,
    cost_ewma_bits: u64,
    acceptance_ewma_bits: u64,
    samples: usize,
}

impl Gemma4DrafterPolicyState {
    const EWMA_ALPHA: f64 = 0.35;
    const LOW_ACCEPTANCE: f64 = 0.50;
    const HIGH_ACCEPTANCE: f64 = 0.85;
    const MIN_COST_SAMPLES: usize = 2;
    const PROBE_WINDOWS: usize = 2;
    const ZERO_DRAFT_MIN_COST_SAMPLES: usize = 8;
    // At >32K, waiting for eight single-draft windows before measuring
    // ordinary decode allows a costly path to consume most of a typical
    // response. Two complete MTP windows already include target, draft,
    // sampling, and cache costs; the separate four-window ordinary probe
    // remains the noise filter for the final decision.
    const LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES: usize = 2;
    const ZERO_DRAFT_PROBE_WINDOWS: usize = 4;
    const INITIAL_PROBE_COOLDOWN_WINDOWS: usize = 8;
    const MAX_PROBE_COOLDOWN_WINDOWS: usize = 64;
    const COST_IMPROVEMENT_RATIO: f64 = 0.95;
    // Ordinary decode is the safe control path. Require only a small measured
    // margin before bypassing MTP so a 5-10% speculative regression is not
    // hidden by overly conservative hysteresis. The four-window probe and
    // EWMA still absorb single-window timing noise.
    const ZERO_DRAFT_COST_IMPROVEMENT_RATIO: f64 = 0.98;

    pub(crate) fn new(max_draft_tokens: usize) -> Self {
        let max_draft_tokens = max_draft_tokens.max(1);
        Self {
            max_draft_tokens,
            current_budget: max_draft_tokens,
            acceptance_ewma: None,
            active_regime: None,
            cost_estimates: Vec::new(),
            probe_budget: None,
            probe_origin_budget: None,
            probe_windows_remaining: 0,
            cooldown_windows: 0,
            next_probe_cooldown_windows: (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2)
                .min(Self::MAX_PROBE_COOLDOWN_WINDOWS),
        }
    }

    pub(crate) fn current_budget(&self) -> usize {
        self.current_budget.min(self.max_draft_tokens)
    }

    pub(crate) fn seed_initial_budget(&mut self, budget: usize) -> bool {
        if self.active_regime.is_some()
            || !self.cost_estimates.is_empty()
            || self.probe_budget.is_some()
        {
            return false;
        }
        let seeded = budget.clamp(1, self.max_draft_tokens);
        let changed = seeded != self.current_budget;
        self.current_budget = seeded;
        changed
    }

    #[cfg(test)]
    pub(crate) fn should_maintain_mtp_cache(&self) -> bool {
        self.current_budget() > 0 || self.probe_budget.is_some()
    }

    pub(crate) fn uses_ordinary_decode(&self) -> bool {
        self.current_budget() == 0 && self.probe_budget.is_none()
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> Gemma4DrafterPolicySnapshot {
        Gemma4DrafterPolicySnapshot {
            max_draft_tokens: self.max_draft_tokens,
            current_budget: self.current_budget,
            acceptance_ewma_bits: self.acceptance_ewma.map(f64::to_bits),
            active_regime: self.active_regime,
            cost_estimates: self
                .cost_estimates
                .iter()
                .map(|estimate| MtpDraftCostEstimateSnapshot {
                    regime: estimate.regime,
                    draft_tokens: estimate.draft_tokens,
                    cost_ewma_bits: estimate.cost_ewma.to_bits(),
                    acceptance_ewma_bits: estimate.acceptance_ewma.to_bits(),
                    samples: estimate.samples,
                })
                .collect(),
            probe_budget: self.probe_budget,
            probe_origin_budget: self.probe_origin_budget,
            probe_windows_remaining: self.probe_windows_remaining,
            cooldown_windows: self.cooldown_windows,
            next_probe_cooldown_windows: self.next_probe_cooldown_windows,
        }
    }

    #[cfg(test)]
    pub(crate) fn restore_snapshot(&mut self, snapshot: Gemma4DrafterPolicySnapshot) -> Result<()> {
        anyhow::ensure!(
            snapshot.max_draft_tokens == self.max_draft_tokens,
            "MTP draft policy snapshot max {} != destination max {}",
            snapshot.max_draft_tokens,
            self.max_draft_tokens
        );
        anyhow::ensure!(
            snapshot.current_budget <= snapshot.max_draft_tokens,
            "MTP draft policy snapshot budget {} is outside [0, {}]",
            snapshot.current_budget,
            snapshot.max_draft_tokens
        );
        self.current_budget = snapshot.current_budget;
        self.acceptance_ewma = snapshot.acceptance_ewma_bits.map(f64::from_bits);
        self.active_regime = snapshot.active_regime;
        self.cost_estimates = snapshot
            .cost_estimates
            .into_iter()
            .map(|estimate| MtpDraftCostEstimate {
                regime: estimate.regime,
                draft_tokens: estimate.draft_tokens,
                cost_ewma: f64::from_bits(estimate.cost_ewma_bits),
                acceptance_ewma: f64::from_bits(estimate.acceptance_ewma_bits),
                samples: estimate.samples,
            })
            .collect();
        self.probe_budget = snapshot.probe_budget;
        self.probe_origin_budget = snapshot.probe_origin_budget;
        self.probe_windows_remaining = snapshot.probe_windows_remaining;
        self.cooldown_windows = snapshot.cooldown_windows;
        self.next_probe_cooldown_windows = snapshot.next_probe_cooldown_windows;
        Ok(())
    }

    pub(crate) fn observe_window(&mut self, window: MtpDraftPolicyWindow) -> MtpDraftBudgetChange {
        let regime = window.regime();
        if self.active_regime != Some(regime) {
            self.active_regime = Some(regime);
            self.acceptance_ewma = None;
            self.cost_estimates.clear();
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            self.cooldown_windows = 0;
            self.next_probe_cooldown_windows =
                (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2).min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
        }
        let old = self.current_budget();

        let acceptance = window.acceptance_rate();
        self.acceptance_ewma = Some(update_ewma(
            self.acceptance_ewma,
            acceptance,
            Self::EWMA_ALPHA,
        ));
        self.record_cost(
            regime,
            window.attempted_draft_tokens,
            window.gemma4_cost_per_committed_token_us(),
            acceptance,
        );
        if window.attempted_draft_tokens != old {
            return MtpDraftBudgetChange::default();
        }

        let zero_draft_probe_min_samples =
            if self.acceptance_ewma.unwrap_or(acceptance) < Self::LOW_ACCEPTANCE {
                Self::MIN_COST_SAMPLES
            } else if matches!(
                regime.context_bucket,
                MtpDraftCapContextBucket::UpTo128k | MtpDraftCapContextBucket::Above128k
            ) {
                Self::LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES
            } else {
                Self::ZERO_DRAFT_MIN_COST_SAMPLES
            };
        if old == 1
            && self.probe_budget.is_none()
            && self.cost_estimate(regime, 0).is_none()
            && self
                .cost_estimate(regime, old)
                .is_some_and(|estimate| estimate.samples >= zero_draft_probe_min_samples)
        {
            self.current_budget = 0;
            if self.acceptance_ewma.unwrap_or(acceptance) < Self::LOW_ACCEPTANCE {
                // At persistently low acceptance, a one-token drafter cannot
                // amortize target verification. Commit directly to ordinary
                // decode so architecture-specific schedulers can leave the
                // speculative path instead of benchmarking a slower
                // zero-draft emulation of it.
                self.probe_budget = None;
                self.probe_origin_budget = None;
                self.probe_windows_remaining = 0;
            } else {
                self.probe_budget = Some(0);
                self.probe_origin_budget = Some(old);
                self.probe_windows_remaining = Self::ZERO_DRAFT_PROBE_WINDOWS;
            }
            return budget_change(old, self.current_budget);
        }
        if old == 0 && self.probe_budget.is_none() {
            return MtpDraftBudgetChange::default();
        }
        if old == 1
            && self.probe_budget.is_none()
            && self.cost_estimate(regime, 0).is_some()
            && self.cooldown_windows == 0
            && self
                .cost_estimate(regime, 1)
                .is_some_and(|estimate| estimate.samples >= Self::ZERO_DRAFT_MIN_COST_SAMPLES)
        {
            // A previous zero-draft sample is stale once the response enters
            // a different acceptance/cost phase. Re-test ordinary decode only
            // while the MTP cache is still synchronized, and discard the old
            // control estimate so it cannot drive an uncontrolled switch.
            self.cost_estimates
                .retain(|estimate| estimate.regime != regime || estimate.draft_tokens != 0);
            self.current_budget = 0;
            self.probe_budget = Some(0);
            self.probe_origin_budget = Some(old);
            self.probe_windows_remaining = Self::ZERO_DRAFT_PROBE_WINDOWS;
            return budget_change(old, self.current_budget);
        }

        let mut next = old;
        let full_accept = window.accepted_draft_tokens == window.attempted_draft_tokens;
        if !full_accept {
            let rejected_probe = self.probe_budget == Some(old);
            next = window.accepted_draft_tokens.saturating_add(1).min(old);
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            if rejected_probe {
                self.back_off_next_probe();
            } else if old == 1 {
                self.cooldown_windows = self.cooldown_windows.saturating_sub(1);
            } else {
                self.arm_initial_probe_cooldown();
            }
        } else if self.probe_budget == Some(old) {
            self.probe_windows_remaining = self.probe_windows_remaining.saturating_sub(1);
            if self.probe_windows_remaining == 0 {
                let origin = self.probe_origin_budget.take();
                self.probe_budget = None;
                next = origin.map_or(old, |origin| {
                    self.preferred_probe_budget(regime, origin, old)
                });
                if origin.is_some_and(|origin| next == origin) && origin != Some(0) {
                    self.back_off_next_probe();
                } else {
                    self.arm_initial_probe_cooldown();
                }
            }
        } else {
            next = self.best_measured_budget(regime, old);
            if next > old && self.acceptance_ewma.unwrap_or(acceptance) < Self::HIGH_ACCEPTANCE {
                next = old;
            }
            if next == old {
                if self.cooldown_windows > 0 {
                    self.cooldown_windows -= 1;
                } else if self
                    .cost_estimate(regime, old)
                    .is_some_and(|estimate| estimate.samples >= Self::MIN_COST_SAMPLES)
                {
                    if let Some(probe) = self.next_probe_budget(regime, old) {
                        if probe < old
                            || self.acceptance_ewma.unwrap_or(acceptance) >= Self::HIGH_ACCEPTANCE
                        {
                            next = probe;
                            self.probe_budget = Some(probe);
                            self.probe_origin_budget = Some(old);
                            self.probe_windows_remaining = Self::PROBE_WINDOWS;
                        }
                    }
                }
            }
        }

        let smoothed_acceptance = self.acceptance_ewma.unwrap_or(acceptance);
        if smoothed_acceptance < Self::LOW_ACCEPTANCE && old > 1 {
            next = next.min(old - 1);
        }
        if acceptance == 0.0 {
            next = next.min(1);
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            // A rejected wider probe should back off before it is retried.
            // At budget 1, however, repeatedly re-arming this cooldown makes
            // an unprofitable MTP path impossible to compare with ordinary
            // decode: every zero-acceptance window resets the countdown. Keep
            // the existing countdown moving toward a fresh zero-draft probe.
            if old > 1 {
                self.arm_initial_probe_cooldown();
            }
        }

        self.current_budget = next.min(self.max_draft_tokens);
        budget_change(old, self.current_budget)
    }

    fn arm_initial_probe_cooldown(&mut self) {
        self.cooldown_windows = Self::INITIAL_PROBE_COOLDOWN_WINDOWS;
        self.next_probe_cooldown_windows =
            (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2).min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
    }

    fn back_off_next_probe(&mut self) {
        self.cooldown_windows = self.next_probe_cooldown_windows.clamp(
            Self::INITIAL_PROBE_COOLDOWN_WINDOWS,
            Self::MAX_PROBE_COOLDOWN_WINDOWS,
        );
        self.next_probe_cooldown_windows = self
            .cooldown_windows
            .saturating_mul(2)
            .min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
    }

    fn record_cost(
        &mut self,
        regime: MtpDraftPolicyRegime,
        draft_tokens: usize,
        cost: f64,
        acceptance: f64,
    ) {
        if let Some(estimate) = self
            .cost_estimates
            .iter_mut()
            .find(|estimate| estimate.regime == regime && estimate.draft_tokens == draft_tokens)
        {
            if estimate.samples == 0 {
                estimate.cost_ewma = cost;
                estimate.acceptance_ewma = acceptance;
            } else {
                estimate.cost_ewma = update_ewma(Some(estimate.cost_ewma), cost, Self::EWMA_ALPHA);
                estimate.acceptance_ewma =
                    update_ewma(Some(estimate.acceptance_ewma), acceptance, Self::EWMA_ALPHA);
            }
            estimate.samples = estimate.samples.saturating_add(1);
        } else {
            // Draft=0 activates the ordinary Q=1 target shape for the first
            // time in this response. That first control window can include a
            // one-off Metal graph compilation which has already been paid by
            // the time the policy makes its decision. Keep the marker but do
            // not let cold compilation bias the steady-state comparison.
            let samples = usize::from(draft_tokens != 0);
            self.cost_estimates.push(MtpDraftCostEstimate {
                regime,
                draft_tokens,
                cost_ewma: cost,
                acceptance_ewma: acceptance,
                samples,
            });
        }
    }

    fn cost_estimate(
        &self,
        regime: MtpDraftPolicyRegime,
        draft_tokens: usize,
    ) -> Option<&MtpDraftCostEstimate> {
        self.cost_estimates
            .iter()
            .find(|estimate| estimate.regime == regime && estimate.draft_tokens == draft_tokens)
    }

    fn best_measured_budget(&self, regime: MtpDraftPolicyRegime, current: usize) -> usize {
        let Some(current_estimate) = self.cost_estimate(regime, current) else {
            return current;
        };
        if current_estimate.samples < Self::MIN_COST_SAMPLES {
            return current;
        }
        self.cost_estimates
            .iter()
            .filter(|estimate| {
                estimate.regime == regime
                    && estimate.draft_tokens > 0
                    && estimate.samples >= Self::MIN_COST_SAMPLES
                    && (estimate.draft_tokens <= current
                        || estimate.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
                    && estimate.cost_ewma
                        < current_estimate.cost_ewma * Self::COST_IMPROVEMENT_RATIO
            })
            .min_by(|left, right| left.cost_ewma.total_cmp(&right.cost_ewma))
            .map_or(current, |estimate| estimate.draft_tokens)
    }

    fn next_probe_budget(&self, regime: MtpDraftPolicyRegime, current: usize) -> Option<usize> {
        let lower = current.checked_sub(1);
        let upper = current
            .checked_add(1)
            .filter(|&budget| budget <= self.max_draft_tokens);
        [lower, upper]
            .into_iter()
            .flatten()
            .filter(|&budget| {
                if budget == 0 && current > 0 {
                    // Wait for a controlled zero-draft probe. Once sampled,
                    // refreshes are scheduled explicitly after cooldown rather
                    // than by the generic adjacent-budget probe path.
                    if self.cost_estimate(regime, 0).is_some()
                        || self.cost_estimate(regime, current).is_none_or(|estimate| {
                            estimate.samples < Self::ZERO_DRAFT_MIN_COST_SAMPLES
                        })
                    {
                        return false;
                    }
                }
                budget <= current
                    || current == 0
                    || self
                        .cost_estimate(regime, budget)
                        .is_none_or(|estimate| estimate.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
            })
            .min_by_key(|&budget| {
                self.cost_estimate(regime, budget)
                    .map_or(0, |estimate| estimate.samples)
            })
    }

    fn preferred_probe_budget(
        &self,
        regime: MtpDraftPolicyRegime,
        origin: usize,
        probe: usize,
    ) -> usize {
        let Some(origin_cost) = self.cost_estimate(regime, origin) else {
            return probe;
        };
        let Some(probe_cost) = self.cost_estimate(regime, probe) else {
            return origin;
        };
        let improvement_ratio = if probe == 0 {
            Self::ZERO_DRAFT_COST_IMPROVEMENT_RATIO
        } else {
            Self::COST_IMPROVEMENT_RATIO
        };
        if probe_cost.samples >= Self::MIN_COST_SAMPLES
            && (probe <= origin
                || origin == 0
                || probe_cost.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
            && probe_cost.cost_ewma < origin_cost.cost_ewma * improvement_ratio
        {
            probe
        } else {
            origin
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct QwenMtpDraftPolicyState {
    max_draft_tokens: usize,
    current_budget: usize,
    acceptance_ewma: Option<f64>,
    active_regime: Option<MtpDraftPolicyRegime>,
    cost_estimates: Vec<MtpDraftCostEstimate>,
    probe_budget: Option<usize>,
    probe_origin_budget: Option<usize>,
    probe_windows_remaining: usize,
    cooldown_windows: usize,
    next_probe_cooldown_windows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QwenMtpDraftPolicySnapshot {
    max_draft_tokens: usize,
    current_budget: usize,
    acceptance_ewma_bits: Option<u64>,
    active_regime: Option<MtpDraftPolicyRegime>,
    cost_estimates: Vec<MtpDraftCostEstimateSnapshot>,
    probe_budget: Option<usize>,
    probe_origin_budget: Option<usize>,
    probe_windows_remaining: usize,
    cooldown_windows: usize,
    next_probe_cooldown_windows: usize,
}

impl QwenMtpDraftPolicyState {
    const EWMA_ALPHA: f64 = 0.35;
    const LOW_ACCEPTANCE: f64 = 0.50;
    const HIGH_ACCEPTANCE: f64 = 0.85;
    const MIN_COST_SAMPLES: usize = 2;
    const PROBE_WINDOWS: usize = 2;
    const ZERO_DRAFT_MIN_COST_SAMPLES: usize = 8;
    // A 32K prompt enters the UpTo128k regime as soon as the prefill token is
    // committed. Waiting for eight d=1 windows can spend most of a short
    // response on an unprofitable exact-verify path before ordinary decode is
    // measured. Keep the pre-Gemma policy at shorter contexts, but sample the
    // ordinary control after one complete long-context window.
    const LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES: usize = 1;
    const ZERO_DRAFT_PROBE_WINDOWS: usize = 4;
    const INITIAL_PROBE_COOLDOWN_WINDOWS: usize = 8;
    const MAX_PROBE_COOLDOWN_WINDOWS: usize = 64;
    const COST_IMPROVEMENT_RATIO: f64 = 0.95;
    // Ordinary decode is the safe control path. Require only a small measured
    // margin before bypassing MTP so a 5-10% speculative regression is not
    // hidden by overly conservative hysteresis. The four-window probe and
    // EWMA still absorb single-window timing noise.
    const ZERO_DRAFT_COST_IMPROVEMENT_RATIO: f64 = 0.98;

    pub(crate) fn new(max_draft_tokens: usize) -> Self {
        let max_draft_tokens = max_draft_tokens.max(1);
        Self {
            max_draft_tokens,
            current_budget: max_draft_tokens,
            acceptance_ewma: None,
            active_regime: None,
            cost_estimates: Vec::new(),
            probe_budget: None,
            probe_origin_budget: None,
            probe_windows_remaining: 0,
            cooldown_windows: 0,
            next_probe_cooldown_windows: (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2)
                .min(Self::MAX_PROBE_COOLDOWN_WINDOWS),
        }
    }

    pub(crate) fn current_budget(&self) -> usize {
        self.current_budget.min(self.max_draft_tokens)
    }

    pub(crate) fn should_maintain_mtp_cache(&self) -> bool {
        self.current_budget() > 0 || self.probe_budget.is_some()
    }

    pub(crate) fn uses_ordinary_decode(&self) -> bool {
        self.current_budget() == 0 && self.probe_budget.is_none()
    }

    pub(crate) fn snapshot(&self) -> QwenMtpDraftPolicySnapshot {
        QwenMtpDraftPolicySnapshot {
            max_draft_tokens: self.max_draft_tokens,
            current_budget: self.current_budget,
            acceptance_ewma_bits: self.acceptance_ewma.map(f64::to_bits),
            active_regime: self.active_regime,
            cost_estimates: self
                .cost_estimates
                .iter()
                .map(|estimate| MtpDraftCostEstimateSnapshot {
                    regime: estimate.regime,
                    draft_tokens: estimate.draft_tokens,
                    cost_ewma_bits: estimate.cost_ewma.to_bits(),
                    acceptance_ewma_bits: estimate.acceptance_ewma.to_bits(),
                    samples: estimate.samples,
                })
                .collect(),
            probe_budget: self.probe_budget,
            probe_origin_budget: self.probe_origin_budget,
            probe_windows_remaining: self.probe_windows_remaining,
            cooldown_windows: self.cooldown_windows,
            next_probe_cooldown_windows: self.next_probe_cooldown_windows,
        }
    }

    pub(crate) fn restore_snapshot(&mut self, snapshot: QwenMtpDraftPolicySnapshot) -> Result<()> {
        anyhow::ensure!(
            snapshot.max_draft_tokens == self.max_draft_tokens,
            "MTP draft policy snapshot max {} != destination max {}",
            snapshot.max_draft_tokens,
            self.max_draft_tokens
        );
        anyhow::ensure!(
            snapshot.current_budget <= snapshot.max_draft_tokens,
            "MTP draft policy snapshot budget {} is outside [0, {}]",
            snapshot.current_budget,
            snapshot.max_draft_tokens
        );
        self.current_budget = snapshot.current_budget;
        self.acceptance_ewma = snapshot.acceptance_ewma_bits.map(f64::from_bits);
        self.active_regime = snapshot.active_regime;
        self.cost_estimates = snapshot
            .cost_estimates
            .into_iter()
            .map(|estimate| MtpDraftCostEstimate {
                regime: estimate.regime,
                draft_tokens: estimate.draft_tokens,
                cost_ewma: f64::from_bits(estimate.cost_ewma_bits),
                acceptance_ewma: f64::from_bits(estimate.acceptance_ewma_bits),
                samples: estimate.samples,
            })
            .collect();
        self.probe_budget = snapshot.probe_budget;
        self.probe_origin_budget = snapshot.probe_origin_budget;
        self.probe_windows_remaining = snapshot.probe_windows_remaining;
        self.cooldown_windows = snapshot.cooldown_windows;
        self.next_probe_cooldown_windows = snapshot.next_probe_cooldown_windows;
        Ok(())
    }

    pub(crate) fn observe_window(&mut self, window: MtpDraftPolicyWindow) -> MtpDraftBudgetChange {
        if qwen_fixed_mtp_draft_depth_is_armed() {
            return MtpDraftBudgetChange::default();
        }
        let regime = window.regime();
        if self.active_regime != Some(regime) {
            self.active_regime = Some(regime);
            self.acceptance_ewma = None;
            self.cost_estimates.clear();
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            self.cooldown_windows = 0;
            self.next_probe_cooldown_windows =
                (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2).min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
        }
        let old = self.current_budget();

        let acceptance = window.acceptance_rate();
        self.acceptance_ewma = Some(update_ewma(
            self.acceptance_ewma,
            acceptance,
            Self::EWMA_ALPHA,
        ));
        self.record_cost(
            regime,
            window.attempted_draft_tokens,
            window.qwen_cost_per_committed_token_us(),
            acceptance,
        );
        if window.attempted_draft_tokens != old {
            return MtpDraftBudgetChange::default();
        }

        let long_context = matches!(
            regime.context_bucket,
            MtpDraftCapContextBucket::UpTo128k | MtpDraftCapContextBucket::Above128k
        );
        let zero_draft_probe_min_samples = if long_context {
            Self::LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES
        } else {
            Self::ZERO_DRAFT_MIN_COST_SAMPLES
        };
        if long_context
            && old > 0
            && self.probe_budget.is_none()
            && self.cost_estimate(regime, 0).is_none()
            && self
                .cost_estimate(regime, old)
                .is_some_and(|estimate| estimate.samples >= 1)
        {
            // At 32K+, first compare the configured production depth directly
            // with ordinary decode. Traversing every adjacent depth before the
            // control path makes the exploration cost dominate short replies.
            self.current_budget = 0;
            self.probe_budget = Some(0);
            self.probe_origin_budget = Some(old);
            self.probe_windows_remaining = Self::ZERO_DRAFT_PROBE_WINDOWS;
            return budget_change(old, self.current_budget);
        }
        if old == 1
            && self.probe_budget.is_none()
            && self.cost_estimate(regime, 0).is_none()
            && self
                .cost_estimate(regime, old)
                .is_some_and(|estimate| estimate.samples >= zero_draft_probe_min_samples)
        {
            self.current_budget = 0;
            self.probe_budget = Some(0);
            self.probe_origin_budget = Some(old);
            self.probe_windows_remaining = Self::ZERO_DRAFT_PROBE_WINDOWS;
            return budget_change(old, self.current_budget);
        }
        if old == 0 && self.probe_budget.is_none() {
            return MtpDraftBudgetChange::default();
        }
        if old == 1
            && self.probe_budget.is_none()
            && self.cost_estimate(regime, 0).is_some()
            && self.cooldown_windows == 0
            && self
                .cost_estimate(regime, 1)
                .is_some_and(|estimate| estimate.samples >= Self::ZERO_DRAFT_MIN_COST_SAMPLES)
        {
            // A previous zero-draft sample is stale once the response enters
            // a different acceptance/cost phase. Re-test ordinary decode only
            // while the MTP cache is still synchronized, and discard the old
            // control estimate so it cannot drive an uncontrolled switch.
            self.cost_estimates
                .retain(|estimate| estimate.regime != regime || estimate.draft_tokens != 0);
            self.current_budget = 0;
            self.probe_budget = Some(0);
            self.probe_origin_budget = Some(old);
            self.probe_windows_remaining = Self::ZERO_DRAFT_PROBE_WINDOWS;
            return budget_change(old, self.current_budget);
        }

        let mut next = old;
        let full_accept = window.accepted_draft_tokens == window.attempted_draft_tokens;
        if !full_accept {
            let rejected_probe = self.probe_budget == Some(old);
            next = if long_context && old > 1 {
                // At long context, every extra verifier position carries the
                // full attention/cache-read cost. If depth d did not fully
                // accept, probe the actually useful depth next instead of
                // spending another window at d.
                window.accepted_draft_tokens.max(1).min(old)
            } else {
                window.accepted_draft_tokens.saturating_add(1).min(old)
            };
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            if rejected_probe {
                self.back_off_next_probe();
            } else if old == 1 {
                self.cooldown_windows = self.cooldown_windows.saturating_sub(1);
            } else {
                self.arm_initial_probe_cooldown();
            }
        } else if self.probe_budget == Some(old) {
            self.probe_windows_remaining = self.probe_windows_remaining.saturating_sub(1);
            if self.probe_windows_remaining == 0 {
                let origin = self.probe_origin_budget.take();
                self.probe_budget = None;
                next = origin.map_or(old, |origin| {
                    self.preferred_probe_budget(regime, origin, old)
                });
                if origin.is_some_and(|origin| next == origin) && origin != Some(0) {
                    self.back_off_next_probe();
                } else {
                    self.arm_initial_probe_cooldown();
                }
            }
        } else {
            next = self.best_measured_budget(regime, old);
            if next > old && self.acceptance_ewma.unwrap_or(acceptance) < Self::HIGH_ACCEPTANCE {
                next = old;
            }
            let adjacent_probe_min_samples = if long_context {
                1
            } else {
                Self::MIN_COST_SAMPLES
            };
            if next == old {
                if self.cooldown_windows > 0 {
                    self.cooldown_windows -= 1;
                } else if self
                    .cost_estimate(regime, old)
                    .is_some_and(|estimate| estimate.samples >= adjacent_probe_min_samples)
                {
                    if let Some(probe) = self.next_probe_budget(regime, old) {
                        if probe < old
                            || self.acceptance_ewma.unwrap_or(acceptance) >= Self::HIGH_ACCEPTANCE
                        {
                            next = probe;
                            if long_context && probe < old {
                                // Adopt the cheaper-depth candidate directly
                                // for one window. The next d=1 observation
                                // immediately compares against ordinary decode,
                                // avoiding a multi-window nested probe at 32K+.
                                self.probe_budget = None;
                                self.probe_origin_budget = None;
                                self.probe_windows_remaining = 0;
                            } else {
                                self.probe_budget = Some(probe);
                                self.probe_origin_budget = Some(old);
                                self.probe_windows_remaining = Self::PROBE_WINDOWS;
                            }
                        }
                    }
                }
            }
        }

        let smoothed_acceptance = self.acceptance_ewma.unwrap_or(acceptance);
        if smoothed_acceptance < Self::LOW_ACCEPTANCE && old > 1 {
            next = next.min(old - 1);
        }
        if acceptance == 0.0 {
            next = next.min(1);
            self.probe_budget = None;
            self.probe_origin_budget = None;
            self.probe_windows_remaining = 0;
            // A rejected wider probe should back off before it is retried.
            // At budget 1, however, repeatedly re-arming this cooldown makes
            // an unprofitable MTP path impossible to compare with ordinary
            // decode: every zero-acceptance window resets the countdown. Keep
            // the existing countdown moving toward a fresh zero-draft probe.
            if old > 1 {
                self.arm_initial_probe_cooldown();
            }
        }

        self.current_budget = next.min(self.max_draft_tokens);
        budget_change(old, self.current_budget)
    }

    fn arm_initial_probe_cooldown(&mut self) {
        self.cooldown_windows = Self::INITIAL_PROBE_COOLDOWN_WINDOWS;
        self.next_probe_cooldown_windows =
            (Self::INITIAL_PROBE_COOLDOWN_WINDOWS * 2).min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
    }

    fn back_off_next_probe(&mut self) {
        self.cooldown_windows = self.next_probe_cooldown_windows.clamp(
            Self::INITIAL_PROBE_COOLDOWN_WINDOWS,
            Self::MAX_PROBE_COOLDOWN_WINDOWS,
        );
        self.next_probe_cooldown_windows = self
            .cooldown_windows
            .saturating_mul(2)
            .min(Self::MAX_PROBE_COOLDOWN_WINDOWS);
    }

    fn record_cost(
        &mut self,
        regime: MtpDraftPolicyRegime,
        draft_tokens: usize,
        cost: f64,
        acceptance: f64,
    ) {
        if let Some(estimate) = self
            .cost_estimates
            .iter_mut()
            .find(|estimate| estimate.regime == regime && estimate.draft_tokens == draft_tokens)
        {
            estimate.cost_ewma = update_ewma(Some(estimate.cost_ewma), cost, Self::EWMA_ALPHA);
            estimate.acceptance_ewma =
                update_ewma(Some(estimate.acceptance_ewma), acceptance, Self::EWMA_ALPHA);
            estimate.samples = estimate.samples.saturating_add(1);
        } else {
            self.cost_estimates.push(MtpDraftCostEstimate {
                regime,
                draft_tokens,
                cost_ewma: cost,
                acceptance_ewma: acceptance,
                samples: 1,
            });
        }
    }

    fn cost_estimate(
        &self,
        regime: MtpDraftPolicyRegime,
        draft_tokens: usize,
    ) -> Option<&MtpDraftCostEstimate> {
        self.cost_estimates
            .iter()
            .find(|estimate| estimate.regime == regime && estimate.draft_tokens == draft_tokens)
    }

    fn best_measured_budget(&self, regime: MtpDraftPolicyRegime, current: usize) -> usize {
        let Some(current_estimate) = self.cost_estimate(regime, current) else {
            return current;
        };
        if current_estimate.samples < Self::MIN_COST_SAMPLES {
            return current;
        }
        self.cost_estimates
            .iter()
            .filter(|estimate| {
                estimate.regime == regime
                    && estimate.draft_tokens > 0
                    && estimate.samples >= Self::MIN_COST_SAMPLES
                    && (estimate.draft_tokens <= current
                        || estimate.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
                    && estimate.cost_ewma
                        < current_estimate.cost_ewma * Self::COST_IMPROVEMENT_RATIO
            })
            .min_by(|left, right| left.cost_ewma.total_cmp(&right.cost_ewma))
            .map_or(current, |estimate| estimate.draft_tokens)
    }

    fn next_probe_budget(&self, regime: MtpDraftPolicyRegime, current: usize) -> Option<usize> {
        let lower = current.checked_sub(1);
        let upper = current
            .checked_add(1)
            .filter(|&budget| budget <= self.max_draft_tokens);
        [lower, upper]
            .into_iter()
            .flatten()
            .filter(|&budget| {
                if budget == 0 && current > 0 {
                    // Wait for a controlled zero-draft probe. Once sampled,
                    // refreshes are scheduled explicitly after cooldown rather
                    // than by the generic adjacent-budget probe path.
                    if self.cost_estimate(regime, 0).is_some()
                        || self.cost_estimate(regime, current).is_none_or(|estimate| {
                            estimate.samples < Self::ZERO_DRAFT_MIN_COST_SAMPLES
                        })
                    {
                        return false;
                    }
                }
                budget <= current
                    || current == 0
                    || self
                        .cost_estimate(regime, budget)
                        .is_none_or(|estimate| estimate.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
            })
            .min_by_key(|&budget| {
                self.cost_estimate(regime, budget)
                    .map_or(0, |estimate| estimate.samples)
            })
    }

    fn preferred_probe_budget(
        &self,
        regime: MtpDraftPolicyRegime,
        origin: usize,
        probe: usize,
    ) -> usize {
        let Some(origin_cost) = self.cost_estimate(regime, origin) else {
            return probe;
        };
        let Some(probe_cost) = self.cost_estimate(regime, probe) else {
            return origin;
        };
        let improvement_ratio = if probe == 0 {
            Self::ZERO_DRAFT_COST_IMPROVEMENT_RATIO
        } else {
            Self::COST_IMPROVEMENT_RATIO
        };
        if probe_cost.samples >= Self::MIN_COST_SAMPLES
            && (probe <= origin
                || origin == 0
                || probe_cost.acceptance_ewma >= Self::HIGH_ACCEPTANCE)
            && probe_cost.cost_ewma < origin_cost.cost_ewma * improvement_ratio
        {
            probe
        } else {
            origin
        }
    }
}

fn update_ewma(current: Option<f64>, sample: f64, alpha: f64) -> f64 {
    current.map_or(sample, |value| value.mul_add(1.0 - alpha, sample * alpha))
}

fn budget_change(old: usize, new: usize) -> MtpDraftBudgetChange {
    MtpDraftBudgetChange {
        reduced: new < old,
        increased: new > old,
    }
}

pub(crate) fn zero_hidden_like_position(hidden: &Array) -> Result<Array> {
    let shape = hidden.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(anyhow!(
            "zero_hidden_like_position: expected hidden shape [1, S, H], got {:?}",
            dims
        ));
    }
    Array::zeros((1_i32, 1_i32, dims[2]), hidden.dtype()).map_err(anyhow::Error::from)
}

pub(crate) fn shift_hidden_for_mtp(
    prev_hidden: &Array,
    hidden: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<Array> {
    let target = target.into();
    let prev_shape = prev_hidden.shape();
    let prev_dims = prev_shape.as_slice();
    let hidden_shape = hidden.shape();
    let hidden_dims = hidden_shape.as_slice();
    if prev_dims.len() != 3 || prev_dims[0] != 1 || prev_dims[1] != 1 {
        return Err(anyhow!(
            "shift_hidden_for_mtp: expected prev_hidden shape [1, 1, H], got {:?}",
            prev_dims
        ));
    }
    if hidden_dims.len() != 3 || hidden_dims[0] != 1 {
        return Err(anyhow!(
            "shift_hidden_for_mtp: expected hidden shape [1, S, H], got {:?}",
            hidden_dims
        ));
    }
    let seq = hidden_dims[1];
    let hidden_size = hidden_dims[2];
    if prev_dims[2] != hidden_size {
        return Err(anyhow!(
            "shift_hidden_for_mtp: prev hidden size {} != hidden size {}",
            prev_dims[2],
            hidden_size
        ));
    }
    if seq == 1 {
        return Ok(prev_hidden.clone());
    }
    let prefix = mlx::ops::indexing::slice_strided_on(
        hidden,
        &[0_i32, 0_i32, 0_i32][..],
        &[1_i32, seq - 1, hidden_size][..],
        &[1_i32, 1_i32, 1_i32][..],
        target,
    )?;
    mlx::ops::shape::concatenate_on(&[prev_hidden, &prefix], 1, target).map_err(anyhow::Error::from)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_mtp_cache_hidden_prefix<M>(
    model: &M,
    mtp: &M::MtpHead,
    mtp_cache: &mut MtpCache,
    prev_hidden: &Array,
    input_tokens: &[u32],
    input_hidden: &Array,
    position_ids: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<()>
where
    M: MtpSpeculativeModel,
{
    if input_tokens.is_empty() {
        return Ok(());
    }
    let target = target.into();
    let hidden_shape = input_hidden.shape();
    let hidden_dims = hidden_shape.as_slice();
    if hidden_dims.len() != 3 || hidden_dims[0] != 1 || hidden_dims[1] != input_tokens.len() as i32
    {
        return Err(anyhow!(
            "commit_mtp_cache_hidden_prefix: hidden shape {:?} does not match {} input tokens",
            hidden_dims,
            input_tokens.len()
        ));
    }
    let shifted_hidden = shift_hidden_for_mtp(prev_hidden, input_hidden, target)?;
    let token_arr: Array = (input_tokens, &[1_i32, input_tokens.len() as i32][..]).try_into()?;
    let mtp_hidden = model.mtp_forward_hidden_on(
        mtp,
        &shifted_hidden,
        &token_arr,
        position_ids,
        None,
        Some(mtp_cache),
        target,
    )?;
    mlx::transforms::eval(&[&mtp_hidden])?;
    Ok(())
}

fn slice_position_ids_position(position_ids: &Array, pos: i32) -> Result<Array> {
    let shape = position_ids.shape();
    let dims = shape.as_slice();
    match dims {
        [1, seq] => {
            if *seq == 1 {
                return Ok(position_ids.clone());
            }
            if pos < 0 || pos >= *seq {
                return Err(anyhow!(
                    "slice_position_ids_position: pos {pos} out of [0, {seq})"
                ));
            }
            mlx::ops::indexing::slice_strided(
                position_ids,
                &[0_i32, pos][..],
                &[1_i32, pos + 1][..],
                &[1_i32, 1_i32][..],
            )
            .map_err(anyhow::Error::from)
        }
        [planes, 1, seq] => {
            if *seq == 1 {
                return Ok(position_ids.clone());
            }
            if pos < 0 || pos >= *seq {
                return Err(anyhow!(
                    "slice_position_ids_position: pos {pos} out of [0, {seq})"
                ));
            }
            mlx::ops::indexing::slice_strided(
                position_ids,
                &[0_i32, 0_i32, pos][..],
                &[*planes, 1_i32, pos + 1][..],
                &[1_i32, 1_i32, 1_i32][..],
            )
            .map_err(anyhow::Error::from)
        }
        _ => Err(anyhow!(
            "slice_position_ids_position: expected position_ids shape [1, S] or [P, 1, S], got {:?}",
            dims
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_mtp_cache_hidden_tail<M>(
    model: &M,
    mtp: &M::MtpHead,
    mtp_cache: &mut MtpCache,
    prev_hidden: &Array,
    input_tokens: &[u32],
    input_hidden: &Array,
    position_ids: &Array,
    target: impl Into<StreamOrDevice>,
) -> Result<()>
where
    M: MtpSpeculativeModel,
{
    if input_tokens.is_empty() {
        return Ok(());
    }
    let tail_idx = input_tokens.len() - 1;
    let tail_prev_hidden = if tail_idx == 0 {
        prev_hidden.clone()
    } else {
        slice_hidden_position(input_hidden, tail_idx as i32 - 1)?
    };
    let tail_hidden = slice_hidden_position(input_hidden, tail_idx as i32)?;
    let tail_position_ids = slice_position_ids_position(position_ids, tail_idx as i32)?;
    commit_mtp_cache_hidden_prefix(
        model,
        mtp,
        mtp_cache,
        &tail_prev_hidden,
        &input_tokens[tail_idx..],
        &tail_hidden,
        &tail_position_ids,
        target,
    )
}

/// Text-only single-request stream for Qwen MTP speculative decoding.
pub struct MtpTextGenerationStream<'m, M>
where
    M: MtpSpeculativeModel,
{
    model: &'m M,
    mtp: &'m M::MtpHead,
    cache: Vec<LayerCache>,
    mtp_cache: MtpCache,
    history: Vec<u32>,
    request: GenerateRequest,
    cfg: MtpSpeculativeConfig,
    pending_tokens: VecDeque<u32>,
    detok: DecodeStream<'m>,
    /// Hidden state for the token immediately before the current pending token.
    last_hidden: Array,
    emitted_new_tokens: usize,
    finished: bool,
    dummy_position_ids: Option<Array>,
    prng_state: Array,
    adaptive_draft_tokens: usize,
    draft_policy: QwenMtpDraftPolicyState,
    stats: MtpSpeculativeStats,
    constraint: Option<ConstraintSession>,
}

impl<'m, M> MtpTextGenerationStream<'m, M>
where
    M: MtpSpeculativeModel,
{
    /// Construct a text-only MTP speculative stream.
    pub fn new_text_only(
        model: &'m M,
        mtp: &'m M::MtpHead,
        tokenizer: &'m Tokenizer,
        request: GenerateRequest,
        cfg: MtpSpeculativeConfig,
    ) -> Result<Self> {
        if request.pixel_values.is_some() {
            return Err(anyhow!(
                "MtpTextGenerationStream::new_text_only called with pixel_values; MTP speculative decoding is text-only"
            ));
        }
        if request.prompt_ids.is_empty() {
            return Err(anyhow!(
                "MtpTextGenerationStream::new_text_only: prompt_ids cannot be empty"
            ));
        }
        if cfg.max_draft_tokens == 0 {
            return Err(anyhow!(
                "MtpTextGenerationStream::new_text_only: max_draft_tokens must be > 0"
            ));
        }
        let prompt_len = request.prompt_ids.len();
        let cap = ((prompt_len + request.max_new_tokens) as i32)
            .max(crate::models::qwen3_5::MIN_KV_CACHE_CAP_FOR_GPU_PERF);
        let dtype = model.cache_dtype();
        let mut cache = model.make_cache(1, cap, dtype)?;
        if let Some(bits) = request.kv_cache_turboquant_bits {
            enable_turboquant_kv_caches(&mut cache, bits)?;
        }
        let mut mtp_cache = model.make_mtp_cache(mtp, 1, cap, dtype)?;
        let dummy_position_ids = if model.requires_position_ids() {
            None
        } else {
            Some(build_position_ids(0, 1)?)
        };

        let chunk_size = request.prefill_chunk_size;
        let prompt_len_i32 = prompt_len as i32;
        let mut pos = 0_i32;
        let mut stats = MtpSpeculativeStats::default();
        let mut last_prompt_hidden = None;
        let mut mtp_prev_hidden: Option<Array> = None;
        while pos < prompt_len_i32 {
            let remaining = prompt_len_i32 - pos;
            let n = if chunk_size == 0 {
                remaining
            } else {
                remaining.min(chunk_size as i32)
            };
            let chunk_ids = &request.prompt_ids[pos as usize..(pos as usize + n as usize)];
            let chunk_arr: Array = (chunk_ids, &[1_i32, n][..]).try_into()?;
            let chunk_pos_ids = match dummy_position_ids.as_ref() {
                Some(dummy) => dummy.clone(),
                None => build_position_ids(pos, n)?,
            };
            let forward_start = Instant::now();
            let hidden = model.forward_text_hidden(
                &chunk_arr,
                &chunk_pos_ids,
                None,
                None,
                Some(&mut cache),
                ().into(),
            )?;
            add_elapsed_us(&mut stats.verify_forward_us, forward_start);
            let prev_hidden = match mtp_prev_hidden.as_ref() {
                Some(hidden) => hidden.clone(),
                None => zero_hidden_like_position(&hidden)?,
            };
            let commit_start = Instant::now();
            commit_mtp_cache_hidden_prefix(
                model,
                mtp,
                &mut mtp_cache,
                &prev_hidden,
                chunk_ids,
                &hidden,
                &chunk_pos_ids,
                (),
            )?;
            add_mtp_prefill_cache_commit_us(&mut stats, commit_start);
            let chunk_last_hidden = slice_hidden_position(&hidden, n - 1)?;
            mtp_prev_hidden = Some(chunk_last_hidden.clone());
            if pos + n == prompt_len_i32 {
                last_prompt_hidden = Some(chunk_last_hidden);
            }
            pos += n;
        }
        let last_prompt_hidden =
            last_prompt_hidden.ok_or_else(|| anyhow!("MTP prefill produced no prompt hidden"))?;

        let projection_start = Instant::now();
        let first_logits =
            model.project_hidden_on(&last_prompt_hidden, StreamOrDevice::default())?;
        add_elapsed_us(&mut stats.projection_us, projection_start);
        let mut constraint = request
            .constraint
            .as_ref()
            .map(|plan| plan.start_session())
            .transpose()?;
        let first_logits = constrain_speculative_logits(&mut constraint, &first_logits, &[])?;
        let mut prng_state = mlx::random::key(request.sampler.seed)?;
        let sampling_start = Instant::now();
        let first_tokens = sample_logits_positions(
            &first_logits,
            request.sampler,
            &request.prompt_ids,
            &mut prng_state,
        )?;
        add_elapsed_us(&mut stats.sampling_us, sampling_start);
        let first_token = *first_tokens
            .first()
            .ok_or_else(|| anyhow!("MTP prefill produced no first token"))?;
        commit_constraint_token(&mut constraint, first_token)?;

        let mut history = request.prompt_ids.clone();
        history.push(first_token);
        let mut pending_tokens = VecDeque::new();
        pending_tokens.push_back(first_token);

        Ok(Self {
            model,
            mtp,
            cache,
            mtp_cache,
            history,
            request,
            cfg,
            pending_tokens,
            detok: tokenizer.decode_stream(true),
            last_hidden: last_prompt_hidden,
            emitted_new_tokens: 0,
            finished: false,
            dummy_position_ids,
            prng_state,
            adaptive_draft_tokens: cfg.max_draft_tokens,
            draft_policy: QwenMtpDraftPolicyState::new(cfg.max_draft_tokens),
            stats,
            constraint,
        })
    }

    /// Return cumulative speculative-window counters for this stream.
    pub fn stats(&self) -> MtpSpeculativeStats {
        self.stats.clone()
    }

    /// Pull the next generated token event.
    pub fn next_token(&mut self) -> Result<Option<GenerateEvent>> {
        if self.finished {
            return Ok(None);
        }

        let token = self
            .pending_tokens
            .pop_front()
            .ok_or_else(|| anyhow!("MTP stream invariant: pending token queue is empty"))?;
        self.emitted_new_tokens += 1;
        let text = self.detok.step(token)?.unwrap_or_default();
        let finish_reason = if self.request.stop_token_ids.contains(&token) {
            Some("stop")
        } else if self.emitted_new_tokens >= self.request.max_new_tokens {
            Some("length")
        } else {
            None
        };

        if finish_reason == Some("length") {
            if let Some(constraint) = self.constraint.as_mut() {
                if constraint.requires_accepting_state_at_length() && !constraint.is_accepting()? {
                    self.finished = true;
                    return Err(anyhow!(
                        "max_new_tokens reached before constrained output became complete"
                    ));
                }
            }
        }

        if finish_reason.is_some() {
            self.finished = true;
            return Ok(Some(GenerateEvent {
                token,
                text,
                finish_reason,
            }));
        }

        if self.pending_tokens.is_empty() {
            self.fill_window(token)?;
        }

        Ok(Some(GenerateEvent {
            token,
            text,
            finish_reason: None,
        }))
    }

    fn fill_window(&mut self, current_token: u32) -> Result<()> {
        let remaining = self
            .request
            .max_new_tokens
            .saturating_sub(self.emitted_new_tokens);
        if remaining == 0 {
            return Ok(());
        }

        let window_started = Instant::now();
        let stats_before_window = self.stats.clone();
        let timing_before = self.stats.draft_cap_timing();
        let context_tokens = self.history.len();
        let draft_budget = self
            .adaptive_draft_tokens
            .min(self.cfg.max_draft_tokens)
            .min(remaining);
        let maintain_mtp_cache = self.draft_policy.should_maintain_mtp_cache();
        let mut draft_constraint = self.constraint.as_ref().map(ConstraintSession::fork);
        let draft_result = self.draft_tokens(current_token, draft_budget, &mut draft_constraint)?;
        let draft_tokens = draft_result.tokens;
        let _draft_distributions = draft_result.distributions;
        let verify_input = verify_input(current_token, &draft_tokens);
        let verify_start_pos = (self.history.len() - 1) as i32;
        let verify_pos_ids = self.position_ids(verify_start_pos, verify_input.len() as i32)?;
        let verify_arr: Array =
            (&verify_input[..], &[1_i32, verify_input.len() as i32][..]).try_into()?;
        let pre_window_hidden = self.last_hidden.clone();

        let base_snapshot = (draft_budget > 0).then(|| {
            self.cache
                .iter()
                .map(LayerCache::snapshot)
                .collect::<Vec<_>>()
        });
        let verify_forward_start = Instant::now();
        let verified_hidden = {
            // A zero-draft window is the ordinary single-token target path.
            // Exact speculative QMM is required only when target positions
            // must be compared against drafted tokens.
            let _verify_qmm = (draft_budget > 0).then(crate::nn::verify_qmm::armed_scope);
            self.model.forward_text_hidden(
                &verify_arr,
                &verify_pos_ids,
                None,
                None,
                Some(&mut self.cache),
                ().into(),
            )?
        };
        add_elapsed_us(&mut self.stats.verify_forward_us, verify_forward_start);
        let resolution = if self.request.sampler.is_pipelinable() && self.constraint.is_none() {
            resolve_greedy_verified_hidden_until_mismatch(
                self.model,
                &verified_hidden,
                &draft_tokens,
                &mut self.stats,
                (),
            )?
        } else {
            let projection_start = Instant::now();
            let verified_logits = self
                .model
                .project_mtp_verify_hidden_on(&verified_hidden, ())?;
            add_elapsed_us(&mut self.stats.projection_us, projection_start);
            let verified_logits = constrain_speculative_logits(
                &mut self.constraint,
                &verified_logits,
                &draft_tokens,
            )?;
            let sampling_start = Instant::now();
            let resolution = if self.request.sampler.is_pipelinable() {
                let verified_ids = mlx::ops::reduction::argmax(&verified_logits, -1, false)?;
                let verified_tokens: Vec<u32> = verified_ids.to_vec()?;
                resolve_speculative_tokens(&draft_tokens, &verified_tokens)?
            } else {
                resolve_exact_deterministic_target_logits(
                    &draft_tokens,
                    &verified_logits,
                    self.request.sampler,
                    &self.history,
                    &mut self.prng_state,
                )?
            };
            add_elapsed_us(&mut self.stats.sampling_us, sampling_start);
            resolution
        };
        self.stats.windows += 1;
        self.stats.drafted_tokens += draft_tokens.len();
        self.stats.accepted_draft_tokens += resolution.accepted_draft_len;
        self.stats.record_exact_sampling(resolution.exact_sampling);
        self.stats
            .record_window_acceptance(draft_tokens.len(), resolution.accepted_draft_len);
        if resolution.needs_rollback {
            self.stats.rollback_count += 1;
        }
        let accepted_len = resolution.accepted_verify_input_len;
        let (accepted_hidden, accepted_position_ids, accepted_last_hidden) = if resolution
            .needs_rollback
        {
            let accepted_position_ids = slice_position_ids_prefix(&verify_pos_ids, accepted_len)?;
            let rollback_start = Instant::now();
            let accepted_hidden = rollback_main_cache_to_accepted_prefix(
                self.model,
                &mut self.cache,
                base_snapshot
                    .as_deref()
                    .ok_or_else(|| anyhow!("MTP rollback snapshot absent"))?,
                MainCacheRollbackInput {
                    accepted_by_row: &[(0, accepted_len)],
                    verify_input: &verify_input,
                    accepted_position_ids: &accepted_position_ids,
                    verified_hidden: &verified_hidden,
                },
                (),
            )?;
            add_elapsed_us(&mut self.stats.main_rollback_us, rollback_start);
            (
                accepted_hidden.clone(),
                accepted_position_ids,
                slice_hidden_position(&accepted_hidden, accepted_len as i32 - 1)?,
            )
        } else {
            (
                verified_hidden.clone(),
                verify_pos_ids.clone(),
                slice_hidden_position(&verified_hidden, accepted_len as i32 - 1)?,
            )
        };
        let accepted_input = verify_input[..accepted_len].to_vec();

        if resolution.needs_rollback {
            let restore_start = Instant::now();
            self.mtp_cache.restore(&draft_result.cache_snapshot)?;
            add_elapsed_us(&mut self.stats.mtp_cache_restore_us, restore_start);
            let commit_start = Instant::now();
            commit_mtp_cache_hidden_prefix(
                self.model,
                self.mtp,
                &mut self.mtp_cache,
                &pre_window_hidden,
                &accepted_input,
                &accepted_hidden,
                &accepted_position_ids,
                (),
            )?;
            add_mtp_decode_cache_commit_us(&mut self.stats, commit_start);
        } else if maintain_mtp_cache {
            let commit_start = Instant::now();
            commit_mtp_cache_hidden_tail(
                self.model,
                self.mtp,
                &mut self.mtp_cache,
                &pre_window_hidden,
                &accepted_input,
                &accepted_hidden,
                &accepted_position_ids,
                (),
            )?;
            add_mtp_decode_cache_commit_us(&mut self.stats, commit_start);
            self.stats.mtp_cache_reuse_count = self.stats.mtp_cache_reuse_count.saturating_add(1);
            self.stats.mtp_cache_reused_tokens = self
                .stats
                .mtp_cache_reused_tokens
                .saturating_add(accepted_input.len().saturating_sub(1));
        }
        self.last_hidden = accepted_last_hidden;

        let mut tokens_to_append = resolution.tokens_to_append;
        if let Some(stop_idx) = tokens_to_append
            .iter()
            .position(|token| self.request.stop_token_ids.contains(token))
        {
            tokens_to_append.truncate(stop_idx + 1);
        }
        tokens_to_append.truncate(remaining);
        if let Some(constraint) = self.constraint.as_ref() {
            constraint.truncate_invalid_speculative_bonus(&mut tokens_to_append)?;
        }
        let committed_tokens = tokens_to_append.len();
        for token in tokens_to_append {
            commit_constraint_token(&mut self.constraint, token)?;
            self.history.push(token);
            self.pending_tokens.push_back(token);
        }

        let total_us = elapsed_us_since(window_started);
        let stats_delta = self.stats.saturating_delta_since(&stats_before_window);
        let change = self
            .draft_policy
            .observe_window(MtpDraftPolicyWindow::from_stats_delta(
                draft_tokens.len(),
                resolution.accepted_draft_len,
                committed_tokens,
                total_us,
                context_tokens,
                1,
                MtpDraftPolicyKvState::Contiguous,
                &stats_delta,
            ));
        if change.reduced {
            self.stats.draft_budget_reductions =
                self.stats.draft_budget_reductions.saturating_add(1);
        } else if change.increased {
            self.stats.draft_budget_increases = self.stats.draft_budget_increases.saturating_add(1);
        }
        self.adaptive_draft_tokens = self.draft_policy.current_budget();
        let timing_delta = self
            .stats
            .draft_cap_timing()
            .saturating_delta_since(timing_before);
        self.stats.record_draft_cap_observation(
            self.cfg.max_draft_tokens,
            &[draft_tokens.len()],
            &[context_tokens],
            resolution.accepted_draft_len,
            committed_tokens,
            usize::from(resolution.needs_rollback),
            total_us,
            timing_delta,
        );

        Ok(())
    }

    fn draft_tokens(
        &mut self,
        current_token: u32,
        draft_budget: usize,
        constraint: &mut Option<ConstraintSession>,
    ) -> Result<MtpDraftResult> {
        let mtp_snapshot = self.mtp_cache.snapshot();
        let mut draft_tokens = Vec::with_capacity(draft_budget);
        let mut draft_history = self.history.clone();
        let mut input_hidden = self.last_hidden.clone();
        let mut input_token = current_token;
        let start_pos = (self.history.len() - 1) as i32;
        let draft_uniforms = if self.request.sampler.is_pipelinable() {
            vec![0.0; draft_budget]
        } else {
            let mut draft_prng = split_speculative_draft_prng(&mut self.prng_state)?;
            draw_uniforms(&mut draft_prng, draft_budget)?
        };
        let mut distributions = Vec::with_capacity(draft_budget);

        for (offset, &draft_uniform) in draft_uniforms.iter().enumerate().take(draft_budget) {
            let token_arr: Array = (&[input_token][..], &[1_i32, 1_i32][..]).try_into()?;
            let position_ids = self.position_ids(start_pos + offset as i32, 1)?;
            let draft_forward_start = Instant::now();
            let output = self.model.mtp_forward_on(
                self.mtp,
                &input_hidden,
                &token_arr,
                &position_ids,
                None,
                Some(&mut self.mtp_cache),
                (),
            )?;
            add_elapsed_us(&mut self.stats.draft_forward_us, draft_forward_start);
            let draft_logits = constrain_speculative_logits(constraint, &output.logits, &[])?;
            let sampling_start = Instant::now();
            let (next_token, distribution) = if self.request.sampler.is_pipelinable() {
                sample_draft_logits_position(
                    &draft_logits,
                    self.request.sampler,
                    &draft_history,
                    None,
                )?
            } else {
                sample_draft_logits_position_with_uniform(
                    &draft_logits,
                    self.request.sampler,
                    &draft_history,
                    draft_uniform,
                )?
            };
            add_elapsed_us(&mut self.stats.sampling_us, sampling_start);
            commit_constraint_token(constraint, next_token)?;
            draft_tokens.push(next_token);
            distributions.push(distribution);
            draft_history.push(next_token);
            input_hidden = output.hidden_states;
            input_token = next_token;
        }

        Ok(MtpDraftResult {
            tokens: draft_tokens,
            distributions,
            cache_snapshot: mtp_snapshot,
        })
    }

    fn position_ids(&self, start_pos: i32, len: i32) -> Result<Array> {
        match self.dummy_position_ids.as_ref() {
            Some(dummy) => Ok(dummy.clone()),
            None => build_position_ids(start_pos, len),
        }
    }
}

fn constrain_speculative_logits(
    constraint: &mut Option<ConstraintSession>,
    logits: &Array,
    draft_tokens: &[u32],
) -> Result<Array> {
    match constraint {
        Some(session) => {
            apply_speculative_token_masks(logits, &[Some(session.speculative_masks(draft_tokens)?)])
        }
        None => Ok(logits.clone()),
    }
}

fn commit_constraint_token(constraint: &mut Option<ConstraintSession>, token: u32) -> Result<()> {
    if let Some(session) = constraint {
        session.commit_token(token)?;
    }
    Ok(())
}

pub(crate) fn verify_input(current_token: u32, draft_tokens: &[u32]) -> Vec<u32> {
    let mut input = Vec::with_capacity(draft_tokens.len() + 1);
    input.push(current_token);
    input.extend_from_slice(draft_tokens);
    input
}

pub(crate) fn sample_logits_positions(
    logits: &Array,
    sampler: Sampler,
    history: &[u32],
    prng_state: &mut Array,
) -> Result<Vec<u32>> {
    let shape = logits.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(anyhow!(
            "sample_logits_positions: expected logits shape [1, S, V], got {:?}",
            dims
        ));
    }
    let seq = dims[1];
    let vocab = dims[2];
    if sampler.is_pipelinable() {
        let ids = mlx::ops::reduction::argmax(logits, -1, false)?;
        let tokens: Vec<u32> = ids.to_vec()?;
        if tokens.len() != seq as usize {
            return Err(anyhow!(
                "sample_logits_positions: greedy argmax returned {} tokens, expected {}",
                tokens.len(),
                seq
            ));
        }
        return Ok(tokens);
    }
    let mut sampled = Vec::with_capacity(seq as usize);
    let mut running_history = history.to_vec();
    for pos in 0..seq {
        let row = mlx::ops::indexing::slice(
            logits,
            &[0_i32, pos, 0_i32][..],
            &[1_i32, pos + 1, vocab][..],
        )?;
        let row = row.reshape((vocab,))?;
        let token = sampler.sample(&row, &running_history, prng_state)?;
        running_history.push(token);
        sampled.push(token);
    }
    Ok(sampled)
}

pub(crate) fn resolve_greedy_verified_hidden_until_mismatch<M>(
    model: &M,
    verified_hidden: &Array,
    draft_tokens: &[u32],
    stats: &mut MtpSpeculativeStats,
    target: impl Into<StreamOrDevice>,
) -> Result<SpeculativeResolution>
where
    M: MtpSpeculativeModel,
{
    let target = target.into();
    let projection_start = Instant::now();
    let verified_logits = model.project_mtp_verify_hidden_on(verified_hidden, target)?;
    add_elapsed_us(&mut stats.projection_us, projection_start);

    let sampling_start = Instant::now();
    let verified_ids = mlx::ops::reduction::argmax(&verified_logits, -1, false)?;
    let verified_tokens: Vec<u32> = verified_ids.to_vec()?;
    add_elapsed_us(&mut stats.sampling_us, sampling_start);
    resolve_speculative_tokens(draft_tokens, &verified_tokens)
}

pub(crate) fn slice_hidden_position(hidden: &Array, pos: i32) -> Result<Array> {
    let shape = hidden.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(anyhow!(
            "slice_hidden_position: expected hidden shape [1, S, H], got {:?}",
            dims
        ));
    }
    let seq = dims[1];
    let hidden_size = dims[2];
    if pos < 0 || pos >= seq {
        return Err(anyhow!(
            "slice_hidden_position: pos {pos} out of [0, {seq})"
        ));
    }
    mlx::ops::indexing::slice_strided(
        hidden,
        &[0_i32, pos, 0_i32][..],
        &[1_i32, pos + 1, hidden_size][..],
        &[1_i32, 1_i32, 1_i32][..],
    )
    .map_err(anyhow::Error::from)
}

pub(crate) fn slice_hidden_prefix(hidden: &Array, len: usize) -> Result<Array> {
    let shape = hidden.shape();
    let dims = shape.as_slice();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(anyhow!(
            "slice_hidden_prefix: expected hidden shape [1, S, H], got {:?}",
            dims
        ));
    }
    if len == 0 || len > dims[1] as usize {
        return Err(anyhow!(
            "slice_hidden_prefix: len {len} out of [1, {}]",
            dims[1]
        ));
    }
    if len == dims[1] as usize {
        return Ok(hidden.clone());
    }
    mlx::ops::indexing::slice_strided(
        hidden,
        &[0_i32, 0_i32, 0_i32][..],
        &[1_i32, len as i32, dims[2]][..],
        &[1_i32, 1_i32, 1_i32][..],
    )
    .map_err(anyhow::Error::from)
}

pub(crate) fn slice_position_ids_prefix(position_ids: &Array, len: usize) -> Result<Array> {
    let shape = position_ids.shape();
    let dims = shape.as_slice();
    if len == 0 {
        return Err(anyhow!("slice_position_ids_prefix: len must be > 0"));
    }
    match dims {
        [1, seq] => {
            if len > *seq as usize {
                return Err(anyhow!(
                    "slice_position_ids_prefix: len {len} exceeds position_ids seq {seq}"
                ));
            }
            if len == *seq as usize {
                return Ok(position_ids.clone());
            }
            mlx::ops::indexing::slice_strided(
                position_ids,
                &[0_i32, 0_i32][..],
                &[1_i32, len as i32][..],
                &[1_i32, 1_i32][..],
            )
            .map_err(anyhow::Error::from)
        }
        [planes, 1, seq] => {
            if len > *seq as usize {
                return Err(anyhow!(
                    "slice_position_ids_prefix: len {len} exceeds position_ids seq {seq}"
                ));
            }
            if len == *seq as usize {
                return Ok(position_ids.clone());
            }
            mlx::ops::indexing::slice_strided(
                position_ids,
                &[0_i32, 0_i32, 0_i32][..],
                &[*planes, 1_i32, len as i32][..],
                &[1_i32, 1_i32, 1_i32][..],
            )
            .map_err(anyhow::Error::from)
        }
        _ => Err(anyhow!(
            "slice_position_ids_prefix: expected position_ids shape [1, S] or [P, 1, S], got {:?}",
            dims
        )),
    }
}

pub(crate) fn restore_layer_cache(
    cache: &mut [LayerCache],
    snapshots: &[LayerCacheSnapshot],
) -> Result<()> {
    if cache.len() != snapshots.len() {
        return Err(anyhow!(
            "restore_layer_cache: cache layers {} != snapshot layers {}",
            cache.len(),
            snapshots.len()
        ));
    }
    for (layer, snapshot) in cache.iter_mut().zip(snapshots.iter()) {
        layer.restore(snapshot)?;
    }
    Ok(())
}

pub(crate) fn layer_cache_supports_accepted_prefix_trim(cache: &[LayerCache]) -> bool {
    cache
        .iter()
        .all(|layer| matches!(layer, LayerCache::Full(_)))
}

pub(crate) fn trim_full_layer_cache_rows_to_accepted_prefix(
    cache: &mut [LayerCache],
    snapshots: &[LayerCacheSnapshot],
    accepted_by_row: &[(usize, usize)],
) -> Result<()> {
    if cache.len() != snapshots.len() {
        return Err(anyhow!(
            "trim_full_layer_cache_rows_to_accepted_prefix: cache layers {} != snapshot layers {}",
            cache.len(),
            snapshots.len()
        ));
    }
    if accepted_by_row.is_empty() {
        return Ok(());
    }

    for (layer_idx, (layer, snapshot)) in cache.iter_mut().zip(snapshots.iter()).enumerate() {
        let (LayerCache::Full(kv), LayerCacheSnapshot::Full(saved)) = (layer, snapshot) else {
            return Err(anyhow!(
                "trim_full_layer_cache_rows_to_accepted_prefix: accepted-prefix trim only supports Full KV layers, layer {layer_idx}"
            ));
        };
        let mut offsets = kv.offsets().to_vec();
        for &(row, accepted_len) in accepted_by_row {
            let base = *saved.offsets().get(row).ok_or_else(|| {
                anyhow!(
                    "trim_full_layer_cache_rows_to_accepted_prefix: row {row} out of snapshot offsets for layer {layer_idx}"
                )
            })?;
            let live = offsets.get_mut(row).ok_or_else(|| {
                anyhow!(
                    "trim_full_layer_cache_rows_to_accepted_prefix: row {row} out of live offsets for layer {layer_idx}"
                )
            })?;
            let accepted_len = i32::try_from(accepted_len).map_err(|_| {
                anyhow!(
                    "trim_full_layer_cache_rows_to_accepted_prefix: accepted_len {accepted_len} exceeds i32"
                )
            })?;
            let target = base.checked_add(accepted_len).ok_or_else(|| {
                anyhow!(
                    "trim_full_layer_cache_rows_to_accepted_prefix: base {base} + accepted_len {accepted_len} overflow"
                )
            })?;
            if target > *live {
                return Err(anyhow!(
                    "trim_full_layer_cache_rows_to_accepted_prefix: target offset {target} exceeds live offset {} for row {row} layer {layer_idx}",
                    *live
                ));
            }
            *live = target;
        }
        kv.restore_offsets(&offsets)?;
    }
    Ok(())
}

pub(crate) struct MainCacheRollbackInput<'a> {
    pub(crate) accepted_by_row: &'a [(usize, usize)],
    pub(crate) verify_input: &'a [u32],
    pub(crate) accepted_position_ids: &'a Array,
    pub(crate) verified_hidden: &'a Array,
}

pub(crate) fn rollback_main_cache_to_accepted_prefix<M: MtpSpeculativeModel>(
    model: &M,
    cache: &mut [LayerCache],
    snapshots: &[LayerCacheSnapshot],
    input: MainCacheRollbackInput<'_>,
    target: impl Into<mlx::StreamOrDevice>,
) -> Result<Array> {
    if input.accepted_by_row.len() != 1 || input.accepted_by_row[0].0 != 0 {
        return Err(anyhow!(
            "rollback_main_cache_to_accepted_prefix: single-row helper got accepted_by_row={:?}",
            input.accepted_by_row
        ));
    }
    let accepted_len = input.accepted_by_row[0].1;
    if accepted_len == 0 || accepted_len > input.verify_input.len() {
        return Err(anyhow!(
            "rollback_main_cache_to_accepted_prefix: accepted_len {accepted_len} outside [1, {}]",
            input.verify_input.len()
        ));
    }

    if layer_cache_supports_accepted_prefix_trim(cache) {
        trim_full_layer_cache_rows_to_accepted_prefix(cache, snapshots, input.accepted_by_row)?;
        return slice_hidden_prefix(input.verified_hidden, accepted_len);
    }

    restore_layer_cache(cache, snapshots)?;
    let accepted_arr: Array = (
        &input.verify_input[..accepted_len],
        &[1_i32, accepted_len as i32][..],
    )
        .try_into()?;
    let _verify_qmm = crate::nn::verify_qmm::armed_scope();
    model.forward_text_hidden(
        &accepted_arr,
        input.accepted_position_ids,
        None,
        None,
        Some(cache),
        target.into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cache::{KVCache, TurboQuantKVBits};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeGreedyProjectModel {
        tokens: Vec<u32>,
        project_calls: AtomicUsize,
        replay_calls: AtomicUsize,
    }

    impl FakeGreedyProjectModel {
        fn new(tokens: Vec<u32>) -> Self {
            Self {
                tokens,
                project_calls: AtomicUsize::new(0),
                replay_calls: AtomicUsize::new(0),
            }
        }

        fn project_calls(&self) -> usize {
            self.project_calls.load(Ordering::Relaxed)
        }

        fn replay_calls(&self) -> usize {
            self.replay_calls.load(Ordering::Relaxed)
        }
    }

    impl Model for FakeGreedyProjectModel {
        fn make_cache(&self, _batch: i32, _cap: i32, _dtype: Dtype) -> Result<Vec<LayerCache>> {
            Ok(Vec::new())
        }

        fn forward_on(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            Err(anyhow!("FakeGreedyProjectModel::forward_on unused"))
        }

        fn batched_prefill(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _attention_mask: &Array,
            _linear_attention_mask: &Array,
            _per_row_lens: &[i32],
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            Err(anyhow!("FakeGreedyProjectModel::batched_prefill unused"))
        }

        fn forward_text_hidden(
            &self,
            input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            self.replay_calls.fetch_add(1, Ordering::Relaxed);
            let dims = input_ids.shape();
            let dims = dims.as_slice();
            if dims.len() != 2 {
                return Err(anyhow!(
                    "FakeGreedyProjectModel::forward_text_hidden expected [B,S], got {dims:?}"
                ));
            }
            if let Some(cache) = cache {
                for layer in cache {
                    if let LayerCache::Linear(gd) = layer {
                        let row_lens = vec![dims[1]; dims[0] as usize];
                        gd.advance(&row_lens)?;
                    }
                }
            }
            Array::zeros((dims[0], dims[1], 1_i32), Dtype::Float32).map_err(anyhow::Error::from)
        }

        fn project_hidden_on(&self, hidden: &Array, _target: StreamOrDevice) -> Result<Array> {
            self.project_calls.fetch_add(1, Ordering::Relaxed);
            let shape = hidden.shape();
            let dims = shape.as_slice();
            let seq = dims[1] as usize;
            if self.tokens.len() != seq {
                return Err(anyhow!(
                    "fake token count {} does not match hidden seq {seq}",
                    self.tokens.len()
                ));
            }
            let vocab = 128_usize;
            let mut logits = vec![0.0_f32; seq * vocab];
            for (pos, &token) in self.tokens.iter().enumerate() {
                logits[pos * vocab + token as usize] = 100.0;
            }
            (&logits[..], &[1_i32, seq as i32, vocab as i32][..])
                .try_into()
                .map_err(anyhow::Error::from)
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }
    }

    impl MtpSpeculativeModel for FakeGreedyProjectModel {
        type MtpHead = ();

        fn load_mtp_head(&self, _loader: &Loader) -> Result<Self::MtpHead> {
            Ok(())
        }

        fn make_mtp_cache(
            &self,
            _mtp: &Self::MtpHead,
            _batch: i32,
            _cap: i32,
            _dtype: Dtype,
        ) -> Result<MtpCache> {
            Err(anyhow!("FakeGreedyProjectModel::make_mtp_cache unused"))
        }

        fn mtp_hidden_size(&self, _mtp: &Self::MtpHead) -> i32 {
            1
        }

        fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> Dtype {
            Dtype::Float32
        }

        fn mtp_forward_hidden_on(
            &self,
            _mtp: &Self::MtpHead,
            _hidden_states: &Array,
            _next_token_ids: &Array,
            _position_ids: &Array,
            _mask: Option<&Array>,
            _mtp_cache: Option<&mut MtpCache>,
            _target: impl Into<StreamOrDevice>,
        ) -> Result<Array> {
            Err(anyhow!(
                "FakeGreedyProjectModel::mtp_forward_hidden_on unused"
            ))
        }

        fn mtp_forward_on(
            &self,
            _mtp: &Self::MtpHead,
            _hidden_states: &Array,
            _next_token_ids: &Array,
            _position_ids: &Array,
            _mask: Option<&Array>,
            _mtp_cache: Option<&mut MtpCache>,
            _target: impl Into<StreamOrDevice>,
        ) -> Result<MtpStepOutput> {
            Err(anyhow!("FakeGreedyProjectModel::mtp_forward_on unused"))
        }
    }

    #[test]
    fn greedy_verify_resolve_batches_projection_before_mismatch_resolution() {
        let model = FakeGreedyProjectModel::new(vec![4, 99, 6, 7]);
        let hidden = Array::zeros((1_i32, 4_i32, 1_i32), Dtype::Float32).expect("hidden");
        let mut stats = MtpSpeculativeStats::default();

        let resolution = resolve_greedy_verified_hidden_until_mismatch(
            &model,
            &hidden,
            &[4, 5, 6],
            &mut stats,
            (),
        )
        .expect("resolution");

        assert_eq!(resolution.accepted_draft_len, 1);
        assert_eq!(resolution.tokens_to_append, vec![4, 99]);
        assert_eq!(resolution.accepted_verify_input_len, 2);
        assert!(resolution.needs_rollback);
        assert_eq!(model.project_calls(), 1);
    }

    #[test]
    fn greedy_verify_resolve_projects_bonus_after_full_accept() {
        let model = FakeGreedyProjectModel::new(vec![4, 5, 6, 7]);
        let hidden = Array::zeros((1_i32, 4_i32, 1_i32), Dtype::Float32).expect("hidden");
        let mut stats = MtpSpeculativeStats::default();

        let resolution = resolve_greedy_verified_hidden_until_mismatch(
            &model,
            &hidden,
            &[4, 5, 6],
            &mut stats,
            (),
        )
        .expect("resolution");

        assert_eq!(resolution.accepted_draft_len, 3);
        assert_eq!(resolution.tokens_to_append, vec![4, 5, 6, 7]);
        assert_eq!(resolution.accepted_verify_input_len, 4);
        assert!(!resolution.needs_rollback);
        assert_eq!(model.project_calls(), 1);
    }

    #[test]
    fn exact_sampling_accepts_deterministic_draft_and_samples_bonus() {
        let logits: Array = (
            &[
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
            ][..],
            &[1_i32, 3_i32, 4_i32][..],
        )
            .try_into()
            .unwrap();
        let mut prng = mlx::random::key(17).unwrap();
        let resolution = resolve_exact_speculative_logits(
            &[1, 1],
            &[
                DraftTokenDistribution::Deterministic,
                DraftTokenDistribution::Deterministic,
            ],
            &logits,
            Sampler::greedy().with_temperature(1.0),
            &[9],
            &mut prng,
        )
        .unwrap();

        assert_eq!(resolution.accepted_draft_len, 2);
        assert_eq!(resolution.tokens_to_append, vec![1, 1, 2]);
        assert_eq!(resolution.accepted_verify_input_len, 3);
        assert!(!resolution.needs_rollback);
        assert_eq!(resolution.exact_sampling.windows, 1);
        assert_eq!(resolution.exact_sampling.acceptance_draws, 2);
        assert_eq!(resolution.exact_sampling.residual_corrections, 0);
        assert_eq!(resolution.exact_sampling.bonus_samples, 1);
    }

    #[test]
    fn exact_sampling_rejects_deterministic_draft_and_uses_residual() {
        let logits: Array = (
            &[
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ][..],
            &[1_i32, 2_i32, 4_i32][..],
        )
            .try_into()
            .unwrap();
        let mut prng = mlx::random::key(23).unwrap();
        let resolution = resolve_exact_speculative_logits(
            &[1],
            &[DraftTokenDistribution::Deterministic],
            &logits,
            Sampler::greedy().with_temperature(1.0),
            &[9],
            &mut prng,
        )
        .unwrap();

        assert_eq!(resolution.accepted_draft_len, 0);
        assert_eq!(resolution.tokens_to_append, vec![2]);
        assert_eq!(resolution.accepted_verify_input_len, 1);
        assert!(resolution.needs_rollback);
        assert_eq!(resolution.exact_sampling.windows, 1);
        assert_eq!(resolution.exact_sampling.acceptance_draws, 1);
        assert_eq!(resolution.exact_sampling.residual_corrections, 1);
        assert_eq!(resolution.exact_sampling.bonus_samples, 0);
    }

    #[test]
    fn exact_sampling_target_coupling_accepts_and_samples_bonus() {
        let target_distributions = [
            SamplingDistribution::new(vec![0.0, 1.0, 0.0, 0.0]).unwrap(),
            SamplingDistribution::new(vec![0.0, 0.0, 1.0, 0.0]).unwrap(),
            SamplingDistribution::new(vec![0.0, 0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut prng = mlx::random::key(29).unwrap();
        let resolution = resolve_exact_deterministic_target_distributions(
            &[1, 2],
            &target_distributions,
            &mut prng,
        )
        .unwrap();

        assert_eq!(resolution.accepted_draft_len, 2);
        assert_eq!(resolution.tokens_to_append, vec![1, 2, 3]);
        assert!(!resolution.needs_rollback);
        assert_eq!(resolution.exact_sampling.acceptance_draws, 2);
        assert_eq!(resolution.exact_sampling.residual_corrections, 0);
        assert_eq!(resolution.exact_sampling.bonus_samples, 1);
    }

    #[test]
    fn exact_sampling_target_coupling_reuses_rejected_target_as_correction() {
        let target_distributions = [
            SamplingDistribution::new(vec![0.0, 0.0, 1.0, 0.0]).unwrap(),
            SamplingDistribution::new(vec![0.0, 0.0, 0.0, 1.0]).unwrap(),
        ];
        let mut prng = mlx::random::key(31).unwrap();
        let resolution = resolve_exact_deterministic_target_distributions(
            &[1],
            &target_distributions,
            &mut prng,
        )
        .unwrap();

        assert_eq!(resolution.accepted_draft_len, 0);
        assert_eq!(resolution.tokens_to_append, vec![2]);
        assert!(resolution.needs_rollback);
        assert_eq!(resolution.exact_sampling.acceptance_draws, 1);
        assert_eq!(resolution.exact_sampling.residual_corrections, 1);
        assert_eq!(resolution.exact_sampling.bonus_samples, 0);
    }

    #[test]
    fn exact_sampling_target_tokens_preserve_target_coupling_counters() {
        let accepted =
            resolve_exact_deterministic_target_tokens(&[1, 2], &[1, 2, 3]).expect("accepted");
        assert_eq!(accepted.accepted_draft_len, 2);
        assert_eq!(accepted.tokens_to_append, vec![1, 2, 3]);
        assert!(!accepted.needs_rollback);
        assert_eq!(accepted.exact_sampling.acceptance_draws, 2);
        assert_eq!(accepted.exact_sampling.residual_corrections, 0);
        assert_eq!(accepted.exact_sampling.bonus_samples, 1);

        let rejected =
            resolve_exact_deterministic_target_tokens(&[1, 2], &[1, 9, 3]).expect("rejected");
        assert_eq!(rejected.accepted_draft_len, 1);
        assert_eq!(rejected.tokens_to_append, vec![1, 9]);
        assert!(rejected.needs_rollback);
        assert_eq!(rejected.exact_sampling.acceptance_draws, 2);
        assert_eq!(rejected.exact_sampling.residual_corrections, 1);
        assert_eq!(rejected.exact_sampling.bonus_samples, 0);
    }

    #[test]
    fn exact_deterministic_coupling_preserves_target_distribution_on_uniform_grid() {
        let target =
            SamplingDistribution::new(vec![0.1, 0.2, 0.3, 0.4]).expect("target distribution");
        let mut counts = [0_usize; 4];
        let mut accepted = 0_usize;
        let mut corrected = 0_usize;

        for index in 0..1_000 {
            let uniform = (index as f32 + 0.5) / 1_000.0;
            let target_token = target
                .sample_with_uniform(uniform)
                .expect("sample target token");
            let resolution = resolve_exact_deterministic_target_tokens(&[3], &[target_token, 0])
                .expect("resolve deterministic draft");
            counts[resolution.tokens_to_append[0] as usize] += 1;
            accepted += resolution.accepted_draft_len;
            corrected += resolution.exact_sampling.residual_corrections;
        }

        assert_eq!(counts, [100, 200, 300, 400]);
        assert_eq!(accepted, 400);
        assert_eq!(corrected, 600);
    }

    #[test]
    fn exact_target_logits_preserve_position_histories_with_penalties() {
        let logits: Array = (
            &[
                0.0_f32, 20.0, 0.0, 0.0, //
                0.0, 0.0, 20.0, 0.0, //
                0.0, 0.0, 0.0, 20.0,
            ][..],
            &[1_i32, 3, 4][..],
        )
            .try_into()
            .unwrap();
        let sampler = Sampler::greedy()
            .with_temperature(0.8)
            .with_top_p(0.95)
            .with_repetition_penalty(1.1)
            .with_frequency_penalty(0.2)
            .with_presence_penalty(0.1);
        let mut prng = mlx::random::key(41).unwrap();
        let resolution = resolve_exact_deterministic_target_logits(
            &[1, 2],
            &logits,
            sampler,
            &[0, 0, 1],
            &mut prng,
        )
        .unwrap();

        assert_eq!(resolution.accepted_draft_len, 2);
        assert_eq!(resolution.tokens_to_append, vec![1, 2, 3]);
        assert!(!resolution.needs_rollback);
    }

    #[test]
    fn speculative_prng_split_is_reproducible_independent_and_shape_preserving() {
        for shape in [&[2_i32][..], &[1_i32, 2_i32][..]] {
            let mut decision_a = mlx::random::key(47).unwrap().reshape(shape).unwrap();
            let mut decision_b = mlx::random::key(47).unwrap().reshape(shape).unwrap();

            let draft_a = split_speculative_draft_prng(&mut decision_a).unwrap();
            let draft_b = split_speculative_draft_prng(&mut decision_b).unwrap();

            assert_eq!(decision_a.shape().as_slice(), shape);
            assert_eq!(decision_b.shape().as_slice(), shape);
            assert_eq!(
                decision_a.to_vec::<u32>().unwrap(),
                decision_b.to_vec::<u32>().unwrap()
            );
            assert_eq!(
                draft_a.to_vec::<u32>().unwrap(),
                draft_b.to_vec::<u32>().unwrap()
            );
            assert_ne!(
                decision_a.to_vec::<u32>().unwrap(),
                draft_a.to_vec::<u32>().unwrap()
            );
        }
    }

    #[test]
    fn exact_sampling_same_seed_replays_resolution_and_prng_state() {
        let logits: Array = (
            &[
                0.0_f32, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 0.5, 1.5, 2.5, 3.5,
            ][..],
            &[1_i32, 3_i32, 4_i32][..],
        )
            .try_into()
            .unwrap();
        let draft_distributions = vec![
            DraftTokenDistribution::Sampled(
                SamplingDistribution::new(vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
            ),
            DraftTokenDistribution::Sampled(
                SamplingDistribution::new(vec![0.4, 0.3, 0.2, 0.1]).unwrap(),
            ),
        ];
        let sampler = Sampler::greedy()
            .with_temperature(0.8)
            .with_top_p(0.95)
            .with_seed(71);
        let mut key_a = mlx::random::key(71)
            .unwrap()
            .reshape((1_i32, 2_i32))
            .unwrap();
        let mut key_b = mlx::random::key(71)
            .unwrap()
            .reshape((1_i32, 2_i32))
            .unwrap();

        let resolution_a = resolve_exact_speculative_logits(
            &[3, 0],
            &draft_distributions,
            &logits,
            sampler,
            &[9, 8],
            &mut key_a,
        )
        .unwrap();
        let resolution_b = resolve_exact_speculative_logits(
            &[3, 0],
            &draft_distributions,
            &logits,
            sampler,
            &[9, 8],
            &mut key_b,
        )
        .unwrap();

        assert_eq!(resolution_a, resolution_b);
        assert_eq!(key_a.shape().as_slice(), &[1, 2]);
        assert_eq!(key_b.shape().as_slice(), &[1, 2]);
        assert_eq!(
            key_a.to_vec::<u32>().unwrap(),
            key_b.to_vec::<u32>().unwrap()
        );
    }

    #[test]
    fn mtp_rollback_main_cache_replays_hybrid_cache_after_mismatch() {
        let model = FakeGreedyProjectModel::new(Vec::new());
        let mut cache = vec![LayerCache::Linear(
            crate::core::cache::GatedDeltaCache::new_with_cap(1, 4, 8, 1, 4, 4, Dtype::Float32, 16)
                .expect("linear cache"),
        )];
        if let LayerCache::Linear(gd) = &mut cache[0] {
            gd.advance(&[4]).expect("base prefix");
        }
        let snapshots = cache.iter().map(LayerCache::snapshot).collect::<Vec<_>>();
        if let LayerCache::Linear(gd) = &mut cache[0] {
            gd.advance(&[3]).expect("verified suffix");
        }

        let verify_input = vec![10_u32, 11, 12];
        let accepted_position_ids =
            crate::core::generate::build_position_ids(4, 2).expect("position ids");
        let verified_hidden =
            Array::zeros((1_i32, 3_i32, 1_i32), Dtype::Float32).expect("verified hidden");

        let accepted_hidden = rollback_main_cache_to_accepted_prefix(
            &model,
            &mut cache,
            &snapshots,
            MainCacheRollbackInput {
                accepted_by_row: &[(0, 2)],
                verify_input: &verify_input,
                accepted_position_ids: &accepted_position_ids,
                verified_hidden: &verified_hidden,
            },
            (),
        )
        .expect("hybrid rollback replay");

        assert_eq!(accepted_hidden.shape().as_slice(), &[1, 2, 1]);
        assert_eq!(model.replay_calls(), 1);
        let LayerCache::Linear(gd) = &cache[0] else {
            panic!("expected linear cache");
        };
        assert_eq!(gd.offsets(), &[6]);
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn accepted_prefix_trim_supports_paged_kv() {
        let mut kv = KVCache::new(1, 1, 2, 2, Dtype::Float32, 8).with_step(4);
        kv.enable_paged(2, 4).expect("enable paged KV");
        let base_k: Array = (&[1.0_f32, 2.0, 3.0, 4.0][..], &[1_i32, 1, 2, 2][..])
            .try_into()
            .unwrap();
        let base_v = &base_k + 100.0_f32;
        kv.update_and_fetch(&base_k, &base_v, &[2])
            .expect("paged base prefix");
        let mut cache = vec![LayerCache::Full(kv)];
        let snapshots = cache.iter().map(LayerCache::snapshot).collect::<Vec<_>>();
        let verify_k: Array = (
            &[5.0_f32, 6.0, 7.0, 8.0, 9.0, 10.0][..],
            &[1_i32, 1, 3, 2][..],
        )
            .try_into()
            .unwrap();
        let verify_v = &verify_k + 100.0_f32;
        let LayerCache::Full(kv) = &mut cache[0] else {
            panic!("full cache");
        };
        kv.update_and_fetch(&verify_k, &verify_v, &[3])
            .expect("paged verify suffix");

        trim_full_layer_cache_rows_to_accepted_prefix(&mut cache, &snapshots, &[(0, 1)])
            .expect("trim paged accepted prefix");

        let LayerCache::Full(kv) = &cache[0] else {
            panic!("full cache");
        };
        assert_eq!(kv.offsets(), &[3]);
        let (keys, values) = kv
            .materialize_current_paged_prefix_on(())
            .expect("materialize trimmed paged prefix");
        assert_eq!(keys.shape().as_slice(), &[1, 1, 3, 2]);
        assert_eq!(values.shape().as_slice(), &[1, 1, 3, 2]);
        assert_eq!(
            keys.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    #[serial_test::serial(mlx_metal)]
    fn accepted_prefix_trim_supports_turboquant_kv() {
        let mut kv = KVCache::new(1, 1, 8, 8, Dtype::Float32, 8)
            .with_step(8)
            .with_turboquant(TurboQuantKVBits::K4V4)
            .expect("enable TurboQuant KV");
        let base_data = (0..16).map(|idx| idx as f32 * 0.1).collect::<Vec<_>>();
        let base_k: Array = (base_data.as_slice(), &[1_i32, 1, 2, 8][..])
            .try_into()
            .unwrap();
        let base_v = &base_k + 1.0_f32;
        kv.update_and_fetch(&base_k, &base_v, &[2])
            .expect("TurboQuant base prefix");
        let mut cache = vec![LayerCache::Full(kv)];
        let snapshots = cache.iter().map(LayerCache::snapshot).collect::<Vec<_>>();
        let verify_data = (0..24)
            .map(|idx| 2.0_f32 + idx as f32 * 0.1)
            .collect::<Vec<_>>();
        let verify_k: Array = (verify_data.as_slice(), &[1_i32, 1, 3, 8][..])
            .try_into()
            .unwrap();
        let verify_v = &verify_k + 1.0_f32;
        let LayerCache::Full(kv) = &mut cache[0] else {
            panic!("full cache");
        };
        kv.update_and_fetch(&verify_k, &verify_v, &[3])
            .expect("TurboQuant verify suffix");

        trim_full_layer_cache_rows_to_accepted_prefix(&mut cache, &snapshots, &[(0, 1)])
            .expect("trim TurboQuant accepted prefix");

        let LayerCache::Full(kv) = &cache[0] else {
            panic!("full cache");
        };
        assert_eq!(kv.offsets(), &[3]);
        let (keys, values, len) = kv
            .dense_prefix_layer_for_row_on(0, ())
            .expect("materialize trimmed TurboQuant prefix");
        assert_eq!(len, 3);
        assert_eq!(keys.shape().as_slice(), &[1, 1, 3, 8]);
        assert_eq!(values.shape().as_slice(), &[1, 1, 3, 8]);
    }

    #[test]
    fn mtp_policy_defaults_qwen35_dense_4b_to_d1() {
        let raw = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 2560,
                "num_hidden_layers": 32
            }
        });

        assert_eq!(default_mtp_draft_tokens_for_config(&raw), 1);
        assert_eq!(
            resolve_mtp_draft_tokens(&raw, MtpDraftTokensArg::Omitted),
            1
        );
    }

    #[test]
    fn mtp_policy_defaults_qwen36_and_qwen38_dense_27b_to_d2() {
        let raw = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "num_hidden_layers": 64
            }
        });

        assert_eq!(default_mtp_draft_tokens_for_config(&raw), 2);
    }

    #[test]
    fn mtp_policy_defaults_qwen36_moe_35b_a3b_to_d2() {
        let raw = serde_json::json!({
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_experts": 256,
                "num_experts_per_tok": 8
            }
        });

        assert_eq!(default_mtp_draft_tokens_for_config(&raw), 2);
    }

    #[test]
    fn mtp_policy_defaults_gemma4_to_d1() {
        for model_type in ["gemma4", "gemma4_unified"] {
            let raw = serde_json::json!({
                "model_type": model_type,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 3584,
                    "num_hidden_layers": 34
                }
            });

            assert_eq!(default_mtp_draft_tokens_for_config(&raw), 1);
            assert_eq!(
                resolve_mtp_draft_tokens(&raw, MtpDraftTokensArg::Omitted),
                1
            );
        }
    }

    #[test]
    fn mtp_policy_preserves_explicit_value() {
        let raw = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "num_hidden_layers": 64
            }
        });

        assert_eq!(
            resolve_mtp_draft_tokens(&raw, MtpDraftTokensArg::Explicit(1)),
            1
        );
    }

    #[test]
    fn mtp_stats_tracks_attempts_and_accepts_by_draft_position() {
        let mut stats = MtpSpeculativeStats::default();

        stats.record_window_acceptance(4, 0);
        stats.record_window_acceptance(4, 2);
        stats.record_window_acceptance(2, 2);

        assert_eq!(stats.draft_attempts_by_position, vec![3, 3, 2, 2]);
        assert_eq!(stats.draft_accepts_by_position, vec![2, 2, 0, 0]);
        assert_eq!(stats.multi_token_windows(), 3);
    }

    #[test]
    fn draft_cap_context_bucket_uses_inclusive_boundaries() {
        assert_eq!(
            MtpDraftCapContextBucket::for_tokens(2_048),
            MtpDraftCapContextBucket::UpTo2k
        );
        assert_eq!(
            MtpDraftCapContextBucket::for_tokens(2_049),
            MtpDraftCapContextBucket::UpTo8k
        );
        assert_eq!(
            MtpDraftCapContextBucket::for_tokens(131_073),
            MtpDraftCapContextBucket::Above128k
        );
    }

    #[test]
    fn draft_cap_observation_aggregates_only_matching_regimes() {
        let mut stats = MtpSpeculativeStats::default();
        let timing = MtpDraftCapTiming {
            draft_forward_us: 10,
            verify_forward_us: 20,
            projection_us: 3,
            sampling_us: 4,
            main_rollback_us: 5,
            decode_cache_commit_us: 6,
            cache_restore_us: 7,
        };

        stats.record_draft_cap_observation(2, &[2, 2], &[1_000, 2_000], 3, 5, 1, 100, timing);
        stats.record_draft_cap_observation(2, &[2, 2], &[1_500, 2_048], 2, 4, 2, 120, timing);
        stats.record_draft_cap_observation(2, &[1, 2], &[2_048, 2_049], 1, 2, 1, 80, timing);

        assert_eq!(stats.draft_cap_observations.len(), 2);
        let homogeneous = &stats.draft_cap_observations[0];
        assert_eq!(homogeneous.windows, 4);
        assert_eq!(homogeneous.accepted_draft_tokens, 5);
        assert_eq!(homogeneous.committed_tokens, 9);
        assert_eq!(homogeneous.total_us, 220);
        assert_eq!(homogeneous.draft_forward_us, 20);

        let mixed = &stats.draft_cap_observations[1];
        assert_eq!(mixed.windows, 2);
        assert!(mixed.mixed_context_buckets);
        assert_eq!(mixed.min_draft_tokens, 1);
        assert_eq!(mixed.max_draft_tokens, 2);
        assert_eq!(mixed.context_bucket, MtpDraftCapContextBucket::UpTo8k);
    }

    fn policy_window(
        draft_tokens: usize,
        committed_tokens: usize,
        total_us: u64,
    ) -> MtpDraftPolicyWindow {
        MtpDraftPolicyWindow {
            attempted_draft_tokens: draft_tokens,
            accepted_draft_tokens: draft_tokens,
            committed_tokens,
            total_us,
            context_tokens: 1_024,
            batch_width: 1,
            kv_state: MtpDraftPolicyKvState::Contiguous,
            ..MtpDraftPolicyWindow::default()
        }
    }

    #[test]
    fn mtp_cost_aware_policy_immediately_reduces_zero_acceptance() {
        let mut policy = Gemma4DrafterPolicyState::new(4);
        let mut window = policy_window(4, 1, 4_000);
        window.accepted_draft_tokens = 0;
        window.main_rollback_us = 500;
        window.mtp_cache_restore_us = 300;

        let change = policy.observe_window(window);

        assert_eq!(policy.current_budget(), 1);
        assert!(change.reduced);
    }

    #[test]
    fn mtp_cost_aware_policy_rejects_a_more_expensive_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(4);

        policy.observe_window(policy_window(4, 4, 400));
        let regime = policy_window(4, 4, 400).regime();
        policy.record_cost(regime, 0, 1_000.0, 1.0);
        assert_eq!(
            policy.observe_window(policy_window(4, 4, 400)).reduced,
            true
        );
        assert_eq!(policy.current_budget(), 3);
        policy.observe_window(policy_window(3, 3, 600));
        let change = policy.observe_window(policy_window(3, 3, 600));

        assert_eq!(policy.current_budget(), 4);
        assert!(change.increased);
    }

    #[test]
    fn mtp_cost_aware_policy_backs_off_rejected_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(4);

        policy.observe_window(policy_window(4, 4, 400));
        let regime = policy_window(4, 4, 400).regime();
        policy.record_cost(regime, 0, 1_000.0, 1.0);
        policy.observe_window(policy_window(4, 4, 400));
        policy.observe_window(policy_window(3, 3, 600));
        policy.observe_window(policy_window(3, 3, 600));
        assert_eq!(policy.current_budget(), 4);

        for _ in 0..16 {
            policy.observe_window(policy_window(4, 4, 400));
            assert_eq!(policy.current_budget(), 4);
        }
        assert!(policy.observe_window(policy_window(4, 4, 400)).reduced);
        assert_eq!(policy.current_budget(), 3);
    }

    #[test]
    fn mtp_cost_aware_policy_transitions_through_single_draft_before_ordinary_decode() {
        let mut policy = Gemma4DrafterPolicyState::new(2);
        policy.observe_window(policy_window(2, 2, 100));
        let mut rejected = policy_window(2, 1, 100);
        rejected.accepted_draft_tokens = 0;
        policy.observe_window(rejected);
        assert_eq!(policy.current_budget(), 1);

        let mut rejected_single = policy_window(1, 1, 100);
        rejected_single.accepted_draft_tokens = 0;
        policy.observe_window(rejected_single);
        policy.observe_window(rejected_single);
        assert_eq!(policy.current_budget(), 0);
        assert!(policy.uses_ordinary_decode());
    }

    #[test]
    fn mtp_cost_aware_policy_seeds_only_an_unobserved_regime() {
        let mut policy = Gemma4DrafterPolicyState::new(4);
        assert!(policy.seed_initial_budget(1));
        assert_eq!(policy.current_budget(), 1);

        policy.observe_window(policy_window(1, 2, 100));
        assert!(!policy.seed_initial_budget(3));
        assert_eq!(policy.current_budget(), 1);
    }

    #[test]
    fn mtp_cost_aware_policy_probes_ordinary_decode_early_for_low_acceptance() {
        let mut policy = Gemma4DrafterPolicyState::new(2);
        let mut rejected_wide = policy_window(2, 1, 2_000);
        rejected_wide.accepted_draft_tokens = 0;
        policy.observe_window(rejected_wide);
        assert_eq!(policy.current_budget(), 1);

        let mut rejected_single = policy_window(1, 1, 1_000);
        rejected_single.accepted_draft_tokens = 0;
        policy.observe_window(rejected_single);
        let change = policy.observe_window(rejected_single);

        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
        assert!(policy.uses_ordinary_decode());
        assert!(!policy.should_maintain_mtp_cache());
    }

    #[test]
    fn mtp_cost_aware_policy_probes_ordinary_decode_early_at_long_context() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        let long_window = MtpDraftPolicyWindow {
            context_tokens: 65_536,
            ..policy_window(1, 2, 200)
        };

        policy.observe_window(long_window);
        let change = policy.observe_window(long_window);

        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
        assert!(policy.should_maintain_mtp_cache());
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(MtpDraftPolicyWindow {
                context_tokens: 65_536,
                ..policy_window(0, 1, 50)
            });
        }
        assert!(policy.uses_ordinary_decode());
    }

    #[test]
    fn mtp_cost_aware_policy_does_not_collapse_wide_full_acceptance_into_ordinary_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(2);

        for _ in 0..64 {
            let budget = policy.current_budget();
            policy.observe_window(policy_window(budget, budget + 1, 200));
        }

        assert_eq!(policy.current_budget(), 2);
        assert!(!policy.uses_ordinary_decode());
    }

    #[test]
    fn mtp_cost_aware_policy_keeps_a_cheaper_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(4);

        policy.observe_window(policy_window(4, 4, 800));
        let regime = policy_window(4, 4, 800).regime();
        policy.record_cost(regime, 0, 1_000.0, 1.0);
        policy.observe_window(policy_window(4, 4, 800));
        policy.observe_window(policy_window(3, 3, 300));
        let change = policy.observe_window(policy_window(3, 3, 300));

        assert_eq!(policy.current_budget(), 3);
        assert!(!change.reduced);
        assert!(!change.increased);
    }

    #[test]
    fn mtp_cost_aware_policy_separates_context_batch_and_kv_regimes() {
        let base = policy_window(2, 2, 200);
        let long_context = MtpDraftPolicyWindow {
            context_tokens: 32_000,
            ..base
        };
        let batched = MtpDraftPolicyWindow {
            batch_width: 4,
            ..base
        };
        let paged = MtpDraftPolicyWindow {
            kv_state: MtpDraftPolicyKvState::PagedActiveKv,
            ..base
        };

        assert_ne!(base.regime(), long_context.regime());
        assert_ne!(base.regime(), batched.regime());
        assert_ne!(base.regime(), paged.regime());
    }

    #[test]
    fn mtp_cost_aware_policy_uses_actual_committed_tokens_and_cache_costs() {
        let cheap = policy_window(2, 2, 200);
        let fewer_commits = policy_window(2, 1, 200);
        let cache_restore = MtpDraftPolicyWindow {
            total_us: 0,
            verify_forward_us: 100,
            mtp_cache_restore_us: 300,
            ..policy_window(2, 2, 0)
        };

        assert_eq!(cheap.gemma4_cost_per_committed_token_us(), 100.0);
        assert_eq!(cheap.qwen_cost_per_committed_token_us(), 100.0);
        assert_eq!(fewer_commits.gemma4_cost_per_committed_token_us(), 200.0);
        assert_eq!(fewer_commits.qwen_cost_per_committed_token_us(), 200.0);
        assert_eq!(cache_restore.gemma4_cost_per_committed_token_us(), 200.0);
        assert_eq!(cache_restore.qwen_cost_per_committed_token_us(), 200.0);
    }

    #[test]
    fn mtp_cost_aware_policy_compares_zero_draft_without_speculative_overhead() {
        let control = MtpDraftPolicyWindow {
            attempted_draft_tokens: 0,
            committed_tokens: 1,
            total_us: 1_000,
            sampling_us: 100,
            ..policy_window(0, 1, 0)
        };

        assert_eq!(control.gemma4_cost_per_committed_token_us(), 100.0);
        assert_eq!(control.qwen_cost_per_committed_token_us(), 1_000.0);
    }

    #[test]
    fn mtp_cost_aware_policy_excludes_cold_ordinary_compile_cost() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        let regime = policy_window(0, 1, 1_000).regime();

        policy.record_cost(regime, 0, 1_000.0, 1.0);
        let cold = policy.cost_estimate(regime, 0).unwrap();
        assert_eq!(cold.samples, 0);

        policy.record_cost(regime, 0, 100.0, 1.0);
        let warm = policy.cost_estimate(regime, 0).unwrap();
        assert_eq!(warm.samples, 1);
        assert_eq!(warm.cost_ewma, 100.0);
    }

    #[test]
    fn qwen_mtp_policy_keeps_the_pre_gemma_control_cost_sampling() {
        let mut policy = QwenMtpDraftPolicyState::new(1);
        let regime = policy_window(0, 1, 1_000).regime();

        policy.record_cost(regime, 0, 1_000.0, 1.0);
        let first = policy.cost_estimate(regime, 0).unwrap();
        assert_eq!(first.samples, 1);
        assert_eq!(first.cost_ewma, 1_000.0);

        policy.record_cost(regime, 0, 100.0, 1.0);
        let second = policy.cost_estimate(regime, 0).unwrap();
        assert_eq!(second.samples, 2);
        assert_eq!(second.cost_ewma, 685.0);
    }

    #[test]
    fn qwen_fixed_draft_depth_scope_disables_adaptation_only_while_armed() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut rejected = policy_window(2, 1, 1_000);
        rejected.accepted_draft_tokens = 0;

        {
            let _fixed = qwen_fixed_mtp_draft_depth_scope();
            let change = policy.observe_window(rejected);
            assert_eq!(change, MtpDraftBudgetChange::default());
            assert_eq!(policy.current_budget(), 2);
        }

        let change = policy.observe_window(rejected);
        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 1);
    }

    #[test]
    fn mtp_cost_aware_policy_snapshot_restores_cost_history() {
        let mut source = Gemma4DrafterPolicyState::new(2);
        source.observe_window(policy_window(2, 2, 200));
        let snapshot = source.snapshot();
        let mut restored = Gemma4DrafterPolicyState::new(2);

        restored.restore_snapshot(snapshot.clone()).unwrap();

        assert_eq!(restored.snapshot(), snapshot);
    }

    #[test]
    fn mtp_cost_aware_policy_keeps_a_cheaper_zero_draft_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES - 1 {
            policy.observe_window(policy_window(1, 2, 200));
        }
        let change = policy.observe_window(policy_window(1, 2, 200));

        assert_eq!(policy.current_budget(), 0);
        assert!(change.reduced);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(policy_window(0, 1, 50));
        }
        assert_eq!(policy.current_budget(), 0);
        assert!(!policy.should_maintain_mtp_cache());
    }

    #[test]
    fn mtp_cost_aware_policy_rejects_a_more_expensive_zero_draft_probe() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            policy.observe_window(policy_window(1, 2, 200));
        }
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS - 1 {
            policy.observe_window(policy_window(0, 1, 99));
        }
        let decision = policy.observe_window(policy_window(0, 1, 99));
        assert!(decision.increased);

        assert_eq!(policy.current_budget(), 1);
        assert!(policy.should_maintain_mtp_cache());
    }

    #[test]
    fn mtp_cost_aware_policy_refreshes_zero_draft_cost_before_switching() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            policy.observe_window(policy_window(1, 2, 200));
        }
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(policy_window(0, 1, 99));
        }
        assert_eq!(policy.current_budget(), 1);

        for _ in 0..16 {
            policy.observe_window(policy_window(1, 2, 300));
            assert_eq!(
                policy.current_budget(),
                1,
                "an old zero-draft sample must not trigger an uncontrolled mode switch"
            );
        }

        let change = policy.observe_window(policy_window(1, 2, 300));
        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS - 1 {
            policy.observe_window(policy_window(0, 1, 200));
        }
        let decision = policy.observe_window(policy_window(0, 1, 200));

        assert!(decision.increased);
        assert_eq!(policy.current_budget(), 1);
    }

    #[test]
    fn mtp_cost_aware_policy_reprobes_ordinary_decode_after_single_draft_rejections() {
        let mut policy = Gemma4DrafterPolicyState::new(1);
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            policy.observe_window(policy_window(1, 2, 200));
        }
        for _ in 0..Gemma4DrafterPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(policy_window(0, 1, 99));
        }
        assert_eq!(policy.current_budget(), 1);

        let mut rejected = policy_window(1, 1, 300);
        rejected.accepted_draft_tokens = 0;
        for _ in 0..16 {
            let change = policy.observe_window(rejected);
            assert!(!change.reduced);
            assert_eq!(policy.current_budget(), 1);
        }

        let change = policy.observe_window(rejected);
        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
    }

    #[test]
    fn qwen_mtp_policy_keeps_the_pre_gemma_zero_draft_decision_flow() {
        let mut policy = QwenMtpDraftPolicyState::new(1);
        for _ in 0..QwenMtpDraftPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            policy.observe_window(policy_window(1, 2, 200));
        }
        assert_eq!(policy.current_budget(), 0);
        assert!(policy.should_maintain_mtp_cache());

        for _ in 0..QwenMtpDraftPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(policy_window(0, 1, 80));
        }

        assert_eq!(policy.current_budget(), 0);
        assert!(policy.uses_ordinary_decode());
        assert!(!policy.should_maintain_mtp_cache());
    }

    #[test]
    fn qwen_mtp_policy_probes_ordinary_decode_early_after_32k() {
        let mut policy = QwenMtpDraftPolicyState::new(1);
        let mut long_window = policy_window(1, 2, 200);
        long_window.context_tokens = 32_769;

        for _ in 0..QwenMtpDraftPolicyState::LONG_CONTEXT_ZERO_DRAFT_MIN_COST_SAMPLES - 1 {
            let change = policy.observe_window(long_window);
            assert!(!change.reduced);
            assert_eq!(policy.current_budget(), 1);
        }
        let change = policy.observe_window(long_window);

        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
        assert!(policy.should_maintain_mtp_cache());
    }

    #[test]
    fn qwen_mtp_policy_compares_partial_long_context_window_with_ordinary_decode() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut long_window = policy_window(2, 2, 200);
        long_window.context_tokens = 32_769;
        long_window.accepted_draft_tokens = 1;

        let change = policy.observe_window(long_window);

        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
        assert_eq!(policy.probe_budget, Some(0));
        assert_eq!(policy.probe_origin_budget, Some(2));
    }

    #[test]
    fn qwen_mtp_policy_preserves_short_context_partial_acceptance_rule() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut short_window = policy_window(2, 2, 200);
        short_window.context_tokens = 8_192;
        short_window.accepted_draft_tokens = 1;

        let change = policy.observe_window(short_window);

        assert!(!change.reduced);
        assert_eq!(policy.current_budget(), 2);
    }

    #[test]
    fn qwen_mtp_policy_compares_full_long_context_window_with_ordinary_decode() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut long_window = policy_window(2, 3, 200);
        long_window.context_tokens = 32_769;

        let change = policy.observe_window(long_window);

        assert!(change.reduced);
        assert_eq!(policy.current_budget(), 0);
        assert_eq!(policy.probe_budget, Some(0));
        assert_eq!(policy.probe_origin_budget, Some(2));
    }

    #[test]
    fn qwen_mtp_policy_keeps_short_context_depth_after_one_full_window() {
        let mut policy = QwenMtpDraftPolicyState::new(2);
        let mut short_window = policy_window(2, 3, 200);
        short_window.context_tokens = 8_192;

        let change = policy.observe_window(short_window);

        assert!(!change.reduced);
        assert_eq!(policy.current_budget(), 2);
    }

    #[test]
    fn qwen_mtp_policy_keeps_mtp_when_ordinary_decode_is_not_cheaper() {
        let mut policy = QwenMtpDraftPolicyState::new(1);
        for _ in 0..QwenMtpDraftPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            policy.observe_window(policy_window(1, 2, 80));
        }
        assert_eq!(policy.current_budget(), 0);

        for _ in 0..QwenMtpDraftPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            policy.observe_window(policy_window(0, 1, 200));
        }

        assert_eq!(policy.current_budget(), 1);
        assert!(!policy.uses_ordinary_decode());
    }

    #[test]
    fn qwen_mtp_policy_snapshot_preserves_zero_draft_probe() {
        let mut source = QwenMtpDraftPolicyState::new(1);
        for _ in 0..QwenMtpDraftPolicyState::ZERO_DRAFT_MIN_COST_SAMPLES {
            source.observe_window(policy_window(1, 2, 200));
        }
        source.observe_window(policy_window(0, 1, 80));
        let snapshot = source.snapshot();

        let mut restored = QwenMtpDraftPolicyState::new(1);
        restored.restore_snapshot(snapshot.clone()).unwrap();

        assert_eq!(restored.snapshot(), snapshot);
        for _ in 1..QwenMtpDraftPolicyState::ZERO_DRAFT_PROBE_WINDOWS {
            let window = policy_window(0, 1, 80);
            source.observe_window(window);
            restored.observe_window(window);
        }
        assert_eq!(restored.snapshot(), source.snapshot());
        assert!(restored.uses_ordinary_decode());
    }
}
