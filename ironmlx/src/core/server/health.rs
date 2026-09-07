//! `/healthz` JSON endpoint (B1-p2.5 G3). Snapshot of scheduler /
//! memory / model state for monitoring + load balancer health probes.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;

use crate::core::cache::{ActiveKvOffloadHealth, ActiveKvOffloadSharedStats};
use crate::core::memory_budget::system_total_ram_bytes;
use crate::core::prompt_lookup::{
    PromptLookupConfig, PromptLookupSourceStats, PromptLookupStats, SHARED_PROMPT_LOOKUP_TTL_MS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HealthStatus {
    #[serde(rename = "healthy")]
    Healthy,
    #[serde(rename = "degraded")]
    Degraded,
    #[serde(rename = "down")]
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthDegradedReason {
    SchedulerQueueHigh,
    KvCacheNearSoftLimit,
    ActiveKvOffloadDegraded,
    PrefixStoreBackpressured,
    ProcessMemoryTelemetryDegraded,
    ProcessMemorySoft,
    ProcessMemoryHard,
    ProcessMemoryEmergency,
    ModelFailed,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub max_position_embeddings: i32,
}

#[derive(Debug, Serialize)]
pub struct SchedulerInfo {
    pub b_max: usize,
    pub b_active: usize,
    pub b_queued: usize,
    pub queue_max: usize,
    pub admit_count: u64,
    pub batch_count: u64,
    pub admission_queue_full_count: u64,
    pub memory_budget_exceeded_count: u64,
}

#[derive(Debug, Serialize)]
pub struct MemoryInfo {
    pub total_ram_bytes: usize,
    /// Literal macOS `Pages free` telemetry. This is observational only and
    /// does not determine health because it excludes reclaimable pages.
    pub free_ram_bytes: usize,
    /// Governor-aligned host headroom: free + inactive + the configured
    /// reclaimable fraction of active pages.
    pub available_ram_bytes: Option<usize>,
    pub kv_cache_active_bytes: usize,
    pub kv_cache_soft_limit_bytes: usize,
    pub kv_cache_logical_cap_tokens: usize,
    pub kv_cache_resident_cap_tokens: usize,
    pub kv_cache_budget_policy: String,
    pub mlx_total_bytes: Option<usize>,
    pub mlx_max_recommended_bytes: Option<usize>,
    pub mlx_active_bytes: usize,
    pub mlx_cache_bytes: usize,
    pub mlx_peak_bytes: usize,
    pub mlx_memory_limit_bytes: usize,
    pub process_governor: crate::core::process_memory::MemoryGovernorSnapshot,
    pub prefix_store: crate::core::cache::AsyncPrefixStoreStats,
    pub immutable_prefix_blocks: crate::core::server::scheduler_actor::ImmutablePrefixBlockHealth,
}

#[derive(Debug, Serialize)]
pub struct MtpHealthInfo {
    pub enabled: bool,
    pub requested_draft_tokens: Option<usize>,
    /// Runtime cap after applying cache and scheduler safety constraints.
    pub draft_tokens: Option<usize>,
    pub prefill_count: u64,
    pub step_count: u64,
    pub fallback_prefill_count: u64,
    pub drafted_tokens: u64,
    pub accepted_draft_tokens: u64,
    pub windows: u64,
    /// Windows that attempted at least two draft tokens.
    pub multi_token_windows: u64,
    pub exact_sampling_windows: u64,
    pub exact_acceptance_draws: u64,
    pub exact_residual_corrections: u64,
    pub exact_bonus_samples: u64,
    pub draft_forward_us: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub sampling_us: u64,
    pub draft_host_sync_count: u64,
    pub draft_host_sync_us: u64,
    pub verify_accept_host_sync_count: u64,
    pub verify_accept_host_sync_us: u64,
    pub main_rollback_us: u64,
    pub cache_commit_us: u64,
    pub prefill_cache_commit_us: u64,
    pub decode_cache_commit_us: u64,
    pub cache_restore_us: u64,
    pub sampled_exact_qualification: NeuralExactQualificationHealth,
}

#[derive(Debug, Serialize)]
pub struct DFlash2HealthInfo {
    pub enabled: bool,
    pub block_size: Option<usize>,
    pub draft_quantization_bits: Option<i32>,
    pub requests: u64,
    pub windows: u64,
    pub drafted_tokens: u64,
    pub accepted_draft_tokens: u64,
    pub rollback_count: u64,
    pub tensor_batch_windows: u64,
    pub tensor_batch_divergent_splits: u64,
    pub tensor_batch_groups_created: u64,
    pub tensor_batch_width_limit: usize,
    pub tensor_batch_max_width: usize,
    pub sampled_requests: u64,
    pub exact_sampling_windows: u64,
    pub exact_acceptance_draws: u64,
    pub exact_residual_corrections: u64,
    pub exact_bonus_samples: u64,
    pub sampling_us: u64,
    pub latest_generation_tps: f64,
    pub latest_acceptance_rate: f64,
    pub peak_memory_bytes: usize,
    pub prefix_cache_enabled: bool,
    pub prefix_cache_max_bytes: Option<usize>,
    pub prefix_cache_entries: usize,
    pub prefix_cache_bytes: usize,
    pub prefix_cache_hits: u64,
    pub prefix_cache_misses: u64,
    pub prefix_cache_saves: u64,
    pub prefix_cache_evictions: u64,
    pub prefix_cache_hit_tokens: u64,
    pub runtime_usage: crate::core::runtime_usage::ModelRuntimeUsageSnapshot,
}

#[derive(Clone)]
pub struct DFlash2HealthConfig {
    enabled: bool,
    block_size: Option<usize>,
    draft_quantization_bits: Option<i32>,
    requests: Arc<AtomicU64>,
    windows: Arc<AtomicU64>,
    drafted_tokens: Arc<AtomicU64>,
    accepted_draft_tokens: Arc<AtomicU64>,
    rollback_count: Arc<AtomicU64>,
    tensor_batch_windows: Arc<AtomicU64>,
    tensor_batch_divergent_splits: Arc<AtomicU64>,
    tensor_batch_groups_created: Arc<AtomicU64>,
    tensor_batch_width_limit: usize,
    tensor_batch_max_width: Arc<AtomicUsize>,
    sampled_requests: Arc<AtomicU64>,
    exact_sampling_windows: Arc<AtomicU64>,
    exact_acceptance_draws: Arc<AtomicU64>,
    exact_residual_corrections: Arc<AtomicU64>,
    exact_bonus_samples: Arc<AtomicU64>,
    sampling_us: Arc<AtomicU64>,
    latest_generation_tps_bits: Arc<AtomicU64>,
    latest_acceptance_rate_bits: Arc<AtomicU64>,
    peak_memory_bytes: Arc<AtomicUsize>,
    prefix_cache_enabled: bool,
    prefix_cache_max_bytes: Option<usize>,
    prefix_cache_entries: Arc<AtomicUsize>,
    prefix_cache_bytes: Arc<AtomicUsize>,
    prefix_cache_hits: Arc<AtomicU64>,
    prefix_cache_misses: Arc<AtomicU64>,
    prefix_cache_saves: Arc<AtomicU64>,
    prefix_cache_evictions: Arc<AtomicU64>,
    prefix_cache_hit_tokens: Arc<AtomicU64>,
    runtime_usage: Arc<crate::core::runtime_usage::ModelRuntimeUsageCounters>,
}

impl DFlash2HealthConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            block_size: None,
            draft_quantization_bits: None,
            requests: Arc::new(AtomicU64::new(0)),
            windows: Arc::new(AtomicU64::new(0)),
            drafted_tokens: Arc::new(AtomicU64::new(0)),
            accepted_draft_tokens: Arc::new(AtomicU64::new(0)),
            rollback_count: Arc::new(AtomicU64::new(0)),
            tensor_batch_windows: Arc::new(AtomicU64::new(0)),
            tensor_batch_divergent_splits: Arc::new(AtomicU64::new(0)),
            tensor_batch_groups_created: Arc::new(AtomicU64::new(0)),
            tensor_batch_width_limit: 0,
            tensor_batch_max_width: Arc::new(AtomicUsize::new(0)),
            sampled_requests: Arc::new(AtomicU64::new(0)),
            exact_sampling_windows: Arc::new(AtomicU64::new(0)),
            exact_acceptance_draws: Arc::new(AtomicU64::new(0)),
            exact_residual_corrections: Arc::new(AtomicU64::new(0)),
            exact_bonus_samples: Arc::new(AtomicU64::new(0)),
            sampling_us: Arc::new(AtomicU64::new(0)),
            latest_generation_tps_bits: Arc::new(AtomicU64::new(0_f64.to_bits())),
            latest_acceptance_rate_bits: Arc::new(AtomicU64::new(0_f64.to_bits())),
            peak_memory_bytes: Arc::new(AtomicUsize::new(0)),
            prefix_cache_enabled: false,
            prefix_cache_max_bytes: None,
            prefix_cache_entries: Arc::new(AtomicUsize::new(0)),
            prefix_cache_bytes: Arc::new(AtomicUsize::new(0)),
            prefix_cache_hits: Arc::new(AtomicU64::new(0)),
            prefix_cache_misses: Arc::new(AtomicU64::new(0)),
            prefix_cache_saves: Arc::new(AtomicU64::new(0)),
            prefix_cache_evictions: Arc::new(AtomicU64::new(0)),
            prefix_cache_hit_tokens: Arc::new(AtomicU64::new(0)),
            runtime_usage: Arc::new(
                crate::core::runtime_usage::ModelRuntimeUsageCounters::default(),
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enabled(
        block_size: usize,
        draft_quantization_bits: Option<i32>,
        requests: Arc<AtomicU64>,
        windows: Arc<AtomicU64>,
        drafted_tokens: Arc<AtomicU64>,
        accepted_draft_tokens: Arc<AtomicU64>,
        rollback_count: Arc<AtomicU64>,
        tensor_batch_windows: Arc<AtomicU64>,
        tensor_batch_divergent_splits: Arc<AtomicU64>,
        tensor_batch_groups_created: Arc<AtomicU64>,
        tensor_batch_width_limit: usize,
        tensor_batch_max_width: Arc<AtomicUsize>,
        sampled_requests: Arc<AtomicU64>,
        exact_sampling_windows: Arc<AtomicU64>,
        exact_acceptance_draws: Arc<AtomicU64>,
        exact_residual_corrections: Arc<AtomicU64>,
        exact_bonus_samples: Arc<AtomicU64>,
        sampling_us: Arc<AtomicU64>,
        latest_generation_tps_bits: Arc<AtomicU64>,
        latest_acceptance_rate_bits: Arc<AtomicU64>,
        peak_memory_bytes: Arc<AtomicUsize>,
        prefix_cache_enabled: bool,
        prefix_cache_max_bytes: Option<usize>,
        prefix_cache_entries: Arc<AtomicUsize>,
        prefix_cache_bytes: Arc<AtomicUsize>,
        prefix_cache_hits: Arc<AtomicU64>,
        prefix_cache_misses: Arc<AtomicU64>,
        prefix_cache_saves: Arc<AtomicU64>,
        prefix_cache_evictions: Arc<AtomicU64>,
        prefix_cache_hit_tokens: Arc<AtomicU64>,
        runtime_usage: Arc<crate::core::runtime_usage::ModelRuntimeUsageCounters>,
    ) -> Self {
        Self {
            enabled: true,
            block_size: Some(block_size),
            draft_quantization_bits,
            requests,
            windows,
            drafted_tokens,
            accepted_draft_tokens,
            rollback_count,
            tensor_batch_windows,
            tensor_batch_divergent_splits,
            tensor_batch_groups_created,
            tensor_batch_width_limit,
            tensor_batch_max_width,
            sampled_requests,
            exact_sampling_windows,
            exact_acceptance_draws,
            exact_residual_corrections,
            exact_bonus_samples,
            sampling_us,
            latest_generation_tps_bits,
            latest_acceptance_rate_bits,
            peak_memory_bytes,
            prefix_cache_enabled,
            prefix_cache_max_bytes,
            prefix_cache_entries,
            prefix_cache_bytes,
            prefix_cache_hits,
            prefix_cache_misses,
            prefix_cache_saves,
            prefix_cache_evictions,
            prefix_cache_hit_tokens,
            runtime_usage,
        }
    }

    pub(crate) fn snapshot(&self) -> DFlash2HealthInfo {
        DFlash2HealthInfo {
            enabled: self.enabled,
            block_size: self.block_size,
            draft_quantization_bits: self.draft_quantization_bits,
            requests: self.requests.load(Ordering::Relaxed),
            windows: self.windows.load(Ordering::Relaxed),
            drafted_tokens: self.drafted_tokens.load(Ordering::Relaxed),
            accepted_draft_tokens: self.accepted_draft_tokens.load(Ordering::Relaxed),
            rollback_count: self.rollback_count.load(Ordering::Relaxed),
            tensor_batch_windows: self.tensor_batch_windows.load(Ordering::Relaxed),
            tensor_batch_divergent_splits: self
                .tensor_batch_divergent_splits
                .load(Ordering::Relaxed),
            tensor_batch_groups_created: self.tensor_batch_groups_created.load(Ordering::Relaxed),
            tensor_batch_width_limit: self.tensor_batch_width_limit,
            tensor_batch_max_width: self.tensor_batch_max_width.load(Ordering::Relaxed),
            sampled_requests: self.sampled_requests.load(Ordering::Relaxed),
            exact_sampling_windows: self.exact_sampling_windows.load(Ordering::Relaxed),
            exact_acceptance_draws: self.exact_acceptance_draws.load(Ordering::Relaxed),
            exact_residual_corrections: self.exact_residual_corrections.load(Ordering::Relaxed),
            exact_bonus_samples: self.exact_bonus_samples.load(Ordering::Relaxed),
            sampling_us: self.sampling_us.load(Ordering::Relaxed),
            latest_generation_tps: f64::from_bits(
                self.latest_generation_tps_bits.load(Ordering::Relaxed),
            ),
            latest_acceptance_rate: f64::from_bits(
                self.latest_acceptance_rate_bits.load(Ordering::Relaxed),
            ),
            peak_memory_bytes: self.peak_memory_bytes.load(Ordering::Relaxed),
            prefix_cache_enabled: self.prefix_cache_enabled,
            prefix_cache_max_bytes: self.prefix_cache_max_bytes,
            prefix_cache_entries: self.prefix_cache_entries.load(Ordering::Relaxed),
            prefix_cache_bytes: self.prefix_cache_bytes.load(Ordering::Relaxed),
            prefix_cache_hits: self.prefix_cache_hits.load(Ordering::Relaxed),
            prefix_cache_misses: self.prefix_cache_misses.load(Ordering::Relaxed),
            prefix_cache_saves: self.prefix_cache_saves.load(Ordering::Relaxed),
            prefix_cache_evictions: self.prefix_cache_evictions.load(Ordering::Relaxed),
            prefix_cache_hit_tokens: self.prefix_cache_hit_tokens.load(Ordering::Relaxed),
            runtime_usage: self.runtime_usage.snapshot(self.prefix_cache_enabled),
        }
    }
}

#[derive(Debug, Default, Serialize)]
pub struct NeuralExactQualificationHealth {
    pub ordinary_cost_samples: u64,
    pub exact_cost_samples: u64,
    pub ordinary_cost_us: u64,
    pub exact_cost_us: u64,
    pub qualified_regimes_current: u64,
    pub rejected_regimes_current: u64,
    pub qualification_changes: u64,
    pub profile_loads: u64,
    pub profile_write_requests: u64,
    pub profile_writes: u64,
    pub profile_write_failures: u64,
    pub profile_write_coalesces: u64,
}

impl From<crate::core::speculative_qualification::NeuralExactQualificationStats>
    for NeuralExactQualificationHealth
{
    fn from(stats: crate::core::speculative_qualification::NeuralExactQualificationStats) -> Self {
        Self {
            ordinary_cost_samples: stats.ordinary_cost_samples,
            exact_cost_samples: stats.exact_cost_samples,
            ordinary_cost_us: stats.ordinary_cost_us,
            exact_cost_us: stats.exact_cost_us,
            qualified_regimes_current: stats.qualified_regimes_current,
            rejected_regimes_current: stats.rejected_regimes_current,
            qualification_changes: stats.qualification_changes,
            profile_loads: stats.profile_loads,
            profile_write_requests: stats.profile_write_requests,
            profile_writes: stats.profile_writes,
            profile_write_failures: stats.profile_write_failures,
            profile_write_coalesces: stats.profile_write_coalesces,
        }
    }
}

#[derive(Clone)]
pub struct MtpHealthConfig {
    enabled: bool,
    requested_draft_tokens: Option<usize>,
    draft_tokens: Option<usize>,
    prefill_count: Arc<AtomicU64>,
    step_count: Arc<AtomicU64>,
    fallback_prefill_count: Arc<AtomicU64>,
    drafted_tokens: Arc<AtomicU64>,
    accepted_draft_tokens: Arc<AtomicU64>,
    windows: Arc<AtomicU64>,
    multi_token_windows: Arc<AtomicU64>,
    exact_sampling_windows: Arc<AtomicU64>,
    exact_acceptance_draws: Arc<AtomicU64>,
    exact_residual_corrections: Arc<AtomicU64>,
    exact_bonus_samples: Arc<AtomicU64>,
    draft_forward_us: Arc<AtomicU64>,
    verify_forward_us: Arc<AtomicU64>,
    projection_us: Arc<AtomicU64>,
    sampling_us: Arc<AtomicU64>,
    draft_host_sync_count: Arc<AtomicU64>,
    draft_host_sync_us: Arc<AtomicU64>,
    verify_accept_host_sync_count: Arc<AtomicU64>,
    verify_accept_host_sync_us: Arc<AtomicU64>,
    main_rollback_us: Arc<AtomicU64>,
    cache_commit_us: Arc<AtomicU64>,
    prefill_cache_commit_us: Arc<AtomicU64>,
    decode_cache_commit_us: Arc<AtomicU64>,
    cache_restore_us: Arc<AtomicU64>,
    neural_exact_qualification_stats: Arc<
        std::sync::Mutex<crate::core::speculative_qualification::NeuralExactQualificationStats>,
    >,
}

impl MtpHealthConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            requested_draft_tokens: None,
            draft_tokens: None,
            prefill_count: Arc::new(AtomicU64::new(0)),
            step_count: Arc::new(AtomicU64::new(0)),
            fallback_prefill_count: Arc::new(AtomicU64::new(0)),
            drafted_tokens: Arc::new(AtomicU64::new(0)),
            accepted_draft_tokens: Arc::new(AtomicU64::new(0)),
            windows: Arc::new(AtomicU64::new(0)),
            multi_token_windows: Arc::new(AtomicU64::new(0)),
            exact_sampling_windows: Arc::new(AtomicU64::new(0)),
            exact_acceptance_draws: Arc::new(AtomicU64::new(0)),
            exact_residual_corrections: Arc::new(AtomicU64::new(0)),
            exact_bonus_samples: Arc::new(AtomicU64::new(0)),
            draft_forward_us: Arc::new(AtomicU64::new(0)),
            verify_forward_us: Arc::new(AtomicU64::new(0)),
            projection_us: Arc::new(AtomicU64::new(0)),
            sampling_us: Arc::new(AtomicU64::new(0)),
            draft_host_sync_count: Arc::new(AtomicU64::new(0)),
            draft_host_sync_us: Arc::new(AtomicU64::new(0)),
            verify_accept_host_sync_count: Arc::new(AtomicU64::new(0)),
            verify_accept_host_sync_us: Arc::new(AtomicU64::new(0)),
            main_rollback_us: Arc::new(AtomicU64::new(0)),
            cache_commit_us: Arc::new(AtomicU64::new(0)),
            prefill_cache_commit_us: Arc::new(AtomicU64::new(0)),
            decode_cache_commit_us: Arc::new(AtomicU64::new(0)),
            cache_restore_us: Arc::new(AtomicU64::new(0)),
            neural_exact_qualification_stats: Arc::new(std::sync::Mutex::new(
                crate::core::speculative_qualification::NeuralExactQualificationStats::default(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enabled(
        requested_draft_tokens: usize,
        draft_tokens: usize,
        prefill_count: Arc<AtomicU64>,
        step_count: Arc<AtomicU64>,
        fallback_prefill_count: Arc<AtomicU64>,
        drafted_tokens: Arc<AtomicU64>,
        accepted_draft_tokens: Arc<AtomicU64>,
        windows: Arc<AtomicU64>,
        multi_token_windows: Arc<AtomicU64>,
        exact_sampling_windows: Arc<AtomicU64>,
        exact_acceptance_draws: Arc<AtomicU64>,
        exact_residual_corrections: Arc<AtomicU64>,
        exact_bonus_samples: Arc<AtomicU64>,
        draft_forward_us: Arc<AtomicU64>,
        verify_forward_us: Arc<AtomicU64>,
        projection_us: Arc<AtomicU64>,
        sampling_us: Arc<AtomicU64>,
        draft_host_sync_count: Arc<AtomicU64>,
        draft_host_sync_us: Arc<AtomicU64>,
        verify_accept_host_sync_count: Arc<AtomicU64>,
        verify_accept_host_sync_us: Arc<AtomicU64>,
        main_rollback_us: Arc<AtomicU64>,
        cache_commit_us: Arc<AtomicU64>,
        prefill_cache_commit_us: Arc<AtomicU64>,
        decode_cache_commit_us: Arc<AtomicU64>,
        cache_restore_us: Arc<AtomicU64>,
        neural_exact_qualification_stats: Arc<
            std::sync::Mutex<crate::core::speculative_qualification::NeuralExactQualificationStats>,
        >,
    ) -> Self {
        Self {
            enabled: true,
            requested_draft_tokens: Some(requested_draft_tokens),
            draft_tokens: Some(draft_tokens),
            prefill_count,
            step_count,
            fallback_prefill_count,
            drafted_tokens,
            accepted_draft_tokens,
            windows,
            multi_token_windows,
            exact_sampling_windows,
            exact_acceptance_draws,
            exact_residual_corrections,
            exact_bonus_samples,
            draft_forward_us,
            verify_forward_us,
            projection_us,
            sampling_us,
            draft_host_sync_count,
            draft_host_sync_us,
            verify_accept_host_sync_count,
            verify_accept_host_sync_us,
            main_rollback_us,
            cache_commit_us,
            prefill_cache_commit_us,
            decode_cache_commit_us,
            cache_restore_us,
            neural_exact_qualification_stats,
        }
    }

    fn snapshot(&self) -> MtpHealthInfo {
        MtpHealthInfo {
            enabled: self.enabled,
            requested_draft_tokens: self.requested_draft_tokens,
            draft_tokens: self.draft_tokens,
            prefill_count: self.prefill_count.load(Ordering::Relaxed),
            step_count: self.step_count.load(Ordering::Relaxed),
            fallback_prefill_count: self.fallback_prefill_count.load(Ordering::Relaxed),
            drafted_tokens: self.drafted_tokens.load(Ordering::Relaxed),
            accepted_draft_tokens: self.accepted_draft_tokens.load(Ordering::Relaxed),
            windows: self.windows.load(Ordering::Relaxed),
            multi_token_windows: self.multi_token_windows.load(Ordering::Relaxed),
            exact_sampling_windows: self.exact_sampling_windows.load(Ordering::Relaxed),
            exact_acceptance_draws: self.exact_acceptance_draws.load(Ordering::Relaxed),
            exact_residual_corrections: self.exact_residual_corrections.load(Ordering::Relaxed),
            exact_bonus_samples: self.exact_bonus_samples.load(Ordering::Relaxed),
            draft_forward_us: self.draft_forward_us.load(Ordering::Relaxed),
            verify_forward_us: self.verify_forward_us.load(Ordering::Relaxed),
            projection_us: self.projection_us.load(Ordering::Relaxed),
            sampling_us: self.sampling_us.load(Ordering::Relaxed),
            draft_host_sync_count: self.draft_host_sync_count.load(Ordering::Relaxed),
            draft_host_sync_us: self.draft_host_sync_us.load(Ordering::Relaxed),
            verify_accept_host_sync_count: self
                .verify_accept_host_sync_count
                .load(Ordering::Relaxed),
            verify_accept_host_sync_us: self.verify_accept_host_sync_us.load(Ordering::Relaxed),
            main_rollback_us: self.main_rollback_us.load(Ordering::Relaxed),
            cache_commit_us: self.cache_commit_us.load(Ordering::Relaxed),
            prefill_cache_commit_us: self.prefill_cache_commit_us.load(Ordering::Relaxed),
            decode_cache_commit_us: self.decode_cache_commit_us.load(Ordering::Relaxed),
            cache_restore_us: self.cache_restore_us.load(Ordering::Relaxed),
            sampled_exact_qualification: (*self
                .neural_exact_qualification_stats
                .lock()
                .expect("neural exact qualification stats mutex poisoned"))
            .into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct PromptLookupSourceHealthInfo {
    pub queries: u64,
    pub hits: u64,
    pub misses: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub zero_accept_windows: u64,
    pub wasted_verify_tokens: u64,
    pub propose_us: u64,
    pub verify_us: u64,
    pub rollback_us: u64,
}

impl PromptLookupSourceHealthInfo {
    fn from_stats(stats: PromptLookupSourceStats) -> Self {
        Self {
            queries: stats.queries,
            hits: stats.hits,
            misses: stats.misses,
            drafted_tokens: stats.drafted_tokens,
            accepted_tokens: stats.accepted_tokens,
            zero_accept_windows: stats.zero_accept_windows,
            wasted_verify_tokens: stats.wasted_verify_tokens,
            propose_us: stats.propose_us,
            verify_us: stats.verify_us,
            rollback_us: stats.rollback_us,
        }
    }

    fn accumulate(&mut self, other: Self) {
        self.queries = self.queries.saturating_add(other.queries);
        self.hits = self.hits.saturating_add(other.hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.drafted_tokens = self.drafted_tokens.saturating_add(other.drafted_tokens);
        self.accepted_tokens = self.accepted_tokens.saturating_add(other.accepted_tokens);
        self.zero_accept_windows = self
            .zero_accept_windows
            .saturating_add(other.zero_accept_windows);
        self.wasted_verify_tokens = self
            .wasted_verify_tokens
            .saturating_add(other.wasted_verify_tokens);
        self.propose_us = self.propose_us.saturating_add(other.propose_us);
        self.verify_us = self.verify_us.saturating_add(other.verify_us);
        self.rollback_us = self.rollback_us.saturating_add(other.rollback_us);
    }
}

#[derive(Debug, Default, Serialize)]
pub struct PromptLookupHealthInfo {
    pub enabled: bool,
    pub min_ngram: Option<usize>,
    pub max_ngram: Option<usize>,
    pub max_draft_tokens: Option<usize>,
    pub history_window_tokens: Option<usize>,
    pub max_index_entries: Option<usize>,
    pub cross_request: Option<bool>,
    pub shared_ttl_ms: Option<u64>,
    pub queries: u64,
    pub hits: u64,
    pub misses: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub rejected_tokens: u64,
    pub zero_accept_windows: u64,
    pub exact_sampling_windows: u64,
    pub exact_acceptance_draws: u64,
    pub exact_residual_corrections: u64,
    pub exact_bonus_samples: u64,
    pub propose_us: u64,
    pub index_build_us: u64,
    pub index_update_us: u64,
    pub index_entries_current: u64,
    pub index_entries_peak: u64,
    pub index_ledger_entries_current: u64,
    pub index_ledger_entries_peak: u64,
    pub index_estimated_bytes_current: u64,
    pub index_estimated_bytes_peak: u64,
    pub index_evictions: u64,
    pub verify_round_us: u64,
    pub verify_forward_us: u64,
    pub projection_us: u64,
    pub exact_batched_verify_windows: u64,
    pub sequential_verify_windows: u64,
    pub verify_accept_host_sync_count: u64,
    pub verify_accept_host_sync_us: u64,
    pub rollback_count: u64,
    pub rollback_us: u64,
    pub mtp_shadow_commit_windows: u64,
    pub mtp_shadow_commit_tokens: u64,
    pub mtp_shadow_commit_us: u64,
    pub miss_fast_path_steps: u64,
    pub ordinary_cost_samples: u64,
    pub lookup_cost_samples: u64,
    pub ordinary_cost_us: u64,
    pub lookup_cost_us: u64,
    pub qualified_regimes_current: u64,
    pub rejected_regimes_current: u64,
    pub qualification_changes: u64,
    pub qualification_profile_loads: u64,
    pub qualification_profile_writes: u64,
    pub qualification_profile_write_drops: u64,
    pub qualification_query_gate_skips: u64,
    pub miss_query_gate_skips: u64,
    pub miss_query_reprobes: u64,
    pub adaptive_draft_width_reductions: u64,
    pub adaptive_draft_width_increases: u64,
    pub adaptive_profitability_width_reductions: u64,
    pub hybrid_neural_windows: u64,
    pub hybrid_lookup_windows: u64,
    pub hybrid_source_switches: u64,
    pub hybrid_lookup_miss_fallbacks: u64,
    pub hybrid_neural_rebases: u64,
    pub hybrid_neural_rebase_us: u64,
    pub local_source: PromptLookupSourceHealthInfo,
    pub shared_source: PromptLookupSourceHealthInfo,
    pub shared_queries: u64,
    pub shared_hits: u64,
    pub shared_misses: u64,
    pub shared_mtp_certified_published_windows: u64,
    pub shared_mtp_certified_published_tokens: u64,
    pub shared_mtp_certified_hits: u64,
    pub shared_mtp_canonical_validation_windows: u64,
    pub shared_mtp_canonical_validation_tokens: u64,
    pub shared_mtp_canonical_validation_us: u64,
    pub shared_mtp_canonical_validation_mismatches: u64,
    pub shared_mtp_canonical_fallbacks: u64,
    pub shared_published_requests: u64,
    pub shared_published_tokens: u64,
    pub shared_entries_current: u64,
    pub shared_entries_peak: u64,
    pub shared_evictions: u64,
    pub shared_pressure_evictions: u64,
    pub shared_clear_count: u64,
    pub shared_cleared_entries: u64,
    pub shared_estimated_bytes_current: u64,
    pub shared_estimated_bytes_peak: u64,
}

impl PromptLookupHealthInfo {
    pub fn aggregate(snapshots: impl IntoIterator<Item = Self>) -> Self {
        let mut aggregate = Self::default();
        let mut config: Option<(usize, usize, usize, usize, usize, bool)> = None;
        let mut config_mismatch = false;
        for snapshot in snapshots {
            aggregate.enabled |= snapshot.enabled;
            if snapshot.enabled {
                let current = snapshot
                    .min_ngram
                    .zip(snapshot.max_ngram)
                    .zip(snapshot.max_draft_tokens)
                    .zip(snapshot.history_window_tokens)
                    .zip(snapshot.max_index_entries)
                    .zip(snapshot.cross_request)
                    .map(
                        |(
                            ((((min_ngram, max_ngram), max_draft_tokens), history), entries),
                            cross_request,
                        )| {
                            (
                                min_ngram,
                                max_ngram,
                                max_draft_tokens,
                                history,
                                entries,
                                cross_request,
                            )
                        },
                    );
                match (config, current) {
                    (None, Some(current)) => config = Some(current),
                    (Some(expected), Some(current)) if expected == current => {}
                    _ => config_mismatch = true,
                }
            }
            aggregate.queries += snapshot.queries;
            aggregate.hits += snapshot.hits;
            aggregate.misses += snapshot.misses;
            aggregate.drafted_tokens += snapshot.drafted_tokens;
            aggregate.accepted_tokens += snapshot.accepted_tokens;
            aggregate.rejected_tokens += snapshot.rejected_tokens;
            aggregate.zero_accept_windows += snapshot.zero_accept_windows;
            aggregate.propose_us += snapshot.propose_us;
            aggregate.index_build_us += snapshot.index_build_us;
            aggregate.index_update_us += snapshot.index_update_us;
            aggregate.index_entries_current += snapshot.index_entries_current;
            aggregate.index_entries_peak += snapshot.index_entries_peak;
            aggregate.index_ledger_entries_current += snapshot.index_ledger_entries_current;
            aggregate.index_ledger_entries_peak += snapshot.index_ledger_entries_peak;
            aggregate.index_estimated_bytes_current += snapshot.index_estimated_bytes_current;
            aggregate.index_estimated_bytes_peak += snapshot.index_estimated_bytes_peak;
            aggregate.index_evictions += snapshot.index_evictions;
            aggregate.verify_round_us += snapshot.verify_round_us;
            aggregate.verify_forward_us += snapshot.verify_forward_us;
            aggregate.projection_us += snapshot.projection_us;
            aggregate.exact_batched_verify_windows += snapshot.exact_batched_verify_windows;
            aggregate.sequential_verify_windows += snapshot.sequential_verify_windows;
            aggregate.verify_accept_host_sync_count += snapshot.verify_accept_host_sync_count;
            aggregate.verify_accept_host_sync_us += snapshot.verify_accept_host_sync_us;
            aggregate.rollback_count += snapshot.rollback_count;
            aggregate.rollback_us += snapshot.rollback_us;
            aggregate.mtp_shadow_commit_windows += snapshot.mtp_shadow_commit_windows;
            aggregate.mtp_shadow_commit_tokens += snapshot.mtp_shadow_commit_tokens;
            aggregate.mtp_shadow_commit_us += snapshot.mtp_shadow_commit_us;
            aggregate.miss_fast_path_steps += snapshot.miss_fast_path_steps;
            aggregate.ordinary_cost_samples += snapshot.ordinary_cost_samples;
            aggregate.lookup_cost_samples += snapshot.lookup_cost_samples;
            aggregate.ordinary_cost_us += snapshot.ordinary_cost_us;
            aggregate.lookup_cost_us += snapshot.lookup_cost_us;
            aggregate.qualified_regimes_current += snapshot.qualified_regimes_current;
            aggregate.rejected_regimes_current += snapshot.rejected_regimes_current;
            aggregate.qualification_changes += snapshot.qualification_changes;
            aggregate.qualification_profile_loads += snapshot.qualification_profile_loads;
            aggregate.qualification_profile_writes += snapshot.qualification_profile_writes;
            aggregate.qualification_profile_write_drops +=
                snapshot.qualification_profile_write_drops;
            aggregate.qualification_query_gate_skips += snapshot.qualification_query_gate_skips;
            aggregate.miss_query_gate_skips += snapshot.miss_query_gate_skips;
            aggregate.miss_query_reprobes += snapshot.miss_query_reprobes;
            aggregate.adaptive_draft_width_reductions += snapshot.adaptive_draft_width_reductions;
            aggregate.adaptive_draft_width_increases += snapshot.adaptive_draft_width_increases;
            aggregate.adaptive_profitability_width_reductions +=
                snapshot.adaptive_profitability_width_reductions;
            aggregate.hybrid_neural_windows += snapshot.hybrid_neural_windows;
            aggregate.hybrid_lookup_windows += snapshot.hybrid_lookup_windows;
            aggregate.hybrid_source_switches += snapshot.hybrid_source_switches;
            aggregate.hybrid_lookup_miss_fallbacks += snapshot.hybrid_lookup_miss_fallbacks;
            aggregate.hybrid_neural_rebases += snapshot.hybrid_neural_rebases;
            aggregate.hybrid_neural_rebase_us += snapshot.hybrid_neural_rebase_us;
            aggregate.local_source.accumulate(snapshot.local_source);
            aggregate.shared_source.accumulate(snapshot.shared_source);
            aggregate.shared_queries += snapshot.shared_queries;
            aggregate.shared_hits += snapshot.shared_hits;
            aggregate.shared_misses += snapshot.shared_misses;
            aggregate.shared_mtp_certified_published_windows +=
                snapshot.shared_mtp_certified_published_windows;
            aggregate.shared_mtp_certified_published_tokens +=
                snapshot.shared_mtp_certified_published_tokens;
            aggregate.shared_mtp_certified_hits += snapshot.shared_mtp_certified_hits;
            aggregate.shared_mtp_canonical_validation_windows +=
                snapshot.shared_mtp_canonical_validation_windows;
            aggregate.shared_mtp_canonical_validation_tokens +=
                snapshot.shared_mtp_canonical_validation_tokens;
            aggregate.shared_mtp_canonical_validation_us +=
                snapshot.shared_mtp_canonical_validation_us;
            aggregate.shared_mtp_canonical_validation_mismatches +=
                snapshot.shared_mtp_canonical_validation_mismatches;
            aggregate.shared_mtp_canonical_fallbacks += snapshot.shared_mtp_canonical_fallbacks;
            aggregate.shared_published_requests += snapshot.shared_published_requests;
            aggregate.shared_published_tokens += snapshot.shared_published_tokens;
            aggregate.shared_entries_current += snapshot.shared_entries_current;
            aggregate.shared_entries_peak += snapshot.shared_entries_peak;
            aggregate.shared_evictions += snapshot.shared_evictions;
            aggregate.shared_pressure_evictions += snapshot.shared_pressure_evictions;
            aggregate.shared_clear_count += snapshot.shared_clear_count;
            aggregate.shared_cleared_entries += snapshot.shared_cleared_entries;
            aggregate.shared_estimated_bytes_current += snapshot.shared_estimated_bytes_current;
            aggregate.shared_estimated_bytes_peak += snapshot.shared_estimated_bytes_peak;
        }
        if !config_mismatch {
            if let Some((min_ngram, max_ngram, max_draft_tokens, history, entries, cross_request)) =
                config
            {
                aggregate.min_ngram = Some(min_ngram);
                aggregate.max_ngram = Some(max_ngram);
                aggregate.max_draft_tokens = Some(max_draft_tokens);
                aggregate.history_window_tokens = Some(history);
                aggregate.max_index_entries = Some(entries);
                aggregate.cross_request = Some(cross_request);
                aggregate.shared_ttl_ms = cross_request.then_some(SHARED_PROMPT_LOOKUP_TTL_MS);
            }
        }
        aggregate
    }
}

#[derive(Clone)]
pub struct PromptLookupHealthConfig {
    config: Option<PromptLookupConfig>,
    stats: Arc<Mutex<Option<PromptLookupStats>>>,
}

impl PromptLookupHealthConfig {
    pub fn disabled() -> Self {
        Self {
            config: None,
            stats: Arc::new(Mutex::new(None)),
        }
    }

    pub fn enabled(
        config: PromptLookupConfig,
        stats: Arc<Mutex<Option<PromptLookupStats>>>,
    ) -> Self {
        Self {
            config: Some(config),
            stats,
        }
    }

    fn snapshot(&self) -> PromptLookupHealthInfo {
        let stats = self
            .stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .unwrap_or_default();
        PromptLookupHealthInfo {
            enabled: self.config.is_some(),
            min_ngram: self.config.map(|config| config.min_ngram),
            max_ngram: self.config.map(|config| config.max_ngram),
            max_draft_tokens: self.config.map(|config| config.max_draft_tokens),
            history_window_tokens: self.config.map(|config| config.history_window_tokens),
            max_index_entries: self.config.map(|config| config.max_index_entries),
            cross_request: self.config.map(|config| config.cross_request),
            shared_ttl_ms: self
                .config
                .is_some_and(|config| config.cross_request)
                .then_some(SHARED_PROMPT_LOOKUP_TTL_MS),
            queries: stats.queries,
            hits: stats.hits,
            misses: stats.misses,
            drafted_tokens: stats.drafted_tokens,
            accepted_tokens: stats.accepted_tokens,
            rejected_tokens: stats.rejected_tokens,
            zero_accept_windows: stats.zero_accept_windows,
            exact_sampling_windows: stats.exact_sampling_windows,
            exact_acceptance_draws: stats.exact_acceptance_draws,
            exact_residual_corrections: stats.exact_residual_corrections,
            exact_bonus_samples: stats.exact_bonus_samples,
            propose_us: stats.propose_us,
            index_build_us: stats.index_build_us,
            index_update_us: stats.index_update_us,
            index_entries_current: stats.index_entries_current,
            index_entries_peak: stats.index_entries_peak,
            index_ledger_entries_current: stats.index_ledger_entries_current,
            index_ledger_entries_peak: stats.index_ledger_entries_peak,
            index_estimated_bytes_current: stats.index_estimated_bytes_current,
            index_estimated_bytes_peak: stats.index_estimated_bytes_peak,
            index_evictions: stats.index_evictions,
            verify_round_us: stats.verify_round_us,
            verify_forward_us: stats.verify_forward_us,
            projection_us: stats.projection_us,
            exact_batched_verify_windows: stats.exact_batched_verify_windows,
            sequential_verify_windows: stats.sequential_verify_windows,
            verify_accept_host_sync_count: stats.verify_accept_host_sync_count,
            verify_accept_host_sync_us: stats.verify_accept_host_sync_us,
            rollback_count: stats.rollback_count,
            rollback_us: stats.rollback_us,
            mtp_shadow_commit_windows: stats.mtp_shadow_commit_windows,
            mtp_shadow_commit_tokens: stats.mtp_shadow_commit_tokens,
            mtp_shadow_commit_us: stats.mtp_shadow_commit_us,
            miss_fast_path_steps: stats.miss_fast_path_steps,
            ordinary_cost_samples: stats.ordinary_cost_samples,
            lookup_cost_samples: stats.lookup_cost_samples,
            ordinary_cost_us: stats.ordinary_cost_us,
            lookup_cost_us: stats.lookup_cost_us,
            qualified_regimes_current: stats.qualified_regimes_current,
            rejected_regimes_current: stats.rejected_regimes_current,
            qualification_changes: stats.qualification_changes,
            qualification_profile_loads: stats.qualification_profile_loads,
            qualification_profile_writes: stats.qualification_profile_writes,
            qualification_profile_write_drops: stats.qualification_profile_write_drops,
            qualification_query_gate_skips: stats.qualification_query_gate_skips,
            miss_query_gate_skips: stats.miss_query_gate_skips,
            miss_query_reprobes: stats.miss_query_reprobes,
            adaptive_draft_width_reductions: stats.adaptive_draft_width_reductions,
            adaptive_draft_width_increases: stats.adaptive_draft_width_increases,
            adaptive_profitability_width_reductions: stats.adaptive_profitability_width_reductions,
            hybrid_neural_windows: stats.hybrid_neural_windows,
            hybrid_lookup_windows: stats.hybrid_lookup_windows,
            hybrid_source_switches: stats.hybrid_source_switches,
            hybrid_lookup_miss_fallbacks: stats.hybrid_lookup_miss_fallbacks,
            hybrid_neural_rebases: stats.hybrid_neural_rebases,
            hybrid_neural_rebase_us: stats.hybrid_neural_rebase_us,
            local_source: PromptLookupSourceHealthInfo::from_stats(stats.local_source),
            shared_source: PromptLookupSourceHealthInfo::from_stats(stats.shared_source),
            shared_queries: stats.shared_queries,
            shared_hits: stats.shared_hits,
            shared_misses: stats.shared_misses,
            shared_mtp_certified_published_windows: stats.shared_mtp_certified_published_windows,
            shared_mtp_certified_published_tokens: stats.shared_mtp_certified_published_tokens,
            shared_mtp_certified_hits: stats.shared_mtp_certified_hits,
            shared_mtp_canonical_validation_windows: stats.shared_mtp_canonical_validation_windows,
            shared_mtp_canonical_validation_tokens: stats.shared_mtp_canonical_validation_tokens,
            shared_mtp_canonical_validation_us: stats.shared_mtp_canonical_validation_us,
            shared_mtp_canonical_validation_mismatches: stats
                .shared_mtp_canonical_validation_mismatches,
            shared_mtp_canonical_fallbacks: stats.shared_mtp_canonical_fallbacks,
            shared_published_requests: stats.shared_published_requests,
            shared_published_tokens: stats.shared_published_tokens,
            shared_entries_current: stats.shared_entries_current,
            shared_entries_peak: stats.shared_entries_peak,
            shared_evictions: stats.shared_evictions,
            shared_pressure_evictions: stats.shared_pressure_evictions,
            shared_clear_count: stats.shared_clear_count,
            shared_cleared_entries: stats.shared_cleared_entries,
            shared_estimated_bytes_current: stats.shared_estimated_bytes_current,
            shared_estimated_bytes_peak: stats.shared_estimated_bytes_peak,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct HealthSnapshot {
    pub status: HealthStatus,
    pub degraded_reasons: Vec<HealthDegradedReason>,
    pub uptime_secs: u64,
    pub model: ModelInfo,
    pub scheduler: SchedulerInfo,
    pub memory: MemoryInfo,
    pub mtp: MtpHealthInfo,
    pub dflash2: DFlash2HealthInfo,
    pub prompt_lookup: PromptLookupHealthInfo,
    pub active_kv_offload: ActiveKvOffloadHealth,
    pub device_name: Option<String>,
    pub version: &'static str,
}

pub struct SchedulerHealthCollector {
    pub start_time: Instant,
    pub b_max: usize,
    pub queue_max: usize,
    pub model_name: String,
    pub max_position_embeddings: i32,
    pub b_active: Arc<AtomicU64>,
    pub b_queued: Arc<AtomicU64>,
    pub admit_count: Arc<AtomicU64>,
    pub batch_count: Arc<AtomicU64>,
    pub admission_queue_full_count: Arc<AtomicU64>,
    pub memory_budget_exceeded_count: Arc<AtomicU64>,
    pub kv_cache_active_bytes: Arc<AtomicUsize>,
    pub kv_cache_soft_limit_bytes: usize,
    pub kv_cache_logical_cap_tokens: usize,
    pub kv_cache_resident_cap_tokens: usize,
    pub kv_cache_budget_policy: String,
    pub mtp: MtpHealthConfig,
    pub dflash2: DFlash2HealthConfig,
    pub prompt_lookup: PromptLookupHealthConfig,
    pub active_kv_offload: ActiveKvOffloadSharedStats,
    pub immutable_prefix_blocks:
        crate::core::server::scheduler_actor::ImmutablePrefixBlockSharedStats,
}

impl SchedulerHealthCollector {
    pub fn snapshot(&self) -> HealthSnapshot {
        let uptime_secs = self.start_time.elapsed().as_secs();
        let total_ram_bytes = system_total_ram_bytes();
        let free_ram_bytes = system_free_ram_bytes();
        let b_active = self.b_active.load(Ordering::Relaxed) as usize;
        let b_queued = self.b_queued.load(Ordering::Relaxed) as usize;
        let admission_full = self.admission_queue_full_count.load(Ordering::Relaxed);
        let mb_exceeded = self.memory_budget_exceeded_count.load(Ordering::Relaxed);
        let kv_active = self.kv_cache_active_bytes.load(Ordering::Relaxed);
        let mlx_memory = mlx::memory::snapshot();
        let process_governor =
            crate::core::process_memory::global_process_memory_governor().sample_process();
        let prefix_store = crate::core::cache::process_async_prefix_store_queue().stats();

        let active_kv_offload = self.active_kv_offload.snapshot();
        let prefix_store_backpressured =
            crate::core::cache::process_async_prefix_store_queue().is_backpressured();
        let (status, degraded_reasons) = classify_status(
            b_queued,
            self.queue_max,
            kv_active,
            self.kv_cache_soft_limit_bytes,
            active_kv_offload.degraded,
            prefix_store_backpressured,
            &process_governor,
        );
        HealthSnapshot {
            status,
            degraded_reasons,
            uptime_secs,
            model: ModelInfo {
                name: self.model_name.clone(),
                max_position_embeddings: self.max_position_embeddings,
            },
            scheduler: SchedulerInfo {
                b_max: self.b_max,
                b_active,
                b_queued,
                queue_max: self.queue_max,
                admit_count: self.admit_count.load(Ordering::Relaxed),
                batch_count: self.batch_count.load(Ordering::Relaxed),
                admission_queue_full_count: admission_full,
                memory_budget_exceeded_count: mb_exceeded,
            },
            memory: MemoryInfo {
                total_ram_bytes,
                free_ram_bytes,
                available_ram_bytes: process_governor.available_ram_bytes,
                kv_cache_active_bytes: kv_active,
                kv_cache_soft_limit_bytes: self.kv_cache_soft_limit_bytes,
                kv_cache_logical_cap_tokens: self.kv_cache_logical_cap_tokens,
                kv_cache_resident_cap_tokens: self.kv_cache_resident_cap_tokens,
                kv_cache_budget_policy: self.kv_cache_budget_policy.clone(),
                mlx_total_bytes: mlx_memory.total_bytes,
                mlx_max_recommended_bytes: mlx_memory.max_recommended_bytes,
                mlx_active_bytes: mlx_memory.active_bytes,
                mlx_cache_bytes: mlx_memory.cache_bytes,
                mlx_peak_bytes: mlx_memory.peak_bytes,
                mlx_memory_limit_bytes: mlx_memory.memory_limit_bytes,
                process_governor,
                prefix_store,
                immutable_prefix_blocks: self.immutable_prefix_blocks.snapshot(),
            },
            mtp: self.mtp.snapshot(),
            dflash2: self.dflash2.snapshot(),
            prompt_lookup: self.prompt_lookup.snapshot(),
            active_kv_offload,
            device_name: mlx_memory.device_name,
            version: env!("CARGO_PKG_VERSION"),
        }
    }
}

pub fn classify_status(
    b_queued: usize,
    queue_max: usize,
    kv_cache_active_bytes: usize,
    kv_cache_soft_limit_bytes: usize,
    active_kv_offload_degraded: bool,
    prefix_store_backpressured: bool,
    process_governor: &crate::core::process_memory::MemoryGovernorSnapshot,
) -> (HealthStatus, Vec<HealthDegradedReason>) {
    let mut reasons = Vec::new();
    let queue_high = queue_max > 0 && b_queued >= queue_max / 2;
    let budget_near = kv_cache_soft_limit_bytes > 0
        && kv_cache_active_bytes >= ((kv_cache_soft_limit_bytes as f64) * 0.9) as usize;
    if queue_high {
        reasons.push(HealthDegradedReason::SchedulerQueueHigh);
    }
    if budget_near {
        reasons.push(HealthDegradedReason::KvCacheNearSoftLimit);
    }
    if active_kv_offload_degraded {
        reasons.push(HealthDegradedReason::ActiveKvOffloadDegraded);
    }
    if prefix_store_backpressured {
        reasons.push(HealthDegradedReason::PrefixStoreBackpressured);
    }
    classify_process_status(process_governor, reasons)
}

pub fn classify_process_status(
    process_governor: &crate::core::process_memory::MemoryGovernorSnapshot,
    mut reasons: Vec<HealthDegradedReason>,
) -> (HealthStatus, Vec<HealthDegradedReason>) {
    use crate::core::process_memory::PressureLevel;

    if process_governor.telemetry_degraded {
        reasons.push(HealthDegradedReason::ProcessMemoryTelemetryDegraded);
    }
    match process_governor.pressure_level {
        PressureLevel::Normal => {}
        PressureLevel::Soft if !process_governor.telemetry_degraded => {
            reasons.push(HealthDegradedReason::ProcessMemorySoft);
        }
        PressureLevel::Soft => {}
        PressureLevel::Hard => reasons.push(HealthDegradedReason::ProcessMemoryHard),
        PressureLevel::Emergency => reasons.push(HealthDegradedReason::ProcessMemoryEmergency),
    }
    let status = if process_governor.pressure_level == PressureLevel::Emergency {
        HealthStatus::Down
    } else if reasons.is_empty() {
        HealthStatus::Healthy
    } else {
        HealthStatus::Degraded
    };
    (status, reasons)
}

pub fn system_free_ram_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(bytes) = macos_free_ram_bytes() {
            return bytes;
        }
        tracing::warn!(
            "failed to query macOS free memory with /usr/bin/vm_stat; using 4 GiB fallback"
        );
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    if let Some(kb_str) = rest.trim().split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }
    4 * 1024 * 1024 * 1024
}

#[cfg(target_os = "macos")]
fn macos_free_ram_bytes() -> Option<usize> {
    let output = std::process::Command::new("/usr/bin/vm_stat")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let output = std::str::from_utf8(&output.stdout).ok()?;
    let mut page_size = 16_384_usize;
    let mut pages_free = None;
    for line in output.lines() {
        if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics: (page size of ") {
            page_size = rest.split(' ').next()?.parse::<usize>().ok()?;
        }
        if let Some(rest) = line.strip_prefix("Pages free:") {
            pages_free = rest.trim().trim_end_matches('.').parse::<usize>().ok();
        }
    }
    pages_free?.checked_mul(page_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    use std::sync::Arc;

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_free_ram_query_uses_fixed_system_path() {
        assert!(macos_free_ram_bytes().is_some());
    }

    #[test]
    fn classify_healthy_when_all_green() {
        let mut governor = crate::core::process_memory::MemoryGovernorSnapshot::default();
        governor.telemetry_degraded = false;
        let (status, reasons) =
            classify_status(0, 32, 1_000_000, 10_000_000, false, false, &governor);
        assert_eq!(status, HealthStatus::Healthy);
        assert!(reasons.is_empty());
    }

    #[test]
    fn classify_degraded_when_queue_half_full() {
        let mut governor = crate::core::process_memory::MemoryGovernorSnapshot::default();
        governor.telemetry_degraded = false;
        let (status, reasons) = classify_status(16, 32, 0, 10, false, false, &governor);
        assert_eq!(status, HealthStatus::Degraded);
        assert_eq!(reasons, vec![HealthDegradedReason::SchedulerQueueHigh]);
    }

    #[test]
    fn raw_free_ram_is_observational_only() {
        let mut governor = crate::core::process_memory::MemoryGovernorSnapshot::default();
        governor.telemetry_degraded = false;
        governor.available_ram_bytes = Some(8 * 1024 * 1024 * 1024);

        // A low raw free-page count is deliberately absent from the health
        // classifier. The governor's pressure level owns memory health.
        let raw_free_ram_bytes = 1;
        let (status, reasons) = classify_status(0, 32, 0, 10_000_000, false, false, &governor);

        assert!(raw_free_ram_bytes < 1024 * 1024 * 1024);
        assert_eq!(status, HealthStatus::Healthy);
        assert!(reasons.is_empty());
    }

    #[test]
    fn classify_degraded_when_budget_near_soft_limit() {
        let mut governor = crate::core::process_memory::MemoryGovernorSnapshot::default();
        governor.telemetry_degraded = false;
        let (status, reasons) =
            classify_status(0, 32, 9_500_000, 10_000_000, false, false, &governor);
        assert_eq!(status, HealthStatus::Degraded);
        assert_eq!(reasons, vec![HealthDegradedReason::KvCacheNearSoftLimit]);
    }

    #[test]
    fn classify_process_pressure_and_telemetry_reasons() {
        use crate::core::process_memory::PressureLevel;

        for (level, expected_status, expected_reason) in [
            (
                PressureLevel::Soft,
                HealthStatus::Degraded,
                HealthDegradedReason::ProcessMemorySoft,
            ),
            (
                PressureLevel::Hard,
                HealthStatus::Degraded,
                HealthDegradedReason::ProcessMemoryHard,
            ),
            (
                PressureLevel::Emergency,
                HealthStatus::Down,
                HealthDegradedReason::ProcessMemoryEmergency,
            ),
        ] {
            let mut governor = crate::core::process_memory::MemoryGovernorSnapshot::default();
            governor.telemetry_degraded = false;
            governor.pressure_level = level;
            let (status, reasons) = classify_process_status(&governor, Vec::new());
            assert_eq!(status, expected_status);
            assert_eq!(reasons, vec![expected_reason]);
        }

        let governor = crate::core::process_memory::MemoryGovernorSnapshot::default();
        let (status, reasons) = classify_process_status(&governor, Vec::new());
        assert_eq!(status, HealthStatus::Degraded);
        assert_eq!(
            reasons,
            vec![HealthDegradedReason::ProcessMemoryTelemetryDegraded]
        );
    }

    fn test_collector(mtp: MtpHealthConfig) -> SchedulerHealthCollector {
        test_collector_with_active_kv(
            mtp,
            ActiveKvOffloadSharedStats::new(&crate::core::cache::ActiveKvOffloadConfig::disabled()),
        )
    }

    fn test_collector_with_active_kv(
        mtp: MtpHealthConfig,
        active_kv_offload: ActiveKvOffloadSharedStats,
    ) -> SchedulerHealthCollector {
        SchedulerHealthCollector {
            start_time: Instant::now(),
            b_max: 1,
            queue_max: 8,
            model_name: "test-model".to_string(),
            max_position_embeddings: 4096,
            b_active: Arc::new(AtomicU64::new(0)),
            b_queued: Arc::new(AtomicU64::new(0)),
            admit_count: Arc::new(AtomicU64::new(0)),
            batch_count: Arc::new(AtomicU64::new(0)),
            admission_queue_full_count: Arc::new(AtomicU64::new(0)),
            memory_budget_exceeded_count: Arc::new(AtomicU64::new(0)),
            kv_cache_active_bytes: Arc::new(AtomicUsize::new(0)),
            kv_cache_soft_limit_bytes: 1,
            kv_cache_logical_cap_tokens: 262_144,
            kv_cache_resident_cap_tokens: 1_024,
            kv_cache_budget_policy: "active_kv_offload".to_string(),
            mtp,
            dflash2: DFlash2HealthConfig::disabled(),
            prompt_lookup: PromptLookupHealthConfig::disabled(),
            active_kv_offload,
            immutable_prefix_blocks:
                crate::core::server::scheduler_actor::ImmutablePrefixBlockSharedStats::new(false),
        }
    }

    #[test]
    fn snapshot_memory_reports_budget_policy_and_caps() {
        let collector = test_collector(MtpHealthConfig::disabled());
        collector.admit_count.store(3, Ordering::Relaxed);
        collector.batch_count.store(2, Ordering::Relaxed);
        let snapshot = collector.snapshot();

        assert_eq!(snapshot.memory.kv_cache_logical_cap_tokens, 262_144);
        assert_eq!(snapshot.memory.kv_cache_resident_cap_tokens, 1_024);
        assert_eq!(snapshot.memory.kv_cache_budget_policy, "active_kv_offload");
        assert_eq!(snapshot.scheduler.admit_count, 3);
        assert_eq!(snapshot.scheduler.batch_count, 2);
    }

    #[test]
    fn snapshot_mtp_reports_disabled_config() {
        let snapshot = test_collector(MtpHealthConfig::disabled()).snapshot();

        assert!(!snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.requested_draft_tokens, None);
        assert_eq!(snapshot.mtp.draft_tokens, None);
        assert_eq!(snapshot.mtp.prefill_count, 0);
        assert_eq!(snapshot.mtp.step_count, 0);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 0);
        assert_eq!(snapshot.mtp.drafted_tokens, 0);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 0);
        assert_eq!(snapshot.mtp.windows, 0);
        assert_eq!(snapshot.mtp.multi_token_windows, 0);
        assert_eq!(snapshot.mtp.draft_forward_us, 0);
        assert_eq!(snapshot.mtp.verify_forward_us, 0);
        assert_eq!(snapshot.mtp.projection_us, 0);
        assert_eq!(snapshot.mtp.sampling_us, 0);
        assert_eq!(snapshot.mtp.main_rollback_us, 0);
        assert_eq!(snapshot.mtp.cache_commit_us, 0);
        assert_eq!(snapshot.mtp.cache_restore_us, 0);
    }

    #[test]
    fn snapshot_dflash2_reports_config_and_live_metrics() {
        let requests = Arc::new(AtomicU64::new(3));
        let windows = Arc::new(AtomicU64::new(17));
        let drafted_tokens = Arc::new(AtomicU64::new(68));
        let accepted_draft_tokens = Arc::new(AtomicU64::new(51));
        let rollback_count = Arc::new(AtomicU64::new(4));
        let tensor_batch_windows = Arc::new(AtomicU64::new(11));
        let tensor_batch_divergent_splits = Arc::new(AtomicU64::new(3));
        let tensor_batch_groups_created = Arc::new(AtomicU64::new(5));
        let tensor_batch_max_width = Arc::new(AtomicUsize::new(4));
        let sampled_requests = Arc::new(AtomicU64::new(2));
        let exact_sampling_windows = Arc::new(AtomicU64::new(9));
        let exact_acceptance_draws = Arc::new(AtomicU64::new(21));
        let exact_residual_corrections = Arc::new(AtomicU64::new(5));
        let exact_bonus_samples = Arc::new(AtomicU64::new(4));
        let sampling_us = Arc::new(AtomicU64::new(12_345));
        let latest_generation_tps_bits = Arc::new(AtomicU64::new(48.5_f64.to_bits()));
        let latest_acceptance_rate_bits = Arc::new(AtomicU64::new(0.75_f64.to_bits()));
        let peak_memory_bytes = Arc::new(AtomicUsize::new(20_000_000_000));
        let prefix_cache_entries = Arc::new(AtomicUsize::new(2));
        let prefix_cache_bytes = Arc::new(AtomicUsize::new(1_500_000_000));
        let prefix_cache_hits = Arc::new(AtomicU64::new(4));
        let prefix_cache_misses = Arc::new(AtomicU64::new(1));
        let prefix_cache_saves = Arc::new(AtomicU64::new(3));
        let prefix_cache_evictions = Arc::new(AtomicU64::new(1));
        let prefix_cache_hit_tokens = Arc::new(AtomicU64::new(16_384));
        let runtime_usage =
            Arc::new(crate::core::runtime_usage::ModelRuntimeUsageCounters::default());
        runtime_usage.record_prefix_cache_lookup(700, 512);
        let mut completed = runtime_usage.start_request(768, Instant::now());
        std::thread::sleep(std::time::Duration::from_millis(1));
        completed.record_output_tokens(256);
        std::thread::sleep(std::time::Duration::from_millis(1));
        completed.complete();
        let mut active = runtime_usage.start_request(64, Instant::now());
        std::thread::sleep(std::time::Duration::from_millis(1));
        active.record_output_tokens(2);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let mut collector = test_collector(MtpHealthConfig::disabled());
        collector.dflash2 = DFlash2HealthConfig::enabled(
            4,
            Some(4),
            requests,
            windows,
            drafted_tokens,
            accepted_draft_tokens,
            rollback_count,
            tensor_batch_windows,
            tensor_batch_divergent_splits,
            tensor_batch_groups_created,
            6,
            tensor_batch_max_width,
            sampled_requests,
            exact_sampling_windows,
            exact_acceptance_draws,
            exact_residual_corrections,
            exact_bonus_samples,
            sampling_us,
            latest_generation_tps_bits,
            latest_acceptance_rate_bits,
            peak_memory_bytes,
            true,
            Some(8_589_934_592),
            prefix_cache_entries,
            prefix_cache_bytes,
            prefix_cache_hits,
            prefix_cache_misses,
            prefix_cache_saves,
            prefix_cache_evictions,
            prefix_cache_hit_tokens,
            runtime_usage,
        );

        let snapshot = collector.snapshot();

        assert!(snapshot.dflash2.enabled);
        assert_eq!(snapshot.dflash2.block_size, Some(4));
        assert_eq!(snapshot.dflash2.draft_quantization_bits, Some(4));
        assert_eq!(snapshot.dflash2.requests, 3);
        assert_eq!(snapshot.dflash2.windows, 17);
        assert_eq!(snapshot.dflash2.drafted_tokens, 68);
        assert_eq!(snapshot.dflash2.accepted_draft_tokens, 51);
        assert_eq!(snapshot.dflash2.rollback_count, 4);
        assert_eq!(snapshot.dflash2.tensor_batch_windows, 11);
        assert_eq!(snapshot.dflash2.tensor_batch_divergent_splits, 3);
        assert_eq!(snapshot.dflash2.tensor_batch_groups_created, 5);
        assert_eq!(snapshot.dflash2.tensor_batch_width_limit, 6);
        assert_eq!(snapshot.dflash2.tensor_batch_max_width, 4);
        assert_eq!(snapshot.dflash2.sampled_requests, 2);
        assert_eq!(snapshot.dflash2.exact_sampling_windows, 9);
        assert_eq!(snapshot.dflash2.exact_acceptance_draws, 21);
        assert_eq!(snapshot.dflash2.exact_residual_corrections, 5);
        assert_eq!(snapshot.dflash2.exact_bonus_samples, 4);
        assert_eq!(snapshot.dflash2.sampling_us, 12_345);
        assert_eq!(snapshot.dflash2.latest_generation_tps, 48.5);
        assert_eq!(snapshot.dflash2.latest_acceptance_rate, 0.75);
        assert_eq!(snapshot.dflash2.peak_memory_bytes, 20_000_000_000);
        assert!(snapshot.dflash2.prefix_cache_enabled);
        assert_eq!(snapshot.dflash2.prefix_cache_max_bytes, Some(8_589_934_592));
        assert_eq!(snapshot.dflash2.prefix_cache_entries, 2);
        assert_eq!(snapshot.dflash2.prefix_cache_bytes, 1_500_000_000);
        assert_eq!(snapshot.dflash2.prefix_cache_hits, 4);
        assert_eq!(snapshot.dflash2.prefix_cache_misses, 1);
        assert_eq!(snapshot.dflash2.prefix_cache_saves, 3);
        assert_eq!(snapshot.dflash2.prefix_cache_evictions, 1);
        assert_eq!(snapshot.dflash2.prefix_cache_hit_tokens, 16_384);
        assert_eq!(snapshot.dflash2.runtime_usage.input_tokens, 832);
        assert_eq!(snapshot.dflash2.runtime_usage.output_tokens, 258);
        assert_eq!(
            snapshot.dflash2.runtime_usage.prefix_cache,
            Some(crate::core::runtime_usage::PrefixCacheUsageSnapshot {
                hit_tokens: 512,
                eligible_tokens: 700,
            })
        );
        let performance = snapshot.dflash2.runtime_usage.performance;
        assert_eq!(performance.completed_requests, 1);
        assert!(performance.live_decode_tokens_per_second.is_some());
        assert!(performance.prefill_tokens_per_second.is_some());
        assert!(performance.decode_tokens_per_second.is_some());
        assert!(performance.session_decode_tokens_per_second.is_some());
        assert!(performance.ttft_ms.is_some());
    }

    #[test]
    fn snapshot_prompt_lookup_reports_config_and_live_stats() {
        let config = PromptLookupConfig {
            min_ngram: 2,
            max_ngram: 5,
            max_draft_tokens: 3,
            history_window_tokens: 4096,
            max_index_entries: 8192,
            cross_request: true,
        };
        let stats = PromptLookupStats {
            queries: 11,
            hits: 7,
            misses: 4,
            drafted_tokens: 19,
            accepted_tokens: 13,
            rejected_tokens: 6,
            exact_sampling_windows: 5,
            exact_acceptance_draws: 14,
            exact_residual_corrections: 3,
            exact_bonus_samples: 2,
            index_ledger_entries_current: 64,
            index_ledger_entries_peak: 96,
            index_estimated_bytes_current: 12_288,
            index_estimated_bytes_peak: 16_384,
            verify_round_us: 17,
            verify_accept_host_sync_count: 7,
            rollback_count: 2,
            miss_fast_path_steps: 3,
            ordinary_cost_samples: 8,
            lookup_cost_samples: 9,
            ordinary_cost_us: 10,
            lookup_cost_us: 11,
            exact_batched_verify_windows: 3,
            sequential_verify_windows: 4,
            qualified_regimes_current: 1,
            rejected_regimes_current: 2,
            qualification_changes: 4,
            qualification_profile_loads: 1,
            qualification_profile_writes: 5,
            qualification_profile_write_drops: 1,
            qualification_query_gate_skips: 6,
            miss_query_gate_skips: 7,
            miss_query_reprobes: 3,
            adaptive_draft_width_reductions: 5,
            adaptive_draft_width_increases: 4,
            adaptive_profitability_width_reductions: 2,
            hybrid_neural_windows: 12,
            hybrid_lookup_windows: 6,
            hybrid_source_switches: 4,
            hybrid_lookup_miss_fallbacks: 3,
            hybrid_neural_rebases: 2,
            hybrid_neural_rebase_us: 29,
            local_source: PromptLookupSourceStats {
                queries: 10,
                hits: 6,
                misses: 4,
                drafted_tokens: 12,
                accepted_tokens: 9,
                zero_accept_windows: 1,
                wasted_verify_tokens: 3,
                propose_us: 31,
                verify_us: 37,
                rollback_us: 2,
            },
            shared_source: PromptLookupSourceStats {
                queries: 9,
                hits: 6,
                misses: 3,
                drafted_tokens: 7,
                accepted_tokens: 4,
                zero_accept_windows: 2,
                wasted_verify_tokens: 3,
                propose_us: 41,
                verify_us: 43,
                rollback_us: 5,
            },
            shared_queries: 9,
            shared_hits: 6,
            shared_misses: 3,
            shared_published_requests: 4,
            shared_published_tokens: 128,
            shared_entries_current: 32,
            shared_entries_peak: 48,
            shared_evictions: 7,
            shared_pressure_evictions: 5,
            shared_clear_count: 2,
            shared_cleared_entries: 11,
            shared_estimated_bytes_current: 4096,
            shared_estimated_bytes_peak: 8192,
            ..PromptLookupStats::default()
        };
        let published = Arc::new(Mutex::new(Some(stats)));
        let mut collector = test_collector(MtpHealthConfig::disabled());
        collector.prompt_lookup = PromptLookupHealthConfig::enabled(config, published);

        let snapshot = collector.snapshot();

        assert!(snapshot.prompt_lookup.enabled);
        assert_eq!(snapshot.prompt_lookup.min_ngram, Some(2));
        assert_eq!(snapshot.prompt_lookup.max_ngram, Some(5));
        assert_eq!(snapshot.prompt_lookup.max_draft_tokens, Some(3));
        assert_eq!(snapshot.prompt_lookup.cross_request, Some(true));
        assert_eq!(
            snapshot.prompt_lookup.shared_ttl_ms,
            Some(SHARED_PROMPT_LOOKUP_TTL_MS)
        );
        assert_eq!(snapshot.prompt_lookup.queries, 11);
        assert_eq!(snapshot.prompt_lookup.hits, 7);
        assert_eq!(snapshot.prompt_lookup.misses, 4);
        assert_eq!(snapshot.prompt_lookup.drafted_tokens, 19);
        assert_eq!(snapshot.prompt_lookup.accepted_tokens, 13);
        assert_eq!(snapshot.prompt_lookup.rejected_tokens, 6);
        assert_eq!(snapshot.prompt_lookup.exact_sampling_windows, 5);
        assert_eq!(snapshot.prompt_lookup.exact_acceptance_draws, 14);
        assert_eq!(snapshot.prompt_lookup.exact_residual_corrections, 3);
        assert_eq!(snapshot.prompt_lookup.exact_bonus_samples, 2);
        assert_eq!(snapshot.prompt_lookup.verify_round_us, 17);
        assert_eq!(snapshot.prompt_lookup.verify_accept_host_sync_count, 7);
        assert_eq!(snapshot.prompt_lookup.rollback_count, 2);
        assert_eq!(snapshot.prompt_lookup.miss_fast_path_steps, 3);
        assert_eq!(snapshot.prompt_lookup.ordinary_cost_samples, 8);
        assert_eq!(snapshot.prompt_lookup.lookup_cost_samples, 9);
        assert_eq!(snapshot.prompt_lookup.ordinary_cost_us, 10);
        assert_eq!(snapshot.prompt_lookup.lookup_cost_us, 11);
        assert_eq!(snapshot.prompt_lookup.exact_batched_verify_windows, 3);
        assert_eq!(snapshot.prompt_lookup.sequential_verify_windows, 4);
        assert_eq!(snapshot.prompt_lookup.qualified_regimes_current, 1);
        assert_eq!(snapshot.prompt_lookup.rejected_regimes_current, 2);
        assert_eq!(snapshot.prompt_lookup.qualification_changes, 4);
        assert_eq!(snapshot.prompt_lookup.qualification_profile_loads, 1);
        assert_eq!(snapshot.prompt_lookup.qualification_profile_writes, 5);
        assert_eq!(snapshot.prompt_lookup.qualification_profile_write_drops, 1);
        assert_eq!(snapshot.prompt_lookup.qualification_query_gate_skips, 6);
        assert_eq!(snapshot.prompt_lookup.miss_query_gate_skips, 7);
        assert_eq!(snapshot.prompt_lookup.miss_query_reprobes, 3);
        assert_eq!(snapshot.prompt_lookup.adaptive_draft_width_reductions, 5);
        assert_eq!(snapshot.prompt_lookup.adaptive_draft_width_increases, 4);
        assert_eq!(
            snapshot
                .prompt_lookup
                .adaptive_profitability_width_reductions,
            2
        );
        assert_eq!(snapshot.prompt_lookup.hybrid_neural_windows, 12);
        assert_eq!(snapshot.prompt_lookup.hybrid_lookup_windows, 6);
        assert_eq!(snapshot.prompt_lookup.hybrid_source_switches, 4);
        assert_eq!(snapshot.prompt_lookup.hybrid_lookup_miss_fallbacks, 3);
        assert_eq!(snapshot.prompt_lookup.hybrid_neural_rebases, 2);
        assert_eq!(snapshot.prompt_lookup.hybrid_neural_rebase_us, 29);
        assert_eq!(snapshot.prompt_lookup.local_source.queries, 10);
        assert_eq!(snapshot.prompt_lookup.local_source.hits, 6);
        assert_eq!(snapshot.prompt_lookup.local_source.misses, 4);
        assert_eq!(snapshot.prompt_lookup.local_source.drafted_tokens, 12);
        assert_eq!(snapshot.prompt_lookup.local_source.accepted_tokens, 9);
        assert_eq!(snapshot.prompt_lookup.local_source.zero_accept_windows, 1);
        assert_eq!(snapshot.prompt_lookup.local_source.wasted_verify_tokens, 3);
        assert_eq!(snapshot.prompt_lookup.local_source.propose_us, 31);
        assert_eq!(snapshot.prompt_lookup.local_source.verify_us, 37);
        assert_eq!(snapshot.prompt_lookup.local_source.rollback_us, 2);
        assert_eq!(snapshot.prompt_lookup.shared_source.queries, 9);
        assert_eq!(snapshot.prompt_lookup.shared_source.hits, 6);
        assert_eq!(snapshot.prompt_lookup.shared_source.misses, 3);
        assert_eq!(snapshot.prompt_lookup.shared_source.drafted_tokens, 7);
        assert_eq!(snapshot.prompt_lookup.shared_source.accepted_tokens, 4);
        assert_eq!(snapshot.prompt_lookup.shared_source.zero_accept_windows, 2);
        assert_eq!(snapshot.prompt_lookup.shared_source.wasted_verify_tokens, 3);
        assert_eq!(snapshot.prompt_lookup.shared_source.propose_us, 41);
        assert_eq!(snapshot.prompt_lookup.shared_source.verify_us, 43);
        assert_eq!(snapshot.prompt_lookup.shared_source.rollback_us, 5);
        assert_eq!(snapshot.prompt_lookup.shared_queries, 9);
        assert_eq!(snapshot.prompt_lookup.shared_hits, 6);
        assert_eq!(snapshot.prompt_lookup.shared_misses, 3);
        assert_eq!(snapshot.prompt_lookup.shared_published_requests, 4);
        assert_eq!(snapshot.prompt_lookup.shared_published_tokens, 128);
        assert_eq!(snapshot.prompt_lookup.shared_entries_current, 32);
        assert_eq!(snapshot.prompt_lookup.shared_entries_peak, 48);
        assert_eq!(snapshot.prompt_lookup.shared_evictions, 7);
        assert_eq!(snapshot.prompt_lookup.shared_pressure_evictions, 5);
        assert_eq!(snapshot.prompt_lookup.shared_clear_count, 2);
        assert_eq!(snapshot.prompt_lookup.shared_cleared_entries, 11);
        assert_eq!(snapshot.prompt_lookup.index_ledger_entries_current, 64);
        assert_eq!(snapshot.prompt_lookup.index_ledger_entries_peak, 96);
        assert_eq!(snapshot.prompt_lookup.index_estimated_bytes_current, 12_288);
        assert_eq!(snapshot.prompt_lookup.index_estimated_bytes_peak, 16_384);
        assert_eq!(snapshot.prompt_lookup.shared_estimated_bytes_current, 4096);
        assert_eq!(snapshot.prompt_lookup.shared_estimated_bytes_peak, 8192);
    }

    #[test]
    fn snapshot_mtp_reports_enabled_config_and_live_counters() {
        let prefill_count = Arc::new(AtomicU64::new(7));
        let step_count = Arc::new(AtomicU64::new(11));
        let fallback_prefill_count = Arc::new(AtomicU64::new(13));
        let drafted_tokens = Arc::new(AtomicU64::new(17));
        let accepted_draft_tokens = Arc::new(AtomicU64::new(19));
        let windows = Arc::new(AtomicU64::new(23));
        let multi_token_windows = Arc::new(AtomicU64::new(17));
        let exact_sampling_windows = Arc::new(AtomicU64::new(5));
        let exact_acceptance_draws = Arc::new(AtomicU64::new(12));
        let exact_residual_corrections = Arc::new(AtomicU64::new(3));
        let exact_bonus_samples = Arc::new(AtomicU64::new(2));
        let draft_forward_us = Arc::new(AtomicU64::new(29));
        let verify_forward_us = Arc::new(AtomicU64::new(31));
        let projection_us = Arc::new(AtomicU64::new(37));
        let sampling_us = Arc::new(AtomicU64::new(41));
        let draft_host_sync_count = Arc::new(AtomicU64::new(0));
        let draft_host_sync_us = Arc::new(AtomicU64::new(0));
        let verify_accept_host_sync_count = Arc::new(AtomicU64::new(23));
        let verify_accept_host_sync_us = Arc::new(AtomicU64::new(42));
        let main_rollback_us = Arc::new(AtomicU64::new(43));
        let cache_commit_us = Arc::new(AtomicU64::new(47));
        let prefill_cache_commit_us = Arc::new(AtomicU64::new(19));
        let decode_cache_commit_us = Arc::new(AtomicU64::new(28));
        let cache_restore_us = Arc::new(AtomicU64::new(53));
        let neural_exact_qualification_stats = Arc::new(std::sync::Mutex::new(
            crate::core::speculative_qualification::NeuralExactQualificationStats {
                ordinary_cost_samples: 8,
                exact_cost_samples: 5,
                rejected_regimes_current: 1,
                qualification_changes: 1,
                ..Default::default()
            },
        ));
        let snapshot = test_collector(MtpHealthConfig::enabled(
            2,
            1,
            prefill_count.clone(),
            step_count.clone(),
            fallback_prefill_count.clone(),
            drafted_tokens.clone(),
            accepted_draft_tokens.clone(),
            windows.clone(),
            multi_token_windows.clone(),
            exact_sampling_windows.clone(),
            exact_acceptance_draws.clone(),
            exact_residual_corrections.clone(),
            exact_bonus_samples.clone(),
            draft_forward_us.clone(),
            verify_forward_us.clone(),
            projection_us.clone(),
            sampling_us.clone(),
            draft_host_sync_count.clone(),
            draft_host_sync_us.clone(),
            verify_accept_host_sync_count.clone(),
            verify_accept_host_sync_us.clone(),
            main_rollback_us.clone(),
            cache_commit_us.clone(),
            prefill_cache_commit_us.clone(),
            decode_cache_commit_us.clone(),
            cache_restore_us.clone(),
            neural_exact_qualification_stats.clone(),
        ))
        .snapshot();

        assert_eq!(snapshot.mtp.requested_draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.draft_tokens, Some(1));

        assert!(snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.draft_tokens, Some(1));
        assert_eq!(snapshot.mtp.prefill_count, 7);
        assert_eq!(snapshot.mtp.step_count, 11);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 13);
        assert_eq!(snapshot.mtp.drafted_tokens, 17);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 19);
        assert_eq!(snapshot.mtp.windows, 23);
        assert_eq!(snapshot.mtp.multi_token_windows, 17);
        assert_eq!(snapshot.mtp.draft_forward_us, 29);
        assert_eq!(snapshot.mtp.verify_forward_us, 31);
        assert_eq!(snapshot.mtp.projection_us, 37);
        assert_eq!(snapshot.mtp.sampling_us, 41);
        assert_eq!(snapshot.mtp.draft_host_sync_count, 0);
        assert_eq!(snapshot.mtp.draft_host_sync_us, 0);
        assert_eq!(snapshot.mtp.verify_accept_host_sync_count, 23);
        assert_eq!(snapshot.mtp.verify_accept_host_sync_us, 42);
        assert_eq!(snapshot.mtp.main_rollback_us, 43);
        assert_eq!(snapshot.mtp.cache_commit_us, 47);
        assert_eq!(snapshot.mtp.prefill_cache_commit_us, 19);
        assert_eq!(snapshot.mtp.decode_cache_commit_us, 28);
        assert_eq!(snapshot.mtp.cache_restore_us, 53);
        assert_eq!(
            snapshot
                .mtp
                .sampled_exact_qualification
                .ordinary_cost_samples,
            8
        );
        assert_eq!(
            snapshot.mtp.sampled_exact_qualification.exact_cost_samples,
            5
        );
        assert_eq!(
            snapshot
                .mtp
                .sampled_exact_qualification
                .rejected_regimes_current,
            1
        );

        prefill_count.store(13, Ordering::Relaxed);
        step_count.store(17, Ordering::Relaxed);
        fallback_prefill_count.store(23, Ordering::Relaxed);
        drafted_tokens.store(29, Ordering::Relaxed);
        accepted_draft_tokens.store(31, Ordering::Relaxed);
        windows.store(37, Ordering::Relaxed);
        multi_token_windows.store(31, Ordering::Relaxed);
        exact_sampling_windows.store(7, Ordering::Relaxed);
        exact_acceptance_draws.store(18, Ordering::Relaxed);
        exact_residual_corrections.store(4, Ordering::Relaxed);
        exact_bonus_samples.store(3, Ordering::Relaxed);
        draft_forward_us.store(41, Ordering::Relaxed);
        verify_forward_us.store(43, Ordering::Relaxed);
        projection_us.store(47, Ordering::Relaxed);
        sampling_us.store(53, Ordering::Relaxed);
        verify_accept_host_sync_count.store(37, Ordering::Relaxed);
        verify_accept_host_sync_us.store(61, Ordering::Relaxed);
        main_rollback_us.store(59, Ordering::Relaxed);
        cache_commit_us.store(61, Ordering::Relaxed);
        prefill_cache_commit_us.store(29, Ordering::Relaxed);
        decode_cache_commit_us.store(32, Ordering::Relaxed);
        cache_restore_us.store(67, Ordering::Relaxed);
        let snapshot = test_collector(MtpHealthConfig::enabled(
            2,
            2,
            prefill_count,
            step_count,
            fallback_prefill_count,
            drafted_tokens,
            accepted_draft_tokens,
            windows,
            multi_token_windows,
            exact_sampling_windows,
            exact_acceptance_draws,
            exact_residual_corrections,
            exact_bonus_samples,
            draft_forward_us,
            verify_forward_us,
            projection_us,
            sampling_us,
            draft_host_sync_count,
            draft_host_sync_us,
            verify_accept_host_sync_count,
            verify_accept_host_sync_us,
            main_rollback_us,
            cache_commit_us,
            prefill_cache_commit_us,
            decode_cache_commit_us,
            cache_restore_us,
            neural_exact_qualification_stats,
        ))
        .snapshot();

        assert_eq!(snapshot.mtp.requested_draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.prefill_count, 13);
        assert_eq!(snapshot.mtp.step_count, 17);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 23);
        assert_eq!(snapshot.mtp.drafted_tokens, 29);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 31);
        assert_eq!(snapshot.mtp.windows, 37);
        assert_eq!(snapshot.mtp.multi_token_windows, 31);
        assert_eq!(snapshot.mtp.exact_sampling_windows, 7);
        assert_eq!(snapshot.mtp.exact_acceptance_draws, 18);
        assert_eq!(snapshot.mtp.exact_residual_corrections, 4);
        assert_eq!(snapshot.mtp.exact_bonus_samples, 3);
        assert_eq!(snapshot.mtp.draft_forward_us, 41);
        assert_eq!(snapshot.mtp.verify_forward_us, 43);
        assert_eq!(snapshot.mtp.projection_us, 47);
        assert_eq!(snapshot.mtp.sampling_us, 53);
        assert_eq!(snapshot.mtp.main_rollback_us, 59);
        assert_eq!(snapshot.mtp.cache_commit_us, 61);
        assert_eq!(snapshot.mtp.prefill_cache_commit_us, 29);
        assert_eq!(snapshot.mtp.decode_cache_commit_us, 32);
        assert_eq!(snapshot.mtp.cache_restore_us, 67);
    }

    #[test]
    fn snapshot_degraded_when_active_kv_reports_error() {
        let active_kv_offload = ActiveKvOffloadSharedStats::new(
            &crate::core::cache::ActiveKvOffloadConfig::enabled(std::env::temp_dir()),
        );
        active_kv_offload.record_error();

        let snapshot =
            test_collector_with_active_kv(MtpHealthConfig::disabled(), active_kv_offload)
                .snapshot();

        assert!(matches!(snapshot.status, HealthStatus::Degraded));
        assert!(snapshot.active_kv_offload.degraded);
    }

    #[test]
    fn health_memory_serializes_mlx_allocator_fields() {
        let snapshot = HealthSnapshot {
            status: HealthStatus::Healthy,
            degraded_reasons: Vec::new(),
            uptime_secs: 7,
            model: ModelInfo {
                name: "test-model".to_string(),
                max_position_embeddings: 4096,
            },
            scheduler: SchedulerInfo {
                b_max: 8,
                b_active: 1,
                b_queued: 0,
                queue_max: 16,
                admit_count: 1,
                batch_count: 1,
                admission_queue_full_count: 0,
                memory_budget_exceeded_count: 0,
            },
            memory: MemoryInfo {
                total_ram_bytes: 64,
                free_ram_bytes: 32,
                available_ram_bytes: Some(48),
                kv_cache_active_bytes: 16,
                kv_cache_soft_limit_bytes: 24,
                kv_cache_logical_cap_tokens: 128,
                kv_cache_resident_cap_tokens: 64,
                kv_cache_budget_policy: "full_resident".to_string(),
                mlx_total_bytes: Some(55),
                mlx_max_recommended_bytes: Some(66),
                mlx_active_bytes: 11,
                mlx_cache_bytes: 22,
                mlx_peak_bytes: 33,
                mlx_memory_limit_bytes: 44,
                process_governor: crate::core::process_memory::MemoryGovernorSnapshot::default(),
                prefix_store: crate::core::cache::AsyncPrefixStoreStats::default(),
                immutable_prefix_blocks:
                    crate::core::server::scheduler_actor::ImmutablePrefixBlockHealth::default(),
            },
            mtp: MtpHealthInfo {
                enabled: false,
                requested_draft_tokens: None,
                draft_tokens: None,
                prefill_count: 0,
                step_count: 0,
                fallback_prefill_count: 0,
                drafted_tokens: 0,
                accepted_draft_tokens: 0,
                windows: 0,
                multi_token_windows: 0,
                exact_sampling_windows: 0,
                exact_acceptance_draws: 0,
                exact_residual_corrections: 0,
                exact_bonus_samples: 0,
                draft_forward_us: 0,
                verify_forward_us: 0,
                projection_us: 0,
                sampling_us: 0,
                draft_host_sync_count: 0,
                draft_host_sync_us: 0,
                verify_accept_host_sync_count: 0,
                verify_accept_host_sync_us: 0,
                main_rollback_us: 0,
                cache_commit_us: 0,
                prefill_cache_commit_us: 0,
                decode_cache_commit_us: 0,
                cache_restore_us: 0,
                sampled_exact_qualification: NeuralExactQualificationHealth::default(),
            },
            dflash2: DFlash2HealthConfig::disabled().snapshot(),
            prompt_lookup: PromptLookupHealthInfo::default(),
            active_kv_offload: ActiveKvOffloadHealth::disabled(),
            device_name: Some("Apple Test GPU".to_string()),
            version: "test",
        };

        let value = serde_json::to_value(snapshot).expect("serialize health snapshot");
        assert_eq!(value["memory"]["mlx_total_bytes"], 55);
        assert_eq!(value["memory"]["available_ram_bytes"], 48);
        assert_eq!(value["mtp"]["multi_token_windows"], 0);
        assert_eq!(value["memory"]["mlx_max_recommended_bytes"], 66);
        assert_eq!(value["memory"]["mlx_active_bytes"], 11);
        assert_eq!(value["memory"]["mlx_cache_bytes"], 22);
        assert_eq!(value["memory"]["mlx_peak_bytes"], 33);
        assert_eq!(value["memory"]["mlx_memory_limit_bytes"], 44);
        assert_eq!(value["memory"]["kv_cache_logical_cap_tokens"], 128);
        assert_eq!(value["memory"]["kv_cache_resident_cap_tokens"], 64);
        assert_eq!(value["memory"]["kv_cache_budget_policy"], "full_resident");
        assert_eq!(value["device_name"], "Apple Test GPU");
    }
}
