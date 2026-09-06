//! SchedulerActor — Tokio task wrapping [`Scheduler`] for serving HTTP
//! requests via mpsc channels.
//!
//! 3b-3 activates multi-request batching via a hybrid admission window:
//! the first admit starts a [`ADMISSION_DEADLINE`] timer; further admits
//! accumulate until either [`Scheduler::active_count`] saturates at
//! `b_max` (saturate path) or the deadline expires (hard limit, no
//! reset on new admits).
//!
//! 3c-3 introduced the rolling decode loop: after first-batch prefill
//! the driver usually biased-selects between `cmd_rx.recv()` (mid-batch
//! admit) and an always-ready step branch. Admission work marks the next
//! decode step as due, so active rows take one [`Scheduler::step`] before
//! the actor accepts more optional mid-batch admission work. Mid admits
//! route through [`Scheduler::admit_mid`] (B=1 temp-cache prefill +
//! adopt-into-main); step branch calls [`Scheduler::step`] +
//! [`Scheduler::gc_finished_rows`]. The loop exits when
//! `active_count == 0` AND `cmd_rx` is empty.
//!
//! The admission window is coordinated by the scheduler actor.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::core::cache::{
    ActiveKvOffloadConfig, ActiveKvOffloadSharedStats, PagedPrefixCacheConfig, PrefixLruCacheConfig,
};
use crate::core::generate::GenerateRequest;
use crate::core::model::Model;
use crate::core::prompt_lookup::{
    PromptLookupConfig, PromptLookupCostAction, PromptLookupCostController,
    PromptLookupDraftLimits, PromptLookupProposalSource, PromptLookupQualificationRegime,
    PromptLookupQualificationRuntimeConfig, PromptLookupQualificationStats, PromptLookupStats,
};
use crate::core::scheduler::{
    ActiveKvParkedRequest, AdmitMidHandle, DenseVlMethods, Gemma4DrafterAdmitMidHandle,
    ImmutablePrefixBlockStats, MtpAdmitMidHandle, Phase, PromptLookupMtpStepOutcome, RequestId,
    Scheduler, StepEvent,
};
use crate::core::server::adaptive_admission::{
    AdaptiveAdmissionPolicy, AdmissionRequestShape, ROLLING_DECODE_STEPS_AFTER_ADMISSION_WORK,
};
use crate::core::speculative::{MtpSpeculativeConfig, MtpSpeculativeModel, MtpSpeculativeStats};
use crate::core::speculative_qualification::{
    NeuralExactAction, NeuralExactCostController, NeuralExactQualificationRuntimeConfig,
    NeuralExactQualificationStats, NeuralExactRegime, NeuralExactSampleCounters, NeuralExactSource,
};
use crate::Result;

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ImmutablePrefixBlockHealth {
    pub enabled: bool,
    pub blocks: u64,
    pub published_blocks: u64,
    pub restored_blocks: u64,
    pub active_block_hits: u64,
    pub idle_block_hits: u64,
    pub lookup_misses: u64,
    pub evicted_blocks: u64,
    pub blocked_evictions: u64,
    pub pressure_evicted_blocks: u64,
    pub ssd_block_hits: u64,
    pub ssd_blocks_loaded: u64,
    pub ssd_blocks_queued: u64,
    pub ssd_blocks_pending: u64,
    pub ssd_store_backpressure: u64,
    pub ssd_load_pressure_skips: u64,
    pub dedup_saved_bytes: u64,
}

#[derive(Clone)]
pub struct ImmutablePrefixBlockSharedStats {
    enabled: bool,
    blocks: Arc<AtomicU64>,
    published_blocks: Arc<AtomicU64>,
    restored_blocks: Arc<AtomicU64>,
    active_block_hits: Arc<AtomicU64>,
    idle_block_hits: Arc<AtomicU64>,
    lookup_misses: Arc<AtomicU64>,
    evicted_blocks: Arc<AtomicU64>,
    blocked_evictions: Arc<AtomicU64>,
    pressure_evicted_blocks: Arc<AtomicU64>,
    ssd_block_hits: Arc<AtomicU64>,
    ssd_blocks_loaded: Arc<AtomicU64>,
    ssd_blocks_queued: Arc<AtomicU64>,
    ssd_blocks_pending: Arc<AtomicU64>,
    ssd_store_backpressure: Arc<AtomicU64>,
    ssd_load_pressure_skips: Arc<AtomicU64>,
    dedup_saved_bytes: Arc<AtomicU64>,
}

impl ImmutablePrefixBlockSharedStats {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            blocks: Arc::new(AtomicU64::new(0)),
            published_blocks: Arc::new(AtomicU64::new(0)),
            restored_blocks: Arc::new(AtomicU64::new(0)),
            active_block_hits: Arc::new(AtomicU64::new(0)),
            idle_block_hits: Arc::new(AtomicU64::new(0)),
            lookup_misses: Arc::new(AtomicU64::new(0)),
            evicted_blocks: Arc::new(AtomicU64::new(0)),
            blocked_evictions: Arc::new(AtomicU64::new(0)),
            pressure_evicted_blocks: Arc::new(AtomicU64::new(0)),
            ssd_block_hits: Arc::new(AtomicU64::new(0)),
            ssd_blocks_loaded: Arc::new(AtomicU64::new(0)),
            ssd_blocks_queued: Arc::new(AtomicU64::new(0)),
            ssd_blocks_pending: Arc::new(AtomicU64::new(0)),
            ssd_store_backpressure: Arc::new(AtomicU64::new(0)),
            ssd_load_pressure_skips: Arc::new(AtomicU64::new(0)),
            dedup_saved_bytes: Arc::new(AtomicU64::new(0)),
        }
    }

    fn store(&self, stats: ImmutablePrefixBlockStats) {
        self.blocks.store(stats.blocks, Ordering::Relaxed);
        self.published_blocks
            .store(stats.published_blocks, Ordering::Relaxed);
        self.restored_blocks
            .store(stats.restored_blocks, Ordering::Relaxed);
        self.active_block_hits
            .store(stats.active_block_hits, Ordering::Relaxed);
        self.idle_block_hits
            .store(stats.idle_block_hits, Ordering::Relaxed);
        self.lookup_misses
            .store(stats.lookup_misses, Ordering::Relaxed);
        self.evicted_blocks
            .store(stats.evicted_blocks, Ordering::Relaxed);
        self.blocked_evictions
            .store(stats.blocked_evictions, Ordering::Relaxed);
        self.pressure_evicted_blocks
            .store(stats.pressure_evicted_blocks, Ordering::Relaxed);
        self.ssd_block_hits
            .store(stats.ssd_block_hits, Ordering::Relaxed);
        self.ssd_blocks_loaded
            .store(stats.ssd_blocks_loaded, Ordering::Relaxed);
        self.ssd_blocks_queued
            .store(stats.ssd_blocks_queued, Ordering::Relaxed);
        self.ssd_blocks_pending
            .store(stats.ssd_blocks_pending, Ordering::Relaxed);
        self.ssd_store_backpressure
            .store(stats.ssd_store_backpressure, Ordering::Relaxed);
        self.ssd_load_pressure_skips
            .store(stats.ssd_load_pressure_skips, Ordering::Relaxed);
        self.dedup_saved_bytes
            .store(stats.dedup_saved_bytes, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> ImmutablePrefixBlockHealth {
        ImmutablePrefixBlockHealth {
            enabled: self.enabled,
            blocks: self.blocks.load(Ordering::Relaxed),
            published_blocks: self.published_blocks.load(Ordering::Relaxed),
            restored_blocks: self.restored_blocks.load(Ordering::Relaxed),
            active_block_hits: self.active_block_hits.load(Ordering::Relaxed),
            idle_block_hits: self.idle_block_hits.load(Ordering::Relaxed),
            lookup_misses: self.lookup_misses.load(Ordering::Relaxed),
            evicted_blocks: self.evicted_blocks.load(Ordering::Relaxed),
            blocked_evictions: self.blocked_evictions.load(Ordering::Relaxed),
            pressure_evicted_blocks: self.pressure_evicted_blocks.load(Ordering::Relaxed),
            ssd_block_hits: self.ssd_block_hits.load(Ordering::Relaxed),
            ssd_blocks_loaded: self.ssd_blocks_loaded.load(Ordering::Relaxed),
            ssd_blocks_queued: self.ssd_blocks_queued.load(Ordering::Relaxed),
            ssd_blocks_pending: self.ssd_blocks_pending.load(Ordering::Relaxed),
            ssd_store_backpressure: self.ssd_store_backpressure.load(Ordering::Relaxed),
            ssd_load_pressure_skips: self.ssd_load_pressure_skips.load(Ordering::Relaxed),
            dedup_saved_bytes: self.dedup_saved_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Commands accepted by the actor. 3b-2 ships only [`Admit`]; later
/// phases may add `Cancel { id }`, `Stats`, etc.
pub enum SchedulerCommand {
    /// Submit a request for batched generation. On success, replies with
    /// the admitted [`RequestId`] and an mpsc receiver that streams
    /// [`StepEvent`]s (one per produced token, until `finish_reason`
    /// becomes `Some(_)` on the final event for this row).
    Admit {
        request: GenerateRequest,
        reply_tx: oneshot::Sender<Result<AdmitReply>>,
    },
}

pub(super) enum SchedulerControlCommand {
    ClearSharedPromptLookup { reply_tx: oneshot::Sender<usize> },
}

/// A request parked in `driver_loop`'s admission queue while the scheduler
/// is at `active_count == b_max`. Drained when `gc_finished_rows` frees a
/// slot, then handed to the rolling mid-admit chunk path.
struct PendingAdmit {
    request: GenerateRequest,
    reply_tx: oneshot::Sender<Result<AdmitReply>>,
    queued_at_profile: Option<Instant>,
}

fn fresh_prefill_batch_limit_for_request<M: Model>(
    request: &GenerateRequest,
    b_max: usize,
    adaptive_policy: AdaptiveAdmissionPolicy,
) -> usize {
    let model_limit = M::fresh_prefill_batch_limit(request.prompt_ids.len(), b_max).clamp(1, b_max);
    adaptive_policy.fresh_batch_limit(admission_request_shape(request), model_limit, b_max)
}

fn fresh_prefill_batch_limit_for_command<M: Model>(
    cmd: &SchedulerCommand,
    b_max: usize,
    adaptive_policy: AdaptiveAdmissionPolicy,
) -> usize {
    let SchedulerCommand::Admit { request, .. } = cmd;
    fresh_prefill_batch_limit_for_request::<M>(request, b_max, adaptive_policy)
}

fn admission_request_shape(request: &GenerateRequest) -> AdmissionRequestShape {
    AdmissionRequestShape {
        prompt_len: request.prompt_ids.len(),
        prefill_chunk_size: request.prefill_chunk_size,
        decode_cadence_mid_chunk_cap: request.decode_cadence_mid_chunk_cap,
        speculative_pipelinable: request.sampler.is_pipelinable(),
    }
}

fn admission_command_shape(cmd: &SchedulerCommand) -> AdmissionRequestShape {
    let SchedulerCommand::Admit { request, .. } = cmd;
    admission_request_shape(request)
}

fn startup_budget_policy(
    effective_cap_max: usize,
    paged_prefix_cache: Option<&PagedPrefixCacheConfig>,
    active_kv_offload: &ActiveKvOffloadConfig,
) -> crate::core::memory_budget::KvBudgetPolicy {
    if !active_kv_offload.enabled {
        return crate::core::memory_budget::KvBudgetPolicy::FullResident;
    }
    let Some(paged_prefix_cache) = paged_prefix_cache else {
        return crate::core::memory_budget::KvBudgetPolicy::FullResident;
    };

    let block_size_i32 = paged_prefix_cache.block_size.max(1);
    let hot_window_pages_i32 = active_kv_offload
        .hot_window_pages_override
        .unwrap_or_else(|| {
            crate::core::scheduler::default_active_kv_hot_window_pages(block_size_i32)
        })
        .max(1);
    let block_size = usize::try_from(block_size_i32).unwrap_or(1);
    let hot_window_pages = usize::try_from(hot_window_pages_i32).unwrap_or(1);
    let resident_cap = hot_window_pages
        .saturating_mul(block_size)
        .min(effective_cap_max.max(1));

    crate::core::memory_budget::KvBudgetPolicy::active_kv_offload(resident_cap)
}

/// Event yielded by the rolling decode loop. Either a new admit command
/// arrived (mid-batch admit), a decode step is due, or the cmd_rx channel
/// was closed (shutdown).
#[allow(clippy::large_enum_variant)] // Admit(SchedulerCommand) intentionally large; boxing would add allocation on hot path
enum RollingEvent {
    Admit(SchedulerCommand),
    AdvanceMidAdmit,
    Step,
    Shutdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RollingMidAdmitSource {
    Direct,
    Queue,
}

#[derive(Clone, Copy, Debug)]
struct MidAdmitProfileContext {
    source: RollingMidAdmitSource,
    queue_wait_ms: Option<f64>,
    queue_len: usize,
}

impl RollingMidAdmitSource {
    fn as_str(self) -> &'static str {
        match self {
            RollingMidAdmitSource::Direct => "direct",
            RollingMidAdmitSource::Queue => "queue",
        }
    }
}

fn rolling_profile_enabled_from_env(value: Option<&str>) -> bool {
    value == Some("1")
}

fn rolling_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        rolling_profile_enabled_from_env(
            std::env::var("IRONMLX_CHUNKED_ROLLING_PROFILE")
                .ok()
                .as_deref(),
        )
    })
}

fn rolling_profile_t_ms(now: Instant) -> f64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = *EPOCH.get_or_init(|| now);
    now.saturating_duration_since(epoch).as_secs_f64() * 1000.0
}

fn rolling_profile_elapsed_ms(start: Instant, end: Instant) -> f64 {
    end.saturating_duration_since(start).as_secs_f64() * 1000.0
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().try_into().unwrap_or(u64::MAX)
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().try_into().unwrap_or(u64::MAX)
}

fn rolling_profile_queue_wait_ms(queued_at: Instant, now: Instant) -> f64 {
    rolling_profile_elapsed_ms(queued_at, now)
}

fn cadence_protected_mid_chunk_size(
    requested_chunk_size: i32,
    active_count_with_mid_admit: usize,
    decode_cadence_mid_chunk_cap: usize,
) -> i32 {
    let requested_chunk_size = requested_chunk_size.max(1);
    if active_count_with_mid_admit > 1 {
        let cap = decode_cadence_mid_chunk_cap.clamp(1, i32::MAX as usize) as i32;
        requested_chunk_size.min(cap)
    } else {
        requested_chunk_size
    }
}

/// Rolling-loop admission fairness policy.
///
/// Any completed admission work (initial prefill or mid-batch prefill)
/// makes a short decode burst due before the actor accepts more optional
/// admission work. This keeps prefill from alternating with every decode
/// token under sustained long-prompt arrivals.
#[derive(Debug, Default)]
struct RollingAdmissionPolicy {
    decode_steps_due_after_admission: usize,
}

impl RollingAdmissionPolicy {
    fn record_admission_work(&mut self) {
        self.record_admission_work_with_decode_steps(ROLLING_DECODE_STEPS_AFTER_ADMISSION_WORK);
    }

    fn record_admission_work_with_decode_steps(&mut self, decode_steps: usize) {
        self.decode_steps_due_after_admission =
            self.decode_steps_due_after_admission.max(decode_steps);
    }

    fn record_decode_step(&mut self) {
        self.decode_steps_due_after_admission =
            self.decode_steps_due_after_admission.saturating_sub(1);
    }

    fn should_force_decode(
        &self,
        phase: Phase,
        has_decodable_rows: bool,
        has_pending_admission_work: bool,
    ) -> bool {
        self.decode_steps_due_after_admission > 0
            && phase == Phase::Decoding
            && has_decodable_rows
            && has_pending_admission_work
    }
}

fn decode_steps_after_mid_admit_chunk(chunk_tokens: usize, cadence_chunk_cap: usize) -> usize {
    let cadence_chunk_cap = cadence_chunk_cap.max(1);
    chunk_tokens
        .max(1)
        .div_ceil(cadence_chunk_cap)
        .saturating_mul(ROLLING_DECODE_STEPS_AFTER_ADMISSION_WORK)
}

fn scheduler_has_decodable_rows<M: Model>(sched: &Scheduler<M>) -> bool {
    sched
        .active()
        .into_iter()
        .any(|state| !state.finished && !state.generated_tokens.is_empty())
}

fn scheduler_available_decode_steps<M: Model>(sched: &Scheduler<M>) -> usize {
    sched
        .active()
        .into_iter()
        .filter(|state| !state.finished && !state.generated_tokens.is_empty())
        .map(|state| {
            state
                .max_new_tokens
                .saturating_sub(state.generated_tokens.len())
        })
        .max()
        .unwrap_or(0)
}

#[derive(Clone)]
struct SchedulerActorMtpCounters {
    mtp_prefill_count: Arc<AtomicU64>,
    mtp_step_count: Arc<AtomicU64>,
    mtp_prefill_fallback_count: Arc<AtomicU64>,
    mtp_drafted_tokens: Arc<AtomicU64>,
    mtp_accepted_draft_tokens: Arc<AtomicU64>,
    mtp_windows: Arc<AtomicU64>,
    mtp_exact_sampling_windows: Arc<AtomicU64>,
    mtp_exact_acceptance_draws: Arc<AtomicU64>,
    mtp_exact_residual_corrections: Arc<AtomicU64>,
    mtp_exact_bonus_samples: Arc<AtomicU64>,
    mtp_draft_forward_us: Arc<AtomicU64>,
    mtp_verify_forward_us: Arc<AtomicU64>,
    mtp_projection_us: Arc<AtomicU64>,
    mtp_sampling_us: Arc<AtomicU64>,
    mtp_draft_host_sync_count: Arc<AtomicU64>,
    mtp_draft_host_sync_us: Arc<AtomicU64>,
    mtp_verify_accept_host_sync_count: Arc<AtomicU64>,
    mtp_verify_accept_host_sync_us: Arc<AtomicU64>,
    mtp_main_rollback_us: Arc<AtomicU64>,
    mtp_cache_commit_us: Arc<AtomicU64>,
    mtp_prefill_cache_commit_us: Arc<AtomicU64>,
    mtp_decode_cache_commit_us: Arc<AtomicU64>,
    mtp_cache_restore_us: Arc<AtomicU64>,
    published_stats: Arc<StdMutex<Option<MtpSpeculativeStats>>>,
    prompt_lookup_published_stats: Arc<StdMutex<Option<PromptLookupStats>>>,
    prompt_lookup_stats_baseline: Arc<StdMutex<Option<PromptLookupStats>>>,
    neural_exact_qualification_stats: Arc<StdMutex<NeuralExactQualificationStats>>,
}

impl SchedulerActorMtpCounters {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mtp_prefill_count: Arc<AtomicU64>,
        mtp_step_count: Arc<AtomicU64>,
        mtp_prefill_fallback_count: Arc<AtomicU64>,
        mtp_drafted_tokens: Arc<AtomicU64>,
        mtp_accepted_draft_tokens: Arc<AtomicU64>,
        mtp_windows: Arc<AtomicU64>,
        mtp_exact_sampling_windows: Arc<AtomicU64>,
        mtp_exact_acceptance_draws: Arc<AtomicU64>,
        mtp_exact_residual_corrections: Arc<AtomicU64>,
        mtp_exact_bonus_samples: Arc<AtomicU64>,
        mtp_draft_forward_us: Arc<AtomicU64>,
        mtp_verify_forward_us: Arc<AtomicU64>,
        mtp_projection_us: Arc<AtomicU64>,
        mtp_sampling_us: Arc<AtomicU64>,
        mtp_draft_host_sync_count: Arc<AtomicU64>,
        mtp_draft_host_sync_us: Arc<AtomicU64>,
        mtp_verify_accept_host_sync_count: Arc<AtomicU64>,
        mtp_verify_accept_host_sync_us: Arc<AtomicU64>,
        mtp_main_rollback_us: Arc<AtomicU64>,
        mtp_cache_commit_us: Arc<AtomicU64>,
        mtp_prefill_cache_commit_us: Arc<AtomicU64>,
        mtp_decode_cache_commit_us: Arc<AtomicU64>,
        mtp_cache_restore_us: Arc<AtomicU64>,
        prompt_lookup_published_stats: Arc<StdMutex<Option<PromptLookupStats>>>,
        neural_exact_qualification_stats: Arc<StdMutex<NeuralExactQualificationStats>>,
    ) -> Self {
        Self {
            mtp_prefill_count,
            mtp_step_count,
            mtp_prefill_fallback_count,
            mtp_drafted_tokens,
            mtp_accepted_draft_tokens,
            mtp_windows,
            mtp_exact_sampling_windows,
            mtp_exact_acceptance_draws,
            mtp_exact_residual_corrections,
            mtp_exact_bonus_samples,
            mtp_draft_forward_us,
            mtp_verify_forward_us,
            mtp_projection_us,
            mtp_sampling_us,
            mtp_draft_host_sync_count,
            mtp_draft_host_sync_us,
            mtp_verify_accept_host_sync_count,
            mtp_verify_accept_host_sync_us,
            mtp_main_rollback_us,
            mtp_cache_commit_us,
            mtp_prefill_cache_commit_us,
            mtp_decode_cache_commit_us,
            mtp_cache_restore_us,
            published_stats: Arc::new(StdMutex::new(None)),
            prompt_lookup_published_stats,
            prompt_lookup_stats_baseline: Arc::new(StdMutex::new(None)),
            neural_exact_qualification_stats,
        }
    }

    fn store_neural_exact_qualification_stats(&self, stats: NeuralExactQualificationStats) {
        *self
            .neural_exact_qualification_stats
            .lock()
            .expect("neural exact qualification stats mutex poisoned") = stats;
    }

    fn store_stats(&self, stats: Option<MtpSpeculativeStats>) {
        let mut published = self
            .published_stats
            .lock()
            .expect("MTP published stats mutex poisoned");
        match stats {
            Some(stats) => {
                let delta = published
                    .as_ref()
                    .map(|before| stats.saturating_delta_since(before))
                    .unwrap_or_else(|| stats.clone());
                self.add_stats_delta(&delta);
                *published = Some(stats);
            }
            None => {
                *published = None;
            }
        }
    }

    fn reset_stats_baseline(&self, prompt_lookup_stats: Option<PromptLookupStats>) {
        self.store_stats(None);
        match prompt_lookup_stats {
            Some(stats) => self.store_prompt_lookup_stats(Some(stats)),
            None => self.store_prompt_lookup_stats(None),
        }
    }

    fn add_stats_delta(&self, stats: &MtpSpeculativeStats) {
        self.mtp_windows
            .fetch_add(stats.windows as u64, Ordering::Relaxed);
        self.mtp_drafted_tokens
            .fetch_add(stats.drafted_tokens as u64, Ordering::Relaxed);
        self.mtp_accepted_draft_tokens
            .fetch_add(stats.accepted_draft_tokens as u64, Ordering::Relaxed);
        self.mtp_exact_sampling_windows
            .fetch_add(stats.exact_sampling_windows as u64, Ordering::Relaxed);
        self.mtp_exact_acceptance_draws
            .fetch_add(stats.exact_acceptance_draws as u64, Ordering::Relaxed);
        self.mtp_exact_residual_corrections
            .fetch_add(stats.exact_residual_corrections as u64, Ordering::Relaxed);
        self.mtp_exact_bonus_samples
            .fetch_add(stats.exact_bonus_samples as u64, Ordering::Relaxed);
        self.mtp_draft_forward_us
            .fetch_add(stats.draft_forward_us, Ordering::Relaxed);
        self.mtp_verify_forward_us
            .fetch_add(stats.verify_forward_us, Ordering::Relaxed);
        self.mtp_projection_us
            .fetch_add(stats.projection_us, Ordering::Relaxed);
        self.mtp_sampling_us
            .fetch_add(stats.sampling_us, Ordering::Relaxed);
        self.mtp_draft_host_sync_count
            .fetch_add(stats.draft_host_sync_count as u64, Ordering::Relaxed);
        self.mtp_draft_host_sync_us
            .fetch_add(stats.draft_host_sync_us, Ordering::Relaxed);
        self.mtp_verify_accept_host_sync_count.fetch_add(
            stats.verify_accept_host_sync_count as u64,
            Ordering::Relaxed,
        );
        self.mtp_verify_accept_host_sync_us
            .fetch_add(stats.verify_accept_host_sync_us, Ordering::Relaxed);
        self.mtp_main_rollback_us
            .fetch_add(stats.main_rollback_us, Ordering::Relaxed);
        self.mtp_cache_commit_us
            .fetch_add(stats.mtp_cache_commit_us, Ordering::Relaxed);
        self.mtp_prefill_cache_commit_us
            .fetch_add(stats.mtp_prefill_cache_commit_us, Ordering::Relaxed);
        self.mtp_decode_cache_commit_us
            .fetch_add(stats.mtp_decode_cache_commit_us, Ordering::Relaxed);
        self.mtp_cache_restore_us
            .fetch_add(stats.mtp_cache_restore_us, Ordering::Relaxed);
    }

    fn store_prompt_lookup_stats(&self, stats: Option<PromptLookupStats>) {
        let mut baseline = self
            .prompt_lookup_stats_baseline
            .lock()
            .expect("PromptLookup stats baseline mutex poisoned");
        let mut published = self
            .prompt_lookup_published_stats
            .lock()
            .expect("PromptLookup published stats mutex poisoned");
        match stats {
            Some(stats) => {
                let delta = baseline
                    .map(|before| stats.saturating_delta_since(before))
                    .unwrap_or(stats);
                published.get_or_insert_default().accumulate_delta(delta);
                *baseline = Some(stats);
            }
            None => {
                *baseline = None;
                if let Some(stats) = published.as_mut() {
                    stats.index_entries_current = 0;
                    stats.index_ledger_entries_current = 0;
                    stats.index_estimated_bytes_current = 0;
                }
            }
        }
    }

    fn store_prompt_lookup_stats_with_qualification(
        &self,
        stats: Option<PromptLookupStats>,
        qualification: PromptLookupQualificationStats,
    ) {
        self.store_prompt_lookup_stats(stats);
        let mut published = self
            .prompt_lookup_published_stats
            .lock()
            .expect("PromptLookup published stats mutex poisoned");
        let stats = published.get_or_insert_default();
        stats.ordinary_cost_samples = qualification.ordinary_cost_samples;
        stats.lookup_cost_samples = qualification.lookup_cost_samples;
        stats.ordinary_cost_us = qualification.ordinary_cost_us;
        stats.lookup_cost_us = qualification.lookup_cost_us;
        stats.qualified_regimes_current = qualification.qualified_regimes_current;
        stats.rejected_regimes_current = qualification.rejected_regimes_current;
        stats.qualification_changes = qualification.qualification_changes;
        stats.qualification_profile_loads = qualification.profile_loads;
        stats.qualification_profile_writes = qualification.profile_writes;
        stats.qualification_profile_write_drops = qualification.profile_write_drops;
        stats.qualification_query_gate_skips = qualification.query_gate_skips;
        stats.miss_query_gate_skips = qualification.miss_query_gate_skips;
        stats.miss_query_reprobes = qualification.miss_query_reprobes;
        stats.adaptive_draft_width_reductions = qualification.adaptive_draft_width_reductions;
        stats.adaptive_draft_width_increases = qualification.adaptive_draft_width_increases;
        stats.adaptive_profitability_width_reductions =
            qualification.adaptive_profitability_width_reductions;
    }

    fn store_prompt_lookup_hybrid_stats(&self, hybrid: PromptLookupHybridStats) {
        let mut published = self
            .prompt_lookup_published_stats
            .lock()
            .expect("PromptLookup published stats mutex poisoned");
        let stats = published.get_or_insert_default();
        stats.hybrid_neural_windows = hybrid.neural_windows;
        stats.hybrid_lookup_windows = hybrid.lookup_windows;
        stats.hybrid_source_switches = hybrid.source_switches;
        stats.hybrid_lookup_miss_fallbacks = hybrid.lookup_miss_fallbacks;
        stats.hybrid_neural_rebases = hybrid.neural_rebases;
        stats.hybrid_neural_rebase_us = hybrid.neural_rebase_us;
    }
}

trait SchedulerActorMtpMode<M>
where
    M: Model + DenseVlMethods,
{
    type MidAdmitHandle: Send + 'static;

    fn allow_rolling_mid_admit(&self) -> bool {
        true
    }

    fn can_start_rolling_mid_admit(&self, _sched: &Scheduler<M>) -> bool {
        true
    }

    fn mid_admit_request_id(handle: &Self::MidAdmitHandle) -> RequestId;

    fn mid_admit_chunk_start(handle: &Self::MidAdmitHandle) -> i32;

    fn mid_admit_prompt_len(handle: &Self::MidAdmitHandle) -> i32;

    fn mid_admit_chunk_size(handle: &Self::MidAdmitHandle) -> i32;

    fn set_mid_admit_chunk_size(handle: &mut Self::MidAdmitHandle, chunk_size: i32);

    fn mid_admit_decode_cadence_mid_chunk_cap(handle: &Self::MidAdmitHandle) -> usize;

    fn prefill_admitted(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>>;

    fn step(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
        admission_pending: bool,
    ) -> Result<Vec<StepEvent>>;

    fn begin_mid_admit(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        request: GenerateRequest,
    ) -> Result<Self::MidAdmitHandle>;

    fn advance_mid_admit_chunk(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        handle: &mut Self::MidAdmitHandle,
    ) -> Result<bool>;

    fn finalize_mid_admit(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        handle: Self::MidAdmitHandle,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<(RequestId, StepEvent)>;
}

struct SchedulerActorNoMtp;

struct SchedulerActorPromptLookup {
    cfg: PromptLookupConfig,
    defer_speculation_once: bool,
    cost_controller: PromptLookupCostController,
    measured_cycle: Option<PromptLookupMeasuredCycle>,
    query_hint: Option<PromptLookupQueryHint>,
    miss_query_hint: Option<PromptLookupMissQueryHint>,
}

const PROMPT_LOOKUP_MISS_INITIAL_REPROBE_TOKENS: usize = 2;
const PROMPT_LOOKUP_MISS_MAX_REPROBE_TOKENS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
struct PromptLookupQueryScope {
    base_regime: PromptLookupQualificationRegime,
    owners: Vec<RequestId>,
    draft_limits: PromptLookupDraftLimits,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PromptLookupQueryHint {
    scope: PromptLookupQueryScope,
    proposal_regime: PromptLookupQualificationRegime,
}

impl PromptLookupQueryHint {
    fn proposal_regime_for(
        &self,
        scope: &PromptLookupQueryScope,
    ) -> Option<PromptLookupQualificationRegime> {
        (self.scope == *scope).then_some(self.proposal_regime)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PromptLookupMissQueryScope {
    base_regime: PromptLookupQualificationRegime,
    request_progress: Vec<(RequestId, usize)>,
    allow_cross_request: bool,
    shared_availability_epoch: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PromptLookupMissQueryHint {
    scope: PromptLookupMissQueryScope,
    reprobe_after_tokens: usize,
}

impl PromptLookupMissQueryHint {
    fn identity_matches(&self, scope: &PromptLookupMissQueryScope) -> bool {
        self.scope.base_regime == scope.base_regime
            && self.scope.allow_cross_request == scope.allow_cross_request
            && self.scope.shared_availability_epoch == scope.shared_availability_epoch
            && self.scope.request_progress.len() == scope.request_progress.len()
            && self
                .scope
                .request_progress
                .iter()
                .zip(&scope.request_progress)
                .all(|((previous_id, previous_len), (current_id, current_len))| {
                    previous_id == current_id && current_len >= previous_len
                })
    }

    fn should_skip(&self, scope: &PromptLookupMissQueryScope) -> bool {
        self.identity_matches(scope)
            && self
                .scope
                .request_progress
                .iter()
                .zip(&scope.request_progress)
                .all(|((_, previous_len), (_, current_len))| {
                    current_len.saturating_sub(*previous_len) < self.reprobe_after_tokens
                })
    }

    fn after_miss(
        scope: PromptLookupMissQueryScope,
        previous: Option<&PromptLookupMissQueryHint>,
    ) -> Self {
        let reprobe_after_tokens = previous
            .filter(|hint| hint.identity_matches(&scope))
            .map_or(PROMPT_LOOKUP_MISS_INITIAL_REPROBE_TOKENS, |hint| {
                hint.reprobe_after_tokens
                    .saturating_mul(2)
                    .min(PROMPT_LOOKUP_MISS_MAX_REPROBE_TOKENS)
            });
        Self {
            scope,
            reprobe_after_tokens,
        }
    }
}

fn prompt_lookup_admission_forces_ordinary(
    measured_cycle_active: bool,
    admission_pending: bool,
) -> bool {
    admission_pending && !measured_cycle_active
}

fn qwen_hybrid_uses_canonical_target(
    source: HybridDraftSource,
    regime: Option<PromptLookupQualificationRegime>,
) -> bool {
    source == HybridDraftSource::PromptLookup
        && regime.is_some_and(|regime| {
            matches!(
                regime.proposal_source,
                Some(PromptLookupProposalSource::Local | PromptLookupProposalSource::Shared)
            )
        })
}

struct PromptLookupMeasuredCycle {
    regime: PromptLookupQualificationRegime,
    action: PromptLookupCostAction,
    elapsed_ns: u64,
    committed_tokens: usize,
    stats_before: PromptLookupStats,
}

struct PromptLookupWindowDecision {
    action: PromptLookupCostAction,
    regime: Option<PromptLookupQualificationRegime>,
    proposal_elapsed_ns: u64,
    stats_before: PromptLookupStats,
    fallback_to_baseline: bool,
}

#[derive(Debug, Default)]
struct PromptLookupEpisode {
    regimes: Vec<PromptLookupQualificationRegime>,
    committed_tokens: usize,
}

impl PromptLookupEpisode {
    fn record(&mut self, regime: PromptLookupQualificationRegime, committed_tokens: usize) {
        if !self.regimes.contains(&regime) {
            self.regimes.push(regime);
        }
        self.committed_tokens = self.committed_tokens.saturating_add(committed_tokens);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HybridDraftSource {
    Neural,
    PromptLookup,
}

#[derive(Debug, Clone, Copy, Default)]
struct PromptLookupHybridStats {
    neural_windows: u64,
    lookup_windows: u64,
    source_switches: u64,
    lookup_miss_fallbacks: u64,
    neural_rebases: u64,
    neural_rebase_us: u64,
}

struct SchedulerActorMtpPromptLookupHybrid<H> {
    neural: SchedulerActorMtp<H>,
    prompt_lookup: SchedulerActorPromptLookup,
    current_source: Option<HybridDraftSource>,
    neural_dirty: bool,
    lookup_window_canonical: bool,
    measured_cycle: Option<PromptLookupMeasuredCycle>,
    lookup_episode: Option<PromptLookupEpisode>,
    stats: PromptLookupHybridStats,
}

struct SchedulerActorGemma4PromptLookupHybrid {
    neural: SchedulerActorGemma4Drafter,
    prompt_lookup: SchedulerActorPromptLookup,
    current_source: Option<HybridDraftSource>,
    neural_dirty: bool,
    measured_cycle: Option<PromptLookupMeasuredCycle>,
    lookup_episode: Option<PromptLookupEpisode>,
    stats: PromptLookupHybridStats,
}

struct SchedulerActorMtp<H> {
    mtp: H,
    cfg: MtpSpeculativeConfig,
    exact_cost_controller: Option<NeuralExactCostController>,
    exact_episode: Option<NeuralExactMeasuredEpisode>,
}

enum SchedulerActorMtpMidAdmitHandle {
    Generic(AdmitMidHandle),
    Mtp(Box<MtpAdmitMidHandle>),
}

struct SchedulerActorGemma4Drafter {
    drafter: Arc<Mutex<crate::models::gemma4::Gemma4AssistantModel>>,
    cfg: MtpSpeculativeConfig,
    exact_cost_controller: Option<NeuralExactCostController>,
    exact_episode: Option<NeuralExactMeasuredEpisode>,
}

struct NeuralExactMeasuredEpisode {
    regime: NeuralExactRegime,
    action: NeuralExactAction,
    elapsed_ns: u64,
    committed_tokens: usize,
    stats_before: MtpSpeculativeStats,
}

enum SchedulerActorGemma4DrafterMidAdmitHandle {
    Generic(AdmitMidHandle),
    Drafter(Box<Gemma4DrafterAdmitMidHandle>),
}

impl<H> SchedulerActorMtp<H> {
    fn new(mtp: H, mtp_draft_tokens: usize) -> Self {
        debug_assert!(mtp_draft_tokens > 0);
        Self {
            mtp,
            cfg: MtpSpeculativeConfig {
                max_draft_tokens: mtp_draft_tokens,
            },
            exact_cost_controller: None,
            exact_episode: None,
        }
    }

    fn new_with_exact_qualification(
        mtp: H,
        mtp_draft_tokens: usize,
        qualification: NeuralExactQualificationRuntimeConfig,
    ) -> Result<Self> {
        let mut mode = Self::new(mtp, mtp_draft_tokens);
        mode.exact_cost_controller = Some(NeuralExactCostController::new(qualification)?);
        Ok(mode)
    }

    fn publish_exact_qualification(&self, counters: &SchedulerActorMtpCounters) {
        if let Some(controller) = self.exact_cost_controller.as_ref() {
            counters.store_neural_exact_qualification_stats(controller.stats());
        }
    }

    fn finish_exact_episode(
        &mut self,
        stats_after: Option<MtpSpeculativeStats>,
        counters: &SchedulerActorMtpCounters,
    ) {
        finish_neural_exact_episode(
            &mut self.exact_episode,
            self.exact_cost_controller.as_mut(),
            stats_after,
        );
        self.publish_exact_qualification(counters);
    }
}

impl<H> SchedulerActorMtpPromptLookupHybrid<H> {
    fn new(
        mtp: H,
        mtp_draft_tokens: usize,
        prompt_lookup: PromptLookupConfig,
        qualification: PromptLookupQualificationRuntimeConfig,
    ) -> Result<Self> {
        Ok(Self {
            neural: SchedulerActorMtp::new(mtp, mtp_draft_tokens),
            prompt_lookup: SchedulerActorPromptLookup::new(prompt_lookup, qualification)?,
            current_source: None,
            neural_dirty: false,
            lookup_window_canonical: false,
            measured_cycle: None,
            lookup_episode: None,
            stats: PromptLookupHybridStats::default(),
        })
    }
}

impl SchedulerActorGemma4PromptLookupHybrid {
    fn new(
        drafter: Arc<Mutex<crate::models::gemma4::Gemma4AssistantModel>>,
        mtp_draft_tokens: usize,
        prompt_lookup: PromptLookupConfig,
        qualification: PromptLookupQualificationRuntimeConfig,
    ) -> Result<Self> {
        Ok(Self {
            neural: SchedulerActorGemma4Drafter::new(drafter, mtp_draft_tokens),
            prompt_lookup: SchedulerActorPromptLookup::new(prompt_lookup, qualification)?,
            current_source: None,
            neural_dirty: false,
            measured_cycle: None,
            lookup_episode: None,
            stats: PromptLookupHybridStats::default(),
        })
    }
}

impl SchedulerActorPromptLookup {
    fn new(
        cfg: PromptLookupConfig,
        qualification: PromptLookupQualificationRuntimeConfig,
    ) -> Result<Self> {
        Ok(Self {
            cfg: cfg.validate()?,
            defer_speculation_once: false,
            cost_controller: PromptLookupCostController::new(qualification)?,
            measured_cycle: None,
            query_hint: None,
            miss_query_hint: None,
        })
    }

    fn miss_query_scope<M: Model>(
        sched: &Scheduler<M>,
        base_regime: Option<PromptLookupQualificationRegime>,
        allow_cross_request: bool,
    ) -> Option<PromptLookupMissQueryScope> {
        let base_regime = base_regime?;
        let request_progress = sched.prompt_lookup_active_request_progress();
        (!request_progress.is_empty()).then_some(PromptLookupMissQueryScope {
            base_regime,
            request_progress,
            allow_cross_request,
            shared_availability_epoch: allow_cross_request
                .then(|| sched.shared_prompt_lookup_availability_epoch())
                .flatten(),
        })
    }

    fn publish_stats<M: Model>(&self, sched: &Scheduler<M>, counters: &SchedulerActorMtpCounters) {
        counters.store_prompt_lookup_stats_with_qualification(
            sched.prompt_lookup_stats(),
            self.cost_controller.stats(),
        );
    }

    fn select_prepared_window<M: Model>(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        allow_cross_request: bool,
        capture_accepted_hidden: bool,
        match_mtp_verify_shape: bool,
        force_ordinary: bool,
    ) -> Result<PromptLookupWindowDecision> {
        let stats_before = sched.prompt_lookup_stats().unwrap_or_default();
        let base_regime = sched.prompt_lookup_qualification_regime();
        let draft_limits = if match_mtp_verify_shape {
            PromptLookupDraftLimits::uniform(self.cfg.max_draft_tokens)
        } else {
            base_regime.map_or_else(
                || PromptLookupDraftLimits::uniform(self.cfg.max_draft_tokens),
                |regime| {
                    self.cost_controller
                        .adaptive_draft_limits(regime, self.cfg.max_draft_tokens)
                },
            )
        }
        .capped(model.max_prompt_lookup_draft_tokens(self.cfg.max_draft_tokens));
        let miss_query_scope = Self::miss_query_scope(sched, base_regime, allow_cross_request);
        let query_scope = base_regime.map(|base_regime| PromptLookupQueryScope {
            base_regime,
            owners: miss_query_scope.as_ref().map_or_else(
                || sched.prompt_lookup_active_request_ids(),
                |scope| scope.request_progress.iter().map(|(id, _)| *id).collect(),
            ),
            draft_limits,
        });
        let hinted_regime = query_scope.as_ref().and_then(|scope| {
            self.query_hint
                .as_ref()
                .and_then(|hint| hint.proposal_regime_for(scope))
        });
        if hinted_regime.is_none() {
            self.query_hint = None;
        }
        if force_ordinary {
            sched.discard_prepared_prompt_lookup_window();
            return Ok(PromptLookupWindowDecision {
                action: PromptLookupCostAction::Ordinary,
                regime: base_regime,
                proposal_elapsed_ns: 0,
                stats_before,
                fallback_to_baseline: false,
            });
        }
        if let Some(regime) = hinted_regime {
            if self.cost_controller.next_action(regime) == PromptLookupCostAction::Ordinary {
                self.cost_controller.record_query_gate_skip();
                sched.discard_prepared_prompt_lookup_window();
                return Ok(PromptLookupWindowDecision {
                    action: PromptLookupCostAction::Ordinary,
                    regime: Some(regime),
                    proposal_elapsed_ns: 0,
                    stats_before,
                    fallback_to_baseline: false,
                });
            }
        }
        if self.miss_query_hint.as_ref().is_some_and(|hint| {
            miss_query_scope
                .as_ref()
                .is_some_and(|scope| hint.should_skip(scope))
        }) {
            self.cost_controller.record_miss_query_gate_skip();
            sched.discard_prepared_prompt_lookup_window();
            return Ok(PromptLookupWindowDecision {
                action: PromptLookupCostAction::Ordinary,
                regime: base_regime,
                proposal_elapsed_ns: 0,
                stats_before,
                fallback_to_baseline: true,
            });
        }
        if let Some(hint) = self.miss_query_hint.as_ref() {
            if miss_query_scope
                .as_ref()
                .is_some_and(|scope| hint.identity_matches(scope))
            {
                self.cost_controller.record_miss_query_reprobe();
            } else {
                self.miss_query_hint = None;
            }
        }

        let proposal_started = Instant::now();
        let mut has_drafts =
            sched.prepare_prompt_lookup_batch_window(allow_cross_request, draft_limits)?;
        if has_drafts && match_mtp_verify_shape {
            has_drafts = sched.align_prepared_prompt_lookup_to_mtp_verify_shape()?;
        }
        let proposal_elapsed_ns = duration_ns(proposal_started.elapsed());
        let Some(regime) = has_drafts
            .then(|| sched.prompt_lookup_prepared_qualification_regime())
            .flatten()
        else {
            self.query_hint = None;
            let miss_query_scope = Self::miss_query_scope(sched, base_regime, allow_cross_request);
            let previous_miss_query_hint = self.miss_query_hint.take();
            self.miss_query_hint = miss_query_scope.map(|scope| {
                PromptLookupMissQueryHint::after_miss(scope, previous_miss_query_hint.as_ref())
            });
            sched.discard_prepared_prompt_lookup_window();
            return Ok(PromptLookupWindowDecision {
                action: PromptLookupCostAction::Ordinary,
                regime: base_regime,
                proposal_elapsed_ns: 0,
                stats_before,
                fallback_to_baseline: true,
            });
        };
        self.miss_query_hint = None;
        self.query_hint = query_scope.map(|scope| PromptLookupQueryHint {
            scope,
            proposal_regime: regime,
        });
        if !sched.prompt_lookup_prepared_window_verify_eligible(model, capture_accepted_hidden)? {
            if match_mtp_verify_shape {
                self.cost_controller.record_lookup_ineligible(regime);
            } else {
                self.cost_controller
                    .record_lookup_ineligible_with_adaptive_draft(regime);
            }
            sched.discard_prepared_prompt_lookup_window();
            return Ok(PromptLookupWindowDecision {
                action: PromptLookupCostAction::Ordinary,
                regime: Some(regime),
                proposal_elapsed_ns: 0,
                stats_before,
                fallback_to_baseline: true,
            });
        }
        let action = self.cost_controller.next_action(regime);
        if action == PromptLookupCostAction::Ordinary {
            sched.discard_prepared_prompt_lookup_window();
        }
        Ok(PromptLookupWindowDecision {
            action,
            regime: Some(regime),
            proposal_elapsed_ns: if action == PromptLookupCostAction::Lookup {
                proposal_elapsed_ns
            } else {
                0
            },
            stats_before,
            fallback_to_baseline: false,
        })
    }
}

impl SchedulerActorGemma4Drafter {
    fn new(
        drafter: Arc<Mutex<crate::models::gemma4::Gemma4AssistantModel>>,
        mtp_draft_tokens: usize,
    ) -> Self {
        debug_assert!(mtp_draft_tokens > 0);
        Self {
            drafter,
            cfg: MtpSpeculativeConfig {
                max_draft_tokens: mtp_draft_tokens,
            },
            exact_cost_controller: None,
            exact_episode: None,
        }
    }

    fn new_with_exact_qualification(
        drafter: Arc<Mutex<crate::models::gemma4::Gemma4AssistantModel>>,
        mtp_draft_tokens: usize,
        qualification: NeuralExactQualificationRuntimeConfig,
    ) -> Result<Self> {
        let mut mode = Self::new(drafter, mtp_draft_tokens);
        mode.exact_cost_controller = Some(NeuralExactCostController::new(qualification)?);
        Ok(mode)
    }

    fn publish_exact_qualification(&self, counters: &SchedulerActorMtpCounters) {
        if let Some(controller) = self.exact_cost_controller.as_ref() {
            counters.store_neural_exact_qualification_stats(controller.stats());
        }
    }

    fn finish_exact_episode(
        &mut self,
        stats_after: Option<MtpSpeculativeStats>,
        counters: &SchedulerActorMtpCounters,
    ) {
        finish_neural_exact_episode(
            &mut self.exact_episode,
            self.exact_cost_controller.as_mut(),
            stats_after,
        );
        self.publish_exact_qualification(counters);
    }
}

fn finish_neural_exact_episode(
    episode: &mut Option<NeuralExactMeasuredEpisode>,
    controller: Option<&mut NeuralExactCostController>,
    stats_after: Option<MtpSpeculativeStats>,
) {
    let (Some(episode), Some(controller)) = (episode.take(), controller) else {
        return;
    };
    let delta = stats_after
        .map(|stats| stats.saturating_delta_since(&episode.stats_before))
        .unwrap_or_default();
    controller.record_sample(
        episode.regime,
        episode.action,
        episode.elapsed_ns,
        episode.committed_tokens,
        NeuralExactSampleCounters {
            drafted_tokens: delta.drafted_tokens as u64,
            accepted_tokens: delta.accepted_draft_tokens as u64,
            exact_windows: delta.exact_sampling_windows as u64,
            residual_corrections: delta.exact_residual_corrections as u64,
        },
    );
}

impl<M> SchedulerActorMtpMode<M> for SchedulerActorNoMtp
where
    M: Model + DenseVlMethods,
{
    type MidAdmitHandle = AdmitMidHandle;

    fn mid_admit_request_id(handle: &Self::MidAdmitHandle) -> RequestId {
        handle.request_id
    }

    fn mid_admit_chunk_start(handle: &Self::MidAdmitHandle) -> i32 {
        handle.chunk_start
    }

    fn mid_admit_prompt_len(handle: &Self::MidAdmitHandle) -> i32 {
        handle.prompt_len
    }

    fn mid_admit_chunk_size(handle: &Self::MidAdmitHandle) -> i32 {
        handle.chunk_size
    }

    fn set_mid_admit_chunk_size(handle: &mut Self::MidAdmitHandle, chunk_size: i32) {
        handle.chunk_size = chunk_size;
    }

    fn mid_admit_decode_cadence_mid_chunk_cap(handle: &Self::MidAdmitHandle) -> usize {
        handle.decode_cadence_mid_chunk_cap
    }

    fn prefill_admitted(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        _counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>> {
        sched.prefill_admitted(model)
    }

    fn step(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        _counters: &SchedulerActorMtpCounters,
        _admission_pending: bool,
    ) -> Result<Vec<StepEvent>> {
        sched.step(model)
    }

    fn begin_mid_admit(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        request: GenerateRequest,
    ) -> Result<Self::MidAdmitHandle> {
        sched.admit_mid_begin(request, model)
    }

    fn advance_mid_admit_chunk(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        handle: &mut Self::MidAdmitHandle,
    ) -> Result<bool> {
        sched.admit_mid_chunk(handle, model)
    }

    fn finalize_mid_admit(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        handle: Self::MidAdmitHandle,
        _counters: &SchedulerActorMtpCounters,
    ) -> Result<(RequestId, StepEvent)> {
        sched.admit_mid_finalize(handle, model)
    }
}

impl<M> SchedulerActorMtpMode<M> for SchedulerActorPromptLookup
where
    M: Model + DenseVlMethods,
{
    type MidAdmitHandle = AdmitMidHandle;

    fn can_start_rolling_mid_admit(&self, sched: &Scheduler<M>) -> bool {
        sched.prompt_lookup_can_start_rolling_mid_admit()
    }

    fn mid_admit_request_id(handle: &Self::MidAdmitHandle) -> RequestId {
        handle.request_id
    }

    fn mid_admit_chunk_start(handle: &Self::MidAdmitHandle) -> i32 {
        handle.chunk_start
    }

    fn mid_admit_prompt_len(handle: &Self::MidAdmitHandle) -> i32 {
        handle.prompt_len
    }

    fn mid_admit_chunk_size(handle: &Self::MidAdmitHandle) -> i32 {
        handle.chunk_size
    }

    fn set_mid_admit_chunk_size(handle: &mut Self::MidAdmitHandle, chunk_size: i32) {
        handle.chunk_size = chunk_size;
    }

    fn mid_admit_decode_cadence_mid_chunk_cap(handle: &Self::MidAdmitHandle) -> usize {
        handle.decode_cadence_mid_chunk_cap
    }

    fn prefill_admitted(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>> {
        let events = sched.prefill_admitted_prompt_lookup(model, self.cfg)?;
        self.defer_speculation_once = true;
        self.measured_cycle = None;
        self.query_hint = None;
        self.miss_query_hint = None;
        self.publish_stats(sched, counters);
        Ok(events)
    }

    fn step(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
        admission_pending: bool,
    ) -> Result<Vec<StepEvent>> {
        let boundary_before = sched.prompt_lookup_can_start_rolling_mid_admit();
        let measured_cycle_active = self.measured_cycle.is_some();
        let decision = if let Some(cycle) = self.measured_cycle.as_ref() {
            PromptLookupWindowDecision {
                action: PromptLookupCostAction::Lookup,
                regime: Some(cycle.regime),
                proposal_elapsed_ns: 0,
                stats_before: cycle.stats_before,
                fallback_to_baseline: false,
            }
        } else {
            let force_ordinary = if self.defer_speculation_once {
                self.defer_speculation_once = false;
                true
            } else {
                prompt_lookup_admission_forces_ordinary(measured_cycle_active, admission_pending)
            };
            self.select_prepared_window(sched, model, true, false, false, force_ordinary)?
        };
        let action = decision.action;
        let regime = decision.regime;
        debug_assert!(
            self.measured_cycle.is_none() || !boundary_before,
            "PromptLookup measured cycle remained active at a window boundary"
        );
        let stats_before = decision.stats_before;
        let started = Instant::now();
        let events = match action {
            PromptLookupCostAction::Ordinary => sched.step_prompt_lookup_ordinary(model)?,
            PromptLookupCostAction::Lookup => sched.step_prompt_lookup(model)?,
        };
        let elapsed_ns =
            duration_ns(started.elapsed()).saturating_add(decision.proposal_elapsed_ns);
        let stats_after = sched.prompt_lookup_stats().unwrap_or_default();
        if let Some(regime) = regime {
            match action {
                PromptLookupCostAction::Ordinary => {
                    self.cost_controller.record_sample_with_adaptive_draft(
                        regime,
                        action,
                        elapsed_ns,
                        events.len(),
                        stats_after.saturating_delta_since(stats_before),
                    );
                }
                PromptLookupCostAction::Lookup => {
                    let cycle = self
                        .measured_cycle
                        .get_or_insert(PromptLookupMeasuredCycle {
                            regime,
                            action,
                            elapsed_ns: 0,
                            committed_tokens: 0,
                            stats_before,
                        });
                    cycle.elapsed_ns = cycle.elapsed_ns.saturating_add(elapsed_ns);
                    cycle.committed_tokens = cycle.committed_tokens.saturating_add(events.len());
                    if sched.prompt_lookup_can_start_rolling_mid_admit() {
                        let cycle = self
                            .measured_cycle
                            .take()
                            .expect("PromptLookup measured cycle initialized above");
                        self.cost_controller.record_sample_with_adaptive_draft(
                            cycle.regime,
                            action,
                            cycle.elapsed_ns,
                            cycle.committed_tokens,
                            stats_after.saturating_delta_since(cycle.stats_before),
                        );
                    }
                }
            }
        }
        self.publish_stats(sched, counters);
        Ok(events)
    }

    fn begin_mid_admit(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        request: GenerateRequest,
    ) -> Result<Self::MidAdmitHandle> {
        sched.admit_mid_begin(request, model)
    }

    fn advance_mid_admit_chunk(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        handle: &mut Self::MidAdmitHandle,
    ) -> Result<bool> {
        sched.admit_mid_chunk(handle, model)
    }

    fn finalize_mid_admit(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        handle: Self::MidAdmitHandle,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<(RequestId, StepEvent)> {
        let result = sched.admit_mid_finalize(handle, model)?;
        sched.register_prompt_lookup_request(result.0, self.cfg)?;
        self.defer_speculation_once = true;
        self.measured_cycle = None;
        self.query_hint = None;
        self.miss_query_hint = None;
        self.publish_stats(sched, counters);
        Ok(result)
    }
}

impl<M> SchedulerActorMtpMode<M> for SchedulerActorMtp<M::MtpHead>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel,
{
    type MidAdmitHandle = SchedulerActorMtpMidAdmitHandle;

    fn can_start_rolling_mid_admit(&self, sched: &Scheduler<M>) -> bool {
        sched.mtp_stats().is_none() || sched.mtp_at_batch_window_boundary()
    }

    fn mid_admit_request_id(handle: &Self::MidAdmitHandle) -> RequestId {
        match handle {
            SchedulerActorMtpMidAdmitHandle::Generic(handle) => handle.request_id,
            SchedulerActorMtpMidAdmitHandle::Mtp(handle) => handle.request_id,
        }
    }

    fn mid_admit_chunk_start(handle: &Self::MidAdmitHandle) -> i32 {
        match handle {
            SchedulerActorMtpMidAdmitHandle::Generic(handle) => handle.chunk_start,
            SchedulerActorMtpMidAdmitHandle::Mtp(handle) => handle.chunk_start,
        }
    }

    fn mid_admit_prompt_len(handle: &Self::MidAdmitHandle) -> i32 {
        match handle {
            SchedulerActorMtpMidAdmitHandle::Generic(handle) => handle.prompt_len,
            SchedulerActorMtpMidAdmitHandle::Mtp(handle) => handle.prompt_len,
        }
    }

    fn mid_admit_chunk_size(handle: &Self::MidAdmitHandle) -> i32 {
        match handle {
            SchedulerActorMtpMidAdmitHandle::Generic(handle) => handle.chunk_size,
            SchedulerActorMtpMidAdmitHandle::Mtp(handle) => handle.chunk_size,
        }
    }

    fn set_mid_admit_chunk_size(handle: &mut Self::MidAdmitHandle, chunk_size: i32) {
        match handle {
            SchedulerActorMtpMidAdmitHandle::Generic(handle) => handle.chunk_size = chunk_size,
            SchedulerActorMtpMidAdmitHandle::Mtp(handle) => handle.chunk_size = chunk_size,
        }
    }

    fn mid_admit_decode_cadence_mid_chunk_cap(handle: &Self::MidAdmitHandle) -> usize {
        match handle {
            SchedulerActorMtpMidAdmitHandle::Generic(handle) => handle.decode_cadence_mid_chunk_cap,
            SchedulerActorMtpMidAdmitHandle::Mtp(handle) => handle.decode_cadence_mid_chunk_cap,
        }
    }

    fn prefill_admitted(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>> {
        self.exact_episode = None;
        let regime = sched.fresh_neural_exact_qualification_regime(NeuralExactSource::QwenMtp);
        let action = regime
            .and_then(|regime| {
                self.exact_cost_controller
                    .as_mut()
                    .map(|controller| controller.next_action(regime))
            })
            .unwrap_or(NeuralExactAction::Exact);
        let stats_before = sched.mtp_stats().unwrap_or_default();
        let started = Instant::now();
        let events = if action == NeuralExactAction::Exact
            && sched.speculative_batch_active_fresh_eligible()
            && sched.native_mtp_fresh_exact_verify_eligible(model, self.cfg)?
        {
            counters.mtp_prefill_count.fetch_add(1, Ordering::Relaxed);
            let events = sched.prefill_admitted_mtp_batch(model, &self.mtp, self.cfg)?;
            counters.store_stats(sched.mtp_stats());
            events
        } else {
            counters
                .mtp_prefill_fallback_count
                .fetch_add(1, Ordering::Relaxed);
            sched.prefill_admitted(model)?
        };
        if let Some(regime) = regime.filter(|_| self.exact_cost_controller.is_some()) {
            self.exact_episode = Some(NeuralExactMeasuredEpisode {
                regime,
                action,
                elapsed_ns: duration_ns(started.elapsed()),
                committed_tokens: events.len(),
                stats_before,
            });
            if sched.active_batch_finished() {
                self.finish_exact_episode(sched.mtp_stats(), counters);
            }
        }
        self.publish_exact_qualification(counters);
        Ok(events)
    }

    fn step(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
        admission_pending: bool,
    ) -> Result<Vec<StepEvent>> {
        let action = self
            .exact_episode
            .as_ref()
            .map(|episode| episode.action)
            .unwrap_or(NeuralExactAction::Exact);
        if action == NeuralExactAction::Exact
            && sched.mtp_stats().is_some()
            && sched.mtp_at_batch_window_boundary()
            && !sched.native_mtp_next_window_exact_verify_eligible(model)?
        {
            counters.store_stats(sched.mtp_stats());
            sched.retire_mtp_at_batch_window_boundary()?;
        }
        let started = Instant::now();
        let events = if action == NeuralExactAction::Exact && sched.mtp_stats().is_some() {
            counters.mtp_step_count.fetch_add(1, Ordering::Relaxed);
            let events = if admission_pending {
                sched.step_mtp_batch_without_postfill(model, &self.mtp)?
            } else {
                sched.step_mtp_batch(model, &self.mtp)?
            };
            counters.store_stats(sched.mtp_stats());
            events
        } else {
            sched.step(model)?
        };
        if let Some(episode) = self.exact_episode.as_mut() {
            episode.elapsed_ns = episode
                .elapsed_ns
                .saturating_add(duration_ns(started.elapsed()));
            episode.committed_tokens = episode.committed_tokens.saturating_add(events.len());
        }
        if sched.active_batch_finished() {
            self.finish_exact_episode(sched.mtp_stats(), counters);
        }
        Ok(events)
    }

    fn begin_mid_admit(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        request: GenerateRequest,
    ) -> Result<Self::MidAdmitHandle> {
        self.exact_episode = None;
        if sched.mtp_stats().is_some() {
            if !sched.native_mtp_rolling_admit_exact_verify_eligible(model, self.cfg, &request)? {
                sched.retire_mtp_at_batch_window_boundary()?;
                return sched
                    .admit_mid_begin(request, model)
                    .map(SchedulerActorMtpMidAdmitHandle::Generic);
            }
            sched
                .admit_mid_begin_mtp(request, model, &self.mtp, self.cfg)
                .map(Box::new)
                .map(SchedulerActorMtpMidAdmitHandle::Mtp)
        } else {
            sched
                .admit_mid_begin(request, model)
                .map(SchedulerActorMtpMidAdmitHandle::Generic)
        }
    }

    fn advance_mid_admit_chunk(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        handle: &mut Self::MidAdmitHandle,
    ) -> Result<bool> {
        match handle {
            SchedulerActorMtpMidAdmitHandle::Generic(handle) => {
                sched.admit_mid_chunk(handle, model)
            }
            SchedulerActorMtpMidAdmitHandle::Mtp(handle) => {
                sched.admit_mid_chunk_mtp(handle.as_mut(), model, &self.mtp)
            }
        }
    }

    fn finalize_mid_admit(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        handle: Self::MidAdmitHandle,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<(RequestId, StepEvent)> {
        match handle {
            SchedulerActorMtpMidAdmitHandle::Generic(handle) => {
                sched.admit_mid_finalize(handle, model)
            }
            SchedulerActorMtpMidAdmitHandle::Mtp(handle) => {
                let result = sched.admit_mid_finalize_mtp(*handle, model, &self.mtp);
                counters.store_stats(sched.mtp_stats());
                result
            }
        }
    }
}

impl<M> SchedulerActorMtpMode<M> for SchedulerActorMtpPromptLookupHybrid<M::MtpHead>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel,
{
    type MidAdmitHandle = SchedulerActorMtpMidAdmitHandle;

    fn can_start_rolling_mid_admit(&self, sched: &Scheduler<M>) -> bool {
        self.measured_cycle.is_none()
            && (sched.mtp_stats().is_none() || sched.mtp_at_batch_window_boundary())
            && sched.prompt_lookup_can_start_rolling_mid_admit()
    }

    fn mid_admit_request_id(handle: &Self::MidAdmitHandle) -> RequestId {
        <SchedulerActorMtp<M::MtpHead> as SchedulerActorMtpMode<M>>::mid_admit_request_id(handle)
    }

    fn mid_admit_chunk_start(handle: &Self::MidAdmitHandle) -> i32 {
        <SchedulerActorMtp<M::MtpHead> as SchedulerActorMtpMode<M>>::mid_admit_chunk_start(handle)
    }

    fn mid_admit_prompt_len(handle: &Self::MidAdmitHandle) -> i32 {
        <SchedulerActorMtp<M::MtpHead> as SchedulerActorMtpMode<M>>::mid_admit_prompt_len(handle)
    }

    fn mid_admit_chunk_size(handle: &Self::MidAdmitHandle) -> i32 {
        <SchedulerActorMtp<M::MtpHead> as SchedulerActorMtpMode<M>>::mid_admit_chunk_size(handle)
    }

    fn set_mid_admit_chunk_size(handle: &mut Self::MidAdmitHandle, chunk_size: i32) {
        <SchedulerActorMtp<M::MtpHead> as SchedulerActorMtpMode<M>>::set_mid_admit_chunk_size(
            handle, chunk_size,
        );
    }

    fn mid_admit_decode_cadence_mid_chunk_cap(handle: &Self::MidAdmitHandle) -> usize {
        <SchedulerActorMtp<M::MtpHead> as SchedulerActorMtpMode<M>>::mid_admit_decode_cadence_mid_chunk_cap(handle)
    }

    fn prefill_admitted(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>> {
        let events = if sched.speculative_batch_active_fresh_eligible()
            && sched.native_mtp_fresh_exact_verify_eligible(model, self.neural.cfg)?
        {
            counters.mtp_prefill_count.fetch_add(1, Ordering::Relaxed);
            let events =
                sched.prefill_admitted_mtp_batch(model, &self.neural.mtp, self.neural.cfg)?;
            counters.store_stats(sched.mtp_stats());
            events
        } else {
            counters
                .mtp_prefill_fallback_count
                .fetch_add(1, Ordering::Relaxed);
            sched.prefill_admitted(model)?
        };
        sched.initialize_prompt_lookup_for_active(self.prompt_lookup.cfg)?;
        self.current_source = Some(HybridDraftSource::Neural);
        self.neural_dirty = false;
        self.lookup_window_canonical = false;
        self.measured_cycle = None;
        self.lookup_episode = None;
        self.prompt_lookup.query_hint = None;
        self.prompt_lookup.miss_query_hint = None;
        self.prompt_lookup.publish_stats(sched, counters);
        counters.store_prompt_lookup_hybrid_stats(self.stats);
        Ok(events)
    }

    fn step(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        counters: &SchedulerActorMtpCounters,
        admission_pending: bool,
    ) -> Result<Vec<StepEvent>> {
        if sched.mtp_stats().is_none() {
            let events = sched.step_prompt_lookup_ordinary(model)?;
            self.prompt_lookup.publish_stats(sched, counters);
            counters.store_prompt_lookup_hybrid_stats(self.stats);
            return Ok(events);
        }

        if sched.mtp_at_batch_window_boundary()
            && !sched.native_mtp_next_window_exact_verify_eligible(model)?
        {
            counters.store_stats(sched.mtp_stats());
            sched.retire_mtp_at_batch_window_boundary()?;
            let events = sched.step_prompt_lookup_ordinary(model)?;
            self.prompt_lookup.publish_stats(sched, counters);
            counters.store_prompt_lookup_hybrid_stats(self.stats);
            return Ok(events);
        }

        let mut source = self.current_source.unwrap_or(HybridDraftSource::Neural);
        let mut canonical_target = false;
        let at_boundary = match source {
            HybridDraftSource::Neural => sched.mtp_at_batch_window_boundary(),
            HybridDraftSource::PromptLookup => sched.prompt_lookup_can_start_rolling_mid_admit(),
        };
        let mut window_decision = None;

        if self.measured_cycle.is_none() && at_boundary {
            let decision = self.prompt_lookup.select_prepared_window(
                sched,
                model,
                true,
                true,
                true,
                admission_pending,
            )?;
            source = if decision.action == PromptLookupCostAction::Lookup {
                HybridDraftSource::PromptLookup
            } else {
                if decision.fallback_to_baseline {
                    self.stats.lookup_miss_fallbacks =
                        self.stats.lookup_miss_fallbacks.saturating_add(1);
                }
                sched.discard_prepared_prompt_lookup_window();
                HybridDraftSource::Neural
            };
            canonical_target = qwen_hybrid_uses_canonical_target(source, decision.regime);
            let previous_source = self.current_source;
            let state_source = if canonical_target {
                HybridDraftSource::Neural
            } else {
                source
            };
            if previous_source.is_some_and(|previous| previous != state_source) {
                self.stats.source_switches = self.stats.source_switches.saturating_add(1);
            }
            self.current_source = Some(state_source);
            if source == HybridDraftSource::PromptLookup {
                self.lookup_window_canonical = false;
            } else if previous_source == Some(HybridDraftSource::PromptLookup) && !self.neural_dirty
            {
                if let Some(episode) = self.lookup_episode.take() {
                    self.prompt_lookup.cost_controller.record_lookup_transition(
                        &episode.regimes,
                        0,
                        episode.committed_tokens,
                    );
                }
            }
            if (source == HybridDraftSource::Neural || canonical_target) && self.neural_dirty {
                let rebase_started = Instant::now();
                sched.rebase_mtp_from_committed_history(model, &self.neural.mtp)?;
                let rebase_elapsed = rebase_started.elapsed();
                self.stats.neural_rebases = self.stats.neural_rebases.saturating_add(1);
                self.stats.neural_rebase_us = self
                    .stats
                    .neural_rebase_us
                    .saturating_add(duration_us(rebase_elapsed));
                if let Some(episode) = self.lookup_episode.take() {
                    self.prompt_lookup.cost_controller.record_lookup_transition(
                        &episode.regimes,
                        duration_ns(rebase_elapsed),
                        episode.committed_tokens,
                    );
                }
                self.neural_dirty = false;
            }
            window_decision = Some(decision);
        }

        let regime = self
            .measured_cycle
            .as_ref()
            .map(|cycle| cycle.regime)
            .or_else(|| {
                window_decision
                    .as_ref()
                    .and_then(|decision| decision.regime)
            })
            .or_else(|| sched.prompt_lookup_qualification_regime());
        if self.measured_cycle.is_none() {
            if let Some(regime) = regime {
                self.measured_cycle = Some(PromptLookupMeasuredCycle {
                    regime,
                    action: window_decision
                        .as_ref()
                        .map_or(PromptLookupCostAction::Ordinary, |decision| decision.action),
                    elapsed_ns: window_decision
                        .as_ref()
                        .map_or(0, |decision| decision.proposal_elapsed_ns),
                    committed_tokens: 0,
                    stats_before: window_decision.as_ref().map_or_else(
                        || sched.prompt_lookup_stats().unwrap_or_default(),
                        |decision| decision.stats_before,
                    ),
                });
            }
        }

        counters.mtp_step_count.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let attempted_source = source;
        let events = match source {
            HybridDraftSource::Neural => {
                let events = sched.step_mtp_batch_without_postfill(model, &self.neural.mtp)?;
                sched.commit_prompt_lookup_events(&events)?;
                counters.store_stats(sched.mtp_stats());
                events
            }
            HybridDraftSource::PromptLookup => {
                if canonical_target {
                    let events = sched.step_mtp_batch_without_postfill_observing_prompt_lookup(
                        model,
                        &self.neural.mtp,
                    )?;
                    counters.store_stats(sched.mtp_stats());
                    self.neural_dirty = false;
                    self.lookup_window_canonical = true;
                    source = HybridDraftSource::Neural;
                    events
                } else {
                    match sched.step_prompt_lookup_with_mtp_verify(model, &self.neural.mtp)? {
                        PromptLookupMtpStepOutcome::Events {
                            events,
                            canonical_shared_full_accept,
                        } => {
                            self.lookup_window_canonical |= canonical_shared_full_accept;
                            events
                        }
                        PromptLookupMtpStepOutcome::FallbackToNeural => {
                            let events =
                                sched.step_mtp_batch_without_postfill(model, &self.neural.mtp)?;
                            sched.commit_prompt_lookup_events(&events)?;
                            counters.store_stats(sched.mtp_stats());
                            if self.current_source != Some(HybridDraftSource::Neural) {
                                self.stats.source_switches =
                                    self.stats.source_switches.saturating_add(1);
                            }
                            self.current_source = Some(HybridDraftSource::Neural);
                            self.neural_dirty = false;
                            self.lookup_window_canonical = false;
                            source = HybridDraftSource::Neural;
                            events
                        }
                    }
                }
            }
        };
        let elapsed_ns = duration_ns(started.elapsed());
        if let Some(cycle) = self.measured_cycle.as_mut() {
            cycle.elapsed_ns = cycle.elapsed_ns.saturating_add(elapsed_ns);
            cycle.committed_tokens = cycle.committed_tokens.saturating_add(events.len());
        }

        let completed = match source {
            HybridDraftSource::Neural => sched.mtp_at_batch_window_boundary(),
            HybridDraftSource::PromptLookup => sched.prompt_lookup_can_start_rolling_mid_admit(),
        };
        if completed {
            match source {
                HybridDraftSource::Neural => {
                    self.stats.neural_windows = self.stats.neural_windows.saturating_add(1);
                }
                HybridDraftSource::PromptLookup => {
                    self.stats.lookup_windows = self.stats.lookup_windows.saturating_add(1);
                    self.neural_dirty = true;
                    self.lookup_window_canonical = false;
                }
            }
            if let Some(cycle) = self.measured_cycle.take() {
                let stats_after = sched.prompt_lookup_stats().unwrap_or_default();
                self.prompt_lookup.cost_controller.record_sample(
                    cycle.regime,
                    cycle.action,
                    cycle.elapsed_ns,
                    cycle.committed_tokens,
                    stats_after.saturating_delta_since(cycle.stats_before),
                );
                if attempted_source == HybridDraftSource::PromptLookup
                    && source == HybridDraftSource::PromptLookup
                {
                    self.lookup_episode
                        .get_or_insert_with(PromptLookupEpisode::default)
                        .record(cycle.regime, cycle.committed_tokens);
                }
            }
        }

        self.prompt_lookup.publish_stats(sched, counters);
        counters.store_prompt_lookup_hybrid_stats(self.stats);
        Ok(events)
    }

    fn begin_mid_admit(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        request: GenerateRequest,
    ) -> Result<Self::MidAdmitHandle> {
        anyhow::ensure!(
            request.sampler.is_pipelinable(),
            "PromptLookup/neural hybrid requires greedy sampling"
        );
        if self.neural_dirty {
            let rebase_started = Instant::now();
            sched.rebase_mtp_from_committed_history(model, &self.neural.mtp)?;
            let rebase_elapsed = rebase_started.elapsed();
            self.stats.neural_rebases = self.stats.neural_rebases.saturating_add(1);
            self.stats.neural_rebase_us = self
                .stats
                .neural_rebase_us
                .saturating_add(duration_us(rebase_elapsed));
            if let Some(episode) = self.lookup_episode.take() {
                self.prompt_lookup.cost_controller.record_lookup_transition(
                    &episode.regimes,
                    duration_ns(rebase_elapsed),
                    episode.committed_tokens,
                );
            }
            self.neural_dirty = false;
        }
        self.current_source = Some(HybridDraftSource::Neural);
        self.lookup_window_canonical = false;
        self.measured_cycle = None;
        self.neural.begin_mid_admit(sched, model, request)
    }

    fn advance_mid_admit_chunk(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        handle: &mut Self::MidAdmitHandle,
    ) -> Result<bool> {
        self.neural.advance_mid_admit_chunk(sched, model, handle)
    }

    fn finalize_mid_admit(
        &mut self,
        sched: &mut Scheduler<M>,
        model: &M,
        handle: Self::MidAdmitHandle,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<(RequestId, StepEvent)> {
        let result = self
            .neural
            .finalize_mid_admit(sched, model, handle, counters)?;
        sched.register_prompt_lookup_request(result.0, self.prompt_lookup.cfg)?;
        self.current_source = Some(HybridDraftSource::Neural);
        self.lookup_window_canonical = false;
        self.measured_cycle = None;
        self.lookup_episode = None;
        self.prompt_lookup.query_hint = None;
        self.prompt_lookup.miss_query_hint = None;
        self.prompt_lookup.publish_stats(sched, counters);
        counters.store_prompt_lookup_hybrid_stats(self.stats);
        Ok(result)
    }
}

impl SchedulerActorMtpMode<crate::models::Gemma4Model> for SchedulerActorGemma4Drafter {
    type MidAdmitHandle = SchedulerActorGemma4DrafterMidAdmitHandle;

    fn mid_admit_request_id(handle: &Self::MidAdmitHandle) -> RequestId {
        match handle {
            SchedulerActorGemma4DrafterMidAdmitHandle::Generic(handle) => handle.request_id,
            SchedulerActorGemma4DrafterMidAdmitHandle::Drafter(handle) => handle.request_id,
        }
    }

    fn mid_admit_chunk_start(handle: &Self::MidAdmitHandle) -> i32 {
        match handle {
            SchedulerActorGemma4DrafterMidAdmitHandle::Generic(handle) => handle.chunk_start,
            SchedulerActorGemma4DrafterMidAdmitHandle::Drafter(handle) => handle.chunk_start,
        }
    }

    fn mid_admit_prompt_len(handle: &Self::MidAdmitHandle) -> i32 {
        match handle {
            SchedulerActorGemma4DrafterMidAdmitHandle::Generic(handle) => handle.prompt_len,
            SchedulerActorGemma4DrafterMidAdmitHandle::Drafter(handle) => handle.prompt_len,
        }
    }

    fn mid_admit_chunk_size(handle: &Self::MidAdmitHandle) -> i32 {
        match handle {
            SchedulerActorGemma4DrafterMidAdmitHandle::Generic(handle) => handle.chunk_size,
            SchedulerActorGemma4DrafterMidAdmitHandle::Drafter(handle) => handle.chunk_size,
        }
    }

    fn set_mid_admit_chunk_size(handle: &mut Self::MidAdmitHandle, chunk_size: i32) {
        match handle {
            SchedulerActorGemma4DrafterMidAdmitHandle::Generic(handle) => {
                handle.chunk_size = chunk_size;
            }
            SchedulerActorGemma4DrafterMidAdmitHandle::Drafter(handle) => {
                handle.chunk_size = chunk_size;
            }
        }
    }

    fn mid_admit_decode_cadence_mid_chunk_cap(handle: &Self::MidAdmitHandle) -> usize {
        match handle {
            SchedulerActorGemma4DrafterMidAdmitHandle::Generic(handle) => {
                handle.decode_cadence_mid_chunk_cap
            }
            SchedulerActorGemma4DrafterMidAdmitHandle::Drafter(handle) => {
                handle.decode_cadence_mid_chunk_cap
            }
        }
    }

    fn prefill_admitted(
        &mut self,
        sched: &mut Scheduler<crate::models::Gemma4Model>,
        model: &crate::models::Gemma4Model,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>> {
        self.exact_episode = None;
        let regime =
            sched.fresh_neural_exact_qualification_regime(NeuralExactSource::Gemma4Assistant);
        let action = regime
            .and_then(|regime| {
                self.exact_cost_controller
                    .as_mut()
                    .map(|controller| controller.next_action(regime))
            })
            .unwrap_or(NeuralExactAction::Exact);
        let stats_before = sched.gemma4_drafter_stats().unwrap_or_default();
        let started = Instant::now();
        let events = if action == NeuralExactAction::Exact
            && sched.speculative_batch_active_fresh_eligible()
        {
            counters.mtp_prefill_count.fetch_add(1, Ordering::Relaxed);
            let drafter = self.drafter.blocking_lock();
            let events = sched.prefill_admitted_gemma4_drafter_batch(model, &drafter, self.cfg)?;
            counters.store_stats(sched.gemma4_drafter_stats());
            events
        } else {
            counters
                .mtp_prefill_fallback_count
                .fetch_add(1, Ordering::Relaxed);
            sched.prefill_admitted(model)?
        };
        if let Some(regime) = regime.filter(|_| self.exact_cost_controller.is_some()) {
            self.exact_episode = Some(NeuralExactMeasuredEpisode {
                regime,
                action,
                elapsed_ns: duration_ns(started.elapsed()),
                committed_tokens: events.len(),
                stats_before,
            });
            if sched.active_batch_finished() {
                self.finish_exact_episode(sched.gemma4_drafter_stats(), counters);
            }
        }
        self.publish_exact_qualification(counters);
        Ok(events)
    }

    fn step(
        &mut self,
        sched: &mut Scheduler<crate::models::Gemma4Model>,
        model: &crate::models::Gemma4Model,
        counters: &SchedulerActorMtpCounters,
        _admission_pending: bool,
    ) -> Result<Vec<StepEvent>> {
        let action = self
            .exact_episode
            .as_ref()
            .map(|episode| episode.action)
            .unwrap_or(NeuralExactAction::Exact);
        let started = Instant::now();
        let events = if action == NeuralExactAction::Exact && sched.gemma4_drafter_stats().is_some()
        {
            counters.mtp_step_count.fetch_add(1, Ordering::Relaxed);
            let drafter = self.drafter.blocking_lock();
            let events = sched.step_gemma4_drafter_batch(model, &drafter)?;
            counters.store_stats(sched.gemma4_drafter_stats());
            events
        } else {
            sched.step(model)?
        };
        if let Some(episode) = self.exact_episode.as_mut() {
            episode.elapsed_ns = episode
                .elapsed_ns
                .saturating_add(duration_ns(started.elapsed()));
            episode.committed_tokens = episode.committed_tokens.saturating_add(events.len());
        }
        if sched.active_batch_finished() {
            self.finish_exact_episode(sched.gemma4_drafter_stats(), counters);
        }
        Ok(events)
    }

    fn begin_mid_admit(
        &mut self,
        sched: &mut Scheduler<crate::models::Gemma4Model>,
        model: &crate::models::Gemma4Model,
        request: GenerateRequest,
    ) -> Result<Self::MidAdmitHandle> {
        self.exact_episode = None;
        if sched.gemma4_drafter_stats().is_some() {
            sched
                .admit_mid_begin_gemma4_drafter(request, model)
                .map(Box::new)
                .map(SchedulerActorGemma4DrafterMidAdmitHandle::Drafter)
        } else {
            sched
                .admit_mid_begin(request, model)
                .map(SchedulerActorGemma4DrafterMidAdmitHandle::Generic)
        }
    }

    fn advance_mid_admit_chunk(
        &mut self,
        sched: &mut Scheduler<crate::models::Gemma4Model>,
        model: &crate::models::Gemma4Model,
        handle: &mut Self::MidAdmitHandle,
    ) -> Result<bool> {
        match handle {
            SchedulerActorGemma4DrafterMidAdmitHandle::Generic(handle) => {
                sched.admit_mid_chunk(handle, model)
            }
            SchedulerActorGemma4DrafterMidAdmitHandle::Drafter(handle) => {
                sched.admit_mid_chunk_gemma4_drafter(handle.as_mut(), model)
            }
        }
    }

    fn finalize_mid_admit(
        &mut self,
        sched: &mut Scheduler<crate::models::Gemma4Model>,
        model: &crate::models::Gemma4Model,
        handle: Self::MidAdmitHandle,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<(RequestId, StepEvent)> {
        match handle {
            SchedulerActorGemma4DrafterMidAdmitHandle::Generic(handle) => {
                sched.admit_mid_finalize(handle, model)
            }
            SchedulerActorGemma4DrafterMidAdmitHandle::Drafter(handle) => {
                let result = sched.admit_mid_finalize_gemma4_drafter(*handle, model);
                counters.store_stats(sched.gemma4_drafter_stats());
                result
            }
        }
    }
}

impl SchedulerActorMtpMode<crate::models::Gemma4Model> for SchedulerActorGemma4PromptLookupHybrid {
    type MidAdmitHandle = SchedulerActorGemma4DrafterMidAdmitHandle;

    fn can_start_rolling_mid_admit(&self, sched: &Scheduler<crate::models::Gemma4Model>) -> bool {
        self.measured_cycle.is_none()
            && sched.gemma4_drafter_at_batch_window_boundary()
            && sched.prompt_lookup_can_start_rolling_mid_admit()
    }

    fn mid_admit_request_id(handle: &Self::MidAdmitHandle) -> RequestId {
        <SchedulerActorGemma4Drafter as SchedulerActorMtpMode<
            crate::models::Gemma4Model,
        >>::mid_admit_request_id(handle)
    }

    fn mid_admit_chunk_start(handle: &Self::MidAdmitHandle) -> i32 {
        <SchedulerActorGemma4Drafter as SchedulerActorMtpMode<
            crate::models::Gemma4Model,
        >>::mid_admit_chunk_start(handle)
    }

    fn mid_admit_prompt_len(handle: &Self::MidAdmitHandle) -> i32 {
        <SchedulerActorGemma4Drafter as SchedulerActorMtpMode<
            crate::models::Gemma4Model,
        >>::mid_admit_prompt_len(handle)
    }

    fn mid_admit_chunk_size(handle: &Self::MidAdmitHandle) -> i32 {
        <SchedulerActorGemma4Drafter as SchedulerActorMtpMode<
            crate::models::Gemma4Model,
        >>::mid_admit_chunk_size(handle)
    }

    fn set_mid_admit_chunk_size(handle: &mut Self::MidAdmitHandle, chunk_size: i32) {
        <SchedulerActorGemma4Drafter as SchedulerActorMtpMode<
            crate::models::Gemma4Model,
        >>::set_mid_admit_chunk_size(handle, chunk_size);
    }

    fn mid_admit_decode_cadence_mid_chunk_cap(handle: &Self::MidAdmitHandle) -> usize {
        <SchedulerActorGemma4Drafter as SchedulerActorMtpMode<
            crate::models::Gemma4Model,
        >>::mid_admit_decode_cadence_mid_chunk_cap(handle)
    }

    fn prefill_admitted(
        &mut self,
        sched: &mut Scheduler<crate::models::Gemma4Model>,
        model: &crate::models::Gemma4Model,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<Vec<StepEvent>> {
        let events = self.neural.prefill_admitted(sched, model, counters)?;
        if sched.gemma4_drafter_stats().is_some() {
            sched.initialize_prompt_lookup_for_active(self.prompt_lookup.cfg)?;
            self.current_source = Some(HybridDraftSource::Neural);
            self.neural_dirty = false;
            self.measured_cycle = None;
            self.lookup_episode = None;
            self.prompt_lookup.query_hint = None;
            self.prompt_lookup.miss_query_hint = None;
            self.prompt_lookup.publish_stats(sched, counters);
            counters.store_prompt_lookup_hybrid_stats(self.stats);
        }
        Ok(events)
    }

    fn step(
        &mut self,
        sched: &mut Scheduler<crate::models::Gemma4Model>,
        model: &crate::models::Gemma4Model,
        counters: &SchedulerActorMtpCounters,
        admission_pending: bool,
    ) -> Result<Vec<StepEvent>> {
        if sched.gemma4_drafter_stats().is_none() {
            return sched.step(model);
        }

        let mut source = self.current_source.unwrap_or(HybridDraftSource::Neural);
        let at_boundary = match source {
            HybridDraftSource::Neural => sched.gemma4_drafter_at_batch_window_boundary(),
            HybridDraftSource::PromptLookup => sched.prompt_lookup_can_start_rolling_mid_admit(),
        };
        let mut window_decision = None;

        if self.measured_cycle.is_none() && at_boundary {
            let decision = self.prompt_lookup.select_prepared_window(
                sched,
                model,
                true,
                false,
                false,
                admission_pending,
            )?;
            source = if decision.action == PromptLookupCostAction::Lookup {
                HybridDraftSource::PromptLookup
            } else {
                if decision.fallback_to_baseline {
                    self.stats.lookup_miss_fallbacks =
                        self.stats.lookup_miss_fallbacks.saturating_add(1);
                }
                sched.discard_prepared_prompt_lookup_window();
                HybridDraftSource::Neural
            };
            if self
                .current_source
                .is_some_and(|previous| previous != source)
            {
                self.stats.source_switches = self.stats.source_switches.saturating_add(1);
            }
            self.current_source = Some(source);
            if source == HybridDraftSource::Neural && self.neural_dirty {
                let rebase_started = Instant::now();
                let drafter = self.neural.drafter.blocking_lock();
                sched.rebase_gemma4_drafter_from_committed_history(model, &drafter)?;
                let rebase_elapsed = rebase_started.elapsed();
                self.stats.neural_rebases = self.stats.neural_rebases.saturating_add(1);
                self.stats.neural_rebase_us = self
                    .stats
                    .neural_rebase_us
                    .saturating_add(duration_us(rebase_elapsed));
                if let Some(episode) = self.lookup_episode.take() {
                    self.prompt_lookup.cost_controller.record_lookup_transition(
                        &episode.regimes,
                        duration_ns(rebase_elapsed),
                        episode.committed_tokens,
                    );
                }
                self.neural_dirty = false;
            }
            window_decision = Some(decision);
        }

        let regime = self
            .measured_cycle
            .as_ref()
            .map(|cycle| cycle.regime)
            .or_else(|| {
                window_decision
                    .as_ref()
                    .and_then(|decision| decision.regime)
            })
            .or_else(|| sched.prompt_lookup_qualification_regime());
        if self.measured_cycle.is_none() {
            if let Some(regime) = regime {
                self.measured_cycle = Some(PromptLookupMeasuredCycle {
                    regime,
                    action: window_decision
                        .as_ref()
                        .map_or(PromptLookupCostAction::Ordinary, |decision| decision.action),
                    elapsed_ns: window_decision
                        .as_ref()
                        .map_or(0, |decision| decision.proposal_elapsed_ns),
                    committed_tokens: 0,
                    stats_before: window_decision.as_ref().map_or_else(
                        || sched.prompt_lookup_stats().unwrap_or_default(),
                        |decision| decision.stats_before,
                    ),
                });
            }
        }

        counters.mtp_step_count.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let events = match source {
            HybridDraftSource::Neural => {
                let drafter = self.neural.drafter.blocking_lock();
                let events = sched.step_gemma4_drafter_batch_without_postfill(model, &drafter)?;
                drop(drafter);
                sched.commit_prompt_lookup_events(&events)?;
                counters.store_stats(sched.gemma4_drafter_stats());
                events
            }
            HybridDraftSource::PromptLookup => sched.step_prompt_lookup_batch_window(model)?,
        };
        let elapsed_ns = duration_ns(started.elapsed());
        if let Some(cycle) = self.measured_cycle.as_mut() {
            cycle.elapsed_ns = cycle.elapsed_ns.saturating_add(elapsed_ns);
            cycle.committed_tokens = cycle.committed_tokens.saturating_add(events.len());
        }

        let completed = match source {
            HybridDraftSource::Neural => sched.gemma4_drafter_at_batch_window_boundary(),
            HybridDraftSource::PromptLookup => sched.prompt_lookup_can_start_rolling_mid_admit(),
        };
        if completed {
            match source {
                HybridDraftSource::Neural => {
                    self.stats.neural_windows = self.stats.neural_windows.saturating_add(1);
                }
                HybridDraftSource::PromptLookup => {
                    self.stats.lookup_windows = self.stats.lookup_windows.saturating_add(1);
                    self.neural_dirty = true;
                }
            }
            if let Some(cycle) = self.measured_cycle.take() {
                let stats_after = sched.prompt_lookup_stats().unwrap_or_default();
                self.prompt_lookup
                    .cost_controller
                    .record_sample_with_adaptive_draft(
                        cycle.regime,
                        cycle.action,
                        cycle.elapsed_ns,
                        cycle.committed_tokens,
                        stats_after.saturating_delta_since(cycle.stats_before),
                    );
                if source == HybridDraftSource::PromptLookup {
                    self.lookup_episode
                        .get_or_insert_with(PromptLookupEpisode::default)
                        .record(cycle.regime, cycle.committed_tokens);
                }
            }
        }

        self.prompt_lookup.publish_stats(sched, counters);
        counters.store_prompt_lookup_hybrid_stats(self.stats);
        Ok(events)
    }

    fn begin_mid_admit(
        &mut self,
        sched: &mut Scheduler<crate::models::Gemma4Model>,
        model: &crate::models::Gemma4Model,
        request: GenerateRequest,
    ) -> Result<Self::MidAdmitHandle> {
        anyhow::ensure!(
            request.sampler.is_pipelinable(),
            "PromptLookup/neural hybrid requires greedy sampling"
        );
        self.neural.begin_mid_admit(sched, model, request)
    }

    fn advance_mid_admit_chunk(
        &mut self,
        sched: &mut Scheduler<crate::models::Gemma4Model>,
        model: &crate::models::Gemma4Model,
        handle: &mut Self::MidAdmitHandle,
    ) -> Result<bool> {
        self.neural.advance_mid_admit_chunk(sched, model, handle)
    }

    fn finalize_mid_admit(
        &mut self,
        sched: &mut Scheduler<crate::models::Gemma4Model>,
        model: &crate::models::Gemma4Model,
        handle: Self::MidAdmitHandle,
        counters: &SchedulerActorMtpCounters,
    ) -> Result<(RequestId, StepEvent)> {
        let result = self
            .neural
            .finalize_mid_admit(sched, model, handle, counters)?;
        sched.register_prompt_lookup_request(result.0, self.prompt_lookup.cfg)?;
        if self.neural_dirty {
            let rebase_started = Instant::now();
            let drafter = self.neural.drafter.blocking_lock();
            sched.rebase_gemma4_drafter_from_committed_history(model, &drafter)?;
            let rebase_elapsed = rebase_started.elapsed();
            self.stats.neural_rebases = self.stats.neural_rebases.saturating_add(1);
            self.stats.neural_rebase_us = self
                .stats
                .neural_rebase_us
                .saturating_add(duration_us(rebase_elapsed));
            if let Some(episode) = self.lookup_episode.take() {
                self.prompt_lookup.cost_controller.record_lookup_transition(
                    &episode.regimes,
                    duration_ns(rebase_elapsed),
                    episode.committed_tokens,
                );
            }
            self.neural_dirty = false;
        }
        self.current_source = Some(HybridDraftSource::Neural);
        self.measured_cycle = None;
        self.lookup_episode = None;
        self.prompt_lookup.query_hint = None;
        self.prompt_lookup.miss_query_hint = None;
        self.prompt_lookup.publish_stats(sched, counters);
        counters.store_prompt_lookup_hybrid_stats(self.stats);
        Ok(result)
    }
}

/// Result returned by [`drive_empty_scheduler_handoff`] encoding what the
/// caller's rolling loop should do next. Matches the existing `continue
/// 'rolling` / `break 'rolling` / `continue 'outer` / `return` patterns
/// without exposing label control to the helper.
///
/// Keeps the empty-batch handoff path reusable from both the existing
/// post-step empty-handoff site and the pre-event Finished-batch
/// finalization at the rolling-loop top.
enum RollingControl {
    /// Re-enter the rolling loop (a new batch was admitted + prefilled).
    ContinueRolling,
    /// Exit the rolling loop into the outer-loop tail cleanup (no
    /// queued or pending admits; outer will block on `cmd_rx.recv()`).
    BreakRolling,
    /// `continue 'outer` — outer loop body resumes from its top
    /// (e.g., poisoned-state recovery).
    ContinueOuter,
    /// `return` from the actor (cmd_rx disconnected; all senders dropped).
    ReturnActor,
}

/// Reply payload for [`SchedulerCommand::Admit`]. Carries the assigned
/// [`RequestId`] and the per-request event receiver.
pub struct AdmitReply {
    pub request_id: RequestId,
    pub event_rx: mpsc::UnboundedReceiver<StepEvent>,
}

/// Handle held by [`crate::core::server::AppState`]. Cheap to clone
/// (`mpsc::Sender` and `Arc<AtomicU64>` are both `Clone`).
#[derive(Clone)]
pub struct SchedulerActorHandle {
    pub cmd_tx: mpsc::Sender<SchedulerCommand>,
    pub(super) control_tx: mpsc::Sender<SchedulerControlCommand>,
    pub(crate) cold_materialization_tracker:
        Arc<OnceLock<Arc<crate::core::process_memory::ColdMaterializationTracker>>>,
    pub(crate) runtime_usage: Arc<crate::core::runtime_usage::ModelRuntimeUsageCounters>,
    /// Test-observable counter. Incremented by the driver every time
    /// `Scheduler::admit` succeeds. Doc-hidden because production code
    /// shouldn't read it — it exists for integration tests to assert
    /// routing decisions (e.g., "VL request did NOT increment the
    /// counter, so it took the GS path"). Cost: one atomic load per
    /// successful admit.
    #[doc(hidden)]
    pub admit_count: Arc<AtomicU64>,
    /// Test-observable counter. Incremented by the driver once per
    /// batch (prefill_admitted invocation, including failed batches —
    /// diagnostic purpose). When multi-admit batching is working,
    /// integration tests expect `batch_count < admit_count`. Doc-hidden.
    #[doc(hidden)]
    pub batch_count: Arc<AtomicU64>,
    /// Test-observable counter. Incremented by `drain_window` when it
    /// exits because `Scheduler::active_count() >= b_max` (saturate path),
    /// NOT when the deadline expires. Used by integration tests to prove
    /// the saturate-trigger fired without relying on wall-time measurement.
    /// Doc-hidden.
    #[doc(hidden)]
    pub saturate_triggered: Arc<AtomicU64>,
    /// Test-observable peak `admission_queue.len()` ever reached. Used by
    /// integration tests to confirm the queue drained (e.g., `peak >= N` for
    /// c=N+b_max admit burst). Doc-hidden — production code shouldn't read it.
    #[doc(hidden)]
    pub queue_depth_peak: Arc<AtomicUsize>,
    /// Test-observable count of admit requests rejected with "admission
    /// queue full" Err (queue_max overflow). Doc-hidden.
    #[doc(hidden)]
    pub queue_rejected: Arc<AtomicU64>,
    /// Count of actor calls to scheduler-internal MTP prefill. Exposed through
    /// `/healthz.mtp.prefill_count` for server-level diagnostics.
    #[doc(hidden)]
    pub mtp_prefill_count: Arc<AtomicU64>,
    /// Count of actor calls to scheduler-internal MTP step. Exposed through
    /// `/healthz.mtp.step_count` for server-level diagnostics.
    #[doc(hidden)]
    pub mtp_step_count: Arc<AtomicU64>,
    /// Count of MTP-enabled prefill calls that fell back to the ordinary
    /// scheduler path because the active batch was not MTP-eligible.
    #[doc(hidden)]
    pub mtp_fallback_prefill_count: Arc<AtomicU64>,
    /// Latest cumulative scheduler MTP drafted-token count.
    #[doc(hidden)]
    pub mtp_drafted_tokens: Arc<AtomicU64>,
    /// Latest cumulative scheduler MTP accepted-draft-token count.
    #[doc(hidden)]
    pub mtp_accepted_draft_tokens: Arc<AtomicU64>,
    /// Latest cumulative scheduler MTP speculative-window count.
    #[doc(hidden)]
    pub mtp_windows: Arc<AtomicU64>,
    #[doc(hidden)]
    pub mtp_exact_sampling_windows: Arc<AtomicU64>,
    #[doc(hidden)]
    pub mtp_exact_acceptance_draws: Arc<AtomicU64>,
    #[doc(hidden)]
    pub mtp_exact_residual_corrections: Arc<AtomicU64>,
    #[doc(hidden)]
    pub mtp_exact_bonus_samples: Arc<AtomicU64>,
    /// Latest cumulative microseconds spent in MTP draft forward passes.
    #[doc(hidden)]
    pub mtp_draft_forward_us: Arc<AtomicU64>,
    /// Latest cumulative microseconds spent in main verifier forward passes.
    #[doc(hidden)]
    pub mtp_verify_forward_us: Arc<AtomicU64>,
    /// Latest cumulative microseconds spent projecting verifier hidden states.
    #[doc(hidden)]
    pub mtp_projection_us: Arc<AtomicU64>,
    /// Latest cumulative microseconds spent sampling logits.
    #[doc(hidden)]
    pub mtp_sampling_us: Arc<AtomicU64>,
    /// Host synchronizations performed while constructing MTP draft chains.
    #[doc(hidden)]
    pub mtp_draft_host_sync_count: Arc<AtomicU64>,
    /// Microseconds blocked on draft-chain host synchronization.
    #[doc(hidden)]
    pub mtp_draft_host_sync_us: Arc<AtomicU64>,
    /// Host synchronizations performed to resolve verified MTP windows.
    #[doc(hidden)]
    pub mtp_verify_accept_host_sync_count: Arc<AtomicU64>,
    /// Microseconds blocked on compact verify-acceptance results.
    #[doc(hidden)]
    pub mtp_verify_accept_host_sync_us: Arc<AtomicU64>,
    /// Latest cumulative microseconds spent rolling back/replaying main KV.
    #[doc(hidden)]
    pub mtp_main_rollback_us: Arc<AtomicU64>,
    /// Latest cumulative microseconds spent committing accepted tokens to MTP KV.
    #[doc(hidden)]
    pub mtp_cache_commit_us: Arc<AtomicU64>,
    /// Latest cumulative microseconds spent building MTP KV during prefill.
    #[doc(hidden)]
    pub mtp_prefill_cache_commit_us: Arc<AtomicU64>,
    /// Latest cumulative microseconds spent committing accepted decode tokens to MTP KV.
    #[doc(hidden)]
    pub mtp_decode_cache_commit_us: Arc<AtomicU64>,
    /// Latest cumulative microseconds spent restoring temporary MTP KV.
    #[doc(hidden)]
    pub mtp_cache_restore_us: Arc<AtomicU64>,
    pub(crate) prompt_lookup_published_stats: Arc<StdMutex<Option<PromptLookupStats>>>,
    pub(crate) neural_exact_qualification_stats: Arc<StdMutex<NeuralExactQualificationStats>>,
    // ── B1-p2.5 G3: /healthz monitoring atomics ──────────────────────────
    /// Live count of in-flight (active) requests in the scheduler slots.
    /// Updated by driver_loop tail on every rolling iteration.
    pub b_active: Arc<AtomicU64>,
    /// Live count of requests parked in the admission queue.
    /// Updated by driver_loop tail on every rolling iteration.
    pub b_queued: Arc<AtomicU64>,
    /// Monotonic count of admits rejected due to admission queue full.
    /// Aliased from `queue_rejected` — single source of truth in driver_loop.
    /// P1.1: Scheduler.admission_queue_full_count field removed (no fetch_add
    /// caller); health collector now reads from this Arc directly. B1-p2.5.
    pub admission_queue_full_count: Arc<AtomicU64>,
    /// Monotonic count of admits rejected due to memory budget exceeded.
    /// Cloned from Scheduler::memory_budget_exceeded_count.
    pub memory_budget_exceeded_count: Arc<AtomicU64>,
    /// Shared Arc into BudgetState::active — live bytes charged to KV cache.
    pub kv_cache_active_bytes: Arc<AtomicUsize>,
    /// KV cache soft limit in bytes (computed at startup; static for lifetime).
    pub kv_cache_soft_limit_bytes: usize,
    /// Logical per-request KV cache cap in tokens.
    pub kv_cache_logical_cap_tokens: usize,
    /// Hot-resident per-request KV cache cap charged to memory budget.
    pub kv_cache_resident_cap_tokens: usize,
    /// Budget policy name used by startup and runtime KV admission.
    pub kv_cache_budget_policy: &'static str,
    /// Shared Active KV offload metrics and runtime status.
    pub active_kv_offload: ActiveKvOffloadSharedStats,
    /// FullPaged immutable content-addressed prefix block pool metrics.
    pub immutable_prefix_blocks: ImmutablePrefixBlockSharedStats,
}

impl SchedulerActorHandle {
    pub fn prompt_lookup_stats(&self) -> Option<PromptLookupStats> {
        *self
            .prompt_lookup_published_stats
            .lock()
            .expect("PromptLookup published stats mutex poisoned")
    }

    pub(crate) fn install_cold_materialization_tracker(
        &self,
        tracker: Arc<crate::core::process_memory::ColdMaterializationTracker>,
    ) -> Result<()> {
        self.cold_materialization_tracker
            .set(tracker)
            .map_err(|_| anyhow::anyhow!("cold materialization tracker already installed"))
    }

    pub async fn clear_shared_prompt_lookup(&self) -> Result<usize> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.control_tx
            .send(SchedulerControlCommand::ClearSharedPromptLookup { reply_tx })
            .await
            .map_err(|_| anyhow::anyhow!("scheduler control channel is closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("scheduler stopped before clearing PromptLookup history"))
    }
}

/// Spawn the driver task and return a handle. The driver runs on
/// `tokio::task::spawn_blocking` because [`Scheduler`] is `!Send`
/// (holds Array fields: KVCache, prng_state) and the model lock is sync.
///
/// # Arguments
/// - `model` — shared model handle (Mutex-protected sync state).
/// - `b_max` — maximum concurrent in-flight requests (Scheduler slot count).
/// - `admission_deadline` — drain-window timeout after the first admit in a
///   batch arrives. Hard limit; new admits do not reset it.
/// - `admission_queue_max` — capacity of the FIFO admission queue. `0`
///   disables queueing (immediate Err on saturation, mirroring pre-3d).
/// - `effective_cap_max` — upper bound on `prompt_len + max_new_tokens`
///   per request. Computed at boot as
///   `min(--max-cache-cap CLI, model.config.max_position_embeddings)`.
///   Passed directly to `Scheduler::new`. B1-p2.3f.
/// - `decode_cadence_mid_chunk_cap` — maximum rolling mid-admit chunk size
///   while existing decode rows are active.
/// - `meta` — model memory-budget metadata for startup validation. B1-p2.5.
pub fn spawn_scheduler_actor<M>(
    model: Arc<Mutex<M>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    spawn_scheduler_actor_with_mode(
        model,
        SchedulerActorNoMtp,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        None,
        None,
        AdaptiveAdmissionPolicy::disabled(),
        ActiveKvOffloadConfig::disabled(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_scheduler_actor_for_prompt_lookup_control<M>(
    model: Arc<Mutex<M>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    spawn_scheduler_actor_with_mode(
        model,
        SchedulerActorNoMtp,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        paged_prefix_cache,
        prefix_lru_cache,
        AdaptiveAdmissionPolicy::prompt_lookup(),
        active_kv_offload,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_scheduler_actor_with_active_kv_offload<M>(
    model: Arc<Mutex<M>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    spawn_scheduler_actor_with_mode(
        model,
        SchedulerActorNoMtp,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        None,
        None,
        AdaptiveAdmissionPolicy::disabled(),
        active_kv_offload,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_scheduler_actor_with_paged_prefix_cache<M>(
    model: Arc<Mutex<M>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: PagedPrefixCacheConfig,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    spawn_scheduler_actor_with_mode(
        model,
        SchedulerActorNoMtp,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        Some(paged_prefix_cache),
        prefix_lru_cache,
        AdaptiveAdmissionPolicy::disabled(),
        ActiveKvOffloadConfig::disabled(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_scheduler_actor_with_paged_prefix_cache_and_active_kv<M>(
    model: Arc<Mutex<M>>,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: PagedPrefixCacheConfig,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    spawn_scheduler_actor_with_mode(
        model,
        SchedulerActorNoMtp,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        Some(paged_prefix_cache),
        prefix_lru_cache,
        AdaptiveAdmissionPolicy::disabled(),
        active_kv_offload,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_scheduler_actor_with_mtp<M>(
    model: Arc<Mutex<M>>,
    mtp: M::MtpHead,
    mtp_draft_tokens: usize,
    exact_qualification: NeuralExactQualificationRuntimeConfig,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
) -> Result<SchedulerActorHandle>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    let mode = SchedulerActorMtp::new_with_exact_qualification(
        mtp,
        mtp_draft_tokens,
        exact_qualification,
    )?;
    Ok(spawn_scheduler_actor_with_mode(
        model,
        mode,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        paged_prefix_cache,
        prefix_lru_cache,
        AdaptiveAdmissionPolicy::qwen_mtp(),
        ActiveKvOffloadConfig::disabled(),
    )?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_scheduler_actor_with_mtp_prompt_lookup<M>(
    model: Arc<Mutex<M>>,
    mtp: M::MtpHead,
    mtp_draft_tokens: usize,
    prompt_lookup: PromptLookupConfig,
    qualification: PromptLookupQualificationRuntimeConfig,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    let mode = SchedulerActorMtpPromptLookupHybrid::new(
        mtp,
        mtp_draft_tokens,
        prompt_lookup,
        qualification,
    )?;
    Ok(spawn_scheduler_actor_with_mode(
        model,
        mode,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        paged_prefix_cache,
        prefix_lru_cache,
        AdaptiveAdmissionPolicy::qwen_mtp(),
        active_kv_offload,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_scheduler_actor_with_prompt_lookup<M>(
    model: Arc<Mutex<M>>,
    cfg: PromptLookupConfig,
    qualification: PromptLookupQualificationRuntimeConfig,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let mode = SchedulerActorPromptLookup::new(cfg, qualification)?;
    Ok(spawn_scheduler_actor_with_mode(
        model,
        mode,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        paged_prefix_cache,
        prefix_lru_cache,
        AdaptiveAdmissionPolicy::prompt_lookup(),
        active_kv_offload,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_scheduler_actor_with_mtp_and_active_kv<M>(
    model: Arc<Mutex<M>>,
    mtp: M::MtpHead,
    mtp_draft_tokens: usize,
    exact_qualification: NeuralExactQualificationRuntimeConfig,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    let mode = SchedulerActorMtp::new_with_exact_qualification(
        mtp,
        mtp_draft_tokens,
        exact_qualification,
    )?;
    Ok(spawn_scheduler_actor_with_mode(
        model,
        mode,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        paged_prefix_cache,
        prefix_lru_cache,
        AdaptiveAdmissionPolicy::qwen_mtp(),
        active_kv_offload,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_scheduler_actor_with_gemma4_drafter(
    model: Arc<Mutex<crate::models::Gemma4Model>>,
    drafter: Arc<Mutex<crate::models::gemma4::Gemma4AssistantModel>>,
    mtp_draft_tokens: usize,
    exact_qualification: NeuralExactQualificationRuntimeConfig,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
) -> Result<SchedulerActorHandle> {
    let mode = SchedulerActorGemma4Drafter::new_with_exact_qualification(
        drafter,
        mtp_draft_tokens,
        exact_qualification,
    )?;
    Ok(spawn_scheduler_actor_with_mode(
        model,
        mode,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        paged_prefix_cache,
        prefix_lru_cache,
        AdaptiveAdmissionPolicy::gemma4_drafter(),
        ActiveKvOffloadConfig::disabled(),
    )?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_scheduler_actor_with_gemma4_drafter_prompt_lookup(
    model: Arc<Mutex<crate::models::Gemma4Model>>,
    drafter: Arc<Mutex<crate::models::gemma4::Gemma4AssistantModel>>,
    mtp_draft_tokens: usize,
    prompt_lookup: PromptLookupConfig,
    qualification: PromptLookupQualificationRuntimeConfig,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle> {
    let mode = SchedulerActorGemma4PromptLookupHybrid::new(
        drafter,
        mtp_draft_tokens,
        prompt_lookup,
        qualification,
    )?;
    Ok(spawn_scheduler_actor_with_mode(
        model,
        mode,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        paged_prefix_cache,
        prefix_lru_cache,
        AdaptiveAdmissionPolicy::gemma4_drafter(),
        active_kv_offload,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_scheduler_actor_with_gemma4_drafter_and_active_kv(
    model: Arc<Mutex<crate::models::Gemma4Model>>,
    drafter: Arc<Mutex<crate::models::gemma4::Gemma4AssistantModel>>,
    mtp_draft_tokens: usize,
    exact_qualification: NeuralExactQualificationRuntimeConfig,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle> {
    let mode = SchedulerActorGemma4Drafter::new_with_exact_qualification(
        drafter,
        mtp_draft_tokens,
        exact_qualification,
    )?;
    Ok(spawn_scheduler_actor_with_mode(
        model,
        mode,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        paged_prefix_cache,
        prefix_lru_cache,
        AdaptiveAdmissionPolicy::gemma4_drafter(),
        active_kv_offload,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn spawn_scheduler_actor_with_mode<M, A>(
    model: Arc<Mutex<M>>,
    mtp_mode: A,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    adaptive_policy: AdaptiveAdmissionPolicy,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M> + Send + 'static,
{
    // ── Step 1: Budget validation on the calling thread. ──────────────────
    // No Scheduler / Array is allocated here — just pure arithmetic + RAM
    // check. Returns Err early if the budget is too tight.
    let budget_policy = startup_budget_policy(
        effective_cap_max,
        paged_prefix_cache.as_ref(),
        &active_kv_offload,
    );
    let budget_state = crate::core::memory_budget::validate_startup_budget_with_policy(
        b_max,
        effective_cap_max,
        &meta,
        budget_policy,
    )?;

    spawn_scheduler_actor_with_mode_and_budget_state(
        model,
        mtp_mode,
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
        paged_prefix_cache,
        prefix_lru_cache,
        adaptive_policy,
        active_kv_offload,
        budget_state,
    )
}

#[allow(clippy::too_many_arguments)]
fn spawn_scheduler_actor_with_mode_and_budget_state<M, A>(
    model: Arc<Mutex<M>>,
    mtp_mode: A,
    b_max: usize,
    admission_deadline: Duration,
    admission_queue_max: usize,
    effective_cap_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    meta: crate::core::memory_budget::ModelMeta,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    adaptive_policy: AdaptiveAdmissionPolicy,
    active_kv_offload: ActiveKvOffloadConfig,
    budget_state: crate::core::memory_budget::BudgetState,
) -> Result<SchedulerActorHandle, crate::core::memory_budget::MemoryBudgetError>
where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M> + Send + 'static,
{
    // ── Step 2: Shared atomics created on the calling thread. ─────────────
    // Cloned for both the handle (returned to caller) and the driver thread.
    // This is the single source of truth — handle + driver share the same
    // Arc instances, so /healthz reads the live values the driver updates.
    //
    // B1-p2.5 P0 fix v2: the previous fix (c043ce9) created Scheduler::new
    // on the calling thread then moved it into spawn_blocking. That caused
    // "MLX runtime error: There is no Stream(gpu, N) in current thread"
    // because Array::zeros (prng_state) was bound to the calling thread's
    // Metal Stream. This fix keeps budget validation + Arc creation on the
    // calling thread while deferring Scheduler::new_with_state (and thus
    // Array allocation) to the spawn_blocking worker thread.
    let memory_budget_exceeded_count = Arc::new(AtomicU64::new(0));
    let cold_materialization_tracker = Arc::new(OnceLock::new());
    let runtime_usage = Arc::new(crate::core::runtime_usage::ModelRuntimeUsageCounters::default());

    // Healthz observables cloned from BudgetState (Arc<AtomicUsize> inside).
    let kv_cache_active_bytes = budget_state.shared_active();
    let kv_cache_soft_limit_bytes = budget_state.soft_limit();
    let kv_cache_logical_cap_tokens = budget_state.logical_cap();
    let kv_cache_resident_cap_tokens = budget_state.resident_cap();
    let kv_cache_budget_policy = budget_state.policy().name();

    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let (control_tx, control_rx) = mpsc::channel(8);
    let admit_count = Arc::new(AtomicU64::new(0));
    let batch_count = Arc::new(AtomicU64::new(0));
    let saturate_triggered = Arc::new(AtomicU64::new(0));
    let queue_depth_peak = Arc::new(AtomicUsize::new(0));
    let queue_rejected = Arc::new(AtomicU64::new(0));
    let mtp_prefill_count = Arc::new(AtomicU64::new(0));
    let mtp_step_count = Arc::new(AtomicU64::new(0));
    let mtp_fallback_prefill_count = Arc::new(AtomicU64::new(0));
    let mtp_drafted_tokens = Arc::new(AtomicU64::new(0));
    let mtp_accepted_draft_tokens = Arc::new(AtomicU64::new(0));
    let mtp_windows = Arc::new(AtomicU64::new(0));
    let mtp_exact_sampling_windows = Arc::new(AtomicU64::new(0));
    let mtp_exact_acceptance_draws = Arc::new(AtomicU64::new(0));
    let mtp_exact_residual_corrections = Arc::new(AtomicU64::new(0));
    let mtp_exact_bonus_samples = Arc::new(AtomicU64::new(0));
    let mtp_draft_forward_us = Arc::new(AtomicU64::new(0));
    let mtp_verify_forward_us = Arc::new(AtomicU64::new(0));
    let mtp_projection_us = Arc::new(AtomicU64::new(0));
    let mtp_sampling_us = Arc::new(AtomicU64::new(0));
    let mtp_draft_host_sync_count = Arc::new(AtomicU64::new(0));
    let mtp_draft_host_sync_us = Arc::new(AtomicU64::new(0));
    let mtp_verify_accept_host_sync_count = Arc::new(AtomicU64::new(0));
    let mtp_verify_accept_host_sync_us = Arc::new(AtomicU64::new(0));
    let mtp_main_rollback_us = Arc::new(AtomicU64::new(0));
    let mtp_cache_commit_us = Arc::new(AtomicU64::new(0));
    let mtp_prefill_cache_commit_us = Arc::new(AtomicU64::new(0));
    let mtp_decode_cache_commit_us = Arc::new(AtomicU64::new(0));
    let mtp_cache_restore_us = Arc::new(AtomicU64::new(0));
    let prompt_lookup_published_stats = Arc::new(StdMutex::new(None));
    let neural_exact_qualification_stats =
        Arc::new(StdMutex::new(NeuralExactQualificationStats::default()));
    // B1-p2.5 G3: live b_active / b_queued updated by driver_loop tail.
    let b_active = Arc::new(AtomicU64::new(0));
    let b_queued = Arc::new(AtomicU64::new(0));
    let active_kv_stats = ActiveKvOffloadSharedStats::new(&active_kv_offload);
    let immutable_prefix_stats = ImmutablePrefixBlockSharedStats::new(paged_prefix_cache.is_some());

    // Clone Arcs for the driver thread.
    let driver_budget_state = budget_state.clone();
    let driver_mb_exceeded = memory_budget_exceeded_count.clone();
    let admit_count_for_task = admit_count.clone();
    let batch_count_for_task = batch_count.clone();
    let saturate_triggered_for_task = saturate_triggered.clone();
    let queue_depth_peak_for_task = queue_depth_peak.clone();
    let queue_rejected_for_task = queue_rejected.clone();
    let mtp_counters_for_task = SchedulerActorMtpCounters::new(
        mtp_prefill_count.clone(),
        mtp_step_count.clone(),
        mtp_fallback_prefill_count.clone(),
        mtp_drafted_tokens.clone(),
        mtp_accepted_draft_tokens.clone(),
        mtp_windows.clone(),
        mtp_exact_sampling_windows.clone(),
        mtp_exact_acceptance_draws.clone(),
        mtp_exact_residual_corrections.clone(),
        mtp_exact_bonus_samples.clone(),
        mtp_draft_forward_us.clone(),
        mtp_verify_forward_us.clone(),
        mtp_projection_us.clone(),
        mtp_sampling_us.clone(),
        mtp_draft_host_sync_count.clone(),
        mtp_draft_host_sync_us.clone(),
        mtp_verify_accept_host_sync_count.clone(),
        mtp_verify_accept_host_sync_us.clone(),
        mtp_main_rollback_us.clone(),
        mtp_cache_commit_us.clone(),
        mtp_prefill_cache_commit_us.clone(),
        mtp_decode_cache_commit_us.clone(),
        mtp_cache_restore_us.clone(),
        prompt_lookup_published_stats.clone(),
        neural_exact_qualification_stats.clone(),
    );
    let b_active_for_task = b_active.clone();
    let b_queued_for_task = b_queued.clone();
    let paged_prefix_cache_for_task = paged_prefix_cache.clone();
    let prefix_lru_cache_for_task = prefix_lru_cache;
    let active_kv_offload_for_task = active_kv_offload.clone();
    let active_kv_stats_for_task = active_kv_stats.clone();
    let immutable_prefix_stats_for_task = immutable_prefix_stats.clone();
    let cold_materialization_tracker_for_task = Arc::clone(&cold_materialization_tracker);
    let runtime_usage_for_task = Arc::clone(&runtime_usage);

    // ── Step 3: Spawn driver — Scheduler::new_with_state constructed INSIDE
    //    spawn_blocking so MLX Array fields (prng_state) are allocated on the
    //    worker thread's Metal Stream. Thread affinity preserved.
    tokio::task::spawn_blocking(move || {
        let mut scheduler = Scheduler::<M>::new_with_state(
            b_max,
            effective_cap_max,
            driver_budget_state,
            driver_mb_exceeded,
            meta,
        )
        .expect("budget already validated above; new_with_state must not fail");
        scheduler.enable_process_memory_governor(
            crate::core::process_memory::global_process_memory_governor(),
        );
        scheduler.share_cold_materialization_tracker(cold_materialization_tracker_for_task);
        scheduler.share_runtime_usage(runtime_usage_for_task);
        if let Some(config) = paged_prefix_cache_for_task {
            scheduler
                .enable_paged_prefix_cache(config)
                .expect("paged prefix cache config was validated before actor spawn");
        }
        if let Some(config) = prefix_lru_cache_for_task {
            let cache = crate::core::cache::process_shared_prefix_lru_cache(config)
                .expect("prefix LRU cache config was validated before actor spawn");
            scheduler
                .enable_shared_prefix_lru_cache(cache)
                .expect("prefix LRU cache config was validated before actor spawn");
        }
        scheduler
            .enable_active_kv_offload(active_kv_offload_for_task, active_kv_stats_for_task.clone())
            .expect("active KV offload config was validated before actor spawn");
        immutable_prefix_stats_for_task.store(scheduler.request_owned_kv_stats().immutable_prefix);
        driver_loop(
            scheduler,
            model,
            mtp_mode,
            mtp_counters_for_task,
            active_kv_stats_for_task,
            immutable_prefix_stats_for_task,
            admission_deadline,
            admission_queue_max,
            cmd_rx,
            control_rx,
            admit_count_for_task,
            batch_count_for_task,
            saturate_triggered_for_task,
            queue_depth_peak_for_task,
            queue_rejected_for_task,
            b_active_for_task,
            b_queued_for_task,
            decode_cadence_mid_chunk_cap,
            adaptive_policy,
        );
    });

    Ok(SchedulerActorHandle {
        cmd_tx,
        control_tx,
        cold_materialization_tracker,
        runtime_usage,
        admit_count,
        batch_count,
        saturate_triggered,
        queue_depth_peak,
        queue_rejected: queue_rejected.clone(),
        mtp_prefill_count,
        mtp_step_count,
        mtp_fallback_prefill_count,
        mtp_drafted_tokens,
        mtp_accepted_draft_tokens,
        mtp_windows,
        mtp_exact_sampling_windows,
        mtp_exact_acceptance_draws,
        mtp_exact_residual_corrections,
        mtp_exact_bonus_samples,
        mtp_draft_forward_us,
        mtp_verify_forward_us,
        mtp_projection_us,
        mtp_sampling_us,
        mtp_draft_host_sync_count,
        mtp_draft_host_sync_us,
        mtp_verify_accept_host_sync_count,
        mtp_verify_accept_host_sync_us,
        mtp_main_rollback_us,
        mtp_cache_commit_us,
        mtp_prefill_cache_commit_us,
        mtp_decode_cache_commit_us,
        mtp_cache_restore_us,
        prompt_lookup_published_stats,
        neural_exact_qualification_stats,
        b_active,
        b_queued,
        // P1.1: alias admission_queue_full_count to queue_rejected Arc —
        // driver_loop is the single fetch_add site; Scheduler field removed.
        admission_queue_full_count: queue_rejected,
        memory_budget_exceeded_count,
        kv_cache_active_bytes,
        kv_cache_soft_limit_bytes,
        kv_cache_logical_cap_tokens,
        kv_cache_resident_cap_tokens,
        kv_cache_budget_policy,
        active_kv_offload: active_kv_stats,
        immutable_prefix_blocks: immutable_prefix_stats,
    })
}

#[allow(clippy::too_many_arguments)]
fn driver_loop<M, A>(
    scheduler: Scheduler<M>,
    model: Arc<Mutex<M>>,
    mut mtp_mode: A,
    mtp_counters: SchedulerActorMtpCounters,
    active_kv_stats: ActiveKvOffloadSharedStats,
    immutable_prefix_stats: ImmutablePrefixBlockSharedStats,
    admission_deadline: Duration,
    admission_queue_max: usize,
    mut cmd_rx: mpsc::Receiver<SchedulerCommand>,
    mut control_rx: mpsc::Receiver<SchedulerControlCommand>,
    admit_count: Arc<AtomicU64>,
    batch_count: Arc<AtomicU64>,
    saturate_triggered: Arc<AtomicU64>,
    queue_depth_peak: Arc<AtomicUsize>,
    queue_rejected: Arc<AtomicU64>,
    b_active: Arc<AtomicU64>,
    b_queued: Arc<AtomicU64>,
    decode_cadence_mid_chunk_cap: usize,
    adaptive_policy: AdaptiveAdmissionPolicy,
) where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M>,
{
    // Receive Scheduler ownership from spawn_scheduler_actor (single instance).
    // P0 fix: previously driver_loop called Scheduler::new a second time,
    // creating fresh Arc atomics disconnected from the handle. B1-p2.5.
    let mut sched = scheduler;
    let b_max = sched.b_max();
    let mut event_txs: HashMap<RequestId, mpsc::UnboundedSender<StepEvent>> = HashMap::new();
    let mut admission_queue: VecDeque<PendingAdmit> = VecDeque::new();
    let mut in_flight_mid_admit: Option<A::MidAdmitHandle> = None;
    let mut parked_active_kv: VecDeque<ActiveKvParkedRequest> = VecDeque::new();
    let rt = tokio::runtime::Handle::current();

    'outer: loop {
        process_scheduler_control_commands(&mut sched, &mut control_rx, &mtp_counters);
        // Defensive: ensure scheduler is in Phase::Idle before
        // blocking on next admit. Most error paths already call evict_all,
        // but this guards any future code path that leaves phase=Finished.
        // If finalize fails, the actor cannot safely admit more requests
        // (the scheduler would be in an unrecoverable state); terminate
        // cleanly rather than emit ERROR per request.
        if sched.phase() == Phase::Finished {
            if let Err(e) =
                finalize_finished_batch_if_any(&mut sched, &mut event_txs, &mtp_counters)
            {
                tracing::error!(
                    "[SchedulerActor] outer-loop finalize failed: {e:?}; \
                     actor cannot reset Finished batch safely — terminating"
                );
                cleanup_parked_active_kv_requests(
                    &sched,
                    &mut parked_active_kv,
                    &mut event_txs,
                    &active_kv_stats,
                    "outer-loop finalize failed",
                );
                event_txs.clear();
                return;
            }
        }

        // Every ContinueOuter path may follow an error recovery that evicted
        // the live batch. Publish the recovered depth before blocking for the
        // next admit so health never retains a stale non-zero active count.
        publish_scheduler_depth(&sched, admission_queue.len(), &b_active, &b_queued);

        // ===== Outer Idle: block waiting for first admit (or shutdown). =====
        // Outer Idle is reached only after evict_all clears all slots; the
        // admission queue is invariantly empty here (any queue elements were
        // drained inside the rolling loop before reaching this point).
        let first_cmd = loop {
            match rt.block_on(tokio::time::timeout(
                Duration::from_millis(250),
                cmd_rx.recv(),
            )) {
                Ok(Some(cmd)) => break cmd,
                Ok(None) => {
                    cleanup_parked_active_kv_requests(
                        &sched,
                        &mut parked_active_kv,
                        &mut event_txs,
                        &active_kv_stats,
                        "scheduler command channel closed",
                    );
                    return;
                }
                Err(_) => {
                    process_scheduler_control_commands(&mut sched, &mut control_rx, &mtp_counters);
                    match sched.apply_process_memory_pressure() {
                        Ok(reclaim)
                            if reclaim.level
                                == crate::core::process_memory::PressureLevel::Normal =>
                        {
                            if let Err(error) =
                                sched.retry_pending_immutable_prefix_stores(usize::MAX)
                            {
                                tracing::warn!(%error, "idle immutable block SSD retry failed");
                            }
                        }
                        Ok(_) => {}
                        Err(error) => {
                            tracing::warn!(%error, "idle scheduler memory-pressure reclaim failed");
                        }
                    }
                    immutable_prefix_stats.store(sched.request_owned_kv_stats().immutable_prefix);
                }
            }
        };
        let fresh_batch_limit =
            fresh_prefill_batch_limit_for_command::<M>(&first_cmd, b_max, adaptive_policy);
        let fresh_batch_shape = admission_command_shape(&first_cmd);
        handle_admit(first_cmd, &mut sched, &mut event_txs, &admit_count);

        if sched.active_count() == 0 {
            // First admit failed (Err) — nothing to prefill. Wait for next.
            continue 'outer;
        }

        // ===== Admission window: drain additional admits until deadline
        //       or the model's fresh-prefill batch limit. Beyond the limit,
        //       push to admission_queue (bounded by admission_queue_max). =====
        if sched.active_count() < fresh_batch_limit {
            rt.block_on(drain_window(
                &mut cmd_rx,
                &mut sched,
                &mut event_txs,
                &mut admission_queue,
                &admit_count,
                &saturate_triggered,
                &queue_depth_peak,
                &queue_rejected,
                fresh_batch_limit,
                b_max,
                admission_queue_max,
                admission_deadline,
                fresh_batch_shape,
                adaptive_policy,
            ));
        }

        prune_abandoned_pending_admits(&mut admission_queue);
        evict_abandoned_active_requests::<M, A>(
            &mut sched,
            &mut event_txs,
            &mut in_flight_mid_admit,
        );
        mtp_counters.store_prompt_lookup_stats(sched.prompt_lookup_stats());
        b_active.store(sched.active_count() as u64, Ordering::Relaxed);
        b_queued.store(admission_queue.len() as u64, Ordering::Relaxed);
        if sched.active_count() == 0 {
            continue 'outer;
        }

        // ===== First-batch prefill. =====
        batch_count.fetch_add(1, Ordering::Relaxed);
        let prefill_profile = rolling_profile_enabled()
            .then(|| (sched.active_count(), admission_queue.len(), Instant::now()));
        let prefill_result = {
            let model_lock = model.blocking_lock();
            mtp_mode.prefill_admitted(&mut sched, &model_lock, &mtp_counters)
        };
        match prefill_result {
            Ok(prefill_events) => {
                if let Some((prefill_active, prefill_queue_len, prefill_timer)) = prefill_profile {
                    let prefill_end = Instant::now();
                    tracing::info!(
                        "[chunked-rolling-profile] event=fresh_prefill t_ms={:.3} active_count={} queue_len={} fresh_batch_limit={} event_count={} elapsed_ms={:.3}",
                        rolling_profile_t_ms(prefill_end),
                        prefill_active,
                        prefill_queue_len,
                        fresh_batch_limit,
                        prefill_events.len(),
                        rolling_profile_elapsed_ms(prefill_timer, prefill_end)
                    );
                }
                for ev in prefill_events {
                    route_event(ev, &event_txs);
                }
            }
            Err(e) => {
                if let Some((prefill_active, prefill_queue_len, prefill_timer)) = prefill_profile {
                    let prefill_end = Instant::now();
                    tracing::info!(
                        "[chunked-rolling-profile] event=fresh_prefill_error t_ms={:.3} active_count={} queue_len={} fresh_batch_limit={} elapsed_ms={:.3}",
                        rolling_profile_t_ms(prefill_end),
                        prefill_active,
                        prefill_queue_len,
                        fresh_batch_limit,
                        rolling_profile_elapsed_ms(prefill_timer, prefill_end)
                    );
                }
                tracing::error!("[SchedulerActor] prefill error: {e:?}");
                if let Err(evict_err) = sched.evict_all() {
                    tracing::warn!(
                        "[SchedulerActor] evict_all after prefill error also failed: \
                         {evict_err:?}; relying on 3b-1 poison flag to reject subsequent admits"
                    );
                }
                mtp_counters.reset_stats_baseline(sched.prompt_lookup_stats());
                cleanup_parked_active_kv_requests(
                    &sched,
                    &mut parked_active_kv,
                    &mut event_txs,
                    &active_kv_stats,
                    "scheduler poisoned after prefill error",
                );
                event_txs.clear();
                // Anything queued during the failed-batch window has nowhere
                // to land — reject with Err so callers see a clear error
                // rather than hanging.
                while let Some(pending) = admission_queue.pop_front() {
                    let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                        "scheduler poisoned after prefill error"
                    )));
                }
                continue 'outer;
            }
        }

        // ===== Rolling decode loop with bounded mid-batch admit + queue drain. =====
        let mut admission_policy = RollingAdmissionPolicy::default();
        admission_policy.record_admission_work();
        'rolling: loop {
            process_scheduler_control_commands(&mut sched, &mut control_rx, &mtp_counters);
            prune_abandoned_pending_admits(&mut admission_queue);
            evict_abandoned_active_requests::<M, A>(
                &mut sched,
                &mut event_txs,
                &mut in_flight_mid_admit,
            );
            discard_abandoned_parked_requests(
                &sched,
                &mut parked_active_kv,
                &mut event_txs,
                &active_kv_stats,
            );
            // Cancellation can empty the scheduler before the rolling-loop
            // tail is reached. Publish the post-eviction state here so
            // /healthz never reports a ghost active request while the actor is
            // already blocked in the outer idle receive.
            b_active.store(sched.active_count() as u64, Ordering::Relaxed);
            b_queued.store(admission_queue.len() as u64, Ordering::Relaxed);
            match sched.apply_process_memory_pressure() {
                Ok(reclaim) if reclaim.should_park_request && in_flight_mid_admit.is_none() => {
                    let _ = try_park_one_active_kv_request(
                        &mut sched,
                        &model,
                        &mut parked_active_kv,
                        &active_kv_stats,
                    );
                    b_active.store(sched.active_count() as u64, Ordering::Relaxed);
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(%error, "scheduler memory-pressure reclaim failed");
                }
            }
            // Pre-event Finished-batch finalization + handoff. If
            // previous iteration's prefill_admitted/step left phase=Finished
            // (e.g. max_tokens=1 workload), handle the completed batch BEFORE
            // dispatching another event. Per Codex Q6: biased select may pick
            // Admit over Step, so this must run before the event pick — or the
            // actor could call admit_mid_begin() in Phase::Finished.
            //
            // `drive_empty_scheduler_handoff` itself calls
            // `finalize_finished_batch_if_any`; do not duplicate finalization
            // here. This avoids two divergent finalize/error paths.
            if sched.phase() == Phase::Finished {
                match drive_empty_scheduler_handoff(
                    &mut sched,
                    &mut cmd_rx,
                    &mut event_txs,
                    &mut admission_queue,
                    &model,
                    &admit_count,
                    &saturate_triggered,
                    &queue_depth_peak,
                    &queue_rejected,
                    &batch_count,
                    &mut mtp_mode,
                    &mtp_counters,
                    &mut parked_active_kv,
                    &active_kv_stats,
                    b_max,
                    admission_queue_max,
                    admission_deadline,
                    adaptive_policy,
                    &rt,
                ) {
                    RollingControl::ContinueRolling => {
                        admission_policy.record_admission_work();
                        continue 'rolling;
                    }
                    RollingControl::BreakRolling => break 'rolling,
                    RollingControl::ContinueOuter => continue 'outer,
                    RollingControl::ReturnActor => return,
                }
            }

            let evt: RollingEvent = if admission_policy.should_force_decode(
                sched.phase(),
                scheduler_has_decodable_rows(&sched),
                in_flight_mid_admit.is_some() || !admission_queue.is_empty(),
            ) {
                RollingEvent::Step
            } else if in_flight_mid_admit.is_some() {
                RollingEvent::AdvanceMidAdmit
            } else {
                rt.block_on(async {
                    tokio::select! {
                        biased;
                        maybe_cmd = cmd_rx.recv() => match maybe_cmd {
                            Some(cmd) => RollingEvent::Admit(cmd),
                            None => RollingEvent::Shutdown,
                        },
                        _ = std::future::ready(()) => RollingEvent::Step,
                    }
                })
            };

            match evt {
                RollingEvent::Shutdown => {
                    cleanup_parked_active_kv_requests(
                        &sched,
                        &mut parked_active_kv,
                        &mut event_txs,
                        &active_kv_stats,
                        "scheduler shutting down",
                    );
                    event_txs.clear();
                    // Reject any queued admits — callers shouldn't hang.
                    while let Some(pending) = admission_queue.pop_front() {
                        let _ = pending
                            .reply_tx
                            .send(Err(anyhow::anyhow!("scheduler shutting down")));
                    }
                    return;
                }
                RollingEvent::Admit(cmd) => {
                    if !mtp_mode.allow_rolling_mid_admit()
                        || !mtp_mode.can_start_rolling_mid_admit(&sched)
                    {
                        enqueue_or_reject(
                            cmd,
                            &mut admission_queue,
                            admission_queue_max,
                            &queue_depth_peak,
                            &queue_rejected,
                        );
                    } else if !can_start_rolling_mid_admit_for_command::<M>(
                        &cmd,
                        &sched,
                        sched.active_count(),
                        b_max,
                        adaptive_policy,
                    ) {
                        let mut pending_cmd = Some(cmd);
                        let can_start_after_park = can_start_rolling_mid_admit_for_command::<M>(
                            pending_cmd.as_ref().expect("pending command present"),
                            &sched,
                            sched.active_count().saturating_sub(1),
                            b_max,
                            adaptive_policy,
                        );
                        if in_flight_mid_admit.is_none()
                            && can_park_for_rolling_admission(&sched)
                            && can_start_after_park
                            && try_park_one_active_kv_request(
                                &mut sched,
                                &model,
                                &mut parked_active_kv,
                                &active_kv_stats,
                            )
                        {
                            if sched.active_count() == 0 {
                                enqueue_or_reject(
                                    pending_cmd.take().expect("pending command present"),
                                    &mut admission_queue,
                                    admission_queue_max,
                                    &queue_depth_peak,
                                    &queue_rejected,
                                );
                                admission_policy.record_admission_work();
                            } else if can_start_rolling_mid_admit_for_command::<M>(
                                pending_cmd.as_ref().expect("pending command present"),
                                &sched,
                                sched.active_count(),
                                b_max,
                                adaptive_policy,
                            ) {
                                let decode_steps = start_mid_admit_one_chunk(
                                    pending_cmd.take().expect("pending command present"),
                                    &mut in_flight_mid_admit,
                                    &mut sched,
                                    &mut event_txs,
                                    &admit_count,
                                    &model,
                                    &mut mtp_mode,
                                    &mtp_counters,
                                    MidAdmitProfileContext {
                                        source: RollingMidAdmitSource::Direct,
                                        queue_wait_ms: None,
                                        queue_len: admission_queue.len(),
                                    },
                                    decode_cadence_mid_chunk_cap,
                                );
                                if decode_steps > 0 {
                                    admission_policy
                                        .record_admission_work_with_decode_steps(decode_steps);
                                }
                            }
                        }
                        if let Some(cmd) = pending_cmd {
                            // Rolling admission limit reached — queue for a later decode turn.
                            enqueue_or_reject(
                                cmd,
                                &mut admission_queue,
                                admission_queue_max,
                                &queue_depth_peak,
                                &queue_rejected,
                            );
                        }
                    } else {
                        let decode_steps = start_mid_admit_one_chunk(
                            cmd,
                            &mut in_flight_mid_admit,
                            &mut sched,
                            &mut event_txs,
                            &admit_count,
                            &model,
                            &mut mtp_mode,
                            &mtp_counters,
                            MidAdmitProfileContext {
                                source: RollingMidAdmitSource::Direct,
                                queue_wait_ms: None,
                                queue_len: admission_queue.len(),
                            },
                            decode_cadence_mid_chunk_cap,
                        );
                        if decode_steps > 0 {
                            admission_policy.record_admission_work_with_decode_steps(decode_steps);
                        }
                    }
                }
                RollingEvent::AdvanceMidAdmit => {
                    let decode_steps = advance_mid_admit_one_chunk(
                        &mut in_flight_mid_admit,
                        &mut sched,
                        &mut event_txs,
                        &admit_count,
                        &model,
                        &mut mtp_mode,
                        &mtp_counters,
                        admission_queue.len(),
                        decode_cadence_mid_chunk_cap,
                    );
                    if decode_steps > 0 {
                        admission_policy.record_admission_work_with_decode_steps(decode_steps);
                    }
                }
                RollingEvent::Step => {
                    let step_profile = rolling_profile_enabled().then(|| {
                        (
                            sched.active_count(),
                            admission_queue.len(),
                            in_flight_mid_admit.is_some(),
                            Instant::now(),
                        )
                    });
                    let step_result = {
                        let model_lock = model.blocking_lock();
                        mtp_mode.step(
                            &mut sched,
                            &model_lock,
                            &mtp_counters,
                            in_flight_mid_admit.is_some() || !admission_queue.is_empty(),
                        )
                    };
                    let step_end = step_profile.map(|_| Instant::now());
                    match step_result {
                        Ok(events) => {
                            let event_count = events.len();
                            for ev in events {
                                route_event(ev, &event_txs);
                            }
                            let evicted_count = match sched.gc_finished_rows(&mut event_txs) {
                                Ok(evicted) => evicted.len(),
                                Err(error) => {
                                    tracing::error!(%error, "request-owned KV release failed");
                                    if let Err(evict_err) = sched.evict_all() {
                                        tracing::warn!(
                                            "[SchedulerActor] evict_all after request-owned KV release failure also failed: {evict_err:?}"
                                        );
                                    }
                                    mtp_counters.reset_stats_baseline(sched.prompt_lookup_stats());
                                    in_flight_mid_admit = None;
                                    cleanup_parked_active_kv_requests(
                                        &sched,
                                        &mut parked_active_kv,
                                        &mut event_txs,
                                        &active_kv_stats,
                                        "scheduler poisoned after request-owned KV release failure",
                                    );
                                    event_txs.clear();
                                    while let Some(pending) = admission_queue.pop_front() {
                                        let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                                            "scheduler poisoned after request-owned KV release failure"
                                        )));
                                    }
                                    b_active.store(sched.active_count() as u64, Ordering::Relaxed);
                                    b_queued.store(admission_queue.len() as u64, Ordering::Relaxed);
                                    continue 'outer;
                                }
                            };
                            mtp_counters.store_prompt_lookup_stats(sched.prompt_lookup_stats());
                            if let (
                                Some((
                                    step_active_before,
                                    step_queue_len,
                                    step_had_in_flight_mid_admit,
                                    step_timer,
                                )),
                                Some(step_end),
                            ) = (step_profile, step_end)
                            {
                                tracing::info!(
                                    "[chunked-rolling-profile] event=decode_step t_ms={:.3} active_before={} active_after={} queue_len={} had_in_flight_mid_admit={} event_count={} evicted_count={} elapsed_ms={:.3}",
                                    rolling_profile_t_ms(step_end),
                                    step_active_before,
                                    sched.active_count(),
                                    step_queue_len,
                                    step_had_in_flight_mid_admit,
                                    event_count,
                                    evicted_count,
                                    rolling_profile_elapsed_ms(step_timer, step_end)
                                );
                            }
                            admission_policy.record_decode_step();
                            // ===== Post-gc queue drain. =====
                            // Free slots → pull from admission_queue head
                            // for one bounded mid-admit. Further queued
                            // requests wait for the next decode turn so
                            // active streams keep making progress.
                            if mtp_mode.allow_rolling_mid_admit()
                                && mtp_mode.can_start_rolling_mid_admit(&sched)
                                && in_flight_mid_admit.is_none()
                                && !admission_queue.is_empty()
                                && sched.active_count() >= b_max
                                && can_park_for_rolling_admission(&sched)
                            {
                                let _ = try_park_one_active_kv_request(
                                    &mut sched,
                                    &model,
                                    &mut parked_active_kv,
                                    &active_kv_stats,
                                );
                            }
                            if mtp_mode.allow_rolling_mid_admit()
                                && mtp_mode.can_start_rolling_mid_admit(&sched)
                                && in_flight_mid_admit.is_none()
                            {
                                let decode_steps = drain_admission_queue(
                                    &mut admission_queue,
                                    &mut in_flight_mid_admit,
                                    &mut sched,
                                    &mut event_txs,
                                    &admit_count,
                                    &model,
                                    &mut mtp_mode,
                                    &mtp_counters,
                                    b_max,
                                    decode_cadence_mid_chunk_cap,
                                    adaptive_policy,
                                );
                                if decode_steps > 0 {
                                    admission_policy
                                        .record_admission_work_with_decode_steps(decode_steps);
                                }
                            }
                        }
                        Err(e) => {
                            if let (
                                Some((
                                    step_active_before,
                                    step_queue_len,
                                    step_had_in_flight_mid_admit,
                                    step_timer,
                                )),
                                Some(step_end),
                            ) = (step_profile, step_end)
                            {
                                tracing::info!(
                                    "[chunked-rolling-profile] event=decode_step_error t_ms={:.3} active_before={} active_after={} queue_len={} had_in_flight_mid_admit={} elapsed_ms={:.3}",
                                    rolling_profile_t_ms(step_end),
                                    step_active_before,
                                    sched.active_count(),
                                    step_queue_len,
                                    step_had_in_flight_mid_admit,
                                    rolling_profile_elapsed_ms(step_timer, step_end)
                                );
                            }
                            tracing::error!("[SchedulerActor] step error: {e:?}");
                            if let Err(evict_err) = sched.evict_all() {
                                tracing::warn!(
                                    "[SchedulerActor] evict_all after step error also failed: \
                                     {evict_err:?}; relying on 3b-1 poison flag to reject subsequent admits"
                                );
                            }
                            mtp_counters.reset_stats_baseline(sched.prompt_lookup_stats());
                            in_flight_mid_admit = None;
                            cleanup_parked_active_kv_requests(
                                &sched,
                                &mut parked_active_kv,
                                &mut event_txs,
                                &active_kv_stats,
                                "scheduler poisoned after step error",
                            );
                            event_txs.clear();
                            while let Some(pending) = admission_queue.pop_front() {
                                let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                                    "scheduler poisoned after step error"
                                )));
                            }
                            b_active.store(sched.active_count() as u64, Ordering::Relaxed);
                            b_queued.store(admission_queue.len() as u64, Ordering::Relaxed);
                            continue 'outer;
                        }
                    }
                }
            }

            // B1-p2.5 G3: update /healthz live counters at tail of every rolling step.
            b_active.store(sched.active_count() as u64, Ordering::Relaxed);
            b_queued.store(admission_queue.len() as u64, Ordering::Relaxed);
            sched.refresh_active_kv_residency_stats();
            active_kv_stats.set_parked_requests(parked_active_kv.len());
            immutable_prefix_stats.store(sched.request_owned_kv_stats().immutable_prefix);

            // ===== Exit rolling loop when active_count == 0 AND queue empty. =====
            // Spec §9 R1: if `active_count() == 0` but admission_queue is
            // non-empty (mid-rolling admit arrived AFTER all rows finished),
            // treat as a "new batch within rolling": evict_all to reset to
            // Idle, then admit from queue + drain_window + prefill_admitted
            // inline (mirrors the existing post-empty path but pulls the
            // first admit from the queue instead of cmd_rx).
            //
            // Extracted into `drive_empty_scheduler_handoff` so the
            // same logic backs the pre-event Finished-batch finalization hook
            // at the rolling-loop top. The helper finalizes any leftover
            // `Phase::Finished` state first, then performs the queued-admit
            // / try_recv / break handoff.
            if sched.active_count() == 0 {
                match drive_empty_scheduler_handoff(
                    &mut sched,
                    &mut cmd_rx,
                    &mut event_txs,
                    &mut admission_queue,
                    &model,
                    &admit_count,
                    &saturate_triggered,
                    &queue_depth_peak,
                    &queue_rejected,
                    &batch_count,
                    &mut mtp_mode,
                    &mtp_counters,
                    &mut parked_active_kv,
                    &active_kv_stats,
                    b_max,
                    admission_queue_max,
                    admission_deadline,
                    adaptive_policy,
                    &rt,
                ) {
                    RollingControl::ContinueRolling => {
                        admission_policy.record_admission_work();
                        continue 'rolling;
                    }
                    RollingControl::BreakRolling => break 'rolling,
                    RollingControl::ContinueOuter => continue 'outer,
                    RollingControl::ReturnActor => return,
                }
            }
        }

        // After rolling loop: reset cache + Phase for next outer iteration.
        if matches!(sched.phase(), Phase::Decoding | Phase::Finished) {
            if let Err(evict_err) = sched.evict_all() {
                tracing::warn!(
                    "[SchedulerActor] evict_all at end of outer failed: {evict_err:?}; \
                     relying on 3b-1 poison flag to reject subsequent admits"
                );
            }
            mtp_counters.reset_stats_baseline(sched.prompt_lookup_stats());
        }
        immutable_prefix_stats.store(sched.request_owned_kv_stats().immutable_prefix);
        in_flight_mid_admit = None;
        cleanup_parked_active_kv_requests(
            &sched,
            &mut parked_active_kv,
            &mut event_txs,
            &active_kv_stats,
            "scheduler outer loop reset",
        );
        event_txs.clear();
    }
}

fn process_scheduler_control_commands<M>(
    sched: &mut Scheduler<M>,
    control_rx: &mut mpsc::Receiver<SchedulerControlCommand>,
    counters: &SchedulerActorMtpCounters,
) where
    M: Model + DenseVlMethods,
{
    while let Ok(command) = control_rx.try_recv() {
        match command {
            SchedulerControlCommand::ClearSharedPromptLookup { reply_tx } => {
                let cleared = sched.clear_shared_prompt_lookup();
                counters.store_prompt_lookup_stats(sched.prompt_lookup_stats());
                let _ = reply_tx.send(cleared);
            }
        }
    }
}

fn publish_scheduler_depth<M>(
    sched: &Scheduler<M>,
    queued: usize,
    b_active: &AtomicU64,
    b_queued: &AtomicU64,
) where
    M: Model,
{
    b_active.store(sched.active_count() as u64, Ordering::Relaxed);
    b_queued.store(queued as u64, Ordering::Relaxed);
}

/// Drain additional `Admit` commands until either the deadline expires or the
/// fresh-batch admission limit is reached. Hard deadline — new admits do NOT
/// reset the timer. Once the limit is reached, additional admits within the
/// window go to the admission queue (bounded by `admission_queue_max`).
#[allow(clippy::too_many_arguments)]
async fn drain_window<M>(
    cmd_rx: &mut mpsc::Receiver<SchedulerCommand>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admission_queue: &mut VecDeque<PendingAdmit>,
    admit_count: &Arc<AtomicU64>,
    saturate_triggered: &Arc<AtomicU64>,
    queue_depth_peak: &Arc<AtomicUsize>,
    queue_rejected: &Arc<AtomicU64>,
    batch_limit: usize,
    b_max: usize,
    queue_max: usize,
    deadline: Duration,
    batch_shape: AdmissionRequestShape,
    adaptive_policy: AdaptiveAdmissionPolicy,
) where
    M: Model + DenseVlMethods + Send + 'static,
{
    let batch_limit = batch_limit.clamp(1, b_max);
    let timer = tokio::time::sleep(deadline);
    tokio::pin!(timer);
    let mut limit_reached = false;
    loop {
        tokio::select! {
            biased;
            _ = &mut timer => return,
            maybe = cmd_rx.recv() => {
                let Some(cmd) = maybe else { return }; // channel closed
                let command_shape = admission_command_shape(&cmd);
                let command_batch_limit =
                    fresh_prefill_batch_limit_for_command::<M>(&cmd, b_max, adaptive_policy);
                if limit_reached
                    || sched.active_count() >= command_batch_limit
                    || !adaptive_policy.can_join_fresh_batch(batch_shape, command_shape)
                {
                    // Fresh batch is full for this model/prompt policy — push
                    // to queue or reject.
                    enqueue_or_reject(
                        cmd,
                        admission_queue,
                        queue_max,
                        queue_depth_peak,
                        queue_rejected,
                    );
                    continue;
                }
                handle_admit(cmd, sched, event_txs, admit_count);
                if sched.active_count() >= batch_limit {
                    if batch_limit >= b_max {
                        saturate_triggered.fetch_add(1, Ordering::Relaxed);
                    }
                    limit_reached = true;
                    // Stay in the loop until deadline so queued admits
                    // arriving during the window's remaining time are
                    // captured. (Pre-3d returned here; 3d keeps draining.)
                }
            }
        }
    }
}

/// Process a single `Admit` command: try `Scheduler::admit`; on success
/// register the per-request event channel and increment admit_count;
/// on failure forward the Err to the caller. Reply-tx send failure
/// (caller abandoned) causes the slot to be evicted as cleanup.
fn handle_admit<M>(
    cmd: SchedulerCommand,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
) where
    M: Model + DenseVlMethods + Send + 'static,
{
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    match sched.admit(request) {
        Ok(id) => {
            admit_count.fetch_add(1, Ordering::Relaxed);
            event_txs.insert(id, event_tx);
            if reply_tx
                .send(Ok(AdmitReply {
                    request_id: id,
                    event_rx,
                }))
                .is_err()
            {
                // Caller dropped reply_rx before we could send.
                // Evict the orphan slot.
                let _ = sched.evict(id);
                event_txs.remove(&id);
            }
        }
        Err(e) => {
            let _ = reply_tx.send(Err(e));
        }
    }
}

fn can_start_rolling_mid_admit_for_request<M: Model>(
    request: &GenerateRequest,
    sched: &Scheduler<M>,
    active_count: usize,
    b_max: usize,
    adaptive_policy: AdaptiveAdmissionPolicy,
) -> bool {
    if active_count >= b_max {
        return false;
    }
    let model_limit = M::fresh_prefill_batch_limit(request.prompt_ids.len(), b_max).clamp(1, b_max);
    adaptive_policy.can_start_rolling_mid_admit(
        admission_request_shape(request),
        active_count,
        model_limit,
        b_max,
        scheduler_available_decode_steps(sched),
    )
}

fn can_start_rolling_mid_admit_for_command<M: Model>(
    cmd: &SchedulerCommand,
    sched: &Scheduler<M>,
    active_count: usize,
    b_max: usize,
    adaptive_policy: AdaptiveAdmissionPolicy,
) -> bool {
    let SchedulerCommand::Admit { request, .. } = cmd;
    can_start_rolling_mid_admit_for_request::<M>(
        request,
        sched,
        active_count,
        b_max,
        adaptive_policy,
    )
}

fn begin_mid_admit<M, A>(
    cmd: SchedulerCommand,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    model: &Arc<Mutex<M>>,
    mtp_mode: &mut A,
    profile_context: MidAdmitProfileContext,
) -> Option<A::MidAdmitHandle>
where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M>,
{
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    let admit_profile = rolling_profile_enabled().then(|| {
        (
            request.prompt_ids.len(),
            request.prefill_chunk_size,
            sched.active_count(),
            Instant::now(),
        )
    });
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    // Phase 1: begin.
    let handle = {
        let m = model.blocking_lock();
        match mtp_mode.begin_mid_admit(sched, &m, request) {
            Ok(h) => h,
            Err(e) => {
                let _ = reply_tx.send(Err(e));
                return None;
            }
        }
    };
    let begin_end = admit_profile.map(|_| Instant::now());
    let id = A::mid_admit_request_id(&handle);
    if let (Some((prompt_len, prefill_chunk_size, active_before, begin_start)), Some(begin_end)) =
        (admit_profile, begin_end)
    {
        tracing::info!(
            "[chunked-rolling-profile] event=mid_begin t_ms={:.3} request_id={} source={} prompt_len={} prefill_chunk_size={} active_before={} active_after={} queue_len={} queue_wait_ms={:.3} elapsed_ms={:.3}",
            rolling_profile_t_ms(begin_end),
            id.0,
            profile_context.source.as_str(),
            prompt_len,
            prefill_chunk_size,
            active_before,
            sched.active_count(),
            profile_context.queue_len,
            profile_context.queue_wait_ms.unwrap_or(-1.0),
            rolling_profile_elapsed_ms(begin_start, begin_end)
        );
    }
    event_txs.insert(id, event_tx);
    if reply_tx
        .send(Ok(AdmitReply {
            request_id: id,
            event_rx,
        }))
        .is_err()
    {
        // Caller dropped reply_rx before the prefill chunks completed.
        let _ = sched.evict(id);
        event_txs.remove(&id);
        return None;
    }

    Some(handle)
}

#[allow(clippy::too_many_arguments)]
fn start_mid_admit_one_chunk<M, A>(
    cmd: SchedulerCommand,
    in_flight_mid_admit: &mut Option<A::MidAdmitHandle>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<M>>,
    mtp_mode: &mut A,
    mtp_counters: &SchedulerActorMtpCounters,
    profile_context: MidAdmitProfileContext,
    decode_cadence_mid_chunk_cap: usize,
) -> usize
where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M>,
{
    if in_flight_mid_admit.is_some() {
        return 0;
    }
    let Some(handle) = begin_mid_admit(cmd, sched, event_txs, model, mtp_mode, profile_context)
    else {
        return 0;
    };
    *in_flight_mid_admit = Some(handle);
    advance_mid_admit_one_chunk(
        in_flight_mid_admit,
        sched,
        event_txs,
        admit_count,
        model,
        mtp_mode,
        mtp_counters,
        profile_context.queue_len,
        decode_cadence_mid_chunk_cap,
    )
}

#[allow(clippy::too_many_arguments)]
fn advance_mid_admit_one_chunk<M, A>(
    in_flight_mid_admit: &mut Option<A::MidAdmitHandle>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<M>>,
    mtp_mode: &mut A,
    mtp_counters: &SchedulerActorMtpCounters,
    queue_len: usize,
    decode_cadence_mid_chunk_cap: usize,
) -> usize
where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M>,
{
    let Some(mut handle) = in_flight_mid_admit.take() else {
        return 0;
    };
    let id = A::mid_admit_request_id(&handle);
    let active_count_before_chunk = sched.active_count();
    let requested_chunk_size = A::mid_admit_chunk_size(&handle);
    let chunk_start_before = A::mid_admit_chunk_start(&handle);
    let effective_chunk_size = cadence_protected_mid_chunk_size(
        requested_chunk_size,
        active_count_before_chunk,
        A::mid_admit_decode_cadence_mid_chunk_cap(&handle),
    );
    debug_assert!(decode_cadence_mid_chunk_cap > 0);
    A::set_mid_admit_chunk_size(&mut handle, effective_chunk_size);
    let chunk_profile = rolling_profile_enabled().then(|| {
        (
            chunk_start_before,
            A::mid_admit_prompt_len(&handle),
            effective_chunk_size,
            active_count_before_chunk,
            Instant::now(),
        )
    });

    let chunk_result = {
        let m = model.blocking_lock();
        mtp_mode.advance_mid_admit_chunk(sched, &m, &mut handle)
    };
    A::set_mid_admit_chunk_size(&mut handle, requested_chunk_size);
    let is_last = match chunk_result {
        Ok(b) => b,
        Err(e) => {
            if let Some((chunk_start, prompt_len, chunk_size, active_count, chunk_timer)) =
                chunk_profile
            {
                let chunk_end_time = Instant::now();
                let chunk_end = A::mid_admit_chunk_start(&handle);
                tracing::info!(
                    "[chunked-rolling-profile] event=mid_chunk_error t_ms={:.3} request_id={} chunk_start={} chunk_end={} chunk_len={} prompt_len={} chunk_size={} active_count={} queue_len={} elapsed_ms={:.3}",
                    rolling_profile_t_ms(chunk_end_time),
                    id.0,
                    chunk_start,
                    chunk_end,
                    chunk_end.saturating_sub(chunk_start),
                    prompt_len,
                    chunk_size,
                    active_count,
                    queue_len,
                    rolling_profile_elapsed_ms(chunk_timer, chunk_end_time)
                );
            }
            tracing::error!("[SchedulerActor] admit_mid_chunk error: {e:?}");
            let _ = sched.evict(id);
            event_txs.remove(&id);
            return ROLLING_DECODE_STEPS_AFTER_ADMISSION_WORK;
        }
    };
    let completed_chunk_tokens =
        usize::try_from(A::mid_admit_chunk_start(&handle).saturating_sub(chunk_start_before))
            .unwrap_or(0);
    let decode_steps =
        decode_steps_after_mid_admit_chunk(completed_chunk_tokens, decode_cadence_mid_chunk_cap);
    if let Some((chunk_start, prompt_len, chunk_size, active_count, chunk_timer)) = chunk_profile {
        let chunk_end_time = Instant::now();
        let chunk_end = A::mid_admit_chunk_start(&handle);
        tracing::info!(
            "[chunked-rolling-profile] event=mid_chunk t_ms={:.3} request_id={} chunk_start={} chunk_end={} chunk_len={} prompt_len={} chunk_size={} is_last={} active_count={} queue_len={} elapsed_ms={:.3}",
            rolling_profile_t_ms(chunk_end_time),
            id.0,
            chunk_start,
            chunk_end,
            chunk_end.saturating_sub(chunk_start),
            prompt_len,
            chunk_size,
            is_last,
            active_count,
            queue_len,
            rolling_profile_elapsed_ms(chunk_timer, chunk_end_time)
        );
    }

    if !is_last {
        *in_flight_mid_admit = Some(handle);
        return decode_steps;
    }

    let finalize_profile =
        rolling_profile_enabled().then(|| (sched.active_count(), Instant::now()));
    let m = model.blocking_lock();
    match mtp_mode.finalize_mid_admit(sched, &m, handle, mtp_counters) {
        Ok((_id, first_event)) => {
            admit_count.fetch_add(1, Ordering::Relaxed);
            if let Some((active_before, finalize_timer)) = finalize_profile {
                let finalize_end = Instant::now();
                tracing::info!(
                    "[chunked-rolling-profile] event=mid_finalize t_ms={:.3} request_id={} active_before={} active_after={} queue_len={} elapsed_ms={:.3}",
                    rolling_profile_t_ms(finalize_end),
                    id.0,
                    active_before,
                    sched.active_count(),
                    queue_len,
                    rolling_profile_elapsed_ms(finalize_timer, finalize_end)
                );
            }
            route_event(first_event, event_txs);
        }
        Err(e) => {
            if let Some((active_before, finalize_timer)) = finalize_profile {
                let finalize_end = Instant::now();
                tracing::info!(
                    "[chunked-rolling-profile] event=mid_finalize_error t_ms={:.3} request_id={} active_before={} active_after={} queue_len={} elapsed_ms={:.3}",
                    rolling_profile_t_ms(finalize_end),
                    id.0,
                    active_before,
                    sched.active_count(),
                    queue_len,
                    rolling_profile_elapsed_ms(finalize_timer, finalize_end)
                );
            }
            tracing::error!("[SchedulerActor] admit_mid_finalize error: {e:?}");
            let _ = sched.evict(id);
            event_txs.remove(&id);
        }
    }
    decode_steps
}

/// Push a pending admit into the queue if there's capacity; otherwise reply
/// with `Err(SchedulerError::QueueFull)` (wrapped in anyhow) and bump
/// `queue_rejected`. Updates `queue_depth_peak` via `fetch_max`.
///
/// HTTP handlers downcast the anyhow Err to [`SchedulerError`] to map
/// QueueFull → HTTP 503 + Retry-After; other errors → HTTP 400.
fn enqueue_or_reject(
    cmd: SchedulerCommand,
    queue: &mut VecDeque<PendingAdmit>,
    queue_max: usize,
    queue_depth_peak: &Arc<AtomicUsize>,
    queue_rejected: &Arc<AtomicU64>,
) {
    let SchedulerCommand::Admit { request, reply_tx } = cmd;
    if reply_tx.is_closed() {
        return;
    }
    prune_abandoned_pending_admits(queue);
    if queue.len() >= queue_max {
        queue_rejected.fetch_add(1, Ordering::Relaxed);
        let _ = reply_tx.send(Err(anyhow::Error::new(
            crate::core::scheduler::SchedulerError::QueueFull {
                capacity: queue_max,
            },
        )));
        return;
    }
    let enqueue_profile = rolling_profile_enabled().then(|| {
        (
            Instant::now(),
            request.prompt_ids.len(),
            request.prefill_chunk_size,
        )
    });
    let queued_at_profile = enqueue_profile.map(|(now, _, _)| now);
    queue.push_back(PendingAdmit {
        request,
        reply_tx,
        queued_at_profile,
    });
    queue_depth_peak.fetch_max(queue.len(), Ordering::Relaxed);
    if let Some((now, prompt_len, prefill_chunk_size)) = enqueue_profile {
        tracing::info!(
            "[chunked-rolling-profile] event=queue_enqueue t_ms={:.3} prompt_len={} prefill_chunk_size={} queue_len={} queue_max={}",
            rolling_profile_t_ms(now),
            prompt_len,
            prefill_chunk_size,
            queue.len(),
            queue_max
        );
    }
}

fn prune_abandoned_pending_admits(queue: &mut VecDeque<PendingAdmit>) -> usize {
    let before = queue.len();
    queue.retain(|pending| !pending.reply_tx.is_closed());
    before.saturating_sub(queue.len())
}

fn evict_abandoned_active_requests<M, A>(
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    in_flight_mid_admit: &mut Option<A::MidAdmitHandle>,
) -> usize
where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M>,
{
    let mut evicted = 0;
    if let Some(handle) = in_flight_mid_admit.as_ref() {
        let id = A::mid_admit_request_id(handle);
        let abandoned = event_txs.get(&id).is_some_and(|tx| tx.is_closed());
        if abandoned {
            match sched.evict(id) {
                Ok(()) => {
                    *in_flight_mid_admit = None;
                    event_txs.remove(&id);
                    evicted += 1;
                }
                Err(error) => {
                    tracing::warn!(request_id = id.0, %error, "failed to evict cancelled mid-admit request");
                }
            }
        }
    }

    let abandoned_ids: Vec<RequestId> = sched
        .active()
        .into_iter()
        .filter_map(|state| {
            event_txs
                .get(&state.id)
                .is_some_and(|tx| tx.is_closed())
                .then_some(state.id)
        })
        .collect();
    for id in abandoned_ids {
        match sched.evict(id) {
            Ok(()) => {
                event_txs.remove(&id);
                evicted += 1;
            }
            Err(error) => {
                tracing::warn!(request_id = id.0, %error, "failed to evict cancelled request");
            }
        }
    }
    evicted
}

fn discard_abandoned_parked_requests<M>(
    sched: &Scheduler<M>,
    parked_active_kv: &mut VecDeque<ActiveKvParkedRequest>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    active_kv_stats: &ActiveKvOffloadSharedStats,
) -> usize
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let mut discarded = 0;
    let mut retained = VecDeque::with_capacity(parked_active_kv.len());
    while let Some(parked) = parked_active_kv.pop_front() {
        let abandoned = event_txs.get(&parked.id).is_none_or(|tx| tx.is_closed());
        if abandoned {
            match sched.discard_active_kv_request(&parked) {
                Ok(()) => {
                    event_txs.remove(&parked.id);
                    discarded += 1;
                }
                Err(error) => {
                    active_kv_stats.record_error();
                    tracing::warn!(
                        request_id = parked.id.0,
                        %error,
                        "failed to discard cancelled parked request"
                    );
                    retained.push_back(parked);
                }
            }
        } else {
            retained.push_back(parked);
        }
    }
    *parked_active_kv = retained;
    active_kv_stats.set_parked_requests(parked_active_kv.len());
    discarded
}

/// Drain at most one mid-batch admit chunk from the admission queue.
/// Full-prompt rolling admits obey the model's `fresh_prefill_batch_limit`.
/// Multi-chunk admits may start in a spare slot beyond that limit because
/// each chunk yields back to the rolling loop before the next chunk runs.
/// Once any admission work happens the caller must return to decode before
/// draining more queue.
///
/// IMPORTANT: `admit_mid` is only legal in `Decoding` phase. If
/// `gc_finished_rows` just transitioned the scheduler to `Finished`
/// (because `active_count` dropped to 0), the caller's rolling-loop
/// exit branch (`active_count == 0 && queue non-empty`) will handle the
/// queued entries via `evict_all` + fresh `prefill_admitted`. Return
/// early here so we do not call `admit_mid` in an illegal phase.
#[allow(clippy::too_many_arguments)]
fn drain_admission_queue<M, A>(
    queue: &mut VecDeque<PendingAdmit>,
    in_flight_mid_admit: &mut Option<A::MidAdmitHandle>,
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admit_count: &Arc<AtomicU64>,
    model: &Arc<Mutex<M>>,
    mtp_mode: &mut A,
    mtp_counters: &SchedulerActorMtpCounters,
    b_max: usize,
    decode_cadence_mid_chunk_cap: usize,
    adaptive_policy: AdaptiveAdmissionPolicy,
) -> usize
where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M>,
{
    // admit_mid is only legal in Decoding phase.
    if sched.phase() != Phase::Decoding {
        return 0;
    }
    if sched.active_count() == 0 {
        return 0;
    }
    if in_flight_mid_admit.is_some() {
        return 0;
    }
    while sched.active_count() < b_max {
        let Some(pending) = queue.front() else {
            return 0;
        };
        if pending.reply_tx.is_closed() {
            queue.pop_front();
            continue;
        }
        if !can_start_rolling_mid_admit_for_request::<M>(
            &pending.request,
            sched,
            sched.active_count(),
            b_max,
            adaptive_policy,
        ) {
            return 0;
        }
        let pending = queue
            .pop_front()
            .expect("queue.front returned Some immediately before pop_front");
        let dequeue_profile = pending.queued_at_profile.map(|queued_at| {
            let now = Instant::now();
            (now, rolling_profile_queue_wait_ms(queued_at, now))
        });
        let queue_wait_ms = dequeue_profile.map(|(_, wait_ms)| wait_ms);
        if let Some((dequeue_at, queue_wait_ms)) = dequeue_profile {
            tracing::info!(
                "[chunked-rolling-profile] event=queue_dequeue t_ms={:.3} prompt_len={} prefill_chunk_size={} queue_len={} queue_wait_ms={:.3}",
                rolling_profile_t_ms(dequeue_at),
                pending.request.prompt_ids.len(),
                pending.request.prefill_chunk_size,
                queue.len(),
                queue_wait_ms
            );
        }
        let cmd = SchedulerCommand::Admit {
            request: pending.request,
            reply_tx: pending.reply_tx,
        };
        let decode_steps = start_mid_admit_one_chunk(
            cmd,
            in_flight_mid_admit,
            sched,
            event_txs,
            admit_count,
            model,
            mtp_mode,
            mtp_counters,
            MidAdmitProfileContext {
                source: RollingMidAdmitSource::Queue,
                queue_wait_ms,
                queue_len: queue.len(),
            },
            decode_cadence_mid_chunk_cap,
        );
        // Re-check phase after each mid-admit — if admit_mid itself
        // exhausted remaining rows and transitioned to Finished, stop.
        if decode_steps > 0 || sched.phase() != Phase::Decoding {
            return decode_steps;
        }
    }
    0
}

fn route_event(ev: StepEvent, event_txs: &HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>) {
    if let Some(tx) = event_txs.get(&ev.id) {
        // Unbounded channel — only fails when the receiver was dropped
        // (handler abandoned). The rolling-loop cancellation sweep evicts
        // that request and releases its KV/governor reservations.
        let _ = tx.send(ev);
    }
}

fn try_park_one_active_kv_request<M>(
    sched: &mut Scheduler<M>,
    model: &Arc<Mutex<M>>,
    parked_active_kv: &mut VecDeque<ActiveKvParkedRequest>,
    active_kv_stats: &ActiveKvOffloadSharedStats,
) -> bool
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if !sched.active_kv_offload_enabled() || sched.phase() != Phase::Decoding {
        return false;
    }
    let candidate_ids: Vec<RequestId> = sched
        .active()
        .into_iter()
        .filter(|state| !state.finished && !state.generated_tokens.is_empty())
        .map(|state| state.id)
        .collect();
    if candidate_ids.is_empty() {
        return false;
    }
    let model_lock = model.blocking_lock();
    for id in candidate_ids {
        match sched.park_active_kv_request(id, &model_lock) {
            Ok(Some(parked)) => {
                tracing::info!(
                    "[active-kv-offload] event=park request_id={} parked_queue_len={}",
                    parked.id.0,
                    parked_active_kv.len() + 1
                );
                parked_active_kv.push_back(parked);
                active_kv_stats.set_parked_requests(parked_active_kv.len());
                return true;
            }
            Ok(None) => {}
            Err(err) => {
                active_kv_stats.record_error();
                tracing::warn!(
                    "[active-kv-offload] event=park_error request_id={} error={err:#}",
                    id.0
                );
            }
        }
    }
    false
}

fn can_park_for_rolling_admission<M: Model>(sched: &Scheduler<M>) -> bool {
    // Rolling mid-admit requires at least one active decode row. Parking the
    // final row cannot make admission progress: the empty-scheduler handoff
    // restores that same parked row before draining queued work, creating a
    // park/restore cycle on every decode step.
    sched.active_count() > 1
}

fn try_restore_one_active_kv_request<M>(
    sched: &mut Scheduler<M>,
    model: &Arc<Mutex<M>>,
    parked_active_kv: &mut VecDeque<ActiveKvParkedRequest>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    active_kv_stats: &ActiveKvOffloadSharedStats,
) -> bool
where
    M: Model + DenseVlMethods + Send + 'static,
{
    if !sched.active_kv_offload_enabled() || sched.active_count() >= sched.b_max() {
        return false;
    }
    if sched.memory_governor_snapshot().is_some_and(|snapshot| {
        snapshot.pressure_level != crate::core::process_memory::PressureLevel::Normal
    }) {
        return false;
    }

    while sched.active_count() < sched.b_max() {
        let Some(parked) = parked_active_kv.pop_front() else {
            active_kv_stats.set_parked_requests(0);
            return false;
        };
        if event_txs.get(&parked.id).is_none_or(|tx| tx.is_closed()) {
            if let Err(error) = sched.discard_active_kv_request(&parked) {
                active_kv_stats.record_error();
                tracing::warn!(
                    request_id = parked.id.0,
                    %error,
                    "failed to discard cancelled parked request before restore"
                );
            } else {
                event_txs.remove(&parked.id);
            }
            active_kv_stats.set_parked_requests(parked_active_kv.len());
            continue;
        }
        let model_lock = model.blocking_lock();
        match sched.restore_active_kv_request(&parked, &model_lock) {
            Ok(id) => {
                tracing::info!(
                    "[active-kv-offload] event=restore request_id={} parked_queue_len={}",
                    id.0,
                    parked_active_kv.len()
                );
                active_kv_stats.set_parked_requests(parked_active_kv.len());
                return true;
            }
            Err(err) => {
                active_kv_stats.record_error();
                tracing::warn!(
                    "[active-kv-offload] event=restore_error request_id={} error={err:#}",
                    parked.id.0
                );
                if let Err(cleanup_err) = sched.discard_active_kv_request(&parked) {
                    active_kv_stats.record_error();
                    tracing::warn!(
                        "[active-kv-offload] event=restore_error_cleanup_failed request_id={} error={cleanup_err:#}",
                        parked.id.0
                    );
                }
                event_txs.remove(&parked.id);
                active_kv_stats.set_parked_requests(parked_active_kv.len());
            }
        }
    }
    false
}

fn cleanup_parked_active_kv_requests<M>(
    sched: &Scheduler<M>,
    parked_active_kv: &mut VecDeque<ActiveKvParkedRequest>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    active_kv_stats: &ActiveKvOffloadSharedStats,
    reason: &str,
) where
    M: Model + DenseVlMethods + Send + 'static,
{
    while let Some(parked) = parked_active_kv.pop_front() {
        if let Err(err) = sched.discard_active_kv_request(&parked) {
            active_kv_stats.record_error();
            tracing::warn!(
                "[active-kv-offload] event=cleanup_error request_id={} reason={} error={err:#}",
                parked.id.0,
                reason
            );
        }
        event_txs.remove(&parked.id);
    }
    active_kv_stats.set_parked_requests(0);
}

/// Finalize a `Phase::Finished` batch: evict slots + release budget +
/// reset to `Phase::Idle`, then close per-request event channels.
///
/// Returns `Ok(true)` if finalization happened (caller MUST go to the
/// empty-scheduler handoff path, NOT continue the normal event pick;
/// per spec § 4.2.1 hard binding).
/// This binding applies in the rolling-loop context; the outer-loop hook calls this directly because admission_queue is invariantly empty at that point.
/// Returns `Ok(false)` if `phase != Finished` (no-op; safe to continue).
/// Returns `Err` if `evict_all` failed (caller should reject queued
/// admits + `continue 'outer` per existing pattern).
///
/// The `Phase::Finished` state arises naturally when `prefill_admitted`
/// completes a batch where every request has
/// `max_new_tokens=1` (the prefill samples first+last token in one
/// pass), which is the standard `iron-bench --max-tokens 1` perf
/// measurement workload.
fn finalize_finished_batch_if_any<M: Model>(
    sched: &mut Scheduler<M>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    mtp_counters: &SchedulerActorMtpCounters,
) -> Result<bool> {
    if sched.phase() != Phase::Finished {
        return Ok(false);
    }
    let evicted_ids: Vec<RequestId> = sched.active().into_iter().map(|state| state.id).collect();
    match sched.evict_all() {
        Ok(()) => {
            mtp_counters.reset_stats_baseline(sched.prompt_lookup_stats());
            for id in evicted_ids {
                event_txs.remove(&id);
            }
            Ok(true)
        }
        Err(e) => {
            tracing::warn!("[SchedulerActor] finalize_finished_batch: evict_all failed: {e:?}");
            Err(e)
        }
    }
}

/// Finalize a just-finished batch if needed, then drain queued admits
/// (or a single pending `cmd_rx.try_recv` admit) into a fresh batch, run
/// `prefill_admitted`, and return how the caller's rolling loop should
/// proceed. Lifts the existing empty-batch transition logic at the
/// rolling-loop tail so it can also be invoked from the pre-event
/// Finished-batch finalization at the rolling-loop top.
///
/// This helper is the single empty-batch handoff path. It first calls
/// [`finalize_finished_batch_if_any`], so callers must not separately
/// finalize before invoking it. After that call the scheduler is either
/// `Phase::Idle` or `Phase::Decoding` (the legacy post-step empty-handoff
/// path used to encounter `Phase::Finished`; the new pre-event hook now
/// shoulders that case via finalize). The helper preserves the current
/// reset semantics for `Decoding`-with-zero-active-rows before starting
/// the next batch but never calls `evict_all` in `Idle` (which is itself
/// an error per scheduler.rs:775-780).
///
/// Behavior per branch:
/// - Queued admit present → pop head, fresh batch via `handle_admit` +
///   `drain_window` + `prefill_admitted`; returns `ContinueRolling`.
/// - Queue empty + `cmd_rx.try_recv()` returns `Ok(cmd)` → fresh batch
///   via the same path; returns `ContinueRolling`.
/// - Queue empty + `try_recv` returns `Empty` → returns `BreakRolling`.
/// - Queue empty + `try_recv` returns `Disconnected` → clear `event_txs`,
///   returns `ReturnActor`.
/// - Any `finalize`, legacy reset, or `prefill_admitted` failure →
///   reject queued admits, clear `event_txs`, returns `ContinueOuter`.
///
/// Replaces the previous `if sched.active_count() == 0 { ... }` block
/// at rolling-loop tail to avoid divergent copies.
#[allow(clippy::too_many_arguments)]
fn drive_empty_scheduler_handoff<M, A>(
    sched: &mut Scheduler<M>,
    cmd_rx: &mut mpsc::Receiver<SchedulerCommand>,
    event_txs: &mut HashMap<RequestId, mpsc::UnboundedSender<StepEvent>>,
    admission_queue: &mut VecDeque<PendingAdmit>,
    model: &Arc<Mutex<M>>,
    admit_count: &Arc<AtomicU64>,
    saturate_triggered: &Arc<AtomicU64>,
    queue_depth_peak: &Arc<AtomicUsize>,
    queue_rejected: &Arc<AtomicU64>,
    batch_count: &Arc<AtomicU64>,
    mtp_mode: &mut A,
    mtp_counters: &SchedulerActorMtpCounters,
    parked_active_kv: &mut VecDeque<ActiveKvParkedRequest>,
    active_kv_stats: &ActiveKvOffloadSharedStats,
    b_max: usize,
    admission_queue_max: usize,
    admission_deadline: Duration,
    adaptive_policy: AdaptiveAdmissionPolicy,
    rt: &tokio::runtime::Handle,
) -> RollingControl
where
    M: Model + DenseVlMethods + Send + 'static,
    A: SchedulerActorMtpMode<M>,
{
    // Finalize any Finished batch BEFORE re-admitting. After
    // this, phase is one of {Idle, Decoding}; never Finished. Callers
    // must not separately finalize.
    match finalize_finished_batch_if_any(sched, event_txs, mtp_counters) {
        Ok(_) => {}
        Err(_e) => {
            cleanup_parked_active_kv_requests(
                sched,
                parked_active_kv,
                event_txs,
                active_kv_stats,
                "scheduler poisoned during Finished-batch finalize",
            );
            while let Some(pending) = admission_queue.pop_front() {
                let _ = pending.reply_tx.send(Err(anyhow::anyhow!(
                    "scheduler poisoned during Finished-batch finalize"
                )));
            }
            event_txs.clear();
            return RollingControl::ContinueOuter;
        }
    }

    if try_restore_one_active_kv_request(sched, model, parked_active_kv, event_txs, active_kv_stats)
    {
        return RollingControl::ContinueRolling;
    }
    if !parked_active_kv.is_empty() {
        std::thread::sleep(Duration::from_millis(10));
        return RollingControl::ContinueRolling;
    }

    if !admission_queue.is_empty() {
        // Reset Decoding-with-zero-active-rows to Idle for fresh batch.
        // (Finished was already handled by finalize above; Idle would
        // itself be an error for `evict_all`.)
        if sched.phase() == Phase::Decoding {
            if let Err(evict_err) = sched.evict_all() {
                tracing::warn!(
                    "[SchedulerActor] evict_all between batches (queue drain) failed: \
                     {evict_err:?}; rejecting queued admits"
                );
                while let Some(pending) = admission_queue.pop_front() {
                    let _ = pending
                        .reply_tx
                        .send(Err(anyhow::anyhow!("scheduler evict_all failed")));
                }
                event_txs.clear();
                return RollingControl::ContinueOuter;
            }
            mtp_counters.reset_stats_baseline(sched.prompt_lookup_stats());
            // Preserve parked Active KV request event channels. At this point
            // active_count is zero; stale finished-batch channels were already
            // removed by gc/finalize paths.
        }
        // Pop first queued admit as the new batch's first admit.
        let pending = admission_queue
            .pop_front()
            .expect("queue non-empty checked");
        let fresh_batch_limit =
            fresh_prefill_batch_limit_for_request::<M>(&pending.request, b_max, adaptive_policy);
        let fresh_batch_shape = admission_request_shape(&pending.request);
        handle_admit(
            SchedulerCommand::Admit {
                request: pending.request,
                reply_tx: pending.reply_tx,
            },
            sched,
            event_txs,
            admit_count,
        );
        if sched.active_count() == 0 {
            // Admit failed; loop to drain more queue (or exit).
            return RollingControl::ContinueRolling;
        }
        if sched.active_count() < fresh_batch_limit {
            // Drain queue head-by-head into the new batch (no deadline —
            // these are already-queued admits, not racing-in cmd_rx).
            // Then optionally drain_window for fresh cmd_rx admits.
            while sched.active_count() < fresh_batch_limit {
                let Some(pending_shape) = admission_queue
                    .front()
                    .map(|pending| admission_request_shape(&pending.request))
                else {
                    break;
                };
                let candidate_limit = admission_queue
                    .front()
                    .map(|pending| {
                        fresh_prefill_batch_limit_for_request::<M>(
                            &pending.request,
                            b_max,
                            adaptive_policy,
                        )
                    })
                    .expect("queue.front returned Some immediately before candidate limit");
                if sched.active_count() >= candidate_limit
                    || !adaptive_policy.can_join_fresh_batch(fresh_batch_shape, pending_shape)
                {
                    break;
                }
                let p = admission_queue
                    .pop_front()
                    .expect("queue.front returned Some immediately before pop_front");
                handle_admit(
                    SchedulerCommand::Admit {
                        request: p.request,
                        reply_tx: p.reply_tx,
                    },
                    sched,
                    event_txs,
                    admit_count,
                );
            }
            // Optionally absorb cmd_rx admits arriving right now.
            if sched.active_count() < fresh_batch_limit {
                rt.block_on(drain_window(
                    cmd_rx,
                    sched,
                    event_txs,
                    admission_queue,
                    admit_count,
                    saturate_triggered,
                    queue_depth_peak,
                    queue_rejected,
                    fresh_batch_limit,
                    b_max,
                    admission_queue_max,
                    admission_deadline,
                    fresh_batch_shape,
                    adaptive_policy,
                ));
            }
        }
        batch_count.fetch_add(1, Ordering::Relaxed);
        let prefill_result = {
            let model_lock = model.blocking_lock();
            mtp_mode.prefill_admitted(sched, &model_lock, mtp_counters)
        };
        match prefill_result {
            Ok(events) => {
                for ev in events {
                    route_event(ev, event_txs);
                }
            }
            Err(e) => {
                tracing::error!("[SchedulerActor] re-prefill (queue drain) error: {e:?}");
                if let Err(evict_err) = sched.evict_all() {
                    tracing::warn!(
                        "[SchedulerActor] evict_all after re-prefill error also failed: \
                         {evict_err:?}; rejecting remaining queued admits"
                    );
                }
                mtp_counters.reset_stats_baseline(sched.prompt_lookup_stats());
                event_txs.clear();
                while let Some(p) = admission_queue.pop_front() {
                    let _ = p.reply_tx.send(Err(anyhow::anyhow!(
                        "scheduler poisoned after re-prefill error"
                    )));
                }
                return RollingControl::ContinueOuter;
            }
        }
        return RollingControl::ContinueRolling;
    }

    // Queue empty + no active rows — same logic as pre-3d.
    match cmd_rx.try_recv() {
        Ok(cmd) => {
            let fresh_batch_limit =
                fresh_prefill_batch_limit_for_command::<M>(&cmd, b_max, adaptive_policy);
            let fresh_batch_shape = admission_command_shape(&cmd);
            if sched.phase() == Phase::Decoding {
                if let Err(evict_err) = sched.evict_all() {
                    tracing::warn!(
                        "[SchedulerActor] evict_all between batches failed: \
                         {evict_err:?}; rejecting incoming admit"
                    );
                    let SchedulerCommand::Admit { reply_tx, .. } = cmd;
                    let _ = reply_tx.send(Err(evict_err));
                    event_txs.clear();
                    return RollingControl::ContinueOuter;
                }
                mtp_counters.reset_stats_baseline(sched.prompt_lookup_stats());
                // Preserve parked Active KV request event channels. At this
                // point active_count is zero; stale finished-batch channels
                // were already removed by gc/finalize paths.
            }
            handle_admit(cmd, sched, event_txs, admit_count);
            if sched.active_count() == 0 {
                return RollingControl::BreakRolling;
            }
            if sched.active_count() < fresh_batch_limit {
                rt.block_on(drain_window(
                    cmd_rx,
                    sched,
                    event_txs,
                    admission_queue,
                    admit_count,
                    saturate_triggered,
                    queue_depth_peak,
                    queue_rejected,
                    fresh_batch_limit,
                    b_max,
                    admission_queue_max,
                    admission_deadline,
                    fresh_batch_shape,
                    adaptive_policy,
                ));
            }
            batch_count.fetch_add(1, Ordering::Relaxed);
            let prefill_result = {
                let model_lock = model.blocking_lock();
                mtp_mode.prefill_admitted(sched, &model_lock, mtp_counters)
            };
            match prefill_result {
                Ok(events) => {
                    for ev in events {
                        route_event(ev, event_txs);
                    }
                }
                Err(e) => {
                    tracing::error!("[SchedulerActor] re-prefill error: {e:?}");
                    if let Err(evict_err) = sched.evict_all() {
                        tracing::warn!(
                            "[SchedulerActor] evict_all after re-prefill error also failed: \
                             {evict_err:?}; relying on 3b-1 poison flag to reject subsequent admits"
                        );
                    }
                    mtp_counters.reset_stats_baseline(sched.prompt_lookup_stats());
                    event_txs.clear();
                    return RollingControl::ContinueOuter;
                }
            }
            RollingControl::ContinueRolling
        }
        Err(tokio::sync::mpsc::error::TryRecvError::Empty) => RollingControl::BreakRolling,
        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
            event_txs.clear();
            RollingControl::ReturnActor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::cache::MtpCache;
    use crate::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
    use crate::core::sampler::Sampler;
    use crate::core::speculative::MtpSpeculativeModel;
    use crate::nn::MtpStepOutput;

    #[derive(Clone, Copy)]
    struct SchedulerActorFakeModel {
        forward_delay: Duration,
        mtp_accepted_prefix_restore: bool,
    }

    #[allow(non_upper_case_globals)]
    const SchedulerActorFakeModel: SchedulerActorFakeModel = SchedulerActorFakeModel {
        forward_delay: Duration::ZERO,
        mtp_accepted_prefix_restore: false,
    };

    /// Keep scheduler behavior tests independent from the host's physical RAM.
    fn test_scheduler(
        b_max: usize,
        effective_cap_max: usize,
    ) -> Scheduler<SchedulerActorFakeModel> {
        let meta = crate::core::memory_budget::test_meta_qwen35();
        let budget_state = crate::core::memory_budget::BudgetState::with_soft_limit(
            crate::core::memory_budget::kv_cache_bytes(b_max, effective_cap_max, &meta),
            effective_cap_max,
            effective_cap_max,
            crate::core::memory_budget::KvBudgetPolicy::FullResident,
        );
        Scheduler::new_with_state(
            b_max,
            effective_cap_max,
            budget_state,
            Arc::new(AtomicU64::new(0)),
            meta,
        )
        .expect("test scheduler startup")
    }

    #[test]
    fn rolling_admission_parking_requires_a_remaining_decode_row() {
        let mut scheduler = test_scheduler(2, 32);
        scheduler.admit(mk_req(11)).expect("admit first row");
        assert!(!can_park_for_rolling_admission(&scheduler));

        scheduler.admit(mk_req(22)).expect("admit second row");
        assert!(can_park_for_rolling_admission(&scheduler));
    }

    impl SchedulerActorFakeModel {
        fn with_forward_delay(forward_delay: Duration) -> Self {
            Self {
                forward_delay,
                mtp_accepted_prefix_restore: false,
            }
        }

        fn with_mtp_accepted_prefix_restore(forward_delay: Duration) -> Self {
            Self {
                forward_delay,
                mtp_accepted_prefix_restore: true,
            }
        }

        fn maybe_delay_forward(&self) {
            if !self.forward_delay.is_zero() {
                std::thread::sleep(self.forward_delay);
            }
        }
    }

    #[derive(Clone, Copy)]
    struct SchedulerActorFakeMtpHead;

    fn write_fake_full_kv(
        input_ids: &mlx::Array,
        per_row_lens: Option<&[i32]>,
        cache: Option<&mut [crate::nn::LayerCache]>,
    ) -> Result<()> {
        let Some(cache) = cache else {
            return Ok(());
        };
        let Some(crate::nn::LayerCache::Full(kv)) = cache.first_mut() else {
            return Ok(());
        };
        let shape = input_ids.shape();
        let dims = shape.as_slice();
        let batch = dims[0];
        let seq = dims[1];
        let owned_lens;
        let lens = match per_row_lens {
            Some(lens) => lens,
            None => {
                owned_lens = vec![seq; batch as usize];
                &owned_lens
            }
        };
        let k = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
            .map_err(|e| anyhow::anyhow!("fake full k failed: {e:?}"))?;
        let v = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
            .map_err(|e| anyhow::anyhow!("fake full v failed: {e:?}"))?;
        kv.update_and_fetch(&k, &v, lens)?;
        Ok(())
    }

    impl Model for SchedulerActorFakeModel {
        fn make_cache(
            &self,
            batch: i32,
            cap: i32,
            dtype: mlx::Dtype,
        ) -> Result<Vec<crate::nn::LayerCache>> {
            Ok(vec![crate::nn::LayerCache::Full(
                crate::core::KVCache::new(batch, 1, 1, 1, dtype, cap),
            )])
        }

        fn forward_on(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            write_fake_full_kv(input_ids, _per_row_lens, cache)?;
            self.maybe_delay_forward();
            fake_logits(input_ids.shape().as_slice()[0] as usize)
        }

        fn batched_prefill(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            per_row_lens: &[i32],
            cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            write_fake_full_kv(input_ids, Some(per_row_lens), cache)?;
            fake_logits(input_ids.shape().as_slice()[0] as usize)
        }

        fn forward_text_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            write_fake_full_kv(input_ids, per_row_lens, cache)?;
            let shape = input_ids.shape();
            let b = shape.as_slice()[0] as usize;
            let s = shape.as_slice()[1] as usize;
            let hidden = 4_usize;
            let flat = vec![0.0_f32; b * s * hidden];
            (&flat[..], &[b as i32, s as i32, hidden as i32][..])
                .try_into()
                .map_err(|e| anyhow::anyhow!("fake hidden Array failed: {e:?}"))
        }

        fn project_hidden_on(
            &self,
            hidden: &mlx::Array,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            let batch = hidden.shape().as_slice()[0] as usize;
            let seq = hidden.shape().as_slice()[1] as usize;
            fake_batched_logits(batch, seq, if seq == 1 { 3 } else { 4 })
        }

        fn fresh_prefill_batch_limit(_prompt_len: usize, b_max: usize) -> usize
        where
            Self: Sized,
        {
            b_max.min(2)
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }

        fn supports_exact_batched_speculative_verify(
            &self,
            _batch_width: usize,
            _context_tokens: usize,
            _verify_width: usize,
        ) -> bool {
            true
        }
    }

    impl DenseVlMethods for SchedulerActorFakeModel {
        fn batched_prefill_vl(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _attention_mask: &mlx::Array,
            _linear_attention_mask: &mlx::Array,
            _per_row_lens: &[i32],
            _per_row_pixel_values: &[Option<&[mlx::Array]>],
            _per_row_grid_thw: &[Option<&[(i32, i32, i32)]>],
            _image_token_id: i32,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            fake_logits(input_ids.shape().as_slice()[0] as usize)
        }

        fn estimate_vision_prefill_peak_bytes(
            &self,
            pixel_values: &[mlx::Array],
            grid_thw: &[(i32, i32, i32)],
        ) -> Result<usize> {
            Ok(pixel_values
                .iter()
                .map(|pixels| pixels.size().saturating_mul(pixels.dtype().byte_size()))
                .sum::<usize>()
                .saturating_add(grid_thw.len())
                .max(1))
        }

        fn compute_vision_embeds(
            &self,
            _pixel_values: &[mlx::Array],
            grid_thw: &[(i32, i32, i32)],
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            let rows: i32 = grid_thw
                .iter()
                .map(|&(t, h, w)| t * (h / 2).max(1) * (w / 2).max(1))
                .sum::<i32>()
                .max(1);
            mlx::Array::zeros((rows, 1_i32), mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("fake vision embeds Array failed: {e:?}"))
        }

        fn forward_vl_chunk(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            fake_logits(input_ids.shape().as_slice()[0] as usize)
        }

        fn forward_vl_hidden(
            &self,
            input_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&mlx::Array>,
            _cache: Option<&mut [crate::nn::LayerCache]>,
            _vision_embeds_slice: Option<&mlx::Array>,
            _image_token_id: i32,
            _target: mlx::StreamOrDevice,
        ) -> Result<mlx::Array> {
            let shape = input_ids.shape();
            let b = shape.as_slice()[0] as usize;
            let s = shape.as_slice()[1] as usize;
            let hidden = 4_usize;
            let flat = vec![0.0_f32; b * s * hidden];
            (&flat[..], &[b as i32, s as i32, hidden as i32][..])
                .try_into()
                .map_err(|e| anyhow::anyhow!("fake VL hidden Array failed: {e:?}"))
        }
    }

    impl MtpSpeculativeModel for SchedulerActorFakeModel {
        type MtpHead = SchedulerActorFakeMtpHead;

        fn load_mtp_head(&self, _loader: &crate::core::Loader) -> Result<Self::MtpHead> {
            Ok(SchedulerActorFakeMtpHead)
        }

        fn make_mtp_cache(
            &self,
            _mtp: &Self::MtpHead,
            batch: i32,
            cap: i32,
            dtype: mlx::Dtype,
        ) -> Result<MtpCache> {
            MtpCache::new_with_cap(1, batch, 1, 1, 1, dtype, cap)
        }

        fn mtp_hidden_size(&self, _mtp: &Self::MtpHead) -> i32 {
            4
        }

        fn mtp_hidden_dtype(&self, _mtp: &Self::MtpHead) -> mlx::Dtype {
            mlx::Dtype::Float32
        }

        fn project_mtp_verify_hidden_on(
            &self,
            hidden: &mlx::Array,
            target: impl Into<mlx::StreamOrDevice>,
        ) -> Result<mlx::Array> {
            if !self.mtp_accepted_prefix_restore {
                return Model::project_hidden_on(self, hidden, target.into());
            }
            let shape = hidden.shape();
            let dims = shape.as_slice();
            fake_batched_logits(dims[0] as usize, dims[1] as usize, 5)
        }

        fn supports_mtp_accepted_prefix_restore(&self) -> bool {
            self.mtp_accepted_prefix_restore
        }

        fn begin_mtp_accepted_prefix_capture(
            &self,
            cache: &mut [crate::nn::LayerCache],
        ) -> Result<()> {
            anyhow::ensure!(
                self.mtp_accepted_prefix_restore,
                "fake MTP accepted-prefix restore is disabled"
            );
            for layer in cache {
                layer.begin_speculative_prefix_capture()?;
            }
            Ok(())
        }

        fn restore_mtp_accepted_prefix_rows_on(
            &self,
            cache: &mut [crate::nn::LayerCache],
            snapshots: &[crate::nn::LayerCacheSnapshot],
            accepted_lens: &[usize],
            _target: mlx::StreamOrDevice,
        ) -> Result<()> {
            anyhow::ensure!(cache.len() == snapshots.len(), "fake cache layer mismatch");
            for (layer, snapshot) in cache.iter_mut().zip(snapshots) {
                let (crate::nn::LayerCache::Full(cache), crate::nn::LayerCacheSnapshot::Full(base)) =
                    (layer, snapshot)
                else {
                    anyhow::bail!("fake accepted-prefix restore requires Full KV");
                };
                anyhow::ensure!(
                    accepted_lens.len() == cache.offsets().len(),
                    "fake accepted prefix rows {} != cache batch {}",
                    accepted_lens.len(),
                    cache.offsets().len()
                );
                let offsets = base
                    .offsets()
                    .iter()
                    .zip(accepted_lens)
                    .map(|(&offset, &accepted)| {
                        offset
                            .checked_add(i32::try_from(accepted)?)
                            .ok_or_else(|| anyhow::anyhow!("fake accepted offset overflow"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                cache.restore_offsets(&offsets)?;
            }
            Ok(())
        }

        fn mtp_forward_hidden_on(
            &self,
            _mtp: &Self::MtpHead,
            hidden_states: &mlx::Array,
            next_token_ids: &mlx::Array,
            _position_ids: &mlx::Array,
            _mask: Option<&mlx::Array>,
            mtp_cache: Option<&mut MtpCache>,
            _target: impl Into<mlx::StreamOrDevice>,
        ) -> Result<mlx::Array> {
            if let Some(cache) = mtp_cache {
                let batch = next_token_ids.shape().as_slice()[0];
                let seq = next_token_ids.shape().as_slice()[1];
                let k = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
                    .map_err(|e| anyhow::anyhow!("fake mtp k failed: {e:?}"))?;
                let v = mlx::Array::zeros((batch, 1_i32, seq, 1_i32), mlx::Dtype::Bfloat16)
                    .map_err(|e| anyhow::anyhow!("fake mtp v failed: {e:?}"))?;
                cache
                    .layer_mut(0)
                    .update_and_fetch(&k, &v, &vec![seq; batch as usize])?;
            }
            Ok(hidden_states.clone())
        }

        fn mtp_forward_on(
            &self,
            mtp: &Self::MtpHead,
            hidden_states: &mlx::Array,
            next_token_ids: &mlx::Array,
            position_ids: &mlx::Array,
            mask: Option<&mlx::Array>,
            mtp_cache: Option<&mut MtpCache>,
            target: impl Into<mlx::StreamOrDevice>,
        ) -> Result<MtpStepOutput> {
            let hidden_states = self.mtp_forward_hidden_on(
                mtp,
                hidden_states,
                next_token_ids,
                position_ids,
                mask,
                mtp_cache,
                target,
            )?;
            let batch = next_token_ids.shape().as_slice()[0] as usize;
            let seq = next_token_ids.shape().as_slice()[1] as usize;
            Ok(MtpStepOutput {
                hidden_states,
                logits: fake_batched_logits(batch, seq, 4)?,
            })
        }
    }

    fn fake_logits(batch: usize) -> Result<mlx::Array> {
        let vocab = 8_usize;
        let mut flat = vec![0.0_f32; batch * vocab];
        for row in 0..batch {
            flat[row * vocab + 3] = 100.0;
        }
        let logits_bv: mlx::Array = (&flat[..], &[batch as i32, vocab as i32][..])
            .try_into()
            .map_err(|e| anyhow::anyhow!("fake logits Array failed: {e:?}"))?;
        logits_bv
            .reshape(&[batch as i32, 1, vocab as i32][..])
            .map_err(|e| anyhow::anyhow!("fake logits reshape failed: {e:?}"))
    }

    fn fake_batched_logits(batch: usize, seq: usize, token: u32) -> Result<mlx::Array> {
        let vocab = 8_usize;
        let mut flat = vec![0.0_f32; batch * seq * vocab];
        for position in 0..batch * seq {
            flat[position * vocab + token as usize] = 100.0;
        }
        (&flat[..], &[batch as i32, seq as i32, vocab as i32][..])
            .try_into()
            .map_err(|e| anyhow::anyhow!("fake batched logits Array failed: {e:?}"))
    }

    fn mk_req(prompt_token: u32) -> GenerateRequest {
        GenerateRequest {
            prompt_ids: vec![prompt_token],
            max_new_tokens: 16,
            sampler: Sampler::greedy(),
            stop_token_ids: vec![2],
            prefill_chunk_size: 0,
            decode_cadence_mid_chunk_cap: 256,
            kv_cache_turboquant_bits: None,
            pixel_values: None,
            image_grid_thw: None,
            image_spatial_merge_size: 2,
            image_token_id: IMAGE_TOKEN_ID,
            constraint: None,
        }
    }

    #[derive(Clone)]
    struct SseDisconnectContractState {
        scheduler: SchedulerActorHandle,
        terminal_events: Arc<AtomicU64>,
    }

    async fn scheduler_disconnect_contract_stream(
        axum::extract::State(state): axum::extract::State<SseDisconnectContractState>,
    ) -> axum::response::Response {
        let (reply_tx, reply_rx) = oneshot::channel();
        state
            .scheduler
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: mk_req(11),
                reply_tx,
            })
            .await
            .expect("send disconnect-contract admission");
        let mut event_rx = reply_rx
            .await
            .expect("disconnect-contract admission reply")
            .expect("disconnect-contract admission accepted")
            .event_rx;

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if state.scheduler.b_active.load(Ordering::Relaxed) == 1
                    && state
                        .scheduler
                        .kv_cache_active_bytes
                        .load(Ordering::Relaxed)
                        > 0
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("scheduler resources must be live before returning SSE response");

        let (tx, rx, disconnect) =
            crate::core::server::api_transport::disconnect_aware_sse_channel(2);
        let terminal_events = state.terminal_events;
        tokio::spawn(async move {
            if tx
                .send(Ok(axum::body::Bytes::from_static(
                    b"data: {\"type\":\"started\"}\n\n",
                )))
                .await
                .is_err()
            {
                return;
            }

            while let Some(event) =
                crate::core::server::api_transport::recv_or_disconnect(&disconnect, &mut event_rx)
                    .await
            {
                let terminal = event.finish_reason.is_some();
                if terminal {
                    terminal_events.fetch_add(1, Ordering::Relaxed);
                }
                let frame = format!("data: {{\"token\":{}}}\n\n", event.token);
                if tx.send(Ok(axum::body::Bytes::from(frame))).await.is_err() {
                    return;
                }
                if terminal {
                    return;
                }
            }
        });

        crate::core::server::api_transport::disconnect_aware_sse_response(rx)
    }

    async fn disconnect_tcp_client_after_first_sse_frame(
        address: std::net::SocketAddr,
        path: &str,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let mut stream = tokio::net::TcpStream::connect(address)
            .await
            .expect("connect contract client");
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}"
        );
        stream
            .write_all(request.as_bytes())
            .await
            .expect("write contract request");

        let response = tokio::time::timeout(Duration::from_secs(2), async {
            let mut response = Vec::new();
            let mut buffer = [0_u8; 512];
            loop {
                let read = stream.read(&mut buffer).await.expect("read SSE response");
                assert!(read > 0, "SSE response closed before its first frame");
                response.extend_from_slice(&buffer[..read]);
                if response
                    .windows(b"data:".len())
                    .any(|part| part == b"data:")
                {
                    return response;
                }
            }
        })
        .await
        .expect("first SSE frame timeout");
        let response = String::from_utf8_lossy(&response);
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");

        drop(stream);
    }

    async fn wait_for_scheduler_resources_to_be_released(handle: &SchedulerActorHandle) {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if handle.b_active.load(Ordering::Relaxed) == 0
                    && handle.b_queued.load(Ordering::Relaxed) == 0
                    && handle.kv_cache_active_bytes.load(Ordering::Relaxed) == 0
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("disconnect must release scheduler slot and KV budget");
    }

    fn mk_vl_req() -> GenerateRequest {
        let mut req = mk_req(11);
        req.prompt_ids = vec![11, IMAGE_TOKEN_ID as u32, 12];
        req.max_new_tokens = 1;
        req.pixel_values = Some(vec![
            mlx::Array::zeros((1_i32, 1_i32), mlx::Dtype::Float32).unwrap()
        ]);
        req.image_grid_thw = Some(vec![(1, 2, 2)]);
        req
    }

    fn queued_pending(prompt_token: u32) -> (PendingAdmit, oneshot::Receiver<Result<AdmitReply>>) {
        let (reply_tx, reply_rx) = oneshot::channel();
        (
            PendingAdmit {
                request: mk_req(prompt_token),
                reply_tx,
                queued_at_profile: None,
            },
            reply_rx,
        )
    }

    fn test_mtp_counters() -> SchedulerActorMtpCounters {
        let counter = || Arc::new(AtomicU64::new(0));
        SchedulerActorMtpCounters::new(
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            counter(),
            Arc::new(StdMutex::new(None)),
            Arc::new(StdMutex::new(NeuralExactQualificationStats::default())),
        )
    }

    fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_mtp_actor_accepts_paged_prefix_cache_config() {
        let root = unique_temp_dir("actor-mtp-prefix");
        let config = PagedPrefixCacheConfig::new(&root, "fake-qwen", 16, 8).expect("prefix config");
        let qualification = NeuralExactQualificationRuntimeConfig::for_test(
            NeuralExactSource::QwenMtp,
            "fake-qwen",
            root.join("qualification.json"),
        );
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let handle = spawn_scheduler_actor_with_mtp(
            model,
            SchedulerActorFakeMtpHead,
            1,
            qualification,
            2,
            Duration::from_millis(1),
            1,
            32,
            256,
            crate::core::memory_budget::test_meta_qwen35(),
            Some(config),
            None,
        )
        .expect("spawn mtp actor with prefix cache");

        assert_eq!(handle.mtp_prefill_count.load(Ordering::Relaxed), 0);
        drop(handle);
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_actor_with_paged_prefix_and_active_kv_accepts_large_logical_cap() {
        let prefix_root = unique_temp_dir("actor-gemma4-prefix");
        let active_kv_root = unique_temp_dir("actor-gemma4-active-kv");
        let config = PagedPrefixCacheConfig::new(&prefix_root, "fake-gemma4", 128, 4096)
            .expect("prefix config");
        let active_kv_offload = ActiveKvOffloadConfig::enabled(active_kv_root.clone());
        let meta = crate::core::memory_budget::test_meta_gemma4_12b();
        let effective_cap_max = 262_144;
        let policy = startup_budget_policy(effective_cap_max, Some(&config), &active_kv_offload);
        let resident_cap = policy.resident_cap(effective_cap_max);
        let resident_bytes = crate::core::memory_budget::kv_cache_bytes(1, resident_cap, &meta);
        let budget_state = crate::core::memory_budget::BudgetState::with_soft_limit(
            resident_bytes,
            effective_cap_max,
            resident_cap,
            policy,
        );
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let handle = spawn_scheduler_actor_with_mode_and_budget_state(
            model,
            SchedulerActorNoMtp,
            1,
            Duration::from_millis(1),
            1,
            effective_cap_max,
            256,
            meta,
            Some(config),
            None,
            AdaptiveAdmissionPolicy::disabled(),
            active_kv_offload,
            budget_state,
        )
        .expect("active KV + paged prefix should allow 256K logical Gemma4 cap");

        drop(handle);
        std::fs::remove_dir_all(prefix_root).ok();
        std::fs::remove_dir_all(active_kv_root).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn actor_active_kv_offload_does_not_park_only_decode_row_for_admission() {
        let root = unique_temp_dir("actor-active-kv");
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel::with_forward_delay(
            Duration::from_millis(25),
        )));
        let handle = spawn_scheduler_actor_with_active_kv_offload(
            model,
            1,
            Duration::from_millis(1),
            4,
            32,
            256,
            crate::core::memory_budget::test_meta_qwen35(),
            ActiveKvOffloadConfig::enabled(root.clone()),
        )
        .expect("spawn actor with active kv offload");

        let (reply_tx_1, reply_rx_1) = oneshot::channel();
        let mut request_1 = mk_req(11);
        request_1.max_new_tokens = 4;
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: request_1,
                reply_tx: reply_tx_1,
            })
            .await
            .expect("send first request");
        let mut events_1 = reply_rx_1
            .await
            .expect("first reply")
            .expect("first admit")
            .event_rx;
        let first_event = tokio::time::timeout(Duration::from_secs(2), events_1.recv())
            .await
            .expect("first event timeout")
            .expect("first event");
        assert_eq!(first_event.finish_reason, None);

        let (reply_tx_2, reply_rx_2) = oneshot::channel();
        let mut request_2 = mk_req(22);
        request_2.max_new_tokens = 1;
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: request_2,
                reply_tx: reply_tx_2,
            })
            .await
            .expect("send second request");
        let mut events_2 = tokio::time::timeout(Duration::from_secs(2), reply_rx_2)
            .await
            .expect("second reply timeout")
            .expect("second reply")
            .expect("second admit")
            .event_rx;

        let mut second_finished = false;
        while let Some(event) = tokio::time::timeout(Duration::from_secs(2), events_2.recv())
            .await
            .expect("second event timeout")
        {
            if event.finish_reason.is_some() {
                second_finished = true;
                break;
            }
        }
        assert!(
            second_finished,
            "second request should finish after the first releases the slot"
        );

        let mut first_finished = false;
        while let Some(event) = tokio::time::timeout(Duration::from_secs(2), events_1.recv())
            .await
            .expect("restored first event timeout")
        {
            if event.finish_reason.is_some() {
                first_finished = true;
                break;
            }
        }
        assert!(first_finished, "first request should finish normally");

        let health = handle.active_kv_offload.snapshot();
        assert_eq!(health.swap_out_count, 0, "last row must not be swapped out");
        assert_eq!(health.swap_in_count, 0, "last row must not be swapped in");
        assert_eq!(health.swap_error_count, 0);
        assert_eq!(health.parked_requests, 0);

        drop(handle);
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mtp_actor_active_kv_offload_does_not_park_only_speculative_row_for_admission() {
        let root = unique_temp_dir("actor-active-kv-mtp");
        let qualification = NeuralExactQualificationRuntimeConfig::for_test(
            NeuralExactSource::QwenMtp,
            "fake-qwen-mtp-active-kv",
            root.join("qualification.json"),
        );
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel::with_forward_delay(
            Duration::from_millis(25),
        )));
        let handle = spawn_scheduler_actor_with_mtp_and_active_kv(
            model,
            SchedulerActorFakeMtpHead,
            1,
            qualification,
            1,
            Duration::from_millis(1),
            4,
            32,
            256,
            crate::core::memory_budget::test_meta_qwen35(),
            None,
            None,
            ActiveKvOffloadConfig::enabled(root.clone()),
        )
        .expect("spawn MTP actor with active KV offload");

        let (reply_tx_1, reply_rx_1) = oneshot::channel();
        let mut request_1 = mk_req(11);
        request_1.max_new_tokens = 4;
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: request_1,
                reply_tx: reply_tx_1,
            })
            .await
            .expect("send first MTP request");
        let mut events_1 = reply_rx_1
            .await
            .expect("first reply")
            .expect("first admit")
            .event_rx;
        let first_event = tokio::time::timeout(Duration::from_secs(2), events_1.recv())
            .await
            .expect("first MTP event timeout")
            .expect("first MTP event");
        assert_eq!(first_event.finish_reason, None);

        let (reply_tx_2, reply_rx_2) = oneshot::channel();
        let mut request_2 = mk_req(22);
        request_2.max_new_tokens = 1;
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: request_2,
                reply_tx: reply_tx_2,
            })
            .await
            .expect("send second request");
        let mut events_2 = tokio::time::timeout(Duration::from_secs(2), reply_rx_2)
            .await
            .expect("second reply timeout")
            .expect("second reply")
            .expect("second admit")
            .event_rx;

        while let Some(event) = tokio::time::timeout(Duration::from_secs(2), events_2.recv())
            .await
            .expect("second event timeout")
        {
            if event.finish_reason.is_some() {
                break;
            }
        }
        let mut first_finished = false;
        while let Some(event) = tokio::time::timeout(Duration::from_secs(2), events_1.recv())
            .await
            .expect("restored MTP event timeout")
        {
            if event.finish_reason.is_some() {
                first_finished = true;
                break;
            }
        }
        assert!(first_finished, "first MTP request should finish normally");

        let health = handle.active_kv_offload.snapshot();
        assert_eq!(health.swap_out_count, 0, "last MTP row must stay resident");
        assert_eq!(health.swap_in_count, 0, "last MTP row must stay resident");
        assert_eq!(health.swap_error_count, 0);
        assert_eq!(health.parked_requests, 0);

        drop(handle);
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mtp_actor_active_kv_offload_parks_when_decode_row_remains() {
        let root = unique_temp_dir("actor-active-kv-mtp-useful-park");
        let qualification = NeuralExactQualificationRuntimeConfig::for_test(
            NeuralExactSource::QwenMtp,
            "fake-qwen-mtp-active-kv",
            root.join("qualification.json"),
        );
        let model = Arc::new(Mutex::new(
            SchedulerActorFakeModel::with_mtp_accepted_prefix_restore(Duration::from_millis(25)),
        ));
        let handle = spawn_scheduler_actor_with_mtp_and_active_kv(
            model,
            SchedulerActorFakeMtpHead,
            1,
            qualification,
            2,
            Duration::from_millis(1),
            8,
            2048,
            256,
            crate::core::memory_budget::test_meta_qwen35(),
            None,
            None,
            ActiveKvOffloadConfig::enabled(root.clone()),
        )
        .expect("spawn MTP actor with active KV offload");

        let (reply_tx_1, reply_rx_1) = oneshot::channel();
        let mut request_1 = mk_req(11);
        request_1.max_new_tokens = 1_000;
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: request_1,
                reply_tx: reply_tx_1,
            })
            .await
            .expect("send first MTP request");
        let mut events_1 = reply_rx_1
            .await
            .expect("first reply")
            .expect("first admit")
            .event_rx;
        let first_event = tokio::time::timeout(Duration::from_secs(2), events_1.recv())
            .await
            .expect("first MTP event timeout")
            .expect("first MTP event");
        assert_eq!(first_event.finish_reason, None);

        let (reply_tx_2, reply_rx_2) = oneshot::channel();
        let mut request_2 = mk_req(22);
        request_2.max_new_tokens = 1_000;
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: request_2,
                reply_tx: reply_tx_2,
            })
            .await
            .expect("send second MTP request");
        let events_2 = reply_rx_2
            .await
            .expect("second reply")
            .expect("second admit")
            .event_rx;

        let (reply_tx_3, reply_rx_3) = oneshot::channel();
        let mut request_3 = mk_req(33);
        request_3.max_new_tokens = 1;
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: request_3,
                reply_tx: reply_tx_3,
            })
            .await
            .expect("send third MTP request");
        let mut events_3 = tokio::time::timeout(Duration::from_secs(2), reply_rx_3)
            .await
            .expect("third reply timeout")
            .expect("third reply")
            .expect("third admit")
            .event_rx;
        let third_event = tokio::time::timeout(Duration::from_secs(2), events_3.recv())
            .await
            .expect("third event timeout")
            .expect("third event");
        assert_eq!(third_event.finish_reason, Some("length"));

        let health = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let health = handle.active_kv_offload.snapshot();
                if health.swap_out_count >= 1
                    && health.swap_in_count >= 1
                    && health.parked_requests == 0
                {
                    break health;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("MTP Active KV park/restore state timeout");
        assert!(
            health.swap_out_count >= 1,
            "expected useful MTP swap out: {health:?}"
        );
        assert!(health.swap_in_count >= 1, "expected useful MTP swap in");
        assert_eq!(health.swap_error_count, 0);
        assert_eq!(health.parked_requests, 0);

        drop(events_1);
        drop(events_2);
        wait_for_scheduler_resources_to_be_released(&handle).await;

        drop(handle);
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn actor_preserves_queue_order_without_parking_only_decode_row() {
        let root = unique_temp_dir("actor-active-kv-fairness");
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel::with_forward_delay(
            Duration::from_millis(25),
        )));
        let handle = spawn_scheduler_actor_with_active_kv_offload(
            model,
            1,
            Duration::from_millis(1),
            4,
            32,
            256,
            crate::core::memory_budget::test_meta_qwen35(),
            ActiveKvOffloadConfig::enabled(root.clone()),
        )
        .expect("spawn actor with active kv offload");

        let (reply_tx_1, reply_rx_1) = oneshot::channel();
        let mut request_1 = mk_req(11);
        request_1.max_new_tokens = 4;
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: request_1,
                reply_tx: reply_tx_1,
            })
            .await
            .expect("send first request");
        let mut events_1 = reply_rx_1
            .await
            .expect("first reply")
            .expect("first admit")
            .event_rx;
        let first_event = tokio::time::timeout(Duration::from_secs(2), events_1.recv())
            .await
            .expect("first event timeout")
            .expect("first event");
        assert_eq!(first_event.finish_reason, None);

        let (reply_tx_2, reply_rx_2) = oneshot::channel();
        let mut request_2 = mk_req(22);
        request_2.max_new_tokens = 4;
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: request_2,
                reply_tx: reply_tx_2,
            })
            .await
            .expect("send second request");
        let mut reply_rx_2 = Box::pin(reply_rx_2);
        loop {
            tokio::select! {
                biased;
                second = &mut reply_rx_2 => {
                    let _ = second.expect("second reply channel").expect("second admit");
                    panic!("second queued request was admitted before the resident first request finished");
                }
                event = events_1.recv() => {
                    let event = event.expect("first request event");
                    if event.finish_reason.is_some() {
                        break;
                    }
                }
            }
        }

        let mut events_2 = tokio::time::timeout(Duration::from_secs(2), &mut reply_rx_2)
            .await
            .expect("second reply timeout")
            .expect("second reply")
            .expect("second admit")
            .event_rx;
        while let Some(event) = tokio::time::timeout(Duration::from_secs(2), events_2.recv())
            .await
            .expect("second event timeout")
        {
            if event.finish_reason.is_some() {
                break;
            }
        }

        let health = handle.active_kv_offload.snapshot();
        assert_eq!(health.swap_out_count, 0);
        assert_eq!(health.swap_in_count, 0);
        assert_eq!(health.swap_error_count, 0);
        assert_eq!(health.parked_requests, 0);

        drop(handle);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn actor_mtp_mode_prefill_and_step_use_mtp_for_eligible_request() {
        let mut scheduler = test_scheduler(1, 32);
        scheduler.admit(mk_req(11)).expect("admit");
        let counters = test_mtp_counters();
        let mut mode = SchedulerActorMtp::new(SchedulerActorFakeMtpHead, 1);

        let prefill_events = mode
            .prefill_admitted(&mut scheduler, &SchedulerActorFakeModel, &counters)
            .expect("mtp prefill");
        assert_eq!(prefill_events.len(), 1);
        assert_eq!(counters.mtp_prefill_count.load(Ordering::Relaxed), 1);
        assert_eq!(
            counters.mtp_prefill_fallback_count.load(Ordering::Relaxed),
            0
        );
        assert!(scheduler.mtp_stats().is_some());

        let step_events = mode
            .step(&mut scheduler, &SchedulerActorFakeModel, &counters, false)
            .expect("mtp step");
        assert_eq!(step_events.len(), 1);
        assert_eq!(counters.mtp_step_count.load(Ordering::Relaxed), 1);
        let drafted = counters.mtp_drafted_tokens.load(Ordering::Relaxed);
        let accepted = counters.mtp_accepted_draft_tokens.load(Ordering::Relaxed);
        assert!(
            drafted >= 1,
            "eligible request must draft at least one token"
        );
        assert!(
            accepted <= drafted,
            "accepted draft tokens cannot exceed drafted tokens"
        );
    }

    #[test]
    fn prompt_lookup_admission_pressure_only_blocks_new_windows() {
        assert!(prompt_lookup_admission_forces_ordinary(false, true));
        assert!(!prompt_lookup_admission_forces_ordinary(true, true));
        assert!(!prompt_lookup_admission_forces_ordinary(false, false));
    }

    #[test]
    fn qwen_hybrid_routes_certified_lookup_to_canonical_target() {
        let base = PromptLookupQualificationRegime::new(2, 1024, Sampler::greedy());
        assert!(qwen_hybrid_uses_canonical_target(
            HybridDraftSource::PromptLookup,
            Some(base.with_proposal(PromptLookupProposalSource::Shared, 3)),
        ));
        assert!(qwen_hybrid_uses_canonical_target(
            HybridDraftSource::PromptLookup,
            Some(base.with_proposal(PromptLookupProposalSource::Local, 3)),
        ));
        assert!(!qwen_hybrid_uses_canonical_target(
            HybridDraftSource::Neural,
            Some(base.with_proposal(PromptLookupProposalSource::Shared, 3)),
        ));
        assert!(!qwen_hybrid_uses_canonical_target(
            HybridDraftSource::PromptLookup,
            Some(base.with_proposal(PromptLookupProposalSource::Mixed, 3)),
        ));
    }

    #[test]
    fn prompt_lookup_query_hint_is_bound_to_base_regime_and_request_owners() {
        let base = PromptLookupQualificationRegime::new(2, 1024, Sampler::greedy());
        let scope = PromptLookupQueryScope {
            base_regime: base,
            owners: vec![RequestId(11), RequestId(12)],
            draft_limits: PromptLookupDraftLimits::uniform(4),
        };
        let proposal_regime = base.with_proposal(PromptLookupProposalSource::Shared, 4);
        let hint = PromptLookupQueryHint {
            scope: scope.clone(),
            proposal_regime,
        };

        assert_eq!(hint.proposal_regime_for(&scope), Some(proposal_regime));
        assert_eq!(
            hint.proposal_regime_for(&PromptLookupQueryScope {
                base_regime: PromptLookupQualificationRegime::new(2, 8193, Sampler::greedy()),
                owners: scope.owners.clone(),
                draft_limits: scope.draft_limits,
            }),
            None
        );
        assert_eq!(
            hint.proposal_regime_for(&PromptLookupQueryScope {
                base_regime: base,
                owners: vec![RequestId(11), RequestId(13)],
                draft_limits: scope.draft_limits,
            }),
            None
        );
        assert_eq!(
            hint.proposal_regime_for(&PromptLookupQueryScope {
                base_regime: base,
                owners: scope.owners.clone(),
                draft_limits: PromptLookupDraftLimits::new(3, 4),
            }),
            None
        );
    }

    #[test]
    fn prompt_lookup_miss_query_hint_uses_bounded_progress_and_invalidates_identity() {
        let base = PromptLookupQualificationRegime::new(2, 1024, Sampler::greedy());
        let scope = PromptLookupMissQueryScope {
            base_regime: base,
            request_progress: vec![(RequestId(11), 100), (RequestId(12), 200)],
            allow_cross_request: true,
            shared_availability_epoch: Some(3),
        };
        let first = PromptLookupMissQueryHint::after_miss(scope.clone(), None);
        assert_eq!(first.reprobe_after_tokens, 2);

        let mut progressed = scope.clone();
        progressed.request_progress[0].1 += 1;
        assert!(first.should_skip(&progressed));
        progressed.request_progress[0].1 += 1;
        assert!(!first.should_skip(&progressed));

        let second = PromptLookupMissQueryHint::after_miss(progressed.clone(), Some(&first));
        assert_eq!(second.reprobe_after_tokens, 4);
        progressed.request_progress[1].1 += 3;
        assert!(second.should_skip(&progressed));
        progressed.request_progress[1].1 += 1;
        assert!(!second.should_skip(&progressed));

        let third = PromptLookupMissQueryHint::after_miss(progressed.clone(), Some(&second));
        assert_eq!(third.reprobe_after_tokens, 8);
        let capped = PromptLookupMissQueryHint::after_miss(progressed.clone(), Some(&third));
        assert_eq!(capped.reprobe_after_tokens, 8);

        let mut changed_epoch = progressed.clone();
        changed_epoch.shared_availability_epoch = Some(4);
        assert!(!capped.identity_matches(&changed_epoch));
        assert!(!capped.should_skip(&changed_epoch));

        let mut changed_owner = progressed;
        changed_owner.request_progress[1].0 = RequestId(13);
        assert!(!capped.identity_matches(&changed_owner));
        assert!(!capped.should_skip(&changed_owner));
    }

    #[test]
    fn prompt_lookup_miss_query_gate_skips_repeated_actor_query() {
        let profile_path = std::env::temp_dir().join(format!(
            "ironmlx-miss-query-gate-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ));
        let qualification = PromptLookupQualificationRuntimeConfig::for_test(
            "miss-query-gate",
            profile_path.clone(),
        );
        let cfg = PromptLookupConfig {
            min_ngram: 2,
            max_ngram: 3,
            max_draft_tokens: 3,
            history_window_tokens: 64,
            max_index_entries: 128,
            cross_request: false,
        };
        let mut mode = SchedulerActorPromptLookup::new(cfg, qualification).expect("lookup mode");
        let mut scheduler = test_scheduler(1, 32);
        let mut request = mk_req(1);
        request.prompt_ids = vec![1, 2, 4, 5, 6];
        scheduler.admit(request).expect("admit");
        let counters = test_mtp_counters();
        mode.prefill_admitted(&mut scheduler, &SchedulerActorFakeModel, &counters)
            .expect("prefill");

        let first = mode
            .select_prepared_window(
                &mut scheduler,
                &SchedulerActorFakeModel,
                false,
                false,
                false,
                false,
            )
            .expect("first miss");
        assert_eq!(first.action, PromptLookupCostAction::Ordinary);
        assert!(first.fallback_to_baseline);
        let queries_after_first = scheduler
            .prompt_lookup_stats()
            .expect("lookup stats")
            .queries;
        assert_eq!(queries_after_first, 1);

        let second = mode
            .select_prepared_window(
                &mut scheduler,
                &SchedulerActorFakeModel,
                false,
                false,
                false,
                false,
            )
            .expect("gated miss");
        assert_eq!(second.action, PromptLookupCostAction::Ordinary);
        assert!(second.fallback_to_baseline);
        assert_eq!(
            scheduler
                .prompt_lookup_stats()
                .expect("lookup stats")
                .queries,
            queries_after_first
        );
        assert_eq!(mode.cost_controller.stats().miss_query_gate_skips, 1);
        assert_eq!(mode.cost_controller.stats().miss_query_reprobes, 0);

        drop(mode);
        std::fs::remove_file(profile_path).ok();
    }

    #[test]
    fn prompt_lookup_query_gate_skips_repeated_baseline_proposal_search() {
        let profile_path = std::env::temp_dir().join(format!(
            "ironmlx-query-gate-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        ));
        let qualification =
            PromptLookupQualificationRuntimeConfig::for_test("query-gate", profile_path.clone());
        let cfg = PromptLookupConfig {
            min_ngram: 2,
            max_ngram: 3,
            max_draft_tokens: 3,
            history_window_tokens: 64,
            max_index_entries: 128,
            cross_request: false,
        };
        let mut mode = SchedulerActorPromptLookup::new(cfg, qualification).expect("lookup mode");
        let mut scheduler = test_scheduler(1, 32);
        let mut request = mk_req(1);
        request.prompt_ids = vec![1, 2, 3, 4, 1, 2];
        scheduler.admit(request).expect("admit");
        let counters = test_mtp_counters();
        mode.prefill_admitted(&mut scheduler, &SchedulerActorFakeModel, &counters)
            .expect("prefill");

        let first = mode
            .select_prepared_window(
                &mut scheduler,
                &SchedulerActorFakeModel,
                false,
                false,
                false,
                false,
            )
            .expect("first proposal");
        assert_eq!(first.action, PromptLookupCostAction::Ordinary);
        assert!(first.regime.is_some_and(|regime| {
            regime.proposal_source == Some(PromptLookupProposalSource::Local)
        }));
        let queries_after_first = scheduler
            .prompt_lookup_stats()
            .expect("lookup stats")
            .queries;
        assert_eq!(queries_after_first, 1);

        let second = mode
            .select_prepared_window(
                &mut scheduler,
                &SchedulerActorFakeModel,
                false,
                false,
                false,
                false,
            )
            .expect("gated proposal");
        assert_eq!(second.action, PromptLookupCostAction::Ordinary);
        assert_eq!(
            scheduler
                .prompt_lookup_stats()
                .expect("lookup stats")
                .queries,
            queries_after_first
        );
        assert_eq!(mode.cost_controller.stats().query_gate_skips, 1);

        let regime = first.regime.expect("proposal regime");
        for _ in 0..8 {
            mode.cost_controller.record_sample(
                regime,
                PromptLookupCostAction::Ordinary,
                100_000,
                1,
                PromptLookupStats::default(),
            );
        }
        assert_eq!(
            mode.cost_controller.next_action(regime),
            PromptLookupCostAction::Lookup
        );

        let probe = mode
            .select_prepared_window(
                &mut scheduler,
                &SchedulerActorFakeModel,
                false,
                false,
                false,
                false,
            )
            .expect("probe proposal");
        assert_eq!(probe.action, PromptLookupCostAction::Lookup);
        assert_eq!(
            scheduler
                .prompt_lookup_stats()
                .expect("lookup stats")
                .queries,
            queries_after_first + 1
        );
        drop(mode);
        std::fs::remove_file(profile_path).ok();
    }

    #[test]
    fn mtp_counters_publish_cumulative_stat_deltas() {
        let counters = test_mtp_counters();
        let first = MtpSpeculativeStats {
            windows: 1,
            drafted_tokens: 2,
            accepted_draft_tokens: 1,
            draft_forward_us: 10,
            verify_forward_us: 20,
            projection_us: 30,
            sampling_us: 40,
            main_rollback_us: 50,
            mtp_cache_commit_us: 60,
            mtp_prefill_cache_commit_us: 25,
            mtp_decode_cache_commit_us: 35,
            mtp_cache_restore_us: 70,
            ..MtpSpeculativeStats::default()
        };
        counters.store_stats(Some(first));

        let second = MtpSpeculativeStats {
            windows: 3,
            drafted_tokens: 6,
            accepted_draft_tokens: 4,
            draft_forward_us: 15,
            verify_forward_us: 35,
            projection_us: 30,
            sampling_us: 55,
            main_rollback_us: 70,
            mtp_cache_commit_us: 90,
            mtp_prefill_cache_commit_us: 40,
            mtp_decode_cache_commit_us: 50,
            mtp_cache_restore_us: 75,
            ..MtpSpeculativeStats::default()
        };
        counters.store_stats(Some(second));

        assert_eq!(counters.mtp_windows.load(Ordering::Relaxed), 3);
        assert_eq!(counters.mtp_drafted_tokens.load(Ordering::Relaxed), 6);
        assert_eq!(
            counters.mtp_accepted_draft_tokens.load(Ordering::Relaxed),
            4
        );
        assert_eq!(counters.mtp_draft_forward_us.load(Ordering::Relaxed), 15);
        assert_eq!(counters.mtp_verify_forward_us.load(Ordering::Relaxed), 35);
        assert_eq!(counters.mtp_projection_us.load(Ordering::Relaxed), 30);
        assert_eq!(counters.mtp_sampling_us.load(Ordering::Relaxed), 55);
        assert_eq!(counters.mtp_main_rollback_us.load(Ordering::Relaxed), 70);
        assert_eq!(counters.mtp_cache_commit_us.load(Ordering::Relaxed), 90);
        assert_eq!(
            counters.mtp_prefill_cache_commit_us.load(Ordering::Relaxed),
            40
        );
        assert_eq!(
            counters.mtp_decode_cache_commit_us.load(Ordering::Relaxed),
            50
        );
        assert_eq!(counters.mtp_cache_restore_us.load(Ordering::Relaxed), 75);

        counters.reset_stats_baseline(None);
        let next_batch = MtpSpeculativeStats {
            windows: 1,
            drafted_tokens: 3,
            accepted_draft_tokens: 2,
            draft_forward_us: 100,
            verify_forward_us: 200,
            projection_us: 300,
            sampling_us: 400,
            main_rollback_us: 500,
            mtp_cache_commit_us: 600,
            mtp_prefill_cache_commit_us: 250,
            mtp_decode_cache_commit_us: 350,
            mtp_cache_restore_us: 700,
            ..MtpSpeculativeStats::default()
        };
        counters.store_stats(Some(next_batch));

        assert_eq!(counters.mtp_windows.load(Ordering::Relaxed), 4);
        assert_eq!(counters.mtp_drafted_tokens.load(Ordering::Relaxed), 9);
        assert_eq!(
            counters.mtp_accepted_draft_tokens.load(Ordering::Relaxed),
            6
        );
        assert_eq!(counters.mtp_draft_forward_us.load(Ordering::Relaxed), 115);
        assert_eq!(counters.mtp_verify_forward_us.load(Ordering::Relaxed), 235);
        assert_eq!(counters.mtp_projection_us.load(Ordering::Relaxed), 330);
        assert_eq!(counters.mtp_sampling_us.load(Ordering::Relaxed), 455);
        assert_eq!(counters.mtp_main_rollback_us.load(Ordering::Relaxed), 570);
        assert_eq!(counters.mtp_cache_commit_us.load(Ordering::Relaxed), 690);
        assert_eq!(
            counters.mtp_prefill_cache_commit_us.load(Ordering::Relaxed),
            290
        );
        assert_eq!(
            counters.mtp_decode_cache_commit_us.load(Ordering::Relaxed),
            400
        );
        assert_eq!(counters.mtp_cache_restore_us.load(Ordering::Relaxed), 775);
    }

    #[test]
    fn prompt_lookup_counters_publish_cumulative_batches_and_live_index_state() {
        let counters = test_mtp_counters();
        counters.store_prompt_lookup_stats(Some(PromptLookupStats {
            queries: 3,
            hits: 2,
            misses: 1,
            drafted_tokens: 7,
            accepted_tokens: 5,
            rejected_tokens: 2,
            index_entries_current: 11,
            index_entries_peak: 13,
            index_ledger_entries_current: 20,
            index_ledger_entries_peak: 25,
            index_estimated_bytes_current: 1_000,
            index_estimated_bytes_peak: 1_200,
            ..PromptLookupStats::default()
        }));
        counters.store_prompt_lookup_stats(Some(PromptLookupStats {
            queries: 5,
            hits: 3,
            misses: 2,
            drafted_tokens: 10,
            accepted_tokens: 8,
            rejected_tokens: 2,
            index_entries_current: 0,
            index_entries_peak: 17,
            index_ledger_entries_current: 0,
            index_ledger_entries_peak: 30,
            index_estimated_bytes_current: 0,
            index_estimated_bytes_peak: 2_000,
            ..PromptLookupStats::default()
        }));
        counters.store_prompt_lookup_stats_with_qualification(
            None,
            PromptLookupQualificationStats {
                ordinary_cost_samples: 8,
                lookup_cost_samples: 9,
                ordinary_cost_us: 10,
                lookup_cost_us: 11,
                qualified_regimes_current: 1,
                rejected_regimes_current: 2,
                qualification_changes: 3,
                profile_loads: 1,
                profile_writes: 4,
                profile_write_drops: 1,
                query_gate_skips: 5,
                miss_query_gate_skips: 6,
                miss_query_reprobes: 7,
                ..PromptLookupQualificationStats::default()
            },
        );
        counters.reset_stats_baseline(None);
        counters.store_prompt_lookup_stats_with_qualification(
            Some(PromptLookupStats {
                queries: 4,
                hits: 1,
                misses: 3,
                drafted_tokens: 6,
                accepted_tokens: 2,
                rejected_tokens: 4,
                index_entries_current: 9,
                index_entries_peak: 12,
                index_ledger_entries_current: 18,
                index_ledger_entries_peak: 22,
                index_estimated_bytes_current: 900,
                index_estimated_bytes_peak: 1_100,
                ..PromptLookupStats::default()
            }),
            PromptLookupQualificationStats {
                ordinary_cost_samples: 8,
                lookup_cost_samples: 9,
                ordinary_cost_us: 10,
                lookup_cost_us: 11,
                qualified_regimes_current: 1,
                rejected_regimes_current: 2,
                qualification_changes: 3,
                profile_loads: 1,
                profile_writes: 4,
                profile_write_drops: 1,
                query_gate_skips: 5,
                miss_query_gate_skips: 6,
                miss_query_reprobes: 7,
                ..PromptLookupQualificationStats::default()
            },
        );

        let stats = counters
            .prompt_lookup_published_stats
            .lock()
            .expect("PromptLookup published stats")
            .expect("PromptLookup stats were published");
        assert_eq!(stats.queries, 9);
        assert_eq!(stats.hits, 4);
        assert_eq!(stats.misses, 5);
        assert_eq!(stats.drafted_tokens, 16);
        assert_eq!(stats.accepted_tokens, 10);
        assert_eq!(stats.rejected_tokens, 6);
        assert_eq!(stats.index_entries_current, 9);
        assert_eq!(stats.index_entries_peak, 17);
        assert_eq!(stats.index_ledger_entries_current, 18);
        assert_eq!(stats.index_ledger_entries_peak, 30);
        assert_eq!(stats.index_estimated_bytes_current, 900);
        assert_eq!(stats.index_estimated_bytes_peak, 2_000);
        assert_eq!(stats.ordinary_cost_samples, 8);
        assert_eq!(stats.lookup_cost_samples, 9);
        assert_eq!(stats.qualified_regimes_current, 1);
        assert_eq!(stats.rejected_regimes_current, 2);
        assert_eq!(stats.qualification_profile_writes, 4);
        assert_eq!(stats.qualification_query_gate_skips, 5);
        assert_eq!(stats.miss_query_gate_skips, 6);
        assert_eq!(stats.miss_query_reprobes, 7);

        counters.reset_stats_baseline(None);
        let stats = counters
            .prompt_lookup_published_stats
            .lock()
            .expect("PromptLookup published stats")
            .expect("PromptLookup stats remain cumulative while idle");
        assert_eq!(stats.queries, 9);
        assert_eq!(stats.index_entries_current, 0);
        assert_eq!(stats.index_entries_peak, 17);
        assert_eq!(stats.index_ledger_entries_current, 0);
        assert_eq!(stats.index_ledger_entries_peak, 30);
        assert_eq!(stats.index_estimated_bytes_current, 0);
        assert_eq!(stats.index_estimated_bytes_peak, 2_000);
    }

    #[test]
    fn prompt_lookup_counters_preserve_shared_pool_baseline_across_batches() {
        let counters = test_mtp_counters();
        let first = PromptLookupStats {
            shared_queries: 2,
            shared_hits: 1,
            shared_misses: 1,
            shared_published_requests: 1,
            shared_entries_current: 10,
            shared_entries_peak: 10,
            ..PromptLookupStats::default()
        };
        counters.store_prompt_lookup_stats(Some(first));
        counters.reset_stats_baseline(Some(first));

        counters.store_prompt_lookup_stats(Some(PromptLookupStats {
            shared_queries: 5,
            shared_hits: 3,
            shared_misses: 2,
            shared_published_requests: 2,
            shared_entries_current: 12,
            shared_entries_peak: 12,
            ..PromptLookupStats::default()
        }));

        let stats = counters
            .prompt_lookup_published_stats
            .lock()
            .expect("PromptLookup published stats")
            .expect("PromptLookup stats were published");
        assert_eq!(stats.shared_queries, 5);
        assert_eq!(stats.shared_hits, 3);
        assert_eq!(stats.shared_misses, 2);
        assert_eq!(stats.shared_published_requests, 2);
        assert_eq!(stats.shared_entries_current, 12);
        assert_eq!(stats.shared_entries_peak, 12);
    }

    #[test]
    fn prompt_lookup_scheduler_stats_do_not_clear_qualification_gauges() {
        let counters = test_mtp_counters();
        counters.store_prompt_lookup_stats_with_qualification(
            Some(PromptLookupStats::default()),
            PromptLookupQualificationStats {
                qualified_regimes_current: 1,
                rejected_regimes_current: 2,
                ..PromptLookupQualificationStats::default()
            },
        );

        counters.store_prompt_lookup_stats(Some(PromptLookupStats::default()));

        let stats = counters
            .prompt_lookup_published_stats
            .lock()
            .expect("PromptLookup published stats")
            .expect("PromptLookup stats were published");
        assert_eq!(stats.qualified_regimes_current, 1);
        assert_eq!(stats.rejected_regimes_current, 2);
    }

    #[test]
    fn recovered_prefill_failure_publishes_idle_scheduler_depth() {
        let mut scheduler = test_scheduler(4, 32);
        scheduler.admit(mk_req(11)).expect("admit");
        scheduler.evict_all().expect("recover failed prefill");

        let b_active = AtomicU64::new(4);
        let b_queued = AtomicU64::new(3);
        publish_scheduler_depth(&scheduler, 0, &b_active, &b_queued);

        assert_eq!(b_active.load(Ordering::Relaxed), 0);
        assert_eq!(b_queued.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn actor_mtp_mode_prefill_uses_mtp_for_eligible_vl_request() {
        let mut scheduler = test_scheduler(1, 32);
        scheduler.admit(mk_vl_req()).expect("admit");
        let counters = test_mtp_counters();
        let mut mode = SchedulerActorMtp::new(SchedulerActorFakeMtpHead, 1);

        let prefill_events = mode
            .prefill_admitted(&mut scheduler, &SchedulerActorFakeModel, &counters)
            .expect("VL MTP prefill");

        assert_eq!(prefill_events.len(), 1);
        assert_eq!(counters.mtp_prefill_count.load(Ordering::Relaxed), 1);
        assert_eq!(
            counters.mtp_prefill_fallback_count.load(Ordering::Relaxed),
            0
        );
        assert!(scheduler.mtp_stats().is_some());
    }

    #[test]
    fn actor_mtp_mode_prefill_uses_exact_mtp_for_sampled_b1_request() {
        let mut scheduler = test_scheduler(1, 32);
        let mut request = mk_req(11);
        request.sampler = Sampler::greedy().with_temperature(0.7);
        scheduler.admit(request).expect("admit");
        let counters = test_mtp_counters();
        let mut mode = SchedulerActorMtp::new(SchedulerActorFakeMtpHead, 1);

        let prefill_events = mode
            .prefill_admitted(&mut scheduler, &SchedulerActorFakeModel, &counters)
            .expect("sampled exact MTP prefill");

        assert_eq!(prefill_events.len(), 1);
        assert_eq!(counters.mtp_prefill_count.load(Ordering::Relaxed), 1);
        assert_eq!(
            counters.mtp_prefill_fallback_count.load(Ordering::Relaxed),
            0
        );
        assert!(counters.mtp_drafted_tokens.load(Ordering::Relaxed) > 0);
        assert!(counters.mtp_exact_sampling_windows.load(Ordering::Relaxed) > 0);
        assert!(scheduler.mtp_stats().is_some());
    }

    #[test]
    fn rolling_policy_forces_decode_burst_after_admission_work() {
        let mut policy = RollingAdmissionPolicy::default();

        assert!(!policy.should_force_decode(Phase::Decoding, true, true));
        policy.record_admission_work();
        for _ in 0..ROLLING_DECODE_STEPS_AFTER_ADMISSION_WORK {
            assert!(policy.should_force_decode(Phase::Decoding, true, true));
            policy.record_decode_step();
        }
        assert!(!policy.should_force_decode(Phase::Decoding, true, true));
    }

    #[test]
    fn rolling_policy_scales_decode_credit_with_mid_admit_chunk_work() {
        assert_eq!(decode_steps_after_mid_admit_chunk(256, 256), 4);
        assert_eq!(decode_steps_after_mid_admit_chunk(257, 256), 8);
        assert_eq!(decode_steps_after_mid_admit_chunk(2048, 256), 32);

        let mut policy = RollingAdmissionPolicy::default();
        policy
            .record_admission_work_with_decode_steps(decode_steps_after_mid_admit_chunk(2048, 256));
        for _ in 0..31 {
            assert!(policy.should_force_decode(Phase::Decoding, true, true));
            policy.record_decode_step();
        }
        assert!(policy.should_force_decode(Phase::Decoding, true, true));
        policy.record_decode_step();
        assert!(!policy.should_force_decode(Phase::Decoding, true, true));
    }

    #[test]
    fn rolling_policy_does_not_force_decode_without_active_decoding_rows() {
        let mut policy = RollingAdmissionPolicy::default();

        policy.record_admission_work();
        assert!(!policy.should_force_decode(Phase::Idle, true, true));
        assert!(!policy.should_force_decode(Phase::Finished, true, true));
        assert!(!policy.should_force_decode(Phase::Decoding, false, true));
    }

    #[test]
    fn rolling_policy_does_not_force_decode_without_known_admission_work() {
        let mut policy = RollingAdmissionPolicy::default();

        policy.record_admission_work();
        assert!(!policy.should_force_decode(Phase::Decoding, true, false));
    }

    #[test]
    fn abandoned_event_receiver_evicts_request_and_releases_slot() {
        let mut scheduler = test_scheduler(1, 32);
        let id = scheduler.admit(mk_req(11)).expect("admit");
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        drop(event_rx);
        let mut event_txs = HashMap::from([(id, event_tx)]);
        let mut in_flight: Option<AdmitMidHandle> = None;

        assert_eq!(
            evict_abandoned_active_requests::<_, SchedulerActorNoMtp>(
                &mut scheduler,
                &mut event_txs,
                &mut in_flight,
            ),
            1
        );
        assert_eq!(scheduler.active_count(), 0);
        assert_eq!(scheduler.phase(), Phase::Idle);
        assert!(event_txs.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn http_sse_disconnect_releases_scheduler_resources_for_all_public_protocols() {
        use axum::{routing::post, Router};

        let model = Arc::new(Mutex::new(SchedulerActorFakeModel::with_forward_delay(
            Duration::from_millis(100),
        )));
        let handle = spawn_scheduler_actor(
            model,
            1,
            Duration::from_millis(1),
            1,
            32,
            256,
            crate::core::memory_budget::test_meta_qwen35(),
        )
        .expect("spawn disconnect-contract scheduler");
        let terminal_events = Arc::new(AtomicU64::new(0));
        let state = SseDisconnectContractState {
            scheduler: handle.clone(),
            terminal_events: terminal_events.clone(),
        };
        let router = Router::new()
            .route(
                "/v1/chat/completions",
                post(scheduler_disconnect_contract_stream),
            )
            .route("/v1/responses", post(scheduler_disconnect_contract_stream))
            .route("/v1/messages", post(scheduler_disconnect_contract_stream))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind disconnect-contract server");
        let address = listener.local_addr().expect("contract server address");
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .await
                .expect("serve disconnect-contract router");
        });

        for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages"] {
            let terminal_before = terminal_events.load(Ordering::Relaxed);
            disconnect_tcp_client_after_first_sse_frame(address, path).await;
            assert_eq!(handle.b_active.load(Ordering::Relaxed), 1);
            assert!(handle.kv_cache_active_bytes.load(Ordering::Relaxed) > 0);

            wait_for_scheduler_resources_to_be_released(&handle).await;
            assert_eq!(
                terminal_events.load(Ordering::Relaxed),
                terminal_before,
                "{path} emitted a terminal SSE event after disconnect"
            );
        }

        assert_eq!(handle.admit_count.load(Ordering::Relaxed), 3);
        server.abort();
        let _ = server.await;
    }

    #[test]
    fn abandoned_queued_admission_does_not_consume_queue_capacity() {
        let (abandoned, abandoned_rx) = queued_pending(11);
        drop(abandoned_rx);
        let (live, _live_rx) = queued_pending(12);
        let mut queue = VecDeque::from([abandoned, live]);

        assert_eq!(prune_abandoned_pending_admits(&mut queue), 1);
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.front().unwrap().request.prompt_ids, vec![12]);
    }

    #[test]
    fn scheduler_has_decodable_rows_requires_generated_token() {
        let mut scheduler = test_scheduler(1, 32);
        scheduler.admit(mk_req(11)).expect("admit");

        assert!(!scheduler_has_decodable_rows(&scheduler));

        let events = scheduler
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("prefill");
        assert_eq!(events.len(), 1);
        assert!(scheduler_has_decodable_rows(&scheduler));
    }

    #[test]
    fn rolling_profile_env_parser_only_enables_explicit_one() {
        assert!(rolling_profile_enabled_from_env(Some("1")));
        assert!(!rolling_profile_enabled_from_env(None));
        assert!(!rolling_profile_enabled_from_env(Some("")));
        assert!(!rolling_profile_enabled_from_env(Some("true")));
        assert!(!rolling_profile_enabled_from_env(Some("0")));
    }

    #[test]
    fn rolling_profile_queue_wait_ms_uses_supplied_clock() {
        let queued_at = std::time::Instant::now();
        let now = queued_at + Duration::from_micros(12_345);

        let wait_ms = rolling_profile_queue_wait_ms(queued_at, now);

        assert!((wait_ms - 12.345).abs() < 1e-9);
    }

    #[test]
    fn cadence_protection_uses_supplied_runtime_cap() {
        assert_eq!(cadence_protected_mid_chunk_size(1024, 2, 384), 384);
        assert_eq!(cadence_protected_mid_chunk_size(1024, 1, 384), 1024);
        assert_eq!(cadence_protected_mid_chunk_size(128, 2, 384), 128);
    }

    #[test]
    fn adaptive_gemma4_drafter_fresh_limit_starts_long_chunked_request_alone() {
        let mut request = mk_req(11);
        request.prompt_ids = (0..4096).collect();
        request.prefill_chunk_size = 2048;

        let limit = fresh_prefill_batch_limit_for_request::<SchedulerActorFakeModel>(
            &request,
            4,
            crate::core::server::adaptive_admission::AdaptiveAdmissionPolicy::gemma4_drafter(),
        );

        assert_eq!(limit, 1);
    }

    #[test]
    fn adaptive_gemma4_drafter_fresh_limit_keeps_short_request_latency_cap() {
        let request = mk_req(11);

        let limit = fresh_prefill_batch_limit_for_request::<SchedulerActorFakeModel>(
            &request,
            4,
            crate::core::server::adaptive_admission::AdaptiveAdmissionPolicy::gemma4_drafter(),
        );

        assert_eq!(limit, 2);
    }

    #[test]
    fn adaptive_qwen_mtp_fresh_limit_batches_greedy_long_chunked_requests() {
        let mut request = mk_req(11);
        request.prompt_ids = (0..4096).collect();
        request.prefill_chunk_size = 2048;

        let limit = fresh_prefill_batch_limit_for_request::<SchedulerActorFakeModel>(
            &request,
            4,
            crate::core::server::adaptive_admission::AdaptiveAdmissionPolicy::qwen_mtp(),
        );

        assert_eq!(limit, 2);
    }

    #[test]
    fn adaptive_qwen_mtp_fresh_limit_starts_non_pipelinable_long_chunked_request_alone() {
        let mut request = mk_req(11);
        request.prompt_ids = (0..4096).collect();
        request.prefill_chunk_size = 2048;
        request.sampler = Sampler::greedy().with_temperature(0.7);

        let limit = fresh_prefill_batch_limit_for_request::<SchedulerActorFakeModel>(
            &request,
            4,
            crate::core::server::adaptive_admission::AdaptiveAdmissionPolicy::qwen_mtp(),
        );

        assert_eq!(limit, 1);
    }

    #[test]
    fn adaptive_qwen_mtp_fresh_limit_keeps_short_request_latency_cap() {
        let request = mk_req(11);

        let limit = fresh_prefill_batch_limit_for_request::<SchedulerActorFakeModel>(
            &request,
            4,
            crate::core::server::adaptive_admission::AdaptiveAdmissionPolicy::qwen_mtp(),
        );

        assert_eq!(limit, 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qwen_mtp_fresh_window_batches_compatible_greedy_long_requests() {
        let mut scheduler = test_scheduler(4, 32768);
        let mut first = mk_req(11);
        first.prompt_ids = (0..4096).collect();
        first.prefill_chunk_size = 2048;
        scheduler.admit(first.clone()).expect("admit first");

        let (tx, mut rx) = mpsc::channel(4);
        let (reply_tx, mut reply_rx) = oneshot::channel();
        let mut second = mk_req(12);
        second.prompt_ids = (0..4096).collect();
        second.prefill_chunk_size = 2048;
        tx.send(SchedulerCommand::Admit {
            request: second,
            reply_tx,
        })
        .await
        .expect("send second");
        drop(tx);

        let mut event_txs = HashMap::new();
        let mut queue = VecDeque::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let saturate_triggered = Arc::new(AtomicU64::new(0));
        let queue_depth_peak = Arc::new(AtomicUsize::new(0));
        let queue_rejected = Arc::new(AtomicU64::new(0));
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let limit =
            fresh_prefill_batch_limit_for_request::<SchedulerActorFakeModel>(&first, 4, policy);

        drain_window(
            &mut rx,
            &mut scheduler,
            &mut event_txs,
            &mut queue,
            &admit_count,
            &saturate_triggered,
            &queue_depth_peak,
            &queue_rejected,
            limit,
            4,
            8,
            Duration::from_millis(1),
            admission_request_shape(&first),
            policy,
        )
        .await;

        assert_eq!(scheduler.active_count(), 2);
        assert_eq!(queue.len(), 0);
        assert!(
            matches!(reply_rx.try_recv(), Ok(Ok(_))),
            "compatible long Qwen MTP request should join the fresh batch"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qwen_mtp_fresh_window_does_not_mix_short_batch_with_long_prompt() {
        let mut scheduler = test_scheduler(4, 32768);
        let first = mk_req(11);
        scheduler.admit(first.clone()).expect("admit first");

        let (tx, mut rx) = mpsc::channel(4);
        let (reply_tx, mut reply_rx) = oneshot::channel();
        let mut second = mk_req(12);
        second.prompt_ids = (0..4096).collect();
        second.prefill_chunk_size = 2048;
        tx.send(SchedulerCommand::Admit {
            request: second,
            reply_tx,
        })
        .await
        .expect("send second");
        drop(tx);

        let mut event_txs = HashMap::new();
        let mut queue = VecDeque::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let saturate_triggered = Arc::new(AtomicU64::new(0));
        let queue_depth_peak = Arc::new(AtomicUsize::new(0));
        let queue_rejected = Arc::new(AtomicU64::new(0));
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let limit =
            fresh_prefill_batch_limit_for_request::<SchedulerActorFakeModel>(&first, 4, policy);

        drain_window(
            &mut rx,
            &mut scheduler,
            &mut event_txs,
            &mut queue,
            &admit_count,
            &saturate_triggered,
            &queue_depth_peak,
            &queue_rejected,
            limit,
            4,
            8,
            Duration::from_millis(1),
            admission_request_shape(&first),
            policy,
        )
        .await;

        assert_eq!(scheduler.active_count(), 1);
        assert_eq!(queue.len(), 1);
        assert!(reply_rx.try_recv().is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn qwen_mtp_fresh_window_does_not_mix_non_pipelinable_long_prompt() {
        let mut scheduler = test_scheduler(4, 32768);
        let mut first = mk_req(11);
        first.prompt_ids = (0..4096).collect();
        first.prefill_chunk_size = 2048;
        scheduler.admit(first.clone()).expect("admit first");

        let (tx, mut rx) = mpsc::channel(4);
        let (reply_tx, mut reply_rx) = oneshot::channel();
        let mut second = mk_req(12);
        second.prompt_ids = (0..4096).collect();
        second.prefill_chunk_size = 2048;
        second.sampler = Sampler::greedy().with_temperature(0.7);
        tx.send(SchedulerCommand::Admit {
            request: second,
            reply_tx,
        })
        .await
        .expect("send second");
        drop(tx);

        let mut event_txs = HashMap::new();
        let mut queue = VecDeque::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let saturate_triggered = Arc::new(AtomicU64::new(0));
        let queue_depth_peak = Arc::new(AtomicUsize::new(0));
        let queue_rejected = Arc::new(AtomicU64::new(0));
        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        let limit =
            fresh_prefill_batch_limit_for_request::<SchedulerActorFakeModel>(&first, 4, policy);

        drain_window(
            &mut rx,
            &mut scheduler,
            &mut event_txs,
            &mut queue,
            &admit_count,
            &saturate_triggered,
            &queue_depth_peak,
            &queue_rejected,
            limit,
            4,
            8,
            Duration::from_millis(1),
            admission_request_shape(&first),
            policy,
        )
        .await;

        assert_eq!(scheduler.active_count(), 1);
        assert_eq!(queue.len(), 1);
        assert!(reply_rx.try_recv().is_err());
    }

    #[test]
    fn qwen_mtp_long_mid_admit_rejects_decode_hot_path() {
        let mut scheduler = test_scheduler(4, 32768);
        let mut active = mk_req(11);
        active.max_new_tokens = 64;
        let active_id = scheduler.admit(active).expect("admit active");
        let first = scheduler
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("prefill active");
        assert_eq!(first.len(), 1);

        let mut pending = mk_req(12);
        pending.prompt_ids = (0..19_785).collect();
        pending.prefill_chunk_size = 2048;
        pending.decode_cadence_mid_chunk_cap = 256;

        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        assert!(
            !can_start_rolling_mid_admit_for_request::<SchedulerActorFakeModel>(
                &pending,
                &scheduler,
                scheduler.active_count(),
                4,
                policy,
            ),
            "64-token active decode budget cannot amortize a 20K rolling prefill"
        );

        scheduler
            .get_mut(active_id)
            .expect("active row")
            .max_new_tokens = 512;
        assert!(
            !can_start_rolling_mid_admit_for_request::<SchedulerActorFakeModel>(
                &pending,
                &scheduler,
                scheduler.active_count(),
                4,
                policy,
            ),
            "20K rolling prefill should not enter the decode hot path even with a 512-token decode budget"
        );
    }

    #[test]
    fn qwen_mtp_long_mid_admit_stays_blocked_after_active_row_removal() {
        let mut scheduler = test_scheduler(4, 32768);
        let mut high_budget_active = mk_req(11);
        high_budget_active.max_new_tokens = 512;
        let high_budget_id = scheduler
            .admit(high_budget_active)
            .expect("admit high-budget active");
        let mut low_budget_active = mk_req(12);
        low_budget_active.max_new_tokens = 64;
        scheduler
            .admit(low_budget_active)
            .expect("admit low-budget active");
        let first = scheduler
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("prefill active rows");
        assert_eq!(first.len(), 2);

        let mut pending = mk_req(13);
        pending.prompt_ids = (0..19_785).collect();
        pending.prefill_chunk_size = 2048;
        pending.decode_cadence_mid_chunk_cap = 256;

        let policy = AdaptiveAdmissionPolicy::qwen_mtp();
        assert!(
            !can_start_rolling_mid_admit_for_request::<SchedulerActorFakeModel>(
                &pending,
                &scheduler,
                scheduler.active_count(),
                4,
                policy,
            ),
            "long Qwen MTP prefill should stay out of active decode even while a high-budget row is active"
        );

        scheduler
            .evict(high_budget_id)
            .expect("remove high-budget active row");
        assert!(
            !can_start_rolling_mid_admit_for_request::<SchedulerActorFakeModel>(
                &pending,
                &scheduler,
                scheduler.active_count(),
                4,
                policy,
            ),
            "after removing the high-budget row, remaining decode budget is insufficient"
        );
    }

    #[test]
    fn drain_admission_queue_limits_successful_mid_admit_to_one_per_turn() {
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let mut sched = test_scheduler(4, 32768);
        sched.admit(mk_req(11)).expect("initial admit");
        let prefill_events = sched
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("initial prefill");
        assert_eq!(prefill_events.len(), 1);
        assert_eq!(sched.phase(), Phase::Decoding);

        let (pending_1, reply_rx_1) = queued_pending(21);
        let (pending_2, reply_rx_2) = queued_pending(22);
        let (pending_3, reply_rx_3) = queued_pending(23);
        let _reply_rxs = [reply_rx_1, reply_rx_2, reply_rx_3];
        let mut queue = VecDeque::from([pending_1, pending_2, pending_3]);
        let mut event_txs = HashMap::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let mut mtp_mode = SchedulerActorNoMtp;
        let mtp_counters = test_mtp_counters();
        let mut in_flight_mid_admit = None;

        let did_admit = drain_admission_queue(
            &mut queue,
            &mut in_flight_mid_admit,
            &mut sched,
            &mut event_txs,
            &admit_count,
            &model,
            &mut mtp_mode,
            &mtp_counters,
            4,
            256,
            AdaptiveAdmissionPolicy::disabled(),
        );

        assert!(did_admit > 0, "expected one queued request to be admitted");
        assert_eq!(
            queue.len(),
            2,
            "queue drain should leave remaining queued requests for later decode turns"
        );
        assert_eq!(
            sched.active_count(),
            2,
            "one active row plus exactly one mid-admitted row"
        );
        assert!(in_flight_mid_admit.is_none());
    }

    #[test]
    fn drain_admission_queue_respects_rolling_prefill_batch_limit() {
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let mut sched = test_scheduler(4, 32768);
        sched.admit(mk_req(11)).expect("initial admit 1");
        sched.admit(mk_req(12)).expect("initial admit 2");
        let prefill_events = sched
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("initial prefill");
        assert_eq!(prefill_events.len(), 2);
        assert_eq!(sched.phase(), Phase::Decoding);
        assert_eq!(sched.active_count(), 2);

        let (pending_1, reply_rx_1) = queued_pending(21);
        let (pending_2, reply_rx_2) = queued_pending(22);
        let _reply_rxs = [reply_rx_1, reply_rx_2];
        let mut queue = VecDeque::from([pending_1, pending_2]);
        let mut event_txs = HashMap::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let mut mtp_mode = SchedulerActorNoMtp;
        let mtp_counters = test_mtp_counters();
        let mut in_flight_mid_admit = None;

        let did_admit = drain_admission_queue(
            &mut queue,
            &mut in_flight_mid_admit,
            &mut sched,
            &mut event_txs,
            &admit_count,
            &model,
            &mut mtp_mode,
            &mtp_counters,
            4,
            256,
            AdaptiveAdmissionPolicy::disabled(),
        );

        assert!(
            did_admit == 0,
            "active rows already reached the model's rolling prefill batch limit"
        );
        assert_eq!(
            queue.len(),
            2,
            "queued requests should wait for decode progress instead of growing active batch"
        );
        assert_eq!(sched.active_count(), 2);
        assert_eq!(admit_count.load(Ordering::Relaxed), 0);
        assert!(in_flight_mid_admit.is_none());
    }

    #[test]
    fn drain_admission_queue_starts_chunked_mid_admit_beyond_rolling_limit() {
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let mut sched = test_scheduler(4, 32768);
        sched.admit(mk_req(11)).expect("initial admit 1");
        sched.admit(mk_req(12)).expect("initial admit 2");
        let prefill_events = sched
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("initial prefill");
        assert_eq!(prefill_events.len(), 2);
        assert_eq!(sched.phase(), Phase::Decoding);
        assert_eq!(sched.active_count(), 2);

        let (reply_tx, reply_rx) = oneshot::channel();
        let mut chunked_req = mk_req(21);
        chunked_req.prompt_ids = vec![21, 22, 23, 24];
        chunked_req.prefill_chunk_size = 2;
        let _reply_rx = reply_rx;
        let mut queue = VecDeque::from([PendingAdmit {
            request: chunked_req,
            reply_tx,
            queued_at_profile: None,
        }]);
        let mut event_txs = HashMap::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let mut mtp_mode = SchedulerActorNoMtp;
        let mtp_counters = test_mtp_counters();
        let mut in_flight_mid_admit = None;

        let did_admit = drain_admission_queue(
            &mut queue,
            &mut in_flight_mid_admit,
            &mut sched,
            &mut event_txs,
            &admit_count,
            &model,
            &mut mtp_mode,
            &mtp_counters,
            4,
            256,
            AdaptiveAdmissionPolicy::disabled(),
        );

        assert!(
            did_admit > 0,
            "chunked queued requests may start prefill under decode-cadence protection"
        );
        assert_eq!(queue.len(), 0);
        assert_eq!(sched.active_count(), 3);
        assert_eq!(
            admit_count.load(Ordering::Relaxed),
            0,
            "the request should not count as admitted until the final chunk samples its first token"
        );
        assert!(
            in_flight_mid_admit.is_some(),
            "multi-chunk mid-admit should yield after one chunk"
        );
    }

    #[test]
    fn drain_admission_queue_caps_chunked_mid_admit_when_decode_rows_are_active() {
        let model = Arc::new(Mutex::new(SchedulerActorFakeModel));
        let mut sched = test_scheduler(4, 32768);
        sched.admit(mk_req(11)).expect("initial admit 1");
        sched.admit(mk_req(12)).expect("initial admit 2");
        let prefill_events = sched
            .prefill_admitted(&SchedulerActorFakeModel)
            .expect("initial prefill");
        assert_eq!(prefill_events.len(), 2);
        assert_eq!(sched.phase(), Phase::Decoding);
        assert_eq!(sched.active_count(), 2);

        let (reply_tx, reply_rx) = oneshot::channel();
        let mut chunked_req = mk_req(21);
        chunked_req.prompt_ids = (0..1025).collect();
        chunked_req.prefill_chunk_size = 1024;
        chunked_req.decode_cadence_mid_chunk_cap = 384;
        let _reply_rx = reply_rx;
        let mut queue = VecDeque::from([PendingAdmit {
            request: chunked_req,
            reply_tx,
            queued_at_profile: None,
        }]);
        let mut event_txs = HashMap::new();
        let admit_count = Arc::new(AtomicU64::new(0));
        let mut mtp_mode = SchedulerActorNoMtp;
        let mtp_counters = test_mtp_counters();
        let mut in_flight_mid_admit = None;

        let did_admit = drain_admission_queue(
            &mut queue,
            &mut in_flight_mid_admit,
            &mut sched,
            &mut event_txs,
            &admit_count,
            &model,
            &mut mtp_mode,
            &mtp_counters,
            4,
            384,
            AdaptiveAdmissionPolicy::disabled(),
        );

        assert!(did_admit > 0, "chunked queued request should start");
        let handle = in_flight_mid_admit
            .as_ref()
            .expect("chunked mid-admit should still be in flight");
        assert_eq!(
            handle.chunk_start, 384,
            "active decode rows should cap the first mid-admit chunk to protect ITL"
        );
        assert_eq!(
            handle.chunk_size, 1024,
            "cadence cap should be temporary and preserve the request chunk size"
        );
    }

    /// Drop the SchedulerActorHandle (and thus cmd_tx); confirm the driver
    /// task exits cleanly. We can't construct a real Qwen35Model in a unit
    /// test, so we never send any commands — we only verify the driver's
    /// `rt.block_on(cmd_rx.recv())` outer loop terminates when all
    /// senders are dropped.
    ///
    /// To keep this test self-contained without a model, we don't call
    /// `spawn_scheduler_actor` (which would require a real model handle).
    /// Instead we directly spawn a minimal stand-in driver that mirrors
    /// the channel-close exit condition.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn driver_shuts_down_when_cmd_channel_closes() {
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<SchedulerCommand>(8);
        let handle = tokio::task::spawn_blocking(move || {
            // Mirrors `driver_loop`'s exit condition without touching a model.
            while let Some(_cmd) = cmd_rx.blocking_recv() {
                // would dispatch here in real driver
            }
        });
        drop(cmd_tx);
        // Driver should exit promptly after senders drop.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .expect("driver did not shut down within 2s")
            .expect("driver join error");
    }

    /// b_max=1 + queue_max=2; admit 3 short requests in rapid succession;
    /// verify the queue grows to peak >= 1 before slots free up.
    /// Real-model heavy — gated by `#[ignore]`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore] // real-model heavy: loads Qwen3.5-4B-MLX-4bit
    async fn admission_queue_push_when_full() {
        use crate::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
        use crate::core::sampler::Sampler;
        use crate::core::{Loader, Tokenizer};
        use std::sync::atomic::Ordering;
        use std::time::Duration;
        use tokio::sync::Mutex;

        let model_dir = std::env::var("IRONMLX_MODEL_DIR").unwrap_or_else(|_| {
            let glob = format!(
                "{}/.ironmlx/models/huggingface/mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
                std::env::var("HOME").unwrap()
            );
            std::fs::read_dir(&glob)
                .expect("snapshots dir")
                .filter_map(|e| e.ok())
                .next()
                .expect("snapshot")
                .path()
                .to_string_lossy()
                .into_owned()
        });
        let loader = Loader::open_multimodal(std::path::Path::new(&model_dir)).unwrap();
        let tokenizer = Tokenizer::from_loader(&loader).unwrap();
        let model = Arc::new(Mutex::new(
            crate::models::Qwen35Model::from_loader(&loader).unwrap(),
        ));
        let meta = model.lock().await.model_meta();

        let handle = spawn_scheduler_actor(
            model.clone(),
            /* b_max */ 1,
            /* admission_deadline */ Duration::from_millis(5),
            /* admission_queue_max */ 2,
            /* effective_cap_max */ 32768,
            /* decode_cadence_mid_chunk_cap */ 256,
            meta,
        )
        .expect("spawn");

        let mk_req = |text: &str| -> GenerateRequest {
            let msgs = vec![crate::core::Message {
                role: "user".into(),
                content: text.into(),
            }];
            let kw = serde_json::json!({"enable_thinking": false});
            let rendered = tokenizer
                .apply_chat_template(&msgs, true, Some(&kw))
                .unwrap();
            let prompt_ids = tokenizer.encode(&rendered, false).unwrap();
            GenerateRequest {
                prompt_ids,
                max_new_tokens: 8,
                sampler: Sampler::greedy(),
                stop_token_ids: tokenizer.eos_token_ids().to_vec(),
                prefill_chunk_size: 0,
                decode_cadence_mid_chunk_cap: 256,
                kv_cache_turboquant_bits: None,
                pixel_values: None,
                image_grid_thw: None,
                image_spatial_merge_size: 2,
                image_token_id: IMAGE_TOKEN_ID,
                constraint: None,
            }
        };

        let mut replies = Vec::new();
        for text in ["Hello", "World", "Goodbye"] {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            handle
                .cmd_tx
                .send(SchedulerCommand::Admit {
                    request: mk_req(text),
                    reply_tx,
                })
                .await
                .expect("cmd_tx.send");
            replies.push(reply_rx);
        }

        let mut counts = Vec::new();
        for rx in replies {
            let admit_reply = rx.await.expect("reply").expect("admit ok");
            let mut event_rx = admit_reply.event_rx;
            let mut n = 0;
            while let Some(ev) = event_rx.recv().await {
                n += 1;
                if ev.finish_reason.is_some() {
                    break;
                }
            }
            counts.push(n);
        }

        for c in &counts {
            assert!(*c >= 1, "expected ≥1 event per request, got {c}");
        }

        let peak = handle.queue_depth_peak.load(Ordering::Relaxed);
        assert!(peak >= 1, "expected queue_depth_peak >= 1, got {peak}");

        let rejected = handle.queue_rejected.load(Ordering::Relaxed);
        assert_eq!(rejected, 0, "expected no rejections, got {rejected}");

        drop(handle);
    }

    /// b_max=1 + queue_max=1; send 3 admits back-to-back. The 3rd one
    /// must be rejected with Err("admission queue full").
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore] // real-model heavy
    async fn admission_queue_overflow_returns_err() {
        use crate::core::generate::{GenerateRequest, IMAGE_TOKEN_ID};
        use crate::core::sampler::Sampler;
        use crate::core::{Loader, Tokenizer};
        use std::sync::atomic::Ordering;
        use std::time::Duration;
        use tokio::sync::Mutex;

        let model_dir = std::env::var("IRONMLX_MODEL_DIR").unwrap_or_else(|_| {
            let glob = format!(
                "{}/.ironmlx/models/huggingface/mlx-community--Qwen3.5-4B-MLX-4bit/snapshots",
                std::env::var("HOME").unwrap()
            );
            std::fs::read_dir(&glob)
                .expect("snapshots dir")
                .filter_map(|e| e.ok())
                .next()
                .expect("snapshot")
                .path()
                .to_string_lossy()
                .into_owned()
        });
        let loader = Loader::open_multimodal(std::path::Path::new(&model_dir)).unwrap();
        let tokenizer = Tokenizer::from_loader(&loader).unwrap();
        let model = Arc::new(Mutex::new(
            crate::models::Qwen35Model::from_loader(&loader).unwrap(),
        ));
        let meta = model.lock().await.model_meta();

        let handle = spawn_scheduler_actor(
            model.clone(),
            /* b_max */ 1,
            /* admission_deadline */ Duration::from_millis(5),
            /* admission_queue_max */ 1,
            /* effective_cap_max */ 32768,
            /* decode_cadence_mid_chunk_cap */ 256,
            meta,
        )
        .expect("spawn");

        let mk_req = |text: &str, max_new: usize| -> GenerateRequest {
            let msgs = vec![crate::core::Message {
                role: "user".into(),
                content: text.into(),
            }];
            let kw = serde_json::json!({"enable_thinking": false});
            let rendered = tokenizer
                .apply_chat_template(&msgs, true, Some(&kw))
                .unwrap();
            let prompt_ids = tokenizer.encode(&rendered, false).unwrap();
            GenerateRequest {
                prompt_ids,
                max_new_tokens: max_new,
                sampler: Sampler::greedy(),
                stop_token_ids: tokenizer.eos_token_ids().to_vec(),
                prefill_chunk_size: 0,
                decode_cadence_mid_chunk_cap: 256,
                kv_cache_turboquant_bits: None,
                pixel_values: None,
                image_grid_thw: None,
                image_spatial_merge_size: 2,
                image_token_id: IMAGE_TOKEN_ID,
                constraint: None,
            }
        };

        let (tx1, rx1) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: mk_req("Hello", 64),
                reply_tx: tx1,
            })
            .await
            .unwrap();

        // Wait briefly so first admit enters Decoding before #2/#3 arrive.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let (tx2, rx2) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: mk_req("World", 8),
                reply_tx: tx2,
            })
            .await
            .unwrap();

        let (tx3, rx3) = tokio::sync::oneshot::channel();
        handle
            .cmd_tx
            .send(SchedulerCommand::Admit {
                request: mk_req("Goodbye", 8),
                reply_tx: tx3,
            })
            .await
            .unwrap();

        let reply3 = tokio::time::timeout(Duration::from_secs(5), rx3)
            .await
            .expect("rx3 timeout")
            .expect("rx3 recv");
        match reply3 {
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("admission queue full"),
                    "expected 'admission queue full' Err, got: {msg}"
                );
            }
            Ok(_) => panic!("expected Err for #3, got Ok"),
        }

        let rejected = handle.queue_rejected.load(Ordering::Relaxed);
        assert!(rejected >= 1, "expected queue_rejected ≥ 1, got {rejected}");

        let _ = tokio::time::timeout(Duration::from_secs(120), async {
            let r1 = rx1.await.unwrap().unwrap();
            let mut e1 = r1.event_rx;
            while let Some(ev) = e1.recv().await {
                if ev.finish_reason.is_some() {
                    break;
                }
            }
            let r2 = rx2.await.unwrap().unwrap();
            let mut e2 = r2.event_rx;
            while let Some(ev) = e2.recv().await {
                if ev.finish_reason.is_some() {
                    break;
                }
            }
        })
        .await
        .expect("rx1/rx2 drain timeout");

        drop(handle);
    }
}
