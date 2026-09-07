//! HTTP server — single-stream OpenAI + Anthropic compatibility.
//!
//! `serve()` owns the model behind a Mutex; concurrent requests serialize
//! waiting for the lock (P4 contract — multi-stream scheduler is P8b).

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::{extract::State, routing::get, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::{oneshot, Mutex};

use crate::core::cache::{
    ActiveKvOffloadConfig, PagedPrefixCacheConfig, PrefixLruCacheConfig, TurboQuantKVBits,
};
use crate::core::model::Model;
use crate::core::sampler::Sampler;
use crate::core::scheduler::DenseVlMethods;
use crate::core::scheduler_autotune::{
    SchedulerAutotuneProfileConfig, SchedulerAutotuneRuntimeProfile,
    SchedulerAutotuneRuntimeRequest,
};
use crate::core::speculative::MtpSpeculativeModel;
use crate::core::tokenizer::Tokenizer;
use crate::Result;

pub(crate) mod adaptive_admission;
pub mod anthropic;
pub(crate) mod api_error;
pub(crate) mod api_transport;
pub mod chat_format;
pub(crate) mod dflash2_actor;
pub mod diffusion_gemma;
pub mod engine;
pub mod health;
pub mod model_manager;
pub(crate) mod openai;
pub(crate) mod responses;
pub mod scheduler_actor;
pub mod security;
pub(crate) mod structured_output;
pub mod vision;

#[derive(Clone)]
pub enum RequestExecutionHandle {
    Scheduler(Arc<scheduler_actor::SchedulerActorHandle>),
    DFlash2(Arc<dflash2_actor::DFlash2ActorHandle>),
}

pub(crate) enum RequestAdmissionError {
    Rejected(anyhow::Error),
    Unavailable,
    ReplyLost,
}

impl RequestExecutionHandle {
    pub(crate) fn is_dflash2(&self) -> bool {
        matches!(self, Self::DFlash2(_))
    }

    pub(crate) fn active_and_queued(&self) -> (usize, usize) {
        let (active, queued) = match self {
            Self::Scheduler(handle) => (&handle.b_active, &handle.b_queued),
            Self::DFlash2(handle) => (&handle.b_active, &handle.b_queued),
        };
        (
            active.load(Ordering::Relaxed) as usize,
            queued.load(Ordering::Relaxed) as usize,
        )
    }

    pub(crate) async fn admit(
        &self,
        request: crate::core::generate::GenerateRequest,
    ) -> std::result::Result<scheduler_actor::AdmitReply, RequestAdmissionError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        match self {
            Self::Scheduler(handle) => handle
                .cmd_tx
                .send(scheduler_actor::SchedulerCommand::Admit { request, reply_tx })
                .await
                .map_err(|_| RequestAdmissionError::Unavailable)?,
            Self::DFlash2(handle) => match handle.enqueue(request, reply_tx) {
                Ok(()) => {}
                Err(dflash2_actor::DFlash2EnqueueError::QueueFull(error)) => {
                    return Err(RequestAdmissionError::Rejected(error));
                }
                Err(dflash2_actor::DFlash2EnqueueError::Unavailable) => {
                    return Err(RequestAdmissionError::Unavailable);
                }
            },
        }
        reply_rx
            .await
            .map_err(|_| RequestAdmissionError::ReplyLost)?
            .map_err(RequestAdmissionError::Rejected)
    }

    pub(crate) async fn clear_shared_prompt_lookup(&self) -> Result<usize> {
        match self {
            Self::Scheduler(handle) => handle.clear_shared_prompt_lookup().await,
            Self::DFlash2(_) => Ok(0),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub repetition_penalty: Option<f32>,
}

impl SamplingDefaults {
    pub fn merge_with_override(self, override_defaults: Self) -> Self {
        Self {
            temperature: override_defaults.temperature.or(self.temperature),
            top_p: override_defaults.top_p.or(self.top_p),
            top_k: override_defaults.top_k.or(self.top_k),
            repetition_penalty: override_defaults
                .repetition_penalty
                .or(self.repetition_penalty),
        }
    }
}

#[derive(Clone)]
pub enum VisionInputConfig {
    Qwen {
        spatial_merge_size: i32,
    },
    Gemma4 {
        vision_config: crate::models::gemma4::Gemma4VisionConfig,
    },
    DiffusionGemma {
        vision_config: crate::models::gemma4::Gemma4VisionConfig,
        image_token_id: Option<i32>,
    },
    MiniCpmV46 {
        /// Effective image-token downsample = 4 (VitMerger 2×2 × Merger 2×2).
        spatial_merge_size: i32,
    },
}

/// HTTP server shared state. The model is wrapped in a tokio Mutex —
/// concurrent requests serialize behind the lock (P4 single-stream contract).
///
/// `request_execution` selects either the mature SchedulerActor or the isolated
/// DFlash2 actor. Ordinary engines may still route long text requests directly
/// through GenerationStream; DFlash2 engines route every request to their actor
/// so unsupported capabilities cannot silently fall back to ordinary decoding.
///
/// P5a-T5: AppState is now generic over `M: Model + DenseVlMethods + Send +
/// 'static`. CLI call sites pass either `Qwen35Model` or `Qwen35MoeModel`
/// based on the checkpoint `model_type`.
///
/// `Clone` is implemented manually so the derive macro doesn't emit an
/// unwanted `M: Clone` bound — all fields clone without needing `M: Clone`
/// because `Arc<Mutex<M>>` and `Arc<...>` are `Clone` unconditionally.
pub struct AppState<M: Model + DenseVlMethods + Send + 'static> {
    pub model: Arc<Mutex<M>>,
    pub tokenizer: Arc<Tokenizer>,
    pub model_id: String,
    /// Default prefill chunk size (max tokens per prefill forward). `0`
    /// disables chunking. Applied to every `GenerateRequest` constructed
    /// by the request handlers.
    pub prefill_chunk_size: usize,
    pub vision_input: VisionInputConfig,
    /// Request execution backend selected when this engine is built.
    pub request_execution: RequestExecutionHandle,
    /// True when the SchedulerActor was started with paged SSD prefix cache.
    pub paged_prefix_cache_enabled: bool,
    /// Maximum concurrent in-flight requests routed to the SchedulerActor.
    pub b_max: usize,
    /// Admission-window deadline (milliseconds) — drain-window timeout.
    pub admission_deadline_ms: u64,
    /// FIFO admission queue capacity.
    pub admission_queue_max: usize,
    /// Effective cap_max = min(--max-cache-cap CLI flag, model.config.max_position_embeddings).
    /// Per-request `prompt_len + max_new_tokens` exceeding this returns HTTP 413. B1-p2.3f.
    pub effective_cap_max: usize,
    /// Runtime scheduler profile. Base config is applied at boot; rules may
    /// select request-level chunk/cadence settings after tokenization.
    pub scheduler_runtime_profile: Arc<SchedulerAutotuneRuntimeProfile>,
    /// Model-level default sampling configuration. Request-level sampling
    /// fields still take precedence.
    pub sampling_defaults: SamplingDefaults,
    /// Loaded model weight bytes captured once at load time so EnginePool
    /// guardrails do not need to lock the model later.
    pub model_weight_bytes: usize,
    /// Metadata-only mmap liability split by first-use component.
    pub static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
    /// Engine-lifetime first-use tracker shared by Scheduler and direct paths.
    pub cold_materialization_tracker: Arc<crate::core::process_memory::ColdMaterializationTracker>,
    /// Optional TurboQuant K/V bit-widths for full-attention KV cache reads.
    pub kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    /// Route eligible greedy requests through SchedulerActor.
    pub force_scheduler_for_greedy: bool,
    /// True when PromptLookup is enabled for this model engine.
    pub prompt_lookup_enabled: bool,
    /// Health snapshot collector for `/healthz`. Holds shared Arc atomics
    /// wired to the SchedulerActor driver loop + BudgetState. B1-p2.5 G3.
    pub health_collector: Arc<health::SchedulerHealthCollector>,
    /// Per-loaded-engine token and prefix-cache usage. A new instance is
    /// created whenever the model is loaded, so unload/reload resets it.
    pub runtime_usage: Arc<crate::core::runtime_usage::ModelRuntimeUsageCounters>,
}

impl<M: Model + DenseVlMethods + Send + 'static> Clone for AppState<M> {
    fn clone(&self) -> Self {
        AppState {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            model_id: self.model_id.clone(),
            prefill_chunk_size: self.prefill_chunk_size,
            vision_input: self.vision_input.clone(),
            request_execution: self.request_execution.clone(),
            paged_prefix_cache_enabled: self.paged_prefix_cache_enabled,
            b_max: self.b_max,
            admission_deadline_ms: self.admission_deadline_ms,
            admission_queue_max: self.admission_queue_max,
            effective_cap_max: self.effective_cap_max,
            scheduler_runtime_profile: self.scheduler_runtime_profile.clone(),
            sampling_defaults: self.sampling_defaults,
            model_weight_bytes: self.model_weight_bytes,
            static_memory_estimate: self.static_memory_estimate,
            cold_materialization_tracker: self.cold_materialization_tracker.clone(),
            kv_cache_turboquant_bits: self.kv_cache_turboquant_bits,
            force_scheduler_for_greedy: self.force_scheduler_for_greedy,
            prompt_lookup_enabled: self.prompt_lookup_enabled,
            health_collector: self.health_collector.clone(),
            runtime_usage: self.runtime_usage.clone(),
        }
    }
}

pub(crate) fn validate_prompt_lookup_sampler(_enabled: bool, sampler: Sampler) -> Result<()> {
    anyhow::ensure!(
        sampler.temperature.is_finite(),
        "sampling temperature must be finite"
    );
    Ok(())
}

impl<M: Model + DenseVlMethods + Send + 'static> AppState<M> {
    pub(crate) fn with_sampling_defaults(mut self, sampling_defaults: SamplingDefaults) -> Self {
        self.sampling_defaults = sampling_defaults;
        self
    }

    pub(crate) fn scheduler_request_config(
        &self,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> SchedulerAutotuneProfileConfig {
        let (active, queued) = self.request_execution.active_and_queued();
        self.scheduler_runtime_profile
            .select_config(SchedulerAutotuneRuntimeRequest {
                prompt_len,
                max_new_tokens,
                effective_concurrency: active.saturating_add(queued).saturating_add(1),
            })
    }

    pub(crate) fn record_request_started(
        &self,
        input_tokens: u32,
        started_at: std::time::Instant,
    ) -> crate::core::runtime_usage::ModelRuntimeRequestTracker {
        let tracker = self
            .runtime_usage
            .start_request(u64::from(input_tokens), started_at);
        if self.paged_prefix_cache_enabled && !self.request_execution.is_dflash2() {
            self.runtime_usage
                .record_prefix_cache_eligible_tokens(u64::from(input_tokens.saturating_sub(1)));
        }
        tracker
    }
}

pub(crate) struct DirectRequestMemoryGuard {
    cold: Option<crate::core::process_memory::ColdMaterializationGuard>,
    vision: Option<crate::core::process_memory::MemoryReservation>,
}

impl DirectRequestMemoryGuard {
    pub(crate) fn commit(mut self) {
        drop(self.vision.take());
        if let Some(cold) = self.cold.take() {
            cold.commit();
        }
    }
}

pub(crate) fn begin_direct_request_memory<M>(
    state: &AppState<M>,
    model: &M,
    request: &crate::core::generate::GenerateRequest,
) -> Result<DirectRequestMemoryGuard>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let governor = crate::core::process_memory::global_process_memory_governor();
    let components = crate::core::process_memory::MaterializationComponents::for_request(
        request.pixel_values.is_some(),
        false,
    );
    let cold = state
        .cold_materialization_tracker
        .begin(components, &governor)
        .map(Some)
        .map_err(|_| {
            let snapshot = governor.snapshot();
            anyhow::Error::new(
                crate::core::scheduler::SchedulerError::ColdMaterializationUnsafe {
                    requested_bytes: components.requested_bytes(state.static_memory_estimate),
                    current_bytes: snapshot.current_usage_bytes,
                    target_bytes: snapshot.hard_watermark_bytes,
                },
            )
        })?;
    let vision = crate::core::scheduler::reserve_vision_prefill_for_request(
        Some(&governor),
        model,
        request.pixel_values.as_deref(),
        request.image_grid_thw.as_deref(),
    )?;
    Ok(DirectRequestMemoryGuard { cold, vision })
}

#[derive(Clone)]
pub(crate) struct Gemma4DrafterAppState {
    pub(crate) base: AppState<crate::models::Gemma4Model>,
    pub(crate) mtp_draft_tokens: usize,
}

#[derive(Clone, Copy)]
struct MtpHealthDraftTokens {
    requested: usize,
    effective: usize,
}

impl Gemma4DrafterAppState {
    pub(crate) fn with_sampling_defaults(mut self, sampling_defaults: SamplingDefaults) -> Self {
        self.base = self.base.with_sampling_defaults(sampling_defaults);
        self
    }
}

pub(crate) fn should_route_to_scheduler<M: Model>(
    prompt_len: usize,
    prefill_chunk_size: usize,
    b_max: usize,
    paged_prefix_cache_enabled: bool,
    force_scheduler: bool,
) -> bool {
    if force_scheduler {
        return true;
    }
    if paged_prefix_cache_enabled {
        return true;
    }
    if prefill_chunk_size == 0 || prompt_len <= prefill_chunk_size {
        return true;
    }
    M::fresh_prefill_batch_limit(prompt_len, b_max) < b_max
}

trait SchedulerActorSpawner<M>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    fn paged_prefix_cache_enabled(&self) -> bool;

    fn force_scheduler_for_greedy(&self) -> bool;

    fn prompt_lookup_config(&self) -> Option<crate::core::prompt_lookup::PromptLookupConfig> {
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn(
        self,
        model: Arc<Mutex<M>>,
        b_max: usize,
        admission_deadline: std::time::Duration,
        admission_queue_max: usize,
        effective_cap_max: usize,
        decode_cadence_mid_chunk_cap: usize,
        meta: crate::core::memory_budget::ModelMeta,
    ) -> Result<scheduler_actor::SchedulerActorHandle>;
}

struct PlainSchedulerActorSpawner {
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
    force_scheduler: bool,
}

impl<M> SchedulerActorSpawner<M> for PlainSchedulerActorSpawner
where
    M: Model + DenseVlMethods + Send + 'static,
{
    fn paged_prefix_cache_enabled(&self) -> bool {
        self.paged_prefix_cache.is_some()
    }

    fn force_scheduler_for_greedy(&self) -> bool {
        self.force_scheduler
    }

    fn spawn(
        self,
        model: Arc<Mutex<M>>,
        b_max: usize,
        admission_deadline: std::time::Duration,
        admission_queue_max: usize,
        effective_cap_max: usize,
        decode_cadence_mid_chunk_cap: usize,
        meta: crate::core::memory_budget::ModelMeta,
    ) -> Result<scheduler_actor::SchedulerActorHandle> {
        if self.force_scheduler {
            return Ok(
                scheduler_actor::spawn_scheduler_actor_for_prompt_lookup_control(
                    model,
                    b_max,
                    admission_deadline,
                    admission_queue_max,
                    effective_cap_max,
                    decode_cadence_mid_chunk_cap,
                    meta,
                    self.paged_prefix_cache,
                    self.prefix_lru_cache,
                    self.active_kv_offload,
                )?,
            );
        }
        if let Some(config) = self.paged_prefix_cache {
            if self.active_kv_offload.enabled {
                return Ok(
                    scheduler_actor::spawn_scheduler_actor_with_paged_prefix_cache_and_active_kv(
                        model,
                        b_max,
                        admission_deadline,
                        admission_queue_max,
                        effective_cap_max,
                        decode_cadence_mid_chunk_cap,
                        meta,
                        config,
                        self.prefix_lru_cache,
                        self.active_kv_offload,
                    )?,
                );
            }
            Ok(
                scheduler_actor::spawn_scheduler_actor_with_paged_prefix_cache(
                    model,
                    b_max,
                    admission_deadline,
                    admission_queue_max,
                    effective_cap_max,
                    decode_cadence_mid_chunk_cap,
                    meta,
                    config,
                    self.prefix_lru_cache,
                )?,
            )
        } else if self.active_kv_offload.enabled {
            Ok(
                scheduler_actor::spawn_scheduler_actor_with_active_kv_offload(
                    model,
                    b_max,
                    admission_deadline,
                    admission_queue_max,
                    effective_cap_max,
                    decode_cadence_mid_chunk_cap,
                    meta,
                    self.active_kv_offload,
                )?,
            )
        } else {
            Ok(scheduler_actor::spawn_scheduler_actor(
                model,
                b_max,
                admission_deadline,
                admission_queue_max,
                effective_cap_max,
                decode_cadence_mid_chunk_cap,
                meta,
            )?)
        }
    }
}

struct MtpSchedulerActorSpawner<H> {
    mtp: H,
    mtp_draft_tokens: usize,
    exact_qualification:
        crate::core::speculative_qualification::NeuralExactQualificationRuntimeConfig,
    prompt_lookup: Option<(
        crate::core::prompt_lookup::PromptLookupConfig,
        crate::core::prompt_lookup::PromptLookupQualificationRuntimeConfig,
    )>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
}

struct PromptLookupSchedulerActorSpawner {
    cfg: crate::core::prompt_lookup::PromptLookupConfig,
    qualification: crate::core::prompt_lookup::PromptLookupQualificationRuntimeConfig,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
}

struct Gemma4DrafterSchedulerActorSpawner {
    drafter: Arc<Mutex<crate::models::gemma4::Gemma4AssistantModel>>,
    mtp_draft_tokens: usize,
    exact_qualification:
        crate::core::speculative_qualification::NeuralExactQualificationRuntimeConfig,
    prompt_lookup: Option<(
        crate::core::prompt_lookup::PromptLookupConfig,
        crate::core::prompt_lookup::PromptLookupQualificationRuntimeConfig,
    )>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
}

impl SchedulerActorSpawner<crate::models::Gemma4Model> for Gemma4DrafterSchedulerActorSpawner {
    fn paged_prefix_cache_enabled(&self) -> bool {
        self.paged_prefix_cache.is_some()
    }

    fn force_scheduler_for_greedy(&self) -> bool {
        true
    }

    fn prompt_lookup_config(&self) -> Option<crate::core::prompt_lookup::PromptLookupConfig> {
        self.prompt_lookup.as_ref().map(|(cfg, _)| *cfg)
    }

    fn spawn(
        self,
        model: Arc<Mutex<crate::models::Gemma4Model>>,
        b_max: usize,
        admission_deadline: std::time::Duration,
        admission_queue_max: usize,
        effective_cap_max: usize,
        decode_cadence_mid_chunk_cap: usize,
        meta: crate::core::memory_budget::ModelMeta,
    ) -> Result<scheduler_actor::SchedulerActorHandle> {
        if let Some((prompt_lookup, qualification)) = self.prompt_lookup {
            return scheduler_actor::spawn_scheduler_actor_with_gemma4_drafter_prompt_lookup(
                model,
                self.drafter,
                self.mtp_draft_tokens,
                prompt_lookup,
                qualification,
                b_max,
                admission_deadline,
                admission_queue_max,
                effective_cap_max,
                decode_cadence_mid_chunk_cap,
                meta,
                self.paged_prefix_cache,
                self.prefix_lru_cache,
                self.active_kv_offload,
            );
        }
        if self.active_kv_offload.enabled {
            Ok(
                scheduler_actor::spawn_scheduler_actor_with_gemma4_drafter_and_active_kv(
                    model,
                    self.drafter,
                    self.mtp_draft_tokens,
                    self.exact_qualification,
                    b_max,
                    admission_deadline,
                    admission_queue_max,
                    effective_cap_max,
                    decode_cadence_mid_chunk_cap,
                    meta,
                    self.paged_prefix_cache,
                    self.prefix_lru_cache,
                    self.active_kv_offload,
                )?,
            )
        } else {
            Ok(scheduler_actor::spawn_scheduler_actor_with_gemma4_drafter(
                model,
                self.drafter,
                self.mtp_draft_tokens,
                self.exact_qualification,
                b_max,
                admission_deadline,
                admission_queue_max,
                effective_cap_max,
                decode_cadence_mid_chunk_cap,
                meta,
                self.paged_prefix_cache,
                self.prefix_lru_cache,
            )?)
        }
    }
}

impl<M> SchedulerActorSpawner<M> for MtpSchedulerActorSpawner<M::MtpHead>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    fn paged_prefix_cache_enabled(&self) -> bool {
        self.paged_prefix_cache.is_some()
    }

    fn force_scheduler_for_greedy(&self) -> bool {
        true
    }

    fn prompt_lookup_config(&self) -> Option<crate::core::prompt_lookup::PromptLookupConfig> {
        self.prompt_lookup.as_ref().map(|(cfg, _)| *cfg)
    }

    fn spawn(
        self,
        model: Arc<Mutex<M>>,
        b_max: usize,
        admission_deadline: std::time::Duration,
        admission_queue_max: usize,
        effective_cap_max: usize,
        decode_cadence_mid_chunk_cap: usize,
        meta: crate::core::memory_budget::ModelMeta,
    ) -> Result<scheduler_actor::SchedulerActorHandle> {
        if let Some((prompt_lookup, qualification)) = self.prompt_lookup {
            return scheduler_actor::spawn_scheduler_actor_with_mtp_prompt_lookup(
                model,
                self.mtp,
                self.mtp_draft_tokens,
                prompt_lookup,
                qualification,
                b_max,
                admission_deadline,
                admission_queue_max,
                effective_cap_max,
                decode_cadence_mid_chunk_cap,
                meta,
                self.paged_prefix_cache,
                self.prefix_lru_cache,
                self.active_kv_offload,
            );
        }
        if self.active_kv_offload.enabled {
            Ok(
                scheduler_actor::spawn_scheduler_actor_with_mtp_and_active_kv(
                    model,
                    self.mtp,
                    self.mtp_draft_tokens,
                    self.exact_qualification,
                    b_max,
                    admission_deadline,
                    admission_queue_max,
                    effective_cap_max,
                    decode_cadence_mid_chunk_cap,
                    meta,
                    self.paged_prefix_cache,
                    self.prefix_lru_cache,
                    self.active_kv_offload,
                )?,
            )
        } else {
            Ok(scheduler_actor::spawn_scheduler_actor_with_mtp(
                model,
                self.mtp,
                self.mtp_draft_tokens,
                self.exact_qualification,
                b_max,
                admission_deadline,
                admission_queue_max,
                effective_cap_max,
                decode_cadence_mid_chunk_cap,
                meta,
                self.paged_prefix_cache,
                self.prefix_lru_cache,
            )?)
        }
    }
}

impl<M> SchedulerActorSpawner<M> for PromptLookupSchedulerActorSpawner
where
    M: Model + DenseVlMethods + Send + 'static,
{
    fn paged_prefix_cache_enabled(&self) -> bool {
        self.paged_prefix_cache.is_some()
    }

    fn force_scheduler_for_greedy(&self) -> bool {
        true
    }

    fn prompt_lookup_config(&self) -> Option<crate::core::prompt_lookup::PromptLookupConfig> {
        Some(self.cfg)
    }

    fn spawn(
        self,
        model: Arc<Mutex<M>>,
        b_max: usize,
        admission_deadline: std::time::Duration,
        admission_queue_max: usize,
        effective_cap_max: usize,
        decode_cadence_mid_chunk_cap: usize,
        meta: crate::core::memory_budget::ModelMeta,
    ) -> Result<scheduler_actor::SchedulerActorHandle> {
        scheduler_actor::spawn_scheduler_actor_with_prompt_lookup(
            model,
            self.cfg,
            self.qualification,
            b_max,
            admission_deadline,
            admission_queue_max,
            effective_cap_max,
            decode_cadence_mid_chunk_cap,
            meta,
            self.paged_prefix_cache,
            self.prefix_lru_cache,
            self.active_kv_offload,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn serve<M>(
    model: M,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize, // 3f
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
    force_scheduler: bool,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    serve_inner(
        model,
        tokenizer,
        model_id,
        network_config,
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        max_cache_cap,
        decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits,
        scheduler_runtime_profile,
        scheduler_autotune_report,
        vision_input_override,
        static_memory_estimate,
        None,
        PlainSchedulerActorSpawner {
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
            force_scheduler,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_with_prompt_lookup<M>(
    model: M,
    cfg: crate::core::prompt_lookup::PromptLookupConfig,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let qualification =
        crate::core::prompt_lookup::PromptLookupQualificationRuntimeConfig::for_scheduler_profile(
            &scheduler_runtime_profile,
        )?;
    serve_inner(
        model,
        tokenizer,
        model_id,
        network_config,
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        max_cache_cap,
        decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits,
        scheduler_runtime_profile,
        scheduler_autotune_report,
        vision_input_override,
        static_memory_estimate,
        None,
        PromptLookupSchedulerActorSpawner {
            cfg,
            qualification,
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_with_mtp<M>(
    model: M,
    mtp: M::MtpHead,
    mtp_draft_tokens: usize,
    prompt_lookup: Option<crate::core::prompt_lookup::PromptLookupConfig>,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
) -> Result<()>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    let prompt_lookup = prompt_lookup
        .map(|cfg| -> Result<_> {
            let qualification = crate::core::prompt_lookup::PromptLookupQualificationRuntimeConfig::for_scheduler_profile_with_baseline(
                &scheduler_runtime_profile,
                crate::core::prompt_lookup::PromptLookupQualificationBaseline::QwenMtp,
            )?;
            Ok((cfg.validate()?, qualification))
        })
        .transpose()?;
    let exact_qualification =
        crate::core::speculative_qualification::NeuralExactQualificationRuntimeConfig::for_scheduler_profile(
            &scheduler_runtime_profile,
            crate::core::speculative_qualification::NeuralExactSource::QwenMtp,
        )?;
    serve_inner(
        model,
        tokenizer,
        model_id,
        network_config,
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        max_cache_cap,
        decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits,
        scheduler_runtime_profile,
        scheduler_autotune_report,
        vision_input_override,
        static_memory_estimate,
        Some(MtpHealthDraftTokens {
            requested: mtp_draft_tokens,
            effective: mtp_draft_tokens,
        }),
        MtpSchedulerActorSpawner {
            mtp,
            mtp_draft_tokens,
            exact_qualification,
            prompt_lookup,
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_with_gemma4_drafter(
    model: crate::models::Gemma4Model,
    drafter: crate::models::gemma4::Gemma4AssistantModel,
    mtp_draft_tokens: usize,
    prompt_lookup: Option<crate::core::prompt_lookup::PromptLookupConfig>,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    active_kv_offload: ActiveKvOffloadConfig,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
) -> Result<()> {
    let state = build_gemma4_drafter_app_state(
        model,
        drafter,
        mtp_draft_tokens,
        prompt_lookup,
        tokenizer,
        model_id,
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        max_cache_cap,
        decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits,
        scheduler_runtime_profile,
        scheduler_autotune_report,
        vision_input_override,
        paged_prefix_cache,
        prefix_lru_cache,
        static_memory_estimate,
        active_kv_offload,
    )
    .await?;

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(gemma4_drafter_healthz_handler))
        .route(
            "/admin/api/prompt-lookup/clear",
            post(gemma4_drafter_clear_prompt_lookup_handler),
        )
        .route(
            "/v1/chat/completions",
            post(openai::gemma4_drafter_chat_completions),
        )
        .route("/v1/responses", post(responses::gemma4_drafter_responses))
        .route("/v1/messages", post(anthropic::gemma4_drafter_messages))
        .with_state(state);

    security::serve_router(app, network_config, "ironmlx Gemma4 drafter server").await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_plain_app_state<M>(
    model: M,
    tokenizer: Tokenizer,
    model_id: String,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<AppState<M>>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    build_app_state(
        model,
        tokenizer,
        model_id,
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        max_cache_cap,
        decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits,
        scheduler_runtime_profile,
        scheduler_autotune_report,
        vision_input_override,
        static_memory_estimate,
        None,
        PlainSchedulerActorSpawner {
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
            force_scheduler: false,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_prompt_lookup_app_state<M>(
    model: M,
    cfg: crate::core::prompt_lookup::PromptLookupConfig,
    tokenizer: Tokenizer,
    model_id: String,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<AppState<M>>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    let qualification =
        crate::core::prompt_lookup::PromptLookupQualificationRuntimeConfig::for_scheduler_profile(
            &scheduler_runtime_profile,
        )?;
    build_app_state(
        model,
        tokenizer,
        model_id,
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        max_cache_cap,
        decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits,
        scheduler_runtime_profile,
        scheduler_autotune_report,
        vision_input_override,
        static_memory_estimate,
        None,
        PromptLookupSchedulerActorSpawner {
            cfg,
            qualification,
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_mtp_app_state<M>(
    model: M,
    mtp: M::MtpHead,
    mtp_draft_tokens: usize,
    prompt_lookup: Option<crate::core::prompt_lookup::PromptLookupConfig>,
    tokenizer: Tokenizer,
    model_id: String,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<AppState<M>>
where
    M: Model + DenseVlMethods + MtpSpeculativeModel + Send + 'static,
    M::MtpHead: Send + 'static,
{
    let prompt_lookup = prompt_lookup
        .map(|cfg| -> Result<_> {
            let qualification = crate::core::prompt_lookup::PromptLookupQualificationRuntimeConfig::for_scheduler_profile_with_baseline(
                &scheduler_runtime_profile,
                crate::core::prompt_lookup::PromptLookupQualificationBaseline::QwenMtp,
            )?;
            Ok((cfg.validate()?, qualification))
        })
        .transpose()?;
    let exact_qualification =
        crate::core::speculative_qualification::NeuralExactQualificationRuntimeConfig::for_scheduler_profile(
            &scheduler_runtime_profile,
            crate::core::speculative_qualification::NeuralExactSource::QwenMtp,
        )?;
    build_app_state(
        model,
        tokenizer,
        model_id,
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        max_cache_cap,
        decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits,
        scheduler_runtime_profile,
        scheduler_autotune_report,
        vision_input_override,
        static_memory_estimate,
        Some(MtpHealthDraftTokens {
            requested: mtp_draft_tokens,
            effective: mtp_draft_tokens,
        }),
        MtpSchedulerActorSpawner {
            mtp,
            mtp_draft_tokens,
            exact_qualification,
            prompt_lookup,
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_gemma4_drafter_app_state(
    model: crate::models::Gemma4Model,
    drafter: crate::models::gemma4::Gemma4AssistantModel,
    mtp_draft_tokens: usize,
    prompt_lookup: Option<crate::core::prompt_lookup::PromptLookupConfig>,
    tokenizer: Tokenizer,
    model_id: String,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    paged_prefix_cache: Option<PagedPrefixCacheConfig>,
    prefix_lru_cache: Option<PrefixLruCacheConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
    active_kv_offload: ActiveKvOffloadConfig,
) -> Result<Gemma4DrafterAppState> {
    let kv_cache_profile = kv_cache_turboquant_bits
        .map(|bits| bits.to_string())
        .unwrap_or_else(|| "unquantized".to_string());
    tracing::info!(
        requested_draft_tokens = mtp_draft_tokens,
        effective_draft_tokens = mtp_draft_tokens,
        scheduler_b_max = b_max,
        kv_cache = %kv_cache_profile,
        paged_prefix_cache_enabled = paged_prefix_cache.is_some(),
        "Gemma4 drafter cap resolved"
    );
    let prompt_lookup = prompt_lookup
        .map(|cfg| -> Result<_> {
            let qualification = crate::core::prompt_lookup::PromptLookupQualificationRuntimeConfig::for_scheduler_profile_with_baseline(
                &scheduler_runtime_profile,
                crate::core::prompt_lookup::PromptLookupQualificationBaseline::Gemma4Assistant,
            )?;
            Ok((cfg.validate()?, qualification))
        })
        .transpose()?;
    let exact_qualification =
        crate::core::speculative_qualification::NeuralExactQualificationRuntimeConfig::for_scheduler_profile(
            &scheduler_runtime_profile,
            crate::core::speculative_qualification::NeuralExactSource::Gemma4Assistant,
        )?;
    let drafter = Arc::new(Mutex::new(drafter));
    let base = build_app_state(
        model,
        tokenizer,
        model_id.clone(),
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        max_cache_cap,
        decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits,
        scheduler_runtime_profile,
        scheduler_autotune_report,
        vision_input_override,
        static_memory_estimate,
        Some(MtpHealthDraftTokens {
            requested: mtp_draft_tokens,
            effective: mtp_draft_tokens,
        }),
        Gemma4DrafterSchedulerActorSpawner {
            drafter,
            mtp_draft_tokens,
            exact_qualification,
            prompt_lookup,
            paged_prefix_cache,
            prefix_lru_cache,
            active_kv_offload,
        },
    )
    .await?;

    Ok(Gemma4DrafterAppState {
        base,
        mtp_draft_tokens,
    })
}

#[allow(clippy::too_many_arguments)]
async fn build_app_state<M, S>(
    model: M,
    tokenizer: Tokenizer,
    model_id: String,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
    mtp_health_draft_tokens: Option<MtpHealthDraftTokens>,
    scheduler_actor_spawner: S,
) -> Result<AppState<M>>
where
    M: Model + DenseVlMethods + Send + 'static,
    S: SchedulerActorSpawner<M>,
{
    let model = Arc::new(Mutex::new(model));
    let admission_deadline = std::time::Duration::from_millis(admission_deadline_ms);

    // 3f + P5a-T5: extract ModelMeta (which now carries max_position_embeddings)
    // inside a single async lock guard so serve<M>() doesn't need a concrete
    // model-specific `config()` method. `blocking_lock` would panic here because
    // `serve` runs inside a Tokio runtime (tests S5 of 3d / 3f-T4 caught this).
    let mut meta = {
        let guard = model.lock().await;
        guard.model_meta()
    };
    meta.weight_bytes =
        effective_model_weight_bytes(meta.weight_bytes, static_memory_estimate.total_cold_bytes());
    let model_max_context: usize = meta.max_position_embeddings.max(0) as usize;
    let effective_cap_max = max_cache_cap.min(model_max_context);
    if max_cache_cap > model_max_context {
        tracing::warn!(
            "max_cache_cap CLI flag {} exceeds model_max_context {} — capping at {}",
            max_cache_cap,
            model_max_context,
            model_max_context
        );
    }
    if scheduler_autotune_report {
        let report = crate::core::scheduler_autotune::build_scheduler_autotune_report(
            crate::core::scheduler_autotune::SchedulerAutotuneInput {
                model_name: model_id.clone(),
                meta,
                prefill_chunk_size,
                b_max,
                admission_deadline_ms,
                admission_queue_max,
                requested_max_cache_cap: max_cache_cap,
                effective_cap_max,
                decode_cadence_mid_chunk_cap,
                total_ram_bytes: crate::core::memory_budget::system_total_ram_bytes(),
            },
            crate::core::scheduler_autotune::prompt_batch_limits_for_model::<M>(b_max),
        );
        tracing::info!(
            target: "ironmlx::scheduler_autotune",
            "\n{}",
            report.render_text()
        );
    }

    let paged_prefix_cache_enabled = scheduler_actor_spawner.paged_prefix_cache_enabled();
    let force_scheduler_for_greedy = scheduler_actor_spawner.force_scheduler_for_greedy();
    let prompt_lookup_config = scheduler_actor_spawner.prompt_lookup_config();
    let prompt_lookup_enabled = prompt_lookup_config.is_some();
    let scheduler_handle = scheduler_actor_spawner.spawn(
        model.clone(),
        b_max,
        admission_deadline,
        admission_queue_max,
        effective_cap_max,
        decode_cadence_mid_chunk_cap,
        meta,
    )?;
    let cold_materialization_tracker =
        crate::core::process_memory::ColdMaterializationTracker::new(static_memory_estimate);
    scheduler_handle
        .install_cold_materialization_tracker(Arc::clone(&cold_materialization_tracker))?;
    let vision_input = vision_input_override.unwrap_or(VisionInputConfig::Qwen {
        spatial_merge_size: meta.spatial_merge_size,
    });

    let mtp_health = mtp_health_draft_tokens
        .map(|draft_tokens| {
            health::MtpHealthConfig::enabled(
                draft_tokens.requested,
                draft_tokens.effective,
                scheduler_handle.mtp_prefill_count.clone(),
                scheduler_handle.mtp_step_count.clone(),
                scheduler_handle.mtp_fallback_prefill_count.clone(),
                scheduler_handle.mtp_drafted_tokens.clone(),
                scheduler_handle.mtp_accepted_draft_tokens.clone(),
                scheduler_handle.mtp_windows.clone(),
                scheduler_handle.mtp_multi_token_windows.clone(),
                scheduler_handle.mtp_exact_sampling_windows.clone(),
                scheduler_handle.mtp_exact_acceptance_draws.clone(),
                scheduler_handle.mtp_exact_residual_corrections.clone(),
                scheduler_handle.mtp_exact_bonus_samples.clone(),
                scheduler_handle.mtp_draft_forward_us.clone(),
                scheduler_handle.mtp_verify_forward_us.clone(),
                scheduler_handle.mtp_projection_us.clone(),
                scheduler_handle.mtp_sampling_us.clone(),
                scheduler_handle.mtp_draft_host_sync_count.clone(),
                scheduler_handle.mtp_draft_host_sync_us.clone(),
                scheduler_handle.mtp_verify_accept_host_sync_count.clone(),
                scheduler_handle.mtp_verify_accept_host_sync_us.clone(),
                scheduler_handle.mtp_main_rollback_us.clone(),
                scheduler_handle.mtp_cache_commit_us.clone(),
                scheduler_handle.mtp_prefill_cache_commit_us.clone(),
                scheduler_handle.mtp_decode_cache_commit_us.clone(),
                scheduler_handle.mtp_cache_restore_us.clone(),
                scheduler_handle.neural_exact_qualification_stats.clone(),
            )
        })
        .unwrap_or_else(health::MtpHealthConfig::disabled);
    let prompt_lookup_health = prompt_lookup_config
        .map(|config| {
            health::PromptLookupHealthConfig::enabled(
                config,
                scheduler_handle.prompt_lookup_published_stats.clone(),
            )
        })
        .unwrap_or_else(health::PromptLookupHealthConfig::disabled);
    let health_collector = build_health_collector(
        model_id.clone(),
        model_max_context,
        b_max,
        admission_queue_max,
        &scheduler_handle,
        mtp_health,
        prompt_lookup_health,
    );
    let runtime_usage = scheduler_handle.runtime_usage.clone();

    Ok(AppState {
        model,
        tokenizer: Arc::new(tokenizer),
        model_id,
        prefill_chunk_size,
        vision_input,
        request_execution: RequestExecutionHandle::Scheduler(Arc::new(scheduler_handle)),
        paged_prefix_cache_enabled,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        effective_cap_max, // 3f
        scheduler_runtime_profile: Arc::new(scheduler_runtime_profile),
        sampling_defaults: SamplingDefaults::default(),
        model_weight_bytes: meta.weight_bytes,
        static_memory_estimate,
        cold_materialization_tracker,
        kv_cache_turboquant_bits,
        force_scheduler_for_greedy,
        prompt_lookup_enabled,
        health_collector,
        runtime_usage,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn serve_with_dflash2<M>(
    model: M,
    draft: crate::models::DFlash2DraftModel,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    tensor_batch_max_width: usize,
    admission_queue_max: usize,
    max_cache_cap: usize,
    block_size: usize,
    draft_quantization_bits: Option<i32>,
    prefix_cache: Option<crate::core::cache::PrefixLruCacheConfig>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
) -> Result<()>
where
    M: Model + DenseVlMethods + crate::models::dflash2::DFlash2Target + Send + 'static,
{
    let model = Arc::new(Mutex::new(model));
    let (mut meta, dflash2_cache_cost) = {
        let guard = model.lock().await;
        (guard.model_meta(), guard.dflash2_target_cache_cost())
    };
    meta.weight_bytes =
        effective_model_weight_bytes(meta.weight_bytes, static_memory_estimate.total_cold_bytes());
    let model_max_context = meta.max_position_embeddings.max(0) as usize;
    let effective_cap_max = max_cache_cap.min(model_max_context);
    let budget_state = crate::core::memory_budget::validate_startup_budget_with_cost(
        b_max,
        effective_cap_max,
        &meta,
        dflash2_cache_cost.bytes_per_token,
        dflash2_cache_cost.fixed_bytes_per_sequence,
    )?;
    let tokenizer = Arc::new(tokenizer);
    let cold_materialization_tracker =
        crate::core::process_memory::ColdMaterializationTracker::new(static_memory_estimate);
    let dflash2_handle = dflash2_actor::spawn_dflash2_actor(
        Arc::clone(&model),
        draft,
        Arc::clone(&tokenizer),
        dflash2_actor::DFlash2ActorConfig {
            block_size,
            b_max,
            admission_deadline: std::time::Duration::from_millis(admission_deadline_ms),
            tensor_batch_max_width,
            admission_queue_max,
            effective_cap_max,
            budget_state,
            cache_cost: dflash2_cache_cost,
            prefix_cache_max_bytes: prefix_cache.map(|config| config.max_bytes),
        },
        Arc::clone(&cold_materialization_tracker),
    );
    let health_collector = Arc::new(health::SchedulerHealthCollector {
        start_time: std::time::Instant::now(),
        b_max,
        queue_max: admission_queue_max,
        model_name: model_id.clone(),
        max_position_embeddings: meta.max_position_embeddings,
        b_active: dflash2_handle.b_active.clone(),
        b_queued: dflash2_handle.b_queued.clone(),
        admit_count: dflash2_handle.admit_count.clone(),
        batch_count: dflash2_handle.batch_count.clone(),
        admission_queue_full_count: dflash2_handle.admission_queue_full_count.clone(),
        memory_budget_exceeded_count: dflash2_handle.memory_budget_exceeded_count.clone(),
        kv_cache_active_bytes: dflash2_handle.kv_cache_active_bytes.clone(),
        kv_cache_soft_limit_bytes: dflash2_handle.kv_cache_soft_limit_bytes,
        kv_cache_logical_cap_tokens: dflash2_handle.kv_cache_logical_cap_tokens,
        kv_cache_resident_cap_tokens: dflash2_handle.kv_cache_resident_cap_tokens,
        kv_cache_budget_policy: dflash2_handle.kv_cache_budget_policy.to_owned(),
        mtp: health::MtpHealthConfig::disabled(),
        dflash2: health::DFlash2HealthConfig::enabled(
            block_size,
            draft_quantization_bits,
            dflash2_handle.admit_count.clone(),
            dflash2_handle.windows.clone(),
            dflash2_handle.drafted_tokens.clone(),
            dflash2_handle.accepted_draft_tokens.clone(),
            dflash2_handle.rollback_count.clone(),
            dflash2_handle.tensor_batch_windows.clone(),
            dflash2_handle.tensor_batch_divergent_splits.clone(),
            dflash2_handle.tensor_batch_groups_created.clone(),
            dflash2_handle.tensor_batch_width_limit,
            dflash2_handle.tensor_batch_max_width.clone(),
            dflash2_handle.sampled_requests.clone(),
            dflash2_handle.exact_sampling_windows.clone(),
            dflash2_handle.exact_acceptance_draws.clone(),
            dflash2_handle.exact_residual_corrections.clone(),
            dflash2_handle.exact_bonus_samples.clone(),
            dflash2_handle.sampling_us.clone(),
            dflash2_handle.latest_generation_tps_bits.clone(),
            dflash2_handle.latest_acceptance_rate_bits.clone(),
            dflash2_handle.peak_memory_bytes.clone(),
            dflash2_handle.prefix_cache_enabled,
            dflash2_handle.prefix_cache_max_bytes,
            dflash2_handle.prefix_cache_entries.clone(),
            dflash2_handle.prefix_cache_bytes.clone(),
            dflash2_handle.prefix_cache_hits.clone(),
            dflash2_handle.prefix_cache_misses.clone(),
            dflash2_handle.prefix_cache_saves.clone(),
            dflash2_handle.prefix_cache_evictions.clone(),
            dflash2_handle.prefix_cache_hit_tokens.clone(),
            dflash2_handle.runtime_usage.clone(),
        ),
        prompt_lookup: health::PromptLookupHealthConfig::disabled(),
        active_kv_offload: crate::core::cache::ActiveKvOffloadSharedStats::new(
            &ActiveKvOffloadConfig::disabled(),
        ),
        immutable_prefix_blocks: scheduler_actor::ImmutablePrefixBlockSharedStats::new(false),
    });
    let runtime_usage = Arc::clone(&dflash2_handle.runtime_usage);
    let state = AppState {
        model,
        tokenizer,
        model_id,
        prefill_chunk_size,
        vision_input: VisionInputConfig::Qwen {
            spatial_merge_size: meta.spatial_merge_size,
        },
        request_execution: RequestExecutionHandle::DFlash2(Arc::new(dflash2_handle)),
        paged_prefix_cache_enabled: prefix_cache.is_some(),
        b_max,
        admission_deadline_ms: 0,
        admission_queue_max,
        effective_cap_max,
        scheduler_runtime_profile: Arc::new(scheduler_runtime_profile),
        sampling_defaults: SamplingDefaults::default(),
        model_weight_bytes: meta.weight_bytes,
        static_memory_estimate,
        cold_materialization_tracker,
        kv_cache_turboquant_bits: None,
        force_scheduler_for_greedy: true,
        prompt_lookup_enabled: false,
        health_collector,
        runtime_usage,
    };
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(healthz_handler))
        .route("/v1/models", get(dflash2_models_handler::<M>))
        .route(
            "/admin/api/prompt-lookup/clear",
            post(clear_prompt_lookup_handler),
        )
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/messages", post(anthropic::messages))
        .with_state(state);

    security::serve_router(app, network_config, "ironmlx DFlash2 server").await
}

#[derive(Debug, Serialize, PartialEq)]
struct DFlash2ModelList {
    object: &'static str,
    data: Vec<DFlash2ModelInfo>,
}

#[derive(Debug, Serialize, PartialEq)]
struct DFlash2ModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

fn dflash2_model_list(model_id: &str) -> DFlash2ModelList {
    DFlash2ModelList {
        object: "list",
        data: vec![DFlash2ModelInfo {
            id: model_id.to_owned(),
            object: "model",
            created: 0,
            owned_by: "ironmlx",
        }],
    }
}

async fn dflash2_models_handler<M>(State(state): State<AppState<M>>) -> Json<DFlash2ModelList>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    Json(dflash2_model_list(&state.model_id))
}

fn effective_model_weight_bytes(meta_weight_bytes: usize, loaded_weight_bytes: usize) -> usize {
    loaded_weight_bytes.max(meta_weight_bytes)
}

#[allow(clippy::too_many_arguments)]
async fn serve_inner<M, S>(
    model: M,
    tokenizer: Tokenizer,
    model_id: String,
    network_config: security::ServerNetworkConfig,
    prefill_chunk_size: usize,
    b_max: usize,
    admission_deadline_ms: u64,
    admission_queue_max: usize,
    max_cache_cap: usize,
    decode_cadence_mid_chunk_cap: usize,
    kv_cache_turboquant_bits: Option<TurboQuantKVBits>,
    scheduler_runtime_profile: SchedulerAutotuneRuntimeProfile,
    scheduler_autotune_report: bool,
    vision_input_override: Option<VisionInputConfig>,
    static_memory_estimate: crate::core::process_memory::StaticMemoryEstimate,
    mtp_health_draft_tokens: Option<MtpHealthDraftTokens>,
    scheduler_actor_spawner: S,
) -> Result<()>
where
    M: Model + DenseVlMethods + Send + 'static,
    S: SchedulerActorSpawner<M>,
{
    let state = build_app_state(
        model,
        tokenizer,
        model_id,
        prefill_chunk_size,
        b_max,
        admission_deadline_ms,
        admission_queue_max,
        max_cache_cap,
        decode_cadence_mid_chunk_cap,
        kv_cache_turboquant_bits,
        scheduler_runtime_profile,
        scheduler_autotune_report,
        vision_input_override,
        static_memory_estimate,
        mtp_health_draft_tokens,
        scheduler_actor_spawner,
    )
    .await?;
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/healthz", get(healthz_handler))
        .route(
            "/admin/api/prompt-lookup/clear",
            post(clear_prompt_lookup_handler),
        )
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/messages", post(anthropic::messages))
        .with_state(state);

    security::serve_router(app, network_config, "ironmlx server").await
}

fn build_health_collector(
    model_id: String,
    model_max_context: usize,
    b_max: usize,
    admission_queue_max: usize,
    scheduler_handle: &scheduler_actor::SchedulerActorHandle,
    mtp: health::MtpHealthConfig,
    prompt_lookup: health::PromptLookupHealthConfig,
) -> Arc<health::SchedulerHealthCollector> {
    Arc::new(health::SchedulerHealthCollector {
        start_time: std::time::Instant::now(),
        b_max,
        queue_max: admission_queue_max,
        model_name: model_id,
        max_position_embeddings: model_max_context as i32,
        b_active: scheduler_handle.b_active.clone(),
        b_queued: scheduler_handle.b_queued.clone(),
        admit_count: scheduler_handle.admit_count.clone(),
        batch_count: scheduler_handle.batch_count.clone(),
        admission_queue_full_count: scheduler_handle.admission_queue_full_count.clone(),
        memory_budget_exceeded_count: scheduler_handle.memory_budget_exceeded_count.clone(),
        kv_cache_active_bytes: scheduler_handle.kv_cache_active_bytes.clone(),
        kv_cache_soft_limit_bytes: scheduler_handle.kv_cache_soft_limit_bytes,
        kv_cache_logical_cap_tokens: scheduler_handle.kv_cache_logical_cap_tokens,
        kv_cache_resident_cap_tokens: scheduler_handle.kv_cache_resident_cap_tokens,
        kv_cache_budget_policy: scheduler_handle.kv_cache_budget_policy.to_string(),
        mtp,
        dflash2: health::DFlash2HealthConfig::disabled(),
        prompt_lookup,
        active_kv_offload: scheduler_handle.active_kv_offload.clone(),
        immutable_prefix_blocks: scheduler_handle.immutable_prefix_blocks.clone(),
    })
}

/// GET /healthz — returns a JSON HealthSnapshot. Reads only Arc atomics;
/// no lock contention with the model or SchedulerActor. B1-p2.5 G3.
#[derive(Debug, Serialize)]
struct SingleActorHealthResponse<T> {
    #[serde(flatten)]
    health: T,
    mode: &'static str,
    models: [(); 0],
}

impl<T> SingleActorHealthResponse<T> {
    fn new(health: T) -> Self {
        Self {
            health,
            mode: "single",
            models: [],
        }
    }
}

async fn healthz_handler<M>(
    axum::extract::State(state): axum::extract::State<AppState<M>>,
) -> axum::Json<SingleActorHealthResponse<health::HealthSnapshot>>
where
    M: Model + DenseVlMethods + Send + 'static,
{
    axum::Json(SingleActorHealthResponse::new(
        state.health_collector.snapshot(),
    ))
}

async fn gemma4_drafter_healthz_handler(
    axum::extract::State(state): axum::extract::State<Gemma4DrafterAppState>,
) -> axum::Json<SingleActorHealthResponse<health::HealthSnapshot>> {
    axum::Json(SingleActorHealthResponse::new(
        state.base.health_collector.snapshot(),
    ))
}

async fn clear_prompt_lookup_handler<M>(
    axum::extract::State(state): axum::extract::State<AppState<M>>,
) -> (axum::http::StatusCode, axum::Json<serde_json::Value>)
where
    M: Model + DenseVlMethods + Send + 'static,
{
    clear_prompt_lookup_response(&state.request_execution, &state.model_id).await
}

async fn gemma4_drafter_clear_prompt_lookup_handler(
    axum::extract::State(state): axum::extract::State<Gemma4DrafterAppState>,
) -> (axum::http::StatusCode, axum::Json<serde_json::Value>) {
    clear_prompt_lookup_response(&state.base.request_execution, &state.base.model_id).await
}

async fn clear_prompt_lookup_response(
    request_execution: &RequestExecutionHandle,
    model_id: &str,
) -> (axum::http::StatusCode, axum::Json<serde_json::Value>) {
    match request_execution.clear_shared_prompt_lookup().await {
        Ok(cleared_entries) => (
            axum::http::StatusCode::OK,
            axum::Json(serde_json::json!({
                "success": true,
                "status": "cleared",
                "model": model_id,
                "cleared_models": 1,
                "cleared_entries": cleared_entries,
            })),
        ),
        Err(error) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({
                "success": false,
                "status": "error",
                "model": model_id,
                "cleared_models": 0,
                "cleared_entries": 0,
                "error": error.to_string(),
            })),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    use std::time::Duration;

    use mlx::{Array, Dtype, StreamOrDevice};
    use tokio::sync::mpsc;
    use tokio::time::sleep;

    use crate::nn::LayerCache;

    #[test]
    fn effective_model_weight_bytes_uses_loaded_tensor_bytes_when_larger() {
        assert_eq!(effective_model_weight_bytes(1_024, 4_096), 4_096);
    }

    #[test]
    fn effective_model_weight_bytes_keeps_meta_estimate_when_larger() {
        assert_eq!(effective_model_weight_bytes(4_096, 1_024), 4_096);
    }

    #[test]
    fn dflash2_model_list_exposes_only_the_public_target_identifier() {
        let list = dflash2_model_list("mlx-community/Qwen3.8-27B-4bit");
        let json = serde_json::to_value(list).expect("serialize model list");

        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().map(Vec::len), Some(1));
        assert_eq!(json["data"][0]["id"], "mlx-community/Qwen3.8-27B-4bit");
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["owned_by"], "ironmlx");
    }

    #[test]
    fn single_actor_health_response_exposes_app_runtime_contract() {
        let response = SingleActorHealthResponse::new(serde_json::json!({
            "status": "healthy",
            "model": {"name": "test-model"},
        }));
        let json = serde_json::to_value(response).expect("serialize health response");

        assert_eq!(json["status"], "healthy");
        assert_eq!(json["model"]["name"], "test-model");
        assert_eq!(json["mode"], "single");
        assert_eq!(json["models"], serde_json::json!([]));
    }

    struct DefaultRouteModel;
    struct LimitedRouteModel;

    impl Model for DefaultRouteModel {
        fn make_cache(&self, _batch: i32, _cap: i32, _dtype: Dtype) -> Result<Vec<LayerCache>> {
            unimplemented!("route tests only call the associated route policy")
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
            unimplemented!("route tests only call the associated route policy")
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
            unimplemented!("route tests only call the associated route policy")
        }

        fn forward_text_hidden(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            unimplemented!("route tests only call the associated route policy")
        }

        fn model_meta(&self) -> crate::core::memory_budget::ModelMeta {
            crate::core::memory_budget::test_meta_qwen35()
        }

        fn num_hidden_layers(&self) -> usize {
            0
        }
    }

    impl Model for LimitedRouteModel {
        fn make_cache(&self, _batch: i32, _cap: i32, _dtype: Dtype) -> Result<Vec<LayerCache>> {
            unimplemented!("route tests only call the associated route policy")
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
            unimplemented!("route tests only call the associated route policy")
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
            unimplemented!("route tests only call the associated route policy")
        }

        fn forward_text_hidden(
            &self,
            _input_ids: &Array,
            _position_ids: &Array,
            _per_row_lens: Option<&[i32]>,
            _decode_mask: Option<&Array>,
            _cache: Option<&mut [LayerCache]>,
            _target: StreamOrDevice,
        ) -> Result<Array> {
            unimplemented!("route tests only call the associated route policy")
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
    }

    #[test]
    fn route_keeps_unlimited_model_long_prompt_on_generation_stream() {
        assert!(!should_route_to_scheduler::<DefaultRouteModel>(
            4096, 2048, 4, false, false,
        ));
    }

    #[test]
    fn route_uses_scheduler_for_model_limited_chunked_long_prompt() {
        assert!(should_route_to_scheduler::<LimitedRouteModel>(
            4096, 2048, 4, false, false,
        ));
    }

    #[test]
    fn route_uses_scheduler_for_long_prompt_when_paged_prefix_cache_enabled() {
        assert!(should_route_to_scheduler::<DefaultRouteModel>(
            4096, 2048, 4, true, false,
        ));
    }

    #[test]
    fn route_uses_scheduler_for_long_prompt_when_greedy_scheduler_is_forced() {
        assert!(should_route_to_scheduler::<DefaultRouteModel>(
            4096, 2048, 1, false, true,
        ));
    }

    #[test]
    fn prompt_lookup_sampler_validation_accepts_exact_non_greedy() {
        assert!(validate_prompt_lookup_sampler(true, Sampler::greedy()).is_ok());
        assert!(
            validate_prompt_lookup_sampler(false, Sampler::greedy().with_temperature(0.7)).is_ok()
        );
        assert!(
            validate_prompt_lookup_sampler(true, Sampler::greedy().with_temperature(0.7)).is_ok()
        );
        let sampler = Sampler {
            temperature: f32::NAN,
            ..Sampler::greedy()
        };
        assert_eq!(
            validate_prompt_lookup_sampler(true, sampler)
                .unwrap_err()
                .to_string(),
            "sampling temperature must be finite"
        );
    }

    fn test_scheduler_handle() -> scheduler_actor::SchedulerActorHandle {
        let (cmd_tx, _cmd_rx) = mpsc::channel(1);
        let (control_tx, _control_rx) = mpsc::channel(1);
        let queue_rejected = Arc::new(AtomicU64::new(0));
        scheduler_actor::SchedulerActorHandle {
            cmd_tx,
            control_tx,
            cold_materialization_tracker: Arc::new(std::sync::OnceLock::new()),
            runtime_usage: Arc::new(
                crate::core::runtime_usage::ModelRuntimeUsageCounters::default(),
            ),
            admit_count: Arc::new(AtomicU64::new(0)),
            batch_count: Arc::new(AtomicU64::new(0)),
            saturate_triggered: Arc::new(AtomicU64::new(0)),
            queue_depth_peak: Arc::new(AtomicUsize::new(0)),
            queue_rejected: queue_rejected.clone(),
            mtp_prefill_count: Arc::new(AtomicU64::new(0)),
            mtp_step_count: Arc::new(AtomicU64::new(0)),
            mtp_fallback_prefill_count: Arc::new(AtomicU64::new(0)),
            mtp_drafted_tokens: Arc::new(AtomicU64::new(0)),
            mtp_accepted_draft_tokens: Arc::new(AtomicU64::new(0)),
            mtp_windows: Arc::new(AtomicU64::new(0)),
            mtp_multi_token_windows: Arc::new(AtomicU64::new(0)),
            mtp_exact_sampling_windows: Arc::new(AtomicU64::new(0)),
            mtp_exact_acceptance_draws: Arc::new(AtomicU64::new(0)),
            mtp_exact_residual_corrections: Arc::new(AtomicU64::new(0)),
            mtp_exact_bonus_samples: Arc::new(AtomicU64::new(0)),
            mtp_draft_forward_us: Arc::new(AtomicU64::new(0)),
            mtp_verify_forward_us: Arc::new(AtomicU64::new(0)),
            mtp_projection_us: Arc::new(AtomicU64::new(0)),
            mtp_sampling_us: Arc::new(AtomicU64::new(0)),
            mtp_draft_host_sync_count: Arc::new(AtomicU64::new(0)),
            mtp_draft_host_sync_us: Arc::new(AtomicU64::new(0)),
            mtp_verify_accept_host_sync_count: Arc::new(AtomicU64::new(0)),
            mtp_verify_accept_host_sync_us: Arc::new(AtomicU64::new(0)),
            mtp_main_rollback_us: Arc::new(AtomicU64::new(0)),
            mtp_cache_commit_us: Arc::new(AtomicU64::new(0)),
            mtp_prefill_cache_commit_us: Arc::new(AtomicU64::new(0)),
            mtp_decode_cache_commit_us: Arc::new(AtomicU64::new(0)),
            mtp_cache_restore_us: Arc::new(AtomicU64::new(0)),
            prompt_lookup_published_stats: Arc::new(std::sync::Mutex::new(None)),
            neural_exact_qualification_stats: Arc::new(std::sync::Mutex::new(
                crate::core::speculative_qualification::NeuralExactQualificationStats::default(),
            )),
            b_active: Arc::new(AtomicU64::new(0)),
            b_queued: Arc::new(AtomicU64::new(0)),
            admission_queue_full_count: queue_rejected,
            memory_budget_exceeded_count: Arc::new(AtomicU64::new(0)),
            kv_cache_active_bytes: Arc::new(AtomicUsize::new(0)),
            kv_cache_soft_limit_bytes: 1,
            kv_cache_logical_cap_tokens: 1,
            kv_cache_resident_cap_tokens: 1,
            kv_cache_budget_policy: "full_resident",
            active_kv_offload: crate::core::cache::ActiveKvOffloadSharedStats::new(
                &crate::core::cache::ActiveKvOffloadConfig::disabled(),
            ),
            immutable_prefix_blocks: scheduler_actor::ImmutablePrefixBlockSharedStats::new(false),
        }
    }

    #[test]
    fn health_collector_mtp_disabled_without_server_mtp_config() {
        let handle = test_scheduler_handle();
        let collector = build_health_collector(
            "test-model".to_string(),
            4096,
            1,
            8,
            &handle,
            health::MtpHealthConfig::disabled(),
            health::PromptLookupHealthConfig::disabled(),
        );
        let snapshot = collector.snapshot();

        assert!(!snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.draft_tokens, None);
        assert_eq!(snapshot.mtp.prefill_count, 0);
        assert_eq!(snapshot.mtp.step_count, 0);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 0);
        assert_eq!(snapshot.mtp.drafted_tokens, 0);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 0);
    }

    #[test]
    fn health_collector_mtp_enabled_uses_scheduler_actor_counters() {
        let handle = test_scheduler_handle();
        handle.mtp_prefill_count.store(3, Ordering::Relaxed);
        handle.mtp_step_count.store(5, Ordering::Relaxed);
        handle
            .mtp_fallback_prefill_count
            .store(7, Ordering::Relaxed);
        handle.mtp_drafted_tokens.store(11, Ordering::Relaxed);
        handle
            .mtp_accepted_draft_tokens
            .store(13, Ordering::Relaxed);
        handle.mtp_windows.store(17, Ordering::Relaxed);
        handle.mtp_multi_token_windows.store(13, Ordering::Relaxed);
        handle.mtp_draft_forward_us.store(19, Ordering::Relaxed);
        handle.mtp_verify_forward_us.store(23, Ordering::Relaxed);
        handle.mtp_projection_us.store(29, Ordering::Relaxed);
        handle.mtp_sampling_us.store(31, Ordering::Relaxed);
        handle.mtp_main_rollback_us.store(37, Ordering::Relaxed);
        handle.mtp_cache_commit_us.store(41, Ordering::Relaxed);
        handle
            .mtp_prefill_cache_commit_us
            .store(17, Ordering::Relaxed);
        handle
            .mtp_decode_cache_commit_us
            .store(24, Ordering::Relaxed);
        handle.mtp_cache_restore_us.store(43, Ordering::Relaxed);
        let collector = build_health_collector(
            "test-model".to_string(),
            4096,
            1,
            8,
            &handle,
            health::MtpHealthConfig::enabled(
                2,
                2,
                handle.mtp_prefill_count.clone(),
                handle.mtp_step_count.clone(),
                handle.mtp_fallback_prefill_count.clone(),
                handle.mtp_drafted_tokens.clone(),
                handle.mtp_accepted_draft_tokens.clone(),
                handle.mtp_windows.clone(),
                handle.mtp_multi_token_windows.clone(),
                handle.mtp_exact_sampling_windows.clone(),
                handle.mtp_exact_acceptance_draws.clone(),
                handle.mtp_exact_residual_corrections.clone(),
                handle.mtp_exact_bonus_samples.clone(),
                handle.mtp_draft_forward_us.clone(),
                handle.mtp_verify_forward_us.clone(),
                handle.mtp_projection_us.clone(),
                handle.mtp_sampling_us.clone(),
                handle.mtp_draft_host_sync_count.clone(),
                handle.mtp_draft_host_sync_us.clone(),
                handle.mtp_verify_accept_host_sync_count.clone(),
                handle.mtp_verify_accept_host_sync_us.clone(),
                handle.mtp_main_rollback_us.clone(),
                handle.mtp_cache_commit_us.clone(),
                handle.mtp_prefill_cache_commit_us.clone(),
                handle.mtp_decode_cache_commit_us.clone(),
                handle.mtp_cache_restore_us.clone(),
                handle.neural_exact_qualification_stats.clone(),
            ),
            health::PromptLookupHealthConfig::disabled(),
        );
        let snapshot = collector.snapshot();

        assert!(snapshot.mtp.enabled);
        assert_eq!(snapshot.mtp.requested_draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.draft_tokens, Some(2));
        assert_eq!(snapshot.mtp.prefill_count, 3);
        assert_eq!(snapshot.mtp.step_count, 5);
        assert_eq!(snapshot.mtp.fallback_prefill_count, 7);
        assert_eq!(snapshot.mtp.drafted_tokens, 11);
        assert_eq!(snapshot.mtp.accepted_draft_tokens, 13);
        assert_eq!(snapshot.mtp.windows, 17);
        assert_eq!(snapshot.mtp.multi_token_windows, 13);
        assert_eq!(snapshot.mtp.draft_forward_us, 19);
        assert_eq!(snapshot.mtp.verify_forward_us, 23);
        assert_eq!(snapshot.mtp.projection_us, 29);
        assert_eq!(snapshot.mtp.sampling_us, 31);
        assert_eq!(snapshot.mtp.main_rollback_us, 37);
        assert_eq!(snapshot.mtp.cache_commit_us, 41);
        assert_eq!(snapshot.mtp.prefill_cache_commit_us, 17);
        assert_eq!(snapshot.mtp.decode_cache_commit_us, 24);
        assert_eq!(snapshot.mtp.cache_restore_us, 43);
    }

    /// Verify two concurrent task acquisitions of the same Mutex serialize.
    /// We don't construct a real Qwen35Model — Mutex<()> exhibits the same
    /// serialization semantics, and that's the load-bearing contract here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mutex_serializes_concurrent_acquirers() {
        let m = Arc::new(Mutex::new(()));
        let m1 = m.clone();
        let m2 = m.clone();

        let timeline: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let t1 = timeline.clone();
        let t2 = timeline.clone();

        let h1 = tokio::spawn(async move {
            let _g = m1.lock().await;
            t1.lock().await.push("1-start");
            sleep(Duration::from_millis(50)).await;
            t1.lock().await.push("1-end");
        });
        sleep(Duration::from_millis(5)).await; // ensure h1 grabs lock first
        let h2 = tokio::spawn(async move {
            let _g = m2.lock().await;
            t2.lock().await.push("2-start");
            t2.lock().await.push("2-end");
        });

        let _ = h1.await;
        let _ = h2.await;

        let tl = timeline.lock().await;
        assert_eq!(*tl, vec!["1-start", "1-end", "2-start", "2-end"]);
    }
}
