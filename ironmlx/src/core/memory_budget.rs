//! Memory budget estimation for the batched scheduler. Computes
//! GQA-aware KV cache bytes from `ModelMeta` and validates
//! `b_max × effective_cap_max × per_token_kv_bytes` against system
//! RAM minus model footprint and safety margin.
//!
//! Used at `Scheduler::new` (startup validation) and `admit_inner`
//! (runtime admission gate) for production memory safety.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub struct ModelMeta {
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub hidden_size: i32,
    pub head_dim: Option<i32>,
    pub weight_bytes: usize,
    /// Maximum sequence length the model supports. Used by `serve()` for
    /// computing `effective_cap_max = min(--max-cache-cap CLI, max_position_embeddings)`.
    /// P5a-T5: added here so `serve<M>()` can read it from the `Model` trait
    /// without requiring a concrete model-specific `config()` method.
    pub max_position_embeddings: i32,
    /// VL vision spatial merge size (= VisionConfig.spatial_merge_size).
    /// Defaults to 2 for text-only models (unused when no images present).
    /// P5a-T5: carried here so generic HTTP handlers don't need a
    /// model-specific `config()` method.
    pub spatial_merge_size: i32,
}

impl ModelMeta {
    pub fn effective_head_dim(&self) -> i32 {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

pub const SAFETY_MARGIN_BYTES: usize = 2 * 1024 * 1024 * 1024;
pub const SOFT_LIMIT_FRAC: f64 = 0.85;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBudgetPolicy {
    FullResident,
    ActiveKvOffload { resident_cap: usize },
}

impl KvBudgetPolicy {
    pub fn active_kv_offload(resident_cap: usize) -> Self {
        Self::ActiveKvOffload {
            resident_cap: resident_cap.max(1),
        }
    }

    pub fn resident_cap(self, logical_cap: usize) -> usize {
        match self {
            Self::FullResident => logical_cap,
            Self::ActiveKvOffload { resident_cap } => resident_cap.min(logical_cap).max(1),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::FullResident => "full_resident",
            Self::ActiveKvOffload { .. } => "active_kv_offload",
        }
    }
}

pub fn kv_bytes_per_token(meta: &ModelMeta) -> usize {
    (meta.num_hidden_layers as usize)
        * (meta.num_key_value_heads as usize)
        * (meta.effective_head_dim() as usize)
        * 2  // K + V
        * 2 // bf16
}

pub fn kv_cache_bytes(b: usize, cap: usize, meta: &ModelMeta) -> usize {
    b * cap * kv_bytes_per_token(meta)
}

pub fn system_total_ram_bytes() -> usize {
    if let Ok(s) = std::env::var("IRONMLX_TOTAL_RAM_BYTES") {
        if let Ok(n) = s.parse::<usize>() {
            return n;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(n) = macos_total_ram_bytes() {
            return n;
        }
        tracing::error!("failed to query macOS hw.memsize with sysctlbyname; using 8 GiB fallback");
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    if let Some(kb_str) = rest.trim().split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }
    8 * 1024 * 1024 * 1024
}

#[cfg(target_os = "macos")]
fn macos_total_ram_bytes() -> Option<usize> {
    let mut value = 0_u64;
    let mut value_size = std::mem::size_of::<u64>();
    // SAFETY: `hw.memsize` is a NUL-terminated, immutable key. `value` and
    // `value_size` are valid writable storage for the duration of the call.
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut value as *mut u64).cast(),
            &mut value_size,
            std::ptr::null_mut(),
            0,
        )
    };
    (result == 0 && value_size == std::mem::size_of::<u64>() && value > 0)
        .then(|| usize::try_from(value).ok())
        .flatten()
}

pub fn available_budget_bytes(meta: &ModelMeta) -> usize {
    available_budget_bytes_for_total_ram(meta, system_total_ram_bytes())
}

fn available_budget_bytes_for_total_ram(meta: &ModelMeta, total_ram_bytes: usize) -> usize {
    total_ram_bytes
        .saturating_sub(meta.weight_bytes)
        .saturating_sub(SAFETY_MARGIN_BYTES)
}

#[derive(Debug, Error)]
#[error(
    "memory budget exceeded: b_max={b_max} × (resident_cap={resident_cap} × \
     {bytes_per_token} bytes/token + {fixed_bytes_per_sequence} fixed bytes/sequence) = \
     {requested_bytes} bytes > available {available_bytes} \
     (logical cap {cap}, policy {policy}, total RAM {total_ram_bytes} - model {model_weight_bytes} - safety margin 2147483648). \
     Lower --b-max or --max-cache-cap."
)]
pub struct MemoryBudgetError {
    pub b_max: usize,
    pub cap: usize,
    pub resident_cap: usize,
    pub policy: &'static str,
    pub bytes_per_token: usize,
    pub fixed_bytes_per_sequence: usize,
    pub requested_bytes: usize,
    pub available_bytes: usize,
    pub total_ram_bytes: usize,
    pub model_weight_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct BudgetState {
    soft_limit: usize,
    active: Arc<AtomicUsize>,
    logical_cap: usize,
    resident_cap: usize,
    policy: KvBudgetPolicy,
}

impl BudgetState {
    pub fn new(total_budget: usize) -> Self {
        Self::with_caps(
            total_budget,
            usize::MAX,
            usize::MAX,
            KvBudgetPolicy::FullResident,
        )
    }

    pub fn with_caps(
        total_budget: usize,
        logical_cap: usize,
        resident_cap: usize,
        policy: KvBudgetPolicy,
    ) -> Self {
        Self::with_soft_limit(
            ((total_budget as f64) * SOFT_LIMIT_FRAC) as usize,
            logical_cap,
            resident_cap,
            policy,
        )
    }

    pub fn with_soft_limit(
        soft_limit: usize,
        logical_cap: usize,
        resident_cap: usize,
        policy: KvBudgetPolicy,
    ) -> Self {
        Self {
            soft_limit,
            active: Arc::new(AtomicUsize::new(0)),
            logical_cap,
            resident_cap,
            policy,
        }
    }

    pub fn soft_limit(&self) -> usize {
        self.soft_limit
    }

    pub fn active_bytes(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }

    pub fn shared_active(&self) -> Arc<AtomicUsize> {
        self.active.clone()
    }

    pub fn logical_cap(&self) -> usize {
        self.logical_cap
    }

    pub fn resident_cap(&self) -> usize {
        self.resident_cap
    }

    pub fn policy(&self) -> KvBudgetPolicy {
        self.policy
    }

    pub fn resident_charge_cap(&self, requested_cap: usize) -> usize {
        requested_cap.min(self.resident_cap)
    }

    /// 试图把 `requested` 加到 active；若加后超 soft_limit 则返回 Err。
    pub fn try_admit(&self, requested: usize) -> Result<(), (usize, usize, usize)> {
        let cur = self.active.load(Ordering::Relaxed);
        if cur + requested > self.soft_limit {
            return Err((cur, requested, self.soft_limit));
        }
        self.active.fetch_add(requested, Ordering::Relaxed);
        Ok(())
    }

    pub fn release(&self, bytes: usize) {
        self.active.fetch_sub(bytes, Ordering::Relaxed);
    }
}

pub fn validate_startup_budget(
    b_max: usize,
    effective_cap_max: usize,
    meta: &ModelMeta,
) -> Result<BudgetState, MemoryBudgetError> {
    validate_startup_budget_with_policy(
        b_max,
        effective_cap_max,
        meta,
        KvBudgetPolicy::FullResident,
    )
}

pub fn validate_startup_budget_with_policy(
    b_max: usize,
    effective_cap_max: usize,
    meta: &ModelMeta,
    policy: KvBudgetPolicy,
) -> Result<BudgetState, MemoryBudgetError> {
    validate_startup_budget_with_cost_for_total_ram(
        b_max,
        effective_cap_max,
        meta,
        kv_bytes_per_token(meta),
        0,
        policy,
        system_total_ram_bytes(),
    )
}

pub(crate) fn validate_startup_budget_with_cost(
    b_max: usize,
    effective_cap_max: usize,
    meta: &ModelMeta,
    bytes_per_token: usize,
    fixed_bytes_per_sequence: usize,
) -> Result<BudgetState, MemoryBudgetError> {
    validate_startup_budget_with_cost_for_total_ram(
        b_max,
        effective_cap_max,
        meta,
        bytes_per_token,
        fixed_bytes_per_sequence,
        KvBudgetPolicy::FullResident,
        system_total_ram_bytes(),
    )
}

#[cfg(test)]
fn validate_startup_budget_with_policy_for_total_ram(
    b_max: usize,
    effective_cap_max: usize,
    meta: &ModelMeta,
    policy: KvBudgetPolicy,
    total_ram_bytes: usize,
) -> Result<BudgetState, MemoryBudgetError> {
    validate_startup_budget_with_cost_for_total_ram(
        b_max,
        effective_cap_max,
        meta,
        kv_bytes_per_token(meta),
        0,
        policy,
        total_ram_bytes,
    )
}

fn validate_startup_budget_with_cost_for_total_ram(
    b_max: usize,
    effective_cap_max: usize,
    meta: &ModelMeta,
    bytes_per_token: usize,
    fixed_bytes_per_sequence: usize,
    policy: KvBudgetPolicy,
    total_ram_bytes: usize,
) -> Result<BudgetState, MemoryBudgetError> {
    let resident_cap = policy.resident_cap(effective_cap_max);
    let requested_per_sequence = resident_cap
        .saturating_mul(bytes_per_token)
        .saturating_add(fixed_bytes_per_sequence);
    let requested = b_max.saturating_mul(requested_per_sequence);
    let available = available_budget_bytes_for_total_ram(meta, total_ram_bytes);
    if requested > available {
        return Err(MemoryBudgetError {
            b_max,
            cap: effective_cap_max,
            resident_cap,
            policy: policy.name(),
            bytes_per_token,
            fixed_bytes_per_sequence,
            requested_bytes: requested,
            available_bytes: available,
            total_ram_bytes,
            model_weight_bytes: meta.weight_bytes,
        });
    }

    let soft_limit = match policy {
        KvBudgetPolicy::FullResident => {
            (((available as f64) * SOFT_LIMIT_FRAC) as usize).max(requested)
        }
        KvBudgetPolicy::ActiveKvOffload { .. } => requested,
    };
    Ok(BudgetState::with_soft_limit(
        soft_limit,
        effective_cap_max,
        resident_cap,
        policy,
    ))
}

/// Realistic Qwen3.5-4B-like ModelMeta for tests.
#[doc(hidden)]
pub fn test_meta_qwen35() -> ModelMeta {
    ModelMeta {
        num_hidden_layers: 28,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        hidden_size: 4096,
        head_dim: None,
        weight_bytes: 3 * 1024 * 1024 * 1024,
        max_position_embeddings: 32768,
        spatial_merge_size: 2,
    }
}

/// Realistic Qwen3.5-35B-A3B-4bit ModelMeta for tests.
///
/// Values from real snapshot text_config (verified P5b T0). The
/// `weight_bytes` is computed via the MoE-aware formula and rounded
/// to 17 GiB which matches `Qwen35MoeModel::approx_weight_bytes`
/// closely for the published config.
#[doc(hidden)]
pub fn test_meta_qwen35_moe() -> ModelMeta {
    ModelMeta {
        num_hidden_layers: 40,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        hidden_size: 2048,
        head_dim: Some(256),
        // approx: attn (4 * 2048^2 * 40 / 2) ≈ 335 MB
        //         routed (3 * 256 * 2048 * 512 * 40 / 2) ≈ 16.1 GB
        //         shared (3 * 2048 * 512 * 40 / 2) ≈ 63 MB
        //         embed + lm_head (2 * 248320 * 2048 / 2) ≈ 0.5 GB
        // total ≈ 17 GB
        weight_bytes: 17 * 1024 * 1024 * 1024,
        max_position_embeddings: 262144,
        spatial_merge_size: 2,
    }
}

#[doc(hidden)]
pub fn test_meta_gemma4_12b() -> ModelMeta {
    ModelMeta {
        num_hidden_layers: 48,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        hidden_size: 3840,
        head_dim: Some(512),
        weight_bytes: 8 * 1024 * 1024 * 1024,
        max_position_embeddings: 262144,
        spatial_merge_size: 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_total_ram_query_does_not_depend_on_path_lookup() {
        let total = macos_total_ram_bytes().expect("sysctlbyname hw.memsize");
        assert!(total > 0);
    }

    fn validate_with_total_ram(
        b_max: usize,
        effective_cap_max: usize,
        meta: &ModelMeta,
        total_ram_bytes: usize,
    ) -> Result<BudgetState, MemoryBudgetError> {
        validate_startup_budget_with_policy_for_total_ram(
            b_max,
            effective_cap_max,
            meta,
            KvBudgetPolicy::FullResident,
            total_ram_bytes,
        )
    }

    fn meta() -> ModelMeta {
        test_meta_qwen35()
    }

    #[test]
    fn kv_bytes_per_token_gqa_aware() {
        // 28 × 8 × 128 (4096/32) × 2 × 2 = 114688
        assert_eq!(kv_bytes_per_token(&meta()), 114_688);
    }

    #[test]
    fn kv_cache_bytes_scales_with_b_and_cap() {
        let bytes = kv_cache_bytes(1, 1024, &meta());
        assert_eq!(bytes, 1024 * 114_688);
        assert_eq!(kv_cache_bytes(2, 1024, &meta()), 2 * bytes);
    }

    #[test]
    fn validate_within_budget_ok() {
        let st = validate_with_total_ram(1, 4096, &meta(), 34_359_738_368).expect("should fit");
        assert!(st.soft_limit() > 0);
    }

    #[test]
    fn validate_over_budget_err() {
        let err = validate_with_total_ram(4, 32768, &meta(), 8_589_934_592)
            .expect_err("4 × 32768 × 114688 should exceed 8 - 3 - 2 = 3 GiB budget");
        let msg = format!("{err}");
        assert!(msg.contains("memory budget exceeded"), "msg: {msg}");
        assert!(msg.contains("Lower --b-max"), "msg: {msg}");
    }

    #[test]
    fn custom_cache_cost_includes_fixed_per_sequence_state() {
        let meta = meta();
        let total_ram = meta
            .weight_bytes
            .saturating_add(SAFETY_MARGIN_BYTES)
            .saturating_add(1_000);
        let accepted = validate_startup_budget_with_cost_for_total_ram(
            2,
            100,
            &meta,
            2,
            300,
            KvBudgetPolicy::FullResident,
            total_ram,
        )
        .expect("2 × (100 × 2 + 300) exactly fits");
        assert_eq!(accepted.soft_limit(), 1_000);

        let rejected = validate_startup_budget_with_cost_for_total_ram(
            2,
            100,
            &meta,
            2,
            301,
            KvBudgetPolicy::FullResident,
            total_ram,
        )
        .expect_err("fixed sequence state must participate in startup admission");
        assert_eq!(rejected.fixed_bytes_per_sequence, 301);
        assert_eq!(rejected.requested_bytes, 1_002);
    }

    #[test]
    fn budget_state_admit_release_round_trip() {
        let st = BudgetState::new(1_000_000);
        assert_eq!(st.active_bytes(), 0);
        st.try_admit(500_000).expect("under soft limit (850k)");
        assert_eq!(st.active_bytes(), 500_000);
        let err = st.try_admit(400_000);
        assert!(err.is_err(), "should reject above soft limit");
        assert_eq!(
            st.active_bytes(),
            500_000,
            "rejected admit leaves state unchanged"
        );
        st.release(500_000);
        assert_eq!(st.active_bytes(), 0);
    }

    #[test]
    fn moe_kv_bytes_per_token_matches_gqa_formula() {
        let m = test_meta_qwen35_moe();
        // 40 layers × 2 KV heads × 256 head_dim × 2 (K+V) × 2 (bf16) = 81920 bytes/token
        let expected = 40 * 2 * 256 * 2 * 2;
        assert_eq!(kv_bytes_per_token(&m), expected as usize);
    }

    #[test]
    fn moe_validate_budget_realistic_32gb_fits() {
        let st = validate_with_total_ram(1, 8192, &test_meta_qwen35_moe(), 34_359_738_368)
            .expect("32GB host should fit 1 stream × 8K context for MoE");
        assert!(st.soft_limit() > 0);
    }

    #[test]
    fn moe_validate_budget_rejects_overcommit_16gb() {
        // 16 GB - 17 GB weights - 2 GB safety margin = negative budget,
        // any cap must be rejected.
        let err = validate_with_total_ram(1, 4096, &test_meta_qwen35_moe(), 17_179_869_184)
            .expect_err("16GB host cannot fit 17GB MoE weights");
        let msg = format!("{err}");
        assert!(msg.contains("memory budget exceeded"), "msg: {msg}");
    }

    #[test]
    fn startup_budget_without_offload_rejects_large_logical_cap() {
        let meta = test_meta_gemma4_12b();
        let error = validate_with_total_ram(1, 262_144, &meta, 137_438_953_472)
            .expect_err("full-resident 256K cache should exceed budget");
        assert_eq!(error.cap, 262_144);
        assert_eq!(error.resident_cap, 262_144);
        assert_eq!(error.policy, "full_resident");
    }

    #[test]
    fn startup_budget_with_offload_charges_hot_resident_cap() {
        let meta = test_meta_gemma4_12b();
        let policy = KvBudgetPolicy::active_kv_offload(8_192);
        let state = validate_startup_budget_with_policy_for_total_ram(
            1,
            262_144,
            &meta,
            policy,
            137_438_953_472,
        )
        .expect("offload hot window should fit");
        assert_eq!(state.logical_cap(), 262_144);
        assert_eq!(state.resident_cap(), 8_192);
        assert_eq!(state.policy(), policy);
        assert!(state.soft_limit() > 0);
    }

    #[test]
    fn startup_budget_soft_limit_allows_configured_full_resident_cap() {
        let meta = test_meta_qwen35();
        let state = validate_with_total_ram(1, 32_768, &meta, 34_359_738_368)
            .expect("configured full resident cap should fit");
        let configured_bytes = kv_cache_bytes(1, 32_768, &meta);
        assert!(
            state.soft_limit() >= configured_bytes,
            "runtime soft limit must allow the startup-validated full-resident cap"
        );
    }

    #[test]
    fn startup_budget_with_offload_soft_limit_allows_resident_budget() {
        let meta = test_meta_gemma4_12b();
        let policy = KvBudgetPolicy::active_kv_offload(8_192);
        let state = validate_startup_budget_with_policy_for_total_ram(
            1,
            262_144,
            &meta,
            policy,
            137_438_953_472,
        )
        .expect("offload hot window should fit");
        let resident_bytes = kv_cache_bytes(1, 8_192, &meta);
        assert_eq!(state.soft_limit(), resident_bytes);
    }
}
