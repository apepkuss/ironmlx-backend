//! Qwen3.5 / Qwen3-Next GatedDeltaNet — recurrent SSM with delta rule.
//!
//! T5: the `gated_delta_step` metal_kernel (2 variants — no-mask + masked).
//! T6: the `GatedDeltaNet` main struct wiring all components.
//!
//! Mirrors mlx-lm's `_make_gated_delta_kernel(has_mask)` from
//! `/Volumes/Dev/mlx-lm/mlx_lm/models/gated_delta.py:13-115`.
//!
//! Templates: `Dk, Dv, Hk, Hv` (i32), `InT, StT` (Dtype).
//! Grid: `(32, Dv, B * Hv)`; threadgroup: `(32, 4, 1)`.

use std::sync::OnceLock;

use anyhow::anyhow;
use mlx::ops::shape::concatenate;
use mlx::{Array, Dtype, MetalKernel, Shape, StreamOrDevice};

use crate::core::cache::gated_delta::GatedDeltaReplayCapture;
use crate::core::cache::GatedDeltaCache;
use crate::core::Loader;
use crate::nn::{Conv1d, Conv1dConfig, Linear, RmsNormGated};
use crate::Result;

/// Configuration for [`GatedDeltaNet`].
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetConfig {
    pub hidden_size: i32,
    pub num_v_heads: i32,
    pub num_k_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub conv_kernel_size: i32,
    pub rms_norm_eps: f32,
}

impl GatedDeltaNetConfig {
    /// Total K-side dim: `num_k_heads × head_k_dim`. Used to size q/k slices
    /// of the qkv projection output and as the K-side stride in the kernel.
    pub fn key_dim(&self) -> i32 {
        self.num_k_heads * self.head_k_dim
    }

    /// Total V-side dim: `num_v_heads × head_v_dim`. Equals the inner dim of
    /// the V projection and the input dim of `out_proj`.
    pub fn value_dim(&self) -> i32 {
        self.num_v_heads * self.head_v_dim
    }

    /// Total projection-output dim for `in_proj_qkv`:
    /// `key_dim × 2 + value_dim` — i.e. concatenated Q + K + V output.
    /// Also the channel count for the depthwise `conv1d`.
    pub fn conv_dim(&self) -> i32 {
        self.key_dim() * 2 + self.value_dim()
    }
}

/// Qwen3.5 / Qwen3-Next "linear attention" branch — recurrent SSM with
/// delta rule and scalar gating.
///
/// Mirrors mlx-lm's `Qwen3NextGatedDeltaNet`
/// (`/Volumes/Dev/mlx-lm/mlx_lm/models/qwen3_5.py:85-205`). Components:
///
/// - `in_proj_qkv` — Q/K/V input projection feeding the depthwise conv.
/// - `in_proj_z` — value gate projection consumed by `RmsNormGated`.
/// - `in_proj_b` / `in_proj_a` — forget and decay signal projections for
///   the delta-rule recurrence.
/// - `conv1d` — depthwise temporal mixing across the Q/K/V channels (then
///   silu via module-level fused compile cell)
/// - `norm` — `RmsNormGated`: `silu(z) * rms_norm(y)` final mixing
/// - `out_proj` — back to `hidden_size`
/// - `a_log` / `dt_bias` — per-head learned parameters for compute_g
pub struct GatedDeltaNet {
    input_projections: GatedDeltaInputProjections,
    conv1d: Conv1d,
    norm: RmsNormGated,
    out_proj: Linear,
    a_log: Array,   // [num_v_heads]
    dt_bias: Array, // [num_v_heads]
    cfg: GatedDeltaNetConfig,
    kernel_no_mask: OnceLock<MetalKernel>,
    kernel_masked: OnceLock<MetalKernel>,
    kernel_no_state_no_mask: OnceLock<MetalKernel>,
    kernel_no_state_masked: OnceLock<MetalKernel>,
    kernel_zero_state_no_mask: OnceLock<MetalKernel>,
    kernel_zero_state_masked: OnceLock<MetalKernel>,
}

enum GatedDeltaInputProjections {
    Separate {
        qkv: Linear,
        z: Linear,
        b: Linear,
        a: Linear,
    },
    Fused {
        projection: Linear,
        qkv: Linear,
        z: Linear,
        b: Linear,
        a: Linear,
        qkv_width: i32,
        z_width: i32,
        b_width: i32,
    },
}

impl GatedDeltaNet {
    /// Production constructor: load all weight tensors + a_log + dt_bias.
    pub fn from_loader(loader: &Loader, prefix: &str, cfg: GatedDeltaNetConfig) -> Result<Self> {
        Self::from_loader_impl(loader, prefix, cfg, false)
    }

    pub(crate) fn from_loader_dflash2(
        loader: &Loader,
        prefix: &str,
        cfg: GatedDeltaNetConfig,
    ) -> Result<Self> {
        Self::from_loader_impl(loader, prefix, cfg, true)
    }

    fn from_loader_impl(
        loader: &Loader,
        prefix: &str,
        cfg: GatedDeltaNetConfig,
        fuse_dflash2: bool,
    ) -> Result<Self> {
        let in_proj_qkv = Linear::from_loader(loader, &format!("{prefix}.in_proj_qkv"))?;
        let in_proj_z = Linear::from_loader(loader, &format!("{prefix}.in_proj_z"))?;
        let in_proj_b = Linear::from_loader(loader, &format!("{prefix}.in_proj_b"))?;
        let in_proj_a = Linear::from_loader(loader, &format!("{prefix}.in_proj_a"))?;
        let input_projections = if fuse_dflash2 {
            let output_widths = [
                in_proj_qkv.out_features(),
                in_proj_z.out_features(),
                in_proj_b.out_features(),
                in_proj_a.out_features(),
            ];
            let projection = Linear::fuse_quantized_outputs(
                &[&in_proj_qkv, &in_proj_z, &in_proj_b, &in_proj_a],
                "DFlash2 fused GDN input projections",
            )?;
            let mut separate = projection
                .split_quantized_outputs(&output_widths, "DFlash2 split GDN input projections")?;
            GatedDeltaInputProjections::Fused {
                projection,
                qkv: separate.remove(0),
                z: separate.remove(0),
                b: separate.remove(0),
                a: separate.remove(0),
                qkv_width: i32::try_from(output_widths[0])?,
                z_width: i32::try_from(output_widths[1])?,
                b_width: i32::try_from(output_widths[2])?,
            }
        } else {
            GatedDeltaInputProjections::Separate {
                qkv: in_proj_qkv,
                z: in_proj_z,
                b: in_proj_b,
                a: in_proj_a,
            }
        };
        let conv1d_cfg = Conv1dConfig {
            in_channels: cfg.conv_dim(),
            out_channels: cfg.conv_dim(),
            kernel_size: cfg.conv_kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: cfg.conv_dim(),
        };
        let conv1d = Conv1d::from_loader(loader, &format!("{prefix}.conv1d"), conv1d_cfg)?;
        let norm = RmsNormGated::from_loader(loader, &format!("{prefix}.norm"), cfg.rms_norm_eps)?;
        let out_proj = Linear::from_loader(loader, &format!("{prefix}.out_proj"))?;
        let a_log = loader.tensor(&format!("{prefix}.A_log"))?.clone();
        let dt_bias = loader.tensor(&format!("{prefix}.dt_bias"))?.clone();

        Ok(Self {
            input_projections,
            conv1d,
            norm,
            out_proj,
            a_log,
            dt_bias,
            cfg,
            kernel_no_mask: OnceLock::new(),
            kernel_masked: OnceLock::new(),
            kernel_no_state_no_mask: OnceLock::new(),
            kernel_no_state_masked: OnceLock::new(),
            kernel_zero_state_no_mask: OnceLock::new(),
            kernel_zero_state_masked: OnceLock::new(),
        })
    }

    /// Test/composition seam: build from pre-built nn building blocks.
    ///
    /// `pub` (not `pub(crate)`) so integration tests in `ironmlx/tests/` can use it.
    /// Hidden from rustdoc via `#[doc(hidden)]`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        in_proj_qkv: Linear,
        in_proj_z: Linear,
        in_proj_b: Linear,
        in_proj_a: Linear,
        conv1d: Conv1d,
        norm: RmsNormGated,
        out_proj: Linear,
        a_log: Array,
        dt_bias: Array,
        cfg: GatedDeltaNetConfig,
    ) -> Self {
        Self {
            input_projections: GatedDeltaInputProjections::Separate {
                qkv: in_proj_qkv,
                z: in_proj_z,
                b: in_proj_b,
                a: in_proj_a,
            },
            conv1d,
            norm,
            out_proj,
            a_log,
            dt_bias,
            cfg,
            kernel_no_mask: OnceLock::new(),
            kernel_masked: OnceLock::new(),
            kernel_no_state_no_mask: OnceLock::new(),
            kernel_no_state_masked: OnceLock::new(),
            kernel_zero_state_no_mask: OnceLock::new(),
            kernel_zero_state_masked: OnceLock::new(),
        }
    }

    pub fn config(&self) -> &GatedDeltaNetConfig {
        &self.cfg
    }

    /// Forward pass with default stream.
    pub fn forward(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut GatedDeltaCache>,
    ) -> Result<Array> {
        // Non-decoder callers (CLI / standalone tests) — pass -1 per spec § 2.5a.
        self.forward_on(x, mask, None, cache, (), -1)
    }

    /// Stream-targeted forward — Qwen3-Next gated delta net algorithm.
    ///
    /// 8 steps:
    ///   1. project qkv, z, a, b
    ///   2. conv1d + silu (with conv_state from cache prepended; cache update)
    ///   3. split + reshape per-head
    ///   4. q/k rms_norm (no weight) + scale
    ///   5. compute_g via mlx::compile
    ///   6. beta = sigmoid(b)
    ///   7. dispatch gated_delta_step kernel + update recurrent cache + advance offset
    ///   8. RmsNormGated(y, z) + reshape + out_proj
    #[allow(clippy::too_many_arguments)]
    pub fn forward_on(
        &self,
        x: &Array,
        mask: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        mut cache: Option<&mut GatedDeltaCache>,
        target: impl Into<StreamOrDevice>,
        layer_idx: i32,
    ) -> Result<Array> {
        let target = target.into();
        // Signature parity with other layer-aware forward paths.
        let _ = layer_idx;

        // Pre-flight validation. Match P3b1 Mrope's "explicit bounds > trust caller"
        // pattern — surface common misuses at the source rather than as a downstream
        // shape/MTL dispatch error.
        let dims_borrow = x.shape();
        let dims = dims_borrow.as_slice();
        if dims.len() != 3 {
            return Err(anyhow!(
                "GatedDeltaNet::forward: x must be rank-3 [B, S, hidden]; got rank {}",
                dims.len()
            ));
        }
        if dims[2] != self.cfg.hidden_size {
            return Err(anyhow!(
                "GatedDeltaNet::forward: x.last_dim={} != hidden_size={}",
                dims[2],
                self.cfg.hidden_size
            ));
        }
        if self.cfg.head_k_dim < 32 || self.cfg.head_k_dim % 32 != 0 {
            return Err(anyhow!(
                "GatedDeltaNet::forward: head_k_dim={} must be a positive multiple of 32 \
                 (Metal kernel requires `n_per_t = Dk/32 >= 1` and full simdgroup coverage)",
                self.cfg.head_k_dim
            ));
        }
        if self.cfg.num_k_heads == 0 || self.cfg.num_v_heads % self.cfg.num_k_heads != 0 {
            return Err(anyhow!(
                "GatedDeltaNet::forward: num_v_heads ({}) must be divisible by num_k_heads ({}) \
                 — kernel uses `hk_idx = hv_idx / (Hv/Hk)` for GQA indexing",
                self.cfg.num_v_heads,
                self.cfg.num_k_heads
            ));
        }

        let batch = dims[0];
        let seq = dims[1];
        if super::batch_stable_qmm::context_is_armed() && batch > 1 {
            let cache = cache
                .as_deref_mut()
                .ok_or_else(|| anyhow!("batch-stable GatedDelta prefill requires a cache"))?;
            return self.forward_batch_rows_isolated_on(
                x,
                mask,
                per_row_lens,
                cache,
                target,
                layer_idx,
            );
        }
        if super::sequence_stable_gated_delta::is_armed() && seq > 1 {
            return self.forward_sequence_stable_on(
                x,
                mask,
                per_row_lens,
                cache,
                target,
                layer_idx,
            );
        }

        // Step 1: reference-equivalent input projections. DFlash2 reuses
        // these tensors to rebuild an accepted recurrent prefix after a
        // rejection. Quantized products can otherwise vary with sequence
        // width, so isolate only the GDN state-input projections while that
        // capture is active; larger MLP/attention projections stay batched.
        let capture_active = cache
            .as_deref()
            .is_some_and(GatedDeltaCache::speculative_prefix_capture_active);
        let defer_captured_state_commit = capture_active
            && super::position_stable_qmm::exact_affine8_b4_q2_is_armed()
            && seq == 2;
        let ((qkv, z), (b, a)) = {
            let _stable_linear = capture_active.then(super::position_stable_linear::scope);
            let _stable_qmm = capture_active.then(super::position_stable_qmm::scope);

            // Step 1a: in_proj_qkv + in_proj_z, then mask-zero qkv at pad
            // positions. The mask multiply stays bundled with qkv projection
            // because it is the immediate downstream consumer before conv1d.
            //
            // Mask-zero rationale (preserved from the prior standalone block):
            // The conv1d is temporal — its output at real-token position t uses
            // input positions `t-(k-1)..t` as history. Under right-padded batched
            // prefill, real qkv occupies positions `[0, L_i)` and the trailing
            // `[L_i, max_len)` positions are pad; conv1d output AT real positions
            // (t < L_i) only consumes earlier real positions (causal kernel), so
            // it stays clean even without zeroing pad qkv. However, conv1d output
            // AT pad positions reads back into the real-tail (positions
            // `[L_i - (k-1), L_i)`), and the kernel post-write of conv_state then
            // captures those pad-slot outputs — so we zero pad qkv up front to
            // keep pad-slot conv1d output benign and avoid leaking pad embeddings
            // (which are non-zero garbage from in_proj_qkv) into the cache
            // update path's per-row slice.
            //
            // The gated_delta_step kernel's per-token mask only skips compute at
            // pad positions; it does not undo conv1d contamination of real
            // positions. Zeroing qkv at pad positions before conv1d gives real
            // tokens the same zero-history as per-stream forward_on.
            //
            // The same argument applies to `z` (used in RmsNormGated at output);
            // however, `z` is only consumed at REAL positions (gated_delta_step
            // emits zero at pad positions), so pad-position `z` values are
            // discarded anyway. We zero `qkv` only.
            let (qkv, z, b, a) = match &self.input_projections {
                GatedDeltaInputProjections::Separate { qkv, z, b, a } => (
                    qkv.forward_on(x, target)?,
                    z.forward_on(x, target)?,
                    b.forward_on(x, target)?,
                    a.forward_on(x, target)?,
                ),
                GatedDeltaInputProjections::Fused {
                    projection,
                    qkv,
                    z,
                    b,
                    a,
                    qkv_width,
                    z_width,
                    b_width,
                } => {
                    if super::product_stable_qmm::is_armed() {
                        let output = projection.forward_on(x, target)?;
                        let mut parts = mlx::ops::shape::split_at_on(
                            &output,
                            &[
                                *qkv_width,
                                *qkv_width + *z_width,
                                *qkv_width + *z_width + *b_width,
                            ],
                            -1,
                            target,
                        )?;
                        if parts.len() != 4 {
                            return Err(anyhow!(
                                "DFlash2 fused GDN input projection returned {} parts",
                                parts.len()
                            ));
                        }
                        (
                            parts.remove(0),
                            parts.remove(0),
                            parts.remove(0),
                            parts.remove(0),
                        )
                    } else {
                        (
                            qkv.forward_on(x, target)?,
                            z.forward_on(x, target)?,
                            b.forward_on(x, target)?,
                            a.forward_on(x, target)?,
                        )
                    }
                }
            };
            let qkv = {
                if let Some(m) = mask {
                    let m_dtype = mlx::ops::cast::astype(m, qkv.dtype())?;
                    let m_broadcast = m_dtype.reshape_on((batch, seq, 1), target)?;
                    &qkv * &m_broadcast
                } else {
                    qkv
                }
            };
            ((qkv, z), (b, a))
        };

        // Step 2a: prepend conv_state
        let conv_input = {
            match cache.as_mut() {
                Some(c) => concatenate(&[c.conv_state(), &qkv], 1)?,
                None => {
                    let zeros = Array::zeros(
                        (batch, self.cfg.conv_kernel_size - 1, self.cfg.conv_dim()),
                        qkv.dtype(),
                    )?;
                    concatenate(&[&zeros, &qkv], 1)?
                }
            }
        };

        // Step 2b: conv1d + silu
        let conv_out = {
            let conv_out = self.conv1d.forward_on(&conv_input, target)?;
            let conv_sig = conv_out.sigmoid()?;
            &conv_out * &conv_sig
        };

        // Step 2c: update conv_state cache.
        //
        // The new conv_state for the next call must capture the last
        // `n_keep = kernel_size - 1` tokens of each row's REAL input. Under
        // right-padded batched prefill the real qkv occupies positions
        // `[k-1, k-1 + L_i)` of conv_input (= old conv_state prepended +
        // qkv with pad zeroed). For row i the real-tail window therefore
        // sits at `[k-1 + L_i - n_keep, k-1 + L_i) == [L_i, L_i + n_keep)`
        // of conv_input — uniform-length and B=1 cases collapse to the
        // last n_keep positions of conv_input (matches pre-right-pad
        // behaviour).
        //
        // When `per_row_lens` is `None` (single-stream / non-batched), we
        // fall back to the simple "last n_keep positions" slice.
        if let Some(c) = cache.as_mut().filter(|_| !defer_captured_state_commit) {
            let n_keep = self.cfg.conv_kernel_size - 1;
            let conv_input_dims = conv_input.shape();
            let total_len = conv_input_dims.as_slice()[1];
            let conv_dim = self.cfg.conv_dim();
            let new_conv_state = match per_row_lens {
                Some(lens) if batch > 1 && !lens.iter().all(|&l| l + n_keep == total_len) => {
                    // Per-row real-tail window starts at position `lens[i]`
                    // in conv_input and spans `n_keep` rows. Express as a
                    // single `take_along_axis` over axis 1 with index tensor
                    // `[B, n_keep, 1]` (broadcasts to [B, n_keep, conv_dim]).
                    if lens.len() as i32 != batch {
                        return Err(anyhow!(
                            "GatedDeltaNet::forward_on: per_row_lens.len()={} != batch={}",
                            lens.len(),
                            batch
                        ));
                    }
                    let mut idx_flat: Vec<u32> = Vec::with_capacity((batch * n_keep) as usize);
                    for &l in lens {
                        for j in 0..n_keep {
                            idx_flat.push((l + j) as u32);
                        }
                    }
                    let idx: Array = (&idx_flat[..], &[batch, n_keep, 1_i32][..])
                        .try_into()
                        .map_err(|e| {
                            anyhow!("GatedDeltaNet::forward_on: idx try_into Array failed: {e:?}")
                        })?;
                    mlx::ops::indexing::take_along_axis_on(&conv_input, &idx, 1, target)?
                }
                _ => mlx::ops::indexing::slice(
                    &conv_input,
                    vec![0_i32, total_len - n_keep, 0].as_slice(),
                    vec![batch, total_len, conv_dim].as_slice(),
                )?,
            };
            c.update_conv(new_conv_state);
        }

        // Step 3: split + reshape per-head
        // conv_out shape: [B, S, conv_dim] = [B, S, key_dim*2 + value_dim]
        // Split at [key_dim, 2*key_dim] → 3 segments [B, S, key_dim], [B, S, key_dim], [B, S, value_dim]
        let (q_per_head, k_per_head, v_per_head) = {
            let split_at = vec![self.cfg.key_dim(), 2 * self.cfg.key_dim()];
            let parts = mlx::ops::shape::split_at_on(&conv_out, &split_at, -1, target)?;
            let q_flat = &parts[0]; // [B, S, num_k_heads * head_k_dim]
            let k_flat = &parts[1]; // [B, S, num_k_heads * head_k_dim]
            let v_flat = &parts[2]; // [B, S, num_v_heads * head_v_dim]

            let q_per_head = q_flat.reshape_on(
                (batch, seq, self.cfg.num_k_heads, self.cfg.head_k_dim),
                target,
            )?;
            let k_per_head = k_flat.reshape_on(
                (batch, seq, self.cfg.num_k_heads, self.cfg.head_k_dim),
                target,
            )?;
            let v_per_head = v_flat.reshape_on(
                (batch, seq, self.cfg.num_v_heads, self.cfg.head_v_dim),
                target,
            )?;
            (q_per_head, k_per_head, v_per_head)
        };

        // Step 4: q/k rms_norm (no weight)
        let (q_scaled, k_scaled) = {
            let inv_scale = 1.0_f32 / (self.cfg.head_k_dim as f32).sqrt();
            let q_normed = mlx::fast::rms_norm_on(&q_per_head, None, 1e-6, target)?;
            let q_scaled = &q_normed * (inv_scale * inv_scale); // panic-on-err, no `?`
            let k_normed = mlx::fast::rms_norm_on(&k_per_head, None, 1e-6, target)?;
            let k_scaled = &k_normed * inv_scale; // panic-on-err, no `?`
            (q_scaled, k_scaled)
        };

        // Step 5: compute_g = exp(-exp(A_log) * softplus(a + dt_bias))
        // softplus stabilised: where(x > 20, x, log(1 + exp(x)))
        let g = {
            let x_sp = &a + &self.dt_bias;
            let twenty: Array = (&[20.0_f32][..], ()).try_into()?;
            let zeros = a.zeros_like()?;
            let safe = zeros.logaddexp(&x_sp)?;
            let cond = x_sp.greater(&twenty)?;
            let sp = cond.where_(&x_sp, &safe)?;
            let a_log_f32 = mlx::ops::cast::astype(&self.a_log, Dtype::Float32)?;
            let exp_alog = a_log_f32.exp()?;
            let neg_exp_alog = mlx::ops::binary::negative(&exp_alog)?;
            let inner = &neg_exp_alog * &sp;
            inner.exp()?
        };

        // Step 6: beta = sigmoid(b)
        let beta = b.sigmoid_on(target)?;

        let y = {
            // Step 7a: build/get the appropriate kernel. The initial
            // prefill chunk has a logically all-zero recurrent state; use
            // a zero-state variant so the kernel does not read or
            // materialize a large fp32 zero buffer.
            let zero_state = match cache.as_deref() {
                Some(c) => c.offsets().iter().all(|&o| o == 0),
                None => true,
            };
            let omit_state_output = defer_captured_state_commit && !zero_state;
            let kernel = if omit_state_output {
                if mask.is_some() {
                    self.kernel_no_state_masked.get_or_init(|| {
                        build_gated_delta_no_state_kernel(true)
                            .expect("build no-state masked kernel")
                    })
                } else {
                    self.kernel_no_state_no_mask.get_or_init(|| {
                        build_gated_delta_no_state_kernel(false).expect("build no-state kernel")
                    })
                }
            } else {
                match (mask.is_some(), zero_state) {
                    (true, true) => self.kernel_zero_state_masked.get_or_init(|| {
                        build_gated_delta_zero_state_kernel(true)
                            .expect("build zero-state masked kernel")
                    }),
                    (false, true) => self.kernel_zero_state_no_mask.get_or_init(|| {
                        build_gated_delta_zero_state_kernel(false)
                            .expect("build zero-state no-mask kernel")
                    }),
                    (true, false) => self.kernel_masked.get_or_init(|| {
                        build_gated_delta_kernel(true).expect("build masked kernel")
                    }),
                    (false, false) => self.kernel_no_mask.get_or_init(|| {
                        build_gated_delta_kernel(false).expect("build no-mask kernel")
                    }),
                }
            };

            // Step 7b: get state_in only after the stream has advanced.
            // Note: `Array::clone()` is cheap (Arc-share refcount inc on
            // `array_desc_`, not a deep memory copy); the regular kernel
            // dispatch needs an `&Array`, and the cache must keep its slot
            // for `update_recurrent` later.
            let state_in = if zero_state {
                None
            } else {
                Some(
                    cache
                        .as_deref()
                        .expect("nonzero GDN state requires cache to exist")
                        .recurrent_state()
                        .clone(),
                )
            };

            // Step 7c: T as 0-dim int32 array.
            let t_arr: Array = (&[seq][..], ()).try_into()?;

            let in_dtype = x.dtype();
            let st_dtype = Dtype::Float32;
            let y_shape = Shape::from(vec![batch, seq, self.cfg.num_v_heads, self.cfg.head_v_dim]);
            let state_shape = Shape::from(vec![
                batch,
                self.cfg.num_v_heads,
                self.cfg.head_v_dim,
                self.cfg.head_k_dim,
            ]);

            // Step 7d: dispatch
            let mut kernel_inputs: Vec<&Array> = vec![&q_scaled, &k_scaled, &v_per_head, &g, &beta];
            if let Some(state_in) = state_in.as_ref() {
                kernel_inputs.push(state_in);
            }
            kernel_inputs.push(&t_arr);
            if let Some(m) = mask {
                kernel_inputs.push(m);
            }

            let mut output_shapes = vec![y_shape];
            let mut output_dtypes = vec![in_dtype];
            if !omit_state_output {
                output_shapes.push(state_shape);
                output_dtypes.push(st_dtype);
            }
            let mut outputs = kernel
                .dispatch_builder()
                .inputs(&kernel_inputs)
                .output_shapes(&output_shapes)
                .output_dtypes(&output_dtypes)
                .grid(32, self.cfg.head_v_dim, batch * self.cfg.num_v_heads)
                .threadgroup(32, 4, 1)
                .template_int("Dk", self.cfg.head_k_dim)
                .template_int("Dv", self.cfg.head_v_dim)
                .template_int("Hk", self.cfg.num_k_heads)
                .template_int("Hv", self.cfg.num_v_heads)
                .template_dtype("InT", in_dtype)
                .template_dtype("StT", st_dtype)
                .stream(target)
                .dispatch()?;

            let y = outputs.take_at(0)?; // [B, S, Hv, Dv]
            let new_state = (!omit_state_output)
                .then(|| outputs.take_at(0))
                .transpose()?; // [B, Hv, Dv, Dk]

            // Step 7e: update cache recurrent_state, advance offset
            if let (Some(c), Some(new_state)) = (cache.as_mut(), new_state) {
                c.update_recurrent(new_state);
                let lens_owned: Vec<i32>;
                let lens_ref: &[i32] = match per_row_lens {
                    Some(l) => l,
                    None => {
                        // Non-batched single-stream caller: lockstep-equivalent uniform.
                        lens_owned = vec![seq; batch as usize];
                        &lens_owned
                    }
                };
                c.advance(lens_ref)?;
            }
            y
        };

        if let Some(c) = cache {
            if c.speculative_prefix_capture_active() {
                c.record_speculative_replay(
                    &conv_input,
                    &q_scaled,
                    &k_scaled,
                    &v_per_head,
                    &g,
                    &beta,
                    mask,
                    target,
                )?;
            }
        }

        // Step 8: RmsNormGated(y, z) + reshape + out_proj
        let out = {
            let z_per_head = z.reshape_on(
                (batch, seq, self.cfg.num_v_heads, self.cfg.head_v_dim),
                target,
            )?;
            let normed = self.norm.forward_on(&y, Some(&z_per_head), target)?;
            let normed_flat = normed.reshape_on((batch, seq, self.cfg.value_dim()), target)?;
            self.out_proj.forward_on(&normed_flat, target)?
        };

        Ok(out)
    }

    pub(crate) fn restore_speculative_prefix_on(
        &self,
        cache: &mut GatedDeltaCache,
        accepted_len: usize,
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        if accepted_len == 0 {
            return Err(anyhow!(
                "GatedDeltaNet speculative accepted prefix cannot be empty"
            ));
        }
        let target = target.into();
        let capture = cache.take_speculative_replay()?;
        let sequence = usize::try_from(capture.q.shape().as_slice()[1])?;
        if accepted_len > sequence {
            return Err(anyhow!(
                "GatedDeltaNet accepted prefix {accepted_len} exceeds captured sequence {sequence}"
            ));
        }
        if capture.prefix_states.len() == sequence {
            cache.restore(&capture.prefix_states[accepted_len - 1])?;
            return Ok(());
        }
        cache.restore(&capture.base)?;

        let accepted_len_i32 = i32::try_from(accepted_len)?;
        let q = slice_sequence_prefix_on(&capture.q, accepted_len_i32, target)?;
        let k = slice_sequence_prefix_on(&capture.k, accepted_len_i32, target)?;
        let v = slice_sequence_prefix_on(&capture.v, accepted_len_i32, target)?;
        let g = slice_sequence_prefix_on(&capture.g, accepted_len_i32, target)?;
        let beta = slice_sequence_prefix_on(&capture.beta, accepted_len_i32, target)?;
        let mask = capture
            .mask
            .as_ref()
            .map(|mask| slice_sequence_prefix_on(mask, accepted_len_i32, target))
            .transpose()?;

        let batch = q.shape().as_slice()[0];
        let kernel = if mask.is_some() {
            self.kernel_masked
                .get_or_init(|| build_gated_delta_kernel(true).expect("build masked kernel"))
        } else {
            self.kernel_no_mask
                .get_or_init(|| build_gated_delta_kernel(false).expect("build no-mask kernel"))
        };
        let t_arr: Array = (&[accepted_len_i32][..], ()).try_into()?;
        let mut kernel_inputs = vec![&q, &k, &v, &g, &beta, cache.recurrent_state(), &t_arr];
        if let Some(mask) = mask.as_ref() {
            kernel_inputs.push(mask);
        }
        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&kernel_inputs)
            .output_shapes(&[
                Shape::from((
                    batch,
                    accepted_len_i32,
                    self.cfg.num_v_heads,
                    self.cfg.head_v_dim,
                )),
                Shape::from((
                    batch,
                    self.cfg.num_v_heads,
                    self.cfg.head_v_dim,
                    self.cfg.head_k_dim,
                )),
            ])
            .output_dtypes(&[q.dtype(), Dtype::Float32])
            .grid(32, self.cfg.head_v_dim, batch * self.cfg.num_v_heads)
            .threadgroup(32, 4, 1)
            .template_int("Dk", self.cfg.head_k_dim)
            .template_int("Dv", self.cfg.head_v_dim)
            .template_int("Hk", self.cfg.num_k_heads)
            .template_int("Hv", self.cfg.num_v_heads)
            .template_dtype("InT", q.dtype())
            .template_dtype("StT", Dtype::Float32)
            .stream(target)
            .dispatch()?;
        let _unused_output = outputs.take_at(0)?;
        let new_state = outputs.take_at(0)?;

        let keep = self.cfg.conv_kernel_size - 1;
        let conv_shape = capture.conv_input.shape();
        let conv_dims = conv_shape.as_slice();
        let new_conv_state = mlx::ops::indexing::slice_strided_on(
            &capture.conv_input,
            &[0_i32, accepted_len_i32, 0][..],
            &[batch, accepted_len_i32 + keep, conv_dims[2]][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        cache.update_conv(new_conv_state);
        cache.update_recurrent(new_state);
        cache.advance(&vec![accepted_len_i32; batch as usize])?;
        Ok(())
    }

    pub(crate) fn restore_speculative_prefix_rows_on(
        &self,
        cache: &mut GatedDeltaCache,
        accepted_lens: &[usize],
        target: impl Into<StreamOrDevice>,
    ) -> Result<()> {
        let target = target.into();
        if accepted_lens.is_empty() {
            return Err(anyhow!(
                "GatedDeltaNet accepted prefix rows cannot be empty"
            ));
        }
        let capture = cache.take_speculative_replay()?;
        let sequence = usize::try_from(capture.q.shape().as_slice()[1])?;
        if accepted_lens.len() != cache.offsets().len() {
            return Err(anyhow!(
                "GatedDeltaNet accepted prefix rows {} != cache batch {}",
                accepted_lens.len(),
                cache.offsets().len()
            ));
        }
        if accepted_lens.iter().any(|&accepted| accepted > sequence) {
            return Err(anyhow!(
                "GatedDeltaNet accepted prefix exceeds captured sequence {sequence}"
            ));
        }
        if accepted_lens.contains(&0) {
            for (row, &accepted_len) in accepted_lens.iter().enumerate() {
                let (conv_state, recurrent_state, cached_len) = if accepted_len == 0 {
                    capture.base.prefix_state_for_row_on(row, target)?
                } else if capture.prefix_states.len() == sequence {
                    capture.prefix_states[accepted_len - 1].prefix_state_for_row_on(row, target)?
                } else {
                    self.replay_captured_prefix_row_on(&capture, row, accepted_len, target)?
                };
                cache.restore_prefix_state_for_row_on(
                    &conv_state,
                    &recurrent_state,
                    row,
                    cached_len,
                    target,
                )?;
            }
            return Ok(());
        }
        if capture.prefix_states.len() != sequence {
            let batch = i32::try_from(accepted_lens.len())?;
            let sequence_i32 = i32::try_from(sequence)?;
            let accepted_mask_values = accepted_lens
                .iter()
                .flat_map(|&accepted| (0..sequence).map(move |position| position < accepted))
                .collect::<Vec<_>>();
            let accepted_mask: Array =
                (accepted_mask_values.as_slice(), &[batch, sequence_i32][..]).try_into()?;
            let replay_mask = if let Some(mask) = capture.mask.as_ref() {
                let disabled = Array::zeros((batch, sequence_i32), Dtype::Bool)?;
                mlx::ops::indexing::where_on(mask, &accepted_mask, &disabled, target)?
            } else {
                accepted_mask
            };
            cache.restore(&capture.base)?;
            let kernel = self
                .kernel_masked
                .get_or_init(|| build_gated_delta_kernel(true).expect("build masked kernel"));
            let t_arr: Array = (&[sequence_i32][..], ()).try_into()?;
            let mut outputs = kernel
                .dispatch_builder()
                .inputs(&[
                    &capture.q,
                    &capture.k,
                    &capture.v,
                    &capture.g,
                    &capture.beta,
                    cache.recurrent_state(),
                    &t_arr,
                    &replay_mask,
                ])
                .output_shapes(&[
                    Shape::from((
                        batch,
                        sequence_i32,
                        self.cfg.num_v_heads,
                        self.cfg.head_v_dim,
                    )),
                    Shape::from((
                        batch,
                        self.cfg.num_v_heads,
                        self.cfg.head_v_dim,
                        self.cfg.head_k_dim,
                    )),
                ])
                .output_dtypes(&[capture.q.dtype(), Dtype::Float32])
                .grid(32, self.cfg.head_v_dim, batch * self.cfg.num_v_heads)
                .threadgroup(32, 4, 1)
                .template_int("Dk", self.cfg.head_k_dim)
                .template_int("Dv", self.cfg.head_v_dim)
                .template_int("Hk", self.cfg.num_k_heads)
                .template_int("Hv", self.cfg.num_v_heads)
                .template_dtype("InT", capture.q.dtype())
                .template_dtype("StT", Dtype::Float32)
                .stream(target)
                .dispatch()?;
            let _unused_output = outputs.take_at(0)?;
            let new_state = outputs.take_at(0)?;

            let keep = self.cfg.conv_kernel_size - 1;
            let conv_dims = capture.conv_input.shape();
            let conv_dims = conv_dims.as_slice();
            let mut conv_rows = Vec::with_capacity(accepted_lens.len());
            for (row, &accepted_len) in accepted_lens.iter().enumerate() {
                let accepted_len = i32::try_from(accepted_len)?;
                let row = i32::try_from(row)?;
                conv_rows.push(mlx::ops::indexing::slice_strided_on(
                    &capture.conv_input,
                    &[row, accepted_len, 0][..],
                    &[row + 1, accepted_len + keep, conv_dims[2]][..],
                    &[1_i32, 1, 1][..],
                    target,
                )?);
            }
            let conv_row_refs = conv_rows.iter().collect::<Vec<_>>();
            let new_conv_state = mlx::ops::shape::concatenate_on(&conv_row_refs, 0, target)?;
            let advances = accepted_lens
                .iter()
                .map(|&accepted| i32::try_from(accepted))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            cache.update_conv(new_conv_state);
            cache.update_recurrent(new_state);
            cache.advance(&advances)?;
            return Ok(());
        }
        for (row, &accepted_len) in accepted_lens.iter().enumerate() {
            let snapshot = &capture.prefix_states[accepted_len - 1];
            let (conv_state, recurrent_state, cached_len) =
                snapshot.prefix_state_for_row_on(row, target)?;
            cache.restore_prefix_state_for_row_on(
                &conv_state,
                &recurrent_state,
                row,
                cached_len,
                target,
            )?;
        }
        Ok(())
    }

    fn replay_captured_prefix_row_on(
        &self,
        capture: &GatedDeltaReplayCapture,
        row: usize,
        accepted_len: usize,
        target: StreamOrDevice,
    ) -> Result<(Array, Array, i32)> {
        let row_i32 = i32::try_from(row)?;
        let accepted_len_i32 = i32::try_from(accepted_len)?;
        let q = slice_sequence_prefix_on(
            &slice_batch_row_or_broadcast(&capture.q, row_i32, target)?,
            accepted_len_i32,
            target,
        )?;
        let k = slice_sequence_prefix_on(
            &slice_batch_row_or_broadcast(&capture.k, row_i32, target)?,
            accepted_len_i32,
            target,
        )?;
        let v = slice_sequence_prefix_on(
            &slice_batch_row_or_broadcast(&capture.v, row_i32, target)?,
            accepted_len_i32,
            target,
        )?;
        let g = slice_sequence_prefix_on(
            &slice_batch_row_or_broadcast(&capture.g, row_i32, target)?,
            accepted_len_i32,
            target,
        )?;
        let beta = slice_sequence_prefix_on(
            &slice_batch_row_or_broadcast(&capture.beta, row_i32, target)?,
            accepted_len_i32,
            target,
        )?;
        let mask = capture
            .mask
            .as_ref()
            .map(|mask| {
                let row_mask = slice_batch_row_or_broadcast(mask, row_i32, target)?;
                slice_sequence_prefix_on(&row_mask, accepted_len_i32, target)
            })
            .transpose()?;
        let (_, base_recurrent_state, base_cached_len) =
            capture.base.prefix_state_for_row_on(row, target)?;

        let kernel = if mask.is_some() {
            self.kernel_masked
                .get_or_init(|| build_gated_delta_kernel(true).expect("build masked kernel"))
        } else {
            self.kernel_no_mask
                .get_or_init(|| build_gated_delta_kernel(false).expect("build no-mask kernel"))
        };
        let t_arr: Array = (&[accepted_len_i32][..], ()).try_into()?;
        let mut kernel_inputs = vec![&q, &k, &v, &g, &beta, &base_recurrent_state, &t_arr];
        if let Some(mask) = mask.as_ref() {
            kernel_inputs.push(mask);
        }
        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&kernel_inputs)
            .output_shapes(&[
                Shape::from((
                    1_i32,
                    accepted_len_i32,
                    self.cfg.num_v_heads,
                    self.cfg.head_v_dim,
                )),
                Shape::from((
                    1_i32,
                    self.cfg.num_v_heads,
                    self.cfg.head_v_dim,
                    self.cfg.head_k_dim,
                )),
            ])
            .output_dtypes(&[q.dtype(), Dtype::Float32])
            .grid(32, self.cfg.head_v_dim, self.cfg.num_v_heads)
            .threadgroup(32, 4, 1)
            .template_int("Dk", self.cfg.head_k_dim)
            .template_int("Dv", self.cfg.head_v_dim)
            .template_int("Hk", self.cfg.num_k_heads)
            .template_int("Hv", self.cfg.num_v_heads)
            .template_dtype("InT", q.dtype())
            .template_dtype("StT", Dtype::Float32)
            .stream(target)
            .dispatch()?;
        let _unused_output = outputs.take_at(0)?;
        let recurrent_state = outputs.take_at(0)?;

        let conv_input = slice_batch_row_or_broadcast(&capture.conv_input, row_i32, target)?;
        let conv_dims = conv_input.shape();
        let conv_dims = conv_dims.as_slice();
        let keep = self.cfg.conv_kernel_size - 1;
        let conv_state = mlx::ops::indexing::slice_strided_on(
            &conv_input,
            &[0_i32, accepted_len_i32, 0][..],
            &[1_i32, accepted_len_i32 + keep, conv_dims[2]][..],
            &[1_i32, 1, 1][..],
            target,
        )?;
        let cached_len = base_cached_len
            .checked_add(accepted_len_i32)
            .ok_or_else(|| anyhow!("GatedDeltaNet accepted prefix offset overflow"))?;
        Ok((conv_state, recurrent_state, cached_len))
    }

    /// Preserve the cache-bearing B1 numerical morphology while the surrounding
    /// decoder layer remains batched. DFlash2 batched prefill arms this route:
    /// the ordinary batched GatedDelta state is close but not bit-exact for
    /// affine4 or affine8, and later tensor verification can amplify that
    /// difference into a token mismatch.
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_rows_isolated_on(
        &self,
        x: &Array,
        mask: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        cache: &mut GatedDeltaCache,
        target: StreamOrDevice,
        layer_idx: i32,
    ) -> Result<Array> {
        let shape = x.shape();
        let [batch, sequence, hidden] = *<&[i32; 3]>::try_from(shape.as_slice())
            .map_err(|_| anyhow!("batch-stable GatedDelta requires [B,S,H]"))?;
        let mut outputs = Vec::with_capacity(batch as usize);
        for row in 0..batch {
            let row_x = mlx::ops::indexing::slice_strided_on(
                x,
                &[row, 0_i32, 0][..],
                &[row + 1, sequence, hidden][..],
                &[1_i32, 1, 1][..],
                target,
            )?;
            let row_mask = mask
                .map(|mask| slice_batch_row_or_broadcast(mask, row, target))
                .transpose()?;
            let row_lens = per_row_lens.map(|lens| [lens[row as usize]]);
            let mut row_cache = GatedDeltaCache::new_with_cap(
                1,
                self.cfg.conv_kernel_size,
                self.cfg.conv_dim(),
                self.cfg.num_v_heads,
                self.cfg.head_v_dim,
                self.cfg.head_k_dim,
                x.dtype(),
                cache.cap(),
            )?;
            row_cache.adopt_row_from(cache, 0, row as usize)?;
            let row_output = self.forward_on(
                &row_x,
                row_mask.as_ref(),
                row_lens.as_ref().map(<[i32; 1]>::as_slice),
                Some(&mut row_cache),
                target,
                layer_idx,
            )?;
            cache.adopt_row_from(&row_cache, row as usize, 0)?;
            outputs.push(row_output);
        }
        let refs = outputs.iter().collect::<Vec<_>>();
        mlx::ops::shape::concatenate_on(&refs, 0, target).map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_sequence_stable_on(
        &self,
        x: &Array,
        mask: Option<&Array>,
        per_row_lens: Option<&[i32]>,
        mut cache: Option<&mut GatedDeltaCache>,
        target: StreamOrDevice,
        layer_idx: i32,
    ) -> Result<Array> {
        let shape = x.shape();
        let [batch, sequence, hidden_size] = *<&[i32; 3]>::try_from(shape.as_slice())
            .map_err(|_| anyhow!("sequence-stable GatedDelta requires [B,Q,H]"))?;
        let mut outputs = Vec::with_capacity(sequence as usize);
        for position in 0..sequence {
            let step = mlx::ops::indexing::slice_strided_on(
                x,
                &[0_i32, position, 0][..],
                &[batch, position + 1, hidden_size][..],
                &[1_i32, 1, 1][..],
                target,
            )?;
            let step_lens = per_row_lens.map(|lens| {
                lens.iter()
                    .map(|&len| i32::from(position < len))
                    .collect::<Vec<_>>()
            });
            let derived_mask;
            let step_mask = if let Some(mask) = mask {
                Some(mlx::ops::indexing::slice_strided_on(
                    mask,
                    &[0_i32, position][..],
                    &[batch, position + 1][..],
                    &[1_i32, 1][..],
                    target,
                )?)
            } else if let Some(lens) = step_lens.as_ref() {
                let values = lens.iter().map(|&len| len != 0).collect::<Vec<_>>();
                derived_mask = (&values[..], &[batch, 1_i32][..]).try_into()?;
                Some(derived_mask)
            } else {
                None
            };
            outputs.push(self.forward_on(
                &step,
                step_mask.as_ref(),
                step_lens.as_deref(),
                cache.as_deref_mut(),
                target,
                layer_idx,
            )?);
        }
        let refs = outputs.iter().collect::<Vec<_>>();
        mlx::ops::shape::concatenate_on(&refs, 1, target).map_err(Into::into)
    }
}

fn slice_batch_row_or_broadcast(array: &Array, row: i32, target: StreamOrDevice) -> Result<Array> {
    let shape = array.shape();
    let dims = shape.as_slice();
    anyhow::ensure!(!dims.is_empty(), "batch-stable slice requires rank > 0");
    if dims[0] == 1 {
        return Ok(array.clone());
    }
    anyhow::ensure!(
        row >= 0 && row < dims[0],
        "batch row {row} outside B={}",
        dims[0]
    );
    let mut starts = vec![0_i32; dims.len()];
    let mut stops = dims.to_vec();
    starts[0] = row;
    stops[0] = row + 1;
    let strides = vec![1_i32; dims.len()];
    mlx::ops::indexing::slice_strided_on(
        array,
        starts.as_slice(),
        stops.as_slice(),
        strides.as_slice(),
        target,
    )
    .map_err(Into::into)
}

fn slice_sequence_prefix_on(array: &Array, length: i32, target: StreamOrDevice) -> Result<Array> {
    let shape = array.shape();
    let dims = shape.as_slice();
    if dims.len() < 2 || length < 0 || length > dims[1] {
        return Err(anyhow!(
            "sequence prefix length {length} is invalid for shape {dims:?}"
        ));
    }
    let starts = vec![0_i32; dims.len()];
    let mut stops = dims.to_vec();
    stops[1] = length;
    let strides = vec![1_i32; dims.len()];
    mlx::ops::indexing::slice_strided_on(
        array,
        starts.as_slice(),
        stops.as_slice(),
        strides.as_slice(),
        target,
    )
    .map_err(Into::into)
}

/// Build the `gated_delta_step` MetalKernel (no-mask or masked variant).
///
/// The shader source is identical between variants except for the per-token
/// guard expression (`mask_clause`). MLX's `metal_kernel` machinery auto-injects
/// `<name>_shape` / `<name>_strides` / `<name>_ndim` for input arrays referenced
/// in the source.
///
/// `T` is passed as a 0-dim int32 array, which MLX treats as `device const
/// int32_t& T` — usable directly as an integer in the shader (e.g.
/// `for (int t = 0; t < T; ++t)`).
pub(crate) fn build_gated_delta_kernel(masked: bool) -> Result<MetalKernel> {
    build_gated_delta_kernel_impl(masked, false, true)
}

fn build_gated_delta_no_state_kernel(masked: bool) -> Result<MetalKernel> {
    build_gated_delta_kernel_impl(masked, false, false)
}

/// Build a `gated_delta_step` kernel for the first chunk of a stream, where
/// recurrent state is known to be all zeros. This variant intentionally has no
/// `state_in` input: it initializes the per-thread register tile to 0 and still
/// emits the same `state_out` shape as the regular kernel.
pub(crate) fn build_gated_delta_zero_state_kernel(masked: bool) -> Result<MetalKernel> {
    build_gated_delta_kernel_impl(masked, true, true)
}

fn build_gated_delta_kernel_impl(
    masked: bool,
    zero_state: bool,
    emit_state: bool,
) -> Result<MetalKernel> {
    let mask_clause = if masked {
        "mask[b_idx * T + t]"
    } else {
        "true"
    };
    let state_in_ptr = if zero_state {
        ""
    } else {
        "auto i_state = state_in + (n * Dv + dv_idx) * Dk;"
    };
    let state_init = if zero_state {
        "state[i] = 0.0f;"
    } else {
        "state[i] = static_cast<float>(i_state[s_idx]);"
    };
    let state_out_ptr = if emit_state {
        "auto o_state = state_out + (n * Dv + dv_idx) * Dk;"
    } else {
        ""
    };
    let state_store = if emit_state {
        r#"
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }"#
    } else {
        ""
    };
    let src = format!(
        r#"
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        {state_in_ptr}
        {state_out_ptr}

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          {state_init}
        }}

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_clause}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            // Note: all 32 simdgroup threads write the same zero value here
            // (no `thread_index_in_simdgroup == 0` guard). Matches mlx-lm
            // reference exactly; wasted write bandwidth is acceptable since
            // masked tokens are rare and all writes are identical.
            y[dv_idx] = static_cast<InT>(0);
          }}
          // Advance pointers to the next time step.
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        {state_store}
        "#,
        mask_clause = mask_clause,
        state_in_ptr = state_in_ptr,
        state_init = state_init,
        state_out_ptr = state_out_ptr,
        state_store = state_store
    );

    let name = match (masked, zero_state, emit_state) {
        (false, false, true) => "ironmlx_gated_delta",
        (true, false, true) => "ironmlx_gated_delta_masked",
        (false, false, false) => "ironmlx_gated_delta_no_state",
        (true, false, false) => "ironmlx_gated_delta_no_state_masked",
        (false, true, true) => "ironmlx_gated_delta_zero_state",
        (true, true, true) => "ironmlx_gated_delta_zero_state_masked",
        (_, true, false) => unreachable!("zero-state kernel must emit state"),
    };

    let inputs: &[&str] = match (masked, zero_state) {
        (false, false) => &["q", "k", "v", "g", "beta", "state_in", "T"],
        (true, false) => &["q", "k", "v", "g", "beta", "state_in", "T", "mask"],
        (false, true) => &["q", "k", "v", "g", "beta", "T"],
        (true, true) => &["q", "k", "v", "g", "beta", "T", "mask"],
    };

    let outputs: &[&str] = if emit_state {
        &["y", "state_out"]
    } else {
        &["y"]
    };
    Ok(MetalKernel::builder(name)
        .inputs(inputs)
        .outputs(outputs)
        .source(&src)
        .ensure_row_contiguous(true)
        .atomic_outputs(false)
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx::{Array, Dtype, Shape};
    use serial_test::serial;

    fn small_gdn_components() -> GatedDeltaNet {
        // Synthetic small model:
        // hidden=32, num_v_heads=4, num_k_heads=2, head_k_dim=32, head_v_dim=8,
        // conv_kernel=4, eps=1e-6
        // NOTE: head_k_dim must be >= 32 so n_per_t = Dk/32 >= 1 (Metal C++ forbids
        // zero-length arrays). Using head_k_dim=32, head_v_dim=8.
        let cfg = GatedDeltaNetConfig {
            hidden_size: 32,
            num_v_heads: 4,
            num_k_heads: 2,
            head_k_dim: 32,
            head_v_dim: 8,
            conv_kernel_size: 4,
            rms_norm_eps: 1e-6,
        };
        // key_dim = 2*32 = 64; value_dim = 4*8 = 32
        // qkv proj output = key_dim*2 + value_dim = 64+64+32 = 160
        // conv_dim = key_dim*2 + value_dim = 160
        // out_proj input = value_dim = 32
        let conv_dim = cfg.conv_dim(); // 160
        let value_dim = cfg.value_dim(); // 32

        let qkv_w = Array::zeros((conv_dim, 32), Dtype::Float32).unwrap();
        let z_w = Array::zeros((value_dim, 32), Dtype::Float32).unwrap();
        let b_w = Array::zeros((cfg.num_v_heads, 32), Dtype::Float32).unwrap();
        let a_w = Array::zeros((cfg.num_v_heads, 32), Dtype::Float32).unwrap();
        let conv_w = Array::zeros((conv_dim, cfg.conv_kernel_size, 1), Dtype::Float32).unwrap();
        let norm_w = mlx::ops::constructors::ones((cfg.head_v_dim,), Dtype::Float32).unwrap();
        let out_w = Array::zeros((32_i32, value_dim), Dtype::Float32).unwrap();
        let a_log = Array::zeros((cfg.num_v_heads,), Dtype::Float32).unwrap();
        let dt_bias = mlx::ops::constructors::ones((cfg.num_v_heads,), Dtype::Float32).unwrap();

        GatedDeltaNet::from_components(
            crate::nn::Linear::new_fp(qkv_w, None),
            crate::nn::Linear::new_fp(z_w, None),
            crate::nn::Linear::new_fp(b_w, None),
            crate::nn::Linear::new_fp(a_w, None),
            crate::nn::Conv1d::new(
                conv_w,
                None,
                crate::nn::Conv1dConfig {
                    in_channels: conv_dim,
                    out_channels: conv_dim,
                    kernel_size: cfg.conv_kernel_size,
                    stride: 1,
                    padding: 0,
                    dilation: 1,
                    groups: conv_dim, // depthwise
                },
            ),
            crate::nn::RmsNormGated::new(norm_w, cfg.rms_norm_eps),
            crate::nn::Linear::new_fp(out_w, None),
            a_log,
            dt_bias,
            cfg,
        )
    }

    fn small_nonzero_gdn_components() -> GatedDeltaNet {
        let cfg = GatedDeltaNetConfig {
            hidden_size: 32,
            num_v_heads: 4,
            num_k_heads: 2,
            head_k_dim: 32,
            head_v_dim: 8,
            conv_kernel_size: 4,
            rms_norm_eps: 1e-6,
        };
        let matrix = |rows: i32, columns: i32, value: f32| {
            let values = vec![value; usize::try_from(rows * columns).unwrap()];
            (values.as_slice(), &[rows, columns][..])
                .try_into()
                .unwrap()
        };
        let conv = |channels: i32, kernel: i32, value: f32| {
            let values = vec![value; usize::try_from(channels * kernel).unwrap()];
            (values.as_slice(), &[channels, kernel, 1_i32][..])
                .try_into()
                .unwrap()
        };
        let conv_dim = cfg.conv_dim();
        let value_dim = cfg.value_dim();
        GatedDeltaNet::from_components(
            crate::nn::Linear::new_fp(matrix(conv_dim, 32, 0.01), None),
            crate::nn::Linear::new_fp(matrix(value_dim, 32, 0.02), None),
            crate::nn::Linear::new_fp(matrix(cfg.num_v_heads, 32, 0.01), None),
            crate::nn::Linear::new_fp(matrix(cfg.num_v_heads, 32, 0.015), None),
            crate::nn::Conv1d::new(
                conv(conv_dim, cfg.conv_kernel_size, 0.025),
                None,
                crate::nn::Conv1dConfig {
                    in_channels: conv_dim,
                    out_channels: conv_dim,
                    kernel_size: cfg.conv_kernel_size,
                    stride: 1,
                    padding: 0,
                    dilation: 1,
                    groups: conv_dim,
                },
            ),
            crate::nn::RmsNormGated::new(
                mlx::ops::constructors::ones((cfg.head_v_dim,), Dtype::Float32).unwrap(),
                cfg.rms_norm_eps,
            ),
            crate::nn::Linear::new_fp(matrix(32, value_dim, 0.01), None),
            Array::zeros((cfg.num_v_heads,), Dtype::Float32).unwrap(),
            mlx::ops::constructors::ones((cfg.num_v_heads,), Dtype::Float32).unwrap(),
            cfg,
        )
    }

    #[test]
    #[serial(mlx_metal)]
    fn gdn_construction_carries_config() {
        let gdn = small_gdn_components();
        let cfg = gdn.config();
        assert_eq!(cfg.num_v_heads, 4);
        assert_eq!(cfg.num_k_heads, 2);
        assert_eq!(cfg.conv_kernel_size, 4);
    }

    #[test]
    #[serial(mlx_metal)]
    fn dflash2_fused_ba_matches_separate_product_stable_projections() {
        let input_width = 64_i32;
        let output_width = 32_i32;
        let group_size = 32_i32;
        let make_linear = |offset: i32| {
            let values = (0..output_width * input_width)
                .map(|index| (((index + offset) % 37) as f32 - 18.0) * 0.01)
                .collect::<Vec<_>>();
            let weight: Array = (values.as_slice(), &[output_width, input_width][..])
                .try_into()
                .expect("weight");
            let quantized =
                mlx::quantization::quantize(&weight, Some(group_size), Some(4), "affine", None)
                    .expect("quantize");
            Linear::new_quant(
                quantized[0].clone(),
                quantized[1].clone(),
                Some(quantized[2].clone()),
                None,
                group_size,
                4,
            )
        };
        let b = make_linear(0);
        let a = make_linear(11);
        let fused =
            Linear::fuse_quantized_outputs(&[&b, &a], "DFlash2 test fused GDN b/a").expect("fuse");
        let input_values = (0..4 * input_width)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.02)
            .collect::<Vec<_>>();
        let input: Array = (input_values.as_slice(), &[1_i32, 4, input_width][..])
            .try_into()
            .expect("input");

        let separate = {
            let _stable = crate::nn::position_stable_qmm::scope();
            let b = b.forward(&input).expect("b");
            let a = a.forward(&input).expect("a");
            mlx::ops::shape::concatenate(&[&b, &a], -1).expect("concatenate")
        };
        let fused = {
            let _stable = crate::nn::position_stable_qmm::scope();
            fused.forward(&input).expect("fused forward")
        };
        let separate = mlx::ops::cast::astype(&separate, Dtype::Float32)
            .expect("cast separate")
            .to_vec::<f32>()
            .expect("materialize separate");
        let fused = mlx::ops::cast::astype(&fused, Dtype::Float32)
            .expect("cast fused")
            .to_vec::<f32>()
            .expect("materialize fused");
        assert_eq!(fused, separate);
    }

    #[test]
    #[serial(mlx_metal)]
    fn gdn_forward_shape_dtype_no_cache() {
        let gdn = small_gdn_components();
        // x: [B=1, S=4, hidden=32] — note: small zeros so the SSM dispatch
        // succeeds even with our trivial weights.
        let x = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        let out = gdn.forward(&x, None, None).expect("forward no cache");
        // out_proj maps value_dim=32 -> 32
        assert_eq!(out.shape().as_slice(), &[1, 4, 32]);
        // dtype may be promoted to fp32 by rms_norm path; just verify finite
        let v: Vec<f32> = mlx::ops::cast::astype(&out, Dtype::Float32)
            .unwrap()
            .to_vec()
            .unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "non-finite output");
    }

    #[test]
    #[serial(mlx_metal)]
    fn gdn_forward_with_cache_advances_offset() {
        let gdn = small_gdn_components();
        let cfg = gdn.config();
        let mut cache = GatedDeltaCache::new_with_cap(
            1, // B
            cfg.conv_kernel_size,
            cfg.conv_dim(),
            cfg.num_v_heads,
            cfg.head_v_dim,
            cfg.head_k_dim,
            Dtype::Float32,
            16, // cap
        )
        .expect("cache");
        let x = Array::zeros((1_i32, 4, 32), Dtype::Float32).unwrap();
        let _out = gdn
            .forward(&x, None, Some(&mut cache))
            .expect("forward with cache");
        assert_eq!(cache.offsets(), &[4]);
    }

    #[test]
    #[serial(mlx_metal)]
    fn captured_b2_q2_and_exact_b4_q2_restore_hidden_and_cache_bitwise() {
        let gdn = small_nonzero_gdn_components();
        let cfg = gdn.config();
        for batch in [2_i32, 4] {
            let sequence = 2_i32;
            let new_cache = || {
                GatedDeltaCache::new_with_cap(
                    batch,
                    cfg.conv_kernel_size,
                    cfg.conv_dim(),
                    cfg.num_v_heads,
                    cfg.head_v_dim,
                    cfg.head_k_dim,
                    Dtype::Float32,
                    16,
                )
                .expect("cache")
            };
            let values = (0..batch * sequence * 32)
                .map(|index| ((index % 23) as f32 - 11.0) * 0.01)
                .collect::<Vec<_>>();
            let input: Array = (values.as_slice(), &[batch, sequence, 32][..])
                .try_into()
                .expect("input");
            let accepted_lens = (0..batch)
                .map(|row| 1 + row as usize % sequence as usize)
                .collect::<Vec<_>>();
            let verify_lens = vec![sequence; batch as usize];

            let mut restored = new_cache();
            restored
                .begin_speculative_prefix_capture()
                .expect("begin capture");
            let exact_b4_scope =
                (batch == 4).then(crate::nn::position_stable_qmm::exact_affine8_b4_q2_scope);
            let restored_output = gdn
                .forward_on(
                    &input,
                    None,
                    Some(&verify_lens),
                    Some(&mut restored),
                    StreamOrDevice::default(),
                    0,
                )
                .expect("captured verify");
            drop(exact_b4_scope);

            let mut full_expected = new_cache();
            let full_expected_output = gdn
                .forward_on(
                    &input,
                    None,
                    Some(&verify_lens),
                    Some(&mut full_expected),
                    StreamOrDevice::default(),
                    0,
                )
                .expect("ordinary B4/Q1-equivalent hidden");
            assert_eq!(
                restored_output.to_vec::<f32>().unwrap(),
                full_expected_output.to_vec::<f32>().unwrap(),
                "verify hidden batch={batch}"
            );
            gdn.restore_speculative_prefix_rows_on(
                &mut restored,
                &accepted_lens,
                StreamOrDevice::default(),
            )
            .expect("restore accepted prefixes");

            let mut expected = new_cache();
            let mask_values = accepted_lens
                .iter()
                .flat_map(|&accepted| {
                    (0..sequence as usize).map(move |position| position < accepted)
                })
                .collect::<Vec<_>>();
            let mask: Array = (mask_values.as_slice(), &[batch, sequence][..])
                .try_into()
                .expect("mask");
            let expected_lens = accepted_lens
                .iter()
                .map(|&accepted| i32::try_from(accepted).unwrap())
                .collect::<Vec<_>>();
            gdn.forward_on(
                &input,
                Some(&mask),
                Some(&expected_lens),
                Some(&mut expected),
                StreamOrDevice::default(),
                0,
            )
            .expect("masked accepted prefixes");

            assert_eq!(restored.offsets(), expected.offsets(), "batch={batch}");
            assert_eq!(
                restored.conv_state().to_vec::<f32>().unwrap(),
                expected.conv_state().to_vec::<f32>().unwrap(),
                "batch={batch}"
            );
            assert_eq!(
                restored.recurrent_state().to_vec::<f32>().unwrap(),
                expected.recurrent_state().to_vec::<f32>().unwrap(),
                "batch={batch}"
            );
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn captured_b2_q2_restores_only_participating_row_bitwise() {
        let gdn = small_nonzero_gdn_components();
        let cfg = gdn.config();
        let new_cache = || {
            GatedDeltaCache::new_with_cap(
                2,
                cfg.conv_kernel_size,
                cfg.conv_dim(),
                cfg.num_v_heads,
                cfg.head_v_dim,
                cfg.head_k_dim,
                Dtype::Float32,
                16,
            )
            .expect("cache")
        };
        let values = (0..2 * 2 * 32)
            .map(|index| ((index % 23) as f32 - 11.0) * 0.01)
            .collect::<Vec<_>>();
        let input: Array = (values.as_slice(), &[2_i32, 2, 32][..])
            .try_into()
            .expect("input");
        let verify_mask: Array = (&[true, true, false, false][..], &[2_i32, 2][..])
            .try_into()
            .expect("verify mask");

        let mut restored = new_cache();
        restored
            .begin_speculative_prefix_capture()
            .expect("begin capture");
        gdn.forward_on(
            &input,
            Some(&verify_mask),
            Some(&[2_i32, 0]),
            Some(&mut restored),
            StreamOrDevice::default(),
            0,
        )
        .expect("captured partial-row verify");
        gdn.restore_speculative_prefix_rows_on(
            &mut restored,
            &[1_usize, 0],
            StreamOrDevice::default(),
        )
        .expect("restore participating row");

        let expected_mask: Array = (&[true, false, false, false][..], &[2_i32, 2][..])
            .try_into()
            .expect("expected mask");
        let mut expected = new_cache();
        gdn.forward_on(
            &input,
            Some(&expected_mask),
            Some(&[1_i32, 0]),
            Some(&mut expected),
            StreamOrDevice::default(),
            0,
        )
        .expect("expected partial-row prefix");

        assert_eq!(restored.offsets(), expected.offsets());
        assert_eq!(restored.offsets(), &[1, 0]);
        assert_eq!(
            restored.conv_state().to_vec::<f32>().unwrap(),
            expected.conv_state().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            restored.recurrent_state().to_vec::<f32>().unwrap(),
            expected.recurrent_state().to_vec::<f32>().unwrap()
        );
    }

    #[test]
    #[serial(mlx_metal)]
    fn gated_delta_step_kernel_links() {
        // Dk must be >= 32 so that n_per_t = Dk/32 >= 1 (Metal C++ forbids
        // zero-length arrays). Use Dk=32, Dv=8, Hk=Hv=1, B=1, T=1.
        let kernel = build_gated_delta_kernel(false).expect("build kernel");

        let q = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let k = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let v = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let g = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let beta = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let state_in = Array::zeros((1_i32, 1, 8, 32), Dtype::Float32).unwrap();
        let t_arr: Array = (&[1_i32][..], ()).try_into().unwrap();

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&[&q, &k, &v, &g, &beta, &state_in, &t_arr])
            .output_shapes(&[
                Shape::from(vec![1, 1, 1, 8]),
                Shape::from(vec![1, 1, 8, 32]),
            ])
            .output_dtypes(&[Dtype::Bfloat16, Dtype::Float32])
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 32)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Bfloat16)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch");

        let _y = outputs.take_at(0).expect("y");
        let _state = outputs.take_at(0).expect("state");
    }

    fn assert_zero_state_kernel_matches_regular(masked: bool) {
        let regular = build_gated_delta_kernel(masked).expect("regular kernel");
        let zero_state = build_gated_delta_zero_state_kernel(masked).expect("zero-state kernel");

        let q_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.001 + 0.01).collect();
        let k_data: Vec<f32> = (0..64).map(|i| (i as f32) * -0.0007 + 0.02).collect();
        let v_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.003 + 0.1).collect();
        let g_data = [0.7_f32, 0.5];
        let beta_data = [0.25_f32, 0.4];

        let q: Array = (q_data.as_slice(), (1_i32, 2, 1, 32)).try_into().unwrap();
        let k: Array = (k_data.as_slice(), (1_i32, 2, 1, 32)).try_into().unwrap();
        let v: Array = (v_data.as_slice(), (1_i32, 2, 1, 8)).try_into().unwrap();
        let g: Array = (g_data.as_slice(), (1_i32, 2, 1)).try_into().unwrap();
        let beta: Array = (beta_data.as_slice(), (1_i32, 2, 1)).try_into().unwrap();
        let state_in = Array::zeros((1_i32, 1, 8, 32), Dtype::Float32).unwrap();
        let t_arr: Array = (&[2_i32][..], ()).try_into().unwrap();
        let mask: Option<Array> = masked.then(|| {
            let mask_data = [true, false];
            (mask_data.as_slice(), (2_i32,)).try_into().unwrap()
        });

        let y_shape = Shape::from(vec![1, 2, 1, 8]);
        let state_shape = Shape::from(vec![1, 1, 8, 32]);
        let output_shapes = [y_shape.clone(), state_shape.clone()];
        let output_dtypes = [Dtype::Float32, Dtype::Float32];

        let mut regular_inputs: Vec<&Array> = vec![&q, &k, &v, &g, &beta, &state_in, &t_arr];
        if let Some(mask) = mask.as_ref() {
            regular_inputs.push(mask);
        }
        let mut regular_out = regular
            .dispatch_builder()
            .inputs(&regular_inputs)
            .output_shapes(&output_shapes)
            .output_dtypes(&output_dtypes)
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 32)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Float32)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch regular");

        let mut zero_inputs: Vec<&Array> = vec![&q, &k, &v, &g, &beta, &t_arr];
        if let Some(mask) = mask.as_ref() {
            zero_inputs.push(mask);
        }
        let mut zero_out = zero_state
            .dispatch_builder()
            .inputs(&zero_inputs)
            .output_shapes(&[y_shape, state_shape])
            .output_dtypes(&output_dtypes)
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 32)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Float32)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch zero-state");

        let regular_y = regular_out.take_at(0).expect("regular y");
        let regular_state = regular_out.take_at(0).expect("regular state");
        let zero_y = zero_out.take_at(0).expect("zero y");
        let zero_state_out = zero_out.take_at(0).expect("zero state");

        let regular_y_vec: Vec<f32> = regular_y.to_vec().unwrap();
        let zero_y_vec: Vec<f32> = zero_y.to_vec().unwrap();
        let regular_state_vec: Vec<f32> = regular_state.to_vec().unwrap();
        let zero_state_vec: Vec<f32> = zero_state_out.to_vec().unwrap();

        for (actual, expected) in zero_y_vec.iter().zip(regular_y_vec.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "zero-state y diverged: actual={actual} expected={expected}"
            );
        }
        for (actual, expected) in zero_state_vec.iter().zip(regular_state_vec.iter()) {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "zero-state recurrent state diverged: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    #[serial(mlx_metal)]
    fn gated_delta_zero_state_kernel_matches_explicit_zero_state() {
        assert_zero_state_kernel_matches_regular(false);
    }

    #[test]
    #[serial(mlx_metal)]
    fn gated_delta_zero_state_masked_kernel_matches_explicit_zero_state() {
        assert_zero_state_kernel_matches_regular(true);
    }

    #[test]
    #[serial(mlx_metal)]
    fn gated_delta_step_masked_zero_path() {
        // mask=0 everywhere: output should be 0, state unchanged.
        // Use non-zero state_in to verify state isn't accidentally modified.
        // Dk=32 (minimum) so n_per_t = 32/32 = 1 (Metal C++ forbids zero-length arrays).
        let kernel = build_gated_delta_kernel(true).expect("build masked kernel");

        // Initial state has values [1.0; 8*32] (Hv=1, Dv=8, Dk=32).
        let init_state_data: Vec<f32> = (0..256).map(|_| 1.0_f32).collect();
        let state_in: Array = (init_state_data.as_slice(), (1_i32, 1, 8, 32))
            .try_into()
            .unwrap();

        let q = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let k = Array::zeros((1_i32, 1, 1, 32), Dtype::Bfloat16).unwrap();
        let v = Array::zeros((1_i32, 1, 1, 8), Dtype::Bfloat16).unwrap();
        let g = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let beta = Array::zeros((1_i32, 1, 1), Dtype::Float32).unwrap();
        let t_arr: Array = (&[1_i32][..], ()).try_into().unwrap();
        // mask: [B*T] = [1*1 = 1] all-zero (masked out)
        let mask = Array::zeros((1_i32,), Dtype::Bool).unwrap();

        let mut outputs = kernel
            .dispatch_builder()
            .inputs(&[&q, &k, &v, &g, &beta, &state_in, &t_arr, &mask])
            .output_shapes(&[
                Shape::from(vec![1, 1, 1, 8]),
                Shape::from(vec![1, 1, 8, 32]),
            ])
            .output_dtypes(&[Dtype::Bfloat16, Dtype::Float32])
            .grid(32, 8, 1)
            .threadgroup(32, 4, 1)
            .template_int("Dk", 32)
            .template_int("Dv", 8)
            .template_int("Hk", 1)
            .template_int("Hv", 1)
            .template_dtype("InT", Dtype::Bfloat16)
            .template_dtype("StT", Dtype::Float32)
            .dispatch()
            .expect("dispatch masked");

        let y = outputs.take_at(0).expect("y");
        let state_out = outputs.take_at(0).expect("state_out");

        // y must be all-zero (else branch sets `y[dv_idx] = 0`).
        let y_f32 = mlx::ops::cast::astype(&y, Dtype::Float32).unwrap();
        let yv: Vec<f32> = y_f32.to_vec().unwrap();
        assert!(
            yv.iter().all(|x| x.abs() < 1e-6),
            "masked output not zero: {:?}",
            yv
        );

        // state_out must equal state_in (no update under mask=0 — kernel writes
        // back the unchanged register-cached state at the end).
        let sv: Vec<f32> = state_out.to_vec().unwrap();
        assert!(
            sv.iter().all(|x| (x - 1.0).abs() < 1e-6),
            "state changed under mask=0: {:?}",
            sv
        );
    }
}
