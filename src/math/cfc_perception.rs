//! Perception CfC — canonical bridge to `iara_cfc::CfcCell`.
//!
//! # Why this module exists
//!
//! The equation of the recurrent CfC that processes the retina (used by `RetinaPerception`, in
//! `perception/retina.rs`) lives in the external crate `iara-cfc` — shared by the whole
//! IARA ecosystem (`~/projects/LLM/iara-runtime/iara-cfc`). That is the ONLY
//! implementation of the recurrence: there is not, and must not come to exist, a second
//! copy of the math anywhere in this crate (Leonardo, 2026-08-13: "sem ter CfC
//! espalhado e afins").
//!
//! This module does NOT reimplement the equation — it re-exports the canonical type and exposes
//! only what `retina_digest.wgsl` needs to mirror the forward pass on the GPU: the dimensions and
//! the flattened buffer layout for upload. Same convention as `math/activations.rs`: "the GPU
//! versions use the same formulas; changes here must be mirrored in the shaders".
//!
//! # The equation (reference — the source of truth is `iara_cfc::CfcCell::step`)
//!
//! ```text
//! pre_xi = w_xi[16×48] @ x        (matvec: for r { acc=0; for c { acc += w[r,c]*x[c] } })
//! pre_hi = w_hi[16×16] @ h
//! pre_f  = w_f [16×16] @ h
//! i_gate[i] = tanh(pre_xi[i] + pre_hi[i] + b_i[i])
//! f_gate[i] = sigmoid(pre_f[i] + b_f[i])
//! decay[i]  = exp(-f_gate[i])
//! h_new[i]  = h[i]*decay[i] + i_gate[i]*(1-decay[i])
//! ```
//!
//! `matvec` accumulates with a single sequential `acc` (`for c in 0..cols { acc += ... }`) —
//! `retina_digest.wgsl` replicates that loop literally (without using WGSL's `dot()`, which
//! lets the compiler reassociate the sum and diverge in ULPs from the CPU result).
//!
//! # Full pipeline before reaching here (not this module — see `perception/retina.rs`)
//!
//! `retina_i8[1764]` (pixel by pixel, `gpu::legacy_encoding`) → col max-pool 126→16 bins
//! per row (`RetinaSensor::sense_retina`) → 3-zone pool far/mid/near
//! (`retina_zones()` + `ZoneConfig::pool`) → `x[48]` → `CfcCell::step` (here).

pub use iara_cfc::{CfcCell, CfcState};

use crate::perception::retina::{PERCEPTION_BINS, PERCEPTION_HIDDEN};

/// Dimension of the perception CfC's recurrent state. Same constant that
/// `RetinaPerception` uses to build the `PerceptionPipeline` — re-exported here
/// (not redefined) so the two sides can never diverge through an isolated edit.
pub const CFC_HIDDEN: usize = PERCEPTION_HIDDEN; // 16

/// Dimension of the CfC input: 3 zones (far/mid/near) × `PERCEPTION_BINS` pooled bins.
/// `PerceptionPipeline::new` computes this as `zone_config.n_zones() * n_features_per_zone`;
/// here it becomes a constant because `retina_zones()` (retina.rs) always uses exactly 3 zones.
pub const CFC_INPUT: usize = 3 * PERCEPTION_BINS; // 48

/// Number of f32 in the flattened weight buffer that `retina_digest.wgsl` expects (binding
/// `cfc_weights`). Layout: `[w_xi(768) | w_hi(256) | w_f(256) | b_i(16) | b_f(16)]` —
/// WITHOUT the `decision_head` of `RetinaPerception::weights_to_vec()`: the GPU kernel only
/// runs the recurrent step that feeds ch192-207, not the 4-class head (which today
/// no `gpu_evolve` path consumes — see the finding "CfC weights never loaded
/// from the genome").
pub const CFC_WEIGHTS_LEN: usize = CFC_HIDDEN * CFC_INPUT   // w_xi
    + CFC_HIDDEN * CFC_HIDDEN // w_hi
    + CFC_HIDDEN * CFC_HIDDEN // w_f
    + CFC_HIDDEN              // b_i
    + CFC_HIDDEN; // b_f

/// Flattens a `CfcCell`'s weights into the layout `retina_digest.wgsl` expects.
///
/// Same ORDER as `RetinaPerception::weights_to_vec()` (w_xi, w_hi, w_f, b_i, b_f),
/// truncated before the `decision_head`. The perception CfC weights are, today, fixed and
/// identical for all beings in all generations (fixed seed in
/// `RetinaPerception::new()`, the `gpu_evolve` `Genome` does not carry `perception_weights`)
/// — which is why a single upload, once per process, is enough.
pub fn flat_weights(cell: &CfcCell) -> Vec<f32> {
    debug_assert_eq!(
        cell.hidden_dim, CFC_HIDDEN,
        "CfcCell hidden_dim != CFC_HIDDEN"
    );
    debug_assert_eq!(cell.input_dim, CFC_INPUT, "CfcCell input_dim != CFC_INPUT");
    let mut w = Vec::with_capacity(CFC_WEIGHTS_LEN);
    w.extend_from_slice(&cell.w_xi);
    w.extend_from_slice(&cell.w_hi);
    w.extend_from_slice(&cell.w_f);
    w.extend_from_slice(&cell.b_i);
    w.extend_from_slice(&cell.b_f);
    debug_assert_eq!(w.len(), CFC_WEIGHTS_LEN);
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::RetinaPerception;

    #[test]
    fn dimensoes_batem_com_retina_perception() {
        let perc = RetinaPerception::new();
        assert_eq!(perc.pipeline.cfc.hidden_dim, CFC_HIDDEN);
        assert_eq!(perc.pipeline.cfc.input_dim, CFC_INPUT);
    }

    #[test]
    fn flat_weights_tem_o_tamanho_esperado() {
        let perc = RetinaPerception::new();
        let w = flat_weights(&perc.pipeline.cfc);
        assert_eq!(w.len(), CFC_WEIGHTS_LEN);
        // 768 + 256 + 256 + 16 + 16 = 1312
        assert_eq!(CFC_WEIGHTS_LEN, 1312);
    }

    #[test]
    fn flat_weights_preserva_ordem_e_valores() {
        let perc = RetinaPerception::new();
        let cell = &perc.pipeline.cfc;
        let w = flat_weights(cell);
        // w_xi first (768 values), then w_hi (256), w_f (256), b_i (16), b_f (16).
        assert_eq!(&w[0..768], &cell.w_xi[..]);
        assert_eq!(&w[768..768 + 256], &cell.w_hi[..]);
        assert_eq!(&w[1024..1024 + 256], &cell.w_f[..]);
        assert_eq!(&w[1280..1280 + 16], &cell.b_i[..]);
        assert_eq!(&w[1296..1296 + 16], &cell.b_f[..]);
    }

    #[test]
    fn duas_instancias_tem_os_mesmos_pesos() {
        // RetinaPerception::new() uses a FIXED seed — critical finding of the perception map:
        // all beings, in all generations, have exactly the same perception CfC
        // weights in gpu_evolve. A single shared buffer is safe TODAY; if
        // that changes (perception_weights plugged into the Genome), this test is the first to
        // break and warn whoever is touching retina_digest.rs.
        let a = flat_weights(&RetinaPerception::new().pipeline.cfc);
        let b = flat_weights(&RetinaPerception::new().pipeline.cfc);
        assert_eq!(
            a, b,
            "seed is supposed to be fixed — weights should be identical across instances"
        );
    }
}
