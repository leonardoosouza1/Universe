//! CfC de percepção — ponte canônica para `iara_cfc::CfcCell`.
//!
//! # Por que este módulo existe
//!
//! A equação do CfC recorrente que processa a retina (usado por `RetinaPerception`, em
//! `perception/retina.rs`) mora na crate externa `iara-cfc` — compartilhada por todo o
//! ecossistema IARA (`~/projects/LLM/iara-runtime/iara-cfc`). Essa é a ÚNICA
//! implementação da recorrência: não existe, e não deve passar a existir, uma segunda
//! cópia da matemática em lugar nenhum deste crate (Leonardo, 2026-08-13: "sem ter CfC
//! espalhado e afins").
//!
//! Este módulo NÃO reimplementa a equação — reexporta o tipo canônico e expõe só o que
//! `retina_digest.wgsl` precisa para espelhar o forward na GPU: as dimensões e o layout
//! de buffer achatado para upload. Mesma convenção de `math/activations.rs`: "as versões
//! GPU usam as mesmas fórmulas; alterações aqui devem ser espelhadas nos shaders".
//!
//! # A equação (referência — a fonte da verdade é `iara_cfc::CfcCell::step`)
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
//! `matvec` acumula com um único `acc` sequencial (`for c in 0..cols { acc += ... }`) —
//! `retina_digest.wgsl` replica esse loop literalmente (sem usar `dot()` do WGSL, que
//! permite ao compilador reassociar a soma e divergir em ULPs do resultado da CPU).
//!
//! # Pipeline completo até aqui chegar (não é este módulo — ver `perception/retina.rs`)
//!
//! `retina_i8[1764]` (pixel a pixel, `gpu::legacy_encoding`) → col max-pool 126→16 bins
//! por linha (`RetinaSensor::sense_retina`) → 3-zone pool far/mid/near
//! (`retina_zones()` + `ZoneConfig::pool`) → `x[48]` → `CfcCell::step` (aqui).

pub use iara_cfc::{CfcCell, CfcState};

use crate::perception::retina::{PERCEPTION_BINS, PERCEPTION_HIDDEN};

/// Dimensão do estado recorrente do CfC de percepção. Mesma constante que
/// `RetinaPerception` usa para construir o `PerceptionPipeline` — reexportada aqui
/// (não redefinida) para que os dois lados nunca possam divergir por edição isolada.
pub const CFC_HIDDEN: usize = PERCEPTION_HIDDEN; // 16

/// Dimensão da entrada do CfC: 3 zonas (far/mid/near) × `PERCEPTION_BINS` bins pooled.
/// `PerceptionPipeline::new` calcula isto como `zone_config.n_zones() * n_features_per_zone`;
/// aqui vira constante porque `retina_zones()` (retina.rs) sempre usa exatamente 3 zonas.
pub const CFC_INPUT: usize = 3 * PERCEPTION_BINS; // 48

/// Número de f32 no buffer de pesos achatado que `retina_digest.wgsl` espera (binding
/// `cfc_weights`). Layout: `[w_xi(768) | w_hi(256) | w_f(256) | b_i(16) | b_f(16)]` —
/// SEM a `decision_head` de `RetinaPerception::weights_to_vec()`: o kernel de GPU só
/// roda o passo recorrente que alimenta ch192-207, não a cabeça de 4 classes (que hoje
/// nenhum caminho do `gpu_evolve` consome — ver achado "pesos do CfC nunca carregados
/// do genoma").
pub const CFC_WEIGHTS_LEN: usize = CFC_HIDDEN * CFC_INPUT   // w_xi
    + CFC_HIDDEN * CFC_HIDDEN // w_hi
    + CFC_HIDDEN * CFC_HIDDEN // w_f
    + CFC_HIDDEN              // b_i
    + CFC_HIDDEN; // b_f

/// Achata os pesos de um `CfcCell` no layout que `retina_digest.wgsl` espera.
///
/// Mesma ORDEM de `RetinaPerception::weights_to_vec()` (w_xi, w_hi, w_f, b_i, b_f),
/// truncada antes da `decision_head`. Os pesos do CfC de percepção são, hoje, fixos e
/// idênticos para todos os seres em todas as gerações (seed fixa em
/// `RetinaPerception::new()`, `Genome` do `gpu_evolve` não carrega `perception_weights`)
/// — por isso um único upload, uma vez por processo, basta.
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
        // w_xi primeiro (768 valores), depois w_hi (256), w_f (256), b_i (16), b_f (16).
        assert_eq!(&w[0..768], &cell.w_xi[..]);
        assert_eq!(&w[768..768 + 256], &cell.w_hi[..]);
        assert_eq!(&w[1024..1024 + 256], &cell.w_f[..]);
        assert_eq!(&w[1280..1280 + 16], &cell.b_i[..]);
        assert_eq!(&w[1296..1296 + 16], &cell.b_f[..]);
    }

    #[test]
    fn duas_instancias_tem_os_mesmos_pesos() {
        // RetinaPerception::new() usa seed FIXA — achado crítico do mapa de percepção:
        // todos os seres, em todas as gerações, têm exatamente os mesmos pesos de CfC
        // de percepção no gpu_evolve. Um buffer único compartilhado é seguro HOJE; se
        // isso mudar (perception_weights plugado no Genome), este teste é o primeiro a
        // quebrar e avisar quem estiver mexendo em retina_digest.rs.
        let a = flat_weights(&RetinaPerception::new().pipeline.cfc);
        let b = flat_weights(&RetinaPerception::new().pipeline.cfc);
        assert_eq!(
            a, b,
            "seed seria fixa — pesos deveriam ser idênticos entre instâncias"
        );
    }
}
