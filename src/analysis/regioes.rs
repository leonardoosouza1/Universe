//! Track regions × activation — does the brain use different neurons on straights and curves?
//!
//! # Why it exists
//!
//! Leonardo: *"the brain will work in different dimensions, neurons and parts
//! depending on where it is on the track, and that needs to be measured."*
//!
//! We had never crossed the two sides that already existed separately:
//!   - activation samples (`--ativacao`), collected in `gpu_evolve/context.rs`
//!     (`l1_samples`/`l2_samples`) — *what* the brain does;
//!   - track geometry, in the same discrete curvature-κ definition that
//!     `src/bin/replay_mapa.rs::Pista::medir` uses for `ω_exigido = v·κ`
//!     (signed angle between consecutive segments ÷ mean length) — *where*
//!     the being is. Reimplemented here (not imported from there) because `replay_mapa` is
//!     a binary, not a lib — the formula is small enough not to justify
//!     a bridge crate just for this.
//!
//! This module only sees numbers: whoever generates the pair (activation sample, physical
//! context) aligned by index is the caller (`GpuCtx::tick`, which pushes both in the
//! SAME iteration — see `regiao_l1`/`regiao_l2` in `context.rs`).
//!
//! The cohesion question ("do the curve-specialized neurons fire TOGETHER?")
//! reuses `contextual::coativacao` — it is exactly the same question as E3
//! (co-activation excess over independence), only restricted to curve ticks
//! and to the neurons that passed the specialization threshold. Not reimplemented.

use crate::analysis::contextual;
use crate::math::stats;

/// Curvature κ (1/m) per waypoint of the track axis, plus nearest-point search
/// from an arbitrary position. It is the SAME discrete definition as `replay_mapa.rs::Pista`:
/// κ = signed angle between the two segments touching the waypoint, divided by the
/// mean length of the two — "nothing estimated" (original comment).
#[derive(Debug, Clone)]
pub struct CurvaturaPista {
    pts: Vec<(f32, f32)>,
    /// SIGNED κ per waypoint — the sign does not matter for straight-vs-curve (we use
    /// `.abs()` in `curvatura_em`), but it is kept raw for possible future use
    /// (e.g. distinguishing left/right).
    kappa: Vec<f32>,
}

impl CurvaturaPista {
    /// `waypoints_xz`: track axis in world coordinates (x, z), closed or not
    /// (works the same — the layout is treated as a cycle via `% n`, which is safe even
    /// for an open list because the closing point has the smallest curvature weight
    /// of anything, and no being passes near it anyway if the track is
    /// open). Fewer than 3 points = degenerate track, all curvatures become 0.
    pub fn medir(waypoints_xz: &[(f32, f32)]) -> Self {
        let n = waypoints_xz.len();
        if n < 3 {
            return Self {
                pts: waypoints_xz.to_vec(),
                kappa: vec![0.0; n],
            };
        }
        let mut kappa = vec![0.0f32; n];
        for i in 0..n {
            let a = waypoints_xz[(i + n - 1) % n];
            let b = waypoints_xz[i];
            let c = waypoints_xz[(i + 1) % n];
            let v1 = (b.0 - a.0, b.1 - a.1);
            let v2 = (c.0 - b.0, c.1 - b.1);
            let l1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt().max(1e-3);
            let l2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt().max(1e-3);
            let cross = v1.0 * v2.1 - v1.1 * v2.0;
            let dot = v1.0 * v2.0 + v1.1 * v2.1;
            let ang = cross.atan2(dot);
            kappa[i] = ang / (0.5 * (l1 + l2));
        }
        Self {
            pts: waypoints_xz.to_vec(),
            kappa,
        }
    }

    /// |κ| (1/m) of the waypoint nearest to `(x, z)`. Linear search — only called on the
    /// sampled ticks (stride of `--ativacao-n`, typically 1 in 64), never in the hot
    /// physics loop; with a few hundred waypoints the cost is irrelevant.
    pub fn curvatura_em(&self, x: f32, z: f32) -> f32 {
        if self.pts.is_empty() {
            return 0.0;
        }
        let mut melhor = 0usize;
        let mut best_d = f32::MAX;
        for (i, p) in self.pts.iter().enumerate() {
            let d = (p.0 - x).powi(2) + (p.1 - z).powi(2);
            if d < best_d {
                best_d = d;
                melhor = i;
            }
        }
        self.kappa[melhor].abs()
    }
}

/// Physical context of ONE activation sample — same index as the samples vector
/// (`l1_samples[i]` ↔ `regiao_l1[i]`). Alignment is the caller's responsibility:
/// this type carries neither tick nor being, only the three numbers the protocol asked for.
#[derive(Debug, Clone, Copy, Default)]
pub struct ContextoAmostra {
    /// Local track |κ| (1/m) at the being's position on this tick. 0 = perfect straight.
    pub curvatura: f32,
    /// Scalar speed (m/s).
    pub velocidade: f32,
    /// Smallest distance to any visible wall (m) — min(left, right, front).
    pub dist_parede: f32,
}

/// Metrics for ONE neuron: how much it changes between straight and curve.
#[derive(Debug, Clone)]
pub struct EspecializacaoNeuronio {
    pub indice: usize,
    /// Mean |activation| over the ticks classified as STRAIGHT.
    pub media_reta: f32,
    /// Mean |activation| over the ticks classified as CURVE.
    pub media_curva: f32,
    /// |media_curva - media_reta| / (media_curva + media_reta + eps). ~1 = only fires
    /// in one of the two regimes; ~0 = fires equally in both.
    pub indice_especializacao: f32,
    /// Pearson between the neuron's |activation| and local curvature, over ALL samples.
    pub correlacao_curvatura: f32,
    /// Same, against speed.
    pub correlacao_velocidade: f32,
    /// Same, against distance to the wall.
    pub correlacao_dist_parede: f32,
}

/// Full report of one layer (L1 or L2) for one generation.
#[derive(Debug, Clone)]
pub struct RelatorioRegioes {
    pub camada: String,
    /// |κ| above this = "in a curve". ALWAYS reported — it is a choice, and a choice
    /// that changes the result has to appear in the report (same trap no. 1 of the
    /// `contextual.rs` protocol: nothing here zeroes out exactly).
    pub limiar_curva: f32,
    pub n_reta: usize,
    pub n_curva: usize,
    pub por_neuronio: Vec<EspecializacaoNeuronio>,
    /// `indice_especializacao` threshold above which a neuron counts as
    /// "specialized". ALWAYS reported, same reason as `limiar_curva`.
    pub limiar_especializacao: f32,
    pub indices_especializados: Vec<usize>,
    /// Co-activation (E3 from `contextual.rs`) restricted to CURVE ticks and the
    /// specialized columns. `None` when there are fewer than 2 specialized neurons or fewer
    /// than 2 curve samples — co-activation among 0 or 1 neurons is not a valid question.
    pub coesao: Option<contextual::CoAtivacao>,
}

const EPS: f32 = 1e-6;

/// Pearson between `xs` and `ys` (same length, truncated to the shorter if they diverge).
/// Generic enough not to have a domain name — same convention as
/// `crate::math::stats` (pure functions, no brain semantics).
fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let mx = stats::mean(&xs[..n]);
    let my = stats::mean(&ys[..n]);
    let mut cov = 0.0f64;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    for i in 0..n {
        let dx = (xs[i] - mx) as f64;
        let dy = (ys[i] - my) as f64;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 1e-12 || vy <= 1e-12 {
        return 0.0; // one of the two series is constant — undefined correlation, not zero-by-chance
    }
    (cov / (vx.sqrt() * vy.sqrt())) as f32
}

/// Crosses activation samples of ONE layer with the aligned physical context.
///
/// `amostras[t]` and `contexto[t]` must come from the SAME tick/being — the caller
/// (`GpuCtx::tick`) guarantees that by pushing both together. `None` if the
/// lengths do not match (misalignment is a collection bug, not data to analyze
/// blindly) or if there are no samples at all.
pub fn analisar(
    camada: &str,
    amostras: &[Vec<f32>],
    contexto: &[ContextoAmostra],
    limiar_curva: f32,
    limiar_especializacao: f32,
    limiar_ativacao_coesao: f32,
) -> Option<RelatorioRegioes> {
    if amostras.is_empty() || amostras.len() != contexto.len() {
        return None;
    }
    let n_neur = amostras[0].len();
    if n_neur == 0 {
        return None;
    }

    let mut reta_idx = Vec::new();
    let mut curva_idx = Vec::new();
    for (t, c) in contexto.iter().enumerate() {
        if c.curvatura >= limiar_curva {
            curva_idx.push(t);
        } else {
            reta_idx.push(t);
        }
    }

    let media_abs = |idxs: &[usize], k: usize| -> f32 {
        if idxs.is_empty() {
            return 0.0;
        }
        idxs.iter().map(|&t| amostras[t][k].abs()).sum::<f32>() / idxs.len() as f32
    };

    let curvaturas: Vec<f32> = contexto.iter().map(|c| c.curvatura).collect();
    let velocidades: Vec<f32> = contexto.iter().map(|c| c.velocidade).collect();
    let dist_paredes: Vec<f32> = contexto.iter().map(|c| c.dist_parede).collect();

    let mut por_neuronio = Vec::with_capacity(n_neur);
    for k in 0..n_neur {
        let media_reta = media_abs(&reta_idx, k);
        let media_curva = media_abs(&curva_idx, k);
        let indice_especializacao =
            (media_curva - media_reta).abs() / (media_curva + media_reta + EPS);
        let ativ_abs: Vec<f32> = amostras.iter().map(|a| a[k].abs()).collect();
        por_neuronio.push(EspecializacaoNeuronio {
            indice: k,
            media_reta,
            media_curva,
            indice_especializacao,
            correlacao_curvatura: pearson(&ativ_abs, &curvaturas),
            correlacao_velocidade: pearson(&ativ_abs, &velocidades),
            correlacao_dist_parede: pearson(&ativ_abs, &dist_paredes),
        });
    }

    let indices_especializados: Vec<usize> = por_neuronio
        .iter()
        .filter(|e| e.indice_especializacao >= limiar_especializacao)
        .map(|e| e.indice)
        .collect();

    // Cohesion: do the specialized neurons fire TOGETHER during the curve, or does each
    // decide on its own? Restricts samples to the CURVE rows and the specialized
    // columns, and reuses `contextual::coativacao` — same question as E3,
    // smaller domain.
    let coesao = if indices_especializados.len() >= 2 && curva_idx.len() >= 2 {
        let submatriz: Vec<Vec<f32>> = curva_idx
            .iter()
            .map(|&t| {
                indices_especializados
                    .iter()
                    .map(|&k| amostras[t][k])
                    .collect()
            })
            .collect();
        Some(contextual::coativacao(
            camada,
            &submatriz,
            limiar_ativacao_coesao,
            5,
        ))
    } else {
        None
    };

    Some(RelatorioRegioes {
        camada: camada.to_string(),
        limiar_curva,
        n_reta: reta_idx.len(),
        n_curva: curva_idx.len(),
        por_neuronio,
        limiar_especializacao,
        indices_especializados,
        coesao,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CurvaturaPista ───────────────────────────────────────────────────────

    #[test]
    fn reta_tem_curvatura_proxima_de_zero() {
        // Collinear points — no angle between consecutive segments.
        let pts: Vec<(f32, f32)> = (0..10).map(|i| (i as f32, 0.0)).collect();
        let c = CurvaturaPista::medir(&pts);
        assert!(
            c.curvatura_em(5.0, 0.0) < 1e-4,
            "a straight should have kappa ~0"
        );
    }

    #[test]
    fn curva_fechada_tem_curvatura_bem_maior_que_reta() {
        // An "L": straight segment from (0,0) to (10,0), then a straight 90° turn to (10,10).
        // The vertex (10,0) has a 90° corner; a point in the middle of the straight does not.
        let mut pts: Vec<(f32, f32)> = (0..=10).map(|i| (i as f32, 0.0)).collect();
        pts.extend((1..=10).map(|i| (10.0, i as f32)));
        let c = CurvaturaPista::medir(&pts);
        let k_reta = c.curvatura_em(5.0, 0.0);
        let k_curva = c.curvatura_em(10.0, 0.0); // the corner
        assert!(
            k_curva > k_reta * 10.0,
            "the corner should be much curvier than the straight: straight={k_reta} curve={k_curva}"
        );
    }

    #[test]
    fn pista_vazia_nao_quebra() {
        let c = CurvaturaPista::medir(&[]);
        assert_eq!(c.curvatura_em(0.0, 0.0), 0.0);
    }

    // ── analisar() ───────────────────────────────────────────────────────────

    /// Builds `contexto` alternating straight/curve by index: even = straight (curvature
    /// 0.0), odd = curve (curvature 1.0). Same convention in the two central tests
    /// of the protocol (item 2 of the task).
    fn contexto_alternado(n: usize) -> Vec<ContextoAmostra> {
        (0..n)
            .map(|t| ContextoAmostra {
                curvatura: if t % 2 == 0 { 0.0 } else { 1.0 },
                velocidade: 10.0,
                dist_parede: 5.0,
            })
            .collect()
    }

    #[test]
    fn neuronio_so_dispara_em_curva_tem_indice_perto_de_1() {
        let n = 20;
        let contexto = contexto_alternado(n);
        // Neuron 0: 0.0 on the straight, 1.0 on the curve. Neuron 1: constant 0.3 (control).
        let amostras: Vec<Vec<f32>> = (0..n)
            .map(|t| {
                let em_curva = t % 2 == 1;
                vec![if em_curva { 1.0 } else { 0.0 }, 0.3]
            })
            .collect();
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.5, 0.05).unwrap();
        assert!(
            r.por_neuronio[0].indice_especializacao > 0.95,
            "expected ~1, got {}",
            r.por_neuronio[0].indice_especializacao
        );
    }

    #[test]
    fn neuronio_dispara_igual_nos_dois_tem_indice_perto_de_0() {
        let n = 20;
        let contexto = contexto_alternado(n);
        let amostras: Vec<Vec<f32>> = (0..n).map(|_| vec![0.7, 0.5]).collect();
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.5, 0.05).unwrap();
        assert!(
            r.por_neuronio[0].indice_especializacao < 0.05,
            "expected ~0, got {}",
            r.por_neuronio[0].indice_especializacao
        );
    }

    #[test]
    fn conta_especializados_acima_do_limiar_reportado() {
        let n = 20;
        let contexto = contexto_alternado(n);
        // 3 neurons: 0 specialized in curves, 1 specialized in straights, 2 neutral.
        let amostras: Vec<Vec<f32>> = (0..n)
            .map(|t| {
                let em_curva = t % 2 == 1;
                vec![
                    if em_curva { 1.0 } else { 0.0 },
                    if em_curva { 0.0 } else { 1.0 },
                    0.5,
                ]
            })
            .collect();
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.8, 0.05).unwrap();
        assert_eq!(r.limiar_especializacao, 0.8);
        assert_eq!(r.indices_especializados.len(), 2);
        assert!(r.indices_especializados.contains(&0));
        assert!(r.indices_especializados.contains(&1));
        assert!(!r.indices_especializados.contains(&2));
    }

    #[test]
    fn correlacao_curvatura_positiva_para_neuronio_alinhado_com_curva() {
        let n = 40;
        let contexto = contexto_alternado(n);
        let amostras: Vec<Vec<f32>> = (0..n)
            .map(|t| {
                vec![contexto[t].curvatura] // activation = the curvature itself → corr = 1
            })
            .collect();
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.5, 0.05).unwrap();
        assert!(
            r.por_neuronio[0].correlacao_curvatura > 0.99,
            "correlation should be ~1, got {}",
            r.por_neuronio[0].correlacao_curvatura
        );
    }

    #[test]
    fn coesao_none_quando_menos_de_2_especializados() {
        let n = 20;
        let contexto = contexto_alternado(n);
        let amostras: Vec<Vec<f32>> = (0..n).map(|_| vec![0.5, 0.5]).collect();
        // extremely high threshold — nobody passes.
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.99, 0.05).unwrap();
        assert!(r.indices_especializados.is_empty());
        assert!(r.coesao.is_none());
    }

    #[test]
    fn coesao_detecta_especializados_que_disparam_juntos_na_curva() {
        let n = 80;
        let contexto = contexto_alternado(n);
        // Neurons 0 and 1: specialized in curves AND coupled — always TOGETHER when
        // they fire, but not in EVERY curve (otherwise both end up with marginal 1.0 inside
        // the curve subset and co-activation becomes identical to independence —
        // 1.0×1.0=1.0 —, exactly the case that `ganho_temporal_zero_quando_tudo_sempre_ativo`
        // in contextual.rs already covers for persistence; here VARIATION within
        // the curve is needed for cohesion to have something to detect, just like
        // `coativacao_detecta_par_acoplado`). Every 4 curve ticks, 1 goes "cold".
        let mut cnt_curva = 0usize;
        let amostras: Vec<Vec<f32>> = (0..n)
            .map(|t| {
                let em_curva = t % 2 == 1;
                if em_curva {
                    let ligado = cnt_curva % 4 != 0;
                    cnt_curva += 1;
                    let v = if ligado { 1.0 } else { 0.0 };
                    vec![v, v, 0.1]
                } else {
                    vec![0.0, 0.0, 0.1]
                }
            })
            .collect();
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.5, 0.5).unwrap();
        assert_eq!(r.indices_especializados.len(), 2);
        let co = r.coesao.expect("should have cohesion with 2 specialized");
        assert!(
            co.excesso > 0.0,
            "excess should be positive, got {}",
            co.excesso
        );
    }

    #[test]
    fn amostras_e_contexto_desalinhados_retorna_none() {
        let amostras = vec![vec![0.0; 2]; 5];
        let contexto = contexto_alternado(3);
        assert!(analisar("teste", &amostras, &contexto, 0.5, 0.5, 0.05).is_none());
    }
}
