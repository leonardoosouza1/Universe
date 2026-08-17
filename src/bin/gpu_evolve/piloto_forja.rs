//! PILOT v2 — runtime of the FORGED BASE (non-linear), reading `piloto_v2.json`.
//!
//! v1 (`piloto_linear.rs`) solves a LINEAR controller by ridge over the (sensed→did)
//! pairs from the champion's replay. v2 swaps the BASE: instead of feeding the ridge
//! with `[x, EMA_a, EMA_b]` (576 features), it feeds a PYRAMID OF GAUSSIANS —
//! 512 "coarse" centers (`Cg`) and 512 "fine" ones (`Cf`), each set read through TWO
//! widths, giving 2048 features. Error on the temporal held-out (ticks 17000..20000):
//! **0.2620 → 0.1794 (−31.5%)**, per action throttle 0.109 · steering 0.137 · brake 0.292
//! (`piloto_v2.json:err_holdout_temporal` and `err_holdout_por_acao`).
//!
//! ## Exact math (FORMATO.md, `"formato": "piloto-v2/1"`)
//! With `x` = the SAME 192 channels L1 receives (post-KAN+CfC, identical to v1):
//! 1. `z[j] = (x[j] - mu[j]) / sd[j]`
//! 2. `dg[a] = Σ_j (z[j] - Cg[a][j])²` · `df[b] = Σ_j (z[j] - Cf[b][j])²`
//! 3. `phi = [ e^(-dg/2sg0²) ‖ e^(-dg/2sg1²) ‖ e^(-df/2sf0²) ‖ e^(-df/2sf1²) ]`
//!    — 2048 values IN THIS ORDER (it is the order of the rows of `W`; swapping the
//!    order of the four blocks is the spec's trap and the test `ordem_dos_quatro_blocos_de_phi`
//!    exists solely to catch it)
//! 4. `pre[k] = (Σ_i phi[i]·W[i][k]) · y_sd[k] + y_mu[k]`
//! 5. `intent[k] = tanh(pre[k])` → `[throttle, steering, brake]`
//!
//! The `tanh` in step 5 is NOT decoration: the fit was done on `arctanh(clip(y, ±0.999))`,
//! so `W` predicts the PRE-activation. Without it the error rises from 0.1794 to 0.2306
//! (measured by the forge, FORMATO.md:16). `clip_arctanh` is in the JSON as a record of
//! the fit; the runtime uses no clip at all. The output is the RAW intent, the SAME
//! delivery point as v1 (pre `kan_eval`/clamps) — the caller continues with the normal
//! pipeline.
//!
//! ## Cost per tick (2×512-center pyramid over 192 channels)
//! - distances: `(512+512) × 192 = 196 608` mult (same number of sub and add)
//! - `2048` exponentials
//! - reading W: `2048 × 3 = 6 144` MAC
//! - **total ≈ 202 752 mult/tick** (`mult_por_tick()` returns that number, checked
//!   in the test). At 64 Hz = **12.98 M mult/s** (v1 spent 576×3 = 1 728 mult/tick,
//!   117× less, but erred 46% more).
//! - MEASURED, not estimated: **143.8 µs/tick** (5 000 ticks in 718.8 ms, 6 956 ticks/s,
//!   `rustc -O` without `target-cpu=native`, Ryzen 7 5800X3D, 1 thread). That is **0.92% of
//!   the budget of one tick at 64 Hz** (15 625 µs) for ONE being — this is the arithmetic
//!   that authorizes calling it negligible, and it holds for the v1 use case (the pilot
//!   drives being 0, the rest follows the brain). Driving the whole POPULATION is NOT
//!   free: 200 beings × 143.8 µs = 28.8 ms/tick, nearly double the budget. For that, the
//!   base has to go to the GPU.
//! - `carregar` (parsing the 4.1 MB JSON): **14.1 ms**, once.
//! - resident memory: `(512+512)×192×4 B = 786 KB` of centers + 24 KB of W.
//! - STATELESS: `agir` is a pure function of `x` (v1 carried 2 EMAs). This is what
//!   allows calling it out of order, repeating a tick, or testing it line by line.
//!
//! ## Fragility measured, not assumed
//! `sd` has 6 degenerate channels (98, 114, 115, 116, 117, 124) with `sd = 1e-12`: they
//! are channels CONSTANT at zero across the 20 001 ticks of the dump (measured:
//! `np.unique(X[:,j]) == [0]`). Inside the dump `z = 0` on them and nothing happens. On a
//! NEW track, if one of them leaves zero, `z ≈ 5e11`, `dg` blows up and ALL of `phi` goes
//! to zero — the output freezes at `tanh(y_mu)`, constant. Hence:
//! - `z` is saturated at ±`Z_MAX` (1e3) only to avoid producing `inf`/`NaN` in the
//!   physics. It changes no number in distribution: the largest `|z|` in the whole dump
//!   is **19.07** and the largest `|center|` is **17.63** — 1e3 is 50× above everything
//!   that exists.
//! - `soma_phi` is "I am inside what I saw" telemetry: across the whole dump it ranges
//!   from **287.9 to 936.7** (p1 = 375.9; tick 17000 = 402.4). `agir_com_portao` abstains
//!   when it falls below `thr_phi` — off by default (`thr_phi = 0`), because turning it
//!   on without measuring would change the validated behaviour.

use std::io::Read;

/// Saturation of `z`. Purely anti-`NaN` — see the "Fragility measured" block above.
const Z_MAX: f32 = 1.0e3;

/// A set of centers sharing a bank of widths (lenses).
/// The pyramid has 2 groups (`Cg` with `sg`, `Cf` with `sf`); the RBF has 1.
struct Grupo {
    /// `[n × c]` FLATTENED row-major: center `a`, channel `j`, is `centros[a * c + j]`.
    /// The JSON parser here scans numbers and ignores the inner brackets, so the
    /// flattening is assumed, not read — which is why `carregar` CHECKS `len == n * c`
    /// and rejects the file if it does not match (the only defence against a
    /// transposed matrix).
    centros: Vec<f32>,
    n: usize,
    /// `1/(2σ²)` per lens, in the order the lenses appear in `phi`.
    inv2s2: Vec<f32>,
}

pub struct PilotoForja {
    base: String,
    c: usize,
    mu: Vec<f32>,
    sd: Vec<f32>,
    grupos: Vec<Grupo>,
    /// `[n_feat × 3]` FLATTENED row-major: `w[i * 3 + k]` = weight of feature `i` on action `k`.
    w: Vec<f32>,
    y_mu: [f32; 3],
    y_sd: [f32; 3],
    n_feat: usize,
    // buffers reused between ticks (zero allocation in the hot loop)
    z: Vec<f32>,
    d2: Vec<f32>,
    phi: Vec<f32>,
    // ── telemetry ───────────────────────────────────────────────────────────
    /// Σφ of the last tick. Range measured in the dump: 287.9 … 936.7.
    pub ultima_soma_phi: f32,
    /// How many times `z` hit `Z_MAX` (≠0 = a degenerate channel left zero).
    pub saturacoes_z: u64,
    pub abstencoes: u64,
    pub decisoes: u64,
    /// Σφ threshold below which `agir_com_portao` abstains. 0 = gate off.
    pub thr_phi: f32,
    /// Indices with degenerate `sd` (< 1e-6): channels the solve saw as constant.
    pub canais_degenerados: Vec<usize>,
}

impl PilotoForja {
    /// Loads `piloto_v2.json`. Zero-dependency parser, same strategy as
    /// `piloto_linear.rs`: find `"key"`, take the next `[`, match brackets and
    /// scan ALL the numbers in the range. Consequence: a matrix `[[..],[..]]` comes out
    /// FLATTENED row-major (the scan ignores the inner brackets). Dimensions come
    /// from the scalars `n_entrada`/`ng`/`nf`/`n_feat` of the JSON itself and are
    /// checked against the length read — a file that does not match is rejected.
    ///
    /// Bases: `"piramide"` (the winner, mandatory) and `"rbf"` behind the same match.
    /// `"linear"` is NOT accepted here on purpose: linear is done by `piloto_linear.rs`,
    /// which applies `clamp` instead of `tanh` — accepting it here would silently
    /// produce numbers different from v1.
    pub fn carregar(path: &str) -> Option<PilotoForja> {
        let mut s = String::new();
        std::fs::File::open(path)
            .ok()?
            .read_to_string(&mut s)
            .ok()?;

        // ── extractors ──────────────────────────────────────────────────────
        let arr = |chave: &str| -> Option<Vec<f32>> {
            let i = s.find(&format!("\"{chave}\""))?;
            let a = s[i..].find('[')? + i;
            let mut nivel = 0i32;
            let mut fim = a;
            for (k, ch) in s[a..].char_indices() {
                match ch {
                    '[' => nivel += 1,
                    ']' => {
                        nivel -= 1;
                        if nivel == 0 {
                            fim = a + k;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let nums: Vec<f32> = s[a..=fim]
                .split(|c: char| {
                    !(c.is_ascii_digit()
                        || c == '.'
                        || c == '-'
                        || c == 'e'
                        || c == 'E'
                        || c == '+')
                })
                .filter(|t| !t.is_empty() && t.chars().any(|c| c.is_ascii_digit()))
                .filter_map(|t| t.parse().ok())
                .collect();
            Some(nums)
        };
        // Scalar (non-array). Do NOT reuse `arr` here: `arr` looks for the next
        // '[', which in a scalar field belongs to ANOTHER key — that is how the
        // v1 gate threshold silently became INFINITY (piloto_linear.rs:88-91).
        let bruto = |chave: &str| -> Option<String> {
            let i = s.find(&format!("\"{chave}\""))?;
            let resto = &s[i + chave.len() + 2..];
            let j = resto.find(':')? + 1;
            let fim = resto[j..]
                .find(|c: char| c == ',' || c == '}')
                .unwrap_or(resto.len() - j)
                + j;
            Some(resto[j..fim].trim().trim_matches('"').to_string())
        };
        let escalar = |chave: &str| -> Option<f32> { bruto(chave)?.parse().ok() };
        let inteiro = |chave: &str| -> Option<usize> { bruto(chave)?.parse().ok() };

        let formato = bruto("formato").unwrap_or_default();
        let base = bruto("base").unwrap_or_default();
        if !formato.starts_with("piloto-v2/") {
            eprintln!(
                "[piloto-forja] unknown format {formato:?} in {path} — expected piloto-v2/*"
            );
            return None;
        }

        let c = inteiro("n_entrada")?;
        let mu = arr("mu")?;
        let sd = arr("sd")?;
        let w = arr("W")?;
        let y_mu_v = arr("y_mu")?;
        let y_sd_v = arr("y_sd")?;
        if mu.len() != c || sd.len() != c || y_mu_v.len() != 3 || y_sd_v.len() != 3 {
            eprintln!(
                "[piloto-forja] unexpected dimensions: n_entrada={c} mu={} sd={} y_mu={} y_sd={}",
                mu.len(),
                sd.len(),
                y_mu_v.len(),
                y_sd_v.len()
            );
            return None;
        }

        // ── bases ───────────────────────────────────────────────────────────
        // The ORDER of this list is the order of the blocks of `phi`, and therefore the
        // order of the rows of `W`. Touching this without touching the solve swaps the
        // actions around.
        let mut grupos: Vec<Grupo> = Vec::new();
        let monta = |chave_c: &str, n: usize, sig: Vec<f32>| -> Option<Grupo> {
            let centros = arr(chave_c)?;
            if centros.len() != n * c {
                eprintln!(
                    "[piloto-forja] {chave_c} has {} numbers, expected {n}×{c}={} \
                           (transposed matrix or wrong dimension in the JSON)",
                    centros.len(),
                    n * c
                );
                return None;
            }
            Some(Grupo {
                centros,
                n,
                inv2s2: sig.iter().map(|s| 1.0 / (2.0 * s * s)).collect(),
            })
        };
        match base.as_str() {
            "piramide" => {
                let ng = inteiro("ng")?;
                let nf = inteiro("nf")?;
                grupos.push(monta("Cg", ng, arr("sg")?)?);
                grupos.push(monta("Cf", nf, arr("sf")?)?);
            }
            "rbf" => {
                // There is no `piloto_v2.json` with base "rbf" on this machine, so this
                // branch is NOT VALIDATED against real data — only against the synthetic
                // tests here. It is open because the math is the pyramid with a single
                // group.
                let ng = inteiro("ng")?;
                grupos.push(monta("Cg", ng, arr("sg")?)?);
            }
            outra => {
                eprintln!(
                    "[piloto-forja] base {outra:?} not supported by this runtime \
                           (linear belongs to piloto_linear.rs, which uses clamp instead of tanh)"
                );
                return None;
            }
        }

        let n_feat: usize = grupos.iter().map(|g| g.n * g.inv2s2.len()).sum();
        if let Some(declarado) = inteiro("n_feat") {
            if declarado != n_feat {
                eprintln!("[piloto-forja] declared n_feat {declarado} ≠ assembled {n_feat}");
                return None;
            }
        }
        if w.len() != n_feat * 3 {
            eprintln!(
                "[piloto-forja] W has {} numbers, expected {n_feat}×3={}",
                w.len(),
                n_feat * 3
            );
            return None;
        }

        let canais_degenerados: Vec<usize> = (0..c).filter(|&j| sd[j].abs() < 1e-6).collect();
        let maior_n = grupos.iter().map(|g| g.n).max().unwrap_or(0);
        let mult = grupos.iter().map(|g| g.n * c).sum::<usize>() + n_feat * 3;
        println!(
            "  [piloto-forja] base={base} {c}→{n_feat}→3 · {} groups · {mult} mult/tick \
                  · err_holdout {} · degenerate {:?}",
            grupos.len(),
            escalar("err_holdout_temporal").unwrap_or(f32::NAN),
            canais_degenerados
        );

        Some(PilotoForja {
            base,
            c,
            mu,
            sd,
            grupos,
            w,
            y_mu: [y_mu_v[0], y_mu_v[1], y_mu_v[2]],
            y_sd: [y_sd_v[0], y_sd_v[1], y_sd_v[2]],
            n_feat,
            z: vec![0.0; c],
            d2: vec![0.0; maior_n],
            phi: vec![0.0; n_feat],
            ultima_soma_phi: 0.0,
            saturacoes_z: 0,
            abstencoes: 0,
            decisoes: 0,
            thr_phi: escalar("thr_phi").unwrap_or(0.0),
            canais_degenerados,
        })
    }

    pub fn n_entradas(&self) -> usize {
        self.c
    }
    pub fn n_features(&self) -> usize {
        self.n_feat
    }
    pub fn base(&self) -> &str {
        &self.base
    }

    /// Multiplications per tick of this configuration (distances + reading W).
    /// 2×512 pyramid over 192 channels = 202 752; at 64 Hz, 12.98 M mult/s.
    pub fn mult_por_tick(&self) -> usize {
        self.grupos.iter().map(|g| g.n * self.c).sum::<usize>() + self.n_feat * 3
    }

    /// One tick. `x` = the 192 channels AS L1 RECEIVES THEM (post-KAN+CfC, the same X of
    /// the solve). Returns `[throttle, steering, brake]` in the SAME convention as the
    /// brain's intent (post-`tanh`, pre `kan_eval`/clamps) — same signature and same
    /// delivery point as `PilotoLinear::agir`.
    pub fn agir(&mut self, x: &[f32]) -> [f32; 3] {
        let c = self.c;
        let n_feat = self.n_feat;
        debug_assert_eq!(
            x.len(),
            c,
            "piloto-forja: X with {} channels, expected {c}",
            x.len()
        );
        // Destructuring to borrow `grupos` (read-only) and `z`/`d2`/`phi`
        // (write) at the same time — `for g in &self.grupos { self.d2[..] = .. }`
        // does not compile, both borrows would come from the same `self`.
        let PilotoForja {
            mu,
            sd,
            grupos,
            w,
            z,
            d2,
            phi,
            y_mu,
            y_sd,
            ultima_soma_phi,
            saturacoes_z,
            ..
        } = self;

        // 1. standardize (with anti-NaN saturation — see the module doc)
        let mut saturou = false;
        for j in 0..c {
            let v = (x[j] - mu[j]) / sd[j];
            let v = if v.is_finite() { v } else { 0.0 };
            if v > Z_MAX || v < -Z_MAX {
                saturou = true;
            }
            z[j] = v.clamp(-Z_MAX, Z_MAX);
        }
        if saturou {
            *saturacoes_z += 1;
        }

        // 2+3. distances and lenses. The outer loop is the GROUP and the inner one the
        //      LENS: that is what produces [dg/sg0 ‖ dg/sg1 ‖ df/sf0 ‖ df/sf1], the
        //      order of the rows of W. Each d² is computed ONCE and read by every lens
        //      of the group (which is where the pyramid gets cheap: 1024 distances,
        //      2048 features).
        let mut p = 0usize;
        let mut soma = 0.0f32;
        for g in grupos.iter() {
            for a in 0..g.n {
                let base = a * c;
                let mut d = 0.0f32;
                for j in 0..c {
                    let dd = z[j] - g.centros[base + j];
                    d += dd * dd;
                }
                d2[a] = d;
            }
            for &inv in &g.inv2s2 {
                for a in 0..g.n {
                    let v = (-d2[a] * inv).exp();
                    phi[p] = v;
                    soma += v;
                    p += 1;
                }
            }
        }
        debug_assert_eq!(p, n_feat);
        *ultima_soma_phi = soma;

        // 4. phi @ W
        let mut pre = [0.0f32; 3];
        for i in 0..n_feat {
            let f = phi[i];
            pre[0] += f * w[i * 3];
            pre[1] += f * w[i * 3 + 1];
            pre[2] += f * w[i * 3 + 2];
        }

        // 5. denormalize and tanh (W predicts the PRE-activation — see the module doc)
        [
            (pre[0] * y_sd[0] + y_mu[0]).tanh(),
            (pre[1] * y_sd[1] + y_mu[1]).tanh(),
            (pre[2] * y_sd[2] + y_mu[2]).tanh(),
        ]
    }

    /// Like `agir`, but with a Σφ GATE: returns `None` when the state lit up no lens
    /// (`Σφ < thr_phi`), and then the brain decides the tick — same contract as
    /// `PilotoLinear::agir_com_portao`.
    ///
    /// Σφ measured across the whole dump (20 001 ticks): min **287.9**, p1 **375.9**,
    /// median 724.7, max 936.7 (tick 17000 = 402.4). `thr_phi` comes from the JSON when
    /// present, otherwise **0 = gate off**: turning it on unilaterally would change the
    /// behaviour that yielded 0.1794, and that is a decision for whoever calibrates,
    /// not the runtime. A conservative value would be ~150 (half of the observed
    /// minimum).
    pub fn agir_com_portao(&mut self, x: &[f32]) -> Option<[f32; 3]> {
        let y = self.agir(x);
        if self.ultima_soma_phi >= self.thr_phi {
            self.decisoes += 1;
            Some(y)
        } else {
            self.abstencoes += 1;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V2: &str = "/media/llm-data/universe-runs/piloto-formula/v2/piloto_v2.json";
    const SANIDADE: &str = "/media/llm-data/universe-runs/piloto-formula/v2/sanidade3.txt";

    /// Minimal synthetic JSON in the piloto-v2/1 format.
    /// `cg`/`cf`: rows of centers already in row-major order.
    fn json_piramide(
        c: usize,
        cg: &[Vec<f32>],
        cf: &[Vec<f32>],
        sg: [f32; 2],
        sf: [f32; 2],
        w: &[[f32; 3]],
        mu: &[f32],
        sd: &[f32],
        y_mu: [f32; 3],
        y_sd: [f32; 3],
    ) -> String {
        let mat = |m: &[Vec<f32>]| -> String {
            m.iter()
                .map(|r| {
                    format!(
                        "[{}]",
                        r.iter()
                            .map(|v| format!("{v:?}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        let wm: String = w
            .iter()
            .map(|r| format!("[{:?},{:?},{:?}]", r[0], r[1], r[2]))
            .collect::<Vec<_>>()
            .join(",");
        let v = |x: &[f32]| {
            x.iter()
                .map(|v| format!("{v:?}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "{{\"formato\": \"piloto-v2/1\", \"base\": \"piramide\", \"n_entrada\": {c}, \"n_saida\": 3, \
             \"ng\": {}, \"nf\": {}, \"n_feat\": {}, \"sg\": [{:?},{:?}], \"sf\": [{:?},{:?}], \
             \"mu\": [{}], \"sd\": [{}], \"y_mu\": [{:?},{:?},{:?}], \"y_sd\": [{:?},{:?},{:?}], \
             \"Cg\": [{}], \"Cf\": [{}], \"W\": [{}]}}",
            cg.len(),
            cf.len(),
            (cg.len() + cf.len()) * 2,
            sg[0],
            sg[1],
            sf[0],
            sf[1],
            v(mu),
            v(sd),
            y_mu[0],
            y_mu[1],
            y_mu[2],
            y_sd[0],
            y_sd[1],
            y_sd[2],
            mat(cg),
            mat(cf),
            wm
        )
    }

    fn escreve(nome: &str, conteudo: &str) -> String {
        let p = std::env::temp_dir().join(nome);
        std::fs::write(&p, conteudo).unwrap();
        p.to_str().unwrap().to_string()
    }

    /// The 5 steps of FORMATO.md in a case that can be checked by hand.
    /// 1 coarse center at 0, 1 fine at 0, x = [1,0] → z = [1,0] → dg = df = 1.
    #[test]
    fn cinco_passos_conferidos_na_mao() {
        let c = 2;
        let j = json_piramide(
            c,
            &[vec![0.0, 0.0]],
            &[vec![0.0, 0.0]],
            [1.0, 2.0],
            [0.5, 4.0],
            // W: only feature 0 (dg with sg[0]) has weight, on action 0.
            &[
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            &[0.0, 0.0],
            &[1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        let mut p = PilotoForja::carregar(&escreve("pf_5passos.json", &j)).expect("load");
        assert_eq!((p.n_entradas(), p.n_features()), (2, 4));
        assert_eq!(p.base(), "piramide");
        let y = p.agir(&[1.0, 0.0]);
        // phi[0] = exp(-1/(2·1²)) = 0.60653066 ; pre[0] = 0.60653066 ; y0 = tanh(pre)
        let esperado = (-0.5f32).exp().tanh();
        assert!(
            (y[0] - esperado).abs() < 1e-6,
            "y0={} expected={esperado}",
            y[0]
        );
        assert_eq!((y[1], y[2]), (0.0, 0.0));
        // Σφ = e^-0.5 + e^-0.125 + e^-2 + e^-0.03125
        let soma: f32 = (-0.5f32).exp() + (-0.125f32).exp() + (-2.0f32).exp() + (-0.03125f32).exp();
        assert!(
            (p.ultima_soma_phi - soma).abs() < 1e-6,
            "Σφ={} expected={soma}",
            p.ultima_soma_phi
        );
        // 1+1 centers × 2 channels + 4 features × 3 = 4 + 12 = 16 mult/tick
        assert_eq!(p.mult_por_tick(), 16);
    }

    /// THE SPEC'S TRAP: `phi` is `[dg/sg0 ‖ dg/sg1 ‖ df/sf0 ‖ df/sf1]`.
    /// Here each of the 4 blocks has a different σ and a different center, and W wires
    /// ONE block per action — if the order is swapped, two values land in the wrong place.
    #[test]
    fn ordem_dos_quatro_blocos_de_phi() {
        let c = 1;
        // Cg = [[0]], Cf = [[3]]; x = 1 → dg = 1, df = 4.
        let j = json_piramide(
            c,
            &[vec![0.0]],
            &[vec![3.0]],
            [1.0, 2.0],
            [4.0, 8.0],
            // feature 0 → action 0 · feature 1 → action 1 · feature 2 → action 2 · feature 3 → action 2 (weight 10)
            &[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 10.0],
            ],
            &[0.0],
            &[1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        let mut p = PilotoForja::carregar(&escreve("pf_ordem.json", &j)).unwrap();
        let y = p.agir(&[1.0]);
        let f0 = (-1.0f32 / 2.0).exp(); // dg=1, sg0=1 → exp(-1/2)
        let f1 = (-1.0f32 / 8.0).exp(); // dg=1, sg1=2 → exp(-1/8)
        let f2 = (-4.0f32 / 32.0).exp(); // df=4, sf0=4 → exp(-4/32)
        let f3 = (-4.0f32 / 128.0).exp(); // df=4, sf1=8 → exp(-4/128)
        assert!(
            (y[0] - f0.tanh()).abs() < 1e-6,
            "block 0 (Cg,sg0) out of place: {}",
            y[0]
        );
        assert!(
            (y[1] - f1.tanh()).abs() < 1e-6,
            "block 1 (Cg,sg1) out of place: {}",
            y[1]
        );
        assert!(
            (y[2] - (f2 + 10.0 * f3).tanh()).abs() < 1e-6,
            "blocks 2/3 (Cf) out of place: {}",
            y[2]
        );
        // f1 > f0 and f3 > f2: if the blocks swapped, y[0] and y[1] would swap values.
        assert!(f1 > f0 && f3 > f2);
    }

    /// `W` and `Cg`/`Cf` arrive FLATTENED from the parser (it scans numbers and ignores
    /// the inner brackets). This test nails the row-major layout: `w[i*3+k]` and
    /// `centros[a*c+j]`. With 2 centers of 3 channels, one center matches x exactly and
    /// the other does not.
    #[test]
    fn achatamento_row_major_de_matriz() {
        let c = 3;
        let j = json_piramide(
            c,
            &[vec![1.0, 2.0, 3.0], vec![-9.0, -9.0, -9.0]], // Cg: center 0 matches x
            &[vec![0.0, 0.0, 0.0]],
            [1.0, 1.0],
            [1.0, 1.0],
            // n_feat = (2+1)*2 = 6. Action 1 reads ONLY feature 0 (center 0, sg0).
            &[
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        let mut p = PilotoForja::carregar(&escreve("pf_flat.json", &j)).unwrap();
        assert_eq!(p.n_features(), 6);
        let y = p.agir(&[1.0, 2.0, 3.0]);
        // d² of center 0 = 0 → phi[0] = 1 → pre[1] = 5 → y1 = tanh(5)
        assert!(
            (y[1] - 5.0f32.tanh()).abs() < 1e-6,
            "y1={} (expected tanh(5)); \
                 if the matrix were read transposed, center 0 would not match",
            y[1]
        );
        // A matrix with the wrong dimension must be REJECTED, not read crooked.
        let ruim = j.replace("\"ng\": 2", "\"ng\": 3");
        assert!(
            PilotoForja::carregar(&escreve("pf_flat_ruim.json", &ruim)).is_none(),
            "accepted Cg with the wrong number of rows"
        );
    }

    /// Σφ gate and z saturation (degenerate channel leaving zero).
    #[test]
    fn portao_por_soma_phi_e_saturacao_de_z() {
        let c = 1;
        let mut j = json_piramide(
            c,
            &[vec![0.0]],
            &[vec![0.0]],
            [1.0, 1.0],
            [1.0, 1.0],
            &[[0.0, 0.0, 0.0]; 4],
            &[0.0],
            &[1.0e-12],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        j = j.replace("\"n_saida\": 3", "\"n_saida\": 3, \"thr_phi\": 1.0");
        assert_eq!(j.matches("thr_phi").count(), 1);
        let mut p = PilotoForja::carregar(&escreve("pf_portao.json", &j)).unwrap();
        assert!(
            (p.thr_phi - 1.0).abs() < 1e-9,
            "threshold read = {}",
            p.thr_phi
        );
        assert_eq!(
            p.canais_degenerados,
            vec![0],
            "sd=1e-12 must be marked degenerate"
        );
        // x = mu = 0 → z = 0 → phi all 1 → Σφ = 4 ≥ 1 → decides
        assert!(p.agir_com_portao(&[0.0]).is_some());
        assert_eq!(p.saturacoes_z, 0);
        // x = 1 with sd = 1e-12 → z = 1e12 → saturates at Z_MAX, φ→0 → Σφ < 1 → abstains
        assert!(p.agir_com_portao(&[1.0]).is_none());
        assert_eq!(p.saturacoes_z, 1, "z saturation was not counted");
        assert!(
            p.ultima_soma_phi.is_finite(),
            "Σφ became NaN/inf: {}",
            p.ultima_soma_phi
        );
        assert_eq!((p.decisoes, p.abstencoes), (1, 1));
    }

    /// SANITY AGAINST THE REAL FILE. Loads `piloto_v2.json` (the same one the forge
    /// generated) and compares the output against the reference generated in Python from
    /// the SAME file (`sanidade3.txt`, produced by `gerar_sanidade3.py` executing the 5
    /// steps of FORMATO.md in float64). Ticks 0, 1, 2 of `X.npy` + tick 17000, which
    /// is the value published in FORMATO.md (`[0.91202, -0.99821, -0.99239]`).
    /// Tolerance 1e-5: the runtime is f32, the reference is f64 (measured difference ≤ 2.8e-6).
    #[test]
    fn sanidade_contra_o_piloto_v2_real() {
        if !std::path::Path::new(V2).exists() || !std::path::Path::new(SANIDADE).exists() {
            eprintln!(
                "\n[WARNING] TEST NOT EXECUTED — missing {V2} or {SANIDADE}.\n\
                       This is the only test that proves the runtime against the real pilot;\n\
                       without /media/llm-data mounted it does not run. It is NOT a PASS.\n"
            );
            return;
        }
        let mut ref_txt = String::new();
        std::fs::File::open(SANIDADE)
            .unwrap()
            .read_to_string(&mut ref_txt)
            .unwrap();
        let mut p = PilotoForja::carregar(V2).expect("load the real piloto_v2.json");
        assert_eq!(p.base(), "piramide");
        assert_eq!((p.n_entradas(), p.n_features()), (192, 2048));
        assert_eq!(p.mult_por_tick(), 1024 * 192 + 2048 * 3);

        let mut tol = 1e-5f32;
        let mut x: Vec<f32> = Vec::new();
        let mut tick = usize::MAX;
        let mut conferidos = 0usize;
        for linha in ref_txt.lines() {
            let mut it = linha.split_whitespace();
            match it.next() {
                Some("TOL") => tol = it.next().unwrap().parse().unwrap(),
                Some("TICK") => tick = it.next().unwrap().parse().unwrap(),
                Some("X") => x = it.map(|t| t.parse().unwrap()).collect(),
                Some("Y") => {
                    let esperado: Vec<f32> = it.map(|t| t.parse().unwrap()).collect();
                    assert_eq!(
                        x.len(),
                        p.n_entradas(),
                        "tick {tick}: X with {} channels",
                        x.len()
                    );
                    assert_eq!(esperado.len(), 3);
                    let y = p.agir(&x);
                    let mut pior = 0.0f32;
                    for k in 0..3 {
                        let dif = (y[k] - esperado[k]).abs();
                        assert!(
                            dif <= tol,
                            "tick {tick} action {k}: rust {} vs python {} (dif {dif:.3e} > tol {tol:.0e})",
                            y[k],
                            esperado[k]
                        );
                        if dif > pior {
                            pior = dif;
                        }
                    }
                    // Σφ measured in the dump: 287.9 … 936.7. Outside that, something changed.
                    assert!(
                        p.ultima_soma_phi > 280.0 && p.ultima_soma_phi < 940.0,
                        "tick {tick}: Σφ = {} outside the measured range [287.9 ; 936.7]",
                        p.ultima_soma_phi
                    );
                    assert_eq!(
                        p.saturacoes_z, 0,
                        "tick {tick}: z saturated on data from the dump itself"
                    );
                    conferidos += 1;
                    println!(
                        "  tick {tick}: rust [{:.7}, {:.7}, {:.7}]  Σφ={:.2}  \
                              largest dif vs python(f64) = {pior:.2e}  (tol {tol:.0e}) ✓",
                        y[0], y[1], y[2], p.ultima_soma_phi
                    );
                }
                _ => {}
            }
        }
        assert_eq!(
            conferidos, 4,
            "expected 4 reference ticks, checked {conferidos}"
        );
        assert_eq!(
            p.canais_degenerados,
            vec![98, 114, 115, 116, 117, 124],
            "degenerate channels changed — the dump or the solve is not the same"
        );
    }
}
