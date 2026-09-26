//! `decide()` — SPEC §5: the 29-term linear score, the G1 mean-centering, the
//! masked 10-way service note, and the first-index-wins argmax that turns 19
//! numbers into one action index `esc ∈ [0, 18]`.
//!
//! SPEC = `/home/leonardo/projects/LLM/neuromod-lab/docs/71-spec-mundo-taxi.md`
//! §5, source `um_passo` / `Linear.extra` (`campo.py:469-490`).
//!
//! **Not touched by this module, on purpose:** `physics.rs`, `passengers.rs`
//! (T2/T4's files).

use crate::constants::{G, NF, NP, S};
use crate::genome::Genome;

/// SPEC §5 (`campo.py:471-481`, `um_passo`):
///
/// ```text
/// sc[g] = Σ_f F[g,f]·W[f]              f = 0..NF            (campo.py:471)
/// sc[g] = sc[g] − mean_h(sc[h])                             (campo.py:472, G1)
/// ss[a] = svn[a]·W[NF+a] + W[NF+S]      a = 0..S             (campo.py:473)
/// ss[a] = −1e9   where sv[a] ≤ 0                             (campo.py:475)
/// tudo  = concat(sc, ss)                                     (campo.py:476)
/// esc   = argmax(tudo), ties → LOWEST index                  (campo.py:481, SPEC §5)
/// ```
///
/// All slices are flat SoA over `n` beings:
/// - `f`: `n * G * NF`, `[being][option][feature]` — `senses.rs::observe`'s
///   `out_f`.
/// - `need`: `n * S`, `svn = sv·nec·paga` — T4 `services::service_state`'s
///   `out_need`.
/// - `avail`: `n * S`, `sv·paga` — used ONLY for the `<= 0` mask, NOT multiplied
///   into the note (SPEC §5: `ss = svn·W + bias` first, `ss[sv≤0] = −1e9`
///   SEPARATELY). T4's `service_state`'s `out_avail` is this array.
/// - `out_choice`: `n`, the winning index `0..=18` (`< G` ⇒ move; `≥ G` ⇒
///   service index `choice − G`).
/// - `out_margin`: `n`, the gap between the best score and the runner-up —
///   computed on VALUE alone, independent of which index the tie-break picked.
/// - `out_scores`: `n * (G + S)`, the full `tudo = [sc(9); ss(10)]` row — kept
///   so T5's identity gate can compare it directly against `nota_por_tique`
///   (SPEC §11), which records exactly this vector every tick
///   (`campo.py:484`, `m.nota_t[m.t] = tudo`).
///
/// **Reduction order (SPEC §5 "reduction order").** The dot product sums
/// `f = 0..NF` sequentially in `f32`, matching the plan's explicit instruction —
/// NOT an `f64` accumulator, which would change the f32 tie distribution the
/// identity gate measures (SPEC §5: "do not accumulate in f64 in v0").
#[allow(clippy::too_many_arguments)] // the arity is the plan's own T3 interface (docs/plano-mundo-taxi.md §T3)
pub fn decide(
    f: &[f32],
    need: &[f32],
    avail: &[f32],
    genome: &Genome,
    n: usize,
    out_choice: &mut [u32],
    out_margin: &mut [f32],
    out_scores: &mut [f32],
) {
    assert_eq!(genome.0.len(), NP, "genome must have NP={NP} weights");
    assert_eq!(f.len(), n * G * NF, "f must be sized n*G*NF");
    assert_eq!(need.len(), n * S, "need must be sized n*S");
    assert_eq!(avail.len(), n * S, "avail must be sized n*S");
    assert_eq!(out_choice.len(), n);
    assert_eq!(out_margin.len(), n);

    let w = &genome.0;
    let total = G + S;
    assert_eq!(
        out_scores.len(),
        n * total,
        "out_scores must be sized n*(G+S)"
    );
    let bias = w[NF + S];

    for b in 0..n {
        let mut sc = [0.0f32; G];
        for (g, sc_g) in sc.iter_mut().enumerate() {
            let base = (b * G + g) * NF;
            let mut s = 0.0f32;
            for feat in 0..NF {
                s += f[base + feat] * w[feat];
            }
            *sc_g = s;
        }
        // G1 (campo.py:472): subtract the mean of the 9 raw scores. This is the
        // step that makes SPEC §4.3's finding E58 true — a body column that is
        // identical across the 9 options shifts every `sc[g]` by the same
        // amount, which this subtraction removes exactly.
        let mean: f32 = sc.iter().sum::<f32>() / G as f32;
        for v in sc.iter_mut() {
            *v -= mean;
        }

        let mut ss = [0.0f32; S];
        for a in 0..S {
            let raw = need[b * S + a] * w[NF + a] + bias;
            ss[a] = if avail[b * S + a] <= 0.0 { -1e9 } else { raw };
        }

        let base_out = b * total;
        out_scores[base_out..base_out + G].copy_from_slice(&sc);
        out_scores[base_out + G..base_out + total].copy_from_slice(&ss);

        // argmax, first-index tie-break (SPEC §5: strict `>`, scan from idx 0 —
        // `torch.argmax` on the GPU does NOT guarantee this).
        let mut best_idx = 0usize;
        let mut best_val = out_scores[base_out];
        for i in 1..total {
            let v = out_scores[base_out + i];
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        out_choice[b] = best_idx as u32;

        // Margin: gap between the top two VALUES, independent of the tie-break
        // above (`campo.py:484`: `top = tudo.topk(2); margem = top[:,0]-top[:,1]`).
        let mut top1 = f32::NEG_INFINITY;
        let mut top2 = f32::NEG_INFINITY;
        for i in 0..total {
            let v = out_scores[base_out + i];
            if v > top1 {
                top2 = top1;
                top1 = v;
            } else if v > top2 {
                top2 = v;
            }
        }
        out_margin[b] = top1 - top2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_services_masked(n: usize) -> (Vec<f32>, Vec<f32>) {
        (vec![0.0; n * S], vec![0.0; n * S]) // avail <= 0 everywhere -> ss = -1e9
    }

    #[test]
    fn the_common_mode_of_the_body_cancels_after_centering() {
        // E58 (SPEC §4.3): body columns (10..NF) are IDENTICAL across the 9
        // options; movement columns (0..10) are distinct per option.
        let n = 1;
        let mut f = vec![0.0f32; n * G * NF];
        for g in 0..G {
            for k in 0..10 {
                f[g * NF + k] = (g * 10 + k) as f32;
            }
            for k in 10..NF {
                f[g * NF + k] = (k as f32) * 0.37; // same value for every g
            }
        }
        let mut genome_a = vec![0.0f32; NP];
        let mut genome_b = vec![0.0f32; NP];
        for k in 0..10 {
            genome_a[k] = 0.1 * (k as f32 + 1.0);
            genome_b[k] = genome_a[k]; // SAME movement weights
        }
        for k in 10..NF {
            genome_a[k] = 0.5;
            genome_b[k] = 99.0; // WILDLY different body weights
        }
        let (need, avail) = all_services_masked(n);

        let mut choice_a = [0u32; 1];
        let mut margin_a = [0.0f32; 1];
        let mut scores_a = vec![0.0f32; n * (G + S)];
        decide(
            &f,
            &need,
            &avail,
            &Genome(genome_a),
            n,
            &mut choice_a,
            &mut margin_a,
            &mut scores_a,
        );

        let mut choice_b = [0u32; 1];
        let mut margin_b = [0.0f32; 1];
        let mut scores_b = vec![0.0f32; n * (G + S)];
        decide(
            &f,
            &need,
            &avail,
            &Genome(genome_b),
            n,
            &mut choice_b,
            &mut margin_b,
            &mut scores_b,
        );

        for g in 0..G {
            assert!(
                (scores_a[g] - scores_b[g]).abs() < 1e-3,
                "sc[{g}] must not depend on the body weights (E58): a={} b={}",
                scores_a[g],
                scores_b[g]
            );
        }
        assert_eq!(choice_a[0], choice_b[0]);
    }

    #[test]
    fn a_tie_in_the_scores_picks_the_lowest_index() {
        let n = 1;
        let mut f = vec![0.0f32; n * G * NF];
        // Only column 0 carries weight; option 1 and option 4 tie for the max.
        let raw = [1.0, 5.0, 2.0, 0.0, 5.0, -3.0, 4.0, 1.0, 0.0];
        for g in 0..G {
            f[g * NF] = raw[g];
        }
        let mut genome = vec![0.0f32; NP];
        genome[0] = 1.0;
        let (need, avail) = all_services_masked(n);

        let mut choice = [0u32; 1];
        let mut margin = [0.0f32; 1];
        let mut scores = vec![0.0f32; n * (G + S)];
        decide(
            &f,
            &need,
            &avail,
            &Genome(genome),
            n,
            &mut choice,
            &mut margin,
            &mut scores,
        );

        assert_eq!(
            choice[0], 1,
            "tie between options 1 and 4 must pick the lower index"
        );
        assert!(
            margin[0].abs() < 1e-5,
            "a tied top score must give a zero margin"
        );
    }

    #[test]
    fn masked_services_never_win() {
        let n = 1;
        let mut f = vec![0.0f32; n * G * NF];
        for g in 0..G {
            f[g * NF] = -100.0; // every movement option scores terribly...
        }
        let mut genome = vec![0.0f32; NP];
        genome[0] = 1.0;
        for a in 0..S {
            genome[NF + a] = 1000.0; // ...and services would score enormously if allowed
        }
        genome[NF + S] = 1000.0;
        let need = vec![1.0f32; n * S];
        let avail = vec![0.0f32; n * S]; // every service masked out (SPEC §5: sv<=0 -> -1e9)

        let mut choice = [0u32; 1];
        let mut margin = [0.0f32; 1];
        let mut scores = vec![0.0f32; n * (G + S)];
        decide(
            &f,
            &need,
            &avail,
            &Genome(genome),
            n,
            &mut choice,
            &mut margin,
            &mut scores,
        );

        assert!(
            choice[0] < G as u32,
            "with every service masked only a movement option can win"
        );
    }

    #[test]
    fn the_margin_is_the_gap_between_the_best_and_the_runner_up() {
        let n = 1;
        let mut f = vec![0.0f32; n * G * NF];
        let raw = [10.0, 3.0, 7.0, -2.0, 1.0, 0.0, 6.5, 2.0, -5.0];
        for g in 0..G {
            f[g * NF] = raw[g];
        }
        let mut genome = vec![0.0f32; NP];
        genome[0] = 1.0;
        let (need, avail) = all_services_masked(n);

        let mut choice = [0u32; 1];
        let mut margin = [0.0f32; 1];
        let mut scores = vec![0.0f32; n * (G + S)];
        decide(
            &f,
            &need,
            &avail,
            &Genome(genome),
            n,
            &mut choice,
            &mut margin,
            &mut scores,
        );

        // Best raw = 10.0 (idx 0), runner-up raw = 7.0 (idx 2); centering
        // subtracts the SAME mean from every sc, so the gap between them
        // survives unchanged.
        assert_eq!(choice[0], 0);
        assert!(
            (margin[0] - 3.0).abs() < 1e-3,
            "expected gap 3.0, got {}",
            margin[0]
        );
    }
}
