/// Estatísticas descritivas — média, variância, percentis, histograma.
///
/// Todas as funções operam em slices f32 e não alocam além dos outputs explícitos.
/// São usadas pelo módulo `analysis/activation.rs` (E4) para monitorar neurônios.

/// Média aritmética de um slice. Retorna 0.0 se vazio.
pub fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}

/// Variância da amostra (divisor N-1). Retorna 0.0 se len < 2.
///
/// N-1 (Bessel) reduz viés para populações pequenas. Nos cálculos de PR usa-se
/// N (veja brain_capacity.rs), mas aqui o contexto é estatística de amostra.
pub fn variance(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / (xs.len() - 1) as f32
}

/// Desvio padrão amostral (raiz da variância N-1).
pub fn std_dev(xs: &[f32]) -> f32 {
    variance(xs).sqrt()
}

/// Desvio-padrão POPULACIONAL (divisor N, sem correção de Bessel), via
/// soma-de-quadrados: `sqrt(E[x²] - E[x]²)`.
///
/// Espelha `brain.wgsl` ETAPA 2 (`params.norm_l2_by_std`, P1.5,
/// `docs/PLANO_EVOLUCAO.md`) — o shader calcula `sum`/`sum_sq` num único laço
/// (sem alocar um segundo vetor de desvios) porque roda em WGSL sem heap;
/// esta função replica o MESMO algoritmo, não `std_dev`/N-1, para que o teste
/// em `math::activations` compare a mesma fórmula que está no shader, não uma
/// aproximação. `max(0.0, ...)` porque a variância por soma-de-quadrados pode
/// dar um epsilon negativo por cancelamento de ponto flutuante quando os
/// valores são quase iguais — mesma guarda que o shader tem.
///
/// Retorna 0.0 se vazio (não 0.0/0.0 = NaN).
pub fn population_std_dev(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f32;
    let sum: f32 = xs.iter().sum();
    let sum_sq: f32 = xs.iter().map(|&x| x * x).sum();
    let m = sum / n;
    let variance = (sum_sq / n - m * m).max(0.0);
    variance.sqrt()
}

/// Percentil p ∈ [0.0, 100.0] usando interpolação linear (método R7 / Excel).
///
/// Ordena o slice in-place — passe uma cópia se precisar preservar a ordem.
/// Retorna 0.0 se vazio.
pub fn percentile(xs: &mut [f32], p: f32) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len() as f32;
    // Índice real via método R7: idx = p/100 * (n-1)
    let idx = (p / 100.0) * (n - 1.0);
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(xs.len() - 1);
    let frac = idx - lo as f32;
    xs[lo] + frac * (xs[hi] - xs[lo])
}

/// Histograma de `n_bins` bins uniformes cobrindo [min, max] dos dados.
///
/// Retorna um Vec<u32> de comprimento `n_bins`.
/// Se todos os valores forem iguais, todos caem no bin central (floor).
/// Se `n_bins == 0` ou o slice for vazio, retorna Vec vazio.
pub fn histogram(xs: &[f32], n_bins: usize) -> Vec<u32> {
    if n_bins == 0 || xs.is_empty() {
        return vec![];
    }
    let min = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    let mut bins = vec![0u32; n_bins];
    for &v in xs {
        let idx = if range < 1e-30 {
            // Todos os valores iguais: vai para o bin do meio
            n_bins / 2
        } else {
            let frac = (v - min) / range;
            // frac ∈ [0.0, 1.0]; mapeia para [0, n_bins-1]
            ((frac * n_bins as f32).floor() as usize).min(n_bins - 1)
        };
        bins[idx] += 1;
    }
    bins
}

/// Mediana do slice (percentile 50.0).
///
/// Ordena in-place — passe uma cópia se necessário.
pub fn median(xs: &mut [f32]) -> f32 {
    percentile(xs, 50.0)
}

// ---------------------------------------------------------------------------
// Testes unitários
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_de_vetor_uniforme() {
        assert_eq!(mean(&[2.0; 5]), 2.0);
    }

    #[test]
    fn mean_de_vazio_e_zero() {
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn mean_de_sequencia_simetrica() {
        // [-2, -1, 0, 1, 2] → média = 0
        let xs: Vec<f32> = (-2..=2).map(|i| i as f32).collect();
        assert!((mean(&xs) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn variance_de_constante_e_zero() {
        assert_eq!(variance(&[3.0; 10]), 0.0);
    }

    #[test]
    fn variance_conhecida() {
        // {0, 1, 2, 3, 4} → média 2, variância amostral = 10/4 = 2.5
        let xs = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let v = variance(&xs);
        assert!((v - 2.5).abs() < 1e-5, "variância = {}", v);
    }

    #[test]
    fn variance_vazio_e_zero() {
        assert_eq!(variance(&[]), 0.0);
    }

    #[test]
    fn percentile_min_e_max() {
        let mut xs = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&mut xs, 0.0) - 1.0).abs() < 1e-5);
        assert!((percentile(&mut xs, 100.0) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn percentile_50_de_impar() {
        let mut xs = [3.0_f32, 1.0, 2.0]; // ordenado: [1, 2, 3], median = 2.0
        let med = percentile(&mut xs, 50.0);
        assert!((med - 2.0).abs() < 1e-5, "mediana = {}", med);
    }

    #[test]
    fn histogram_soma_total() {
        // Todos os valores devem estar num bin
        let xs = [0.0_f32, 0.5, 1.0];
        let h = histogram(&xs, 5);
        assert_eq!(h.iter().sum::<u32>(), 3);
    }

    #[test]
    fn histogram_distribuicao_uniforme() {
        // 100 valores igualmente espaçados em [0, 1] → 10 bins com ~10 cada
        let xs: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let h = histogram(&xs, 10);
        assert_eq!(h.len(), 10);
        assert_eq!(h.iter().sum::<u32>(), 100);
        // Cada bin deve ter aproximadamente 10 ±3 (tolerante a bordas)
        for (i, &count) in h.iter().enumerate() {
            assert!(
                count >= 7 && count <= 13,
                "bin {} tem {} (esperado ~10)",
                i,
                count
            );
        }
    }

    #[test]
    fn histogram_valores_iguais_nao_trava() {
        let xs = [5.0_f32; 20];
        let h = histogram(&xs, 8);
        assert_eq!(h.iter().sum::<u32>(), 20, "todos devem estar em algum bin");
    }

    #[test]
    fn histogram_vazio_retorna_vazio() {
        let h = histogram(&[], 5);
        assert!(h.is_empty());
    }

    #[test]
    fn histogram_n_bins_zero_retorna_vazio() {
        let h = histogram(&[1.0, 2.0, 3.0], 0);
        assert!(h.is_empty());
    }

    #[test]
    fn std_dev_de_constante_e_zero() {
        assert_eq!(std_dev(&[7.0; 5]), 0.0);
    }

    #[test]
    fn population_std_dev_de_constante_e_zero() {
        assert_eq!(population_std_dev(&[7.0; 48]), 0.0);
    }

    #[test]
    fn population_std_dev_vazio_e_zero() {
        assert_eq!(population_std_dev(&[]), 0.0);
    }

    #[test]
    fn population_std_dev_conhecido() {
        // {0,1,2,3} → média 1.5, variância POPULACIONAL (÷N) = 1.25, desvio ≈ 1.1180
        let xs = [0.0_f32, 1.0, 2.0, 3.0];
        let d = population_std_dev(&xs);
        assert!(
            (d - 1.25f32.sqrt()).abs() < 1e-5,
            "desvio populacional = {}",
            d
        );
    }

    #[test]
    fn population_std_dev_e_menor_que_amostral_para_mesmo_conjunto() {
        // Divisor N < N-1 → desvio populacional <= amostral (N>1).
        let xs = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        assert!(population_std_dev(&xs) < std_dev(&xs));
    }

    #[test]
    fn median_de_tres_elementos() {
        let mut xs = [5.0_f32, 1.0, 3.0];
        assert!((median(&mut xs) - 3.0).abs() < 1e-5);
    }
}
