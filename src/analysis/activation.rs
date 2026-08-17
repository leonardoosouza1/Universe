//! Métricas de ativação por neurônio + guardas de runtime (E4).
//!
//! REFACTOR_PLAN.md §E4 / MATH_INVENTORY.md lacunas 1-4, 6, 12: o projeto tinha ZERO
//! visibilidade sobre o que acontece DENTRO do cérebro rápido (L1=96, L2=48) — só a
//! saída final (`final_intent`) e as 3 saídas do lento (`participation_ratio` em
//! `main.rs`). Este módulo fecha essa lacuna: recebe amostras já lidas da GPU (o
//! caller, `gpu_evolve/context.rs`, é quem sabe como ler `buf_cfc`/`buf_intent`) e
//! devolve estatística por neurônio + alarmes prontos para o relatório.
//!
//! # Por que não duplicar os limiares aqui
//!
//! O REFACTOR_PLAN original propunha um `pub mod thresholds` LOCAL a este arquivo.
//! O E1 já criou `crate::math::thresholds` com os mesmos limiares (nomes diferentes,
//! mesma função) — ter dois lugares com o "limiar de neurônio morto" é exatamente o
//! tipo de duplicidade que gera divergência silenciosa (um lugar muda, o outro não).
//! Este módulo importa `math::thresholds` como fonte única.
//!
//! # Dois cortes, dois níveis
//!
//! Há uma distinção que vale a pena deixar explícita porque os nomes são parecidos:
//! - `DEAD_EPS`/`SATURATION_CUTOFF` — corte POR AMOSTRA (este `h` neste tick está
//!   morto/saturado?). Produz `NeuronStats::dead_pct`/`saturated_pct`, uma fração
//!   contínua por neurônio.
//! - `DEAD_NEURONS_WARN_PCT`/`SATURATED_NEURONS_WARN_PCT` — corte POR CAMADA (que
//!   fração dos NEURÔNIOS da camada está consistentemente morta/saturada?). Um
//!   neurônio só entra nessa contagem se sua própria fração por-amostra passar de
//!   `NEURON_DEAD_CLASSIFY_PCT`/`NEURON_SATURATED_CLASSIFY_PCT` (ver comentário nas
//!   constantes) — não acumular os dois níveis faria QUALQUER neurônio que morresse
//!   uma vez, mesmo que raro, contar como "morto" na fração da camada.

use crate::math::{linalg, quant, stats, thresholds};

/// Bins do histograma agregado da camada (REFACTOR_PLAN.md §E4: "histograma 32 bins").
pub const HISTOGRAM_BINS: usize = 32;

/// Corte de saturação POR AMOSTRA: |h| > 0.99 é considerado saturado.
///
/// Distinto de `thresholds::SATURATED_NEURONS_WARN_PCT` (fração da CAMADA saturada
/// que dispara alarme). `tanh(2.65) ≈ 0.99` — MATH_INVENTORY.md lacuna 3.
const SATURATION_CUTOFF: f32 = 0.99;

/// Um neurônio só entra na contagem de "neurônios mortos da camada" se estiver
/// morto em pelo menos esta fração das amostras — não a primeira vez que passa por
/// zero. 90% é conservador: mesmo um canal que "descansa" boa parte do tempo (ex.:
/// detector de evento raro) sobra fora desta classificação.
const NEURON_DEAD_CLASSIFY_PCT: f32 = 90.0;
/// Idem para saturação: neurônio preso no teto em ≥90% das amostras.
const NEURON_SATURATED_CLASSIFY_PCT: f32 = 90.0;

/// Estatísticas de um neurônio, coletadas sobre uma janela de amostras (ver
/// `collect_activation_stats` para a semântica exata de "amostra" — mistura tempo e
/// população, não é uma trajetória de ser único).
#[derive(Debug, Clone)]
pub struct NeuronStats {
    pub neuron_id: usize,
    pub mean: f32,
    pub sigma: f32,
    pub p5: f32,
    pub p50: f32,
    pub p95: f32,
    /// Fração de amostras com `|h| < DEAD_EPS` — 0.0 a 100.0.
    pub dead_pct: f32,
    /// Fração de amostras com `|h| > 0.99` — 0.0 a 100.0.
    pub saturated_pct: f32,
    /// SNR de quantizar este neurônio para i8 em escala de ativação (127), em dB.
    pub snr_i8: f32,
}

/// Estatísticas agregadas de uma camada inteira.
#[derive(Debug, Clone)]
pub struct LayerStats {
    pub layer_name: String,
    /// Número de linhas de amostra usadas (após descartar NaN/Inf).
    pub n_samples: usize,
    pub neurons: Vec<NeuronStats>,
    /// Maior valor singular dos pesos de entrada da camada (0.0 se `w_rec` não foi
    /// fornecido). Ver `collect_activation_stats` para como é calculado a partir de
    /// uma matriz retangular (L1 não é quadrada: 192→96).
    /// `None` quando a camada nao teve matriz de pesos entregue — nao confundir com zero.
    pub spectral_radius: Option<f32>,
    /// Rank efetivo (95% da energia espectral) dos mesmos pesos. 0 se `w_rec` ausente.
    /// `None` quando nao medido (ver [`Self::spectral_radius`]).
    pub effective_rank: Option<usize>,
    /// Entropia da política (nats), só quando `policy_actions` foi passado.
    pub policy_entropy: Option<f32>,
    /// Histograma agregado (todas as amostras de todos os neurônios), 32 bins.
    pub histogram: Vec<u32>,
}

/// Tipo de violação de limiar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlarmKind {
    DeadNeurons,
    Saturated,
    SpectralRadiusHigh,
    SnrLow,
    NanDetected,
    PolicyEntropyLow,
}

/// Alarme: violação de limiar detectada nesta geração.
#[derive(Debug, Clone)]
pub struct ActivationAlarm {
    pub layer: String,
    pub kind: AlarmKind,
    pub value: f32,
    pub threshold: f32,
}

/// Espectro de valores singulares de `w` (`n_neurons × in_dim`, linha-maior) via
/// Gram + Jacobi: raio espectral (maior valor singular, por power-iteration — barato
/// e é o que interessa para o alarme de instabilidade) e rank efetivo (precisa do
/// espectro inteiro, por isso o Jacobi completo).
///
/// Retorna `(0.0, 0)` se `w` não tiver o formato esperado.
fn spectral_summary(w: &[f32], n_neurons: usize) -> (f32, usize) {
    if n_neurons == 0 || w.is_empty() || w.len() % n_neurons != 0 {
        return (0.0, 0);
    }
    let in_dim = w.len() / n_neurons;
    if in_dim == 0 {
        return (0.0, 0);
    }
    // `linalg::largest_singular_value` já faz gram+power-iteration+sqrt (fonte única —
    // é a MESMA conta que `gpu::evolution::Genome::random` usa pra normalizar na
    // inicialização, P0.5). O rank efetivo precisa do espectro inteiro, então o Gram
    // é recomputado aqui uma vez a mais para `eigenvalues_symmetric`; matrizes de L1
    // (96×192) tornam isso desprezível frente ao custo do resto do relatório.
    let spectral_radius = linalg::largest_singular_value(w, n_neurons, in_dim);
    let gram = linalg::gram_matrix(w, n_neurons, in_dim);
    let eigs = linalg::eigenvalues_symmetric(&gram, n_neurons);
    let mut singular_vals: Vec<f32> = eigs.iter().map(|&e| e.max(0.0).sqrt()).collect();
    singular_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let effective_rank = linalg::effective_rank(&singular_vals, 0.95);

    (spectral_radius, effective_rank)
}

/// Entropia de Shannon (nats) da distribuição conjunta de (throttle, steer, brake),
/// binada em 5×5×5 = 125 células sobre `[0,1] × [-1,1] × [0,1]`.
///
/// Binar as TRÊS saídas juntas (não uma entropia por canal) é intencional: um
/// volante cravado com acelerador variando ainda seria "baixa entropia" para o que
/// importa (o carro faz sempre a mesma curva), e olhar só o canal do volante
/// esconderia isso se as outras duas variassem.
fn policy_entropy_nats(actions: &[[f32; 3]]) -> f32 {
    const BINS: usize = 5;
    if actions.is_empty() {
        return 0.0;
    }
    let mut counts = std::collections::HashMap::<(usize, usize, usize), u32>::new();
    for a in actions {
        let bt = ((a[0].clamp(0.0, 1.0) * BINS as f32) as usize).min(BINS - 1);
        let bs = (((a[1].clamp(-1.0, 1.0) + 1.0) * 0.5 * BINS as f32) as usize).min(BINS - 1);
        let bb = ((a[2].clamp(0.0, 1.0) * BINS as f32) as usize).min(BINS - 1);
        *counts.entry((bt, bs, bb)).or_insert(0) += 1;
    }
    let total = actions.len() as f32;
    let mut h = 0.0f32;
    for &c in counts.values() {
        let p = c as f32 / total;
        h -= p * p.ln();
    }
    h
}

/// Coleta estatísticas de ativação de um conjunto de amostras.
///
/// `samples[k]` é uma linha de `n_neurons` valores — CADA LINHA pode vir de um ser
/// diferente e/ou de um tick diferente. Este módulo não distingue as duas coisas: a
/// estatística por neurônio (média, percentis, %morto, %saturado) é sobre o
/// CONJUNTO de amostras, misturando população e tempo, no mesmo espírito de como
/// `[lento]` (main.rs) já mede `participation_ratio` sobre amostras de vários seres
/// em vários ticks. Linhas com NaN/Inf são descartadas ANTES de chegar aqui (ver
/// guarda de runtime em `gpu_evolve/context.rs::tick()`), então esta função assume
/// que `samples` já está limpo — mas ainda audita e reporta um alarme se algo
/// escapou (camada de defesa dupla, custo desprezível).
///
/// `w_rec`: pesos de entrada da camada, linha-maior `[n_neurons × in_dim]` (o nome
/// "recorrente" é herança do REFACTOR_PLAN original — L1 não é recorrente no sentido
/// de RNN, mas a soma "CfC(canais 0..96) ++ KAN(canais 96..192) → L1" faz o mesmo
/// papel que W_rec faria numa análise de estabilidade: é a matriz cujo maior valor
/// singular decide se o sinal amplifica ou atenua). `None` quando a camada não tem
/// pesos de entrada acessíveis aqui (ex.: SlowBrain, cujo `W` mora só na GPU).
///
/// `policy_actions`: (throttle, steer, brake) por amostra, na MESMA ordem/stride das
/// linhas de `samples` — usado só para `policy_entropy`; passe `None` para camadas
/// que não são a saída motora.
pub fn collect_activation_stats(
    layer_name: &str,
    samples: &[Vec<f32>],
    w_rec: Option<&[f32]>,
    n_neurons: usize,
    policy_actions: Option<&[[f32; 3]]>,
) -> (LayerStats, Vec<ActivationAlarm>) {
    let mut alarms: Vec<ActivationAlarm> = Vec::new();

    // Defesa dupla: qualquer NaN/Inf que tenha escapado do filtro do caller é
    // reportado (alarme) e a linha é excluída da estatística — sem isto uma única
    // amostra ruim contaminaria mean/percentile/histogram da camada inteira.
    let mut clean: Vec<&Vec<f32>> = Vec::with_capacity(samples.len());
    let mut n_nan_rows = 0usize;
    for row in samples {
        if row.iter().any(|v| !v.is_finite()) {
            n_nan_rows += 1;
        } else {
            clean.push(row);
        }
    }
    if n_nan_rows > 0 {
        alarms.push(ActivationAlarm {
            layer: layer_name.to_string(),
            kind: AlarmKind::NanDetected,
            value: n_nan_rows as f32,
            threshold: 0.0,
        });
    }

    let n_samples = clean.len();
    let mut neurons: Vec<NeuronStats> = Vec::with_capacity(n_neurons);
    let mut all_vals: Vec<f32> = Vec::with_capacity(n_samples * n_neurons.max(1));

    for j in 0..n_neurons {
        let col: Vec<f32> = clean.iter().filter_map(|row| row.get(j).copied()).collect();
        all_vals.extend_from_slice(&col);

        let mean = stats::mean(&col);
        let sigma = stats::std_dev(&col);
        let mut for_percentile = col.clone();
        let p5 = stats::percentile(&mut for_percentile, 5.0);
        let p50 = stats::percentile(&mut for_percentile, 50.0);
        let p95 = stats::percentile(&mut for_percentile, 95.0);
        let dead_pct = if col.is_empty() {
            0.0
        } else {
            100.0
                * col
                    .iter()
                    .filter(|&&v| v.abs() < thresholds::DEAD_EPS)
                    .count() as f32
                / col.len() as f32
        };
        let saturated_pct = if col.is_empty() {
            0.0
        } else {
            100.0 * col.iter().filter(|&&v| v.abs() > SATURATION_CUTOFF).count() as f32
                / col.len() as f32
        };
        let snr_i8 = quant::snr_db_act(&col);

        neurons.push(NeuronStats {
            neuron_id: j,
            mean,
            sigma,
            p5,
            p50,
            p95,
            dead_pct,
            saturated_pct,
            snr_i8,
        });
    }

    let histogram = stats::histogram(&all_vals, HISTOGRAM_BINS);

    // `None` = a camada nao teve matriz de pesos entregue a analise. Isso NAO e o mesmo
    // que "raio espectral zero": antes este ramo devolvia (0.0, 0) e o relatorio imprimia
    // "raio_espectral=0.000 rank_efetivo=0/48", que se le como colapso total da camada.
    // Custou uma conclusao invertida em 2026-08-13 (L2 parecia morta e estava apenas nao
    // medida). Ausencia de medida agora viaja como Option ate o relatorio, que imprime n/a.
    let (spectral_radius, effective_rank) = match w_rec {
        Some(w) => {
            let (r, k) = spectral_summary(w, n_neurons);
            (Some(r), Some(k))
        }
        None => (None, None),
    };

    let policy_entropy = policy_actions.and_then(|a| {
        if a.is_empty() {
            None
        } else {
            Some(policy_entropy_nats(a))
        }
    });

    // ── Alarmes de camada (fração de NEURÔNIOS, não de amostras — ver doc do módulo) ──
    if n_neurons > 0 {
        let dead_frac_layer = 100.0
            * neurons
                .iter()
                .filter(|s| s.dead_pct >= NEURON_DEAD_CLASSIFY_PCT)
                .count() as f32
            / n_neurons as f32;
        if dead_frac_layer > thresholds::DEAD_NEURONS_WARN_PCT {
            alarms.push(ActivationAlarm {
                layer: layer_name.to_string(),
                kind: AlarmKind::DeadNeurons,
                value: dead_frac_layer,
                threshold: thresholds::DEAD_NEURONS_WARN_PCT,
            });
        }
        let sat_frac_layer = 100.0
            * neurons
                .iter()
                .filter(|s| s.saturated_pct >= NEURON_SATURATED_CLASSIFY_PCT)
                .count() as f32
            / n_neurons as f32;
        if sat_frac_layer > thresholds::SATURATED_NEURONS_WARN_PCT {
            alarms.push(ActivationAlarm {
                layer: layer_name.to_string(),
                kind: AlarmKind::Saturated,
                value: sat_frac_layer,
                threshold: thresholds::SATURATED_NEURONS_WARN_PCT,
            });
        }
    }

    // So alarma sobre o que foi de fato medido: camada sem matriz entregue nao gera
    // alarme nenhum (nem falso positivo de "zero", nem falso negativo silencioso).
    if let Some(sr) = spectral_radius {
        if sr > thresholds::SPECTRAL_MAX {
            alarms.push(ActivationAlarm {
                layer: layer_name.to_string(),
                kind: AlarmKind::SpectralRadiusHigh,
                value: sr,
                threshold: thresholds::SPECTRAL_MAX,
            });
        }
    }

    if let Some(min_snr) = neurons
        .iter()
        .map(|s| s.snr_i8)
        .filter(|v| v.is_finite())
        .fold(None, |acc: Option<f32>, v| {
            Some(acc.map_or(v, |a| a.min(v)))
        })
    {
        if min_snr < thresholds::SNR_I8_MIN {
            alarms.push(ActivationAlarm {
                layer: layer_name.to_string(),
                kind: AlarmKind::SnrLow,
                value: min_snr,
                threshold: thresholds::SNR_I8_MIN,
            });
        }
    }

    if let Some(pe) = policy_entropy {
        if pe < thresholds::POLICY_ENTROPY_MIN {
            alarms.push(ActivationAlarm {
                layer: layer_name.to_string(),
                kind: AlarmKind::PolicyEntropyLow,
                value: pe,
                threshold: thresholds::POLICY_ENTROPY_MIN,
            });
        }
    }

    (
        LayerStats {
            layer_name: layer_name.to_string(),
            n_samples,
            neurons,
            spectral_radius,
            effective_rank,
            policy_entropy,
            histogram,
        },
        alarms,
    )
}

/// Imprime relatório de stats para stderr (não bloqueia, não tranca Mutex — cada
/// chamada só lê os `Vec`/`struct` que já foram montados por `collect_activation_stats`).
pub fn print_stats_report(stats: &LayerStats, alarms: &[ActivationAlarm]) {
    eprintln!(
        "[ativacao:{}] amostras={} neuronios={} raio_espectral={} rank_efetivo={} entropia_politica={}",
        stats.layer_name,
        stats.n_samples,
        stats.neurons.len(),
        stats
            .spectral_radius
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        stats
            .effective_rank
            .map(|k| format!("{k}/{}", stats.neurons.len()))
            .unwrap_or_else(|| "n/a (pesos nao entregues)".to_string()),
        stats
            .policy_entropy
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
    );
    for a in alarms {
        eprintln!(
            "[ativacao] ALARME {} {:?}: valor={:.3} limiar={:.3}",
            a.layer, a.kind, a.value, a.threshold,
        );
    }
}

// ---------------------------------------------------------------------------
// Testes unitários
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn linha(vs: &[f32]) -> Vec<f32> {
        vs.to_vec()
    }

    #[test]
    fn neuronio_morto_detectado() {
        // 3 neurônios, 10 amostras: neurônio 0 sempre ~0 (morto), 1 e 2 variam.
        let mut samples = Vec::new();
        for i in 0..10 {
            let t = i as f32 / 9.0;
            samples.push(linha(&[1e-6, t, -t]));
        }
        let (stats, _alarms) = collect_activation_stats("teste", &samples, None, 3, None);
        assert!(
            stats.neurons[0].dead_pct > 99.0,
            "neurônio 0 devia estar ~100% morto, {}",
            stats.neurons[0].dead_pct
        );
        assert!(stats.neurons[1].dead_pct < 50.0);
    }

    #[test]
    fn neuronio_saturado_detectado() {
        let mut samples = Vec::new();
        for _ in 0..10 {
            samples.push(linha(&[0.999, 0.0]));
        }
        let (stats, alarms) = collect_activation_stats("teste", &samples, None, 2, None);
        assert!(stats.neurons[0].saturated_pct > 99.0);
        assert!(
            alarms.iter().any(|a| a.kind == AlarmKind::Saturated),
            "camada com 1/2 neurônios saturados (50% > 30% warn) devia alarmar: {:?}",
            alarms
        );
    }

    #[test]
    fn nan_na_amostra_vira_alarme_e_e_excluido() {
        let samples = vec![
            linha(&[0.1, 0.2]),
            linha(&[f32::NAN, 0.2]),
            linha(&[0.3, 0.4]),
        ];
        let (stats, alarms) = collect_activation_stats("teste", &samples, None, 2, None);
        assert_eq!(
            stats.n_samples, 2,
            "a linha com NaN deve ser excluída da contagem"
        );
        assert!(alarms.iter().any(|a| a.kind == AlarmKind::NanDetected));
    }

    #[test]
    fn camada_vazia_nao_gera_pane() {
        let (stats, alarms) = collect_activation_stats("vazia", &[], None, 4, None);
        assert_eq!(stats.n_samples, 0);
        assert_eq!(stats.neurons.len(), 4);
        assert!(alarms.iter().all(|a| a.kind != AlarmKind::NanDetected));
    }

    #[test]
    fn spectral_summary_matriz_ortogonal_normalizada() {
        // W = I (4×4): valores singulares todos = 1.0 → raio espectral = 1.0,
        // rank efetivo = 4 (energia espalhada igualmente).
        let n = 4usize;
        let mut w = vec![0.0f32; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        let (sr, er) = spectral_summary(&w, n);
        assert!(
            (sr - 1.0).abs() < 0.05,
            "raio espectral esperado ~1.0, obteve {sr}"
        );
        assert_eq!(er, n);
    }

    #[test]
    fn spectral_summary_matriz_retangular_nao_quadrada() {
        // W (2×4): duas linhas idênticas ortogonais a duas colunas de zero.
        // rows=2 < cols=4 — testa o caminho "L1 não é quadrada" (192→96 na prática).
        let w = vec![2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let (sr, er) = spectral_summary(&w, 2);
        assert!(
            (sr - 3.0).abs() < 0.05,
            "maior valor singular esperado 3.0, obteve {sr}"
        );
        assert_eq!(er, 2);
    }

    #[test]
    fn spectral_radius_alto_dispara_alarme() {
        // Uma linha com norma 10 garante valor singular > SPECTRAL_MAX(1.0).
        let mut w = vec![0.0f32; 3 * 3];
        w[0] = 10.0;
        let samples = vec![linha(&[0.0, 0.0, 0.0])];
        let (stats, alarms) = collect_activation_stats("teste", &samples, Some(&w), 3, None);
        assert!(stats.spectral_radius.unwrap() > thresholds::SPECTRAL_MAX);
        assert!(
            alarms
                .iter()
                .any(|a| a.kind == AlarmKind::SpectralRadiusHigh)
        );
    }

    #[test]
    fn entropia_politica_baixa_com_acao_constante() {
        let acoes: Vec<[f32; 3]> = (0..50).map(|_| [0.8_f32, 0.63, 0.0]).collect();
        let mut samples = Vec::new();
        for _ in 0..50 {
            samples.push(linha(&[0.0]));
        }
        let (stats, alarms) = collect_activation_stats("L2", &samples, None, 1, Some(&acoes));
        let pe = stats
            .policy_entropy
            .expect("deve calcular entropia quando policy_actions é Some");
        assert!(
            pe < 0.05,
            "ação constante deve ter entropia ~0, obteve {pe}"
        );
        assert!(alarms.iter().any(|a| a.kind == AlarmKind::PolicyEntropyLow));
    }

    #[test]
    fn entropia_politica_alta_com_acao_uniforme() {
        // Varre throttle/steer/brake por toda a faixa — entropia deve ficar bem
        // acima do limiar (POLICY_ENTROPY_MIN = 0.5 nats).
        let mut acoes = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let t = i as f32 / 4.0;
                    let s = (j as f32 / 4.0) * 2.0 - 1.0;
                    let b = k as f32 / 4.0;
                    acoes.push([t, s, b]);
                }
            }
        }
        let samples: Vec<Vec<f32>> = (0..acoes.len()).map(|_| linha(&[0.0])).collect();
        let (stats, _alarms) = collect_activation_stats("L2", &samples, None, 1, Some(&acoes));
        let pe = stats.policy_entropy.unwrap();
        assert!(
            pe > thresholds::POLICY_ENTROPY_MIN,
            "125 ações distintas uniformes devem ter entropia > {}, obteve {}",
            thresholds::POLICY_ENTROPY_MIN,
            pe
        );
    }

    #[test]
    fn histograma_tem_32_bins_e_soma_total() {
        let samples: Vec<Vec<f32>> = (0..20)
            .map(|i| linha(&[(i as f32 / 19.0) * 2.0 - 1.0]))
            .collect();
        let (stats, _) = collect_activation_stats("teste", &samples, None, 1, None);
        assert_eq!(stats.histogram.len(), HISTOGRAM_BINS);
        assert_eq!(stats.histogram.iter().sum::<u32>(), 20);
    }

    #[test]
    fn sem_w_rec_raio_e_rank_ficam_zero() {
        let samples = vec![linha(&[0.1, 0.2])];
        let (stats, alarms) = collect_activation_stats("SlowBrain", &samples, None, 2, None);
        // Sem matriz entregue: ausencia de medida, nao zero (ver doc do campo).
        assert_eq!(stats.spectral_radius, None);
        assert_eq!(stats.effective_rank, None);
        assert!(
            !alarms
                .iter()
                .any(|a| a.kind == AlarmKind::SpectralRadiusHigh)
        );
    }
}
