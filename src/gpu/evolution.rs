//! Seleção e mutação de genomas entre gerações — roda na CPU.
//!
//! O CUSTO está nos ticks (GPU). A evolução acontece entre gerações,
//! uma vez por N ticks. Portanto pode rodar na CPU sem penalidade significativa.
//!
//! # Genoma
//!
//! Um `Genome` carrega os pesos de L1 (BitLinear f32) e L2 (FloatLinear f32) do
//! `SovereignEngine`, além dos knots KAN de saída. São os mesmos campos que o
//! `EliteDna` do motor principal, mas sem os metadados de serialização.
//!
//! # Algoritmo evolutivo
//!
//! - **Seleção**: torneio (k=3) — seleciona o melhor entre 3 candidatos aleatórios.
//!   Menos propenso à deriva genética que a seleção proporcional.
//! - **Mutação gaussiana**: ruído N(0, σ) por peso, com σ=0.05 por padrão.
//!   Cada peso tem probabilidade `p_mutate` de ser perturbado.
//! - **Elitismo**: os melhores `n_elite` genomas passam sem mutação.
//!   Preserva o progresso; σ cobre a exploração.
//!
//! # Inicialização aleatória
//!
//! `Genome::random()` usa LCG com seed derivada do índice do ser. Magnitude σ=0.3
//! pra L1/L2. Pesos zerados (BitLinear::new()) → rede nula → nenhum movimento;
//! σ=0.3 coloca pesos na escala que o motor.rs usa (~0.1–0.5 após treinamento).

use crate::config::iara::brain::{SENSORY_WAVE_SIZE, L1_NEURONS, L2_NEURONS};
use crate::kan::KAN_OUT_KNOTS;
use crate::engine::SovereignEngine;

// ── Dimensões derivadas ────────────────────────────────────────────────────────
pub const N_L1_WEIGHTS: usize = SENSORY_WAVE_SIZE * L1_NEURONS; // 223 × 96 = 21.408
pub const N_L2_WEIGHTS: usize = L1_NEURONS * L2_NEURONS;         // 96  × 48 =  4.608
pub const N_KAN_KNOTS:  usize = 3 * KAN_OUT_KNOTS;               // 3   × 12 =     36

/// Genoma completo de um ser: L1 + L2 + KAN de saída.
#[derive(Clone)]
pub struct Genome {
    /// Pesos L1 (instinct_layer): SENSORY_WAVE_SIZE × L1_NEURONS f32.
    /// Magnitude típica pós-treinamento: ~0.1–0.5. Init aleatório com σ=0.3.
    pub l1: Vec<f32>,
    /// Pesos L2 (cortex_l2): L1_NEURONS × L2_NEURONS f32.
    pub l2: Vec<f32>,
    /// Knots KAN de saída: 3 outputs × KAN_OUT_KNOTS f32.
    /// Init com KanOutputHead::new() (tanh ramp — equivale a identidade suave).
    pub kan_knots: [[f32; KAN_OUT_KNOTS]; 3],
    /// Pesos do cérebro LENTO na ordem de `SlowBrain::all_weights()`:
    /// dense1.weights(1024) | dense1.bias(16) | dense2.weights(48) | dense2.bias(3).
    ///
    /// Sem isto o lento não é um órgão, é uma constante: `SlowBrain::new()` usa
    /// sementes FIXAS, então todos os seres nasceriam com os mesmos pesos lentos e a
    /// modulação seria idêntica para a população inteira — a evolução não teria como
    /// ajustá-la nem como se livrar dela. Herdável e mutável, ela pode ser selecionada
    /// ou descartada pelo próprio processo.
    pub slow: Vec<f32>,
}

impl Genome {
    /// Gera genoma com pesos aleatórios (N(0, σ_l1)).
    ///
    /// σ_l1 = 0.30 para L1/L2 — colocada na escala de pesos reais observados
    /// no CHAMPION.dna (|w| médio ≈ 0.15, max ≈ 0.7). Pesos zerados = rede nula.
    /// σ = 0.3 garante pelo menos ~30% dos pesos não-nulos com ação não-trivial.
    ///
    /// KAN knots = default (tanh ramp) — funciona como identidade suave e não
    /// precisa ser evoluído nos primeiros estágios.
    pub fn random(seed: u64) -> Self {
        // LCG de 64 bits (Knuth MMIX) — determinístico por seed.
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x6C62272E07BB0142);
        let mut rng_f32 = move || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f32) / (1u64 << 53) as f32;
            u * 2.0 - 1.0  // uniforme [-1, 1]
        };

        // Box-Muller para N(0, σ).
        let sigma_l1 = 0.30_f32;
        let gauss = |rng: &mut dyn FnMut() -> f32| -> f32 {
            let u1 = (rng() * 0.5 + 0.5).max(1e-7_f32);
            let u2 = rng() * 0.5 + 0.5;
            let g = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
            (g * sigma_l1).clamp(-3.0 * sigma_l1, 3.0 * sigma_l1)  // clamp em 3σ
        };

        let l1: Vec<f32> = (0..N_L1_WEIGHTS).map(|_| gauss(&mut rng_f32)).collect();
        let l2: Vec<f32> = (0..N_L2_WEIGHTS).map(|_| gauss(&mut rng_f32)).collect();
        let kan_knots = crate::evolution::dna_genesis::default_kan_knots();

        // Lento: parte da inicialização padrão e passa a divergir por mutação. Ele nasce
        // igual em todos os seres de propósito — a diferença tem que ser SELECIONADA,
        // não sorteada, senão a primeira geração já mistura dois efeitos.
        let slow = crate::engine::SlowBrain::new().all_weights();
        Self { l1, l2, kan_knots, slow }
    }

    /// Carrega genoma a partir de um `SovereignEngine` existente.
    pub fn from_engine(eng: &SovereignEngine) -> Self {
        let l1 = eng.instinct_layer.weights.clone();
        let l2 = eng.cortex_l2.weights.clone();
        // KAN knots residem em engine.kan_filters (per-canal), mas os knots de
        // SAÍDA (throttle/steer/brake) não ficam no engine — usamos o default.
        let kan_knots = crate::evolution::dna_genesis::default_kan_knots();
        let slow = eng.slow_brain.all_weights();
        Self { l1, l2, kan_knots, slow }
    }

    /// Escreve genoma em um `SovereignEngine` (substitui pesos).
    pub fn apply_to_engine(&self, eng: &mut SovereignEngine) {
        let n1 = self.l1.len().min(eng.instinct_layer.weights.len());
        eng.instinct_layer.weights[..n1].copy_from_slice(&self.l1[..n1]);
        let n2 = self.l2.len().min(eng.cortex_l2.weights.len());
        eng.cortex_l2.weights[..n2].copy_from_slice(&self.l2[..n2]);
        if !self.slow.is_empty() { eng.slow_brain.set_all_weights(&self.slow); }
        // KAN knots são usados pelo ciclo de evolução diretamente (não no engine).
    }
}

/// Parâmetros do algoritmo evolutivo.
pub struct EvoParams {
    /// Taxa de mutação por peso: probabilidade de cada peso ser perturbado.
    pub p_mutate: f32,
    /// Desvio-padrão do ruído gaussiano adicionado a cada peso mutado.
    pub sigma: f32,
    /// Número de genomas elite que passam sem mutação.
    pub n_elite: usize,
    /// Tamanho do torneio para seleção de pais.
    pub tournament_k: usize,
}

impl Default for EvoParams {
    fn default() -> Self {
        Self {
            p_mutate: 0.02,    // 2% dos pesos por geração
            sigma: 0.05,       // perturbação gaussiana σ=0.05
            n_elite: 3,        // top 3 passam sem mutação
            tournament_k: 3,   // torneio de 3
        }
    }
}

/// Resultado de uma geração.
pub struct GenResult {
    pub generation: usize,
    pub dist_max:  f64,  // metros, melhor indivíduo
    pub dist_med:  f64,  // metros, mediana
    pub n_ge1v:    usize, // seres com ≥1 volta completa
    pub n_ge2v:    usize, // seres com ≥2 voltas completas
    pub mortality: f32,  // fração de seres que não andaram (dist < 10m)
}

/// Executa um ciclo evolutivo: seleciona + muta.
///
/// `fitnesses[i]` = distância acumulada em metros do ser i na geração anterior.
/// Retorna nova geração de genomas na mesma ordem (0..pop_size).
pub fn evolve(
    genomes:    &[Genome],
    fitnesses:  &[f64],
    params:     &EvoParams,
    generation: usize,   // índice da geração (para seed determinística)
) -> Vec<Genome> {
    let n = genomes.len();
    assert_eq!(fitnesses.len(), n, "fitnesses e genomes devem ter mesmo tamanho");

    // ── Ordena por fitness decrescente ─────────────────────────────────────────
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut nova: Vec<Genome> = Vec::with_capacity(n);

    // ── Elitismo: os melhores passam inalterados ────────────────────────────────
    for &idx in order.iter().take(params.n_elite) {
        nova.push(genomes[idx].clone());
    }

    // ── Seleção por torneio + mutação ───────────────────────────────────────────
    // Struct local para encapsular o RNG sem aliasing de closures.
    struct Rng { state: u64, spare: Option<f32> }
    impl Rng {
        fn new(seed: u64) -> Self { Self { state: seed, spare: None } }
        fn u64(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }
        fn f32(&mut self) -> f32 { ((self.u64() >> 11) as f32) / (1u64 << 53) as f32 }
        // Box-Muller N(0,1).
        fn gauss(&mut self) -> f32 {
            if let Some(v) = self.spare.take() { return v; }
            loop {
                let u1 = self.f32().max(1e-7_f32);
                let u2 = self.f32();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = std::f32::consts::TAU * u2;
                self.spare = Some(r * theta.sin());
                return r * theta.cos();
            }
        }
    }

    let seed = (generation as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0xCAFE_BABE_DEAD_BEEF);
    let mut rng = Rng::new(seed);

    while nova.len() < n {
        // Torneio: seleciona melhor entre k candidatos aleatórios.
        let parent_idx = {
            let mut best = rng.u64() as usize % n;
            for _ in 1..params.tournament_k {
                let cand = rng.u64() as usize % n;
                if fitnesses[cand] > fitnesses[best] { best = cand; }
            }
            best
        };

        let pai = &genomes[parent_idx];
        let mut filho = pai.clone();

        // Mutação gaussiana em L1 (2% dos pesos por geração, σ configurável).
        for w in filho.l1.iter_mut() {
            if rng.f32() < params.p_mutate {
                *w += rng.gauss() * params.sigma;
                *w = w.clamp(-2.0, 2.0);
            }
        }
        // Mutação gaussiana em L2.
        for w in filho.l2.iter_mut() {
            if rng.f32() < params.p_mutate {
                *w += rng.gauss() * params.sigma;
                *w = w.clamp(-2.0, 2.0);
            }
        }
        // Mutação do lento. σ maior porque são mil pesos contra vinte e seis mil do
        // rápido — com a mesma taxa o lento receberia perturbação praticamente nula.
        for w in filho.slow.iter_mut() {
            if rng.f32() < params.p_mutate * 3.0 {
                *w += rng.gauss() * params.sigma * 1.5;
                *w = w.clamp(-2.0, 2.0);
            }
        }
        // KAN knots: mutação esparsa (0.5% por knot, σ=0.05).
        for row in filho.kan_knots.iter_mut() {
            for k in row.iter_mut() {
                if rng.f32() < 0.005 {
                    *k = (*k + rng.gauss() * 0.05).clamp(-1.0, 1.0);
                }
            }
        }

        nova.push(filho);
    }

    nova
}

/// Calcula estatísticas de uma geração a partir das distâncias acumuladas.
pub fn gen_stats(generation: usize, dist_acum: &[f64], lap_dist: f64) -> GenResult {
    let n = dist_acum.len();
    let dist_max = dist_acum.iter().cloned().fold(0.0_f64, f64::max);
    let dist_med = {
        let mut v = dist_acum.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let n_ge1v  = dist_acum.iter().filter(|&&d| d >= lap_dist).count();
    let n_ge2v  = dist_acum.iter().filter(|&&d| d >= lap_dist * 2.0).count();
    let n_morto = dist_acum.iter().filter(|&&d| d < 10.0).count();
    let mortality = n_morto as f32 / n as f32;
    GenResult { generation, dist_max, dist_med, n_ge1v, n_ge2v, mortality }
}

/// Calcula inclinação linear (OLS) e seu desvio-padrão (erro do estimador).
///
/// `xs[i]` = geração i (float), `ys[i]` = mediana da geração i.
/// Retorna `(slope, slope_stderr)`. Se n < 2 retorna (0.0, 0.0).
pub fn linear_slope(ys: &[f64]) -> (f64, f64) {
    let n = ys.len();
    if n < 2 { return (0.0, 0.0); }
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mean_x = xs.iter().sum::<f64>() / n as f64;
    let mean_y = ys.iter().sum::<f64>() / n as f64;
    let sxx: f64 = xs.iter().map(|&x| (x - mean_x).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| (x - mean_x) * (y - mean_y)).sum();
    if sxx < 1e-10 { return (0.0, 0.0); }
    let slope = sxy / sxx;
    // SSR = variância residual
    let ssr: f64 = xs.iter().zip(ys.iter())
        .map(|(&x, &y)| (y - (mean_y + slope * (x - mean_x))).powi(2))
        .sum();
    let var_resid = ssr / (n as f64 - 2.0).max(1.0);
    let slope_stderr = (var_resid / sxx).sqrt();
    (slope, slope_stderr)
}
