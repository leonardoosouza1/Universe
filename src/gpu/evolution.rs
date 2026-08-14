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
//!
//! # Inicialização espectral de L1 (P0.5)
//!
//! O sorteio N(0, σ=0.30) i.i.d. acima, medido com `analysis::activation` em 4
//! corridas independentes, produz ganho espectral de L1 (maior valor singular da
//! matriz feedforward 192→96, a mesma grandeza que `AlarmKind::SpectralRadiusHigh`
//! audita) em ~7.0 — bem acima do `|λ_max| <= ~1` que a teoria de echo-state pede
//! para o sinal sensorial não ser amplificado antes da primeira mutação. Depois de
//! sortear `l1`, `Genome::random` mede esse ganho com
//! `math::linalg::largest_singular_value` e re-escala TODOS os pesos de L1 por um
//! único fator escalar (`thresholds::SPECTRAL_INIT_TARGET / medido`) — um escalar
//! preserva a direção relativa de cada peso, só ajusta a magnitude, e como a medida é
//! LINEAR na matriz (`largest_singular_value(c·W) = c · largest_singular_value(W)`,
//! ver teste `largest_singular_value_escala_linear_com_a_matriz` em
//! `math/linalg.rs`), o fator atinge o alvo exatamente, não por aproximação.
//!
//! Controlado por [`SPECTRAL_INIT_ON`] (padrão: ligado) — o braço de controle do A/B
//! é `--sem-init-espectral` no `gpu_evolve`. Só toca `l1`: L2 não entra no alarme
//! (`collect_activation_stats("L2", ..., w_rec: None, ...)` em `gpu_evolve/main.rs`),
//! então não há "o que o alarme mede" para L2 mirar. `load_genome` (genoma vindo do
//! disco) NUNCA passa por `Genome::random` — constrói o `Genome` direto por struct
//! literal — então um campeão salvo antes desta mudança carrega com o ganho espectral
//! que ele sempre teve, intocado.

use crate::config::iara::brain::{SENSORY_WAVE_SIZE, L1_NEURONS, L2_NEURONS};
use crate::kan::KAN_OUT_KNOTS;
use crate::engine::SovereignEngine;
use crate::math::{linalg, thresholds};
use std::sync::atomic::{AtomicBool, Ordering};

/// Liga/desliga a inicialização espectral de L1 (P0.5) em `Genome::random`.
///
/// Padrão `true` (LIGADA) — nascer com ganho espectral ≈ `thresholds::SPECTRAL_INIT_TARGET`
/// é o comportamento recomendado, não um experimento à parte. `gpu_evolve
/// --sem-init-espectral` desliga (`store(false, ...)`) para rodar o braço de controle
/// do A/B: população nascendo com o σ=0.30 i.i.d. cru, sem correção — o mesmo cérebro
/// de antes da hipótese P0.5.
///
/// É um `AtomicBool` global (não um parâmetro de `Genome::random`) pelo mesmo motivo de
/// `world::NO_HOLES`/`world::EASY_TRACK`: `Genome::random(seed)` é chamado em várias
/// dezenas de lugares (população inicial, retomada de checkpoint, testes) e mudar a
/// assinatura obrigaria a tocar todos eles só para propagar uma flag de CLI que é lida
/// uma vez no início do processo.
pub static SPECTRAL_INIT_ON: AtomicBool = AtomicBool::new(true);

/// Fator escalar que leva um ganho `medido` para `alvo` (== `alvo / medido`).
///
/// Função pura, sem estado global — extraída de `Genome::random` para poder ser
/// testada sem depender de [`SPECTRAL_INIT_ON`] (que é global e mutável; um teste que
/// o alternasse arriscaria interferir com qualquer outro teste que chame
/// `Genome::random` em paralelo, ver R8 do CLAUDE.md do projeto — "canário: números
/// idênticos onde deveriam divergir").
///
/// `medido <= 1e-6` (L1 inteiro ~0, nunca observado com σ=0.30 e 18.432 pesos) devolve
/// `1.0` — nada para escalar, e `0.0 × 1.0 = 0.0` continua nulo, não uma divisão por
/// quase-zero explodindo o peso.
fn spectral_rescale_factor(medido: f32, alvo: f32) -> f32 {
    if medido > 1e-6 { alvo / medido } else { 1.0 }
}

// ── Dimensões derivadas ────────────────────────────────────────────────────────
pub const N_L1_WEIGHTS: usize = SENSORY_WAVE_SIZE * L1_NEURONS; // 192 × 96 = 18.432
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
    /// Banco de τ: taxa de decaimento do CfC por canal (s⁻¹), `N_CFC_CHANNELS` valores.
    ///
    /// Nasce uniforme em `DECAY_DEFAULT` (τ = 2,0 s, o valor único de antes) e só vira
    /// heterogêneo por `randomize_tau`. Assim a condição de controle é literalmente o
    /// cérebro histórico rodando pelo código novo — mesmo shader, mesma fórmula, só o
    /// vetor difere.
    pub tau: Vec<f32>,
}

/// Canais do CfC — o filtro cobre os primeiros 96 da onda sensorial.
pub const N_CFC_CHANNELS: usize = 96;

impl Genome {
    /// Gera genoma com pesos aleatórios (N(0, σ_l1)).
    ///
    /// σ_l1 = 0.30 para L1/L2 — colocada na escala de pesos reais observados
    /// no CHAMPION.dna (|w| médio ≈ 0.15, max ≈ 0.7). Pesos zerados = rede nula.
    /// σ = 0.3 garante pelo menos ~30% dos pesos não-nulos com ação não-trivial.
    ///
    /// Depois de sortear L1, e se [`SPECTRAL_INIT_ON`] estiver ligado (padrão), L1 é
    /// re-escalado para nascer com ganho espectral ≈ `thresholds::SPECTRAL_INIT_TARGET`
    /// — ver o módulo "Inicialização espectral de L1 (P0.5)" no topo do arquivo.
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

        let mut l1: Vec<f32> = (0..N_L1_WEIGHTS).map(|_| gauss(&mut rng_f32)).collect();
        let l2: Vec<f32> = (0..N_L2_WEIGHTS).map(|_| gauss(&mut rng_f32)).collect();

        // P0.5 — inicialização espectral: mede o ganho que o sorteio acima acabou de
        // produzir (mesma conta que `analysis::activation` usa pro alarme,
        // `SpectralRadiusHigh`) e re-escala L1 por um único fator escalar para que o
        // ganho medido bata em `thresholds::SPECTRAL_INIT_TARGET`. Um fator escalar
        // sobre a matriz inteira escala o maior valor singular pelo mesmo fator
        // (linear — ver teste em `math/linalg.rs`), então o alvo é atingido
        // exatamente, não por tentativa. Só toca L1: é a única matriz que o alarme
        // audita (L2 usa `w_rec: None` na chamada de `collect_activation_stats`).
        //
        // `SPECTRAL_INIT_ON` é o braço de controle do A/B (`--sem-init-espectral`,
        // padrão LIGADO). Genoma vindo do disco (`load_genome`) nunca passa por aqui.
        if SPECTRAL_INIT_ON.load(Ordering::Relaxed) {
            let medido = linalg::largest_singular_value(&l1, L1_NEURONS, SENSORY_WAVE_SIZE);
            let fator = spectral_rescale_factor(medido, thresholds::SPECTRAL_INIT_TARGET);
            for w in l1.iter_mut() { *w *= fator; }
        }

        // FASE CANAIS (2026-08-14) — zera a coluna de L1 dos canais em
        // `perception::channels::DEAD_L1_INPUTS` (hoje só TireTemp, ch70): canais
        // cujo kernel escreve mas o valor é estruturalmente constante (sem dado real
        // por trás — ver justificativa no registro). O cérebro não nasce gastando
        // peso de entrada com estática. Roda DEPOIS da re-escala espectral de
        // propósito: mede/normaliza o ganho da matriz i.i.d. inteira primeiro, e só
        // então zera as ~96 entradas (de 18.432) dos canais mortos — perturbação
        // pequena o bastante (<1% dos pesos) para não tirar o ganho medido de
        // [0.8, 1.2] (ver teste `genoma_novo_nasce_com_ganho_espectral_perto_de_1`).
        // NÃO é poda de neurônio — a ablação 96→64 (PLANO_EVOLUCAO.md) provou que
        // nenhum neurônio de L1 é descartável; isto só impede um peso de ENTRADA de
        // ler um canal que nunca varia.
        for &(ch, _motivo) in crate::perception::channels::DEAD_L1_INPUTS {
            for neuron in 0..L1_NEURONS {
                l1[neuron * SENSORY_WAVE_SIZE + ch] = 0.0;
            }
        }

        let kan_knots = crate::evolution::dna_genesis::default_kan_knots();

        // Lento: parte da inicialização padrão e passa a divergir por mutação. Ele nasce
        // igual em todos os seres de propósito — a diferença tem que ser SELECIONADA,
        // não sorteada, senão a primeira geração já mistura dois efeitos.
        let slow = crate::engine::SlowBrain::new().all_weights();
        // τ uniforme = comportamento herdado. Heterogêneo só por `randomize_tau`, para que
        // o braço de controle do experimento seja o cérebro antigo, não um terceiro caso.
        let tau = vec![crate::cfc::DECAY_DEFAULT; N_CFC_CHANNELS];
        Self { l1, l2, kan_knots, slow, tau }
    }

    /// Sorteia o banco de τ log-uniforme em [25 ms, 4 s] — a condição heterogênea.
    pub fn randomize_tau(&mut self, seed: u64) {
        self.tau = crate::cfc::log_uniform_decays(seed, N_CFC_CHANNELS);
    }

    /// Achata o banco de τ de vários genomas no layout que `BrainGpu::upload_tau` espera.
    pub fn pack_taus(genomes: &[Genome]) -> Vec<f32> {
        let mut out = Vec::with_capacity(genomes.len() * N_CFC_CHANNELS);
        for g in genomes { out.extend_from_slice(&g.tau); }
        out
    }

    /// Carrega genoma a partir de um `SovereignEngine` existente.
    pub fn from_engine(eng: &SovereignEngine) -> Self {
        let l1 = eng.instinct_layer.weights.clone();
        let l2 = eng.cortex_l2.weights.clone();
        // KAN knots residem em engine.kan_filters (per-canal), mas os knots de
        // SAÍDA (throttle/steer/brake) não ficam no engine — usamos o default.
        let kan_knots = crate::evolution::dna_genesis::default_kan_knots();
        let slow = eng.slow_brain.all_weights();
        let tau = vec![crate::cfc::DECAY_DEFAULT; N_CFC_CHANNELS];
        Self { l1, l2, kan_knots, slow, tau }
    }

    /// Escreve genoma em um `SovereignEngine` (substitui pesos).
    ///
    /// Falha explicitamente se L1/L2 divergirem — truncamento silencioso já causou
    /// pesos perdidos sem aviso, resultado: seres nasciam com genoma parcialmente zerado
    /// enquanto métricas de seleção pareciam normais.
    /// Quantos neurônios de L1 ficam VIVOS. `0` = todos (comportamento normal).
    ///
    /// Leonardo: *"podemos avaliar o quanto de espaço eles realmente precisam, e se o
    /// modelo ficar com pouco espaço a gente vai aumentando."*
    ///
    /// A medida de capacidade diz que L1 usa 6,7 das 96 dimensões (7%) — mas métrica não
    /// é prova. Zerar os pesos de saída dos neurônios acima do corte e medir os metros
    /// transforma a estimativa em experimento: se cortar 96→16 não custar desempenho, os
    /// 93% ociosos estão demonstrados, não inferidos.
    ///
    /// Zera-se a SAÍDA (a coluna de L2 que lê aquele neurônio) e não a entrada, porque é
    /// isso que remove a contribuição dele para o comportamento; zerar a entrada deixaria
    /// o neurônio emitindo o seu viés.
    pub fn ablar_l1_acima_de(&mut self, vivos: usize) {
        use crate::config::iara::brain::{L1_NEURONS, L2_NEURONS};
        if vivos == 0 || vivos >= L1_NEURONS { return; }
        for j in 0..L2_NEURONS {
            for n in vivos..L1_NEURONS {
                let idx = j * L1_NEURONS + n;
                if idx < self.l2.len() { self.l2[idx] = 0.0; }
            }
        }
    }

    pub fn apply_to_engine(&self, eng: &mut SovereignEngine) {
        assert_eq!(
            self.l1.len(), eng.instinct_layer.weights.len(),
            "L1 tem {} pesos, engine espera {} — divergência de SENSORY_WAVE_SIZE ou L1_NEURONS",
            self.l1.len(), eng.instinct_layer.weights.len()
        );
        eng.instinct_layer.weights.copy_from_slice(&self.l1);
        assert_eq!(
            self.l2.len(), eng.cortex_l2.weights.len(),
            "L2 tem {} pesos, engine espera {} — divergência de L1_NEURONS ou L2_NEURONS",
            self.l2.len(), eng.cortex_l2.weights.len()
        );
        eng.cortex_l2.weights.copy_from_slice(&self.l2);
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
        // Mutação do banco de τ: MULTIPLICATIVA, em log.
        //
        // Aditiva seria errada aqui. A faixa vai de 0,25 a 40 s⁻¹ — um σ que mexe de
        // verdade num canal de 0,25 atropela um de 40, e vice-versa. O que tem escala
        // constante é a razão: ×1,15 significa a mesma coisa em qualquer ponto da faixa.
        //
        // Taxa 3× a dos pesos pelo mesmo motivo do lento: são 96 valores contra 26 mil,
        // e com a taxa base o banco de τ mal seria tocado numa vida de 30 gerações.
        let d_lo = 1.0 / crate::cfc::TAU_MAX_S;
        let d_hi = 1.0 / crate::cfc::TAU_MIN_S;
        for d in filho.tau.iter_mut() {
            if rng.f32() < params.p_mutate * 3.0 {
                *d = (*d * (rng.gauss() * 0.20).exp()).clamp(d_lo, d_hi);
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

// ---------------------------------------------------------------------------
// Testes unitários — P0.5 (inicialização espectral de L1).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `spectral_rescale_factor` é pura function matemática — o teste central da
    /// inicialização espectral que não precisa tocar `SPECTRAL_INIT_ON` nem sortear
    /// pesos: dado o ganho ~7.0 medido empiricamente (CLAUDE.md do projeto, 4 corridas
    /// independentes), o fator devolvido tem que levar exatamente a 1.0.
    #[test]
    fn spectral_rescale_factor_atinge_alvo_exatamente() {
        let fator = spectral_rescale_factor(7.0, 1.0);
        assert!((fator - 1.0 / 7.0).abs() < 1e-6, "fator = {fator}, esperado 1/7");
        assert!((7.0 * fator - 1.0).abs() < 1e-6, "aplicar o fator deveria levar a 1.0");
    }

    /// Ganho ~0 (L1 nulo, nunca observado na prática) não deve gerar divisão explosiva
    /// — o fator cai para 1.0 (identidade: nada para escalar).
    #[test]
    fn spectral_rescale_factor_ganho_quase_zero_nao_explode() {
        assert_eq!(spectral_rescale_factor(0.0, 1.0), 1.0);
        assert_eq!(spectral_rescale_factor(1e-9, 1.0), 1.0);
    }

    /// P0.5, requisito 4: genoma novo (flag padrão LIGADA) nasce com ganho espectral
    /// de L1 medido em [0.8, 1.2] — não exatamente 1.0 porque a normalização mede o
    /// GANHO DA MATRIZ INTEIRA sorteada (deve bater ~1.0 dentro do erro numérico do
    /// power-iteration, ~100 iterações, mais o efeito do clamp em 3σ na cauda da
    /// gaussiana), mas longe o bastante do ~7.0 sem correção para provar que a
    /// re-escala está de fato acontecendo.
    ///
    /// Não mexe em `SPECTRAL_INIT_ON` (fica no padrão `true` do processo de teste) —
    /// ver doc de `spectral_rescale_factor` sobre por que evitar alternar o atomic
    /// global dentro de testes.
    #[test]
    fn genoma_novo_nasce_com_ganho_espectral_perto_de_1() {
        assert!(
            SPECTRAL_INIT_ON.load(Ordering::Relaxed),
            "pré-condição do teste: a flag deveria estar no padrão (ligada) neste processo"
        );
        for seed in [1u64, 2, 3, 42, 999_999] {
            let g = Genome::random(seed);
            let ganho = linalg::largest_singular_value(&g.l1, L1_NEURONS, SENSORY_WAVE_SIZE);
            assert!(
                (0.8..=1.2).contains(&ganho),
                "seed {seed}: ganho espectral de L1 = {ganho}, esperado em [0.8, 1.2]"
            );
        }
    }

    /// FASE CANAIS (2026-08-14): genoma novo nasce com a coluna de L1 dos canais
    /// `DEAD_L1_INPUTS` (hoje só TireTemp, ch70) inteiramente zerada — nenhum dos
    /// `L1_NEURONS` pesos que leem esse canal escapa do zeramento, em nenhuma seed.
    #[test]
    fn genoma_novo_zera_l1_dos_canais_mortos() {
        use crate::perception::channels::DEAD_L1_INPUTS;
        assert!(!DEAD_L1_INPUTS.is_empty(), "pré-condição: precisa de ao menos 1 canal morto pra testar algo");
        for seed in [1u64, 7, 999] {
            let g = Genome::random(seed);
            for &(ch, _motivo) in DEAD_L1_INPUTS {
                for neuron in 0..L1_NEURONS {
                    let w = g.l1[neuron * SENSORY_WAVE_SIZE + ch];
                    assert_eq!(w, 0.0, "seed {seed}: l1[neuron={neuron}, ch={ch}] = {w}, esperado 0.0");
                }
            }
        }
    }

    /// FASE CANAIS: zerar os canais mortos não deve mudar canais VIZINHOS — sanidade
    /// contra um erro de índice (off-by-one na coluna) que zerasse o canal errado.
    /// Não mexe em `SPECTRAL_INIT_ON` (mesmo cuidado de
    /// `genoma_novo_nasce_com_ganho_espectral_perto_de_1` — ver doc de
    /// `spectral_rescale_factor` sobre por que evitar alternar o atomic global
    /// dentro de testes, R8 do CLAUDE.md do projeto). Um vizinho ficar em 0.0
    /// exatamente após reescala espectral tem probabilidade praticamente nula
    /// (gaussiana contínua × fator escalar não-nulo); múltiplas seeds reduzem
    /// ainda mais o risco de falso-negativo.
    #[test]
    fn zerar_canal_morto_nao_afeta_canal_vizinho_vivo() {
        use crate::perception::channels::DEAD_L1_INPUTS;
        for seed in [1u64, 7, 999] {
            let g = Genome::random(seed);
            for &(ch, _) in DEAD_L1_INPUTS {
                if ch + 1 < SENSORY_WAVE_SIZE {
                    let vizinho = g.l1[0 * SENSORY_WAVE_SIZE + (ch + 1)];
                    assert_ne!(vizinho, 0.0, "seed {seed}: canal vizinho ch{} não deveria ter sido zerado", ch + 1);
                }
            }
        }
    }

    // P0.5, requisito 4 (parte 2, "herança intocada"): o teste que carrega um genoma
    // legado de verdade do disco e confirma que `load_genome` não re-escala mora em
    // `src/bin/gpu_evolve/tests.rs::genoma_legado_carregado_do_disco_nao_e_reescalado`
    // — precisa de `load_genome`/`save_genome`, que vivem no binário `gpu_evolve`, não
    // nesta crate. A garantia estrutural é simples de conferir por leitura: `load_genome`
    // monta `Genome { l1, l2, kan_knots, slow, tau }` por struct literal direto
    // (genome_io.rs), nunca chamando `Genome::random` nem `spectral_rescale_factor`.
}
