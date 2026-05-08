use bevy::prelude::*;
use crate::simulation::Racer;
use crate::world::VoxelWorld;
use crate::config::iara::vision::RETINA_TOTAL;
use crate::config::iara::brain::{SENSORY_WAVE_SIZE, L1_NEURONS};
use crate::config::UniverseConfig;
use rand::prelude::*;
use std::io::Write;

use super::dna_genesis::{EliteDna, MorphoGenes, HypernetSeed, load_champion_dna};
use super::logs::open_report_log;
use super::settings::{RaceSettings, DevSettings};

/// Stride para indexação de pesos: weights[output * SWS + input_channel].
/// DEVE ser = SENSORY_WAVE_SIZE (input_dim do BitLinear). Era hardcoded 128,
/// causando desalinhamento quando SENSORY_WAVE_SIZE mudou pra 223.
const SWS: usize = SENSORY_WAVE_SIZE;

// TIER 4.4 — GENES NEUTROS (ZONA PROTEGIDA).
// Primeiros ~5% dos índices do instinct_fast são reservados como
// "zona neutra/protegida": isentos de mutação E isentos de crossover.
const PROTECTED_ZONE_SIZE: usize = 400;

/// Constrói o vetor de pesos do instinct_layer pré-semeado com os axiomas A1–A15.
/// Usado tanto na geração 0 (sem pool/DNA) quanto periodicamente em gens seguintes
/// para reinjetar instintos perdidos por deriva genética.
// FIX: slow brain (32→8, 256 pesos) ficava 100% zero por ~125 gens.
// Priming diagonal: cada output k soma banda de 4 inputs → força sinal > 0
// e evolução afina a partir de base funcional.
fn build_slow_priming_weights() -> Vec<f32> {
    let mut w = vec![0.0f32; 32 * 8];
    for out in 0..8 {
        for i in 0..4 {
            let idx = out * 32 + (out * 4 + i);
            if idx < w.len() {
                w[idx] = 1.0;
            }
        }
    }
    w
}

fn build_priming_weights() -> Vec<f32> {
    let mut weights = vec![0.0f32; SWS * L1_NEURONS];

    // Motor.rs faz POOLING da retina: fóvea 32 px → 8 bins (ch 0..7),
    // periférica 32 px → 8 bins (ch 8..15). Ch 16..63 ZERADOS pelo motor.
    // Priming DEVE atacar os bins novos, não os pixels antigos.
    //
    // Periférica bins (uniforme, 4 px cada): 8=px32-35 ... 15=px60-63.
    // Bins 8-9 = esquerda extremo/lateral. Bins 14-15 = direita. 10-13 = centro.
    //
    // A1: Wall à esquerda (perif bin 8, 9) → vira à DIREITA (positivo).
    // v4: 6/4→3/2. Periférica SEMPRE vê wall em pista <14m. Magnitude alta
    // causava zig-zag perpétuo. Agora é sugestão leve, evolução refina.
    weights[1 * SWS + 8] = 3.0;
    weights[1 * SWS + 9] = 2.0;
    // A2: Wall à direita (perif bin 14, 15) → vira à ESQUERDA (negativo)
    weights[1 * SWS + 14] = -2.0;
    weights[1 * SWS + 15] = -3.0;
    // A3: Alinhamento com fluxo (canal 64) → acelera (magnitude alta p/ sobrepor pooling sparse)
    weights[0 * SWS + 64] = 12.0;
    // A4: Parede no centro periférica (apenas bin 11, 12 = centro exato) → corta throttle LEVE.
    // Magnitudes baixas: ganho vem da fóvea (A7b). Periférica center sempre
    // acende leve em pistas estreitas — forte demais = carro trava parado.
    weights[0 * SWS + 11] = -3.0;
    weights[0 * SWS + 12] = -3.0;
    // A4d: BRAKE removido do priming (causa travamento permanente em pistas narrow).
    // Brake emergente vem dos whiskers (94-97) que só acendem em colisão iminente.
    // A5: Dor de grama (73) → reduz throttle
    // Sensor 73 = -128 SE grass, 0 caso contrário. w=+5 × (-128) = sinal negativo p/ throttle = corta. ✓
    weights[0 * SWS + 73] = 5.0;
    // A6: Comp-stress (125) → acelera
    weights[0 * SWS + 125] = 3.0;
    // A7: Freio pânico removido do priming — whiskers (94-97) já acendem BRAKE em proximidade real.
    // A7b: Fóvea centro (bin 3, 4 = px 13..19 centro). Retina: road=NEGATIVO, wall=POSITIVO.
    // Road clear (value −128) × weight (−6) = +768 contrib = ACELERA. Wall ahead (value +)
    // × weight (−) = negativo = corta throttle.
    // v4: -8→-4. Fóvea center vê wall frequentemente em narrow → -8 cortava throttle demais.
    weights[0 * SWS + 3] = -4.0;
    weights[0 * SWS + 4] = -4.0;
    // A7c REMOVIDO: bins extremos de fóvea SEMPRE veem paredes em pistas narrow (wall 7m lateral
    // entra em bin 0/7 a 15m à frente). Steer forçado pra um lado → deriva contra wall oposta.
    // Dark_center (90-93) já guia steering via centroide angular do road.
    // A8: Deriva lateral (66) → contra-steer
    // v4: -5→-3. Contra-steer mais suave, deixa evolução aprender drift.
    weights[1 * SWS + 66] = -3.0;
    // A9: Curvatura antecipada (78/79/80).
    // cross(forward, flow_ahead) em xz-plane: POSITIVO = flow vira à DIREITA.
    // Steer out 1: positivo = RIGHT. Logo sinal deve ser +, não -. FIX v3.
    // v4: curvatura mantida moderada — é o axioma MAIS útil (antecipa curvas).
    weights[1 * SWS + 78] = 5.0;
    weights[1 * SWS + 79] = 8.0;
    weights[1 * SWS + 80] = 4.0;
    // A10: Lane offset (81) → centraliza. v4: 5→2. Sugestão leve.
    // Evolução aprende centralização melhor que priming fixo em narrow.
    weights[1 * SWS + 81] = 2.0;
    // A11: Urgência de colisão (82) → freia + corta motor.
    // MAGNITUDE boost (v2): 10→15 brake, -6→-10 throttle.
    // Match nível A16/A18 (gaze/dark) que usam 4× duplicação.
    // Ver melhorias/survival-pressure.md §A/B validado — carros perceb TTC mas não freiam.
    weights[2 * SWS + 82] = 15.0;
    weights[0 * SWS + 82] = -10.0;
    // A11b: REFORÇO — replicar em canais vizinhos pra vencer quantize ternária.
    // Canal 82 é único então duplicamos via peso em outras linhas de output igual.
    // Isso não duplica input — apenas soma contribuição nas mesmas saídas via vizinhos
    // 85/86 (memória parede) que já correlacionam com TTC alto:
    weights[2 * SWS + 85] = 8.0;   // mem-wall-L + brake (já tem steer em A13)
    weights[2 * SWS + 86] = 8.0;   // mem-wall-R + brake
    // A12: Fluxo óptico (83/84) → steer anti-aproximação. v4: 5→3.
    weights[1 * SWS + 83] = 3.0;
    weights[1 * SWS + 84] = -3.0;
    // A13: Memória de parede (85/86). v4: 8→4. Sempre ativo em narrow, precisa ser suave.
    weights[1 * SWS + 85] = 4.0;
    weights[1 * SWS + 86] = -4.0;
    // A14: Slip angle (87) → contra-volante
    weights[1 * SWS + 87] = 4.0;
    // A15: Proximidade sustentada (88) → corta throttle + FREIA (v2).
    // Antes só cortava throttle; carro ficava em coast, não freava até bater.
    // v2: adiciona brake forte — "se tô grudado na parede há tempo, freia E recua intent".
    weights[0 * SWS + 88] = -7.0;
    weights[2 * SWS + 88] = 8.0;

    // A16 (NOVO): DARK-CENTER TRACKER (canais 90-93) → SIGA O ESCURO.
    // Resposta à crítica do usuário: "o carro não tenta centralizar o preto".
    // Canal = centro de massa angular dos pixels de pista, normalizado [-1..+1].
    // Positivo ⇒ pista à direita ⇒ steer RIGHT. Negativo ⇒ pista à esquerda ⇒ LEFT.
    // v2: magnitude reduzida 15.0→4.0. Valor 15.0 causava steer direita sistemático
    // em 100% dos modelos spawn. Lane offset (A10: ch81=+12.0) é sinal primário.
    // Dark center é refinamento, não dominador.
    for i in 90..94 {
        weights[1 * SWS + i] = 4.0;
    }

    // A17: WHISKERS DIAGONAIS (canais 94-97) — reflexo de curva fechada.
    //   94 = forward-left 30°  | 95 = forward-right 30°
    //   96 = forward-left 60°  | 97 = forward-right 60°
    //  v4: STEERING reduzido 18→8 / 14→6. Em narrow track whiskers SEMPRE detectam
    //  parede lateral → magnitude alta = oscilação perpétua. Agora é sugestão,
    //  evolução aprende o quanto virar. BRAKE mantido forte (seguro freiar).
    weights[1 * SWS + 94] = 8.0;    // FL 30° → steer RIGHT (era 18)
    weights[1 * SWS + 96] = 6.0;    // FL 60° → steer RIGHT (era 14)
    weights[1 * SWS + 95] = -8.0;   // FR 30° → steer LEFT  (era -18)
    weights[1 * SWS + 97] = -6.0;   // FR 60° → steer LEFT  (era -14)
    // BRAKE mantido forte — freiar é sempre seguro, steering não.
    weights[2 * SWS + 94] = 10.0;
    weights[2 * SWS + 95] = 10.0;
    weights[2 * SWS + 96] = 8.0;
    weights[2 * SWS + 97] = 8.0;
    // Corta throttle moderado (era -8, agora -5).
    weights[0 * SWS + 94] = -5.0;
    weights[0 * SWS + 95] = -5.0;

    // A19 (V6): pre_brake e wall_imminent priming SEGURO (canais novos).
    // open_l/r/best_escape ficam sem priming pra evitar conflito de steer signals
    // — net evolui peso natural via mutation (ainda mais lento mas estável).
    // Brain trace mostrou raw_thr SEMPRE +0.33. Cars nunca freiam.
    // Bump 2x priming brake — força output 2 dominar quando perigo.
    weights[2 * SWS + 28] = 10.0;
    weights[0 * SWS + 28] = -8.0;
    weights[2 * SWS + 44] = 12.0;
    weights[0 * SWS + 44] = -10.0;
    // Plus retina center ahead (ch 3, 4 fovea center pooled bins) → brake.
    weights[2 * SWS + 3] = 12.0;
    weights[2 * SWS + 4] = 12.0;
    // Plus retina center periph (8..15 = horizonte center) → brake.
    weights[2 * SWS + 11] = 8.0;
    weights[2 * SWS + 12] = 8.0;
    // A22: approach detection signed. Parede VINDO esquerda → steer DIREITA.
    // ch 114 high (positivo) = aproximação esquerda → output 1 positivo (right).
    weights[1 * SWS + 114] = 4.0;
    // ch 115 high = aproximação direita → output 1 negativo (left).
    weights[1 * SWS + 115] = -4.0;

    // A20: trapped_score (ch 48..51) → ré como OPÇÃO (não force).
    for i in 48..52 {
        weights[0 * SWS + i] = -2.5;
        weights[2 * SWS + i] = -1.5;
    }

    // A21 (DNA diagnostic): canais com peso < 0.03 = ignored. Forçar uso.
    // Speed (ch 65) é VITAL pra decisão "frear pré-curva". Net atual ignora.
    weights[2 * SWS + 65] = 4.0;   // speed alta → brake (composição "rápido demais")
    // Open_left_width × 4 (ch 36..39) → steer LEFT (saída 1 negativo). v4: 3.5→2.0
    for i in 36..40 { weights[1 * SWS + i] = -2.0; }
    // Open_right_width × 4 (ch 40..43) → steer RIGHT (positivo). v4: 3.5→2.0
    for i in 40..44 { weights[1 * SWS + i] = 2.0; }
    // Best_escape × 4 (ch 60..63) signed → steer matching. v4: 5→3.
    for i in 60..64 { weights[1 * SWS + i] = 3.0; }

    // A26: RACING LINE FOLLOWER (Tesla HD-map analog).
    // v4: -10/10→-6/6. Menos agressivo — evolução refina o quanto seguir a racing line.
    weights[1 * SWS + 108] = -6.0;  // off-line right → steer LEFT pra retornar
    weights[1 * SWS + 110] = 6.0;   // line vai direita → steer right preemptivo
    weights[1 * SWS + 109] = 3.0;   // d_off improving → manter direção

    // A25: WIDTH-INVARIANT (Tesla-like generalização) — track-width-relative reactions.
    // ch 118 fwd_dist/hw: BAIXO (parede dentro de 1 track-width) → brake forte.
    // ch 119 corridor/hw: BAIXO (corredor estreito) → reduz throttle.
    // ch 123 whisker_min/hw: BAIXO (lateral perto) → brake moderado.
    // Inversão: peso negativo na entrada → ativa output quando canal BAIXO.
    weights[2 * SWS + 118] = -16.0; // canal baixo (wall close relative) → brake
    weights[0 * SWS + 118] = 8.0;   // canal alto (wall far relative) → throttle OK
    weights[0 * SWS + 119] = 6.0;   // corridor amplo → throttle full
    weights[2 * SWS + 119] = -8.0;  // corridor narrow → brake
    weights[2 * SWS + 123] = -10.0; // whisker proximity normalized → brake

    // A24: FOVEAL OPTICAL FLOW — parede vindo na frente (ch 116) = brake máximo.
    // Resolve "vira na parede do nada": canal mostra parede aproximando ANTES dela
    // estar perto, não só presente.
    weights[2 * SWS + 116] = 14.0;  // approach frontal signed → brake forte
    weights[0 * SWS + 116] = -10.0; // → corta throttle
    weights[2 * SWS + 117] = 8.0;   // motion magnitude → brake moderado
    weights[0 * SWS + 117] = -6.0;  // → reduz throttle

    // A23: LOOKAHEAD steer (curv 100m/180m signed) — DNA atual ignora estes (mortos).
    // Sem isso, carro só vê parede via whiskers/fovea = REATIVO. Vira "do nada".
    // curv signed positivo = curva à direita → steer right (positivo).
    weights[1 * SWS + 80] = 8.0;   // curv100m signed → steer matching
    weights[1 * SWS + 32] = 6.0;   // curv180m signed → steer preemptivo
    // Curva forte (qualquer lado) → frear ANTES da parede aparecer.
    weights[2 * SWS + 79] = 10.0;  // brake-prep magnitude → brake
    weights[0 * SWS + 79] = -6.0;  // brake-prep → corta throttle

    // A18: GAZE TRACKER (canais 98-101) — fóvea sacádica aponta pro caminho aberto.
    // v4: 14→5. Total era 56 (14×4 canais), dominava todo o steering.
    // Em narrow track, gaze oscilava violentamente. Agora é refinamento,
    // não dominador. Evolução aprende o quanto confiar no gaze.
    for i in 98..102 {
        weights[1 * SWS + i] = 5.0;
    }

    weights
}

/// Spawna carros no índice `[idx_start..idx_end)`. Permite spawn em blocos entre frames
/// (evita freeze de spawn de 2000 entidades em 1 frame).
pub fn spawn_swarm(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    race: &mut RaceSettings,
    world: &VoxelWorld,
    dev: &DevSettings,
    config: &UniverseConfig,
    lab: &crate::lab::LabParams,
    idx_start: usize,
    idx_end: usize,
) {
    let mut rng = rand::rng();
    let wing = meshes.add(Cuboid::new(0.6, 0.8, 2.0));
    let cockpit = meshes.add(Sphere::new(0.4));
    let names = ["Senna", "Hamilton", "Prost", "Schumi", "Alonso", "Vettel", "Max", "Lauda", "Leclerc", "Norris"];
    let car_models = ["Porsche 911 GT3", "Ferrari 296 GT3", "McLaren 720S GT3", "Aston Martin Vantage", "Audi R8 LMS"];

    let start_flow = world.get_flow_at(world.start_pos.x, world.start_pos.z);
    let start_angle = start_flow.y.atan2(start_flow.x);

    let mut current_best_w = race.best_weights.clone();
    let mut current_best_s = race.best_scale;
    let mut current_best_slow_w = race.best_slow_weights.clone();
    let mut current_best_slow_s = race.best_slow_scale;
    // Tier 6.3 LITE: morpho do champion DNA (se carregado); default caso contrário.
    let mut current_best_morpho: MorphoGenes = MorphoGenes::default();
    // Tier 6.4 LITE: hypernet seed do champion DNA. Default = inactive (zero-cost).
    let mut current_best_hypernet: HypernetSeed = HypernetSeed::default();
    // v3.3: kan_knots herdados do CHAMPION.dna (default = tanh ramp).
    let mut current_best_kan_knots: [[f32; crate::kan::KAN_OUT_KNOTS]; 3] =
        crate::evolution::dna_genesis::default_kan_knots();

    // Mutation rate com decay exponencial por geração: convergência gradual.
    // Floor em 0.005 mantém exploração residual mesmo após 100+ gerações.
    // v9: floor 0.005→0.012 pra sustentar exploração após 100+ gens.
    let effective_mutation_rate = (dev.mutation_rate * 0.97_f32.powi(race.generation as i32 - 1))
        .max(0.012) as f64;

    // Carrega CHAMPION.dna (v3 com magic "IARA" ou legacy v1/v2).
    // PER-WIDTH: tenta CHAMPION_W{width}m.dna primeiro (memória por pista,
    // não desaprende ao mudar de pista), fallback genérico CHAMPION.dna.
    if current_best_w.is_none() {
        let cw = config.current_track().track_width as i32;
        let per_width_path = format!("CHAMPION_W{}m.dna", cw);
        let dna_loaded = load_champion_dna(&per_width_path)
            .or_else(|| load_champion_dna("CHAMPION.dna"));
        if let Some(dna) = dna_loaded {
            current_best_w = Some(if let Some(ref craw) = dna.cortex_raw {
                craw.iter().map(|&w| w as f32 / 63.5).collect()
            } else {
                dna.fast_weights.iter().map(|&w| w as f32 / 63.5).collect()
            });
            // V7: load cortex_l2 weights (Layer 2 deep). Default zeros se ausente
            // (DNAs pré-V7) — cortex_l2 inicializa zeros, mutação preenche.
            if let Some(ref l2) = dna.cortex_l2_raw {
                race.best_l2_weights = Some(l2.iter().map(|&w| w as f32 / 63.5).collect());
            }
            current_best_s = dna.fast_scale;
            // SLWF lossless f32 chunk preferido. Fallback ternary se DNA antigo.
            if let Some(sf32) = dna.slow_f32.as_ref() {
                current_best_slow_w = Some(sf32.clone());
                current_best_slow_s = dna.slow_scale;
            } else if let Some(sw) = dna.slow_weights {
                // Legacy: ternário {-1,0,+1} → magnitude priming 0.33.
                current_best_slow_w = Some(sw.iter().map(|&w| (w as f32) * 0.33).collect());
                current_best_slow_s = dna.slow_scale;
            }

            // Tier 6.3 LITE: absorve morpho genes serializados.
            current_best_morpho = dna.morpho;
            // Tier 6.4 LITE: absorve hypernet seed (default inactive p/ arquivos antigos).
            current_best_hypernet = dna.hypernet;
            // v3.3: absorve KAN knots (preserva spline aprendido entre runs).
            current_best_kan_knots = dna.kan_knots;
            // v5: absorve Trinity gate + reflex weights (TRIN chunk). Restaura swarm
            // aprendido após reboot — sem isso, gate/reflex re-iniciam uniform e
            // perdem o aprendizado da safety net.
            if let Some(ref t) = dna.trinity {
                race.best_gate_weights = Some(t.gate_weights.to_vec());
                race.best_gate_bias = Some(t.gate_bias.to_vec());
                // Dequantize i8 TRIN bytes → f32 runtime weights. Mapping i8/127 covers
                // [-1.0, 1.0]. Cars trained pre-V5 had ternary {-1,0,+1} which maps cleanly.
                let mut rw: Vec<f32> = Vec::with_capacity(96);
                for &b in &t.reflex_steer    { rw.push(b as f32 / 127.0); }
                for &b in &t.reflex_throttle { rw.push(b as f32 / 127.0); }
                for &b in &t.reflex_brake    { rw.push(b as f32 / 127.0); }
                race.best_reflex_weights = Some(rw);
            }
            // Se o DNA foi treinado em outra seed/largura, loga aviso e descarta champion_distance
            // (senão o carro novo nunca consegue coroar num track diferente).
            if dna.version >= 3 {
                let same_track = dna.seed as u64 == race.seed
                    && (dna.track_width - config.current_track().track_width).abs() < 0.1;
                if !same_track {
                    race.champion_distance = 0.0;
                    if let Some(mut file) = open_report_log() {
                        let _ = writeln!(file, "⚠️ CHAMPION.dna veio de outra pista (seed {} width {:.1}) — threshold resetado", dna.seed, dna.track_width);
                    }
                }
                // V6: NÃO pump race.generation pra dna.generation — evolution state
                // (elite_pool, fitness_history, env transitions) começa vazio em UI
                // reload, gen pumped causava mismatch (rain transitions, time dilation
                // skip_n=2 em gen≥10, etc.) que degradava DNA carregado.
            }
        }
    }

    // Helper: tournament k=3 seleciona 1 pai entre k competidores aleatórios do pool.
    let tournament = |pool: &[EliteDna], rng: &mut rand::rngs::ThreadRng| -> usize {
        let mut best_idx = rng.random_range(0..pool.len());
        for _ in 1..3 {
            let cand = rng.random_range(0..pool.len());
            if pool[cand].fitness > pool[best_idx].fitness { best_idx = cand; }
        }
        best_idx
    };

    // Cluster-aware tournament p2 picker: runs standard tournament and rejects
    // any candidate whose niching cluster matches p1's. After 5 failed attempts
    // the fallback accepts a same-cluster parent (pool may be dominated by one
    // niche early on — better to reproduce than stall). Cluster ids are parallel
    // to race.elite_pool; truncated to effective_pool to stay within index range.
    let pick_p2_diff_cluster = |pool: &[EliteDna], clusters: &[usize], p1: usize, rng: &mut rand::rngs::ThreadRng| -> usize {
        let c1 = clusters.get(p1).copied().unwrap_or(usize::MAX);
        for _ in 0..5 {
            let cand = tournament(pool, rng);
            if cand == p1 { continue; }
            let cc = clusters.get(cand).copied().unwrap_or(usize::MAX);
            if cc != c1 { return cand; }
        }
        // Fallback: same-cluster allowed after 5 rejections.
        let cand = tournament(pool, rng);
        if cand == p1 && pool.len() > 1 { (p1 + 1) % pool.len() } else { cand }
    };

    // Helper: BLOCK-WISE crossover (Tier 4.1 — reproducao sexuada).
    // Troca blocos inteiros entre pais em vez de byte-a-byte. Preserva coerencia
    // funcional: cada bloco = linha de entrada de um neuronio (fast: 128 inputs →
    // block_size=128 da a linha inteira; aqui usamos 64 p/ meia-linha, mantendo
    // granularidade razoavel sem fragmentar demais). Para slow_weights usamos
    // block_size=32 (metade dos 8 outputs × 32 inputs).
    let crossover_block = |a: &[f32], b: &[f32], block_size: usize, rng: &mut rand::rngs::ThreadRng| -> Vec<f32> {
        let n = a.len();
        let mut child = vec![0.0f32; n];
        let mut i = 0;
        while i < n {
            let end = (i + block_size).min(n);
            let from = if rng.random_bool(0.5) { a } else { b };
            for j in i..end { child[j] = from[j]; }
            i = end;
        }
        child
    };

    // SPECIES SIMILARITY (f32 weights): fraction de posições onde signos batem.
    let species_similarity = |a: &[f32], b: &[f32]| -> f32 {
        let n = a.len().min(b.len()).max(1);
        let mut agree = 0usize;
        for i in 0..n {
            let sa = a[i].signum();
            let sb = b[i].signum();
            if sa == sb || (sa == 0.0 && sb == 0.0) { agree += 1; }
        }
        agree as f32 / n as f32
    };

    // Helper: BURST MUTATION por canal de entrada.
    // Layout do fast layer: weights[out * 128 + in], 64 outputs × 128 inputs.
    // Burst = escolhe um canal de entrada c e substitui TODAS as 64 conexões
    // (para cada output) por valores aleatórios em [-1, 1]. Quebra neutralidade do BitNet:
    // mutação pontual em layer ternário tem alta probabilidade de virar 0 após quantize —
    // burst força variação coerente de um sensor inteiro.
    let burst_mutate_input_channel = |weights: &mut Vec<f32>, scar_mask_union: &[u8], rng: &mut rand::rngs::ThreadRng| {
        let n_outputs = L1_NEURONS;
        let n_inputs = SWS;
        let c = rng.random_range(0..n_inputs);
        for out in 0..n_outputs {
            let idx = out * n_inputs + c;
            if idx < weights.len() {
                // TIER 4.4: zona protegida isenta de burst mutation também.
                if idx < PROTECTED_ZONE_SIZE { continue; }
                // TIER 6.6: respeita scars — não toca em pesos travados.
                if scar_mask_union.get(idx).copied().unwrap_or(0) > 0 { continue; }
                weights[idx] = rng.random_range(-1.0..1.0);
            }
        }
    };

    // TIER 1.2 — AXIOM DECAY.
    // Antes: reinject_axioms forçava pesos priming a cada gen (freezava evolução).
    // Agora: axiom_strength cai linear de 1.0 (gen 0) até 0.0 (gen 50) — a partir de
    // gen 50 evolução é totalmente livre. Emergency brake (axiom_boost_counter > 0)
    // restaura strength mínimo 0.5 se fitness degradar > 50%.
    //
    // Blend: new = current * (1 - s) + axiom * s, depois round + clamp ternário.
    // Estratégia: só age em canais críticos (mesma lista de antes) em todos os 3 outputs
    // (throttle/steer/brake). Outros canais ficam totalmente livres pra evolução.
    let gen_f = race.generation as f32;
    // v15: floor 0.15 → 0.35. Gen 1 produziu 7 laps (priming ninja), evolução
    // degrada após gen 50 (axiom 0.15 insuficiente). Floor alto mantém prior
    // forte sem congelar evolução (65% espaço livre nos pesos).
    let mut axiom_strength = (1.0 - gen_f / lab.axiom_decay_divisor).max(lab.axiom_floor);
    if race.axiom_boost_counter > 0 {
        axiom_strength = axiom_strength.max(0.5);
    }
    race.current_axiom_strength = axiom_strength;
    let apply_axiom_decay = |weights: &mut Vec<f32>| {
        if axiom_strength < 0.01 {
            return; // gen 50+ (sem brake) → evolução totalmente livre.
        }
        let priming = build_priming_weights();
        let critical: &[usize] = &[
            0, 3, 4, 7,             // fóvea bins (centro + extremos)
            8, 9, 10, 11, 12, 13, 14, 15, // periférica bins 8 (pós-pooling)
            64,                     // alinhamento fluxo
            79, 80, 82,             // curvatura + ttc
            85, 86,                 // parede memória L/R
            90, 91, 92, 93,         // dark-center A16
            94, 95, 96, 97,         // whiskers A17
            98, 99, 100, 101,       // gaze A18
        ];
        for &ch in critical {
            for out in 0..3 {
                let idx = out * SWS + ch;
                if idx >= weights.len() { continue; }
                let current = weights[idx];
                let axiom = priming[idx];
                if axiom.abs() < 0.01 { continue; } // sem axioma neste slot — deixa evoluir livre.
                let blended = current * (1.0 - axiom_strength) + axiom * axiom_strength;
                // Após quantize ternária cada peso vira ±1 ou 0; garante que o sinal
                // do blend sobreviva via round+clamp pra ternário.
                weights[idx] = blended.round().clamp(-1.0, 1.0);
            }
        }
    };

    // Tier 6.4 LITE: mutação do hypernet seed 1x por gen (primeiro batch).
    // 2% chance de flip do active; se ativo, gaussian noise σ=0.05 nos coeffs.
    // Mutação é runtime-only nesta LITE slice: persistência via .dna requer
    // follow-up em pipeline.rs (fora do escopo). Zero-cost quando inactive.
    if idx_start == 0 {
        current_best_hypernet.mutate(|| rng.random::<f32>());
    }
    // Expose for pipeline save (Tier 6.3/6.4 persistência).
    race.current_morpho = current_best_morpho;
    race.current_hypernet = current_best_hypernet;

    let pool_size = race.elite_pool.len();
    // Snapshot niching clusters (parallel to elite_pool). Empty until pipeline
    // populates it on first post-race pass; gen 0 spawn hits this before any
    // clustering, so default is empty and pick_p2_diff_cluster degrades cleanly
    // to standard tournament.
    let elite_clusters: Vec<usize> = race.elite_clusters.clone();

    // Tier 4.2: acumula sample count de inheritance + soma para log agregado no final.
    let mut epi_inherited_count: usize = 0;
    let mut epi_cortisol_sum: f32 = 0.0;
    let mut epi_dopamine_sum: f32 = 0.0;
    let mut epi_curiosity_sum: f32 = 0.0;

    for i in idx_start..idx_end.min(config.swarm_count) {
        let mut engine = crate::engine::SovereignEngine::new();
        // Tier 4.2: herança epigenética parcial (10%) — se houver parent(s).
        // Preenchido nas branches de reprodução; None = organismo de priming/boot.
        let mut epigenetic_parents: Option<(f32, f32, f32)> = None;
        // Tier 6.3 LITE: morpho do novo racer. Preenchido nas branches conforme fonte
        // (champion / elite clone / parents crossover / priming). Mutado e clampado.
        let mut morpho_this: MorphoGenes = MorphoGenes::default();

        // --- HALL OF FAME (CHAMPION FALLBACK) ---
        // Slot 0: SEMPRE carrega o CHAMPION DNA se existe. Imune à deriva geracional
        // (até mesmo se o elite_pool inteiro for ruim, o HoF preserva o melhor de todos os tempos).
        let slot_is_champion = i == 0 && current_best_w.is_some();

        // SIMPLE ELITISM v2: slot 0 = champion puro, slots 1..(N-15) = champion + mutação,
        // últimos 15 slots = PRIMING REFRESH (fresh DNA) pra evitar monoculture.
        // User reportou: gen 2 TODOS batem mesma curva (clones do bug do champion).
        // Priming refresh dá diversidade — se champion stuck em local minimum,
        // priming cars exploram alternativas.
        let priming_slot = i >= config.swarm_count.saturating_sub(5);
        #[cfg(feature = "simple_elitism")]
        let simple_mut_slot = !slot_is_champion && !priming_slot && current_best_w.is_some();
        #[cfg(not(feature = "simple_elitism"))]
        let simple_mut_slot = false;

        if priming_slot {
            // PRIMING REFRESH: fresh DNA com axiomas. Diversidade vs monoculture.
            let priming = build_priming_weights();
            // Adiciona ruído pra cada slot ser único (sem isso 15 priming idênticos).
            let mut jittered = priming;
            for w in jittered.iter_mut() {
                if rng.random_bool(0.02) { *w += (rng.random::<f32>() - 0.5) * 0.5; }
            }
            engine.instinct_layer.quantize(&jittered);
            engine.instinct_layer.set_scale(1.0);
            // FIX BUG: priming refresh slots não primavam slow brain — System 2 ficava zerado.
            // Resultado: champion eleito desse slot tinha slow weights = 0 → save todo zero.
            let slow_priming: Vec<f32> = build_slow_priming_weights()
                .into_iter().map(|w| w / 3.0).collect();
            engine.slow_brain.layer.set_weights(&slow_priming);
        } else if slot_is_champion || simple_mut_slot {
            if let Some(ref champion) = current_best_w {
                engine.instinct_layer.quantize(champion);
                engine.instinct_layer.set_scale(current_best_s);
                if let Some(ref l2) = race.best_l2_weights {
                    let n = l2.len().min(engine.cortex_l2.weights.len());
                    engine.cortex_l2.weights[..n].copy_from_slice(&l2[..n]);
                }
                // CHAMPION VERIFY: printar checksum simples do DNA carregado em slot 0.
                // User pediu validação que DNA campeão realmente carrega entre gens.
                if i == 0 {
                    let cksum: f32 = champion.iter().take(64).sum();
                    let l2_cksum: f32 = race.best_l2_weights.as_ref()
                        .map(|v| v.iter().take(64).sum::<f32>()).unwrap_or(0.0);
                    let nonzero_l1 = champion.iter().filter(|&&w| w.abs() > 0.01).count();
                    println!("✓ SLOT 0 CHAMPION gen={} L1_sum={:.3} L2_sum={:.3} L1_active={} (DNA carregado)",
                        race.generation, cksum, l2_cksum, nonzero_l1);
                }
                // Hypernet DISABLED for f32 cortex — expansion era ternary {-1,0,1}/127
                // ≈ ±0.008 magnitude que sobrescrevia weights treinados f32 (~±2.0)
                // destruindo a rede no spawn. Hypernet ficará pra V2 com basis f32 real.
                let _ = current_best_hypernet;
                if let Some(ref l3) = race.best_l3_weights {
                    engine.cortex_l3.set_weights(l3);
                }
                // BUG FIX: carregar do current_best_slow_w que veio do DNA (sf32), e não do race
                // que estava vazio no start.
                if let Some(ref slow) = current_best_slow_w {
                    engine.slow_brain.layer.set_weights(slow);
                }
            }
            // Tier 6.3 LITE: champion preserva morpho exata (sem mutação no HoF).
            morpho_this = current_best_morpho;

            // simple_elitism: slot != 0 sofre mutação gaussiana suave (f32 runtime).
            // Box-Muller σ=0.05 — pequenos ajustes em vez de flips disruptivos.
            if simple_mut_slot {
                let rate_boosted = (effective_mutation_rate * 2.0).min(0.25);
                // STRATIFIED mutation: slots top próximos do champion (clones quase
                // puros), slots fundo exploram. User report: gen N+1 majority não
                // replica champion da gen N. Antes σ uniform 0.06+ destruía herança.
                let g = race.generation.max(1);
                let base_decay = 0.04_f32 + 0.06_f32 * 0.88_f32.powi((g as i32 - 1).max(0));
                let sigma_cortex = match i {
                    0..=4   => base_decay * 0.15,  // clones quase exatos (σ ≈ 0.006)
                    5..=14  => base_decay * 0.5,   // small variance (σ ≈ 0.02)
                    15..=29 => base_decay * 1.0,   // medium (σ ≈ 0.04)
                    _       => base_decay * 2.0,   // explore (σ ≈ 0.08-0.20)
                };
                for w in engine.instinct_layer.weights.iter_mut() {
                    if rng.random_bool(rate_boosted) {
                        let u1 = (rng.random::<f32>()).max(1e-7);
                        let u2 = rng.random::<f32>();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        *w = (*w + sigma_cortex * z).clamp(-2.0, 2.0);
                    }
                }
                for w in engine.cortex_l2.weights.iter_mut() {
                    if rng.random_bool(rate_boosted) {
                        let u1 = (rng.random::<f32>()).max(1e-7);
                        let u2 = rng.random::<f32>();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        *w = (*w + sigma_cortex * z).clamp(-2.0, 2.0);
                    }
                }
                // Mutate cortex_l3 (Layer 3 — deep abstract).
                for w in engine.cortex_l3.weights.iter_mut() {
                    if rng.random_bool(rate_boosted) {
                        let u1 = (rng.random::<f32>()).max(1e-7);
                        let u2 = rng.random::<f32>();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        *w = (*w + sigma_cortex * z).clamp(-2.0, 2.0);
                    }
                }
                // Mutação leve no slow brain também (scale menor porque é float).
                for w in engine.slow_brain.layer.weights.iter_mut() {
                    if rng.random_bool(rate_boosted * 0.5) {
                        *w += (rng.random::<f32>() - 0.5) * 0.3;
                    }
                }
                // Morpho jittered leve — mantém proximidade do champion
                // (refator sem dep em MorphoGenes::jitter extra).
            }

            // ---- Trinity: copy champion gate + reflex into this engine ----
            if let Some(ref gw) = race.best_gate_weights {
                let n = gw.len().min(engine.gate.weights.len());
                engine.gate.weights[..n].copy_from_slice(&gw[..n]);
            }
            if let Some(ref gb) = race.best_gate_bias {
                let n = gb.len().min(engine.gate.bias.len());
                engine.gate.bias[..n].copy_from_slice(&gb[..n]);
            }
            if let Some(ref rw) = race.best_reflex_weights {
                // SpecialistBrain has 3 FloatLinears: steer (32), throttle (32), brake (32) = 96 f32 weights.
                let steer_len = engine.reflex.steer.weights.len();
                let throttle_len = engine.reflex.throttle.weights.len();
                let brake_len = engine.reflex.brake.weights.len();
                let total = steer_len + throttle_len + brake_len;
                if rw.len() >= total {
                    engine.reflex.steer.weights.copy_from_slice(&rw[0..steer_len]);
                    engine.reflex.throttle.weights.copy_from_slice(&rw[steer_len..steer_len + throttle_len]);
                    engine.reflex.brake.weights.copy_from_slice(&rw[steer_len + throttle_len..steer_len + throttle_len + brake_len]);
                }
            }

            // ---- Trinity: mutate gate + reflex on simple_mut_slot ----
            if simple_mut_slot {
                let rate_boosted = (effective_mutation_rate * 2.0).min(0.25);
                let rate = (rate_boosted * 0.5) as f64; // half rate vs cortex — gate/reflex more sensitive
                for w in engine.gate.weights.iter_mut() {
                    if rng.random_bool(rate) {
                        *w += (rng.random::<f32>() - 0.5) * 0.2;
                    }
                }
                for b in engine.gate.bias.iter_mut() {
                    if rng.random_bool(rate) {
                        *b += (rng.random::<f32>() - 0.5) * 0.1;
                    }
                }
                // Reflex weights são f32 (FloatLinear). Mutate gaussian smooth — pequenos
                // ajustes em vez de flip ternário disruptivo. σ=0.05, clamp [-2.0, 2.0].
                for layer in [&mut engine.reflex.steer, &mut engine.reflex.throttle, &mut engine.reflex.brake] {
                    for w in layer.weights.iter_mut() {
                        if rng.random_bool(rate) {
                            // Box-Muller approx via 2 uniforms; cheap noise.
                            let u1 = (rng.random::<f32>()).max(1e-7);
                            let u2 = rng.random::<f32>();
                            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                            *w = (*w + 0.05 * z).clamp(-2.0, 2.0);
                        }
                    }
                }
            }
            // ---- end Trinity ----
        } else if pool_size >= 2 {
            // --- REPRODUÇÃO DIVERSIFICADA NO GRID (30 slots de largada) ---
            // Gargalo anterior: só 5 grid spawns × 2 elite clones = 2 DNAs testados
            // da largada por gen. DNA de 935m ficava na população distribuída sem
            // nunca ser validado em "start from zero". Agora espalhamos o pool
            // inteiro pelos 30 slots grid:
            //   Slot 0:      CHAMPION (já tratado acima)
            //   Slot 1-10:   top-10 elite clones (cada elite testado no grid)
            //   Slot 11-20:  PRIMING REFRESH (instintos frescos na largada)
            //   Slot 21-29:  tournament crossover (filhos diversos no grid)
            //   Slot 30+:    tournament crossover (distributed, cobertura de pista)
            let elite_clones = pool_size.min(10);
            let refresh_slots = 10;
            let elite_range_end = 1 + elite_clones; // [1, 1+elite_clones)

            if i >= 1 && i < elite_range_end {
                let elite = &race.elite_pool[i - 1];
                let exact: Vec<f32> = elite.fast_weights.iter().map(|&w| w as f32).collect();
                engine.instinct_layer.quantize(&exact);
                engine.instinct_layer.set_scale(elite.fast_scale);
                
                engine.slow_brain.layer.set_weights(&elite.slow_weights);
                    engine.cortex_l2.set_weights(&elite.cortex_l2);
                    engine.cortex_l3.set_weights(&elite.cortex_l3);
                let _ = elite.slow_scale;
                // Tier 4.2: clone herda 10% do estado hormonal médio do único parent.
                epigenetic_parents = Some((elite.cortisol_mean, elite.dopamine_mean, elite.curiosity_mean));
                // Tier 6.3 LITE: clone herda morpho do elite + mutação leve.
                morpho_this = elite.morpho;
                morpho_this.mutate(|| rng.random::<f32>());
            } else if i < elite_range_end + refresh_slots {
                // PRIMING REFRESH: axiomas frescos + ruído leve para diversidade.
                // Preserva instintos mas ainda permite exploração local.
                let priming = build_priming_weights();
                let jittered: Vec<f32> = priming.into_iter().map(|w| {
                    if rng.random_bool(0.005) { rng.random_range(-1.0..1.0) } else { w }
                }).collect();
                engine.instinct_layer.quantize(&jittered);
                // FIX: quantize gera scale=abs_mean (~0.055 para priming esparso) →
                // sinal fica ~0.02 no forward, tanh saturado em ~0.04 → throttle zero.
                // Força scale=1.0 pra priming ter força competitiva e poder vencer
                // tournament. Sem isso axiomas nunca chegam no campeão (gen 13 mostrou
                // canal 79 curvatura em 4.7%, era pra estar >30%).
                engine.instinct_layer.set_scale(1.0);
                // FIX: slow_brain também recebe priming diagonal pra sair do zero absoluto.
                // Option 4: FloatLinear — priming /3.0 porque ternarização saturava em ±1.
                let slow_priming: Vec<f32> = build_slow_priming_weights()
                    .into_iter().map(|w| w / 3.0).collect();
                engine.slow_brain.layer.set_weights(&slow_priming);
            } else {
                // Tier 4.1 — MATE PREFERENCE.
                // Seleciona p1 via tournament (top pool). Escolhe p2 ponderando por
                // similaridade de species:
                //   similaridade > 0.7 (mesma species): peso 0.7 (especializacao)
                //   similaridade <= 0.7 (species diferentes): peso 0.3 (diversidade)
                // Pool efetivo: top-10 do elite_pool (ja ordenado por fitness).
                let effective_pool = pool_size.min(10);
                let p1 = tournament(&race.elite_pool[..effective_pool], &mut rng);
                // NICHING: p2 via cluster-aware rejection tournament (different
                // BehaviorFingerprint cluster from p1, up to 5 attempts, fallback
                // to same-cluster). If clusters not yet populated (gen 1 boot),
                // fall back to legacy species-similarity mate preference.
                let clusters_slice: &[usize] = if elite_clusters.len() >= effective_pool {
                    &elite_clusters[..effective_pool]
                } else { &[] };
                let p2 = if !clusters_slice.is_empty() {
                    pick_p2_diff_cluster(&race.elite_pool[..effective_pool], clusters_slice, p1, &mut rng)
                } else {
                    let parent_a_ref = &race.elite_pool[p1];
                    let mut weights_pref: Vec<f32> = Vec::with_capacity(effective_pool);
                    for j in 0..effective_pool {
                        if j == p1 { weights_pref.push(0.0); continue; }
                        let sim = species_similarity(&parent_a_ref.fast_weights, &race.elite_pool[j].fast_weights);
                        let w = if sim > 0.7 { 0.7 } else { 0.3 };
                        weights_pref.push(w);
                    }
                    let total_w: f32 = weights_pref.iter().sum();
                    if total_w > 1e-6 {
                        let mut target = rng.random_range(0.0..total_w);
                        let mut pick = 0usize;
                        for (j, &w) in weights_pref.iter().enumerate() {
                            target -= w;
                            if target <= 0.0 { pick = j; break; }
                            pick = j;
                        }
                        pick
                    } else if pool_size > 1 { (p1 + 1) % pool_size } else { p1 }
                };
                let parent_a = &race.elite_pool[p1];
                let parent_b = &race.elite_pool[p2];

                // Tier 4.1 stats: log da mate preference usada (1x por gen, primeiro slot crossover).
                if i == elite_range_end + refresh_slots {
                    let sim = species_similarity(&parent_a.fast_weights, &parent_b.fast_weights);
                    let kind = if sim > 0.7 { "same-species" } else { "cross-species" };
                    if let Some(mut file) = open_report_log() {
                        let _ = writeln!(file, "[MATE] gen {} pool={} p1={} p2={} sim={:.2} ({})",
                            race.generation, effective_pool, p1, p2, sim, kind);
                    }
                }

                let mut crossed_fast = crossover_block(&parent_a.fast_weights, &parent_b.fast_weights, 64, &mut rng);
                // TIER 4.4 — ZONA PROTEGIDA: primeiros 819 índices (10%) são copiados
                // direto do parent_a, sem block-alternate. Preserva axiomas ancestrais.
                for idx in 0..PROTECTED_ZONE_SIZE.min(crossed_fast.len()) {
                    crossed_fast[idx] = parent_a.fast_weights[idx] as f32;
                }
                // TIER 6.6 — WEIGHT SCARS: peso com scar TTL>0 em QUALQUER pai é lockdown pra mutação.
                // União: se parent_a.scar_mask[i] > 0 OR parent_b.scar_mask[i] > 0 → protege.
                // O valor herdado no crossover já ficou; mutação é o que skipamos.
                let scar_a = &parent_a.scar_mask;
                let scar_b = &parent_b.scar_mask;
                let mut mutated_fast: Vec<f32> = crossed_fast.into_iter().enumerate().map(|(idx, w)| {
                    // TIER 4.4: zona protegida isenta de mutação.
                    if idx < PROTECTED_ZONE_SIZE { return w; }
                    let scarred = scar_a.get(idx).copied().unwrap_or(0) > 0
                        || scar_b.get(idx).copied().unwrap_or(0) > 0;
                    if !scarred && rng.random_bool(effective_mutation_rate) {
                        rng.random_range(-1.0..1.0)
                    } else { w }
                }).collect();
                // 20% dos slots crossover sofrem burst por canal de entrada (além da mutação pontual).
                if rng.random_bool(0.20) {
                    // União de scars dos dois pais (mesma lógica da mutação pontual acima).
                    let scar_union: Vec<u8> = scar_a.iter().zip(scar_b.iter())
                        .map(|(&a, &b)| a.max(b)).collect();
                    burst_mutate_input_channel(&mut mutated_fast, &scar_union, &mut rng);
                }
                // TIER 1.2: decay axiomático — strength linear 1→0 até gen 50.
                apply_axiom_decay(&mut mutated_fast);
                engine.instinct_layer.quantize(&mutated_fast);
                engine.instinct_layer.set_scale((parent_a.fast_scale + parent_b.fast_scale) * 0.5);

                // slow_weights are i8 (slow_brain still ternary). Convert to f32 for crossover.
                let pa_slow_f: Vec<f32> = parent_a.slow_weights.iter().map(|&w| w as f32).collect();
                let pb_slow_f: Vec<f32> = parent_b.slow_weights.iter().map(|&w| w as f32).collect();
                let crossed_slow = crossover_block(&pa_slow_f, &pb_slow_f, 32, &mut rng);
                let mut mutated_slow: Vec<f32> = crossed_slow.into_iter().map(|w| w / 63.5).collect();
                for w in mutated_slow.iter_mut() {
                    if rng.random_bool(effective_mutation_rate) {
                        // Box-Muller gaussian σ=0.02.
                        let u1 = (rng.random::<f32>()).max(1e-7);
                        let u2 = rng.random::<f32>();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        *w = (*w + 0.02 * z).clamp(-2.0, 2.0);
                    }
                }
                engine.slow_brain.layer.set_weights(&mutated_slow);
                // Tier 4.2: herança hormonal MÉDIA dos 2 pais (reprodução sexuada).
                let cort = (parent_a.cortisol_mean + parent_b.cortisol_mean) * 0.5;
                let dopa = (parent_a.dopamine_mean + parent_b.dopamine_mean) * 0.5;
                let curi = (parent_a.curiosity_mean + parent_b.curiosity_mean) * 0.5;
                epigenetic_parents = Some((cort, dopa, curi));
                // Tier 6.3 LITE: morpho = média dos pais + mutação gaussiana.
                let pa_m = parent_a.morpho;
                let pb_m = parent_b.morpho;
                morpho_this = MorphoGenes {
                    mass:      (pa_m.mass      + pb_m.mass)      * 0.5,
                    wheelbase: (pa_m.wheelbase + pb_m.wheelbase) * 0.5,
                    downforce: (pa_m.downforce + pb_m.downforce) * 0.5,
                    grip_mult: (pa_m.grip_mult + pb_m.grip_mult) * 0.5,
                }.sanitized();
                morpho_this.mutate(|| rng.random::<f32>());
            }
        } else if let Some(ref best) = current_best_w {
            // Tier 6.3 LITE: fallback sem pool → herda morpho do champion (+ mutação leve se i>0).
            morpho_this = current_best_morpho;
            if i >= 3 { morpho_this.mutate(|| rng.random::<f32>()); }
            // --- FALLBACK: DNA de CHAMPION.dna sem pool (primeira geração após boot) ---
            // Slots 0,1,2 = clones exatos do champion (3x elitism buffer). Slots 3+ mutam.
            if i < 3 {
                let exact: Vec<f32> = best.iter().map(|&w| w as f32).collect();
                engine.instinct_layer.quantize(&exact);
                engine.instinct_layer.set_scale(current_best_s);
                if let Some(ref best_slow) = current_best_slow_w {
                    // FIX: best_slow já é f32 normalizado (load /63.5 em line 265). Não dividir de novo.
                    engine.slow_brain.layer.set_weights(best_slow);
                }
                if let Some(ref l2) = race.best_l2_weights {
                    engine.cortex_l2.set_weights(l2);
                }
                if let Some(ref l3) = race.best_l3_weights {
                    engine.cortex_l3.set_weights(l3);
                }
            } else {
                let mut mutated: Vec<f32> = best.iter().enumerate().map(|(idx, &w)| {
                    // TIER 4.4: zona protegida isenta de mutação.
                    if idx < PROTECTED_ZONE_SIZE { return w as f32; }
                    if rng.random_bool(effective_mutation_rate) { rng.random_range(-1.0..1.0) } else { w as f32 }
                }).collect();
                apply_axiom_decay(&mut mutated);
                engine.instinct_layer.quantize(&mutated);
                engine.instinct_layer.set_scale(current_best_s);
                if let Some(ref best_slow) = current_best_slow_w {
                    // Option 4: f32 base + gaussian mutation σ=0.02.
                    // FIX: best_slow já é f32 normalizado (load fez /63.5 em line 265).
                    // Bug anterior: dividia /63.5 de novo → 0.33 / 63.5 ≈ 0.005 → save round 0.
                    // Slow brain morria toda gen por double normalization.
                    let mut mutated_slow: Vec<f32> = best_slow.clone();
                    for w in mutated_slow.iter_mut() {
                        if rng.random_bool(effective_mutation_rate) {
                            let u1 = (rng.random::<f32>()).max(1e-7);
                            let u2 = rng.random::<f32>();
                            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                            *w = (*w + 0.02 * z).clamp(-2.0, 2.0);
                        }
                    }
                    engine.slow_brain.layer.set_weights(&mutated_slow);
                }
            }
        } else {
            // --- PRIMING GENÉTICO (GERAÇÃO 0, SEM POOL/DNA) ---
            // Axiomas A1–A15 definidos em build_priming_weights(). Magnitudes grandes
            // pra sobreviver quantize ternária (bits ±1).
            let weights = build_priming_weights();
            engine.instinct_layer.quantize(&weights);
            engine.instinct_layer.set_scale(1.0);
            // FIX: gen 0 sem DNA também prima slow brain (Option 4: f32, /3.0 damp).
            let slow_priming: Vec<f32> = build_slow_priming_weights()
                .into_iter().map(|w| w / 3.0).collect();
            engine.slow_brain.layer.set_weights(&slow_priming);
        }

        // Tier 4.2: seeda metabolismo do filho com 10% do estado hormonal do(s) parent(s).
        // Cria linhagens com "trauma" (cortisol alto) ou "calma" (cortisol baixo).
        if let Some((p_cort, p_dopa, p_curi)) = epigenetic_parents {
            engine.metabolism.seed_epigenetic(p_cort, p_dopa, p_curi);
            epi_inherited_count += 1;
            epi_cortisol_sum += p_cort;
            epi_dopamine_sum += p_dopa;
            epi_curiosity_sum += p_curi;
        }

        let color = if i == 0 && current_best_w.is_some() { Color::srgb(0.0, 1.0, 0.8) } else { Color::srgb(rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>()) };
        let racer_mat = materials.add(StandardMaterial { base_color: color, emissive: LinearRgba::from(color) * 4.0, perceptual_roughness: 0.1, metallic: 0.9, ..default() });

        // --- SPAWN DISTRIBUÍDO AO LONGO DA PISTA (Andrychowicz et al, 2017) ---
        // Carros 0..29 (champion + 10 elites + 10 priming + 9 crossover) começam na
        // largada — 30 tentativas independentes do DNA bom, o suficiente pra que pelo
        // menos um sobreviva à largada e prove skill transferível.
        // Carros 30+ spawnam em pontos espalhados pela pista (coverage).
        // Fitness usa NET progress (track_progress - spawn_progress), então carros
        // que nasceram na curva e sobreviveram 30m PROVARAM habilidade de curva.
        // Isso quebra o platô deceptivo de "só reta dá fitness alto".
        // simple_elitism: TODOS no grid. Sem distributed spawn.
        // Cada carro é champion+mutação, todos partem do zero = comparação justa.
        #[cfg(feature = "simple_elitism")]
        const GRID_SPAWNS: usize = usize::MAX;
        #[cfg(not(feature = "simple_elitism"))]
        const GRID_SPAWNS: usize = 30; // rollback v7 (grid 50 regressed)
        let (spawn_pos, spawn_rot, spawn_progress) = if i < GRID_SPAWNS {
            // Grid spawn ao longo dos waypoints (segue a pista). Cada row recua 6m
            // em arco da pista, lateral spread por col mantém dentro do road width.
            // Antes: avançava reto da start_pos → curvas Catmull-Rom punham carros
            // fora da pista. Agora: pega waypoint segment N atrás e posiciona lá.
            #[cfg(feature = "simple_elitism")]
            const GRID_COLS: usize = 4;
            #[cfg(not(feature = "simple_elitism"))]
            const GRID_COLS: usize = 2;
            let row = i / GRID_COLS;
            let col = i % GRID_COLS;
            let col_offset = col as f32 - (GRID_COLS as f32 - 1.0) / 2.0;
            let back_dist = row as f32 * 6.0; // metros pra trás ao longo da pista

            // Localiza waypoint do segmento da start_pos.
            let n_wp = world.waypoints_world.len();
            let start_progress = world.start_progress;
            let lap = world.lap_distance.max(1.0);
            // Progresso alvo (atrás da start, com wrap).
            let target_progress = (start_progress - back_dist + lap) % lap;
            // Acha segmento que cobre target_progress via waypoint_arcs.
            let mut seg = 0usize;
            for k in 0..n_wp.saturating_sub(1) {
                if world.waypoint_arcs[k] <= target_progress
                   && target_progress < *world.waypoint_arcs.get(k+1).unwrap_or(&lap) {
                    seg = k;
                    break;
                }
            }
            let wp_a = world.waypoints_world[seg];
            let wp_b = world.waypoints_world[(seg + 1).min(n_wp - 1)];
            let seg_arc_a = world.waypoint_arcs[seg];
            let seg_arc_b = *world.waypoint_arcs.get(seg+1).unwrap_or(&lap);
            let t = if seg_arc_b > seg_arc_a {
                ((target_progress - seg_arc_a) / (seg_arc_b - seg_arc_a)).clamp(0.0, 1.0)
            } else { 0.0 };
            let center_2d = wp_a + (wp_b - wp_a) * t;
            let dir_2d = (wp_b - wp_a).normalize_or_zero();
            let perp_2d = Vec2::new(-dir_2d.y, dir_2d.x);
            let pos_2d = center_2d + perp_2d * (col_offset * 3.0);
            let p = Vec3::new(pos_2d.x, world.start_pos.y, pos_2d.y);
            let angle = dir_2d.y.atan2(dir_2d.x);
            // GRID spawn: spawn_progress=0 marca como "largada" pra is_grid_spawn
            // gate de crown. Net progress depois subtrai 0 → puro track_progress.
            // (Antes target_progress confundia gate, impedia DNA crown).
            let _ = target_progress;
            (p, Quat::from_rotation_y(-angle), 0.0_f32)
        } else {
            // Distribui ao longo dos waypoints. Fração = 25%, 50%, 75%, 87%... em faixa circular.
            let distributed_idx = (i - GRID_SPAWNS) as f32;
            let total_distributed = (config.swarm_count - GRID_SPAWNS).max(1) as f32;
            let fraction = 0.15 + (distributed_idx / total_distributed) * 0.7; // [0.15..0.85]
            let wp_count = world.waypoints_world.len().max(2);
            let target_wp = ((fraction * wp_count as f32) as usize).min(wp_count - 2);
            let wp = world.waypoints_world[target_wp];
            let wp_next = world.waypoints_world[target_wp + 1];
            let dir = (wp_next - wp).normalize_or_zero();
            let p = Vec3::new(wp.x, world.start_pos.y, wp.y);
            let angle = dir.y.atan2(dir.x);
            let progress = world.get_track_progress(p.x, p.z);
            (p, Quat::from_rotation_y(-angle), progress)
        };
        let elite_range_end = 1 + race.elite_pool.len();
        let mut spawn_model_idx = i % 5;
        if slot_is_champion {
            if let Some(idx) = race.best_car_model_idx {
                spawn_model_idx = idx as usize;
            }
        } else if i >= 1 && i < elite_range_end {
            if i - 1 < race.elite_pool.len() {
                let elite = &race.elite_pool[i - 1];
                spawn_model_idx = elite.car_model_idx as usize;
            }
        }
        let (racer_entity, car_model) = match spawn_model_idx % 5 {
            0 => (crate::prefabs::spawn_porsche_911_gt3(commands, Transform::from_translation(spawn_pos).with_rotation(spawn_rot), &mut *meshes, racer_mat.clone()), car_models[0]),
            1 => (crate::prefabs::spawn_ferrari_296_gt3(commands, Transform::from_translation(spawn_pos).with_rotation(spawn_rot), &mut *meshes, racer_mat.clone()), car_models[1]),
            2 => (crate::prefabs::spawn_mclaren_720s_gt3(commands, Transform::from_translation(spawn_pos).with_rotation(spawn_rot), &mut *meshes, racer_mat.clone()), car_models[2]),
            3 => (crate::prefabs::spawn_aston_vantage_gt3(commands, Transform::from_translation(spawn_pos).with_rotation(spawn_rot), &mut *meshes, racer_mat.clone()), car_models[3]),
            _ => (crate::prefabs::spawn_audi_r8_lms_gt3(commands, Transform::from_translation(spawn_pos).with_rotation(spawn_rot), &mut *meshes, racer_mat.clone()), car_models[4]),
        };

        commands.entity(racer_entity).insert(
            Racer {
                name: if i == 0 && current_best_w.is_some() { "👑 CHAMPION".to_string() } else { format!("{}-{}", names[i % names.len()], rng.random_range(10..99)) },
                car_model: car_model.to_string(),
                car_model_idx: (spawn_model_idx % 5) as u8,
                engine, is_dead: false, distance: 0.0, phase: 0.0, amplitude: 0.0,
                pain_memories: Vec::new(), stagnation: 0.0, prev_pos: spawn_pos, fuel: 1.0, hp: 100.0, total_laps: 0,
                memory: [0.0; 5], visual_cortex: [0; RETINA_TOTAL],
                tire_temp: 0.0, engine_power: 1.0, grass_time: 0.0, wall_time: 0.0, aquaplaning_timer: 0.0,
                joy_integral: 0.0, joy_streak: 0.0, wall_hits: 0, wall_cooldown: 0.0, last_waypoint: 0, track_progress: spawn_progress, grace_ticks: 0,
                reverse_timer: 0.0,
                progress_checkpoint: spawn_progress,
                progress_stall: 0.0,
                wall_ema_l: 15.0,
                wall_ema_r: 15.0,
                wall_dist_prev_l: 15.0,
                wall_dist_prev_r: 15.0,
                wall_dist_prev_f: 50.0,
                proximity_sustain: 0.0,
                post_reverse_lockout: 0.0,
                spawn_progress,
                path: vec![spawn_pos],
                path_timer: 0.0,
                brain_trace: Vec::with_capacity(15),
                trace_timer: 0.0,
                gaze_yaw: 0.0,
                max_progress: spawn_progress,
                slow_time: 0.0,
                race_time: 0.0,
                prev_lap_prog: spawn_progress.rem_euclid(world.lap_distance.max(1.0)),
                post_lap_grace: 0,
                post_lap_checkpoint: 0.0,
                death_reason: None,
                death_frame: 0,
                death_pos: Vec3::ZERO,
                peak_speed: 0.0,
                telemetry: crate::simulation::racer::RacerTelemetry::new(),
                brain_tick_counter: 0,
                last_action: (0.0, 0.0),
                // Tier 4.3: alterna nicho por índice. 50% defensive / 50% aggressive.
                niche: (i % 2) as u8,
                // Tier 5.1 LITE: data layer — zerado no spawn; populado via
                // EnvironmentalState::closest_racers em follow-up (não wired).
                perceived_rivals: [0.0; 6],
                // Tier 6.3 LITE: morpho genes herdados/mutados conforme branch de reprodução.
                morpho: morpho_this.sanitized(),
                // v13 KAN output head: 5% injection (v12 10% → HOLE=5181 regression).
                // Fresh heads untreinadas caíam do track. Mut rate 0.30→0.15.
                kan_out: {
                    let mut head = crate::kan::KanOutputHead::new();
                    // Fallback (pool vazio, CHAMPION.dna carregado): slot 0 usa knots do DNA;
                    // outros slots herdam + mutação moderada p/ diversidade.
                    if race.elite_pool.is_empty() {
                        if current_best_w.is_some() {
                            head.knots = current_best_kan_knots;
                            if i >= 3 {
                                use rand::prelude::*;
                                let mut local_rng = rand::rng();
                                head.mutate_spline_knots(|| local_rng.random::<f32>(), 0.15);
                            }
                        }
                    } else {
                        use rand::prelude::*;
                        let mut local_rng = rand::rng();
                        let pool_sz = race.elite_pool.len();
                        if i == 0 && current_best_w.is_some() {
                            head.knots = race.elite_pool[0].kan_knots;
                        } else if local_rng.random::<f32>() < 0.05 {
                            // INJEÇÃO DIVERSIDADE: 5% fresh head + mut moderada (v13).
                            head.mutate_spline_knots(|| local_rng.random::<f32>(), 0.15);
                        } else {
                            // Blend 2 parents aleatórios + mut leve.
                            let pa = local_rng.random_range(0..pool_sz);
                            let pb = local_rng.random_range(0..pool_sz);
                            let ka = &race.elite_pool[pa].kan_knots;
                            let kb = &race.elite_pool[pb].kan_knots;
                            for o in 0..3 {
                                for k in 0..crate::kan::KAN_OUT_KNOTS {
                                    head.knots[o][k] = if local_rng.random::<f32>() < 0.5 {
                                        ka[o][k]
                                    } else {
                                        kb[o][k]
                                    };
                                }
                            }
                            head.mutate_spline_knots(|| local_rng.random::<f32>(), 0.04);
                        }
                    }
                    head
                },
                stable_seed: if i == 0 { 42 } else { rng.random::<u64>() },
            }
        ).with_children(|p| {
            p.spawn((Mesh3d(wing.clone()), MeshMaterial3d(racer_mat.clone()), Transform::from_xyz(-1.2, 0.4, 0.0)));
            p.spawn((Mesh3d(cockpit.clone()), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::WHITE, emissive: LinearRgba::WHITE * 2.0, ..default() })), Transform::from_xyz(0.5, 0.3, 0.0)));

            // FAROL DO CAMPO VISUAL (FOV de 135 graus, alcance 40 metros igual à Retina KAN)
            p.spawn((
                SpotLight {
                    color: Color::WHITE,
                    intensity: 150_000.0, // Luz alta para alcance prolongado (150m)
                    range: 150.0,
                    radius: 0.0,
                    shadows_enabled: false,
                    inner_angle: 0.8,
                    outer_angle: 1.178, // (135 graus) / 2 em radianos
                    ..default()
                },
                Transform::from_xyz(1.5, 0.5, 0.0) // No nariz do carro
                    .with_rotation(Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2).mul_quat(Quat::from_rotation_x(-0.05))) // Vira pra frente (-Z -> +X)
            ));
        });
    }

    // Tier 4.4: log da zona protegida (1x por gen, no primeiro batch de spawn).
    if idx_start == 0 {
        if let Some(mut file) = open_report_log() {
            let _ = writeln!(file, "[PROTECTED] gen {} protected zone: {} weights exempt",
                race.generation, PROTECTED_ZONE_SIZE);
        }
    }

    // Tier 4.2: log da herança epigenética média desta geração (ajuda a visualizar
    // evolução de linhagens "traumatizadas" vs "calmas" vs "exploradoras").
    if epi_inherited_count > 0 {
        if let Some(mut file) = open_report_log() {
            let n = epi_inherited_count as f32;
            let _ = writeln!(file,
                "[EPIG] gen {} inheritance n={} parent_mean cort={:.3} dopa={:.3} curi={:.3} (child baseline = 10% desses)",
                race.generation, epi_inherited_count,
                epi_cortisol_sum / n, epi_dopamine_sum / n, epi_curiosity_sum / n
            );
        }
    }
}
