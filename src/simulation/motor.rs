use bevy::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use crate::world::{VoxelType, VoxelWorldResource};
use crate::environment::EnvironmentalState;
use super::racer::{Racer, Skidmark};

// --- Tier 6.5: Time Dilation metrics (process-wide) ---
// Contam brain firings vs skips acumulados desde boot. Logados 1x/gen
// em pipeline (acesso via getters). Zero overhead (atomic relaxed).
pub(crate) static TIME_DILATION_FIRES: AtomicU64 = AtomicU64::new(0);
pub(crate) static TIME_DILATION_SKIPS: AtomicU64 = AtomicU64::new(0);

pub fn time_dilation_metrics() -> (u64, u64) {
    (
        TIME_DILATION_FIRES.load(AtomicOrdering::Relaxed),
        TIME_DILATION_SKIPS.load(AtomicOrdering::Relaxed),
    )
}

/// Tier 1.3 — Axiom como prior externo (Ng-Harada-Russell 1999 potential-based shaping).
///
/// Política procedural que espelha os 17-18 axiomas de priming usados em
/// `build_priming_weights()` (spawn.rs). Retorna bias aditivo para cada um dos
/// 64 outputs do instinct_layer — é somado a `final_intent` após o forward do
/// brain, escalado por `race.current_axiom_strength` (decay 1.0→0.0 ao longo
/// de 50 gens). Teorema NHR: bias potencial-based não altera policy ótima,
/// apenas acelera convergência.
///
/// Layout dos outputs (iguais ao priming block):
///   0 = accel/throttle, 1 = steer (+direita / -esquerda), 2 = brake.
///
/// Canais de input consultados:
///   32..37  = perif. esquerda (A1 wall-left)        42..53 = centro (A4 wall-front)
///   58..63  = perif. direita  (A2 wall-right)       46..49 = centro exato (A4d/A7 brake)
///   64      = alinhamento fluxo (A3)                73     = grass pain (A5)
///   66      = deriva lateral (A8)                   78..80 = curvatura (A9)
///   81      = lane offset (A10)                     82     = colision urgency (A11)
///   83,84   = fluxo óptico (A12)                    85,86  = parede memória L/R (A13)
///   87      = slip angle (A14)                      88     = proximity sustain (A15)
///   90..93  = dark-center tracker (A16)             94..97 = whiskers diagonais (A17)
///   98..101 = gaze tracker (A18)                  108 = racing-line offset (A21)
/// Builds a 32-dim i8 input slice for the reflex SpecialistBrain by sampling key sensory channels.
/// Conservative subset — enough for a simple safety-aware reflex.
fn slow_input_for_reflex(sensory_wave: &[i8; 128]) -> [i8; 32] {
    let mut out = [0_i8; 32];
    for (dst, &src) in out.iter_mut().zip(sensory_wave.iter().take(32)) {
        *dst = src;
    }
    out
}

fn axiom_policy(inputs: &[i8; 128]) -> [f32; 64] {
    let mut bias = [0.0f32; 64];
    const WALL_T: i8 = 40;  // ~0.31 normalizado — parede próxima mas não encostada

    // A1: parede esquerda (perif 32..37) → steer direita (+)
    let wall_l: i32 = (32..38).map(|i| inputs[i] as i32).sum();
    if wall_l > WALL_T as i32 * 2 {
        bias[1] += 0.30;
    }
    // A2: parede direita (perif 58..63) → steer esquerda (-)
    let wall_r: i32 = (58..64).map(|i| inputs[i] as i32).sum();
    if wall_r > WALL_T as i32 * 2 {
        bias[1] -= 0.30;
    }
    // A3: alinhamento com fluxo (64) → acelera
    bias[0] += (inputs[64] as f32 / 127.0) * 0.15;
    // A4: parede no centro (perif 42..54) → corta throttle
    let wall_c: i32 = (42..54).map(|i| inputs[i] as i32).sum();
    if wall_c > WALL_T as i32 * 4 {
        bias[0] -= 0.25;
    }
    // A4d/A7: centro exato (46..50) → brake forte (parede direto à frente)
    let wall_front: i32 = (46..50).map(|i| inputs[i] as i32).sum();
    if wall_front > WALL_T as i32 * 2 {
        bias[2] += 0.30;
    }
    // A5: dor de grama (73) → reduz throttle
    if inputs[73] > WALL_T {
        bias[0] -= 0.15;
    }
    // A8: deriva lateral (66) → contra-steer
    bias[1] -= (inputs[66] as f32 / 127.0) * 0.10;
    // A9: curvatura antecipada (78/79/80). FIX v3: sinal estava invertido.
    // cross(forward, flow_ahead) xz-plane: + = curva à DIREITA. Steer + = direita.
    // v4: boost coeffs 0.08/0.18/0.06 → 0.20/0.40/0.25 pra romper plateau do 1º corner
    // (330m). Lookahead 10/25/50m cria steering preditivo em vez de reativo.
    bias[1] += (inputs[78] as f32 / 127.0) * 0.20;
    bias[1] += (inputs[79] as f32 / 127.0) * 0.32;  // v5: 0.40→0.32 (menos overshoot)
    bias[1] += (inputs[80] as f32 / 127.0) * 0.25;
    // v5: pre-brake só em curva SEVERA (threshold) em vez de proporcional sempre.
    let curv_mag = ((inputs[79] as f32).abs() + (inputs[80] as f32).abs()) / 254.0;
    bias[0] -= curv_mag * 0.10; // v5: 0.25→0.10 (não mata throttle no ápice)
    // A10: lane offset (81) — valor positivo = mais espaço à direita ⇒ estou à ESQUERDA ⇒
    // preciso virar pra DIREITA = steer positivo. Coef boostado 0.08→0.22 (v2) pra virar
    // o sinal primário de centralização em pistas narrow.
    bias[1] += (inputs[81] as f32 / 127.0) * 0.22;
    // A11: urgência de colisão (82) → freia + corta motor
    bias[2] += (inputs[82] as f32 / 127.0) * 0.20;
    bias[0] -= (inputs[82] as f32 / 127.0) * 0.12;
    // A12: fluxo óptico L/R (83/84) → steer anti-aproximação
    bias[1] += (inputs[83] as f32 / 127.0) * 0.08;
    bias[1] -= (inputs[84] as f32 / 127.0) * 0.08;
    // A13: parede memória L/R (85/86)
    bias[1] += (inputs[85] as f32 / 127.0) * 0.12;
    bias[1] -= (inputs[86] as f32 / 127.0) * 0.12;
    // A14: slip angle (87) → contra-volante. v5: 0.08→0.20 (estabilizar saída de curva).
    bias[1] += (inputs[87] as f32 / 127.0) * 0.20;
    // A15: proximidade sustentada (88) → corta throttle
    if inputs[88] > WALL_T {
        bias[0] -= 0.12;
    }
    // A16: dark-center tracker (90..93) → siga o escuro (centro de pista)
    let dark_center: i32 = (90..94).map(|i| inputs[i] as i32).sum();
    bias[1] += (dark_center as f32 / (127.0 * 4.0)) * 0.28;
    // A17: whiskers diagonais (94..97) — parede perto em diagonal
    //   94 = FL 30°, 95 = FR 30°, 96 = FL 60°, 97 = FR 60°
    if inputs[94] > WALL_T {
        bias[1] += 0.28;                                // FL → steer direita forte
        bias[2] += 0.18; bias[0] -= 0.16;
    }
    if inputs[96] > WALL_T {
        bias[1] += 0.20;                                // FL larga
        bias[2] += 0.14;
    }
    if inputs[95] > WALL_T {
        bias[1] -= 0.28;                                // FR → steer esquerda forte
        bias[2] += 0.18; bias[0] -= 0.16;
    }
    if inputs[97] > WALL_T {
        bias[1] -= 0.20;
        bias[2] += 0.14;
    }
    // A18: gaze tracker (98..101) → fóvea aponta pra onde há pista aberta
    let gaze: i32 = (98..102).map(|i| inputs[i] as i32).sum();
    bias[1] += (gaze as f32 / (127.0 * 4.0)) * 0.25;

    // A19 REVERTIDO v6b: testado em 200 gens — champion peak 621m vs v5 736m.
    // Anti-stall forçava throttle em estados pós-curva que não deveriam acelerar.
    // Decay window 20→40 e rate 0.95→0.98 MANTIDOS (pipeline.rs).

    // A20: drift commit — slip alto + curvatura alta à frente + velocidade mínima →
    // compromete com throttle em vez de amarelar. Canais: 87 slip, |79|+|80|/254 curv, 65 speed.
    let slip_abs = (inputs[87] as f32 / 127.0).abs();
    let curv_mag_a20 = ((inputs[79] as f32).abs() + (inputs[80] as f32).abs()) / 254.0;
    let speed_norm = (inputs[65] as f32 / 127.0).abs();
    if slip_abs > 0.5 && curv_mag_a20 > 0.4 && speed_norm > 0.35 {
        bias[0] += 0.18;
        bias[2] -= 0.15;
    }

    // A21: racing line follow. Canal 108 = lateral offset signed (positivo =
    // carro à ESQUERDA da linha ideal → precisa steer pra DIREITA pra fechar).
    // Convenção casa com A10 (canal 81): positivo → bias[1] positivo (steer +right).
    // Axiom decay (race.current_axiom_strength) é aplicado no call site, então
    // aqui só semeamos o prior. v13: 0.35→0.20 (HOLE=5181 em v12, overcorrection
    // pro apex puxava carro pra fora em curva fechada).
    const CH_LINE: usize = 108;
    let line_err = inputs[CH_LINE] as f32 / 127.0;
    bias[1] += line_err * 0.20;

    bias
}

pub(super) fn update_racers(
    mut commands: Commands,
    mut query: Query<(
        Entity,
        &mut Racer,
        &mut Transform,
        &mut MeshMaterial3d<StandardMaterial>,
        Option<&mut crate::physics::EngineComponent>,
        Option<&mut crate::physics::PhysicsBody>
    )>,
    world_res: Res<VoxelWorldResource>,
    env: Res<EnvironmentalState>,
    dev: Res<crate::evolution::DevSettings>,
    mut race: ResMut<crate::evolution::RaceSettings>,
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut skidmark_cache: Local<Option<(Handle<Mesh>, Handle<StandardMaterial>)>>,
    mut retina_tick_counter: Local<u32>,
    // Cache of previous lateral-offset-from-racing-line per entity so the
    // derivative channel (109) can be computed without extending Racer
    // (spawn.rs is frozen by task constraints).
    mut prev_line_offset: Local<HashMap<Entity, f32>>,
    mut gizmos: Gizmos,
) {
    // Rodando em FixedUpdate: dt = Time<Fixed>::delta = 1/64s constante. Bevy já
    // substepa o schedule N vezes por frame real conforme time_scale (Virtual).
    // Sem cap, sem vdt separado — fixed step garante determinismo e sync com física.
    let dt = time.delta_secs();
    let vdt = dt;
    let world = &world_res.0;

    // --- TOGGLE GIZMOS (KeyG) ---
    // Wrap todo o bloco de gizmos visuais (flag, paths, pilares) em dev.show_gizmos.
    // Permite limpar UI pra "modo jogo" quando jogador quer só dirigir.
    if dev.show_gizmos {
    // --- RENDERIZAR MARCO-ZERO (A BANDEIRA DO CAMPEÃO GLOBAL) ---
    if let Some(flag_pos) = race.global_best_position {
        // Haste de 15 metros cravada no chão
        gizmos.line(flag_pos, flag_pos + Vec3::Y * 15.0, Color::srgb(1.0, 0.84, 0.0));
        // Globo/Bandeira dourada no topo
        gizmos.sphere(flag_pos + Vec3::Y * 15.0, 2.0, Color::srgb(1.0, 0.84, 0.0));
        // Anel no chão para visibilidade de longe
        gizmos.circle(
            Isometry3d::new(flag_pos, Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
            5.0, Color::srgba(1.0, 0.84, 0.0, 0.6),
        );
    }

    // --- TRAÇADO DO ÚLTIMO CAMPEÃO (magenta, persiste até ser superado) ---
    if race.champion_path.len() >= 2 {
        let path = &race.champion_path;
        let y_off = Vec3::Y * 1.5;
        for w in path.windows(2) {
            gizmos.line(w[0] + y_off, w[1] + y_off, Color::srgba(1.0, 0.2, 0.9, 0.9));
        }
        if let Some(&start) = path.first() {
            gizmos.sphere(start + Vec3::Y * 2.0, 1.5, Color::srgb(1.0, 0.84, 0.0)); // 👑 dourado
            gizmos.line(start, start + Vec3::Y * 12.0, Color::srgb(1.0, 0.84, 0.0));
        }
        if let Some(&end) = path.last() {
            gizmos.sphere(end + Vec3::Y * 2.0, 1.5, Color::srgb(1.0, 0.1, 0.1)); // 🔴 fim
            gizmos.line(end, end + Vec3::Y * 12.0, Color::srgb(1.0, 0.1, 0.1));
        }
    }

    // --- TRAÇADO DO MELHOR DA ÚLTIMA GERAÇÃO ---
    // 🔵 CIANO = grid-spawn (elegível pra coroação — se for maior que magenta, vira campeão)
    // 🟠 LARANJA = distributed spawn (NÃO coroa — é só especialista de um trecho)
    // Assim se a laranja passa a magenta, você sabe que é fake progress.
    if race.last_gen_path.len() >= 2 {
        let path = &race.last_gen_path;
        let y_off = Vec3::Y * 0.6;
        let (line_col, sphere_col) = if race.last_gen_was_grid {
            (Color::srgba(0.0, 0.9, 1.0, 0.75), Color::srgb(0.0, 0.9, 1.0))
        } else {
            (Color::srgba(1.0, 0.5, 0.0, 0.75), Color::srgb(1.0, 0.5, 0.0))
        };
        for w in path.windows(2) {
            gizmos.line(w[0] + y_off, w[1] + y_off, line_col);
        }
        if let Some(&start) = path.first() {
            gizmos.sphere(start + Vec3::Y * 0.8, 1.0, sphere_col);
        }
        if let Some(&end) = path.last() {
            gizmos.sphere(end + Vec3::Y * 0.8, 1.0, sphere_col);
        }
    }

    // --- RENDERIZAR WAYPOINTS/CHECKPOINTS DA PISTA ---
    // Decima: pista procedural tem 100+ waypoints (Catmull-Rom). Mostrar todos
    // = 100+ pilares atravessando paredes (gizmos drawn on top). Mostrar só 1 em
    // cada 8 (densidade dos anchors originais) e altura reduzida pra 3m.
    let wps = &world.waypoints_world;
    let stride = (wps.len() / 18).max(1); // ~14-18 pilares
    for i in (0..wps.len()).step_by(stride) {
        let wp = Vec3::new(wps[i].x, 0.5, wps[i].y);

        // Pilar de checkpoint (3m de altura, fino).
        let pillar_color = if i == 0 {
            Color::srgb(0.0, 1.0, 0.3) // Verde = Largada/Chegada
        } else {
            Color::srgba(0.0, 0.7, 1.0, 0.5) // Ciano sutil
        };
        gizmos.line(wp, wp + Vec3::Y * 3.0, pillar_color);
        gizmos.sphere(wp + Vec3::Y * 3.5, 0.8, pillar_color);
    }
    } // end dev.show_gizmos

    // --- PRÉ-CÁLCULO (Primeiro passo read-only sobre a mesma query) ---
    let mut best_score = -10000.0;
    let mut current_best = race.selected_racer.unwrap_or(Entity::PLACEHOLDER);
    let mut racer_positions: Vec<(Entity, Vec3, Vec3)> = Vec::new();

    for (e, r, t, _, _, _) in query.iter() {
        // Atualiza a Bandeira do Campeão Global se este piloto superou a marca de todos os tempos
        // Somado com 0.05 para evitar micro-atualizações visuais
        if r.track_progress > race.global_best_distance + 0.05 {
            race.global_best_distance = r.track_progress;
            race.global_best_position = Some(t.translation);
        }

        let score = if r.track_progress < 15.0 {
            r.track_progress * 10.0 - 10000.0
        } else {
            let avg_speed = r.track_progress / r.engine.current_time.max(0.1);
            let lap_bonus = r.total_laps as f32 * 10000.0;
            lap_bonus + (r.track_progress * 100.0) + (avg_speed * 50.0) + (r.hp * 2.0)
        };
        if score > best_score {
            best_score = score;
            current_best = e;
        }
        if !r.is_dead {
            racer_positions.push((e, t.translation, t.right().as_vec3()));
        }
    }

    // --- PARALELO: PRECOMPUTA RETINAS DE TODOS CARROS VIVOS ---
    // Retina = 64 raycasts DDA × 2000 carros = 128k rays/tick. Em time_scale=10 são
    // ~10 substeps/frame real → 1.28M rays/frame. Pesado mesmo paralelizado via rayon.
    //
    // Otimização: cache + skip. Recomputa retina só a cada RETINA_SKIP ticks (3 =
    // ~47ms virtual de latência). Nos substeps restantes reusa racer.visual_cortex
    // (guardado do último recompute). Carro a 60 m/s se desloca ~3m entre recomputes
    // — tolerável pra percepção (walls têm largura 3m).
    const RETINA_SKIP: u32 = 2;
    *retina_tick_counter = retina_tick_counter.wrapping_add(1);
    let recompute_retinas = (*retina_tick_counter % RETINA_SKIP) == 0;

    let retinas: HashMap<Entity, [i8; 64]> = if recompute_retinas {
        let retina_inputs: Vec<(Entity, Vec3, Vec3, Vec3)> = query
            .iter()
            .filter(|(_, r, _, _, _, _)| !r.is_dead)
            .map(|(e, _, t, _, _, _)| (e, t.translation, t.right().as_vec3(), t.back().as_vec3()))
            .collect();
        retina_inputs
            .par_iter()
            .map(|&(e, pos, fwd, right)| {
                (e, world.render_binocular_retina(pos + Vec3::Y * 0.4, fwd, right, 0.0))
            })
            .collect()
    } else {
        // Reuso: pega visual_cortex (retina do último recompute) de cada carro.
        query
            .iter()
            .filter(|(_, r, _, _, _, _)| !r.is_dead)
            .map(|(e, r, _, _, _, _)| (e, r.visual_cortex))
            .collect()
    };

    // --- PRECOMPUTA RANK + SPATIAL HASH (elimina O(N²) do loop principal) ---
    // Antes: cada carro sortava todos (O(N log N) × N) + scan comp_stress/draft O(N) × N
    //      = ~4M ops/tick pra 2000 carros. Agora: 1 sort O(N log N) + lookup O(1).
    //   Spatial hash 20m cells: query vizinhos retorna só 9 células (média ~8 carros cada).
    let mut rank_sorted: Vec<(Entity, f32)> = racer_positions.iter().map(|&(e, p, _)| (e, p.x)).collect();
    rank_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let rank_by_entity: HashMap<Entity, usize> = rank_sorted.iter().enumerate().map(|(i, &(e, _))| (e, i)).collect();

    const SPATIAL_CELL: f32 = 20.0;
    let mut spatial: HashMap<(i32, i32), Vec<(Entity, Vec3, Vec3)>> = HashMap::new();
    for &entry in &racer_positions {
        let cx = (entry.1.x / SPATIAL_CELL).floor() as i32;
        let cz = (entry.1.z / SPATIAL_CELL).floor() as i32;
        spatial.entry((cx, cz)).or_default().push(entry);
    }

    for (entity, mut racer, mut transform, mat_handle, mut opt_engine, mut opt_body) in query.iter_mut() {
        if racer.is_dead {
            if let Some(engine_comp) = opt_engine.as_mut() { engine_comp.throttle = 0.0; }
            if let Some(body) = opt_body.as_mut() {
                body.velocity *= 0.95_f32.powf(dt * 60.0); // Atrito forte: para em ~2s
                body.angular_velocity *= 0.9_f32.powf(dt * 60.0);
                if body.velocity.length() < 0.5 { body.velocity = Vec3::ZERO; }
            }
            continue;
        }

        // Grace tick-based: imune a time_scale. ~600 frames = 10s reais @ 60Hz.
        racer.grace_ticks = racer.grace_ticks.saturating_add(1);
        // race_time em SEGUNDOS VIRTUAIS: a 8x time_scale, 1s real = 8s virtuais aqui.
        // É o relógio que o USUÁRIO vê na tela. Punições de grace usam isto.
        racer.race_time += vdt;

        let pos = transform.translation;
        let flow = world.get_flow_at(pos.x, pos.z);
        let track_phase = flow.y.atan2(flow.x);

        // --- 1. VISÃO DINÂMICA (COM CICLO DIA/NOITE) ---
        let mut sensory_wave = [0i8; 128]; // Stack alloc (sem heap)

        // --- VISÃO BIOLÓGICA ESTEREOSCÓPICA (Fóvea Dupla) ---
        let forward = transform.right().as_vec3();
        let right = transform.back().as_vec3();

        // Retina FIXA forward (sacádica desativada). Dado precomputado PARALELAMENTE acima
        // via rayon par_iter — este loop serial apenas consome o resultado por entidade.
        let _ = (forward, right);
        let retina_binocular = *retinas.get(&entity).unwrap_or(&[-128i8; 64]);
        // RETINA POOLED — 8 bins fóvea (NÃO-UNIFORMES, center-dense) + 8 bins periférica (uniforme).
        //
        // Fóvea cobre 50° (índices 0..32, -25° a +25°). Bins center-dense: pixels centrais
        // (±5°) ocupam 1-2 bins dedicados, bordas ficam em bins largos. Dá agudeza pra
        // curvas apertadas (precisão ±5° critical) sem estourar canais.
        //
        //   Bin 0: px 0..3    (-25° a -19°) — borda esquerda larga
        //   Bin 1: px 3..8    (-19° a -12°)
        //   Bin 2: px 8..13   (-12° a  -4°) — meio-esquerda
        //   Bin 3: px 13..16  (-4°  a   0°) — CENTRO (3 px, ~5° precisão)
        //   Bin 4: px 16..19  ( 0°  a  +4°) — CENTRO
        //   Bin 5: px 19..24  (+4°  a +12°)
        //   Bin 6: px 24..29  (+12° a +19°)
        //   Bin 7: px 29..32  (+19° a +25°) — borda direita larga
        const FOV_BINS: [(usize, usize); 8] = [
            (0, 3), (3, 8), (8, 13), (13, 16),
            (16, 19), (19, 24), (24, 29), (29, 32),
        ];
        for (bin, &(lo, hi)) in FOV_BINS.iter().enumerate() {
            let mut fov_max: i8 = -128;
            for p in lo..hi {
                let f = retina_binocular[p];
                if f > fov_max { fov_max = f; }
            }
            sensory_wave[bin] = fov_max;
        }
        // Periférica uniforme — papel de ALERTA, não precisão. 32 → 8 bins max-pool.
        for bin in 0..8 {
            let start = bin * 4;
            let mut per_max: i8 = -128;
            for k in 0..4 {
                let p = retina_binocular[32 + start + k];
                if p > per_max { per_max = p; }
            }
            sensory_wave[8 + bin] = per_max;
        }
        // Canais 16..63 zerados explicitamente (garantia contra leak de frame anterior).
        for i in 16..64 { sensory_wave[i] = 0; }

        // --- SACCADE COM MEMÓRIA: média ponderada de pixels escuros + hold se cego ---
        // Usa TODOS os pixels escuros (valor negativo = road) no setor frontal (±60°),
        // pondera por "quão escuro" e calcula centroide angular. Se NADA está escuro
        // (tudo parede/grama = visão bloqueada), MANTÉM o gaze anterior — é a memória
        // de "a pista estava pra direita há pouco, continua indo pra lá pra encontrar".
        // 180°/32 pixels ≈ 5.625°/pixel. ±60° ⇒ índices 6..26.
        let per_fov = std::f32::consts::PI;
        let mut dark_angle_sum: f32 = 0.0;
        let mut dark_weight_sum: f32 = 0.0;
        for i in 6..26 {
            let v = retina_binocular[32 + i] as i32;
            if v < 0 {
                let strength = (-v) as f32; // -128 (perto) → 128 de peso; -80 (longe) → 80
                let angle = ((i as f32 / 31.0) * 2.0 - 1.0) * (per_fov / 2.0);
                dark_angle_sum += angle * strength;
                dark_weight_sum += strength;
            }
        }
        // Saccade/gaze_yaw DESATIVADO (user feedback: "problema é o foco").
        // Mantemos dark_center abaixo como priming de steer (canais 90-93) —
        // ele já carrega o sinal "pista vai pra lá, vira pra lá" sem girar o olho.
        racer.gaze_yaw = 0.0;
        sensory_wave[98] = 0;
        sensory_wave[99] = 0;
        sensory_wave[100] = 0;
        sensory_wave[101] = 0;
        let _ = (dark_angle_sum, dark_weight_sum, per_fov);

        // --- DARK-CENTER TRACKER (Canais 90-93): "Center the Black" ---
        // Centro de massa angular dos pixels ESCUROS (pista) na retina.
        // Positivo = pista vai pra direita → steer RIGHT.
        // Negativo = pista vai pra esquerda → steer LEFT.
        //
        // Duplicado em 4 canais (90..93) para que, após quantize ternária (todos
        // os pesos não-zero viram ±1), o sinal tenha "voto" equivalente a 4× um
        // axioma normal. É o jeito de amplificar um único sinal numa rede 1.58-bit.
        //
        // Essa foi a crítica do usuário: "os carros não tentam centralizar o preto".
        // O sinal anterior (ch 81 lane_offset) existia mas só votava ±1 contra 40+
        // canais concorrentes. Aqui damos força proporcional à sua importância.
        // BUG FIX (user report): centroide global entre 2 blobs escuros (curva em
        // S, pista atrás) caía em PAREDE. Modelo virava pro meio da parede achando
        // que era pista. Solução: sliding window — encontra janela com MAIOR soma
        // de darkness (=blob mais largo/profundo de pista visível) e usa SEU centro.
        const W: usize = 5; // janela 5 pixels (~28° angular)
        let mut best_window_start = 0usize;
        let mut best_window_sum: f32 = -1.0;
        let mut best_window_centroid: f32 = 0.0;
        for start in 0..=(32 - W) {
            let mut sum: f32 = 0.0;
            let mut weighted_idx: f32 = 0.0;
            for k in 0..W {
                let v = retina_binocular[32 + start + k] as i32;
                let d = (-v as f32).max(0.0);
                sum += d;
                weighted_idx += d * (start + k) as f32;
            }
            if sum > best_window_sum {
                best_window_sum = sum;
                best_window_start = start;
                best_window_centroid = if sum > 1.0 { weighted_idx / sum } else { (start + W/2) as f32 };
            }
        }
        let _ = best_window_start;
        // Centroide da janela vencedora normalizado em [-1, 1] (idx 15.5 = centro).
        let dark_center_norm: f32 = if best_window_sum > 1.0 {
            ((best_window_centroid - 15.5) / 15.5).clamp(-1.0, 1.0)
        } else { 0.0 };
        let dark_center_i8 = (dark_center_norm * 127.0).clamp(-127.0, 127.0) as i8;
        sensory_wave[90] = dark_center_i8;
        sensory_wave[91] = dark_center_i8;
        sensory_wave[92] = dark_center_i8;

        // BUG FIX 2 (user report): "modelo escolhe menor lado". Adiciona canais EXPLÍCITOS:
        // Soma de "darkness" (road = i8 negativo) por metade. Não exige contiguidade
        // do centro — antes parede colada ao center idx zerava sinal mesmo com road longe.
        let mut left_dark_sum: i32 = 0;
        let mut right_dark_sum: i32 = 0;
        for i in 0..16 {
            let v = retina_binocular[32 + i] as i32;
            if v < 0 { left_dark_sum += -v; }
        }
        for i in 16..32 {
            let v = retina_binocular[32 + i] as i32;
            if v < 0 { right_dark_sum += -v; }
        }
        // Max possível: 16 pixels × 128 = 2048. Normalize.
        let open_l_norm = (left_dark_sum as f32 / 2048.0).clamp(0.0, 1.0);
        let open_r_norm = (right_dark_sum as f32 / 2048.0).clamp(0.0, 1.0);
        let open_l_i8 = (open_l_norm * 127.0) as i8;
        let open_r_i8 = (open_r_norm * 127.0) as i8;
        // Vitais: 4× pra bootstrap rápido. Sinais novos precisam força inicial.
        for i in 36..40 { sensory_wave[i] = open_l_i8; }
        for i in 40..44 { sensory_wave[i] = open_r_i8; }

        // BEST ESCAPE direction (NOVO — bug user "vai sempre pra parede").
        // Encontra LONGEST contiguous run de pixels escuros (road) na periférica.
        // Output = ângulo signed [-1, 1] do centro do run em relação ao center idx.
        // Sinal forte e UNÍVOCO de "vai pra cá pra escapar parede".
        let mut best_run_start = 0usize;
        let mut best_run_len = 0usize;
        let mut cur_run_start = 0usize;
        let mut cur_run_len = 0usize;
        for i in 0..32 {
            let v = retina_binocular[32 + i] as i32;
            if v < -10 { // escuro = road
                if cur_run_len == 0 { cur_run_start = i; }
                cur_run_len += 1;
                if cur_run_len > best_run_len {
                    best_run_start = cur_run_start;
                    best_run_len = cur_run_len;
                }
            } else {
                cur_run_len = 0;
            }
        }
        let escape_norm: f32 = if best_run_len >= 2 {
            let center_of_run = best_run_start as f32 + (best_run_len as f32 - 1.0) * 0.5;
            ((center_of_run - 15.5) / 15.5).clamp(-1.0, 1.0)
        } else { 0.0 };
        let escape_i8 = (escape_norm * 127.0).clamp(-127.0, 127.0) as i8;
        // 4× duplicação — sinal vital, deve dominar steer em bistable choice.
        for i in 60..64 { sensory_wave[i] = escape_i8; }

        // BUG FIX 3 (user report): "pista pequena bate direto". Wall imminent
        // binary signal forte — pixels centrais (10..22) com retina alta = parede frontal.
        let mut wall_central_max: i32 = -128;
        for i in 10..22 {
            let v = retina_binocular[32 + i] as i32;
            if v > wall_central_max { wall_central_max = v; }
        }
        // > 80 = parede a < 30m; > 110 = colisão iminente.
        let wall_imminent_i8: i8 = if wall_central_max > 110 { 127 }
                                   else if wall_central_max > 80 { 90 }
                                   else if wall_central_max > 40 { 50 }
                                   else { 0 };
        for i in 44..48 { sensory_wave[i] = wall_imminent_i8; }
        sensory_wave[93] = dark_center_i8;

        // --- FLUXO ÓPTICO TEMPORAL (delta entre frames) ---
        // Mede variação entre retina atual e anterior, por metade. Sinal de "coisa
        // aproximando". Canais 83 (esquerda) e 84 (direita).
        let mut motion_left: i32 = 0;
        let mut motion_right: i32 = 0;
        for i in 0..16 {
            motion_left += (retina_binocular[i] as i32 - racer.visual_cortex[i] as i32).abs();
            motion_left += (retina_binocular[32 + i] as i32 - racer.visual_cortex[32 + i] as i32).abs();
        }
        for i in 16..32 {
            motion_right += (retina_binocular[i] as i32 - racer.visual_cortex[i] as i32).abs();
            motion_right += (retina_binocular[32 + i] as i32 - racer.visual_cortex[32 + i] as i32).abs();
        }
        // 32 pixels × max delta 255 ≈ 8160. Normaliza para [0,127].
        sensory_wave[83] = ((motion_left / 64).min(127)) as i8;
        sensory_wave[84] = ((motion_right / 64).min(127)) as i8;

        racer.visual_cortex = retina_binocular;

        // --- CAOS VISUAL (Medo de Obstáculos) ---
        // O modelo deve procurar o "Preto" (Pista). Ver muito "Branco" (Parede) causa Pânico.
        // Pós pooled retina: itera 16 bins (8 fóvea + 8 periférica).
        let mut visual_stress = 0.0;
        for i in 0..16 {
            if sensory_wave[i] > 0 { visual_stress += (sensory_wave[i] as f32) / 127.0; }
        }
        visual_stress /= 16.0; // Média de brancura (0.0 a 1.0)

        // Punição psicológica direta por olhar para paredes
        racer.joy_integral -= visual_stress * 5.0 * dt;

        let panic_level = if visual_stress > 0.2 { (visual_stress - 0.2) * 1.5 } else { 0.0 };

        // --- RUÍDO NOTURNO E PÂNICO: Incerteza sensorial ---
        use rand::prelude::*;
        let mut rng = rand::rng();

        if env.is_night() || panic_level > 0.0 {
            let magnitude = (15.0 + panic_level * 50.0) as i8;
            for i in 0..16 {
                let noise = rng.random_range(-magnitude..magnitude);
                sensory_wave[i] = sensory_wave[i].saturating_add(noise);
            }
        }

        // --- 2. CANAIS DE INSTINTO E TELEMETRIA (Sensoriometria) ---
        let alignment = (racer.phase - track_phase).cos();
        sensory_wave[64] = (alignment * 127.0) as i8;

        let speed = if let Some(ref body) = opt_body { body.velocity.length() } else { 0.0 };
        let avg_tire_temp = if let Some(ref body) = opt_body {
            body.wheels.iter().map(|w| w.tire_temp).sum::<f32>() / 4.0
        } else { 0.3 };
        sensory_wave[65] = (speed * 2.0).clamp(0.0, 127.0) as i8;

        let mut is_drifting = false;
        if let Some(ref body) = opt_body {
            let lateral_dir = transform.back().as_vec3();
            let lateral_vel = body.velocity.dot(lateral_dir);
            sensory_wave[66] = (lateral_vel * 10.0).clamp(-127.0, 127.0) as i8;
            if lateral_vel.abs() > 4.0 && speed > 5.0 && pos.y < 1.0 { is_drifting = true; }
        }

        // Percepção de Dor Mecânica (HP), Tensão no Motor (RPM) e Marcha Atual
        sensory_wave[67] = ((racer.hp / 100.0) * 127.0).clamp(-128.0, 127.0) as i8;
        // --- RANK + COMP_STRESS via precomputados ---
        let my_rank = rank_by_entity.get(&entity).copied().unwrap_or(0);
        let is_leader = my_rank == 0;

        // Canal 125: Estresse Competitivo (0 líder, 127 atrás de alguém próximo, 64 meio pelotão)
        // Spatial hash query: só 9 células ao redor (~8 carros cada) em vez de N inteiro.
        let mut comp_stress = 0;
        if !is_leader {
            comp_stress = 64;
            let cx = (pos.x / SPATIAL_CELL).floor() as i32;
            let cz = (pos.z / SPATIAL_CELL).floor() as i32;
            'outer: for dx in -1..=1 {
                for dz in -1..=1 {
                    if let Some(cell) = spatial.get(&(cx + dx, cz + dz)) {
                        for &(other_e, other_pos, _) in cell {
                            if other_e == entity { continue; }
                            let to_other = other_pos - pos;
                            if to_other.length() < 20.0 && to_other.normalize_or_zero().dot(forward) > 0.3 {
                                comp_stress = 127;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        sensory_wave[125] = comp_stress as i8;

        // Canal 124: Dor de Centrífuga (Anti-Giro)
        // Se estiver girando muito (>1.5 rad/s) e com pouca velocidade (<2.0 m/s), DOR MÁXIMA.
        let is_spinning = if let Some(body) = &opt_body { body.angular_velocity.y.abs() > 1.5 && speed < 2.0 } else { false };
        sensory_wave[124] = if is_spinning { -128 } else { 0 };

        // --- AGITAÇÃO SENSORIO-REFERENCIAL (RUÍDO DE ESTRESSE) ---
        // Injeta ruído exploratório nos sensores quando está estressado para sair de estados presos
        if comp_stress > 0 {
            let noise_amplitude = (comp_stress as f32 / 127.0) * 5.0;
            for i in 0..16 {
                let noise = (rand::random::<f32>() - 0.5) * noise_amplitude;
                sensory_wave[i] = (sensory_wave[i] as f32 + noise).clamp(-128.0, 127.0) as i8;
            }
        }

        // --- CANAIS ORIGINAIS ---
        if let Some(engine) = &opt_engine {
            let rpm_ratio = engine.current_rpm / engine.max_rpm;
            sensory_wave[68] = (rpm_ratio * 127.0).clamp(0.0, 127.0) as i8;
            sensory_wave[69] = (engine.current_gear as i8 * 20).clamp(0, 127);
        }

        // --- NOVOS SENSORES DE TELEMETRIA (PROPRIOCEPÇÃO) ---
        // Canal 70: Carga Térmica dos Pneus (0-127 = 0%-100%)
        sensory_wave[70] = (avg_tire_temp * 127.0).clamp(0.0, 127.0) as i8;

        // Canal 71: Downforce Sensor (D = 0.5 * ρ * v² * Cl * A)
        let downforce_normalized = (0.5 * 1.225 * speed * speed * 0.18 * 3.0 * 2.0) / 15000.0; // Normalizado ~15kN max
        sensory_wave[71] = (downforce_normalized * 127.0).clamp(0.0, 127.0) as i8;

        // Canal 72: Drafting/Slipstream (127 se piloto à frente dentro de 10m)
        // Spatial hash: só 9 células em vez de N completo.
        let mut in_draft = false;
        let cx = (pos.x / SPATIAL_CELL).floor() as i32;
        let cz = (pos.z / SPATIAL_CELL).floor() as i32;
        'draft: for dx in -1..=1 {
            for dz in -1..=1 {
                if let Some(cell) = spatial.get(&(cx + dx, cz + dz)) {
                    for &(other_e, other_pos, _) in cell {
                        if other_e == entity { continue; }
                        let to_other = other_pos - pos;
                        let dist = to_other.length();
                        let dot_forward = to_other.normalize_or_zero().dot(forward);
                        if dist < 10.0 && dot_forward > 0.5 {
                            in_draft = true;
                            break 'draft;
                        }
                    }
                }
            }
        }
        sensory_wave[72] = if in_draft { 127 } else { 0 };

        // Canal 73: Dor de Grama (Neural Pain constante quando na grama)
        let voxel_current = world.get_voxel_at(pos.x, pos.z);
        sensory_wave[73] = if voxel_current == VoxelType::Grass { -128 } else { 0 };

        // Canal 74: Peso do Volante (Steering Weight - Caster Dinâmico)
        let steer_limit = 1.0 / (1.0 + speed * 0.02);
        sensory_wave[74] = (steer_limit * 127.0).clamp(0.0, 127.0) as i8;

        // Canal 75: Clima / Wetness
        sensory_wave[75] = (env.weather_wetness * 127.0).clamp(0.0, 127.0) as i8;

        // --- NOVOS SENSORES DE PROXIMIDADE LATERAL (TRACK OFFSET 2.0) ---
        // Permite "sentir" a margem da pista mesmo sem olhar.
        let left_angle = racer.phase + std::f32::consts::FRAC_PI_2;
        let right_angle = racer.phase - std::f32::consts::FRAC_PI_2;
        let dist_l = world.raycast(pos.x, pos.z, left_angle, 15.0);
        let dist_r = world.raycast(pos.x, pos.z, right_angle, 15.0);
        sensory_wave[76] = ((dist_l / 15.0) * 127.0) as i8;
        sensory_wave[77] = ((dist_r / 15.0) * 127.0) as i8;

        // --- WHISKERS DIAGONAIS (canais 94-97) — visão de canto de L ---
        // Em curva de 90° fechada a retina vê "aberto" na direção diagonal (corte da parede
        // externa), mas os raycasts a 90° puro não pegam a parede interna aproximando.
        // Whiskers a ±30° e ±60° do heading cobrem esse ponto cego. Sinal invertido:
        // 127 = parede encostando, 0 = 10m+ livre. Range curto (10m) pra sinal forte
        // sobreviver quantize ternária e disparar reflexo imediato.
        const WHISKER_RANGE: f32 = 10.0;
        let ang30 = std::f32::consts::FRAC_PI_6;   // 30°
        let ang60 = std::f32::consts::FRAC_PI_3;   // 60°
        let d_fl30 = world.raycast(pos.x, pos.z, racer.phase + ang30, WHISKER_RANGE);
        let d_fr30 = world.raycast(pos.x, pos.z, racer.phase - ang30, WHISKER_RANGE);
        let d_fl60 = world.raycast(pos.x, pos.z, racer.phase + ang60, WHISKER_RANGE);
        let d_fr60 = world.raycast(pos.x, pos.z, racer.phase - ang60, WHISKER_RANGE);
        let invert = |d: f32| -> i8 { ((1.0 - (d / WHISKER_RANGE).clamp(0.0, 1.0)) * 127.0) as i8  };
        sensory_wave[94] = invert(d_fl30);
        sensory_wave[95] = invert(d_fr30);
        sensory_wave[96] = invert(d_fl60);
        sensory_wave[97] = invert(d_fr60);

        // --- CURVATURA ANTECIPADA (LOOKAHEAD DE PISTA) ---
        // Amostra o flow field à frente a 3 distâncias e mede o desvio angular
        // em relação ao heading atual. Positivo = pista curva à ESQUERDA;
        // negativo = curva à DIREITA. É o "GPS interno" dos próximos ~2s.
        let forward_2d = Vec2::new(forward.x, forward.z);
        for (slot, look_dist) in [(78usize, 10.0_f32), (79, 25.0), (80, 50.0)] {
            let sample = pos + forward * look_dist;
            let flow_ahead = world.get_flow_at(sample.x, sample.z);
            // cross-z (2D perp dot): forward × flow_ahead
            let cross = forward_2d.x * flow_ahead.y - forward_2d.y * flow_ahead.x;
            sensory_wave[slot] = (cross * 127.0).clamp(-127.0, 127.0) as i8;
        }

        // Canal 81: LANE OFFSET — assimetria lateral pra posição na pista [-1..+1].
        // Positivo = mais espaço à direita → estou à ESQUERDA da pista.
        // Negativo = mais espaço à esquerda → estou à DIREITA da pista.
        let lane_denom = (dist_l + dist_r).max(0.1);
        let lane_offset = (dist_r - dist_l) / lane_denom;
        sensory_wave[81] = (lane_offset * 127.0).clamp(-127.0, 127.0) as i8;

        // --- RACING LINE SENSORS (canais 108/109/110) ---
        // Teacher signal para a rede aprender a linha ideal (wide-apex-wide).
        // 108 = lateral offset (m, signed, clamped to ±half_width, then scaled).
        //       Positivo = carro à ESQUERDA da linha → precisa virar pra DIREITA.
        // 109 = derivada do offset (offset anterior - atual). Positivo = fechando
        //       a distância pra linha (bom). Codifica "tô corrigindo na direção certa".
        // 110 = lookahead offset ~5m à frente na linha ideal. Anticipa turn-in
        //       quando o offset instantâneo é 0 mas a linha já começou a curvar.
        let pos2d = Vec2::new(pos.x, pos.z);
        let fwd2d = Vec2::new(forward.x, forward.z);
        let line_offset_m = if !world.racing_line.is_empty() {
            world.racing_line.lateral_offset_from_line(pos2d, fwd2d)
        } else { 0.0 };
        let hw = world.racing_line.half_width.max(1.0);
        let line_off_norm = (line_offset_m / hw).clamp(-1.0, 1.0);
        sensory_wave[108] = (line_off_norm * 127.0).clamp(-127.0, 127.0) as i8;

        let prev_off = prev_line_offset.get(&entity).copied().unwrap_or(line_offset_m);
        // "offset anterior - atual": positivo quando o erro diminuiu neste tick.
        let d_off = (prev_off - line_offset_m) / hw;
        sensory_wave[109] = (d_off * 127.0 * 8.0).clamp(-127.0, 127.0) as i8; // gain 8x pq delta por tick é pequeno
        prev_line_offset.insert(entity, line_offset_m);

        let lookahead_m = if !world.racing_line.is_empty() {
            world.racing_line.lookahead_offset(pos2d, 5.0)
        } else { 0.0 };
        sensory_wave[110] = ((lookahead_m / hw).clamp(-1.0, 1.0) * 127.0) as i8;

        // Canal 82: URGÊNCIA DE COLISÃO FRONTAL — speed / forward_wall_dist.
        // Alto = vou bater logo. Combina TTC inverso num único sinal [0..127].
        let fwd_probe = world.raycast(pos.x, pos.z, racer.phase, 30.0).max(0.5);
        let ttc_inv = (speed / fwd_probe).clamp(0.0, 10.0); // 10 = ~0.1s de TTC
        sensory_wave[82] = ((ttc_inv / 10.0) * 127.0) as i8;

        // --- MEMÓRIA SENSORIAL PERSISTENTE ---
        // EMA (alpha ~0.15/frame @ 60Hz ≈ ~0.5s de meia-vida) suaviza picos de ruído
        // e mantém sensação de "tem parede perto" mesmo se o raycast momentâneo falhar.
        let ema_alpha = (dt * 3.0).clamp(0.0, 1.0);
        racer.wall_ema_l = racer.wall_ema_l * (1.0 - ema_alpha) + dist_l * ema_alpha;
        racer.wall_ema_r = racer.wall_ema_r * (1.0 - ema_alpha) + dist_r * ema_alpha;
        // Invertido: 127 = encostado, 0 = 15m+ livre. Canal "alerta" persistente.
        sensory_wave[85] = (((1.0 - (racer.wall_ema_l / 15.0).clamp(0.0, 1.0)) * 127.0)) as i8;
        sensory_wave[86] = (((1.0 - (racer.wall_ema_r / 15.0).clamp(0.0, 1.0)) * 127.0)) as i8;

        // Canal 87: SLIP ANGLE — ângulo entre velocity e forward (sinal de drift/aquaplanagem).
        // Positivo = velocity aponta à direita do heading (drift pra direita).
        let slip_signal = if let Some(ref body) = opt_body {
            if body.velocity.length_squared() > 1.0 {
                let v2d = Vec2::new(body.velocity.x, body.velocity.z).normalize_or_zero();
                let f2d = forward_2d.normalize_or_zero();
                // cross-z: f × v
                f2d.x * v2d.y - f2d.y * v2d.x
            } else { 0.0 }
        } else { 0.0 };
        sensory_wave[87] = (slip_signal * 127.0).clamp(-127.0, 127.0) as i8;

        // Canal 88: PROXIMIDADE SUSTENTADA — acumula tempo raspando qualquer lateral (<3m).
        // Sinaliza "tô raspando há tempo demais, precisa afastar".
        let raspando = racer.wall_ema_l < 3.0 || racer.wall_ema_r < 3.0;
        if raspando {
            racer.proximity_sustain = (racer.proximity_sustain + dt).min(3.0);
        } else {
            racer.proximity_sustain = (racer.proximity_sustain - dt * 0.5).max(0.0);
        }
        sensory_wave[88] = ((racer.proximity_sustain / 3.0) * 127.0) as i8;

        // Canal 89: estado de ré/cooldown do frame ANTERIOR. 127 em ré, 64 em cooldown, 0 normal.
        // (a atualização real do timer acontece mais abaixo, após a rede decidir)
        sensory_wave[89] = if racer.reverse_timer > 0.0 { 127 }
                           else if racer.post_reverse_lockout > 0.0 { 64 }
                           else { 0 };

        // --- SKIDMARKS DE DRIFT ---
        if is_drifting && rand::random::<f32>() < 0.15 { // rate limit to ~10 FPS
            let (mark_mesh, mark_mat) = skidmark_cache.get_or_insert_with(|| {
                (
                    meshes.add(Rectangle::new(1.0, 1.0)),
                    materials.add(StandardMaterial {
                        base_color: Color::srgba(0.0, 0.0, 0.0, 0.7),
                        alpha_mode: AlphaMode::Blend,
                        unlit: true,
                        ..default()
                    })
                )
            });

            let mark_length = (speed * 0.06).max(0.5);
            let mut mark_t = *transform;
            mark_t.translation = pos - mark_t.right().as_vec3() * 0.9;
            mark_t.translation.y = 0.51;
            mark_t.rotation *= Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
            mark_t.scale = Vec3::new(mark_length, 0.4, 1.0);

            let _e = commands.spawn((Mesh3d(mark_mesh.clone()), MeshMaterial3d(mark_mat.clone()), mark_t, Skidmark { timer: 5.0 })).id();
            // Not adding to world_entities because they decay on their own and
            // cleaning them up explicitly could block the CPU unnecessarily during reset.
        }

        // Memória temporal
        sensory_wave[120] = (racer.memory[0] * 127.0) as i8;
        sensory_wave[121] = (racer.memory[1] * 127.0) as i8;
        sensory_wave[122] = (racer.memory[2] * 127.0) as i8;

        // --- SISTEMA DOPAMINÉRGICO (Reward Shaping Contínuo) ---
        // joy_tick = progresso_alinhado * superfície. Streak compõe flow state.
        // Evolução seleciona quem MAXIMIZA a integral — grama/parede são PERMITIDAS, só rendem menos/custam.
        let progress_speed = (speed * alignment.max(0.0) / 30.0).clamp(0.0, 1.0); // 30m/s ≈ 108km/h = pico
        let surface_factor = match voxel_current {
            VoxelType::Road => 1.0,
            VoxelType::Grass => 0.2,
            VoxelType::Wall => -0.5,
            VoxelType::Hole => -1.0, // Hole é pior que parede — reforça evitar.
        };
        let joy_tick = progress_speed * surface_factor;
        if joy_tick > 0.5 {
            racer.joy_streak = (racer.joy_streak + dt).min(5.0);
        } else {
            racer.joy_streak = (racer.joy_streak - dt * 2.0).max(0.0); // decai 2x rápido
        }
        let dopamine_mult = 1.0 + racer.joy_streak * 0.5; // até 3.5x em 5s de flow
        racer.joy_integral += joy_tick * dopamine_mult * dt;

        // Canais 126/127: expõe a dopamina como sinal sensorial (meta-cognição evoluível)
        sensory_wave[126] = (joy_tick * 127.0).clamp(-127.0, 127.0) as i8;
        sensory_wave[127] = ((racer.joy_streak / 5.0) * 127.0) as i8;

        // --- TIER 3.1: DOPAMINA (Reward Prediction Error) ---
        // actual_reward ≈ delta_track_progress[m] * 10 - wall_contact_cost.
        // Proxy para delta: speed * alignment * dt (componente forward de avanço).
        // wall_contact_cost dispara com superfície de parede (voxel == Wall).
        let delta_track_progress = (speed * alignment.max(0.0) * dt).max(0.0);
        let wall_contact_cost = if matches!(voxel_current, VoxelType::Wall) { 5.0 * dt } else { 0.0 };
        let actual_reward = delta_track_progress * 10.0 - wall_contact_cost;
        racer.engine.metabolism.update_dopamine(actual_reward);

        // --- TIER 3.2: CORTISOL (medo integral) ---
        // Integra proximidade da parede mais próxima entre EMA esquerda/direita.
        // wall_ema_l/r já são distâncias em metros (decay ~1s).
        let min_wall_distance = racer.wall_ema_l.min(racer.wall_ema_r);
        racer.engine.metabolism.update_cortisol(min_wall_distance, dt);

        // --- TIER 3.3: CURIOSIDADE (novelty em retina) ---
        // Mede variância do visual_cortex nos últimos 16 frames. Monotonia → sobe.
        // visual_cortex já foi atualizado logo acima neste tick (recompute branch)
        // ou preserva o valor do substep anterior (fallback). Ambos os casos servem
        // como sinal retiniano válido pro histórico de curiosidade.
        let retina_snapshot = racer.visual_cortex;
        racer.engine.metabolism.update_curiosity(&retina_snapshot, dt);

        // --- 3. EXECUÇÃO SOBERANA ---
        // Tier 6.5 — Time Dilation sob stress:
        //   cortisol baixo (<0.5)  → skip_n = 2 (brain fires 30Hz, economia de compute).
        //   cortisol alto (≥0.5)   → skip_n = 1 (brain fires 60Hz, atenção máxima sob perigo).
        // Racer preserva engine.final_intent entre ticks → skipped frames reusam última decisão.
        // Gated atrás de gen ≥ 10 pra não desestabilizar bootstrap evolutivo.
        racer.brain_tick_counter = racer.brain_tick_counter.wrapping_add(1);
        let cortisol = racer.engine.metabolism.cortisol;
        // Time dilation desabilitada: skip_n=2 em gen>=10 reduzia brain freq pra
        // 30Hz e degradava DNA reload (cars precisam reagir 60Hz pra cap).
        let _ = cortisol;
        let skip_n: u8 = 1;
        let should_fire = (racer.brain_tick_counter % skip_n) == 0;

        // --- ETAPA PERCEPTION v2: full upgrade (with retrain) ---
        // Added channels (treinado do zero):
        //   16..19 = yaw_rate × 4 (peso ternário)
        //   20..23 = pitch_rate × 4
        //   24..27 = roll_rate × 4
        //   28..31 = pre_brake_signal × 4 (speed × |curv|)
        //   78     = curv25_signed (cortex + axiom)
        //   79     = curv25_signed (Sentinel + axiom — duplicado)
        //   80     = curv60_signed (lookahead)
        {
            let yaw = if let Some(ref body) = opt_body { body.angular_velocity.y } else { 0.0 };
            let pitch_r = if let Some(ref body) = opt_body { body.angular_velocity.x } else { 0.0 };
            let roll = if let Some(ref body) = opt_body { body.angular_velocity.z } else { 0.0 };
            let yaw_i8   = ((yaw     / 3.0).clamp(-1.0, 1.0) * 127.0) as i8;
            let pitch_i8 = ((pitch_r / 3.0).clamp(-1.0, 1.0) * 127.0) as i8;
            let roll_i8  = ((roll    / 3.0).clamp(-1.0, 1.0) * 127.0) as i8;
            // Single-channel proprio (f32 cortex aprende magnitude via weight).
            sensory_wave[16] = yaw_i8;
            sensory_wave[17] = pitch_i8;
            sensory_wave[18] = roll_i8;

            // Curvature lookahead — signed angles.
            let curv_at = |ahead_m: f32| -> f32 {
                let probe = Vec3::new(pos.x, 0.0, pos.z) + forward * ahead_m;
                let f = world.get_flow_at(probe.x, probe.z);
                if f.length_squared() > 1e-4 {
                    let cross_y = forward.x * f.y - forward.z * f.x;
                    let dot = (forward.x * f.x + forward.z * f.y).clamp(-1.0, 1.0);
                    cross_y.atan2(dot)
                } else { 0.0 }
            };
            let yn = std::f32::consts::FRAC_PI_2;
            // Lookahead estendido: 40m próximo, 100m médio, 180m longe (alcance visão 600m).
            let c40_signed = curv_at(40.0);
            let c100_signed = curv_at(100.0);
            let c180_signed = curv_at(180.0);
            let c40 = ((c40_signed / yn).clamp(-1.0, 1.0) * 127.0) as i8;
            let c100 = ((c100_signed / yn).clamp(-1.0, 1.0) * 127.0) as i8;
            let c180 = ((c180_signed / yn).clamp(-1.0, 1.0) * 127.0) as i8;
            sensory_wave[78] = c40;
            sensory_wave[79] = c40;
            sensory_wave[80] = c100;
            sensory_wave[32] = c180;

            // Pre-brake: speed × max curv across 3 horizons. Curva longa (180m) pesa
            // 0.5 (cars têm tempo). Curva próxima (40m) pesa 1.0 (urgente).
            let speed_norm = (speed / 80.0).clamp(0.0, 1.0);
            let curv_abs_max = (c40_signed.abs() / yn)
                .max(c100_signed.abs() / yn * 0.85)
                .max(c180_signed.abs() / yn * 0.5)
                .clamp(0.0, 1.0);
            let pre_brake = (speed_norm * curv_abs_max * 127.0).clamp(0.0, 127.0) as i8;
            sensory_wave[28] = pre_brake;
        }

        if should_fire {
            racer.engine.execute_cycle(&sensory_wave, track_phase, dt);
            // Tier 1.3 — Axiom como prior externo (potential-based shaping).
            // Injeta bias procedural no output do brain APÓS o forward. Escala decai
            // com a geração (race.current_axiom_strength: 1.0 gen 0 → 0.0 gen 50+).
            // Clamp em [-1, 1] preserva range do tanh já aplicado em final_intent.
            let ax_strength = race.current_axiom_strength;
            if ax_strength > 1e-4 {
                let bias = axiom_policy(&sensory_wave);
                let n = racer.engine.final_intent.len().min(64);
                for i in 0..n {
                    let v = racer.engine.final_intent[i] + bias[i] * ax_strength;
                    racer.engine.final_intent[i] = v.clamp(-1.0, 1.0);
                }
            }
            // ---- Trinity decide (Gate + Reflex + Sentinel) ----
            let cortex_intent = [
                racer.engine.final_intent.get(0).copied().unwrap_or(0.0),
                racer.engine.final_intent.get(1).copied().unwrap_or(0.0),
                racer.engine.final_intent.get(2).copied().unwrap_or(0.0),
            ];
            let speed_kmh = speed * 3.6;
            // Wall raycast proxies derived from retina samples (i8 0..127). Larger sum = closer.
            let wall_l_proxy = {
                let s: i32 = (32..38).map(|i| sensory_wave[i] as i32).sum();
                let norm = (s as f32 / (6.0 * 127.0)).clamp(0.0, 1.0);
                (1.0 - norm) * 5.0
            };
            let wall_r_proxy = {
                let s: i32 = (58..64).map(|i| sensory_wave[i] as i32).sum();
                let norm = (s as f32 / (6.0 * 127.0)).clamp(0.0, 1.0);
                (1.0 - norm) * 5.0
            };
            let fwd_obstacle_norm: f32 = {
                let s: i32 = (10..22).map(|i| sensory_wave[i] as i32).sum();
                (s as f32 / (12.0 * 127.0)).clamp(0.0, 1.0)
            };
            let ttc_forward = if speed > 0.5 {
                let clear_m = (1.0 - fwd_obstacle_norm) * 30.0;
                (clear_m / speed).clamp(0.0, 99.0)
            } else { 99.0 };
            // yaw_rate: pull from body angular velocity if available (cheap local).
            let yaw_rate: f32 = if let Some(ref body) = opt_body { body.angular_velocity.y } else { 0.0 };
            // slip_avg: read directly from wheel state (1-tick lag from physics, fine).
            let slip_avg: f32 = if let Some(ref body) = opt_body {
                body.wheels.iter().map(|w| w.slip_ratio.abs() + w.slip_angle.abs()).sum::<f32>() / 4.0
            } else { 0.0 };
            // curv25 read from sensory_wave[79..80] (filled in proprio block above the brain fire).
            let curv25: f32 = ((sensory_wave[79] as f32).abs() + (sensory_wave[80] as f32).abs()) / 254.0 * 60.0;
            let safety_state = crate::safety::SafetyState {
                speed_kmh,
                yaw_rate,
                slip_avg,
                ttc_forward,
                wall_left: wall_l_proxy,
                wall_right: wall_r_proxy,
                // slow_time increments 1.0/tick @ ~64Hz when speed<2m/s; convert to real seconds.
                stall_timer: racer.slow_time / 64.0,
                // wall_time already in real seconds; gate on near-stop so it reads "pinned vs scraping".
                pinado_timer: if speed_kmh < 8.0 { racer.wall_time } else { 0.0 },
                rain: env.weather_wetness,
                curv25,
                current_thr: cortex_intent[0],
            };
            let chaos_in = crate::chaos::ChaosInput {
                yaw_rate,
                slip_avg,
                retina_temporal_var: 0.0, // wired later from motion_left/motion_right
            };
            let reflex_in = slow_input_for_reflex(&sensory_wave);
            let trinity = racer.engine.decide(cortex_intent, &reflex_in, &safety_state, &chaos_in);
            // Mutate final_intent so downstream readers see gate-blended values.
            if let Some(slot) = racer.engine.final_intent.get_mut(0) { *slot = trinity[0]; }
            if let Some(slot) = racer.engine.final_intent.get_mut(1) { *slot = trinity[1]; }
            if let Some(slot) = racer.engine.final_intent.get_mut(2) { *slot = trinity[2]; }
            // ---- end Trinity ----
            // Atualiza last_action com a saída fresca (throttle, steer).
            let ti = racer.engine.final_intent.get(0).copied().unwrap_or(0.0);
            let si = racer.engine.final_intent.get(1).copied().unwrap_or(0.0);
            racer.last_action = (ti, si);
            TIME_DILATION_FIRES.fetch_add(1, AtomicOrdering::Relaxed);
        } else {
            // Skipped: engine.final_intent já contém última intent (persiste entre ticks).
            // Nada a fazer — downstream vai ler os mesmos valores.
            TIME_DILATION_SKIPS.fetch_add(1, AtomicOrdering::Relaxed);
        }

        // --- CÓRTEX LENTO (System 2) — agrega contexto e emite bias modulador ---
        // Input 32-dim: estado homeostático + pooled retina + saídas anteriores do fast.
        // Roda subsampled (1× a cada 32 ticks). Output em `racer.engine.slow_brain.output`.
        let lap_progress = if world.lap_distance > 0.0 { (racer.distance / world.lap_distance).fract() } else { 0.0 };
        let mut slow_input = [0i8; 32];
        slow_input[0]  = sensory_wave[126];                                    // joy_tick
        slow_input[1]  = sensory_wave[127];                                    // joy_streak
        slow_input[2]  = ((racer.hp / 100.0) * 127.0).clamp(0.0, 127.0) as i8; // hp
        slow_input[3]  = (racer.fuel * 127.0).clamp(0.0, 127.0) as i8;          // fuel
        slow_input[4]  = ((speed / 80.0) * 127.0).clamp(0.0, 127.0) as i8;      // speed / 80m/s
        slow_input[5]  = (racer.tire_temp * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[6]  = (racer.engine_power * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[7]  = (alignment * 127.0) as i8;
        slow_input[8]  = (lap_progress * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[9]  = (racer.total_laps as i8 * 25).min(127);
        slow_input[10] = ((racer.wall_time / 5.0) * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[11] = ((racer.grass_time / 5.0) * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[12] = (racer.stagnation * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[13] = if env.is_night() { 127 } else { 0 };
        // 14..21: retina pooled — 4 regiões no chão (row 0) + 4 no horizonte (row 1)
        for reg in 0..4 {
            let mut sum_g: i32 = 0;
            let mut sum_h: i32 = 0;
            for j in 0..8 {
                sum_g += racer.visual_cortex[reg * 8 + j] as i32;
                sum_h += racer.visual_cortex[32 + reg * 8 + j] as i32;
            }
            slow_input[14 + reg] = (sum_g / 8).clamp(-128, 127) as i8;
            slow_input[18 + reg] = (sum_h / 8).clamp(-128, 127) as i8;
        }
        // 22..25: últimas intenções do fast (throttle, steer, brake, shift)
        slow_input[22] = (racer.memory[0] * 127.0) as i8;
        slow_input[23] = (racer.memory[1] * 127.0) as i8;
        slow_input[24] = (racer.memory[2] * 127.0) as i8;
        slow_input[25] = (racer.engine.final_intent[3] * 127.0) as i8;
        // 26: memória auto-recorrente (slow[0] do tick anterior)
        slow_input[26] = (racer.engine.slow_brain.output[0] * 127.0) as i8;
        // 27..28, 30: LOOKAHEAD DE CURVATURA — flow da pista a 30/60/90m à frente.
        // Signed yaw [-π/2, +π/2] → normalizado pra [-127, +127]. Negativo = curva à esquerda,
        // positivo = direita. System 2 agora SABE que a próxima curva vira e em qual direção,
        // pode pré-frear e pré-girar volante antes do fast ver a parede.
        let forward_vec = forward; // alias already available
        let pos_fwd = Vec3::new(pos.x, 0.0, pos.z);
        let signed_yaw_to_flow = |ahead_m: f32| -> f32 {
            let probe = pos_fwd + forward_vec * ahead_m;
            let flow2 = world.get_flow_at(probe.x, probe.z);
            if flow2.length_squared() < 1e-4 { return 0.0; }
            let flow3 = Vec3::new(flow2.x, 0.0, flow2.y).normalize_or_zero();
            let cross_y = forward_vec.x * flow3.z - forward_vec.z * flow3.x;
            let dot = forward_vec.dot(flow3);
            cross_y.atan2(dot.max(-1.0).min(1.0))
        };
        let curv_30 = signed_yaw_to_flow(30.0);
        let curv_60 = signed_yaw_to_flow(60.0);
        let curv_90 = signed_yaw_to_flow(90.0);
        let yaw_norm = std::f32::consts::FRAC_PI_2;
        slow_input[27] = ((curv_30 / yaw_norm).clamp(-1.0, 1.0) * 127.0) as i8;
        slow_input[28] = ((curv_60 / yaw_norm).clamp(-1.0, 1.0) * 127.0) as i8;
        slow_input[29] = if racer.grace_ticks < 600 { 127 } else { 0 }; // in_grace (tick-based)
        slow_input[30] = ((curv_90 / yaw_norm).clamp(-1.0, 1.0) * 127.0) as i8;
        // canal 31: distância à parede frontal normalizada (0 = encostado, 127 = livre >6m).
        let forward_wall_dist_sensor = world.raycast(pos.x, pos.z, racer.phase, 30.0);
        slow_input[31] = ((forward_wall_dist_sensor / 6.0).clamp(0.0, 1.0) * 127.0) as i8;
        racer.engine.slow_brain.tick(&slow_input);

        // Usar final_intent estabilizado pelo CfC e KAN em vez de pesos crus (Evita ZigZag)
        let intent = &racer.engine.final_intent;

        // ================================================================
        // === CAMADA SIMBÓLICA: REFLEXOS DE SOBREVIVÊNCIA ================
        // ================================================================
        // A rede neural tem pesos aleatórios. Sem um "instinto base" de se
        // mover, ela nunca bootstrapeia. O Reflexo Simbólico garante que:
        //   1) O carro SEMPRE tenta andar para frente (throttle mínimo)
        //   2) Freio só ativa com intenção EXPLÍCITA positiva (>0.5)
        //   3) A IA modula INTENSIDADE e DIREÇÃO, não decide SE move
        //
        // A evolução seleciona quem modula MELHOR, não quem descobre que
        // precisa apertar o acelerador.
        // ================================================================

        // Throttle 100% neural — rede tem que APRENDER a acelerar via priming A3 (flow alignment).
        // Removido survival_throttle 0.3 mínimo: tirava pressão seletiva sobre aprender throttle.
        // KAN output head: spline suave por canal antes do clamp final — elimina saturação ±1
        // hard do raw logit ([-2,2] esperado) e dá curva evoluível por carro.
        let neural_throttle = racer.kan_out.eval(0, intent[0]);
        let base_pedal = ((neural_throttle + 1.0) * 0.5).clamp(0.0, 1.0); // Remap [-1,1] → [0,1]

        // --- MODULAÇÃO DO SLOW BRAIN (System 2 sussurra pro System 1) ---
        // 8 outputs: aggression, throttle_cap, risk_damp, steer_bias, brake_anticip,
        //            line_hint, mem0 (feedback recorrente via slow_input[26]), mem1 (reservado).
        // Inicia neutro (outputs=0 → multiplicadores = 1.0). Evolução descobre uso.
        let slow = &racer.engine.slow_brain.output;
        let aggression = slow[0];                             // -1..+1
        let throttle_cap = (1.0 + slow[1] * 0.3).clamp(0.5, 1.0); // 0.7..1.0
        // Tier 3.2: cortisol intensifica defensividade — risk_damp amplificado até +30%
        // (cap final 0.3 se mantém, mas baseline sobe com medo sustentado).
        let cortisol = racer.engine.metabolism.risk_aversion();
        let risk_damp = (slow[2].max(0.0) + cortisol * 0.3).min(0.3); // 0..0.3
        // slow[3..5] amplitudes BAIXAS inicialmente — pesos random injetam ruído no steer.
        // Evolução pode amplificar via learning rate (outputs saturam +1/-1 quando útil).
        // Começar pequeno evita sabotar gen 0-3 antes da seleção descobrir a função.
        let steer_bias = slow[3].clamp(-1.0, 1.0) * 0.12;     // pre-turn bias (±0.12)
        let brake_anticip = slow[4].max(0.0) * 0.20;          // pre-brake (0..0.20)
        let line_hint = slow[5].clamp(-1.0, 1.0) * 0.06;      // apex pick (±0.06)

        // Tier 3.2: medo freia (throttle -30% max) e suaviza volante (steer -20% max).
        let cortisol_throttle_mult = 1.0 - cortisol * 0.3;
        let cortisol_steer_mult = 1.0 - cortisol * 0.2;
        let pedal_intent = (base_pedal * (1.0 + aggression * 0.5) * cortisol_throttle_mult)
            .clamp(0.0, throttle_cap);
        // KAN spline no steer raw ANTES do smoothing (v8) — suaviza curva por carro evoluído.
        let steer_spline = racer.kan_out.eval(1, intent[1]);
        // Gain compensation: cérebro fala em "unidades físicas reais" — multiplicamos
        // pelo inverso de dynamic_steer_limit (cap 2×) ANTES do clamp, p/ que saturação
        // do neural ±1 corresponda a full-lock verdadeiro em qualquer velocidade.
        // Ver melhorias/gain-asymmetry.md. OFF por default (invalida scales do champion legado).
        #[cfg(feature = "gain_comp_steer")]
        let gain_comp = (1.0 + speed * 0.02).min(2.0);
        #[cfg(not(feature = "gain_comp_steer"))]
        let gain_comp = 1.0_f32;
        let steer_raw_unclamped = (steer_spline + steer_bias + line_hint) * cortisol_steer_mult * gain_comp;
        // v8: smoothing temporal via prev_steer_mem (memory[1] snapshot).
        // Combate saturação ±1 winner-take-all alternada frame a frame.
        let prev_steer_mem = racer.memory[1];
        let new_steer = steer_raw_unclamped.clamp(-1.0, 1.0);
        let steer_raw = prev_steer_mem * 0.65 + new_steer * 0.35;
        let steer_intent = steer_raw.clamp(-1.0, 1.0) * (1.0 - risk_damp);

        // Sistema de Sobreposição: Freio tem prioridade se a intenção for clara (>0.2).
        // KAN spline no brake raw antes do threshold — mesma razão do throttle/steer.
        let brake_spline = racer.kan_out.eval(2, intent[2]);
        // Brake symmetric: dead-zone central ±0.05 em vez de threshold assimétrico 0.2.
        // Permite modulação fina mas evita brake-constante de carros freshborn (spline≈0).
        // Ver melhorias/gain-asymmetry.md §Assimetria de throttle → brake.
        #[cfg(feature = "brake_symmetric")]
        let raw_brake = {
            let dead = 0.05_f32;
            if brake_spline > dead { ((brake_spline - dead) / (1.0 - dead)).clamp(0.0, 1.0) }
            else { 0.0 }
        };
        #[cfg(not(feature = "brake_symmetric"))]
        let raw_brake = if brake_spline > 0.2 { ((brake_spline - 0.2) * 1.25).clamp(0.0, 1.0) } else { 0.0 };
        let combined_brake = (raw_brake + brake_anticip).clamp(0.0, 1.0);
        let mut brake_intent = if speed < 2.0 { 0.0 } else { combined_brake };

        // SENTINEL BRAKE REFLEX — non-learnable hard brake.
        // Inspirado no Sentinel do SACI: gating físico sobrepõe decisão neural
        // quando perigo é fisicamente inescapável. Ver melhorias/gain-asymmetry.md.
        // Disparo restrito: carro ATIVAMENTE acelerando contra parede (pedal>0.4)
        // em speed alto com TTC curto. Evita amplificar stagnation em cruising.
        #[cfg(feature = "sentinel_brake")]
        {
            const SENTINEL_TTC: f32 = 0.5;
            const SENTINEL_MIN_SPEED: f32 = 10.0;
            const SENTINEL_MIN_PEDAL: f32 = 0.4;
            if speed > SENTINEL_MIN_SPEED && pedal_intent > SENTINEL_MIN_PEDAL {
                let fwd = world.raycast(pos.x, pos.z, racer.phase, 30.0);
                let ttc = fwd / speed;
                if ttc < SENTINEL_TTC {
                    let severity = (1.0 - ttc / SENTINEL_TTC).clamp(0.0, 1.0);
                    brake_intent = brake_intent.max(severity);
                }
            }
        }

        // Modula o acelerador: se estiver freando, reduz proporcionalmente o pedal
        let final_pedal_intent = (pedal_intent - brake_intent).max(0.0);

        // --- SISTEMA DE FUGA DE EMERGÊNCIA (ECAS) & STIGMERGY ---
        // Se a parede está perto (< 4m) na lateral, o reflexo físico FORÇA a roda pro lado oposto
        // Feedback visual: desenha setas vermelhas indicando a "repulsa" das paredes
        let mut emerg_steer = 0.0;
        let left_angle = racer.phase + std::f32::consts::FRAC_PI_2;
        let right_angle = racer.phase - std::f32::consts::FRAC_PI_2;
        // ECAS danger 4m→2.5m: em pista estreita, 4m tava disparando constante e brigava
        // com steering aprendido. Agora só reflexo em emergência real (<2.5m lateral).
        let danger_dist = 2.5;
        let p_dist_l = world.raycast(pos.x, pos.z, left_angle, 10.0);
        let p_dist_r = world.raycast(pos.x, pos.z, right_angle, 10.0);

        // ECAS AGORA É DICA, NÃO OVERRIDE. Severity multiplier 2.0 → 0.5.
        // A rede ainda precisa APRENDER a virar, ECAS só empurra o volante na hora do choque.
        if p_dist_l < danger_dist {
            let severity = 1.0 - (p_dist_l / danger_dist);
            emerg_steer -= severity * 0.5;
            gizmos.arrow(pos - transform.right().as_vec3() * p_dist_l, pos, Color::srgba(1.0, 0.0, 0.0, severity));
        }
        if p_dist_r < danger_dist {
            let severity = 1.0 - (p_dist_r / danger_dist);
            emerg_steer += severity * 0.5;
            gizmos.arrow(pos + transform.right().as_vec3() * p_dist_r, pos, Color::srgba(1.0, 0.0, 0.0, severity));
        }

        // ================================================================
        // === SEGURANÇA FRONTAL & RÉ AUTOMÁTICA ==========================
        // ================================================================
        // Probe frontal: quão perto está a parede na direção do nariz do carro.
        // Dois usos:
        //   1) SAFETY CAP: throttle forward sobe linearmente com a folga à frente.
        //      Encostado na parede => zero pedal; 6m+ de folga => pedal livre.
        //      Evita que a IA se encrave acelerando contra concreto.
        //   2) GATILHO DE RÉ: se muito perto (<2m) e praticamente parado,
        //      dispara reverse_timer de 1.5s. Durante a ré, shift vai para
        //      gear 0 e o pedal aplica torque de recuo. Visualmente: seta laranja.
        const SAFE_DIST: f32 = 3.0;      // folga mínima para throttle pleno (pista estreita friendly)
        const REVERSE_TRIGGER_DIST: f32 = 3.0;   // detecta antes (era 2.5)
        const REVERSE_MIN_DURATION: f32 = 0.4;   // toque curto (era 3.0)
        const REVERSE_MAX_DURATION: f32 = 1.5;   // cap duro (era 5.0)
        const REVERSE_EXIT_CLEAR: f32 = 2.5;     // só precisa afastar 2.5m (era 4.0)
        const REVERSE_PEDAL: f32 = 0.75;         // >0.5 garante stall recovery
        const POST_REVERSE_LOCKOUT: f32 = 0.6;   // cooldown curto (era 1.5)

        let forward_wall_dist = world.raycast(pos.x, pos.z, racer.phase, 30.0);
        let safety_cap = (forward_wall_dist / SAFE_DIST).clamp(0.0, 1.0);

        // --- SISTEMA DE RÉ DESATIVADO ---
        // Removido: a ré virou muleta (carros ficavam em loop frente/ré).
        // Agora: bateu = morre. Pressão seletiva máxima contra colisões.
        // Mantemos as constantes/campos pra facilitar reabilitar no futuro se necessário.
        let is_reversing = false;
        let _ = (REVERSE_TRIGGER_DIST, REVERSE_MIN_DURATION, REVERSE_MAX_DURATION,
                 REVERSE_EXIT_CLEAR, REVERSE_PEDAL, POST_REVERSE_LOCKOUT);
        racer.reverse_timer = 0.0;
        racer.post_reverse_lockout = 0.0;

        // (canal 89 já foi setado antes do execute_cycle, usando o estado do frame anterior)

        if forward_wall_dist < SAFE_DIST {
            let severity = 1.0 - safety_cap;
            let front_point = pos + transform.right().as_vec3() * forward_wall_dist;
            gizmos.arrow(front_point, pos, Color::srgba(1.0, 0.6, 0.0, severity.max(0.2)));
        }
        if is_reversing {
            gizmos.sphere(pos + Vec3::new(0.0, 2.0, 0.0), 0.5, Color::srgb(1.0, 0.4, 0.0));
        }

        // Fuga Espectral (Stigmergia) Removida - Agora a IA aprende pela rede e seleção natural

        // --- TIER 3.3: EXPLORATION NOISE (curiosity > 0.7 && stuck) ---
        // Quando curiosidade alta E organismo estagnado (>60 frames sem max_progress),
        // injeta ruído gaussiano no steering pra escapar de local optima.
        // scale 0.1 * curiosity (cap 0.1). Usa rand::rng() (consistente com motor.rs).
        let curiosity = racer.engine.metabolism.exploration_drive();
        let exploration_steer = if curiosity > 0.7 && racer.stagnation > 60.0 && !is_reversing {
            use rand::prelude::*;
            let mut rng = rand::rng();
            // Box-Muller aproximado via soma de 4 uniformes (determinístico-friendly,
            // evita dep em rand_distr). Média 0, std ≈ 0.58.
            let u: f32 = (0..4).map(|_| rng.random::<f32>() - 0.5).sum();
            u * 0.1 * curiosity
        } else {
            0.0
        };

        // Durante ré, força steer = 0 (backup reto) pra evitar rotacionar pra outra parede.
        // A rede retoma controle quando a ré termina.
        let final_steer_intent = if is_reversing {
            0.0
        } else {
            (steer_intent + emerg_steer + exploration_steer).clamp(-1.0, 1.0)
        };

        // Marchas: Auto-shift inteligente baseado em RPM e velocidade
        let shift_up_intent = racer.engine.final_intent[3];
        let shift_down_intent = racer.engine.final_intent[4];
        let mut shift_req = 0;

        if let Some(ref eng) = opt_engine {
            // A embreagem está patinando se velocidade da roda for baixa comparado ao giro
            let wheel_idx = eng.current_gear;
            let wheel_ratio = if wheel_idx <= 1 { 0.0 } else { 12.0 - ((wheel_idx - 2) as f32 * 2.0) };
            let wheel_rpm = speed * wheel_ratio.max(3.0) * 30.0;
            let is_clutch_engaged = wheel_rpm > 1200.0 && eng.current_gear >= 2;

            // Manobra de ré ativa: empurra gear para 0 (Reverse) e segura lá.
            if is_reversing {
                if eng.current_gear > 0 { shift_req = -1; }
            }
            // Auto-clutch: Sai de Neutro/Ré para 1ª SEMPRE quando parado (desativado na ré)
            else if eng.current_gear <= 1 {
                shift_req = 1;
            }
            // Guardian Anti-Stall: Se parado mas numa marcha alta (2ª, 3ª), CAI DE MARCHA IMEDIATAMENTE!
            // Impede a IA de matar o motor por ficar em marchas longas sem andar.
            else if eng.current_gear > 2 && speed < 3.0 {
                shift_req = -1;
            }
            // Auto-upshift: Sobe marcha ONLY se a embreagem estiver colada
            else if is_clutch_engaged && eng.current_rpm > eng.max_rpm * 0.85 && eng.current_gear < 6 {
                shift_req = 1;
            }
            // Auto-downshift: Desce marcha em RPM crítico
            else if is_clutch_engaged && eng.current_rpm < eng.max_rpm * 0.30 && eng.current_gear > 2 && speed > 3.0 {
                shift_req = -1;
            }
            // IA pode sobrepor com shift manual APENAS em velocidade colada (is_clutch_engaged)
            else if is_clutch_engaged {
                if shift_up_intent > 0.5 && racer.memory[3] <= 0.5 && eng.current_gear < 6 { shift_req = 1; }
                else if shift_down_intent > 0.5 && racer.memory[4] <= 0.5 && eng.current_gear > 2 { shift_req = -1; }
            }
        }

        racer.memory[0] = pedal_intent;
        racer.memory[1] = steer_raw; // v8: RNN state; lido via ch121/slow23
        racer.memory[2] = brake_intent;
        racer.memory[3] = shift_up_intent;
        racer.memory[4] = shift_down_intent;

        // --- INÉRCIA DE ATUADOR (Direção Hidráulica Rápida) ---
        // Em ré: aplica pedal reverso fixo e solta o freio, sobrescrevendo o intent neural.
        // Em modo normal: multiplica pelo safety_cap (zero se encostado na parede, 1 se >6m).
        // Pedal: durante ré, aplica REVERSE_PEDAL fixo. Em lockout pós-ré, capa em 0.3
        // (para evitar re-acelerada imediata de volta contra a parede). Normal: safety_cap.
        let pedal_target = if is_reversing {
            REVERSE_PEDAL
        } else if racer.post_reverse_lockout > 0.0 {
            (final_pedal_intent * safety_cap).min(0.3)
        } else {
            final_pedal_intent * safety_cap
        };
        let brake_target = if is_reversing { 0.0 } else { brake_intent };
        racer.memory[0] += (pedal_target - racer.memory[0]) * (dt * 10.0).min(1.0);
        racer.memory[1] += (final_steer_intent - racer.memory[1]) * (dt * 20.0).min(1.0); // De 8.0 para 20.0 (Quase instantâneo)
        racer.memory[2] += (brake_target - racer.memory[2]) * (dt * 15.0).min(1.0);

        let mut physical_throttle = racer.memory[0];
        let physical_steer = racer.memory[1]; // Sem trava de volante! Foco total nas curvas
        let physical_brake = racer.memory[2];

        // --- DIAGNÓSTICO: SNAPSHOT DO CÉREBRO (~2Hz) ---
        // Captura entradas/saídas e atenuações para dump posterior nos reports.
        racer.trace_timer -= dt;
        if racer.trace_timer <= 0.0 && racer.brain_trace.len() < 15 {
            racer.trace_timer = 0.5;
            // Top-5 canais mais ativos (|valor|)
            let sw = &sensory_wave;
            let mut pairs: Vec<(u8, i8)> = (0..128u8)
                .map(|i| (i, sw[i as usize]))
                .collect();
            pairs.sort_by(|a, b| (b.1 as i16).abs().cmp(&(a.1 as i16).abs()));
            let mut top5 = [(0u8, 0i8); 5];
            for (k, p) in pairs.iter().take(5).enumerate() { top5[k] = *p; }

            // Pico dos canais centrais da retina (10..22) — parede bem em frente
            let retina_center_max = (10..22usize)
                .chain((32+10)..(32+22))
                .map(|i| sw[i].unsigned_abs() as i8)
                .max().unwrap_or(0);

            // dynamic_steer_limit e risk_damp já calculados acima
            let slow = &racer.engine.slow_brain.output;
            let snap_risk_damp = slow[2].max(0.0).min(0.3);
            let snap_steer_limit = 1.0 / (1.0 + speed * 0.02);
            let snap_phys_steer = physical_steer * snap_steer_limit;

            let engine_time = racer.engine.current_time;
            let ri_thr = racer.engine.final_intent[0];
            let ri_str = racer.engine.final_intent[1];
            let ri_brk = racer.engine.final_intent[2];
            racer.brain_trace.push(crate::simulation::racer::BrainSnapshot {
                t: engine_time,
                speed_kmh: speed * 3.6,
                wall_prox_l: sw[85],
                wall_prox_r: sw[86],
                ttc_inv: sw[82],
                curv_25: sw[79],
                retina_center_max,
                raw_throttle: ri_thr,
                raw_steer: ri_str,
                raw_brake: ri_brk,
                post_damp_steer: final_steer_intent,
                physical_steer: snap_phys_steer,
                steer_limit: snap_steer_limit,
                risk_damp: snap_risk_damp,
                top_channels: top5,
            });
        }

        // BOS removido — o sistema de penalidades já pune sobreposição
        // A camada simbólica garante que brake só ativa com intenção explícita

        // --- DANO PROGRESSIVO MECÂNICO E TERMODINÂMICA DE ESTRESSE ---
        // LIMP MODE na ré: não multiplica por engine_power, garantindo torque suficiente
        // pra descolar da parede mesmo com HP baixo (mantém piso mínimo 0.6x do pedal).
        if !is_reversing {
            physical_throttle *= racer.engine_power;
        } else {
            physical_throttle *= racer.engine_power.max(0.6);
        }

        // Overlap penalty afrouxado: 30 → 5 HP/s. Permite trail-braking legítimo em curvas.
        // Ainda pune conflict severo, mas não mata em 3s como antes.
        // Não aplicar durante ré (pedal real vem de REVERSE_PEDAL, não do neural).
        let conflict = !is_reversing && final_pedal_intent > 0.4 && brake_intent > 0.4;
        if conflict {
            racer.hp -= dt * 5.0;
            gizmos.sphere(pos + Vec3::new(0.0, 1.5, 0.0), 0.3, Color::srgb(1.0, 0.5, 0.0));
        }

        // --- OVER-REV PENALTY & CLUTCH BURN (Termodinâmica de motor) ---
        // Bootstrap grace: primeiros 10s sem penalidades duras — IA precisa de tempo
        // para o priming genético se expressar antes de ser culpada.
        let in_grace = racer.grace_ticks < 600;
        if let Some(ref eng) = opt_engine {
            if eng.current_rpm > eng.max_rpm * 0.95 {
                racer.hp -= dt * 5.0; // Redline sempre pune — é sinal claro, não estagnação
            }
            // Burnout: reduzido e só fora do grace
            if !in_grace && speed < 1.0 && eng.current_rpm > eng.max_rpm * 0.7 {
                racer.hp -= dt * 2.0; // era 5.0
            }
        }

        // --- PENALIDADE DE ESTAGNAÇÃO / ANDAR DEVAGAR ---
        // Só ativa após grace e fora de manobra de ré (recuar é lento mas legítimo).
        if !in_grace && !is_reversing && speed < 1.0 {
            racer.hp -= dt * 5.0;
        }

        // --- MODELO TÉRMICO DOS PNEUS (agora per-wheel em physics.rs) ---
        // Ler a média das rodas para a telemetria do Racer
        if let Some(ref body) = opt_body {
            racer.tire_temp = body.wheels.iter().map(|w| w.tire_temp).sum::<f32>() / 4.0;
        }

        // --- GRAMA: Penalidade de Tempo ("DOR" Neural Constante) ---
        if voxel_current == VoxelType::Grass {
            racer.grass_time += dt;
        }

        // --- HOLE: Morte instantânea ---
        // VoxelType::Hole sob centro do carro → kill. Força a rede a desviar ativamente
        // de buracos (pixel distinto na retina ~90, enquanto road ~-80 e wall ~127).
        // Grace window preserva bootstrap: primeiros 180 frames reais sem penalidade.
        if voxel_current == VoxelType::Hole && !racer.is_dead && racer.grace_ticks >= 180 {
            racer.is_dead = true;
            racer.hp = 0.0;
            racer.death_reason = Some(crate::simulation::racer::DeathReason::Hole);
            racer.death_frame = racer.grace_ticks;
            racer.death_pos = pos;
            race.death_markers.push(pos);
            if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
            commands.entity(entity).insert(Visibility::Hidden);
            println!("💀 {} HOLE frame {} pos=({:.1},{:.1})", racer.name, racer.grace_ticks, pos.x, pos.z);
        }

        // --- NPC HIT: Dano pesado, kill se HP zero ---
        // Cooldown reutilizado do wall_cooldown evita double-dip no mesmo hit.
        if !racer.is_dead && racer.grace_ticks >= 180 && env.npc_hit(Vec2::new(pos.x, pos.z)) && racer.wall_cooldown == 0.0 {
            racer.telemetry.hit_npc_count += 1;
            racer.wall_cooldown = 0.8;
            racer.hp -= 50.0;
            if racer.hp <= 0.0 {
                racer.is_dead = true;
                racer.hp = 0.0;
                racer.death_reason = Some(crate::simulation::racer::DeathReason::NpcHit);
                racer.death_frame = racer.grace_ticks;
                racer.death_pos = pos;
                race.death_markers.push(pos);
                if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
                commands.entity(entity).insert(Visibility::Hidden);
                println!("💀 {} NPC_HIT frame {}", racer.name, racer.grace_ticks);
            }
        }

        // --- DINÂMICA DE CASTER (STEERING WEIGHT DEPENDENTE DA VELOCIDADE) ---
        // Steer_limit = MaxSteer / (1.0 + speed * 0.02)
        // Impede virar 90° a 200km/h, forçando suavidade de F1
        let dynamic_steer_limit = 1.0 / (1.0 + speed * 0.02);

        // engine_power degrada com HP (Dano Progressivo). Se zerar o HP, o motor morre (0.0 de power)
        racer.engine_power = (racer.hp / 100.0).clamp(0.0, 1.0);

        // Morte por falha mecânica CATASTRÓFICA (HP <= 0.0) ou tempo esgotado.
        if racer.hp <= 0.0 && !racer.is_dead {
            racer.is_dead = true;
            if racer.death_reason.is_none() {
                racer.death_reason = Some(crate::simulation::racer::DeathReason::HpZero);
                racer.death_frame = racer.grace_ticks;
                racer.death_pos = pos;
            }
            // Cria MEMÓRIA DE DOR no mundo para as próximas gerações
            race.death_markers.push(pos);

            if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
            // --- HIDE-ON-DEATH: libera a tela com 500 carros ---
            // Mantemos a entidade (pipeline precisa ler Racer pro fitness/pool),
            // mas esconde o mesh + filhos. InheritedVisibility propaga pro wing,
            // cockpit e farol. Custo de render: zero.
            commands.entity(entity).insert(Visibility::Hidden);
        } else if !racer.is_dead {
            // Cor reflete estado mecânico: vermelho = danificado, ciano normal
            if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                let health_ratio = (racer.hp / 100.0).clamp(0.0, 1.0);
                mat.base_color = Color::srgb(1.0, health_ratio, health_ratio);
            }
        }

        // --- 4. ENGINE E FÍSICA ---
        let flow_pull = (track_phase - racer.phase).sin() * dev.flow_influence;
        let steering_speed = (physical_steer * dynamic_steer_limit) * 2.5 + flow_pull;

        if let (Some(engine_comp), Some(body)) = (&mut opt_engine, &mut opt_body) {
            engine_comp.throttle = physical_throttle;
            engine_comp.brake = physical_brake;
            // Limita o ângulo real da roda traseira dependendo da velocidade
            engine_comp.steer = physical_steer * dynamic_steer_limit;
            engine_comp.manual_shift_req = shift_req;

            // Motor afogado: NÃO mata instantaneamente!
            // A IA precisa aprender a sequência de partida (reduzir pra marcha 0/1 + segurar acelerador).
            // Drena HP lentamente (bateria do starter) como pressão evolutiva para reiniciar rápido.
            if engine_comp.is_stalled {
                racer.hp -= 1.0 * dt; // Bateria drena lentamente: ~100s até morrer
            }

            // YAW FÍSICO ATIVADO!
            // A rotação do carro agora é dominada inteiramente pelas forças dos pneus em physics.rs.
            // A IA vira a roda da frente -> gera força lateral no pneu -> gera torque -> vira o chassi.

            racer.amplitude = body.velocity.length();
            if racer.amplitude > racer.peak_speed { racer.peak_speed = racer.amplitude; }
            racer.phase = transform.rotation.to_euler(EulerRot::YXZ).0 * -1.0;

            // --- TIER 1.1: TELEMETRIA DENSA (reward shaping) ---
            // Amostra a cada fixed tick (64Hz). voxel==Road só premia exploração útil (anti-novelty-exploit).
            let tel_on_road = voxel_current == VoxelType::Road;
            racer.telemetry.record(
                final_steer_intent,
                body.velocity.length(),
                body.angular_velocity.y,
                tel_on_road,
                transform.translation.x,
                transform.translation.z,
            );

            // Distância verdadeira: Projeção da velocidade na direção da pista (Flow)
            let flow_3d = Vec3::new(flow.x, 0.0, flow.y).normalize_or_zero();
            let track_velocity = body.velocity.dot(flow_3d);

            // Punição de Fitness por Zigzag + Penalidade de Grama
            let lateral_velocity = body.velocity.cross(flow_3d).length();

            // SÓ ACUMULA DISTÂNCIA SE NÃO ESTÁ NA PAREDE
            // Isso elimina completamente a estratégia de "ricochete produtivo"
            let on_wall = world.get_voxel_at(transform.translation.x, transform.translation.z) == VoxelType::Wall;
            if !on_wall {
                racer.distance += (track_velocity - lateral_velocity * 0.5) * dt;
            }

            // --- AMOSTRAGEM DE PATH (~4Hz) para desenhar traçado do campeão ---
            racer.path_timer -= dt;
            if racer.path_timer <= 0.0 {
                racer.path_timer = 0.25;
                if racer.path.len() < 1500 {
                    racer.path.push(transform.translation);
                }
            }

            // --- ANTI-PLATEAU / ANTI-DONUT ---
            // Passar 10s sem ganhar 3m → drena 10HP/s (morte em ~10s de loop).
            // Afrouxado pra pistas estreitas: carro precisa de tempo pra manobrar em curva fechada,
            // e 5s/20HP/s antes matava qualquer um que parasse pra pensar numa L de 90°.
            if racer.track_progress > racer.progress_checkpoint + 3.0 {
                racer.progress_checkpoint = racer.track_progress;
                racer.progress_stall = 0.0;
            } else {
                racer.progress_stall += dt;
            }
            if !in_grace && !is_reversing && racer.progress_stall > 10.0 {
                racer.hp -= 10.0 * dt;
            }

            // Punição Severa por Andar na Contramão — mas não durante manobra de ré de emergência.
            if !is_reversing && racer.distance < -5.0 {
                racer.hp -= 100.0 * dt;
            }

            // Punição por fazer "zerinho" (Spinning in place donuts)
            if body.angular_velocity.y.abs() > 1.5 && track_velocity.abs() < 5.0 {
                racer.hp -= 30.0 * dt; // Se ficar girando mais que 1.5 rad/s parado, morre em ~3s
            }

            // Punição de Desgaste de Pneu (Destruição Térmica): Evitar Burnouts Livres
            // Se as rodas estão escorregando loucamente (Slip Ratio enorme) e o carro não avança muito
            let avg_slip = body.wheels.iter().map(|w| w.slip_ratio.abs() + w.slip_angle.abs()).sum::<f32>() / 4.0;
            if avg_slip > 0.5 && track_velocity.abs() < 15.0 {
                // "Estourar os Pneus": Drena vida absurdamente rápido se derrapar à toa acelerando a 80% parado
                racer.hp -= avg_slip * 10.0 * dt;
            }

            // === DRIFT REWARD (derrapagem limpa) ===
            // Carro deslizando MAS mantendo velocidade na direção do track = dopamina bônus.
            // Requer: slip_angle médio alto + forward speed real + throttle engajado + alinhamento ok.
            // Burnout parado (acima) continua sendo punido — esses dois não se sobrepõem.
            let avg_slip_angle = body.wheels.iter().map(|w| w.slip_angle.abs()).sum::<f32>() / 4.0;
            if avg_slip_angle > 0.15 && track_velocity > 10.0 && pedal_intent > 0.3 && alignment > 0.3 {
                // Escala: drift magnitude × velocidade × alinhamento com pista.
                // Cap em 0.6 rad (~34°) pra não premiar spins absurdos.
                let drift_bonus = avg_slip_angle.min(0.6) * track_velocity * alignment * 0.08;
                racer.joy_integral += drift_bonus * dt;
                // Fumaça visível: dobra chance de skidmark durante clean drift.
                if rand::random::<f32>() < 0.25 {
                    racer.tire_temp = (racer.tire_temp + 0.02 * dt).min(1.0);
                }
            }

            // === DRAFTING: Redução de arrasto 30% se atrás de outro carro ===
            if in_draft {
                let drag_reduction = body.velocity.normalize_or_zero() * body.velocity.length_squared() * 0.18 * 0.3;
                body.forces += drag_reduction;
            }
        } else {
            // Fallback kinematic
            let target_amp = pedal_intent * dev.wave_amplitude;
            racer.amplitude = racer.amplitude + (target_amp - racer.amplitude) * 2.0 * dt;
            racer.phase = racer.phase + steering_speed * dt;
            let wave_vector = Vec3::new(racer.phase.cos(), 0.0, racer.phase.sin()) * racer.amplitude;
            transform.translation += wave_vector * dt;
            transform.rotation = Quat::from_rotation_y(-racer.phase);
            racer.distance += (wave_vector * dt).length();
        }

        racer.wall_cooldown = (racer.wall_cooldown - dt).max(0.0);
        let mut hit_wall = false;

        // Verifica colisões usando as mesmas probes físicas para não ser salvo pelo bounce
        if world.get_voxel_at(transform.translation.x, transform.translation.z) == VoxelType::Wall {
            hit_wall = true;
        } else {
            // Se o centro está livre, checa os 4 cantos (nariz, traseira, laterais)
            // Constantes aproximadas do chassis (wheelbase, track_width) — ignora opt_body por ora.
            let wb = 2.6_f32;
            let tw = 1.6_f32;

            let pos = transform.translation;
            let forward = transform.right().as_vec3();
            let right = -transform.back().as_vec3();

            let probe_points = [
                pos + forward * (wb * 0.5),   // Nariz
                pos - forward * (wb * 0.5),   // Traseira
                pos + right * (tw * 0.5),     // Lateral direita
                pos - right * (tw * 0.5),     // Lateral esquerda
            ];

            for p in &probe_points {
                if world.get_voxel_at(p.x, p.z) == VoxelType::Wall {
                    hit_wall = true;
                    break;
                }
            }
        }

        // GRACE em frames REAIS — virtual time é caótico nos primeiros frames (FPS
        // irregular em swarm grande, max_delta do Bevy, etc). 180 frames reais ≈ 3s real
        // ≈ tempo suficiente pra rede propagar e carro sair da largada em QUALQUER time_scale.
        let out_of_grace = racer.grace_ticks >= 180;

        // === POST-LAP GRACE (calculado cedo — wall check abaixo precisa de HP grace) ===
        if racer.post_lap_grace > 0 { racer.post_lap_grace = racer.post_lap_grace.saturating_sub(1); }
        let advanced_past_line = racer.post_lap_checkpoint > 0.0
            && racer.track_progress >= racer.post_lap_checkpoint + 25.0;
        let in_post_lap_grace = racer.post_lap_grace > 0 || (racer.post_lap_checkpoint > 0.0 && !advanced_past_line);
        let in_post_lap_hp_grace = racer.post_lap_grace > 480;

        if hit_wall && !racer.is_dead {
            racer.wall_time += dt; // dt capado; wall_time em segundos REAIS (lento, estável).
            // PINADO: 0.8s REAIS raspando parede = morte.
            let pinned = out_of_grace && !in_post_lap_hp_grace && racer.wall_time > 0.8;
            if pinned {
                racer.is_dead = true;
                racer.hp = 0.0;
                racer.death_reason = Some(crate::simulation::racer::DeathReason::Pinado);
                racer.death_frame = racer.grace_ticks;
                racer.death_pos = pos;
                race.death_markers.push(pos);
                if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
                commands.entity(entity).insert(Visibility::Hidden);
                println!("💀 {} PINADO frame {} (wall_time={:.2}s)", racer.name, racer.grace_ticks, racer.wall_time);
            } else if racer.wall_cooldown == 0.0 {
                racer.wall_hits += 1;
                racer.wall_cooldown = 0.6;
                // Damage proporcional à componente NORMAL da velocidade na parede.
                // Scrape paralelo (velocity dot wall_normal ≈ 0) = 8 dmg.
                // Batida frontal 30m/s = 60 dmg (out_of_grace) ou 35 (grace).
                // Aproximação: usa lateral_vel (normal à pista) como proxy da velocidade normal.
                let impact_speed = if let Some(ref body) = opt_body {
                    let lateral_dir = transform.back().as_vec3();
                    body.velocity.dot(lateral_dir).abs()
                } else { 10.0 };
                let impact_norm = (impact_speed / 30.0).clamp(0.0, 1.0);
                let max_dmg = if out_of_grace { 60.0 } else { 35.0 };
                let mut dmg = 8.0 + impact_norm * (max_dmg - 8.0);
                if in_post_lap_hp_grace { dmg *= 0.3; }
                racer.hp -= dmg;
                if racer.hp <= 0.0 {
                    racer.is_dead = true;
                    racer.hp = 0.0;
                    racer.death_reason = Some(crate::simulation::racer::DeathReason::Batida);
                    racer.death_frame = racer.grace_ticks;
                    racer.death_pos = pos;
                    race.death_markers.push(pos);
                    if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
                    commands.entity(entity).insert(Visibility::Hidden);
                    println!("💀 {} BATIDA frame {} ({} hits)", racer.name, racer.grace_ticks, racer.wall_hits);
                } else if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                    let health_ratio = (racer.hp / 100.0).clamp(0.0, 1.0);
                    mat.base_color = Color::srgb(1.0, health_ratio, health_ratio);
                }
            }
        } else {
            // Decay lento: dt * 0.1 = precisa de ~10s reais contínuos sem hit pra zerar.
            racer.wall_time = (racer.wall_time - dt * 0.1).max(0.0);
        }

        // --- CHECKPOINT RECOVERY (HEAL GATE) ---
        // A cada 25m de pista (arc-length real), checkpoint alcançado.
        // User feedback: carros morriam antes de chegar no próximo marco.
        //  - Intervalo 50→25m: cura mais frequente, sobrevivem enquanto aprendem.
        //  - HP boost 25→40: compensa wall hit (60 dmg) em 2 waypoints (antes: 3).
        //  - Reset completo de estagnação: stagnation/progress_stall/progress_checkpoint.
        //    Antes só zerava wall_hits — carro ainda morria por progress_stall antigo
        //    mesmo cruzando waypoint.
        let current_wp = (racer.track_progress / 25.0).floor().max(0.0) as usize;

        if current_wp > racer.last_waypoint {
            racer.last_waypoint = current_wp;
            racer.hp = (racer.hp + 40.0).min(100.0);
            racer.wall_hits = 0;
            racer.stagnation = 0.0;
            racer.progress_stall = 0.0;
            racer.progress_checkpoint = racer.track_progress;
            if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                mat.base_color = Color::srgb(0.5, 1.0, 1.0);
            }
        }

        // === STAGNATION MONOTÔNICA (em FRAMES reais) ===
        // max_progress SÓ CRESCE. Sem avanço > 2m em 180 frames (≈3s reais), morte.
        // Frame-based = estável independente de time_scale / FPS / max_delta do Bevy.
        let prev_max = racer.max_progress;
        racer.max_progress = racer.max_progress.max(racer.track_progress);
        if out_of_grace && !in_post_lap_grace {
            if racer.max_progress > prev_max + 0.01 {
                // v2: checkpoint 2m→0.5m (stagnation zera com progresso fino, não só chunks grandes).
                // Carros que vão lentamente mas avançam não morrem por STAG.
                if racer.max_progress > racer.progress_checkpoint + 0.5 {
                    racer.progress_checkpoint = racer.max_progress;
                    racer.stagnation = 0.0;
                } else {
                    racer.stagnation += 1.0;
                }
            } else {
                racer.stagnation += 1.0;
            }
            // v2: threshold 240→480 frames (~7.5s sim). Mais paciência pra cars em curva lenta.
            if racer.stagnation > 480.0 {
                racer.is_dead = true;
                racer.death_reason = Some(crate::simulation::racer::DeathReason::Stagnation);
                racer.death_frame = racer.grace_ticks;
                racer.death_pos = pos;
                race.death_markers.push(pos);
                if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
                commands.entity(entity).insert(Visibility::Hidden);
                println!("💀 {} STAGNATION aos frame {} (sem avanço {} frames)", racer.name, racer.grace_ticks, racer.stagnation as u32);
            }
        } else {
            racer.stagnation = 0.0;
            racer.progress_checkpoint = racer.max_progress;
        }

        // === SPEED BLEED (em FRAMES reais) ===
        // Velocidade < 2 m/s por 120 frames (≈2s reais) E sem avanço → sangra 100 HP/s real.
        // Guard `making_progress`: curvas lentas (hairpin) não devem matar — muitos carros
        // passavam reduzindo a <3 m/s no apex e morriam por bleed mesmo avançando. Agora só
        // bleed quando parados DE VERDADE (sem crescer max_progress).
        let cur_speed = if let Some(ref body) = opt_body { body.velocity.length() } else { 0.0 };
        let making_progress = racer.max_progress > prev_max + 0.005;
        if out_of_grace && !in_post_lap_grace && cur_speed < 2.0 && !making_progress {
            racer.slow_time += 1.0;
            if racer.slow_time > 180.0 {
                racer.hp -= 100.0 * dt;
                if racer.hp <= 0.0 {
                    racer.is_dead = true;
                    racer.hp = 0.0;
                    racer.death_reason = Some(crate::simulation::racer::DeathReason::SpeedBleed);
                    racer.death_frame = racer.grace_ticks;
                    racer.death_pos = pos;
                    race.death_markers.push(pos);
                    if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
                    commands.entity(entity).insert(Visibility::Hidden);
                    println!("💀 {} SPEED_BLEED aos frame {} (slow_time={} frames)", racer.name, racer.grace_ticks, racer.slow_time as u32);
                }
            }
        } else {
            racer.slow_time = (racer.slow_time - 2.0).max(0.0);
        }

        // Fuel burn base + quadrático no pedal. Com survival_pressure: curva custa mais.
        #[cfg(feature = "survival_pressure")]
        let corner_cost = {
            let steer_abs = physical_steer.abs();
            steer_abs * pedal_intent * 0.003
        };
        #[cfg(not(feature = "survival_pressure"))]
        let corner_cost = 0.0_f32;
        racer.fuel -= (0.002 + pedal_intent.powi(2) * 0.004 + corner_cost) * dt;

        // Idle brake penalty: brake forte + parado dreno HP pequeno.
        // Caçador sem presa não pode hibernar freando.
        #[cfg(feature = "survival_pressure")]
        if brake_intent > 0.5 && speed < 3.0 && !is_reversing {
            racer.hp -= 2.0 * dt;
        }
        if racer.fuel <= 0.0 && !racer.is_dead {
            racer.is_dead = true;
            racer.death_reason = Some(crate::simulation::racer::DeathReason::Fuel);
            racer.death_frame = racer.grace_ticks;
            racer.death_pos = pos;
            commands.entity(entity).insert(Visibility::Hidden);
        }

        // Contagem de Voltas — DIRECTION-AGNOSTIC via wrap detection + track_progress MONOTÔNICO.
        //
        // Bug fix crítico: antes, track_progress = lap_prog + total_laps*lap_dist. Quando carro
        // cruzava a linha MAS lap não contava (check x-axis falhava OU required_progress bloqueava
        // spawn distribuído), track_progress REGREDIA de ~450 pra ~0 → max_progress parava →
        // stagnation matava. Carro sumia "na linha de chegada" sem completar a volta.
        //
        // Agora: track_progress acumula delta_arc SEMPRE (detecta wrap forward/backward no arc).
        // Fitness usa (track_progress - spawn_progress), total_laps é DESACOPLADO e só incrementa
        // quando NET travel >= (total_laps+1) * 0.85 * lap_dist — sem fraudar spawn distribuído.
        if !world.waypoints_world.is_empty() && world.lap_distance > 0.0 {
            let cur_lap_prog = world.get_track_progress(transform.translation.x, transform.translation.z);
            let prev_lap_prog = racer.prev_lap_prog;
            let lap_dist = world.lap_distance;

            // Delta assinado com wrap detection. Se arc "pulou" de >0.5*lap pra <0.5*lap: forward wrap.
            let mut delta = cur_lap_prog - prev_lap_prog;
            let wrapped_forward = delta < -lap_dist * 0.5;
            let wrapped_backward = delta > lap_dist * 0.5;
            if wrapped_forward { delta += lap_dist; }
            else if wrapped_backward { delta -= lap_dist; }

            // Cap delta pra evitar salto absurdo (glitch de get_track_progress ou teleporte).
            let delta_clamped = delta.clamp(-lap_dist * 0.1, lap_dist * 0.1);
            racer.track_progress += delta_clamped;

            if wrapped_forward {
                let net = racer.track_progress - racer.spawn_progress;
                let required = (racer.total_laps as f32 + 1.0) * lap_dist * 0.85;
                if net >= required {
                    racer.total_laps += 1;
                    racer.distance = 0.0;
                    if racer.total_laps % 3 == 0 { racer.fuel = 1.0; }
                    // Grace imediato pós-lap: carro passa pela linha parando, max_progress
                    // congela em ~lap_dist, precisa de janela pra re-acelerar sem morrer.
                    // 240f @ 60fps = 4s reais — suficiente pra sair do stall sem premiar preguiça.
                    racer.post_lap_grace = 600;
                    racer.post_lap_checkpoint = racer.track_progress;
                    racer.stagnation = 0.0;
                    racer.slow_time = 0.0;
                    racer.progress_checkpoint = racer.max_progress;
                    racer.hp = (racer.hp + 50.0).min(100.0);
                    racer.wall_hits = 0;
                    racer.wall_time = 0.0;
                    println!("🏁 LAP {} | {} | gen_frame={} | pos=({:.1},{:.1}) | net_prog={:.0}m",
                        racer.total_laps, racer.name, racer.grace_ticks,
                        transform.translation.x, transform.translation.z, net);
                }
                // WRAP sem LAP silenciado — spawn distribuído gera ruído; lap real já sinaliza progresso.
            }
            racer.prev_lap_prog = cur_lap_prog;
        }

        // --- 6. ESPIONAGEM NEURAL (GIZMOS DO CAMPO DE VISÃO) ---
        // Toggle KeyH (dev.show_vision) — esconde cones verdes pra visual limpo.
        if dev.show_vision && (race.selected_racer.is_some() && race.selected_racer.unwrap() == entity ||
           (race.selected_racer.is_none() && current_best == entity)) {

            let forward = *transform.right();
            let up = *transform.up();
            let eye_pos = transform.translation + Vec3::Y * 0.4;

            // Visualmente: SÓ fóvea verde 180° tan-warp. Periférica + whiskers continuam
            // alimentando cérebro (canais 32-63 + 94-97) mas gizmo fica limpo.
            let fov_half = 90.0_f32.to_radians();
            const TAN_K: f32 = 1.25_f32;
            let tan_k_val = TAN_K.tan();
            let fovea_len = 60.0;
            let fovea_color = Color::srgba(0.0, 1.0, 0.0, 0.22);

            for i in 0..32 {
                let normalized_i = (i as f32 / 31.0) * 2.0 - 1.0;
                let warped = (normalized_i * TAN_K).tan() / tan_k_val;
                let yaw_angle = warped * fov_half + racer.gaze_yaw;
                let ray_dir = Quat::from_axis_angle(up, yaw_angle) * forward;
                gizmos.line(eye_pos, eye_pos + ray_dir * fovea_len, fovea_color);
            }
            let _ = up;
        }
    }

}
