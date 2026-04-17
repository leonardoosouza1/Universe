use bevy::prelude::*;
use crate::world::{VoxelType, VoxelWorldResource};
use crate::environment::EnvironmentalState;
use super::racer::{Racer, Skidmark};

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
    time: Res<Time<Virtual>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut skidmark_cache: Local<Option<(Handle<Mesh>, Handle<StandardMaterial>)>>,
    mut gizmos: Gizmos,
) {
    let dt = time.delta_secs().min(0.02);
    let world = &world_res.0;

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
    let wps = &world.waypoints_world;
    for i in 0..wps.len() {
        let wp = Vec3::new(wps[i].x, 0.5, wps[i].y);

        // Pilar de checkpoint (10m de altura)
        let pillar_color = if i == 0 {
            Color::srgb(0.0, 1.0, 0.3) // Verde = Largada/Chegada
        } else {
            Color::srgba(0.0, 0.7, 1.0, 0.7) // Ciano = checkpoint normal
        };
        gizmos.line(wp, wp + Vec3::Y * 10.0, pillar_color);
        gizmos.sphere(wp + Vec3::Y * 10.0, 1.5, pillar_color);

        // Linhas conectando os waypoints (tracejado da pista ideal)
        if i < wps.len() - 1 {
            let next = Vec3::new(wps[i+1].x, 0.5, wps[i+1].y);
            gizmos.line(wp + Vec3::Y * 0.3, next + Vec3::Y * 0.3, Color::srgba(0.0, 0.7, 1.0, 0.25));
        }
    }

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

        let pos = transform.translation;
        let flow = world.get_flow_at(pos.x, pos.z);
        let track_phase = flow.y.atan2(flow.x);

        // --- 1. VISÃO DINÂMICA (COM CICLO DIA/NOITE) ---
        let mut sensory_wave = [0i8; 128]; // Stack alloc (sem heap)

        // --- VISÃO BIOLÓGICA ESTEREOSCÓPICA (Fóvea Dupla) ---
        let forward = transform.right().as_vec3();
        let right = transform.back().as_vec3();

        let retina_binocular = world.render_binocular_retina(pos + Vec3::new(0.0, 0.4, 0.0), forward, right);
        for i in 0..64 { sensory_wave[i] = retina_binocular[i]; }

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
        let mut dark_sum_i: f32 = 0.0;
        let mut dark_weight: f32 = 0.0;
        for i in 0..32 {
            // Fovea + periférica combinadas para robustez.
            let d_fov = (-retina_binocular[i] as f32).max(0.0);
            let d_per = (-retina_binocular[32 + i] as f32).max(0.0);
            let d = d_fov + d_per;
            dark_sum_i += d * (i as f32 - 15.5);
            dark_weight += d;
        }
        let dark_center_norm = if dark_weight > 1.0 { (dark_sum_i / (dark_weight * 15.5)).clamp(-1.0, 1.0) } else { 0.0 };
        let dark_center_i8 = (dark_center_norm * 127.0).clamp(-127.0, 127.0) as i8;
        sensory_wave[90] = dark_center_i8;
        sensory_wave[91] = dark_center_i8;
        sensory_wave[92] = dark_center_i8;
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
        let mut visual_stress = 0.0;
        for i in 0..64 {
            if sensory_wave[i] > 0 { visual_stress += (sensory_wave[i] as f32) / 127.0; }
        }
        visual_stress /= 64.0; // Média de brancura (0.0 a 1.0)

        // Punição psicológica direta por olhar para paredes
        racer.joy_integral -= visual_stress * 5.0 * dt;

        let panic_level = if visual_stress > 0.2 { (visual_stress - 0.2) * 1.5 } else { 0.0 };

        // --- RUÍDO NOTURNO E PÂNICO: Incerteza sensorial ---
        use rand::prelude::*;
        let mut rng = rand::rng();

        if env.is_night() || panic_level > 0.0 {
            let magnitude = (15.0 + panic_level * 50.0) as i8;
            for i in 0..64 {
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
        // --- DETERMINAR RANK PARA ESTRESSE COMPETITIVO ---
        let mut sorted_racers: Vec<(Entity, f32)> = racer_positions.iter().map(|&(e, p, _)| (e, p.x)).collect();
        sorted_racers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let my_rank = sorted_racers.iter().position(|&(e, _)| e == entity).unwrap_or(0);
        let is_leader = my_rank == 0;

        // Canal 125: Estresse Competitivo (0 se líder, 127 se atrás de alguém próximo, 64 se meio do pelotão)
        let mut comp_stress = 0;
        if !is_leader {
            comp_stress = 64;
            for &(other_e, other_pos, _) in &racer_positions {
                if other_e == entity { continue; }
                let to_other = other_pos - pos;
                if to_other.length() < 20.0 && to_other.normalize_or_zero().dot(forward) > 0.3 {
                    comp_stress = 127; // "Tira ele da frente!"
                    break;
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
            for i in 0..64 {
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
        let mut in_draft = false;
        for &(other_e, other_pos, _other_fwd) in &racer_positions {
            if other_e == entity { continue; }
            let to_other = other_pos - pos;
            let dist = to_other.length();
            let dot_forward = to_other.normalize_or_zero().dot(forward);
            if dist < 10.0 && dot_forward > 0.5 {
                in_draft = true;
                break;
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

        // --- 3. EXECUÇÃO SOBERANA ---
        racer.engine.execute_cycle(&sensory_wave, track_phase, dt);

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
        // 26..28: memória do próprio slow brain (suas saídas anteriores)
        slow_input[26] = (racer.engine.slow_brain.output[0] * 127.0) as i8;
        slow_input[27] = (racer.engine.slow_brain.output[1] * 127.0) as i8;
        slow_input[28] = (racer.engine.slow_brain.output[2] * 127.0) as i8;
        slow_input[29] = if racer.grace_ticks < 600 { 127 } else { 0 }; // in_grace (tick-based)
        // bias DC único — dois canais de 127 duplicavam o mesmo sinal e saturavam
        // o acumulador antes mesmo da evolução usar os outros canais.
        slow_input[30] = 127;
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
        let neural_throttle = intent[0].clamp(-1.0, 1.0);
        let base_pedal = ((neural_throttle + 1.0) * 0.5).clamp(0.0, 1.0); // Remap [-1,1] → [0,1]

        // --- MODULAÇÃO DO SLOW BRAIN (System 2 sussurra pro System 1) ---
        // Inicia neutro (outputs=0 → multiplicadores = 1.0). Evolução descobre uso.
        let slow = &racer.engine.slow_brain.output;
        let aggression = slow[0];                             // -1..+1
        let throttle_cap = (1.0 + slow[1] * 0.3).clamp(0.5, 1.0); // 0.7..1.0
        let risk_damp = slow[2].max(0.0).min(0.3);            // 0..0.3
        let pedal_intent = (base_pedal * (1.0 + aggression * 0.5)).clamp(0.0, throttle_cap);
        let steer_intent = (intent[1].clamp(-1.0, 1.0)) * (1.0 - risk_damp);

        // Sistema de Sobreposição: Freio tem prioridade se a intenção for clara (>0.2)
        let raw_brake = if intent[2] > 0.2 { ((intent[2] - 0.2) * 1.25).clamp(0.0, 1.0) } else { 0.0 };
        let brake_intent = if speed < 2.0 { 0.0 } else { raw_brake };

        // Modula o acelerador: se estiver freando, reduz proporcionalmente o pedal
        let final_pedal_intent = (pedal_intent - brake_intent).max(0.0);

        // --- SISTEMA DE FUGA DE EMERGÊNCIA (ECAS) & STIGMERGY ---
        // Se a parede está perto (< 4m) na lateral, o reflexo físico FORÇA a roda pro lado oposto
        // Feedback visual: desenha setas vermelhas indicando a "repulsa" das paredes
        let mut emerg_steer = 0.0;
        let left_angle = racer.phase + std::f32::consts::FRAC_PI_2;
        let right_angle = racer.phase - std::f32::consts::FRAC_PI_2;
        let danger_dist = 4.0;
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
        const SAFE_DIST: f32 = 6.0;      // folga mínima para throttle pleno
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

        // Durante ré, força steer = 0 (backup reto) pra evitar rotacionar pra outra parede.
        // A rede retoma controle quando a ré termina.
        let final_steer_intent = if is_reversing {
            0.0
        } else {
            (steer_intent + emerg_steer).clamp(-1.0, 1.0)
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

        // --- DINÂMICA DE CASTER (STEERING WEIGHT DEPENDENTE DA VELOCIDADE) ---
        // Steer_limit = MaxSteer / (1.0 + speed * 0.02)
        // Impede virar 90° a 200km/h, forçando suavidade de F1
        let dynamic_steer_limit = 1.0 / (1.0 + speed * 0.02);

        // engine_power degrada com HP (Dano Progressivo). Se zerar o HP, o motor morre (0.0 de power)
        racer.engine_power = (racer.hp / 100.0).clamp(0.0, 1.0);

        // Morte por falha mecânica CATASTRÓFICA (HP <= 0.0) ou tempo esgotado.
        if racer.hp <= 0.0 && !racer.is_dead {
            racer.is_dead = true;
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
            racer.phase = transform.rotation.to_euler(EulerRot::YXZ).0 * -1.0;

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
            // Se o carro está vivo e se movendo, mas track_progress NÃO avança,
            // está fazendo círculos/gaming. Dreno severo de HP após grace.
            // Regra: a cada 3m de progresso REAL, reseta o timer.
            // Passar 5s sem ganhar 3m → drena 20HP/s (mata em ~5s de loop).
            if racer.track_progress > racer.progress_checkpoint + 3.0 {
                racer.progress_checkpoint = racer.track_progress;
                racer.progress_stall = 0.0;
            } else {
                racer.progress_stall += dt;
            }
            if !in_grace && !is_reversing && racer.progress_stall > 5.0 {
                racer.hp -= 20.0 * dt;
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

        if hit_wall && !racer.is_dead {
            racer.wall_time += dt;
            // BATEU NO MURO = MORREU. Pressão seletiva máxima.
            // Qualquer contato com parede depois do grace é letal.
            // Durante o grace (10s), é "contato educacional": HP -50 e empurra mas não mata
            // (dá chance das gens iniciais explorarem sem apagar todas no primeiro frame).
            if racer.grace_ticks >= 600 {
                racer.is_dead = true;
                racer.hp = 0.0;
                race.death_markers.push(pos);
                if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
                commands.entity(entity).insert(Visibility::Hidden);
            } else if racer.wall_cooldown == 0.0 {
                // Durante grace: dano pesado mas não instantâneo
                racer.wall_hits += 1;
                racer.wall_cooldown = 1.0;
                racer.hp -= 50.0;
                if racer.hp <= 0.0 {
                    racer.is_dead = true;
                    racer.hp = 0.0;
                    race.death_markers.push(pos);
                    if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
                    commands.entity(entity).insert(Visibility::Hidden);
                } else if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                    let health_ratio = (racer.hp / 100.0).clamp(0.0, 1.0);
                    mat.base_color = Color::srgb(1.0, health_ratio, health_ratio);
                }
            }
        }

        // --- CHECKPOINT RECOVERY ---
        // A cada 50 metros avançados NA PISTA (arc-length real), checkpoint alcançado.
        // Usa track_progress (não euclidiano) — imune a backward-motion quebrar o contador.
        let current_wp = (racer.track_progress / 50.0).floor().max(0.0) as usize;

        if current_wp > racer.last_waypoint {
            racer.last_waypoint = current_wp;
            // Alcançou um checkpoint: recupera HP e zera strikes da parede!
            racer.hp = (racer.hp + 25.0).min(100.0);
            racer.wall_hits = 0;
            // Feedback visual de cura
            if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                mat.base_color = Color::srgb(0.5, 1.0, 1.0); // Brilha ciano rapidamente
            }
        }

        // Stagnation anti-idle: só mata se VERDADEIRAMENTE parado.
        // REMOVIDA a checagem `alignment < 0.0` — matava carros que estavam virando
        // curva fechada (cos da diferença angular fica negativo no meio de um 90°+).
        // Resultado visível: toda gen morria em 12-15s na primeira curva.
        // Progresso real (track_progress) já é coberto pelo progress_stall → hp drain.
        let alignment = (racer.phase - track_phase).cos();
        if racer.amplitude < 3.0 && racer.engine.current_time > 25.0 {
            racer.stagnation += dt;
            if racer.stagnation > 1.5 {
                racer.is_dead = true;
                commands.entity(entity).insert(Visibility::Hidden);
            }
        } else {
            racer.stagnation = (racer.stagnation - dt).max(0.0);
        }
        let _ = alignment; // mantido pra uso futuro (telemetria/curriculum)

        racer.fuel -= (0.002 + pedal_intent.powi(2) * 0.004) * dt;
        if racer.fuel <= 0.0 {
            racer.is_dead = true;
            commands.entity(entity).insert(Visibility::Hidden);
        }

        // Contagem de Voltas — cruzamento da lap_line + validação por arc-length REAL.
        // racer.track_progress ainda tem o valor do frame anterior aqui (ainda NÃO atualizamos
        // pelo lap_prog deste frame). No frame do cruzamento, o lap_prog atual deu wrap pra ~0,
        // mas o frame anterior estava em ~lap_distance — por isso o check usa o valor antigo.
        let lap_line_x = world.start_pos.x;
        let required_progress = (racer.total_laps as f32 + 1.0) * world.lap_distance * 0.90;
        if racer.prev_pos.x < lap_line_x && transform.translation.x >= lap_line_x && racer.track_progress >= required_progress {
            racer.total_laps += 1; racer.distance = 0.0;
            if racer.total_laps % 3 == 0 { racer.fuel = 1.0; }
        }

        // Comita o novo track_progress DEPOIS da contagem de voltas —
        // total_laps já pode ter sido incrementado acima, então a fórmula usa o valor atualizado.
        if !world.waypoints_world.is_empty() {
            let lap_prog = world.get_track_progress(transform.translation.x, transform.translation.z);
            racer.track_progress = lap_prog + (racer.total_laps as f32) * world.lap_distance;
        }

        // --- 6. ESPIONAGEM NEURAL (GIZMOS DO CAMPO DE VISÃO) ---
        if race.selected_racer.is_some() && race.selected_racer.unwrap() == entity ||
           (race.selected_racer.is_none() && current_best == entity) {

            let fov = 100.0_f32.to_radians();
            let forward = *transform.right();
            let eye_pos = transform.translation + Vec3::Y * 0.4;

            for i in 0..32 {
                let normalized_i = (i as f32 / 31.0) * 2.0 - 1.0;
                let yaw_angle = normalized_i * (fov / 2.0);
                let ray_dir = Quat::from_axis_angle(*transform.up(), yaw_angle) * forward;
                // 150.0 metros é o limite físico real (hit_dist) do LiDAR no world.rs
                gizmos.line(eye_pos, eye_pos + ray_dir * 150.0, Color::srgba(0.0, 1.0, 0.0, 0.15));
            }
        }
    }

}
