use bevy::prelude::*;
use crate::engine::SovereignEngine;
use crate::world::{VoxelType, VoxelWorldResource};
use crate::environment::EnvironmentalState;

#[derive(Component)]
pub struct Racer {
    pub name: String,
    pub car_model: String,
    pub engine: SovereignEngine,
    pub is_dead: bool,
    pub amplitude: f32, 
    pub phase: f32,     
    pub pain_memories: Vec<PainSynapse>,
    pub prev_pos: Vec3,
    pub distance: f32,
    pub fuel: f32,
    pub hp: f32,
    pub total_laps: usize,
    pub stagnation: f32,
    pub memory: [f32; 5], // [throttle, steer, brake, prev_shift_up, prev_shift_down]
    pub visual_cortex: [i8; 64],
    // --- NOVOS CAMPOS (Sovereign Drive Evolution) ---
    pub tire_temp: f32,       // Carga térmica dos pneus 0.0-1.0
    pub engine_power: f32,    // Degradação mecânica 0.1-1.0 (1.0 = full)
    pub grass_time: f32,      // Tempo acumulado na grama (penalidade fitness)
    pub wall_time: f32,        // Tempo acumulado raspando parede (penalidade fitness)
    pub aquaplaning_timer: f32, // Timer de aquaplanagem (0.5s de perda de controle)
    pub joy_integral: f32,    // Dopamina acumulada: Σ(joy_tick * streak_mult * dt) — driver de fitness
    pub joy_streak: f32,      // Tempo contínuo em flow state (cap 5s); zera em "dor"
}

pub struct PainSynapse {
    pub position: Vec3,
    pub pain_phase: f32, 
    pub intensity: f32,
}

#[derive(Component)]
pub struct Skidmark {
    pub timer: f32,
}

pub struct RacerPlugin;

impl Plugin for RacerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (update_racers, camera_system, skidmark_cleanup_system));
    }
}

fn skidmark_cleanup_system(
    mut commands: Commands,
    mut query: Query<(Entity, &mut Skidmark)>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    for (e, mut skid) in query.iter_mut() {
        skid.timer -= dt;
        if skid.timer <= 0.0 { commands.entity(e).despawn(); }
    }
}

fn update_racers(
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
        if r.distance > race.global_best_distance + 0.05 {
            race.global_best_distance = r.distance;
            race.global_best_position = Some(t.translation);
        }

        let score = if r.distance < 15.0 { 
            r.distance * 10.0 - 10000.0 
        } else { 
            let avg_speed = r.distance / r.engine.current_time.max(0.1);
            let lap_bonus = r.total_laps as f32 * 10000.0;
            lap_bonus + (r.distance * 100.0) + (avg_speed * 50.0) + (r.hp * 2.0)
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
        slow_input[29] = if racer.engine.current_time <= 10.0 { 127 } else { 0 }; // in_grace
        slow_input[30] = 127; // bias constante (input DC para aprender offset)
        slow_input[31] = 127;
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
        
        // Throttle neural
        let neural_throttle = intent[0].clamp(-1.0, 1.0);
        let survival_throttle = 0.3; // Reflexo: sempre quer andar
        let base_pedal = (survival_throttle + neural_throttle * 0.7).clamp(0.0, 1.0);

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
        
        if p_dist_l < danger_dist {
            let severity = 1.0 - (p_dist_l / danger_dist);
            emerg_steer -= severity * 2.0; // Vira direita (negativo) para fugir
            gizmos.arrow(pos - transform.right().as_vec3() * p_dist_l, pos, Color::srgba(1.0, 0.0, 0.0, severity));
        }
        if p_dist_r < danger_dist {
            let severity = 1.0 - (p_dist_r / danger_dist);
            emerg_steer += severity * 2.0; // Vira esquerda (positivo) para fugir
            gizmos.arrow(pos + transform.right().as_vec3() * p_dist_r, pos, Color::srgba(1.0, 0.0, 0.0, severity));
        }

        // Fuga Espectral (Stigmergia) Removida - Agora a IA aprende pela rede e seleção natural

        let final_steer_intent = (steer_intent + emerg_steer).clamp(-1.0, 1.0);
        
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

            // Auto-clutch: Sai de Neutro/Ré para 1ª SEMPRE quando parado
            if eng.current_gear <= 1 {
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
        racer.memory[0] += (final_pedal_intent - racer.memory[0]) * (dt * 10.0).min(1.0);
        racer.memory[1] += (final_steer_intent - racer.memory[1]) * (dt * 20.0).min(1.0); // De 8.0 para 20.0 (Quase instantâneo)
        racer.memory[2] += (brake_intent - racer.memory[2]) * (dt * 15.0).min(1.0);

        let mut physical_throttle = racer.memory[0];
        let physical_steer = racer.memory[1]; // Sem trava de volante! Foco total nas curvas
        let physical_brake = racer.memory[2];

        // BOS removido — o sistema de penalidades já pune sobreposição
        // A camada simbólica garante que brake só ativa com intenção explícita

        // --- DANO PROGRESSIVO MECÂNICO E TERMODINÂMICA DE ESTRESSE ---
        physical_throttle *= racer.engine_power;

        // Overlap penalty: Se forçar freio e acelerador juntos, estraga o motor/freio!
        let conflict = final_pedal_intent > 0.4 && brake_intent > 0.4;
        if conflict {
            racer.hp -= dt * 30.0; // Dreno agressivo de HP
            // Feedback visual: Gizmo de esforço/estresse
            gizmos.sphere(pos + Vec3::new(0.0, 1.5, 0.0), 0.3, Color::srgb(1.0, 0.5, 0.0));
        }
        
        // --- OVER-REV PENALTY & CLUTCH BURN (Termodinâmica de motor) ---
        // Bootstrap grace: primeiros 10s sem penalidades duras — IA precisa de tempo
        // para o priming genético se expressar antes de ser culpada.
        let in_grace = racer.engine.current_time <= 10.0;
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
        // Só ativa após grace. Drena mais devagar (era 10.0, agora 5.0).
        if !in_grace && speed < 1.0 {
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

            // Punição Severa por Andar na Contramão
            if racer.distance < -5.0 {
                // Dor extrema e morte rápida se andar pra trás
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

        racer.prev_pos = transform.translation;

        // --- 5. DOR E TRAUMA (Dano Progressivo por Energia Cinética) ---
        let voxel = world.get_voxel_at(transform.translation.x, transform.translation.z);
        
        if voxel == VoxelType::Wall {
            // PAREDE = MORTE INSTANTÂNEA
            // Sinal evolutivo mais claro possível: tocou, morreu.
            // Sem HP, sem ricochet, sem estratégia de "raspar na parede".
            racer.is_dead = true;
            racer.hp = 0.0;
            if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
        }
        
        // Stagnation: Ficar parado ou em contramão. Aumentado para 20s/12s para aprender partida.
        let alignment = (racer.phase - track_phase).cos();
        if (racer.amplitude < 3.0 && racer.engine.current_time > 20.0) || (racer.engine.current_time > 12.0 && alignment < 0.0) { 
            racer.stagnation += dt;
            if racer.stagnation > 1.0 { racer.is_dead = true; }
        } else { 
            // Decaimento lento em vez de reset instantâneo para evitar que "tremidas" resetem o timer
            racer.stagnation = (racer.stagnation - dt).max(0.0); 
        }

        racer.fuel -= (0.002 + pedal_intent.powi(2) * 0.004) * dt;
        if racer.fuel <= 0.0 { racer.is_dead = true; }
        
        // Contagem de Voltas
        let lap_line_x = world.start_pos.x;
        if racer.prev_pos.x < lap_line_x && transform.translation.x >= lap_line_x && racer.distance > 400.0 {
            racer.total_laps += 1; racer.distance = 0.0;
            if racer.total_laps % 3 == 0 { racer.fuel = 1.0; }
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

fn camera_system(
    mut query: Query<&mut Transform, With<Camera3d>>,
    racer_query: Query<(Entity, &Racer, &Transform), Without<Camera3d>>,
    mut race: ResMut<crate::evolution::RaceSettings>,
    mut scroll_evr: EventReader<bevy::input::mouse::MouseWheel>,
    mut motion_evr: EventReader<bevy::input::mouse::MouseMotion>,
    mouse_btn: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();

    // Zoom
    for ev in scroll_evr.read() {
        race.camera_distance = (race.camera_distance - ev.y * 15.0).clamp(10.0, 400.0);
    }

    // Toggle free cam
    if keys.just_pressed(KeyCode::KeyC) {
        race.is_free_cam = !race.is_free_cam;
    }
    
    // Ler os movimentos do mouse para não acumular
    let mut mouse_delta = Vec2::ZERO;
    for ev in motion_evr.read() {
        if mouse_btn.pressed(MouseButton::Right) {
            mouse_delta += ev.delta;
        }
    }

    if let Ok(mut cam_t) = query.single_mut() {
        if race.is_free_cam {
            let mut move_dir = Vec3::ZERO;
            let forward = cam_t.forward().as_vec3();
            let right = cam_t.right().as_vec3();
            let up = Vec3::Y;
            
            if keys.pressed(KeyCode::KeyW) { move_dir += forward; }
            if keys.pressed(KeyCode::KeyS) { move_dir -= forward; }
            if keys.pressed(KeyCode::KeyA) { move_dir -= right; }
            if keys.pressed(KeyCode::KeyD) { move_dir += right; }
            if keys.pressed(KeyCode::KeyE) { move_dir += up; }
            if keys.pressed(KeyCode::KeyQ) { move_dir -= up; }
            
            if move_dir.length_squared() > 0.0 {
                let speed = if keys.pressed(KeyCode::ShiftLeft) { 300.0 } else { 100.0 };
                cam_t.translation += move_dir.normalize() * speed * dt;
            }
            
            // Rotação com o Mouse (Segurando Botão Direito)
            if mouse_delta.length_squared() > 0.0 {
                let yaw = -mouse_delta.x * 0.003;
                let pitch = -mouse_delta.y * 0.003;
                
                // Rotaciona Yaw (global Y) - usamos rotação ao redor do Y do mundo para não bugar o horizonte
                cam_t.rotate_axis(Dir3::Y, yaw);
                // Rotaciona Pitch (local X)
                cam_t.rotate_local_x(pitch);
            }
            
            // Mantendo fallback para as setas do teclado
            if keys.pressed(KeyCode::ArrowLeft) { cam_t.rotate_axis(Dir3::Y, 2.0 * dt); }
            if keys.pressed(KeyCode::ArrowRight) { cam_t.rotate_axis(Dir3::Y, -2.0 * dt); }
            if keys.pressed(KeyCode::ArrowUp) { cam_t.rotate_local_x(2.0 * dt); }
            if keys.pressed(KeyCode::ArrowDown) { cam_t.rotate_local_x(-2.0 * dt); }
        } else {
            let mut target = race.selected_racer;
            if target.is_none() {
                if let Some((e, _, _)) = racer_query.iter().filter(|(_, r, _)| !r.is_dead).max_by(|a, b| a.1.distance.partial_cmp(&b.1.distance).unwrap()) {
                    target = Some(e);
                }
            }
            if let Some(e) = target {
                if let Ok((_, _, t)) = racer_query.get(e) {
                    let offset = Vec3::new(-0.6, 0.8, -0.6).normalize() * race.camera_distance;
                    let goal = t.translation + offset;
                    cam_t.translation = cam_t.translation.lerp(goal, 0.05);
                    cam_t.look_at(t.translation, Vec3::Y);
                }
            }
        }
    }
}
