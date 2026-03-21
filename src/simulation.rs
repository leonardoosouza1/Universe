use bevy::prelude::*;
use crate::engine::SovereignEngine;
use crate::world::{VoxelType, VoxelWorldResource};

#[derive(Component)]
pub struct Racer {
    pub name: String,
    pub engine: SovereignEngine,
    pub is_dead: bool,
    pub distance: f32,
    pub amplitude: f32, 
    pub phase: f32,     
    pub pain_memories: Vec<PainSynapse>,
    pub prev_pos: Vec3,
    pub fuel: f32,
    pub total_laps: usize,
    pub stagnation: f32,
}

pub struct PainSynapse {
    pub position: Vec3,
    pub pain_phase: f32, 
    pub intensity: f32,
}

pub struct RacerPlugin;

impl Plugin for RacerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (update_racers, camera_system));
    }
}

fn update_racers(
    mut query: Query<(Entity, &mut Racer, &mut Transform, &mut MeshMaterial3d<StandardMaterial>)>,
    world_res: Res<VoxelWorldResource>,
    dev: Res<crate::evolution::DevSettings>,
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let dt = time.delta_secs().min(0.02);
    let world = &world_res.0;

    // Coletar posições de todos os pilotos vivos para interação de enxame
    let particle_positions: Vec<(Entity, Vec3)> = query.iter()
        .filter(|(_, r, _, _)| !r.is_dead)
        .map(|(e, _, t, _)| (e, t.translation))
        .collect();

    for (entity, mut racer, mut transform, mat_handle) in query.iter_mut() {
        if racer.is_dead { continue; }

        let pos = transform.translation;
        let flow = world.get_flow_at(pos.x, pos.z);
        let track_phase = flow.y.atan2(flow.x);

        // --- 1. VISÃO DINÂMICA + RADAR DE ENXAME ---
        let vision_range = 25.0 + racer.amplitude * 1.8;
        let mut sensory_wave = vec![0i8; 128];
        let num_rays = 21;
        let mut max_harmony_dist = 0.0;
        let mut harmony_angle = 0.0;

        for i in 0..num_rays {
            let angle_offset = (i as f32 / (num_rays - 1) as f32 - 0.5) * 2.8; 
            let d = world.raycast(pos.x, pos.z, racer.phase + angle_offset, vision_range);
            let energy = (vision_range / d.max(1.0)).powi(2);
            sensory_wave[i] = (energy * 0.2).clamp(0.0, 127.0) as i8;
            if d > max_harmony_dist { max_harmony_dist = d; harmony_angle = angle_offset; }
        }

        // INTERAÇÃO ENTRE PARTÍCULAS (REPULSÃO QUÂNTICA)
        let mut repulsion_vec = Vec2::ZERO;
        for (other_entity, other_pos) in &particle_positions {
            if entity == *other_entity { continue; }
            let dist = pos.distance(*other_pos);
            if dist < 8.0 {
                // Força de Repulsão Inversa
                let diff = (pos - *other_pos).xz().normalize_or_zero();
                let push = (8.0 / dist.max(1.0)).powi(2);
                repulsion_vec += diff * push;
                
                // Injetar "Dor de Proximidade" no Canal 50 do cérebro
                sensory_wave[50] = (push * 10.0).clamp(0.0, 127.0) as i8;
            }
        }

        // --- 2. EXECUÇÃO ---
        let sync_err = (track_phase - racer.phase).sin();
        let alignment = (racer.phase - track_phase).cos();
        sensory_wave[41] = (sync_err * 127.0) as i8;
        sensory_wave[42] = (harmony_angle * 100.0).clamp(-127.0, 127.0) as i8;
        sensory_wave[43] = (alignment * 127.0) as i8;

        racer.engine.execute_cycle(&sensory_wave, track_phase, dt);
        let intent = racer.engine.instinct_layer.forward(&sensory_wave);
        let pedal_intent = (intent[0] + 0.5).clamp(0.0, 1.0);
        let steer_intent = intent[1].clamp(-1.0, 1.0);

        // --- 3. MOVIMENTAÇÃO POR ONDAS E REPULSÃO ---
        let target_amp = pedal_intent * dev.wave_amplitude;
        racer.amplitude = racer.amplitude + (target_amp - racer.amplitude) * 2.0 * dt;
        
        let flow_pull = sync_err * dev.flow_influence;
        racer.phase = racer.phase + (steer_intent * 6.0 + flow_pull) * dt;

        let mut wave_vector = Vec2::new(racer.phase.cos(), racer.phase.sin()) * racer.amplitude;
        // Aplicar a Repulsão Quântica das outras partículas
        wave_vector += repulsion_vec * 5.0;

        transform.translation += Vec3::new(wave_vector.x, 0.0, wave_vector.y) * dt;
        transform.rotation = Quat::from_rotation_y(-racer.phase);

        racer.distance += (wave_vector * dt).length();
        racer.prev_pos = transform.translation;

        // --- 4. ELIMINAÇÃO ---
        let voxel = world.get_voxel_at(transform.translation.x, transform.translation.z);
        if voxel == VoxelType::Wall || voxel == VoxelType::Grass || (racer.engine.current_time > 2.0 && alignment < 0.0) {
            racer.is_dead = true;
            if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
        }
        
        if racer.amplitude < 10.0 && racer.engine.current_time > 4.0 { 
            racer.stagnation += dt;
            if racer.stagnation > 1.0 { racer.is_dead = true; }
        } else { racer.stagnation = 0.0; }

        racer.fuel -= (0.002 + pedal_intent.powi(2) * 0.004) * dt;
        if racer.fuel <= 0.0 { racer.is_dead = true; }
        
        let lap_line_x = world.start_pos.x;
        if racer.prev_pos.x < lap_line_x && transform.translation.x >= lap_line_x && racer.distance > 400.0 {
            racer.total_laps += 1; racer.distance = 0.0;
            if racer.total_laps % 3 == 0 { racer.fuel = 1.0; }
        }
    }
}

fn camera_system(
    mut query: Query<&mut Transform, With<Camera3d>>,
    racer_query: Query<(Entity, &Racer, &Transform), Without<Camera3d>>,
    race_settings: Res<crate::evolution::RaceSettings>,
) {
    let mut target = race_settings.selected_racer;
    if target.is_none() {
        if let Some((e, _, _)) = racer_query.iter().filter(|(_, r, _)| !r.is_dead).max_by(|a, b| a.1.distance.partial_cmp(&b.1.distance).unwrap()) {
            target = Some(e);
        }
    }
    if let Some(e) = target {
        if let Ok((_, _, t)) = racer_query.get(e) {
            if let Ok(mut cam_t) = query.get_single_mut() {
                let goal = t.translation + Vec3::new(-90.0, 110.0, -90.0);
                cam_t.translation = cam_t.translation.lerp(goal, 0.04);
                cam_t.look_at(t.translation, Vec3::Y);
            }
        }
    }
}
