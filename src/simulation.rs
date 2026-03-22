use bevy::prelude::*;
use crate::engine::SovereignEngine;
use crate::resonance::ResonantNode;
use std::f32::consts::PI;

// CONSTANTES FÍSICAS REAIS PARA CARBONO (GRAFENO)
const CARBON_BOND_DIST: f32 = 14.2; // 142 pm escalonado
const MORSE_DEPTH: f32 = 5000.0; // Energia de ligação forte do C-C
const ANGULAR_STIFFNESS: f32 = 4000.0; // Força para manter os 120 graus
const THERMAL_DECAY: f32 = 0.995; // Para o resfriamento (Annealing)

#[derive(Clone, Copy, Default)]
pub struct SubPart {
    pub rel_pos: Vec3,
    pub velocity: Vec3,
    pub orbit_axis: Vec3,
}

#[derive(Component)]
pub struct Atom {
    pub element: String,
    pub mass: f32,
    pub charge: f32,
    pub radius: f32,
    pub valence: usize,
    pub velocity: Vec3,
    pub electrons: Vec<SubPart>,
}

#[derive(Component)]
pub struct ScientistIA {
    pub engine: SovereignEngine,
    pub sync_bit: i8,
    pub tunnel_stability: i32,
    pub temperature: f32, // Temperatura local do agente
    pub brain_resonator: ResonantNode,
    pub target_atom: Option<Entity>,
}

#[derive(Component)]
pub struct MainCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
}

pub struct AtomPlugin;

impl Plugin for AtomPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (atomic_physics_system, main_camera_system));
    }
}

fn atomic_physics_system(
    mut scientists: Query<&mut ScientistIA>,
    mut atoms: Query<(Entity, &mut Atom, &mut Transform, &mut MeshMaterial3d<StandardMaterial>)>,
    mut gizmos: Gizmos,
    mut materials: ResMut<Assets<StandardMaterial>>,
    time: Res<Time>,
) {
    let dt = time.delta_secs().min(0.015); 
    
    // 1. GESTÃO DOS AGENTES (O cérebro da forja)
    let mut avg_temp = 0.0;
    let sci_count = scientists.iter().count() as f32;

    for mut scientist in scientists.iter_mut() {
        let stability = scientist.tunnel_stability as f32 / 1000.0;
        
        if stability < 0.4 {
            scientist.temperature = (scientist.temperature + 5.0 * dt).min(50.0);
        } else {
            scientist.temperature *= (dt * -0.5).exp(); // Resfriamento lento (THERMAL_K_DECAY)
        }
        avg_temp += scientist.temperature;
    }
    avg_temp /= sci_count.max(1.0);

    let atom_data: Vec<(Entity, Vec3, f32, usize)> = atoms.iter()
        .map(|(e, a, t, _)| (e, t.translation, a.radius, a.valence))
        .collect();

    let mut total_stability = 0;

    // 2. FÍSICA ATÔMICA (O Laboratório de Grafeno)
    for (entity, mut atom, mut transform, mat_handle) in atoms.iter_mut() {
        let mut force = Vec3::ZERO;
        let pos = transform.translation;
        let a_radius = atom.radius;
        let a_valence = atom.valence;
        
        // Coletar vizinhos próximos
        let mut neighbors: Vec<_> = atom_data.iter()
            .filter(|(e, _, _, _)| *e != entity)
            .map(|(e, p, r, v)| (e, p, r, v, p.distance(pos)))
            .collect();
        neighbors.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap());

        let mut bonded_dirs = Vec::new();

        for (other_e, other_pos, _other_radius, _other_valence, dist) in neighbors {
            if dist < 0.1 { continue; }
            let dir = (*other_pos - pos).normalize();

            // POTENCIAL DE MORSE (Ligação Química sp2)
            let x = dist - CARBON_BOND_DIST;
            let d_morse = 2.0 * MORSE_DEPTH * (1.0 - (-x).exp()) * (-x).exp();
            
            if bonded_dirs.len() < a_valence && dist < CARBON_BOND_DIST * 1.5 {
                force += dir * d_morse; // Força de atração/repulsão Morse
                bonded_dirs.push(dir);
                total_stability += 1;
                
                if entity.index() < other_e.index() {
                    gizmos.line(pos, *other_pos, Color::srgba(1.0, 0.3, 0.3, 0.6));
                }
            } else if dist < CARBON_BOND_DIST {
                // Repulsão de sobreposição (Pauli)
                force -= dir * (CARBON_BOND_DIST - dist) * 10000.0;
            }
        }

        // --- FORÇA ANGULAR DE 120 GRAUS (A Chave do Grafeno) ---
        if bonded_dirs.len() >= 2 {
            for i in 0..bonded_dirs.len() {
                for j in (i+1)..bonded_dirs.len() {
                    let v1 = bonded_dirs[i];
                    let v2 = bonded_dirs[j];
                    let angle = v1.dot(v2).acos();
                    let target_angle = 120.0 * (PI / 180.0);
                    let diff = target_angle - angle;
                    
                    let correction = v1.cross(v2).cross(v1) * diff * ANGULAR_STIFFNESS;
                    force += correction;
                }
            }
        }

        // --- DINÂMICA TÉRMICA ---
        let mut rng = rand::rng();
        use rand::Rng;
        let jitter = Vec3::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)) * avg_temp;
        force += jitter;

        // --- MOVIMENTO ---
        let accel = force / atom.mass;
        atom.velocity = (atom.velocity + accel * dt) * 0.8; 
        transform.translation += atom.velocity * dt;
        
        transform.translation.y *= 0.9; // Plano 2D

        // --- VISUAL (NÚCLEO E ELÉTRONS) ---
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            let hue = (bonded_dirs.len() as f32 / 3.0) * 120.0;
            mat.base_color = Color::hsla(hue, 0.8, 0.5, 1.0);
            mat.emissive = LinearRgba::from(mat.base_color) * (avg_temp * 0.1);
        }
        
        // FIX: Extract radius before the mutable borrow loop
        let r_limit = a_radius / 8.0;
        let core_pos = transform.translation;

        for electron in atom.electrons.iter_mut() {
            let tangent = electron.rel_pos.cross(electron.orbit_axis).normalize_or_zero();
            electron.rel_pos += (tangent * 30.0 - electron.rel_pos.normalize() * 10.0) * dt;
            if electron.rel_pos.length() > r_limit {
                electron.rel_pos = electron.rel_pos.normalize() * r_limit;
            }
            gizmos.sphere(core_pos + electron.rel_pos, 0.4, Color::srgb(0.0, 1.0, 1.0));
        }
    }

    for mut scientist in scientists.iter_mut() {
        scientist.tunnel_stability = (total_stability / 2).min(1000);
    }
}

fn main_camera_system(
    mut query: Query<(&mut Transform, &mut MainCamera)>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<bevy::input::mouse::MouseMotion>,
    mut scroll: EventReader<bevy::input::mouse::MouseWheel>,
) {
    let Ok((mut transform, mut cam)) = query.single_mut() else { return };
    let mut delta = Vec2::ZERO;
    for ev in motion.read() { delta += ev.delta; }
    let scroll_y = scroll.read().map(|ev| ev.y).sum::<f32>();

    if mouse.pressed(MouseButton::Right) {
        cam.yaw -= delta.x * 0.005;
        cam.pitch = (cam.pitch - delta.y * 0.005).clamp(-1.5, 1.5);
    }
    cam.dist = (cam.dist - scroll_y * 60.0).clamp(50.0, 2000.0);

    let rot = Quat::from_rotation_y(cam.yaw) * Quat::from_rotation_x(cam.pitch);
    transform.translation = rot * Vec3::new(0.0, 50.0, cam.dist);
    transform.look_at(Vec3::ZERO, Vec3::Y);
}
