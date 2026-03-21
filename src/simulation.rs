use bevy::prelude::*;
use crate::engine::SovereignEngine;
use crate::world::{VoxelWorld, VoxelWorldResource};
use crate::resonance::ResonantNode;
use crate::periodic_table::{get_element_by_name};

const SCALE: f32 = 10.0;
const COULOMB_K: f32 = 500.0;
const PAULI_K: f32 = 2000.0;

#[derive(Clone, Copy, Default)]
pub struct SubPart {
    pub rel_pos: Vec3,
    pub velocity: Vec3,
    pub phase: f32,
}

#[derive(Component)]
pub struct Atom {
    pub element: String,
    pub mass: f32,
    pub charge: f32, // Carga do Núcleo
    pub radius: f32,
    pub electronegativity: f32,
    pub velocity: Vec3,
    pub electrons: Vec<SubPart>,
    pub energy_level: f32, // Condução de energia
}

#[derive(Component)]
pub struct ScientistIA {
    pub engine: SovereignEngine,
    pub sync_bit: i8,
    pub tunnel_stability: i32,
    pub vortex_expansion: i32,
    pub brain_resonator: ResonantNode,
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
    let Ok(mut scientist) = scientists.single_mut() else { return };
    let dt = time.delta_secs().min(0.02); // Estabilidade física

    let mut atom_positions: Vec<(Entity, Vec3, f32, f32, String)> = atoms.iter()
        .map(|(e, a, t, _)| (e, t.translation, a.charge, a.radius, a.element.clone()))
        .collect();

    let mut total_stability = 0;

    for (entity, mut atom, mut transform, mat_handle) in atoms.iter_mut() {
        let mut force = Vec3::ZERO;
        let pos = transform.translation;

        // 1. INTERAÇÃO INTER-ATÔMICA (Ligações e Repulsão)
        for (other_e, other_pos, other_charge, other_radius, other_el) in &atom_positions {
            if entity == *other_e { continue; }

            let dir = *other_pos - pos;
            let dist = dir.length();
            if dist < 0.1 { continue; }
            let unit_dir = dir.normalize();

            // Coulomb (Atração/Repulsão de longo alcance)
            let f_coulomb = (atom.charge * other_charge * COULOMB_K) / (dist * dist);
            force -= unit_dir * f_coulomb;

            // Repulsão de Pauli / Van der Waals (Curto alcance)
            let min_dist = (atom.radius + other_radius) / 50.0;
            if dist < min_dist {
                let overlap = min_dist - dist;
                force -= unit_dir * overlap * PAULI_K;
            }

            // CONDUÇÃO DE ENERGIA (Especialmente Prata/Silver)
            if (atom.element == "Silver" || other_el == "Silver") && dist < min_dist * 1.5 {
                // Transferência de energia (frequência/fase) entre condutores
                if atom.energy_level > 0.1 {
                    gizmos.line(pos, *other_pos, Color::srgba(0.8, 0.9, 1.0, 0.3));
                }
            }
        }

        // 2. DINÂMICA DOS ELÉTRONS (SubParts orbitais)
        let num_electrons = atom.electrons.len();
        for i in 0..num_electrons {
            let electron = &mut atom.electrons[i];
            
            // Força Central (Núcleo puxa elétron)
            let dist_to_core = electron.rel_pos.length();
            let pull = (atom.charge * 100.0) / (dist_to_core.max(1.0).powi(2));
            let dir_to_core = -electron.rel_pos.normalize_or_zero();
            
            // Força Centrífuga Simulada (Órbita)
            let orbit_axis = Vec3::Y;
            let tangent = electron.rel_pos.cross(orbit_axis).normalize_or_zero();
            
            electron.velocity += (dir_to_core * pull + tangent * 2.0) * dt;
            electron.rel_pos += electron.velocity * dt;
            
            // Manter na "nuvem" eletrônica baseada no raio atômico
            let target_r = atom.radius / 10.0;
            if electron.rel_pos.length() > target_r * 1.5 {
                electron.rel_pos = electron.rel_pos.normalize() * target_r;
            }

            // Renderizar Elétrons como pequenos gizmos
            gizmos.sphere(pos + electron.rel_pos, 0.5, Color::srgb(0.0, 1.0, 1.0));
        }

        // 3. MOVIMENTO DO NÚCLEO
        let accel = force / atom.mass;
        atom.velocity = (atom.velocity + accel * dt) * 0.95; // Amortecimento
        transform.translation += atom.velocity * dt;

        // 4. ATUALIZAÇÃO VISUAL
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            let el_data = get_element_by_name(&atom.element);
            mat.base_color = el_data.color;
            mat.emissive = LinearRgba::from(el_data.color) * (1.0 + atom.energy_level);
        }
    }
    
    // O Cientista monitora a "Estabilidade da Malha"
    scientist.tunnel_stability = (total_stability / 10).min(1000);
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
    cam.dist = (cam.dist - scroll_y * 60.0).clamp(100.0, 5000.0);

    let rot = Quat::from_rotation_y(cam.yaw) * Quat::from_rotation_x(cam.pitch);
    transform.translation = rot * Vec3::new(0.0, 200.0, cam.dist);
    transform.look_at(Vec3::ZERO, Vec3::Y);
}
