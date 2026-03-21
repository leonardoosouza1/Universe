use bevy::prelude::*;
use crate::engine::SovereignEngine;

const SCALE: i32 = 1000;
const IRON_LATTICE: i32 = 286;
const IRON_BOND: i32 = 248;

#[derive(Clone, Copy, Default)]
pub struct SubPart {
    pub x: i32, pub y: i32, pub z: i32,
    pub vx: i32, pub vy: i32, pub vz: i32,
    pub phase: i8,
}

#[derive(Component)]
pub struct Atom {
    pub name: String,
    pub ix: i32, pub iy: i32, pub iz: i32, 
    pub vx: i32, pub vy: i32, pub vz: i32,
    pub valence_electrons: Vec<SubPart>,
    pub is_locked: bool,
}

#[derive(Component)]
pub struct ScientistIA {
    pub engine: SovereignEngine,
    pub universe_id: usize,
    pub center: Vec3,
    pub sync_bit: i8,
    pub integrity: i32,
    pub discovery_hardness: f32,
    pub discovery_conductive: f32,
    pub discovery_purity: f32,
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
        app.add_systems(Update, (iron_bar_synthesis, main_camera_system));
    }
}

fn iron_bar_synthesis(
    mut scientists: Query<&mut ScientistIA>,
    mut atoms: Query<(Entity, &mut Atom, &mut Transform, &mut MeshMaterial3d<StandardMaterial>)>,
    mut gizmos: Gizmos,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Ok(mut scientist) = scientists.get_single_mut() else { return };
    
    let mut sensory = [0i8; 128];
    sensory[0] = (scientist.integrity / 10).clamp(0, 127) as i8;
    let intent = scientist.engine.get_intent(&sensory);
    scientist.sync_bit = (intent[0] % 127) as i8;

    let atom_data: Vec<(Entity, i32, i32, i32, i8, bool)> = atoms.iter()
        .map(|(e, a, _, _)| (e, a.ix, a.iy, a.iz, a.valence_electrons[0].phase, a.is_locked))
        .collect();

    let mut total_sync = 0;

    for (entity, mut atom, mut transform, mat_handle) in atoms.iter_mut() {
        if atom.is_locked {
            if let Some(mat) = materials.get_mut(&mat_handle.0) {
                mat.base_color = Color::srgb(0.5, 0.5, 0.6);
                mat.emissive = LinearRgba::from(mat.base_color) * 30.0;
            }
            continue;
        }

        let mut fx = 0i32; let mut fy = 0i32; let mut fz = 0i32;
        let mut local_bonds = 0;

        let target_x = (atom.ix / IRON_LATTICE) * IRON_LATTICE;
        let target_y = (atom.iy / IRON_LATTICE) * IRON_LATTICE;
        let target_z = (atom.iz / IRON_LATTICE) * IRON_LATTICE;
        
        fx += (target_x.clamp(-1200, 1200) - atom.ix) * 2;
        fy += (target_y.clamp(-300, 300) - atom.iy) * 4;
        fz += (target_z.clamp(-300, 300) - atom.iz) * 4;

        for (other_e, ox, oy, oz, o_phase, _) in &atom_data {
            if entity == *other_e { continue; }
            let dx = ox - atom.ix;
            let dy = oy - atom.iy;
            let dz = oz - atom.iz;
            let dist_sq = (dx*dx + dy*dy + dz*dz) / SCALE;

            if dist_sq < 1200 {
                if (atom.valence_electrons[0].phase as i32 - *o_phase as i32).abs() < 3 {
                    let error = dist_sq - (IRON_BOND * IRON_BOND / SCALE);
                    fx += dx * error / 1500;
                    fy += dy * error / 1500;
                    fz += dz * error / 1500;
                    local_bonds += 1;
                    if entity.index() < other_e.index() {
                        let p2 = Vec3::new(*ox as f32 / 10.0, *oy as f32 / 10.0, *oz as f32 / 10.0);
                        gizmos.line(transform.translation, p2, Color::srgba(0.0, 1.0, 1.0, 0.4));
                    }
                }
            }
        }

        atom.vx = (atom.vx + fx) - (atom.vx >> 3);
        atom.vy = (atom.vy + fy) - (atom.vy >> 3);
        atom.vz = (atom.vz + fz) - (atom.vz >> 3);
        atom.ix += atom.vx / 10;
        atom.iy += atom.vy / 10;
        atom.iz += atom.vz / 10;

        if local_bonds >= 6 && atom.vx.abs() < 3 { atom.is_locked = true; }

        transform.translation = Vec3::new(atom.ix as f32 / 10.0, atom.iy as f32 / 10.0, atom.iz as f32 / 10.0);
        gizmos.sphere(transform.translation, 5.0, Color::srgb(1.0, 0.2, 0.1));
        total_sync += local_bonds;
    }

    scientist.integrity = (total_sync / 10).min(1000);
    scientist.discovery_hardness = (scientist.integrity as f32 / 1000.0) * 211.0;
    scientist.discovery_conductive = (scientist.integrity as f32 / 1000.0) * 0.9;
    scientist.discovery_purity = scientist.integrity as f32 / 1000.0;
}

fn main_camera_system(
    mut query: Query<(&mut Transform, &mut MainCamera)>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<bevy::input::mouse::MouseMotion>,
    mut scroll: EventReader<bevy::input::mouse::MouseWheel>,
) {
    let Ok((mut transform, mut cam)) = query.get_single_mut() else { return };
    let mut delta = Vec2::ZERO;
    for ev in motion.read() { delta += ev.delta; }
    let scroll_y = scroll.read().map(|ev| ev.y).sum::<f32>();

    if mouse.pressed(MouseButton::Right) {
        cam.yaw -= delta.x * 0.005;
        cam.pitch = (cam.pitch - delta.y * 0.005).clamp(-1.5, 1.5);
    }
    cam.dist = (cam.dist - scroll_y * 40.0).clamp(50.0, 3000.0);

    let rot = Quat::from_rotation_y(cam.yaw) * Quat::from_rotation_x(cam.pitch);
    transform.translation = rot * Vec3::Z * cam.dist;
    transform.look_at(Vec3::ZERO, Vec3::Y);
}
