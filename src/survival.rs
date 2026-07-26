//! CONTROLADOR LEVE DE SOBREVIVÊNCIA (Leonardo 2026-07-08).
//! Modo `--vida`: motor de corrida + física de carro DESLIGADOS. Os seres viram CRIATURAS com máquina de
//! estados: procurar → chegar → PARAR e ANIMAR (comendo/bebendo) → a comida SOME (indisponível pros outros)
//! e regenera depois. Fome/sede/medo. Leve → roda com muitos seres. Reusa a entidade Racer (Transform, hp).
use bevy::prelude::*;
use std::sync::atomic::Ordering;
use crate::AppState;
use crate::simulation::Racer;
use crate::world::{VoxelWorldResource, VoxelWorld, SURVIVAL};

const PI: f32 = std::f32::consts::PI;
const TAU: f32 = std::f32::consts::TAU;
const LOD_RADIUS: f32 = 300.0;
const EAT_SECS: f32 = 1.4;     // tempo parado comendo
const DRINK_SECS: f32 = 1.2;   // tempo parado bebendo

#[derive(Component)]
pub struct Forager {
    pub heading: f32,
    pub energy: f32,    // 0..100 (fome)
    pub thirst: f32,    // 0..100 (sede) — 100 = hidratado
    pub age: f32,
    pub rng: u32,
    pub dopamine: f32,  // 0..1 — sobe ao comer/beber
    pub cortisol: f32,  // 0..1 — sobe com predador perto
    pub group: u8,      // manada (clustering)
    pub busy: f32,      // >0 = PARADO comendo/bebendo (anima)
    pub target: Option<Entity>, // comida sendo comida (None enquanto busy = bebendo)
}

/// Planta comestível. `available=false` = já pega por alguém / regenerando (some da tela).
#[derive(Component)]
pub struct Food { pub available: bool, pub regrow: f32 }

/// Lago — beber mata a sede (não é consumido).
#[derive(Component)]
pub struct Water;

/// PREDADOR — caça a presa; presa foge.
#[derive(Component)]
pub struct Predator { pub heading: f32, pub rng: u32 }

#[derive(Resource, Default)]
struct FoodSpawned(bool);

fn survival_on() -> bool { SURVIVAL.load(Ordering::Relaxed) }
fn nextf(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*s >> 8) & 0xFFFFFF) as f32 / 16_777_216.0
}
fn terrain_y(world: &VoxelWorld, x: f32, z: f32) -> f32 {
    let cx = (x / world.cell_size).round() as i32;
    let cz = (z / world.cell_size).round() as i32;
    if cx >= 0 && cx < world.size as i32 && cz >= 0 && cz < world.size as i32 {
        world.height_map[cz as usize * world.size + cx as usize]
    } else { 0.0 }
}

pub struct SurvivalPlugin;
impl Plugin for SurvivalPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FoodSpawned>()
           .add_systems(Update, (spawn_world, survival_init, cluster_system, predator_step, regrow_food, forage_step).chain()
                .run_if(in_state(AppState::Simulation))
                .run_if(survival_on));
    }
}

/// Espalha COMIDA (plantas), ÁGUA (lagos) e PREDADORES pelo mundo aberto (uma vez).
fn spawn_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    world: Res<VoxelWorldResource>,
    mut spawned: ResMut<FoodSpawned>,
) {
    if spawned.0 { return; }
    let world_m = world.0.size as f32 * world.0.cell_size;
    let stem_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.35, 0.26, 0.12), perceptual_roughness: 0.95, ..default() });
    let leaf_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.30, 0.85, 0.25),
        emissive: LinearRgba::new(0.1, 0.6, 0.05, 1.0).into(), perceptual_roughness: 0.6, ..default() });
    let berry_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.75, 0.15),
        emissive: LinearRgba::new(1.2, 0.9, 0.1, 1.0).into(), perceptual_roughness: 0.4, ..default() });
    let mut s: u32 = 0x1234_9ABC;
    let n_food = 90;
    for _ in 0..n_food {
        let x = 50.0 + nextf(&mut s) * (world_m - 100.0);
        let z = 50.0 + nextf(&mut s) * (world_m - 100.0);
        let y = terrain_y(&world.0, x, z);
        commands.spawn((Mesh3d(meshes.add(Cylinder::new(0.35, 3.0))), MeshMaterial3d(stem_mat.clone()),
            Transform::from_xyz(x, y + 1.5, z), crate::SimulationEntity));
        commands.spawn((Mesh3d(meshes.add(Sphere::new(1.7))), MeshMaterial3d(leaf_mat.clone()),
            Transform::from_xyz(x, y + 3.4, z), crate::SimulationEntity));
        commands.spawn((Mesh3d(meshes.add(Sphere::new(0.75))), MeshMaterial3d(berry_mat.clone()),
            Transform::from_xyz(x, y + 4.6, z), Food { available: true, regrow: 0.0 }, crate::SimulationEntity));
    }

    // ÁGUA
    let water_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.10, 0.45, 0.75), emissive: LinearRgba::new(0.02, 0.10, 0.20, 1.0).into(),
        perceptual_roughness: 0.08, reflectance: 0.6, ..default() });
    let n_water = 14;
    for _ in 0..n_water {
        let x = 60.0 + nextf(&mut s) * (world_m - 120.0);
        let z = 60.0 + nextf(&mut s) * (world_m - 120.0);
        let y = terrain_y(&world.0, x, z);
        let r = 12.0 + nextf(&mut s) * 22.0;
        commands.spawn((Mesh3d(meshes.add(Cylinder::new(r, 0.4))), MeshMaterial3d(water_mat.clone()),
            Transform::from_xyz(x, y + 0.15, z), Water, crate::SimulationEntity));
    }

    // PREDADORES
    let pred_body = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.08, 0.06), perceptual_roughness: 0.7, ..default() });
    let pred_eye = materials.add(StandardMaterial { base_color: Color::srgb(1.0, 0.5, 0.0), emissive: LinearRgba::new(2.5, 0.6, 0.0, 1.0).into(), ..default() });
    let n_pred = 8;
    for _ in 0..n_pred {
        let x = 80.0 + nextf(&mut s) * (world_m - 160.0);
        let z = 80.0 + nextf(&mut s) * (world_m - 160.0);
        let y = terrain_y(&world.0, x, z);
        let heading = nextf(&mut s) * TAU;
        commands.spawn((Mesh3d(meshes.add(Sphere::new(2.0))), MeshMaterial3d(pred_body.clone()),
            Transform::from_xyz(x, y + 1.6, z), Predator { heading, rng: s.wrapping_mul(2654435761).wrapping_add(17) }, crate::SimulationEntity))
        .with_children(|c| {
            c.spawn((Mesh3d(meshes.add(Sphere::new(1.2))), MeshMaterial3d(pred_body.clone()), Transform::from_xyz(2.4, 0.6, 0.0)));
            c.spawn((Mesh3d(meshes.add(Cone::new(0.7, 1.6))), MeshMaterial3d(pred_body.clone()), Transform::from_xyz(3.6, 0.5, 0.0).with_rotation(Quat::from_rotation_z(-PI/2.0))));
            c.spawn((Mesh3d(meshes.add(Sphere::new(0.28))), MeshMaterial3d(pred_eye.clone()), Transform::from_xyz(3.0, 1.0, 0.5)));
            c.spawn((Mesh3d(meshes.add(Sphere::new(0.28))), MeshMaterial3d(pred_eye.clone()), Transform::from_xyz(3.0, 1.0, -0.5)));
        });
    }
    spawned.0 = true;
    println!("🌿 {} plantas · 💧 {} lagos · 🐺 {} predadores.", n_food, n_water, n_pred);
}

fn survival_init(
    mut commands: Commands,
    world: Res<VoxelWorldResource>,
    mut q: Query<(Entity, &mut Transform), (With<Racer>, Without<Forager>)>,
    mut seed: Local<u32>,
) {
    if *seed == 0 { *seed = 0x9E37_79B9; }
    let world_m = world.0.size as f32 * world.0.cell_size;
    for (e, mut t) in q.iter_mut() {
        let x = 60.0 + nextf(&mut *seed) * (world_m - 120.0);
        let z = 60.0 + nextf(&mut *seed) * (world_m - 120.0);
        t.translation.x = x; t.translation.z = z; t.translation.y = terrain_y(&world.0, x, z) + 0.9;
        let heading = nextf(&mut *seed) * TAU;
        commands.entity(e).insert(Forager {
            heading, energy: 60.0 + nextf(&mut *seed) * 30.0, thirst: 60.0 + nextf(&mut *seed) * 30.0,
            age: 0.0, rng: seed.wrapping_add(e.index()).wrapping_mul(2654435761),
            dopamine: 0.0, cortisol: 0.0, group: 0, busy: 0.0, target: None,
        });
    }
}

/// Comida escondida (pega) regenera depois de um tempo e reaparece.
fn regrow_food(time: Res<Time>, mut q: Query<(&mut Food, &mut Visibility)>) {
    let dt = time.delta_secs().min(0.05);
    for (mut food, mut vis) in q.iter_mut() {
        if !food.available {
            food.regrow -= dt;
            if food.regrow <= 0.0 { food.available = true; *vis = Visibility::Inherited; }
        }
    }
}

fn predator_step(
    time: Res<Time>, world: Res<VoxelWorldResource>,
    prey_q: Query<&Transform, (With<Forager>, Without<Predator>)>,
    mut preds: Query<(&mut Transform, &mut Predator)>,
) {
    let dt = time.delta_secs().min(0.05);
    let world_m = world.0.size as f32 * world.0.cell_size;
    let prey: Vec<Vec2> = prey_q.iter().take(800).map(|t| Vec2::new(t.translation.x, t.translation.z)).collect();
    for (mut t, mut p) in preds.iter_mut() {
        let pos = Vec2::new(t.translation.x, t.translation.z);
        let (mut nd, mut n) = (f32::MAX, pos);
        for &q in &prey { let d = pos.distance(q); if d < nd { nd = d; n = q; } }
        if nd < 120.0 {
            let dir = n - pos; let desired = dir.y.atan2(dir.x);
            let mut dth = desired - p.heading;
            while dth > PI { dth -= TAU; } while dth < -PI { dth += TAU; }
            p.heading += dth.clamp(-2.0 * dt, 2.0 * dt);
        } else { p.heading += (nextf(&mut p.rng) - 0.5) * 0.6 * dt; }
        let speed = 6.0;
        let mut nx = t.translation.x + p.heading.cos() * speed * dt;
        let mut nz = t.translation.z + p.heading.sin() * speed * dt;
        if nx < 40.0 || nx > world_m - 40.0 { p.heading = PI - p.heading; nx = t.translation.x; }
        if nz < 40.0 || nz > world_m - 40.0 { p.heading = -p.heading; nz = t.translation.z; }
        t.translation.x = nx; t.translation.z = nz; t.translation.y = terrain_y(&world.0, nx, nz) + 1.6;
        t.rotation = Quat::from_rotation_y(-p.heading);
    }
}

fn cluster_system(
    time: Res<Time>, mut timer: Local<f32>, mut centroids: Local<Vec<Vec2>>,
    mut q: Query<(&Transform, &mut Forager)>,
) {
    *timer -= time.delta_secs();
    if *timer > 0.0 { return; }
    *timer = 1.0;
    const K: usize = 6;
    let pts: Vec<Vec2> = q.iter().map(|(t, _)| Vec2::new(t.translation.x, t.translation.z)).collect();
    if pts.len() < K { return; }
    if centroids.len() != K { *centroids = (0..K).map(|i| pts[i * pts.len() / K]).collect(); }
    for _ in 0..4 {
        let mut sum = vec![Vec2::ZERO; K]; let mut cnt = vec![0u32; K];
        for &p in &pts {
            let (mut bi, mut bd) = (0usize, f32::MAX);
            for (i, &c) in centroids.iter().enumerate() { let d = p.distance_squared(c); if d < bd { bd = d; bi = i; } }
            sum[bi] += p; cnt[bi] += 1;
        }
        for i in 0..K { if cnt[i] > 0 { centroids[i] = sum[i] / cnt[i] as f32; } }
    }
    for (t, mut f) in q.iter_mut() {
        let p = Vec2::new(t.translation.x, t.translation.z);
        let (mut bi, mut bd) = (0u8, f32::MAX);
        for (i, &c) in centroids.iter().enumerate() { let d = p.distance_squared(c); if d < bd { bd = d; bi = i as u8; } }
        f.group = bi;
    }
}

/// MÁQUINA DE ESTADOS: fugir > (parado comendo/bebendo) > procurar comida/água > vaguear.
fn forage_step(
    time: Res<Time>,
    world: Res<VoxelWorldResource>,
    cam_q: Query<&Transform, (With<Camera3d>, Without<Forager>, Without<Food>, Without<Water>, Without<Predator>)>,
    waters_q: Query<&Transform, (With<Water>, Without<Forager>, Without<Food>, Without<Predator>)>,
    preds_q: Query<&Transform, (With<Predator>, Without<Forager>, Without<Food>, Without<Water>)>,
    mut foods: Query<(Entity, &Transform, &mut Food, &mut Visibility), (Without<Forager>, Without<Water>, Without<Predator>)>,
    mut beings: Query<(&mut Transform, &mut Forager, &mut Racer), (Without<Food>, Without<Water>, Without<Predator>)>,
) {
    let dt = time.delta_secs().min(0.05);
    let cam = cam_q.iter().next().map(|t| Vec2::new(t.translation.x, t.translation.z)).unwrap_or(Vec2::ZERO);
    let world_m = world.0.size as f32 * world.0.cell_size;
    let preds: Vec<Vec2> = preds_q.iter().map(|t| Vec2::new(t.translation.x, t.translation.z)).collect();
    let waters: Vec<Vec2> = waters_q.iter().map(|t| Vec2::new(t.translation.x, t.translation.z)).collect();
    // comida disponível (entity + pos)
    let avail: Vec<(Entity, Vec2)> = foods.iter()
        .filter(|(_, _, f, _)| f.available)
        .map(|(e, t, _, _)| (e, Vec2::new(t.translation.x, t.translation.z)))
        .collect();
    let mut claimed: Vec<Entity> = Vec::new();

    for (mut t, mut f, mut racer) in beings.iter_mut() {
        if racer.is_dead { t.scale = Vec3::ONE; continue; }
        f.age += dt;
        f.energy -= 3.0 * dt;
        f.thirst -= 1.6 * dt;
        f.dopamine = (f.dopamine - 0.5 * dt).max(0.0);
        let pos = Vec2::new(t.translation.x, t.translation.z);
        let far = pos.distance(cam) > LOD_RADIUS;

        // predador mais próximo → medo
        let (mut pd, mut pnear) = (f32::MAX, pos);
        for &pp in &preds { let d = pos.distance(pp); if d < pd { pd = d; pnear = pp; } }
        let threat = if pd < 90.0 { 1.0 - pd / 90.0 } else { 0.0 };
        f.cortisol = (f.cortisol * 0.97 + threat * dt * 3.0).clamp(0.0, 1.0);
        let fleeing = pd < 70.0;

        // ---- ESTADO: comendo/bebendo (PARADO + animação) ----
        if f.busy > 0.0 && !fleeing {
            f.busy -= dt;
            // animação: pulsa a escala (mordidas/goles) e abaixa um tiquinho
            let bob = ((EAT_SECS - f.busy) * 14.0).sin() * 0.18;
            t.scale = Vec3::splat(1.0 + bob.abs() * 0.4);
            t.translation.y = terrain_y(&world.0, pos.x, pos.y) + 0.9 - bob.max(0.0) * 0.5;
            if f.busy <= 0.0 {
                // TERMINOU: se comida, ela SOME (indisponível) e regenera; se água, mata sede
                if let Some(fe) = f.target {
                    if let Ok((_, _, mut food, mut vis)) = foods.get_mut(fe) {
                        food.available = false; food.regrow = 12.0; *vis = Visibility::Hidden;
                    }
                    f.energy = (f.energy + 45.0).min(100.0);
                } else {
                    f.thirst = 100.0;
                }
                f.dopamine = (f.dopamine + 0.6).min(1.0);
                f.target = None; t.scale = Vec3::ONE;
            }
            // fome/medo/morte continuam contando mesmo parado
            if f.energy <= 0.0 { f.energy = 0.0; racer.hp -= 18.0 * dt; }
            if racer.hp <= 0.0 { racer.hp = 0.0; racer.is_dead = true; }
            continue;
        }
        t.scale = Vec3::ONE;

        // ---- decide alvo ----
        let hungry = f.energy < 70.0;
        let thirsty = f.thirst < 45.0;
        // acha comida disponível mais próxima (não reivindicada neste frame)
        let (mut fd, mut fpos, mut fe) = (f32::MAX, pos, None);
        if hungry {
            for &(e, p) in &avail {
                if claimed.contains(&e) { continue; }
                let d = pos.distance(p);
                if d < fd { fd = d; fpos = p; fe = Some(e); }
            }
        }
        // água mais próxima
        let (mut wd, mut wpos) = (f32::MAX, pos);
        if thirsty { for &wp in &waters { let d = pos.distance(wp); if d < wd { wd = d; wpos = wp; } } }

        // rumo alvo: medo > (sede muito baixa) > fome > vaguear
        let (mut goal, mut has_goal) = (pos, false);
        if fleeing {
            goal = pos + (pos - pnear); has_goal = true;
        } else if thirsty && (f.thirst < 25.0 || !hungry) && wd < f32::MAX {
            goal = wpos; has_goal = true;
        } else if hungry && fe.is_some() {
            goal = fpos; has_goal = true;
        }

        if has_goal {
            let dir = goal - pos;
            let desired = dir.y.atan2(dir.x);
            let mut dth = desired - f.heading;
            while dth > PI { dth -= TAU; } while dth < -PI { dth += TAU; }
            let rate = if fleeing { 5.0 } else if far { 1.2 } else { 3.0 };
            f.heading += dth.clamp(-rate * dt, rate * dt);
        } else {
            f.heading += (nextf(&mut f.rng) - 0.5) * if far { 0.4 } else { 1.6 } * dt;
        }

        // ---- chegou no alvo? come/bebe (entra em busy, PARA) ----
        if !fleeing {
            if hungry && fd < 5.0 { if let Some(e) = fe { f.busy = EAT_SECS; f.target = Some(e); claimed.push(e); } }
            else if thirsty && wd < 24.0 { f.busy = DRINK_SECS; f.target = None; }
        }
        if f.busy > 0.0 { continue; } // vai animar no próximo frame

        // ---- movimento cinemático ----
        let speed = if fleeing { 11.0 } else { 8.0 };
        let mut nx = t.translation.x + f.heading.cos() * speed * dt;
        let mut nz = t.translation.z + f.heading.sin() * speed * dt;
        if nx < 40.0 || nx > world_m - 40.0 { f.heading = PI - f.heading; nx = t.translation.x; }
        if nz < 40.0 || nz > world_m - 40.0 { f.heading = -f.heading; nz = t.translation.z; }
        t.translation.x = nx; t.translation.z = nz;
        t.translation.y = terrain_y(&world.0, nx, nz) + 0.9;
        t.rotation = Quat::from_rotation_y(-f.heading);

        // predador pegou → mordida
        if pd < 5.0 { racer.hp -= 30.0 * dt; f.energy = (f.energy - 25.0 * dt).max(0.0); }
        if f.energy <= 0.0 { f.energy = 0.0; racer.hp -= 18.0 * dt; }
        else if racer.hp < 100.0 && pd >= 5.0 { racer.hp = (racer.hp + 6.0 * dt).min(100.0); }
        if racer.hp <= 0.0 { racer.hp = 0.0; racer.is_dead = true; }
    }
}
