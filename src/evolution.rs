use bevy::prelude::*;
use crate::simulation::{Atom, ScientistIA, SubPart};
use crate::world::{VoxelWorld, VoxelWorldResource};
use crate::resonance::ResonantNode;
use crate::periodic_table::{get_element_by_name};
use rand::prelude::*;

#[derive(Resource, Default)]
pub struct ElementMaterials {
    pub map: std::collections::HashMap<String, Handle<StandardMaterial>>,
}

#[derive(Resource)]
pub struct RaceSettings {
    pub generation: usize,
    pub best_weights: Option<Vec<i8>>,
    pub best_scale: i32,
    pub best_recipe: String,
    pub best_fitness: i32,
    pub should_reset: bool,
    pub fossils: Vec<AtomFossil>,
}

#[derive(Clone)]
pub struct AtomFossil {
    pub name: String,
    pub ix: i32, pub iy: i32, pub iz: i32,
}

#[derive(Resource)]
pub struct DevSettings {
    pub mutation_rate: f32,
    pub wave_amplitude: f32,
    pub flow_influence: f32,
    pub time_scale: f32,
}

impl Default for DevSettings {
    fn default() -> Self {
        Self { mutation_rate: 0.05, wave_amplitude: 40.0, flow_influence: 0.5, time_scale: 1.0 }
    }
}

pub struct EvolutionPlugin;
impl Plugin for EvolutionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ElementMaterials>();
        app.add_systems(Startup, initial_spawn);
        app.add_systems(Update, evolution_pipeline);
    }
}

fn initial_spawn(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut el_materials: ResMut<ElementMaterials>,
    race: Res<RaceSettings>,
    dev: Res<DevSettings>,
    world_res: Res<VoxelWorldResource>,
) {
    spawn_multiverse(&mut commands, &mut meshes, &mut materials, &mut el_materials, &race, &dev, &world_res.0);
}

pub fn spawn_multiverse(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    _el_materials: &mut ElementMaterials,
    race: &RaceSettings,
    _dev: &DevSettings,
    _world: &VoxelWorld,
) {
    let mut rng = rand::rng();
    let core_mesh = meshes.add(Sphere::new(2.5));

    // SPAWN DE 50 MODELOS (AGENTES)
    for _ in 0..50 {
        let mut engine = crate::engine::SovereignEngine::new();
        if let Some(ref best) = race.best_weights {
            engine.instinct_layer.weights = best.clone();
            engine.instinct_layer.mutate(0.05);
        }

        commands.spawn(ScientistIA {
            engine,
            sync_bit: 0,
            tunnel_stability: 0,
            temperature: 20.0, // Começa quente para fundir
            brain_resonator: ResonantNode::new(1.0),
            target_atom: None,
        });
    }

    // SPAWN DE ÁTOMOS DE CARBONO (GRAFENO)
    // Spawning em um plano 2D para facilitar a formação da folha
    let el = get_element_by_name("Carbon");
    
    for _ in 0..150 {
        let pos = Vec3::new(
            rng.random_range(-80.0..80.0),
            rng.random_range(-5.0..5.0), // Quase plano
            rng.random_range(-80.0..80.0)
        );

        let mut electrons = Vec::new();
        for _ in 0..3 { // sp2 tem 3 elétrons de valência principais na folha
            let orbit_axis = Vec3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0)
            ).normalize_or_zero();

            electrons.push(SubPart {
                rel_pos: Vec3::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)).normalize() * (el.radius / 10.0),
                velocity: Vec3::ZERO,
                orbit_axis,
            });
        }

        commands.spawn((
            Atom { 
                element: el.name.to_string(),
                mass: el.mass,
                charge: el.atomic_number as f32,
                radius: el.radius,
                valence: 3, // sp2 hibridização para Grafeno
                velocity: Vec3::ZERO,
                electrons,
            },
            Mesh3d(core_mesh.clone()), MeshMaterial3d(materials.add(StandardMaterial::default())),
            Transform::from_translation(pos),
        ));
    }
}

fn evolution_pipeline(
    mut commands: Commands,
    mut race: ResMut<RaceSettings>,
    dev: Res<DevSettings>,
    mut el_materials: ResMut<ElementMaterials>,
    world_res: Res<VoxelWorldResource>,
    query: Query<(Entity, &Atom)>,
    scientists: Query<(Entity, &ScientistIA)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if race.should_reset {
        for (e, _) in query.iter() { commands.entity(e).despawn(); }
        for (e, _) in scientists.iter() { commands.entity(e).despawn(); }
        race.generation += 1; race.should_reset = false;
        spawn_multiverse(&mut commands, &mut meshes, &mut materials, &mut el_materials, &race, &dev, &world_res.0);
    }
}
