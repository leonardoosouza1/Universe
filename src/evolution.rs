use bevy::prelude::*;
use crate::simulation::{Atom, ScientistIA, SubPart};
use crate::world::{VoxelWorld, VoxelWorldResource};
use crate::resonance::ResonantNode;
use crate::periodic_table::{get_element_by_name, PERIODIC_TABLE};
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
        app.add_systems(Update, evolution_pipeline);
    }
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
    let core_mesh = meshes.add(Sphere::new(2.0));

    let mut engine = crate::engine::SovereignEngine::new();
    if let Some(ref best) = race.best_weights {
        engine.instinct_layer.weights = best.clone();
        engine.instinct_layer.mutate(0.05);
    }

    commands.spawn(ScientistIA {
        engine,
        sync_bit: 0,
        tunnel_stability: 0,
        vortex_expansion: 400,
        brain_resonator: ResonantNode::new(1.0),
    });

    // SPAWN DE MATERIAIS: Carbono e Prata para testar condutividade e malha
    let elements_to_spawn = ["Carbon", "Silver"];
    
    for i in 0..150 {
        let el_name = elements_to_spawn[i % 2];
        let el = get_element_by_name(el_name);
        
        let ix = rng.random_range(-150..150) as f32;
        let iy = rng.random_range(-150..150) as f32;
        let iz = rng.random_range(-150..150) as f32;

        let mut electrons = Vec::new();
        for _ in 0..el.valence_electrons {
            electrons.push(SubPart {
                rel_pos: Vec3::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)).normalize() * (el.radius / 10.0),
                velocity: Vec3::ZERO,
                phase: rng.random_range(0.0..6.28),
            });
        }

        commands.spawn((
            Atom { 
                element: el.name.to_string(),
                mass: el.mass,
                charge: el.valence_electrons as f32, // Simplificação: carga efetiva
                radius: el.radius,
                electronegativity: el.electronegativity,
                velocity: Vec3::ZERO,
                electrons,
                energy_level: if el.name == "Silver" { 1.0 } else { 0.0 },
            },
            Mesh3d(core_mesh.clone()), MeshMaterial3d(materials.add(StandardMaterial::default())),
            Transform::from_translation(Vec3::new(ix, iy, iz)),
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
