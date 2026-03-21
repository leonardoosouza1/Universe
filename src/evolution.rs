use bevy::prelude::*;
use crate::simulation::{Atom, ScientistIA, SubPart};
use crate::world::{VoxelWorld, VoxelWorldResource};
use crate::periodic_table::get_element_by_name;
use rand::prelude::*;

#[derive(Resource, Default)]
pub struct ElementMaterials {
    pub map: std::collections::HashMap<String, Handle<StandardMaterial>>,
}

#[derive(Resource)]
pub struct RaceSettings {
    pub generation: usize,
    pub best_weights: Option<Vec<i8>>,
    pub best_scale: f32,
    pub best_recipe: String,
    pub best_fitness: f32,
    pub should_reset: bool,
    pub fossils: Vec<AtomFossil>,
}

#[derive(Clone)]
pub struct AtomFossil {
    pub name: String,
    pub position: Vec3,
    pub phase: f32,
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
    let core_mesh = meshes.add(Sphere::new(4.0));

    // UM ÚNICO CIENTISTA
    let mut engine = crate::engine::SovereignEngine::new();
    if let Some(ref best) = race.best_weights {
        engine.instinct_layer.weights = best.clone();
        engine.instinct_layer.mutate(0.05);
    }

    commands.spawn(ScientistIA {
        engine,
        universe_id: 0,
        sync_bit: 0,
        integrity: 0,
        center: Vec3::ZERO,
        discovery_hardness: 0.0,
        discovery_conductive: 0.0,
        discovery_purity: 0.0,
    });

    // 300 ÁTOMOS DE FERRO NO CENTRO
    for _ in 0..300 {
        let ix = rng.random_range(-1000..1000);
        let iy = rng.random_range(-400..400);
        let iz = rng.random_range(-400..400);

        let mut valence = Vec::new();
        valence.push(SubPart { phase: rng.random_range(0..127), ..default() });

        commands.spawn((
            Atom { 
                name: "Iron".to_string(),
                ix, iy, iz,
                vx: 0, vy: 0, vz: 0,
                valence_electrons: valence,
                is_locked: false,
            },
            Mesh3d(core_mesh.clone()), MeshMaterial3d(materials.add(StandardMaterial::default())),
            Transform::from_translation(Vec3::ZERO),
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
