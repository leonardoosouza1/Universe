//! PONTE survival↔inteligência DENTRO do engine (Leonardo 2026-07-08).
//! O mundo-semente validado (bytebrain: 768B/região, germinação fiel) vira ÁRVORES DE CONHECIMENTO
//! plantadas ao longo da pista. Quando um SER chega perto, a semente GERMINA (o fato aparece), a árvore
//! ACENDE e o ser ganha HP — sobreviver = colher conhecimento. É a ponte dos 4 lotes, visível no engine
//! nativo do race (mesma UI, muda o mundo). O conteúdo é germinado offline (assets/world_knowledge.json).
use bevy::prelude::*;
use serde::Deserialize;
use crate::{SimulationEntity, AppState};
use crate::simulation::Racer;
use crate::world::VoxelWorldResource;

#[derive(Deserialize, Clone)]
pub struct KnowledgeRegion {
    pub regiao: String,
    pub fato: String,
    #[serde(default)] pub fiel: bool,
}
#[derive(Deserialize)]
struct KnowledgeFile { regioes: Vec<KnowledgeRegion> }

/// Marca a árvore de conhecimento. `glow` = material da copa, aceso ao germinar.
#[derive(Component)]
pub struct KnowledgeTree {
    pub regiao: String,
    pub fato: String,
    pub learned: bool,
    pub glow: Handle<StandardMaterial>,
}

/// Códice global do que já foi germinado no mundo (pra HUD/telemetria).
#[derive(Resource, Default)]
pub struct Codex { pub aprendidos: Vec<(String, String)> }

#[derive(Resource, Default)]
struct KnowledgeSpawned(bool);

/// Marca o texto do HUD que lista os fatos germinados (atualizado ao vivo).
#[derive(Component)]
struct KnowledgeHud;

pub struct KnowledgePlugin;
impl Plugin for KnowledgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Codex>()
           .init_resource::<KnowledgeSpawned>()
           .add_systems(Update, (spawn_knowledge_world, germinate_on_approach, update_knowledge_hud)
               .run_if(in_state(AppState::Simulation)));
    }
}

fn load_knowledge() -> Vec<KnowledgeRegion> {
    for p in ["assets/world_knowledge.json",
              "/home/leonardo/projects/LLM/Universe/assets/world_knowledge.json"] {
        if let Ok(s) = std::fs::read_to_string(p) {
            if let Ok(f) = serde_json::from_str::<KnowledgeFile>(&s) { return f.regioes; }
        }
    }
    Vec::new()
}

/// Planta as árvores UMA vez, quando o mundo (waypoints) já existe. Espalha ao longo da pista.
fn spawn_knowledge_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    world: Res<VoxelWorldResource>,
    mut spawned: ResMut<KnowledgeSpawned>,
    mut codex: ResMut<Codex>,
    trees_q: Query<(), With<KnowledgeTree>>,
    hud_q: Query<(), With<KnowledgeHud>>,
) {
    // já plantado E as árvores ainda existem → nada a fazer. Se um reset de geração as apagou, replanta.
    if spawned.0 && !trees_q.is_empty() { return; }
    let wps = &world.0.waypoints_world;
    if wps.is_empty() { return; }                     // espera o mundo gerar
    let regioes = load_knowledge();
    if regioes.is_empty() { spawned.0 = true; return; }
    codex.aprendidos.clear();                          // novo ciclo de forrageio de conhecimento
    let n = regioes.len();
    let trunk_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.22, 0.10), perceptual_roughness: 0.95, ..default()
    });
    for (i, reg) in regioes.iter().enumerate() {
        // espalha nos primeiros ~25% da pista (trilha de conhecimento que os seres alcançam cedo)
        let span = (wps.len() * 25) / 100;
        let wi = (3 + i * span.max(1) / n) % wps.len();
        let wp = wps[wi];
        let nxt = wps[(wi + 1) % wps.len()];
        let dir = (nxt - wp).normalize_or_zero();
        let perp = Vec2::new(-dir.y, dir.x);
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };   // alterna os lados p/ o ser passar perto de todas
        let base = wp + perp * 5.0 * side;                 // 5m ao lado da linha
        let (x, z) = (base.x, base.y);
        // copa com material ÚNICO (pra acender ao germinar)
        let glow = materials.add(StandardMaterial {
            base_color: Color::srgb(0.16, 0.40, 0.16),
            emissive: LinearRgba::new(0.0, 0.12, 0.0, 1.0).into(),   // brilho fraco = ainda não germinado
            perceptual_roughness: 0.85, ..default()
        });
        // tronco (leva o componente KnowledgeTree)
        commands.spawn((
            Mesh3d(meshes.add(Cylinder::new(0.6, 7.0))),
            MeshMaterial3d(trunk_mat.clone()),
            Transform::from_xyz(x, 3.5, z),
            KnowledgeTree { regiao: reg.regiao.clone(), fato: reg.fato.clone(), learned: false, glow: glow.clone() },
            SimulationEntity,
        ));
        // copa
        commands.spawn((
            Mesh3d(meshes.add(Cone::new(3.2, 8.0))),
            MeshMaterial3d(glow),
            Transform::from_xyz(x, 9.0, z),
            SimulationEntity,
        ));
    }
    spawned.0 = true;
    println!("🌱 Mundo-semente: {} árvores de conhecimento plantadas ao longo da pista (germinam ao serem visitadas).", n);

    // HUD (uma vez): lista os fatos germinados ao vivo
    if hud_q.is_empty() {
        commands.spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(12.0), left: Val::Px(12.0),
                width: Val::Px(440.0), padding: UiRect::all(Val::Px(8.0)),
                flex_direction: FlexDirection::Column, ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.06, 0.02, 0.55)),
        )).with_children(|p| {
            p.spawn((Text::new("🌱 CONHECIMENTO GERMINADO  (mundo-semente → seres)"),
                TextFont { font_size: 15.0, ..default() }, TextColor(Color::srgb(0.45, 1.0, 0.55))));
            p.spawn((Text::new("(nenhum ainda — os seres precisam visitar as árvores de conhecimento)"),
                TextFont { font_size: 13.0, ..default() }, TextColor(Color::srgb(0.85, 0.95, 0.85)), KnowledgeHud));
        });
    }
}

/// Atualiza o HUD com os fatos germinados assim que o Códice muda.
fn update_knowledge_hud(codex: Res<Codex>, mut q: Query<&mut Text, With<KnowledgeHud>>) {
    if !codex.is_changed() { return; }
    let Ok(mut text) = q.single_mut() else { return; };
    if codex.aprendidos.is_empty() {
        text.0 = "(nenhum ainda — os seres precisam visitar as árvores de conhecimento)".to_string();
        return;
    }
    let mut s = String::new();
    for (reg, fato) in &codex.aprendidos {
        s.push_str(&format!("✓ {}: \"{}\"\n", reg, fato));
    }
    s.push_str(&format!("— {}/8 regiões germinadas", codex.aprendidos.len()));
    text.0 = s;
}

/// Cada frame: se um ser vivo chega perto de uma árvore não-germinada, ela GERMINA (acende + fato + HP).
fn germinate_on_approach(
    mut racers: Query<(&Transform, &mut Racer), Without<KnowledgeTree>>,
    mut trees: Query<(&Transform, &mut KnowledgeTree)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut codex: ResMut<Codex>,
) {
    for (tt, mut tree) in trees.iter_mut() {
        if tree.learned { continue; }
        let tp = Vec2::new(tt.translation.x, tt.translation.z);
        for (rt, mut racer) in racers.iter_mut() {
            if racer.is_dead { continue; }
            let rp = Vec2::new(rt.translation.x, rt.translation.z);
            if rp.distance(tp) < 13.0 {
                tree.learned = true;
                racer.hp = (racer.hp + 20.0).min(100.0);
                if let Some(m) = materials.get_mut(&tree.glow) {
                    m.emissive = LinearRgba::new(0.15, 3.5, 0.30, 1.0).into();   // ACENDE verde
                    m.base_color = Color::srgb(0.30, 0.85, 0.35);
                }
                codex.aprendidos.push((tree.regiao.clone(), tree.fato.clone()));
                println!("🌳 [{}] germinou → \"{}\"  · {} aprendeu (HP {:.0}, {}/{} regiões)",
                    tree.regiao, tree.fato, racer.name, racer.hp, codex.aprendidos.len(), trees_hint());
                break;
            }
        }
    }
}

fn trees_hint() -> usize { 8 }
