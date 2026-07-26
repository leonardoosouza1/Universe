//! INSPEÇÃO + VISTA (Leonardo 2026-07-08): clicar num ser mostra HP, energia, dopamina, cortisol e GRUPO.
//! Também abre a câmera e remove o "fog" (que fazia o mapa sumir) no modo sobrevivência.
use bevy::prelude::*;
use std::sync::atomic::Ordering;
use crate::AppState;
use crate::simulation::Racer;
use crate::survival::Forager;

fn survival_on() -> bool { crate::world::SURVIVAL.load(Ordering::Relaxed) }

#[derive(Resource, Default)]
pub struct SelectedBeing(pub Option<Entity>);

#[derive(Component)]
struct InspectRoot;
#[derive(Component)]
struct InspectText;

pub struct InspectPlugin;
impl Plugin for InspectPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SelectedBeing>()
           .add_systems(Update, (survival_view, pick_system, inspect_panel)
                .run_if(in_state(AppState::Simulation))
                .run_if(survival_on));
    }
}

/// Uma vez: tira o fog (mapa somindo) e abre a câmera pra ver mais longe. Gráfico básico/leve.
fn survival_view(
    mut commands: Commands,
    mut done: Local<bool>,
    cam_q: Query<Entity, With<Camera3d>>,
    light_q: Query<Entity, With<DirectionalLight>>,
    mut race: ResMut<crate::evolution::RaceSettings>,
) {
    if *done { return; }
    let mut any = false;
    for e in cam_q.iter() {
        commands.entity(e)
            .remove::<bevy::pbr::DistanceFog>()
            .remove::<bevy::pbr::VolumetricFog>();
        any = true;
    }
    for e in light_q.iter() {
        commands.entity(e).remove::<bevy::pbr::VolumetricLight>();
    }
    if any {
        race.camera_distance = 130.0; // vista aberta (>80 = top-down), sem ser longe demais
        *done = true;
    }
}

/// Clique esquerdo → seleciona o ser mais próximo do raio da câmera.
fn pick_system(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cam_q: Query<(&Camera, &GlobalTransform)>,
    beings: Query<(Entity, &Transform), With<Forager>>,
    mut selected: ResMut<SelectedBeing>,
) {
    if !buttons.just_pressed(MouseButton::Left) { return; }
    let Ok(window) = windows.single() else { return; };
    let Some(cursor) = window.cursor_position() else { return; };
    let Ok((cam, cam_tf)) = cam_q.single() else { return; };
    let Ok(ray) = cam.viewport_to_world(cam_tf, cursor) else { return; };
    let dir = *ray.direction;
    let mut best: Option<(Entity, f32)> = None;
    for (e, t) in beings.iter() {
        let to = t.translation - ray.origin;
        let proj = to.dot(dir);
        if proj < 0.0 { continue; }
        let closest = ray.origin + dir * proj;
        if closest.distance(t.translation) < 6.0 {
            if best.map_or(true, |(_, bp)| proj < bp) { best = Some((e, proj)); }
        }
    }
    if let Some((e, _)) = best { selected.0 = Some(e); }
}

/// Painel (canto sup. direito): mostra o estado do ser selecionado ao vivo.
fn inspect_panel(
    mut commands: Commands,
    selected: Res<SelectedBeing>,
    beings: Query<(&Forager, &Racer)>,
    mut text_q: Query<&mut Text, With<InspectText>>,
    root_q: Query<(), With<InspectRoot>>,
) {
    if root_q.is_empty() {
        commands.spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0), right: Val::Px(12.0),
                width: Val::Px(270.0), padding: UiRect::all(Val::Px(10.0)),
                flex_direction: FlexDirection::Column, ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.06, 0.12, 0.72)),
            InspectRoot,
        )).with_children(|p| {
            p.spawn((Text::new("🔍 INSPECIONAR SER"),
                TextFont { font_size: 17.0, ..default() }, TextColor(Color::srgb(0.4, 1.0, 0.9))));
            p.spawn((Text::new("clique num ser pra ver\nvida · dopamina · cortisol · grupo"),
                TextFont { font_size: 13.0, ..default() }, TextColor(Color::srgb(0.85, 0.9, 0.95)), InspectText));
        });
        return;
    }
    let Ok(mut text) = text_q.single_mut() else { return; };
    match selected.0.and_then(|e| beings.get(e).ok()) {
        Some((f, r)) => {
            let bars = |v: f32| { let n = ((v * 10.0).round() as usize).min(10); "█".repeat(n) };
            let estado = if r.is_dead { "† MORTO" }
                else if f.cortisol > 0.5 { "🏃 FUGINDO" }
                else if f.busy > 0.0 && f.target.is_some() { "🍽 COMENDO" }
                else if f.busy > 0.0 { "💧 BEBENDO" }
                else if f.energy < 70.0 { "🔎 procurando comida" }
                else { "🚶 andando" };
            text.0 = format!(
                "GRUPO/manada: {}\nEstado: {}\nVida (HP):  {:>3.0}  {}\nEnergia:    {:>3.0}  {}\nSede:       {:>3.0}  {}\nDopamina:  {:.2}  {}\nCortisol:  {:.2}  {}\nIdade: {:.0}s",
                f.group, estado,
                r.hp, bars(r.hp / 100.0),
                f.energy, bars(f.energy / 100.0),
                f.thirst, bars(f.thirst / 100.0),
                f.dopamine, bars(f.dopamine),
                f.cortisol, bars(f.cortisol),
                f.age,
            );
        }
        None => {
            text.0 = "clique num ser pra ver\nvida · dopamina · cortisol · grupo".to_string();
        }
    }
}
