use bevy::prelude::*;
use crate::simulation::{ScientistIA};
use crate::evolution::{RaceSettings};

#[derive(Component)]
pub struct HudRoot;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_ui);
        app.add_systems(Update, update_hud);
    }
}

fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        },
        HudRoot,
    ));
}

fn update_hud(
    mut commands: Commands,
    query: Query<(Entity, Option<&Children>), With<HudRoot>>,
    race: Res<RaceSettings>,
    scientist_query: Query<&ScientistIA>,
) {
    let Ok((ui_entity, children)) = query.single() else { return };
    if let Some(children) = children {
        for &child in children { commands.entity(child).despawn(); }
    }

    let num_agents = scientist_query.iter().count();
    let total_stability: f32 = scientist_query.iter().map(|s| s.tunnel_stability as f32).sum::<f32>() / num_agents.max(1) as f32;
    let avg_temp: f32 = scientist_query.iter().map(|s| s.temperature).sum::<f32>() / num_agents.max(1) as f32;

    commands.entity(ui_entity).with_children(|p| {
        p.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(30.0), top: Val::Px(30.0),
                width: Val::Px(350.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(20.0)),
                border: UiRect::all(Val::Px(2.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.05, 0.1, 0.9)),
            BorderColor(Color::srgb(0.0, 1.0, 0.5)),
        )).with_children(|panel| {
            panel.spawn((Text::new("IARA | GRAPHENE FORGE"), TextFont { font_size: 24.0, ..default() }, TextColor(Color::srgb(0.0, 1.0, 0.5))));
            panel.spawn((Text::new(format!("GEN: {}", race.generation)), TextFont { font_size: 14.0, ..default() }, TextColor(Color::srgb(0.7, 0.7, 0.8))));

            panel.spawn((
                Node { margin: UiRect::vertical(Val::Px(10.0)), height: Val::Px(1.0), width: Val::Percent(100.0), ..default() },
                BackgroundColor(Color::srgba(0.0, 1.0, 0.5, 0.3)),
            ));

            // STABILITY BAR
            let integrity = (total_stability / 5.0).clamp(0.0, 100.0);
            panel.spawn(Node { flex_direction: FlexDirection::Column, margin: UiRect::vertical(Val::Px(10.0)), ..default() }).with_children(|bar_cont| {
                bar_cont.spawn((Text::new(format!("HEXAGONAL INTEGRITY: {:.1}%", integrity)), TextFont { font_size: 14.0, ..default() }, TextColor(Color::WHITE)));
                bar_cont.spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(8.0),
                        margin: UiRect::top(Val::Px(4.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.05, 0.05, 0.1)),
                )).with_children(|inner| {
                    inner.spawn((
                        Node {
                            width: Val::Percent(integrity),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        BackgroundColor(Color::hsla( integrity * 1.2, 0.8, 0.5, 1.0)),
                    ));
                });
            });

            // THERMAL STATUS
            let status_text = if avg_temp > 15.0 { "FORGING MODE (HIGH HEAT)" } else { "CRYSTALLIZING (ANNEALING)" };
            panel.spawn((Text::new(status_text), TextFont { font_size: 16.0, ..default() }, TextColor(Color::srgb(1.0, 0.5, 0.0))));
            panel.spawn((Text::new(format!("CORE TEMP: {:.1} K", avg_temp * 10.0)), TextFont { font_size: 18.0, ..default() }, TextColor(Color::WHITE)));

            panel.spawn(Node { margin: UiRect::top(Val::Px(15.0)), ..default() });
            panel.spawn((Text::new("STATUS: SP2 HYBRIDIZATION ACTIVE"), TextFont { font_size: 12.0, ..default() }, TextColor(Color::srgba(0.0, 1.0, 0.5, 0.5))));
        });
    });
}
