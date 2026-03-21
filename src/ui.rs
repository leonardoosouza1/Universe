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
    
    // Cleanup old UI elements
    if let Some(children) = children {
        for &child in children {
            commands.entity(child).despawn();
        }
    }

    commands.entity(ui_entity).with_children(|p| {
        p.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(20.0), top: Val::Px(20.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(20.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.05, 0.1, 0.95)),
        )).with_children(|panel| {
            panel.spawn((Text::new("IARA FIBONACCI VORTEX"), TextFont { font_size: 28.0, ..default() }, TextColor(Color::srgb(0.0, 1.0, 1.0))));
            panel.spawn((Text::new(format!("BATCH GEN: {}", race.generation)), TextFont { font_size: 16.0, ..default() }, TextColor(Color::WHITE)));

            panel.spawn(Node { margin: UiRect::vertical(Val::Px(15.0)), ..default() });

            if let Ok(s) = scientist_query.single() {
                let status = if s.tunnel_stability > 800 { "QUANTUM TUNNEL STABLE" } else { "STABILIZING VORTEX..." };

                panel.spawn((Text::new(status), TextFont { font_size: 20.0, ..default() }, TextColor(Color::srgb(1.0, 0.0, 1.0))));
                panel.spawn((Text::new(format!("TUNNEL STABILITY: {:.1}%", s.tunnel_stability as f32 / 10.0)), TextFont { font_size: 16.0, ..default() }, TextColor(Color::WHITE)));
                panel.spawn((Text::new("FIBONACCI TUNNEL: ACTIVE"), TextFont { font_size: 14.0, ..default() }, TextColor(Color::srgb(0.0, 1.0, 0.5))));
                panel.spawn((Text::new(format!("VORTEX EXPANSION: {}", s.vortex_expansion)), TextFont { font_size: 16.0, ..default() }, TextColor(Color::srgb(0.5, 0.8, 1.0))));
                panel.spawn((Text::new(format!("SYNC BIT: 0x{:02X}", s.sync_bit)), TextFont { font_size: 16.0, ..default() }, TextColor(Color::srgb(0.8, 1.0, 0.5))));
            }
        });
    });
}
