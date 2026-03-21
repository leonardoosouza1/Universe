use bevy::prelude::*;
use crate::simulation::{ScientistIA, Atom};
use crate::evolution::{RaceSettings};

#[derive(Component)]
pub struct LeaderboardUI;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_hud);
    }
}

fn update_hud(
    mut commands: Commands,
    query: Query<(Entity, Option<&Children>), With<LeaderboardUI>>,
    scientist_query: Query<&ScientistIA>,
    atom_query: Query<&Atom>,
    race: Res<RaceSettings>,
) {
    let Ok((ui_entity, children)) = query.get_single() else { return };
    if let Some(children) = children { for child in children.iter() { commands.entity(child).despawn(); } }

    commands.entity(ui_entity).with_children(|p| {
        p.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(20.0), top: Val::Px(20.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(20.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.05, 0.1, 0.9)),
        )).with_children(|panel| {
            panel.spawn((Text::new("IARA IRON FORGE LAB"), TextFont { font_size: 28.0, ..default() }, TextColor(Color::srgb(0.8, 0.8, 0.9))));
            panel.spawn((Text::new(format!("BATCH GEN: {}", race.generation)), TextFont { font_size: 16.0, ..default() }, TextColor(Color::WHITE)));
            
            panel.spawn(Node { margin: UiRect::vertical(Val::Px(15.0)), ..default() });

            if let Some(s) = scientist_query.get_single().ok() {
                let locked_count = atom_query.iter().filter(|a| a.is_locked).count();
                let status = if locked_count > 250 { "IRON BAR SOLIDIFIED" } else { "FORGING..." };
                
                panel.spawn((Text::new(status), TextFont { font_size: 20.0, ..default() }, TextColor(Color::srgb(1.0, 0.5, 0.0))));
                panel.spawn((Text::new(format!("SOLID ATOMS: {} / 300", locked_count)), TextFont { font_size: 16.0, ..default() }, TextColor(Color::WHITE)));
                panel.spawn((Text::new(format!("TENSILE STRENGTH: {:.1} GPa", s.discovery_hardness)), TextFont { font_size: 16.0, ..default() }, TextColor(Color::srgb(0.5, 0.8, 1.0))));
                panel.spawn((Text::new(format!("CRYSTAL PURITY: {:.1}%", s.integrity as f32 / 10.0)), TextFont { font_size: 16.0, ..default() }, TextColor(Color::srgb(0.8, 1.0, 0.5))));
            }
        });
    });
}
