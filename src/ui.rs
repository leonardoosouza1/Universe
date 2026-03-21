use bevy::prelude::*;
use crate::simulation::Racer;
use crate::evolution::RaceSettings;

#[derive(Component)]
pub struct LeaderboardUI;

#[derive(Component)]
pub struct RacerButton { pub racer_entity: Entity }

#[derive(Component)]
pub struct RestartButton;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (update_hud, handle_ui_interactions));
    }
}

fn update_hud(
    mut commands: Commands,
    query: Query<(Entity, Option<&Children>), With<LeaderboardUI>>,
    racer_query: Query<(Entity, &Racer)>,
    race: Res<RaceSettings>,
) {
    // Bevy 0.16.1 returns Result for single()
    let (ui_entity, children) = query.single().unwrap();
    if let Some(children) = children {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }

    let mut sorted: Vec<(Entity, &Racer)> = racer_query.iter().collect();
    sorted.sort_by(|a, b| b.1.distance.partial_cmp(&a.1.distance).unwrap());

    commands.entity(ui_entity).with_children(|p| {
        p.spawn((Text::new(format!("IARA F1 ENGINE | GEN {}", race.generation)), TextFont { font_size: 18.0, ..default() }, TextColor(Color::srgb(1.0, 0.8, 0.0))));
        
        // Restart Button
        p.spawn((
            Button,
            Node { width: Val::Px(200.0), height: Val::Px(25.0), margin: UiRect::vertical(Val::Px(5.0)), justify_content: JustifyContent::Center, align_items: AlignItems::Center, ..default() },
            BackgroundColor(Color::srgb(0.8, 0.2, 0.2)),
            RestartButton,
        )).with_children(|btn| {
            btn.spawn((Text::new("RESTART G"), TextFont { font_size: 12.0, ..default() }, TextColor(Color::WHITE)));
        });

        for (entity, r) in sorted.iter().take(12) {
            let color = if r.is_dead { Color::srgb(0.5, 0.5, 0.5) } else { Color::WHITE };
            p.spawn((
                Button,
                Node { width: Val::Px(220.0), margin: UiRect::all(Val::Px(1.0)), padding: UiRect::all(Val::Px(2.0)), ..default() },
                BackgroundColor(Color::srgba(0.2, 0.2, 0.2, 0.5)),
            )).insert(RacerButton { racer_entity: *entity })
            .with_children(|btn| {
                btn.spawn((
                    Text::new(format!("{} | G:{} | V:{:.0}", r.name, r.gear, r.velocity)),
                    TextFont { font_size: 13.0, ..default() },
                    TextColor(color),
                ));
            });
        }
    });
}

fn handle_ui_interactions(
    mut race: ResMut<RaceSettings>,
    query: Query<(&Interaction, Option<&RacerButton>, Option<&RestartButton>), (Changed<Interaction>, With<Button>)>,
) {
    for (interaction, racer_btn, restart_btn) in &query {
        if *interaction == Interaction::Pressed {
            if let Some(r_btn) = racer_btn {
                race.selected_racer = Some(r_btn.racer_entity);
            }
            if restart_btn.is_some() {
                race.should_reset = true;
            }
        }
    }
}
