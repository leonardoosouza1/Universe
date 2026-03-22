use bevy::prelude::*;
use crate::simulation::Racer;
use crate::evolution::{RaceSettings, DevSettings};

#[derive(Component)]
pub struct LeaderboardUI;

#[derive(Component)]
pub struct RacerButton { pub racer_entity: Entity }

#[derive(Component)]
pub struct DevButton { pub action: DevAction }

#[derive(Clone, Copy)]
pub enum DevAction {
    ResetGen,
    MutUp, MutDown,
    AmpUp, AmpDown,
    FlowUp, FlowDown,
}

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
    dev: Res<DevSettings>,
) {
    let Ok((ui_entity, children)) = query.get_single() else { return; };
    if let Some(children) = children {
        for child in children.iter() {
            if let Some(e) = commands.get_entity(*child) {
                e.despawn_recursive();
            }
        }
    }


    let mut sorted: Vec<(Entity, &Racer)> = racer_query.iter().collect();
    sorted.sort_by(|a, b| b.1.distance.partial_cmp(&a.1.distance).unwrap());

    commands.entity(ui_entity).with_children(|p| {
        // --- PAINEL DEV (ESQUERDA) ---
        p.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(10.0),
                top: Val::Px(10.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.85)),
        )).with_children(|dev_panel| {
            dev_panel.spawn((Text::new("IARA DEV CTRL"), TextFont { font_size: 18.0, ..default() }, TextColor(Color::srgb(0.0, 1.0, 1.0))));
            dev_panel.spawn((Text::new(format!("GEN: {}", race.generation)), TextFont { font_size: 14.0, ..default() }, TextColor(Color::WHITE)));
            
            // Botão Reset
            dev_panel.spawn((
                Button,
                Node { width: Val::Px(180.0), height: Val::Px(30.0), margin: UiRect::vertical(Val::Px(5.0)), justify_content: JustifyContent::Center, align_items: AlignItems::Center, ..default() },
                BackgroundColor(Color::srgb(0.7, 0.1, 0.1)),
                DevButton { action: DevAction::ResetGen },
            )).with_children(|b| { b.spawn((Text::new("HARD RESET GEN"), TextFont { font_size: 12.0, ..default() }, TextColor(Color::WHITE))); });

            // Controles de Parâmetros
            let params = [
                ("MUTATION", dev.mutation_rate, DevAction::MutDown, DevAction::MutUp),
                ("SPEED/AMP", dev.wave_amplitude, DevAction::AmpDown, DevAction::AmpUp),
                ("FLOW PULL", dev.flow_influence, DevAction::FlowDown, DevAction::FlowUp),
            ];

            for (label, val, down, up) in params {
                dev_panel.spawn(Node { margin: UiRect::vertical(Val::Px(5.0)), flex_direction: FlexDirection::Column, ..default() })
                .with_children(|c| {
                    c.spawn((Text::new(format!("{}: {:.2}", label, val)), TextFont { font_size: 12.0, ..default() }, TextColor(Color::srgb(0.8, 0.8, 0.8))));
                    c.spawn(Node { flex_direction: FlexDirection::Row, ..default() }).with_children(|row| {
                        for (l, act) in [("-", down), ("+", up)] {
                            row.spawn((
                                Button,
                                Node { width: Val::Px(85.0), height: Val::Px(25.0), margin: UiRect::horizontal(Val::Px(2.0)), justify_content: JustifyContent::Center, align_items: AlignItems::Center, ..default() },
                                BackgroundColor(Color::srgb(0.3, 0.3, 0.3)),
                                DevButton { action: act },
                            )).with_children(|b| { b.spawn((Text::new(l), TextFont { font_size: 12.0, ..default() }, TextColor(Color::WHITE))); });
                        }
                    });
                });
            }
        });

        // --- LEADERBOARD (DIREITA) ---
        p.spawn(Node {
            position_type: PositionType::Absolute,
            right: Val::Px(10.0),
            top: Val::Px(10.0),
            flex_direction: FlexDirection::Column,
            ..default()
        }).with_children(|list| {
            for (i, (entity, r)) in sorted.iter().take(10).enumerate() {
                let color = if r.is_dead { Color::srgb(0.4, 0.4, 0.4) } else { Color::WHITE };
                list.spawn((
                    Button,
                    Node { width: Val::Px(250.0), margin: UiRect::all(Val::Px(1.0)), padding: UiRect::all(Val::Px(4.0)), flex_direction: FlexDirection::Column, ..default() },
                    BackgroundColor(Color::srgba(0.1, 0.1, 0.2, 0.7)),
                )).insert(RacerButton { racer_entity: *entity })
                .with_children(|btn| {
                    btn.spawn((Text::new(format!("#{:02} {} | LAP:{}", i+1, r.name, r.total_laps)), TextFont { font_size: 13.0, ..default() }, TextColor(color)));
                    btn.spawn((Text::new(format!("{:.1} km/h", r.amplitude * 3.6)), TextFont { font_size: 11.0, ..default() }, TextColor(color)));
                    btn.spawn((Node { width: Val::Percent(r.fuel * 100.0), height: Val::Px(2.0), ..default() }, BackgroundColor(Color::srgb(0.0, 0.5, 1.0))));
                });
            }
        });
    });
}

fn handle_ui_interactions(
    mut race: ResMut<RaceSettings>,
    mut dev: ResMut<DevSettings>,
    query: Query<(&Interaction, Option<&RacerButton>, Option<&DevButton>), (Changed<Interaction>, With<Button>)>,
) {
    for (interaction, racer_btn, dev_btn) in &query {
        if *interaction == Interaction::Pressed {
            if let Some(r_btn) = racer_btn { 
                race.selected_racer = Some(r_btn.racer_entity); 
                race.is_free_cam = false;
            }
            if let Some(d_btn) = dev_btn {
                match d_btn.action {
                    DevAction::ResetGen => race.should_reset = true,
                    DevAction::MutUp => dev.mutation_rate = (dev.mutation_rate + 0.05).min(1.0),
                    DevAction::MutDown => dev.mutation_rate = (dev.mutation_rate - 0.05).max(0.0),
                    DevAction::AmpUp => dev.wave_amplitude += 5.0,
                    DevAction::AmpDown => dev.wave_amplitude = (dev.wave_amplitude - 5.0).max(10.0),
                    DevAction::FlowUp => dev.flow_influence += 0.1,
                    DevAction::FlowDown => dev.flow_influence = (dev.flow_influence - 0.1).max(0.0),
                }
            }
        }
    }
}
