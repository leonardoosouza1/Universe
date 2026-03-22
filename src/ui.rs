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
    racer_query: Query<(Entity, &Racer, &crate::physics::EngineComponent)>,
    race: Res<RaceSettings>,
    dev: Res<DevSettings>,
) {
    let Ok((ui_entity, children)) = query.single() else { return; };
    if let Some(children) = children {
        for child in children.iter() {
            if let Ok(mut e) = commands.get_entity(child) {
                e.despawn();
            }
        }
    }

    let mut sorted: Vec<(Entity, &Racer, &crate::physics::EngineComponent)> = racer_query.iter().collect();
    sorted.sort_by(|a, b| {
        let score_a = if a.1.distance <= 0.0 { a.1.distance * 100.0 } else { a.1.distance.powi(2) / a.1.engine.current_time.max(0.1) };
        let score_b = if b.1.distance <= 0.0 { b.1.distance * 100.0 } else { b.1.distance.powi(2) / b.1.engine.current_time.max(0.1) };
        score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
    });

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
            for (i, (entity, r, _)) in sorted.iter().take(10).enumerate() {
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

        // --- TELEMETRY (ESQUERDA INFERIOR) ---
        // Se o usuário não clicou em ninguém, espiona o Líder (1º colocado)
        let selected = race.selected_racer.unwrap_or_else(|| {
            sorted.first().map(|(e, _, _)| *e).unwrap_or(Entity::PLACEHOLDER)
        });
        
        if let Ok((_, r, eng_comp)) = racer_query.get(selected) {
            p.spawn(Node {
                    position_type: PositionType::Absolute,
                    bottom: Val::Px(10.0),
                    left: Val::Px(10.0),
                    width: Val::Px(350.0),
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(Val::Px(15.0)),
                    ..default()
                }).insert(BackgroundColor(Color::srgba(0.05, 0.08, 0.12, 0.95)))
                .with_children(|panel| {
                    panel.spawn((Text::new(format!("CÓRTEX CLONE: {}", r.name)), TextFont { font_size: 16.0, ..default() }, TextColor(Color::srgb(0.0, 1.0, 0.8))));
                    
                    // --- NOVOS DADOS NUMÉRICOS (TELEMETRIA VEICULAR) ---
                    panel.spawn(Node { flex_direction: FlexDirection::Row, justify_content: JustifyContent::SpaceBetween, margin: UiRect::vertical(Val::Px(4.0)), ..default() })
                        .with_children(|stats_row| {
                            let speed = r.amplitude * 3.6;
                            let tire_color = if r.tire_temp < 0.3 { Color::srgb(0.0, 0.5, 1.0) } else if r.tire_temp > 0.7 { Color::srgb(1.0, 0.0, 0.0) } else { Color::srgb(0.0, 1.0, 0.0) };
                            
                            stats_row.spawn(Node { flex_direction: FlexDirection::Column, ..default() }).with_children(|col1| {
                                col1.spawn((Text::new(format!("VEL: {:.0} km/h", speed)), TextFont { font_size: 14.0, ..default() }, TextColor(Color::WHITE)));
                                col1.spawn((Text::new(format!("MARCHA: {}", eng_comp.current_gear)), TextFont { font_size: 14.0, ..default() }, TextColor(Color::WHITE)));
                                col1.spawn((Text::new(format!("RPM: {:.0}", eng_comp.current_rpm)), TextFont { font_size: 14.0, ..default() }, TextColor(if eng_comp.current_rpm > eng_comp.max_rpm * 0.9 { Color::srgb(1.0, 0.0, 0.0) } else { Color::WHITE })));
                            });
                            
                            stats_row.spawn(Node { flex_direction: FlexDirection::Column, ..default() }).with_children(|col2| {
                                col2.spawn((Text::new(format!("PNEU: {:.0}°C", r.tire_temp * 100.0)), TextFont { font_size: 14.0, ..default() }, TextColor(tire_color)));
                                col2.spawn((Text::new(format!("MTR PWR: {:.0}%", r.engine_power * 100.0)), TextFont { font_size: 14.0, ..default() }, TextColor(if r.engine_power < 0.5 { Color::srgb(1.0, 0.5, 0.0) } else { Color::WHITE })));
                                col2.spawn((Text::new(format!("GRAMA: {:.1}s", r.grass_time)), TextFont { font_size: 14.0, ..default() }, TextColor(if r.grass_time > 0.0 { Color::srgb(1.0, 0.0, 0.0) } else { Color::WHITE })));
                            });
                        });
                        
                    // Arrays de Vida e Pedais
                    let pedals = [
                        ("HP (Integridade Estrutural)", (r.hp / 100.0).clamp(0.0, 1.0), Color::srgb(0.0, 0.6, 1.0)),
                        ("THROTTLE (Aceleração)", r.memory[0], Color::srgb(0.0, 1.0, 0.0)), 
                        ("BRAKE (Freio)", r.memory[2], Color::srgb(1.0, 0.0, 0.0))
                    ];
                    for (label, val, col) in pedals {
                        panel.spawn(Node { width: Val::Percent(100.0), height: Val::Px(14.0), margin: UiRect::top(Val::Px(6.0)), ..default() }).with_children(|bar| {
                            bar.spawn((Text::new(label), TextFont { font_size: 11.0, ..default() })).insert(Node { width: Val::Px(140.0), ..default() });
                            bar.spawn(Node { width: Val::Percent(val * 100.0), height: Val::Percent(100.0), ..default() }).insert(BackgroundColor(col));
                        });
                    }
                    
                    // Volante (Steer)
                    panel.spawn(Node { width: Val::Percent(100.0), height: Val::Px(14.0), margin: UiRect::top(Val::Px(6.0)), ..default() }).with_children(|bar| {
                        bar.spawn((Text::new("STEERING (Volante)"), TextFont { font_size: 11.0, ..default() })).insert(Node { width: Val::Px(140.0), ..default() });
                        let steer_val = (r.memory[1] + 1.0) / 2.0; // -1 to 1 => 0 to 1
                        bar.spawn(Node { width: Val::Percent(100.0), height: Val::Percent(100.0), ..default() }).insert(BackgroundColor(Color::srgb(0.2, 0.2, 0.2))).with_children(|bg| {
                            bg.spawn(Node { 
                                width: Val::Px(4.0), height: Val::Percent(100.0), 
                                left: Val::Percent(steer_val * 100.0), position_type: PositionType::Absolute,
                                ..default() 
                            }).insert(BackgroundColor(Color::WHITE));
                        });
                    });

                    // Retina Visual
                    panel.spawn((Text::new("CÓRTEX (L | R) Muro Sup | Chão Inf"), TextFont { font_size: 11.0, ..default() }, TextColor(Color::WHITE)))
                         .insert(Node { margin: UiRect::top(Val::Px(12.0)), ..default() });
                         
                    panel.spawn(Node { width: Val::Percent(100.0), height: Val::Px(30.0), flex_direction: FlexDirection::Column, margin: UiRect::top(Val::Px(4.0)), ..default() }).with_children(|retina_col| {
                        // Linha 1: Horizonte (Muros), Linha 0: Chão (Asfalto/Grama)
                        for row in (0..2).rev() { 
                            retina_col.spawn(Node { width: Val::Percent(100.0), height: Val::Percent(50.0), flex_direction: FlexDirection::Row, ..default() }).with_children(|retina_row| {
                                for x in 0..32 {
                                    let i = (row * 32) + x;
                                    let brightness = ((r.visual_cortex[i] as f32 + 128.0) / 255.0).clamp(0.0, 1.0);
                                    let color = Color::srgb(brightness, brightness, brightness);
                                    retina_row.spawn(Node {
                                        width: Val::Percent(100.0 / 32.0),
                                        height: Val::Percent(100.0),
                                        ..default()
                                    }).insert(BackgroundColor(color));
                                }
                            });
                        }
                    });
                });
            }
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
