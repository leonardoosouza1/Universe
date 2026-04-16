use bevy::prelude::*;
use crate::config::{UniverseConfig, TrackProfile};
use crate::AppState;
use rand::Rng;

#[derive(Component)]
pub struct MenuEntity;

#[derive(Component)]
pub enum MenuButton {
    StartSimulation,
    AddTrack,
    DeleteTrack(usize),
    SelectTrack(usize),
    TogglePhysics(PhysicsToggle),
    SwarmUp,
    SwarmDown,
}

#[derive(Component)]
pub struct SwarmText;

#[derive(Component)]
pub struct PhysicsToggleIndicator(pub PhysicsToggle);

#[derive(Component)]
pub struct TrackIndicator(pub usize);

#[derive(Clone, Copy)]
pub enum PhysicsToggle {
    Stall,
    Thermal,
    Aero,
    Downforce,
    Tcs,
    Abs,
    Aqua,
    Wear,
}

pub struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(AppState::MainMenu), setup_menu)
           .add_systems(Update, (
               handle_menu_interactions,
               sync_menu_ui,
               handle_button_hover,
           ).run_if(in_state(AppState::MainMenu)))
           .add_systems(OnExit(AppState::MainMenu), cleanup_menu);
    }
}

fn setup_menu(mut commands: Commands, config: Res<UniverseConfig>) {
    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        BackgroundColor(Color::srgba(0.02, 0.02, 0.05, 1.0)),
        MenuEntity,
    )).with_children(|root| {
        // Título
        root.spawn((
            Text::new("UNIVERSE - LAB CONTROL"),
            TextFont { font_size: 40.0, ..default() },
            TextColor(Color::srgb(0.0, 1.0, 0.8)),
        ));

        root.spawn(Node {
            width: Val::Px(800.0),
            height: Val::Px(500.0),
            margin: UiRect::top(Val::Px(20.0)),
            flex_direction: FlexDirection::Row,
            ..default()
        }).with_children(|container| {
            // Coluna Esquerda: Tracks
            container.spawn(Node {
                width: Val::Percent(40.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            }).with_children(|left| {
                left.spawn((Text::new("TRACK MANAGER"), TextFont { font_size: 20.0, ..default() }, TextColor(Color::WHITE)));
                
                for (i, track) in config.tracks.iter().enumerate() {
                    let is_selected = i == config.selected_track_index;
                    let bg_color = if is_selected { Color::srgb(0.0, 0.4, 0.3) } else { Color::srgb(0.2, 0.2, 0.2) };
                    
                    left.spawn((
                        Button,
                        Node { width: Val::Percent(100.0), margin: UiRect::vertical(Val::Px(2.0)), padding: UiRect::all(Val::Px(5.0)), flex_direction: FlexDirection::Row, justify_content: JustifyContent::SpaceBetween, ..default() },
                        BackgroundColor(bg_color),
                        MenuButton::SelectTrack(i),
                        TrackIndicator(i),
                    )).with_children(|btn| {
                        btn.spawn((Text::new(&track.name), TextFont { font_size: 14.0, ..default() }));
                        if config.tracks.len() > 1 {
                            btn.spawn((
                                Button,
                                Node { width: Val::Px(20.0), height: Val::Px(20.0), ..default() },
                                BackgroundColor(Color::srgb(0.5, 0.1, 0.1)),
                                MenuButton::DeleteTrack(i),
                            )).with_children(|del| { del.spawn(Text::new("X")); });
                        }
                    });
                }
                
                left.spawn((
                    Button,
                    Node { width: Val::Percent(100.0), height: Val::Px(30.0), margin: UiRect::top(Val::Px(10.0)), justify_content: JustifyContent::Center, align_items: AlignItems::Center, ..default() },
                    BackgroundColor(Color::srgb(0.3, 0.3, 0.3)),
                    MenuButton::AddTrack,
                )).with_children(|b| { b.spawn(Text::new("+ NEW PROCEDURAL TRACK")); });
            });

            // Coluna Direita: Physics & Swarm
            container.spawn(Node {
                width: Val::Percent(60.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            }).with_children(|right| {
                right.spawn((Text::new("PHYSICS & AGENTS"), TextFont { font_size: 20.0, ..default() }, TextColor(Color::WHITE)));
                
                let p = &config.physics;
                let physics_options = [
                    ("Engine Stall", p.enable_stall, PhysicsToggle::Stall),
                    ("Thermal Simulation", p.enable_thermal, PhysicsToggle::Thermal),
                    ("Aerodynamic Drag", p.enable_aero_drag, PhysicsToggle::Aero),
                    ("Downforce", p.enable_downforce, PhysicsToggle::Downforce),
                    ("Traction Control (TCS)", p.enable_tcs, PhysicsToggle::Tcs),
                    ("ABS Braking", p.enable_abs, PhysicsToggle::Abs),
                    ("Aquaplaning", p.enable_aquaplaning, PhysicsToggle::Aqua),
                    ("Tire Wear", p.enable_tire_wear, PhysicsToggle::Wear),
                ];

                for (label, val, toggle) in physics_options {
                    right.spawn(Node { flex_direction: FlexDirection::Row, align_items: AlignItems::Center, margin: UiRect::vertical(Val::Px(2.0)), ..default() }).with_children(|row| {
                        row.spawn((
                            Button,
                            Node { width: Val::Px(20.0), height: Val::Px(20.0), margin: UiRect::right(Val::Px(10.0)), ..default() },
                            BackgroundColor(if val { Color::srgb(0.0, 1.0, 0.0) } else { Color::srgb(1.0, 0.0, 0.0) }),
                            MenuButton::TogglePhysics(toggle),
                            PhysicsToggleIndicator(toggle),
                        ));
                        row.spawn((Text::new(label), TextFont { font_size: 14.0, ..default() }));
                    });
                }

                // Swarm Count
                right.spawn(Node { margin: UiRect::top(Val::Px(20.0)), flex_direction: FlexDirection::Row, align_items: AlignItems::Center, ..default() }).with_children(|row| {
                    row.spawn((
                        Text::new(format!("SWARM SIZE: {}", config.swarm_count)), 
                        TextFont { font_size: 16.0, ..default() },
                        SwarmText
                    ));
                    row.spawn((Button, Node { width: Val::Px(30.0), margin: UiRect::left(Val::Px(10.0)), ..default() }, BackgroundColor(Color::srgb(0.3, 0.3, 0.3)), MenuButton::SwarmDown)).with_children(|b| { b.spawn(Text::new("-")); });
                    row.spawn((Button, Node { width: Val::Px(30.0), margin: UiRect::left(Val::Px(5.0)), ..default() }, BackgroundColor(Color::srgb(0.3, 0.3, 0.3)), MenuButton::SwarmUp)).with_children(|b| { b.spawn(Text::new("+")); });
                });
            });
        });

        // Botão Start
        root.spawn((
            Button,
            Node {
                width: Val::Px(250.0),
                height: Val::Px(60.0),
                margin: UiRect::top(Val::Px(30.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.0, 0.5, 0.4)),
            MenuButton::StartSimulation,
        )).with_children(|b| {
            b.spawn((Text::new("INICIAR SIMULAÇÃO"), TextFont { font_size: 24.0, ..default() }));
        });
    });
}

fn handle_menu_interactions(
    mut next_state: ResMut<NextState<AppState>>,
    mut config: ResMut<UniverseConfig>,
    query: Query<(&Interaction, &MenuButton), (Changed<Interaction>, With<Button>)>,
) {
    for (interaction, button) in &query {
        if *interaction == Interaction::Pressed {
            match button {
                MenuButton::StartSimulation => {
                    let _ = config.save();
                    next_state.set(AppState::Simulation);
                }
                MenuButton::SelectTrack(i) => {
                    config.selected_track_index = *i;
                }
                MenuButton::AddTrack => {
                    let mut rng = rand::rng();
                    let new_track = TrackProfile {
                        name: format!("New Circuit {}", config.tracks.len() + 1),
                        seed: rng.random(),
                        track_width: 18.0,
                        complexity: 1.0,
                    };
                    config.tracks.push(new_track);
                    config.selected_track_index = config.tracks.len() - 1;
                }
                MenuButton::DeleteTrack(i) => {
                    if config.tracks.len() > 1 {
                        config.tracks.remove(*i);
                        if config.selected_track_index >= config.tracks.len() {
                            config.selected_track_index = config.tracks.len() - 1;
                        }
                    }
                }
                MenuButton::TogglePhysics(toggle) => {
                    let p = &mut config.physics;
                    match toggle {
                        PhysicsToggle::Stall => p.enable_stall = !p.enable_stall,
                        PhysicsToggle::Thermal => p.enable_thermal = !p.enable_thermal,
                        PhysicsToggle::Aero => p.enable_aero_drag = !p.enable_aero_drag,
                        PhysicsToggle::Downforce => p.enable_downforce = !p.enable_downforce,
                        PhysicsToggle::Tcs => p.enable_tcs = !p.enable_tcs,
                        PhysicsToggle::Abs => p.enable_abs = !p.enable_abs,
                        PhysicsToggle::Aqua => p.enable_aquaplaning = !p.enable_aquaplaning,
                        PhysicsToggle::Wear => p.enable_tire_wear = !p.enable_tire_wear,
                    }
                }
                MenuButton::SwarmUp => config.swarm_count = (config.swarm_count + 1).min(100),
                MenuButton::SwarmDown => config.swarm_count = (config.swarm_count - 1).max(1),
            }
        }
    }
}

fn cleanup_menu(mut commands: Commands, query: Query<Entity, With<MenuEntity>>) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
}

fn sync_menu_ui(
    config: Res<UniverseConfig>,
    mut swarm_text: Query<&mut Text, With<SwarmText>>,
    mut physics_indicators: Query<(&mut BackgroundColor, &PhysicsToggleIndicator), Without<TrackIndicator>>,
    mut track_indicators: Query<(&mut BackgroundColor, &TrackIndicator), Without<PhysicsToggleIndicator>>,
) {
    if let Ok(mut text) = swarm_text.single_mut() {
        text.0 = format!("SWARM SIZE: {}", config.swarm_count);
    }

    for (mut bg, indicator) in &mut physics_indicators {
        let val = match indicator.0 {
            PhysicsToggle::Stall => config.physics.enable_stall,
            PhysicsToggle::Thermal => config.physics.enable_thermal,
            PhysicsToggle::Aero => config.physics.enable_aero_drag,
            PhysicsToggle::Downforce => config.physics.enable_downforce,
            PhysicsToggle::Tcs => config.physics.enable_tcs,
            PhysicsToggle::Abs => config.physics.enable_abs,
            PhysicsToggle::Aqua => config.physics.enable_aquaplaning,
            PhysicsToggle::Wear => config.physics.enable_tire_wear,
        };
        bg.0 = if val { Color::srgb(0.0, 0.8, 0.2) } else { Color::srgb(0.8, 0.1, 0.1) };
    }

    for (mut bg, indicator) in &mut track_indicators {
        let is_selected = indicator.0 == config.selected_track_index;
        bg.0 = if is_selected { Color::srgb(0.0, 0.4, 0.3) } else { Color::srgb(0.2, 0.2, 0.2) };
    }
}

fn handle_button_hover(
    mut query: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<Button>)>,
) {
    for (interaction, mut bg) in &mut query {
        match *interaction {
            Interaction::Hovered => {
                let current = bg.0.to_srgba();
                bg.0 = Color::srgba(current.red + 0.1, current.green + 0.1, current.blue + 0.1, current.alpha);
            }
            Interaction::None => {
                // O sync_menu_ui cuidará das cores base para toggles e tracks.
                // Para botões genéricos, poderíamos armazenar a cor original, 
                // mas para este lab, o sync resolve a maioria.
            }
            _ => {}
        }
    }
}
