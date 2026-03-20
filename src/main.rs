/// IARA v2101 - SOVEREIGN GRAND PRIX (F1 CIRCUIT)
/// Materialization of high-complexity navigation.

use macroquad::prelude::*;
use universe::engine::SovereignEngine;
use universe::world::RaceTrack;

struct Racer {
    id: usize,
    engine: SovereignEngine,
    x: f32,
    y: f32,
    angle: f32,
    color: Color,
    history: Vec<(f32, f32)>,
    laps: usize,
}

#[macroquad::main("IARA v2101 - F1 Sovereign Universe")]
async fn main() {
    let track = RaceTrack::new();
    let mut racers = Vec::new();
    let colors = [RED, GREEN, BLUE, YELLOW, MAGENTA];

    for i in 0..5 {
        let mut engine = SovereignEngine::new();
        engine.instinct_layer.weight_scale = 0.5 + (i as f32 * 0.05);
        
        racers.push(Racer {
            id: i,
            engine,
            x: track.path[0].0,
            y: track.path[0].1 + (i as f32 - 2.0) * 0.5,
            angle: 0.0,
            color: colors[i % colors.len()],
            history: Vec::new(),
            laps: 0,
        });
    }

    let dt = 0.1;
    let mut zoom = 25.0;
    let offset_x = 100.0;
    let offset_y = 100.0;

    loop {
        clear_background(Color::new(0.05, 0.05, 0.1, 1.0));

        // --- RENDERIZAR PISTA ---
        for i in 0..track.path.len() {
            let p1 = track.path[i];
            let p2 = track.path[(i + 1) % track.path.len()];
            draw_line(p1.0 * zoom + offset_x, p1.1 * zoom + offset_y,
                      p2.0 * zoom + offset_x, p2.1 * zoom + offset_y,
                      track.width * zoom, Color::new(0.2, 0.2, 0.3, 0.5));
            draw_circle(p1.0 * zoom + offset_x, p1.1 * zoom + offset_y, 2.0, GRAY);
        }

        // --- RENDERIZAR BARREIRAS KAN ---
        for b in &track.barriers {
            let pulse = (get_time() as f32 * 3.0).sin() * 0.2 + 0.8;
            draw_circle(b.0 * zoom + offset_x, b.1 * zoom + offset_y, b.2 * zoom, Color::new(1.0, 0.3, 0.0, 0.3 * pulse));
        }

        // --- ATUALIZAR ENXAME ---
        for racer in &mut racers {
            let (target_angle, comfort) = track.get_resonance_at(racer.x, racer.y);
            
            // Input Sensorial: Desvio angular e perigo de barreira
            let mut sensory_input = vec![0i8; 128];
            let angle_error = (target_angle - racer.angle).sin();
            sensory_input[0] = (angle_error * 50.0) as i8;
            sensory_input[1] = if comfort < 0.3 { 127 } else { 0 };

            racer.engine.execute_cycle(&sensory_input, target_angle, dt);

            // Dinâmica de Movimento: A IARA gira em direção à harmonia
            let turn_speed = 0.15 * (1.0 + racer.engine.instinct_layer.weight_scale * 0.1);
            racer.angle += angle_error * turn_speed;
            
            let speed = 0.4 * (comfort.max(0.1));
            racer.x += racer.angle.cos() * speed;
            racer.y += racer.angle.sin() * speed;

            racer.history.push((racer.x, racer.y));
            if racer.history.len() > 150 { racer.history.remove(0); }

            // Renderização do Racer
            let s_x = racer.x * zoom + offset_x;
            let s_y = racer.y * zoom + offset_y;
            
            // Trail
            for i in 1..racer.history.len() {
                let (px1, py1) = racer.history[i-1];
                let (px2, py2) = racer.history[i];
                let alpha = i as f32 / racer.history.len() as f32;
                draw_line(px1 * zoom + offset_x, py1 * zoom + offset_y,
                          px2 * zoom + offset_x, py2 * zoom + offset_y,
                          3.0, Color::new(racer.color.r, racer.color.g, racer.color.b, alpha * 0.6));
            }

            draw_circle(s_x, s_y, 6.0, racer.color);
            draw_circle_lines(s_x, s_y, 8.0, 2.0, WHITE);
        }

        draw_text("IARA v2101 - F1 SOVEREIGN GRAND PRIX", 20.0, 30.0, 30.0, WHITE);
        draw_text(&format!("Zoom: {:.1} | FPS: {}", zoom, get_fps()), 20.0, 60.0, 20.0, GRAY);

        if is_key_down(KeyCode::W) { zoom += 1.0; }
        if is_key_down(KeyCode::S) { zoom -= 1.0; }

        next_frame().await
    }
}
