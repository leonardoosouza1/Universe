mod engine;
mod metabolism;
mod layers;
mod cfc;
mod kan;
mod resonance;
mod memory;
mod types;
mod reality;
mod repominer;
mod mesh;

use crate::engine::SovereignEngine;
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::page::ScreenshotParams;
use futures::StreamExt;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use axum::{routing::get, extract::ws::{Message, WebSocket, WebSocketUpgrade}, response::IntoResponse, Router};
use tower_http::services::ServeDir;
use serde::Serialize;
use base64::{Engine as _, engine::general_purpose};
use image::{io::Reader as ImageReader, GenericImageView};
use std::io::Cursor;
use sysinfo::System;
use std::env;

#[derive(Serialize, Clone)]
struct LabState {
    energy: f32,
    fatigue: f32,
    coherence: f32,
    sensory_wave: Vec<i8>,
    neuron_outputs: Vec<f32>,
    dna_sample: Vec<i8>,
    screenshot: Option<String>,
    log: String,
    vision_res: usize,
    is_ad: bool,
    delta: f32,
}

lazy_static::lazy_static! {
    static ref GLOBAL_STATE: Arc<Mutex<LabState>> = Arc::new(Mutex::new(LabState {
        energy: 0.0, fatigue: 0.0, coherence: 0.0,
        sensory_wave: Vec::new(), neuron_outputs: Vec::new(),
        dna_sample: vec![0; 1024], screenshot: None,
        log: "Activating Hyper-Perception...".to_string(),
        vision_res: 0,
        is_ad: false,
        delta: 0.0,
    }));
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let target_url = if args.len() > 1 { &args[1] } else { "https://www.youtube.com" };

    println!("--- IARA LAB: HYPER-PERCEPTION MODE (RGB MATRIX) ---");

    let mut sys = System::new_all();
    sys.refresh_memory();
    let total_ram_gb = sys.total_memory() / 1024 / 1024 / 1024;
    
    // Configuração Ultra: 64x32 Grid = 2048 Inputs
    let (cols, rows) = (64, 32);
    let input_size = cols * rows;
    let hidden_size = if total_ram_gb > 16 { 2048 } else { 1024 };
    let output_size = 128;

    let app = Router::new().fallback_service(ServeDir::new("public")).route("/ws", get(ws_handler));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });

    let user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";
    let (browser, mut handler) = Browser::launch(BrowserConfig::builder()
        .no_sandbox()
        .window_size(1920, 1080)
        .arg(format!("--user-agent={}", user_agent))
        .build()?).await?;

    tokio::spawn(async move {
        while let Some(h) = handler.next().await { if h.is_err() { break; } }
    });

    let page = browser.new_page(target_url).await?;
    tokio::time::sleep(Duration::from_secs(5)).await;

    let mut iara = SovereignEngine::new_scaled(input_size, hidden_size, output_size);
    let mut last_cycle = Instant::now();
    let mut prev_luma_grid = vec![0u8; input_size];

    loop {
        let start = Instant::now();
        let current_url = page.url().await.ok().flatten().unwrap_or_default();

        // A) Captura de Screenshot (Para o Canal Vermelho/Movimento)
        let mut b64_vision = None;
        let mut current_luma_grid = vec![0u8; input_size];
        
        if let Ok(Ok(bytes)) = tokio::time::timeout(Duration::from_millis(1000), page.screenshot(ScreenshotParams::builder().full_page(false).build())).await {
            b64_vision = Some(general_purpose::STANDARD.encode(&bytes));
            if let Ok(img) = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?.decode() {
                let scaled = img.thumbnail(cols as u32, rows as u32);
                for (i, (_x, _y, pixel)) in scaled.pixels().enumerate() {
                    if i < input_size {
                        current_luma_grid[i] = (pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114) as u8;
                    }
                }
            }
        }

        // B) SEMANTIC DOM SENSING (Para os Canais Verde e Azul)
        let mut sensory_wave = vec![0i8; input_size];
        
        let dom_script = format!("
            (() => {{
                const cols = {}; const rows = {};
                const cellW = window.innerWidth / cols;
                const cellH = window.innerHeight / rows;
                let grid = new Array(cols * rows).fill(0);
                
                const elements = document.querySelectorAll('video, a, button, [role=\"button\"], .ytd-thumbnail');
                elements.forEach(el => {{
                    const r = el.getBoundingClientRect();
                    if (r.width === 0 || r.top > window.innerHeight) return;
                    
                    let energy = 50; // Link default
                    if (el.tagName === 'VIDEO') energy = 127;
                    else if (el.tagName === 'BUTTON' || el.getAttribute('role') === 'button') energy = 100;

                    let sX = Math.max(0, Math.floor(r.left / cellW));
                    let eX = Math.min(cols - 1, Math.floor(r.right / cellW));
                    let sY = Math.max(0, Math.floor(r.top / cellH));
                    let eY = Math.min(rows - 1, Math.floor(r.bottom / cellH));

                    for (let y = sY; y <= eY; y++) {{
                        for (let x = sX; x <= eX; x++) {{
                            const idx = y * cols + x;
                            if (energy > grid[idx]) grid[idx] = energy;
                        }}
                    }}
                }});
                return grid;
            }})()
        ", cols, rows);

        let semantic_grid = match tokio::time::timeout(Duration::from_millis(500), page.evaluate(dom_script)).await {
            Ok(Ok(val)) => val.into_value::<Vec<i8>>().unwrap_or_else(|_| vec![0; input_size]),
            _ => vec![0; input_size]
        };

        // C) FUSÃO DE CONSCIÊNCIA (RGB Mapping)
        for i in 0..input_size {
            // Vermelho (Movimento): Diferença de Luma
            let motion = (current_luma_grid[i] as i16 - prev_luma_grid[i] as i16).abs() as i8;
            // Verde (Semântica): Grid do DOM
            let semantic = semantic_grid[i];
            
            // Combinamos os dois na onda sensorial:
            // A IARA sente o movimento como um pulso positivo e a estrutura como a base da onda
            sensory_wave[i] = (semantic as i16 + motion as i16 - 64).clamp(-127, 127) as i8;
        }
        prev_luma_grid = current_luma_grid;

        // D) Brain Processing
        let dt = last_cycle.elapsed().as_secs_f32();
        last_cycle = Instant::now();
        iara.execute_cycle(&sensory_wave, 1.0, dt);
        
        let neuron_outputs = iara.instinct_layer.forward(&sensory_wave);
        let dna_sample = iara.instinct_layer.weights.iter().take(1024).cloned().collect();

        // E) STABILITY & RECOVERY
        let visual_delta: f32 = sensory_wave.iter().map(|&x| (x as f32).abs()).sum::<f32>() / input_size as f32;
        
        let mut action_log = "Hyper-Processing...".to_string();
        let mut is_ad = false;

        let script = "
            (() => {
                const skipBtn = document.querySelector('.ytp-ad-skip-button, .ytp-ad-skip-button-modern');
                if (skipBtn) { skipBtn.click(); return { msg: 'ACTION: NEURAL BYPASS (AD)', isAd: true }; }
                
                const video = document.querySelector('video');
                if (video && video.paused && !document.querySelector('.ad-showing')) {
                    video.play().catch(() => video.muted = true);
                    return { msg: 'STABILITY: RE-PLAYING REALITY', isAd: false };
                }
                
                return { msg: 'STATUS: ABSORBING HYPER-DATA', isAd: document.querySelector('.ad-showing') !== null };
            })()
        ";

        if let Ok(val) = page.evaluate(script).await {
            if let Some(obj) = val.value() {
                action_log = obj.get("msg").and_then(|v| v.as_str()).unwrap_or("...").to_string();
                is_ad = obj.get("isAd").and_then(|v| v.as_bool()).unwrap_or(false);
            }
        }

        // F) Sync
        {
            if let Ok(mut state) = GLOBAL_STATE.lock() {
                state.energy = iara.metabolism.cache_stress;
                state.fatigue = iara.metabolism.adenosine;
                state.coherence = iara.heart_resonance.get_coherence(1.0);
                state.sensory_wave = sensory_wave;
                state.neuron_outputs = neuron_outputs;
                state.dna_sample = dna_sample;
                if let Some(vision) = b64_vision { state.screenshot = Some(vision); }
                state.log = format!("{} | URL: {}", action_log, current_url);
                state.is_ad = is_ad;
                state.delta = visual_delta;
                state.vision_res = cols;
            }
        }

        tokio::time::sleep(Duration::from_millis(400)).await;
    }
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse { ws.on_upgrade(handle_socket) }
async fn handle_socket(mut socket: WebSocket) {
    loop {
        let state_json = { if let Ok(state) = GLOBAL_STATE.lock() { serde_json::to_string(&*state).unwrap() } else { continue; } };
        if socket.send(Message::Text(state_json)).await.is_err() { break; }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}
