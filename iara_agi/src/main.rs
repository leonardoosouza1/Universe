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
        log: "System Awakening...".to_string(),
        vision_res: 0,
        is_ad: false,
        delta: 0.0,
    }));
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let target_url = if args.len() > 1 { &args[1] } else { "https://www.youtube.com" };

    println!("--- IARA LAB: STABLE SOVEREIGNTY ---");

    let mut sys = System::new_all();
    sys.refresh_memory();
    let total_ram_gb = sys.total_memory() / 1024 / 1024 / 1024;
    let (input_size, hidden_size, output_size, vision_side) = if total_ram_gb > 24 {
        (4096, 1024, 256, 64) 
    } else {
        (1024, 512, 128, 32)  
    };

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

    println!("[NAV] Initializing portal connection...");
    let page = browser.new_page(target_url).await?;
    tokio::time::sleep(Duration::from_secs(8)).await;

    let mut iara = SovereignEngine::new_scaled(input_size, hidden_size, output_size);
    let mut last_cycle = Instant::now();
    let mut last_reload = Instant::now();
    let mut prev_sensory_wave = vec![0i8; input_size];
    let mut static_frame_count = 0;

    loop {
        let start = Instant::now();
        let current_url = page.url().await.ok().flatten().unwrap_or_default();
        let is_watching = current_url.contains("watch?v=");

        // A) Visual Capture
        let mut sensory_wave = vec![0i8; input_size];
        let mut b64_vision = None;

        if let Ok(Ok(bytes)) = tokio::time::timeout(Duration::from_millis(1500), page.screenshot(ScreenshotParams::builder().full_page(false).build())).await {
            b64_vision = Some(general_purpose::STANDARD.encode(&bytes));
            if let Ok(img) = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?.decode() {
                let scaled = img.thumbnail(vision_side as u32, vision_side as u32);
                for (i, (_x, _y, pixel)) in scaled.pixels().enumerate() {
                    if i < input_size {
                        let luma = (pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114) as u8;
                        sensory_wave[i] = (luma as i16 - 128) as i8;
                    }
                }
            }
        }

        // B) Brain Processing
        let dt = last_cycle.elapsed().as_secs_f32();
        last_cycle = Instant::now();
        iara.execute_cycle(&sensory_wave, 1.0, dt);
        
        let neuron_outputs = iara.instinct_layer.forward(&sensory_wave);
        let dna_sample = iara.instinct_layer.weights.iter().take(1024).cloned().collect();

        // C) STABILITY LOGIC
        let visual_delta: f32 = sensory_wave.iter().zip(prev_sensory_wave.iter())
            .map(|(&a, &b)| (a as f32 - b as f32).abs())
            .sum::<f32>() / input_size as f32;
        prev_sensory_wave = sensory_wave.clone();

        // Só conta estática se o browser terminou de carregar o básico
        if is_watching && visual_delta < 0.8 { static_frame_count += 1; } 
        else { static_frame_count = 0; }

        let mut action_log = "Processing...".to_string();
        let mut is_ad_detected = false;
        let can_reload = last_reload.elapsed().as_secs() > 20;

        let script = format!("
            (() => {{
                const isWatching = {};
                const staticTicks = {};
                const canReload = {};
                
                // 1. Erros de Interface
                const errorDetectors = ['#error-screen', '.ytp-error', 'button[aria-label=\"Tentar novamente\"]'];
                const hasError = errorDetectors.some(s => document.querySelector(s));

                // RELOAD CONTROLADO (Evita loop de F5)
                if ((hasError || staticTicks > 60) && canReload && document.readyState === 'complete') {{
                    window.location.reload();
                    return {{ msg: 'STABILITY: TRIGGERING REFRESH', isAd: false, reloaded: true }};
                }}

                const skipBtn = document.querySelector('.ytp-ad-skip-button, .ytp-ad-skip-button-modern');
                if (skipBtn) {{ skipBtn.click(); return {{ msg: 'ACTION: SKIPPING AD', isAd: true, reloaded: false }}; }}

                const isAd = document.querySelector('.ad-showing, .ad-interrupting') !== null;

                if (isWatching && !isAd && staticTicks > 15 && document.readyState === 'complete') {{
                    const video = document.querySelector('video');
                    if (video) {{
                        video.play().catch(() => video.muted = true);
                        const player = document.querySelector('#movie_player');
                        if (player) player.click();
                        return {{ msg: 'STABILITY: FORCING PLAY', isAd: false, reloaded: false }};
                    }}
                }}

                if (isAd) return {{ msg: 'STATUS: AD INTERFERENCE', isAd: true, reloaded: false }};
                return {{ msg: 'STATUS: ABSORBING DATA', isAd: false, reloaded: false }};
            }})()
        ", is_watching, static_frame_count, can_reload);

        if let Ok(val) = page.evaluate(script).await {
            if let Some(obj) = val.value() {
                action_log = obj.get("msg").and_then(|v| v.as_str()).unwrap_or("...").to_string();
                is_ad_detected = obj.get("isAd").and_then(|v| v.as_bool()).unwrap_or(false);
                if obj.get("reloaded").and_then(|v| v.as_bool()).unwrap_or(false) {
                    last_reload = Instant::now();
                    static_frame_count = 0;
                }
            }
        }

        // D) Dashboard Sync
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
                state.is_ad = is_ad_detected;
                state.delta = visual_delta;
                state.vision_res = vision_side;
            }
        }

        let elapsed = start.elapsed();
        if elapsed < Duration::from_millis(500) { tokio::time::sleep(Duration::from_millis(500) - elapsed).await; }
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
