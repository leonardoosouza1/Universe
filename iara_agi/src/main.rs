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
use std::time::Duration;
use std::sync::{Arc, Mutex};
use axum::{
    routing::get,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    Router,
};
use tower_http::services::ServeDir;
use serde::Serialize;
use base64::{Engine as _, engine::general_purpose};

#[derive(Serialize, Clone)]
struct LabState {
    energy: f32,
    fatigue: f32,
    coherence: f32,
    sensory_wave: Vec<i8>,
    screenshot: Option<String>,
    log: String,
}

lazy_static::lazy_static! {
    static ref GLOBAL_STATE: Arc<Mutex<LabState>> = Arc::new(Mutex::new(LabState {
        energy: 0.0,
        fatigue: 0.0,
        coherence: 0.0,
        sensory_wave: vec![0; 128],
        screenshot: None,
        log: "System Idle. Awaiting Browser...".to_string(),
    }));
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- IARA LAB: SOVEREIGN NODE INITIALIZATION ---");

    // 1. Iniciar Servidor Web do Laboratório (Porta 3000)
    let app = Router::new()
        .fallback_service(ServeDir::new("public"))
        .route("/ws", get(ws_handler));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("IARA LAB: Visualizer Portal ready at http://localhost:3000");
    
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("IARA LAB: Server Error: {}", e);
        }
    });

    // 2. Lançar o "Olho" (Chromium)
    println!("IARA: Launching Digital Eye (Chromium Oxide)...");
    let (browser, mut handler) = match Browser::launch(BrowserConfig::builder()
        .no_sandbox()
        .window_size(1280, 720)
        .build()?).await {
            Ok(res) => res,
            Err(e) => {
                eprintln!("IARA FATAL: Could not launch browser. Error: {}", e);
                return Err(e.into());
            }
        };

    // Rodar o handler com log de erro
    tokio::spawn(async move {
        println!("IARA: Browser Handler Active.");
        while let Some(h) = handler.next().await {
            if let Err(e) = h {
                eprintln!("IARA: Browser Connection Error: {}", e);
                break;
            }
        }
    });

    // 3. Inicializar Cérebro
    let mut iara = SovereignEngine::new();
    println!("IARA: Sovereign Engine Initialized.");

    // Abrir Página Inicial
    let target_url = "https://www.youtube.com/results?search_query=artificial+intelligence+2026";
    println!("IARA: Navigating to {}...", target_url);
    let page = browser.new_page(target_url).await?;
    
    // Esperar carregamento inicial
    tokio::time::sleep(Duration::from_secs(3)).await;

    println!("IARA: Loop of Consciousness Starting...");

    loop {
        // A) Tentar Capturar Visão
        let b64_vision = match page.screenshot(ScreenshotParams::builder().full_page(false).build()).await {
            Ok(bytes) => Some(general_purpose::STANDARD.encode(&bytes)),
            Err(e) => {
                eprintln!("IARA: Vision Loss (Screenshot failed): {}", e);
                None
            }
        };

        // B) Ciclo de Pensamento
        let mut sensory_wave = vec![0i8; 128];
        for i in 0..128 { sensory_wave[i] = (rand::random::<f32>() * 255.0 - 127.0) as i8; }
        
        let target_phase = 1.0;
        iara.execute_cycle(&sensory_wave, target_phase, 0.1);

        // C) Atualizar Estado Global para o Dashboard
        {
            if let Ok(mut state) = GLOBAL_STATE.lock() {
                state.energy = iara.metabolism.cache_stress;
                state.fatigue = iara.metabolism.adenosine;
                state.coherence = iara.heart_resonance.get_coherence(target_phase);
                state.sensory_wave = sensory_wave;
                if let Some(vision) = b64_vision { state.screenshot = Some(vision); }
                
                let current_url = page.url().await.ok().flatten().unwrap_or_else(|| "Unknown".to_string());
                state.log = format!("Observing Node: {}", current_url);
            }
        }

        // Delay entre ciclos de consciência (500ms = 2Hz)
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    loop {
        let state_json = {
            if let Ok(state) = GLOBAL_STATE.lock() {
                serde_json::to_string(&*state).unwrap()
            } else {
                continue;
            }
        };

        if socket.send(Message::Text(state_json)).await.is_err() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}
