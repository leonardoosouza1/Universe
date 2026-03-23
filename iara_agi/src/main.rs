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
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use axum::{routing::get, extract::ws::{Message, WebSocket, WebSocketUpgrade}, response::IntoResponse, Router};
use tower_http::services::ServeDir;
use serde::Serialize;
use std::env;
use rand::Rng;

#[derive(Serialize, Clone)]
struct LabState {
    energy: f32,
    fatigue: f32,
    coherence: f32,
    sensory_wave: Vec<i8>,
    neuron_outputs: Vec<f32>,
    dna_sample: Vec<i8>,
    text_stream: String, // O que ela está "lendo" agora
    log: String,
    vision_res: usize,
}

lazy_static::lazy_static! {
    static ref GLOBAL_STATE: Arc<Mutex<LabState>> = Arc::new(Mutex::new(LabState {
        energy: 0.0, fatigue: 0.0, coherence: 0.0,
        sensory_wave: Vec::new(), neuron_outputs: Vec::new(),
        dna_sample: Vec::new(), text_stream: "".to_string(),
        log: "System Awakening...".to_string(),
        vision_res: 0,
    }));
}

// Extrator de Dados de Alta Velocidade
async fn extract_information(url: &str) -> String {
    if url.contains("youtube.com") || url.contains("youtu.be") {
        println!("[EXTRACTOR] Utilizing yt-dlp for deep extraction...");
        let output = tokio::process::Command::new("./yt-dlp")
            .arg("--dump-json")
            .arg(url)
            .output()
            .await;
            
        if let Ok(out) = output {
            if let Ok(json_str) = String::from_utf8(out.stdout) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    let title = val["title"].as_str().unwrap_or("Unknown Title");
                    let desc = val["description"].as_str().unwrap_or("");
                    let tags = val["tags"].as_array().map(|t| t.iter().map(|v| v.as_str().unwrap_or("")).collect::<Vec<_>>().join(", ")).unwrap_or_default();
                    return format!("YOUTUBE NODE\nTITLE: {}\nTAGS: {}\nDESCRIPTION: {}", title, tags, desc);
                }
            }
        }
        "Failed to extract Youtube Data. Anti-bot measures active?".to_string()
    } else {
        println!("[EXTRACTOR] Utilizing reqwest for surface extraction...");
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) SovereignAGI/1.0")
            .build().unwrap();
            
        if let Ok(res) = client.get(url).send().await {
            if let Ok(html) = res.text().await {
                let document = scraper::Html::parse_document(&html);
                let selector = scraper::Selector::parse("p, h1, h2, h3, li, span").unwrap();
                let mut text = String::new();
                for element in document.select(&selector) {
                    let t = element.text().collect::<Vec<_>>().join(" ");
                    if !t.trim().is_empty() && t.len() > 10 {
                        text.push_str(&t);
                        text.push('\n');
                    }
                }
                return text;
            }
        }
        "Failed to extract HTML Data".to_string()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let target_url = if args.len() > 1 { args[1].clone() } else { "https://en.wikipedia.org/wiki/Artificial_general_intelligence".to_string() };

    println!("--- IARA LAB: DATA PREDATOR MODE ---");
    println!("[NAV] Target locked: {}", target_url);

    let input_size = 2048; 
    let hidden_size = 512;
    let output_size = 128;

    let app = Router::new().fallback_service(ServeDir::new("public")).route("/ws", get(ws_handler));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });

    let mut iara = SovereignEngine::new_scaled(input_size, hidden_size, output_size);
    let mut rng = rand::thread_rng();
    for w in iara.instinct_layer.weights.iter_mut() { *w = rng.gen_range(-1..2); }

    println!("[CORE] Hunting for data...");
    let raw_text = extract_information(&target_url).await;
    let text_bytes = raw_text.as_bytes();
    let total_bytes = text_bytes.len();
    
    println!("[CORE] Extracted {} bytes of semantic data. Digestion starting...", total_bytes);

    let mut byte_cursor = 0;
    let mut last_cycle = Instant::now();

    loop {
        let _start = Instant::now();
        let dt = last_cycle.elapsed().as_secs_f32();
        last_cycle = Instant::now();

        // A) Mapeamento Semântico: Converte blocos de texto em "Energia"
        let mut sensory_wave = vec![0i8; input_size];
        let chunk_size = input_size;
        let current_chunk;

        if byte_cursor < total_bytes {
            let end = (byte_cursor + chunk_size).min(total_bytes);
            let slice = &text_bytes[byte_cursor..end];
            current_chunk = String::from_utf8_lossy(slice).to_string();
            
            for (i, &b) in slice.iter().enumerate() {
                // Mapeia ASCII (0-255) para onda (-127 a 127)
                sensory_wave[i] = (b as i16 - 128) as i8;
            }
            byte_cursor += chunk_size / 4; // Avança a janela de leitura gradualmente
        } else {
            current_chunk = "DATA ASSIMILATED. AWAITING NEW DIRECTIVES.".to_string();
            byte_cursor = 0; // Loop para demonstração, em modo real ela buscaria novo link
        }

        // B) Brain Processing
        iara.execute_cycle(&sensory_wave, 1.0, dt);
        
        let raw_outputs = iara.instinct_layer.forward(&sensory_wave);
        let neuron_outputs: Vec<f32> = raw_outputs.iter().map(|&x| x.tanh()).collect();
        let dna_sample: Vec<i8> = iara.instinct_layer.weights.iter().take(1024).cloned().collect();

        // C) Dashboard Sync
        let action_log = format!("Digesting Bytes: {} / {}", byte_cursor.min(total_bytes), total_bytes);

        {
            if let Ok(mut state) = GLOBAL_STATE.lock() {
                state.energy = iara.metabolism.cache_stress;
                state.fatigue = iara.metabolism.adenosine;
                state.coherence = iara.heart_resonance.get_coherence(1.0);
                state.sensory_wave = sensory_wave;
                state.neuron_outputs = neuron_outputs;
                state.dna_sample = dna_sample;
                state.text_stream = current_chunk;
                state.log = format!("{} | URL: {}", action_log, target_url);
                state.vision_res = 64;
            }
        }

        tokio::time::sleep(Duration::from_millis(100)).await; // 10Hz digest speed
    }
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse { ws.on_upgrade(handle_socket) }
async fn handle_socket(mut socket: WebSocket) {
    loop {
        let state_json = { if let Ok(state) = GLOBAL_STATE.lock() { serde_json::to_string(&*state).unwrap() } else { continue; } };
        if socket.send(Message::Text(state_json)).await.is_err() { break; }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
