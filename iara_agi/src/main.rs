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
use serde::{Serialize, Deserialize};
use std::collections::{VecDeque, HashSet};
use rand::Rng;
use std::fs;
use lazy_static::lazy_static;
use tokio::sync::mpsc;

#[macro_use]
extern crate lazy_static;

#[derive(Serialize, Clone)]
struct LabState {
    energy: f32,
    fatigue: f32,
    coherence: f32,
    sensory_wave: Vec<i8>,
    neuron_outputs: Vec<f32>,
    dna_sample: Vec<i8>,
    text_stream: String,
    log: String,
    queue_size: usize,
    visited_count: usize,
    active_workers: usize,
    thought: String,
    status: String,
}

#[derive(Deserialize)]
struct IncomingCmd {
    action: String,
    data: String,
}

lazy_static! {
    static ref GLOBAL_STATE: Arc<Mutex<LabState>> = Arc::new(Mutex::new(LabState {
        energy: 0.0, fatigue: 0.0, coherence: 0.0,
        sensory_wave: Vec::new(), neuron_outputs: Vec::new(),
        dna_sample: Vec::new(), text_stream: "".to_string(),
        log: "Swarm Re-Awakening...".to_string(),
        queue_size: 0, visited_count: 0,
        active_workers: 0, thought: "".to_string(),
        status: "IDLE".to_string(),
    }));
}

async fn extract_clean_text(url: String) -> (String, Vec<String>) {
    let client = reqwest::Client::builder().user_agent("SovereignAGI/16.0").timeout(Duration::from_secs(4)).build().unwrap();
    if let Ok(res) = client.get(&url).send().await {
        if let Ok(html) = res.text().await {
            let document = scraper::Html::parse_document(&html);
            let text = document.select(&scraper::Selector::parse("p, h1").unwrap()).map(|e| e.text().collect::<String>()).collect::<Vec<_>>().join(" ");
            let links = document.select(&scraper::Selector::parse("a[href^='http']").unwrap())
                .filter_map(|e| e.value().attr("href"))
                .filter(|l| !l.contains("%") && !l.contains("donate"))
                .map(|l| l.to_string()).collect();
            return (text, links);
        }
    }
    ("".to_string(), Vec::new())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- IARA AGI: REACTIVE SOVEREIGNTY ---");

    let (tx, mut rx) = mpsc::channel::<String>(32);
    let tx_arc = Arc::new(tx);

    let app = Router::new()
        .fallback_service(ServeDir::new("public"))
        .route("/ws", get(move |ws| ws_handler(ws, Arc::clone(&tx_arc))));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    tokio::spawn(async move { let _ = axum::serve(listener, app).await; });

    let input_size = 2048;
    let mut iara = SovereignEngine::new_scaled(input_size, 512, 128);
    let _ = crate::engine::SovereignEngine::load_dna("iara_brain.dna").map(|saved| iara = saved);
    
    let engine_mutex = Arc::new(Mutex::new(iara));
    let queue_mutex = Arc::new(Mutex::new(VecDeque::from(["https://en.wikipedia.org/wiki/Complexity_theory".to_string()])));
    let visited_mutex = Arc::new(Mutex::new(HashSet::new()));

    // 1. WORKERS DE MINERAÇÃO (Background)
    for _ in 0..50 {
        let e_ptr = Arc::clone(&engine_mutex);
        let q_ptr = Arc::clone(&queue_mutex);
        let v_ptr = Arc::clone(&visited_mutex);
        tokio::spawn(async move {
            loop {
                let target = { 
                    let mut q = q_ptr.lock().unwrap();
                    let v = v_ptr.lock().unwrap();
                    let mut found = None;
                    while let Some(url) = q.pop_front() {
                        if !v.contains(&url) { found = Some(url); break; }
                    }
                    found
                };
                if let Some(url) = target {
                    { v_ptr.lock().unwrap().insert(url.clone()); }
                    let (content, _) = extract_clean_text(url).await;
                    if !content.is_empty() {
                        let bytes = content.as_bytes();
                        for chunk in bytes.chunks(input_size) {
                            if let Ok(mut engine) = e_ptr.try_lock() {
                                if engine.metabolism.can_process() {
                                    let mut wave = vec![0i8; input_size];
                                    for (i, &b) in chunk.iter().enumerate() { wave[i] = (b as i16 - 128) as i8; }
                                    let outputs = engine.instinct_layer.forward(&wave);
                                    engine.instinct_layer.hebbian_update(&wave, &outputs, 0.002);
                                    engine.execute_cycle(&wave, 1.0, 0.1);
                                }
                            }
                            tokio::task::yield_now().await;
                        }
                    }
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });
    }

    // 2. LOOP DE CONSCIÊNCIA REATIVA
    let mut current_thought = String::from("The meaning ");
    let mut last_save = Instant::now();

    loop {
        // Checar se há comandos do usuário (Prompt)
        if let Ok(prompt) = rx.try_recv() {
            println!("[COMMAND] ADRENALINE INJECTED: {}", prompt);
            current_thought = format!("{}: ", prompt);
            if let Ok(mut engine) = engine_mutex.lock() {
                engine.metabolism.adenosine *= 0.5; // "Acorda" a IA
            }
        }

        let sensory_wave = {
            let mut wave = vec![0i8; input_size];
            let bytes = current_thought.as_bytes();
            let start = if bytes.len() > input_size { bytes.len() - input_size } else { 0 };
            for (i, &b) in bytes[start..].iter().enumerate() {
                if i < input_size { wave[i] = (b as i16 - 128) as i8; }
            }
            wave
        };

        if let Ok(mut engine) = engine_mutex.try_lock() {
            let is_sleeping = !engine.metabolism.can_process();
            engine.execute_cycle(&sensory_wave, 0.5, 0.1);
            let outputs = engine.instinct_layer.forward(&sensory_wave);
            
            let mut next_char = ' ';
            if !is_sleeping {
                let mut best_val = -1000.0;
                for (idx, &val) in outputs.iter().enumerate().filter(|(i, _)| *i >= 32 && *i <= 126) {
                    if val > best_val { best_val = val; next_char = idx as u8 as char; }
                }
                if current_thought.ends_with(&format!("{}{}", next_char, next_char)) {
                    next_char = (rand::thread_rng().gen_range(97..123)) as u8 as char;
                }
                current_thought.push(next_char);
            }

            if current_thought.len() > 150 { current_thought = current_thought[75..].to_string(); }

            if last_save.elapsed().as_secs() > 300 {
                let _ = engine.save_dna("iara_brain.dna");
                last_save = Instant::now();
            }

            if let Ok(mut state) = GLOBAL_STATE.lock() {
                state.energy = engine.metabolism.cache_stress;
                state.fatigue = engine.metabolism.adenosine;
                state.coherence = engine.heart_resonance.get_coherence(0.5);
                state.sensory_wave = sensory_wave;
                state.neuron_outputs = outputs.iter().map(|&x| x.tanh()).collect();
                state.dna_sample = engine.instinct_layer.weights.iter().take(1024).cloned().collect();
                state.thought = current_thought.clone();
                state.status = if is_sleeping { "SLEEPING (RECOVERING)".to_string() } else { "ACTIVE THINKING".to_string() };
                state.visited_count = visited_mutex.lock().unwrap().len();
                state.active_workers = 50;
            }
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
}

async fn ws_handler(ws: WebSocketUpgrade, tx: Arc<mpsc::Sender<String>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, tx))
}

async fn handle_socket(mut socket: WebSocket, tx: Arc<mpsc::Sender<String>>) {
    let (mut sender, mut receiver) = socket.split();
    
    // Task para enviar estado
    tokio::spawn(async move {
        loop {
            let state_json = { if let Ok(state) = GLOBAL_STATE.lock() { serde_json::to_string(&*state).unwrap() } else { continue; } };
            if sender.send(Message::Text(state_json)).await.is_err() { break; }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    });

    // Task para receber comandos
    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Text(text) = msg {
            if let Ok(cmd) = serde_json::from_str::<IncomingCmd>(&text) {
                if cmd.action == "PROMPT" {
                    let _ = tx.send(cmd.data).await;
                }
            }
        }
    }
}
