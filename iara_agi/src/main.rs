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
use std::collections::{VecDeque, HashSet};
use lazy_static::lazy_static;
use tokio::io::AsyncReadExt;

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
    mode: String,
}

lazy_static! {
    static ref GLOBAL_STATE: Arc<Mutex<LabState>> = Arc::new(Mutex::new(LabState {
        energy: 0.0, fatigue: 0.0, coherence: 0.0,
        sensory_wave: Vec::new(), neuron_outputs: Vec::new(),
        dna_sample: Vec::new(), text_stream: "".to_string(),
        log: "Swarm Awakening...".to_string(),
        queue_size: 0,
        visited_count: 0,
        active_workers: 0,
        mode: "Idle".to_string(),
    }));
}

struct SwarmCore {
    engine: Arc<Mutex<SovereignEngine>>,
    queue: Arc<Mutex<VecDeque<String>>>,
    visited: Arc<Mutex<HashSet<String>>>,
}

fn is_relevant(link: &str) -> bool {
    let link_lower = link.to_lowercase();
    let blacklisted = ["donate.", "login", "register", "facebook", "twitter", "linkedin", "instagram", "share=", "print", "special:", "action=", "uselang="];
    if blacklisted.iter().any(|&b| link_lower.contains(b)) { return false; }
    let keywords = ["science", "tech", "evolution", "quantum", "neural", "physics", "ai", "wiki", "theory", "bio", "robot", "philosophy", "mathematics"];
    keywords.iter().any(|&k| link_lower.contains(k))
}

async fn digest_video_stream(url: String, engine_ptr: Arc<Mutex<SovereignEngine>>) {
    let cmd = format!("./yt-dlp -f ba --ffmpeg-location ./ffmpeg -o - \"{}\" | ./ffmpeg -i pipe:0 -f s16le -ac 1 -ar 16000 pipe:1 -loglevel quiet", url);
    let mut child = match tokio::process::Command::new("sh").arg("-c").arg(cmd).stdout(std::process::Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(_) => return,
    };
    let mut stdout = child.stdout.take().unwrap();
    let mut buffer = [0u8; 4096];
    while let Ok(n) = stdout.read(&mut buffer).await {
        if n == 0 { break; }
        let mut wave = vec![0i8; 2048];
        for i in 0..(n/2).min(2048) {
            let sample = i16::from_le_bytes([buffer[i*2], buffer[i*2+1]]);
            wave[i] = (sample >> 8) as i8;
        }
        if let Ok(mut engine) = engine_ptr.try_lock() {
            engine.execute_cycle(&wave, 1.0, 0.05);
        }
    }
    let _ = child.kill().await;
}

async fn extract_web_data(url: String) -> (String, Vec<String>) {
    let client = reqwest::Client::builder().user_agent("SovereignAGI/8.0-Stable").timeout(Duration::from_secs(5)).build().unwrap();
    if let Ok(res) = client.get(&url).send().await {
        if let Ok(html) = res.text().await {
            let document = scraper::Html::parse_document(&html);
            let text = document.select(&scraper::Selector::parse("p, h1, h2").unwrap()).map(|e| e.text().collect::<String>()).collect::<Vec<_>>().join(" ");
            let links = document.select(&scraper::Selector::parse("a[href^='http']").unwrap())
                .filter_map(|e| e.value().attr("href")).filter(|l| is_relevant(l)).map(|l| l.to_string()).collect();
            return (text, links);
        }
    }
    ("".to_string(), Vec::new())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- IARA AGI: ULTRA-STABLE SWARM ---");
    let app = Router::new().fallback_service(ServeDir::new("public")).route("/ws", get(ws_handler));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    tokio::spawn(async move { let _ = axum::serve(listener, app).await; });

    let input_size = 2048;
    let mut iara = SovereignEngine::new_scaled(input_size, 512, 128);
    let _ = crate::layers::BitLinear::load_from_file("iara_brain.json").map(|saved| iara.instinct_layer = saved);
    let core = Arc::new(SwarmCore {
        engine: Arc::new(Mutex::new(iara)),
        queue: Arc::new(Mutex::new(VecDeque::new())),
        visited: Arc::new(Mutex::new(HashSet::new())),
    });
    core.queue.lock().unwrap().push_back("https://en.wikipedia.org/wiki/Complexity_theory".to_string());

    for worker_id in 0..20 {
        let core_ptr = Arc::clone(&core);
        tokio::spawn(async move {
            let mut last_cycle = Instant::now();
            loop {
                let target = { 
                    let mut q = core_ptr.queue.lock().unwrap();
                    let v = core_ptr.visited.lock().unwrap();
                    let mut found = None;
                    while let Some(url) = q.pop_front() {
                        if !v.contains(&url) { found = Some(url); break; }
                    }
                    found
                };

                if let Some(url) = target {
                    { core_ptr.visited.lock().unwrap().insert(url.clone()); }
                    let (content, links, is_yt) = if url.contains("youtube.com") || url.contains("youtu.be") {
                        (String::new(), Vec::new(), true)
                    } else {
                        let (c, l) = extract_web_data(url.clone()).await;
                        (c, l, false)
                    };

                    if is_yt {
                        digest_video_stream(url, Arc::clone(&core_ptr.engine)).await;
                    } else if content.len() > 200 {
                        {
                            let mut q = core_ptr.queue.lock().unwrap();
                            let v = core_ptr.visited.lock().unwrap();
                            if q.len() < 800 {
                                for link in links.into_iter().take(5) {
                                    if !v.contains(&link) { q.push_back(link); }
                                }
                            }
                        }
                        let bytes = content.as_bytes();
                        for chunk in bytes.chunks(input_size) {
                            let dt = last_cycle.elapsed().as_secs_f32();
                            last_cycle = Instant::now();
                            let mut wave = vec![0i8; input_size];
                            for (i, &b) in chunk.iter().enumerate() { wave[i] = (b as i16 - 128) as i8; }
                            if let Ok(mut engine) = core_ptr.engine.try_lock() {
                                engine.execute_cycle(&wave, 1.0, dt);
                                if worker_id == 0 {
                                    if let Ok(mut state) = GLOBAL_STATE.lock() {
                                        state.visited_count = core_ptr.visited.lock().unwrap().len();
                                        state.queue_size = core_ptr.queue.lock().unwrap().len();
                                        state.text_stream = String::from_utf8_lossy(chunk).chars().take(150).collect();
                                        state.log = format!("SWARM ACTIVE | Mining: {}", url);
                                        state.mode = "CLEAN_TEXT".to_string();
                                        state.active_workers = 20;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                }
            }
        });
    }

    loop {
        tokio::time::sleep(Duration::from_secs(300)).await;
        let _ = core.engine.lock().unwrap().instinct_layer.save_to_file("iara_brain.json");
    }
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse { ws.on_upgrade(handle_socket) }
async fn handle_socket(mut socket: WebSocket) {
    loop {
        let state_json = { if let Ok(state) = GLOBAL_STATE.lock() { serde_json::to_string(&*state).unwrap() } else { continue; } };
        if socket.send(Message::Text(state_json)).await.is_err() { break; }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}
