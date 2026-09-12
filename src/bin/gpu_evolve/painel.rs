//! Real-time HTTP panel — REFACTOR_PLAN.md §E5 / §3, enabled by `--panel`.
//!
//! Minimalist HTTP server (`std::net::TcpListener`, no framework) on its own thread
//! inside the `gpu_evolve` process. It exists so Leonardo can SEE the race while it runs —
//! without it, the only way to follow along is reading `stdout` line by line. State shared via
//! `Arc<Mutex<..>>`; one client at a time is enough (it is diagnostics, not production).
//!
//! # Cost when off
//!
//! Without `--panel` no thread is launched, no `Arc`/`Mutex` is allocated — `main.rs` only
//! builds `Option<PainelHandle>` and every update point is `if let Some(p) = &painel`.
//!
//! # Two levels of activation detail (E4)
//!
//! `PainelState` carries only the 4 L1 summary numbers (`dead_pct_l1` etc, exactly the fields
//! of REFACTOR_PLAN.md §3.2) — it is what `/api/status` returns, consumed by the numeric
//! indicator at the top of the panel. `/api/activation` returns `ActivationSnapshot`, which has
//! BOTH layers (L1 and L2) with a 32-bin histogram each — it is what the alarm table and the
//! panel's histogram consume. `update_activation()` writes to both at once, so they
//! never diverge.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use serde::Serialize;

/// "Per-generation" + "per-tick" state — everything `/api/status` returns. Fields match
/// REFACTOR_PLAN.md §3.2, in the same order.
#[derive(Default, Serialize, Clone)]
pub struct PainelState {
    // --- generation status (updated at the end of each generation) ---
    // `gen` is a reserved word since edition 2024 (generators syntax) — the field is
    // named `geracao` for that reason, not by naming preference.
    pub geracao: u32,
    pub mediana_m: f64,
    pub elite_m: f64,
    pub mortos_pct: f32,
    /// Mean progress / sensory pilot progress. `gpu_evolve` does not compute this
    /// ratio anywhere in the production loop today (it is `replay_mapa.rs::F58` that does
    /// it, OFFLINE, over an already-recorded `.replay`) — stays at 0.0 until some real caller
    /// feeds `update_obediencia()`. The spec's "if available": it is not.
    pub obediencia_pct: f32,
    /// Fraction of the population that has already crossed EACH sector, in order (`setores[s]` =
    /// fraction with `checkpoint > s`). It is not literally "mean fitness per sector" — we chose
    /// this progress curve because it is what `checkpoint: Vec<u32>` already contains without
    /// recomputing anything, and it answers the question the real race cares about: WHERE the
    /// population gets stuck. Documented here so the field name does not get misread.
    pub setores: Vec<f32>,
    /// OLS slope (m/gen) of the median up to this generation — `universe::gpu::evolution::linear_slope`
    /// over the accumulated history (0.0 with fewer than 2 generations).
    pub slope: f64,

    // --- tick snapshot (updated every N ticks, default N=64) ---
    pub tick: u64,
    pub vivos: u32,
    pub vel_mediana: f32,
    pub life_mediana: f32,
    pub fuel_mediana: f32,

    // --- activation (updated at the end of each generation, only with --activation) ---
    pub activation_alarms: Vec<String>,
    pub dead_pct_l1: f32,
    pub saturated_pct_l1: f32,
    pub spectral_radius: f32,
    pub snr_i8: f32,

    /// Track layout in world coordinates (x, z) — sent once, in `set_track()`,
    /// and reused for the rest of the race: the world does not change between generations. It rides
    /// along with `/api/status` (not a separate endpoint) because the HTML panel already reads that
    /// endpoint at 1 Hz and the layout is small (a few KB) — replicating it on another route would only duplicate the poll.
    pub waypoints: Vec<[f32; 2]>,
}

/// One being, at one instant — what `/api/beings` returns.
#[derive(Default, Serialize, Clone)]
pub struct BeingSnapshot {
    pub id: usize,
    pub x: f32,
    pub z: f32,
    pub vel: f32,
    /// Life normalized in [0, 1] (`life_left / life_cap()`), not raw ticks — the HTML
    /// colors the dot directly from this value without needing to know the cap.
    pub life: f32,
    pub vivo: bool,
}

/// One layer (L1 or L2) inside `/api/activation`.
#[derive(Default, Serialize, Clone)]
pub struct LayerSnapshot {
    pub name: String,
    pub dead_pct: f32,
    pub saturated_pct: f32,
    pub spectral_radius: f32,
    pub effective_rank: usize,
    pub snr_i8: f32,
    /// 32 bins, same aggregate as `analysis::activation::LayerStats::histogram`.
    pub histogram: Vec<u32>,
    /// INSTANTANEOUS activation of each neuron of the observed being, in [-1, 1].
    ///
    /// Leonardo: *"cerebro.html cannot be static — I would like to see the brain in
    /// operation, how it becomes active on the track, what activates."* The aggregates above
    /// (dead/saturated/histogram) tell the layer's HISTORY; this vector is the NOW:
    /// which neurons are firing this tick, while the being drives. It is what
    /// turns the panel from a report into an observation.
    pub live: Vec<f32>,
    /// Mean activation per neuron over the generation — separates "fired now" from
    /// "always fires". A neuron with high mean and zero variance is a bias, not a sensor.
    pub mean_per_neuron: Vec<f32>,
}

/// What `/api/activation` returns — more detail than the 4 summary fields of `PainelState`.
#[derive(Default, Serialize, Clone)]
pub struct ActivationSnapshot {
    pub layers: Vec<LayerSnapshot>,
    pub alarms: Vec<String>,
}

/// HTML embedded in the binary — zero I/O at runtime, zero CDN (canvas + pure JS).
const HTML: &str = include_str!("painel.html");

/// Panel handle: what `main.rs`/`context.rs` call to publish data. Launching
/// (`launch`) only makes sense with `--panel`; without the flag `main.rs` never builds this.
pub struct PainelHandle {
    state: Arc<Mutex<PainelState>>,
    beings: Arc<Mutex<Vec<BeingSnapshot>>>,
    activation: Arc<Mutex<ActivationSnapshot>>,
}

impl PainelHandle {
    /// Starts the `TcpListener` on a separate thread and returns the handle immediately — does
    /// not block `main()`. If the port is already in use, warns on stderr and turns the panel off
    /// (the rest of the process keeps running: a diagnostic must not take down the race).
    pub fn launch(port: u16) -> Self {
        let state = Arc::new(Mutex::new(PainelState::default()));
        let beings = Arc::new(Mutex::new(Vec::<BeingSnapshot>::new()));
        let activation = Arc::new(Mutex::new(ActivationSnapshot::default()));

        let state_l = Arc::clone(&state);
        let beings_l = Arc::clone(&beings);
        let activation_l = Arc::clone(&activation);

        std::thread::spawn(move || {
            let listener = match TcpListener::bind(("0.0.0.0", port)) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!(
                        "[painel] ERROR opening port {port}: {e} — panel OFF, \
                               the race continues without it."
                    );
                    return;
                }
            };
            println!(
                "[painel] serving at http://localhost:{port}  (Ctrl+C does not take down just the panel — it kills the whole process)"
            );
            for conn in listener.incoming() {
                match conn {
                    Ok(stream) => serve_one(stream, &state_l, &beings_l, &activation_l),
                    Err(_) => continue, // connection dropped mid-accept — ignore and move on.
                }
            }
        });

        Self {
            state,
            beings,
            activation,
        }
    }

    /// Track layout — call ONCE, right after `VoxelWorld::new()`, before the
    /// generation loop.
    pub fn set_track(&self, waypoints_world: &[bevy::math::Vec2]) {
        let mut st = self.state.lock().unwrap();
        st.waypoints = waypoints_world.iter().map(|w| [w.x, w.y]).collect();
    }

    /// End of generation: the numbers that today already become `println!("gen ...")` in `main.rs`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_gen(
        &self,
        geracao: u32,
        mediana_m: f64,
        elite_m: f64,
        mortos_pct: f32,
        obediencia_pct: f32,
        setores: Vec<f32>,
        slope: f64,
    ) {
        let mut st = self.state.lock().unwrap();
        st.geracao = geracao;
        st.mediana_m = mediana_m;
        st.elite_m = elite_m;
        st.mortos_pct = mortos_pct;
        st.obediencia_pct = obediencia_pct;
        st.setores = setores;
        st.slope = slope;
    }

    /// L1/L2 activation (E4, only called when `--activation` is on). Feeds both the
    /// summary in `PainelState` (L1) and the full detail of `/api/activation` (L1+L2).
    pub fn update_activation(&self, layers: Vec<LayerSnapshot>, alarms: Vec<String>) {
        if let Some(l1) = layers.iter().find(|l| l.name == "L1") {
            let mut st = self.state.lock().unwrap();
            st.dead_pct_l1 = l1.dead_pct;
            st.saturated_pct_l1 = l1.saturated_pct;
            st.spectral_radius = l1.spectral_radius;
            st.snr_i8 = l1.snr_i8;
            st.activation_alarms = alarms.clone();
        }
        let mut act = self.activation.lock().unwrap();
        act.layers = layers;
        act.alarms = alarms;
    }

    /// Every N ticks (`main.rs` decides N, default 64): positions + speed + life of
    /// ALL beings, plus the median aggregates that go into `PainelState`.
    pub fn update_tick(&self, tick: u64, snaps: Vec<BeingSnapshot>, fuel_mediana: f32) {
        let vivos = snaps.iter().filter(|b| b.vivo).count() as u32;
        let vel_mediana = mediana(snaps.iter().filter(|b| b.vivo).map(|b| b.vel));
        let life_mediana = mediana(snaps.iter().filter(|b| b.vivo).map(|b| b.life));
        {
            let mut st = self.state.lock().unwrap();
            st.tick = tick;
            st.vivos = vivos;
            st.vel_mediana = vel_mediana;
            st.life_mediana = life_mediana;
            st.fuel_mediana = fuel_mediana;
        }
        *self.beings.lock().unwrap() = snaps;
    }
}

/// Median of an f32 iterator — an empty population (nobody alive) returns 0.0 instead of
/// panicking in `sort_by`'s `unwrap()`.
fn mediana(xs: impl Iterator<Item = f32>) -> f32 {
    let mut v: Vec<f32> = xs.collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Extracts the path from the first line of the HTTP request (`GET /api/status HTTP/1.1`).
/// Minimal parsing on purpose — this server only needs to distinguish 4 fixed routes.
fn extract_path(req: &str) -> &str {
    req.lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
}

/// Serves ONE connection and closes. Minimal HTTP/1.1: status line + Content-Type + Content-Length
/// + body. Explicit `Connection: close` — without keep-alive there is no reason for the client to wait
/// for another response on the same connection, and closing early avoids hanging connections piling up.
fn serve_one(
    mut stream: TcpStream,
    state: &Arc<Mutex<PainelState>>,
    beings: &Arc<Mutex<Vec<BeingSnapshot>>>,
    activation: &Arc<Mutex<ActivationSnapshot>>,
) {
    let mut buf = [0u8; 1024];
    let n = match stream.read(&mut buf) {
        Ok(n) if n > 0 => n,
        _ => return,
    };
    let req = String::from_utf8_lossy(&buf[..n]);
    let path = extract_path(&req);

    let (status, ct, body): (&str, &str, String) = match path {
        "/" => ("200 OK", "text/html; charset=utf-8", HTML.to_string()),
        "/api/status" => {
            let st = state.lock().unwrap().clone();
            (
                "200 OK",
                "application/json",
                serde_json::to_string(&st).unwrap_or_else(|_| "{}".into()),
            )
        }
        "/api/activation" => {
            let act = activation.lock().unwrap().clone();
            (
                "200 OK",
                "application/json",
                serde_json::to_string(&act).unwrap_or_else(|_| "{}".into()),
            )
        }
        "/api/beings" => {
            let b = beings.lock().unwrap().clone();
            (
                "200 OK",
                "application/json",
                serde_json::to_string(&b).unwrap_or_else(|_| "[]".into()),
            )
        }
        _ => ("404 Not Found", "text/plain", "not found".to_string()),
    };

    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{body}",
        body.len(),
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}
