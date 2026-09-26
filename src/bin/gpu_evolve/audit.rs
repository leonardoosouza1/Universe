//! VISION AUDIT (3 axes) — enabled by `--audit-vision`.
//!
//! A channel is APPROVED if and only if:
//!   (a) it varies over TIME (σ_temporal > 2.0 across ticks)
//!   (b) it varies BETWEEN BEINGS at the last tick (σ_between > 2.0 with beings spread out)
//!   (c) it has a documented SOURCE (some WGSL kernel writes that index)
//!
//! The previous version measured (b) a single time at tick 0, when the beings are still
//! in identical positions — it produced false dead verdicts for correct proprioceptive
//! channels.
//!
//! Accumulation structure: per tick, records being 0's wave (temporal variance) and the
//! between-beings variance of the current tick (last tick = spread population = best
//! measurement).

use universe::perception::channels::source_for as channel_source_for;

pub const N_CH: usize = universe::config::iara::brain::SENSORY_WAVE_SIZE;

/// Static table: which channels each WGSL kernel writes.
///
/// Delegated to `perception::channels::source_for()` — single source of truth.
/// Kept as a local wrapper so this function's internal call sites don't break.
#[inline]
fn canal_tem_fonte(ch: usize) -> &'static str {
    channel_source_for(ch)
}

/// Audit accumulator — Mutex-protected for access from the tick loop.
struct AuditAcum {
    /// Temporal history of being 0: n_ticks × N_CH i8 values.
    hist_ser0: Vec<[i8; N_CH]>,
    /// Snapshot of the last tick with ≥4 ALIVE beings: N_CH × n_beings.
    ///
    /// Original criterion (b): "last tick of the race". Problem: if everyone dies, the
    /// last tick has beings frozen in an identical state (brake=1, throttle=0) →
    /// σ_between≈0 → the channel fails unfairly. Replaced by "last tick with ≥4 alive",
    /// which guarantees the population is still moving and differentiated.
    ultimo_tick_4vivos: Vec<[i8; N_CH]>,
    n_ticks: usize,
}

impl AuditAcum {
    fn new() -> Self {
        Self {
            hist_ser0: Vec::new(),
            ultimo_tick_4vivos: Vec::new(),
            n_ticks: 0,
        }
    }
}

static AUDIT_ACUM: std::sync::LazyLock<std::sync::Mutex<AuditAcum>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(AuditAcum::new()));

/// Bounds the accumulator's memory (risk R6 of REFACTOR_PLAN.md): without this,
/// `--audit-vision` in a battery of 100 generations × 8.000 ticks accumulates
/// 800.000 entries × 208 bytes ≈ 166 MB in `hist_ser0`, never freeing.
///
/// Called once at the START of each generation (main.rs). Side effect: the final
/// report now reflects the LAST generation run, not the sum of all of them —
/// which is the useful behaviour (an already evolved population, not diluted by
/// the first random individuals).
pub fn nova_geracao() {
    let mut g = AUDIT_ACUM.lock().unwrap();
    g.hist_ser0.clear();
    g.ultimo_tick_4vivos.clear();
    g.n_ticks = 0;
}

/// Called every tick when `--audit-vision` is active.
/// Records being 0's wave (temporal variance) and a snapshot of everyone (between-beings
/// variance). `n_vivos` = how many beings have life_left > 0 this tick.
pub fn auditoria_acumular(ondas: &[[i8; N_CH]], n: usize, ser_vivo: usize, n_vivos: usize) {
    let mut g = AUDIT_ACUM.lock().unwrap();
    if n == 0 {
        return;
    }
    // Wave of the first ALIVE being — stable temporal trajectory (a dead being stays constant).
    g.hist_ser0.push(ondas[ser_vivo.min(n - 1)]);
    // Snapshot of the current tick only if ≥4 beings are alive.
    // New criterion (b): "last tick with ≥4 alive" instead of "last tick of the race".
    // If everyone dies, the last tick has frozen beings (brake=1) → σ_between≈0 → false failure.
    // With ≥4 the population is still differentiated and the variance measures the channel,
    // not the death mechanism.
    if n_vivos >= 4 {
        g.ultimo_tick_4vivos = ondas[..n].to_vec();
    }
    g.n_ticks += 1;
}

/// Prints the final report after all the ticks.
pub fn auditoria_relatorio() {
    let g = AUDIT_ACUM.lock().unwrap();
    let n_ticks = g.n_ticks;
    if n_ticks == 0 {
        println!("[audit] no tick accumulated — enable --audit-vision");
        return;
    }
    let n_seres = g.ultimo_tick_4vivos.len();
    println!();
    println!(
        "╔══ VISION AUDIT — FINAL REPORT ({n_ticks} ticks, {n_seres} beings, {N_CH} channels) ══"
    );

    // ── (a) Temporal variance: standard deviation across ticks, being 0 ──────
    let mut sigma_temporal = [0.0f32; N_CH];
    for ch in 0..N_CH {
        if g.hist_ser0.is_empty() {
            break;
        }
        let vals: Vec<f32> = g.hist_ser0.iter().map(|f| f[ch] as f32).collect();
        let m = vals.iter().sum::<f32>() / vals.len() as f32;
        let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f32>() / vals.len() as f32;
        sigma_temporal[ch] = var.sqrt();
    }

    // ── (b) Between-beings variance: standard deviation at the last tick with ≥4 alive ─
    //
    // Previous criterion: "last tick of the race". If everyone dies, the last tick
    // has beings frozen at (brake=1, throttle=0) → σ_between≈0 → false failure.
    // New criterion: "last tick with ≥4 alive beings", where the population is still
    // differentiated and the variance measures the channel, not the death mechanism.
    let mut sigma_between = [0.0f32; N_CH];
    if !g.ultimo_tick_4vivos.is_empty() {
        let n = g.ultimo_tick_4vivos.len() as f32;
        for ch in 0..N_CH {
            let vals: Vec<f32> = g.ultimo_tick_4vivos.iter().map(|f| f[ch] as f32).collect();
            let m = vals.iter().sum::<f32>() / n;
            let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f32>() / n;
            sigma_between[ch] = var.sqrt();
        }
    }

    // ── (c) Source ────────────────────────────────────────────────────────────
    // Classifies by failure reason.
    let thr = 2.0_f32; // minimum σ to consider "varies"
    let mut aprovados: Vec<usize> = Vec::new();
    let mut falha_temporal: Vec<usize> = Vec::new();
    let mut falha_between: Vec<usize> = Vec::new();
    let mut falha_fonte: Vec<usize> = Vec::new();
    let mut falha_multi: Vec<usize> = Vec::new(); // multiple failures

    for ch in 0..N_CH {
        let ok_t = sigma_temporal[ch] >= thr;
        let ok_e = sigma_between[ch] >= thr;
        let ok_f = canal_tem_fonte(ch) != "sem_fonte";
        match (ok_t, ok_e, ok_f) {
            (true, true, true) => aprovados.push(ch),
            (false, true, true) => falha_temporal.push(ch),
            (true, false, true) => falha_between.push(ch),
            (true, true, false) => falha_fonte.push(ch),
            _ if !ok_f => falha_fonte.push(ch),
            _ => falha_multi.push(ch),
        }
    }

    println!("║");
    println!("║ Approval criteria:");
    println!("║   (a) σ_temporal ≥ {thr:.0} (varies across ticks, being 0)");
    println!("║   (b) σ_between  ≥ {thr:.0} (varies between beings — last tick with ≥4 alive)");
    println!("║   (c) source     = a WGSL kernel documents a write to the channel");
    println!("║");
    println!("║ APPROVED: {}/{N_CH}", aprovados.len());

    // Prints contiguous ranges of approved channels
    if !aprovados.is_empty() {
        let faixas = faixas_contiguas(&aprovados);
        let faixas_str: Vec<String> = faixas
            .iter()
            .map(|(a, b)| {
                if a == b {
                    format!("{a}")
                } else {
                    format!("{a}-{b}")
                }
            })
            .collect();
        println!("║   {}", faixas_str.join(" "));
    }

    println!("║");
    println!("║ FAILED by failure kind:");

    if !falha_temporal.is_empty() {
        println!(
            "║   [no temporal variance — constant in time] {} channels:",
            falha_temporal.len()
        );
        let faixas = faixas_contiguas(&falha_temporal);
        for (a, b) in &faixas {
            if a == b {
                println!(
                    "║     ch{a:3}  σ_t={:.1} σ_e={:.1}  source={}",
                    sigma_temporal[*a],
                    sigma_between[*a],
                    canal_tem_fonte(*a)
                );
            } else {
                println!(
                    "║     ch{a:3}..{b:3}  (ex: σ_t={:.1}  σ_e={:.1})",
                    sigma_temporal[*a], sigma_between[*a]
                );
            }
        }
    }

    if !falha_between.is_empty() {
        println!(
            "║   [no variance between beings at the last tick] {} channels:",
            falha_between.len()
        );
        let faixas = faixas_contiguas(&falha_between);
        for (a, b) in &faixas {
            if a == b {
                println!(
                    "║     ch{a:3}  σ_t={:.1} σ_e={:.1}  source={}",
                    sigma_temporal[*a],
                    sigma_between[*a],
                    canal_tem_fonte(*a)
                );
            } else {
                println!(
                    "║     ch{a:3}..{b:3}  (ex: σ_t={:.1}  σ_e={:.1})",
                    sigma_temporal[*a], sigma_between[*a]
                );
            }
        }
    }

    if !falha_fonte.is_empty() {
        println!("║   [no documented source] {} channels:", falha_fonte.len());
        for &ch in &falha_fonte {
            println!(
                "║     ch{ch:3}  σ_t={:.1} σ_e={:.1}  source={}",
                sigma_temporal[ch],
                sigma_between[ch],
                canal_tem_fonte(ch)
            );
        }
    }

    if !falha_multi.is_empty() {
        println!("║   [multiple failures] {} channels:", falha_multi.len());
        let faixas = faixas_contiguas(&falha_multi);
        for (a, b) in &faixas {
            if a == b {
                println!(
                    "║     ch{a:3}  σ_t={:.1} σ_e={:.1}  source={}",
                    sigma_temporal[*a],
                    sigma_between[*a],
                    canal_tem_fonte(*a)
                );
            } else {
                println!(
                    "║     ch{a:3}..{b:3}  (ex: σ_t={:.1}  σ_e={:.1}  source={})",
                    sigma_temporal[*a],
                    sigma_between[*a],
                    canal_tem_fonte(*a)
                );
            }
        }
    }

    // Most informative channels (highest σ_temporal)
    let mut ranking_t: Vec<(usize, f32)> = (0..N_CH).map(|ch| (ch, sigma_temporal[ch])).collect();
    ranking_t.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let topo: Vec<String> = ranking_t
        .iter()
        .take(10)
        .map(|(ch, s)| format!("ch{ch}(σ_t{s:.0})"))
        .collect();
    println!("║");
    println!("║ Top-10 σ_temporal: {}", topo.join(" "));

    println!("╚══");
}

/// Exists only so `auditoria_relatorio()` can decide whether there is accumulated data
/// even when the `--audit-vision` flag has already been consulted and cleared elsewhere.
pub fn n_ticks_acumulados() -> usize {
    AUDIT_ACUM.lock().unwrap().n_ticks
}

fn faixas_contiguas(idxs: &[usize]) -> Vec<(usize, usize)> {
    if idxs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut ini = idxs[0];
    let mut ant = idxs[0];
    for &c in idxs.iter().skip(1) {
        if c != ant + 1 {
            out.push((ini, ant));
            ini = c;
        }
        ant = c;
    }
    out.push((ini, ant));
    out
}
