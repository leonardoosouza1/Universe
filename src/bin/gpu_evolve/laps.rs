//! Per-lap telemetry (PRD C0, `docs/PRD-CEREBRO-NEUROMOD.md` §C0).
//!
//! The hippocampus experiment is decided by lap 2 against lap 1 and nothing here could
//! answer it: `fantasma.rs:101-113` keeps only the FIRST crossing of each sector (a
//! reference trail, not a per-lap ledger) and `ReplayFrame` only lets it be derived
//! offline, for the few beings a run records. This is the missing ledger — one
//! [`LapRecord`] per COMPLETED lap per being; the lap in progress is dropped by
//! [`LapTracker::finish`], because an unfinished lap has no comparable time or distance
//! and counting it would inflate the very ratio the experiment reads.
//!
//! Pure: no GPU, no globals, no clock. Fed at sites that already compute the values
//! (`main.rs` crossing and lap close, `context.rs` surface/damage/dry), so the wiring
//! copies no formula; [`write_jsonl`] is the only I/O and takes its path explicitly.

use crate::statics::N_CHECKPOINTS;
use std::io::Write;

/// Sector slots in a [`LapRecord`]; single source of truth is `statics::N_CHECKPOINTS`.
/// `--portal-curva` runs more gates than this (43 curve portals against 24 uniform
/// sectors); [`LapTracker::on_sector`] folds with `% N_SECTORS`, the same fold the
/// crossing site applies before pushing into `cruzamentos` (`main.rs:3060`) — under that
/// flag several portals share a slot: totals stay right, the per-slot split does not.
pub const N_SECTORS: usize = N_CHECKPOINTS as usize;

/// One COMPLETED lap of one being. `lap` is 1-based; ticks are the generation's own tick
/// index (64 Hz), so `tick_end - tick_start` is the lap time; `dist` is odometer metres
/// between them; `sector_ticks[s]` is ticks inside sector `s`; `damage` sums the per-tick
/// hidden-damage increments; `dry_events` counts tank-empty crossings.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct LapRecord {
    pub lap: u32,
    pub tick_start: u32,
    pub tick_end: u32,
    pub dist: f32,
    pub sector_ticks: [u32; N_SECTORS],
    pub grass_ticks: u32,
    pub damage: f32,
    pub dry_events: u32,
}

/// The lap in progress; becomes a [`LapRecord`] when `on_lap` closes it. Every `on_tick`
/// is charged to `sector`, the sector the being is in.
#[derive(Clone, Default)]
struct OpenLap {
    lap: u32,
    tick_start: u32,
    dist_start: f32,
    sector: usize,
    sector_ticks: [u32; N_SECTORS],
    grass_ticks: u32,
    damage: f32,
    dry_events: u32,
}

impl OpenLap {
    fn new(lap: u32, tick_start: u32, dist_start: f32, sector: usize) -> Self {
        Self {
            lap,
            tick_start,
            dist_start,
            sector,
            ..Default::default()
        }
    }
}

/// Per-being lap ledger for ONE generation: create, feed through the tick loop, `finish`.
pub struct LapTracker {
    done: Vec<Vec<LapRecord>>,
    open: Vec<OpenLap>,
}

impl LapTracker {
    /// Every being starts lap 1 at tick 0, in sector 0, odometer at zero.
    pub fn new(n_beings: usize) -> Self {
        Self {
            done: vec![Vec::new(); n_beings],
            open: vec![OpenLap::new(1, 0, 0.0, 0); n_beings],
        }
    }

    /// The being entered `sector`: ticks from here on are charged there. The crossing tick
    /// stays in the signature (the call site holds it) but is not read — `sector_ticks`
    /// COUNTS `on_tick` calls, so it stays right for a being that dies mid-sector.
    /// Out-of-range `being` ignored, `sector` folded: telemetry must not abort a run.
    pub fn on_sector(&mut self, being: usize, sector: u32, _tick: u32) {
        if let Some(open) = self.open.get_mut(being) {
            open.sector = sector as usize % N_SECTORS;
        }
    }

    /// The being closed a lap at `tick`, odometer reading `dist` metres. Closes the open
    /// record and opens the next at the same tick, odometer and sector (laps tile).
    pub fn on_lap(&mut self, being: usize, tick: u32, dist: f32) {
        let (Some(open), Some(done)) = (self.open.get_mut(being), self.done.get_mut(being)) else {
            return;
        };
        let next = OpenLap::new(open.lap + 1, tick, dist, open.sector);
        let closed = std::mem::replace(open, next);
        done.push(LapRecord {
            lap: closed.lap,
            tick_start: closed.tick_start,
            tick_end: tick,
            dist: dist - closed.dist_start,
            sector_ticks: closed.sector_ticks,
            grass_ticks: closed.grass_ticks,
            damage: closed.damage,
            dry_events: closed.dry_events,
        });
    }

    /// One simulated tick of one being, charged to the sector it is currently in.
    pub fn on_tick(&mut self, being: usize, on_grass: bool, damage_delta: f32, dry: bool) {
        let Some(open) = self.open.get_mut(being) else {
            return;
        };
        open.sector_ticks[open.sector] += 1;
        open.grass_ticks += u32::from(on_grass);
        open.damage += damage_delta;
        open.dry_events += u32::from(dry);
    }

    /// Completed laps per being. The lap in progress is dropped — see the module header.
    pub fn finish(self) -> Vec<Vec<LapRecord>> {
        self.done
    }
}

/// Population summary of one generation — what the `laps:` log line prints. `median_laps`
/// counts ALL beings, zero-lap ones included; `lap2_over_lap1_dist` is the median of
/// `lap2.dist / lap1.dist` over the `n_ge2` beings with at least two completed laps.
#[derive(Debug, Clone, PartialEq)]
pub struct LapSummary {
    pub median_laps: f32,
    pub lap2_over_lap1_dist: Option<f32>,
    pub n_ge2: usize,
}

impl LapSummary {
    /// The generation's one-line summary, in the format the PRD fixes:
    /// `laps: med 1.80 | lap2/lap1 dist +3.1 % | >=2V 31/60`.
    pub fn log_line(&self, n_beings: usize) -> String {
        let ratio = match self.lap2_over_lap1_dist {
            Some(r) => format!("{:+.1} %", (r - 1.0) * 100.0),
            None => "n/a".to_string(),
        };
        format!(
            "laps: med {:.2} | lap2/lap1 dist {} | >=2V {}/{}",
            self.median_laps, ratio, self.n_ge2, n_beings
        )
    }
}

/// Summarises what [`LapTracker::finish`] returned. Empty input gives `0 / None / 0`: a
/// generation where nobody closed a lap is a measurement, not a failure. A zero-metre lap
/// 1 is dropped from the ratio instead of becoming an infinity.
pub fn summary(records: &[Vec<LapRecord>]) -> LapSummary {
    let mut laps: Vec<f32> = records.iter().map(|r| r.len() as f32).collect();
    let mut ratios: Vec<f32> = records
        .iter()
        .filter_map(|r| match (r.first(), r.get(1)) {
            (Some(l1), Some(l2)) if l1.dist > 0.0 => Some(l2.dist / l1.dist),
            _ => None,
        })
        .collect();
    LapSummary {
        median_laps: median(&mut laps).unwrap_or(0.0),
        lap2_over_lap1_dist: median(&mut ratios),
        n_ge2: records.iter().filter(|r| r.len() >= 2).count(),
    }
}

/// Median, averaging the two middle values on an even count — deliberately NOT the
/// `v[len / 2]` upper median of the `[setores]` line (`main.rs:3477`): this one feeds a
/// paired comparison where rounding 1.5 up to 2 hides the half-lap difference looked for.
/// On an odd count both indices land on the same element, so the value is exact.
fn median(v: &mut [f32]) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    Some(0.5 * (v[(n - 1) / 2] + v[n / 2]))
}

/// Writes the ledger to `<out>/laps_gen<N>.jsonl` as JSON Lines, one object per being:
/// `{"being":i,"seed":s,"laps":[…]}`; `seeds[i]` is the population seed
/// (`population_seed_for`), a missing entry written as `0`. The two scalars are formatted
/// by hand (both numeric, nothing to escape) to keep that key order — `json!` sorts.
pub fn write_jsonl(path: &str, records: &[Vec<LapRecord>], seeds: &[u64]) -> std::io::Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    for (i, laps) in records.iter().enumerate() {
        let seed = seeds.get(i).copied().unwrap_or(0);
        let laps_json = serde_json::to_string(laps).map_err(std::io::Error::other)?;
        writeln!(w, "{{\"being\":{i},\"seed\":{seed},\"laps\":{laps_json}}}")?;
    }
    w.flush()
}
