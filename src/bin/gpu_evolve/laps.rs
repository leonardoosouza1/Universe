//! Per-lap telemetry (PRD C0, `docs/PRD-CEREBRO-NEUROMOD.md` §C0).
//!
//! The hippocampus experiment is decided by lap 2 against lap 1 and nothing here could
//! answer it: `fantasma.rs:101-113` keeps only the FIRST crossing of each sector (a
//! reference trail, not a per-lap ledger) and `ReplayFrame` only lets it be derived
//! offline, for the few beings a run records. This is the missing ledger — one
//! [`LapRecord`] per lap per being: the laps it CLOSED, then always one more for the lap
//! still open when the generation ends, flagged `partial` (`"partial":true` on disk).
//!
//! It was DROPPED until 2026-09-12, and that cost the per-sector metric its coverage: a
//! being closing lap 1 and dying mid lap 2 offered no second pass, so `u0c_valida_hip`
//! gen 0 measured 27 of 60 beings while 54 of 60 closed lap 1. An unfinished lap has no
//! time or distance — `tick_end - tick_start` and `dist` cover a fragment — so counting
//! it would inflate the very ratio the experiment reads: [`summary`] and
//! `tools/laps.py` keep partials out of every lap COUNT and out of `lap2/lap1`. What a
//! partial carries honestly is `sector_ticks` — every sector it reached is a real pass.
//!
//! Pure: no GPU, no globals, no clock. Fed at sites that already compute the values
//! (`main.rs` crossing and lap close, `context.rs` surface/damage/dry), so the wiring
//! copies no formula; [`write_jsonl`] is the only I/O and takes its path explicitly.

use crate::statics::N_CHECKPOINTS;
use std::io::Write;

/// Sector slots in a [`LapRecord`]; single source of truth is `statics::N_CHECKPOINTS`.
/// `--curve-portal` runs more gates than this (43 curve portals against 24 uniform
/// sectors); [`LapTracker::on_sector`] folds with `% N_SECTORS`, the same fold the
/// crossing site applies before pushing into `cruzamentos` (`main.rs:3060`) — under that
/// flag several portals share a slot: totals stay right, the per-slot split does not.
pub const N_SECTORS: usize = N_CHECKPOINTS as usize;

/// One lap of one being. `lap` is 1-based; ticks are the generation's own tick index
/// (64 Hz), so `tick_end - tick_start` is the lap time; `dist` is odometer metres between
/// them; `sector_ticks[s]` is ticks inside sector `s`; `damage` sums the per-tick
/// hidden-damage increments; `dry_events` counts tank-empty crossings.
///
/// On a `partial` record `tick_end` is the last tick the being was alive and `dist` the
/// metres of the fragment — only `sector_ticks` compares. It is the LAST field, and the
/// only new one, so a ledger written before it still reads key for key.
///
/// Its wire name was `parcial` for one day (2026-09-12) and is `partial` from the
/// English pass of the same day; `tools/laps.py` reads BOTH keys and treats a record
/// with neither as closed, so every ledger ever written still reports the same numbers.
// check-language: allow -- the sentence above quotes the retired wire name.
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
    pub partial: bool,
}

/// The lap in progress; becomes a [`LapRecord`] when `on_lap` closes it or when `finish`
/// writes it as a partial. Every `on_tick` is charged to `sector` and moves
/// `last_tick`/`last_dist` — where a partial's fragment ends, which nothing else could
/// reconstruct: a corpse stops being ticked, so the tick count of the generation is NOT
/// where this being stopped.
#[derive(Clone, Default)]
struct OpenLap {
    lap: u32,
    tick_start: u32,
    dist_start: f32,
    last_tick: u32,
    last_dist: f32,
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
            last_tick: tick_start,
            last_dist: dist_start,
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
            partial: false,
        });
    }

    /// One simulated tick of one being, charged to the sector it is currently in. `tick`
    /// and `dist` (odometer AFTER this step, the reading `on_lap` gets) are only stored:
    /// they are where a partial lap ends, never a term in a count.
    pub fn on_tick(
        &mut self,
        being: usize,
        tick: u32,
        dist: f32,
        on_grass: bool,
        damage_delta: f32,
        dry: bool,
    ) {
        let Some(open) = self.open.get_mut(being) else {
            return;
        };
        open.last_tick = tick;
        open.last_dist = dist;
        open.sector_ticks[open.sector] += 1;
        open.grass_ticks += u32::from(on_grass);
        open.damage += damage_delta;
        open.dry_events += u32::from(dry);
    }

    /// Every lap per being: the closed ones, then the still-open one as a `partial`
    /// record — ALWAYS, so a being has `closed + 1` records and coverage is readable in
    /// the ledger instead of inferred from a missing line. A being that never crossed a
    /// sector writes a partial holding only slot 0 (pre-crossing time, which
    /// `tools/laps.py` drops by default); one that never moved, a partial of zeros.
    /// Both are measurements — writing nothing is what left the old ledger unable to
    /// tell "reached no sector" from "never ran".
    pub fn finish(self) -> Vec<Vec<LapRecord>> {
        let Self { mut done, open } = self;
        for (laps, o) in done.iter_mut().zip(open) {
            laps.push(LapRecord {
                lap: o.lap,
                tick_start: o.tick_start,
                tick_end: o.last_tick,
                dist: o.last_dist - o.dist_start,
                sector_ticks: o.sector_ticks,
                grass_ticks: o.grass_ticks,
                damage: o.damage,
                dry_events: o.dry_events,
                partial: true,
            });
        }
        done
    }
}

/// Population summary of one generation — what the `laps:` log line prints. Every number
/// counts CLOSED laps only; the partial record every being now ends with is skipped, so
/// the line reads exactly as it did before partials were written. `median_laps` counts
/// ALL beings, zero-lap ones included; `lap2_over_lap1_dist` is the median of
/// `lap2.dist / lap1.dist` over the `n_ge2` beings with at least two closed laps.
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

/// Summarises what [`LapTracker::finish`] returned, PARTIALS DROPPED. Empty input gives
/// `0 / None / 0`: a generation where nobody closed a lap is a measurement, not a
/// failure. A zero-metre lap 1 is dropped from the ratio instead of becoming an infinity.
pub fn summary(records: &[Vec<LapRecord>]) -> LapSummary {
    let closed: Vec<Vec<&LapRecord>> = records
        .iter()
        .map(|r| r.iter().filter(|l| !l.partial).collect())
        .collect();
    let mut laps: Vec<f32> = closed.iter().map(|r| r.len() as f32).collect();
    let mut ratios: Vec<f32> = closed
        .iter()
        .filter_map(|r| match (r.first(), r.get(1)) {
            (Some(l1), Some(l2)) if l1.dist > 0.0 => Some(l2.dist / l1.dist),
            _ => None,
        })
        .collect();
    LapSummary {
        median_laps: median(&mut laps).unwrap_or(0.0),
        lap2_over_lap1_dist: median(&mut ratios),
        n_ge2: closed.iter().filter(|r| r.len() >= 2).count(),
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
/// `{"being":i,"seed":s,"laps":[…]}` — `laps` always ends with the `partial` record, so
/// it is never empty, not even for a being that never moved; `seeds[i]` is the seed
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
