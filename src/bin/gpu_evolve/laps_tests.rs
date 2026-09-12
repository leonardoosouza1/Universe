//! Acceptance tests for `laps.rs` (PRD C0, `docs/PRD-CEREBRO-NEUROMOD.md` §C0).
//!
//! A separate file, not a `#[cfg(test)]` block inside `laps.rs`, only because that
//! module is capped at 200 lines by the PRD. Run by `cargo test --release --bin
//! gpu_evolve` like `tests.rs`, which an integration test under `tests/` would not be:
//! `laps` is a module of the binary, not of the library.

use crate::laps::{LapRecord, LapTracker, N_SECTORS, summary};

/// Drives one being through `ticks` ticks of sector `sector`, then crosses into the
/// next one. Returns the tick index it stopped at, so a caller can chain sectors.
fn run_sector(t: &mut LapTracker, being: usize, sector: u32, from_tick: u32, ticks: u32) -> u32 {
    for _ in 0..ticks {
        t.on_tick(being, false, 0.0, false);
    }
    let tick = from_tick + ticks;
    t.on_sector(being, sector + 1, tick);
    tick
}

/// The synthetic sequence the PRD asks for: crossings and one closed lap produce the
/// expected record, ticks landing in the sector the being was in.
#[test]
fn a_synthetic_lap_produces_the_expected_record() {
    let mut t = LapTracker::new(1);
    // Sector 0 for 10 ticks, sector 1 for 20, sector 2 for 5, then the lap closes.
    let tick = run_sector(&mut t, 0, 0, 0, 10);
    let tick = run_sector(&mut t, 0, 1, tick, 20);
    let tick = run_sector(&mut t, 0, 2, tick, 5);
    // Two more ticks inside sector 3, one of them on grass and taking damage, and the
    // lap closes at tick 37 with the odometer at 300 m.
    t.on_tick(0, true, 0.25, false);
    t.on_tick(0, false, 0.25, true);
    t.on_lap(0, tick + 2, 300.0);

    let records = t.finish();
    assert_eq!(records.len(), 1, "one being in, one entry out");
    let mut expected_sectors = [0u32; N_SECTORS];
    expected_sectors[0] = 10;
    expected_sectors[1] = 20;
    expected_sectors[2] = 5;
    expected_sectors[3] = 2;
    let expected = LapRecord {
        lap: 1,
        tick_start: 0,
        tick_end: 37,
        dist: 300.0,
        sector_ticks: expected_sectors,
        grass_ticks: 1,
        damage: 0.5,
        dry_events: 1,
    };
    assert_eq!(records[0], vec![expected]);
    println!(
        "[laps] lap 1: {} ticks, {:.0} m, sectors 0-3 = {:?}",
        records[0][0].tick_end - records[0][0].tick_start,
        records[0][0].dist,
        &records[0][0].sector_ticks[0..4]
    );
}

/// A being that drives but never closes a lap records nothing: the partial lap is not a
/// `LapRecord`, because it has no comparable time or distance.
#[test]
fn a_being_that_never_closes_a_lap_records_nothing() {
    let mut t = LapTracker::new(2);
    for _ in 0..500 {
        t.on_tick(0, true, 0.01, false);
    }
    t.on_sector(0, 1, 100);
    t.on_sector(0, 2, 200);
    let records = t.finish();
    assert!(records[0].is_empty(), "500 ticks, no lap closed, no record");
    assert!(
        records[1].is_empty(),
        "a being that never moved has no laps"
    );
    let s = summary(&records);
    assert_eq!(s.median_laps, 0.0);
    assert_eq!(s.n_ge2, 0);
    assert_eq!(s.lap2_over_lap1_dist, None);
    println!("[laps] no lap closed: {}", s.log_line(records.len()));
}

/// Builds a being's completed laps from one distance per lap, 1000 ticks each.
fn being_with_laps(dists: &[f32]) -> Vec<LapRecord> {
    dists
        .iter()
        .enumerate()
        .map(|(k, &dist)| LapRecord {
            lap: k as u32 + 1,
            tick_start: k as u32 * 1000,
            tick_end: (k as u32 + 1) * 1000,
            dist,
            sector_ticks: [0; N_SECTORS],
            grass_ticks: 0,
            damage: 0.0,
            dry_events: 0,
        })
        .collect()
}

/// `lap2_over_lap1_dist` is `None` while nobody has two laps, and the median ratio once
/// three beings do — the beings with a single lap must not enter the statistic.
#[test]
fn lap2_over_lap1_is_none_until_three_beings_have_two_laps() {
    let one_lap_only = vec![
        being_with_laps(&[1000.0]),
        being_with_laps(&[900.0]),
        being_with_laps(&[]),
    ];
    assert_eq!(summary(&one_lap_only).lap2_over_lap1_dist, None);

    // Ratios 1.10, 1.02 and 0.90; the median is 1.02, not the mean (1.0067).
    let two_laps = vec![
        being_with_laps(&[1000.0, 1100.0]),
        being_with_laps(&[1000.0, 1020.0]),
        being_with_laps(&[1000.0, 900.0]),
        being_with_laps(&[1000.0]),
        being_with_laps(&[]),
    ];
    let s = summary(&two_laps);
    let ratio = s.lap2_over_lap1_dist.expect("three beings have two laps");
    assert!(
        (ratio - 1.02).abs() < 1e-6,
        "median of [0.90, 1.02, 1.10] is 1.02, got {ratio}"
    );
    assert_eq!(s.n_ge2, 3);
    assert_eq!(s.median_laps, 2.0, "laps per being sorted: [0, 1, 2, 2, 2]");
    println!("[laps] three beings with two laps: {}", s.log_line(5));
    assert_eq!(
        s.log_line(5),
        "laps: med 2.00 | lap2/lap1 dist +2.0 % | >=2V 3/5",
        "the PRD fixes this line format; a parser downstream depends on it"
    );
}

/// The summary of zero beings is a measurement, not a panic — an early generation where
/// the whole population dies before the first sector must still print its line.
#[test]
fn a_summary_of_zero_beings_does_not_panic() {
    let s = summary(&[]);
    assert_eq!(s.median_laps, 0.0);
    assert_eq!(s.lap2_over_lap1_dist, None);
    assert_eq!(s.n_ge2, 0);
    assert_eq!(
        s.log_line(0),
        "laps: med 0.00 | lap2/lap1 dist n/a | >=2V 0/0"
    );
    println!("[laps] empty population: {}", s.log_line(0));
}

/// `--portal-curva` crosses gates whose index goes past 24, and an out-of-range being
/// index can only be a caller bug: neither may abort a generation.
#[test]
fn out_of_range_sector_and_being_are_absorbed() {
    let mut t = LapTracker::new(1);
    t.on_sector(0, 30, 5); // 30 % 24 = 6
    t.on_tick(0, false, 0.0, false);
    t.on_sector(9, 1, 5); // no such being
    t.on_tick(9, true, 1.0, true);
    t.on_lap(9, 6, 10.0);
    t.on_lap(0, 6, 42.0);
    let records = t.finish();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0][0].sector_ticks[6], 1, "30 folded onto slot 6");
    assert_eq!(records[0][0].dist, 42.0);
}

/// The JSONL the C5 analysis reads: one line per being, keys `being`/`seed`/`laps`, and a
/// being with no lap present as an empty list instead of missing from the file. Same
/// `temp_dir` + `remove_file` pattern as the genome round trip (`tests.rs:59-63`).
#[test]
fn the_jsonl_has_one_line_per_being_with_the_documented_keys() {
    let records = vec![being_with_laps(&[1000.0, 1100.0]), being_with_laps(&[])];
    let path = std::env::temp_dir().join("universe_laps_gen_test.jsonl");
    let path = path.to_str().expect("temp path is utf-8");
    crate::laps::write_jsonl(path, &records, &[7, 9]).expect("wrote");
    let text = std::fs::read_to_string(path).expect("read back");
    let _ = std::fs::remove_file(path);

    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "one line per being, corpses included");
    let first: serde_json::Value = serde_json::from_str(lines[0]).expect("line 0 is JSON");
    assert_eq!(first["being"], 0);
    assert_eq!(first["seed"], 7);
    assert_eq!(first["laps"][1]["lap"], 2);
    assert_eq!(first["laps"][1]["dist"], 1100.0);
    let sectors = first["laps"][0]["sector_ticks"].as_array().expect("array");
    assert_eq!(sectors.len(), N_SECTORS);
    let second: serde_json::Value = serde_json::from_str(lines[1]).expect("line 1 is JSON");
    assert_eq!(second["being"], 1);
    assert_eq!(second["seed"], 9);
    assert_eq!(second["laps"].as_array().expect("empty list").len(), 0);
    println!("[laps] jsonl line 0: {}", lines[0]);
}
