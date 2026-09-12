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
    for k in 0..ticks {
        t.on_tick(being, from_tick + k, 0.0, false, 0.0, false);
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
    t.on_tick(0, tick, 0.0, true, 0.25, false);
    t.on_tick(0, tick + 1, 0.0, false, 0.25, true);
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
        partial: false,
    };
    // The lap that never started is still written: 2 records, the second an empty partial
    // opened by `on_lap` at tick 37 and never ticked (`closed + 1`, always).
    assert_eq!(
        records[0].len(),
        2,
        "the closed lap plus its open successor"
    );
    assert_eq!(records[0][0], expected);
    let tail = &records[0][1];
    assert!(
        tail.partial,
        "the last record of a being is always the open lap"
    );
    assert_eq!((tail.lap, tail.tick_start, tail.tick_end), (2, 37, 37));
    assert_eq!(
        tail.sector_ticks, [0u32; N_SECTORS],
        "no tick reached lap 2"
    );
    println!(
        "[laps] lap 1: {} ticks, {:.0} m, sectors 0-3 = {:?}",
        records[0][0].tick_end - records[0][0].tick_start,
        records[0][0].dist,
        &records[0][0].sector_ticks[0..4]
    );
}

/// A being that drives but never closes a lap records ONE partial — deliberately written
/// even when it holds only slot 0, so the ledger shows coverage instead of a missing line
/// (`laps.rs::finish`). It still counts as zero laps everywhere the summary looks.
#[test]
fn a_being_that_never_closes_a_lap_records_one_partial() {
    let mut t = LapTracker::new(2);
    for k in 0..500 {
        t.on_tick(0, k, 2.0 * k as f32, true, 0.01, false);
    }
    t.on_sector(0, 1, 100);
    t.on_sector(0, 2, 200);
    let records = t.finish();
    assert_eq!(records[0].len(), 1, "500 ticks, no lap closed, one partial");
    let p = &records[0][0];
    assert!(p.partial);
    assert_eq!((p.lap, p.tick_start, p.tick_end), (1, 0, 499));
    assert_eq!(p.dist, 998.0, "odometer of the last tick charged");
    assert_eq!(p.sector_ticks[0], 500, "all 500 before the first crossing");
    assert_eq!(p.sector_ticks[1], 0, "the crossings came after the ticks");
    assert_eq!(p.grass_ticks, 500);
    // A being that never moved: one partial of zeros, not an absent line.
    assert_eq!(
        records[1].len(),
        1,
        "even a being that never moved files one"
    );
    assert_eq!(records[1][0].sector_ticks, [0u32; N_SECTORS]);
    assert_eq!(records[1][0].dist, 0.0);
    let s = summary(&records);
    assert_eq!(s.median_laps, 0.0);
    assert_eq!(s.n_ge2, 0);
    assert_eq!(s.lap2_over_lap1_dist, None);
    println!("[laps] no lap closed: {}", s.log_line(records.len()));
}

/// The case the U6 amendment needs: one closed lap and HALF of the second. Two records,
/// the second `partial`, only the sectors the half-lap reached non-zero — those are the
/// second passes `tools/laps.py` may compare, sector by sector.
#[test]
fn one_lap_and_a_half_yields_a_partial_with_only_the_visited_sectors() {
    let mut t = LapTracker::new(1);
    // Lap 1: sectors 0, 1 and 2 at 100 ticks each, closing on entering sector 3.
    let tick = run_sector(&mut t, 0, 0, 0, 100);
    let tick = run_sector(&mut t, 0, 1, tick, 100);
    let tick = run_sector(&mut t, 0, 2, tick, 100);
    t.on_lap(0, tick, 1200.0);
    t.on_sector(0, 0, tick); // the lap closes into slot 0 again
    // Lap 2, half of it: slot 0 for 90 ticks, slot 1 for 80, then the being dies.
    for k in 0..90 {
        t.on_tick(0, tick + k, 1200.0 + 4.0 * k as f32, false, 0.0, false);
    }
    t.on_sector(0, 1, tick + 90);
    for k in 0..80 {
        t.on_tick(0, tick + 90 + k, 1560.0 + 4.0 * k as f32, false, 0.0, false);
    }

    let records = t.finish();
    assert_eq!(records[0].len(), 2, "one closed lap plus the half-lap");
    assert!(!records[0][0].partial, "lap 1 closed");
    let half = &records[0][1];
    assert!(half.partial, "lap 2 never closed");
    assert_eq!(half.lap, 2);
    assert_eq!((half.tick_start, half.tick_end), (300, 469));
    assert_eq!(half.dist, 676.0, "1876 - 1200 m of fragment, not a lap");
    assert_eq!(half.sector_ticks[0], 90);
    assert_eq!(half.sector_ticks[1], 80);
    let visited = half.sector_ticks.iter().filter(|&&v| v > 0).count();
    assert_eq!(visited, 2, "only the two sectors it reached are non-zero");
    // The whole point: the summary must not see a lap here.
    let s = summary(&records);
    assert_eq!(s.median_laps, 1.0, "one CLOSED lap, the partial is not one");
    assert_eq!(s.n_ge2, 0, ">=2V must not count a half-lap");
    assert_eq!(s.lap2_over_lap1_dist, None, "no lap 2 distance to divide");
    println!(
        "[laps] 1.5 laps: {} | partial sectors 0-2 = {:?}",
        s.log_line(1),
        &half.sector_ticks[0..3]
    );
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
            partial: false,
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

    // Same population as `LapTracker::finish` really hands over — every being carrying
    // its open lap as well. The line must print the SAME numbers, digit for digit: the
    // u1 ledgers were measured with it, and a partial is not a lap.
    let with_partials: Vec<Vec<LapRecord>> = two_laps
        .iter()
        .map(|laps| {
            let mut v = laps.clone();
            let lap = v.len() as u32 + 1;
            v.push(LapRecord {
                lap,
                tick_start: (lap - 1) * 1000,
                tick_end: (lap - 1) * 1000 + 400,
                dist: 380.0,
                sector_ticks: [7; N_SECTORS],
                grass_ticks: 3,
                damage: 0.1,
                dry_events: 0,
                partial: true,
            });
            v
        })
        .collect();
    assert_eq!(
        summary(&with_partials),
        s,
        "partials must not move med, lap2/lap1 or >=2V"
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

/// `--curve-portal` crosses gates whose index goes past 24, and an out-of-range being
/// index can only be a caller bug: neither may abort a generation.
#[test]
fn out_of_range_sector_and_being_are_absorbed() {
    let mut t = LapTracker::new(1);
    t.on_sector(0, 30, 5); // 30 % 24 = 6
    t.on_tick(0, 5, 42.0, false, 0.0, false);
    t.on_sector(9, 1, 5); // no such being
    t.on_tick(9, 5, 1.0, true, 1.0, true);
    t.on_lap(9, 6, 10.0);
    t.on_lap(0, 6, 42.0);
    let records = t.finish();
    assert_eq!(records.len(), 1, "the phantom being creates no entry");
    assert_eq!(records[0][0].sector_ticks[6], 1, "30 folded onto slot 6");
    assert_eq!(records[0][0].dist, 42.0);
    assert_eq!(records[0].len(), 2, "the closed lap plus its empty partial");
}

/// The JSONL the C5 analysis reads: one line per being, keys `being`/`seed`/`laps`, and a
/// being with no lap present as an empty list instead of missing from the file. Same
/// `temp_dir` + `remove_file` pattern as the genome round trip (`tests.rs:59-63`).
#[test]
fn the_jsonl_has_one_line_per_being_with_the_documented_keys() {
    let records = vec![being_with_laps(&[1000.0, 1100.0]), being_with_laps(&[])];
    // `being_with_laps` builds closed laps only, so line 1 is the pre-partial shape the
    // OLD binary wrote: the key set below is what the u1 ledgers already on disk carry.
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
    assert_eq!(first["laps"][0]["partial"], false, "wire name is `partial`");
    // The exact BYTES of a closed lap. `serde_json::Value` cannot check this — its map is
    // a `BTreeMap` and sorts the keys on parse (the same reason `write_jsonl` formats
    // `being`/`seed` by hand) — so the assertion is on the raw line: every field the OLD
    // binary wrote, in its order, with `,"partial":false` appended and nothing else moved.
    let zeros = ["0"; N_SECTORS].join(",");
    let expected_bytes = format!(
        "{{\"lap\":1,\"tick_start\":0,\"tick_end\":1000,\"dist\":1000.0,\
         \"sector_ticks\":[{zeros}],\"grass_ticks\":0,\"damage\":0.0,\
         \"dry_events\":0,\"partial\":false}}"
    );
    assert!(
        lines[0].contains(&expected_bytes),
        "the bytes of a closed lap moved:\n{}\nexpected to contain\n{}",
        lines[0],
        expected_bytes
    );
    let second: serde_json::Value = serde_json::from_str(lines[1]).expect("line 1 is JSON");
    assert_eq!(second["being"], 1);
    assert_eq!(second["seed"], 9);
    assert_eq!(second["laps"].as_array().expect("empty list").len(), 0);
    println!("[laps] jsonl line 0: {}", lines[0]);
}

/// A partial lap on the wire: `"partial":true` and the unvisited sectors at 0, which is
/// exactly what `tools/laps.py` reads to decide a sector saw a second pass.
#[test]
fn the_jsonl_marks_a_partial_lap_and_keeps_its_unvisited_sectors_at_zero() {
    let mut t = LapTracker::new(1);
    let tick = run_sector(&mut t, 0, 0, 0, 50);
    t.on_lap(0, tick, 500.0);
    t.on_sector(0, 0, tick);
    for k in 0..20 {
        t.on_tick(0, tick + k, 500.0 + k as f32, false, 0.0, false);
    }
    let path = std::env::temp_dir().join("universe_laps_gen_partial_test.jsonl");
    let path = path.to_str().expect("temp path is utf-8");
    crate::laps::write_jsonl(path, &t.finish(), &[3]).expect("wrote");
    let text = std::fs::read_to_string(path).expect("read back");
    let _ = std::fs::remove_file(path);

    let line: serde_json::Value =
        serde_json::from_str(text.lines().next().expect("a line")).expect("the line is JSON");
    assert_eq!(line["laps"][0]["partial"], false);
    assert_eq!(line["laps"][1]["partial"], true);
    assert_eq!(line["laps"][1]["sector_ticks"][0], 20);
    assert_eq!(line["laps"][1]["sector_ticks"][1], 0);
    assert_eq!(line["laps"][1]["tick_end"], 69);
    println!("[laps] jsonl partial: {}", text.trim());
}
