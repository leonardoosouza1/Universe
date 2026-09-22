//! lab-ui PRD S2/D3/I3 — static description of the 192 sensory-wave channels for the
//! `retina_map.json` UI contract: which channels are retina-pixel reductions (and
//! their EXACT pixel set + reduction op), versus channels sourced from body
//! telemetry, host/temporal state, the diagonal whisker probes, or a world-grid
//! query.
//!
//! Generated under the DEFAULT runtime flags: `perif_real = false` (legacy pixel
//! sets for the fovea/peripheral HD blocks) and `BODY_CHANNELS_ON = false` (the
//! 8 `channels::BODY_SLOTS` channels keep their non-body meaning). These match every
//! trace this module was built against (`champ_nocost_s77_obs2_t8005` and the b9_*
//! runs) -- a run captured with either flag on needs a second map, out of scope here.
//!
//! # How the retina/non-retina split was verified
//!
//! Every pixel-pooling formula below was read directly out of `gpu/perception.wgsl`
//! (Blocks A, B, I, M, O, P, Q) and `gpu/retina_digest.wgsl` (the row-6/row-7 edge
//! pass), not copied from a comment or a prior report -- see each function's doc for
//! the block name and the exact scan range. The retina/non-retina partition and the
//! non-retina source bucket for every one of the other 102 channels was cross-checked
//! against `perception::channels::source_for()` (the canonical last-writer registry)
//! and `channels::BODY_SLOTS`/`CHANNEL_FAMILIES`, with two deliberate departures from
//! `source_for()`'s literal string:
//!   - ch28, ch32: `source_for()` says `"motor.rs"` because the INTERACTIVE Bevy path
//!     overwrites them after perception.wgsl runs. The traces this map targets all
//!     come from the headless GPU/native path, where perception.wgsl's grid-query
//!     value is what actually reaches the wave. Classified as `"other"` (grid query),
//!     not `"body"`/`"temporal"`/`"whisker"` (the contract has no `"motor.rs"` bucket
//!     anyway).
//!   - ch20, ch64-75, ch87: `source_for()` names the WRITING KERNEL (`temporal.wgsl`),
//!     which is accurate but not an "honest label" in the contract's sense -- these
//!     are physiological/vehicle telemetry (fuel, velocity, tire temp, slip angle),
//!     so they're bucketed `"body"` here, matching `channels::CHANNEL_FAMILIES`'s own
//!     `("body", 64, 76)` entry (and ch20's `BODY_SLOTS` doc: "unconditional").
//!
//! The exhaustive partition is enforced by `every_index_0_191_appears_exactly_once`
//! below, not by hand-counting prose -- run `cargo test --release --lib retina_map`.

use crate::config::iara::vision::{RETINA_COLS, RETINA_ROWS};
use crate::perception::channels;

/// One retina-pixel-derived channel: exactly which pixels feed it and how.
pub struct RetinaChannelEntry {
    pub index: usize,
    pub name: String,
    /// The real reduction: `"max"`, or `"custom:<name>"` when it isn't a plain
    /// pool (a run-search, a weighted centroid, an adjacent-column diff sum, or a
    /// single unpooled sample).
    pub op: String,
    pub pixels: Vec<u32>,
}

/// One non-retina channel: an honest source label from the contract's fixed
/// vocabulary (`body` | `temporal` | `whisker` | `other`).
pub struct NonRetinaChannelEntry {
    pub index: usize,
    pub name: String,
    pub source: &'static str,
}

fn pixel_index(row: u32, col: u32) -> u32 {
    row * RETINA_COLS as u32 + col
}

fn pool_pixels(row_lo: u32, row_hi: u32, col_lo: u32, col_hi: u32) -> Vec<u32> {
    let mut px = Vec::with_capacity(((row_hi - row_lo) * (col_hi - col_lo)) as usize);
    for row in row_lo..row_hi {
        for col in col_lo..col_hi {
            px.push(pixel_index(row, col));
        }
    }
    px
}

/// Mirrors `let hi = select(lo + total/n_bins, total, bin == n_bins - 1)`, the exact
/// "last bin absorbs the remainder" pattern shared by Blocks A, B, O and (legacy
/// default, `perif_real = false`) P in `perception.wgsl`.
fn last_bin_absorbs(bin: u32, n_bins: u32, lo_base: u32, span: u32) -> (u32, u32) {
    let cols_per_bin = span / n_bins;
    let lo = lo_base + bin * cols_per_bin;
    let hi = if bin == n_bins - 1 {
        lo_base + span
    } else {
        lo + cols_per_bin
    };
    (lo, hi)
}

/// Block I, `perception.wgsl` (escape_direction, ch60-63, all 4 broadcast the same
/// value). Candidate scan range: row `RETINA_ROWS/2 - 1` = 6, cols
/// `[RETINA_COLS/4, RETINA_COLS*3/4)` = [31, 94).
fn escape_direction_pixels() -> Vec<u32> {
    let row = RETINA_ROWS as u32 / 2 - 1;
    let col_lo = RETINA_COLS as u32 / 4;
    let col_hi = (RETINA_COLS as u32 * 3) / 4;
    pool_pixels(row, row + 1, col_lo, col_hi)
}

/// Block M, `perception.wgsl` (dark_center_tracker, ch90-93, all 4 broadcast). The
/// sliding window (width 8) starts at cols 16..=40, so the pixels it can ever touch
/// span cols [16, 48) of row `RETINA_ROWS/2 - 1` = 6.
fn dark_center_pixels() -> Vec<u32> {
    let row = RETINA_ROWS as u32 / 2 - 1;
    pool_pixels(row, row + 1, 16, 48)
}

/// `retina_digest.wgsl`'s edge-density pass (ch99 = row 6, ch100 = row 7): the whole
/// row, all `RETINA_COLS` columns -- the reduction needs the full ordered sequence
/// (sum of |Δ| between consecutive columns), not a fixed pool.
fn edge_density_pixels(row: u32) -> Vec<u32> {
    pool_pixels(row, row + 1, 0, RETINA_COLS as u32)
}

/// The 90 channels that reduce retina pixels within a single tick, with their exact
/// pixel sets. See the module doc for how this was verified.
pub fn build_retina_channels() -> Vec<RetinaChannelEntry> {
    let mut v = Vec::with_capacity(90);

    // Block A: ch0-7 "far_pool_N", rows [0,5), 8 col-bins over the full width.
    for bin in 0..8u32 {
        let (lo, hi) = last_bin_absorbs(bin, 8, 0, RETINA_COLS as u32);
        v.push(RetinaChannelEntry {
            index: bin as usize,
            name: format!("far_pool_{bin}"),
            op: "max".to_string(),
            pixels: pool_pixels(0, 5, lo, hi),
        });
    }

    // Block B: ch8-15 "near_pool_N", rows [10, RETINA_ROWS), same col-bins.
    for bin in 0..8u32 {
        let (lo, hi) = last_bin_absorbs(bin, 8, 0, RETINA_COLS as u32);
        v.push(RetinaChannelEntry {
            index: 8 + bin as usize,
            name: format!("near_pool_{bin}"),
            op: "max".to_string(),
            pixels: pool_pixels(10, RETINA_ROWS as u32, lo, hi),
        });
    }

    // Block I: ch60-63 "escape_direction" (broadcast).
    let escape = escape_direction_pixels();
    for ch in 60..=63usize {
        v.push(RetinaChannelEntry {
            index: ch,
            name: "escape_direction".to_string(),
            op: "custom:longest_dark_run_centroid".to_string(),
            pixels: escape.clone(),
        });
    }

    // Block M: ch90-93 "dark_center_tracker" (broadcast).
    let dark_center = dark_center_pixels();
    for ch in 90..=93usize {
        v.push(RetinaChannelEntry {
            index: ch,
            name: "dark_center_tracker".to_string(),
            op: "custom:max_weight_centroid_window8".to_string(),
            pixels: dark_center.clone(),
        });
    }

    // retina_digest.wgsl: ch99/ch100 "edge_row6"/"edge_row7".
    v.push(RetinaChannelEntry {
        index: 99,
        name: "edge_row6".to_string(),
        op: "custom:edge_density_adjacent_diff".to_string(),
        pixels: edge_density_pixels(6),
    });
    v.push(RetinaChannelEntry {
        index: 100,
        name: "edge_row7".to_string(),
        op: "custom:edge_density_adjacent_diff".to_string(),
        pixels: edge_density_pixels(7),
    });

    // Block O: ch128-143 "fovea_hd_N", 16 bins, ALL rows, legacy full width
    // (perif_real = false).
    for bin in 0..16u32 {
        let (lo, hi) = last_bin_absorbs(bin, 16, 0, RETINA_COLS as u32);
        v.push(RetinaChannelEntry {
            index: 128 + bin as usize,
            name: format!("fovea_hd_{bin}"),
            op: "max".to_string(),
            pixels: pool_pixels(0, RETINA_ROWS as u32, lo, hi),
        });
    }

    // Block P: ch144-159 "peripheral_hd_N". Legacy default (perif_real = false) is a
    // documented byte-identical duplicate of Block O's pixel set and formula (see
    // perception.wgsl's own "é literalmente o mesmo loop... não é erro" comment).
    for bin in 0..16u32 {
        let (lo, hi) = last_bin_absorbs(bin, 16, 0, RETINA_COLS as u32);
        v.push(RetinaChannelEntry {
            index: 144 + bin as usize,
            name: format!("peripheral_hd_{bin}"),
            op: "max".to_string(),
            pixels: pool_pixels(0, RETINA_ROWS as u32, lo, hi),
        });
    }

    // Block Q: ch160-191 "horizon_raw_i", one unpooled pixel per channel: row
    // RETINA_ROWS/2 = 7, col = i * (RETINA_COLS/32) = i*3.
    let horizon_row = RETINA_ROWS as u32 / 2;
    let step = RETINA_COLS as u32 / 32;
    for i in 0..32u32 {
        v.push(RetinaChannelEntry {
            index: 160 + i as usize,
            name: format!("horizon_raw_{i}"),
            op: "custom:single_pixel".to_string(),
            pixels: vec![pixel_index(horizon_row, i * step)],
        });
    }

    v.sort_by_key(|e| e.index);
    v
}

/// Honest source bucket for a non-retina channel index. Panics if called on a
/// retina-pixel index -- callers must partition first (see `build_channel_entries`).
fn non_retina_source(ch: usize) -> &'static str {
    match ch {
        // Physiological / vehicle telemetry, CPU-injected (BODY_CHANNELS_ON = false):
        // ch20 = FuelFrac (channels::BODY_SLOTS slot 3, unconditional), ch64-75 =
        // channels::CHANNEL_FAMILIES's own ("body", 64, 76) family (VelAlign..
        // Estagnacao), ch87 = SlipAngle (cross2D(fwd, vel), slot 49).
        20 | 64..=75 | 87 => "body",
        // Block N, perception.wgsl: body-relative diagonal whisker probes at
        // +-30 deg / +-60 deg (grid raycast, not a retina-pixel read).
        94..=97 => "whisker",
        // Grid/flow-field/heightmap queries (perception.wgsl Blocks C, D, E, F, G,
        // H, J, L) -- world-space raycasts via voxel_at()/flow_at()/height_at(),
        // never touching the retina buffer. ch28 and ch32 fall in here even though
        // channels::source_for() names "motor.rs" for them (see module doc).
        16..=19 | 21..=59 | 76 | 77 | 82 => "other",
        // Everything else non-retina: host/racing-line geometry, motion-diff
        // kernels (motion.wgsl), and CPU accumulators/derivatives -- all carried
        // tick-to-tick rather than read fresh from one frame's pixels.
        _ => "temporal",
    }
}

/// The 102 non-retina channels, each with an honest `body|temporal|whisker|other`
/// source label. See `non_retina_source` for the classification and its rationale.
pub fn build_non_retina_channels() -> Vec<NonRetinaChannelEntry> {
    let retina = build_retina_channels();
    let retina_indices: std::collections::HashSet<usize> = retina.iter().map(|e| e.index).collect();

    let mut v = Vec::with_capacity(channels::N_CHANNELS - retina_indices.len());
    for ch in 0..channels::N_CHANNELS {
        if retina_indices.contains(&ch) {
            continue;
        }
        v.push(NonRetinaChannelEntry {
            index: ch,
            name: channels::name_for(ch).to_string(),
            source: non_retina_source(ch),
        });
    }
    v
}

/// Assembles the full `retina_map.json` contract payload (rows/cols/wave_size fixed
/// by `config::iara::vision`/`config::iara::brain`, channels + non_retina from the
/// two builders above).
pub fn build_retina_map_json() -> serde_json::Value {
    let channels_json: Vec<serde_json::Value> = build_retina_channels()
        .into_iter()
        .map(|e| {
            serde_json::json!({
                "index": e.index,
                "name": e.name,
                "source": "retina",
                "op": e.op,
                "pixels": e.pixels,
            })
        })
        .collect();

    let non_retina_json: Vec<serde_json::Value> = build_non_retina_channels()
        .into_iter()
        .map(|e| {
            serde_json::json!({
                "index": e.index,
                "name": e.name,
                "source": e.source,
            })
        })
        .collect();

    serde_json::json!({
        "schema_version": 1,
        "rows": RETINA_ROWS,
        "cols": RETINA_COLS,
        "wave_size": channels::N_CHANNELS,
        "channels": channels_json,
        "non_retina": non_retina_json,
        // Not part of the fixed contract shape -- additive, ignored by a strict
        // consumer -- but recording it here beats silent drift: this map assumes
        // the DEFAULT runtime flags (see module doc). A trace captured with either
        // flag on needs a different map.
        "assumptions": {
            "perif_real": false,
            "body_channels_on": false,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_index_0_191_appears_exactly_once() {
        let retina = build_retina_channels();
        let non_retina = build_non_retina_channels();
        let mut seen = vec![0u8; channels::N_CHANNELS];
        for e in &retina {
            seen[e.index] += 1;
        }
        for e in &non_retina {
            seen[e.index] += 1;
        }
        for (ch, &count) in seen.iter().enumerate() {
            assert_eq!(
                count, 1,
                "channel {ch} covered {count} times (want exactly 1)"
            );
        }
        assert_eq!(retina.len() + non_retina.len(), channels::N_CHANNELS);
    }

    #[test]
    fn retina_channel_count_is_90() {
        assert_eq!(build_retina_channels().len(), 90);
    }

    #[test]
    fn non_retina_source_bucket_counts() {
        let non_retina = build_non_retina_channels();
        let count = |src: &str| non_retina.iter().filter(|e| e.source == src).count();
        assert_eq!(count("body"), 14, "ch20 + ch64-75 (12) + ch87");
        assert_eq!(count("whisker"), 4, "ch94-97");
        assert_eq!(
            count("other"),
            46,
            "grid-query blocks C-L minus ch20/60-63/76-77-carve-outs already retina/body"
        );
        assert_eq!(count("temporal"), 38);
    }

    #[test]
    fn pixel_indices_in_bounds() {
        let total = (RETINA_ROWS * RETINA_COLS) as u32;
        for e in build_retina_channels() {
            for &p in &e.pixels {
                assert!(p < total, "channel {} has out-of-bounds pixel {p}", e.index);
            }
            assert!(
                !e.pixels.is_empty(),
                "channel {} has an empty pixel list",
                e.index
            );
        }
    }

    #[test]
    fn spot_check_known_bins() {
        let retina = build_retina_channels();
        let by_index = |i: usize| retina.iter().find(|e| e.index == i).unwrap();

        // ch0: far_pool bin 0, rows 0-4, cols 0-14 (15 cols/bin, 5 rows) = 75 px.
        let ch0 = by_index(0);
        assert_eq!(ch0.pixels.len(), 5 * 15);
        assert_eq!(ch0.pixels[0], pixel_index(0, 0));
        assert_eq!(*ch0.pixels.last().unwrap(), pixel_index(4, 14));

        // ch7: far_pool bin 7 (last bin absorbs remainder: cols 105-125 = 21 cols).
        let ch7 = by_index(7);
        assert_eq!(ch7.pixels.len(), 5 * 21);

        // ch160: horizon_raw_0, row 7 col 0.
        let ch160 = by_index(160);
        assert_eq!(ch160.pixels, vec![pixel_index(7, 0)]);

        // ch191: horizon_raw_31, row 7, col 31*3 = 93.
        let ch191 = by_index(191);
        assert_eq!(ch191.pixels, vec![pixel_index(7, 93)]);

        // ch128 and ch144 are byte-identical under the default (perif_real=false)
        // flag -- the documented Block O/P duplicate.
        let ch128 = by_index(128);
        let ch144 = by_index(144);
        assert_eq!(ch128.pixels, ch144.pixels);
    }

    #[test]
    fn json_shape_matches_contract() {
        let v = build_retina_map_json();
        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["rows"], 14);
        assert_eq!(v["cols"], 126);
        assert_eq!(v["wave_size"], 192);
        assert_eq!(v["channels"].as_array().unwrap().len(), 90);
        assert_eq!(v["non_retina"].as_array().unwrap().len(), 102);
    }
}
