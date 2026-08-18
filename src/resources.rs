//! D7 "scarce resource" — energy stations: M fuel stations < N beings.
//!
//! Why this module exists: the difficulty-ladder battery measured 0% mortality in
//! 34 of 35 generations with fuel, noise, damage and weather all enabled — the world
//! does not select brains. D7 replaces the per-sector refill (an infinite fuel source)
//! with M fixed stations on the track, M < N beings, so that for the first time the
//! action of one being (draining a station) alters the world another being finds.
//!
//! This is a PURE state machine, in the pattern of `gpu_evolve/escada.rs` and
//! `gpu_evolve/fantasma.rs`: no I/O, no statics, no CLI knowledge, no GPU. Everything
//! is deterministic in its inputs, so every function here is testable offline.
//!
//! # Integrator contract (src/bin/gpu_evolve/, NOT wired by this module)
//!
//! - Per generation: `place_stations(seed_gen, m, &world.waypoints_world,
//!   &world.waypoint_arcs, track_width, cfg.capacity)` — same seed, same track ⇒ same
//!   stations for every being of the generation (same comparability contract as the
//!   D2 shared start).
//! - Per tick, inside the CPU fuel loop (context.rs, the block that already does
//!   `states[i].fuel -= …`): call `step(&mut stations, &beings, &cfg)` ONCE and add
//!   `takes[i]` to each being's fuel. `step` processes beings in index order and
//!   recharges stations afterwards; do not also call `recharge_all` separately.
//! - Per tick, for perception slots (ch114-117 under the flag): `nearest_charged`
//!   gives index, distance and charge fraction of the nearest station that still has
//!   ≥ `MIN_USEFUL_CHARGE_FRAC` of its capacity.
//! - Units: positions in metres (world XZ plane), fuel and charge in tank units
//!   (the same unit as `states[i].fuel`), rates are PER TICK at 64 Hz.
//! - The mutable station Vec lives in `GpuCtx` (state), never in a static (statics
//!   are for dose, not state — statics.rs ownership table).

use bevy::math::Vec2;

use crate::rng::Lcg;

/// Default station capacity, in fuel (tank) units. 1.0 ≈ 5 tanks at `--fuel-auto 5.0`
/// (tank 0.195 measured on the dose curve, docs/ESCADA_DIFICULDADE.md).
pub const DEFAULT_CAPACITY: f32 = 1.0;
/// Default recharge, fuel per tick. DELIBERATELY below idle consumption
/// (0.002/64 = 3.125e-5 per tick, context.rs:1136), so camping on a drained station
/// is a net fuel LOSS — the anti-degeneration lock of the D7 spec.
pub const DEFAULT_RECHARGE_PER_TICK: f32 = 0.00002;
/// Default drive-through radius in metres: passing close already drains.
pub const DEFAULT_RADIUS_M: f32 = 6.0;
/// Default transfer, fuel per tick. At 18.75 m/s a being stays ~41 ticks inside the
/// 6 m radius and takes ~0.082 (42% of a k=5 tank); filling more requires slowing
/// down — that tension is what the transfer dose controls.
pub const DEFAULT_TRANSFER_PER_TICK: f32 = 0.002;
/// A station below this fraction of capacity is invisible to `nearest_charged`:
/// steering a being toward an almost-empty station is worse than showing the next one.
pub const MIN_USEFUL_CHARGE_FRAC: f32 = 0.10;
/// Lateral offset amplitude as a fraction of track width. 0.3 keeps stations off the
/// racing line but away from walls by construction (the reverse-maneuver interaction
/// flagged in the spec risks).
pub const LATERAL_OFFSET_FRAC: f32 = 0.3;

/// Dose knobs of the D7 pressure. Owned by the CLI/statics on the integrator side;
/// this module only ever reads them.
#[derive(Clone, Copy, Debug)]
pub struct StationConfig {
    /// Max and initial charge of each station, fuel units.
    pub capacity: f32,
    /// Passive recharge per tick, fuel units (saturates at `capacity`).
    pub recharge_per_tick: f32,
    /// Drive-through radius, metres.
    pub radius_m: f32,
    /// Max fuel transferred to one being per tick, fuel units.
    pub transfer_per_tick: f32,
}

impl Default for StationConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_CAPACITY,
            recharge_per_tick: DEFAULT_RECHARGE_PER_TICK,
            radius_m: DEFAULT_RADIUS_M,
            transfer_per_tick: DEFAULT_TRANSFER_PER_TICK,
        }
    }
}

/// One fuel station. `pos` is fixed for the generation; only `charge` mutates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergyStation {
    /// World position (metres, XZ plane — same frame as `waypoints_world`).
    pub pos: Vec2,
    /// Current charge, fuel units, in `[0, capacity]`.
    pub charge: f32,
}

/// What a being needs to interact with the station field for one tick.
#[derive(Clone, Copy, Debug)]
pub struct BeingFuel {
    /// World position, metres.
    pub pos: Vec2,
    /// Current fuel, tank units.
    pub fuel: f32,
    /// Tank capacity, tank units (`fuel` never exceeds it).
    pub tank: f32,
    /// Dead beings neither drain nor perceive stations.
    pub alive: bool,
}

/// Result of the nearest-charged-station query, ready for the ch114-117 slots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NearestStation {
    /// Index into the station Vec.
    pub index: usize,
    /// Euclidean distance in metres.
    pub dist_m: f32,
    /// `charge / capacity`, in `[0, 1]`.
    pub charge_frac: f32,
}

/// Interpolate the point at cumulative arc-length `s` along the waypoint axis, and
/// the unit tangent of the segment it falls on.
///
/// `arcs` is the cumulative arc per waypoint exactly as `world.rs:206-211` builds it
/// (`arcs[i]` = length from wp[0] to wp[i], `arcs[0] == 0`). `s` is clamped to
/// `[0, arcs.last()]`. Panics if fewer than 2 waypoints (a track axis needs a segment).
pub fn point_at_arc(waypoints: &[Vec2], arcs: &[f32], s: f32) -> (Vec2, Vec2) {
    assert!(
        waypoints.len() >= 2 && arcs.len() == waypoints.len(),
        "track axis needs >= 2 waypoints and one cumulative arc per waypoint \
         (got {} waypoints, {} arcs)",
        waypoints.len(),
        arcs.len()
    );
    let total = *arcs.last().unwrap();
    let s = s.clamp(0.0, total);
    // Find the segment [i, i+1] with arcs[i] <= s <= arcs[i+1]. Linear scan: M <= 16
    // stations per generation, placement runs once per generation — O(M·W) is trivial.
    let mut i = 0;
    while i + 2 < arcs.len() && arcs[i + 1] < s {
        i += 1;
    }
    let seg_len = (arcs[i + 1] - arcs[i]).max(1e-6);
    let t = (s - arcs[i]) / seg_len;
    let dir = (waypoints[i + 1] - waypoints[i]).normalize_or_zero();
    (waypoints[i].lerp(waypoints[i + 1], t), dir)
}

/// Deterministic station placement for one generation.
///
/// M anchors equally spaced in arc-length, all shifted by the same seed-derived
/// fraction of one spacing (`arc_j = (j + frac) · lap/M`), then pushed sideways off
/// the axis by a per-station lateral offset in `±LATERAL_OFFSET_FRAC · track_width`.
/// Same `(seed, waypoints, track_width)` ⇒ byte-identical stations, for every being
/// of the generation — the comparability contract of the shared start (D2).
///
/// All stations start full (`charge == capacity`).
pub fn place_stations(
    seed: u64,
    count: usize,
    waypoints: &[Vec2],
    arcs: &[f32],
    track_width: f32,
    capacity: f32,
) -> Vec<EnergyStation> {
    if count == 0 {
        return Vec::new();
    }
    let lap = *arcs.last().expect("cumulative arcs must not be empty");
    let mut rng = Lcg::new(seed ^ 0xD7_57A7_10); // domain-separated from other seed users
    let frac = rng.uniform(); // one shared phase: keeps spacing exactly lap/M
    let spacing = lap / count as f32;
    (0..count)
        .map(|j| {
            let s = (j as f32 + frac) * spacing;
            let (axis, tangent) = point_at_arc(waypoints, arcs, s % lap);
            // Left normal of the segment, same convention as world.rs racing-line code.
            let normal = Vec2::new(-tangent.y, tangent.x);
            let lateral = rng.symmetric() * LATERAL_OFFSET_FRAC * track_width;
            EnergyStation {
                pos: axis + normal * lateral,
                charge: capacity,
            }
        })
        .collect()
}

/// Nearest station that is still worth visiting (charge ≥ 10% of capacity).
///
/// Returns `None` when every station is drained (or there are none) — the perception
/// slots then report "no station" (bearing 0, distance saturated, charge 0).
pub fn nearest_charged(
    stations: &[EnergyStation],
    capacity: f32,
    pos: Vec2,
) -> Option<NearestStation> {
    let min_charge = MIN_USEFUL_CHARGE_FRAC * capacity;
    stations
        .iter()
        .enumerate()
        .filter(|(_, st)| st.charge >= min_charge)
        .map(|(i, st)| (i, st.pos.distance(pos)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(index, dist_m)| NearestStation {
            index,
            dist_m,
            charge_frac: (stations[index].charge / capacity).clamp(0.0, 1.0),
        })
}

/// Drain one tick of fuel from `station` into a being, if the being is in radius.
///
/// Returns the fuel actually transferred: `min(transfer, charge, tank − fuel)`,
/// never negative. The station loses exactly what the being gains (conservation).
pub fn try_drain(
    station: &mut EnergyStation,
    being_pos: Vec2,
    fuel: f32,
    tank: f32,
    cfg: &StationConfig,
) -> f32 {
    if station.pos.distance(being_pos) > cfg.radius_m {
        return 0.0;
    }
    let headroom = (tank - fuel).max(0.0);
    let take = cfg
        .transfer_per_tick
        .min(station.charge)
        .min(headroom)
        .max(0.0);
    station.charge -= take;
    take
}

/// Passive recharge of every station, one tick, saturating at capacity.
pub fn recharge_all(stations: &mut [EnergyStation], cfg: &StationConfig) {
    for st in stations {
        st.charge = (st.charge + cfg.recharge_per_tick).min(cfg.capacity);
    }
}

/// One full tick of the station field: drain for every living being, then recharge.
///
/// Returns the fuel gained per being (index-aligned with `beings`); the caller adds
/// it to `states[i].fuel`. Beings are processed in INDEX ORDER — when charge is
/// scarce inside one tick, the lower index drinks first. That order is deterministic
/// and matches the existing CPU fuel loop, so runs stay reproducible.
/// Each being drains only its nearest in-radius station on a given tick.
pub fn step(stations: &mut [EnergyStation], beings: &[BeingFuel], cfg: &StationConfig) -> Vec<f32> {
    let takes = beings
        .iter()
        .map(|b| {
            if !b.alive {
                return 0.0;
            }
            // Nearest in-radius station with any charge; O(N·M) with N=48, M<=16 —
            // the same order of work as the comp_stress pass already paid per tick.
            let target = stations
                .iter()
                .enumerate()
                .filter(|(_, st)| st.charge > 0.0)
                .map(|(i, st)| (i, st.pos.distance(b.pos)))
                .filter(|(_, d)| *d <= cfg.radius_m)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i);
            match target {
                Some(i) => try_drain(&mut stations[i], b.pos, b.fuel, b.tank, cfg),
                None => 0.0,
            }
        })
        .collect();
    recharge_all(stations, cfg);
    takes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Square track axis, 4×100 m sides — enough geometry for arc interpolation.
    fn square_track() -> (Vec<Vec2>, Vec<f32>) {
        let waypoints = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(100.0, 100.0),
            Vec2::new(0.0, 100.0),
            Vec2::new(0.0, 0.0),
        ];
        let mut arcs = vec![0.0];
        let mut total = 0.0;
        for i in 0..waypoints.len() - 1 {
            total += waypoints[i].distance(waypoints[i + 1]);
            arcs.push(total);
        }
        (waypoints, arcs)
    }

    #[test]
    fn same_seed_gives_identical_stations_and_different_seed_does_not() {
        let (wps, arcs) = square_track();
        let a = place_stations(77, 8, &wps, &arcs, 12.0, 1.0);
        let b = place_stations(77, 8, &wps, &arcs, 12.0, 1.0);
        assert_eq!(a, b, "same seed must reproduce stations byte-identically");
        let c = place_stations(78, 8, &wps, &arcs, 12.0, 1.0);
        assert_ne!(a, c, "a different seed must move the stations");
        assert_eq!(a.len(), 8);
    }

    #[test]
    fn stations_sit_within_lateral_band_of_the_axis() {
        let (wps, arcs) = square_track();
        let width = 12.0;
        let stations = place_stations(42, 16, &wps, &arcs, width, 1.0);
        let max_off = LATERAL_OFFSET_FRAC * width + 1e-3;
        for st in &stations {
            // Distance to the nearest axis segment must be within the lateral band.
            let d = (0..wps.len() - 1)
                .map(|i| {
                    let (a, b) = (wps[i], wps[i + 1]);
                    let ab = b - a;
                    let t = ((st.pos - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
                    st.pos.distance(a + ab * t)
                })
                .fold(f32::MAX, f32::min);
            assert!(
                d <= max_off,
                "station at {:?} is {d:.2} m off-axis, band is {max_off:.2} m",
                st.pos
            );
        }
    }

    #[test]
    fn charge_is_conserved_and_bounded() {
        let (wps, arcs) = square_track();
        let cfg = StationConfig::default();
        let mut stations = place_stations(7, 4, &wps, &arcs, 12.0, cfg.capacity);
        let initial: f32 = stations.iter().map(|s| s.charge).sum();
        // Park one thirsty being on top of station 0 for many ticks.
        let being = BeingFuel {
            pos: stations[0].pos,
            fuel: 0.0,
            tank: 1000.0, // headroom never binds: isolates the charge bound
            alive: true,
        };
        let ticks = 2_000;
        let mut drained = 0.0;
        for _ in 0..ticks {
            drained += step(&mut stations, &[being], &cfg)[0];
        }
        let max_supply = initial + 4.0 * cfg.recharge_per_tick * ticks as f32;
        assert!(
            drained <= max_supply + 1e-4,
            "drained {drained} exceeds supply {max_supply}: fuel created from nothing"
        );
        for st in &stations {
            assert!(st.charge >= 0.0, "charge went negative: {}", st.charge);
            assert!(st.charge <= cfg.capacity + 1e-6, "charge above capacity");
        }
        // Camping lock: the drained station's regen (2e-5) is below idle burn
        // (3.125e-5, context.rs:1136), so parking there cannot be fuel-positive.
        assert!(
            cfg.recharge_per_tick < 0.002 / 64.0,
            "regen must stay below idle consumption or camping becomes viable"
        );
    }

    #[test]
    fn recharge_saturates_at_capacity() {
        let cfg = StationConfig {
            recharge_per_tick: 0.5,
            ..StationConfig::default()
        };
        let mut stations = vec![EnergyStation {
            pos: Vec2::ZERO,
            charge: 0.8,
        }];
        for _ in 0..10 {
            recharge_all(&mut stations, &cfg);
        }
        assert_eq!(
            stations[0].charge, cfg.capacity,
            "charge must saturate exactly at capacity"
        );
    }

    #[test]
    fn nearest_query_skips_drained_stations() {
        let capacity = 1.0;
        let stations = vec![
            EnergyStation {
                pos: Vec2::new(1.0, 0.0),
                charge: 0.05, // 5% < 10% threshold: invisible
            },
            EnergyStation {
                pos: Vec2::new(50.0, 0.0),
                charge: 0.9,
            },
        ];
        let hit =
            nearest_charged(&stations, capacity, Vec2::ZERO).expect("one charged station exists");
        assert_eq!(hit.index, 1, "the drained near station must be skipped");
        assert!((hit.dist_m - 50.0).abs() < 1e-4);
        assert!((hit.charge_frac - 0.9).abs() < 1e-6);

        let all_drained = vec![EnergyStation {
            pos: Vec2::ZERO,
            charge: 0.0,
        }];
        assert_eq!(
            nearest_charged(&all_drained, capacity, Vec2::ZERO),
            None,
            "no charged station means no target"
        );
    }

    #[test]
    fn drain_respects_tank_headroom_and_radius() {
        let cfg = StationConfig::default();
        let mut st = EnergyStation {
            pos: Vec2::ZERO,
            charge: 1.0,
        };
        // Full tank: nothing moves.
        assert_eq!(try_drain(&mut st, Vec2::ZERO, 0.195, 0.195, &cfg), 0.0);
        assert_eq!(st.charge, 1.0);
        // Almost full: only the headroom moves, not the whole transfer.
        let take = try_drain(&mut st, Vec2::ZERO, 0.1945, 0.195, &cfg);
        assert!((take - 0.0005).abs() < 1e-7, "take was {take}");
        // Out of radius: nothing moves regardless of hunger.
        let before = st.charge;
        assert_eq!(
            try_drain(
                &mut st,
                Vec2::new(cfg.radius_m + 0.1, 0.0),
                0.0,
                0.195,
                &cfg
            ),
            0.0
        );
        assert_eq!(st.charge, before);
    }

    #[test]
    fn dead_beings_do_not_drink_and_index_order_breaks_ties() {
        let cfg = StationConfig {
            transfer_per_tick: 0.6,
            recharge_per_tick: 0.0,
            ..StationConfig::default()
        };
        let mut stations = vec![EnergyStation {
            pos: Vec2::ZERO,
            charge: 1.0,
        }];
        let beings = [
            BeingFuel {
                pos: Vec2::ZERO,
                fuel: 0.0,
                tank: 10.0,
                alive: false, // dead: must take nothing
            },
            BeingFuel {
                pos: Vec2::ZERO,
                fuel: 0.0,
                tank: 10.0,
                alive: true,
            },
            BeingFuel {
                pos: Vec2::ZERO,
                fuel: 0.0,
                tank: 10.0,
                alive: true,
            },
        ];
        let takes = step(&mut stations, &beings, &cfg);
        assert_eq!(takes[0], 0.0, "dead being drank");
        assert!(
            (takes[1] - 0.6).abs() < 1e-6,
            "first living index drinks first"
        );
        assert!(
            (takes[2] - 0.4).abs() < 1e-6,
            "second in index order gets the remainder: got {}",
            takes[2]
        );
    }
}
