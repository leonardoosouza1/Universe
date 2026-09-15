//! Passenger pickup, delivery, fare and respawn (SPEC §7) — `passo()`'s items 29-31
//! (`campo.py:436-457`). See `docs/plano-mundo-taxi.md` §T4 and
//! `/home/leonardo/projects/LLM/neuromod-lab/docs/71-spec-mundo-taxi.md` §7.
//!
//! Like `services.rs`, this module takes its own documented flat-slice input
//! struct ([`PassengerState`]) rather than T2's not-yet-written `Beings` — see
//! `services.rs`'s module doc for why. Field names match the plan's `Beings`
//! sketch (plan §3.1: `carrying`, `dest_x`/`dest_y` → here `dest: [f32; 2]`,
//! `board_x`/`board_y` → `board_pos`, `origin_district`, `since_pickup`, `px`,
//! `pxd`, `px_ok`, `money`, `revenue`, `deliveries`, `map_idx`).
//!
//! # RNG: a crate-local PRNG, not `ChaCha8Rng`
//!
//! The plan's own T4 interface sketch types the respawn RNG as
//! `Option<&mut ChaCha8Rng>` (plan §T4), but this task's hard rules forbid adding
//! any crate beyond what T1 already depends on (`serde`, `serde_json`,
//! `thiserror`) — no `rand`/`rand_chacha`. [`TaxiRng`] is a small xorshift64*
//! generator instead. This is SPEC-compliant: §10 states plainly that the
//! battery's own RNG does not need to be bit-identical to torch's Philox, only
//! statistically comparable (SPEC §11.3) — and the identity gate (`rng: None`)
//! never touches RNG at all (SPEC §7.2's quirk: the vaga is just marked inactive,
//! the sampled replacement position is never read back).

use crate::constants::{Config, NPX};
use crate::services::price_mul_at;
use crate::world::World;

/// A tiny, deterministic, seedable PRNG — xorshift64* (Vigna 2014, non-cryptographic).
/// Used only in "battery" mode (`rng: Some`); the identity gate never constructs one.
/// See this module's doc comment for why this replaces the plan's `ChaCha8Rng`.
pub struct TaxiRng {
    state: u64,
}

impl TaxiRng {
    /// xorshift64* requires a nonzero state; a zero seed is remapped to a fixed
    /// nonzero constant rather than panicking.
    pub fn new(seed: u64) -> Self {
        TaxiRng {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// The per-being passenger fields `step_passengers` reads and mutates. Field
/// names match the plan's `Beings` sketch (plan §3.1) one for one (modulo the
/// `pos_x`/`pos_y` split there vs `[f32; 2]` here — see this crate's final
/// report), so T5's wiring is a direct borrow, not a redesign.
pub struct PassengerState<'a> {
    /// `n`, CURRENT (post-movement) position — read-only here, physics owns it.
    pub pos: &'a [[f32; 2]],
    /// `n`, `pd_` — `-1.0` empty, `+1.0` carrying a passenger.
    pub carrying: &'a mut [f32],
    /// `n`, `dest` — the passenger's drop-off point, set on pickup.
    pub dest: &'a mut [[f32; 2]],
    /// `n`, `emb` — where the CURRENT passenger was picked up, set on pickup.
    pub board_pos: &'a mut [[f32; 2]],
    /// `n`, `orig_d` — the district `board_pos` was in, set on pickup, read by
    /// the fare at delivery (SPEC §7.4: the fare uses the PICKUP district).
    pub origin_district: &'a mut [u32],
    /// `n`, `t_ult` — ticks since the last pickup; `0.0` the tick a pickup
    /// happens, `+1.0` every other tick (unconditionally, SPEC §7.2).
    pub since_pickup: &'a mut [f32],
    /// `n`, `dinh` — money; increased by the fare on delivery.
    pub money: &'a mut [f32],
    /// `n`, `rec` — lifetime revenue; increased by the fare on delivery.
    pub revenue: &'a mut [f32],
    /// `n`, `ent` — lifetime delivery count.
    pub deliveries: &'a mut [f32],
    /// Which `World` (and which segment of `price_mul`) each being is on.
    pub map_idx: &'a [u32],
    /// `n * NPX * 2`, layout `[being][slot][xy]` (plan §3.1) — passenger-slot
    /// positions.
    pub px: &'a mut [f32],
    /// `n * NPX * 2`, same layout — passenger-slot destinations.
    pub pxd: &'a mut [f32],
    /// `n * NPX`, `1` active / `0` inactive (`u32`, not `bool` — plan §3.1: no
    /// `bool` in a future WGSL storage buffer).
    pub px_ok: &'a mut [u32],
}

fn slot_xy(being: usize, slot: usize) -> usize {
    (being * NPX + slot) * 2
}

fn slot_flag(being: usize, slot: usize) -> usize {
    being * NPX + slot
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// SPEC §7.2 (boarding), §7.3 (respawn) and §7.4 (delivery) — `passo()` items
/// 29-31. `free` is `livre` (SPEC §3.1 item 3, precomputed by physics for this
/// tick); `acting_service` is `pode` (`livre & (esc >= 9)`, the same boolean
/// `service_state`'s caller derives from the decided action). `rng: None` is the
/// identity-gate mode (SPEC §7.2: a used slot is just marked inactive, its
/// sampled replacement is never read); `rng: Some` is battery mode (respawn on,
/// slots resample on pickup).
///
/// Runs boarding for every being, THEN respawn for every slot, THEN delivery for
/// every being — the SPEC's own item order (29, 30, 31) — so a being that boards
/// and is immediately within its new destination's radius is delivered in the
/// SAME call (`a_passenger_can_board_and_be_delivered_in_the_same_tick`).
pub fn step_passengers(
    st: &mut PassengerState,
    w: &[World],
    price_mul: &[f32],
    cfg: &Config,
    free: &[bool],
    acting_service: &[bool],
    mut rng: Option<&mut TaxiRng>,
) {
    let n = st.pos.len();

    // --- item 29: boarding (SPEC §7.2) ---
    for i in 0..n {
        let mut best_d = f32::INFINITY;
        let mut kx = 0usize;
        for k in 0..NPX {
            let active = st.px_ok[slot_flag(i, k)] != 0;
            let d = if active {
                let xy = slot_xy(i, k);
                dist([st.px[xy], st.px[xy + 1]], st.pos[i])
            } else {
                1e9
            };
            if d < best_d {
                best_d = d;
                kx = k;
            }
        }

        // `recusa` is always false in v0 (SPEC §1), so it is not a parameter here.
        let pega =
            free[i] && st.carrying[i] < 0.0 && best_d <= cfg.service_radius_m && !acting_service[i];

        if pega {
            let xy = slot_xy(i, kx);
            st.dest[i] = [st.pxd[xy], st.pxd[xy + 1]];
            st.board_pos[i] = st.pos[i];
            st.origin_district[i] =
                w[st.map_idx[i] as usize].district_at(st.pos[i][0], st.pos[i][1]);
            st.carrying[i] = 1.0;

            // Resample the used slot only in battery mode — the identity gate
            // never reads it back (SPEC §7.2's quirk, this module's doc comment).
            if let Some(r) = rng.as_deref_mut() {
                let side = w[st.map_idx[i] as usize].side;
                let (new_px, new_pxd) = sample_new_slot(r, st.pos[i], cfg, side);
                st.px[xy] = new_px[0];
                st.px[xy + 1] = new_px[1];
                st.pxd[xy] = new_pxd[0];
                st.pxd[xy + 1] = new_pxd[1];
            }
            st.px_ok[slot_flag(i, kx)] = 0;
        }

        // Unconditional (SPEC §7.2, `campo.py:450`): resets on pickup, otherwise
        // always advances, even for a parked or dead being.
        st.since_pickup[i] = if pega { 0.0 } else { st.since_pickup[i] + 1.0 };
    }

    // --- item 30: respawn (SPEC §7.3) — off entirely when `rng` is `None`. ---
    if let Some(r) = rng.as_deref_mut() {
        for i in 0..n {
            let world = &w[st.map_idx[i] as usize];
            let district = world.district_at(st.pos[i][0], st.pos[i][1]);
            let demand = world.demand[district as usize];
            let p = demand / cfg.respawn_denominator;
            for k in 0..NPX {
                if r.next_f32() < p {
                    st.px_ok[slot_flag(i, k)] = 1;
                }
            }
        }
    }

    // --- item 31: delivery (SPEC §7.4) ---
    for i in 0..n {
        let cheg =
            free[i] && st.carrying[i] >= 0.0 && dist(st.pos[i], st.dest[i]) <= cfg.service_radius_m;
        if cheg {
            let origin = st.origin_district[i];
            let aliq = w[st.map_idx[i] as usize].tax[origin as usize].clamp(0.0, 0.6);
            let trip_distance = dist(st.dest[i], st.board_pos[i]);
            // The CURRENT day's multiplier for the PICKUP district, not the
            // drop-off one (`the_fare_uses_the_district_of_the_pickup_not_of_the_drop_off`).
            let pm = price_mul_at(price_mul, w, st.map_idx[i], origin);
            let bruta = (cfg.fare_base + cfg.fare_per_distance * trip_distance) * pm;
            let tarifa = bruta * (1.0 - aliq);

            st.money[i] += tarifa;
            st.revenue[i] += tarifa;
            st.deliveries[i] += 1.0;
            st.carrying[i] = -1.0;
        }
    }
}

/// `_perto(pos, 1)` then `_destino(po)` (SPEC §7.1, `campo.py:222-235`): a new
/// passenger slot born in a ring `[8, raio_px]` around `pos`, with a destination
/// concentrated nearby but long-tailed. Only called in battery mode, on a
/// pickup — see this module's doc comment on why the identity gate never needs
/// this (its RNG stream cannot match torch's regardless).
fn sample_new_slot(
    rng: &mut TaxiRng,
    pos: [f32; 2],
    cfg: &Config,
    side: f32,
) -> ([f32; 2], [f32; 2]) {
    let angle = rng.next_f32() * std::f32::consts::TAU;
    let radius = 8.0 + (cfg.passenger_spawn_radius_m - 8.0) * rng.next_f32();
    let px = [
        (pos[0] + radius * angle.cos()).clamp(0.0, side),
        (pos[1] + radius * angle.sin()).clamp(0.0, side),
    ];

    let angle2 = rng.next_f32() * std::f32::consts::TAU;
    let u = rng.next_f32();
    let radius2 = side * (0.05 + 0.45 * u * u);
    let pxd = [
        (px[0] + radius2 * angle2.cos()).clamp(0.0, side),
        (px[1] + radius2 * angle2.sin()).clamp(0.0, side),
    ];

    (px, pxd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{GRID, TEX};

    fn tiny_world(k: usize) -> World {
        World {
            side: 100.0,
            cost_tex: vec![1.0; TEX * TEX],
            district_field: vec![0; GRID * GRID],
            centers: vec![[0.0, 0.0]; k],
            tax: vec![0.0; k],
            price_mul_base: vec![1.0; k],
            rent: vec![0.0; k],
            demand: vec![0.0; k],
            price_phase: vec![0.0; k],
            services: [vec![], vec![], vec![], vec![], vec![]],
        }
    }

    /// Owned per-being data for one test being with `NPX` slots.
    struct OneBeing {
        pos: [[f32; 2]; 1],
        carrying: [f32; 1],
        dest: [[f32; 2]; 1],
        board_pos: [[f32; 2]; 1],
        origin_district: [u32; 1],
        since_pickup: [f32; 1],
        money: [f32; 1],
        revenue: [f32; 1],
        deliveries: [f32; 1],
        map_idx: [u32; 1],
        px: Vec<f32>,
        pxd: Vec<f32>,
        px_ok: Vec<u32>,
    }

    impl OneBeing {
        fn empty() -> Self {
            OneBeing {
                pos: [[0.0, 0.0]],
                carrying: [-1.0],
                dest: [[0.0, 0.0]],
                board_pos: [[0.0, 0.0]],
                origin_district: [0],
                since_pickup: [0.0],
                money: [0.0],
                revenue: [0.0],
                deliveries: [0.0],
                map_idx: [0],
                px: vec![0.0; NPX * 2],
                pxd: vec![0.0; NPX * 2],
                px_ok: vec![0; NPX],
            }
        }

        fn state(&mut self) -> PassengerState<'_> {
            PassengerState {
                pos: &self.pos,
                carrying: &mut self.carrying,
                dest: &mut self.dest,
                board_pos: &mut self.board_pos,
                origin_district: &mut self.origin_district,
                since_pickup: &mut self.since_pickup,
                money: &mut self.money,
                revenue: &mut self.revenue,
                deliveries: &mut self.deliveries,
                map_idx: &self.map_idx,
                px: &mut self.px,
                pxd: &mut self.pxd,
                px_ok: &mut self.px_ok,
            }
        }
    }

    #[test]
    fn a_passenger_can_board_and_be_delivered_in_the_same_tick() {
        let w = tiny_world(1);
        let price_mul = [1.0];
        let cfg = Config::default();
        let mut being = OneBeing::empty();
        being.pos = [[10.0, 10.0]];
        being.px_ok[0] = 1;
        being.px[0] = 12.0; // slot 0 position: 2 m from pos, within raio=6
        being.px[1] = 10.0;
        being.pxd[0] = 14.0; // slot 0 destination: 4 m from pos, within raio=6 too
        being.pxd[1] = 10.0;

        let free = [true];
        let acting_service = [false];
        step_passengers(
            &mut being.state(),
            &[w],
            &price_mul,
            &cfg,
            &free,
            &acting_service,
            None,
        );

        assert_eq!(
            being.carrying[0], -1.0,
            "must be delivered by the end of the tick"
        );
        assert_eq!(being.deliveries[0], 1.0);
        // bruta = (3.0 + 0.25*4.0)*1.0 = 4.0; aliq = 0.0 -> tarifa = 4.0.
        assert!(
            (being.money[0] - 4.0).abs() < 1e-5,
            "got {}",
            being.money[0]
        );
        assert!((being.revenue[0] - 4.0).abs() < 1e-5);
        assert_eq!(being.px_ok[0], 0, "the used slot must be deactivated");
    }

    #[test]
    fn the_fare_uses_the_district_of_the_pickup_not_of_the_drop_off() {
        let mut w = tiny_world(2);
        // ix < 64 -> district 0 (pickup side); ix >= 64 -> district 1 (current side).
        for iy in 0..GRID {
            for ix in 64..GRID {
                w.district_field[iy * GRID + ix] = 1;
            }
        }
        w.tax = vec![0.3, 0.0];
        let price_mul = [2.0, 1.0]; // world 0's only segment: district0=2.0, district1=1.0
        let cfg = Config::default();

        let mut being = OneBeing::empty();
        being.carrying = [1.0]; // already carrying, embarked earlier
        being.origin_district = [0]; // embarked in district 0
        being.board_pos = [[10.0, 10.0]]; // where it embarked (district 0)
        being.pos = [[90.0, 10.0]]; // current position (district 1)
        being.dest = [[92.0, 10.0]]; // within raio=6 of the current position

        let free = [true];
        let acting_service = [false];
        step_passengers(
            &mut being.state(),
            &[w],
            &price_mul,
            &cfg,
            &free,
            &acting_service,
            None,
        );

        // trip_distance = |92-10| = 82; bruta = (3 + 0.25*82)*2.0 = 47.0 (district 0's
        // pmul=2.0, NOT district 1's 1.0); tarifa = 47.0*(1-0.3) = 32.9.
        assert!(
            (being.money[0] - 32.9).abs() < 1e-4,
            "expected the pickup district's tax/multiplier, got {}",
            being.money[0]
        );
    }

    #[test]
    fn a_used_slot_never_comes_back_when_respawn_is_off() {
        let w = tiny_world(1);
        let price_mul = [1.0];
        let cfg = Config::default();
        let mut being = OneBeing::empty();
        being.pos = [[10.0, 10.0]];
        being.px_ok[0] = 1;
        being.px[0] = 12.0;
        being.px[1] = 10.0;
        being.pxd[0] = 500.0; // far away: delivery never fires this test
        being.pxd[1] = 500.0;

        let free = [true];
        let acting_service = [false];

        step_passengers(
            &mut being.state(),
            &[w.clone()],
            &price_mul,
            &cfg,
            &free,
            &acting_service,
            None,
        );
        assert_eq!(being.px_ok[0], 0, "boarding must deactivate the slot");

        // Run several more ticks with respawn off: the slot must stay inactive.
        for _ in 0..5 {
            step_passengers(
                &mut being.state(),
                &[w.clone()],
                &price_mul,
                &cfg,
                &free,
                &acting_service,
                None,
            );
        }
        assert_eq!(
            being.px_ok[0], 0,
            "respawn is off: a used slot never reactivates"
        );
    }

    #[test]
    fn respawn_can_reactivate_a_slot_when_demand_and_rng_favour_it() {
        let mut w = tiny_world(1);
        w.demand = vec![1000.0]; // absurdly high demand: p = demand/respawn should be >= 1
        let price_mul = [1.0];
        let cfg = Config::default(); // respawn_denominator = 20.0
        let mut being = OneBeing::empty();
        being.px_ok = vec![0; NPX]; // every slot inactive

        let free = [true];
        let acting_service = [false];
        let mut rng = TaxiRng::new(42);
        step_passengers(
            &mut being.state(),
            &[w],
            &price_mul,
            &cfg,
            &free,
            &acting_service,
            Some(&mut rng),
        );

        assert!(
            being.px_ok.iter().any(|&ok| ok == 1),
            "demand/respawn >= 1 must reactivate at least one slot"
        );
    }

    #[test]
    fn taxi_rng_is_deterministic_for_a_given_seed() {
        let mut a = TaxiRng::new(7);
        let mut b = TaxiRng::new(7);
        let seq_a: Vec<f32> = (0..8).map(|_| a.next_f32()).collect();
        let seq_b: Vec<f32> = (0..8).map(|_| b.next_f32()).collect();
        assert_eq!(seq_a, seq_b, "same seed must produce the same sequence");
        for v in &seq_a {
            assert!(
                (0.0..1.0).contains(v),
                "next_f32 must stay in [0, 1), got {v}"
            );
        }

        let mut c = TaxiRng::new(8);
        let seq_c: Vec<f32> = (0..8).map(|_| c.next_f32()).collect();
        assert_ne!(
            seq_a, seq_c,
            "different seeds must (almost certainly) diverge"
        );
    }
}
