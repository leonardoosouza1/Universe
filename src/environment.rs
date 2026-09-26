use bevy::prelude::*;

/// Tier 5.4 — Moving obstacles (deterministic AI).
/// NPC that follows track waypoints at constant speed.
/// Does not learn — it is an obstacle that forces the champion to anticipate/overtake.
/// Deterministic per generation (initial positions via splitmix(generation_index)).
#[derive(Clone, Debug)]
pub struct NpcObstacle {
    /// 2D world position (x, z in world; y = track).
    pub pos: Vec2,
    /// Current velocity (lerp vector toward the next waypoint).
    pub velocity: Vec2,
    /// Index of the current target waypoint inside `waypoints_world`.
    pub waypoint_idx: usize,
    /// Target constant speed (m/s). ~8 m/s = slower than evolved champions.
    pub speed: f32,
}

impl NpcObstacle {
    /// NPC↔racer collision radius (hit penalty when distance < NPC_HIT_RADIUS).
    pub const HIT_RADIUS: f32 = 2.0;
    /// Waypoint advance radius (when the NPC gets < ADVANCE_RADIUS, it moves to the next).
    pub const ADVANCE_RADIUS: f32 = 5.0;

    /// Deterministic tick: lerp toward the current waypoint; advances on arrival.
    /// `waypoints` must be the same slice used by `Track::waypoints_world`.
    pub fn tick(&mut self, waypoints: &[Vec2], dt: f32) {
        if waypoints.is_empty() {
            self.velocity = Vec2::ZERO;
            return;
        }
        let target = waypoints[self.waypoint_idx % waypoints.len()];
        let delta = target - self.pos;
        let dist = delta.length();
        if dist < Self::ADVANCE_RADIUS {
            self.waypoint_idx = (self.waypoint_idx + 1) % waypoints.len();
        }
        let dir = if dist > 1e-4 {
            delta / dist
        } else {
            Vec2::ZERO
        };
        self.velocity = dir * self.speed;
        self.pos += self.velocity * dt;
    }
}

/// Deterministic splitmix64 — seed → stable pseudo-random u64 value.
/// Used to pick the NPCs' spawn waypoint from `generation_index`.
fn splitmix64(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    x
}

/// Curriculum gate: moving NPCs only activate from this generation on.
pub const NPC_ACTIVATION_GEN: u32 = 20;
/// NPC count per generation when active.
pub const NPC_COUNT: usize = 3;
/// Constant NPC speed (m/s). Slow enough to be overtaken by an evolved
/// champion (~30-50 m/s) but fast enough to get in the way in curves.
pub const NPC_SPEED: f32 = 8.0;

/// Dynamic Environmental State: The Living World
/// Changes every generation to force evolutionary generalization.
#[derive(Resource)]
pub struct EnvironmentalState {
    /// Asphalt temperature in °C (15.0 - 55.0)
    pub asphalt_temp: f32,
    /// Rain level (0.0 = dry, 1.0 = storm)
    pub weather_wetness: f32,
    /// Wind force vector (N) applied to all bodies
    pub wind: Vec3,
    /// Time of day (0.0 - 24.0)
    pub time_of_day: f32,
    /// Global tire wear accumulated this generation
    pub tire_wear_global: f32,
    /// Time elapsed in this generation
    pub generation_time: f32,
    /// Internal generation counter (auto-incremented in each randomize_level/reset_fixed).
    /// Used for deterministic seeding of the mid-race rain transition.
    pub generation_index: u32,
    /// Tier 5.2 — mid-race rain transition started this generation?
    pub rain_transition_started: bool,
    /// Tier 5.2 — target wetness of the transition (typically 0.8).
    pub rain_target_wetness: f32,
    /// Tier 5.2 — transition speed in wetness units / second.
    pub rain_transition_speed: f32,
    /// Tier 5.2 — is the transition scheduled for this generation? (30% chance, gen >= 5)
    pub rain_transition_scheduled: bool,
    /// Tier 5.4 — this generation's moving NPC obstacles. Empty when
    /// `generation_index < NPC_ACTIVATION_GEN` or before the first `spawn_npcs`.
    pub npcs: Vec<NpcObstacle>,
}

impl Default for EnvironmentalState {
    fn default() -> Self {
        Self {
            asphalt_temp: 35.0,   // Optimal temperature
            weather_wetness: 0.0, // Dry
            wind: Vec3::ZERO,
            time_of_day: 14.0, // Mid-afternoon
            tire_wear_global: 0.0,
            generation_time: 0.0,
            generation_index: 0,
            rain_transition_started: false,
            rain_target_wetness: 0.0,
            rain_transition_speed: 0.0,
            rain_transition_scheduled: false,
            npcs: Vec::new(),
        }
    }
}

impl EnvironmentalState {
    /// Fixed reference environment (deterministic).
    /// Used in validation gens for coronation without environmental bias.
    pub fn reset_fixed(&mut self) {
        self.asphalt_temp = 35.0;
        self.weather_wetness = 0.0;
        self.wind = Vec3::ZERO;
        self.time_of_day = 14.0;
        self.tire_wear_global = 0.0;
        self.generation_time = 0.0;
        // Validation gens (level 0): no rain transition (fixed conditions).
        self.generation_index = self.generation_index.saturating_add(1);
        self.rain_transition_started = false;
        self.rain_target_wetness = 0.0;
        self.rain_transition_speed = 0.0;
        self.rain_transition_scheduled = false;
        // Validation gens: no NPCs (fixed conditions, no moving obstacles).
        self.npcs.clear();
    }

    /// Generates conditions based on `level` (curriculum).
    ///   0 = fixed (reset_fixed) — used in validation gens.
    ///   1 = MILD — light rain, temp 28-40, wind 0-2.
    ///   2 = MEDIUM — temp 22-48, rain 0-0.5, wind 0-5.
    ///   3 = FULL — temp 15-55, rain 0-1, wind ±8.
    pub fn randomize_level(&mut self, level: u8) {
        use rand::prelude::*;
        if level == 0 {
            self.reset_fixed();
            return;
        }
        let mut rng = rand::rng();
        match level {
            1 => {
                self.asphalt_temp = rng.random_range(28.0..40.0);
                self.weather_wetness = rng.random_range(0.0..0.2);
                self.wind = Vec3::new(
                    rng.random_range(-2.0..2.0),
                    0.0,
                    rng.random_range(-2.0..2.0),
                );
                self.time_of_day = rng.random_range(10.0..16.0);
            }
            2 => {
                self.asphalt_temp = rng.random_range(22.0..48.0);
                self.weather_wetness = rng.random_range(0.0..0.5);
                self.wind = Vec3::new(
                    rng.random_range(-5.0..5.0),
                    0.0,
                    rng.random_range(-5.0..5.0),
                );
                self.time_of_day = rng.random_range(8.0..20.0);
            }
            _ => {
                self.asphalt_temp = rng.random_range(15.0..55.0);
                self.weather_wetness = rng.random_range(0.0..1.0);
                self.wind = Vec3::new(
                    rng.random_range(-8.0..8.0),
                    0.0,
                    rng.random_range(-8.0..8.0),
                );
                self.time_of_day = rng.random_range(0.0..24.0);
            }
        }
        self.tire_wear_global = 0.0;
        self.generation_time = 0.0;

        // Tier 5.2 — reset the mid-race transition state at the start of each generation.
        self.generation_index = self.generation_index.saturating_add(1);
        self.rain_transition_started = false;
        self.rain_target_wetness = 0.0;
        self.rain_transition_speed = 0.0;
        // Tier 5.4 — NPCs are (re)seeded externally via `spawn_npcs` after
        // the track is available. We clear here to avoid carry-over between gens.
        self.npcs.clear();

        // Rain DISABLED — it swapped the environment between gens, cars trained dry crashing
        // in rain. Caused a cliff drop in the median (user report).
        self.rain_transition_scheduled = if false {
            let mut x = self.generation_index as u64;
            x = x.wrapping_mul(0x9E3779B97F4A7C15);
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58476D1CE4E5B9);
            x ^= x >> 27;
            let frac = (x as f32 / u64::MAX as f32).abs();
            frac < 0.30
        } else {
            false
        };
    }

    /// Compat: keeps the old API. `mild=true` → level 1, false → level 3.
    pub fn randomize(&mut self, mild: bool) {
        self.randomize_level(if mild { 1 } else { 3 });
    }

    /// Decides the curriculum level based on champion distance and lap length.
    /// Avoids environmental noise before a basic policy exists.
    ///   champion < 0.5 lap → level 1
    ///   champion 0.5 – 1.5 laps → level 2
    ///   champion ≥ 1.5 laps → level 3
    pub fn level_for_champion(champion_distance: f32, lap_distance: f32) -> u8 {
        if lap_distance <= 0.0 {
            return 1;
        }
        let laps = champion_distance / lap_distance;
        if laps < 0.5 {
            1
        } else if laps < 1.5 {
            2
        } else {
            3
        }
    }

    /// Grip factor based on asphalt temperature.
    /// Optimal at 35°C. Drops to 0.6x at 15°C (cold) and 0.75x at 55°C (overheated).
    pub fn grip_temp_factor(&self) -> f32 {
        // Inverted parabola centered at 35°C
        let deviation = (self.asphalt_temp - 35.0) / 20.0; // Normalizes the deviation
        (1.0 - 0.4 * deviation * deviation).clamp(0.55, 1.0)
    }

    /// Grip factor based on rain.
    /// Dry = 1.0 (μ=0.85 base), Heavy rain = ~0.41 (μ≈0.35)
    pub fn grip_rain_factor(&self) -> f32 {
        1.0 - 0.59 * self.weather_wetness
    }

    /// Combined environmental grip factor (multiplies the floor's base grip).
    pub fn combined_grip_factor(&self) -> f32 {
        self.grip_temp_factor() * self.grip_rain_factor()
    }

    /// Returns true if aquaplaning conditions were reached.
    /// Speed in m/s. Aquaplaning when >120km/h (33.3 m/s) and rain >0.7
    pub fn is_aquaplaning(&self, speed_ms: f32) -> bool {
        speed_ms > 33.3 && self.weather_wetness > 0.7
    }

    /// Returns true if it is night (18:00 - 06:00)
    pub fn is_night(&self) -> bool {
        self.time_of_day >= 18.0 || self.time_of_day < 6.0
    }

    /// Vision range adjusted by the day/night cycle
    pub fn vision_range(&self) -> f32 {
        if self.is_night() { 12.0 } else { 25.0 }
    }

    /// Tier 5.4 — Deterministic spawn of NPC obstacles on the waypoints.
    /// Only produces NPCs when `generation_index >= NPC_ACTIVATION_GEN` (curriculum gate).
    /// Initial indices are chosen via splitmix(gen_index + k) → deterministic.
    /// Expected to be called ONCE per generation, after the track is ready.
    pub fn spawn_npcs(&mut self, waypoints: &[Vec2]) {
        self.npcs.clear();
        if waypoints.len() < 2 || self.generation_index < NPC_ACTIVATION_GEN {
            return;
        }
        let n = waypoints.len() as u64;
        for k in 0..NPC_COUNT {
            let seed = (self.generation_index as u64)
                .wrapping_mul(1000)
                .wrapping_add(k as u64 + 1);
            // Distribute well spread: start at a random offset and space by 1/NPC_COUNT of the circuit.
            let base = (splitmix64(seed) % n) as usize;
            let stride = (waypoints.len() / NPC_COUNT).max(1);
            let idx = (base + k * stride) % waypoints.len();
            let next_idx = (idx + 1) % waypoints.len();
            let pos = waypoints[idx];
            let dir = (waypoints[next_idx] - pos).normalize_or_zero();
            self.npcs.push(NpcObstacle {
                pos,
                velocity: dir * NPC_SPEED,
                waypoint_idx: next_idx,
                speed: NPC_SPEED,
            });
        }
    }

    /// Tier 5.4 — Advances all NPCs one dt step following waypoints.
    /// No-op if `npcs` is empty (curriculum gate / no track).
    pub fn tick_npcs(&mut self, waypoints: &[Vec2], dt: f32) {
        for npc in self.npcs.iter_mut() {
            npc.tick(waypoints, dt);
        }
    }

    /// Tier 5.4 — Returns `true` if position `racer_pos` collides with some NPC
    /// (distance < `NpcObstacle::HIT_RADIUS`). Consumers (motor.rs) may
    /// use this to increment `telemetry.hit_npc_count`.
    pub fn npc_hit(&self, racer_pos: Vec2) -> bool {
        self.npcs.iter().any(|n| {
            n.pos.distance_squared(racer_pos) < NpcObstacle::HIT_RADIUS * NpcObstacle::HIT_RADIUS
        })
    }

    /// Tier 5.1 LITE — Social perception: returns the 3 nearest rivals as
    /// (dx, dy) pairs normalized by a 50m clamp. Excludes the racer itself (if
    /// `self_pos` coincides with some entry of `all_positions`, the exact match is
    /// discarded — tolerance 1e-4m).
    ///
    /// Layout: `[(dx0, dy0), (dx1, dy1), (dx2, dy2)]`, where index 0 is the nearest
    /// rival. Empty slots (fewer than 3 rivals) stay at `(0.0, 0.0)`.
    /// Components clamped to `[-1.0, 1.0]` after division by `RIVAL_CLAMP_RADIUS`.
    ///
    /// Data layer Tier 5.1 LITE — not yet wired into sensor sampling (channels 102-107).
    pub fn closest_racers(self_pos: Vec2, all_positions: &[Vec2]) -> [(f32, f32); 3] {
        const RIVAL_CLAMP_RADIUS: f32 = 50.0;
        const EPS_SQ: f32 = 1e-8; // (1e-4)² — filters "myself" out of the slice.

        let mut candidates: Vec<(f32, Vec2)> = all_positions
            .iter()
            .filter_map(|&p| {
                let delta = p - self_pos;
                let d2 = delta.length_squared();
                if d2 <= EPS_SQ {
                    None
                } else {
                    Some((d2, delta))
                }
            })
            .collect();
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = [(0.0f32, 0.0f32); 3];
        for (slot, (_d2, delta)) in out.iter_mut().zip(candidates.into_iter().take(3)) {
            let nx = (delta.x / RIVAL_CLAMP_RADIUS).clamp(-1.0, 1.0);
            let ny = (delta.y / RIVAL_CLAMP_RADIUS).clamp(-1.0, 1.0);
            *slot = (nx, ny);
        }
        out
    }

    /// Updates the generation timer
    pub fn tick(&mut self, dt: f32) {
        self.generation_time += dt;

        // Tier 5.2 — Weather transition mid-race.
        // Trigger window: 30.0..30.5s (avoids missing the start on high dt at speed 16x).
        // Gate: only if the generation's draw enabled it (rain_transition_scheduled).
        if self.rain_transition_scheduled
            && !self.rain_transition_started
            && self.generation_time > 30.0
            && self.generation_time < 30.5
        {
            self.rain_transition_started = true;
            self.rain_target_wetness = 0.8;
            self.rain_transition_speed = 0.05; // 0.8 in ~16s
        }

        if self.rain_transition_started && self.weather_wetness < self.rain_target_wetness {
            self.weather_wetness = (self.weather_wetness + self.rain_transition_speed * dt)
                .min(self.rain_target_wetness);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_0_produces_fixed_state() {
        let mut e = EnvironmentalState::default();
        e.asphalt_temp = 0.0;
        e.weather_wetness = 1.0;
        e.wind = Vec3::new(10.0, 0.0, 10.0);
        e.randomize_level(0);
        assert!((e.asphalt_temp - 35.0).abs() < 1e-5);
        assert_eq!(e.weather_wetness, 0.0);
        assert_eq!(e.wind, Vec3::ZERO);
        assert!((e.time_of_day - 14.0).abs() < 1e-5);
    }

    #[test]
    fn level_curriculum_is_monotonic_in_range_width() {
        let mut e1 = EnvironmentalState::default();
        e1.randomize_level(1);
        assert!((28.0..=40.0).contains(&e1.asphalt_temp));
        assert!(e1.weather_wetness <= 0.2 + 1e-5);
        assert!(e1.wind.x.abs() <= 2.0 + 1e-5);

        let mut e2 = EnvironmentalState::default();
        e2.randomize_level(2);
        assert!((22.0..=48.0).contains(&e2.asphalt_temp));
        assert!(e2.weather_wetness <= 0.5 + 1e-5);

        let mut e3 = EnvironmentalState::default();
        e3.randomize_level(3);
        assert!((15.0..=55.0).contains(&e3.asphalt_temp));
    }

    #[test]
    fn level_for_champion_table() {
        let lap = 900.0;
        assert_eq!(EnvironmentalState::level_for_champion(0.0, lap), 1);
        assert_eq!(EnvironmentalState::level_for_champion(200.0, lap), 1);
        assert_eq!(EnvironmentalState::level_for_champion(450.0, lap), 2);
        assert_eq!(EnvironmentalState::level_for_champion(900.0, lap), 2);
        assert_eq!(EnvironmentalState::level_for_champion(1400.0, lap), 3);
        assert_eq!(EnvironmentalState::level_for_champion(3000.0, lap), 3);
    }

    #[test]
    fn level_for_champion_zero_lap_distance_safe() {
        assert_eq!(EnvironmentalState::level_for_champion(100.0, 0.0), 1);
    }

    #[test]
    fn npcs_gate_until_activation_gen() {
        let wps: Vec<Vec2> = (0..10).map(|i| Vec2::new(i as f32 * 10.0, 0.0)).collect();
        let mut e = EnvironmentalState::default();
        e.generation_index = NPC_ACTIVATION_GEN - 1;
        e.spawn_npcs(&wps);
        assert!(
            e.npcs.is_empty(),
            "NPCs must not spawn before gen {}",
            NPC_ACTIVATION_GEN
        );

        e.generation_index = NPC_ACTIVATION_GEN;
        e.spawn_npcs(&wps);
        assert_eq!(e.npcs.len(), NPC_COUNT);
    }

    #[test]
    fn npc_tick_is_deterministic() {
        let wps: Vec<Vec2> = (0..10).map(|i| Vec2::new(i as f32 * 10.0, 0.0)).collect();
        let mut a = EnvironmentalState::default();
        let mut b = EnvironmentalState::default();
        a.generation_index = NPC_ACTIVATION_GEN;
        b.generation_index = NPC_ACTIVATION_GEN;
        a.spawn_npcs(&wps);
        b.spawn_npcs(&wps);
        for _ in 0..50 {
            a.tick_npcs(&wps, 1.0 / 60.0);
            b.tick_npcs(&wps, 1.0 / 60.0);
        }
        for (na, nb) in a.npcs.iter().zip(b.npcs.iter()) {
            assert!((na.pos - nb.pos).length() < 1e-4);
        }
    }

    #[test]
    fn npc_hit_detects_proximity() {
        let wps: Vec<Vec2> = (0..10).map(|i| Vec2::new(i as f32 * 10.0, 0.0)).collect();
        let mut e = EnvironmentalState::default();
        e.generation_index = NPC_ACTIVATION_GEN;
        e.spawn_npcs(&wps);
        let npc_pos = e.npcs[0].pos;
        assert!(e.npc_hit(npc_pos));
        assert!(!e.npc_hit(npc_pos + Vec2::new(100.0, 100.0)));
    }

    #[test]
    fn closest_racers_ordering_and_clamp() {
        // self at (0,0); rivals at 10m, 30m, 5m, 80m (outside the clamp), and "myself" (0,0).
        let self_pos = Vec2::ZERO;
        let others = vec![
            Vec2::new(10.0, 0.0), // 10m east
            Vec2::new(0.0, 30.0), // 30m north
            Vec2::new(-5.0, 0.0), // 5m west  ← nearest
            Vec2::new(80.0, 0.0), // 80m — clamp at 1.0
            Vec2::new(0.0, 0.0),  // "myself" — must be filtered
        ];
        let out = EnvironmentalState::closest_racers(self_pos, &others);

        // Slot 0 = nearest (5m west) → dx = -5/50 = -0.1, dy = 0.
        assert!((out[0].0 - (-0.1)).abs() < 1e-5, "slot0 dx: {}", out[0].0);
        assert!(out[0].1.abs() < 1e-5, "slot0 dy: {}", out[0].1);

        // Slot 1 = 10m east → dx = 0.2, dy = 0.
        assert!((out[1].0 - 0.2).abs() < 1e-5, "slot1 dx: {}", out[1].0);
        assert!(out[1].1.abs() < 1e-5);

        // Slot 2 = 30m north → dx = 0, dy = 0.6.
        assert!(out[2].0.abs() < 1e-5);
        assert!((out[2].1 - 0.6).abs() < 1e-5, "slot2 dy: {}", out[2].1);

        // All values must be inside the clamp.
        for (dx, dy) in out.iter() {
            assert!((-1.0..=1.0).contains(dx));
            assert!((-1.0..=1.0).contains(dy));
        }
    }

    #[test]
    fn closest_racers_empty_and_fewer_than_three() {
        let self_pos = Vec2::new(100.0, 100.0);
        // Empty → all zeros.
        let out = EnvironmentalState::closest_racers(self_pos, &[]);
        assert_eq!(out, [(0.0, 0.0); 3]);

        // Only 1 rival → slots 1 and 2 zeroed.
        let one = vec![Vec2::new(110.0, 100.0)];
        let out = EnvironmentalState::closest_racers(self_pos, &one);
        assert!((out[0].0 - 0.2).abs() < 1e-5);
        assert_eq!(out[1], (0.0, 0.0));
        assert_eq!(out[2], (0.0, 0.0));
    }

    #[test]
    fn closest_racers_clamps_far_rivals() {
        // A very distant rival (>50m) must saturate at ±1.0.
        let out = EnvironmentalState::closest_racers(Vec2::ZERO, &[Vec2::new(500.0, -500.0)]);
        assert!((out[0].0 - 1.0).abs() < 1e-5);
        assert!((out[0].1 - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn grip_factors_within_sane_bounds() {
        let mut e = EnvironmentalState::default();
        for lvl in 0..=3 {
            e.randomize_level(lvl);
            let g = e.combined_grip_factor();
            assert!(
                (0.20..=1.01).contains(&g),
                "grip {} out of bounds at level {}",
                g,
                lvl
            );
        }
    }
}
