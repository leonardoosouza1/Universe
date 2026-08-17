use bevy::prelude::*;

/// Tier 5.4 — Obstáculos móveis (AI determinista).
/// NPC que segue waypoints da pista com velocidade constante.
/// Não aprende — é um obstáculo que força o campeão a antecipar/ultrapassar.
/// Determinístico por geração (posições iniciais via splitmix(generation_index)).
#[derive(Clone, Debug)]
pub struct NpcObstacle {
    /// Posição 2D no mundo (x, z em mundo; y = pista).
    pub pos: Vec2,
    /// Velocidade atual (vetor lerp em direção ao próximo waypoint).
    pub velocity: Vec2,
    /// Índice do waypoint-alvo atual dentro de `waypoints_world`.
    pub waypoint_idx: usize,
    /// Velocidade constante alvo (m/s). ~8 m/s = mais lento que campeões evoluídos.
    pub speed: f32,
}

impl NpcObstacle {
    /// Raio de colisão NPC↔racer (hit penalty quando distance < NPC_HIT_RADIUS).
    pub const HIT_RADIUS: f32 = 2.0;
    /// Raio de avanço de waypoint (quando NPC chega a < ADVANCE_RADIUS, passa pro próximo).
    pub const ADVANCE_RADIUS: f32 = 5.0;

    /// Tick determinístico: lerp em direção ao waypoint atual; avança quando chega.
    /// `waypoints` deve ser o mesmo slice usado por `Track::waypoints_world`.
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

/// Splitmix64 determinístico — seed → u64 pseudoaleatório estável.
/// Usado para escolher waypoint de spawn dos NPCs a partir de `generation_index`.
fn splitmix64(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    x
}

/// Gate curricular: NPCs móveis só ativam a partir dessa geração.
pub const NPC_ACTIVATION_GEN: u32 = 20;
/// Contagem de NPCs por geração quando ativos.
pub const NPC_COUNT: usize = 3;
/// Velocidade constante dos NPCs (m/s). Lento o suficiente para ser ultrapassado
/// por um campeão evoluído (~30-50 m/s) mas rápido o suficiente para atrapalhar em curvas.
pub const NPC_SPEED: f32 = 8.0;

/// Estado Ambiental Dinâmico: O Mundo Vivo
/// Muda a cada geração para forçar generalização evolucionária.
#[derive(Resource)]
pub struct EnvironmentalState {
    /// Temperatura do asfalto em °C (15.0 - 55.0)
    pub asphalt_temp: f32,
    /// Nível de chuva (0.0 = seco, 1.0 = tempestade)
    pub weather_wetness: f32,
    /// Vetor de força do vento (N) aplicado a todos os corpos
    pub wind: Vec3,
    /// Hora do dia (0.0 - 24.0)
    pub time_of_day: f32,
    /// Desgaste global dos pneus acumulado nesta geração
    pub tire_wear_global: f32,
    /// Tempo decorrido nesta geração
    pub generation_time: f32,
    /// Contador interno de geração (auto-incrementado em cada randomize_level/reset_fixed).
    /// Usado para seeding determinístico da transição de chuva mid-race.
    pub generation_index: u32,
    /// Tier 5.2 — transição de chuva mid-race iniciada nesta geração?
    pub rain_transition_started: bool,
    /// Tier 5.2 — wetness alvo da transição (tipicamente 0.8).
    pub rain_target_wetness: f32,
    /// Tier 5.2 — velocidade da transição em unidades de wetness / segundo.
    pub rain_transition_speed: f32,
    /// Tier 5.2 — transição está agendada para esta geração? (30% de chance, gen >= 5)
    pub rain_transition_scheduled: bool,
    /// Tier 5.4 — NPCs obstáculos móveis desta geração. Vazio quando
    /// `generation_index < NPC_ACTIVATION_GEN` ou antes do primeiro `spawn_npcs`.
    pub npcs: Vec<NpcObstacle>,
}

impl Default for EnvironmentalState {
    fn default() -> Self {
        Self {
            asphalt_temp: 35.0,   // Temperatura ótima
            weather_wetness: 0.0, // Seco
            wind: Vec3::ZERO,
            time_of_day: 14.0, // Meio da tarde
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
    /// Ambiente fixo de referência (determinístico).
    /// Usado em validation gens para coroação sem viés ambiental.
    pub fn reset_fixed(&mut self) {
        self.asphalt_temp = 35.0;
        self.weather_wetness = 0.0;
        self.wind = Vec3::ZERO;
        self.time_of_day = 14.0;
        self.tire_wear_global = 0.0;
        self.generation_time = 0.0;
        // Validation gens (level 0): sem transição de chuva (condições fixas).
        self.generation_index = self.generation_index.saturating_add(1);
        self.rain_transition_started = false;
        self.rain_target_wetness = 0.0;
        self.rain_transition_speed = 0.0;
        self.rain_transition_scheduled = false;
        // Validation gens: sem NPCs (condições fixas, sem obstáculos móveis).
        self.npcs.clear();
    }

    /// Gera condições baseadas em `level` (curriculum).
    ///   0 = fixo (reset_fixed) — usado em validation gens.
    ///   1 = MILD — chuva leve, temp 28-40, vento 0-2.
    ///   2 = MEDIUM — temp 22-48, chuva 0-0.5, vento 0-5.
    ///   3 = FULL — temp 15-55, chuva 0-1, vento ±8.
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

        // Tier 5.2 — reset de estado de transição mid-race no início de cada geração.
        self.generation_index = self.generation_index.saturating_add(1);
        self.rain_transition_started = false;
        self.rain_target_wetness = 0.0;
        self.rain_transition_speed = 0.0;
        // Tier 5.4 — NPCs são (re)semeados externamente via `spawn_npcs` depois que
        // o track estiver disponível. Limpamos aqui para evitar carry-over entre gens.
        self.npcs.clear();

        // Rain DISABLED — trocava environment entre gens, cars trained dry crashing
        // em rain. Causava cliff drop em mediana (user report).
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

    /// Compat: mantém API antiga. `mild=true` → level 1, false → level 3.
    pub fn randomize(&mut self, mild: bool) {
        self.randomize_level(if mild { 1 } else { 3 });
    }

    /// Decide nível curricular baseado em distância do campeão e comprimento da volta.
    /// Evita ruído ambiental antes da política básica existir.
    ///   champion < 0.5 volta → level 1
    ///   champion 0.5 – 1.5 voltas → level 2
    ///   champion ≥ 1.5 voltas → level 3
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

    /// Fator de grip baseado na temperatura do asfalto.
    /// Ótimo em 35°C. Cai para 0.6x em 15°C (frio) e 0.75x em 55°C (superaquecido).
    pub fn grip_temp_factor(&self) -> f32 {
        // Parábola invertida centrada em 35°C
        let deviation = (self.asphalt_temp - 35.0) / 20.0; // Normaliza desvio
        (1.0 - 0.4 * deviation * deviation).clamp(0.55, 1.0)
    }

    /// Fator de grip baseado na chuva.
    /// Seco = 1.0 (μ=0.85 base), Chuva forte = ~0.41 (μ≈0.35)
    pub fn grip_rain_factor(&self) -> f32 {
        1.0 - 0.59 * self.weather_wetness
    }

    /// Fator combinado de grip ambiental (multiplica o grip base do piso).
    pub fn combined_grip_factor(&self) -> f32 {
        self.grip_temp_factor() * self.grip_rain_factor()
    }

    /// Retorna true se condições de aquaplanagem foram atingidas.
    /// Speed em m/s. Aquaplanagem quando >120km/h (33.3 m/s) e chuva >0.7
    pub fn is_aquaplaning(&self, speed_ms: f32) -> bool {
        speed_ms > 33.3 && self.weather_wetness > 0.7
    }

    /// Retorna true se é noite (18:00 - 06:00)
    pub fn is_night(&self) -> bool {
        self.time_of_day >= 18.0 || self.time_of_day < 6.0
    }

    /// Range de visão ajustado pelo ciclo dia/noite
    pub fn vision_range(&self) -> f32 {
        if self.is_night() { 12.0 } else { 25.0 }
    }

    /// Tier 5.4 — Spawn determinístico de NPCs obstáculos sobre os waypoints.
    /// Só produz NPCs quando `generation_index >= NPC_ACTIVATION_GEN` (gate curricular).
    /// Os índices iniciais são escolhidos via splitmix(gen_index + k) → determinístico.
    /// Espera ser chamado UMA vez por geração, após o track estar pronto.
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
            // Distribuir bem espalhado: começa de offset aleatório e espaça por 1/NPC_COUNT do circuito.
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

    /// Tier 5.4 — Avança todos os NPCs um passo dt seguindo waypoints.
    /// No-op se `npcs` estiver vazio (gate curricular / sem track).
    pub fn tick_npcs(&mut self, waypoints: &[Vec2], dt: f32) {
        for npc in self.npcs.iter_mut() {
            npc.tick(waypoints, dt);
        }
    }

    /// Tier 5.4 — Retorna `true` se a posição `racer_pos` colide com algum NPC
    /// (distância < `NpcObstacle::HIT_RADIUS`). Consumidores (motor.rs) podem
    /// usar isso para incrementar `telemetry.hit_npc_count`.
    pub fn npc_hit(&self, racer_pos: Vec2) -> bool {
        self.npcs.iter().any(|n| {
            n.pos.distance_squared(racer_pos) < NpcObstacle::HIT_RADIUS * NpcObstacle::HIT_RADIUS
        })
    }

    /// Tier 5.1 LITE — Percepção social: retorna os 3 rivais mais próximos como
    /// pares (dx, dy) normalizados por clamp de 50m. Exclui o próprio racer (se
    /// `self_pos` coincidir com alguma entrada de `all_positions`, o match exato é
    /// descartado — tolerância 1e-4m).
    ///
    /// Layout: `[(dx0, dy0), (dx1, dy1), (dx2, dy2)]`, onde o índice 0 é o rival
    /// mais próximo. Slots vazios (menos de 3 rivais) ficam em `(0.0, 0.0)`.
    /// Componentes clampeadas em `[-1.0, 1.0]` após divisão por `RIVAL_CLAMP_RADIUS`.
    ///
    /// Data layer Tier 5.1 LITE — ainda não wired em sensor sampling (canais 102-107).
    pub fn closest_racers(self_pos: Vec2, all_positions: &[Vec2]) -> [(f32, f32); 3] {
        const RIVAL_CLAMP_RADIUS: f32 = 50.0;
        const EPS_SQ: f32 = 1e-8; // (1e-4)² — filtra "eu mesmo" do slice.

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

    /// Atualiza o timer da geração
    pub fn tick(&mut self, dt: f32) {
        self.generation_time += dt;

        // Tier 5.2 — Weather transition mid-race.
        // Janela de disparo: 30.0..30.5s (evita perder o start em dt altos em speed 16x).
        // Gate: só se sorteio da geração habilitou (rain_transition_scheduled).
        if self.rain_transition_scheduled
            && !self.rain_transition_started
            && self.generation_time > 30.0
            && self.generation_time < 30.5
        {
            self.rain_transition_started = true;
            self.rain_target_wetness = 0.8;
            self.rain_transition_speed = 0.05; // 0.8 em ~16s
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
            "NPCs não devem spawnar antes da gen {}",
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
        // self em (0,0); rivais a 10m, 30m, 5m, 80m (fora do clamp), e "eu mesmo" (0,0).
        let self_pos = Vec2::ZERO;
        let others = vec![
            Vec2::new(10.0, 0.0), // 10m leste
            Vec2::new(0.0, 30.0), // 30m norte
            Vec2::new(-5.0, 0.0), // 5m oeste  ← mais próximo
            Vec2::new(80.0, 0.0), // 80m — clamp em 1.0
            Vec2::new(0.0, 0.0),  // "eu mesmo" — deve ser filtrado
        ];
        let out = EnvironmentalState::closest_racers(self_pos, &others);

        // Slot 0 = mais próximo (5m oeste) → dx = -5/50 = -0.1, dy = 0.
        assert!((out[0].0 - (-0.1)).abs() < 1e-5, "slot0 dx: {}", out[0].0);
        assert!(out[0].1.abs() < 1e-5, "slot0 dy: {}", out[0].1);

        // Slot 1 = 10m leste → dx = 0.2, dy = 0.
        assert!((out[1].0 - 0.2).abs() < 1e-5, "slot1 dx: {}", out[1].0);
        assert!(out[1].1.abs() < 1e-5);

        // Slot 2 = 30m norte → dx = 0, dy = 0.6.
        assert!(out[2].0.abs() < 1e-5);
        assert!((out[2].1 - 0.6).abs() < 1e-5, "slot2 dy: {}", out[2].1);

        // Todos os valores devem estar dentro do clamp.
        for (dx, dy) in out.iter() {
            assert!((-1.0..=1.0).contains(dx));
            assert!((-1.0..=1.0).contains(dy));
        }
    }

    #[test]
    fn closest_racers_empty_and_fewer_than_three() {
        let self_pos = Vec2::new(100.0, 100.0);
        // Vazio → tudo zero.
        let out = EnvironmentalState::closest_racers(self_pos, &[]);
        assert_eq!(out, [(0.0, 0.0); 3]);

        // Apenas 1 rival → slots 1 e 2 zerados.
        let one = vec![Vec2::new(110.0, 100.0)];
        let out = EnvironmentalState::closest_racers(self_pos, &one);
        assert!((out[0].0 - 0.2).abs() < 1e-5);
        assert_eq!(out[1], (0.0, 0.0));
        assert_eq!(out[2], (0.0, 0.0));
    }

    #[test]
    fn closest_racers_clamps_far_rivals() {
        // Rival muito longe (>50m) deve saturar em ±1.0.
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
                "grip {} fora de bounds no level {}",
                g,
                lvl
            );
        }
    }
}
