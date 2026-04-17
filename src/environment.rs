use bevy::prelude::*;

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
}

impl Default for EnvironmentalState {
    fn default() -> Self {
        Self {
            asphalt_temp: 35.0,     // Temperatura ótima
            weather_wetness: 0.0,   // Seco
            wind: Vec3::ZERO,
            time_of_day: 14.0,     // Meio da tarde
            tire_wear_global: 0.0,
            generation_time: 0.0,
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

    /// Atualiza o timer da geração
    pub fn tick(&mut self, dt: f32) {
        self.generation_time += dt;
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
    fn grip_factors_within_sane_bounds() {
        let mut e = EnvironmentalState::default();
        for lvl in 0..=3 {
            e.randomize_level(lvl);
            let g = e.combined_grip_factor();
            assert!((0.20..=1.01).contains(&g), "grip {} fora de bounds no level {}", g, lvl);
        }
    }
}
