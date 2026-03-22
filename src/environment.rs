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
    /// Gera condições aleatórias para uma nova geração.
    /// `mild`: true para as primeiras gerações (bootstrap), condições clementes.
    pub fn randomize(&mut self, mild: bool) {
        use rand::prelude::*;
        let mut rng = rand::rng();

        if mild {
            // Bootstrap: seco, quente, sem vento, dia
            self.asphalt_temp = rng.random_range(28.0..40.0);
            self.weather_wetness = 0.0;
            self.wind = Vec3::ZERO;
            self.time_of_day = rng.random_range(10.0..16.0);
        } else {
            self.asphalt_temp = rng.random_range(15.0..55.0);
            self.weather_wetness = rng.random_range(0.0..1.0);
            self.wind = Vec3::new(
                rng.random_range(-8.0..8.0),
                0.0,
                rng.random_range(-8.0..8.0),
            );
            self.time_of_day = rng.random_range(0.0..24.0);
        }

        self.tire_wear_global = 0.0;
        self.generation_time = 0.0;
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
