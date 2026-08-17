//! # Solo (Soil) — Campo de energia onde organismos evoluem
//!
//! O Solo é o campo de fitness landscape comprimido. Ele captura:
//! - **Topografia**: quais regiões do espaço de pesos são "montanhas" (alto fitness)
//! - **Nutrientes**: quais padrões sinápticos correlacionam com sobrevivência
//! - **Clima**: pressão seletiva e taxa de mutação ambiental
//!
//! O Solo é compartilhado pela população — todos crescem no mesmo terreno.
//! É atualizado a cada N gerações a partir das métricas de fitness.
//!
//! ## Analogia
//!
//! Assim como uma árvore real absorve nutrientes do solo, um organismo
//! IARA "absorve" padrões de peso do campo de energia. O Solo é a
//! memória coletiva da espécie — o que funcionou para os ancestrais.

use crate::math::fourier::FourierField2D;

/// Campo de energia 2D representando o fitness landscape.
///
/// Internamente usa Fourier 2D para compressão máxima (~128 bytes
/// para um campo que captura a topografia essencial do problema).
#[derive(Clone, Debug)]
pub struct Soil {
    /// Campo de energia (Fourier 2D comprimido).
    /// Eixos: (estrutura cerebral × modulação neuroendócrina).
    pub energy_field: FourierField2D,

    /// Pressão seletiva: 0.0 = exploração livre, 1.0 = elitismo extremo.
    /// Afeta como fortemente o Solo influencia a geração de novos organismos.
    pub selection_pressure: f32,

    /// Temperatura ambiental: taxa base de mutação.
    /// Alta temperatura = mais mutações = diversidade.
    pub temperature: f32,

    /// "pH" do solo: balanço entre exploitation vs exploration.
    /// < 0.5 = ácido (exploitation), > 0.5 = alcalino (exploration).
    pub ph: f32,

    /// Geração na qual o Solo foi atualizado pela última vez.
    pub last_update_gen: u32,

    /// Número de organismos que contribuíram para este campo.
    pub sample_count: u32,
}

impl Soil {
    /// Cria um Solo neutro (sem informação prévia).
    pub fn neutral(width: usize, height: usize) -> Self {
        // Campo uniforme (sem variação) = sem preferência
        let data = vec![0.0f32; width * height];
        let energy_field = FourierField2D::from_grid(&data, width, height, 4, 4);

        Self {
            energy_field,
            selection_pressure: 0.5,
            temperature: 1.0,
            ph: 0.5,
            last_update_gen: 0,
            sample_count: 0,
        }
    }

    /// Atualiza o Solo a partir de uma amostra de fitness da população.
    ///
    /// `fitness_landscape[i]` = fitness do organismo i, organizado em grid.
    /// O Solo aprende quais regiões são "férteis" (alto fitness).
    pub fn absorb(
        &mut self,
        fitness_landscape: &[f32],
        width: usize,
        height: usize,
        generation: u32,
    ) {
        // Média exponencial: blend entre campo antigo e novos dados
        let alpha = 0.3; // peso dos dados novos

        let new_field = FourierField2D::from_grid(
            fitness_landscape,
            width,
            height,
            self.energy_field.kx_max,
            self.energy_field.ky_max,
        );

        // Blend nos coeficientes de Fourier
        for (old, new) in self
            .energy_field
            .modes
            .iter_mut()
            .zip(new_field.modes.iter())
        {
            old.0 = old.0 * (1.0 - alpha as f32) + new.0 * alpha as f32;
            old.1 = old.1 * (1.0 - alpha as f32) + new.1 * alpha as f32;
        }

        self.last_update_gen = generation;
        self.sample_count += 1;

        // Ajuste automático de pH baseado na diversidade do fitness
        if fitness_landscape.len() > 1 {
            let mean: f32 = fitness_landscape.iter().sum::<f32>() / fitness_landscape.len() as f32;
            let variance: f32 = fitness_landscape
                .iter()
                .map(|&f| (f - mean) * (f - mean))
                .sum::<f32>()
                / fitness_landscape.len() as f32;
            let cv = if mean.abs() > 1e-6 {
                variance.sqrt() / mean.abs()
            } else {
                0.5
            };
            // Alto CV (diversidade) → pH sobe (mais exploration)
            // Baixo CV (convergência) → pH desce (mais exploitation)
            self.ph = (self.ph * 0.9 + cv.clamp(0.0, 1.0) * 0.1).clamp(0.0, 1.0);
        }
    }

    /// Amostra a energia no ponto (x, y) normalizado [0, 1).
    pub fn energy_at(&self, x: f32, y: f32) -> f32 {
        self.energy_field.sample(x, y) * self.selection_pressure
    }

    /// Serializa para bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let field_bytes = self.energy_field.serialize();
        let mut buf = Vec::with_capacity(20 + field_bytes.len());

        buf.extend_from_slice(&self.selection_pressure.to_le_bytes());
        buf.extend_from_slice(&self.temperature.to_le_bytes());
        buf.extend_from_slice(&self.ph.to_le_bytes());
        buf.extend_from_slice(&self.last_update_gen.to_le_bytes());
        buf.extend_from_slice(&self.sample_count.to_le_bytes());
        buf.extend_from_slice(&(field_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&field_bytes);

        buf
    }

    /// Desserializa de bytes.
    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < 24 {
            return None;
        }

        let selection_pressure = f32::from_le_bytes(buf[0..4].try_into().ok()?);
        let temperature = f32::from_le_bytes(buf[4..8].try_into().ok()?);
        let ph = f32::from_le_bytes(buf[8..12].try_into().ok()?);
        let last_update_gen = u32::from_le_bytes(buf[12..16].try_into().ok()?);
        let sample_count = u32::from_le_bytes(buf[16..20].try_into().ok()?);
        let field_len = u32::from_le_bytes(buf[20..24].try_into().ok()?) as usize;

        if buf.len() < 24 + field_len {
            return None;
        }

        let energy_field = FourierField2D::deserialize(&buf[24..24 + field_len])?;

        Some(Self {
            energy_field,
            selection_pressure,
            temperature,
            ph,
            last_update_gen,
            sample_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_soil_has_no_energy() {
        let soil = Soil::neutral(8, 8);
        // Campo neutro deve ser ~0 em todos os pontos
        let e = soil.energy_at(0.5, 0.5);
        assert!(e.abs() < 0.1, "Neutral soil should have ~0 energy: {}", e);
    }

    #[test]
    fn absorb_updates_field() {
        let mut soil = Soil::neutral(4, 4);

        // Fitness landscape com pico no centro
        let mut landscape = vec![0.0f32; 16];
        landscape[5] = 10.0; // centro
        landscape[6] = 8.0;
        landscape[9] = 7.0;

        soil.absorb(&landscape, 4, 4, 1);
        assert_eq!(soil.last_update_gen, 1);
        assert_eq!(soil.sample_count, 1);
    }

    #[test]
    fn serialize_deserialize_round_trip() {
        let mut soil = Soil::neutral(4, 4);
        soil.selection_pressure = 0.7;
        soil.temperature = 1.5;
        soil.ph = 0.3;

        let bytes = soil.serialize();
        let restored = Soil::deserialize(&bytes).unwrap();

        assert!((restored.selection_pressure - 0.7).abs() < 1e-6);
        assert!((restored.temperature - 1.5).abs() < 1e-6);
        assert!((restored.ph - 0.3).abs() < 1e-6);
    }
}
