/// IARA v2101 - Sovereign World: The F1 Resonance Circuit
/// Based on Codex v2101: Darwinismo Matemático e Geometria de Splines

use std::f32::consts::PI;

pub struct RaceTrack {
    /// Points defining the center of the track (Racing Line)
    pub path: Vec<(f32, f32)>,
    pub width: f32,
    /// KAN Barriers along the track edges or inside curves
    pub barriers: Vec<(f32, f32, f32)>, 
}

impl RaceTrack {
    pub fn new() -> Self {
        // Criando um circuito estilo Interlagos/F1 (Formato de 'L' ou Ferradura)
        let path = vec![
            (0.0, 0.0),   // Largada
            (20.0, 0.0),  // Reta inicial
            (30.0, 5.0),  // Curva larga 1
            (35.0, 15.0), // Curva fechada (Hairpin)
            (25.0, 20.0), // Reta oposta
            (10.0, 15.0), // Curva larga 2
            (0.0, 10.0),  // S do Senna
            (0.0, 0.0),   // Fechamento (Loop)
        ];

        let mut barriers = Vec::new();
        // Adicionar barreiras KAN nas "zebras" das curvas para forçar precisão
        barriers.push((32.0, 5.0, 2.0));  // Curva 1
        barriers.push((37.0, 15.0, 1.5)); // Apex do Hairpin
        barriers.push((5.0, 12.0, 2.0));  // S do Senna

        Self {
            path,
            width: 6.0,
            barriers,
        }
    }

    /// Retorna a ressonância baseada na distância até o ponto mais próximo do path
    pub fn get_resonance_at(&self, x: f32, y: f32) -> (f32, f32) {
        let mut min_dist = f32::MAX;
        let mut closest_idx = 0;

        // Encontrar o segmento mais próximo
        for i in 0..self.path.len() {
            let p = self.path[i];
            let dist = ((p.0 - x).powi(2) + (p.1 - y).powi(2)).sqrt();
            if dist < min_dist {
                min_dist = dist;
                closest_idx = i;
            }
        }

        // Direção ideal: Tangente ao próximo ponto do path
        let next_idx = (closest_idx + 1) % self.path.len();
        let target = self.path[next_idx];
        let angle_to_target = (target.1 - y).atan2(target.0 - x);

        // Penalidade por sair da largura da pista
        let off_track = (min_dist / (self.width / 2.0)).powi(2);
        
        // Barreiras KAN
        let mut repulsion = 0.0;
        let mut phase_shift = 0.0;
        for b in &self.barriers {
            let dist = ((b.0 - x).powi(2) + (b.1 - y).powi(2)).sqrt();
            if dist < b.2 * 2.0 {
                let intensity = 1.0 - (dist / (b.2 * 2.0));
                repulsion += intensity;
                phase_shift += intensity * PI;
            }
        }

        (angle_to_target + phase_shift, (1.0 - off_track) - repulsion)
    }
}
