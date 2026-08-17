/// IARA v2101 - Reality Engine (SACI-VISION-ACTUATOR)
/// Based on Codex Part 5 & 6: Materialization

pub struct RealityEngine {
    pub width: usize,
    pub height: usize,
    /// 2-bit Draft Buffer (00 = Empty, 01 = Low, 10 = Mid, 11 = High Energy)
    pub draft_buffer: Vec<u8>,
}

impl RealityEngine {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            draft_buffer: vec![0; (width * height + 3) / 4], // 2 bits per pixel packed in u8
        }
    }

    /// Materialization: Translate latent DNA (valid_intent) into visual draft.
    /// Axiom: "Tradução direta de vetores latentes em pixels reais."
    pub fn materialize(&mut self, latent_dna: &[f32]) {
        for (i, &energy) in latent_dna.iter().enumerate() {
            if i >= self.width * self.height {
                break;
            }

            let pixel_val = if energy > 5.0 {
                0b11 // High Energy (White/Bright)
            } else if energy > 2.0 {
                0b10 // Mid
            } else if energy > 0.0 {
                0b01 // Low
            } else {
                0b00 // Empty
            };

            self.set_pixel_draft(i, pixel_val);
        }
    }

    fn set_pixel_draft(&mut self, idx: usize, val: u8) {
        let byte_idx = idx / 4;
        let bit_shift = (idx % 4) * 2;
        self.draft_buffer[byte_idx] &= !(0b11 << bit_shift);
        self.draft_buffer[byte_idx] |= (val & 0b11) << bit_shift;
    }

    /// Visualizes the internal "thought" as a primitive ASCII representation
    pub fn render_ascii(&self) -> String {
        let mut output = String::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                let byte_idx = idx / 4;
                let bit_shift = (idx % 4) * 2;
                let val = (self.draft_buffer[byte_idx] >> bit_shift) & 0b11;

                let char = match val {
                    0b11 => '█',
                    0b10 => '▓',
                    0b01 => '░',
                    _ => ' ',
                };
                output.push(char);
            }
            output.push('\n');
        }
        output
    }
}
