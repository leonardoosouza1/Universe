/// IARA v2101 - CfC (Closed-form Continuous-time) Cell
/// Based on Codex v2101 - Liquid Neural Networks for Temporal Flow
///
/// Tier 2.1: usado como PRE-PROCESSOR da retina (dim-preserving: 64→64).
/// Fornece memória temporal entre frames — organismo detecta movimento
/// ("parede se aproxima" vs "parede estática") ao invés de reagir apenas
/// ao frame atual.
///
/// Update rule (por canal, in-place):
///   state[i] = state[i] * exp(-decay_rate * dt) + input[i] * gain * dt
///   filtered[i] = tanh(state[i])
///
/// State é CLAMPED em [-clamp_range, +clamp_range] pra evitar blow-up
/// (failure mode documentado em melhorias/02-cerebro-dormente.md§2.1).

#[derive(Clone)]
pub struct CfCCell {
    pub hidden_size: usize,
    pub state: Vec<f32>,
    pub decay_rate: f32,
    pub input_gain: f32,
    pub clamp_range: f32,
    /// Denoising noise magnitude (byte-lab transfer: SSM_NOISE=0.15 fixed exposure bias).
    /// Injects small perturbations in state to make creatures robust to self-induced drift.
    pub denoise_mag: f32,
    denoise_counter: u32,
    /// Fadiga adaptativa por canal (spike-frequency adaptation).
    ///
    /// Estado lento que sobe quando |state[i]| é alto, subtrai do input efetivo
    /// quebrando realimentação positiva e evitando ponto fixo irreversível.
    ///
    /// Equação por canal i:
    ///   fatigue[i] += alpha * (|state[i]| - fatigue[i]) * dt
    ///   x_efetivo = x - beta * fatigue[i]
    ///
    /// alpha=0.3: τ ≈ 3.3s ≈ 210 ticks @ 64Hz (sobe devagar, não confunde ruído de tick).
    /// beta=0.4: com fatigue=1.8 subtrai 0.72 do input [-1,1] — suficiente para quebrar saturaçao.
    pub fatigue: Vec<f32>,
    /// Parâmetro alpha da fadiga (velocidade de integração). Padrão: 0.3.
    pub fatigue_alpha: f32,
    /// Parâmetro beta da fadiga (força da subtração). Padrão: 0.4.
    pub fatigue_beta: f32,
}

impl CfCCell {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            state: vec![0.0; hidden_size],
            decay_rate: 0.5,     // exp(-0.5 * dt) por tick
            input_gain: 1.0,
            clamp_range: 2.0,
            denoise_mag: 0.10,   // byte-lab validated: 15% noise fixes exposure bias
            denoise_counter: 0,
            fatigue: vec![0.0; hidden_size],
            fatigue_alpha: 1.0,  // τ ≈ 1s @ 64Hz — sobe em ~64 ticks para 63% do alvo
            // β=0.4: com o encolhimento suave o ponto fixo de entrada máxima sustentada
            // é s = 2(1 - β·s) ⇒ s = 2/(1+2β). Com β=0.4 dá s=1.111 (saída 102/127, o
            // canal continua informando); com β=0.8 dá s=0.769 (saída 82) e o CfC deixa
            // de passar sinal sustentado, quebrando `update_stabilizes_on_constant_input`.
            // 0.4 é o valor que a documentação deste arquivo sempre declarou — o código
            // tinha ido para 0.8 sem atualizar nem o comentário nem o teste. Segue como
            // parâmetro de varredura, não como constante sagrada.
            fatigue_beta:  0.4,
        }
    }

    /// Reseta o estado temporal (chamado em spawn/death do racer).
    pub fn reset(&mut self) {
        for s in self.state.iter_mut() {
            *s = 0.0;
        }
        // Zera fadiga junto com o estado — ser recém-nascido começa sem fadiga acumulada.
        for f in self.fatigue.iter_mut() {
            *f = 0.0;
        }
        self.denoise_counter = 0; // Deterministic denoising start per creature
    }

    /// Seed the denoising counter with a unique value per car.
    /// Champion clones with identical weights but different denoise phases
    /// take slightly different paths — truly independent attempts.
    pub fn seed_denoise(&mut self, seed: u32) {
        self.denoise_counter = seed;
    }

    /// Update dim-preserving: recebe input i8 [hidden_size], atualiza state,
    /// escreve filtered i8 [hidden_size] = tanh(state) * 127.
    /// Input e output em i8 pra respeitar o pipeline ternário do BitLinear.
    ///
    /// Inclui fadiga adaptativa (spike-frequency adaptation): subtrai beta*fatigue[i]
    /// do input efetivo antes de aplicar o CfC, quebrando pontos fixos de saturação.
    pub fn update(&mut self, input: &[i8], output: &mut [i8], dt: f32) {
        debug_assert_eq!(input.len(), self.hidden_size);
        debug_assert_eq!(output.len(), self.hidden_size);
        let decay = (-self.decay_rate * dt).exp();
        let gain_dt = self.input_gain * dt;
        let clamp = self.clamp_range;
        // Denoising: cheap deterministic noise via counter-based hash (no RNG needed).
        // Byte-lab discovery: 15% state noise during forward pass fixes exposure bias —
        // creatures become robust to self-induced sensory drift from their own actions.
        self.denoise_counter = self.denoise_counter.wrapping_add(1);
        let noise_mag = self.denoise_mag;
        let alpha = self.fatigue_alpha;
        let beta  = self.fatigue_beta;
        for i in 0..self.hidden_size {
            let x_raw = (input[i] as f32) / 127.0;
            // Fadiga adaptativa por LIMIAR SUAVE — atenua, nunca inverte.
            //
            // A forma anterior era `x = x_raw - beta * fatigue`, subtração crua. Ela
            // quebra os dois contratos que definem um CfC, e os testes deste arquivo
            // pegaram os dois:
            //   · sem entrada nenhuma, a fadiga integra |state| e passa a EMPURRAR o
            //     estado para o negativo — o filtro cria sinal do nada;
            //   · com +127 sustentado a fadiga chega a ~1,8 e subtrai 1,44 de x=1,0,
            //     então uma entrada positiva constante produz saída NEGATIVA.
            // O comentário original registrava a inversão ("input efetivo negativo")
            // como se fosse recurso.
            //
            // Adaptação por frequência de disparo atenua a resposta, não troca o sinal
            // dela. `sign(x)·max(0, |x| - beta·fat)` é o encolhimento suave — o mesmo
            // operador da Lei do Garimpo e do OMP, aqui no eixo do tempo: só passa o que
            // excede o custo acumulado.
            let threshold = beta * self.fatigue[i];
            let x = x_raw.signum() * (x_raw.abs() - threshold).max(0.0);
            let mut s = self.state[i] * decay + x * gain_dt;
            // Inject deterministic noise (~15% of channels per tick, cheap hash).
            if noise_mag > 0.0 {
                let h = self.denoise_counter.wrapping_mul(2654435761).wrapping_add(i as u32);
                if h % 7 == 0 { // ~14% of channels hit per tick
                    // Pseudo-uniform noise in [-noise_mag, +noise_mag] from hash bits.
                    let noise = ((h >> 8) as f32 / 16777216.0 - 0.5) * 2.0 * noise_mag;
                    s += noise;
                }
            }
            if s > clamp { s = clamp; } else if s < -clamp { s = -clamp; }
            // Atualiza fadiga: filtro passa-baixa sobre |state| (spike-frequency adaptation).
            // fat += alpha * (|s| - fat) * dt
            self.fatigue[i] += alpha * (s.abs() - self.fatigue[i]) * dt;
            self.state[i] = s;
            let filtered = s.tanh();
            output[i] = (filtered * 127.0).clamp(-127.0, 127.0) as i8;
        }
    }

    /// Legacy API (mantida pra não quebrar imports experimentais).
    pub fn flow(&mut self, dt: f32) -> Vec<f32> {
        let decay = (-self.decay_rate * dt).exp();
        for s in self.state.iter_mut() {
            *s *= decay;
        }
        self.state.clone()
    }

    pub fn inject(&mut self, input: &[f32]) {
        for (s, &i) in self.state.iter_mut().zip(input.iter()) {
            *s = (*s + i).clamp(-self.clamp_range, self.clamp_range);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_stabilizes_on_constant_input() {
        let mut cfc = CfCCell::new(4);
        cfc.denoise_mag = 0.0; // disable noise for deterministic test
        let input = [127i8, 0, -127, 64];
        let mut out = [0i8; 4];
        // 600 passos de 1/60s → 10s virtuais, bem acima do time-constant 1/decay=2s.
        for _ in 0..600 {
            cfc.update(&input, &mut out, 1.0 / 60.0);
        }
        // Steady state: s = x * gain / decay_rate = x * 1.0 / 0.5 = 2x, clamped em ±2.
        // tanh(2) ≈ 0.964 → ≈122 para +127, -122 para -127.
        assert!(out[0] > 100, "sustained +127 input should drive filtered high ({})", out[0]);
        assert!(out[2] < -100, "sustained -127 input should drive filtered low ({})", out[2]);
        assert!(out[1].abs() < 5, "zero input should stay near zero ({})", out[1]);
    }

    #[test]
    fn state_decays_to_zero_without_input() {
        let mut cfc = CfCCell::new(2);
        cfc.denoise_mag = 0.0; // disable noise for deterministic test
        cfc.state[0] = 1.5;
        cfc.state[1] = -1.5;
        let input = [0i8; 2];
        let mut out = [0i8; 2];
        for _ in 0..1200 {
            cfc.update(&input, &mut out, 1.0 / 60.0);
        }
        assert!(cfc.state[0].abs() < 0.05, "state should decay near zero ({})", cfc.state[0]);
        assert!(cfc.state[1].abs() < 0.05, "state should decay near zero ({})", cfc.state[1]);
    }

    #[test]
    fn reset_zeros_state() {
        let mut cfc = CfCCell::new(8);
        for s in cfc.state.iter_mut() { *s = 1.23; }
        cfc.reset();
        assert!(cfc.state.iter().all(|&s| s == 0.0));
    }
}
