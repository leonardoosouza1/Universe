//! Gerador determinístico único do projeto.
//!
//! # Por que este arquivo existe
//!
//! O mesmo LCG estava escrito à mão em 7 lugares (`engine.rs` ×4, `gpu_validate.rs` ×2,
//! `dna_genesis.rs` ×1) e **todos com o mesmo erro de sinal**:
//!
//! ```text
//! ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
//! //  \_____ 31 bits _____/   \__ 2^31 __/
//! //  o quociente já está em [0,1) — subtrair 1 dá [-1,0), nunca positivo
//! ```
//!
//! Faltava o `* 2.0`. Consequência medida em `slow_probe`: **1024/1024 pesos do
//! `dense1` e 48/48 do `dense2` nasciam negativos**, o que transforma cada camada num
//! somador de sinal fixo — as três saídas do cérebro lento viravam a MESMA soma
//! (correlação +0,9999, primeira componente principal explicando 100% da variância) e
//! nunca positivas, deixando `throttle_cap` travado em 1,0 permanentemente.
//!
//! Um bug de sinal copiado 7 vezes é o argumento para ter isto num lugar só: agora
//! existe um teste que falha se a simetria quebrar de novo.

/// LCG de 64 bits (constantes de Knuth/PCG), determinístico e sem dependência externa.
/// Usado onde a reprodutibilidade importa mais que a qualidade estatística: inicialização
/// de pesos, ruído de denoise, fixtures de teste.
#[derive(Clone, Debug)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniforme em `[0, 1)`.
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }

    /// Uniforme em `[-1, 1)` — **média zero**. É esta que a inicialização de pesos quer.
    #[inline]
    pub fn symmetric(&mut self) -> f32 {
        self.uniform() * 2.0 - 1.0
    }

    /// Normal(0, 1) por Box-Muller. Estava copiada 4× em `spawn.rs`, sempre igual.
    #[inline]
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// `true` com probabilidade `p`.
    #[inline]
    pub fn bernoulli(&mut self, p: f32) -> bool {
        self.uniform() < p
    }

    /// Vetor de pesos simétricos com escala `sigma`. Atalho para o caso mais comum.
    pub fn weights(&mut self, n: usize, sigma: f32) -> Vec<f32> {
        (0..n).map(|_| self.symmetric() * sigma).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// O teste que o projeto não tinha. Ele falha com o idioma antigo (`- 1.0` sem o
    /// `* 2.0`), que é exatamente como o bug sobreviveu a 7 cópias.
    #[test]
    fn symmetric_covers_both_signs() {
        let mut r = Lcg::new(0xDEAD_BEEF);
        let samples: Vec<f32> = (0..10_000).map(|_| r.symmetric()).collect();
        let positives = samples.iter().filter(|&&v| v > 0.0).count();
        assert!(
            (4_000..6_000).contains(&positives),
            "init assimétrico: {positives}/10000 positives — o bug de sinal voltou"
        );
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 0.05, "média {mean} longe de zero");
    }

    #[test]
    fn uniform_stays_in_unit_interval() {
        let mut r = Lcg::new(7);
        for _ in 0..10_000 {
            let v = r.uniform();
            assert!((0.0..1.0).contains(&v), "uniform fora de [0,1): {v}");
        }
    }

    #[test]
    fn symmetric_has_uniform_variance() {
        // Var(U[-1,1]) = 1/3. Com o bug ([-1,0)) daria 1/12 — foi por isso que o teste
        // do DCT em dna_genesis.rs passava por motivo errado.
        let mut r = Lcg::new(99);
        let a: Vec<f32> = (0..50_000).map(|_| r.symmetric()).collect();
        let m = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|v| (v - m).powi(2)).sum::<f32>() / a.len() as f32;
        assert!(
            (var - 1.0 / 3.0).abs() < 0.02,
            "variância {var}, esperado 1/3"
        );
    }

    #[test]
    fn gaussian_has_zero_mean_and_unit_stddev() {
        let mut r = Lcg::new(0xC0FFEE);
        let a: Vec<f32> = (0..50_000).map(|_| r.gaussian()).collect();
        let m = a.iter().sum::<f32>() / a.len() as f32;
        let sd = (a.iter().map(|v| (v - m).powi(2)).sum::<f32>() / a.len() as f32).sqrt();
        assert!(m.abs() < 0.03, "média {m}");
        assert!((sd - 1.0).abs() < 0.05, "desvio {sd}");
    }

    #[test]
    fn is_deterministic_for_same_seed() {
        let a: Vec<f32> = {
            let mut r = Lcg::new(42);
            (0..100).map(|_| r.symmetric()).collect()
        };
        let b: Vec<f32> = {
            let mut r = Lcg::new(42);
            (0..100).map(|_| r.symmetric()).collect()
        };
        assert_eq!(a, b);
    }
}
