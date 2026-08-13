/// Funções de ativação e helpers de clamp — tudo puramente escalar, sem alocação.
///
/// Estas funções refletem as fórmulas documentadas em MATH_INVENTORY.md §1.1.
/// As versões GPU (WGSL) usam as mesmas fórmulas; alterações aqui devem ser
/// espelhadas nos shaders correspondentes (contrariamente às regras E1-E3, não
/// toque os .wgsl — este módulo serve como referência do host).

/// Sigmoid padrão: σ(x) = 1 / (1 + e^{-x}).
///
/// Usada historicamente em gating e normalização de probabilidade.
/// Para o CfC o não-linear primário é `tanh`, não sigmoid.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Tangente hiperbólica — não-linear padrão do CfC e do BitLinear.
///
/// Mapeamento: [-∞, +∞] → (-1, +1).
/// A 64 Hz com estado clamped em [-2, 2]: tanh(±2) ≈ ±0.964 → ±122/127.
#[inline]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Aproximação rápida de tanh: x / (1 + |x|).
///
/// Erro < 2% para |x| < 3. Usada pelo shader perception_temporal.wgsl (F42).
/// NÃO usar onde precisão é exigida; motor.rs e cfc.rs usam tanh exato.
#[inline]
pub fn tanh_approx(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// ReLU: max(0, x).
#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Leaky ReLU: max(alpha * x, x) com alpha < 1.
///
/// alpha = 0.01 é o padrão para manter gradiente mínimo em regiões negativas.
#[inline]
pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 { x } else { alpha * x }
}

/// Clamp simétrico em [-limit, +limit].
#[inline]
pub fn clamp_sym(x: f32, limit: f32) -> f32 {
    x.clamp(-limit, limit)
}

/// Converte estado f32 para i8 via tanh, multiplicando por 127.
///
/// Fórmula: `clamp(tanh(x) * 127, -127, 127) as i8`.
/// Usada pelo CfC (F07) e pelo BitLinear (F10).
#[inline]
pub fn state_to_i8(x: f32) -> i8 {
    (x.tanh() * 127.0).clamp(-127.0, 127.0) as i8
}

/// Encolhimento suave (soft-thresholding): sign(x) * max(0, |x| - threshold).
///
/// Idêntico ao operador da Lei do Garimpo no eixo temporal.
/// Usado na fadiga adaptativa do CfC (F03): atenua sem inverter o sinal.
#[inline]
pub fn soft_threshold(x: f32, threshold: f32) -> f32 {
    x.signum() * (x.abs() - threshold).max(0.0)
}

// ---------------------------------------------------------------------------
// Testes unitários
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_de_zero_e_meio() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6, "sigmoid(0) deve ser 0.5, obteve {}", s);
    }

    #[test]
    fn sigmoid_monotona_crescente() {
        assert!(sigmoid(-2.0) < sigmoid(0.0));
        assert!(sigmoid(0.0) < sigmoid(2.0));
    }

    #[test]
    fn tanh_de_zero_e_zero() {
        assert_eq!(tanh(0.0), 0.0);
    }

    #[test]
    fn tanh_simetrico() {
        // tanh é ímpar: tanh(-x) = -tanh(x)
        let x = 1.234_f32;
        assert!((tanh(-x) + tanh(x)).abs() < 1e-6);
    }

    #[test]
    fn tanh_approx_erro_pequeno_perto_de_zero() {
        // x/(1+|x|) é acurada perto de x=0 (erro abaixo de 2% para |x| ≤ 0.1).
        // Para |x| > 0.1 o erro cresce rapidamente: é uma aproximação de primeira
        // ordem, útil em shaders (sem exp) mas imprecisa em geral.
        // Veja MATH_INVENTORY.md F42: perception_temporal.wgsl usa esta forma.
        for xi in [-0.1_f32, -0.05, 0.0, 0.05, 0.1] {
            let approx = tanh_approx(xi);
            let exact = xi.tanh();
            let err = (approx - exact).abs();
            assert!(err < 0.015, "tanh_approx({}) erro={} perto de zero", xi, err);
        }
    }

    #[test]
    fn tanh_approx_mesmo_sinal_que_tanh() {
        // A aproximação nunca deve inverter o sinal
        for xi in [-3.0_f32, -1.0, 0.0, 1.0, 3.0] {
            let approx = tanh_approx(xi);
            let exact = xi.tanh();
            if exact.abs() > 1e-6 {
                assert_eq!(approx.signum(), exact.signum(),
                    "tanh_approx({}) trocou sinal: approx={} exact={}", xi, approx, exact);
            }
        }
    }

    #[test]
    fn relu_de_negativo_e_zero() {
        assert_eq!(relu(-5.0), 0.0);
        assert_eq!(relu(3.0), 3.0);
    }

    #[test]
    fn leaky_relu_preserva_gradiente_negativo() {
        let alpha = 0.01_f32;
        let out = leaky_relu(-10.0, alpha);
        assert!((out - (-0.1)).abs() < 1e-6, "leaky_relu(-10, 0.01) deve ser -0.1, obteve {}", out);
    }

    #[test]
    fn clamp_sym_respeita_limite() {
        assert_eq!(clamp_sym(5.0, 2.0), 2.0);
        assert_eq!(clamp_sym(-5.0, 2.0), -2.0);
        assert_eq!(clamp_sym(1.0, 2.0), 1.0);
    }

    #[test]
    fn state_to_i8_maximo_clamped() {
        // tanh(10) ≈ 1.0 → 127; tanh(-10) ≈ -1.0 → -127
        assert_eq!(state_to_i8(10.0), 127);
        assert_eq!(state_to_i8(-10.0), -127);
    }

    #[test]
    fn state_to_i8_de_zero_e_zero() {
        assert_eq!(state_to_i8(0.0), 0);
    }

    #[test]
    fn soft_threshold_atenua_sem_inverter() {
        // x=0.5, threshold=0.3 → sign(0.5)*max(0, 0.5-0.3) = 0.2
        let out = soft_threshold(0.5, 0.3);
        assert!((out - 0.2).abs() < 1e-6, "soft_threshold(0.5, 0.3) = {}", out);
        // sinal não deve inverter
        assert!(out >= 0.0, "sinal negativo para entrada positiva!");
    }

    #[test]
    fn soft_threshold_negativo_atenua_sem_inverter() {
        let out = soft_threshold(-0.5, 0.3);
        assert!((out - (-0.2)).abs() < 1e-6, "soft_threshold(-0.5, 0.3) = {}", out);
        assert!(out <= 0.0, "sinal positivo para entrada negativa!");
    }

    #[test]
    fn soft_threshold_abaixo_do_limiar_e_zero() {
        let out = soft_threshold(0.1, 0.5);
        assert_eq!(out, 0.0, "abaixo do limiar deve ser zero");
    }
}
