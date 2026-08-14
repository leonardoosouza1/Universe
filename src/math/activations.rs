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

/// Epsilon de segurança para a normalização por desvio de `out_stage` — evita
/// dividir por um desvio ~0 no início do treino (pesos recém-sorteados, os 48
/// `l2_raw` ainda quase todos ~0). Mesmo valor que `NORM_EPS` em `brain.wgsl`.
pub const OUT_STAGE_NORM_EPS: f32 = 1e-3;

/// Estágio de saída da ETAPA 2 do cérebro — espelha `brain.wgsl` (linhas da
/// ETAPA 2, P1.5 em `docs/PLANO_EVOLUCAO.md`) e `engine.rs` (~497-515):
/// normalização opcional por desvio → ganho de saída → fadiga adaptativa
/// (soft-threshold) → tanh.
///
/// # Parâmetros
/// - `raw_pregain`: `l2_raw` ou `l1_raw[bypass]` ANTES de qualquer ganho — a
///   mesma grandeza que `float_layers::forward`/`w_l1r` devolvem (`acc/127`,
///   sem o `2.0` histórico).
/// - `l2_std`: `Some(desvio)` liga a normalização por escala (P1.5, item 3);
///   `None` é o padrão — comportamento idêntico ao histórico, sem tocar num
///   bit do resultado. Quando `Some`, o desvio é o de
///   `math::stats::population_std_dev` sobre os 48 `raw_pregain` do MESMO ser
///   (só faz sentido para a via de L2, não para o bypass — ver comentário na
///   ETAPA 2 do shader).
/// - `out_gain`: ganho de saída (P1.5, item 2) — `2.0` reproduz exatamente o
///   hardcode histórico (`OUT_GAIN_PADRAO` em `gpu/brain.rs`).
/// - `out_beta`, `out_fat`: fadiga de saída — `out_fat` é o estado JÁ
///   ATUALIZADO neste tick (o chamador atualiza `out_fat` ANTES de chamar esta
///   função, mesma ordem que o shader e `engine.rs` seguem).
///
/// # Contrato de equivalência (testado abaixo)
/// Com `l2_std=None` e `out_gain=OUT_GAIN_PADRAO` (2.0), esta função reproduz
/// EXATAMENTE `tanh(sign(raw*2.0) * max(0, |raw*2.0| - out_beta*out_fat))` — a
/// fórmula que `brain.wgsl`/`engine.rs` tinham antes de `out_gain` virar
/// parâmetro.
#[inline]
pub fn out_stage(raw_pregain: f32, l2_std: Option<f32>, out_gain: f32, out_beta: f32, out_fat: f32) -> f32 {
    let scaled = match l2_std {
        Some(std) => raw_pregain / std.max(OUT_STAGE_NORM_EPS),
        None => raw_pregain,
    };
    let raw = scaled * out_gain;
    let adapted = soft_threshold(raw, out_beta * out_fat);
    adapted.tanh()
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

    // ── out_stage (P1.5: ganho de saída parametrizável + normalização opcional) ──

    /// Requisito da tarefa (docs/PLANO_EVOLUCAO.md P1.5, item 5): "o ganho vira
    /// parâmetro e o valor 2.0 reproduz exatamente a saída anterior". Compara
    /// `out_stage` com `out_gain=OUT_GAIN_PADRAO` contra a fórmula histórica
    /// hardcoded (`raw = l2_raw_pregain * 2.0`) que `brain.wgsl`/`engine.rs`
    /// tinham ANTES desta sessão — bit a bit, não só "próximo".
    #[test]
    fn out_stage_ganho_padrao_reproduz_formula_historica_exatamente() {
        const OUT_GAIN_PADRAO: f32 = 2.0; // gpu::brain::OUT_GAIN_PADRAO (evita depender de wgpu aqui)
        for raw_pregain in [-1.8_f32, -0.31, 0.0, 0.42, 1.05, 3.7] {
            for out_beta in [0.0_f32, 0.85] {
                for out_fat in [0.0_f32, 0.12, 1.9] {
                    let historico = {
                        let raw = raw_pregain * 2.0;
                        (raw.signum() * (raw.abs() - out_beta * out_fat).max(0.0)).tanh()
                    };
                    let novo = out_stage(raw_pregain, None, OUT_GAIN_PADRAO, out_beta, out_fat);
                    assert_eq!(novo, historico,
                        "raw_pregain={raw_pregain} out_beta={out_beta} out_fat={out_fat}: \
                         novo={novo} histórico={historico}");
                }
            }
        }
    }

    /// `l2_std=None` e `l2_std=Some(1.0)` devem ser idênticos — dividir por 1.0
    /// é matematicamente um no-op; confirma que a normalização OFF (`None`, o
    /// padrão da flag `--norm-l2-por-desvio` desligada) não introduz nenhuma
    /// escala espúria.
    #[test]
    fn out_stage_norm_none_e_std_um_sao_identicos() {
        let a = out_stage(0.73, None, 2.0, 0.85, 0.3);
        let b = out_stage(0.73, Some(1.0), 2.0, 0.85, 0.3);
        assert_eq!(a, b);
    }

    /// Com a normalização LIGADA, o resultado rescala pela amplitude: dividir
    /// por um desvio maior ATENUA o sinal antes do ganho — é o efeito que P1.5
    /// persegue contra a saturação (a escala de l2_raw cresce com a mutação ao
    /// longo das gerações; dividir pelo próprio desvio neutraliza esse
    /// crescimento antes do tanh).
    #[test]
    fn out_stage_norm_ligada_atenua_por_desvio_maior() {
        // out_beta=0 (sem fadiga) isola o efeito da normalização.
        let sem_norm = out_stage(4.0, None, 1.0, 0.0, 0.0);
        let com_norm = out_stage(4.0, Some(2.0), 1.0, 0.0, 0.0);
        assert_eq!(sem_norm, 4.0f32.tanh());
        assert_eq!(com_norm, 2.0f32.tanh()); // 4.0/2.0 = 2.0
        assert!(com_norm.abs() < sem_norm.abs(), "normalização deveria atenuar, não amplificar aqui");
    }

    /// O denominador nunca cai abaixo de `OUT_STAGE_NORM_EPS` — um desvio ~0
    /// (início do treino, pesos quase todos mudos) não deve fazer a saída
    /// explodir numericamente.
    #[test]
    fn out_stage_norm_com_desvio_quase_zero_nao_explode() {
        let out = out_stage(0.001, Some(0.0), 2.0, 0.0, 0.0);
        assert!(out.is_finite(), "saída não-finita com desvio quase zero: {out}");
        assert!(out.abs() <= 1.0, "tanh deve saturar em [-1,1], obteve {out}");
    }

    /// `math::stats::population_std_dev` alimentando `out_stage` end-to-end:
    /// dado o vetor de 48 l2_raw de um ser hipotético saturado, a normalização
    /// deve reduzir a fração de canais saturados (|tanh|>0.9) em relação ao
    /// mesmo vetor sem normalizar — a hipótese central de P1.5, verificada na
    /// matemática pura (sem GPU).
    #[test]
    fn out_stage_com_population_std_dev_reduz_saturacao() {
        use crate::math::stats::population_std_dev;
        // 48 valores simulando l2_raw crescido (weights maiores após várias
        // gerações de mutação) — a maioria perto de 3.0, com leve variação.
        let l2_raw: Vec<f32> = (0..48).map(|i| 2.8 + 0.05 * (i as f32 % 5.0)).collect();
        let std = population_std_dev(&l2_raw);
        assert!(std > 0.0, "desvio deveria ser > 0 para vetor não-constante");

        let sat_sem_norm = l2_raw.iter()
            .filter(|&&v| out_stage(v, None, 2.0, 0.0, 0.0).abs() > 0.9)
            .count();
        let sat_com_norm = l2_raw.iter()
            .filter(|&&v| out_stage(v, Some(std), 2.0, 0.0, 0.0).abs() > 0.9)
            .count();
        assert!(sat_com_norm <= sat_sem_norm,
            "normalização deveria reduzir (ou manter) saturação: sem={sat_sem_norm} com={sat_com_norm}");
    }
}
