/// Limiares nomeados para análise de ativações e saúde do modelo.
///
/// TODOS os limiares do sistema devem ser definidos aqui, com documentação
/// explicando por que o valor foi escolhido. Nenhum arquivo externo deve usar
/// literais numéricos para estas grandezas — importe as constantes deste módulo.
///
/// Fonte: REFACTOR_PLAN.md §E1 (seção thresholds.rs) e análise empírica dos
/// experimentos Universe (sessões 2026-04-xx a 2026-08-13).

/// Fração de saturação da capacidade acima da qual o modelo está "saturado".
///
/// PR/largura > SATURATION_WARN → o cérebro está usando >85% das dimensões;
/// crescer pode ser justificado (segunda condição: ganho real via Lei do Garimpo).
///
/// 0.85 = 85% — escolhido em `brain_capacity.rs::growth_verdict` como limiar
/// padrão de `saturation_threshold`.
pub const SATURATION_WARN: f32 = 0.85;

/// Limiar para considerar um neurônio "morto" (sem ativação).
///
/// |h| < DEAD_EPS por TODOS os ticks na janela de observação → neurônio morto.
/// 1e-4 é escolhido para ser acima do ruído de ponto flutuante (1e-7) mas abaixo
/// de qualquer ativação genuína (mínimo observado de canais vivos: ~0.01).
pub const DEAD_EPS: f32 = 1e-4;

/// Raio espectral máximo permitido para W_rec sem acionar alarme.
///
/// |λ_max| > SPECTRAL_MAX → dinâmica recorrente potencialmente instável.
/// 1.0 é o limiar teórico (echo-state networks, condição de echo-state property).
/// Na prática, como W_rec da GPU opera com CfC (decaimento explícito), valores
/// levemente acima de 1.0 são absorvidos pelo decaimento, mas o alarme serve de
/// aviso para investigação.
pub const SPECTRAL_MAX: f32 = 1.0;

/// Alvo do maior valor singular de L1 na INICIALIZAÇÃO do genoma (P0.5,
/// `gpu::evolution::Genome::random`) — mesma grandeza que `SPECTRAL_MAX` audita depois
/// do fato (`analysis::activation::spectral_summary`), agora usada como ALVO de
/// nascimento em vez de limiar de alarme.
///
/// Motivação: 4 corridas independentes de `Genome::random` com σ=0.30 i.i.d. cru
/// mediram ganho espectral de L1 em ~7.0 (7.101/7.050/7.048/6.983 —
/// `docs/PLANO_EVOLUCAO.md` §P0.5). A teoria de echo-state networks pede
/// `|λ_max| <= ~1` para dinâmica de entrada estável; a 7× o sinal sensorial é
/// amplificado antes mesmo da primeira mutação, e a evolução mede-se sempre
/// carregando esse viés de fundo (o campeão observado ficou 99,25% igual ao sorteio
/// inicial — cosseno 0.9925 — sinal de que a seleção empurra o init ruim, não conserta).
///
/// 1.0 == `SPECTRAL_MAX` por definição (o alarme dispara ACIMA do que a init MIRA),
/// mas mantida como constante separada: as duas grandezas significam coisas diferentes
/// (limiar de alarme pós-fato vs. alvo de nascimento) e uma mudança futura numa não
/// deve acoplar silenciosamente a outra.
pub const SPECTRAL_INIT_TARGET: f32 = 1.0;

/// Se verdadeiro, qualquer NaN detectado nas ativações aborta a análise.
///
/// NaN em ativações = bug grave (divisão por zero, overflow de gradiente, etc.).
/// Manter em `true` em produção — custo de detecção é O(n) por análise.
pub const NAN_ABORT: bool = true;

/// SNR mínimo aceitável da quantização i8 (em dB).
///
/// SNR < SNR_I8_MIN → pesos mal distribuídos (fora do range [-2, 2]) ou
/// quantização introduzindo muito ruído.
/// 6.0 dB = razão sinal/ruído de 4:1 — abaixo disso o modelo pode ter
/// comportamento imprevisível após quantização para i8.
pub const SNR_I8_MIN: f32 = 6.0;

/// Fração de neurônios mortos acima da qual dispara alarme.
///
/// Mais de 20% mortos → problema de inicialização, vanishing gradient ou
/// taxa de mutação muito alta. Validado empiricamente: modelos saudáveis
/// têm < 5% de neurônios mortos após algumas gerações.
pub const DEAD_NEURONS_WARN_PCT: f32 = 20.0;

/// Fração de neurônios saturados acima da qual dispara alarme.
///
/// |h| > 0.99 em > 30% dos ticks → o canal está preso num extremo e
/// não transporta informação gradiente útil.
pub const SATURATED_NEURONS_WARN_PCT: f32 = 30.0;

/// Entropia mínima da política (distribuição steer/throttle/brake).
///
/// Entropia < POLICY_ENTROPY_MIN → o agente está tomando quase sempre a
/// mesma ação (volante cravado, freio constante, etc.).
/// 0.5 nats é suficientemente baixo para capturar colapso sem gerar falsos
/// positivos em manobras legítimas sustentadas.
pub const POLICY_ENTROPY_MIN: f32 = 0.5;

// ---------------------------------------------------------------------------
// Testes unitários — verificam que os valores estão na faixa esperada.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturation_warn_entre_zero_e_um() {
        assert!(SATURATION_WARN > 0.0 && SATURATION_WARN < 1.0,
            "SATURATION_WARN deve estar em (0, 1), valor = {}", SATURATION_WARN);
    }

    #[test]
    fn dead_eps_positivo_e_pequeno() {
        assert!(DEAD_EPS > 0.0 && DEAD_EPS < 0.01,
            "DEAD_EPS deve ser pequeno e positivo, valor = {}", DEAD_EPS);
    }

    #[test]
    fn spectral_max_positivo() {
        assert!(SPECTRAL_MAX > 0.0,
            "SPECTRAL_MAX deve ser positivo, valor = {}", SPECTRAL_MAX);
    }

    #[test]
    fn spectral_init_target_positivo_e_igual_ao_alarme() {
        assert!(SPECTRAL_INIT_TARGET > 0.0,
            "SPECTRAL_INIT_TARGET deve ser positivo, valor = {}", SPECTRAL_INIT_TARGET);
        // Por definição documentada: o alarme dispara acima do que a init mira.
        assert_eq!(SPECTRAL_INIT_TARGET, SPECTRAL_MAX,
            "alvo de init e limiar de alarme deveriam coincidir numericamente por definição");
    }

    #[test]
    fn snr_min_positivo() {
        assert!(SNR_I8_MIN > 0.0,
            "SNR_I8_MIN deve ser positivo, valor = {}", SNR_I8_MIN);
    }

    #[test]
    fn dead_neurons_warn_pct_entre_0_e_100() {
        assert!(DEAD_NEURONS_WARN_PCT > 0.0 && DEAD_NEURONS_WARN_PCT <= 100.0);
    }

    #[test]
    fn saturated_neurons_warn_pct_entre_0_e_100() {
        assert!(SATURATED_NEURONS_WARN_PCT > 0.0 && SATURATED_NEURONS_WARN_PCT <= 100.0);
    }

    #[test]
    fn policy_entropy_min_positivo() {
        assert!(POLICY_ENTROPY_MIN > 0.0);
    }
}
