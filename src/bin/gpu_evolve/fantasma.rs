//! D6 da escada de dificuldade — FANTASMA DO CAMPEÃO (ver `docs/ESCADA_DIFICULDADE.md`,
//! linha `| D6 | fantasma do campeão anterior na pista | --fantasma | spec |`).
//!
//! Pressão de tempo viva por setor, **sem contato físico**: o campeão da geração
//! anterior não é desenhado na pista nem colide com ninguém — ele vira uma trilha de
//! "quanto tempo levou para chegar em cada setor", e cada ser da geração atual é
//! comparado contra essa trilha no MESMO setor. Não há posição x/y, só setor × tick.
//!
//! ## A moeda cruzada da pressão (por quê `bonus_vida`, não `penalidade_vida`)
//!
//! Lei da Moeda Cruzada (validada no ecossistema, ver `types.rs::BeingState::life_left`
//! e `CLAUDE.md`): toda ação protetora tem de custar/pagar em moeda DIFERENTE da que
//! está sendo medida, senão a política degenerada domina. Aqui a pressão do fantasma é
//! medida em TEMPO (setor × tick) mas o efeito é pago em VIDA — não em fitness direta e
//! não em distância — exatamente a mesma escolha que já funcionou para a parede
//! (custo em tempo, não em metros).
//!
//! Dentro dessa escolha, o vetor é ASSIMÉTRICO de propósito: estar à frente do
//! fantasma GANHA vida extra; estar atrás não perde NADA além do que a vida-ganha
//! normal (`--life-earn`, ganha vida por metro) já cobra. Empilhar uma segunda
//! punição em cima da mesma correlação (lento = atrás do fantasma = ganha pouca
//! vida-por-metro) é dose dobrada na MESMA falha — a lição de teto6/energia2
//! registrada em `ESCADA_DIFICULDADE.md` (janela de dose certa: mortalidade
//! 5-20%, não mais). Pressão só positiva mantém o incentivo (perseguir o fantasma
//! rende) sem arriscar estourar a janela de mortalidade.
//!
//! ## Setores não alcançados pelo fantasma
//!
//! O campeão anterior pode ter morrido/estagnado antes de cruzar todos os setores.
//! Para um setor que ele nunca alcançou, comparar tempo não faz sentido (não há T
//! contra o que comparar) — nesse caso o ser que chega lá está, por definição,
//! indo além de onde o melhor da geração passada chegou. Tratamos isso como
//! vantagem FIXA (`VANTAGEM_SETOR_NAO_ALCANCADO`, +5,0 s), grande o bastante para
//! saturar `bonus_vida` no teto (não precisa ser exata — só precisa ser "claramente
//! à frente" sem depender de infinito real chegando em contas seguintes).
//!
//! ## Ciclo de vida (mesmo processo, sem disco)
//!
//! `Fantasma` é `Clone`; a intenção é: no fim da geração N, o orquestrador chama
//! `Fantasma::gravar(...)` a partir dos cruzamentos do melhor ser e guarda o valor
//! (ex.: `Option<Fantasma>` numa variável do laço de gerações); na geração N+1, cada
//! tick consulta esse mesmo valor via `vantagem()`/`bonus_vida()`. Não atravessa
//! processos, não tem serialização em bytes — é RAM pura entre gerações.

/// Número de setores do circuito. Espelha `N_CHECKPOINTS` de `statics.rs` (24).
///
/// Duplicado aqui de propósito: este arquivo tem posse exclusiva e compila sozinho
/// (`rustc --edition 2021 --crate-type lib`, só `std`), então não pode depender de
/// `crate::statics`. Se `N_CHECKPOINTS` mudar de 24, este valor precisa acompanhar —
/// deixado documentado para quem ligar os dois módulos.
pub const N_SETORES: u32 = 24;

/// Ticks de física por segundo (64 Hz, constante do projeto — ver `context.rs`,
/// `statics.rs`: "640 ticks = 10 s @ 64 Hz").
const TICKS_POR_SEGUNDO: f32 = 64.0;

/// Vantagem atribuída quando o setor pedido está além de onde o fantasma chegou
/// (ou fora do array). Fixa e finita de propósito: não é preciso saber "quão à
/// frente", só que é uma vantagem grande — `bonus_vida` já satura bem antes disso.
pub const VANTAGEM_SETOR_NAO_ALCANCADO: f32 = 5.0;

/// Vantagem (em segundos) a partir da qual `bonus_vida` já está no teto.
const VANTAGEM_CAP_S: f32 = 2.0;

/// Teto de vida extra concedida por tick quando a vantagem é ≥ `VANTAGEM_CAP_S`.
/// Mesma unidade que `life_earn`/`wall_life` em `main.rs`: ticks de vida por tick
/// de simulação (aqui vale sempre o dobro do padrão de vida-ganha por avanço, é
/// intencionalmente pequeno perto do fôlego inicial de 640 ticks/10s).
pub const BONUS_VIDA_CAP: f32 = 2.0;

/// Tempo (em ticks) que o CAMPEÃO da geração anterior levou para alcançar cada um
/// dos `N_SETORES` setores do circuito, a partir da largada (tick 0).
///
/// `tempos_setor[s] == f32::INFINITY` significa "o campeão nunca alcançou o setor
/// `s` nesta vida" (morreu, ficou preso, ou a lista de cruzamentos gravada não
/// cobria esse setor).
#[derive(Clone, Debug, Default)]
pub struct Fantasma {
    pub tempos_setor: Vec<f32>,
}

impl Fantasma {
    /// Constrói o fantasma a partir dos cruzamentos de setor do melhor ser da
    /// geração que terminou: cada item é `(setor, tick)`, o tick em que o ser
    /// entrou naquele setor pela primeira vez.
    ///
    /// A lista pode ser:
    /// - **incompleta**: setores nunca cruzados simplesmente não aparecem, e ficam
    ///   `f32::INFINITY` em `tempos_setor`.
    /// - **desordenada ou com repetição** (ex.: o mesmo índice de setor cruzado de
    ///   novo numa volta seguinte, já que `checkpoint` em `main.rs` conta em módulo
    ///   `N_CHECKPOINTS`): mantemos o PRIMEIRO cruzamento (menor tick) de cada
    ///   setor — é o que define "quando o campeão passou por ali" na primeira
    ///   volta, que é a referência que interessa para pressão sector-a-sector.
    ///
    /// Índices `setor >= N_SETORES` são ignorados (defensivo — não deveria
    /// acontecer se o chamador já aplica `% N_CHECKPOINTS`, mas não é motivo para
    /// entrar em pânico).
    pub fn gravar(cruzamentos: &[(u32, u32)]) -> Fantasma {
        let mut tempos_setor = vec![f32::INFINITY; N_SETORES as usize];
        for &(setor, tick) in cruzamentos {
            if let Some(slot) = tempos_setor.get_mut(setor as usize) {
                let tick_f = tick as f32;
                if tick_f < *slot {
                    *slot = tick_f;
                }
            }
        }
        Fantasma { tempos_setor }
    }

    /// Vantagem do ser sobre o fantasma no `setor_atual`, em SEGUNDOS.
    ///
    /// Positivo = o ser está à frente (chegou antes de quando o fantasma chegou
    /// naquele setor). Negativo = está atrás. `vantagem = (T − tick_atual) / 64`,
    /// onde `T` é o tick em que o fantasma alcançou `setor_atual`.
    ///
    /// Se o fantasma nunca alcançou `setor_atual` (ou o índice está fora do que foi
    /// gravado), devolve a vantagem fixa `VANTAGEM_SETOR_NAO_ALCANCADO` — ver o
    /// comentário do módulo.
    pub fn vantagem(&self, setor_atual: u32, tick_atual: u32) -> f32 {
        let t_fantasma = self
            .tempos_setor
            .get(setor_atual as usize)
            .copied()
            .unwrap_or(f32::INFINITY);
        if !t_fantasma.is_finite() {
            return VANTAGEM_SETOR_NAO_ALCANCADO;
        }
        (t_fantasma - tick_atual as f32) / TICKS_POR_SEGUNDO
    }

    /// Bônus de vida (em ticks/tick, mesma unidade de `life_earn`) para uma dada
    /// vantagem em segundos (tipicamente a saída de `vantagem()`).
    ///
    /// Assimétrico de propósito (ver comentário do módulo — moeda cruzada): à
    /// frente do fantasma GANHA vida extra, linear de 0 a `VANTAGEM_CAP_S` (2 s) de
    /// vantagem, saturando em `BONUS_VIDA_CAP` (2 vida/tick) a partir daí. Atrás do
    /// fantasma (vantagem ≤ 0) NÃO perde vida — devolve 0,0. Punir quem está atrás
    /// já é papel da vida-ganha normal por avanço; duplicar a punição na mesma
    /// moeda (vida) para a mesma causa (lentidão) é dose dobrada, não dose nova.
    pub fn bonus_vida(&self, vantagem_s: f32) -> f32 {
        (vantagem_s.max(0.0) / VANTAGEM_CAP_S).min(1.0) * BONUS_VIDA_CAP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fantasma que cruzou os `N_SETORES` setores em ritmo constante de 100 ticks
    /// por setor (setor 0 em tick 0, setor 1 em tick 100, ...).
    fn fantasma_constante() -> Fantasma {
        let cruzamentos: Vec<(u32, u32)> = (0..N_SETORES).map(|s| (s, s * 100)).collect();
        Fantasma::gravar(&cruzamentos)
    }

    #[test]
    fn gravar_fantasma_completo_preenche_todos_os_setores_finitos() {
        let f = fantasma_constante();
        assert_eq!(f.tempos_setor.len(), N_SETORES as usize);
        assert!(f.tempos_setor.iter().all(|t| t.is_finite()));
        assert_eq!(f.tempos_setor[0], 0.0);
        assert_eq!(f.tempos_setor[23], 2300.0);
    }

    #[test]
    fn ser_a_frente_do_fantasma_completo_ganha_vantagem_e_bonus_positivos() {
        let f = fantasma_constante();
        // Fantasma alcançou o setor 5 em tick 500 (5*100); ser chega em tick 436,
        // 64 ticks (1 s) mais cedo.
        let vant = f.vantagem(5, 436);
        assert!(
            (vant - 1.0).abs() < 1e-4,
            "vantagem esperada ~1.0s, veio {vant}"
        );
        let bonus = f.bonus_vida(vant);
        assert!(
            (bonus - 1.0).abs() < 1e-4,
            "bonus esperado ~1.0, veio {bonus}"
        );
    }

    #[test]
    fn ser_atras_do_fantasma_completo_tem_vantagem_negativa_e_bonus_zero() {
        let f = fantasma_constante();
        // Fantasma alcançou o setor 10 em tick 1000; ser chega em 1128 (2s atrasado).
        let vant = f.vantagem(10, 1128);
        assert!(
            (vant + 2.0).abs() < 1e-4,
            "vantagem esperada ~-2.0s, veio {vant}"
        );
        assert_eq!(
            f.bonus_vida(vant),
            0.0,
            "atrás não perde nada, mas também não ganha"
        );
    }

    #[test]
    fn fantasma_incompleto_marca_setores_nao_cruzados_como_infinito() {
        // Fantasma morreu/estagnou antes de sair do setor 11 (só cruzou 0..=11).
        let cruzamentos: Vec<(u32, u32)> = (0..=11u32).map(|s| (s, s * 80)).collect();
        let f = Fantasma::gravar(&cruzamentos);
        assert!(f.tempos_setor[11].is_finite());
        assert!(f.tempos_setor[12].is_infinite());
        assert!(f.tempos_setor[23].is_infinite());

        // Ser que chega no setor 15 (o fantasma nunca chegou lá) recebe a
        // vantagem fixa, não NaN nem pânico por dividir infinito.
        let vant = f.vantagem(15, 999_999);
        assert_eq!(vant, VANTAGEM_SETOR_NAO_ALCANCADO);
        assert_eq!(f.bonus_vida(vant), BONUS_VIDA_CAP);
    }

    #[test]
    fn setor_zero_na_largada_compara_corretamente() {
        // Fantasma cruza o setor 0 exatamente na largada (tick 0) — caso comum,
        // já que `checkpoint[i] == esperado == 0` costuma valer desde o tick 0
        // (ver main.rs: setor calculado a partir da posição, esperado começa em 0).
        let cruzamentos = vec![(0u32, 0u32), (1u32, 90u32)];
        let f = Fantasma::gravar(&cruzamentos);
        assert_eq!(f.tempos_setor[0], 0.0);

        // Ser também larga no tick 0: empate exato, vantagem 0, sem bônus (mas
        // também sem punição).
        assert_eq!(f.vantagem(0, 0), 0.0);
        assert_eq!(f.bonus_vida(0.0), 0.0);

        // Ser demora 1s (64 ticks) a mais para "sair" do setor 0: 1s atrás.
        let vant = f.vantagem(0, 64);
        assert!((vant + 1.0).abs() < 1e-4);
        assert_eq!(f.bonus_vida(vant), 0.0);
    }

    #[test]
    fn setor_zero_nunca_gravado_e_tratado_como_nao_alcancado() {
        // Lista de cruzamentos que só começa no setor 1 — setor 0 nunca aparece
        // explicitamente (ex.: bug de gravação, ou pista onde o primeiro evento
        // registrado já é o setor 1). Não deve travar: vira "não alcançado".
        let cruzamentos = vec![(1u32, 90u32)];
        let f = Fantasma::gravar(&cruzamentos);
        assert!(f.tempos_setor[0].is_infinite());
        assert_eq!(f.vantagem(0, 500), VANTAGEM_SETOR_NAO_ALCANCADO);
    }

    #[test]
    fn cruzamento_duplicado_do_mesmo_setor_mantem_o_menor_tick() {
        // Simula duas entradas pro setor 3 (ex.: módulo entre voltas, ou lista
        // fora de ordem) — o PRIMEIRO cruzamento (tick menor) é o que vale.
        let cruzamentos = vec![(3u32, 300u32), (3u32, 50u32)];
        let f = Fantasma::gravar(&cruzamentos);
        assert_eq!(f.tempos_setor[3], 50.0);
    }

    #[test]
    fn setor_alem_do_indice_gravado_degrada_para_nao_alcancado() {
        // Chamador que esquecer o módulo (setor_atual >= N_SETORES) não deve
        // causar pânico de índice — `vantagem` degrada graciosamente.
        let f = Fantasma::gravar(&[(0, 0)]);
        let vant = f.vantagem(999, 10);
        assert_eq!(vant, VANTAGEM_SETOR_NAO_ALCANCADO);
    }

    #[test]
    fn bonus_vida_e_linear_ate_o_cap_e_depois_satura() {
        let f = Fantasma::default();
        assert_eq!(f.bonus_vida(0.0), 0.0);
        assert!((f.bonus_vida(1.0) - 1.0).abs() < 1e-4); // meio do caminho -> metade do cap
        assert_eq!(f.bonus_vida(2.0), BONUS_VIDA_CAP); // exatamente no cap
        assert_eq!(f.bonus_vida(10.0), BONUS_VIDA_CAP); // além do cap, continua capado
        assert_eq!(f.bonus_vida(-3.0), 0.0); // atrás não perde nada
    }

    #[test]
    fn fantasma_default_ou_vazio_da_vantagem_maxima_em_qualquer_setor() {
        // Sem fantasma anterior (ex.: geração 0, ninguém para comparar ainda):
        // `Default` produz `tempos_setor` vazio, que se comporta como "todo
        // setor não alcançado" — todo ser recebe o bônus MÁXIMO. Inofensivo: o
        // teto é pequeno (+2 vida/tick) e a pressão real liga sozinha assim que
        // houver um fantasma de verdade na geração seguinte.
        let f = Fantasma::default();
        assert!(f.tempos_setor.is_empty());
        assert_eq!(f.vantagem(0, 0), VANTAGEM_SETOR_NAO_ALCANCADO);
        assert_eq!(f.bonus_vida(f.vantagem(0, 0)), BONUS_VIDA_CAP);
    }
}
