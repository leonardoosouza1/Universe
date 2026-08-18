//! Controlador da escada de dificuldade — máquina de estados PURA (sem I/O, sem
//! `static`/`AtomicXxx` globais, só `std`). Implementa a seção "Currículo da
//! escada" de `docs/ESCADA_DIFICULDADE.md` (linhas 39-45): o orquestrador chama
//! [`Escada::avaliar`] uma vez por geração/bateria com os números que ele já
//! mede (promoção, mortalidade, recuperação da mediana) e recebe um
//! [`Veredito`] dizendo se aperta, mantém ou recua o degrau ativo.
//!
//! POSSE EXCLUSIVA deste arquivo: standalone, `mod escada;` é declarado e fiado
//! pelo orquestrador (main.rs) — este módulo não lê CLI, não conhece flags
//! (`--fuel-auto-k`, `--ruido-sensorial`, `--dano`, `--clima`, `--fantasma`),
//! não sabe rodar bateria. Só decide o PRÓXIMO degrau a partir dos números que
//! recebe por parâmetro.
//!
//! Fontes citadas inline: `docs/ESCADA_DIFICULDADE.md` (linhas do arquivo em
//! 2026-08-16) e a lição de histerese de currículo por população
//! (`~/.claude/.../memory/project_curriculo_populacao.md`, Universe
//! 2026-07-31): currículo dirigido por um indivíduo de sorte prendia a
//! população; o conserto foi avançar por fração da população, com histerese e
//! caminho de volta. Esta escada aplica a mesma lição no eixo gen-a-gen.

use std::fmt;

// ---------------------------------------------------------------------------
// Degrau: mecanismo + dose
// ---------------------------------------------------------------------------

/// Um degrau da escada: o mecanismo ativo (docs/ESCADA_DIFICULDADE.md linhas
/// 18-27, table D1-D8) and the dose that goes with it.
///
/// Only the rungs with a closed dose join the default plan. D2 (random start) is
/// deliberately OUT of the enum: it runs from generation 0 alongside D1 and is never
/// promoted on merit (docs:40 — "starts with D1(k=1.4)+D2"), so it is not a rung the
/// ladder switches on or tightens.
///
/// D7 and D8 JOINED on 2026-08-17 (BODY PHASE), when they got a mechanism and a
/// closed dose in `docs/SPECS_MUNDO_HOSTIL.md`. Before that, the rule stated just
/// above (an undefined dose does not become a variant) is what kept them out, and it
/// still holds for the rest. The PREDATOR stays out on purpose: its own spec makes
/// `Degrau::Predator(f)` conditional on the dose canary closing first, and a variant
/// with no measured dose is a rung the ladder would promote blind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Degrau {
    /// D1 — fuel-auto k apertando (docs:20). Dose = valor de k.
    FuelK(f32),
    /// D3 — ruído sensorial σ + dropout de canal por tick (docs:22). Dose = σ.
    Ruido(f32),
    /// D4 — dano oculto acumulado, RPM alto + parede degradam torque/direção
    /// na vida (docs:23, detalhe 29-37). Dose = multiplicador da rampa de
    /// acúmulo (1,0 = dose cheia do doc: +0,00015×excesso_rpm, +0,0004 colado
    /// na parede).
    Dano(f32),
    /// D5 — clima: μ por trecho + vento lateral (docs:24). Dose = piso de
    /// aderência μ (1,0 = seco; menor = mais escorregadio, mesmo eixo do
    /// nível de clima que já existia em environment.rs antes da lição do
    /// currículo por população).
    Clima(f32),
    /// D6 — fantasma do campeão anterior na pista (docs:25). Sem dose própria
    /// — é liga/desliga.
    Fantasma,
    /// D7 — scarce resource (docs:26): M energy stations with M < N beings.
    /// The dose is M, and it TIGHTENS DOWNWARD (fewer stations = more contention),
    /// the opposite direction from noise and damage. First COMPETITIVE pressure of
    /// the ladder: the only rung where what one being does (draining a station)
    /// changes the world another being finds. Spec doses: 16 -> 12 -> 8 -> 6 -> 4.
    Resource(u32),
    /// D8 — perception costs energy (docs:27): a per-channel-family subscription
    /// paid in LIFE, proportional to the use L1 makes of the family. The dose is
    /// life/tick at reference use and it tightens UPWARD. Fine steps (0.15 -> 0.30
    /// -> 0.45 -> 0.60) because the fuel curve taught that the cliff lives between
    /// two coarse jumps (docs:103-104).
    SensorCost(f32),
}

impl fmt::Display for Degrau {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Degrau::FuelK(k) => write!(f, "D1 fuel-auto k={:.2}", k),
            Degrau::Ruido(sigma) => write!(f, "D3 ruído sensorial σ={:.2}", sigma),
            Degrau::Dano(dose) => write!(f, "D4 dano oculto dose={:.2}", dose),
            Degrau::Clima(mu) => write!(f, "D5 clima μ_min={:.2}", mu),
            Degrau::Fantasma => write!(f, "D6 fantasma do campeão anterior"),
            Degrau::Resource(m) => write!(f, "D7 scarce resource M={m} stations"),
            Degrau::SensorCost(d) => write!(f, "D8 perception costs energy dose={d:.2}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Veredito
// ---------------------------------------------------------------------------

/// Resultado de uma avaliação: o que o orquestrador deve fazer no PRÓXIMO
/// gen/bateria. `Sobe`/`Rece` carregam o degrau que passa a ficar ativo — o
/// orquestrador não precisa chamar `degrau_ativo()` de novo pra saber o quê
/// religar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Veredito {
    /// Fica no degrau atual: dose certa, zona morta (nem boa nem ruim o
    /// bastante), ou já no limite do plano (início/fim) e não há pra onde ir.
    Mantem,
    /// Aperta/liga o próximo degrau. `Degrau` = novo degrau ativo.
    Sobe(Degrau),
    /// Recua um degrau (histerese). `Degrau` = novo degrau ativo.
    Rece(Degrau),
}

// ---------------------------------------------------------------------------
// Limiares — cada um com a fonte
// ---------------------------------------------------------------------------

/// Janela da "dose certa" (docs/ESCADA_DIFICULDADE.md:15): "promoção cai pra
/// 30-60%/gen". Dentro da janela = o degrau está calibrado, a escada não mexe.
pub const PROMOCAO_DOSE_MIN: f32 = 0.30;
pub const PROMOCAO_DOSE_MAX: f32 = 0.60;
/// Janela da "dose certa" (docs:15): "mortalidade 5-20%".
pub const MORTALIDADE_DOSE_MIN: f32 = 0.05;
pub const MORTALIDADE_DOSE_MAX: f32 = 0.20;

/// Lado fácil da janela (docs:15): promoção acima do teto da dose certa E
/// mortalidade abaixo do teto — a dose está fácil demais, hora de apertar.
pub const PROMOCAO_BOA_MIN: f32 = 0.60;
pub const MORTALIDADE_BOA_MAX: f32 = 0.20;

/// Lado difícil da janela (fora da dose certa, do lado ruim): promoção abaixo
/// do piso OU mortalidade muito acima do teto — a dose está pesada demais.
/// Não há linha explícita no doc pra esses dois números; são o espelho do
/// piso/teto da dose certa (10% = um terço do piso 30%; 30% = 1,5× o teto
/// 20%), a mesma folga que separa "boa" do teto da janela.
pub const PROMOCAO_RUIM_MAX: f32 = 0.10;
pub const MORTALIDADE_RUIM_MIN: f32 = 0.30;

/// Nº de avaliações boas SEGUIDAS pra subir, regime normal. Task: "promoção
/// >60% E mortalidade <20% por 3 avaliações seguidas → Sobe".
pub const APROVACOES_PARA_SUBIR: u32 = 3;
/// Nº de avaliações boas seguidas pra subir, regime PÓS-RECUO (histerese).
/// Fonte: lição de `project_curriculo_populacao.md` (Universe, 2026-07-31) —
/// currículo dirigido pelo extremo (um indivíduo/uma geração de sorte) sobe e
/// desce a cada avaliação e nada consolida; o conserto de lá foi histerese +
/// caminho de volta. Aqui: depois de recuar, a escada exige o DOBRO menos um
/// de aprovações antes de tentar subir nesse degrau de novo, senão ela oscila
/// degrau-cima/degrau-baixo indefinidamente numa dose limítrofe.
pub const APROVACOES_PARA_SUBIR_POS_RECUO: u32 = 5;
/// Nº de avaliações ruins SEGUIDAS pra recuar. Task: "por 5 seguidas → Rece".
pub const REPROVACOES_PARA_RECUAR: u32 = 5;

// ---------------------------------------------------------------------------
// Escada
// ---------------------------------------------------------------------------

/// Estado completo da escada. Todo campo é público porque o orquestrador
/// grava/recupera este struct em snapshot de bateria (mesmo espírito de
/// `Genome` — estado que atravessa gerações não pode ficar preso atrás de
/// getters).
#[derive(Debug, Clone, PartialEq)]
pub struct Escada {
    /// Sequência de degraus, na ordem em que são promovidos.
    pub plano: Vec<Degrau>,
    /// Índice do degrau ativo em `plano`.
    pub ativo: usize,
    /// Avaliações boas seguidas (streak). Qualquer avaliação que não seja
    /// "boa" zera este contador — inclusive dose-certa e zona morta.
    pub aprovacoes_seguidas: u32,
    /// Avaliações ruins seguidas (streak). Qualquer avaliação que não seja
    /// "ruim" zera este contador.
    pub reprovacoes_seguidas: u32,
    /// Quantas aprovações seguidas são necessárias pra subir na PRÓXIMA
    /// tentativa. Começa em `APROVACOES_PARA_SUBIR`; um `Rece` eleva pra
    /// `APROVACOES_PARA_SUBIR_POS_RECUO` (histerese) e volta ao normal assim
    /// que a escada consegue subir de novo (a penalidade é paga uma vez, não
    /// permanente — senão um recuo antigo travaria a escada pra sempre).
    pub limiar_subida: u32,
}

impl Escada {
    /// Plano padrão do doc (docs/ESCADA_DIFICULDADE.md, seção "Currículo da
    /// escada", linhas 39-45): "começa D1(k=1,4)... cada promoção-de-mundo
    /// APERTA o degrau ativo ou LIGA o próximo (D1 aperta k → D3 liga → D4
    /// liga → ...), um de cada vez, nunca dois juntos — degrau a degrau,
    /// corpo antes de pista".
    ///
    /// D1 aperta em 3 doses (1,4 → 1,2 → 1,0) antes de passar a bola pra D3 —
    /// "FuelK aperta o mesmo mecanismo = estágios separados" em vez de um
    /// único degrau D1 com salto direto de 1,4 pra 1,0, pra manter a régua
    /// "um degrau de cada vez" também DENTRO do próprio D1. Depois D3 → D4 →
    /// D5 → D6 ligam em sequência, doses conforme o doc.
    pub fn padrao() -> Self {
        Escada {
            plano: vec![
                // Faixas ESTENDIDAS (2026-08-16): com o plano curto anterior a escada
                // esgotava o aperto de cada mecanismo e passava adiante sem nunca entrar
                // na janela de dose (mortalidade 5-20%) — medido: 34 de 35 gerações com
                // mortalidade 0% e QUATRO mecanismos ligados ao mesmo tempo. Agora cada
                // mecanismo tem degraus suficientes para chegar a MORDER antes de ceder
                // a vez ao próximo. (O que tornava o combustível inócuo era outra coisa:
                // tanque dimensionado por VOLTA e recarregado por SETOR — ver fuel-auto
                // em main.rs.)
                // Doses medidas na CONFIG DA BATERIA (20k ticks, largada aleatória,
                // rotação de pistas), não numa config de smoke — a primeira versão
                // começava em k=2,0 porque foi calibrada com 6000 ticks e largada
                // fixa, e a bateria colapsou na primeira geração (mediana 84 m).
                // Curva medida: k=8,0 → promoção 98% · 5,0 → 42% (janela) · 3,0 → 0%
                // · 2,0 → 0%; sem combustível → 100%. O penhasco é entre 5 e 3, então
                // a escada anda FINO nessa faixa em vez de saltar por cima dela.
                Degrau::FuelK(8.0),
                Degrau::FuelK(6.5),
                Degrau::FuelK(5.5),
                Degrau::FuelK(5.0),
                Degrau::FuelK(4.5),
                Degrau::FuelK(4.0),
                Degrau::Ruido(4.0),
                Degrau::Ruido(8.0),
                Degrau::Ruido(14.0),
                Degrau::Dano(1.0),
                Degrau::Dano(2.0),
                Degrau::Dano(3.5),
                Degrau::Clima(0.85),
                Degrau::Clima(0.70),
                Degrau::Clima(0.55),
                Degrau::Fantasma,
                // D7 and D8 (BODY PHASE, 2026-08-17), doses from
                // docs/SPECS_MUNDO_HOSTIL.md.
                //
                // They come AFTER the ghost because they are the first two pressures
                // that are not "the world got harder": D7 is competitive (what one being
                // does changes another's world) and D8 charges for perception itself.
                // The order is the spec's: "D7 enters after D6 in the curriculum" and
                // "the sensor rung enters the SENSOR PHASE, after D7".
                //
                // D7 tightens DOWNWARD 16->12->8->6->4. The spec's supply/demand
                // arithmetic says M=16 already leaves half the population out (53% of
                // full demand) and M=4 covers 25% — no dose is decoration and no dose is
                // impossible by construction, which are the two failure modes that
                // energy round 2 and the fuel-auto denominator already produced here.
                Degrau::Resource(16),
                Degrau::Resource(12),
                Degrau::Resource(8),
                Degrau::Resource(6),
                Degrau::Resource(4),
                // D8 is deliberately fine: the fuel cliff sat between k=5 and k=3 and is
                // invisible to anyone stepping 8 at a time. The spec stops at 0.60
                // because above ~1.2 the ceiling arithmetic says a being recovering from
                // a slow corner enters an irreversible deficit and the population
                // collapses in the first generation.
                Degrau::SensorCost(0.15),
                Degrau::SensorCost(0.30),
                Degrau::SensorCost(0.45),
                Degrau::SensorCost(0.60),
            ],
            ativo: 0,
            aprovacoes_seguidas: 0,
            reprovacoes_seguidas: 0,
            limiar_subida: APROVACOES_PARA_SUBIR,
        }
    }

    /// Degrau ativo agora.
    pub fn degrau_ativo(&self) -> &Degrau {
        &self.plano[self.ativo]
    }

    /// Linha de telemetria pronta pra `println!` (docs:43: "Telemetria
    /// [escada] por gen com o degrau ativo").
    pub fn descrever(&self) -> String {
        format!(
            "[escada] estágio {}/{}: {} (aprovações {}/{}, reprovações {}/{})",
            self.ativo + 1,
            self.plano.len(),
            self.degrau_ativo(),
            self.aprovacoes_seguidas,
            self.limiar_subida,
            self.reprovacoes_seguidas,
            REPROVACOES_PARA_RECUAR,
        )
    }

    /// Avalia uma geração/bateria e retorna o veredito, atualizando os
    /// contadores internos (streaks) e, se subir/recuar, `ativo` +
    /// `limiar_subida`.
    ///
    /// - `promocao_pct`: fração da população (0..=1, NÃO percentual 0..=100)
    ///   que promoveu na dose atual — a mesma estatística de POPULAÇÃO (não o
    ///   campeão) que o currículo por população usa; ver
    ///   `project_curriculo_populacao.md`: dirigir pelo extremo é sorte.
    /// - `mortalidade`: fração da população que morreu (0..=1).
    /// - `recuperou`: a mediana caiu e voltou a subir em ≤15 gens
    ///   (docs:15-16, terceira perna da "dose certa"). Só entra no critério
    ///   de `Mantem`-por-dose-certa; `Sobe`/`Rece` usam só promoção+
    ///   mortalidade porque o doc não amarra recuperação a eles, só nomeia a
    ///   janela de dose certa com os três critérios juntos.
    pub fn avaliar(&mut self, promocao_pct: f32, mortalidade: f32, recuperou: bool) -> Veredito {
        // MORTALIDADE É TETO, NÃO PISO (conserto de 2026-08-16, com medição).
        //
        // O doc pedia "mortalidade 5-20%" como parte da janela. Medido na config
        // real da bateria: a pressão de combustível age TRAVANDO, não matando —
        // varredura k = {8,0 · 5,0 · 3,0 · 2,0} deu promoção {98% · 42% · 0% · 0%}
        // com mortalidade 0% em TODAS. Exigir mortalidade ≥5% tornava a janela
        // inalcançável e a escada só sabia "boa" ou "ruim": passava reto pela dose
        // certa. O piso vira teto — mortalidade alta ainda reprova (`ruim`), mas
        // não é mais requisito para reconhecer a dose calibrada.
        let dose_certa = (PROMOCAO_DOSE_MIN..=PROMOCAO_DOSE_MAX).contains(&promocao_pct)
            && mortalidade <= MORTALIDADE_DOSE_MAX
            && recuperou;
        let boa = promocao_pct > PROMOCAO_BOA_MIN && mortalidade < MORTALIDADE_BOA_MAX;
        let ruim = promocao_pct < PROMOCAO_RUIM_MAX || mortalidade > MORTALIDADE_RUIM_MIN;

        if dose_certa {
            self.aprovacoes_seguidas = 0;
            self.reprovacoes_seguidas = 0;
            return Veredito::Mantem;
        }

        if boa {
            self.reprovacoes_seguidas = 0;
            self.aprovacoes_seguidas += 1;
            if self.aprovacoes_seguidas >= self.limiar_subida {
                self.aprovacoes_seguidas = 0;
                return self.subir();
            }
            return Veredito::Mantem;
        }

        if ruim {
            self.aprovacoes_seguidas = 0;
            self.reprovacoes_seguidas += 1;
            if self.reprovacoes_seguidas >= REPROVACOES_PARA_RECUAR {
                self.reprovacoes_seguidas = 0;
                return self.recuar();
            }
            return Veredito::Mantem;
        }

        // Zona morta: nem dose certa, nem boa o bastante, nem ruim o
        // bastante (ex.: promoção 20%, mortalidade 10% — abaixo do piso da
        // janela mas não reprovável). "Seguidas" é literal: qualquer
        // avaliação fora dos dois extremos quebra a sequência, senão uma boa
        // isolada no meio de reprovações continuaria contando pro streak
        // errado.
        self.aprovacoes_seguidas = 0;
        self.reprovacoes_seguidas = 0;
        Veredito::Mantem
    }

    /// Sobe um degrau, ou segura se já estiver no fim do plano.
    fn subir(&mut self) -> Veredito {
        if self.ativo + 1 >= self.plano.len() {
            // End of plan: hold. The top is D8's hardest dose; the next mechanism
            // to enter here is the predator, and only after its dose canary closes.
            return Veredito::Mantem;
        }
        self.ativo += 1;
        // Penalidade de histerese paga (se havia): volta ao regime normal.
        self.limiar_subida = APROVACOES_PARA_SUBIR;
        Veredito::Sobe(*self.degrau_ativo())
    }

    /// Recua um degrau, ou segura se já estiver no início do plano.
    fn recuar(&mut self) -> Veredito {
        if self.ativo == 0 {
            // Já no primeiro degrau: segura. Não há como recuar mais.
            return Veredito::Mantem;
        }
        self.ativo -= 1;
        self.limiar_subida = APROVACOES_PARA_SUBIR_POS_RECUO; // histerese
        Veredito::Rece(*self.degrau_ativo())
    }
}

// ---------------------------------------------------------------------------
// Testes
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// promoção >60% + mortalidade <20% (dentro de "boa").
    const BOA_PROMO: f32 = 0.70;
    const BOA_MORT: f32 = 0.10;
    /// promoção <10% (dentro de "ruim").
    const RUIM_PROMO: f32 = 0.05;
    const RUIM_MORT: f32 = 0.10;
    /// dentro da janela de dose certa.
    const CERTA_PROMO: f32 = 0.45;
    const CERTA_MORT: f32 = 0.12;

    /// Plano FIXO de teste: a máquina de estados é o que está sob teste, não as
    /// doses de produção. Antes os testes chamavam `escada_de_teste()` e travavam
    /// a sequência real — quando o plano precisou de degraus mais finos (a escada
    /// passava por todos os mecanismos sem nunca morder: 34 de 35 gerações com
    /// mortalidade 0%), cinco testes quebraram sem que nada da lógica mudasse.
    /// Teste que quebra ao ajustar uma dose está medindo a dose, não o código.
    fn escada_de_teste() -> Escada {
        Escada {
            plano: vec![
                Degrau::FuelK(1.4),
                Degrau::FuelK(1.2),
                Degrau::FuelK(1.0),
                Degrau::Ruido(4.0),
                Degrau::Dano(1.0),
                Degrau::Clima(0.85),
                Degrau::Fantasma,
            ],
            ativo: 0,
            aprovacoes_seguidas: 0,
            reprovacoes_seguidas: 0,
            limiar_subida: APROVACOES_PARA_SUBIR,
        }
    }

    #[test]
    fn plano_de_producao_aperta_monotonicamente() {
        // O teste antigo travava a LISTA exata do plano; qualquer ajuste de dose o
        // quebrava sem que a lógica mudasse. O que importa checar é a PROPRIEDADE:
        // dentro de cada mecanismo as doses só podem ficar mais duras, na direção
        // certa de cada uma (combustível e aderência DESCEM, ruído e dano SOBEM), e
        // o fantasma fecha a escada.
        let e = Escada::padrao();
        assert_eq!(e.ativo, 0);
        assert!(
            matches!(e.plano[0], Degrau::FuelK(_)),
            "escada começa pelo combustível"
        );
        assert!(
            matches!(e.plano.last().unwrap(), Degrau::SensorCost(_)),
            "the SENSOR PHASE (D8) closes the ladder; the ghost stopped being the top \
             when D7/D8 got a closed dose"
        );
        let (mut ult_fuel, mut ult_ruido) = (f32::INFINITY, 0.0f32);
        let (mut ult_dano, mut ult_clima) = (0.0f32, f32::INFINITY);
        let (mut ult_recurso, mut ult_sensor) = (u32::MAX, 0.0f32);
        let (mut n_fuel, mut n_ruido, mut n_dano, mut n_clima) = (0, 0, 0, 0);
        let (mut n_recurso, mut n_sensor) = (0, 0);
        // Order between mechanisms: D7 only after the ghost, D8 only after D7
        // (docs/SPECS_MUNDO_HOSTIL.md). A competitive rung before the ghost would move
        // TWO axes on one promotion, which is exactly what "one at a time" forbids.
        let (mut viu_fantasma, mut viu_recurso) = (false, false);
        for d in &e.plano {
            match d {
                Degrau::FuelK(k) => {
                    assert!(
                        *k < ult_fuel,
                        "combustível tem de apertar: {k} não é < {ult_fuel}"
                    );
                    ult_fuel = *k;
                    n_fuel += 1;
                }
                Degrau::Ruido(x) => {
                    assert!(*x > ult_ruido, "ruído tem de subir");
                    ult_ruido = *x;
                    n_ruido += 1;
                }
                Degrau::Dano(x) => {
                    assert!(*x > ult_dano, "dano tem de subir");
                    ult_dano = *x;
                    n_dano += 1;
                }
                Degrau::Clima(mu) => {
                    assert!(*mu < ult_clima, "aderência tem de cair");
                    ult_clima = *mu;
                    n_clima += 1;
                }
                Degrau::Fantasma => {
                    viu_fantasma = true;
                }
                Degrau::Resource(m) => {
                    assert!(viu_fantasma, "D7 (resource) must come after the ghost");
                    assert!(
                        *m < ult_recurso,
                        "the resource tightens DOWNWARD (fewer stations = more \
                         contention): {m} is not < {ult_recurso}"
                    );
                    ult_recurso = *m;
                    viu_recurso = true;
                    n_recurso += 1;
                }
                Degrau::SensorCost(dose) => {
                    assert!(viu_recurso, "D8 (sensor cost) must come after D7");
                    assert!(*dose > ult_sensor, "the sensor cost must rise");
                    ult_sensor = *dose;
                    n_sensor += 1;
                }
            }
        }
        // D8's top dose respects the ceiling computed BEFORE running: above
        // ~1.2 life/tick the spec's arithmetic says a being recovering from a corner
        // enters an irreversible deficit and the population collapses in generation 1.
        assert!(
            ult_sensor <= 1.2,
            "sensor dose {ult_sensor} exceeds the computed ceiling (1.2 life/tick)"
        );
        // Cada mecanismo precisa de degraus suficientes para CHEGAR a morder antes
        // de ceder a vez — a lição das 34 gerações com mortalidade 0%.
        for (nome, n) in [
            ("fuel", n_fuel),
            ("ruído", n_ruido),
            ("dano", n_dano),
            ("clima", n_clima),
            ("resource", n_recurso),
            ("sensor cost", n_sensor),
        ] {
            assert!(
                n >= 3,
                "mecanismo {nome} com só {n} degrau(s): aperta pouco antes de passar adiante"
            );
        }
    }

    #[test]
    fn sobe_apos_3_boas_seguidas() {
        let mut e = escada_de_teste();
        assert_eq!(e.avaliar(BOA_PROMO, BOA_MORT, false), Veredito::Mantem);
        assert_eq!(e.avaliar(BOA_PROMO, BOA_MORT, false), Veredito::Mantem);
        let v = e.avaliar(BOA_PROMO, BOA_MORT, false);
        assert_eq!(v, Veredito::Sobe(Degrau::FuelK(1.2)));
        assert_eq!(e.ativo, 1);
        assert_eq!(e.aprovacoes_seguidas, 0);
    }

    #[test]
    fn nao_sobe_com_apenas_2_boas() {
        let mut e = escada_de_teste();
        e.avaliar(BOA_PROMO, BOA_MORT, false);
        let v = e.avaliar(BOA_PROMO, BOA_MORT, false);
        assert_eq!(v, Veredito::Mantem);
        assert_eq!(e.ativo, 0, "2 aprovações não bastam, limiar é 3");
        assert_eq!(e.aprovacoes_seguidas, 2);
    }

    #[test]
    fn recua_apos_5_ruins_seguidas() {
        let mut e = escada_de_teste();
        // sobe até o degrau 2 (índice 2, FuelK 1.0) pra ter margem de recuar.
        for _ in 0..3 {
            e.avaliar(BOA_PROMO, BOA_MORT, false);
        }
        for _ in 0..3 {
            e.avaliar(BOA_PROMO, BOA_MORT, false);
        }
        assert_eq!(e.ativo, 2);
        assert_eq!(*e.degrau_ativo(), Degrau::FuelK(1.0));

        for _ in 0..4 {
            let v = e.avaliar(RUIM_PROMO, RUIM_MORT, false);
            assert_eq!(v, Veredito::Mantem, "menos de 5 reprovações ainda segura");
        }
        let v = e.avaliar(RUIM_PROMO, RUIM_MORT, false);
        assert_eq!(v, Veredito::Rece(Degrau::FuelK(1.2)));
        assert_eq!(e.ativo, 1);
        assert_eq!(e.reprovacoes_seguidas, 0);
    }

    #[test]
    fn histerese_pos_recuo_exige_5_para_subir_de_novo() {
        let mut e = escada_de_teste();
        // leva a escada pro degrau 1 e depois recua pro 0 via 5 reprovações.
        e.avaliar(BOA_PROMO, BOA_MORT, false);
        e.avaliar(BOA_PROMO, BOA_MORT, false);
        e.avaliar(BOA_PROMO, BOA_MORT, false); // Sobe -> ativo=1
        assert_eq!(e.ativo, 1);
        for _ in 0..4 {
            e.avaliar(RUIM_PROMO, RUIM_MORT, false);
        }
        let v = e.avaliar(RUIM_PROMO, RUIM_MORT, false); // 5ª -> Rece
        assert_eq!(v, Veredito::Rece(Degrau::FuelK(1.4)));
        assert_eq!(e.ativo, 0);
        assert_eq!(e.limiar_subida, APROVACOES_PARA_SUBIR_POS_RECUO);

        // 3 boas NÃO bastam mais (histerese exige 5).
        for _ in 0..3 {
            let v = e.avaliar(BOA_PROMO, BOA_MORT, false);
            assert_eq!(v, Veredito::Mantem, "pós-recuo exige 5, não 3");
        }
        assert_eq!(e.ativo, 0);

        // a 4ª e 5ª completam a histerese.
        e.avaliar(BOA_PROMO, BOA_MORT, false);
        let v = e.avaliar(BOA_PROMO, BOA_MORT, false);
        assert_eq!(v, Veredito::Sobe(Degrau::FuelK(1.2)));
        assert_eq!(e.ativo, 1);
        // penalidade paga: volta ao regime normal.
        assert_eq!(e.limiar_subida, APROVACOES_PARA_SUBIR);
    }

    #[test]
    fn fim_do_plano_segura() {
        let mut e = escada_de_teste();
        e.ativo = e.plano.len() - 1; // Fantasma, último degrau
        for _ in 0..3 {
            let v = e.avaliar(BOA_PROMO, BOA_MORT, false);
            assert_eq!(v, Veredito::Mantem, "não há degrau além do plano");
        }
        assert_eq!(e.ativo, e.plano.len() - 1);
        assert_eq!(*e.degrau_ativo(), Degrau::Fantasma);
    }

    #[test]
    fn inicio_do_plano_segura_ao_recuar() {
        let mut e = escada_de_teste();
        assert_eq!(e.ativo, 0);
        for _ in 0..5 {
            let v = e.avaliar(RUIM_PROMO, RUIM_MORT, false);
            assert_eq!(
                v,
                Veredito::Mantem,
                "já no primeiro degrau, não há pra onde recuar"
            );
        }
        assert_eq!(e.ativo, 0);
    }

    #[test]
    fn dose_certa_mantem_e_zera_os_dois_contadores() {
        let mut e = escada_de_teste();
        e.avaliar(BOA_PROMO, BOA_MORT, false);
        e.avaliar(BOA_PROMO, BOA_MORT, false);
        assert_eq!(e.aprovacoes_seguidas, 2);
        let v = e.avaliar(CERTA_PROMO, CERTA_MORT, true);
        assert_eq!(v, Veredito::Mantem);
        assert_eq!(
            e.aprovacoes_seguidas, 0,
            "dose certa quebra o streak de boas"
        );
        assert_eq!(e.ativo, 0);
    }

    #[test]
    fn zona_morta_sem_recuperar_quebra_sequencia_de_boas() {
        // dentro da janela numérica de promoção/mortalidade mas SEM
        // recuperação -> não é dose certa nem boa nem ruim -> zona morta,
        // ainda assim quebra o streak.
        let mut e = escada_de_teste();
        e.avaliar(BOA_PROMO, BOA_MORT, false);
        e.avaliar(BOA_PROMO, BOA_MORT, false);
        assert_eq!(e.aprovacoes_seguidas, 2);
        let v = e.avaliar(CERTA_PROMO, CERTA_MORT, false); // recuperou=false
        assert_eq!(v, Veredito::Mantem);
        assert_eq!(e.aprovacoes_seguidas, 0);
    }

    #[test]
    fn descrever_contem_estagio_e_total() {
        let e = escada_de_teste();
        let linha = e.descrever();
        assert!(linha.starts_with("[escada] estágio 1/7:"));
        assert!(linha.contains("FuelK") == false); // usa o Display, não o Debug
        assert!(linha.contains("D1 fuel-auto k=1.40"));
    }
}
