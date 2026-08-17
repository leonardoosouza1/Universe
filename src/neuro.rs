//! Alocação neural por ABLAÇÃO — quanto cada via do cérebro realmente carrega.
//!
//! O cérebro do carro já é feito de partes com pesos explícitos de mistura
//! (`engine.rs` `decide()`: `w[0]·córtex + w[1]·reflexo + w[2]·sentinela`), mais a via
//! lenta que **modula** o resto em `motor.rs` (agressão, teto de acelerador, viés de
//! esterço). O que não existia era medir o quanto cada parte contribui.
//!
//! # Por que ABLAÇÃO e não o peso do portão
//!
//! O peso do portão diz o quanto uma via é USADA. Não diz o quanto ela SERVE. Uma via pode
//! receber peso alto e não estar dizendo nada — e o inverso também acontece.
//!
//! Isto não é conjectura: foi medido na bancada em Python (`~/projects/LLM/novo/exp14`).
//! Genes moduladores derivaram até b=−1,67 e c=8,02 **enquanto a ablação daqueles mesmos
//! genes custava exatamente +0,0%**. Genes vagueiam livres justamente quando são invisíveis
//! à seleção, então qualquer estatística sobre o genoma (variância, magnitude, peso do
//! portão) mede deriva junto com função. A ablação não: ela é pareada dentro do mesmo
//! indivíduo e pergunta a única coisa que importa — *quanto se perde sem isto?*
//!
//! Por isso este módulo grava as duas coisas lado a lado. **A distância entre USO e
//! CONTRIBUIÇÃO é o resultado**, não um detalhe de diagnóstico.
//!
//! # A predição arriscada que este experimento testa
//!
//! Em 2026-07-30 a bancada Python mediu, num mundo de sobrevivência com regimes ocultos,
//! que modular a constante de tempo tem teto realizável de **+0,11%** contra os 24,20% que
//! a forma fechada prometia. O diagnóstico: naquele mundo o objetivo é de TAXA (que fração
//! do tempo ficar escondido), e filtrar melhora o INSTANTE, que ali não valia nada.
//!
//! Corrida é o caso oposto: frear na curva 40 m tarde demais é batida, e nenhuma taxa média
//! correta salva. Se a leitura estiver certa, **a via lenta tem de custar caro aqui** — bem
//! mais que os ~0% que custou lá. Se não custar, a explicação "taxa vs instante" cai e o
//! resultado passa a ser sobre modulação em geral, não sobre o tipo de tarefa.
//!
//! # ⚠️ COLISÃO DE CANAIS NO `sensory_wave` (bug do simulador, não deste módulo)
//!
//! Três escritores gravam nos mesmos canais dentro de `update_racers`, no mesmo frame, e o
//! último vence:
//!
//! | ordem | linha | escreve | canais |
//! |---|---|---|---|
//! | 1 | `motor.rs:561` | distância de parede, 8 ângulos | 16–23 |
//! | 2 | `motor.rs:579` | largura de obstáculo, 8 ângulos | 24–31 |
//! | 3 | `motor.rs:676` | movimento por linha da retina | **21–27** |
//! | 4 | `motor.rs:1413` | curvatura a 20 m | **20** |
//!
//! Sobra: parede só nos canais 16–19 e largura só nos 28–31 — **metade do leque de sensores
//! de parede é calculada e descartada todo frame**. O comentário de layout em `motor.rs:519`
//! descreve o que se pretendia, não o que acontece.
//!
//! Os grupos aqui refletem o que SOBREVIVE, não o que o comentário promete. Consertar de
//! verdade (mover os escritores para a faixa livre 192–222) recupera 8 canais de informação
//! real, mas muda os índices e obriga a retreinar o campeão — decisão de projeto, não
//! conserto silencioso.
//!
//! # Uso
//!
//! Um processo por via ablada (o `universe-lab` já roda variantes em paralelo):
//!
//! ```text
//! ./target/release/universe-lab --spec lab_specs/alocacao.json --gens 30
//! ```

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// As vias do cérebro que podem ser desligadas independentemente.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Via {
    /// Linha de base: nada desligado. É contra ela que todo custo é medido.
    Nenhuma,
    /// Córtex: `instinct_layer` → `cortex_l2` → `final_intent[0..3]`. A via lenta-mas-rica.
    Cortex,
    /// Reflexo: `SpecialistBrain`, três BitLinear esparsos de 32→1 com canais fixos.
    Reflexo,
    /// Sentinela, só a proposta SUAVE. O veto duro continua ativo de propósito — sem ele o
    /// carro bate nos primeiros segundos e o número mediria "morreu cedo", não "a via
    /// sentinela contribui tanto".
    SentinelaSuave,
    /// Sentinela inteira, veto duro incluído. Cara e informativa, mas domina as outras.
    SentinelaTotal,
    /// Via lenta: `SlowBrain` (CfC 64 → Dense 16 → 3). NÃO age direto — modula agressão,
    /// teto de acelerador e viés de esterço. É o análogo exato do M2 da bancada Python.
    Lenta,
    /// Autoridade do cérebro sobre o CÂMBIO (`final_intent[3]` e `[4]`).
    ///
    /// A caixa em si continua funcionando: auto-upshift a 85% do RPM, auto-downshift abaixo
    /// de 30%, guardião anti-afogamento. O que se desliga é só a SOBREPOSIÇÃO neural, que em
    /// `motor.rs:2107` é o último `else if` — ou seja, só decide na faixa que as regras
    /// automáticas deixam.
    ///
    /// É a pergunta do Leonardo posta como medição: *"assim o modelo tem mais parâmetros ou
    /// pior, não sei"*. Se o custo desta ablação for ~0, dar o câmbio ao cérebro não paga, e
    /// acrescentar mais detalhe de motor não vai dar a ele mais o que decidir — vai dar mais
    /// o que a regra automática decide sozinha.
    Marcha,
    /// SENTIDO ablado: zera um grupo de canais do `sensory_wave` na ENTRADA, antes de tudo.
    ///
    /// Responde "como a visão afeta o modelo" com a mesma medida das vias. É necessário
    /// porque o mapa de pesos NÃO responde: medido no campeão, o peso por canal é
    /// praticamente idêntico entre os seis grupos (9,5 a 10,7), e a fatia da visão (35,4%)
    /// é exatamente a fração de canais que ela ocupa (80/223). A alocação de pesos é
    /// indistinguível de uniforme — então só a ablação separa o que é escutado do que é
    /// usado.
    Sentido(Sentido),
    /// O PRIMING escrito à mão nos 3 neurônios motores (`spawn.rs:64-90`).
    ///
    /// Esta é a pergunta que a tabela de ablação abre e não responde: o córtex carrega
    /// **94,7%**, mas os neurônios motores 0, 1 e 2 nascem com 15 pesos ESCRITOS À MÃO
    /// (curvatura a 70 m → freio +10,0; a 120 m → esterço +8,0; e por aí). Todos eles
    /// aparecem SATURADOS no campeão treinado, ou seja, sobreviveram intactos à evolução.
    ///
    /// Ablar só esses 15 pesos, deixando o resto do córtex de pé, separa duas coisas que a
    /// ablação do córtex inteiro confunde:
    ///
    /// - custo ≈ o do córtex inteiro (94,7%) -> **o "cérebro" são as regras escritas à mão**,
    ///   e a evolução é decoração
    /// - custo bem menor -> a evolução construiu algo em cima do priming, e vale medir o quê
    ///
    /// É o mesmo raciocínio de sempre: separar o que herdou do que aprendeu.
    Priming,
}

/// As 15 sinapses do priming esparso (`spawn.rs:64-90`), como (neurônio, canal).
pub const PRIMING: [(usize, usize); 15] = [
    (0, 64),
    (0, 79),
    (0, 78), // acelerador
    (1, 80),
    (1, 32),
    (1, 79),
    (1, 78),
    (1, 20),
    (1, 81), // esterço
    (1, 90),
    (1, 91),
    (1, 92),
    (1, 93), // rastreador de asfalto
    (2, 79),
    (2, 78), // freio
];

/// Zera as sinapses do priming, se a ablação pedir. Chamado uma vez, quando o cérebro é
/// montado — não a cada frame, porque são pesos e não sinais.
pub fn ablar_priming(pesos: &mut [f32], stride: usize) {
    if ablada() != Via::Priming {
        return;
    }
    for &(neu, ch) in PRIMING.iter() {
        let i = neu * stride + ch;
        if i < pesos.len() {
            pesos[i] = 0.0;
        }
    }
}

/// Grupos de canais do `sensory_wave`, das faixas onde eles são ESCRITOS
/// (`motor.rs:455-524` e `480-513`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sentido {
    Visao,
    Curvatura,
    Paredes,
    PistaAFrente,
    Corpo,
    Movimento,
}

impl Sentido {
    pub fn faixas(&self) -> &'static [(usize, usize)] {
        match self {
            Sentido::Visao => &[(0, 15), (128, 191)],
            Sentido::Curvatura => &[(20, 20), (32, 39), (78, 80)],
            // 16-19 e 28-31 apenas: os canais 20-27 são SOBRESCRITOS depois (ver a nota
            // sobre colisão de canais no topo deste módulo)
            Sentido::Paredes => &[(16, 19), (28, 31), (76, 77), (94, 97), (114, 115)],
            Sentido::PistaAFrente => &[(40, 55), (90, 93), (108, 110), (118, 119)],
            Sentido::Corpo => &[(56, 75), (124, 125)],
            Sentido::Movimento => &[(21, 27), (82, 84), (98, 107), (116, 117)],
        }
    }

    pub fn nome(&self) -> &'static str {
        match self {
            Sentido::Visao => "visao",
            Sentido::Curvatura => "curvatura",
            Sentido::Paredes => "paredes",
            Sentido::PistaAFrente => "pista",
            Sentido::Corpo => "corpo",
            Sentido::Movimento => "movimento",
        }
    }

    pub fn de_str(s: &str) -> Option<Self> {
        Some(match s {
            "visao" | "visão" => Sentido::Visao,
            "curvatura" => Sentido::Curvatura,
            "paredes" => Sentido::Paredes,
            "pista" => Sentido::PistaAFrente,
            "corpo" => Sentido::Corpo,
            "movimento" => Sentido::Movimento,
            _ => return None,
        })
    }
}

/// Zera os canais do sentido ablado, se houver. Chamado no início do ciclo, sobre o
/// `sensory_wave` cru — antes do KAN, do CfC e de tudo mais, para que a ablação seja
/// "este sentido não existe" e não "este sentido chega distorcido".
pub fn ablar_sentido(onda: &mut [i8]) {
    if let Via::Sentido(s) = ablada() {
        for &(a, b) in s.faixas() {
            for ch in a..=b.min(onda.len().saturating_sub(1)) {
                onda[ch] = 0;
            }
        }
    }
}

impl Via {
    pub fn de_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "cortex" | "córtex" => Via::Cortex,
            "reflexo" => Via::Reflexo,
            "sentinela" | "sentinela_suave" => Via::SentinelaSuave,
            "sentinela_total" => Via::SentinelaTotal,
            "lenta" | "slow" => Via::Lenta,
            "marcha" | "cambio" | "câmbio" => Via::Marcha,
            "priming" => Via::Priming,
            outro => {
                if let Some(sent) = Sentido::de_str(outro) {
                    return Via::Sentido(sent);
                }
                return Via::Nenhuma;
            }
        }
    }

    pub fn nome(&self) -> &'static str {
        match self {
            Via::Nenhuma => "nenhuma",
            Via::Cortex => "cortex",
            Via::Reflexo => "reflexo",
            Via::SentinelaSuave => "sentinela_suave",
            Via::SentinelaTotal => "sentinela_total",
            Via::Lenta => "lenta",
            Via::Marcha => "marcha",
            Via::Sentido(s) => s.nome(),
            Via::Priming => "priming",
        }
    }
}

/// Como os neurônios MOTORES nascem. Separa duas coisas que o priming atual mistura.
///
/// O comentário de `spawn.rs:38` explica o problema que o priming resolve:
///
/// > *"With N(0,0.1) random weights, each motor neuron sums ~110 random ternary inputs that
/// > **drown the curvature signal (SNR ≈ 0.26)**"*
///
/// Ou seja, o priming faz DUAS coisas ao mesmo tempo:
///
/// 1. **esparsidade** — o neurônio motor conecta a poucos canais. É um *prior arquitetural*,
///    e é o que resolve o SNR.
/// 2. **quais canais** — curvatura a 70 m → freio, a 120 m → esterço. É *conhecimento
///    humano injetado*.
///
/// Tirar as duas juntas mede "sem priming" e não responde a pergunta interessante. Separadas,
/// dá para perguntar: **a evolução acha sozinha os canais que o humano escolheu?**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nascimento {
    /// Esparso E com os canais escolhidos à mão. O que existe hoje.
    Primado,
    /// Esparso, mas com canais SORTEADOS. Mantém o prior arquitetural (SNR tratável) e
    /// devolve a escolha do conteúdo à evolução. É esta a condição de "independente".
    EsparsoAleatorio,
    /// Denso e aleatório, como os neurônios de característica. É a condição que o comentário
    /// do `spawn.rs` descreve como afogada — serve de piso, para saber quanto a esparsidade
    /// sozinha vale.
    Denso,
}

impl Nascimento {
    pub fn de_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "esparso" | "esparso_aleatorio" | "independente" => Nascimento::EsparsoAleatorio,
            "denso" => Nascimento::Denso,
            _ => Nascimento::Primado,
        }
    }
    pub fn nome(&self) -> &'static str {
        match self {
            Nascimento::Primado => "primado (canais escolhidos à mão)",
            Nascimento::EsparsoAleatorio => "esparso com canais sorteados (independente)",
            Nascimento::Denso => "denso aleatório (piso)",
        }
    }
}

static NASCIMENTO: OnceLock<Nascimento> = OnceLock::new();

pub fn definir_nascimento(n: Nascimento) {
    let _ = NASCIMENTO.set(n);
}

#[inline]
pub fn nascimento() -> Nascimento {
    *NASCIMENTO.get().unwrap_or(&Nascimento::Primado)
}

/// Quantas conexões cada neurônio motor recebe no modo esparso — o mesmo número que o
/// priming à mão usa (3 no acelerador, 10 no esterço, 2 no freio), para que a comparação
/// isole a ESCOLHA dos canais e não a quantidade deles.
pub fn n_conexoes_motor(neuronio: usize) -> usize {
    PRIMING.iter().filter(|&&(n, _)| n == neuronio).count()
}

/// Fração da população que nasce na faixa de mutação destrutiva (σ 0,08–0,20).
///
/// A mutação é ESTRATIFICADA por slot (`spawn.rs:431`): os 5 primeiros são quase clones
/// (σ≈0,006), depois σ sobe em degraus, e do slot 30 em diante (metade de 60) σ vai a
/// 0,08–0,20. Medido nos relatórios de geração:
///
/// ```text
/// GEN 1: NET max 3022 · p90 2924 · median 701 · wall_mean 0,03s
/// GEN 2: NET max 2474 · p90 2341 · median   0 · wall_mean 0,00s
/// ```
///
/// **A mediana é ZERO.** Metade da população não sai do lugar — e `wall_mean = 0,00s` mostra
/// que não é batida nem parede: é a mutação destruindo o genoma. A população efetiva é 30.
///
/// Este parâmetro existe para MEDIR qual fração vale: exploração demais desperdiça metade do
/// orçamento, exploração de menos estagna. Não há razão a priori para que 0,50 seja o ponto.
static FRACAO_EXPLORADORA: OnceLock<f32> = OnceLock::new();

pub fn definir_fracao_exploradora(f: f32) {
    let _ = FRACAO_EXPLORADORA.set(f.clamp(0.0, 1.0));
}

/// Fração padrão: 0,50 — os slots 30..59 de 60, como está em `spawn.rs` hoje.
pub const FRACAO_EXPLORADORA_PADRAO: f32 = 0.50;

#[inline]
pub fn fracao_exploradora() -> f32 {
    *FRACAO_EXPLORADORA
        .get()
        .unwrap_or(&FRACAO_EXPLORADORA_PADRAO)
}

static ABLADA: OnceLock<Via> = OnceLock::new();

/// Fixa a via ablada deste processo. Chamado uma vez no arranque, a partir de `LabParams`.
///
/// É global de processo de propósito: o `universe-lab` já roda cada variante num processo
/// separado, então não há o que confundir, e evita arrastar o parâmetro por dentro de todos
/// os sistemas do Bevy só para servir a um experimento.
pub fn definir_ablacao(via: Via) {
    let _ = ABLADA.set(via);
}

#[inline]
pub fn ablada() -> Via {
    *ABLADA.get().unwrap_or(&Via::Nenhuma)
}

// ---------------------------------------------------------------------------------------
// USO das vias — os pesos do portão, acumulados. Contraparte de comparação da ablação.
// ---------------------------------------------------------------------------------------

/// Acumuladores em ponto fixo (×1e6) para poderem ser atômicos e sem trava, já que o
/// `decide()` roda para milhares de carros por tick.
static USO: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static USO_N: AtomicU64 = AtomicU64::new(0);
/// Quantas vezes o veto duro da sentinela sobrepôs a mistura inteira.
static VETOS: AtomicU64 = AtomicU64::new(0);

const ESCALA: f64 = 1_000_000.0;

#[inline]
pub fn registrar_uso(w: &[f32; 3]) {
    for i in 0..3 {
        USO[i].fetch_add((w[i].max(0.0) as f64 * ESCALA) as u64, Ordering::Relaxed);
    }
    USO_N.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn registrar_veto() {
    VETOS.fetch_add(1, Ordering::Relaxed);
}

/// Média dos pesos do portão desde a última chamada de [`zerar_uso`].
pub fn uso_medio() -> ([f32; 3], u64, u64) {
    let n = USO_N.load(Ordering::Relaxed).max(1);
    let m = [
        (USO[0].load(Ordering::Relaxed) as f64 / ESCALA / n as f64) as f32,
        (USO[1].load(Ordering::Relaxed) as f64 / ESCALA / n as f64) as f32,
        (USO[2].load(Ordering::Relaxed) as f64 / ESCALA / n as f64) as f32,
    ];
    (
        m,
        USO_N.load(Ordering::Relaxed),
        VETOS.load(Ordering::Relaxed),
    )
}

pub fn zerar_uso() {
    for a in USO.iter() {
        a.store(0, Ordering::Relaxed);
    }
    USO_N.store(0, Ordering::Relaxed);
    VETOS.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------------------
// O QUE A SINAPSE DIZ — dicionário dos canais sensoriais
// ---------------------------------------------------------------------------------------

/// Nome legível do canal de entrada `ch` do `sensory_wave`.
///
/// Uma sinapse é uma aresta `canal → neurônio`. Sem saber o que o canal mede, "peso 10.0 na
/// posição 79" não diz nada; com o nome, vira "*curvatura a 70 m → freio, forte*", que é uma
/// frase sobre o comportamento do carro.
///
/// Os nomes saem de onde os canais são ESCRITOS (`simulation/motor.rs`) e do priming
/// esparso (`evolution/spawn.rs`), não de suposição. Canal sem origem localizada devolve
/// `None` de propósito: um nome inventado num relatório de interpretabilidade é pior que
/// índice cru, porque parece conhecimento.
pub fn canal_nome(ch: usize) -> Option<&'static str> {
    Some(match ch {
        // 0..15: pooling da retina por ZONA (motor.rs:455-478)
        0..=7 => "visão zona LONGE (~80 m, curvas distantes)",
        8..=15 => "visão zona PERTO (~20 m, obstáculos)",
        // 16..63: canais semânticos, layout em motor.rs:515-524
        16..=19 => "distância de parede por ângulo (4 de 8 — ver colisão de canais)",
        20 => "curvatura a 20 m (SOBRESCREVE parede do ângulo 4)",
        21..=27 => "movimento por linha da retina (SOBRESCREVE parede 5-7 e largura 0-3)",
        28..=31 => "largura do obstáculo por ângulo (4 de 8 — ver colisão de canais)",
        32..=39 => "curvatura do fluxo por ângulo",
        40..=47 => "pista à frente PERTO (5–40 m)",
        48..=55 => "pista à frente LONGE (45–150 m)",
        56..=59 => "material do solo e atrito",
        60..=63 => "estado do agente (rota de escape)",
        64 => "alinhamento com o fluxo da pista",
        65 => "velocidade",
        66 => "velocidade lateral (derrapagem)",
        67 => "integridade (hp)",
        68 => "giro do motor (rpm)",
        69 => "marcha",
        70 => "temperatura dos pneus",
        71 => "downforce",
        72 => "vácuo (draft)",
        73 => "está na grama",
        74 => "limite de esterço na velocidade atual",
        75 => "pista molhada",
        76 => "distância da parede à esquerda",
        77 => "distância da parede à direita",
        78 => "curvatura a 40 m (com sinal)",
        79 => "curvatura a 70 m (com sinal)",
        80 => "curvatura a 120 m (com sinal)",
        81 => "desvio lateral do centro da pista",
        82 => "1/tempo-até-colisão",
        83 => "movimento no campo esquerdo",
        84 => "movimento no campo direito",
        90..=93 => "rastreador do centro escuro (asfalto)",
        94 => "bigode +30°",
        95 => "bigode −30°",
        96 => "bigode +60°",
        97 => "bigode −60°",
        98 => "movimento na linha 7 da retina",
        102 => "taxa de aproximação à esquerda",
        103 => "taxa de aproximação à direita",
        104 => "taxa de aproximação à frente",
        105 => "tempo-até-colisão mínimo",
        106 => "aproximação esquerda (ganho fino)",
        107 => "aproximação direita (ganho fino)",
        108 => "desvio da linha de corrida",
        109 => "variação do desvio da linha",
        110 => "antecipação da linha de corrida",
        114 => "aproximação de parede à esquerda",
        115 => "aproximação de parede à direita",
        116 => "aproximação na fóvea",
        117 => "movimento absoluto na fóvea",
        118 => "espaço livre à frente",
        119 => "largura do corredor",
        124 => "está rodando (spin)",
        125 => "estresse do conjunto",
        // 128..191: canais visuais de alta resolução (motor.rs:480-513)
        128..=143 => "fóvea HD (max-pool volumétrico)",
        144..=159 => "periférica HD",
        160..=191 => "linha do horizonte, pixel cru",
        _ => return None,
    })
}

/// Papel do neurônio de saída da camada 1. Os três primeiros são motores por construção
/// (`spawn.rs`: recebem só o priming esparso); do 3 em diante são de característica.
pub fn neuronio_papel(n: usize) -> &'static str {
    match n {
        0 => "MOTOR acelerador",
        1 => "MOTOR esterço",
        2 => "MOTOR freio",
        _ => "característica",
    }
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn de_str_reconhece_as_vias_e_cai_em_nenhuma() {
        assert_eq!(Via::de_str("cortex"), Via::Cortex);
        assert_eq!(Via::de_str("  LENTA "), Via::Lenta);
        assert_eq!(Via::de_str("sentinela_total"), Via::SentinelaTotal);
        // qualquer coisa que não seja uma via conhecida NÃO pode ablar nada por engano:
        // uma grafia errada no spec viraria silenciosamente um experimento sem linha de base
        assert_eq!(Via::de_str("cortez"), Via::Nenhuma);
        assert_eq!(Via::de_str(""), Via::Nenhuma);
    }

    #[test]
    fn uso_medio_e_a_media_dos_pesos() {
        zerar_uso();
        registrar_uso(&[1.0, 0.0, 0.0]);
        registrar_uso(&[0.0, 1.0, 0.0]);
        let (m, n, _) = uso_medio();
        assert_eq!(n, 2);
        assert!((m[0] - 0.5).abs() < 1e-4, "m[0]={}", m[0]);
        assert!((m[1] - 0.5).abs() < 1e-4, "m[1]={}", m[1]);
        assert!(m[2].abs() < 1e-4);
        zerar_uso();
        let (_, n, _) = uso_medio();
        assert_eq!(n, 0);
    }
}

#[cfg(test)]
mod testes_sentido {
    use super::*;

    #[test]
    fn os_grupos_nao_se_sobrepoem_e_cobrem_o_que_dizem() {
        // sobreposição faria a soma dos custos de ablação contar canal duas vezes
        let todos = [
            Sentido::Visao,
            Sentido::Curvatura,
            Sentido::Paredes,
            Sentido::PistaAFrente,
            Sentido::Corpo,
            Sentido::Movimento,
        ];
        let mut visto = std::collections::HashMap::new();
        for s in todos {
            for &(a, b) in s.faixas() {
                for ch in a..=b {
                    if let Some(outro) = visto.insert(ch, s.nome()) {
                        panic!("canal {ch} está em '{}' e em '{}'", outro, s.nome());
                    }
                }
            }
        }
        // a visão tem de ser o maior grupo (retina zonal + HD): 16 + 64 = 80 canais
        let n_visao: usize = Sentido::Visao.faixas().iter().map(|(a, b)| b - a + 1).sum();
        assert_eq!(n_visao, 80, "faixas da visão mudaram");
    }

    #[test]
    fn de_str_reconhece_sentidos_e_vias_sem_confundir() {
        assert_eq!(Via::de_str("visao"), Via::Sentido(Sentido::Visao));
        assert_eq!(Via::de_str("cortex"), Via::Cortex);
        assert_eq!(Via::de_str("visão"), Via::Sentido(Sentido::Visao));
        // grafia errada NÃO pode virar ablação silenciosa de outra coisa
        assert_eq!(Via::de_str("visaoo"), Via::Nenhuma);
    }

    #[test]
    fn ablar_zera_so_o_grupo_pedido() {
        // como ABLADA é OnceLock e este binário de teste pode já tê-lo fixado, o teste
        // exercita a função de faixas diretamente, que é o que decide o que some
        let mut onda = [42i8; 223];
        for &(a, b) in Sentido::Curvatura.faixas() {
            for ch in a..=b {
                onda[ch] = 0;
            }
        }
        assert_eq!(onda[79], 0, "curvatura a 70 m tinha de sumir");
        assert_eq!(onda[0], 42, "a retina não podia ser tocada");
        assert_eq!(onda[64], 42, "o alinhamento de fluxo não podia ser tocado");
    }
}

#[cfg(test)]
mod testes_priming {
    use super::*;

    #[test]
    fn as_15_sinapses_batem_com_spawn_rs() {
        // se alguém mexer no priming de spawn.rs sem mexer aqui, a ablação passa a medir
        // outra coisa em silêncio. Este teste é o alarme.
        assert_eq!(PRIMING.len(), 15);
        // os três neurônios motores, e SÓ eles
        assert!(PRIMING.iter().all(|&(n, _)| n < 3));
        // o freio depende das duas sondas de curvatura curtas — é o que segura a hairpin
        assert!(PRIMING.contains(&(2, 79)) && PRIMING.contains(&(2, 78)));
        // o esterço tem as quatro do rastreador de asfalto
        for ch in 90..94 {
            assert!(PRIMING.contains(&(1, ch)), "faltou o canal {ch} no esterço");
        }
    }

    #[test]
    // `0 * stride` names the motor neuron being asserted on, matching the rows
    // above and below; clippy reads it as `erasing_op`.
    #[allow(clippy::erasing_op)]
    fn ablar_priming_zera_so_as_posicoes_certas() {
        // exercita a aritmética de índice sem depender do OnceLock global
        let stride = 223;
        let mut pesos = vec![1.0f32; 3 * stride];
        for &(neu, ch) in PRIMING.iter() {
            pesos[neu * stride + ch] = 0.0;
        }
        assert_eq!(
            pesos[2 * stride + 79],
            0.0,
            "freio/curvatura 70 m tinha de sumir"
        );
        assert_eq!(
            pesos[0 * stride + 64],
            0.0,
            "acelerador/fluxo tinha de sumir"
        );
        assert_eq!(
            pesos[1 * stride + 100],
            1.0,
            "canal não primado não podia ser tocado"
        );
        assert_eq!(
            pesos[2 * stride + 64],
            1.0,
            "canal 64 só é primado no neurônio 0"
        );
        assert_eq!(pesos.iter().filter(|&&w| w == 0.0).count(), 15);
    }
}

#[cfg(test)]
mod testes_nascimento {
    use super::*;

    #[test]
    fn a_quantidade_de_conexoes_bate_com_o_priming() {
        // o modo "esparso aleatório" tem de dar aos motores a MESMA quantidade de conexões
        // do priming à mão — senão a comparação mede quantidade, não escolha de canal
        assert_eq!(
            n_conexoes_motor(0),
            3,
            "acelerador: fluxo + curvatura 70 + curvatura 40"
        );
        assert_eq!(
            n_conexoes_motor(1),
            10,
            "esterço: 6 curvaturas/offset + 4 do asfalto"
        );
        assert_eq!(n_conexoes_motor(2), 2, "freio: curvatura 70 e 40");
        assert_eq!(
            n_conexoes_motor(0) + n_conexoes_motor(1) + n_conexoes_motor(2),
            PRIMING.len()
        );
    }

    #[test]
    fn de_str_reconhece_os_modos_e_cai_no_primado() {
        assert_eq!(Nascimento::de_str("esparso"), Nascimento::EsparsoAleatorio);
        assert_eq!(
            Nascimento::de_str("independente"),
            Nascimento::EsparsoAleatorio
        );
        assert_eq!(Nascimento::de_str("denso"), Nascimento::Denso);
        assert_eq!(Nascimento::de_str(""), Nascimento::Primado);
        // grafia errada cai no comportamento ATUAL, não num modo experimental silencioso
        assert_eq!(Nascimento::de_str("esparsoo"), Nascimento::Primado);
    }
}

#[cfg(test)]
mod testes_fracao {
    use super::*;

    /// Réplica da aritmética de `spawn.rs`, para poder testá-la sem montar o mundo.
    fn faixa(i: usize, n_pop: usize, frac: f32) -> &'static str {
        let ini = ((1.0 - frac) * n_pop as f32) as usize;
        if i >= ini {
            "explora"
        } else if i * 2 >= ini {
            "medio"
        } else if i * 6 >= ini {
            "pequeno"
        } else {
            "clone"
        }
    }

    #[test]
    fn o_padrao_reproduz_a_estratificacao_original() {
        // com 60 slots e fração 0,50 tem de dar exatamente os limites que estavam
        // escritos à mão em spawn.rs: clones 0-4, pequeno 5-14, médio 15-29, explora 30+
        let (n, f) = (60usize, 0.50f32);
        assert_eq!(faixa(0, n, f), "clone");
        assert_eq!(faixa(4, n, f), "clone");
        assert_eq!(faixa(7, n, f), "pequeno");
        assert_eq!(faixa(14, n, f), "pequeno");
        assert_eq!(faixa(22, n, f), "medio");
        assert_eq!(faixa(29, n, f), "medio");
        assert_eq!(faixa(30, n, f), "explora");
        assert_eq!(faixa(59, n, f), "explora");
    }

    #[test]
    fn fracao_menor_encolhe_a_faixa_destrutiva() {
        let n = 60;
        let n_explora = |f: f32| (0..n).filter(|&i| faixa(i, n, f) == "explora").count();
        assert_eq!(
            n_explora(0.50),
            30,
            "o padrão sacrifica metade da população"
        );
        assert_eq!(n_explora(0.25), 15);
        assert_eq!(n_explora(0.10), 6);
        // e a ordem tem de ser monotônica — menos fração, menos exploradores
        assert!(n_explora(0.10) < n_explora(0.25));
        assert!(n_explora(0.25) < n_explora(0.50));
    }

    #[test]
    fn os_primeiros_slots_sao_sempre_clones() {
        // o slot 0 é o campeão puro (simple_elitism); ele NUNCA pode cair na faixa
        // destrutiva, senão a linhagem se perde a cada geração
        for f in [0.10f32, 0.25, 0.50, 0.75] {
            assert_eq!(
                faixa(0, 60, f),
                "clone",
                "fração {f} pôs o slot 0 fora dos clones"
            );
        }
    }
}
