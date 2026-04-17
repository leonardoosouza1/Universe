//! # Substrate — a engine metafísica generalizada
//!
//! Este módulo formaliza a tese do projeto: **a simulação não é um jogo com IA
//! em cima. É um mundo físico onde percepção, instinto e afeto são propriedades
//! emergentes do substrato.**
//!
//! Hoje o único experimento instanciado é uma corrida (`Racer` + `VoxelWorld` +
//! `pista`). Amanhã pode ser um organismo forrageando campos de energia, uma
//! formiga navegando feromônios, ou qualquer outra cognição situada em um campo.
//!
//! A engine gráfica (Bevy) **renderiza** o substrato — ela não o *simula*. O
//! substrato vive nos dados (voxels, sinapses de dor, pesos ternários); o GPU
//! só traduz esses dados para fótons que batem na retina humana. O carro também
//! tem uma retina — mas a dele é um DDA raycast sobre o mesmo campo de voxels.
//! Humano e agente veem o MESMO mundo pelo MESMO mecanismo físico: luz refletida
//! do campo de energia.
//!
//! ## As três abstrações
//!
//! - [`World`] — o campo físico. Voxels, luz, feromônios, gradientes. Resposta
//!   única pra pergunta "o que existe em (x, y, z)?".
//! - [`Percept`] — o recorte do mundo que um agente consegue captar. Retina
//!   binocular DDA, campo escalar de cheiro, pressão local, temperatura.
//! - [`Agent`] — um corpo situado. Percebe, sente (dor/afeto), decide, age.
//!   O que distingue agentes é a *forma de perceber* e a *ética ternária* de
//!   decidir — não "qual jogo ele joga".
//!
//! ## Por que ternário (BitLinear 1.58-bit) pertence aqui
//!
//! Pesos ternários colapsam magnitudes em `{-1, 0, +1}`. Isso não é otimização —
//! é uma **ética discreta**: cada neurônio responde à pergunta "em relação a
//! este estímulo, sou a favor, neutro, ou contra?". Não há "48.3% a favor".
//! Agentes neste substrato decidem por votação categórica.
//!
//! ## Priming (axiomas a priori)
//!
//! Antes da evolução, cada agente nasce com estruturas mínimas de sobrevivência
//! (A1..A15 na corrida: "parede à esquerda → vira direita"). Isto é Kant
//! aplicado: existem formas a priori do entendimento que precedem a experiência.
//! Sem o priming, a aleatoriedade inicial extermina a população antes da
//! seleção operar.

use bevy::prelude::Vec3;

/// O campo físico. Responde "o que existe aqui?" para qualquer ponto.
///
/// Implementações concretas: [`crate::world::VoxelWorld`] (pista com asfalto,
/// grama, parede). Futuros: `PheromoneField`, `ResourceField`, `SemanticSpace`.
pub trait World {
    /// Tipo da "matéria" deste mundo (ex: [`crate::world::VoxelType`]).
    type Substance;

    /// Amostra o mundo em um ponto 3D.
    fn sample_at(&self, pos: Vec3) -> Self::Substance;

    /// Gradiente local — direção "natural" do fluxo neste ponto. Na corrida é
    /// o vetor da pista; em um mundo de cheiro seria o gradiente de feromônio.
    fn flow_at(&self, pos: Vec3) -> Vec3;
}

/// O recorte sensorial que um agente extrai do mundo.
///
/// Implementações concretas: retina binocular DDA (2×32 pixels, FOV dual-scale),
/// sensores laterais, fluxo óptico. O importante: **todo `Percept` é derivado
/// do `World` pelo mesmo mecanismo físico** — fótons, campos, colisões.
pub trait Percept {
    /// Projeta o percept no vetor de entrada do cérebro (canais 0..N).
    /// Por convenção, inteiros com sinal de 8 bits; o instinct_layer ternário
    /// quantiza tudo ao entrar.
    fn project(&self) -> &[i8];
}

/// Um corpo situado no mundo. Percebe, sofre, decide, age.
///
/// Implementações concretas: [`crate::simulation::Racer`]. Futuros: Forager,
/// Predator, Writer (um agente que percebe um espaço semântico e emite tokens).
///
/// ## Contrato de ciclo
///
/// Cada tick, o harness externo (Bevy ECS) chama:
///
/// 1. `perceive(world)` — agente amostra campo → sensory_wave
/// 2. O brain faz: `sensory_wave → instinct_layer (ternário) → intent`
/// 3. `act(intent)` — agente aplica forças no mundo via physics
/// 4. `feel(delta)` — agente atualiza dor/joy/fitness a partir do que aconteceu
pub trait Agent {
    type Intent;
    type Affect;

    /// Aposição: o agente olha para o mundo e extrai seu percept.
    fn perceive<W: World>(&mut self, world: &W);

    /// Volição: o agente transduz percept em intenção motora.
    fn decide(&mut self) -> Self::Intent;

    /// Sofrimento/Prazer: o agente integra feedback afetivo do passo anterior.
    /// Na corrida: `joy_integral` cresce com flow state, `pain_memories`
    /// marcam colisões. Este sinal é o DRIVER da seleção evolucionária —
    /// fitness é a pegada afetiva da vida.
    fn feel(&mut self, affect: Self::Affect);
}

/// Marco filosófico — não é usado em runtime, mas é o manifesto do projeto.
///
/// Nenhum dos axiomas abaixo é "opinião do programador". São *invariantes*
/// do substrato que definem o que é possível observar nele.
pub mod axioms {
    /// A1 — **Tudo é campo.** Não existe "objeto" fundamental; apenas regiões
    /// do voxel field com propriedades distintas (parede = alta densidade,
    /// grama = atrito alto, asfalto = baixa resistência).
    pub const FIELD_PRIMACY: &str = "World is a field, not a scene.";

    /// A2 — **Perceber é colidir com luz.** O agente não recebe Vec3 ground-truth;
    /// recebe fótons que ricocheteiam no campo e batem em uma retina.
    pub const PERCEPTION_IS_COLLISION: &str = "To see is to be hit by light.";

    /// A3 — **Decisão é votação ternária.** Cada peso é um voto categórico.
    /// Não há gradientes contínuos de preferência — há {contra, neutro, a favor}.
    pub const ETHICS_IS_TERNARY: &str = "To decide is to vote -1, 0, or +1.";

    /// A4 — **Fitness é afeto integrado no tempo.** Evolução seleciona não
    /// "quem chegou mais longe" mas "quem acumulou mais alegria e menos dor".
    pub const FITNESS_IS_AFFECT: &str = "Fitness is felt life, integrated.";

    /// A5 — **Instintos precedem experiência.** O priming A1..A15 é a estrutura
    /// a priori sem a qual a aleatoriedade extermina a população.
    pub const A_PRIORI_PRIMING: &str = "Instincts come before experience.";
}
