//! Estado global do binário `gpu_evolve` — flags/parâmetros lidos dentro do laço de
//! `tick()` (em `context.rs`), longe de onde os argumentos de linha de comando são
//! parseados (em `main.rs`).
//!
//! # Por que statics em vez de campos de struct
//!
//! `GpuCtx::tick()` roda por-ser, num laço apertado, e vários destes valores só existem
//! por flag de CLI (default = comportamento antigo). Passá-los como parâmetro exigiria
//! furar a assinatura de `tick()` a cada novo experimento — o histórico deste binário é
//! de crescer flag por flag. Statics resolvem isso ao custo de estado global implícito
//! (ver risco R2/R7 do REFACTOR_PLAN.md — não promover nenhum destes para a lib sem
//! encapsular numa struct de config primeiro).
//!
//! # Quem lê / quem escreve (tabela de propriedade)
//!
//! | Static | Escrito por | Lido por |
//! |---|---|---|
//! | `SLOW_ABLATED` | `main()` (flag `--sem-lento`) | `slow_ablated()` em `context.rs::tick()` |
//! | `FUEL_ON`, `FUEL_TANK`, `REFUEL_LAPS`, `SETOR_PAGA` | `main()` (flags `--fuel*`) | `main()` (laço de geração, mecânica de combustível) |
//! | `AUDITAR` | `main()` (flag `--auditar-visao`) | `context.rs::tick()` (decide se acumula auditoria) + `main()` (relatório final) |
//! | `SLOW_STEER_AUTH` | `main()` (flag `--slow-steer`) | `context.rs::tick()` (viés do volante do cérebro lento) |
//! | `PORTAL_CURVA` | `main()` (flag `--portal-curva`) | `main()` (laço de geração, escolha de setor) |
//! | `STEER_ALPHA` | `main()` (flag `--steer-alpha`) | `context.rs::tick()` (alisamento do volante) |
//! | `WALL_LIFE_DRAIN`, `STAG_LIFE_DRAIN` | `main()` (flags `--wall-life`/`--stag-life-drain`) | `context.rs::tick()` (custo de vida) |
//! | `LIFE_CAP_TICKS` | `main()` (flag `--life-cap`) | `context.rs::tick()` + `main()`, via `life_cap()` |
//! | `LIFE_EARN` | `main()` (flag `--life-earn`) | `main()` (laço de geração, ganho de vida por avanço) |
//! | `PILOTO_GANHO`, `PILOTO_THR` | `main()` (flags `--piloto*`) | `context.rs::tick()` (override do piloto fantasma) |
//! | `STEER_FIXO`, `THR_FIXO`, `STEER_ESCALA` | `main()` (flags de diagnóstico bruto) | `context.rs::tick()` (override final, depois de tudo) |
//! | `PILOTO_SENSOR` | `main()` (flag `--piloto-sensor`) | `context.rs::tick()` (piloto sensorial, raios de parede) |
//! | `ATIVACAO_ON`, `ATIVACAO_N` | `main()` (flags `--ativacao`/`--ativacao-n`) | `context.rs::tick()` (decide se/quando amostra L1/L2) |

use std::sync::atomic::Ordering;

/// f32 atômico simples sobre AtomicU32 — só precisa de store/load, sem aritmética.
pub struct AtomicF32(std::sync::atomic::AtomicU32);
impl AtomicF32 {
    pub const fn new() -> Self { Self(std::sync::atomic::AtomicU32::new(0)) }
    pub fn store(&self, v: f32, o: Ordering) { self.0.store(v.to_bits(), o) }
    pub fn load(&self, o: Ordering) -> f32 { f32::from_bits(self.0.load(o)) }
}

/// Ablação da metade lenta: quando ligada, as três saídas do lento são zeradas antes
/// do consumo. O órgão continua rodando (mesmo custo, mesmo estado, mesma evolução dos
/// pesos) — só não INFORMA. É a ablação pareada correta: isola o efeito do sinal, não
/// o efeito de existir. Comparar contra um binário sem o órgão mudaria duas coisas.
pub static SLOW_ABLATED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn slow_ablated() -> bool { SLOW_ABLATED.load(Ordering::Relaxed) }

// VOXEL_WALL (material de parede = 2, mesma ordem de `VoxelType`) foi removido daqui —
// o único consumidor era o laço de `min_wall_ahead` em `context.rs::tick()`, migrado
// para `gpu/retina_digest.wgsl` (que tem sua própria cópia da constante, `VOXEL_WALL:
// u32 = 2u`, documentada lá). Ver `gpu::retina_digest` para a equivalência completa.
/// Distância à frente abaixo da qual o ser conta como "colado" na parede.
pub const WALL_NEAR_M: f32 = 3.0;
/// Liga o combustível (mecanismo restaurado das branches que completavam voltas).
pub static FUEL_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// A cada quantas voltas reabastece. Na f1-world era 3.
pub static REFUEL_LAPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
/// Tamanho do tanque. 1,0 é o original da `f1-world`.
///
/// Com consumo `(0,002 + acel²·0,004)/s`, um tanque de 1,0 dura de 166 s (pé no fundo) a
/// 500 s (marcha lenta). Numa geração de 8.000 ticks (125 s) ele NUNCA acaba — o mecanismo
/// existe e não morde. É por isso que o tanque precisa ser varrido junto com a duração:
/// combustível só cria incentivo se puder faltar antes do fim da geração.
pub static FUEL_TANK: AtomicF32 = AtomicF32::new();
/// Se cada SETOR reabastece (padrão) ou só a volta inteira (`--refuel-lap-only`).
pub static SETOR_PAGA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
pub static AUDITAR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Modo do câmbio (`--cambio`): 0 = travado (histórico: piso marcha 2, NUNCA sobe —
/// teto físico v_max = 21,5 m/s com ratio 3,8×3,7, max_rpm 8750, raio 0,33; o pico
/// real medido da população era 21,9 m/s, colado no limitador), 1 = auto por RPM
/// (paridade com motor.rs:2476-2482), 2 = neural (auto + override intent[3]/[4],
/// paridade com motor.rs:2484-2487). Padrão 0 até o A/B validar — mudar dinâmica
/// por default sem medir é proibido (docs/PLANO_EVOLUCAO.md).
pub static CAMBIO_MODO: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Autoridade da metade lenta sobre o volante (padrão 0,20 histórico).
pub static SLOW_STEER_AUTH: AtomicF32 = AtomicF32::new();
/// Portais nas CURVAS em vez de setores uniformes.
pub static PORTAL_CURVA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Setores do circuito que precisam ser cruzados EM ORDEM.
///
/// 24 setores em 2.928 m dá ~122 m cada: perto o bastante para o próximo estar sempre ao
/// alcance de quem está avançando, longe o bastante para não premiar tremer no lugar.
pub const N_CHECKPOINTS: u32 = 24;
/// Peso do comando NOVO no alisamento do volante (1,0 = sem alisamento).
pub static STEER_ALPHA: AtomicF32 = AtomicF32::new();
/// Ticks de vida consumidos a mais por tick colado. Global porque o custo é aplicado
/// dentro do laço de leitura da retina, longe do parsing de argumentos.
pub static WALL_LIFE_DRAIN: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
/// Ticks de vida consumidos a mais por tick PARADO (stagnation_ticks > 128). 0 = sem custo.
///
/// Conserto 1 (2026-08-12): o ser para em 272 m com vida = 3010 ticks e fica ESTÁTICO até
/// hp=0 (tick 4020). Com life-earn, parar é de graça — acumula vida ao andar e a gasta
/// lentamente estagnado. Esse dreno torna a paralisia FATAL: parado por 1 s extra gasta
/// drain × 64 ticks de vida. Com drain=2 e 3010 ticks restantes, o ser tem ~23 s para
/// tentar sair; com drain=4 tem ~12 s. Análogo ao wall_drain mas trigado por AUSÊNCIA
/// de movimento. Risco baixo se drain ≤ 2: quem reduz para curva tem stag=0 (speed≥1,5).
pub static STAG_LIFE_DRAIN: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
/// Fôlego inicial quando a vida é ganha: 640 ticks = 10 s @ 64 Hz.
pub const LIFE_START: i64 = 640;
/// Teto de fôlego acumulável, em ticks. Padrão 1920 = 30 s.
///
/// Era constante e virou o LIMITE REAL do experimento sem eu perceber: com earn=6 os seres
/// convergiam para 9,2 m/s e viviam exatamente 1920 ticks — o teto, não a competência deles.
/// A aritmética do equilíbrio diz que a 9,2 m/s e earn=6 a vida deveria ser
/// `640/(1 - v·earn/64)` = 4654 ticks. O banco truncado é que impedia.
///
/// É a mesma armadilha do orçamento de 8.000 ticks: uma constante que ninguém conferiu
/// contra a tarefa vira o resultado.
pub static LIFE_CAP_TICKS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(1920);
#[inline]
pub fn life_cap() -> i64 { LIFE_CAP_TICKS.load(Ordering::Relaxed) }
/// Ticks de vida por metro avançado (lido dentro do laço, longe do parsing).
pub static LIFE_EARN: AtomicF32 = AtomicF32::new();

/// Ganho do PILOTO FANTASMA (`--piloto`). Zero = desligado (cérebro no comando).
///
/// Controlador puramente geométrico que ignora o cérebro e dirige pela racing line. Existe
/// para separar culpa do CORPO da culpa do CÉREBRO, que é a pergunta que nenhuma métrica de
/// dentro do cérebro responde: se o piloto fantasma faz a curva, o corpo obedece e o
/// problema é neural; se nem ele faz, há travamento no corpo ou na física.
///
/// O sinal do ganho é empírico — a convenção de `lateral_offset_from_line` não está
/// documentada, e chutar sinal é como se erra aqui. Rodar com +2 e −2 e ficar com o que
/// anda: é uma corrida de 3 gerações, mais barato que ler a convenção errado.
pub static PILOTO_GANHO: AtomicF32 = AtomicF32::new();
/// Acelerador do piloto fantasma (`--piloto-thr`). Padrão 0,45.
pub static PILOTO_THR: AtomicF32 = AtomicF32::new();
/// `--steer-fixo <v>`: crava o volante em `v`, DEPOIS de tudo — cérebro, KAN, alisamento,
/// viés do lento, override de ré. É o teste mais bruto possível: se o carro não descreve um
/// círculo com o volante no batente, o defeito é do corpo, não do modelo.
pub static STEER_FIXO: AtomicF32 = AtomicF32::new();
/// `--thr-fixo <v>`: idem para o acelerador.
pub static THR_FIXO: AtomicF32 = AtomicF32::new();
/// `--steer-escala <k>`: multiplica o volante do CÉREBRO por k antes de ir à física.
///
/// Medido na ablação: o corpo responde bem ao esterço (yaw 0,29 → 0,66 → 1,25 → 1,44 rad/s
/// para volante 0,10 → 0,25 → 0,50 → 0,75). A curva mais fechada do trecho exige 0,42 rad/s,
/// ou seja, volante ~0,20 basta. Com o cérebro no comando o carro gira 1,17-1,45 rad/s —
/// três vezes o necessário. Não é falta de autoridade, é excesso: escalar para baixo testa
/// se o gargalo é ganho de controle.
pub static STEER_ESCALA: AtomicF32 = AtomicF32::new();
/// `--piloto-sensor <g>`: controlador de vinte linhas que dirige SÓ por raios de parede.
/// Mede o teto da informação sensorial. Sinal do ganho é empírico.
pub static PILOTO_SENSOR: AtomicF32 = AtomicF32::new();

/// `--ativacao`: liga a coleta de métricas de ativação L1/L2 (E4, `analysis::activation`).
///
/// Lido por `context.rs::tick()` para decidir se faz o readback EXTRA de
/// `buf_cfc` (o de `final_intent` já acontece todo tick de qualquer forma — ver
/// comentário no ponto de amostragem). Desligado (padrão), custo é zero: nenhum
/// buffer novo é lido, nenhuma amostra é empilhada.
pub static ATIVACAO_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// `--ativacao-n <N>`: período de amostragem em ticks. Padrão 64 (REFACTOR_PLAN.md §E4).
pub static ATIVACAO_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(64);
