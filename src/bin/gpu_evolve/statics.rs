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
//!
//! ## Hostile world (BODY PHASE, 2026-08-17) — English rows, new code
//!
//! | Static | Written by | Read by |
//! |---|---|---|
//! | `universe::gpu::perception::BODY_CHANNELS_ON` (library, reached here through [`body_channels_on`]) | `main()` (flag `--interocepcao`, and any flag that needs the block) | `context.rs::tick()` (decides whether to pack + upload the body block) and `gpu::perception` (builds the `body_on` uniform field) |
//! | `STATIONS_N`, `STATION_CAPACITY`, `STATION_REGEN`, `STATION_RADIUS_M`, `STATION_TRANSFER` | `main()` (flags `--recurso-*`) | `main()` (places the stations once per generation) + `context.rs::tick()` (drain/recharge + the station channels) |
//! | `PREDATOR_SPEED_FRAC`, `PREDATOR_DRAIN`, `PREDATOR_RADIUS_M`, `PREDATOR_PERCEPT_M`, `PREDATOR_GRACE_TICKS`, `PREDATOR_SPAWN_BACK_M` | `main()` (flags `--predador*`) | `main()` (spawns once per generation) + `context.rs::tick()` (step, life drain, predator channels) |
//! | `TIRE_THERMAL_GAIN` | `main()` (flag `--pneu-termico`) | `main()` (builds the per-being `ThermalParams` once per generation) + `context.rs::tick()` (integrates the temperature and writes `env_grip`) |
//! | `SENSOR_COST`, `SENSOR_COST_K`, `SENSOR_COST_CAP` | `main()` (flags `--custo-sensor*`) | `main()` (recomputes the per-being subscription once per generation, next to the wiring cost) — the tick only reads the resulting `ctx.sensor_cost` vector |
//! | `PERFIL_KERNEL` | `main()` (flag `--perfil-kernel <N>`) | `context.rs::tick()` (gates the per-N-tick GPU kernel timing printout). The `GpuTimer` itself is NOT here — it owns a `QuerySet` and buffers, which is device-backed state, not a CLI knob, so it lives on `GpuCtx` (`src/gpu/timing.rs`) |
//!
//! State, as opposed to dose, is deliberately NOT here: the station charges, the
//! predator pose, the per-being tyre temperature and the per-being subscription all
//! live in `GpuCtx` / `BeingState`. A static is a knob the CLI turns; simulation
//! state in a process global is how two runs in one process contaminate each other.

use std::sync::atomic::Ordering;

/// f32 atômico simples sobre AtomicU32 — só precisa de store/load, sem aritmética.
pub struct AtomicF32(std::sync::atomic::AtomicU32);
impl AtomicF32 {
    pub const fn new() -> Self {
        Self(std::sync::atomic::AtomicU32::new(0))
    }
    pub fn store(&self, v: f32, o: Ordering) {
        self.0.store(v.to_bits(), o)
    }
    pub fn load(&self, o: Ordering) -> f32 {
        f32::from_bits(self.0.load(o))
    }
}

/// Ablação da metade lenta: quando ligada, as três saídas do lento são zeradas antes
/// do consumo. O órgão continua rodando (mesmo custo, mesmo estado, mesma evolução dos
/// pesos) — só não INFORMA. É a ablação pareada correta: isola o efeito do sinal, não
/// o efeito de existir. Comparar contra um binário sem o órgão mudaria duas coisas.
pub static SLOW_ABLATED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[inline]
pub fn slow_ablated() -> bool {
    SLOW_ABLATED.load(Ordering::Relaxed)
}

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
/// Telemetria por geração: pico de velocidade (m/s) e marcha máxima atingida.
/// Zerados no início de cada geração (main.rs), atualizados no laço de decisão
/// (context.rs, single-thread — load/compare/store basta). Provam se o câmbio
/// vive: travado NUNCA passa de 21,5 m/s nem de marcha 2.
/// Marcha máxima do auto/neural (`--cambio-teto`, padrão 6). O A/B mostrou que
/// liberar até a 6ª (69 m/s) de uma vez DESTRÓI: retina cega >28 m/s. Teto 3
/// (30,5 m/s) sobe o limite ~40% ficando na borda da percepção — degrau de
/// currículo do CORPO, não do cérebro.
pub static CAMBIO_TETO: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(6);
/// `--cambio-curriculo`: o teto COMEÇA em 2 (≡ travado) e sobe +1 a cada promoção
/// por mérito (mesmo gatilho da rotação de pista), até `CAMBIO_TETO_MAX` (o valor
/// de `--cambio-teto`). A promoção que sobe o teto NÃO roda a pista — um degrau
/// de cada vez, corpo primeiro, pista depois (teto3 fixo: readaptação incompleta
/// em 30 gens, Σ≥1V/gen 6-22 vs 24-36 do travado; teto6 direto: colapso 30-100×).
pub static CAMBIO_CURRICULO: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
pub static CAMBIO_TETO_MAX: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(6);
pub static VEL_PICO: AtomicF32 = AtomicF32::new();
pub static MARCHA_PICO: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
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
/// FÍSICA POR PARTES (Leonardo 2026-08-15: "penalizar por partes, com flags de
/// ligar/desligar, N baterias validando de verdade"). Peça 1: CUSTO DE RPM ALTO.
/// O campeão vive 73% dos ticks no limitador (medido no laboratório) — de graça.
/// Na física real, giro alto desgasta motor e bebe combustível. Moeda CRUZADA
/// (lei validada no ecossistema): custa VIDA (tempo), nunca a própria fitness.
/// Custo por tick = CUSTO_RPM × excesso, onde excesso = (rpm/max − 0,9)/0,1
/// (0 no limiar de 90%, 1 no limitador cravado). 0 = desligado (padrão).
pub static CUSTO_RPM: AtomicF32 = AtomicF32::new();
/// E-REG1 (docs/REGIOES_CEREBRAIS.md, M1): custo de FIAÇÃO por distância.
/// Neurônio L1 j tem posição j/96 num anel; canal de entrada c tem posição c/192
/// (a ordem dos canais JÁ agrupa famílias). dist = circular ∈ [0, 0,5].
/// Custo por tick do ser = λ × (Σ|w|·dist / Σ|w|) × 2  ∈ [0, λ], pago em VIDA.
/// Peso forte em fio longo encarece → seleção puxa cada neurônio pra perto do
/// que ele escuta; regiões emergem (ou a tese falha — Q medido decide). 0 = off.
pub static CUSTO_FIACAO: AtomicF32 = AtomicF32::new();
/// M4: liga o maestro — cérebro lento → 4 gates de região com RESERVA finita
/// (doc-09 §6.2). gate_b = 1 + 0,5·tanh(Σ w·slow_out + b) atenuado pela reserva;
/// usar o gate longe de 1,0 GASTA reserva (repõe devagar) → burnout mensurável.
pub static NEUROMOD_REGIOES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// `--fuel-auto k`: o 4º VALOR da regra de três do Leonardo (2026-08-16).
/// MEDIDO na rodada 2 da energia: tanque 0,30 durou ~100 s → consumo 0,003/s;
/// 1875 m em 100 s → v típica 18,75 m/s. Logo o tanque VIÁVEL para uma volta L:
///   X = (L / 18,75) × 0,003 × k     (k = folga; 1,0 = exato, sem margem)
/// Recalculado A CADA troca de pista — a dose escala com o mundo, nunca mais
/// impossível-por-construção (rodada 2) nem inócua (rodada 1). 0 = desligado.
pub static FUEL_AUTO_K: AtomicF32 = AtomicF32::new();
/// D4 (ESCADA_DIFICULDADE.md): DANO OCULTO acumulado — RPM alto e parede
/// degradam o corpo DENTRO da vida (throttle ×(1−0,5·dano), direção ×(1−0,3·dano)),
/// sem canal sensorial direto. É o estado lento oculto do EXP10 — o regime onde
/// o maestro M4 tem teto alto. 0 = desligado; 1 = escala padrão dos acúmulos.
pub static DANO_ESCALA: AtomicF32 = AtomicF32::new();
/// E-REG1b: custo de fiação só na CAUDA — top-N conexões mais fortes por neurônio
/// (0 = todas, como E-REG1). Motivo (negativo do E-REG1 com mecanismo): a média
/// sobre 18k pesos tem variância interindividual ≈ 0 (dist 0,2491 vs 0,2493 ≈
/// acaso 0,25) — imposto constante não seleciona (mesma lei do Landauer/controle).
/// A cauda é onde genomas DIFEREM: seleção ganha gradiente.
pub static FIACAO_CAUDA: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
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
pub fn life_cap() -> i64 {
    LIFE_CAP_TICKS.load(Ordering::Relaxed)
}
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

// ─── HOSTILE WORLD (BODY PHASE, 2026-08-17) ────────────────────────────────────
//
// Everything below is dose, never state (see the ownership table in the header).
// Every one of these is off at its default, so a run without the new flags is the
// old run — the bit-for-bit guarantee the A/B needs (CLAUDE.md rule 3).

/// Typical sustained speed of a healthy being, m/s. MEASURED, not chosen: the
/// energy round-2 calibration recorded 1875 m in ~100 s, and the `--fuel-auto`
/// rule of three above is built on the same number. It is the denominator that
/// turns `--predador <f>` from "m/s" into "fraction of what a being can do", so a
/// dose reads the same on every track.
pub const TYPICAL_SPEED_MS: f32 = 18.75;

/// Whether the BODY BLOCK is feeding the sensory wave this run.
///
/// There is ONE flag, and it lives in the library (`gpu::perception`), because the
/// perception kernels build their `body_on` uniform field from it once per tick. A
/// second copy here would be a second source of truth for a value two kernels must
/// agree on, which is precisely how a channel ends up written by both kernels
/// (`atomicOr` corruption) or by neither. This function exists so the ownership
/// table has a row and `context.rs` has one name to call.
#[inline]
pub fn body_channels_on() -> bool {
    universe::gpu::perception::BODY_CHANNELS_ON.load(Ordering::Relaxed)
}

/// Reference distance of the station proximity ramp (`1 - min(d/R, 1)` → ch115),
/// metres. Fixed by the D7 specification; it is a host-side dose by contract, so
/// changing it never touches WGSL (`perception/channels.rs`, BODY BLOCK header).
pub const STATION_PROX_REF_M: f32 = 300.0;

/// D7 `--recurso-escasso <M>`: number of energy stations on the track. 0 = off.
///
/// The point of the degree is that M < N beings, so for the first time what one
/// being does (draining a station) changes the world another being finds. It also
/// REPLACES the per-sector refill — leaving both on would keep an infinite fuel
/// source next to a scarce one, which is a pressure that is off wearing the costume
/// of a pressure that is on (the `--fuel-auto` denominator bug, ESCADA_DIFICULDADE.md).
pub static STATIONS_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// `--recurso-carga <C>`: max and initial charge of each station, fuel units.
pub static STATION_CAPACITY: AtomicF32 = AtomicF32::new();
/// `--recurso-regen <R>`: passive recharge per tick, fuel units. Deliberately below
/// idle consumption (0.002/64 = 3.125e-5/tick) so camping on a drained station is a
/// net LOSS — the anti-degeneration lock of the D7 spec.
pub static STATION_REGEN: AtomicF32 = AtomicF32::new();
/// `--recurso-raio <m>`: drive-through radius, metres.
pub static STATION_RADIUS_M: AtomicF32 = AtomicF32::new();
/// `--recurso-transfer <f>`: fuel per tick transferred to one being inside the radius.
pub static STATION_TRANSFER: AtomicF32 = AtomicF32::new();

/// `--predador <f>`: top speed of the predator as a fraction of [`TYPICAL_SPEED_MS`].
/// 0 = off. Escapability is by construction while f < 1: a healthy being outruns it
/// on a straight, so the pressure bites the slow and the cornering, and the dose stays
/// sweepable instead of being a coin flip.
pub static PREDATOR_SPEED_FRAC: AtomicF32 = AtomicF32::new();
/// `--predador-dreno <D>`: life ticks per tick drained at zero distance. The ramp is
/// continuous and quadratic, `D * (1 - d/R)^2`, so contact is never an instant kill —
/// a step function would make mortality a threshold artefact rather than a dose.
pub static PREDATOR_DRAIN: AtomicF32 = AtomicF32::new();
/// `--predador-raio <m>`: radius of the drain ramp, metres.
pub static PREDATOR_RADIUS_M: AtomicF32 = AtomicF32::new();
/// `--predador-percep <m>`: reference distance of the ch124 proximity ramp, metres.
pub static PREDATOR_PERCEPT_M: AtomicF32 = AtomicF32::new();
/// `--predador-gracia <ticks>`: immunity at the start of a generation. Without it the
/// arithmetic says a newborn (LIFE_START = 640) dies in 128 ticks = 2 s at D = 4, i.e.
/// before it could possibly have fled — that is the metric hitting a limit, not a being
/// failing (CLAUDE.md rule 4).
pub static PREDATOR_GRACE_TICKS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(1280);
/// `--predador-spawn <m>`: how far BEHIND the generation's start line the predator is
/// placed, in metres of track arc.
pub static PREDATOR_SPAWN_BACK_M: AtomicF32 = AtomicF32::new();

/// `--pneu-termico <gain>`: depth of the tyre-thermal grip curve. 0 = off.
///
/// The dose scales how far grip can fall away from the optimal temperature:
/// `floor = 1 - gain * (1 - 0.70)`. gain = 0 gives floor 1.0, i.e. `grip_factor ≡ 1`
/// and the mechanism is exactly off; gain = 1 is the legacy floor 0.70. The DC gain is
/// re-normalised to 1.0 at every dose (`tire_thermal::normalize_dc`), so a bigger dose
/// is a WIDER SWING, never less grip on average — otherwise the A/B would measure "less
/// grip is harder" instead of the dynamic, the confound this project already paid for.
pub static TIRE_THERMAL_GAIN: AtomicF32 = AtomicF32::new();

/// D8 `--custo-sensor <DOSE>`: life/tick charged for perception at reference use.
/// 0 = off. The subscription is per FAMILY of channels (`channels::CHANNEL_FAMILIES`),
/// so pruning one family removes a whole term. E-REG1 failed because its tax was the
/// mean of 18 432 weights, whose inter-individual variance goes to zero by the CLT —
/// a constant tax does not select. A family sum has somewhere to go.
pub static SENSOR_COST: AtomicF32 = AtomicF32::new();
/// `--custo-sensor-k <K>`: size of the per-family top-K tail. Default 4.
pub static SENSOR_COST_K: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(4);
/// `--custo-sensor-cap <C>`: ceiling of the normalised per-family use `u_F`. Default
/// 2.0, which also bounds the whole tax at `2 * DOSE` — the ceiling the metric plan is
/// computed against.
pub static SENSOR_COST_CAP: AtomicF32 = AtomicF32::new();

/// `--perfil-kernel <N>`: print per-kernel GPU milliseconds every N ticks. 0 = off
/// (the default — this instrument is diagnostic-only and must not change what a run
/// measures). First per-kernel timing this project has ever had; see
/// `src/gpu/timing.rs` for why no external profiler could produce this before.
pub static PERFIL_KERNEL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
#[inline]
pub fn perfil_kernel() -> u32 {
    PERFIL_KERNEL.load(Ordering::Relaxed)
}
