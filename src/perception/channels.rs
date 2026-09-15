/// Registro canônico dos 192 canais da onda sensorial.
///
/// Fonte de verdade única: elimina literais numéricos espalhados no código.
/// Cada variante nomeada representa um canal com semântica documentada.
/// Canais da retina (grupos 0-19, 29-63, 76-77, 82, 90-97, 128-191) são
/// escritos por perception.wgsl — não precisam de variante individual pois
/// são acessados em loops sobre o bloco retina, não por índice simbólico.
///
/// # Como isto foi verificado
///
/// Cada variante abaixo foi conferida contra o `write_channel(being, N, ...)`
/// que efetivamente grava o canal em `perception.wgsl` / `perception_temporal.wgsl`
/// / `perception_motion.wgsl` (grep de todos os `write_channel(being` nesses três
/// arquivos), e contra o código CPU que alimenta esses kernels em `gpu_evolve.rs`
/// (arrays `prop`, `extra`, `extra2`, `rv`). Comentários de cabeçalho nos próprios
/// .wgsl (ex.: o bloco de slots 20-31 em `perception_temporal.wgsl`) documentam
/// nomes de variável ANTIGOS que não batem mais com o código atual — não confiar
/// neles isoladamente. A primeira passagem deste arquivo (antes desta revisão)
/// tinha 4 famílias de canal com índice trocado ou fórmula errada, todas achadas
/// e corrigidas aqui:
///   - ch124-127: `JoyTick`/`JoyStreak`/`Spinning`/`CompStress` estavam em ordem
///     errada (comparado a `write_channel(being, 124..127, ...)` real).
///   - ch102-104: `ApproachRate0/1/2` tinham direção (esq/dir/frontal) trocada.
///   - ch105: nomeado `TtcFrontal`, mas a fórmula é o MÍNIMO entre esq/dir/frontal
///     (não é "frontal") — renomeado para `TtcMin`.
///   - ch106-107: nomeados `TtcLateral0/1`, mas a fórmula é approach-rate de
///     novo (ganho 64/15), não time-to-collision — renomeados para
///     `ApproachRateFineL/R`.
///   - ch83-84: documentados como "linha 6"/"linha 7", mas são fluxo óptico
///     hemisférico ESQUERDO/DIREITO (soma das rows 6 e 8, metade esq/dir).
///   - ch98: documentado como "aceleração", mas é a MESMA fórmula de motion
///     delta de ch21-27 aplicada à retina row 7 — renomeado `MotionDelta7`.
///   - ch73 (`OnGrass`) e ch124 (`Spinning`): codificação é `{0, -128}`, não
///     `{0, 127}` como as outras flags — `select(0, -128, cond)` no shader.
///   - ch65 (`SpeedFwd`) e ch69 (`Gear`): fórmula tinha o fator de escala errado
///     no comentário (×2.0 e ×20.0 respectivamente, não "/30m/s" nem cast puro).
///
/// # FASE CANAIS (2026-08-14) — auditoria dos 37 canais reprovados
///
/// Ponto de partida: `--audit-vision` (3 eixos) reprovou 37/192 canais em
/// `gpu_evolve` (o binário de TREINO — não usa `motor.rs`/Bevy). Investigação
/// canal a canal, sem rodar GPU (2 baterias de currículo ocupando-a no momento —
/// tudo abaixo é leitura de código + `cargo test`, não medição):
///
/// - **"sem fonte em kernel nenhum" (99, 100, 121, 122):** FALSO-POSITIVO. Já
///   alimentados via `update_extra2_temporal()` (context.rs) → `temporal[54-57]`
///   → `perception_temporal.wgsl` (write_channel 99/100/121/122). Conferido
///   linha a linha (context.rs:614-663 × perception_temporal.wgsl:409-425): os
///   slots batem exatamente. O diagnóstico é anterior a este commit de wiring
///   (`git log` mostra a fiação já no HEAD antes desta sessão) — nenhuma mudança
///   de código foi necessária aqui, só esta nota para não reabrir a investigação.
/// - **ch87, 124, 126, 127 (slip/spinning/joy):** mesmo caso — já alimentados
///   pelo mesmo mecanismo (`extra2[0,3,1,2]`). `joy_tick`/`joy_streak` conferidos
///   como computados de verdade a cada tick em `main.rs:991-1028` (progresso ×
///   superfície), não um stub constante.
/// - **ch125 (`CompStress`): BUG real, corrigido.** A fórmula tinha
///   `if i > 0 {0.5} else {0.0}` como piso — dependia do ÍNDICE do ser, não da
///   física. Como a auditoria rastreia UM ser vivo ao longo do tempo (tende a
///   ser o de índice mais baixo), esse ser ficava preso em {0.0, 1.0} e o canal
///   parecia constante mesmo "alimentado". Corrigido em `context.rs` para depender
///   só de distância real (ver comentário lá).
/// - **ch101, ch114-117: ALIMENTAR — eram mesmo só motor.rs (Bevy CPU).**
///   `gpu_evolve` não importa `motor.rs`; nesse binário esses 5 canais realmente
///   não tinham fonte. Ligados em `perception_temporal.wgsl` (ch101, reusa
///   `curv_at_d` já validado para 111-113) e `perception_motion.wgsl` (ch114-117,
///   reusa retina_cur/retina_prev que o kernel já lê para ch21-27/83-84). Normalizadores
///   novos (FOVEA_NORM, MAX_OFFSET_RATE) são ESTIMATIVAS não validadas por medição —
///   ver comentário em cada shader.
/// - **ch109 (`LineOffsetDeriv`): BUG real, corrigido.** Estava hardcoded 0.0 (CPU
///   nunca implementou a derivada). O slot [8] do buffer temporal já guardava o
///   valor do tick anterior (escrito pelo próprio shader, nunca lido) — a derivada
///   passou a ser computada dentro do shader, sem precisar de novo dado da CPU.
/// - **ch70 (`TireTemp`): [REVERTIDO 2026-08-15 — ver doc de DEAD_L1_INPUTS:
///   o canal está VIVO; physics_power_c.wgsl implementa a térmica.] Veredito
///   original (baseado no kernel errado): DESLIGAR — não há dado real por trás, e não é bug
///   corrigível nesta fase.** `physics_c.wgsl:290` confirma: o slot de temperatura
///   do pneu é um PASSTHROUGH puro (`c.wheels[1].x` copiado sem alteração) — o
///   comentário do próprio shader diz "térmica no Kernel B não implementada para
///   poupar reg". Implementar o modelo térmico é `PLANO_EVOLUCAO.md` P2 item 2/3
///   (fase de CORPO, escopo bem maior que fiação de canal). Até lá o valor nasce e
///   morre igual — zerado no L1 do genoma (`DEAD_L1_INPUTS` abaixo, consumido por
///   `Genome::random` em `gpu/evolution.rs`) para o cérebro não nascer gastando
///   atenção com estática. NÃO é poda de neurônio (a ablação 96→64 provou que
///   neurônio não é descartável, ver PLANO_EVOLUCAO.md) — é só não alimentar peso
///   de entrada que lê canal constante. Reversível: basta remover a entrada de
///   `DEAD_L1_INPUTS` quando P2 implementar a térmica do pneu.
/// - **Demais 21 canais da lista "escritos mas constantes"** (19, 25, 26, 41, 43,
///   44, 46, 56, 58, 59, 64, 69, 73, 83, 84, 89, 94-98): lidos e comparados contra
///   canais IRMÃOS da mesma fórmula que NÃO foram reprovados (ex.: ch56/58/59
///   [material lookahead] vs ch57, que usa a mesma codificação; ch19 [wall_fov ai=3]
///   vs ch16-18/20-23, mesmíssima fórmula por ângulo). O padrão observado é
///   ASSIMÉTRICO e não-sistemático dentro de cada família de fórmula idêntica —
///   incompatível com bug de código (que afetaria a família inteira ou nenhum
///   membro) e compatível com NATUREZA: população atual dirige bem numa pista sem
///   buracos (obediência 0,74), então sinais de "evento raro" (grama, quase-colisão,
///   whisker <10m, rodopio, ré) ficam perto de zero na maior parte da corrida — é
///   informação real (crítica quando dispara), só rara sob as condições medidas.
///   `ch70` (`TireTemp`) é a ÚNICA exceção onde achei evidência ESTRUTURAL (não
///   dependente de trajetória) de constância — daí ser o único do lote com DESLIGAR.
///   Os outros 20 permanecem com peso de entrada vivo — não confirmei "natureza"
///   por medição (não pude rodar `--auditar-visao`), só por leitura de código; se
///   uma bateria futura mostrar que algum deles é MESMO morto (constante mesmo com
///   população variada/pista com buracos), reabrir aqui com o número medido.
///
/// # Arquitetura dos 192 canais
///
/// ```text
///  0..=19   → perception.wgsl   (retina near + fovea primária)
///  20       → motor.rs          (canal de controle CPU)
///  21..=27  → motion.wgsl       (motion delta por linha)
///  28       → motor.rs
///  29..=31  → perception.wgsl
///  32       → motor.rs
///  33..=63  → perception.wgsl   (retina/flow estendido)
///  64..=75  → temporal.wgsl     (propriocepção + navegação)
///  76..=77  → perception.wgsl   (distância lateral)
///  78..=80  → temporal.wgsl     (curvatura da pista por comprimento de arco)
///  81       → temporal.wgsl
///  82       → perception.wgsl   (urgência frontal)
///  83..=84  → motion.wgsl       (flow ótico hemisférico esq/dir)
///  85..=86  → temporal.wgsl     (EMA distância lateral)
///  87       → temporal.wgsl     (slip_angle, injetado pela CPU)
///  88..=89  → temporal.wgsl
///  90..=97  → perception.wgsl   (whiskers + near-wall)
///  98       → motion.wgsl       (motion delta retina row 7)
///  99..=100 → temporal.wgsl     (edge density linhas 6/7, injetado pela CPU)
///  101      → temporal.wgsl     (curvatura 260m — FASE CANAIS 2026-08-14, era motor.rs-só)
///  102..=113→ temporal.wgsl     (approach rate, TTC mínimo, racing line, curvatura)
///  114..=117→ motion.wgsl       (approach hemisférico + fóvea — FASE CANAIS 2026-08-14,
///                                era motor.rs-só; motor.rs continua escrevendo no
///                                caminho Bevy interativo, sem conflito de binário)
///  118..=120→ temporal.wgsl     (racing line dist norms + memory CfC)
///  121..=122→ temporal.wgsl     (steer/brake_mem)
///  123      → temporal.wgsl     (whisker normalizado)
///  124..=127→ temporal.wgsl     (spinning/comp_stress/joy_tick/joy_streak injetados pela CPU)
///  128..=191→ perception.wgsl   (retina estendida / horizon scan)
/// ```
///
/// Nota: o `canal_tem_fonte()` original também listava `192..=207` como
/// `temporal.wgsl`, mas esses índices estão ACIMA de N_CHANNELS=192 e nunca
/// são acessados no loop de auditoria — preservados aqui como `source_for()`
/// retorna "sem_fonte" para ch >= N_CHANNELS.

/// Total de canais na onda sensorial — deve bater com
/// `config::iara::brain::SENSORY_WAVE_SIZE`.
pub const N_CHANNELS: usize = 192;

/// Canais nomeados com semântica documentada.
///
/// Variantes com discriminante explícito = índice na onda [0, 191].
/// Para compilar: Rust exige que discriminantes explícitos em enums sem campos
/// sejam únicos. Usamos `#[repr(usize)]` e `index()` via `as usize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Channel {
    /// Remaining fuel as a fraction of tank capacity. Native GPU body channel;
    /// written from the resident temporal body buffer. [0, 127].
    FuelFrac = 20,
    // ── Motion per-row (motion.wgsl ch 21-27) ────────────────────────────────
    /// Delta de posição retina linha 0 entre ticks consecutivos.
    /// GPU: encoded = clamp((delta_dist_m / 2000.0) * 127, 0, 127).
    MotionDelta0 = 21,
    /// Delta de posição retina linha 1.
    MotionDelta1 = 22,
    /// Delta de posição retina linha 2.
    MotionDelta2 = 23,
    /// Delta de posição retina linha 3.
    MotionDelta3 = 24,
    /// Delta de posição retina linha 4.
    MotionDelta4 = 25,
    /// Delta de posição retina linha 5.
    MotionDelta5 = 26,
    /// Delta de posição retina linha 6.
    MotionDelta6 = 27,

    // ── Propriocepção temporal (temporal.wgsl ch 64-75 via buf_temporal[20-31]) ─
    // Mapeamento confirmado no código gpu_evolve.rs:586-637 (array `prop`, 12
    // elementos) cruzado com os write_channel(being, 64..=75, ...) de
    // perception_temporal.wgsl:324-349:
    //   prop[0]  → ch64 = align = dot(vel,fwd)/speed, clamp(-1,1)
    //   prop[1]  → ch65 = speed = |vel| em m/s (RAW, sem normalizar no host)
    //   prop[2]  → ch66 = lat_vel = dot(vel, lat)
    //   prop[3]  → ch67 = vida_frac = life_left / life_cap
    //   prop[4]  → ch68 = rpm_norm = current_rpm / max_rpm
    //   prop[5]  → ch69 = gear = current_gear as f32 (RAW; ×20 é no shader)
    //   prop[6]  → ch70 = average normalised tyre temperature
    //   prop[7]  → ch71 = temp_motor = (engine_temp - 20) / 130
    //   prop[8]  → ch72 = risco_cinetico = speed_norm × curv_aparente
    //   prop[9]  → ch73 = on_grass (voxel==0 → 1.0, mas o shader codifica -128)
    //   prop[10] → ch74 = |steer|
    //   prop[11] → ch75 = estagnado = stagnation_ticks / 64
    /// Alinhamento velocidade/direção frontal: dot(vel,fwd)/|vel|. [-127,127].
    /// +127 = alinhado, -127 = indo de ré. Não é velocidade fwd bruta.
    VelAlign = 64,
    /// Módulo da velocidade em m/s. Shader: clamp(speed_m_s * 2.0, 0, 127) —
    /// ou seja, satura em 63,5 m/s (não 30 m/s). [0, 127].
    SpeedFwd = 65,
    /// Velocidade lateral: dot(vel, lat). Shader: clamp(lat_vel * 10.0, -127, 127).
    VelLat = 66,
    /// Vida restante como fração do teto. [0, 127] = [0%, 100% de life_cap].
    VidaFrac = 67,
    /// RPM normalizado pelo max_rpm. [0, 127] = [0%, 100%].
    RpmNorm = 68,
    /// Marcha atual. Shader: clamp(current_gear * 20.0, 0, 127) — NÃO é o
    /// índice de marcha puro, é escalado ×20 para ocupar a faixa i8.
    Gear = 69,
    /// Normalised average tyre temperature: 0 = cold, 0.5 = optimum, 1 = hot. [0, 127].
    /// BODY BLOCK: with `BODY_CHANNELS_ON` the SOURCE becomes slot 6 of the block
    /// (CPU-side thermal model), same encoding — see `BODY_SLOTS`.
    TireTemp = 70,
    /// Temperatura do motor normalizada ((engine_temp - 20) / 130). [0, 127].
    /// BODY BLOCK: with `BODY_CHANNELS_ON` the SOURCE becomes slot 7 of the block,
    /// same formula — see `BODY_SLOTS`.
    TempMotor = 71,
    /// Demanda lateral normalizada: clamp(v² × |κ30m| / (0.9g), 0, 1). [0, 127].
    RiscoCinetico = 72,
    /// Flag de grama. Shader: `select(0, -128, voxel==0)` — codificação é
    /// {0, -128}, NÃO {0, 127} como as outras flags do registro.
    OnGrass = 73,
    /// |steer| atual clamped. [0, 127].
    SteerAbs = 74,
    /// Estagnação: stagnation_ticks / 64. [0, 127].
    Estagnacao = 75,

    // ── Distância lateral esq/dir (perception.wgsl ch 76-77) ─────────────────
    /// Distância à parede esquerda. GPU: norm por 40m. CPU (motor.rs, legado):
    /// norm por 15m — DIVERGÊNCIA confirmada (MATH_INVENTORY F33).
    DistLateral0 = 76,
    /// Distância à parede direita. Mesma divergência GPU/CPU de `DistLateral0`.
    DistLateral1 = 77,

    // ── Urgência frontal (perception.wgsl ch 82) ──────────────────────────────
    /// Urgência frontal. GPU: 1 - dist/30. CPU (motor.rs, legado): speed/dist
    /// (TTC) — semântica diverge (MATH_INVENTORY F36).
    UrgenciaFrontal = 82,

    // ── Flow ótico hemisférico (motion.wgsl ch 83-84) ─────────────────────────
    // NÃO é "linha 6"/"linha 7": soma |delta| das rows 6 E 8 (RETINA_ROWS/2 ± 1),
    // colunas 0..63 (metade esquerda) para ch83 e 63..126 (metade direita) para
    // ch84. Confirmado em perception_motion.wgsl:130-168 (threads 8 e 9).
    /// Fluxo óptico hemisférico ESQUERDO: soma |delta_dist| das rows 6 e 8,
    /// colunas 0..63, / HEMI_NORM(30.0) * 127. [0, 127].
    MotionMag0 = 83,
    /// Fluxo óptico hemisférico DIREITO: mesma fórmula de `MotionMag0`,
    /// colunas 63..126.
    MotionMag1 = 84,

    // ── Approach hemisférico + fóvea (motion.wgsl ch 114-117) ────────────────
    // FASE CANAIS (2026-08-14): antes só motor.rs (Bevy CPU) alimentava —
    // `gpu_evolve` não importa motor.rs, então ficavam sem fonte nesse binário.
    // Ligados em perception_motion.wgsl reusando retina_cur/retina_prev (mesmos
    // dados de MotionMag0/1, thread 8-10) — ver comentário "FASE CANAIS" lá.
    /// Approach hemisférico ESQUERDO, COM SINAL: positivo = parede se
    /// aproximando (distância caindo), rows 6+8, colunas 0..63. [-127, 127].
    /// BODY BLOCK: with `BODY_CHANNELS_ON` this channel changes MEANING —
    /// it becomes `StationBearing` (slot 0). See `BODY_SLOTS`.
    ApproachHemiL = 114,
    /// Approach hemisférico DIREITO, mesma fórmula de `ApproachHemiL`,
    /// colunas 63..126. [-127, 127].
    /// BODY BLOCK: becomes `StationProx` (slot 1) while the flag is on.
    ApproachHemiR = 115,
    /// Approach na fóvea (±15° centrais), COM SINAL: positivo = parede vindo de
    /// frente. Row 6, colunas 24..40. [-127, 127].
    /// BODY BLOCK: becomes `StationCharge` (slot 2) while the flag is on.
    FoveaApproach = 116,
    /// Movimento absoluto na fóvea (qualquer direção), mesma amostra de
    /// `FoveaApproach`. [0, 127].
    /// BODY BLOCK: becomes `OwnFuel` (slot 3) while the flag is on.
    FoveaMotionAbs = 117,

    // ── EMA distância lateral (temporal.wgsl ch 85-86) ───────────────────────
    /// EMA da distância lateral esquerda, invertida: encoded = (1 - ema/40)*127
    /// → 127 = parede encostando, 0 = livre. alpha = clamp(dt*3,0,1) (meia-vida
    /// ~21 ticks). CPU (motor.rs, legado) norm por 15m — DIVERGE (F34).
    EmaDistL = 85,
    /// EMA da distância lateral direita. Mesma fórmula de `EmaDistL`.
    EmaDistR = 86,

    // ── Slip angle injetado pela CPU (temporal.wgsl slot 49, ch 87) ──────────
    /// Ângulo de deslizamento = cross2D(fwd, vel). Injetado via update_extra2_temporal.
    SlipAngle = 87,

    // ── Sustain de proximidade e flag de ré (temporal.wgsl ch 88-89) ────────────
    // Confirmado em perception_temporal.wgsl:229-237 e 350-358.
    /// Tempo acumulado raspando na parede (ema_l<3m OU ema_r<3m), normalizado
    /// por 3 s. [0, 127].
    ProxSustain = 88,
    /// Flag de marcha-ré: 127 = em ré, 0 = marcha à frente.
    IsReversing = 89,

    // ── Motion delta retina row 7 (motion.wgsl ch 98) ─────────────────────────
    /// MESMA fórmula de `MotionDelta0..6`, aplicada à retina row 7 — NÃO é uma
    /// aceleração (derivada segunda); é o 8º canal da família motion-delta,
    /// escrito por um thread separado no mesmo kernel. [0, 127].
    /// BODY BLOCK: with `BODY_CHANNELS_ON` this channel changes MEANING — it
    /// becomes `PredatorBearing` (slot 4) and is written by temporal.wgsl, with
    /// motion.wgsl going silent. See `BODY_SLOTS`.
    MotionDelta7 = 98,

    // ── Edge density retina (temporal.wgsl slots 54-55, ch 99-100) ───────────
    /// Densidade de borda horizontal linha 6 da retina. Norm por 6000
    /// (confirmado em gpu_evolve.rs: soma de |Δpixel| adjacentes / 6000).
    EdgeRow6 = 99,
    /// Densidade de borda horizontal linha 7 da retina. Mesma fórmula de `EdgeRow6`.
    EdgeRow7 = 100,

    // ── Approach rate + TTC mínimo (temporal.wgsl ch 102-107) ────────────────
    // Confirmado em perception_temporal.wgsl:195-217 (comentário "ch 102-104:
    // approach rate lateral esq/dir e frontal" + "ch 106-107: delta-dist
    // lateral ±15m/s"). ar > 0 = parede se aproximando; ar < 0 = se afastando.
    /// Approach rate PAREDE ESQUERDA: (prev_dist_l - dist_l)/DT, clamp
    /// (ar/30*127, -127, 127). Com SINAL: negativo = afastando. [-127, 127].
    ApproachRate0 = 102,
    /// Approach rate PAREDE DIREITA. Mesma fórmula de `ApproachRate0`.
    ApproachRate1 = 103,
    /// Approach rate PAREDE FRONTAL. Mesma fórmula de `ApproachRate0`.
    ApproachRate2 = 104,
    /// TTC MÍNIMO entre as 3 direções (esq/dir/frontal): 127*(1 - ttc_min/2),
    /// onde ttc_min = min(dist/ar, 2.0) para cada direção com ar>1.0, senão 2.0.
    /// 127 = colisão iminente, 0 = seguro (≥2s). NÃO é "frontal" — é o mínimo. [0, 127].
    TtcMin = 105,
    /// Approach rate PAREDE ESQUERDA, segunda codificação (ganho 64/15 ≈ 4,27×,
    /// satura em ≈±29,7 m/s em vez de ±30 m/s — resolução quase idêntica a
    /// `ApproachRate0`, não é time-to-collision). [-127, 127].
    ApproachRateFineL = 106,
    /// Approach rate PAREDE DIREITA, segunda codificação. Mesma fórmula de
    /// `ApproachRateFineL`.
    ApproachRateFineR = 107,
    /// Desvio normalizado da racing line [-1,1] → i8. (buf_temporal[10]).
    LineOffsetRacingLine = 108,
    /// Derivada do desvio da racing line, como TAXA (offset/segundo).
    /// FIX 2026-08-14 (FASE CANAIS): ANTES sempre 0.0 — a CPU nunca implementou
    /// o cálculo (slot [11] do buffer temporal ficava hardcoded). Agora computada
    /// dentro de perception_temporal.wgsl a partir do slot [8] (valor do tick
    /// anterior, que o próprio shader já salvava sem nunca ler de volta).
    /// MAX_OFFSET_RATE=8,0/s é estimativa não validada por medição. [-127, 127].
    LineOffsetDeriv = 109,
    /// Lookahead offset da racing line normalizado [-1,1]. (buf_temporal[12]).
    LineLookAheadRL = 110,
    // ── Curvatura a 260m (temporal.wgsl ch 101) ──────────────────────────────
    // FASE CANAIS (2026-08-14): antes só motor.rs alimentava (`sensory_wave[101]
    // = c260`); ligado em perception_temporal.wgsl reusando `curv_at_d`, a MESMA
    // função já validada para Curv350m/200m/275m abaixo — só muda a distância.
    /// Curvatura da pista a 260m à frente (via curv_at_d no temporal.wgsl). [-127, 127].
    Curv260m = 101,

    /// Curvatura da pista a 350m à frente (via curv_at_d no temporal.wgsl). [-127, 127].
    Curv350m = 111,
    /// Curvatura da pista a 200m à frente.
    Curv200m = 112,
    /// Curvatura da pista a 275m à frente.
    Curv275m = 113,

    // ── Racing line dist CPU (temporal.wgsl ch 118-119, 123) ─────────────────
    // Mapeamento confirmado em gpu_evolve.rs:552-561 (array `rv`) cruzado com
    // perception_temporal.wgsl:275-277:
    //   ch118 = cpu_fwd_norm  (buf_temporal[13]): dist_f / (hw*3)
    //   ch119 = cpu_corridor  (buf_temporal[14]): (dist_l+dist_r)/(hw*4)
    //   ch123 = cpu_whisker   (buf_temporal[15]): min(dist_l,dist_r)/(hw*1.5)
    /// Distância frontal normalizada: dist_f / (half_width * 3). [0, 127].
    FwdDistNorm = 118,
    /// Largura do corredor lateral normalizada: (dist_l+dist_r)/(half_width*4). [0, 127].
    CorridorNorm = 119,
    /// Estado memory0 do CfC de percepção temporal, atualizado pela CPU via
    /// update_memory0() após a física, lido e copiado à onda pelo shader.
    /// [-127, 127]. Escrito por temporal.wgsl:260 via `memory0 = temporal[tb+7]`.
    Memory0Cfc = 120,

    // ── Steer/brake mem (temporal.wgsl slots 56-57, ch 121-122) ──────────────
    /// Memória de volante do tick anterior (phys.steer). [-127, 127].
    SteerMem = 121,
    /// Memória de freio do tick anterior (phys.brake_cmd). [0, 127].
    BrakeMem = 122,

    // ── Whisker normalizado (temporal.wgsl ch 123) ───────────────────────────
    // Confirmado em perception_temporal.wgsl:277: cpu_whisker_norm (buf_temporal[15])
    /// Lado mais estreito: min(dist_l, dist_r) / (half_width * 1.5). [0, 127].
    WhiskerNorm = 123,

    // ── Sinais injetados pela CPU (temporal.wgsl slots 49-53, ch 124-127) ────
    // Mapeamento confirmado em gpu_evolve.rs:863-869 (array `extra2`) cruzado
    // com perception_temporal.wgsl:391-407 e o cabeçalho de slots ([49]-[53]):
    //   extra2[3]=is_spinning  → slot 52 → ch124
    //   extra2[4]=comp_stress  → slot 53 → ch125
    //   extra2[1]=joy_tick     → slot 50 → ch126
    //   extra2[2]=joy_streak/5 → slot 51 → ch127
    // A primeira versão deste registro tinha os quatro TROCADOS (JoyTick=124,
    // JoyStreak=125, Spinning=126, CompStress=127) — corrigido aqui.
    /// Flag de rodopio: |av_y|>1.5 rad/s E speed<2.0 m/s. Shader:
    /// `select(0, -128, is_spinning>0.5)` — codificação {0, -128}, NÃO {0,127}.
    /// BODY BLOCK: with `BODY_CHANNELS_ON` it changes MEANING AND RANGE — it
    /// becomes `PredatorProx` (slot 5), [0,127] instead of {0,-128}. See `BODY_SLOTS`.
    Spinning = 124,
    /// Estresse de proximidade (congestionamento): 0.0 = líder/ninguém perto,
    /// 0.5 = no meio do pelotão sem vizinho perto, 1.0 = alguém <20m à frente
    /// (ângulo <~72° do fwd). NÃO é "velocidade × curvatura" — é distância a
    /// outros seres (gpu_evolve.rs:844-861). [0, 127].
    CompStress = 125,
    /// Joy tick: progress_speed × surface_factor, faixa [-1,1] no host →
    /// [-127,127] no canal (COM sinal — recuar reduz joy).
    JoyTick = 126,
    /// Joy streak: segundos de flow contínuo / 5.0, [0,1] no host → [0,127].
    JoyStreak = 127,
}

impl Channel {
    /// Índice numérico do canal na onda [0, 191].
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Nome legível do canal (identificador simbólico).
    pub const fn name(self) -> &'static str {
        match self {
            Channel::FuelFrac => "FuelFrac",
            Channel::MotionDelta0 => "MotionDelta0",
            Channel::MotionDelta1 => "MotionDelta1",
            Channel::MotionDelta2 => "MotionDelta2",
            Channel::MotionDelta3 => "MotionDelta3",
            Channel::MotionDelta4 => "MotionDelta4",
            Channel::MotionDelta5 => "MotionDelta5",
            Channel::MotionDelta6 => "MotionDelta6",
            Channel::VelAlign => "VelAlign",
            Channel::SpeedFwd => "SpeedFwd",
            Channel::VelLat => "VelLat",
            Channel::VidaFrac => "VidaFrac",
            Channel::RpmNorm => "RpmNorm",
            Channel::Gear => "Gear",
            Channel::TireTemp => "TireTemp",
            Channel::TempMotor => "TempMotor",
            Channel::RiscoCinetico => "RiscoCinetico",
            Channel::OnGrass => "OnGrass",
            Channel::SteerAbs => "SteerAbs",
            Channel::Estagnacao => "Estagnacao",
            Channel::DistLateral0 => "DistLateral0",
            Channel::DistLateral1 => "DistLateral1",
            Channel::UrgenciaFrontal => "UrgenciaFrontal",
            Channel::MotionMag0 => "MotionMag0",
            Channel::MotionMag1 => "MotionMag1",
            Channel::ApproachHemiL => "ApproachHemiL",
            Channel::ApproachHemiR => "ApproachHemiR",
            Channel::FoveaApproach => "FoveaApproach",
            Channel::FoveaMotionAbs => "FoveaMotionAbs",
            Channel::EmaDistL => "EmaDistL",
            Channel::EmaDistR => "EmaDistR",
            Channel::SlipAngle => "SlipAngle",
            Channel::ProxSustain => "ProxSustain",
            Channel::IsReversing => "IsReversing",
            Channel::MotionDelta7 => "MotionDelta7",
            Channel::EdgeRow6 => "EdgeRow6",
            Channel::EdgeRow7 => "EdgeRow7",
            Channel::ApproachRate0 => "ApproachRate0",
            Channel::ApproachRate1 => "ApproachRate1",
            Channel::ApproachRate2 => "ApproachRate2",
            Channel::TtcMin => "TtcMin",
            Channel::ApproachRateFineL => "ApproachRateFineL",
            Channel::ApproachRateFineR => "ApproachRateFineR",
            Channel::LineOffsetRacingLine => "LineOffsetRacingLine",
            Channel::LineOffsetDeriv => "LineOffsetDeriv",
            Channel::LineLookAheadRL => "LineLookAheadRL",
            Channel::Curv260m => "Curv260m",
            Channel::Curv350m => "Curv350m",
            Channel::Curv200m => "Curv200m",
            Channel::Curv275m => "Curv275m",
            Channel::FwdDistNorm => "FwdDistNorm",
            Channel::CorridorNorm => "CorridorNorm",
            Channel::Memory0Cfc => "Memory0Cfc",
            Channel::SteerMem => "SteerMem",
            Channel::BrakeMem => "BrakeMem",
            Channel::WhiskerNorm => "WhiskerNorm",
            Channel::Spinning => "Spinning",
            Channel::CompStress => "CompStress",
            Channel::JoyTick => "JoyTick",
            Channel::JoyStreak => "JoyStreak",
        }
    }

    /// Faixa esperada em i8: (min, max).
    pub const fn range_i8(self) -> (i8, i8) {
        match self {
            // Motion e flow: sempre não-negativos (magnitude de |delta|)
            Channel::MotionDelta0
            | Channel::MotionDelta1
            | Channel::MotionDelta2
            | Channel::MotionDelta3
            | Channel::MotionDelta4
            | Channel::MotionDelta5
            | Channel::MotionDelta6
            | Channel::MotionMag0
            | Channel::MotionMag1
            | Channel::MotionDelta7 => (0, 127),

            // Approach hemisférico/fóvea: com sinal (positivo = aproximando);
            // FoveaMotionAbs é magnitude, não-negativa.
            Channel::ApproachHemiL | Channel::ApproachHemiR | Channel::FoveaApproach => (-127, 127),
            Channel::FoveaMotionAbs => (0, 127),

            // Alinhamento e velocidade lateral: com sinal
            Channel::VelAlign | Channel::VelLat => (-127, 127),

            // Velocidade fwd (módulo): não-negativo
            Channel::SpeedFwd | Channel::FuelFrac => (0, 127),

            // Propriocepção não-negativa
            Channel::VidaFrac
            | Channel::RpmNorm
            | Channel::Gear
            | Channel::TireTemp
            | Channel::TempMotor
            | Channel::RiscoCinetico
            | Channel::SteerAbs
            | Channel::Estagnacao => (0, 127),

            // Flags com codificação NEGATIVA (select(0, -128, cond)) — não são
            // {0,127} como as outras flags do registro.
            Channel::OnGrass | Channel::Spinning => (-128, 0),

            // Distância lateral e urgência
            Channel::DistLateral0
            | Channel::DistLateral1
            | Channel::UrgenciaFrontal
            | Channel::EmaDistL
            | Channel::EmaDistR => (0, 127),

            // Slip angle: com sinal
            Channel::SlipAngle => (-127, 127),

            // Prox sustain e flag de ré: não-negativos
            Channel::ProxSustain | Channel::IsReversing => (0, 127),

            // Edge density: não-negativa
            Channel::EdgeRow6 | Channel::EdgeRow7 => (0, 127),

            // Approach rate (todas as codificações, incl. as "Fine"): COM SINAL
            // — ar>0 aproximando, ar<0 afastando. Só o TTC mínimo é não-negativo.
            Channel::ApproachRate0
            | Channel::ApproachRate1
            | Channel::ApproachRate2
            | Channel::ApproachRateFineL
            | Channel::ApproachRateFineR => (-127, 127),

            Channel::TtcMin => (0, 127),

            // Racing line offsets: com sinal
            Channel::LineOffsetRacingLine | Channel::LineOffsetDeriv | Channel::LineLookAheadRL => {
                (-127, 127)
            }

            // Racing line dist norms: não-negativos
            Channel::FwdDistNorm | Channel::CorridorNorm => (0, 127),

            // Memory CfC de percepção: com sinal
            Channel::Memory0Cfc => (-127, 127),

            // Curvatura da pista a múltiplas distâncias: com sinal
            Channel::Curv260m | Channel::Curv350m | Channel::Curv200m | Channel::Curv275m => {
                (-127, 127)
            }

            // Steer/brake mem: steer tem sinal, brake não-negativo
            Channel::SteerMem => (-127, 127),
            Channel::BrakeMem => (0, 127),

            // Whisker norm: não-negativo
            Channel::WhiskerNorm => (0, 127),

            // Comp stress e joy streak: não-negativos
            Channel::CompStress | Channel::JoyStreak => (0, 127),

            // Joy tick: com sinal (recuar reduz)
            Channel::JoyTick => (-127, 127),
        }
    }

    /// Classificação Kernel/Host/Morto de quem ESCREVE o canal na onda final.
    ///
    /// Derivada mecanicamente do texto de `source_for()` (mesma fonte de verdade
    /// usada pela auditoria) — não é uma segunda tabela mantida à mão, para não
    /// divergir dela. `Kernel(nome)` = a última escrita no canal acontece dentro
    /// de um shader WGSL. `Host` = a última escrita é feita pela CPU (motor.rs,
    /// ou injeção via `update_extra_temporal`/`update_extra2_temporal`/`update_memory0`
    /// em `gpu_evolve.rs`, documentada no comentário de `source_for()`).
    ///
    /// CUIDADO: isto responde "quem grava o byte na onda", não "quem calculou o
    /// valor bruto". Vários canais `Kernel` (ex.: `Memory0Cfc`, `EdgeRow6`/`EdgeRow7`)
    /// têm o valor bruto calculado na CPU e só copiado para a onda pelo shader —
    /// ver o comentário específico de cada slot em `source_for()`. Nenhuma variante
    /// nomeada é `Morto` hoje: os 192 índices têm fonte documentada (confirmado
    /// pelo teste `source_for_cobre_todos_os_canais_conhecidos`).
    pub fn source(self) -> ChannelSource {
        let s = source_for(self.index());
        if s == "sem_fonte" {
            ChannelSource::Morto("nenhum kernel nem código CPU documenta escrita neste canal")
        } else if s.contains("injetado pela CPU") || s.contains("motor.rs") || s.contains("física")
        {
            ChannelSource::Host
        } else {
            // Nome do arquivo .wgsl vem antes do primeiro espaço/parêntese, ex.:
            // "temporal.wgsl (slots 54-55: ...)" → "temporal.wgsl".
            let kernel = s.split(|c: char| c == ' ' || c == '(').next().unwrap_or(s);
            ChannelSource::Kernel(kernel)
        }
    }

    /// Fórmula curta (semântica de codificação) do canal, para consulta rápida
    /// sem precisar abrir o .wgsl. Verificada linha a linha contra os
    /// `write_channel(being, N, ...)` reais — ver o comentário "Como isto foi
    /// verificado" no topo do arquivo.
    pub const fn formula_doc(self) -> &'static str {
        match self {
            Channel::FuelFrac => "fuel / tank capacity, [0,127] = [empty,full]",
            Channel::MotionDelta0
            | Channel::MotionDelta1
            | Channel::MotionDelta2
            | Channel::MotionDelta3
            | Channel::MotionDelta4
            | Channel::MotionDelta5
            | Channel::MotionDelta6 => {
                "clamp((delta_dist_m / 2000.0) * 127, 0, 127) — delta de posição da linha da retina entre ticks"
            }
            Channel::VelAlign => {
                "dot(vel, fwd) / speed, clamp(-1,1)*127: +127 alinhado, -127 de ré"
            }
            Channel::SpeedFwd => "clamp(speed_m_s * 2.0, 0, 127) — satura em 63,5 m/s",
            Channel::VelLat => "clamp(dot(vel, lat) * 10.0, -127, 127)",
            Channel::VidaFrac => "life_left / life_cap, [0,127] = [0%,100%]",
            Channel::RpmNorm => "current_rpm / max_rpm, [0,127] = [0%,100%]",
            Channel::Gear => "clamp(current_gear * 20.0, 0, 127) — NÃO é cast puro",
            Channel::TireTemp => "avg(wheels[*].temp) / 120°C, [0,127]",
            Channel::TempMotor => "(engine_temp - 20) / 130, [0,127]",
            Channel::RiscoCinetico => "clamp(v² × |κ média nos próximos 30m| / (0.9g), 0, 1)",
            Channel::OnGrass => "select(0, -128, voxel == 0) — {0,-128}, NÃO {0,127}",
            Channel::SteerAbs => "|steer| atual, clamped [0,127]",
            Channel::Estagnacao => "stagnation_ticks / 64, [0,127]",
            Channel::DistLateral0 | Channel::DistLateral1 => {
                "GPU: clamp(dist/40*127,0,127). CPU legado: clamp(dist/15*127,0,127) — DIVERGE (F33)"
            }
            Channel::UrgenciaFrontal => {
                "GPU: clamp((1 - fwd_probe/30)*127,0,127). CPU legado: speed/fwd_probe (TTC) — DIVERGE (F36)"
            }
            Channel::MotionMag0 => {
                "soma |delta| das rows 6+8, colunas 0..63 (metade ESQ) / HEMI_NORM(30) * 127"
            }
            Channel::MotionMag1 => {
                "soma |delta| das rows 6+8, colunas 63..126 (metade DIR) / HEMI_NORM(30) * 127"
            }
            Channel::ApproachHemiL => {
                "soma SIGNED (dist_prev-dist_cur) das rows 6+8, cols 0..63 / HEMI_NORM(30) * 127 — positivo=aproximando"
            }
            Channel::ApproachHemiR => "mesma fórmula de ApproachHemiL, colunas 63..126",
            Channel::FoveaApproach => {
                "soma SIGNED (dist_prev-dist_cur), row 6, cols 24..40 (~±15°) / FOVEA_NORM(4) * 127"
            }
            Channel::FoveaMotionAbs => {
                "soma |dist_prev-dist_cur|, mesma amostra de FoveaApproach / FOVEA_NORM(4) * 127"
            }
            Channel::EmaDistL | Channel::EmaDistR => {
                "clamp((1 - ema/40)*127, 0, 127); ema=ema*(1-α)+dist*α, α=clamp(dt*3,0,1) — invertido (perto=alto)"
            }
            Channel::SlipAngle => "cross2D(fwd, vel) — ângulo de deslizamento, injetado pela CPU",
            Channel::ProxSustain => {
                "acumula DT enquanto ema_l<3m ou ema_r<3m, satura em 3s, /3*127"
            }
            Channel::IsReversing => "flag de marcha-ré: 127 = em ré, 0 = à frente",
            Channel::MotionDelta7 => {
                "MESMA fórmula de MotionDelta0..6 (delta_dist/2000*127), aplicada à row 7"
            }
            Channel::EdgeRow6 => "densidade de borda horizontal linha 6 da retina, norm por 6000",
            Channel::EdgeRow7 => "densidade de borda horizontal linha 7 da retina, norm por 6000",
            Channel::ApproachRate0 => {
                "(prev_dist_l - dist_l)/DT, clamp(ar/30*127,-127,127) — PAREDE ESQUERDA"
            }
            Channel::ApproachRate1 => {
                "(prev_dist_r - dist_r)/DT, clamp(ar/30*127,-127,127) — PAREDE DIREITA"
            }
            Channel::ApproachRate2 => {
                "(prev_dist_f - dist_f)/DT, clamp(ar/30*127,-127,127) — PAREDE FRONTAL"
            }
            Channel::TtcMin => {
                "127*(1 - min(ttc_l,ttc_r,ttc_f)/2) — MÍNIMO das 3 direções, não só frontal"
            }
            Channel::ApproachRateFineL => {
                "clamp(ar_l/15*64,-127,127) — 2ª codificação de ApproachRate0, ganho maior"
            }
            Channel::ApproachRateFineR => {
                "clamp(ar_r/15*64,-127,127) — 2ª codificação de ApproachRate1, ganho maior"
            }
            Channel::LineOffsetRacingLine => "desvio normalizado da racing line [-1,1] → i8",
            Channel::LineOffsetDeriv => {
                "(cur-prev_offset)/DT / MAX_OFFSET_RATE(8) * 127 — FIX 2026-08-14, era hardcoded 0.0 (ver doc da variante)"
            }
            Channel::LineLookAheadRL => "lookahead offset da racing line normalizado [-1,1]",
            Channel::Curv260m => {
                "variação da tangente da pista entre agora e +260m de arco, dividida por 90°, [-127,127]"
            }
            Channel::Curv350m => {
                "variação da tangente da pista até +350m de arco / 90°, [-127,127]"
            }
            Channel::Curv200m => {
                "variação da tangente da pista até +200m de arco / 90°, [-127,127]"
            }
            Channel::Curv275m => {
                "variação da tangente da pista até +275m de arco / 90°, [-127,127]"
            }
            Channel::FwdDistNorm => "dist_f / (half_width * 3), [0,127]",
            Channel::CorridorNorm => "(dist_l + dist_r) / (half_width * 4), [0,127]",
            Channel::Memory0Cfc => {
                "memory0 = temporal[tb+7] — estado do CfC de percepção, atualizado pela CPU"
            }
            Channel::SteerMem => "phys.steer do tick anterior, [-127,127]",
            Channel::BrakeMem => "phys.brake_cmd do tick anterior, [0,127]",
            Channel::WhiskerNorm => "min(dist_l, dist_r) / (half_width * 1.5), [0,127]",
            Channel::Spinning => "select(0, -128, |av_y|>1.5 && speed<2.0) — {0,-128}, NÃO {0,127}",
            Channel::CompStress => {
                "0.0=líder/ninguém, 0.5=pelotão, 1.0=alguém <20m à frente — NÃO é veloc.×curvatura"
            }
            Channel::JoyTick => "progress_speed * surface_factor, [-1,1]*127 — COM sinal",
            Channel::JoyStreak => "joy_streak_segundos / 5.0, [0,1]*127",
        }
    }
}

/// Categoria de quem escreve o canal na onda final — ver `Channel::source()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelSource {
    /// Escrito dentro de um shader WGSL — carrega o nome do arquivo.
    Kernel(&'static str),
    /// Escrito pela CPU (motor.rs ou injeção via update_*_temporal em gpu_evolve.rs).
    Host,
    /// Sem fonte documentada — nenhum kernel nem código CPU escreve este canal.
    Morto(&'static str),
}

/// Canais cujo peso de entrada L1 nasce ZERADO no genoma — DESLIGAR da FASE CANAIS
/// (2026-08-14), consumido por `Genome::random()` em `gpu/evolution.rs`.
///
/// Diferente de `ChannelSource::Morto` (nenhum kernel escreve o canal): estes canais
/// TÊM um kernel que escreve — mas o valor escrito é estruturalmente constante
/// (não depende de trajetória, pista ou população; ver justificativa por canal
/// abaixo), então o peso de entrada correspondente em L1 é lixo previsível: o
/// cérebro nasceria gastando atenção com estática. Ver o comentário "FASE CANAIS"
/// no topo deste arquivo para o raciocínio completo (por que isto NÃO é poda de
/// neurônio, e por que só `TireTemp` está aqui e não os outros 20 canais "escritos
/// mas constantes" da auditoria).
///
/// Cada entrada é `(índice_do_canal, motivo)`. Reversível: remover a entrada quando
/// o motivo deixar de valer (ex.: P2 implementar a térmica do pneu).
/// REFUTADO POR MEDIÇÃO (2026-08-15, bateria de 139 validações): ch70 (TireTemp)
/// está VIVO nos dois campeões observados (valores 1-79, std ~7, trajetória
/// estruturada) — o veredito "passthrough" da auditoria leu physics_c.wgsl, mas o
/// pipeline real do gpu_evolve termina em physics_power_c.wgsl, onde a térmica É
/// implementada (linha ~271: temp += calor_derrapagem + calor_freio − resfriamento)
/// e AFETA o grip (linha ~230). Zerar os pesos deste canal cegava o ser para um
/// sinal que modula a própria aderência. Lista vazia mantida (a máquina
/// zero_dead_inputs continua no lugar para o próximo canal genuinamente morto).
pub const DEAD_L1_INPUTS: &[(usize, &str)] = &[];

/// Canonical channel families: `(name, start_inclusive, end_exclusive)`.
///
/// One export instead of two copies. The same table lived only in
/// `tools/destilabilidade.py:121-138` and the D8 specification asks for it to become
/// an export of this module (`docs/SPECS_MUNDO_HOSTIL.md`, D8 "Mechanism"): a second
/// copy of a channel range is exactly the failure `src/CLAUDE.md` rule 1 forbids, and
/// D8 charges a per-family subscription, so a family boundary that drifts between the
/// analysis tool and the pressure would tax a family the report does not name.
///
/// The ranges tile `0..N_CHANNELS` with no gap and no overlap — `families_tile_the_wave`
/// fails the build otherwise. Two consequences worth stating: every channel pays under
/// exactly one family, and `sum(|F|) == N_CHANNELS`, so the D8 weight `|F| / 192` sums
/// to 1 and the cost of a being at reference use is exactly `DOSE` life/tick.
pub const CHANNEL_FAMILIES: &[(&str, usize, usize)] = &[
    ("retina", 0, 16),
    ("wall_dist", 16, 32),
    ("wall_curv", 32, 40),
    ("road_near_far", 40, 56),
    ("material_escape", 56, 64),
    ("body", 64, 76),
    ("proximity", 76, 96),
    ("derived", 96, 120),
    ("memory", 120, 123),
    ("whisker_spin_stress", 123, 126),
    ("dopamine", 126, 128),
    ("fovea", 128, 144),
    ("peripheral", 144, 160),
    ("horizon", 160, 192),
];

/// Family index of channel `ch`, or `None` when `ch >= N_CHANNELS`.
pub fn family_of(ch: usize) -> Option<usize> {
    CHANNEL_FAMILIES
        .iter()
        .position(|&(_, lo, hi)| ch >= lo && ch < hi)
}

// ─── BODY BLOCK (BODY PHASE, 2026-08-17) — hostile-world + interoception ────────
//
// CONTRACT FOR THE INTEGRATOR (owner of context.rs/main.rs):
//
// A single mechanism carries every new CPU-born signal of the hostile world and of
// the being's own body to the sensory wave: the CPU computes `BODY_K` f32 values
// per being per tick and calls `PerceptionGpu::update_body_channels()`
// (src/gpu/perception.rs), which writes them into `buf_temporal` slots
// [`gpu::perception::BODY_SLOT_BASE` .. `BODY_SLOT_BASE + BODY_K`).
// `perception_temporal.wgsl` converts each slot to the i8 channel listed below
// ONLY when the flag `gpu::perception::BODY_CHANNELS_ON` is set (same pattern as
// `PERIF_REAL`, perception.rs:108 — `main.rs:990` is the store site for
// `PERIF_REAL`: `universe::gpu::perception::PERIF_REAL.store(true, Ordering::Relaxed)`).
//
// ONE MECHANISM, NOT ONE PER SPEC — REGISTERED DECISION (mission rule 7). The D7
// spec asked to widen `update_extra2_temporal` from `[f32; 9]` to `[f32; 13]`, and
// the predator spec asked to widen the SAME array to `[f32; 11]`
// (docs/SPECS_MUNDO_HOSTIL.md:132: "alargar [f32;9]→[f32;11]"). Two specs widening
// one array to two different lengths is a merge conflict by construction, and each
// new pressure would widen it again. Instead the body block is ONE generic array
// with a fixed, documented `BODY_K` and its own slot range, so a new body signal
// costs a row in `BODY_SLOTS` and nothing else. `update_extra2_temporal` stays at
// `[f32; 9]` and keeps its slots 49-57 — untouched, so nothing already measured moves.
//
// HOST NORMALISES, SHADER ONLY ENCODES. Every value in this block arrives from the
// CPU already normalised to the range stated in its row (`[-1,1]` or `[0,1]`); the
// shader does nothing but `clamp(v * 127.0)`. Consequence for the integrator: the
// reference distance of a proximity ramp is a HOST-side dose, not a constant frozen
// in a kernel — the D7 station ramp is fixed at 300 m, and the predator ramp is the
// `--predator-perception` dose (default 60 m, docs/SPECS_MUNDO_HOSTIL.md:114). Changing
// a dose must never require touching WGSL.
//
// OWNERSHIP HANDOFF, REGISTERED DECISION (mission rule 7): channels 98 and
// 114-117 are written by `perception_motion.wgsl` when the flag is OFF (the
// measured-constant approach/fovea family) and by `perception_temporal.wgsl`
// (from this block) when the flag is ON. Both kernels read the SAME uniform
// field `body_on`, built once per tick from the single static — two kernels can
// never disagree, so no double-write (atomicOr corruption) and no mute channel.
// Flag OFF is bit-for-bit the old behavior (project rule 3: new perception
// enters behind a default-off flag).
//
// Why these channels: ch98, ch114-117 and ch124 are the six channels MEASURED
// constant for a whole run (tools/destilabilidade.py degenerate list) — the
// natural space for new signals. ch70/ch71 keep their meaning; the flag only
// switches their SOURCE to the CPU-side real models (tire thermal / engine
// temp) carried in this block.
//
// Declared out and why (do not silently drop sibling requirements):
// - life_left interoception: ALREADY live as ch67 `VidaFrac` — context.rs:591-592
//   feeds it into prop[3]: `let vida_frac = (states[i].life_left as f32 /
//   life_cap().max(1) as f32).clamp(0.0, 1.0);`, and context.rs:521 pins the slot
//   with `assert!(Channel::VidaFrac.index() == 67, "prop[3] = vida_frac")`.
//   Not duplicated.
// - engine_temp: ALSO already live, as ch71 — context.rs:593 `let temp_motor =
//   ((p.engine_temp - 20.0) / 130.0).clamp(0.0, 1.0);` into prop[7], pinned by
//   context.rs:525 `assert!(Channel::TempMotor.index() == 71, "prop[7] = temp_motor")`.
//   The mission brief listed engine_temp and life_left as orphan states that never
//   reach the brain; the code says otherwise, so slot 7 below is a SOURCE move, not
//   a new signal (see its row). The two states that are genuinely orphan are `fuel`
//   (slot 3 closes it) and `oil_temp`.
// - damage ("dano") as a RATE: a per-tick damage rate is not derivable from ch67,
//   which is a level and the brain has no previous-tick copy of it. It is left out
//   for lack of a channel, not for lack of value — the six MEASURED-dead channels
//   are exactly ch98, ch114-117 and ch124, and all six are allocated below.
// - oil_temp: same reason (no dead channel left).
// Next candidates for either would have to come from ch144-159, which the D8 spec
// reserves as its pruning positive control — do not take them without registering it.

/// Number of f32 body values injected per being per tick. Fixed and part of the
/// CPU↔GPU contract — must match `[f32; BODY_K]` in
/// `PerceptionGpu::update_body_channels` and the slot reads in
/// `perception_temporal.wgsl`.
pub const BODY_K: usize = 8;

/// Canonical map of the body block: `(slot, channel, name_when_on, doc)`.
///
/// `slot` is the index inside the `[f32; BODY_K]` array handed to
/// `update_body_channels` (and the offset from `BODY_SLOT_BASE` in
/// `buf_temporal`). `channel` is the sensory-wave index written when
/// `BODY_CHANNELS_ON` is set. Every doc states the HOST range and the i8
/// encoding, like the enum variants above do.
///
/// Bearings use the same signed encoding as the wall probes: host value in
/// [-1, 1], shader does `i32(clamp(v * 127.0, -127.0, 127.0))`.
///
/// SIGN, STATED AS A BASIS AND NOT AS A WORD (integrator correction, BODY PHASE):
/// the host computes `atan2(rel · lat, rel · fwd) / PI`, where `lat` is the body
/// basis vector the physics state carries — `lat = [fwd.z, 0, -fwd.x]`
/// (`gpu/physics.rs:105`). Positive therefore means "toward `+lat`". The row docs
/// used to say "negative = LEFT", which is a claim about the world's handedness that
/// no `file:line` in this repository states; the basis is checkable and the word is
/// not. `bearing_sign_follows_the_lat_basis` in `gpu_evolve/context.rs` pins it, so
/// the sign cannot flip silently — and a flipped bearing is invisible to every
/// aggregate statistic, which is why it gets a test instead of a comment.
/// Proximities are a host-side ramp `1 - min(d / R, 1)` in [0, 1] (1 = on top of
/// it, 0 = at or beyond R), shader `i32(clamp(v * 127.0, 0.0, 127.0))`. R is the
/// integrator's dose and lives on the host, never in WGSL — see the block header.
///
/// The 0 that a proximity emits when there is no target is INDISTINGUISHABLE from
/// "target exactly at range R". That is deliberate and matches the wall probes
/// (`EmaDistL`: far = 0), and it is why every proximity here is paired with a
/// bearing: a live target at R has a bearing, an absent one is pinned to 0 too.
/// Both being 0 is the "nothing there" code.
pub const BODY_SLOTS: &[(usize, usize, &str, &str)] = &[
    (
        0,
        114,
        "StationBearing",
        "signed bearing to the nearest CHARGED energy station (D7), [-1,1] = [-pi,pi]/pi -> [-127,127]; 0 when no station qualifies",
    ),
    (
        1,
        115,
        "StationProx",
        "proximity to that station: host ramp 1 - min(d/R, 1) with R = 300 m (D7 spec), [0,1] -> [0,127]; 0 when none",
    ),
    (
        2,
        116,
        "StationCharge",
        "charge fraction of that station, charge/C in [0,1] -> [0,127]",
    ),
    (
        3,
        117,
        "OwnFuel",
        "own fuel / tank capacity, [0,1] -> [0,127] — interoception, closes the orphan `fuel` state (physics computes it, brain never felt it)",
    ),
    (
        4,
        98,
        "PredatorBearing",
        "signed bearing to the nearest predator, [-1,1] -> [-127,127]; 0 when no predator active",
    ),
    (
        5,
        124,
        "PredatorProx",
        "proximity to that predator: host ramp 1 - min(d/R, 1) with R = --predator-perception (default 60 m), [0,1] -> [0,127]; replaces the {0,-128} spinning flag while the flag is on",
    ),
    (
        6,
        70,
        "TireTempReal",
        "REAL tire temperature from the CPU-side thermal model, avg(temp)/120C in [0,1] -> [0,127]; overrides prop[6] while on — SOURCE move, ch70 is already alive (139-validation battery, see DEAD_L1_INPUTS above)",
    ),
    (
        7,
        71,
        "EngineTempReal",
        "engine temperature (engine_temp - 20)/130 in [0,1] -> [0,127] — SOURCE move, NOT a new signal: context.rs:593 already feeds this exact formula through prop[7]. Pass the same value and ch71 is bit-identical with the flag on; the slot exists so every body signal has ONE assembly point (M4/neuromodulation needs the body state in one place)",
    ),
];

/// Body-block slot that feeds sensory channel `ch`, or `None` if `ch` is not part
/// of the block. Linear scan over `BODY_K` = 8 rows: this is registry lookup, not
/// a hot path.
///
/// Use it instead of writing the slot number down a second time — a second copy of
/// a channel index is exactly the failure `src/CLAUDE.md` rule 1 forbids.
pub fn body_slot_for_channel(ch: usize) -> Option<usize> {
    BODY_SLOTS
        .iter()
        .find(|&&(_, c, _, _)| c == ch)
        .map(|&(slot, _, _, _)| slot)
}

/// Sensory channel written by body-block slot `slot`, or `None` if out of range.
pub fn body_channel_for_slot(slot: usize) -> Option<usize> {
    BODY_SLOTS
        .iter()
        .find(|&&(s, _, _, _)| s == slot)
        .map(|&(_, ch, _, _)| ch)
}

/// Name of `ch` while `BODY_CHANNELS_ON` is set — the block's name when the channel
/// belongs to it, and the ordinary `name_for(ch)` otherwise.
///
/// Exists so a tool (audit, `--audit-vision`, the distillability report) prints the
/// meaning the channel actually carries in the run being inspected instead of the
/// flag-off name. Reporting `Spinning` for a channel that is carrying predator
/// proximity is the "identical numbers where they should differ" alarm in reverse:
/// different meaning under the same label.
pub fn name_for_with_body(ch: usize, body_on: bool) -> &'static str {
    if body_on {
        if let Some(&(_, _, name, _)) = BODY_SLOTS.iter().find(|&&(_, c, _, _)| c == ch) {
            return name;
        }
    }
    name_for(ch)
}

/// `source_for(ch)`, corrected for the body block when `body_on`.
///
/// With the flag ON the ownership of ch98 and ch114-117 moves from
/// `perception_motion.wgsl` to `perception_temporal.wgsl`, and ch70/71/124 change
/// which host array feeds them. An audit that keeps calling plain `source_for()`
/// during a body-block run reports the wrong kernel — it would blame motion.wgsl
/// for a channel motion.wgsl is deliberately not writing.
pub fn source_for_with_body(ch: usize, body_on: bool) -> &'static str {
    if body_on && body_slot_for_channel(ch).is_some() {
        return "temporal.wgsl (BODY BLOCK, injected by the CPU: update_body_channels)";
    }
    source_for(ch)
}

/// Retorna o kernel/arquivo que escreve o canal `ch` na onda sensorial.
///
/// Esta função substitui e formaliza a lógica de `canal_tem_fonte()` que
/// estava inline em `gpu_evolve.rs`. A fonte é o arquivo responsável pela
/// última escrita (GPU sobrescreve CPU quando ambos escrevem no mesmo canal).
///
/// Canais fora do range [0, N_CHANNELS) retornam "sem_fonte".
pub fn source_for(ch: usize) -> &'static str {
    // IMPORTANTE: ranges do mais específico para o mais geral.
    // Vários canais têm fontes múltiplas (GPU + motor.rs sobrescrevem);
    // listamos a GPU (última escrita) quando há sobreposição.
    match ch {
        20 => "temporal.wgsl (BODY slot 3, injected by the CPU: fuel/tank)",
        // motion.wgsl escreve ch 21-27, 83-84, 98
        21..=27 => "motion.wgsl",
        83 | 84 => "motion.wgsl",
        98 => "motion.wgsl",
        // temporal.wgsl: ch 64-75, 81, 85-86, 88-89, 101-113, 118-120, 123
        64..=75 => "temporal.wgsl",
        81 => "temporal.wgsl",
        85 | 86 => "temporal.wgsl",
        88 | 89 => "temporal.wgsl",
        // ch101: track-tangent change at +260m of arc length. The host computes it
        // from the racing line and temporal.wgsl owns the final channel write.
        // NOTA: a string abaixo evita a substring "motor.rs" de propósito — ela
        // dispara o ramo Host em Channel::source() (ver match logo abaixo desta
        // função), e este canal É Kernel agora (temporal.wgsl escreve por último).
        101 => "temporal.wgsl (host arc-length racing-line lookahead, +260m)",
        102..=113 => "temporal.wgsl",
        78..=80 => "temporal.wgsl (host arc-length racing-line lookahead, heritable horizons)",
        118..=120 => "temporal.wgsl",
        123 => "temporal.wgsl",
        // temporal.wgsl (injeção CPU — gravados pelo host via update_*_temporal):
        //   ch87  = slip_angle  (slot 49)
        //   ch99  = edge_row6   (slot 54)
        //   ch100 = edge_row7   (slot 55)
        //   ch121 = steer_mem   (slot 56)
        //   ch122 = brake_mem   (slot 57)
        //   ch124 = is_spinning (slot 52)
        //   ch125 = comp_stress (slot 53)
        //   ch126 = joy_tick    (slot 50)
        //   ch127 = joy_streak  (slot 51)
        87 => "temporal.wgsl (slot 49, injetado pela CPU: slip_angle = cross2D fwd×vel)",
        99 | 100 => {
            "temporal.wgsl (slots 54-55, injetado pela CPU: edge density linhas 6/7 da retina)"
        }
        121 | 122 => "temporal.wgsl (slots 56-57, injetado pela CPU: steer/brake_mem do phys)",
        124..=127 => "temporal.wgsl (slots 50-53, injetado pela CPU: joy/spinning/stress)",
        // motor.rs (CPU, caminho Bevy interativo) escreve: ch 28, 32. The native
        // training path owns 78-80 in temporal.wgsl.
        28 | 32 => "motor.rs",
        // ch114-117: approach hemisférico + fóvea via motion.wgsl — FASE CANAIS
        // 2026-08-14 (antes só o caminho Bevy interativo alimentava; ainda
        // alimenta lá também, sem conflito — binários distintos).
        // NOTA: evita a substring "motor.rs" de propósito (ver nota em ch101 acima).
        114..=117 => {
            "motion.wgsl (approach hemisférico + fóvea — FASE CANAIS 2026-08-14, era só caminho Bevy)"
        }
        // perception.wgsl: ch 0-19, 29-31, 33-63, 76-77, 82, 90-97, 128-191
        0..=19 => "perception.wgsl",
        29..=31 => "perception.wgsl",
        33..=63 => "perception.wgsl",
        76 | 77 => "perception.wgsl",
        82 => "perception.wgsl",
        90..=97 => "perception.wgsl",
        128..=191 => "perception.wgsl",
        // sem fonte documentada (canais nunca escritos por nenhum kernel)
        _ => "sem_fonte",
    }
}

/// Nome simbólico do canal `ch`, ou `"ch{N}"` para canais anônimos da retina.
pub fn name_for(ch: usize) -> &'static str {
    // Canais nomeados (variantes do enum):
    match ch {
        20 => Channel::FuelFrac.name(),
        21 => Channel::MotionDelta0.name(),
        22 => Channel::MotionDelta1.name(),
        23 => Channel::MotionDelta2.name(),
        24 => Channel::MotionDelta3.name(),
        25 => Channel::MotionDelta4.name(),
        26 => Channel::MotionDelta5.name(),
        27 => Channel::MotionDelta6.name(),
        64 => Channel::VelAlign.name(),
        65 => Channel::SpeedFwd.name(),
        66 => Channel::VelLat.name(),
        67 => Channel::VidaFrac.name(),
        68 => Channel::RpmNorm.name(),
        69 => Channel::Gear.name(),
        70 => Channel::TireTemp.name(),
        71 => Channel::TempMotor.name(),
        72 => Channel::RiscoCinetico.name(),
        73 => Channel::OnGrass.name(),
        74 => Channel::SteerAbs.name(),
        75 => Channel::Estagnacao.name(),
        76 => Channel::DistLateral0.name(),
        77 => Channel::DistLateral1.name(),
        82 => Channel::UrgenciaFrontal.name(),
        83 => Channel::MotionMag0.name(),
        84 => Channel::MotionMag1.name(),
        101 => Channel::Curv260m.name(),
        114 => Channel::ApproachHemiL.name(),
        115 => Channel::ApproachHemiR.name(),
        116 => Channel::FoveaApproach.name(),
        117 => Channel::FoveaMotionAbs.name(),
        85 => Channel::EmaDistL.name(),
        86 => Channel::EmaDistR.name(),
        87 => Channel::SlipAngle.name(),
        88 => Channel::ProxSustain.name(),
        89 => Channel::IsReversing.name(),
        98 => Channel::MotionDelta7.name(),
        99 => Channel::EdgeRow6.name(),
        100 => Channel::EdgeRow7.name(),
        102 => Channel::ApproachRate0.name(),
        103 => Channel::ApproachRate1.name(),
        104 => Channel::ApproachRate2.name(),
        105 => Channel::TtcMin.name(),
        106 => Channel::ApproachRateFineL.name(),
        107 => Channel::ApproachRateFineR.name(),
        108 => Channel::LineOffsetRacingLine.name(),
        109 => Channel::LineOffsetDeriv.name(),
        110 => Channel::LineLookAheadRL.name(),
        111 => Channel::Curv350m.name(),
        112 => Channel::Curv200m.name(),
        113 => Channel::Curv275m.name(),
        118 => Channel::FwdDistNorm.name(),
        119 => Channel::CorridorNorm.name(),
        120 => Channel::Memory0Cfc.name(),
        121 => Channel::SteerMem.name(),
        122 => Channel::BrakeMem.name(),
        123 => Channel::WhiskerNorm.name(),
        124 => Channel::Spinning.name(),
        125 => Channel::CompStress.name(),
        126 => Channel::JoyTick.name(),
        127 => Channel::JoyStreak.name(),
        // Fallback por BLOCO (layout de perception.wgsl:19-46 + motor.rs) — cobre os
        // 192 índices com o nome da FAMÍLIA quando o canal não tem variante no enum.
        // "anonimo" escondia 133/192 canais em toda ferramenta de visualização.
        0..=15 => "retina_pool",
        16..=23 => "parede_dist_fov",
        24..=31 => "parede_largura_fov",
        32..=39 => "parede_curv_fov",
        40..=47 => "estrada_perto_5-40m",
        48..=55 => "estrada_longe_45-150m",
        56..=59 => "material_atual",
        60..=63 => "direcao_de_fuga",
        78..=81 => "curvatura_lookahead",
        90..=93 => "rastreador_centro_escuro",
        94..=97 => "whisker_diagonal",
        128..=143 => "fovea_hd",
        144..=159 => "periferica_hd",
        160..=191 => "horizonte_raw",
        _ => "anonimo",
    }
}

/// Doc por índice — tabela de `name_for` devolvendo `formula_doc()`; famílias em PT.
pub fn doc_for(ch: usize) -> &'static str {
    // Canais nomeados (variantes do enum):
    match ch {
        20 => Channel::FuelFrac.formula_doc(),
        21 => Channel::MotionDelta0.formula_doc(),
        22 => Channel::MotionDelta1.formula_doc(),
        23 => Channel::MotionDelta2.formula_doc(),
        24 => Channel::MotionDelta3.formula_doc(),
        25 => Channel::MotionDelta4.formula_doc(),
        26 => Channel::MotionDelta5.formula_doc(),
        27 => Channel::MotionDelta6.formula_doc(),
        64 => Channel::VelAlign.formula_doc(),
        65 => Channel::SpeedFwd.formula_doc(),
        66 => Channel::VelLat.formula_doc(),
        67 => Channel::VidaFrac.formula_doc(),
        68 => Channel::RpmNorm.formula_doc(),
        69 => Channel::Gear.formula_doc(),
        70 => Channel::TireTemp.formula_doc(),
        71 => Channel::TempMotor.formula_doc(),
        72 => Channel::RiscoCinetico.formula_doc(),
        73 => Channel::OnGrass.formula_doc(),
        74 => Channel::SteerAbs.formula_doc(),
        75 => Channel::Estagnacao.formula_doc(),
        76 => Channel::DistLateral0.formula_doc(),
        77 => Channel::DistLateral1.formula_doc(),
        82 => Channel::UrgenciaFrontal.formula_doc(),
        83 => Channel::MotionMag0.formula_doc(),
        84 => Channel::MotionMag1.formula_doc(),
        101 => Channel::Curv260m.formula_doc(),
        114 => Channel::ApproachHemiL.formula_doc(),
        115 => Channel::ApproachHemiR.formula_doc(),
        116 => Channel::FoveaApproach.formula_doc(),
        117 => Channel::FoveaMotionAbs.formula_doc(),
        85 => Channel::EmaDistL.formula_doc(),
        86 => Channel::EmaDistR.formula_doc(),
        87 => Channel::SlipAngle.formula_doc(),
        88 => Channel::ProxSustain.formula_doc(),
        89 => Channel::IsReversing.formula_doc(),
        98 => Channel::MotionDelta7.formula_doc(),
        99 => Channel::EdgeRow6.formula_doc(),
        100 => Channel::EdgeRow7.formula_doc(),
        102 => Channel::ApproachRate0.formula_doc(),
        103 => Channel::ApproachRate1.formula_doc(),
        104 => Channel::ApproachRate2.formula_doc(),
        105 => Channel::TtcMin.formula_doc(),
        106 => Channel::ApproachRateFineL.formula_doc(),
        107 => Channel::ApproachRateFineR.formula_doc(),
        108 => Channel::LineOffsetRacingLine.formula_doc(),
        109 => Channel::LineOffsetDeriv.formula_doc(),
        110 => Channel::LineLookAheadRL.formula_doc(),
        111 => Channel::Curv350m.formula_doc(),
        112 => Channel::Curv200m.formula_doc(),
        113 => Channel::Curv275m.formula_doc(),
        118 => Channel::FwdDistNorm.formula_doc(),
        119 => Channel::CorridorNorm.formula_doc(),
        120 => Channel::Memory0Cfc.formula_doc(),
        121 => Channel::SteerMem.formula_doc(),
        122 => Channel::BrakeMem.formula_doc(),
        123 => Channel::WhiskerNorm.formula_doc(),
        124 => Channel::Spinning.formula_doc(),
        125 => Channel::CompStress.formula_doc(),
        126 => Channel::JoyTick.formula_doc(),
        127 => Channel::JoyStreak.formula_doc(),
        // Fallback por BLOCO (layout de perception.wgsl:19-46 + motor.rs) — cobre os
        // 192 índices com o nome da FAMÍLIA quando o canal não tem variante no enum.
        // "sem documentação" escondia 133/192 canais em toda ferramenta de visualização.
        0..=15 => "pooling da retina: resumo grosso da imagem (16 células)",
        16..=23 => "distância até a parede em 8 ângulos (alcance 60 m; menor = perto)",
        24..=31 => "largura livre da pista por ângulo, 15 m à frente",
        32..=39 => "curvatura da parede por ângulo (fluxo × direção)",
        40..=47 => "quanto de ESTRADA à frente em 8 amostras de 5-40 m (curva imediata)",
        48..=55 => "estrada à frente de 45-150 m (planejamento de reta)",
        56..=59 => "material sob as rodas (asfalto/grama/parede)",
        60..=63 => "ângulo do corredor de estrada mais longo (por onde escapar)",
        78..=81 => "curvatura da pista à frente em várias distâncias",
        90..=93 => "rastreia o centro escuro (asfalto) na linha média da retina",
        94..=97 => "bigodes ±30°/±60°: obstáculo nas diagonais",
        128..=143 => "VISÃO central HD: 16 colunas do centro do olhar",
        144..=159 => "VISÃO periférica HD: 16 colunas das bordas",
        160..=191 => {
            "VISÃO do horizonte: 32 amostras da linha do olhar (para-brisa; escuro=asfalto)"
        }
        _ => "sem documentação",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Garante que N_CHANNELS bate com SENSORY_WAVE_SIZE do config.
    /// O número 192 deve ser a mesma constante usada pelos shaders.
    #[test]
    fn n_channels_e_192() {
        assert_eq!(
            N_CHANNELS, 192,
            "N_CHANNELS mudou — sincronizar com config::iara::brain::SENSORY_WAVE_SIZE"
        );
    }

    /// Garante que nenhuma variante do enum tem índice fora de [0, N_CHANNELS).
    #[test]
    fn todos_os_indices_dentro_do_range() {
        let canais = all_named_channels();
        for ch in canais {
            assert!(
                ch.index() < N_CHANNELS,
                "Channel::{} tem index {} >= N_CHANNELS={}",
                ch.name(),
                ch.index(),
                N_CHANNELS
            );
        }
    }

    /// Garante que não há duas variantes com o mesmo índice (colisão).
    #[test]
    fn sem_colisao_de_indice() {
        let canais = all_named_channels();
        let mut vistos = std::collections::HashSet::new();
        for ch in &canais {
            assert!(
                vistos.insert(ch.index()),
                "Colisão de índice {} em Channel::{}",
                ch.index(),
                ch.name()
            );
        }
    }

    /// Garante que source_for() não retorna "sem_fonte" para nenhum canal que
    /// está documentado como escrito — ou seja, todos os canais das variantes
    /// nomeadas têm fonte.
    #[test]
    fn source_for_cobre_todos_os_canais_conhecidos() {
        for ch in all_named_channels() {
            let src = source_for(ch.index());
            assert_ne!(
                src,
                "sem_fonte",
                "Channel::{} (index={}) não tem fonte documentada",
                ch.name(),
                ch.index()
            );
        }
    }

    /// Garante que nenhuma variante nomeada é classificada como `Morto` — os 192
    /// índices têm fonte documentada (mesma garantia de `source_for_cobre_todos_os_canais_conhecidos`,
    /// agora através do método tipado `Channel::source()`).
    #[test]
    fn source_nao_marca_variante_nomeada_como_morto() {
        for ch in all_named_channels() {
            assert!(
                !matches!(ch.source(), ChannelSource::Morto(_)),
                "Channel::{} (index={}) foi classificado Morto — deveria ter fonte",
                ch.name(),
                ch.index()
            );
        }
    }

    /// Garante que toda variante nomeada tem uma fórmula curta documentada (não vazia).
    #[test]
    fn formula_doc_nao_vazia_para_todos_os_canais() {
        for ch in all_named_channels() {
            assert!(
                !ch.formula_doc().is_empty(),
                "Channel::{} (index={}) sem formula_doc()",
                ch.name(),
                ch.index()
            );
        }
    }

    /// Garante que a classificação Host bate com os canais documentados como
    /// injetados pela CPU (o achado original que motivou este arquivo: canais
    /// com valor calculado no host mas escrito ao buffer pelo shader).
    #[test]
    fn source_host_para_canais_injetados_pela_cpu() {
        let injetados_pela_cpu = [
            Channel::SlipAngle,
            Channel::EdgeRow6,
            Channel::EdgeRow7,
            Channel::SteerMem,
            Channel::BrakeMem,
            Channel::Spinning,
            Channel::CompStress,
            Channel::JoyTick,
            Channel::JoyStreak,
        ];
        for ch in injetados_pela_cpu {
            assert_eq!(
                ch.source(),
                ChannelSource::Host,
                "Channel::{} deveria ser Host (injetado via update_*_temporal)",
                ch.name()
            );
        }
    }

    /// Garante que index() é a identidade (sem cast incorreto) e ancora os
    /// quatro índices que estavam trocados na primeira versão deste arquivo
    /// (ver comentário "Como isto foi verificado" no topo).
    #[test]
    fn index_consistente_com_valor_discriminante() {
        assert_eq!(Channel::VelAlign.index(), 64);
        assert_eq!(Channel::SpeedFwd.index(), 65);
        assert_eq!(Channel::MotionDelta0.index(), 21);
        assert_eq!(Channel::WhiskerNorm.index(), 123);
        assert_eq!(Channel::MotionDelta7.index(), 98);
        assert_eq!(Channel::FwdDistNorm.index(), 118);
        assert_eq!(Channel::CorridorNorm.index(), 119);
        assert_eq!(Channel::Memory0Cfc.index(), 120);
        assert_eq!(Channel::ProxSustain.index(), 88);
        assert_eq!(Channel::IsReversing.index(), 89);
        assert_eq!(Channel::LineOffsetRacingLine.index(), 108);
        assert_eq!(Channel::LineOffsetDeriv.index(), 109);
        assert_eq!(Channel::LineLookAheadRL.index(), 110);
        // Família joy/spin/stress — CORRIGIDA (estava JoyTick=124, JoyStreak=125,
        // Spinning=126, CompStress=127 na primeira versão deste arquivo).
        assert_eq!(Channel::Spinning.index(), 124);
        assert_eq!(Channel::CompStress.index(), 125);
        assert_eq!(Channel::JoyTick.index(), 126);
        assert_eq!(Channel::JoyStreak.index(), 127);
        // Approach rate / TTC — renomeados e reordenados (TtcFrontal→TtcMin,
        // TtcLateral0/1→ApproachRateFineL/R).
        assert_eq!(Channel::ApproachRate0.index(), 102);
        assert_eq!(Channel::ApproachRate1.index(), 103);
        assert_eq!(Channel::ApproachRate2.index(), 104);
        assert_eq!(Channel::TtcMin.index(), 105);
        assert_eq!(Channel::ApproachRateFineL.index(), 106);
        assert_eq!(Channel::ApproachRateFineR.index(), 107);
        // FASE CANAIS (2026-08-14): canais recém-ligados no caminho GPU de gpu_evolve.
        assert_eq!(Channel::Curv260m.index(), 101);
        assert_eq!(Channel::ApproachHemiL.index(), 114);
        assert_eq!(Channel::ApproachHemiR.index(), 115);
        assert_eq!(Channel::FoveaApproach.index(), 116);
        assert_eq!(Channel::FoveaMotionAbs.index(), 117);
    }

    /// `DEAD_L1_INPUTS` (FASE CANAIS) referencia índices válidos e motivos não-vazios
    /// — sanidade básica do registro que `Genome::random` consome para zerar L1.
    #[test]
    fn dead_l1_inputs_indices_validos_e_motivo_nao_vazio() {
        for &(ch, motivo) in DEAD_L1_INPUTS {
            assert!(
                ch < N_CHANNELS,
                "DEAD_L1_INPUTS tem índice {ch} >= N_CHANNELS={N_CHANNELS}"
            );
            assert!(
                !motivo.is_empty(),
                "DEAD_L1_INPUTS ch{ch} sem motivo documentado"
            );
        }
    }

    // ─── CHANNEL FAMILIES (D8 charges a subscription per family) ────────────

    /// The families tile `0..N_CHANNELS` exactly: no gap, no overlap, in order.
    ///
    /// D8 taxes `DOSE * sum_F (|F| / N_CHANNELS) * u_F`. A gap would make a channel
    /// free to keep (an untaxed family is a hiding place for the pruning the pressure
    /// is supposed to select against); an overlap would tax one channel twice and the
    /// weights would no longer sum to 1, so "cost at reference use == DOSE" — the
    /// number the metric ceiling is computed from — would silently be false.
    #[test]
    fn families_tile_the_wave() {
        let mut next = 0usize;
        for &(name, lo, hi) in CHANNEL_FAMILIES {
            assert_eq!(lo, next, "family {name} starts at {lo}, expected {next}");
            assert!(hi > lo, "family {name} is empty ({lo}..{hi})");
            next = hi;
        }
        assert_eq!(
            next, N_CHANNELS,
            "families stop at {next}, the sensory wave is {N_CHANNELS} wide"
        );
        let widths: usize = CHANNEL_FAMILIES.iter().map(|&(_, lo, hi)| hi - lo).sum();
        assert_eq!(widths, N_CHANNELS, "family widths must sum to the wave");
    }

    /// Every channel belongs to exactly one family, and out-of-range answers `None`.
    #[test]
    fn family_of_is_total_inside_the_wave() {
        for ch in 0..N_CHANNELS {
            let f = family_of(ch).unwrap_or_else(|| panic!("ch{ch} has no family"));
            let (_, lo, hi) = CHANNEL_FAMILIES[f];
            assert!(ch >= lo && ch < hi, "ch{ch} mapped to the wrong family");
        }
        assert_eq!(family_of(N_CHANNELS), None);
    }

    // ─── BODY BLOCK (BODY PHASE, 2026-08-17) ────────────────────────────────

    /// The block's slots are exactly `0..BODY_K`, each appearing once, in order.
    /// A duplicated or skipped slot would make `update_body_channels` write one
    /// value over another with no error — the same class of silent corruption as a
    /// wrong stride (`gpu/CLAUDE.md`, "host and shader must agree on layout").
    #[test]
    fn body_slots_cover_0_to_body_k_with_no_gap() {
        assert_eq!(
            BODY_SLOTS.len(),
            BODY_K,
            "BODY_SLOTS has {} rows but BODY_K = {BODY_K}",
            BODY_SLOTS.len()
        );
        for (i, &(slot, _, _, _)) in BODY_SLOTS.iter().enumerate() {
            assert_eq!(
                slot, i,
                "BODY_SLOTS[{i}] declares slot {slot}, expected {i}"
            );
        }
    }

    /// Two slots writing the same channel = one overwrites the other in the shader,
    /// silently. And a channel index outside the wave writes into another being's
    /// data (the `retina_digest` failure already paid for, `gpu/CLAUDE.md`).
    #[test]
    fn body_slots_have_unique_channels_inside_the_wave() {
        let mut seen: Vec<usize> = Vec::new();
        for &(slot, ch, name, doc) in BODY_SLOTS {
            assert!(
                ch < N_CHANNELS,
                "BODY_SLOTS slot {slot} points at ch{ch} >= N_CHANNELS={N_CHANNELS}"
            );
            assert!(
                !seen.contains(&ch),
                "BODY_SLOTS: ch{ch} appears in two slots (the second would overwrite the first)"
            );
            seen.push(ch);
            assert!(!name.is_empty(), "BODY_SLOTS slot {slot} has no name");
            assert!(!doc.is_empty(), "BODY_SLOTS slot {slot} has no doc");
        }
    }

    /// The block may only land on channels that were MEASURED dead (ch98, 114-117,
    /// 124 — `piloto_forja.rs:49`, "`sd` has 6 degenerate channels (98, 114, 115,
    /// 116, 117, 124) with `sd = 1e-12`") or on the two documented SOURCE moves
    /// (ch70, ch71 — same meaning, same encoding, different feeder).
    ///
    /// This is the test that stops the block from quietly eating a live channel: if
    /// someone adds a row pointing at, say, ch65 (speed, L1 magnitude ≈ 23), the
    /// build fails here instead of the champion silently going blind.
    #[test]
    fn body_slots_only_take_measured_dead_channels_or_source_moves() {
        const MEASURED_DEAD: [usize; 6] = [98, 114, 115, 116, 117, 124];
        const SOURCE_MOVES: [usize; 2] = [70, 71];
        for &(slot, ch, name, _) in BODY_SLOTS {
            assert!(
                MEASURED_DEAD.contains(&ch) || SOURCE_MOVES.contains(&ch),
                "BODY_SLOTS slot {slot} ({name}) takes ch{ch}, which is neither a \
                 measured-dead channel {MEASURED_DEAD:?} nor a documented source move \
                 {SOURCE_MOVES:?}"
            );
        }
    }

    /// The two lookups are inverses of each other over the whole block.
    #[test]
    fn body_slot_and_channel_are_inverses() {
        for &(slot, ch, _, _) in BODY_SLOTS {
            assert_eq!(body_slot_for_channel(ch), Some(slot));
            assert_eq!(body_channel_for_slot(slot), Some(ch));
        }
        // A channel outside the block has no slot, with the flag on or off.
        assert_eq!(body_slot_for_channel(Channel::SpeedFwd.index()), None);
        assert_eq!(body_channel_for_slot(BODY_K), None);
    }

    /// With the flag OFF nothing changes — the audit keeps seeing the old names and
    /// the old kernels. Project rule 3: default-off means bit-for-bit the old world.
    #[test]
    fn body_off_changes_no_channel_name_or_source() {
        for ch in 0..N_CHANNELS {
            assert_eq!(name_for_with_body(ch, false), name_for(ch));
            assert_eq!(source_for_with_body(ch, false), source_for(ch));
        }
    }

    /// With the flag ON, exactly the block's channels change name and source — and
    /// no others. ch98/114-117 must stop being reported as `motion.wgsl`: motion is
    /// deliberately silent on them, and blaming it would send the next debugger to
    /// the wrong kernel.
    #[test]
    fn body_on_switches_name_and_source_of_exactly_the_block_channels() {
        for ch in 0..N_CHANNELS {
            let in_block = body_slot_for_channel(ch).is_some();
            let name_changed = name_for_with_body(ch, true) != name_for(ch);
            let source_changed = source_for_with_body(ch, true) != source_for(ch);
            assert_eq!(
                name_changed, in_block,
                "ch{ch}: name changed={name_changed} but in_block={in_block}"
            );
            assert_eq!(
                source_changed, in_block,
                "ch{ch}: source changed={source_changed} but in_block={in_block}"
            );
        }
        // Anchors: the handoff away from motion.wgsl, and a name that changes meaning.
        assert_eq!(source_for(98), "motion.wgsl");
        assert!(source_for_with_body(98, true).starts_with("temporal.wgsl"));
        assert_eq!(name_for_with_body(124, true), "PredatorProx");
        assert_eq!(name_for_with_body(124, false), "Spinning");
    }

    /// Retorna todas as variantes do enum (lista mantida manualmente).
    fn all_named_channels() -> Vec<Channel> {
        vec![
            Channel::MotionDelta0,
            Channel::MotionDelta1,
            Channel::MotionDelta2,
            Channel::MotionDelta3,
            Channel::MotionDelta4,
            Channel::MotionDelta5,
            Channel::MotionDelta6,
            Channel::VelAlign,
            Channel::SpeedFwd,
            Channel::VelLat,
            Channel::VidaFrac,
            Channel::RpmNorm,
            Channel::Gear,
            Channel::TireTemp,
            Channel::TempMotor,
            Channel::RiscoCinetico,
            Channel::OnGrass,
            Channel::SteerAbs,
            Channel::Estagnacao,
            Channel::DistLateral0,
            Channel::DistLateral1,
            Channel::UrgenciaFrontal,
            Channel::MotionMag0,
            Channel::MotionMag1,
            Channel::ApproachHemiL,
            Channel::ApproachHemiR,
            Channel::FoveaApproach,
            Channel::FoveaMotionAbs,
            Channel::EmaDistL,
            Channel::EmaDistR,
            Channel::SlipAngle,
            Channel::ProxSustain,
            Channel::IsReversing,
            Channel::MotionDelta7,
            Channel::EdgeRow6,
            Channel::EdgeRow7,
            Channel::ApproachRate0,
            Channel::ApproachRate1,
            Channel::ApproachRate2,
            Channel::TtcMin,
            Channel::ApproachRateFineL,
            Channel::ApproachRateFineR,
            Channel::LineOffsetRacingLine,
            Channel::LineOffsetDeriv,
            Channel::LineLookAheadRL,
            Channel::Curv260m,
            Channel::Curv350m,
            Channel::Curv200m,
            Channel::Curv275m,
            Channel::FwdDistNorm,
            Channel::CorridorNorm,
            Channel::Memory0Cfc,
            Channel::SteerMem,
            Channel::BrakeMem,
            Channel::WhiskerNorm,
            Channel::Spinning,
            Channel::CompStress,
            Channel::JoyTick,
            Channel::JoyStreak,
        ]
    }
}
