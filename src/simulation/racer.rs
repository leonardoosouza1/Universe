use crate::config::iara::vision::RETINA_TOTAL;
use crate::engine::SovereignEngine;
use bevy::prelude::*;
use std::collections::{HashSet, VecDeque};

/// Tier 1.1 — Reward shaping denso.
/// Ring buffers (128 amostras ≈ 2s @ 64Hz FixedUpdate) + voxel grid de exploração.
/// Populado em motor.rs a cada tick, consumido em pipeline::calc_fitness.
pub const TELEMETRY_CAP: usize = 128;
pub const TELEMETRY_DT: f32 = 1.0 / 64.0; // FixedUpdate step — usado nas integrais.
pub const TELEMETRY_VOXEL_SIZE: f32 = 10.0; // célula 10m p/ novelty.

/// # PRESSÃO DE SOBREVIVÊNCIA — as duas escalas que hoje a desligam
///
/// Estes dois valores estavam inline em `spawn.rs:829` como `fuel: 1000.0, hp: 100000.0`,
/// **exatamente 1000× a escala que todo o resto do código supõe**. Três evidências:
///
/// | onde | o que supõe |
/// |---|---|
/// | `motor.rs:2370` `hp -= 1.0*dt` "~100s até morrer" | HP ≈ 100 |
/// | `motor.rs:2470` `hp -= 30.0*dt` "morre em ~3s" · dano de parede máx 40 | HP ≈ 100 |
/// | `motor.rs:1572` `slow_input[3] = fuel * 127.0` **com clamp em 127** | combustível ∈ [0,1] |
///
/// A terceira é a mais cara: com `fuel = 1000` esse canal fica saturado em 127 para sempre,
/// e o cérebro tem um sensor de combustível que é uma CONSTANTE — informação zero.
///
/// Consequência medida: os carros só morrem de batida (buraco, parede, estagnação). HP e
/// combustível nunca mordem. Não há a tensão de sobrevivência que o projeto descreve.
///
/// # Valores atuais: pressão LIGADA, começando generosa
///
/// Escolha do Leonardo (*"ambiente mais propício à vida e depois ir dificultando"*): **3× a
/// escala coerente**, não a escala coerente direto. O motivo é o modo de falha documentado
/// em `PROJETO_DE_MUNDO.md` — com 70–99% de mortalidade já existente, ligar a pressão cheia
/// de uma vez faria do mundo uma armadilha, e em armadilha nada se aprende.
///
/// ⚠️ **CORREÇÃO MEDIDA:** `HP_INICIAL` era quase INÓCUO. A cura de checkpoint em
/// `motor.rs` fazia `hp = (hp + 40).min(100.0)` — teto FIXO em 100 —, então a primeira
/// passagem por um checkpoint (a cada 25 m) travava o HP em 100 qualquer que fosse o valor
/// de nascimento. Prova: a distribuição de causas de morte é idêntica com HP=100.000 e com
/// HP=300 (HP ~59%, SpeedBleed ~40% nos dois). O teto da cura agora acompanha esta constante.
///
/// A economia real de HP é: **+40% do tanque a cada 25 m percorridos**, contra drenos de
/// 1/s (bateria), 30/s (rodando parado) e 60/s (velocidade < 4 m/s sem progresso).
/// Ou seja: **mover ou morrer**. E é isso que a autópsia mostra — 98,7% das mortes são
/// HP ou SpeedBleed, e BATIDA são 3 em 11.450.
///
/// O que muda em relação ao infinito de antes:
/// - o canal `slow_input[3]` deixa de ser constante e passa a informar de verdade — foi
///   preciso normalizá-lo pelo tanque inicial (`motor.rs`), senão ele só sairia da
///   saturação depois de 2/3 gastos
/// - derrapar (`hp -= slip*10*dt`) passa a custar: ~10 s de patinação contínua matam
/// - bater passa a acumular: dano de parede vai a 40, então ~8 batidas fortes matam
///
/// Para APERTAR depois, o próximo degrau é `HP_INICIAL = 100.0` e `FUEL_INICIAL = 1.0`, que
/// é a escala que os danos supõem — e que fecha uma tensão bonita: a 0,006/s o tanque de 1.0
/// dura ~167 s ≈ **2,7 voltas**, enquanto o reabastecimento só vem na **3ª**. O ser tem de
/// ser eficiente o bastante para chegar ao posto antes de secar. Com o tanque de 3.0 de hoje
/// dá ~8 voltas, então o combustível ainda não morde — é generoso de propósito. Apertar só quando a população sustentar o
/// nível atual — o mesmo critério do currículo de ambiente
/// (`pipeline::decide_next_env_level_pop`), pelo mesmo motivo.
pub const FUEL_INICIAL: f32 = 3.0;
// HP_INICIAL elevado de 300 para 1000.
//
// Cálculo dos piores casos (todos os drenos simultâneos, sem regeneração):
//   motor afogado    0,1 HP/s
//   anti-donut       0,2 HP/s
//   donut/zerinho    0,5 HP/s
//   derrapagem       0,2 HP/s  (avg_slip ≈ 1,0)
//   raspagem parede  0,3 HP/s
//   estagnação       0,5 HP/s
//   TOTAL MÁXIMO:    1,8 HP/s  → HP=1000 dura ~556 s ≈ 9 min
//
// Caso ruim típico (parede + donut + stag, sem afogado/slip):
//   0,3 + 0,5 + 0,5 = 1,3 HP/s → ~770 s ≈ 12 min
//
// Com regeneração (1,0 HP/s enquanto avança): saldo é -0,3 HP/s no pior caso ativo.
//
// O relógio real continua sendo o FUEL (~150 s / tanque de FUEL_INICIAL=3.0 a consumo
// base de ~0,002/s × mult_fome=10). Nenhum dos drenos de HP consegue matar antes do
// tanque secar — exatamente o que o usuário pediu: "não vamos deixar ele morrer, os
// testes são relevantes com os modelos vivendo muito e dando várias voltas".
pub const HP_INICIAL: f32 = 1000.0;

#[derive(Clone)]
pub struct RacerTelemetry {
    pub steering_history: VecDeque<f32>,     // steer_intent [-1, 1]
    pub speed_history: VecDeque<f32>,        // m/s
    pub on_road_mask: VecDeque<bool>,        // true quando voxel == Road
    pub yaw_rate_history: VecDeque<f32>,     // rad/s
    pub visited_voxels: HashSet<(i32, i32)>, // células (cx, cz) visitadas ON-ROAD
    pub hit_npc_count: u32,                  // Tier 5.4 — colisões com NPCs obstáculos móveis
}

impl RacerTelemetry {
    pub fn new() -> Self {
        Self {
            steering_history: VecDeque::with_capacity(TELEMETRY_CAP),
            speed_history: VecDeque::with_capacity(TELEMETRY_CAP),
            on_road_mask: VecDeque::with_capacity(TELEMETRY_CAP),
            yaw_rate_history: VecDeque::with_capacity(TELEMETRY_CAP),
            visited_voxels: HashSet::new(),
            hit_npc_count: 0,
        }
    }

    fn push_capped<T>(buf: &mut VecDeque<T>, val: T) {
        if buf.len() == TELEMETRY_CAP {
            buf.pop_front();
        }
        buf.push_back(val);
    }

    pub fn record(
        &mut self,
        steer: f32,
        speed: f32,
        yaw_rate: f32,
        on_road: bool,
        pos_x: f32,
        pos_z: f32,
    ) {
        Self::push_capped(&mut self.steering_history, steer);
        Self::push_capped(&mut self.speed_history, speed);
        Self::push_capped(&mut self.yaw_rate_history, yaw_rate);
        Self::push_capped(&mut self.on_road_mask, on_road);
        if on_road {
            let cx = (pos_x / TELEMETRY_VOXEL_SIZE).floor() as i32;
            let cz = (pos_z / TELEMETRY_VOXEL_SIZE).floor() as i32;
            self.visited_voxels.insert((cx, cz));
        }
    }

    /// Variância populacional do histórico de steering (jitter).
    pub fn steering_variance(&self) -> f32 {
        let n = self.steering_history.len();
        if n < 2 {
            return 0.0;
        }
        let mean: f32 = self.steering_history.iter().sum::<f32>() / n as f32;
        let var: f32 = self
            .steering_history
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / n as f32;
        var
    }

    /// Σ speed * on_road * DT — distância "útil" percorrida na pista (≈metros on-road).
    pub fn speed_on_track(&self) -> f32 {
        self.speed_history
            .iter()
            .zip(self.on_road_mask.iter())
            .map(|(&s, &on)| if on { s * TELEMETRY_DT } else { 0.0 })
            .sum()
    }

    /// Σ (1 - |yaw_rate|.min(1)) * DT on-road — premia controle de yaw.
    pub fn stability_integral(&self) -> f32 {
        self.yaw_rate_history
            .iter()
            .zip(self.on_road_mask.iter())
            .map(|(rate, on)| {
                if *on {
                    (1.0 - rate.abs().min(1.0)) * TELEMETRY_DT
                } else {
                    0.0
                }
            })
            .sum()
    }
}

impl Default for RacerTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeathReason {
    HpZero,     // HP drained outside wall event
    Pinado,     // Wall-time > 0.8s
    Batida,     // Wall hit chain → HP 0
    Stagnation, // max_progress não cresceu
    SpeedBleed, // < 2 m/s + sem progresso → HP drain
    Fuel,       // fuel <= 0
    Hole,       // Caiu em buraco (VoxelType::Hole) → morte instantânea
    NpcHit,     // Colisão violenta com NPC obstáculo → HP 0
}

impl DeathReason {
    pub fn label(&self) -> &'static str {
        match self {
            DeathReason::HpZero => "HP",
            DeathReason::Pinado => "PINADO",
            DeathReason::Batida => "BATIDA",
            DeathReason::Stagnation => "STAG",
            DeathReason::SpeedBleed => "SBLD",
            DeathReason::Fuel => "FUEL",
            DeathReason::Hole => "HOLE",
            DeathReason::NpcHit => "NPC",
        }
    }
}

#[derive(Component)]
pub struct Racer {
    pub name: String,
    pub car_model: String,
    pub car_model_idx: u8,
    pub engine: SovereignEngine,
    pub is_dead: bool,
    pub amplitude: f32,
    pub phase: f32,
    pub pain_memories: Vec<PainSynapse>,
    pub prev_pos: Vec3,
    pub distance: f32,
    pub fuel: f32,
    pub hp: f32,
    pub total_laps: usize,
    pub stagnation: f32,
    pub memory: [f32; 5], // [throttle, steer, brake, prev_shift_up, prev_shift_down]
    pub visual_cortex: [i8; RETINA_TOTAL],
    // --- NOVOS CAMPOS (Sovereign Drive Evolution) ---
    pub tire_temp: f32,                  // Carga térmica dos pneus 0.0-1.0
    pub engine_power: f32,               // Degradação mecânica 0.1-1.0 (1.0 = full)
    pub grass_time: f32,                 // Tempo acumulado na grama (penalidade fitness)
    pub wall_time: f32,                  // Tempo acumulado raspando parede (penalidade fitness)
    pub aquaplaning_timer: f32,          // Timer de aquaplanagem (0.5s de perda de controle)
    pub joy_integral: f32, // Dopamina acumulada: Σ(joy_tick * streak_mult * dt) — driver de fitness
    pub joy_streak: f32,   // Tempo contínuo em flow state (cap 5s); zera em "dor"
    pub wall_hits: u32,    // Contador de impactos na parede (3 strikes = dead)
    pub wall_cooldown: f32, // Tempo até registrar novo hit na parede
    pub last_waypoint: usize, // Último waypoint alcançado para cura
    pub track_progress: f32, // Progresso real ao longo da polilinha da pista (arc-length)
    pub grace_ticks: u32, // Contador de frames desde spawn; imune a time_scale (grace real de ~10s)
    pub reverse_timer: f32, // >0 enquanto em manobra de ré de emergência; decrementa com dt
    pub progress_checkpoint: f32, // Último track_progress que marcou avanço real (+3m)
    pub progress_stall: f32, // Segundos acumulados SEM ganhar 3m de pista (anti-donut)
    pub wall_ema_l: f32,  // EMA da distância à parede esquerda (decay ~1s) — sensação persistente
    pub wall_ema_r: f32,  // EMA da distância à parede direita
    pub wall_dist_prev_l: f32, // Distância à parede esquerda frame anterior (m) — para cálculo approach rate
    pub wall_dist_prev_r: f32, // Distância à parede direita frame anterior (m)
    pub wall_dist_prev_f: f32, // Distância à parede frontal frame anterior (m)
    pub proximity_sustain: f32, // Segundos contínuos raspando alguma lateral (alerta sustentado)
    pub post_reverse_lockout: f32, // Cooldown pós-ré com throttle capado (evita "ré-acelera-bate")
    pub spawn_progress: f32, // track_progress da posição de spawn — fitness usa (track_progress - spawn_progress)
    pub path: Vec<Vec3>, // Histórico de posições (amostrado ~4Hz) — usado para desenhar traçado do campeão
    pub path_timer: f32, // Cooldown entre amostras de path
    pub brain_trace: Vec<BrainSnapshot>, // Snapshots periódicos do cérebro (diagnóstico)
    pub trace_timer: f32, // Cooldown entre snapshots
    pub gaze_yaw: f32,   // Ângulo da fóvea relativo ao forward (rad). Atualizado por saccade
    // reativo — aponta pro caminho mais aberto detectado pela periférica.
    // Clamp ±π/3. Canal 98 expõe pra rede.
    /// Comprimento da volta em metros, copiado de `world.lap_distance` junto com a
    /// atualização de progresso. Existe para `calc_fitness` computar VOLTAS FRACIONÁRIAS sem
    /// precisar enxergar o mundo. Zero = ainda não escrito, e aí a fitness cai no
    /// comportamento antigo (contagem inteira) — o que mantém a comparação A/B honesta.
    pub lap_dist_m: f32,
    pub max_progress: f32, // track_progress máximo já alcançado — MONOTÔNICO. Imune a oscilação.
    // Base do stagnation check: sem crescimento > 2m em 1.5s virtuais, morte.
    pub slow_time: f32, // SEGUNDOS VIRTUAIS com velocidade < 3 m/s após grace. > 0.5s → sangria pesada.
    pub race_time: f32, // SEGUNDOS VIRTUAIS vivos — imune a time_scale. Usado pra grace real
    // (grace_ticks conta frames reais, inadequado quando sim roda a 8x).
    pub prev_lap_prog: f32, // In-lap arc position do frame anterior — usado pra detectar wrap
    // direction-agnostic (prev > 0.85*lap_dist && cur < 0.15*lap_dist).
    pub post_lap_grace: u32, // Frames restantes de imunidade a STAGNATION/SPEED_BLEED após lap.
    // Carro que completa volta para no waypoint[0] → max_progress congela →
    // morria por stagnation antes de re-acelerar. Grace dá tempo pra sair.
    pub post_lap_checkpoint: f32, // track_progress no instante do último lap. Grace só acaba quando
    // carro avança 25m ALÉM desse checkpoint — garante saída da start line.
    pub death_reason: Option<DeathReason>, // Motivo causal de morte (autopsia).
    pub death_frame: u32,                  // grace_ticks no instante da morte (≈ frame real).
    pub death_pos: Vec3,                   // Posição mundo da morte (zona na pista).
    pub peak_speed: f32,                   // Velocidade máxima registrada (m/s) pro reporte.
    pub telemetry: RacerTelemetry,         // Tier 1.1: ring buffers densos pra reward shaping.
    // --- Tier 6.5: Time Dilation sob stress ---
    /// Contador de ticks pra modular frequência de inferência do cérebro.
    /// Wraps em u8 (256 ticks ≈ 4s @ 64Hz — mais que suficiente para skip mod).
    pub brain_tick_counter: u8,
    /// Última ação (throttle, steer) emitida pelo cérebro. Reusada quando
    /// skip_n > 1 pra manter comando estável entre inferências (time dilation).
    /// Nota: engine.final_intent[] já persiste entre ticks por ser membro da SovereignEngine,
    /// mas mantemos cópia canonical aqui pro contrato documental do Tier 6.5.
    pub last_action: (f32, f32),
    // --- Tier 4.3: Nicho-splitting (runtime-only, NOT serialized) ---
    /// 0 = defensive (penaliza wall_time 2x), 1 = aggressive (premia speed_on_track 1.5x).
    /// Alternado no spawn (i % 2). Shared-pool: ambos os nichos competem por mesmos elites,
    /// cross-breed emerge naturalmente. Annotation-only pra lineage introspection.
    pub niche: u8,
    // --- Tier 5.1 LITE: Multi-car social perception ---
    /// Posições relativas (dx, dy) dos 3 rivais mais próximos, normalizadas por
    /// clamp de 50m → [-1.0, 1.0]. Layout: [dx0, dy0, dx1, dy1, dx2, dy2], ordenados
    /// do mais próximo pro mais distante. Slots vazios ficam em 0.0.
    /// Data layer only — populado via `EnvironmentalState::closest_racers` mas ainda
    /// NÃO wired no motor/sensor sampling (wire-up em follow-up). Canais retina 102-107.
    pub perceived_rivals: [f32; 6],
    // --- Tier 6.3 LITE: morphological co-evolution (body params do DNA) ---
    /// Genes corporais (massa, wheelbase, downforce, grip_mult) carregados do DNA no spawn.
    /// Por ora apenas ARMAZENADOS — wiring físico em follow-up. Ver `crate::evolution::dna_genesis::MorphoGenes`.
    pub morpho: crate::evolution::dna_genesis::MorphoGenes,
    // --- KAN output head: spline suave por output (throttle/steer/brake) ---
    /// WHY: elimina zig-zag ±1 saturado do clamp duro; evoluído via mutate_spline_knots.
    pub kan_out: crate::kan::KanOutputHead,
    /// Seed estável para PRNG local (ruído sensorial/exploração determinístico).
    pub stable_seed: u64,
    /// Last sensory snapshot (64 semantic channels) for visualization/diagnostics.
    /// Updated each frame in update_racers() for UI to display what agent sees.
    pub sensory_snapshot: [i8; 64],
}

/// Frame de diagnóstico do cérebro do racer.
/// Capturado a ~2Hz e dumpeado nos reports em momentos chave (morte, campeão).
/// Revela se o sinal sensorial está chegando, se o volante responde, e
/// quanto foi atenuado entre o neurônio e a física.
#[derive(Clone)]
pub struct BrainSnapshot {
    pub t: f32,                      // Tempo de vida do carro
    pub speed_kmh: f32,              // Velocidade
    pub wall_prox_l: i8,             // Canal 85: proximidade parede esquerda (0=longe, 127=colada)
    pub wall_prox_r: i8,             // Canal 86: proximidade parede direita
    pub ttc_inv: i8,                 // Canal 82: urgência de colisão frontal
    pub curv_25: i8,                 // Canal 79: curvatura antecipada 25m
    pub retina_center_max: i8, // Pico dos canais centrais da retina (10..22) — parede no centro
    pub raw_throttle: f32,     // final_intent[0] do cérebro
    pub raw_steer: f32,        // final_intent[1] (pré-atenuação)
    pub raw_brake: f32,        // final_intent[2]
    pub post_damp_steer: f32,  // steer_intent depois do risk_damp do slow brain
    pub physical_steer: f32,   // O que REALMENTE chegou no volante (com dynamic_steer_limit)
    pub steer_limit: f32,      // Fator dinâmico 1/(1+v*0.02)
    pub risk_damp: f32,        // Atenuação do slow brain (0..0.3)
    pub top_channels: [(u8, i8); 5], // Top-5 canais mais ativados (|valor|), com índice+valor

    // --- Vision replay data (compressed retina + sensory channels) ---
    pub retina_fov: Vec<i8>, // FOV_BINS × RETINA_ROWS (dynamic size, supports variable retina dimensions)
    pub sensory_compact: [i8; 32], // Top-32 telemetry channels: speed, alignment, curvature, walls, etc
}

pub struct PainSynapse {
    pub position: Vec3,
    pub pain_phase: f32,
    pub intensity: f32,
}

#[derive(Component)]
pub struct Skidmark {
    pub timer: f32,
}
