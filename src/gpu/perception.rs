//! # Percepção na GPU — host Rust para `percepcao.wgsl`
//!
//! ## O que este módulo faz
//!
//! Mantém os buffers wgpu necessários para o kernel de percepção e expõe uma função
//! `reduzir()` que, dado o buffer de retina física (saída de `RetinaGpu::render_raw()`),
//! devolve um `Vec<[i8; WAVE_SIZE]>` — um sensory_wave parcial por ser.
//!
//! "Parcial" porque alguns canais dependem de estado temporal por ser (visual_cortex,
//! wall_ema_l/r, approach_rate, etc.) ou de estado inter-seres (drafting, comp_stress).
//! Esses continuam na CPU. Veja `CANAIS_NA_CPU` abaixo.
//!
//! ## Layout do buffer de saída
//!
//! O kernel escreve 56 u32 por ser (ceil(223/4) = 56). Cada byte é um canal i8
//! empacotado little-endian:
//!
//! ```text
//! canal ch → saida[ser * 56 + ch/4], byte = ch % 4
//! ```
//!
//! ## Integração com o motor.rs (agente D)
//!
//! 1. Motor.rs chama `RetinaGpu::render()` → buffer físico em VRAM.
//! 2. Motor.rs chama `percepcao.reduzir(&poses)` — passa poses (posição + heading) de
//!    todos os seres. O kernel usa o buffer de retina que já está em VRAM.
//! 3. Motor.rs lê `wave[ser]` como base do sensory_wave e sobrescreve os canais
//!    temporais/inter-seres na CPU (ver `CANAIS_NA_CPU`).
//!
//! A transferência eliminada: 1764 bytes × N seres (retina física) → ~223 bytes × N seres.
//! Com 60 seres e 64 Hz: de ~6,8 MB/s para ~856 KB/s de PCIe no sentido GPU→CPU.
//!
//! ## Canais que ficam na CPU — razão documentada
//!
//! | ch | nome | motivo |
//! |---|---|---|
//! | 21-27, 98 | motion per-row | `visual_cortex` anterior (estado temporal por ser, não está na GPU) |
//! | 29-31, 33-35, 99-100 | edge density | idem |
//! | 44-47 → 44-51 | wall_imminent, trapped_score | dependem de speed → canal 65 (estado do corpo) |
//! | 52-59 | depth_per_row, material_code | motor.rs sobrescreve depois — GPU escreve 48-51 (FAR), CPU sobrescreve 52-55 |
//! | 64-75 | telemetria proprioceptiva | RPM, velocidade, temperatura (estado do corpo físico) |
//! | 72 | lateral acceleration demand | host racing-line geometry + body speed |
//! | 83-84, 114-117 | fluxo óptico | `visual_cortex` anterior |
//! | 85-86 | wall EMA | acumulado por ser entre ticks (`wall_ema_l/r`) |
//! | 87 | slip angle | estado físico (`body.velocity`) |
//! | 88 | proximity_sustain | acumulado por ser |
//! | 89 | estado de ré | acumulado por ser |
//! | 78-80, 101, 111-113 | track curve lookahead | host follows racing line by arc length |
//! | 102-107 | approach rate | `wall_dist_prev_l/r/f` — estado temporal |
//! | 108-110 | racing line | `racing_line.half_width` + estado temporal `prev_line_offset` |
//! | 111 | curvatura 350m | motor.rs escreve depois |
//! | 118-119, 123 | width-invariant | `racing_line.half_width` |
//! | 120-122 | memória interna | `racer.memory[0..2]` |
//! | 124 | spinning | `body.angular_velocity` |
//! | 125 | comp_stress | hash espacial entre seres |
//! | 126-127 | dopamina/joy streak | acumulado por ser |
//! | 82 | urgência frontal | GPU escreve distância; CPU sobrescreve com TTC real (usa speed) |
//! | 192+ | injeção CfC percepção | estado recorrente temporal por ser |
//!
//! ## Canais com semântica alterada vs motor.rs (revalidar)
//!
//! | ch | mudança |
//! |---|---|
//! | 0-15, 128-191 | pool sobre física reconstruída via `codificacao_legada()`. Equivalente ao motor.rs legado. Se migrar para física pura, revalidar. |
//! | 82 | GPU usa `1 − fwd_probe/30` (distância); CPU usa `speed / fwd_probe` (TTC). CPU sobrescreve. |
//! | 48-55 (FAR) | altura usa nearest-neighbor (GPU) vs bilinear (CPU). Erro ≤ 0.3m em rampa 10%. |
//!
//! ## Conflitos de slots no motor.rs — o que sobrescreve o quê
//!
//! O motor.rs escreve canais em ordem não-óbvia. Slots que a GPU escreve e depois a CPU
//! sobrescreve: 21-27 (motion), 29-35 (edge+curvatura), 32 (curvatura 180m), 36-43 (open_l/r),
//! 44-47 (wall_imminent), 48-51 (trapped_score), 52-55 (depth_per_row), 56-63 (agent_state/escape).
//! A integração D deve garantir que a CPU escreva SOBRE esses slots depois de copiar o wave da GPU.

#[cfg(test)]
#[path = "perception_whisker_test.rs"]
mod whisker_tests;

use std::borrow::Cow;

use bevy::math::Vec2;
use wgpu::util::DeviceExt;

use crate::config::iara::brain::SENSORY_WAVE_SIZE;
use crate::config::iara::vision::RETINA_TOTAL;

/// Número de u32 por ser no buffer de saída. ceil(SENSORY_WAVE_SIZE / 4).
pub const WAVE_U32S: usize = (SENSORY_WAVE_SIZE + 3) / 4; // (192 + 3) / 4 = 48

/// Pose de um ser para o kernel de percepção.
/// Campos em coordenadas de mundo — mesmos campos que `PoseGpu` da retina, mais `phase`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PosePerception {
    /// Posição mundo (x, y, z em metros).
    pub pos_x: f32,
    pub pos_y: f32,
    pub pos_z: f32,
    /// `racer.phase` — heading absoluto em radianos. Usado pelos whiskers (ch 94-97).
    pub phase: f32,
    /// Forward 2D normalizado (x, z).
    pub fwd_x: f32,
    pub fwd_z: f32,
    /// Right 2D normalizado (x, z).
    pub rgt_x: f32,
    pub rgt_z: f32,
}

/// Parâmetros uniformes do kernel — layout deve bater exatamente com `Params` em percepcao.wgsl.
/// Liga a visão periférica REAL (flag `--visao-periferica-real`). Off por padrão:
/// percepção nova entra atrás de flag, com A/B pareado (regra do projeto).
pub static PERIF_REAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// BODY BLOCK (BODY PHASE, 2026-08-17) — turns on the CPU-fed body/hostile-world
/// channels described in `perception::channels::BODY_SLOTS`. **Off by default**
/// (project rule 3: new perception enters behind a default-off flag, with a paired
/// A/B — `main.rs:715-718`).
///
/// # The single switch, and why there is exactly one
///
/// This one static becomes the uniform field `body_on`, which is built ONCE per
/// tick (`reduce`/`bind_and_dispatch`) into the ONE `buf_params` shared by all
/// three perception pipelines. `perception_temporal.wgsl` starts writing ch98 and
/// ch114-117 when it is set, and `perception_motion.wgsl` stops writing them —
/// from the same field, in the same uniform, read in the same tick. Two independent
/// params could disagree for one tick and produce either a double `atomicOr` (bit
/// garbage, no error) or a mute channel (also no error). One field cannot.
///
/// Read with `Relaxed`, like `PERIF_REAL`: it is stored once at start-up by `main`
/// before any dispatch, never toggled mid-run.
pub static BODY_CHANNELS_ON: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Uniform shared by the three perception pipelines.
///
/// # Layout is a contract with THREE shaders
///
/// 32 bytes, and it must stay a multiple of 16: WGSL rounds a uniform struct up to
/// its 16-byte alignment, so a host struct of 5 f32 (20 bytes) against a shader
/// struct that WGSL sizes as 32 would be a binding smaller than the shader demands.
/// The explicit `_pad0.._pad2` make both sides read the same 32 bytes with no
/// implicit padding to reason about. Same precedent as `brain.wgsl`'s `Params`
/// (`tests/wgsl_validation.rs`: "gained `out_gain` + `norm_l2_by_std` (+ padding to
/// keep the uniform a multiple of 16 bytes)").
///
/// Growing this struct is the SAFE direction and only that one — `min_binding_size:
/// None` (`uniform_entry`) lets the buffer be larger than a shader's declaration, so
/// a kernel that declares fewer fields keeps reading the original bytes
/// bit-identically (`src/gpu/CLAUDE.md`). Shrinking, reordering or inserting a field
/// in the middle is the silent-corruption case. **If you change this struct, change
/// the `Params` struct in perception.wgsl, perception_temporal.wgsl AND
/// perception_motion.wgsl in the same commit** — all three bind this same buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsGpu {
    grid_size: u32,
    n_beings: u32,
    cell_size: f32,
    /// 1.0 = visão periférica REAL (ver perception.wgsl). 0.0 = legado (a periférica
    /// era cópia bit-a-bit da fóvea — 16 canais redundantes de 192).
    perif_real: f32,
    /// 1.0 = BODY BLOCK on (see `BODY_CHANNELS_ON`). Read by the temporal kernel
    /// (starts writing the block's channels) and by the motion kernel (stops writing
    /// ch98/114-117) — one field, so they can never disagree.
    body_on: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Número de f32 no buffer de estado temporal por ser.
/// Layout:
///  [0]  wall_dist_l          — distância lateral esquerda no tick anterior (m)
///  [1]  wall_dist_r          — idem, direita
///  [2]  wall_dist_f          — idem, frontal
///  [3]  wall_ema_l           — EMA da distância esquerda
///  [4]  wall_ema_r           — idem, direita
///  [5]  proximity_sustain    — tempo acumulado raspando
///  [6]  joy_streak           — (não usado no GPU; reservado)
///  [7]  memory0              — atualizado pela CPU após física
///  [8]  prev_line_offset_n   — desvio normalizado [-1,1] do tick anterior (para d_off)
///  [9]  _pad
///  [10] cpu_line_off_norm    — ch108: desvio norm [-1,1], injetado pela CPU
///  [11] cpu_d_off            — ch109: derivada do desvio, escalada x8, injetado pela CPU
///  [12] cpu_la_off           — ch110: lookahead offset norm [-1,1], injetado pela CPU
///  [13] cpu_fwd_norm         — ch118: dist frontal / (hw*3), norm [0,1]
///  [14] cpu_corridor         — ch119: (dist_l+dist_r)/(2*hw*2), norm [0,1]
///  [15] cpu_whisker_norm     — ch123: min(dist_l,dist_r)/(hw*1.5), norm [0,1]
///  [16] cpu_dist_l           — distância lateral esq ATUAL (m), injetada pela CPU
///  [17] cpu_dist_r           — distância lateral dir ATUAL (m), injetada pela CPU
///  [18] cpu_dist_f           — distância frontal ATUAL (m), injetada pela CPU
///  [19] _pad2
///
/// Telemetria proprioceptiva (ch 64-75) — injetada via update_proprioception_temporal():
///  [20] prop_alignment        — cos(ângulo heading vs tangente da pista) [-1,1] → ch64
///  [21] prop_speed_m_s        — |vel| em m/s [0, 80] → ch65 (CRÍTICO: modelo sabe se anda)
///  [22] prop_lat_vel          — dot(vel, lat) em m/s [-20, 20] → ch66 (drift)
///  [23] prop_hp_frac          — HP restante / HP_inicial [0,1] → ch67
///  [24] prop_rpm_ratio        — current_rpm / max_rpm [0,1] → ch68
///  [25] prop_gear             — current_gear [0,7] → ch69
///  [26] prop_tire_temp_norm   — average normalised tyre temperature [0,1] → ch70
///  [27] prop_temp_motor       — engine_temp [20-150°C → 0-1] → ch71 (slot era downforce, sobrescrito)
///  [28] prop_curve_risk       — clamp(v²|κ30m|/(0.9g),0,1) → ch72
///  [29] prop_on_grass         — 1.0 se voxel atual é grama, else 0.0 → ch73
///  [30] prop_steer_abs        — |steer| [0,1] → ch74 (efeito limitador)
///  [31] prop_stagnation_ticks — ticks sem avançar [0-64 → 0-1] → ch75 (slot era wetness, sobrescrito)
///
/// NOVOS slots v2 (expandidos de 32 → 50):
///  [32] reverse_timer         — tempo restante de ré (s); > 0 = em ré → ch89 = 127
///  [33-48] cfc_hidden[0..15] — estado recorrente do CfC de percepção → ch192-207
///           O CfC de percepção (RetinaPerception) roda na CPU com a retina lida por
///           readback, e os 16 valores do hidden são injetados aqui antes do dispatch.
///           No shader: wave[192+i] = clamp(tanh(hidden[i]) * 127 * 0.3, -127, 127)
///  [49] _pad3
///
/// NOVOS slots v3 (expandidos de 50 → 54) — canais dopaminérgicos e de stress:
///  [50] joy_tick      — progresso*superfície normalizado [-1,1] → ch126
///  [51] joy_streak    — segundos de flow acumulado / 5.0 [0,1] → ch127
///  [52] is_spinning   — 1.0 se |av_y|>1.5 && speed<2.0 → ch124 = -128
///  [53] comp_stress   — proximidade normalizada [0,1] (0=líder, 1=alguém a <20m) → ch125
///
/// Adicionalmente, slots usados para ch87, ch99, ch100, ch121, ch122 (sem slot próprio —
/// injetados via update_extra2_temporal em array separado de 5 f32):
///  update_extra2_temporal grava em [49] _pad3 (reutilizado) NÃO — usa array separado
///  porque os canais 87/99/100/121/122 são calculados no readback e chegam tarde.
///  Esses sinais são entregues via update_extra2_temporal() gravando nos slots [49-53]
///  SE ainda não houver conflito. Nota: [49] era _pad3 e agora vira:
///  [49] slip_signal   — ch87: cross2D(fwd, vel_norm) [-1,1]
///  E os slots [50-53] acima (v3) são usados para ch124-127.
///  Para ch99/100 e ch121/122: entregues via update_extra3_temporal nos slots [54-57].
///
/// NOVOS slots v4 (expandidos de 54 → 58):
///  [54] edge_row6     — edge density linha 6 da retina [0,1] → ch99
///  [55] edge_row7     — edge density linha 7 da retina [0,1] → ch100
///  [56] steer_mem     — phys.steer suavizado [-1,1] → ch121
///  [57] brake_mem     — phys.brake_cmd [0,1] → ch122
///
/// NOVOS slots v5 (expandidos de 58 → 66) — BODY BLOCK (BODY PHASE, 2026-08-17):
///  [58-65] the `BODY_K` = 8 values of the body / hostile-world block, injected by
///          `update_body_channels()`. The slot->channel map is the canonical
///          registry `perception::channels::BODY_SLOTS` — deliberately NOT repeated
///          here (`src/CLAUDE.md` rule 1: a channel index comes from the registry,
///          one copy only).
///          Read by the temporal shader only when `params.body_on == 1.0`.
///
/// Campos [16-18] são injetados pela CPU em `update_dist_temporal()` ANTES do dispatch.
/// Campos [20-31] são injetados pela CPU em `update_proprioception_temporal()` ANTES do dispatch.
/// Campos [32] e [33-48] são injetados em `update_extra_temporal()` ANTES do dispatch.
/// Campos [49-57] são injetados em `update_extra2_temporal()` APÓS readback (dentro do tick).
/// Slots [58-65] are injected by `update_body_channels()` BEFORE the dispatch.
/// Slots [66-72] carry arc-length curvature lookaheads computed with the same
/// racing-line projection already used for slots [10-15].
/// Elimina loops sobre `grid` no shader temporal, prevenindo o crash do compilador ACO
/// do RADV que ocorria com loops sobre storage buffers em workgroup_size=1.
pub const CURVE_SLOT_BASE: usize = 58 + crate::perception::channels::BODY_K;
pub const CURVE_SLOT_COUNT: usize = 7;
pub const TEMPORAL_STRIDE: usize = CURVE_SLOT_BASE + CURVE_SLOT_COUNT;

fn initial_temporal_state(n: usize) -> Vec<f32> {
    let mut values = vec![0.0; n * TEMPORAL_STRIDE];
    for being in 0..n {
        let base = being * TEMPORAL_STRIDE;
        values[base] = 40.0;
        values[base + 1] = 40.0;
        values[base + 2] = 50.0;
        values[base + 3] = 40.0;
        values[base + 4] = 40.0;
        values[base + 16] = 40.0;
        values[base + 17] = 40.0;
        values[base + 18] = 50.0;
        values[base + 20] = 1.0;
        values[base + 23] = 1.0;
        values[base + 25] = 1.0;
        values[base + 26] = 0.5;
    }
    values
}

/// First `buf_temporal` slot of the BODY BLOCK — the base the shader adds
/// `BODY_SLOTS`'s slot index to.
///
/// It is the old `TEMPORAL_STRIDE` (58): the block was APPENDED, so every slot
/// already measured keeps its index and no existing channel moves. Locked to the
/// stride by `body_block_fecha_o_stride` below — the guarded-stride pattern that
/// `EXTRA_OUT_STRIDE = 19` uses in `retina_digest.rs:88`, after writes past the end
/// of one being's region landed on the NEXT being's channels 0-15.
///
/// CRITICAL: `perception_temporal.wgsl` hardcodes the same two numbers (its
/// `TEMPORAL_STRIDE` and `BODY_SLOT_BASE` consts). WGSL cannot import a Rust const,
/// so the agreement is by test, not by compiler — see `body_block_fecha_o_stride`.
pub const BODY_SLOT_BASE: usize = 58;

/// Contexto de compute persistente para o kernel de percepção.
///
/// Criado uma vez por corrida (ou por troca de mundo). Compartilha o mesmo device/queue
/// da `RetinaGpu` — NUNCA criar um segundo device para a mesma GPU; wgpu impõe 1 por processo.
///
/// Por isso `PerceptionGpu::new()` recebe `device` e `fila` por referência, não cria os seus.
///
/// # Três kernels encadeados
///
/// O shader combinado (~1050 linhas) travava o compilador ACO do RADV em
/// `create_compute_pipeline` com SIGSEGV — o mesmo problema que ocorreu com
/// `physics.wgsl` (resolvido dividindo em três kernels). A solução é a mesma:
///
/// 1. `perception.wgsl` — canais espaciais/retina (7 bindings, layout_spatial)
/// 2. `perception_temporal.wgsl` — canais temporais (8 bindings, layout_temporal)
/// 3. `perception_motion.wgsl` — ch 21-27 fluxo óptico (4 bindings, layout_motion)
///    Lê retina atual + retina anterior and replaces the legacy spatial values in ch 21-27.
///    A retina anterior é preservada via copy_retina_prev() ANTES do dispatch da retina.
///
/// Os três compartilham `buf_output`; the motion pass owns the final value of ch 21-27.
/// `bind_and_dispatch` despacha todos no mesmo `CommandEncoder` sem readback entre eles.
pub struct PerceptionGpu {
    // ── Pipeline espacial (perception.wgsl) — 8 bindings (binding 7 = buf_sensors,
    // FASE SENSORES degrau 1; era 7 bindings antes deste campo existir) ────────────
    pipeline_spatial: wgpu::ComputePipeline,
    layout_spatial: wgpu::BindGroupLayout,
    // ── Pipeline temporal (perception_temporal.wgsl) — 8 bindings ───────────────
    pipeline_temporal: wgpu::ComputePipeline,
    layout_temporal: wgpu::BindGroupLayout,
    // ── Pipeline motion (perception_motion.wgsl) — 4 bindings ───────────────────
    // Calcula ch 21-27 (motion_per_row) comparando retina atual vs anterior.
    // Weight L1 total dos canais 21-27 no CHAMPION.dna = 91.1 (alta importância).
    pipeline_motion: wgpu::ComputePipeline,
    layout_motion: wgpu::BindGroupLayout,
    /// Grade de voxels (estática por pista, subida uma vez). Mesmos dados de `RetinaGpu`.
    buf_grid: wgpu::Buffer,
    /// Campo de fluxo (estático por pista). Dois f32 por célula (x, z).
    buf_flow: wgpu::Buffer,
    /// Mapa de altura (estático por pista). Um f32 por célula.
    buf_heights: wgpu::Buffer,
    /// Poses dos seres (atualizado por tick via write_buffer). Tamanho: capacidade × PosePercep.
    buf_poses: wgpu::Buffer,
    /// Banco de sensores herdáveis por ser (FASE SENSORES, degrau 1: FOV do sweep de
    /// parede/curvatura, alcance de parede, alcance lateral, 3 lookaheads de curvatura —
    /// ver `gpu::evolution::N_SENSOR_GENES`/`sensor_idx`). Constante durante a geração,
    /// como τ: herdado do genoma, enviado uma vez por `upload_sensors`. Nasce em
    /// `SENSOR_DEFAULTS` repetido (os hardcodes de hoje) para que quem nunca chamar
    /// `upload_sensors` (ex.: um caminho de teste que só usa `reduce` sem preparar
    /// genomas) reproduza o comportamento antigo — mesmo padrão de `buf_tau` em
    /// `brain.rs`. Binding 7 de `layout_spatial` (só o pipeline ESPACIAL lê; os lookaheads
    /// de curvatura de `perception_temporal.wgsl` ficam fora de escopo nesta fase, ver
    /// doc de `N_SENSOR_GENES`).
    buf_sensors: wgpu::Buffer,
    /// Saída: WAVE_U32S u32 por ser. Copiado para buf_readback antes de mapear.
    buf_output: wgpu::Buffer,
    /// Buffer MAP_READ, tamanho = buf_output. Separado porque STORAGE | MAP_READ é inválido.
    buf_readback: wgpu::Buffer,
    /// Estado temporal por ser — residente na VRAM entre ticks.
    /// Layout: TEMPORAL_STRIDE f32 por ser (ver constante acima).
    /// Inicializado com prev_dist=15m e ema=15m para evitar spike de approach_rate no tick 0.
    /// Campos [10-15] são preenchidos pela CPU antes do dispatch (racing_line pré-computada).
    /// Nunca sofre readback no laço quente — o kernel lê + atualiza sem sair da GPU.
    buf_temporal: wgpu::Buffer,
    /// Retina do tick anterior — mantida em VRAM para calcular motion (ch 21-27).
    /// Tamanho: capacity × RETINA_TOTAL × 4 bytes. Inicializada com zeros (sem movimento no tick 0).
    /// Atualizada via copy_retina_prev(enc, buf_retina_fisica) ANTES do dispatch da retina,
    /// preservando a retina do tick t-1 antes de ser sobrescrita pela retina do tick t.
    buf_retina_prev: wgpu::Buffer,
    /// Tamanho em bytes do buf_retina_prev / buf_retina_fisica.
    retina_bytes: u64,
    /// Capacidade máxima de seres (N que foi alocado).
    pub capacity: usize,
    /// Número de células da grade (grid_size²).
    grid_size: u32,
    cell_size: f32,
}

impl PerceptionGpu {
    /// Cria o contexto, subindo grade + fluxo + alturas uma única vez.
    ///
    /// `device` e `fila` são emprestados de `RetinaGpu` — mesmo device, sem conflito.
    /// `retina_fisica_buf` é o `buf_output` da retina que ficará em VRAM; passamos o
    /// handle para o bind group (binding 5), eliminando qualquer cópia.
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        world: &crate::world::VoxelWorld,
        capacity: usize,
    ) -> Option<Self> {
        // ── Pipeline espacial: perception.wgsl (7 bindings, sem temporal) ─────
        // Dividido de um shader de ~1050 linhas que crashava o compilador ACO do RADV.
        // Mesma solução usada em physics.wgsl (dividido em 3 kernels).
        eprintln!("[perception] create_shader_module (spatial)...");
        let shader_spatial = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("perception-spatial"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("perception.wgsl"))),
        });
        eprintln!("[perception] shader_module (spatial) OK");

        // Layout espacial: 7 bindings (0-6). Binding 7 (temporal) removido deste shader.
        // O compilador ACO do RADV crashava com o shader combinado de ~1050 linhas;
        // ao dividir em dois shaders menores, cada um compila sem travar.
        let layout_spatial = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("perception-spatial-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // grade
                storage_entry(2, true),  // fluxo
                storage_entry(3, true),  // alturas
                storage_entry(4, true),  // poses
                storage_entry(5, true),  // retina física
                storage_entry(6, false), // buf_output (read_write, atomic)
                // binding 7 aqui = buf_sensors (FASE SENSORES, degrau 1). Namespace de
                // binding é POR LAYOUT — não colide com o binding 7 de `layout_temporal`
                // (que é o estado temporal, um buffer inteiramente diferente num pipeline
                // separado). Ver doc de `buf_sensors` no struct.
                storage_entry(7, true), // sensores herdáveis (read-only)
            ],
        });
        eprintln!("[perception] create_pipeline_layout (spatial)...");
        let pipe_layout_spatial = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_spatial],
            push_constant_ranges: &[],
        });
        eprintln!("[perception] create_compute_pipeline (spatial)...");
        let pipeline_spatial = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("perception-spatial-pipeline"),
            layout: Some(&pipe_layout_spatial),
            module: &shader_spatial,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        eprintln!("[perception] pipeline (spatial) OK");

        // ── Pipeline temporal: perception_temporal.wgsl (8 bindings) ─────────
        // Segundo kernel: lê estado temporal (binding 7) e escreve canais temporais
        // no mesmo buf_output (binding 6). Encadeado no mesmo CommandEncoder.
        eprintln!("[perception] create_shader_module (temporal)...");
        let shader_temporal = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("perception-temporal"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "perception_temporal.wgsl"
            ))),
        });
        eprintln!("[perception] shader_module (temporal) OK");

        // Layout temporal: 8 bindings (0-7), idêntico ao layout original.
        // Os bindings 0-6 são os mesmos; binding 7 é o estado temporal (read_write).
        let layout_temporal = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("perception-temporal-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // grade (mesmo buffer)
                storage_entry(2, true),  // fluxo (mesmo buffer)
                storage_entry(3, true),  // alturas (mesmo buffer)
                storage_entry(4, true),  // poses (mesmo buffer)
                storage_entry(5, true),  // retina física (mesmo buffer)
                storage_entry(6, false), // buf_output (mesmo buffer, escrita nos canais temporais)
                storage_entry(7, false), // estado temporal — lê prev_dist/EMA, escreve atualizado
            ],
        });
        eprintln!("[perception] create_pipeline_layout (temporal)...");
        let pipe_layout_temporal = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_temporal],
            push_constant_ranges: &[],
        });
        eprintln!("[perception] create_compute_pipeline (temporal)...");
        let pipeline_temporal = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("perception-temporal-pipeline"),
            layout: Some(&pipe_layout_temporal),
            module: &shader_temporal,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        eprintln!("[perception] pipeline (temporal) OK");

        // ── Pipeline de motion: perception_motion.wgsl (4 bindings) ──────────
        // Calcula ch 21-27 (fluxo óptico por linha) comparando retina atual vs anterior.
        // Weight L1 dos canais 21-27 no CHAMPION.dna = 91.1 — essencial para navegação em curvas.
        // Dividido em kernel separado para evitar o crash ACO do RADV (shader grande → SIGSEGV).
        eprintln!("[perception] create_shader_module (motion)...");
        let shader_motion = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("perception-motion"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("perception_motion.wgsl"))),
        });
        eprintln!("[perception] shader_module (motion) OK");

        // Layout motion: 4 bindings (params, retina_cur, retina_prev, output).
        // Não precisa de grade/fluxo/alturas/poses/temporal — só retina delta.
        let layout_motion = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("perception-motion-layout"),
            entries: &[
                uniform_entry(0),        // params (n_beings)
                storage_entry(1, true),  // retina_cur (tick atual)
                storage_entry(2, true),  // retina_prev (tick anterior)
                storage_entry(3, false), // buf_output (read_write, atomicOr)
            ],
        });
        let pipe_layout_motion = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_motion],
            push_constant_ranges: &[],
        });
        eprintln!("[perception] create_compute_pipeline (motion)...");
        let pipeline_motion = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("perception-motion-pipeline"),
            layout: Some(&pipe_layout_motion),
            module: &shader_motion,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        eprintln!("[perception] pipeline (motion) OK");

        // ── Buffers estáticos: grade, fluxo, alturas ─────────────────────────
        // grade: u32 por célula (VoxelType como u32).
        // COPY_DST (além de STORAGE): permite `reload_world()` trocar a pista via
        // `queue.write_buffer` sem recriar o `wgpu::Buffer` nem os bind groups que o
        // referenciam. Só é seguro porque `VoxelWorld::new` usa `size=320`/`cell_size=3.0`
        // CONSTANTES independentes da seed (world.rs) — o tamanho em bytes destes três
        // buffers nunca muda entre pistas, então o write cabe sempre (P1, rotação de
        // pistas — docs/PLANO_EVOLUCAO.md).
        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("percep-grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // fluxo: vec2<f32> por célula → [f32, f32] empacotado.
        // VoxelWorld::flow é Vec<Vec2>; Vec2 = [f32; 2] no layout de memória do Bevy
        // (bevy::math::Vec2 = glam::Vec2 = repr(C) { x: f32, y: f32 }).
        // bytemuck::cast_slice funciona diretamente porque Vec2 é Pod.
        let flow_flat: Vec<f32> = world.flow.iter().flat_map(|v| [v.x, v.y]).collect();
        let buf_flow = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("percep-flow"),
            contents: bytemuck::cast_slice(&flow_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // alturas: f32 por célula.
        let buf_heights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("percep-heights"),
            contents: bytemuck::cast_slice(&world.height_map),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ── Buffers dinâmicos: poses, saida, leitura ─────────────────────────
        let cap = capacity.max(64);
        let bytes_poses = (cap * std::mem::size_of::<PosePerception>()) as u64;
        let bytes_output = (cap * WAVE_U32S * std::mem::size_of::<u32>()) as u64;

        let buf_poses = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("percep-poses"),
            size: bytes_poses,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("percep-output"),
            size: bytes_output,
            // COPY_DST necessário para enc.clear_buffer() que zera antes do dispatch atomicOr.
            // COPY_SRC necessário para enc.copy_buffer_to_buffer() pós-dispatch.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("percep-readback"),
            size: bytes_output,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Banco de sensores (FASE SENSORES, degrau 1) ───────────────────────
        // Nasce em SENSOR_DEFAULTS repetido por ser — o hardcode de hoje (FOV=50°,
        // wall_range=60m, lat_range=40m, curv=[40,70,120]m) — para que o caminho que
        // nunca chama `upload_sensors` (testes, `gpu_validate` antes do primeiro upload)
        // reproduza o comportamento antigo, exatamente como `buf_tau` nasce em
        // DECAY_DEFAULT em `brain.rs`.
        let sensors_init: Vec<f32> = crate::gpu::evolution::SENSOR_DEFAULTS
            .iter()
            .cycle()
            .take(cap * crate::gpu::evolution::N_SENSOR_GENES)
            .copied()
            .collect();
        let buf_sensors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("percep-sensors"),
            contents: bytemuck::cast_slice(&sensors_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ── Buffer de estado temporal por ser ─────────────────────────────────
        // TEMPORAL_STRIDE f32 por ser, zerado na criação.
        // Zeros iniciais: EMA começa em 0m (será forçado para dist_l/r real nos primeiros ticks).
        // prev_dist começa em 15m (sem approach rate no tick 0 → canais 102-107 ficam zerados).
        // Após 1 tick, todos os estados refletem valores reais.
        // TEMPORAL_STRIDE expandido de 50 → 58:
        //   slots 32: reverse_timer (ch89)
        //   slots 33-48: cfc_hidden[0..15] (ch192-207)
        //   slot 49: slip_signal (ch87)
        //   slots 50-53: joy_tick, joy_streak, is_spinning, comp_stress → ch124-127
        //   slots 54-57: edge_row6, edge_row7, steer_mem, brake_mem → ch99, ch100, ch121, ch122
        // Distances and proprioception start at the same conservative neutral values on
        // construction and on every generation reset. All remaining slots are neutral zero.
        let temporal_init = initial_temporal_state(cap);
        let buf_temporal = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("percep-temporal"),
            contents: bytemuck::cast_slice(&temporal_init),
            // read_write: o kernel lê e escreve; COPY_SRC/DST para diagnóstico/readback futuro.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // A racing_line é pré-computada pela CPU e injetada em buf_temporal[10-15]
        // antes de cada dispatch (ver update_racing_line_temporal).

        // ── Buffer de retina anterior (para cálculo de motion ch 21-27) ──────
        // Tamanho: cap × RETINA_TOTAL × 4 bytes (mesmo que buf_retina_fisica da RetinaGpu).
        // COPY_DST: recebe cópia da retina antes de ser sobrescrita pela retina atual.
        // STORAGE: lido pelo kernel motion.
        // Inicializado com zeros → no tick 0, motion = |cur - 0| → ligeiramente elevado
        // mas aceitável (modelo não treinou com tick 0 zerado, mas converge em 1 tick).
        let retina_bytes = (cap * RETINA_TOTAL * std::mem::size_of::<u32>()) as u64;
        let buf_retina_prev = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("percep-retina-prev"),
            size: retina_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            pipeline_spatial,
            layout_spatial,
            pipeline_temporal,
            layout_temporal,
            pipeline_motion,
            layout_motion,
            buf_grid,
            buf_flow,
            buf_heights,
            buf_poses,
            buf_sensors,
            buf_output,
            buf_readback,
            buf_temporal,
            buf_retina_prev,
            retina_bytes,
            capacity: cap,
            grid_size: world.size as u32,
            cell_size: world.cell_size,
        })
    }

    /// Clears all perception history that belongs to a previous life.
    ///
    /// The temporal buffer is small and receives its documented neutral values once per
    /// generation. The previous retina is cleared by a GPU command, avoiding a host upload
    /// of the full image and keeping the reset outside the per-tick path.
    pub fn reset_temporal(&self, device: &wgpu::Device, queue: &wgpu::Queue, n: usize) {
        let n = n.min(self.capacity);
        if n == 0 {
            return;
        }
        let temporal = initial_temporal_state(n);
        queue.write_buffer(&self.buf_temporal, 0, bytemuck::cast_slice(&temporal));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("perception-generation-reset"),
        });
        let retina_bytes = (n * RETINA_TOTAL * std::mem::size_of::<u32>()) as u64;
        encoder.clear_buffer(&self.buf_retina_prev, 0, Some(retina_bytes));
        queue.submit(Some(encoder.finish()));
    }

    /// Clears only one being's temporal perception state and previous retina.
    ///
    /// Ecology reuses dead slots during an episode. Resetting the prefix with
    /// [`reset_temporal`] would erase the visual history of living beings, so births use
    /// this slot-scoped variant instead.
    pub fn reset_temporal_slot(&self, device: &wgpu::Device, queue: &wgpu::Queue, being: usize) {
        if being >= self.capacity {
            return;
        }
        let neutral = initial_temporal_state(1);
        let temporal_offset = (being * TEMPORAL_STRIDE * std::mem::size_of::<f32>()) as u64;
        queue.write_buffer(
            &self.buf_temporal,
            temporal_offset,
            bytemuck::cast_slice(&neutral),
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("perception-slot-reset"),
        });
        let retina_offset = (being * RETINA_TOTAL * std::mem::size_of::<u32>()) as u64;
        let retina_bytes = (RETINA_TOTAL * std::mem::size_of::<u32>()) as u64;
        encoder.clear_buffer(&self.buf_retina_prev, retina_offset, Some(retina_bytes));
        queue.submit(Some(encoder.finish()));
    }

    /// Troca a grade/fluxo/alturas para uma pista NOVA sem recriar buffers, bind
    /// groups nem pipelines — só `queue.write_buffer` (P1: rotação de pistas por
    /// geração, docs/PLANO_EVOLUCAO.md).
    ///
    /// Só é válido porque `VoxelWorld::new` usa `size`/`cell_size` CONSTANTES
    /// (world.rs: 320/3.0) independentes da seed — os três buffers têm sempre o
    /// mesmo tamanho em bytes, então o `world` novo nunca pode ter grade de
    /// tamanho diferente da que os buffers já têm alocada. O `debug_assert!`
    /// documenta essa invariante e pega em teste/debug qualquer violação futura
    /// (ex.: alguém tornar `size` dependente da seed); em release o custo é zero.
    pub fn reload_world(&self, queue: &wgpu::Queue, world: &crate::world::VoxelWorld) {
        debug_assert_eq!(
            world.size as u32, self.grid_size,
            "reload_world: pista nova tem grade de tamanho diferente — buffer não realoca"
        );
        debug_assert!(
            (world.cell_size - self.cell_size).abs() < 1e-6,
            "reload_world: pista nova tem cell_size diferente — buffer não realoca"
        );
        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        queue.write_buffer(&self.buf_grid, 0, bytemuck::cast_slice(&grid_u32));
        let flow_flat: Vec<f32> = world.flow.iter().flat_map(|v| [v.x, v.y]).collect();
        queue.write_buffer(&self.buf_flow, 0, bytemuck::cast_slice(&flow_flat));
        queue.write_buffer(
            &self.buf_heights,
            0,
            bytemuck::cast_slice(&world.height_map),
        );
    }

    /// Roda o kernel de percepção para N seres, consumindo o buffer de retina física já em VRAM.
    ///
    /// `buf_retina_fisica` — o `buf_output` de `RetinaGpu` (layout: N × 1764 × u32, packed mm|mat).
    /// `poses` — um por ser, mesmo tamanho e ordem que os seres processados pela retina.
    ///
    /// Retorna um vetor de arrays `[i8; SENSORY_WAVE_SIZE]` com os canais portados pela GPU.
    /// Os canais não portados estão a zero — o motor.rs preenche o restante na CPU.
    ///
    /// # Falhas silenciosas
    ///
    /// Se `poses.len() > capacidade`, trunca. Igual ao comportamento de `RetinaGpu::render()`.
    pub fn reduce(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        poses: &[PosePerception],
        buf_retina_fisica: &wgpu::Buffer,
    ) -> Vec<[i8; SENSORY_WAVE_SIZE]> {
        let n = poses.len().min(self.capacity);
        if n == 0 {
            return Vec::new();
        }

        // ── Atualiza poses ────────────────────────────────────────────────────
        queue.write_buffer(&self.buf_poses, 0, bytemuck::cast_slice(&poses[..n]));

        // ── Uniforme (criado por tick; barato — é só 16 bytes) ───────────────
        let params = ParamsGpu {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            perif_real: if PERIF_REAL.load(std::sync::atomic::Ordering::Relaxed) {
                1.0
            } else {
                0.0
            },
            // BODY BLOCK: read ONCE here, so the temporal and motion kernels of this
            // tick see the same value (see `BODY_CHANNELS_ON`).
            body_on: if BODY_CHANNELS_ON.load(std::sync::atomic::Ordering::Relaxed) {
                1.0
            } else {
                0.0
            },
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("percep-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // ── Bind group espacial (8 bindings: 0-6 + 7=buf_sensors, FASE SENSORES) ──
        let bg_spatial = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("percep-bg-spatial"),
            layout: &self.layout_spatial,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_flow.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_heights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_poses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_retina_fisica.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buf_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buf_sensors.as_entire_binding(),
                },
            ],
        });

        // ── Bind group temporal (8 bindings, inclui temporal) ────────────────
        let bg_temporal = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("percep-bg-temporal"),
            layout: &self.layout_temporal,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_flow.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_heights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_poses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_retina_fisica.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buf_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buf_temporal.as_entire_binding(),
                },
            ],
        });

        // ── Dois dispatches encadeados no mesmo encoder ───────────────────────
        // 1º: spatial (workgroup_size=64, 1 wg/ser) → canais 0-97, 128-191
        // 2º: temporal (workgroup_size=1, 1 wg/ser) → canais 85,86,88,102-110,118-120,123
        // Barreira implícita entre compute passes garante visibilidade do buf_output zerado.
        let bytes_out = (n * WAVE_U32S * 4) as u64;
        let mut enc = device.create_command_encoder(&Default::default());
        // ZERAGEM OBRIGATÓRIA: ambos os kernels usam atomicOr.
        // Zero inicial + OR por byte = valor final correto. Sem zero: bits anteriores persistem.
        enc.clear_buffer(&self.buf_output, 0, Some(bytes_out));
        {
            let mut passe = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-percep-spatial"),
                timestamp_writes: None,
            });
            passe.set_pipeline(&self.pipeline_spatial);
            passe.set_bind_group(0, &bg_spatial, &[]);
            passe.dispatch_workgroups(n as u32, 1, 1);
        }
        // Fim do pass espacial → barreira implícita: buf_output visível para o próximo pass.
        {
            let mut passe = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-percep-temporal"),
                timestamp_writes: None,
            });
            passe.set_pipeline(&self.pipeline_temporal);
            passe.set_bind_group(0, &bg_temporal, &[]);
            // workgroup_size=1: 1 thread por ser → 60 workgroups em paralelo na CU.
            passe.dispatch_workgroups(n as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.buf_output, 0, &self.buf_readback, 0, bytes_out);
        queue.submit(Some(enc.finish()));

        // ── Leitura síncrona (block_on → poll) ───────────────────────────────
        // Mesmo padrão de `RetinaGpu::render()`. Em produção, o agente D pode
        // fazer pipeline assíncrono — retina despacha tick N, percepção despacha tick N,
        // CPU trabalha tick N-1 enquanto GPU está ocupada com tick N.
        let slice = self.buf_readback.slice(0..bytes_out);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);

        // ── Desempacota u32 → [i8; 223] ──────────────────────────────────────
        // Layout: wave[ser][ch] = words[ser * WAVE_U32S + ch/4], byte (ch%4).
        // Os bytes estão em two's complement — cast de u8 para i8 é bit-a-bit correto.
        let mut out = vec![[0i8; SENSORY_WAVE_SIZE]; n];
        for being in 0..n {
            let base = being * WAVE_U32S;
            for ch in 0..SENSORY_WAVE_SIZE {
                let word = words[base + ch / 4];
                let byte = ((word >> ((ch % 4) * 8)) & 0xFF) as u8;
                out[being][ch] = byte as i8;
            }
        }

        drop(data);
        self.buf_readback.unmap();
        out
    }

    /// Devolve referência ao buffer de saída (buf_output) para inspeção ou pipeline assíncrono.
    /// O agente D pode usar isto para evitar a cópia de volta se o próximo kernel (cerebro.wgsl)
    /// consumir o sensory_wave diretamente em VRAM.
    pub fn buf_output(&self) -> &wgpu::Buffer {
        &self.buf_output
    }

    /// Devolve referência ao buf_readback (MAP_READ | COPY_DST) para quem quer fazer
    /// map_async + poll(Wait) + recv() manualmente sem usar read_after_submit().
    /// Pré-condição: bind_and_dispatch() já copiou buf_output→buf_readback no encoder.
    pub fn buf_readback_ref(&self) -> &wgpu::Buffer {
        &self.buf_readback
    }

    /// Envia o banco de sensores: `n × N_SENSOR_GENES` valores, em ordem de ser (FASE
    /// SENSORES, degrau 1 — ver doc de `buf_sensors` e `gpu::evolution::N_SENSOR_GENES`).
    ///
    /// Chamado uma vez por geração, junto com os pesos e o τ — os sensores são herdados
    /// do genoma e não mudam durante a vida do ser. Quem não chamar fica com
    /// `SENSOR_DEFAULTS` da inicialização (os hardcodes de hoje), então o braço de
    /// controle do experimento (`SENSORS_EVOLVE_ON` desligado) é o comportamento
    /// histórico — mesmo padrão de `BrainGpu::upload_tau`.
    pub fn upload_sensors(&self, queue: &wgpu::Queue, sensors: &[f32], n: usize) {
        use crate::gpu::evolution::N_SENSOR_GENES;
        assert!(
            n <= self.capacity,
            "n={n} excede capacidade {}",
            self.capacity
        );
        // Comprimento errado seria um bug MUDO: o kernel leria sensores do ser seguinte
        // (ou lixo do fim do buffer) e a "evolução dos sensores" viraria ruído embaralhado
        // sem sintoma visível — mesma classe de erro já vista neste projeto no stride da
        // onda sensorial (ver CLAUDE.md, gap 1).
        assert_eq!(
            sensors.len(),
            n * N_SENSOR_GENES,
            "banco de sensores tem {} valores, esperava {} ({n} seres × {N_SENSOR_GENES} genes)",
            sensors.len(),
            n * N_SENSOR_GENES
        );
        queue.write_buffer(&self.buf_sensors, 0, bytemuck::cast_slice(sensors));
    }

    /// Atualiza `memory0` (índice 7 do estado temporal) para N seres via write_buffer parcial.
    ///
    /// # Por que existe
    ///
    /// O kernel de percepção escreve os canais temporais (85, 86, 88, 102-110, 120) e atualiza
    /// o estado temporal residindo na VRAM. Porém `memory[0]` (throttle acumulado) é computado
    /// pelo cérebro na CPU depois do dispatch — o kernel só pode ler o valor DO TICK ANTERIOR,
    /// que é exatamente o comportamento correto (o cérebro usa a memória do frame passado como
    /// entrada, e depois a CPU grava o novo valor para o próximo tick).
    ///
    /// Chamado uma vez por tick, APÓS a física atualizar os intents.
    /// `memory0_vals`: fatia com N valores em [-1, 1] (o throttle intencional).
    pub fn update_memory0(&self, queue: &wgpu::Queue, memory0_vals: &[f32]) {
        let n = memory0_vals.len().min(self.capacity);
        if n == 0 {
            return;
        }
        // Cada ser ocupa TEMPORAL_STRIDE f32; memory0 está no índice 7.
        // Para atualizar apenas memory0 sem sobreescrever os outros campos, precisamos
        // de writes pontuais. Usamos write_buffer para cada ser — custo: N × 4 bytes de overhead.
        // Com N=60 e 64Hz: 60 × 4 B = 240 B/tick, negligível.
        // ALTERNATIVA: um buffer auxiliar e kernel de cópia — desnecessário a este nível.
        for i in 0..n {
            let byte_offset = (i * TEMPORAL_STRIDE + 7) as u64 * 4;
            queue.write_buffer(
                &self.buf_temporal,
                byte_offset,
                bytemuck::bytes_of(&memory0_vals[i]),
            );
        }
    }

    /// Injeta os 6 valores de racing_line pré-computados pela CPU nos campos [10-15] de
    /// `buf_temporal`, ANTES do dispatch do kernel de percepção.
    ///
    /// # Por que existe
    ///
    /// O kernel original tentava fazer a projeção O(N) na racing_line dentro do shader.
    /// Loops sobre storage buffers de comprimento variável crashavam o compilador ACO/NIR do RADV
    /// (SIGSEGV em `create_compute_pipeline`). Solução: CPU pré-computa, GPU apenas lê.
    ///
    /// # Layout de `racing_vals[i]`
    ///
    /// ```text
    /// [0] cpu_line_off_norm  — ch108: desvio lateral normalizado [-1,1]
    /// [1] cpu_d_off          — ch109: derivada do desvio (escalada), [-1,1]
    /// [2] cpu_la_off         — ch110: lookahead offset norm [-1,1]
    /// [3] cpu_fwd_norm       — ch118: dist frontal / (half_width*3), [0,1]
    /// [4] cpu_corridor       — ch119: (dl+dr)/(2*hw*2), [0,1]
    /// [5] cpu_whisker_norm   — ch123: min(dl,dr)/(hw*1.5), [0,1]
    /// ```
    pub fn update_racing_line_temporal(&self, queue: &wgpu::Queue, racing_vals: &[[f32; 6]]) {
        let n = racing_vals.len().min(self.capacity);
        if n == 0 {
            return;
        }
        // Campos [10-15] = índices 10 a 15 em buf_temporal para cada ser.
        // Escrevemos 6 f32 contíguos de uma vez (24 bytes) por ser — mais eficiente
        // do que 6 write_buffer separados.
        for i in 0..n {
            let byte_offset = (i * TEMPORAL_STRIDE + 10) as u64 * 4;
            queue.write_buffer(
                &self.buf_temporal,
                byte_offset,
                bytemuck::cast_slice(&racing_vals[i]),
            );
        }
    }

    /// Write geometrically correct curvature horizons to temporal slots 66-72.
    /// Values are signed heading deltas normalised by 90 degrees. They are computed
    /// along track arc length on the host because that projection already exists for
    /// the racing-line channels; the seven values then remain in VRAM for the shader.
    pub fn update_curve_temporal(&self, queue: &wgpu::Queue, values: &[[f32; 7]]) {
        let n = values.len().min(self.capacity);
        for (being, value) in values.iter().take(n).enumerate() {
            let byte_offset = (being * TEMPORAL_STRIDE + CURVE_SLOT_BASE) as u64 * 4;
            queue.write_buffer(&self.buf_temporal, byte_offset, bytemuck::cast_slice(value));
        }
    }

    /// Injeta as distâncias laterais/frontais calculadas pela CPU nos campos [16-18] de
    /// `buf_temporal`, ANTES do dispatch do kernel temporal.
    ///
    /// # Por que existe
    ///
    /// O kernel `perception_temporal.wgsl` precisava de loops sobre o storage buffer `grid`
    /// para calcular dist_l/r/f — esses loops com `break` travavam o compilador ACO/NIR do
    /// RADV em `create_compute_pipeline` (SIGSEGV). Solução: CPU calcula as distâncias com
    /// os mesmos arrays de amostragem (LAT_SAMPLES + FWD_SAMPLES) e injeta os 3 valores em
    /// buf_temporal[16-18]. O shader temporal lê esses campos diretamente — sem nenhum loop.
    ///
    /// # Layout de `dist_vals[i]`
    ///
    /// ```text
    /// [0] cpu_dist_l — menor dist lateral esq a Wall (m), LAT_SAMPLES [1,3,6,9,12,16,20,26,33,40]
    /// [1] cpu_dist_r — menor dist lateral dir a Wall (m)
    /// [2] cpu_dist_f — menor dist frontal a Wall/Hole (m), FWD_SAMPLES [3..30]
    /// ```
    pub fn update_dist_temporal(&self, queue: &wgpu::Queue, dist_vals: &[[f32; 3]]) {
        let n = dist_vals.len().min(self.capacity);
        if n == 0 {
            return;
        }
        // Campos [16-18] = índices 16 a 18 em buf_temporal.
        // 3 f32 contíguos (12 bytes) por ser em um único write_buffer.
        for i in 0..n {
            let byte_offset = (i * TEMPORAL_STRIDE + 16) as u64 * 4;
            queue.write_buffer(
                &self.buf_temporal,
                byte_offset,
                bytemuck::cast_slice(&dist_vals[i]),
            );
        }
    }

    /// Injeta telemetria proprioceptiva (ch 64-75) em `buf_temporal[20-31]` ANTES do dispatch.
    ///
    /// # Motivação
    ///
    /// O shader `perception.wgsl` (espacial) não tem acesso ao estado físico do ser (velocidade,
    /// RPM, marcha, temperatura dos pneus) — esses dados ficam na CPU em `PhysicsStateGpu`.
    /// O CHAMPION.dna usa ch65 (speed, mag=23.3 normalizada) e ch71 (downforce) como canais de
    /// alto peso no L1. Sem eles, o modelo não sabe se está parado ou a 80 km/h e tende a
    /// throttle=0 (qualquer aceleração sem feedback de velocidade leva a comportamento instável).
    ///
    /// # Layout de `vals[i]` — 12 f32 por ser
    ///
    /// ```text
    /// [0]  prop_alignment   — cos(angle heading vs pista tangente) ∈ [-1,1]
    /// [1]  prop_speed_m_s   — |vel| em m/s ∈ [0,80]
    /// [2]  prop_lat_vel     — dot(vel,lat) em m/s ∈ [-20,20]
    /// [3]  prop_hp_frac     — HP restante / HP_inicial ∈ [0,1]
    /// [4]  prop_rpm_ratio   — current_rpm / max_rpm ∈ [0,1]
    /// [5]  prop_gear        — current_gear ∈ [0,7] (mapeado como f32)
    /// [6]  prop_tire_temp_n — average normalised tyre temperature ∈ [0,1]
    /// [7]  prop_downforce_n — downforce / weight ∈ [0,1] (reservado)
    /// [8]  prop_in_draft    — 1.0 se em draft, else 0.0 (reservado)
    /// [9]  prop_on_grass    — 1.0 se voxel atual é grama, else 0.0
    /// [10] prop_steer_abs   — |steer| ∈ [0,1]
    /// [11] prop_wetness     — wetness do clima ∈ [0,1] (reservado)
    /// ```
    pub fn update_proprioception_temporal(&self, queue: &wgpu::Queue, vals: &[[f32; 12]]) {
        let n = vals.len().min(self.capacity);
        if n == 0 {
            return;
        }
        // Campos [20-31] = 12 f32 contíguos (48 bytes) por ser.
        // Um write_buffer por ser para gravar os 48 bytes no offset correto.
        // Custo total: N × 48 bytes ≈ 2880 bytes @ 60 seres — negligível.
        for i in 0..n {
            let byte_offset = (i * TEMPORAL_STRIDE + 20) as u64 * 4;
            queue.write_buffer(
                &self.buf_temporal,
                byte_offset,
                bytemuck::cast_slice(&vals[i]),
            );
        }
    }

    /// Injeta `reverse_timer` (slot 32) e `cfc_hidden[0..15]` (slots 33-48) no
    /// `buf_temporal` ANTES do dispatch do kernel temporal.
    ///
    /// # Motivação
    ///
    /// - `reverse_timer` (ch 89): estado por ser acumulado na CPU. Quando > 0, o ser está
    ///   em manobra de ré de emergência. O shader escreve ch89 = 127 se timer > 0, 0 caso contrário.
    ///
    /// - `cfc_hidden[0..15]` (ch 192-207): estado recorrente do CfC de percepção
    ///   (`RetinaPerception::inject_into_wave`, offset=192, strength=0.3). Pesa 7,1% no L1
    ///   do CHAMPION.dna — o maior canal ausente até este commit. O CfC roda na CPU com
    ///   a retina lida via readback, e os 16 valores float são armazenados aqui entre ticks.
    ///   O shader aplica `tanh(h) * 127 * 0.3` replicando exatamente o motor.rs.
    ///
    /// # Layout de `extra_vals[i]`
    ///
    /// ```text
    /// [0]    reverse_timer — tempo restante em ré (s), ≥ 0.0
    /// [1..16] cfc_hidden  — hidden state do CfC de percepção (16 floats)
    /// ```
    pub fn update_extra_temporal(&self, queue: &wgpu::Queue, extra_vals: &[[f32; 17]]) {
        let n = extra_vals.len().min(self.capacity);
        if n == 0 {
            return;
        }
        // Slot 32 (reverse_timer) + slots 33-48 (cfc_hidden[0..15]) = 17 f32 contíguos.
        // Um write_buffer por ser grava os 68 bytes no offset correto.
        // Custo total: N × 68 bytes ≈ 4080 bytes @ 60 seres — negligível.
        for i in 0..n {
            let byte_offset = (i * TEMPORAL_STRIDE + 32) as u64 * 4;
            queue.write_buffer(
                &self.buf_temporal,
                byte_offset,
                bytemuck::cast_slice(&extra_vals[i]),
            );
        }
    }

    /// Injeta sinais dopaminérgicos, de dor e estresse (slots 49-53) ANTES do dispatch.
    ///
    /// Chamado ANTES de `bind_and_dispatch`, depois de calcular os valores na CPU.
    /// Esses canais eram escritos apenas no motor.rs; agora chegam ao caminho GPU também.
    ///
    /// # Layout de `extra2_vals[i]` (9 f32)
    ///
    /// ```text
    /// [0] slip_signal  — cross2D(fwd_norm, vel_norm) [-1,1] → ch87
    /// [1] joy_tick     — progress_speed * surface_factor [-1,1] → ch126
    /// [2] joy_streak   — joy_streak_s / 5.0 [0,1] → ch127
    /// [3] is_spinning  — 1.0 se |av_y|>1.5 && speed<2.0, else 0.0 → ch124
    /// [4] comp_stress  — proximidade normalizada [0,1] → ch125
    /// [5] edge_row6    — edge density linha 6 da retina [0,1] → ch99
    /// [6] edge_row7    — edge density linha 7 da retina [0,1] → ch100
    /// [7] steer_mem    — phys.steer [-1,1] → ch121
    /// [8] brake_mem    — phys.brake_cmd [0,1] → ch122
    /// ```
    pub fn update_extra2_temporal(&self, queue: &wgpu::Queue, extra2_vals: &[[f32; 9]]) {
        let n = extra2_vals.len().min(self.capacity);
        if n == 0 {
            return;
        }
        // Slots 49-57 (9 f32 contíguos). Um write_buffer por ser = 36 bytes.
        for i in 0..n {
            let byte_offset = (i * TEMPORAL_STRIDE + 49) as u64 * 4;
            queue.write_buffer(
                &self.buf_temporal,
                byte_offset,
                bytemuck::cast_slice(&extra2_vals[i]),
            );
        }
    }

    /// BODY BLOCK — injects the `BODY_K` body / hostile-world values of every being
    /// into `buf_temporal[BODY_SLOT_BASE .. BODY_SLOT_BASE + BODY_K]`, BEFORE the
    /// dispatch.
    ///
    /// # Contract with the integrator (owner of context.rs / main.rs)
    ///
    /// - `values[i][slot]` is the value of `BODY_SLOTS[slot]` for being `i`, in the
    ///   order the beings are dispatched — the same indexing as `update_memory0`
    ///   and `update_proprioception_temporal`. Beings past `capacity` are dropped.
    /// - The HOST normalises; the shader only encodes. Bearings arrive in [-1, 1]
    ///   (bearing/π, negative = left of `fwd`), everything else in [0, 1]. The
    ///   reference distance of a proximity ramp is a host-side dose, never a kernel
    ///   constant — so changing `--predator-perception` never touches WGSL.
    /// - Values outside those ranges are NOT rejected here: they clamp in the shader,
    ///   exactly like every other channel. Passing metres where [0,1] was asked for
    ///   therefore produces a saturated channel, not an error — a canary that a
    ///   channel is pinned at 127 is the alarm for it.
    /// - Call it EVERY tick while `BODY_CHANNELS_ON` is set. `buf_temporal` is
    ///   resident: skip a tick and the shader re-reads the PREVIOUS tick's values
    ///   without any error, which reads as a working-but-stale signal.
    /// - With `BODY_CHANNELS_ON` unset, calling this is harmless (the shader never
    ///   reads the slots) and skipping it is free. There is no need to gate the call.
    ///
    /// # Cost
    ///
    /// One `write_buffer` of `BODY_K * 4` = 32 bytes per being — 48 beings = 1 536
    /// bytes/tick, the same order as `update_proprioception_temporal`'s 48 bytes ×
    /// N. Kept as one write per being (instead of one big write) for the same reason
    /// as every sibling here: the slots of consecutive beings are not contiguous,
    /// they are `TEMPORAL_STRIDE` apart.
    pub fn update_body_channels(
        &self,
        queue: &wgpu::Queue,
        values: &[[f32; crate::perception::channels::BODY_K]],
    ) {
        let n = values.len().min(self.capacity);
        if n == 0 {
            return;
        }
        for i in 0..n {
            queue.write_buffer(
                &self.buf_temporal,
                body_slot_byte_offset(i),
                bytemuck::cast_slice(&values[i]),
            );
        }
    }

    /// Grava o dispatch da percepção num `CommandEncoder` externo SEM fazer submit nem poll.
    ///
    /// # Por que existe
    ///
    /// O método `reduce()` fazia seu próprio encoder + submit + poll(Wait), custando ~0,76 ms/tick
    /// (medido: encoder overhead + sincronização de fila). Ao gravar apenas no encoder fornecido
    /// pelo chamador (retina → percepção → único submit), eliminamos esse readback intermediário.
    ///
    /// O resultado fica em `self.buf_output` (VRAM) e em `self.buf_readback` (prontos para map),
    /// mas o map só pode acontecer DEPOIS do submit + poll do encoder que foi passado aqui.
    ///
    /// # Fluxo esperado pelo chamador (motor.rs)
    ///
    /// ```text
    /// 1. encoder criado externamente
    /// 2. retina.dispatch_into(enc, ...)          ← dispatch retina
    /// 3. percept.bind_and_dispatch(enc, ...)     ← dispatch percepção (depende do buf_output da retina)
    /// 4. enc.copy_buffer_to_buffer(buf_output → buf_readback)
    /// 5. queue.submit(enc.finish())              ← UM submit
    /// 6. device.poll(Wait)                       ← UM bloqueio
    /// 7. ler buf_readback                        ← resultado
    /// ```
    ///
    /// # Parâmetros
    ///
    /// - `enc`: encoder externo onde os comandos serão gravados.
    /// - `poses`: poses dos seres vivos neste tick.
    /// - `buf_retina_fisica`: buffer de saída da retina (VRAM, já preenchido pelo dispatch anterior).
    /// - `queue`: fila para `write_buffer` das poses (sem submit).
    ///
    /// Retorna o número de bytes do buf_output preenchidos (para o copy posterior).
    pub fn bind_and_dispatch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        poses: &[PosePerception],
        buf_retina_fisica: &wgpu::Buffer,
    ) -> u64 {
        let n = poses.len().min(self.capacity);
        if n == 0 {
            return 0;
        }

        // Atualiza poses via write_buffer (CPU→GPU, não bloqueia até submit).
        queue.write_buffer(&self.buf_poses, 0, bytemuck::cast_slice(&poses[..n]));

        self.encode_prepared(
            enc,
            device,
            n,
            buf_retina_fisica,
            BODY_CHANNELS_ON.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    pub fn poses_buffer(&self) -> &wgpu::Buffer {
        &self.buf_poses
    }
    pub fn temporal_buffer(&self) -> &wgpu::Buffer {
        &self.buf_temporal
    }

    /// Reuse spatial/temporal/motion kernels with poses/extras already resident.
    /// Scenario mode disables the legacy BODY block; taxi owns its explicit remap.
    pub fn encode_resident(
        &self,
        enc: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        n: usize,
        buf_retina_fisica: &wgpu::Buffer,
    ) -> u64 {
        self.encode_prepared(enc, device, n, buf_retina_fisica, false)
    }

    fn encode_prepared(
        &self,
        enc: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        n: usize,
        buf_retina_fisica: &wgpu::Buffer,
        legacy_body_on: bool,
    ) -> u64 {
        assert!(n <= self.capacity, "resident perception capacity");
        if n == 0 {
            return 0;
        }
        // Uniform criado por tick — 16 bytes, custo negligível.
        let params = ParamsGpu {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            perif_real: if PERIF_REAL.load(std::sync::atomic::Ordering::Relaxed) {
                1.0
            } else {
                0.0
            },
            // BODY BLOCK: read ONCE here, so the temporal and motion kernels of this
            // tick see the same value (see `BODY_CHANNELS_ON`).
            body_on: if legacy_body_on { 1.0 } else { 0.0 },
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("percep-params-chain"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // ── Bind group espacial (8 bindings: 0-6 + 7=buf_sensors, FASE SENSORES) ──
        let bg_spatial = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("percep-chain-bg-spatial"),
            layout: &self.layout_spatial,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_flow.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_heights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_poses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_retina_fisica.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buf_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buf_sensors.as_entire_binding(),
                },
            ],
        });

        // ── Bind group temporal (8 bindings, inclui estado temporal) ─────────
        let bg_temporal = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("percep-chain-bg-temporal"),
            layout: &self.layout_temporal,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_flow.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_heights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_poses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_retina_fisica.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buf_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buf_temporal.as_entire_binding(),
                },
            ],
        });

        let bytes_out = (n * WAVE_U32S * 4) as u64;

        // ZERAGEM OBRIGATÓRIA: ambos os kernels usam atomicOr; zero + OR = valor correto.
        enc.clear_buffer(&self.buf_output, 0, Some(bytes_out));

        // Dispatch espacial: canais 0-97, 128-191
        {
            let mut passe = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-percep-spatial-chain"),
                timestamp_writes: None,
            });
            passe.set_pipeline(&self.pipeline_spatial);
            passe.set_bind_group(0, &bg_spatial, &[]);
            passe.dispatch_workgroups(n as u32, 1, 1);
        }
        // Barreira implícita (fim do compute pass) — buf_output do spatial visível para temporal.

        // Dispatch temporal: canais 85, 86, 88, 102-110, 118-120, 123
        {
            let mut passe = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-percep-temporal-chain"),
                timestamp_writes: None,
            });
            passe.set_pipeline(&self.pipeline_temporal);
            passe.set_bind_group(0, &bg_temporal, &[]);
            // workgroup_size=1: 1 thread por ser, 60 workgroups em paralelo.
            passe.dispatch_workgroups(n as u32, 1, 1);
        }

        // ── Dispatch motion: canais 21-27 (motion_per_row, fluxo óptico) ──────
        // Requer: buf_retina_prev foi copiado via copy_retina_prev(enc, buf_retina_fisica)
        // ANTES do dispatch da retina neste encoder.
        // Weight L1 total ch21-27 = 91.1 no CHAMPION.dna — crítico para curvas.
        let bg_motion = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("percep-chain-bg-motion"),
            layout: &self.layout_motion,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_retina_fisica.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_retina_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_output.as_entire_binding(),
                },
            ],
        });
        {
            let mut passe = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-percep-motion-chain"),
                timestamp_writes: None,
            });
            passe.set_pipeline(&self.pipeline_motion);
            passe.set_bind_group(0, &bg_motion, &[]);
            // workgroup_size=11: one workgroup per being; rows 0-7 plus
            // hemispheric-left, hemispheric-right and foveal motion.
            passe.dispatch_workgroups(n as u32, 1, 1);
        }

        // Copia resultado para buf_readback no mesmo encoder (sem submit intermediário).
        enc.copy_buffer_to_buffer(&self.buf_output, 0, &self.buf_readback, 0, bytes_out);

        bytes_out
    }

    /// Copia `buf_retina_fisica` → `buf_retina_prev` no encoder externo, preservando a retina
    /// do tick atual como "anterior" para o próximo tick.
    ///
    /// # Quando chamar
    ///
    /// ANTES do dispatch da retina neste mesmo encoder. Assim:
    ///   1. `percept.copy_retina_prev(enc, retina.buf_output())`   ← preserva tick t-1
    ///   2. `retina.dispatch_into(enc, ...)`                       ← retina escreve tick t
    ///   3. `percept.bind_and_dispatch(enc, ...)`                  ← motion lê tick t vs t-1
    ///
    /// No tick 0, `buf_retina_prev` contém zeros (inicializado em new()), e a retina atual
    /// é o frame real. Motion = |frame_real - 0| = frame_real — ligeiramente elevado no tick 0
    /// mas convergirá no tick 1.
    pub fn copy_retina_prev(
        &self,
        enc: &mut wgpu::CommandEncoder,
        buf_retina_fisica: &wgpu::Buffer,
    ) {
        // Cópia GPU→GPU dentro do encoder: zero overhead de CPU, latência ≈ PCIe write latency.
        // Tamanho: capacity × RETINA_TOTAL × 4 bytes. Para 60 seres: 60×1764×4 = 423.360 bytes.
        // BW de cópia intra-VRAM da RX 6750 XT: ~200 GB/s → 423 KB leva ~2 μs — negligível.
        // Usa min(src.size, dst.size) para não ultrapassar nenhum dos dois buffers.
        // buf_retina_fisica: alocado pelo chamador para N seres reais (ex: 60×1764×4 = 423.360 B).
        // buf_retina_prev: alocado para capacity.max(64) seres (ex: 64×1764×4 = 451.584 B).
        // Copiamos apenas o que a fonte tem — excedente em dst fica como zero.
        let copy_bytes = buf_retina_fisica.size().min(self.retina_bytes);
        enc.copy_buffer_to_buffer(buf_retina_fisica, 0, &self.buf_retina_prev, 0, copy_bytes);
    }

    /// Lê o buf_readback após um poll(Wait) do encoder que continha bind_and_dispatch.
    ///
    /// Deve ser chamado APENAS depois que o CommandEncoder externo foi submetido e
    /// `device.poll(Wait)` retornou. Sem isso o buffer não estará mapeado e o método panickará.
    ///
    /// Retorna um vetor de [i8; SENSORY_WAVE_SIZE] com os canais portados pela GPU.
    /// Lê a onda sensorial de volta. EXIGE o `device` para servir o mapeamento.
    ///
    /// A versão anterior chamava `map_async` e ia direto para `recv()`, com o comentário
    /// "o caller já fez poll(Wait) no encoder externo". Não basta: `map_async` enfileira
    /// uma requisição NOVA, que precisa do próprio poll para o callback ser chamado. O
    /// `recv()` esperava para sempre — a função NUNCA funcionou, e por isso o
    /// `cortex_evolve` faz o map/poll/recv à mão em vez de chamá-la.
    ///
    /// É o mesmo deadlock já documentado duas vezes nesta base: **poll DEPOIS do
    /// map_async, nunca antes**.
    pub fn read_after_submit(
        &self,
        device: &wgpu::Device,
        n: usize,
        bytes_out: u64,
    ) -> Vec<[i8; SENSORY_WAVE_SIZE]> {
        let n = n.min(self.capacity);
        if n == 0 || bytes_out == 0 {
            return Vec::new();
        }

        let slice = self.buf_readback.slice(0..bytes_out);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // O poll QUE FALTAVA: sem ele o callback nunca é servido e o recv trava.
        device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);

        let mut out = vec![[0i8; SENSORY_WAVE_SIZE]; n];
        for being in 0..n {
            let base = being * WAVE_U32S;
            for ch in 0..SENSORY_WAVE_SIZE {
                let word = words[base + ch / 4];
                let byte = ((word >> ((ch % 4) * 8)) & 0xFF) as u8;
                out[being][ch] = byte as i8;
            }
        }

        drop(data);
        self.buf_readback.unmap();
        out
    }
}

// ─── BODY BLOCK — packing tests (no GPU) + GPU parity (#[ignore]) ──────────────
// A separate file because perception.rs is already past the 600-line cap of
// docs/STYLE.md 2; same pattern as `retina_digest.rs:345-348`.
#[cfg(test)]
#[path = "perception_body_test.rs"]
mod perception_body_test;

// ─── BODY BLOCK — packing and encoding (pure, testable without a GPU) ────────

/// Byte offset of being `i`'s body block inside `buf_temporal`.
///
/// Split out of `update_body_channels` so the packing can be tested WITHOUT a GPU:
/// the arithmetic is the whole risk here. An off-by-one in the base or a stride that
/// disagrees with the shader writes into the neighbouring slot with no error and no
/// panic — the exact failure mode `gpu/CLAUDE.md` records for `retina_digest`
/// ("wrote 16 extra words into `buf_output` … those writes landed on the *next*
/// being's channels 0-15").
#[inline]
pub fn body_slot_byte_offset(being: usize) -> u64 {
    (being * TEMPORAL_STRIDE + BODY_SLOT_BASE) as u64 * 4
}

/// Host-side twin of the shader's encoding for a SIGNED body value:
/// `i32(clamp(v * 127.0, -127.0, 127.0))`.
///
/// Not used by the production path (the shader does the conversion on the GPU) —
/// it exists as the oracle the parity test compares the GPU against, and as the
/// answer to "what will the brain actually see for this value?" without a GPU.
/// Note the range is [-127, 127], not [-128, 127]: same as every other signed
/// channel in the registry.
#[inline]
pub fn encode_body_signed(v: f32) -> i8 {
    (v * 127.0).clamp(-127.0, 127.0) as i8
}

/// Host-side twin of the shader's encoding for an UNSIGNED body value:
/// `i32(clamp(v * 127.0, 0.0, 127.0))`. See `encode_body_signed`.
#[inline]
pub fn encode_body_unsigned(v: f32) -> i8 {
    (v * 127.0).clamp(0.0, 127.0) as i8
}

// ─── Helpers de bind group layout (copiados de mod.rs para manter o módulo autônomo) ──

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ─── Equivalência CPU — como testar sem rodar nada ─────────────────────────────

/// # Como verificar equivalência sem executar a GPU
///
/// ## Teste 1: equivalência canal por canal (unit test offline)
///
/// Crie um `VoxelWorld` sintético (grade 10×10, flow zerado, alturas zeradas).
/// Coloque um ser em (15.0, 0.0, 15.0) apontando para (1.0, 0.0) com phase=0.
/// Execute a CPU (motor.rs, apenas os blocos de canal a ser testado) E o kernel
/// via `PercepcaoGpu::reduzir()`. Compare canal por canal.
///
/// Canais esperados equivalentes (tolerância ≤ 1 em i8):
/// - ch 16-23, 24-31, 32-39, 40-47, 48-51, 56-59, 60-63, 76-77, 78-80, 82, 90-97
///
/// Canais que DIVERGEM por design (e por quê):
/// - ch 0-15, 128-191: equivalente se `codificacao_legada()` for idêntica ao Rust —
///   verificar bit a bit com `codificacao_legada` em mod.rs.
/// - ch 82: GPU usa `1 − fwd_probe/30`; CPU usa `speed / fwd_probe`. Com speed=10m/s
///   e fwd_probe=15m: CPU = 10/15 × 127 ≈ 85, GPU = (1−15/30) × 127 ≈ 64. Diverge.
///   A CPU DEVE sobrescrever este canal.
///
/// ## Teste 2: roundtrip de empacotamento
///
/// ```ignore
/// // Preenche um buffer fake com valores conhecidos.
/// let wave_cpu = [42i8; 223];
/// // Simula o empacotamento que o WGSL faria.
/// let mut words = [0u32; 56];
/// for ch in 0..223 {
///     let byte_val = wave_cpu[ch] as u8 as u32;
///     words[ch / 4] |= byte_val << ((ch % 4) * 8);
/// }
/// // Simula a leitura em Rust.
/// for ch in 0..223 {
///     let word = words[ch / 4];
///     let byte = ((word >> ((ch % 4) * 8)) & 0xFF) as u8;
///     let val = byte as i8;
///     assert_eq!(val, wave_cpu[ch], "ch={}", ch);
/// }
/// ```
///
/// ## Teste 3: equivalência do `codificacao_legada` WGSL vs Rust
///
/// Para N=100 pares (dist_m, material) aleatórios, comparar `codificacao_legada()` em
/// mod.rs com a versão WGSL executada por um micro-kernel auxiliar. Tolerância: diferença
/// máxima 1 em i8 (pelas diferentes precisões de `sqrt`/`clamp` entre GPU e CPU).
///
/// ## Teste 4: equivalência end-to-end (integração — requer GPU)
///
/// Rodado pelo agente D (`gpu_valida` binary):
/// 1. Carrega pista real, popula N seres com posições conhecidas.
/// 2. Executa percepção CPU (blocos portados do motor.rs) em Rust puro.
/// 3. Executa `PercepcaoGpu::reduzir()`.
/// 4. Compara canal por canal, excluindo canais documentados como divergentes.
/// 5. Aceita ≤ 1 em |i8_cpu - i8_gpu| para canais equivalentes.
///
/// ## Teste de regressão de performance (requer 2440 amostras de PERF)
///
/// Mede tempo do laço de sensory_wave na CPU antes e depois da integração.
/// Ganho esperado: depende de quais canais foram portados. Os canais 0-15 e 128-191
/// custam O(RETINA_ROWS × RETINA_COLS) = O(1764) por bin — são os mais caros.
/// Estimativa conservadora: 40-60% dos canais portados → redução de 30-40% no tempo
/// de percepção (1.9% → ~1.1-1.3% do total). Medir, não estimar.

impl PerceptionGpu {
    /// Temporal resident state and previous-retina history for runtime checkpoints.
    pub fn snapshot_buffers(&self) -> Vec<crate::gpu::state_snapshot::NamedBuffer<'_>> {
        vec![
            ("perception.temporal", &self.buf_temporal),
            ("perception.retina_prev", &self.buf_retina_prev),
        ]
    }
}

// ─── Struct de exposição pública para integração ────────────────────────────────

/// Recurso Bevy análogo a `RetinaGpuRes` — mantém o contexto de percepção GPU entre ticks.
///
/// IMPORTANTE: o mesmo `RetinaGpu` que produz o buffer de retina física deve ser passado
/// aqui. Use `RetinaGpu::buf_output_raw()` (a ser adicionado pelo agente D no mod.rs)
/// para passar o buffer sem cópia.
#[derive(bevy::prelude::Resource, Default)]
pub struct PerceptionGpuRes {
    pub ctx: Option<PerceptionGpu>,
    /// Assinatura do mundo — detecta troca de pista e reinicia buffers.
    pub signature: u64,
}

impl PerceptionGpuRes {
    /// Retorna contexto válido ou None (sem GPU, cai na CPU).
    ///
    /// Precisa do `device` e `fila` da `RetinaGpu` para compartilhar o mesmo dispositivo.
    /// Não deve criar um segundo dispositivo — wgpu só suporta 1 por processo em muitos drivers.
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &crate::world::VoxelWorld,
        signature: u64,
        capacity: usize,
    ) -> Option<&PerceptionGpu> {
        let needs_update = self.ctx.is_none()
            || self.signature != signature
            || self
                .ctx
                .as_ref()
                .map(|c| c.capacity < capacity)
                .unwrap_or(true);
        if needs_update {
            self.ctx = PerceptionGpu::new(device, queue, world, capacity.max(64));
            self.signature = signature;
            if self.ctx.is_some() {
                println!(
                    "🔭 percepção na GPU: {} seres (kernel percepcao.wgsl)",
                    capacity
                );
            } else {
                println!("⚠️  percepção GPU falhou — canais ficam na CPU");
            }
        }
        self.ctx.as_ref()
    }
}

/// Converte `bevy::prelude::Vec3` + forward/right (Vec2) + phase para `PosePercep`.
///
/// Chamado no motor.rs antes do dispatch para montar o slice de poses em um único passo.
/// `forward_2d` = Vec2 { x: transform.right().x, y: transform.right().z } (motor.rs usa .right() como forward).
/// `right_2d`   = Vec2 { x: transform.back().x,  y: transform.back().z }.
pub fn pose_from_transform(
    pos: bevy::math::Vec3,
    forward_2d: Vec2,
    right_2d: Vec2,
    phase: f32,
) -> PosePerception {
    PosePerception {
        pos_x: pos.x,
        pos_y: pos.y,
        pos_z: pos.z,
        phase,
        fwd_x: forward_2d.x,
        fwd_z: forward_2d.y,
        rgt_x: right_2d.x,
        rgt_z: right_2d.y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_reset_uses_the_same_neutral_state_for_every_being() {
        let state = initial_temporal_state(2);
        assert_eq!(state.len(), 2 * TEMPORAL_STRIDE);
        for being in 0..2 {
            let base = being * TEMPORAL_STRIDE;
            assert_eq!(&state[base..base + 5], &[40.0, 40.0, 50.0, 40.0, 40.0]);
            assert_eq!(&state[base + 16..base + 19], &[40.0, 40.0, 50.0]);
            assert_eq!(state[base + 20], 1.0);
            assert_eq!(state[base + 23], 1.0);
            assert_eq!(state[base + 25], 1.0);
            assert_eq!(state[base + 26], 0.5);
        }
    }

    /// Testa o roundtrip de empacotamento i8 ↔ u32 ↔ i8 sem precisar de GPU.
    /// Garante que o desempacotador em Rust inverte o que o WGSL escreve.
    #[test]
    fn roundtrip_packing_i8_u32() {
        // Simula o empacotamento que escreve_canal() faria no WGSL.
        let wave_cpu: Vec<i8> = (0..SENSORY_WAVE_SIZE as i32)
            .map(|i| ((i * 13 + 7) % 255 - 127) as i8)
            .collect();

        let mut words = [0u32; WAVE_U32S];
        for ch in 0..SENSORY_WAVE_SIZE {
            let byte_val = wave_cpu[ch] as u8 as u32;
            let shift = (ch % 4) * 8;
            words[ch / 4] |= byte_val << shift;
        }

        // Simula a leitura em Rust (igual ao loop em reduzir()).
        for ch in 0..SENSORY_WAVE_SIZE {
            let word = words[ch / 4];
            let byte = ((word >> ((ch % 4) * 8)) & 0xFF) as u8;
            let val = byte as i8;
            assert_eq!(val, wave_cpu[ch], "roundtrip falhou no canal {}", ch);
        }
    }

    /// Testa a equivalência de `codificacao_legada` com valores de borda.
    /// O WGSL replica esta função — se os valores baterem aqui, a GPU terá o mesmo resultado
    /// dentro de precisão float32.
    #[test]
    fn legacy_encoding_edge_values() {
        use crate::gpu::Pixel;
        // Wall a 0m → 127
        let v = crate::gpu::legacy_encoding(Pixel {
            dist_m: 0.0,
            material: 2,
        });
        assert_eq!(v, 127i8);
        // Road a 0m → -128
        let v = crate::gpu::legacy_encoding(Pixel {
            dist_m: 0.0,
            material: 1,
        });
        assert_eq!(v, -128i8);
        // Grass a 0m → 40
        let v = crate::gpu::legacy_encoding(Pixel {
            dist_m: 0.0,
            material: 0,
        });
        assert_eq!(v, 40i8);
        // Wall a 30m → 55 (limit point entre as duas curvas de Wall)
        let v = crate::gpu::legacy_encoding(Pixel {
            dist_m: 30.0,
            material: 2,
        });
        assert_eq!(v, 55i8, "Wall a 30m deveria ser 55 (limite das curvas)");
        // Wall a 120m → 0
        let v = crate::gpu::legacy_encoding(Pixel {
            dist_m: 120.0,
            material: 2,
        });
        assert_eq!(v, 0i8, "Wall a 120m deveria ser 0");
    }

    /// Verifica que WAVE_U32S cobre todos os canais sem overflow.
    #[test]
    fn wave_u32s_cobre_todos_os_canais() {
        // SENSORY_WAVE_SIZE = 192 (onda atual, ver perception/channels.rs). O "223"
        // do comentário antigo era a onda legada (cortex_sheet N_INPUTS=223) — mentiroso
        // aqui, pois este WAVE_U32S é derivado de SENSORY_WAVE_SIZE=192, não 223.
        assert_eq!(WAVE_U32S, 48, "48 × 4 = 192 bytes = 192 canais, sem folga");
        assert!(WAVE_U32S * 4 >= SENSORY_WAVE_SIZE);
    }

    /// Verifica que PosePercep tem o tamanho e alinhamento esperados.
    #[test]
    fn pose_perception_layout() {
        assert_eq!(
            std::mem::size_of::<PosePerception>(),
            32,
            "8 campos f32 = 32 bytes, bytemuck exige alinhamento correto"
        );
        assert_eq!(std::mem::align_of::<PosePerception>(), 4);
    }
}

#[cfg(test)]
#[path="perception_fuel_test.rs"]
mod perception_fuel_test;
