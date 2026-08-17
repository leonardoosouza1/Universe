use std::fs::File;
/// Sistema de replay quadro a quadro — formato único para motor.rs e cortex_evolve.
///
/// # Formato binário `.replay`
///
/// Cabeçalho fixo 64 bytes:
///   magic: [u8; 8] = b"IARARPLY"
///   version: u16 = 1
///   source: u8   = 0 (motor.rs) ou 1 (cortex_evolve)
///   n_beings: u8
///   tick_rate: f32  (Hz nominal, ex: 64.0)
///   world_seed: u64
///   _pad: [u8; 42]
///
/// A seguir, frames intercalados:
///   frame_tag: u8 = 0xFE  (início de frame)
///   being_idx: u8
///   ReplayFrame serializado binário (bytemuck::Pod)
///
/// Escritor garante SENDO_IDX 0 = campeão.
///
/// # Leitura
///
/// Carregar via `ReplayReader::open` → iterator de `(being_idx, ReplayFrame)`.
///
/// # Convenção de canais neural_summary
///
/// [0] throttle_raw     — saída crua do cortex antes de axiom/governor
/// [1] steer_raw        — idem
/// [2] brake_raw        — idem
/// [3] throttle_final   — comando aplicado na física
/// [4] steer_final      — idem
/// [5] brake_final      — idem
/// [6] governor_delta   — |throttle_raw - throttle_final| (quanto o governor mudou)
/// [7] sentinel_active  — 0.0 ou 1.0 (se sentinel hard-vetou neste frame)
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

// ── Constantes do formato ────────────────────────────────────────────────────

pub const REPLAY_MAGIC: &[u8; 8] = b"IARARPLY";
pub const REPLAY_VERSION: u16 = 1;
pub const SOURCE_MOTOR: u8 = 0;
pub const SOURCE_CORTEX: u8 = 1;
pub const FRAME_TAG: u8 = 0xFE;

// ── Frame principal ──────────────────────────────────────────────────────────

/// Um frame de replay — 128 bytes, alinhado e Pod.
///
/// Gravado por CADA ser em CADA tick onde a análise interessa.
/// Para corridas longas (24000 ticks × 40 seres) = 960.000 frames ≈ 123 MB.
/// Filtragem opcional: gravar só campeão (IDX=0) reduz para 3 MB.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ReplayFrame {
    // --- Estado físico ---
    /// Posição mundo (x, y, z) em metros
    pub pos: [f32; 3],
    /// Orientação (forward.x, forward.z) — 2D pois y sempre ~0
    pub fwd: [f32; 2],
    /// Velocidade escalar em m/s
    pub speed: f32,
    /// Taxa de rotação yaw em rad/s
    pub yaw_rate: f32,
    /// Progresso na pista em metros (arc-length)
    pub arc_prog: f32,
    /// Número de voltas completas
    pub laps: u16,
    /// Marcha atual (0=neutro/ré, 1-6=marchas)
    pub gear: u8,
    /// HP normalizado [0..127] onde 127 = HP_INICIAL
    pub hp_norm: u8,
    /// Combustível normalizado [0..127]
    pub fuel_norm: u8,
    /// Tick absoluto desde início da geração
    pub tick: u32,

    // --- Sensores de visão (distâncias em m, codificadas) ---
    /// Distância à parede esquerda em metros (cap 15m)
    pub dist_wall_l: f32,
    /// Distância à parede direita em metros
    pub dist_wall_r: f32,
    /// Distância à parede frontal em metros (cap 30m)
    pub dist_wall_f: f32,
    /// Curvatura 40m à frente — signed [-1..1] (positivo = curva direita)
    pub curv_40m: f32,
    /// Curvatura 70m à frente
    pub curv_70m: f32,
    /// Curvatura 120m à frente
    pub curv_120m: f32,
    /// Alinhamento com o fluxo da pista [-1..1] (1 = perfeitamente alinhado)
    pub flow_align: f32,
    /// Offset lateral da racing-line em metros (positivo = à esquerda da linha)
    pub line_offset: f32,

    // --- Intent do cérebro (cru, antes de governor/sentinel) ---
    /// Throttle bruto [-1..1]
    pub intent_thr: f32,
    /// Steer bruto [-1..1]
    pub intent_str: f32,
    /// Brake bruto [0..1]
    pub intent_brk: f32,

    // --- Comando final aplicado na física ---
    /// Throttle final (após KAN, governor, sentinel, ECAS)
    pub final_thr: f32,
    /// Steer final
    pub final_str: f32,
    /// Brake final
    pub final_brk: f32,

    // --- Diagnóstico neural compacto ---
    /// Número de dimensões saturadas (|x|>0.95) no final_intent (0..96)
    pub dims_saturated: u8,
    /// Magnitude média do final_intent (quantizado 0..127)
    pub intent_mag_q: u8,
    /// Axiom strength neste tick quantizada [0..127]
    pub axiom_str_q: u8,
    /// Flags de safety: bit0=governor_ativo, bit1=sentinel_hard, bit2=ecas, bit3=reversing
    pub safety_flags: u8,

    // --- Retina compacta (8 fóvea + 8 periférica, i8) ---
    pub retina_fov: [i8; 8],
    pub retina_peri: [i8; 8],

    // --- Padding para 128 bytes ---
    pub _pad: [u8; 8],
}

// Garante tamanho fixo — falha de compilação se mudar sem ajustar.
const _: () = assert!(std::mem::size_of::<ReplayFrame>() == 128);

// SAFETY: ReplayFrame é #[repr(C)] com apenas tipos primitivos, sem padding implícito.
unsafe impl bytemuck::Pod for ReplayFrame {}
unsafe impl bytemuck::Zeroable for ReplayFrame {}

// ── Cabeçalho ────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReplayHeader {
    pub magic: [u8; 8],
    pub version: u16,
    pub source: u8,
    pub n_beings: u8,
    pub tick_rate: f32,
    pub world_seed: u64,
    pub _pad: [u8; 40],
}

const _: () = assert!(std::mem::size_of::<ReplayHeader>() == 64);

unsafe impl bytemuck::Pod for ReplayHeader {}
unsafe impl bytemuck::Zeroable for ReplayHeader {}

// ── Escritor ─────────────────────────────────────────────────────────────────

pub struct ReplayWriter {
    writer: BufWriter<File>,
    pub frames_written: u64,
}

impl ReplayWriter {
    /// Abre arquivo e escreve cabeçalho.
    pub fn create(
        path: &Path,
        source: u8,
        n_beings: u8,
        tick_rate: f32,
        world_seed: u64,
    ) -> io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let hdr = ReplayHeader {
            magic: *REPLAY_MAGIC,
            version: REPLAY_VERSION,
            source,
            n_beings,
            tick_rate,
            world_seed,
            _pad: [0u8; 40],
        };
        writer.write_all(bytemuck::bytes_of(&hdr))?;

        Ok(Self {
            writer,
            frames_written: 0,
        })
    }

    /// Escreve um frame de um ser específico.
    pub fn write_frame(&mut self, being_idx: u8, frame: &ReplayFrame) -> io::Result<()> {
        self.writer.write_all(&[FRAME_TAG, being_idx])?;
        self.writer.write_all(bytemuck::bytes_of(frame))?;
        self.frames_written += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

// ── Leitor ───────────────────────────────────────────────────────────────────

pub struct ReplayReader {
    data: Vec<u8>,
    cursor: usize,
    pub header: ReplayHeader,
}

impl ReplayReader {
    /// Carrega arquivo inteiro em memória (arquivos tipicamente < 200 MB).
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        if data.len() < 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "arquivo muito curto",
            ));
        }

        let header_bytes: [u8; 64] = data[..64].try_into().unwrap();
        // Desalinhado pelo mesmo motivo do quadro: o array local não tem garantia de
        // alinhamento do tipo alvo, e depender disso é uma bomba-relógio de plataforma.
        let header: ReplayHeader = bytemuck::pod_read_unaligned(&header_bytes);

        if &header.magic != REPLAY_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "magic inválido — não é replay IARA",
            ));
        }
        if header.version != REPLAY_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "versão {} não suportada (esperado {})",
                    header.version, REPLAY_VERSION
                ),
            ));
        }

        Ok(Self {
            data,
            cursor: 64,
            header,
        })
    }

    /// Retorna próximo (being_idx, ReplayFrame) ou None se EOF.
    pub fn next_frame(&mut self) -> Option<(u8, ReplayFrame)> {
        const FRAME_SIZE: usize = 128;
        const ENTRY_SIZE: usize = 2 + FRAME_SIZE; // tag + idx + frame

        if self.cursor + ENTRY_SIZE > self.data.len() {
            return None;
        }

        let tag = self.data[self.cursor];
        if tag != FRAME_TAG {
            // Corrupção: skip bytes até achar tag ou EOF
            eprintln!(
                "REPLAY: tag inválida 0x{:02x} em offset {} — buscando próxima",
                tag, self.cursor
            );
            while self.cursor < self.data.len() && self.data[self.cursor] != FRAME_TAG {
                self.cursor += 1;
            }
            return self.next_frame();
        }

        let being_idx = self.data[self.cursor + 1];
        let frame_bytes: &[u8; FRAME_SIZE] = self.data
            [self.cursor + 2..self.cursor + 2 + FRAME_SIZE]
            .try_into()
            .ok()?;
        // Leitura DESALINHADA obrigatória: cada entrada ocupa 130 bytes (tag + índice +
        // quadro de 128) e o quadro começa em `cursor + 2`, então seu deslocamento é
        // 64 + k·130 + 2 — nunca múltiplo de 4. `bytemuck::from_bytes` exige alinhamento
        // do tipo alvo e entra em pânico já no primeiro quadro; era isso que impedia
        // qualquer replay de ser lido.
        let frame: ReplayFrame = bytemuck::pod_read_unaligned(frame_bytes);

        self.cursor += ENTRY_SIZE;
        Some((being_idx, frame))
    }

    /// Coleta todos os frames de um being específico em ordem.
    pub fn frames_of(&mut self, target_idx: u8) -> Vec<ReplayFrame> {
        self.cursor = 64; // reset para início dos dados
        let mut out = Vec::new();
        while let Some((idx, frame)) = self.next_frame() {
            if idx == target_idx {
                out.push(frame);
            }
        }
        out
    }

    /// Retorna todos os frames de todos os beings agrupados por idx.
    pub fn all_frames(&mut self) -> Vec<Vec<ReplayFrame>> {
        let n = self.header.n_beings as usize;
        let mut out: Vec<Vec<ReplayFrame>> = vec![Vec::new(); n.max(1)];
        self.cursor = 64;
        while let Some((idx, frame)) = self.next_frame() {
            let i = idx as usize;
            if i < out.len() {
                out[i].push(frame);
            }
        }
        out
    }
}

// ── Análise quadro a quadro ──────────────────────────────────────────────────

/// Perfil de uma seção da pista (dividida em N arcos iguais).
pub struct SectionProfile {
    pub arc_start: f32,
    pub arc_end: f32,
    pub avg_speed: f32,
    pub max_speed: f32,
    pub min_speed: f32,
    /// Tempo médio gasto nessa seção (estimado como arc_len / avg_speed)
    pub time_est: f32,
    /// Frames onde governor interveio (|thr_raw - thr_final| > 0.05)
    pub gov_count: u32,
    pub total_count: u32,
}

/// Analisa um conjunto de frames e produz perfil por seção.
///
/// `n_sections`: quantas fatias iguais dividir o arco total.
/// Retorna None se frames vazio.
pub fn section_profile(frames: &[ReplayFrame], n_sections: usize) -> Option<Vec<SectionProfile>> {
    if frames.is_empty() || n_sections == 0 {
        return None;
    }

    let arc_max = frames
        .iter()
        .map(|f| f.arc_prog)
        .fold(f32::NEG_INFINITY, f32::max);
    let arc_min = frames
        .iter()
        .map(|f| f.arc_prog)
        .fold(f32::INFINITY, f32::min);
    if arc_max <= arc_min {
        return None;
    }

    let section_len = (arc_max - arc_min) / n_sections as f32;
    let mut profiles: Vec<SectionProfile> = (0..n_sections)
        .map(|i| SectionProfile {
            arc_start: arc_min + i as f32 * section_len,
            arc_end: arc_min + (i + 1) as f32 * section_len,
            avg_speed: 0.0,
            max_speed: 0.0,
            min_speed: f32::MAX,
            time_est: 0.0,
            gov_count: 0,
            total_count: 0,
        })
        .collect();

    for f in frames {
        let idx = (((f.arc_prog - arc_min) / section_len) as usize).min(n_sections - 1);
        let p = &mut profiles[idx];
        p.avg_speed += f.speed;
        p.max_speed = p.max_speed.max(f.speed);
        p.min_speed = p.min_speed.min(f.speed);
        p.total_count += 1;

        // Governor interveio se throttle final for significativamente menor que raw
        if (f.intent_thr - f.final_thr).abs() > 0.05 {
            p.gov_count += 1;
        }
    }

    for p in &mut profiles {
        if p.total_count > 0 {
            p.avg_speed /= p.total_count as f32;
            p.time_est = section_len / p.avg_speed.max(0.5);
        }
        if p.min_speed == f32::MAX {
            p.min_speed = 0.0;
        }
    }

    Some(profiles)
}

/// Evento crítico detectado na análise.
#[derive(Debug)]
pub struct CriticalEvent {
    pub tick: u32,
    pub arc: f32,
    pub kind: EventKind,
    pub speed: f32,
    pub intent_str: f32,
    pub final_str: f32,
}

#[derive(Debug)]
pub enum EventKind {
    /// Governor cortou throttle (delta > limiar)
    GovernorCut { thr_before: f32, thr_after: f32 },
    /// Sentinel hard-veto ativo
    SentinelVeto,
    /// ECAS interveio (safety_flags bit2)
    EcasActive,
    /// Carro em ré (safety_flags bit3)
    Reversing,
    /// Velocidade caiu > 5 m/s em 10 frames (perda brusca)
    SpeedDrop { delta: f32 },
    /// Saturação neural alta (>20 dims saturadas)
    NeuralSaturation { dims: u8 },
}

/// Extrai todos os eventos críticos de uma sequência de frames.
pub fn detect_events(frames: &[ReplayFrame]) -> Vec<CriticalEvent> {
    let mut events = Vec::new();

    for (i, f) in frames.iter().enumerate() {
        // Governor cut
        let gov_delta = f.intent_thr - f.final_thr;
        if gov_delta > 0.10 {
            events.push(CriticalEvent {
                tick: f.tick,
                arc: f.arc_prog,
                kind: EventKind::GovernorCut {
                    thr_before: f.intent_thr,
                    thr_after: f.final_thr,
                },
                speed: f.speed,
                intent_str: f.intent_str,
                final_str: f.final_str,
            });
        }

        // Sentinel hard veto (bit1)
        if f.safety_flags & 0x02 != 0 {
            events.push(CriticalEvent {
                tick: f.tick,
                arc: f.arc_prog,
                kind: EventKind::SentinelVeto,
                speed: f.speed,
                intent_str: f.intent_str,
                final_str: f.final_str,
            });
        }

        // ECAS (bit2)
        if f.safety_flags & 0x04 != 0 {
            events.push(CriticalEvent {
                tick: f.tick,
                arc: f.arc_prog,
                kind: EventKind::EcasActive,
                speed: f.speed,
                intent_str: f.intent_str,
                final_str: f.final_str,
            });
        }

        // Reversing (bit3)
        if f.safety_flags & 0x08 != 0 {
            events.push(CriticalEvent {
                tick: f.tick,
                arc: f.arc_prog,
                kind: EventKind::Reversing,
                speed: f.speed,
                intent_str: f.intent_str,
                final_str: f.final_str,
            });
        }

        // Speed drop: compara com frame de 10 atrás
        if i >= 10 {
            let prev = &frames[i - 10];
            let drop = prev.speed - f.speed;
            if drop > 5.0 {
                events.push(CriticalEvent {
                    tick: f.tick,
                    arc: f.arc_prog,
                    kind: EventKind::SpeedDrop { delta: drop },
                    speed: f.speed,
                    intent_str: f.intent_str,
                    final_str: f.final_str,
                });
            }
        }

        // Saturação neural
        if f.dims_saturated > 20 {
            events.push(CriticalEvent {
                tick: f.tick,
                arc: f.arc_prog,
                kind: EventKind::NeuralSaturation {
                    dims: f.dims_saturated,
                },
                speed: f.speed,
                intent_str: f.intent_str,
                final_str: f.final_str,
            });
        }
    }

    events
}

/// Diff entre dois replays (caminho motor.rs vs caminho cortex_evolve).
pub struct ReplayDiff {
    pub arc: f32,
    /// Velocidade no caminho A (motor.rs)
    pub speed_a: f32,
    /// Velocidade no caminho B (cortex_evolve)
    pub speed_b: f32,
    /// Diferença de steer final (A - B)
    pub steer_diff: f32,
    /// Diferença de throttle final (A - B)
    pub thr_diff: f32,
    /// Posição A - posição B (distância entre trajetórias)
    pub pos_diverge: f32,
}

/// Compara dois conjuntos de frames (mesma pista, seres diferentes ou caminhos diferentes).
/// Alinha por arc_prog — interpola frame B para mesmo arco que frame A.
pub fn diff_replays(frames_a: &[ReplayFrame], frames_b: &[ReplayFrame]) -> Vec<ReplayDiff> {
    if frames_a.is_empty() || frames_b.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut bi = 0usize;

    for fa in frames_a {
        // Avança bi até o frame de B mais próximo em arc_prog
        while bi + 1 < frames_b.len()
            && (frames_b[bi + 1].arc_prog - fa.arc_prog).abs()
                < (frames_b[bi].arc_prog - fa.arc_prog).abs()
        {
            bi += 1;
        }
        let fb = &frames_b[bi];

        let pos_a = glam_like_dist(fa.pos, fb.pos);

        out.push(ReplayDiff {
            arc: fa.arc_prog,
            speed_a: fa.speed,
            speed_b: fb.speed,
            steer_diff: fa.final_str - fb.final_str,
            thr_diff: fa.final_thr - fb.final_thr,
            pos_diverge: pos_a,
        });
    }

    out
}

fn glam_like_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ── Gerador de HTML player ───────────────────────────────────────────────────

/// Template HTML embutido em tempo de compilação (src/assets/replay_player.html).
/// Contém toda a lógica de visualização; os dados são injetados como window.REPLAY_*.
static HTML_TEMPLATE: &str = include_str!("assets/replay_player.html");

/// Constrói HTML autocontido com player frame-a-frame.
///
/// `track_path`: polilinha da pista em coordenadas de mundo [(x, z)].
/// `being_frames`: Vec de sequências de frames por ser (índice 0 = campeão).
/// `labels`: nomes opcionais para cada ser.
///
/// Injeta `window.REPLAY_TRACK` e `window.REPLAY_BEINGS` antes do `<script>` do template.
pub fn generate_html_player(
    track_path: &[[f32; 2]],
    being_frames: &[Vec<ReplayFrame>],
    labels: &[&str],
    output_path: &Path,
) -> io::Result<()> {
    let file = File::create(output_path)?;
    let mut w = BufWriter::new(file);

    // --- Calcula bounds da pista para normalização ---
    let (mut min_x, mut max_x, mut min_z, mut max_z) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for &[x, z] in track_path {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }
    // Inclui posições de todos os seres nas bounds
    for frames in being_frames {
        for f in frames {
            min_x = min_x.min(f.pos[0]);
            max_x = max_x.max(f.pos[0]);
            min_z = min_z.min(f.pos[2]);
            max_z = max_z.max(f.pos[2]);
        }
    }

    let range_x = (max_x - min_x).max(1.0);
    let range_z = (max_z - min_z).max(1.0);

    // --- Serializa dados para JSON embutido no HTML ---
    // track_points: array de [px, py] normalizados para [0..1000, 0..1000]
    let mut track_json = String::from("[");
    for (i, &[x, z]) in track_path.iter().enumerate() {
        let px = ((x - min_x) / range_x * 980.0 + 10.0) as i32;
        let py = ((z - min_z) / range_z * 980.0 + 10.0) as i32;
        if i > 0 {
            track_json.push(',');
        }
        track_json.push_str(&format!("[{},{}]", px, py));
    }
    track_json.push(']');

    // beings_data: array de objetos, cada um com array de frames
    let mut beings_json = String::from("[");
    for (bi, frames) in being_frames.iter().enumerate() {
        if bi > 0 {
            beings_json.push(',');
        }
        let label = labels.get(bi).copied().unwrap_or("?");
        beings_json.push_str(&format!("{{\"label\":\"{}\",\"frames\":[", label));

        for (fi, f) in frames.iter().enumerate() {
            if fi > 0 {
                beings_json.push(',');
            }
            let px = ((f.pos[0] - min_x) / range_x * 980.0 + 10.0) as i32;
            let py = ((f.pos[2] - min_z) / range_z * 980.0 + 10.0) as i32;
            let fwd_px = (f.fwd[0] * 15.0) as i32;
            let fwd_py = (f.fwd[1] * 15.0) as i32;
            beings_json.push_str(&format!(
                "{{\"t\":{},\"x\":{},\"y\":{},\"fx\":{},\"fy\":{},\
                 \"spd\":{:.1},\"thr\":{:.2},\"str\":{:.2},\"brk\":{:.2},\
                 \"ithr\":{:.2},\"istr\":{:.2},\"ibrk\":{:.2},\
                 \"cl\":{:.1},\"cr\":{:.1},\"cf\":{:.1},\
                 \"c40\":{:.2},\"c70\":{:.2},\"c120\":{:.2},\
                 \"fl\":{:.2},\"lo\":{:.2},\
                 \"ds\":{},\"sf\":{},\"laps\":{},\"arc\":{:.1}}}",
                f.tick,
                px,
                py,
                fwd_px,
                fwd_py,
                f.speed,
                f.final_thr,
                f.final_str,
                f.final_brk,
                f.intent_thr,
                f.intent_str,
                f.intent_brk,
                f.dist_wall_l,
                f.dist_wall_r,
                f.dist_wall_f,
                f.curv_40m,
                f.curv_70m,
                f.curv_120m,
                f.flow_align,
                f.line_offset,
                f.dims_saturated,
                f.safety_flags,
                f.laps,
                f.arc_prog,
            ));
        }
        beings_json.push_str("]}");
    }
    beings_json.push(']');

    // --- Injetar dados no template HTML ---
    // O template espera window.REPLAY_TRACK e window.REPLAY_BEINGS
    // definidos ANTES do seu <script>. Inserimos um bloco <script> com os
    // dados logo antes do </head> do template.
    let data_script = format!(
        "<script>\nwindow.REPLAY_TRACK = {};\nwindow.REPLAY_BEINGS = {};\n</script>\n",
        track_json, beings_json
    );

    // Insere o bloco de dados antes de </head>
    let output_html = HTML_TEMPLATE.replacen("</head>", &format!("{}</head>", data_script), 1);

    w.write_all(output_html.as_bytes())?;
    w.flush()?;
    Ok(())
}
