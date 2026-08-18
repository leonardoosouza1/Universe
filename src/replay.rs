use std::fs::File;
/// Frame-by-frame replay system — single format for motor.rs and cortex_evolve.
///
/// # `.replay` binary format
///
/// Fixed 64-byte header:
///   magic: [u8; 8] = b"IARARPLY"
///   version: u16 = 1
///   source: u8   = 0 (motor.rs) or 1 (cortex_evolve)
///   n_beings: u8
///   tick_rate: f32  (nominal Hz, e.g. 64.0)
///   world_seed: u64
///   _pad: [u8; 42]
///
/// Then, interleaved frames:
///   frame_tag: u8 = 0xFE  (start of frame)
///   being_idx: u8
///   ReplayFrame serialized binary (bytemuck::Pod)
///
/// The writer guarantees BEING_IDX 0 = champion.
///
/// # Reading
///
/// Load via `ReplayReader::open` → iterator of `(being_idx, ReplayFrame)`.
///
/// # neural_summary channel convention
///
/// [0] throttle_raw     — raw cortex output before axiom/governor
/// [1] steer_raw        — idem
/// [2] brake_raw        — idem
/// [3] throttle_final   — command applied to the physics
/// [4] steer_final      — idem
/// [5] brake_final      — idem
/// [6] governor_delta   — |throttle_raw - throttle_final| (how much the governor changed)
/// [7] sentinel_active  — 0.0 or 1.0 (whether the sentinel hard-vetoed this frame)
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

// ── Format constants ─────────────────────────────────────────────────────────

pub const REPLAY_MAGIC: &[u8; 8] = b"IARARPLY";
pub const REPLAY_VERSION: u16 = 1;
pub const SOURCE_MOTOR: u8 = 0;
pub const SOURCE_CORTEX: u8 = 1;
pub const FRAME_TAG: u8 = 0xFE;

// ── Main frame ───────────────────────────────────────────────────────────────

/// One replay frame — 128 bytes, aligned and Pod.
///
/// Recorded by EACH being at EACH tick the analysis cares about.
/// For long races (24000 ticks × 40 beings) = 960.000 frames ≈ 123 MB.
/// Optional filtering: recording only the champion (IDX=0) reduces it to 3 MB.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ReplayFrame {
    // --- Physical state ---
    /// World position (x, y, z) in meters
    pub pos: [f32; 3],
    /// Orientation (forward.x, forward.z) — 2D since y is always ~0
    pub fwd: [f32; 2],
    /// Scalar speed in m/s
    pub speed: f32,
    /// Yaw rotation rate in rad/s
    pub yaw_rate: f32,
    /// Progress along the track in meters (arc-length)
    pub arc_prog: f32,
    /// Number of completed laps
    pub laps: u16,
    /// Current gear (0=neutral/reverse, 1-6=gears)
    pub gear: u8,
    /// Normalized HP [0..127] where 127 = HP_INICIAL
    pub hp_norm: u8,
    /// Normalized fuel [0..127]
    pub fuel_norm: u8,
    /// Absolute tick since the start of the generation
    pub tick: u32,

    // --- Vision sensors (distances in m, encoded) ---
    /// Distance to the left wall in meters (cap 15m)
    pub dist_wall_l: f32,
    /// Distance to the right wall in meters
    pub dist_wall_r: f32,
    /// Distance to the frontal wall in meters (cap 30m)
    pub dist_wall_f: f32,
    /// Curvature 40m ahead — signed [-1..1] (positive = right turn)
    pub curv_40m: f32,
    /// Curvature 70m ahead
    pub curv_70m: f32,
    /// Curvature 120m ahead
    pub curv_120m: f32,
    /// Alignment with the track flow [-1..1] (1 = perfectly aligned)
    pub flow_align: f32,
    /// Lateral offset from the racing line in meters (positive = left of the line)
    pub line_offset: f32,

    // --- Brain intent (raw, before governor/sentinel) ---
    /// Raw throttle [-1..1]
    pub intent_thr: f32,
    /// Raw steer [-1..1]
    pub intent_str: f32,
    /// Raw brake [0..1]
    pub intent_brk: f32,

    // --- Final command applied to the physics ---
    /// Final throttle (after KAN, governor, sentinel, ECAS)
    pub final_thr: f32,
    /// Final steer
    pub final_str: f32,
    /// Final brake
    pub final_brk: f32,

    // --- Compact neural diagnostics ---
    /// Number of saturated dimensions (|x|>0.95) in final_intent (0..96)
    pub dims_saturated: u8,
    /// Mean magnitude of final_intent (quantized 0..127)
    pub intent_mag_q: u8,
    /// Axiom strength this tick, quantized [0..127]
    pub axiom_str_q: u8,
    /// Safety flags: bit0=governor_active, bit1=sentinel_hard, bit2=ecas, bit3=reversing
    pub safety_flags: u8,

    // --- Compact retina (8 fovea + 8 peripheral, i8) ---
    pub retina_fov: [i8; 8],
    pub retina_peri: [i8; 8],

    // --- Padding to 128 bytes ---
    pub _pad: [u8; 8],
}

// Guarantees a fixed size — compilation failure if it changes without adjustment.
const _: () = assert!(std::mem::size_of::<ReplayFrame>() == 128);

// SAFETY: ReplayFrame is #[repr(C)] with only primitive types, no implicit padding.
unsafe impl bytemuck::Pod for ReplayFrame {}
unsafe impl bytemuck::Zeroable for ReplayFrame {}

// ── Header ───────────────────────────────────────────────────────────────────

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

// ── Writer ───────────────────────────────────────────────────────────────────

pub struct ReplayWriter {
    writer: BufWriter<File>,
    pub frames_written: u64,
}

impl ReplayWriter {
    /// Opens the file and writes the header.
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

    /// Writes one frame of a specific being.
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

// ── Reader ───────────────────────────────────────────────────────────────────

pub struct ReplayReader {
    data: Vec<u8>,
    cursor: usize,
    pub header: ReplayHeader,
}

impl ReplayReader {
    /// Loads the whole file into memory (files typically < 200 MB).
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        if data.len() < 64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too short"));
        }

        let header_bytes: [u8; 64] = data[..64].try_into().unwrap();
        // Unaligned for the same reason as the frame: the local array has no alignment
        // guarantee for the target type, and depending on it is a platform time bomb.
        let header: ReplayHeader = bytemuck::pod_read_unaligned(&header_bytes);

        if &header.magic != REPLAY_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid magic — not an IARA replay",
            ));
        }
        if header.version != REPLAY_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "version {} not supported (expected {})",
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

    /// Returns the next (being_idx, ReplayFrame) or None on EOF.
    pub fn next_frame(&mut self) -> Option<(u8, ReplayFrame)> {
        const FRAME_SIZE: usize = 128;
        const ENTRY_SIZE: usize = 2 + FRAME_SIZE; // tag + idx + frame

        if self.cursor + ENTRY_SIZE > self.data.len() {
            return None;
        }

        let tag = self.data[self.cursor];
        if tag != FRAME_TAG {
            // Corruption: skip bytes until a tag or EOF is found
            eprintln!(
                "REPLAY: invalid tag 0x{:02x} at offset {} — searching for the next",
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
        // MANDATORY UNALIGNED read: each entry occupies 130 bytes (tag + index +
        // 128-byte frame) and the frame starts at `cursor + 2`, so its offset is
        // 64 + k·130 + 2 — never a multiple of 4. `bytemuck::from_bytes` requires the
        // target type's alignment and panics on the very first frame; this is what
        // prevented any replay from being read.
        let frame: ReplayFrame = bytemuck::pod_read_unaligned(frame_bytes);

        self.cursor += ENTRY_SIZE;
        Some((being_idx, frame))
    }

    /// Collects every frame of a specific being, in order.
    pub fn frames_of(&mut self, target_idx: u8) -> Vec<ReplayFrame> {
        self.cursor = 64; // reset to the start of the data
        let mut out = Vec::new();
        while let Some((idx, frame)) = self.next_frame() {
            if idx == target_idx {
                out.push(frame);
            }
        }
        out
    }

    /// Returns every frame of every being, grouped by idx.
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

// ── Frame-by-frame analysis ──────────────────────────────────────────────────

/// Profile of one track section (divided into N equal arcs).
pub struct SectionProfile {
    pub arc_start: f32,
    pub arc_end: f32,
    pub avg_speed: f32,
    pub max_speed: f32,
    pub min_speed: f32,
    /// Average time spent in this section (estimated as arc_len / avg_speed)
    pub time_est: f32,
    /// Frames where the governor intervened (|thr_raw - thr_final| > 0.05)
    pub gov_count: u32,
    pub total_count: u32,
}

/// Analyzes a set of frames and produces a per-section profile.
///
/// `n_sections`: how many equal slices to divide the total arc into.
/// Returns None if frames is empty.
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

        // The governor intervened if the final throttle is significantly lower than raw
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

/// Critical event detected in the analysis.
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
    /// The governor cut the throttle (delta > threshold)
    GovernorCut { thr_before: f32, thr_after: f32 },
    /// Sentinel hard-veto active
    SentinelVeto,
    /// ECAS intervened (safety_flags bit2)
    EcasActive,
    /// Car in reverse (safety_flags bit3)
    Reversing,
    /// Speed dropped > 5 m/s in 10 frames (abrupt loss)
    SpeedDrop { delta: f32 },
    /// High neural saturation (>20 saturated dims)
    NeuralSaturation { dims: u8 },
}

/// Extracts every critical event from a sequence of frames.
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

        // Speed drop: compares with the frame 10 behind
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

        // Neural saturation
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

/// Diff between two replays (motor.rs path vs cortex_evolve path).
pub struct ReplayDiff {
    pub arc: f32,
    /// Speed on path A (motor.rs)
    pub speed_a: f32,
    /// Speed on path B (cortex_evolve)
    pub speed_b: f32,
    /// Final steer difference (A - B)
    pub steer_diff: f32,
    /// Final throttle difference (A - B)
    pub thr_diff: f32,
    /// Position A - position B (distance between trajectories)
    pub pos_diverge: f32,
}

/// Compares two sets of frames (same track, different beings or different paths).
/// Aligns by arc_prog — interpolates frame B to the same arc as frame A.
pub fn diff_replays(frames_a: &[ReplayFrame], frames_b: &[ReplayFrame]) -> Vec<ReplayDiff> {
    if frames_a.is_empty() || frames_b.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut bi = 0usize;

    for fa in frames_a {
        // Advances bi to the frame of B closest in arc_prog
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

// ── HTML player generator ────────────────────────────────────────────────────

/// HTML template embedded at compile time (src/assets/replay_player.html).
/// Contains all the visualization logic; the data is injected as window.REPLAY_*.
static HTML_TEMPLATE: &str = include_str!("assets/replay_player.html");

/// Builds a self-contained HTML with a frame-by-frame player.
///
/// `track_path`: track polyline in world coordinates [(x, z)].
/// `being_frames`: Vec of frame sequences per being (index 0 = champion).
/// `labels`: optional names for each being.
///
/// Injects `window.REPLAY_TRACK` and `window.REPLAY_BEINGS` before the template's `<script>`.
pub fn generate_html_player(
    track_path: &[[f32; 2]],
    being_frames: &[Vec<ReplayFrame>],
    labels: &[&str],
    output_path: &Path,
) -> io::Result<()> {
    let file = File::create(output_path)?;
    let mut w = BufWriter::new(file);

    // --- Computes the track bounds for normalization ---
    let (mut min_x, mut max_x, mut min_z, mut max_z) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for &[x, z] in track_path {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }
    // Includes every being's positions in the bounds
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

    // --- Serializes data as JSON embedded in the HTML ---
    // track_points: array of [px, py] normalized to [0..1000, 0..1000]
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

    // beings_data: array of objects, each with an array of frames
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

    // --- Inject the data into the HTML template ---
    // The template expects window.REPLAY_TRACK and window.REPLAY_BEINGS
    // defined BEFORE its <script>. We insert a <script> block with the
    // data right before the template's </head>.
    let data_script = format!(
        "<script>\nwindow.REPLAY_TRACK = {};\nwindow.REPLAY_BEINGS = {};\n</script>\n",
        track_json, beings_json
    );

    // Inserts the data block before </head>
    let output_html = HTML_TEMPLATE.replacen("</head>", &format!("{}</head>", data_script), 1);

    w.write_all(output_html.as_bytes())?;
    w.flush()?;
    Ok(())
}
