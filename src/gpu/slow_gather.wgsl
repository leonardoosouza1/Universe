// ── SLOW BRAIN INPUT GATHER ──────────────────────────────────────────────────
//
// # Why this kernel exists
//
// In `gpu_evolve` the 223-channel sensory wave lives ENTIRELY on the GPU: perception
// writes into `buf_wave` and the fast brain kernel reads from there, without going
// through the CPU. The slow brain needs 48 of those channels. Bringing the wave back
// to the host just to pick 48 values would cost one readback per tick — it would be
// going back to the CPU on exactly the path that was ported to avoid it.
//
// So the selection happens here, on the GPU: 48 threads, each one copies one channel.
//
// # The channel map is DECLARED, not guessed
//
// `CHANNEL_MAP[k]` says which wave channel the slow brain's input `k` comes from. A
// negative value means "this input has no source on this path" and the kernel writes
// zero. That is deliberate: in `motor.rs` the slow brain receives quantities from the
// being's body (HP, fuel, tyre temperature, time on wall/grass, episodic memory) that
// simply DO NOT EXIST in `gpu_evolve`, whose per-being state is only physics +
// timers. Filling those slots with a similar-looking channel would be inventing
// signal; zero is honest and the participation ratio will show how many dimensions
// were left.
//
// The wave arrives packed: 4 i8 channels per u32 (WAVE_U32S = ceil(192/4) = 48 u32
// per being), which is the same packing the fast brain kernel consumes.

const INPUT_SIZE: u32 = 48u;
const WAVE_U32S:  u32 = 48u;   // ceil(192/4). Was 56u (223-channel wave) — the slow brain read
                               // perception with the wrong stride, scrambled input. Fixed.

struct GatherParams {
    n_beings: u32,
    _pad0:    u32,
    _pad1:    u32,
    _pad2:    u32,
}

@group(0) @binding(0) var<uniform>             gparams:    GatherParams;
@group(0) @binding(1) var<storage, read>       buf_wave:   array<u32>;
@group(0) @binding(2) var<storage, read_write> buf_input:  array<i32>;

// Wave-channel → slow brain input map.
//
// The sources follow the blocks documented in `perception.wgsl`. Indices 0..13 and
// 22..26 have no source because they correspond to body state and episodic memory
// that `gpu_evolve` does not model.
const CHANNEL_MAP: array<i32, 48> = array<i32, 48>(
    // 0..3   joy_tick, joy_streak, hp, fuel — no source on this path
    -1, -1, -1, -1,
    // 4      speed
    33,
    // 5..13  tyre temperature, power, alignment, 2 reserved, time on
    //        wall/grass, stagnation, night — no source
    -1, -1, 32, -1, -1, -1, -1, -1, -1,
    // 14..17 near wall per region (4 retina rows, 40/25/15/8 m)
    36, 37, 38, 39,
    // 18..21 far wall per region
    40, 41, 42, 43,
    // 22..26 episodic memory, intent[3], previous slow brain output — no source
    -1, -1, -1, -1, -1,
    // 27..28 curvature at 30 and 60 m
    78, 79,
    // 29     grace window — no source
    -1,
    // 30     curvature at 90 m
    80,
    // 31     distance to the wall ahead
    82,
    // 32..47 high-definition vision bins (fovea)
    128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 141, 142, 143,
);

@compute @workgroup_size(48)
fn gather(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let being: u32 = wg.x;
    let k:     u32 = lid.x;
    if being >= gparams.n_beings || k >= INPUT_SIZE { return; }

    let src: i32 = CHANNEL_MAP[k];
    var v: i32 = 0;
    if src >= 0 {
        let ch: u32 = u32(src);
        // Unpacks the i8 of channel `ch`: 4 channels per u32, byte `ch % 4`.
        let word: u32 = buf_wave[being * WAVE_U32S + ch / 4u];
        let byte: u32 = (word >> ((ch % 4u) * 8u)) & 0xFFu;
        // Sign extension from 8 to 32 bits.
        if byte >= 128u { v = i32(byte) - 256; } else { v = i32(byte); }
    }
    buf_input[being * INPUT_SIZE + k] = v;
}
