// Event-driven LIF fly map.  The three entry points share one numerical contract with
// crate::fly_map::spiking::SpikingState:
//   propagate  -- delayed spikes walk the presynaptic CSR into an integer accumulator;
//   integrate  -- one LIF substep consumes that accumulator and writes a packed spike ring;
//   readout    -- an exponential mean of plus/minus efferent spike rates.
//
// The accumulator stores signed synapse-count units, not f32. FlyWire weights are integer
// synapse counts; using an atomic integer makes concurrent source rows deterministic in
// value (the final sum is integer-exact) and avoids the nonexistent portable float atomicAdd.

struct Params {
    n: u32,
    n_beings: u32,
    capacity: u32,
    n_words: u32,
    n_types: u32,
    delay_steps: u32,
    slot: u32,
    eye_on: u32,
    eye_mode: u32, // 0 photoreceptor, 1 motion energy
    step_lo: u32,
    step_hi: u32,
    seed_lo: u32,
    seed_hi: u32,
    dt_ms: f32,
    v0: f32,
    v_rst: f32,
    v_th: f32,
    decay_v: f32,
    decay_syn: f32,
    w_syn: f32,
    poisson_factor: f32,
    background_hz: f32,
    decay_read: f32,
    read_norm: f32,
    retina_stride: u32,
    refractory_steps: u32,
    _pad1: u32,
    _pad2: u32,
    debug_being: u32,
    integrate_groups_x: u32,
    _debug_pad1: u32,
    _debug_pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;

@group(0) @binding(1) var<storage, read> pre_indptr: array<u32>;
@group(0) @binding(2) var<storage, read> pre_indices: array<u32>;
@group(0) @binding(3) var<storage, read> pre_weight_units: array<i32>;
@group(0) @binding(4) var<storage, read> type_of: array<u32>;
@group(0) @binding(5) var<storage, read> type_mask: array<f32>;
@group(0) @binding(6) var<storage, read_write> spike_ring: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> syn_acc: array<atomic<i32>>;
// The selection's gain/scale genes are shared by K1/K2. Scale is applied after the
// integer atomic delivery (so the accumulator remains exact); gain modulates K2's
// Poisson rate.
@group(0) @binding(8) var<storage, read> gain_scale: array<f32>;
@group(0) @binding(9) var<storage, read> pixel_of_neuron: array<u32>;
@group(0) @binding(10) var<storage, read> no_refractory: array<u32>;
@group(0) @binding(11) var<storage, read> retina_cur: array<u32>;
@group(0) @binding(12) var<storage, read> retina_prev: array<u32>;
@group(0) @binding(13) var<storage, read_write> v: array<f32>;
@group(0) @binding(14) var<storage, read_write> i_syn: array<f32>;
@group(0) @binding(15) var<storage, read_write> refractory: array<u32>;
@group(0) @binding(16) var<storage, read_write> spike_counts: array<atomic<u32>>;
@group(0) @binding(17) var<storage, read> out_indptr: array<u32>;
@group(0) @binding(18) var<storage, read> out_members: array<u32>;
@group(0) @binding(19) var<storage, read> out_pair: array<u32>;
@group(0) @binding(20) var<storage, read_write> read_count: array<f32>;
@group(0) @binding(21) var<storage, read_write> command: array<f32>;
@group(0) @binding(22) var<storage, read_write> debug_v: array<f32>;
@group(0) @binding(23) var<storage, read_write> debug_counts: array<u32>;

fn ring_index(slot: u32, neuron: u32, word: u32) -> u32 {
    return (slot * params.n + neuron) * params.n_words + word;
}

fn mix(x0: u32) -> u32 {
    var x = x0;
    x = (x ^ (x >> 16u)) * 0x7FEB352Du;
    x = (x ^ (x >> 15u)) * 0x846CA68Bu;
    return x ^ (x >> 16u);
}

fn uniform01(neuron: u32, being: u32) -> f32 {
    let h = mix(
        neuron * 0x9E3779B9u + being * 0x85EBCA6Bu +
        params.step_lo * 0xC2B2AE35u + params.step_hi * 0x27D4EB2Fu +
        params.seed_lo + params.seed_hi
    );
    return f32(h >> 8u) / 16777216.0;
}

fn packed_brightness(packed: u32) -> f32 {
    let d = f32(packed >> 8u) / 1000.0;
    let mat = packed & 0xFFu;
    let depth_fade = clamp(d / 1200.0, 0.0, 1.0);
    var raw: f32;
    if mat == 2u { // Wall
        if d < 30.0 {
            raw = 127.0 - sqrt(clamp(d / 30.0, 0.0, 1.0)) * 72.0;
        } else if d < 120.0 {
            raw = 55.0 - clamp((d - 30.0) / 90.0, 0.0, 1.0) * 55.0;
        } else if d < 1200.0 {
            raw = -clamp((d - 120.0) / 480.0, 0.0, 1.0) * 25.0;
        } else {
            raw = -25.0 - clamp((d - 1200.0) / 1200.0, 0.0, 1.0) * 25.0;
        }
    } else if mat == 0u { // Grass
        raw = 40.0 - depth_fade * 40.0;
    } else if mat == 1u { // Road
        raw = -128.0 + depth_fade * 96.0;
    } else if mat == 3u { // Hole
        raw = 90.0 - clamp(d / 50.0, 0.0, 1.0) * 30.0;
    } else { // Being (4): matches gpu::legacy_encoding and the rate path.
        raw = -29.0;
    }
    // Rust's f32 -> i8 cast truncates toward zero; WGSL has trunc for the same contract.
    raw = trunc(clamp(raw, -128.0, 127.0));
    return clamp((raw + 128.0) / 255.0, 0.0, 1.0);
}

fn eye_rate_factor(neuron: u32, being: u32) -> f32 {
    let pixel = pixel_of_neuron[neuron];
    if params.eye_on == 0u || pixel == 0xFFFFFFFFu {
        return 0.0;
    }
    let index = being * params.retina_stride + pixel;
    let current = packed_brightness(retina_cur[index]);
    if params.eye_mode == 0u {
        return 1.0 - current;
    }
    let previous = packed_brightness(retina_prev[index]);
    // Motion mode is motion energy. Direction selectivity requires a separate polarity
    // table; pixel_of_neuron alone cannot encode it.
    return abs(current - previous);
}

fn is_no_refractory(neuron: u32) -> bool {
    return (no_refractory[neuron / 32u] & (1u << (neuron & 31u))) != 0u;
}

@compute @workgroup_size(64)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let source_word = gid.x;
    let total = params.n * params.n_words;
    if source_word >= total {
        return;
    }
    let source = source_word / params.n_words;
    let word = source_word % params.n_words;
    let packed = atomicLoad(&spike_ring[ring_index(params.slot, source, word)]);
    if packed == 0u {
        return;
    }
    let t = type_of[source];
    for (var bit: u32 = 0u; bit < 32u; bit = bit + 1u) {
        if (packed & (1u << bit)) == 0u {
            continue;
        }
        let being = word * 32u + bit;
        if being >= params.n_beings {
            continue;
        }
        if type_mask[being * params.n_types + t] < 0.5 {
            continue;
        }
        let start = pre_indptr[source];
        let end = pre_indptr[source + 1u];
        for (var edge: u32 = start; edge < end; edge = edge + 1u) {
            let post = pre_indices[edge];
            atomicAdd(&syn_acc[post * params.capacity + being], pre_weight_units[edge]);
        }
    }
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.n * params.capacity;
    let linear = gid.y * params.integrate_groups_x * 64u + gid.x;
    if linear >= total {
        return;
    }
    let neuron = linear / params.capacity;
    let being = linear % params.capacity;
    let delivered = atomicExchange(&syn_acc[linear], 0);
    var vv = v[linear];
    let scale = max(gain_scale[being * 2u + 1u], 0.0);
    var syn = i_syn[linear] + f32(delivered) * params.w_syn * scale;
    var rfc = refractory[linear];
    var spiked = false;
    if being < params.n_beings {
        if rfc == 0u {
            vv = params.v0 + (vv - params.v0) * params.decay_v + syn * (1.0 - params.decay_v);
            syn = syn * params.decay_syn;
            let pixel = pixel_of_neuron[neuron];
            if params.eye_on != 0u && pixel != 0xFFFFFFFFu {
                let gain = max(gain_scale[being * 2u], 0.0);
                let rate = params.background_hz * gain * eye_rate_factor(neuron, being);
                let p = clamp(1.0 - exp(-max(rate, 0.0) * params.dt_ms * 1e-3), 0.0, 1.0);
                if uniform01(neuron, being) < p {
                    vv = vv + params.w_syn * params.poisson_factor;
                }
            }
            spiked = vv > params.v_th;
        } else {
            // Absolute refractory: do not integrate fresh drive or emit another event.
            rfc = rfc - 1u;
        }

        if spiked {
            vv = params.v_rst;
            syn = 0.0;
            if is_no_refractory(neuron) {
                rfc = 0u;
            } else {
                // Remaining-step counter; the next rfc_steps substeps are blocked.
                rfc = params.refractory_steps;
            }
            // Counts are cleared once per Universe tick and accumulated across all
            // LIF substeps.  Storing 1 here would make K3 observe only the final
            // substep, silently shrinking rates by ~tick_substeps.
            atomicAdd(&spike_counts[linear], 1u);
            let ring_word = (params.slot * params.n + neuron) * params.n_words + being / 32u;
            atomicOr(&spike_ring[ring_word], 1u << (being & 31u));
        }
    }
    v[linear] = vv;
    i_syn[linear] = syn;
    refractory[linear] = rfc;
}

fn group_mean(group: u32, being: u32) -> f32 {
    if group == 0xFFFFFFFFu || group >= arrayLength(&out_indptr) - 1u {
        return 0.0;
    }
    let start = out_indptr[group];
    let end = out_indptr[group + 1u];
    if end <= start {
        return 0.0;
    }
    var sum: f32 = 0.0;
    for (var i: u32 = start; i < end; i = i + 1u) {
        let neuron = out_members[i];
        sum = sum + f32(atomicLoad(&spike_counts[neuron * params.capacity + being]));
    }
    return sum / f32(end - start);
}

@compute @workgroup_size(64)
fn readout(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    let being = slot / 3u;
    let command_idx = slot % 3u;
    if being >= params.n_beings {
        return;
    }
    let plus = out_pair[being * 6u + command_idx * 2u];
    let minus = out_pair[being * 6u + command_idx * 2u + 1u];
    let drive = group_mean(plus, being) - group_mean(minus, being);
    let old = read_count[slot];
    let next = old * params.decay_read + (1.0 - params.decay_read) * drive;
    read_count[slot] = next;
    command[slot] = tanh(next / max(params.read_norm, 1e-6));
}

// Compacts one selected being from the being-minor state layout. This pass exists only
// for the explicit trace flag; ordinary ticks never bind these buffers or dispatch it.
@compute @workgroup_size(64)
fn debug_snapshot(@builtin(global_invocation_id) gid: vec3<u32>) {
    let neuron = gid.x;
    if neuron >= params.n || params.debug_being >= params.n_beings {
        return;
    }
    let idx = neuron * params.capacity + params.debug_being;
    debug_v[neuron] = v[idx];
    debug_counts[neuron] = atomicLoad(&spike_counts[idx]);
}
