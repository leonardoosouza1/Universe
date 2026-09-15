// Sampled diagnostics: one invocation reduces a being's resident retina.
struct Params { n: u32, pixels: u32, cols: u32, pad: u32 }
struct Pose { position: vec4<f32>, directions: vec4<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> retina: array<u32>;
@group(0) @binding(2) var<storage, read> poses: array<Pose>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let b = id.x;
    if b >= params.n { return; }
    var left = 0u;
    var right = 0u;
    var nearest_visible = 1200.0;
    for (var p = 0u; p < params.pixels; p++) {
        let raw = retina[b * params.pixels + p];
        if (raw & 255u) == 4u {
            if p % params.cols < params.cols / 2u { left++; } else { right++; }
            nearest_visible = min(nearest_visible, f32(raw >> 8u) / 1000.0);
        }
    }
    var nearest_body = 1200.0;
    var overlaps = 0u;
    for (var other = 0u; other < params.n; other++) {
        if other == b { continue; }
        let d = distance(poses[b].position.xz, poses[other].position.xz);
        nearest_body = min(nearest_body, d);
        if d < 1.55 { overlaps++; }
    }
    let offset = b * 8u;
    result[offset] = f32(left + right) / f32(params.pixels);
    result[offset + 1u] = (f32(right) - f32(left)) / f32(max(left + right, 1u));
    result[offset + 2u] = f32(left);
    result[offset + 3u] = f32(right);
    result[offset + 4u] = nearest_visible;
    result[offset + 5u] = nearest_body;
    result[offset + 6u] = f32(overlaps);
    result[offset + 7u] = 0.0;
}
