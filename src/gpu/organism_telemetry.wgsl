// organism telemetry window v1; slots 104..122 mirror driving_diagnostics::Window.
struct Params { n: u32, tick: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> physics_in: array<f32>;
@group(0) @binding(2) var<storage, read> physics_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> body: array<f32>;

const PI: u32 = 92u;
const PO: u32 = 56u;
const BS: u32 = 128u;
const W: u32 = 104u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let b = id.x;
  if (b >= p.n) { return; }
  let i = b * PI;
  let o = b * PO;
  let s = b * BS;
  let alive = body[s + 29u] > 0.5;
  let reset = (p.tick % 64u) == 1u;

  // Reset only the rolling window. Slot 123 is the contact edge memory.
  if (reset) {
    for (var k = 0u; k < 19u; k++) { body[s + W + k] = 0.0; }
  }
  body[s + W] += 1.0;
  if (!alive) { return; }
  body[s + W + 1u] += 1.0;

  // Numeric mask contract: physics_c/traffic write f32(bits), never a bit pattern.
  let mask = u32(max(physics_out[o + 39u], 0.0));
  let touching = mask != 0u;
  if (touching) { body[s + W + 2u] += 1.0; }
  let previous_mask = u32(max(body[s + 123u], 0.0));
  if (touching && previous_mask == 0u) { body[s + W + 4u] += 1.0; }
  body[s + 123u] = f32(mask);

  let speed = length(vec2<f32>(physics_out[o], physics_out[o + 2u]));
  let brake = clamp(physics_in[i + 52u], 0.0, 1.0);
  let throttle = clamp(physics_in[i + 51u], 0.0, 1.0);
  if (brake > 0.05) { body[s + W + 5u] += 1.0; }
  if (brake > 0.05 && throttle > 0.1) { body[s + W + 6u] += 1.0; }
  if (speed < 0.5) { body[s + W + 7u] += 1.0; }
  if (bitcast<u32>(physics_in[i + 55u]) == 0u) { body[s + W + 8u] += 1.0; }
  if (body[s + 103u] <= 3.0) { body[s + W + 9u] += 1.0; }
  body[s + W + 10u] += speed;
  let dx = physics_out[o + 36u] - physics_in[i + 36u];
  let dz = physics_out[o + 38u] - physics_in[i + 38u];
  let distance = length(vec2<f32>(dx, dz));
  body[s + W + 11u] += distance;

  let impact_speed = max(0.0, physics_out[o + 43u]);
  let impulse = max(0.0, physics_out[o + 47u]);
  // Kernel C's canonical collision energy is output slot 55.
  let collision_energy = max(0.0, physics_out[o + 55u]);
  body[s + W + 12u] += collision_energy;
  body[s + W + 13u] += impulse;
  if (impulse > 0.0) { body[s + W + 3u] += 1.0; }
  body[s + W + 14u] = max(body[s + W + 14u], impact_speed);

  var slip_peak = 0.0;
  var slip_angle_peak = 0.0;
  for (var wheel = 0u; wheel < 4u; wheel++) {
    let wo = o + 4u + wheel * 8u;
    slip_peak = max(slip_peak, abs(physics_out[wo + 1u]));
    slip_angle_peak = max(slip_angle_peak, abs(physics_out[wo + 2u]));
  }
  body[s + W + 15u] = max(body[s + W + 15u], slip_peak);
  body[s + W + 16u] = max(body[s + W + 16u], slip_angle_peak);
  body[s + W + 17u] += throttle;
  body[s + W + 18u] += brake;

  // Lifetime counters (98..102) are initialized with the newborn body and
  // intentionally never participate in the 64-tick reset above.
  body[s + 98u] += distance;
  if (touching) { body[s + 99u] += 1.0; }
  body[s + 100u] += collision_energy;
  if (impulse > 0.0) { body[s + 101u] += 1.0; }
  if (brake > 0.05) { body[s + 102u] += 1.0; }
}
