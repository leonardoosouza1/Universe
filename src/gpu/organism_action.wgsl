// Resident native action mapper. This is a direct port of native_action.rs:
// KAN head -> slow/hip modulation -> arbitration/clamps -> steer smoothing.
struct Params {
  n: u32,
  flags: u32, // bit 0: linear head, bit 1: signed throttle arbitration
  steering_alpha: f32,
  slow_steering_authority: f32,
  gearbox_mode: u32,
  gear_cap: u32,
  dt: f32,
  _pad: f32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> intent: array<f32>;
@group(0) @binding(2) var<storage, read> slow: array<f32>;
@group(0) @binding(3) var<storage, read> heads: array<f32>;
@group(0) @binding(4) var<storage, read_write> previous_steer: array<f32>;
@group(0) @binding(5) var<storage, read> hip_bias: array<f32>;
struct GearState { cooldown: f32, prev_up: f32, prev_down: f32, gear: u32 };
@group(0) @binding(6) var<storage, read_write> gear_state: array<GearState>;
@group(0) @binding(7) var<storage, read_write> physics: array<f32>;

@group(0) @binding(8) var<storage,read_write> trace:array<f32>;

const INTENT_STRIDE: u32 = 96u;
const SLOW_STRIDE: u32 = 3u;
const HEAD_STRIDE: u32 = 36u;
const KNOTS: u32 = 12u;
const DOMAIN: f32 = 2.0;
const SLOW_BRAKE_AUTHORITY: f32 = 0.8;
const PHYSICS_STRIDE: u32 = 92u;
const FLAG_ALLOW_LOW_SPEED_BRAKE: u32 = 4u;

fn spline(b: u32, output: u32, x: f32) -> f32 {
  let xc = clamp(x, -DOMAIN, DOMAIN);
  let fpos = ((xc + DOMAIN) / (2.0 * DOMAIN)) * f32(KNOTS - 1u);
  let i0 = u32(floor(fpos));
  let i1 = min(i0 + 1u, KNOTS - 1u);
  let frac = fpos - f32(i0);
  let base = b * HEAD_STRIDE + output * KNOTS;
  return clamp(heads[base + i0] + (heads[base + i1] - heads[base + i0]) * frac, -1.0, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= p.n) { return; }
  let ib = b * INTENT_STRIDE;
  let sb = b * SLOW_STRIDE;
  let pb = b * PHYSICS_STRIDE;
  let hb = b * 3u;
  let after0 = select(spline(b, 0u, intent[ib]), clamp(intent[ib], -1.0, 1.0), (p.flags & 1u) != 0u);
  let after1 = select(spline(b, 1u, intent[ib + 1u]), clamp(intent[ib + 1u], -1.0, 1.0), (p.flags & 1u) != 0u);
  let after2 = select(spline(b, 2u, intent[ib + 2u]), clamp(intent[ib + 2u], -1.0, 1.0), (p.flags & 1u) != 0u);
  let slow_inhibition = max(slow[sb + 1u], 0.0);
  let throttle_cap = 1.0 - slow_inhibition * SLOW_BRAKE_AUTHORITY;
  let throttle_base = select(max(after0, 0.0), after0, (p.flags & 2u) != 0u);
  let throttle = clamp(throttle_base + slow[sb] * 0.5 + hip_bias[hb], 0.0, throttle_cap);
  let steer = clamp(after1 + slow[sb + 2u] * p.slow_steering_authority + hip_bias[hb + 1u], -1.0, 1.0);
  let fast_brake = max(after2 + hip_bias[hb + 2u], 0.0);
  let brake_before_gate = min(fast_brake + slow_inhibition * SLOW_BRAKE_AUTHORITY, 1.0);
  let speed = length(vec2<f32>(physics[pb], physics[pb + 2u]));
  // Legacy mode suppresses the brake below 2 m/s. The scenario-only flag
  // removes that suppression so a taxi can finish braking below 0.5 m/s.
  let suppress_low_speed = select(
    speed < 2.0,
    false,
    (p.flags & FLAG_ALLOW_LOW_SPEED_BRAKE) != 0u,
  );
  let brake = select(brake_before_gate, 0.0, brake_before_gate <= 0.0 || suppress_low_speed);
  let filtered = clamp(previous_steer[b] * (1.0 - p.steering_alpha) + steer * p.steering_alpha, -1.0, 1.0);
  physics[pb + 51u] = throttle;
  physics[pb + 53u] = filtered;
  physics[pb + 52u] = brake;
  previous_steer[b] = filtered;
  let tr=b*16u;
  trace[tr]=after0;trace[tr+1u]=after1;trace[tr+2u]=after2;
  trace[tr+3u]=throttle;trace[tr+4u]=steer;trace[tr+5u]=brake_before_gate;
  trace[tr+6u]=speed;trace[tr+7u]=filtered;
  trace[tr+8u]=select(0.0,1.0,suppress_low_speed && brake_before_gate>0.0);
  for(var k=9u;k<16u;k++){trace[tr+k]=0.0;}


  // Resident gearbox state mirrors context.rs: automatic thresholds are 85%/30%
  // RPM, engagement requires >3.5 m/s, and each change locks the box for 0.35 s.
  // Neural rows 3/4 are edge-triggered. Action row 8 is the explicit near-stop
  // reverse request; it never reverses a moving vehicle.
  if (p.gearbox_mode != 0u) {
    var gs = gear_state[b];
    let rpm = max(0.0, physics[pb + 49u]);
    let max_rpm = max(1.0, physics[pb + 50u]);
    let near_stop = speed < 0.5;
    if (near_stop && intent[ib + 8u] > 0.5) {
      gs.gear = 0u;
    } else if (near_stop && intent[ib + 8u] <= 0.5 && gs.gear == 0u) {
      gs.gear = 2u;
    }
    gs.cooldown = max(0.0, gs.cooldown - p.dt);
    if (gs.cooldown <= 0.0 && !(near_stop && intent[ib + 8u] > 0.5)) {
      let engaged = gs.gear >= 2u && speed > 3.5;
      var request: i32 = 0;
      if (engaged && rpm > max_rpm * 0.85 && gs.gear < p.gear_cap) {
        request = 1;
      } else if (engaged && rpm < max_rpm * 0.30 && gs.gear > 2u) {
        request = -1;
      } else if (p.gearbox_mode == 2u && engaged) {
        if (intent[ib + 3u] > 0.5 && gs.prev_up <= 0.5 && gs.gear < p.gear_cap) {
          request = 1;
        } else if (intent[ib + 4u] > 0.5 && gs.prev_down <= 0.5 && gs.gear > 2u) {
          request = -1;
        }
      }
      if (request != 0) {
        gs.gear = u32(clamp(i32(gs.gear) + request, 2, i32(p.gear_cap)));
        gs.cooldown = 0.35;
      }
      gs.prev_up = intent[ib + 3u];
      gs.prev_down = intent[ib + 4u];
    }
    physics[pb + 55u] = bitcast<f32>(gs.gear);
    gear_state[b] = gs;
  }
}
