// organism-taxi-v1 bounded body kernel.
// State is 128 f32/being; genes 16 f32; physics input 92 f32; output 56 f32.
// One authoritative writer owns energy, water, material, wear, fatigue and temperature.

struct Params { n: u32, _pad: u32, dt: f32, config_flags: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> state: array<f32>;
@group(0) @binding(2) var<storage, read> genes: array<f32>;
@group(0) @binding(3) var<storage, read_write> io: array<f32>;

const STATE_STRIDE: u32 = 128u;
const GENE_STRIDE: u32 = 16u;
const INPUT_STRIDE: u32 = 92u;
const OUTPUT_STRIDE: u32 = 56u;
const DISABLE_OLD_THERMAL: u32 = 256u;

fn clamp01(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn idx(being: u32, slot: u32) -> u32 { return being * STATE_STRIDE + slot; }
fn gidx(being: u32, slot: u32) -> u32 { return being * GENE_STRIDE + slot; }
fn iidx(being: u32, slot: u32) -> u32 { return being * INPUT_STRIDE + slot; }
fn oidx(being: u32, slot: u32) -> u32 { return being * OUTPUT_STRIDE + slot; }

@compute @workgroup_size(64)
fn encode_pre(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= p.n) { return; }
  let s = b * STATE_STRIDE;
  let g = b * GENE_STRIDE;
  io[iidx(b, 87u)] = bitcast<f32>(bitcast<u32>(io[iidx(b, 87u)]) | p.config_flags | 64u);
  state[s+57u]=io[iidx(b,51u)];state[s+58u]=io[iidx(b,53u)];state[s+59u]=io[iidx(b,52u)];state[s+60u]=1.0;
  if (state[s + 29u] < 0.5) {
    // A dead body cannot continue to power the legacy physics path.
    io[iidx(b, 48u)] = 0.0;
    io[iidx(b, 51u)] = 0.0;
    io[iidx(b, 52u)] = 1.0;
    io[iidx(b, 54u)] = 0.0;
    state[s+57u]=0.0;state[s+59u]=1.0;
    return;
  }
  // Capture physical birth bases exactly once. Subsequent ticks express from
  // these values, avoiding compounding gene scales on the physics input.
  if (state[s + 47u] == 0.0) {
    state[s + 47u] = max(1.0, io[iidx(b, 48u)]);
    state[s + 48u] = max(1.0, io[iidx(b, 68u)]);
    state[s + 49u] = max(1.0, io[iidx(b, 62u)]);
  }
  // Effective capacity is recomputed from immutable birth bases every tick.
  // This prevents compounding a gene expression after reset.
  let maturity = clamp01(state[s + 4u]);
  let wear_factor = 1.0 - 0.7 * clamp01(state[s + 6u]);
  // Fatigue is an acute capacity limit; wear and senescence remain separate
  // chronic limits. The 0.8 coefficient is the bounded model assumption recorded
  // in organism.rs: fatigue at one leaves 20% of the birth engine capacity.
  let fatigue_factor = 1.0 - 0.8 * clamp01(state[s + 5u]);
  let age_factor = 1.0 - 0.35 * clamp01(state[s + 13u]);
  state[s + 32u] = state[s + 16u] * mix(0.75, 1.0, maturity) * wear_factor * age_factor * fatigue_factor;
  state[s + 33u] = state[s + 17u] * mix(0.75, 1.0, maturity) * (1.0 - 0.7 * state[s + 7u]);
  state[s + 34u] = state[s + 18u];
  state[s + 35u] = state[s + 19u];
  state[s + 36u] = state[s + 20u];
  state[s + 37u] = max(0.05, state[s + 21u] * (1.0 - 0.5 * state[s + 6u]));
  state[s + 38u] = max(0.0, state[s + 22u] * maturity);
  state[s + 39u] = max(0.0, state[s + 23u] * (1.0 - state[s + 13u]));
  state[s + 40u] = max(0.25, state[s + 24u]);
  let throttle = abs(io[iidx(b, 51u)]);
  let efficiency = clamp(0.4 + 0.2 * genes[g + 12u], 0.1, 0.8);
  state[s + 41u] = 120.0 * state[s + 40u];
  // Bound crankshaft work at maximum possible engine speed before dispatch.
  let max_power = state[s + 47u] * state[s + 32u] * throttle * (io[iidx(b, 50u)] + 250.0) * 0.104719755;
  let affordable = max(0.0, state[s] - state[s + 41u] * p.dt);
  let motor_scale = min(1.0, affordable * efficiency / max(0.000001, max_power * p.dt));
  let gear = bitcast<u32>(io[iidx(b, 55u)]);
  var ratio = 1.15;
  switch gear { case 0u: {ratio=3.5;} case 1u: {ratio=0.0;} case 2u: {ratio=3.8;}
    case 3u: {ratio=2.6;} case 4u: {ratio=1.9;} case 5u: {ratio=1.45;} default: {} }
  state[s + 50u] = ratio * 3.7;
  // This is the sole body-side capability expression. It writes the input
  // consumed by physics and marks the legacy thermal writer as disabled.
  io[iidx(b, 48u)] = state[s + 47u] * state[s + 32u] * motor_scale;
  io[iidx(b, 68u)] = state[s + 48u] * state[s + 33u];
  // Taxi slot 65 is an explicit onboard/load flag. Passenger mass is applied
  // from immutable birth mass each tick, so boarding cannot compound mass.
  let passenger_mass = select(0.0, 75.0, state[s + 65u] > 0.5);
  io[iidx(b, 62u)] = state[s + 49u] * state[s + 40u] + passenger_mass;
  state[s+53u]=io[iidx(b,48u)];state[s+54u]=io[iidx(b,68u)];state[s+55u]=io[iidx(b,62u)];
  io[iidx(b, 87u)] = bitcast<f32>(bitcast<u32>(io[iidx(b, 87u)]) | p.config_flags | 64u);
}

@compute @workgroup_size(64)
fn encode_post(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= p.n) { return; }
  let s = b * STATE_STRIDE;
  let o = b * OUTPUT_STRIDE;
  if (state[s + 29u] < 0.5) {
    io[o+53u]=state[s+9u]-273.15;io[o+54u]=state[s+56u]-273.15;
    return;
  }
  let dt = p.dt;
  let torque = abs(io[o + 49u]);
  let speed = length(vec2<f32>(io[o + 0u], io[o + 2u]));
  let rpm = max(0.0, io[o + 48u]);
  // drive_torque_out is wheel-side torque: undo the drivetrain ratio before RPM.
  let crank_torque = select(0.0, torque / max(0.000001, state[s + 50u]), state[s + 50u] > 0.0);
  let mechanical_power = crank_torque * rpm * 0.104719755;
  let efficiency = clamp(0.4 + 0.2 * genes[b * GENE_STRIDE + 12u], 0.1, 0.8);
  let integrity = clamp01(state[s + 8u]);
  let damage = clamp01(1.0 - integrity);

  let chemical_power = mechanical_power / efficiency;
  let basal_paid = min(state[s], state[s + 41u] * dt);
  state[s] -= basal_paid;
  let motion_request = chemical_power * dt;
  let motion_paid = min(state[s], motion_request);
  state[s] -= motion_paid;
  state[s + 51u] = basal_paid;
  state[s + 52u] = motion_paid;
  state[s + 63u] = max(0.0, motion_request - motion_paid);
  state[s + 3u] += dt;
  let durability = 1.0 + genes[b * GENE_STRIDE + 13u];
  state[s + 6u] = clamp01(state[s + 6u] + dt * (crank_torque / max(1.0, state[s + 47u])) * 0.0001 / max(0.25, durability));
  state[s + 7u] = clamp01(state[s + 7u] + dt * speed * 0.00001 / max(0.25, durability));

  let growth_request = dt * state[s + 39u] * (1.0 - state[s + 4u]);
  let growth_energy = growth_request * 1000.0;
  let growth_material = growth_request * 0.2;
  let temperature_gate = select(0.0, 1.0, state[s + 9u] > 285.0 && state[s + 9u] < 345.0);
  let water_gate = select(0.0, 1.0, state[s + 1u] > 0.2 * state[s + 35u]);
  let growth_scale = min(temperature_gate * water_gate, min(1.0, min(state[s + 0u] / max(1.0, growth_energy), state[s + 2u] / max(0.001, growth_material))));
  let growth = growth_request * clamp01(growth_scale);
  state[s + 0u] = max(0.0, state[s + 0u] - growth * 1000.0);
  state[s + 2u] = max(0.0, state[s + 2u] - growth * 0.2);
  state[s + 4u] = clamp01(state[s + 4u] + growth);
  state[s + 12u] += growth * 1000.0;

  let resting = speed < 0.25 && torque < 1.0;
  let repair_request = min(state[s + 6u] + state[s + 7u], dt * state[s + 38u] * (1.0 + genes[b * GENE_STRIDE + 14u]));
  let repair_engine = min(state[s + 6u], repair_request * 0.5);
  let repair_tyre = min(state[s + 7u], repair_request - repair_engine);
  let repair_energy = (repair_engine + repair_tyre) * 500.0;
  let repair_material = (repair_engine + repair_tyre) * 0.5;
  let repair_temp_gate = state[s + 9u] > 285.0 && state[s + 9u] < 345.0;
  let repair_scale = select(0.0, min(1.0, min(state[s + 0u] / max(1.0, repair_energy), state[s + 2u] / max(0.001, repair_material))), resting && repair_temp_gate && repair_request > 0.0);
  let paid_engine = repair_engine * repair_scale;
  let paid_tyre = repair_tyre * repair_scale;
  state[s + 6u] = max(0.0, state[s + 6u] - paid_engine);
  state[s + 7u] = max(0.0, state[s + 7u] - paid_tyre);
  state[s + 0u] = max(0.0, state[s + 0u] - (paid_engine + paid_tyre) * 500.0);
  state[s + 2u] = max(0.0, state[s + 2u] - (paid_engine + paid_tyre) * 0.5);
  state[s + 11u] += (paid_engine + paid_tyre) * 500.0;
  state[s + 42u] = paid_engine;
  state[s + 43u] = paid_tyre;
  state[s + 44u] = (paid_engine + paid_tyre) * 500.0;
  state[s + 45u] = (paid_engine + paid_tyre) * 0.5;
  state[s + 46u] = repair_scale;

  let ambient = 293.15;
  let old_thermal = select(max(0.0, io[o + 53u] + 273.15 - ambient) * 100.0, 0.0, (p.config_flags & DISABLE_OLD_THERMAL) != 0u);
  let basal_power = basal_paid / dt;
  let paid_chemical_power = motion_paid / max(dt, 0.0001);
  let waste_power = max(0.0, paid_chemical_power - min(mechanical_power, paid_chemical_power));
  let heat = waste_power + basal_power + old_thermal
    + state[s + 44u] / dt + growth * 1000.0 / dt;
  let fatigue_load = clamp01((chemical_power / 10000.0) + speed * 0.001);
  // Declared bounded model time scales: 120 s at maximum load to fatigue=1;
  // 60 s of unloaded rest to recover from fatigue=1. Both rates are per second.
  let fatigue_recovery = select(0.0, 1.0 / 60.0, speed < 0.25 && torque < 1.0);
  state[s + 5u] = clamp01(state[s + 5u] + dt * (fatigue_load / 120.0 - fatigue_recovery));
  let water_fraction = clamp01(state[s + 1u] / max(0.001, state[s + 35u]));
  let cooling_coeff = 120.0 * state[s + 37u] * (0.25 + 0.75 * water_fraction) * (1.0 + speed);
  var water_vapor = 0.0;
  if (state[s + 9u] > 373.15 && state[s + 1u] > 0.0) {
    water_vapor = min(state[s + 1u], heat * dt / 2260000.0);
  }
  // Resistance is a positive multiplier: default resistance 1 leaks, while
  // higher resistance reduces leakage without accidentally clamping it to zero.
  let leak = select(0.0, damage * (1.0 / max(0.25, state[s + 26u])) * 0.001 * dt, damage > 0.0);
  let latent_power = water_vapor * 2260000.0 / max(dt, 0.0001);
  state[s + 1u] = max(0.0, state[s + 1u] - water_vapor - leak);

  // Two-temperature implicit Euler balance. Body heat capacity and oil heat
  // capacity are fixed model coefficients (12 kJ/K and 6 kJ/K); exchange and
  // oil-to-air cooling conserve energy between compartments before cooling to air.
  let body_capacity = 12000.0 + state[s + 25u] * 100.0;
  let oil_capacity = 6000.0;
  let exchange = 240.0;
  let oil_cooling = cooling_coeff;
  let body_source = old_thermal + basal_power + state[s + 44u] / dt + growth * 1000.0 / dt - latent_power;
  let oil_source = waste_power;
  let a = body_capacity / dt + exchange;
  let bmat = oil_capacity / dt + exchange + oil_cooling;
  let rhs_body = body_capacity / dt * state[s + 9u] + body_source;
  let rhs_oil = oil_capacity / dt * state[s + 56u] + oil_source + oil_cooling * ambient;
  let determinant = max(1.0, a * bmat - exchange * exchange);
  let body_temperature = (rhs_body * bmat + exchange * rhs_oil) / determinant;
  let oil_temperature = (a * rhs_oil + exchange * rhs_body) / determinant;
  state[s + 9u] = body_temperature;
  state[s + 56u] = oil_temperature;
  state[s + 13u] = clamp01(max(0.0, (state[s + 3u] - 3600.0) / 3600.0) / max(0.25, 1.0 + genes[b * GENE_STRIDE + 11u]));
  let thermal_damage = max(0.0, state[s + 9u] - (373.15 + state[s + 25u])) * dt * 0.000001;
  let collision_damage = max(0.0, io[o + 55u]) * 0.0000001;
  state[s + 8u] = clamp01(state[s + 8u] - thermal_damage - collision_damage);
  state[s + 10u] = clamp01(1.0 - state[s + 8u]);
  state[s + 14u] = mechanical_power;
  state[s + 15u] = heat;
  // Physics owns the mechanical output; this body kernel owns the temperatures.
  io[o + 53u] = state[s + 9u] - 273.15;
  io[o + 54u] = state[s + 56u] - 273.15;
  if (state[s + 0u] <= 0.0 || state[s + 1u] <= 0.0 || state[s + 8u] <= 0.0) {
    state[s + 29u] = 0.0;
  }
}
