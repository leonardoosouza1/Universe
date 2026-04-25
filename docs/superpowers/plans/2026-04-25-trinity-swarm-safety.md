# Trinity Swarm + Tesla-Style Safety — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace monolithic brain with Trinity (Cortex/Reflex/Sentinel) + learned MoE Gate, add chaos detection, drive PINADO deaths < 40% and accelerate champion progression.

**Architecture:** Pipeline: sensors → Cortex (slow, plans) + Reflex (fast, survives) + Sentinel (rules, vetoes) → Gate (MoE softmax) blends soft outputs → Sentinel hard veto last → motor.rs. Chaos detector feeds Gate using physical (yaw/slip/retina) + Kuramoto (resonance.rs) signals.

**Tech Stack:** Rust 1.85, Bevy 0.16, BitLinear (existing), FloatLinear (existing), specialist.rs (existing skeleton), resonance.rs (existing, unused).

**Spec:** `docs/superpowers/specs/2026-04-25-trinity-swarm-safety-design.md`

---

## File Structure

**New:**
- `src/safety/mod.rs` — module entry, re-exports Sentinel
- `src/safety/sentinel.rs` — Tier 0/1/2 rules, returns hard_veto + soft_proposal + flags
- `src/chaos.rs` — chaos score from physics + Kuramoto coherence
- `src/gate.rs` — MoE FloatLinear 16→3 + softmax

**Modified:**
- `src/lib.rs` — add `pub mod safety; pub mod chaos; pub mod gate;`
- `src/engine.rs` — `SovereignEngine` gains `gate: GateBrain`, `reflex: SpecialistBrain`; new method `decide(sensors, state) -> [f32; 3]`
- `src/resonance.rs` — add `coherence(nodes: &[ResonantNode]) -> f32` free fn
- `src/simulation/motor.rs` — replace direct `final_intent` reads with `engine.decide()` call
- `src/evolution/dna.rs` — serialize/deserialize gate + reflex weights
- `src/evolution/spawn.rs` — mutate gate + reflex weights
- `src/evolution/pipeline.rs` — apply fitness penalties (hard_veto_count, gate_entropy)
- `Cargo.toml` — add feature `trinity_swarm` default ON, deprecate `sentinel_brake`

**Test:**
- `src/safety/sentinel.rs` — unit tests inline (`#[cfg(test)]`)
- `src/chaos.rs` — unit tests inline
- `src/gate.rs` — unit tests inline
- `tests/trinity_integration.rs` — headless 1-car integration

---

## Task 1: Sentinel module — Tier 0/1/2 rules

**Files:**
- Create: `src/safety/mod.rs`
- Create: `src/safety/sentinel.rs`
- Modify: `src/lib.rs` — add `pub mod safety;`

- [ ] **Step 1.1: Create `src/safety/mod.rs`**

```rust
pub mod sentinel;
pub use sentinel::{Sentinel, SafetyState, IntentVec, SentinelOutput, SafetyFlags};
```

- [ ] **Step 1.2: Create `src/safety/sentinel.rs` with type definitions and failing tests**

```rust
//! Tesla-style safety layer. Pure rules, no NN.
//! Tier 0 = HARD veto (returns Some(IntentVec) overriding everything).
//! Tier 1/2 = SOFT proposal (gate decides weight).

use bitflags::bitflags;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IntentVec {
    pub throttle: f32,  // [-1, 1] (negative = reverse)
    pub steer: f32,     // [-1, 1]
    pub brake: f32,     // [0, 1]
}

#[derive(Debug, Clone, Copy)]
pub struct SafetyState {
    pub speed_kmh: f32,
    pub yaw_rate: f32,         // rad/s
    pub slip_avg: f32,         // [0, 1]
    pub ttc_forward: f32,      // s, INF if clear
    pub wall_left: f32,        // raycast distance m
    pub wall_right: f32,
    pub stall_timer: f32,      // s spent below 2 km/h with obstacle ahead
    pub pinado_timer: f32,     // s pinned (V<2 + obstacle)
    pub rain: f32,             // [0, 1]
    pub curv25: f32,           // upcoming curvature
    pub current_thr: f32,      // brain proposed thr (for tier2 TCS check)
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct SafetyFlags: u16 {
        const ANTI_STALL  = 1 << 0;
        const FCA         = 1 << 1;
        const SPIN        = 1 << 2;
        const ESP         = 1 << 3;
        const LANE_KEEP   = 1 << 4;
        const REVERSE_OUT = 1 << 5;
        const TCS         = 1 << 6;
        const AQUA        = 1 << 7;
        const CORNER_LIM  = 1 << 8;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SentinelOutput {
    pub hard_veto: Option<IntentVec>,
    pub soft_proposal: IntentVec,
    pub flags: SafetyFlags,
}

pub struct Sentinel;

impl Sentinel {
    pub fn evaluate(s: &SafetyState) -> SentinelOutput {
        let mut out = SentinelOutput::default();

        // ----- Tier 0 HARD vetos (priority: spin > FCA > anti-stall) -----
        if s.yaw_rate.abs() > 3.0 {
            out.flags |= SafetyFlags::SPIN;
            out.hard_veto = Some(IntentVec {
                throttle: 0.0,
                brake: 0.5,
                steer: -s.yaw_rate.signum() * 0.5,
            });
            return out;
        }
        if s.ttc_forward < 0.3 && s.speed_kmh > 15.0 {
            out.flags |= SafetyFlags::FCA;
            out.hard_veto = Some(IntentVec { throttle: 0.0, brake: 1.0, steer: 0.0 });
            return out;
        }
        if s.stall_timer > 0.3 && s.speed_kmh < 2.0 {
            out.flags |= SafetyFlags::ANTI_STALL;
            // sign chosen by which wall is closer (escape away from it)
            let escape_steer = if s.wall_left < s.wall_right { 0.5 } else { -0.5 };
            out.hard_veto = Some(IntentVec {
                throttle: -0.5,
                brake: 0.0,
                steer: escape_steer,
            });
            return out;
        }

        // ----- Tier 1 SOFT proposals -----
        let mut prop = IntentVec::default();
        if s.yaw_rate.abs() > 1.5 {
            out.flags |= SafetyFlags::ESP;
            prop.throttle += -0.7; // counter-throttle proposal
            prop.brake += 0.3;
        }
        if s.wall_left < 0.5 {
            out.flags |= SafetyFlags::LANE_KEEP;
            prop.steer += 0.3;
        } else if s.wall_right < 0.5 {
            out.flags |= SafetyFlags::LANE_KEEP;
            prop.steer -= 0.3;
        }
        if s.pinado_timer > 0.3 && s.speed_kmh < 2.0 {
            out.flags |= SafetyFlags::REVERSE_OUT;
            prop.throttle += -0.4;
        }

        // ----- Tier 2 SOFT proposals -----
        if s.slip_avg > 0.4 && s.current_thr > 0.5 {
            out.flags |= SafetyFlags::TCS;
            prop.throttle += -0.4;
        }
        if s.rain > 0.7 && s.speed_kmh > 40.0 {
            out.flags |= SafetyFlags::AQUA;
            prop.throttle += -0.3;
        }
        if s.curv25 > 30.0 && s.speed_kmh > 60.0 {
            out.flags |= SafetyFlags::CORNER_LIM;
            prop.brake += 0.3;
        }

        // clamp soft proposal
        prop.throttle = prop.throttle.clamp(-1.0, 1.0);
        prop.steer = prop.steer.clamp(-1.0, 1.0);
        prop.brake = prop.brake.clamp(0.0, 1.0);
        out.soft_proposal = prop;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_state() -> SafetyState {
        SafetyState {
            speed_kmh: 30.0, yaw_rate: 0.0, slip_avg: 0.0, ttc_forward: 10.0,
            wall_left: 5.0, wall_right: 5.0, stall_timer: 0.0, pinado_timer: 0.0,
            rain: 0.0, curv25: 0.0, current_thr: 0.5,
        }
    }

    #[test]
    fn anti_stall_triggers_after_300ms() {
        let mut s = base_state();
        s.speed_kmh = 1.0; s.stall_timer = 0.4;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::ANTI_STALL));
        let v = out.hard_veto.expect("hard veto required");
        assert!(v.throttle < 0.0);
    }

    #[test]
    fn fca_critical_overrides_throttle() {
        let mut s = base_state();
        s.ttc_forward = 0.2; s.speed_kmh = 50.0;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::FCA));
        assert_eq!(out.hard_veto.unwrap().brake, 1.0);
    }

    #[test]
    fn spin_correction_opposite_yaw() {
        let mut s = base_state();
        s.yaw_rate = 3.5;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::SPIN));
        assert!(out.hard_veto.unwrap().steer < 0.0);
    }

    #[test]
    fn no_hard_veto_in_calm_state() {
        let out = Sentinel::evaluate(&base_state());
        assert!(out.hard_veto.is_none());
        assert_eq!(out.flags, SafetyFlags::empty());
    }

    #[test]
    fn esp_soft_proposal_on_moderate_yaw() {
        let mut s = base_state();
        s.yaw_rate = 2.0;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::ESP));
        assert!(out.soft_proposal.brake > 0.0);
    }

    #[test]
    fn tcs_cuts_throttle_on_slip() {
        let mut s = base_state();
        s.slip_avg = 0.5; s.current_thr = 0.8;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::TCS));
        assert!(out.soft_proposal.throttle < 0.0);
    }

    #[test]
    fn priority_spin_over_fca() {
        let mut s = base_state();
        s.yaw_rate = 3.5; s.ttc_forward = 0.1; s.speed_kmh = 40.0;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::SPIN));
        assert!(!out.flags.contains(SafetyFlags::FCA));
    }
}
```

- [ ] **Step 1.3: Add bitflags dep to Cargo.toml**

Add line under `[dependencies]`:
```toml
bitflags = "2.6"
```

- [ ] **Step 1.4: Wire module in lib.rs**

Add to `src/lib.rs`:
```rust
pub mod safety;
```

- [ ] **Step 1.5: Run tests — expect PASS**

Run: `cargo test --lib safety::sentinel`
Expected: 7 tests pass.

- [ ] **Step 1.6: Commit**

```bash
git add src/safety/ src/lib.rs Cargo.toml Cargo.lock
git commit -m "feat(safety): add Sentinel Tier 0/1/2 rules with hard veto + soft proposal"
```

---

## Task 2: Chaos detector + Kuramoto coherence

**Files:**
- Create: `src/chaos.rs`
- Modify: `src/resonance.rs` — add `coherence` free function
- Modify: `src/lib.rs` — add `pub mod chaos;`

- [ ] **Step 2.1: Add `coherence` free fn to `src/resonance.rs`**

Append at end of file:
```rust
/// Kuramoto order parameter: |Σ e^iφ_k| / N. Returns [0, 1].
/// 1.0 = all in phase. 0.0 = uniformly distributed phases (chaotic).
pub fn coherence(nodes: &[ResonantNode]) -> f32 {
    if nodes.is_empty() { return 1.0; }
    let n = nodes.len() as f32;
    let (sin_sum, cos_sum) = nodes.iter().fold((0.0_f32, 0.0_f32), |(s, c), node| {
        (s + node.phase.sin(), c + node.phase.cos())
    });
    ((sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / n).min(1.0)
}

#[cfg(test)]
mod coherence_tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn coherence_full_when_aligned() {
        let mut nodes = vec![ResonantNode::new(1.0); 4];
        for n in &mut nodes { n.phase = 0.5; }
        assert!((coherence(&nodes) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn coherence_zero_when_uniform() {
        let mut nodes = vec![ResonantNode::new(1.0); 4];
        for (i, n) in nodes.iter_mut().enumerate() {
            n.phase = (i as f32) * (2.0 * PI / 4.0);
        }
        assert!(coherence(&nodes) < 0.05);
    }

    #[test]
    fn coherence_empty_returns_one() {
        assert_eq!(coherence(&[]), 1.0);
    }
}
```

- [ ] **Step 2.2: Run resonance tests — expect PASS**

Run: `cargo test --lib resonance::coherence_tests`
Expected: 3 tests pass.

- [ ] **Step 2.3: Create `src/chaos.rs`**

```rust
//! Chaos signal: blends physical instability with Kuramoto neural decoherence.
//! Output [0, 1]: 0 = calm, 1 = total chaos. Drives Gate input.

use crate::resonance::{ResonantNode, coherence};

#[derive(Debug, Clone, Copy)]
pub struct ChaosInput {
    pub yaw_rate: f32,           // rad/s
    pub slip_avg: f32,           // [0, 1]
    pub retina_temporal_var: f32, // 0..N magnitude of frame-to-frame retina change, normalized
}

#[derive(Debug, Clone, Copy)]
pub struct ChaosScore {
    pub physical: f32,    // [0, 1]
    pub neural: f32,      // [0, 1] = 1 - coherence
    pub coherence: f32,   // raw kuramoto
    pub combined: f32,    // max(physical, neural)
}

pub fn score(input: &ChaosInput, oscillators: &[ResonantNode]) -> ChaosScore {
    let phys = 0.4 * (input.yaw_rate.abs() / 2.0).tanh()
             + 0.4 * (input.slip_avg / 0.5).tanh()
             + 0.2 * input.retina_temporal_var.clamp(0.0, 1.0);
    let phys = phys.clamp(0.0, 1.0);
    let coh = coherence(oscillators);
    let neural = 1.0 - coh;
    let combined = phys.max(neural);
    ChaosScore { physical: phys, neural, coherence: coh, combined }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn calm_input() -> ChaosInput {
        ChaosInput { yaw_rate: 0.0, slip_avg: 0.0, retina_temporal_var: 0.0 }
    }

    #[test]
    fn calm_state_low_chaos() {
        let nodes = vec![ResonantNode::new(1.0); 4];
        let s = score(&calm_input(), &nodes);
        assert!(s.combined < 0.2);
    }

    #[test]
    fn high_yaw_high_physical_chaos() {
        let mut i = calm_input(); i.yaw_rate = 5.0;
        let s = score(&i, &[]);
        assert!(s.physical > 0.3);
    }

    #[test]
    fn decoherent_oscillators_high_neural_chaos() {
        let mut nodes = vec![ResonantNode::new(1.0); 4];
        for (i, n) in nodes.iter_mut().enumerate() {
            n.phase = (i as f32) * (2.0 * PI / 4.0);
        }
        let s = score(&calm_input(), &nodes);
        assert!(s.neural > 0.9);
        assert!(s.combined > 0.9);
    }

    #[test]
    fn combined_takes_max() {
        let mut i = calm_input(); i.yaw_rate = 5.0;
        let nodes = vec![ResonantNode::new(1.0); 4]; // coherent
        let s = score(&i, &nodes);
        assert_eq!(s.combined, s.physical);
    }
}
```

- [ ] **Step 2.4: Wire module in lib.rs**

Add to `src/lib.rs` next to safety:
```rust
pub mod chaos;
```

- [ ] **Step 2.5: Run chaos tests**

Run: `cargo test --lib chaos`
Expected: 4 tests pass.

- [ ] **Step 2.6: Commit**

```bash
git add src/chaos.rs src/resonance.rs src/lib.rs
git commit -m "feat(chaos): add chaos detector blending physical instability with Kuramoto coherence"
```

---

## Task 3: MoE Gate brain

**Files:**
- Create: `src/gate.rs`
- Modify: `src/lib.rs` — add `pub mod gate;`

- [ ] **Step 3.1: Create `src/gate.rs`**

```rust
//! Mixture-of-Experts gate. 16 inputs → 3 softmax outputs.
//! Outputs [w_cortex, w_reflex, w_sentinel] summing to 1.
//! Replaceable by float_layers::FloatLinear once that API stabilizes;
//! for now, keep dependency surface small with an inline tiny dense layer.

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

pub const GATE_INPUTS: usize = 16;
pub const GATE_OUTPUTS: usize = 3;

#[derive(Debug, Clone)]
pub struct GateBrain {
    pub weights: [f32; GATE_INPUTS * GATE_OUTPUTS], // row-major: out * INPUTS + in
    pub bias:    [f32; GATE_OUTPUTS],
}

impl GateBrain {
    /// Initialize uniform: bias=0, weights small random.
    /// Result of softmax(0) = [1/3, 1/3, 1/3] when weights are zero — ideal cold-start.
    pub fn new_uniform(seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut weights = [0.0_f32; GATE_INPUTS * GATE_OUTPUTS];
        for w in &mut weights {
            *w = (rng.random::<f32>() - 0.5) * 0.1; // tiny perturbation
        }
        Self { weights, bias: [0.0; GATE_OUTPUTS] }
    }

    pub fn forward(&self, input: &[f32; GATE_INPUTS]) -> [f32; GATE_OUTPUTS] {
        let mut logits = self.bias;
        for o in 0..GATE_OUTPUTS {
            let row = o * GATE_INPUTS;
            let mut acc = logits[o];
            for i in 0..GATE_INPUTS {
                acc += self.weights[row + i] * input[i];
            }
            logits[o] = acc;
        }
        // softmax
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        let mut exps = [0.0_f32; GATE_OUTPUTS];
        for o in 0..GATE_OUTPUTS {
            let e = (logits[o] - max).exp();
            exps[o] = e;
            sum += e;
        }
        if !sum.is_finite() || sum == 0.0 {
            return [0.0, 0.0, 1.0]; // NaN fallback → sentinel-only
        }
        for o in 0..GATE_OUTPUTS { exps[o] /= sum; }
        exps
    }

    /// Shannon entropy of softmax output. Used for fitness reward (avoid collapse).
    pub fn entropy(weights: &[f32; GATE_OUTPUTS]) -> f32 {
        let mut h = 0.0_f32;
        for &w in weights {
            if w > 1e-6 { h -= w * w.ln(); }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_init_softmax_near_third() {
        let g = GateBrain { weights: [0.0; 48], bias: [0.0; 3] };
        let out = g.forward(&[0.0; GATE_INPUTS]);
        for w in out { assert!((w - 1.0/3.0).abs() < 1e-5); }
    }

    #[test]
    fn softmax_sums_to_one() {
        let g = GateBrain::new_uniform(42);
        let out = g.forward(&[0.5; GATE_INPUTS]);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn nan_input_falls_back_to_sentinel() {
        let mut g = GateBrain::new_uniform(7);
        for w in &mut g.weights { *w = f32::INFINITY; }
        let out = g.forward(&[1.0; GATE_INPUTS]);
        assert_eq!(out, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn entropy_max_when_uniform() {
        let h = GateBrain::entropy(&[1.0/3.0; 3]);
        assert!((h - (3.0_f32).ln()).abs() < 1e-5);
    }

    #[test]
    fn entropy_zero_when_collapsed() {
        let h = GateBrain::entropy(&[1.0, 0.0, 0.0]);
        assert!(h < 1e-5);
    }
}
```

- [ ] **Step 3.2: Wire module**

Add to `src/lib.rs`:
```rust
pub mod gate;
```

- [ ] **Step 3.3: Run tests**

Run: `cargo test --lib gate`
Expected: 5 tests pass.

- [ ] **Step 3.4: Commit**

```bash
git add src/gate.rs src/lib.rs
git commit -m "feat(gate): add MoE gate (16 inputs → 3 softmax) with NaN fallback to sentinel"
```

---

## Task 4: Wire Trinity into SovereignEngine

**Files:**
- Modify: `src/engine.rs` — embed `gate`, `reflex`, expose `decide()`
- Modify: `src/specialist.rs` — verify `forward_specialist` signature returns `[f32; 3]`

- [ ] **Step 4.1: Read current SovereignEngine struct head**

Run: `sed -n '70,130p' /home/leonardo/projects/LLM/Universe/src/engine.rs`
Note: `instinct_layer` (BitLinear 128×64), `slow_brain`, `final_intent`. Add new fields.

- [ ] **Step 4.2: Add fields and method to SovereignEngine**

In `src/engine.rs`, add fields to `SovereignEngine` struct (after existing fields):
```rust
    pub gate: crate::gate::GateBrain,
    pub reflex: crate::specialist::SpecialistBrain,
    pub last_chaos: f32,
    pub hard_veto_count: u32,
    pub soft_override_count: u32,
    pub gate_entropy_acc: f32,
    pub gate_entropy_n: u32,
```

In `SovereignEngine::new(...)` initializer (find existing function, add):
```rust
    gate: crate::gate::GateBrain::new_uniform(0xDA7E),
    reflex: crate::specialist::SpecialistBrain::new_default(),
    last_chaos: 0.0,
    hard_veto_count: 0,
    soft_override_count: 0,
    gate_entropy_acc: 0.0,
    gate_entropy_n: 0,
```

- [ ] **Step 4.3: Verify `SpecialistBrain::new_default` exists or add it**

Run: `grep -n "pub fn new" /home/leonardo/projects/LLM/Universe/src/specialist.rs`
If `new_default` not present, add to `impl SpecialistBrain`:
```rust
    pub fn new_default() -> Self {
        Self::new(0xBEEF)
    }
```
(The seed-based constructor must already exist; if it does not, add a `new(seed)` returning a brain with small random weights matching the existing internal layout.)

- [ ] **Step 4.4: Add `decide` method to SovereignEngine**

In `src/engine.rs` `impl SovereignEngine`, append:
```rust
    /// Trinity output. Returns final [throttle, steer, brake] after Gate blend + Sentinel hard veto.
    /// Caller still drives `execute_cycle` for sensors → cortex; `decide` reads the cached
    /// `final_intent` (cortex output) and combines with reflex + sentinel.
    pub fn decide(
        &mut self,
        cortex_intent: [f32; 3],
        reflex_input: &[i8],
        safety_state: &crate::safety::SafetyState,
        chaos_input: &crate::chaos::ChaosInput,
        oscillators: &[crate::resonance::ResonantNode],
    ) -> [f32; 3] {
        use crate::safety::{Sentinel, IntentVec};

        let chaos = crate::chaos::score(chaos_input, oscillators);
        self.last_chaos = chaos.combined;

        let reflex_raw = self.reflex.forward_specialist(reflex_input); // [thr, steer, brake]
        let sentinel_out = Sentinel::evaluate(safety_state);

        // Gate input vector (16 dims, normalized roughly to ±1)
        let gi = [
            chaos.combined,
            (safety_state.yaw_rate / 3.0).clamp(-1.0, 1.0),
            (safety_state.slip_avg).clamp(0.0, 1.0),
            (1.0 / (1.0 + safety_state.ttc_forward)).clamp(0.0, 1.0),
            (safety_state.speed_kmh / 100.0).clamp(0.0, 1.0),
            chaos_input.retina_temporal_var.clamp(0.0, 1.0),
            chaos.coherence,
            (1.0 / (1.0 + safety_state.wall_left)).clamp(0.0, 1.0),
            (1.0 / (1.0 + safety_state.wall_right)).clamp(0.0, 1.0),
            if sentinel_out.flags.intersects(
                crate::safety::SafetyFlags::ESP
                | crate::safety::SafetyFlags::LANE_KEEP
                | crate::safety::SafetyFlags::REVERSE_OUT
            ) { 1.0 } else { 0.0 },
            if sentinel_out.flags.intersects(
                crate::safety::SafetyFlags::TCS
                | crate::safety::SafetyFlags::AQUA
                | crate::safety::SafetyFlags::CORNER_LIM
            ) { 1.0 } else { 0.0 },
            if self.hard_veto_count > 0 { 1.0 } else { 0.0 },
            0.0, 0.0, 0.0, 0.0, // ai/gen/lap/fuel — wired later by caller via mutator if needed
        ];
        let w = self.gate.forward(&gi);
        self.gate_entropy_acc += crate::gate::GateBrain::entropy(&w);
        self.gate_entropy_n += 1;

        let cortex_iv = IntentVec { throttle: cortex_intent[0], steer: cortex_intent[1], brake: cortex_intent[2] };
        let reflex_iv = IntentVec { throttle: reflex_raw[0], steer: reflex_raw[1], brake: reflex_raw[2] };
        let sentinel_iv = sentinel_out.soft_proposal;

        let mixed = IntentVec {
            throttle: w[0]*cortex_iv.throttle + w[1]*reflex_iv.throttle + w[2]*sentinel_iv.throttle,
            steer:    w[0]*cortex_iv.steer    + w[1]*reflex_iv.steer    + w[2]*sentinel_iv.steer,
            brake:    w[0]*cortex_iv.brake    + w[1]*reflex_iv.brake    + w[2]*sentinel_iv.brake,
        };

        let final_iv = if let Some(veto) = sentinel_out.hard_veto {
            self.hard_veto_count = self.hard_veto_count.saturating_add(1);
            veto
        } else {
            if !sentinel_out.flags.is_empty() {
                self.soft_override_count = self.soft_override_count.saturating_add(1);
            }
            mixed
        };

        [
            final_iv.throttle.clamp(-1.0, 1.0),
            final_iv.steer.clamp(-1.0, 1.0),
            final_iv.brake.clamp(0.0, 1.0),
        ]
    }
```

- [ ] **Step 4.5: Compile check**

Run: `cargo check`
Expected: zero errors. Warnings about unused code OK.

- [ ] **Step 4.6: Add integration test stub**

Create `tests/trinity_integration.rs`:
```rust
use universe::{
    engine::SovereignEngine,
    safety::SafetyState,
    chaos::ChaosInput,
    resonance::ResonantNode,
};

fn calm_safety() -> SafetyState {
    SafetyState {
        speed_kmh: 30.0, yaw_rate: 0.0, slip_avg: 0.0, ttc_forward: 10.0,
        wall_left: 5.0, wall_right: 5.0, stall_timer: 0.0, pinado_timer: 0.0,
        rain: 0.0, curv25: 0.0, current_thr: 0.5,
    }
}

fn calm_chaos() -> ChaosInput {
    ChaosInput { yaw_rate: 0.0, slip_avg: 0.0, retina_temporal_var: 0.0 }
}

#[test]
fn decide_returns_clamped_outputs() {
    let mut engine = SovereignEngine::new();
    let nodes = vec![ResonantNode::new(1.0); 4];
    let input = [0_i8; 32];
    let out = engine.decide([0.5, 0.0, 0.0], &input, &calm_safety(), &calm_chaos(), &nodes);
    for v in out {
        assert!(v >= -1.0 && v <= 1.0);
    }
}

#[test]
fn hard_veto_on_imminent_collision() {
    let mut engine = SovereignEngine::new();
    let mut s = calm_safety(); s.ttc_forward = 0.1; s.speed_kmh = 50.0;
    let nodes = vec![ResonantNode::new(1.0); 4];
    let input = [0_i8; 32];
    let out = engine.decide([1.0, 0.0, 0.0], &input, &s, &calm_chaos(), &nodes);
    assert_eq!(out[0], 0.0);   // throttle vetoed
    assert_eq!(out[2], 1.0);   // brake forced
    assert!(engine.hard_veto_count >= 1);
}

#[test]
fn anti_stall_engages_reverse() {
    let mut engine = SovereignEngine::new();
    let mut s = calm_safety(); s.speed_kmh = 1.0; s.stall_timer = 0.5; s.wall_left = 0.4;
    let nodes = vec![ResonantNode::new(1.0); 4];
    let input = [0_i8; 32];
    let out = engine.decide([1.0, 0.0, 0.0], &input, &s, &calm_chaos(), &nodes);
    assert!(out[0] < 0.0); // reverse forced
}
```

If `SovereignEngine::new()` requires args, mirror the actual signature found in step 4.1 — adapt the test calls.

- [ ] **Step 4.7: Run integration tests**

Run: `cargo test --test trinity_integration`
Expected: 3 tests pass. If `SovereignEngine::new` signature differs, adjust the test call accordingly and re-run.

- [ ] **Step 4.8: Commit**

```bash
git add src/engine.rs src/specialist.rs tests/trinity_integration.rs
git commit -m "feat(engine): integrate Trinity (Gate+Reflex+Sentinel) decide() in SovereignEngine"
```

---

## Task 5: Drive motor.rs through `engine.decide()`

**Files:**
- Modify: `src/simulation/motor.rs` — replace direct `final_intent` reads with `engine.decide()` and supply state inputs

- [ ] **Step 5.1: Locate read site**

Run: `grep -n "final_intent\[" /home/leonardo/projects/LLM/Universe/src/simulation/motor.rs`
Confirm the lines at ~849-855 read `final_intent[0..3]` after `execute_cycle`.

- [ ] **Step 5.2: Build SafetyState + ChaosInput from existing locals**

Just before the read of `final_intent`, insert:
```rust
            // ----- Trinity decide -----
            let cortex_intent = [
                racer.engine.final_intent.get(0).copied().unwrap_or(0.0),
                racer.engine.final_intent.get(1).copied().unwrap_or(0.0),
                racer.engine.final_intent.get(2).copied().unwrap_or(0.0),
            ];
            let safety_state = crate::safety::SafetyState {
                speed_kmh: speed * 3.6,
                yaw_rate,
                slip_avg: ((slip_front_avg.abs() + slip_rear_avg.abs()) * 0.5).clamp(0.0, 1.0),
                ttc_forward: if speed > 0.5 { (fwd_dist / speed).min(99.0) } else { 99.0 },
                wall_left: wall_l_dist,
                wall_right: wall_r_dist,
                stall_timer: racer.stall_timer,
                pinado_timer: racer.pinado_timer,
                rain: env.rain,
                curv25: curv25,
                current_thr: cortex_intent[0],
            };
            let chaos_in = crate::chaos::ChaosInput {
                yaw_rate,
                slip_avg: safety_state.slip_avg,
                retina_temporal_var: retina_var.clamp(0.0, 1.0),
            };
            let oscillators: &[crate::resonance::ResonantNode] = &racer.engine.oscillators;
            let trinity = racer.engine.decide(
                cortex_intent,
                &slow_input, // reflex uses 32-i8 short context — same as slow_input here
                &safety_state,
                &chaos_in,
                oscillators,
            );
            let ti = trinity[0];
            let si = trinity[1];
            let bi = trinity[2];
```

Then **delete** the prior block that read `final_intent[0..2]` and computed `ti`, `si` directly.

**Variable provenance (resolve while editing):** the block above references `speed`, `yaw_rate`, `slip_front_avg`, `slip_rear_avg`, `fwd_dist`, `wall_l_dist`, `wall_r_dist`, `racer.stall_timer`, `racer.pinado_timer`, `env.rain`, `curv25`, `retina_var`, `slow_input`, `racer.engine.oscillators`. If a name is missing in the current motor.rs scope:
- For physical scalars (`yaw_rate`, `slip_*`, `wall_*`, `fwd_dist`, `curv25`): they exist in motor.rs already — use the actual local names.
- For `racer.stall_timer` / `pinado_timer`: if absent on `Racer`, add `pub stall_timer: f32` and `pub pinado_timer: f32` to the Racer struct in `src/simulation/racer.rs`, init to 0.0 in `Racer::new`, and update each tick:
  ```rust
  if speed * 3.6 < 2.0 && fwd_dist < 3.0 { racer.stall_timer += dt; } else { racer.stall_timer = 0.0; }
  if speed * 3.6 < 2.0 && (fwd_dist < 1.5 || wall_l_dist < 0.5 || wall_r_dist < 0.5) {
      racer.pinado_timer += dt;
  } else {
      racer.pinado_timer = 0.0;
  }
  ```
- For `retina_var`: if not present, compute as `let retina_var = (retina_now.iter().zip(racer.engine.last_retina.iter()).map(|(a,b)| (*a as f32 - *b as f32).abs()).sum::<f32>() / 68.0 / 127.0).clamp(0.0, 1.0);` after retina is collected. (Use existing field if available.)
- For `racer.engine.oscillators`: if `SovereignEngine` does not yet hold a `Vec<ResonantNode>`, add `pub oscillators: Vec<ResonantNode>` initialized to `vec![ResonantNode::new(1.0); 8]` in `SovereignEngine::new`, and tick them inside `execute_cycle` using `synchronize` against the average top-K instinct activation phase. If wiring oscillators meaningfully is too large here, ship with a static (always-coherent) array — `decide` will just return chaos.neural ≈ 0 and the gate still works.

- [ ] **Step 5.3: Compile + test**

Run: `cargo check && cargo test --lib`
Expected: zero errors; existing tests still pass.

- [ ] **Step 5.4: Smoke run headless 1 gen**

Run: `cargo run --release --bin universe -- --headless --gens 1 --swarm 80 --speed 16` (use the actual headless command from the project; if a different flag exists, substitute). Watch for crashes only — do not validate metrics yet.
Expected: race completes without panic.

- [ ] **Step 5.5: Commit**

```bash
git add src/simulation/motor.rs src/simulation/racer.rs src/engine.rs
git commit -m "feat(motor): drive intents through engine.decide() Trinity pipeline"
```

---

## Task 6: Mutate gate + reflex weights in spawn

**Files:**
- Modify: `src/evolution/spawn.rs` — extend champion-clone logic to also clone & mutate `gate.weights`, `gate.bias`, `reflex` weights.
- Modify: `src/evolution/dna.rs` — serialize gate + reflex into DNA payload.

- [ ] **Step 6.1: Extend DNA serialization**

In `src/evolution/dna.rs`, locate the binary serialize fn. After the cortex/slow_brain blocks, append a versioned block:

```rust
    // --- Trinity block (DNA v3.1) ---
    out.extend_from_slice(b"TRIN"); // magic
    out.extend_from_slice(&(engine.gate.weights.len() as u32).to_le_bytes());
    for w in &engine.gate.weights { out.extend_from_slice(&w.to_le_bytes()); }
    for w in &engine.gate.bias { out.extend_from_slice(&w.to_le_bytes()); }
    out.extend_from_slice(&(engine.reflex.weights_len() as u32).to_le_bytes());
    for w in engine.reflex.iter_weights() { out.extend_from_slice(&w.to_le_bytes()); }
```

Add to `SpecialistBrain` (`src/specialist.rs`) helper accessors if absent:
```rust
    pub fn weights_len(&self) -> usize { /* total weight count across the 3 sub-nets */ todo!("count steer+thr+brake weights") }
    pub fn iter_weights(&self) -> impl Iterator<Item=&f32> { /* concat iters */ todo!("chain steer+thr+brake weight slices") }
    pub fn iter_weights_mut(&mut self) -> impl Iterator<Item=&mut f32> { todo!("chain mut") }
```
Implement them concretely matching the actual sub-net layout in specialist.rs (do not leave `todo!` in committed code — replace with real chains, e.g. `self.steer.iter().chain(self.throttle.iter()).chain(self.brake.iter())`).

In the deserialize fn, after the cortex/slow read, peek the next 4 bytes; if equal to `b"TRIN"`, parse the trinity block and populate `engine.gate` + `engine.reflex`. If absent (legacy DNA), leave them at default — backward compat with `CHAMPION.dna`.

- [ ] **Step 6.2: Mutate gate + reflex in spawn**

In `src/evolution/spawn.rs`, find the `simple_mut_slot` block that mutates `instinct_layer.weights` and `slow_brain.layer.weights`. After the slow_brain block, add:
```rust
            // Mutate gate (f32 delta)
            for w in engine.gate.weights.iter_mut() {
                if rng.random_bool((rate_boosted * 0.5) as f64) {
                    *w += (rng.random::<f32>() - 0.5) * 0.2;
                }
            }
            for b in engine.gate.bias.iter_mut() {
                if rng.random_bool((rate_boosted * 0.5) as f64) {
                    *b += (rng.random::<f32>() - 0.5) * 0.1;
                }
            }
            // Mutate reflex (f32 delta)
            for w in engine.reflex.iter_weights_mut() {
                if rng.random_bool((rate_boosted * 0.5) as f64) {
                    *w += (rng.random::<f32>() - 0.5) * 0.2;
                }
            }
```

- [ ] **Step 6.3: Compile + lib tests**

Run: `cargo check && cargo test --lib`

- [ ] **Step 6.4: Commit**

```bash
git add src/evolution/spawn.rs src/evolution/dna.rs src/specialist.rs
git commit -m "feat(evolution): mutate gate + reflex weights; serialize trinity block in DNA"
```

---

## Task 7: Fitness penalties for safety dependency + entropy

**Files:**
- Modify: `src/evolution/pipeline.rs` — apply penalties at end-of-race fitness aggregation.

- [ ] **Step 7.1: Locate fitness aggregation**

Run: `grep -n "fn fitness\|race.fitness\|fitness =" /home/leonardo/projects/LLM/Universe/src/evolution/pipeline.rs | head -10`
Find where `racer.fitness` is finalized after race ends.

- [ ] **Step 7.2: Add penalty hook**

After the existing fitness computation (per-racer), insert:
```rust
            // Trinity safety penalties / rewards.
            let hv = racer.engine.hard_veto_count as f32;
            let so = racer.engine.soft_override_count as f32;
            let ent = if racer.engine.gate_entropy_n > 0 {
                racer.engine.gate_entropy_acc / racer.engine.gate_entropy_n as f32
            } else { 0.0 };
            const LN3: f32 = 1.0986123; // ln(3) = max entropy of 3-class softmax
            let entropy_norm = (ent / LN3).clamp(0.0, 1.0); // 1.0 = uniform, 0.0 = collapsed
            racer.fitness -= hv * 5.0;
            racer.fitness -= so * 0.3;
            // entropy bonus: 0 when collapsed, +20 when fully uniform — discourages collapse without forcing flatness
            racer.fitness += entropy_norm * 20.0;
            if racer.engine.last_chaos > 0.5 && racer.alive {
                racer.fitness += 10.0; // chaos recovery bonus
            }
```

- [ ] **Step 7.3: Compile + tests**

Run: `cargo check && cargo test --lib`

- [ ] **Step 7.4: Commit**

```bash
git add src/evolution/pipeline.rs
git commit -m "feat(evolution): trinity fitness penalties (hard veto, soft override) + entropy bonus"
```

---

## Task 8: Cargo features cleanup

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 8.1: Add `trinity_swarm` feature, deprecate `sentinel_brake`**

In `Cargo.toml`:
```toml
[features]
default = ["survival_pressure", "simple_elitism", "trinity_swarm"]
# Trinity swarm + Tesla-style safety. See docs/superpowers/specs/2026-04-25-trinity-swarm-safety-design.md
trinity_swarm = []
simple_elitism = []
# DEPRECATED: absorbed by Sentinel Tier 0 FCA. Enable only for A/B regression vs trinity_swarm.
sentinel_brake = []
gain_comp_steer = []
brake_symmetric = []
survival_pressure = []
```

- [ ] **Step 8.2: Verify build with default features**

Run: `cargo build --release`
Expected: clean build.

- [ ] **Step 8.3: Commit**

```bash
git add Cargo.toml
git commit -m "chore(cargo): default-on trinity_swarm, deprecate sentinel_brake feature"
```

---

## Task 9: A/B validation run (8 gens)

**Files:**
- None modified. This task validates results.

- [ ] **Step 9.1: Reset champion baseline**

Save current `CHAMPION.dna`:
```bash
cp CHAMPION.dna CHAMPION.dna.pre-trinity
rm CHAMPION.dna   # force fresh bootstrap
```

- [ ] **Step 9.2: Run 8 gens with trinity_swarm default ON**

Run: `cargo run --release --bin universe -- --headless --gens 8 --swarm 80 --speed 16`
(Substitute the project's actual headless flags if they differ.)

- [ ] **Step 9.3: Inspect race_reports.md**

Read tail of `race_reports.md`. Capture for each gen:
- NET max
- PINADO%
- BATIDA count
- hard_veto_count avg (if logged; if not, this is a follow-up to add to logs)

- [ ] **Step 9.4: Decision**

Compare vs simple_elitism baseline (champion 472m at gen 8):
- PASS: PINADO% < 40 AND champion progresses monotonically AND new champion > 600m at gen 8.
- FAIL: any criterion missed → file a follow-up "tuning" plan; do NOT roll back yet.

- [ ] **Step 9.5: Commit results**

```bash
git add race_reports.md melhorias/
git commit -m "test(trinity): A/B run 8 gens — record results"
```

If a `melhorias/trinity-ab.md` summary helps, write it now:
- gen-by-gen table (NET max, PINADO%, BATIDA, hard_veto avg)
- comparison vs simple_elitism v2 baseline
- next tuning suggestions

---

## Self-Review checklist (run before handing off)

- [ ] Spec coverage: every spec section has ≥ 1 task — Cortex preserved (Task 4 keeps existing instinct_layer), Reflex (Task 4), Sentinel (Task 1), Gate (Task 3), Chaos (Task 2), Fitness (Task 7), DNA evolution (Task 6), Cargo features (Task 8), A/B (Task 9). ✓
- [ ] No placeholders in committed code (Task 6 explicitly flags `todo!` must be replaced with real impl). ✓
- [ ] Type consistency: `IntentVec` used everywhere (Task 1 defines, Task 4 uses), `SafetyFlags` bitflags consistent, `GATE_INPUTS=16`/`GATE_OUTPUTS=3` used in Task 3 + 4, `[f32; 3]` reflex output matches `forward_specialist` signature. ✓
- [ ] DNA backward compat: legacy CHAMPION.dna without TRIN block still loads (Task 6.1). ✓

---

## Execution

Plan saved to `docs/superpowers/plans/2026-04-25-trinity-swarm-safety.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks
2. **Inline Execution** — execute tasks in this session with checkpoints

Which approach?
