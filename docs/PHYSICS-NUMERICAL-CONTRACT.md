# Resident taxi numerical profiles

The default remains `explicit-v1-instant-tcs` for existing checkpoint comparisons.
`--organism-stable-wheels` explicitly selects `implicit-2stage-v2-tcs-gain-100ms`.
It requires resident taxi and the corrected power-conjugate tyre frame. Do not
compare checkpoints across these profiles without recording the change.

## Executed boundary

`physics_a.wgsl` computes wheel-side requested drive torque and normal loads.
`physics_b.wgsl` computes four tyre forces and wheel angular speeds.
`physics_c.wgsl` integrates chassis velocity/position and collision response.
`organism_orientation.wgsl` rotates the body frame and feeds acquired state back.
The brain, action mapper, task judge and recorded 64 Hz clock are unchanged.
There are no added host readbacks or whole-world substeps. Only the local wheel
solve uses two half steps inside kernel B.

## Wheel equation (bit 11)

The existing explicit wheel step evaluates force at the old angular speed.
The optional solver uses backward Euler:

`I * (omega_next - omega_old) / dt = drive_torque - radius * F(omega_next) - brake_torque`.

The existing Pacejka longitudinal force and combined-friction circle define `F`.
Chassis velocity and normal load are held within this solve. Brake torque opposes
rotation and can hold zero through a Coulomb impulse interval. Each half step uses twenty bounded bisection iterations to solve a bracket
enclosing the residual root. Chassis force is the average of both half-step
forces; recorded slip and angular speed come from the final half step. No claim
of a globally unique root or fully implicit chassis integration is made.

## Traction-control response (bit 12)

The existing slip controller still determines the requested torque reduction.
TCS gain cuts are immediate. Recovery of that gain uses
`alpha = 1 - exp(-dt / 0.100 s)`. Pedal/engine/gear requests are multiplied by the
gain without another smoothing stage. Braking/neutral/shift interruption cannot
retain positive drive, and a non-slipping pedal request is not attenuated.
The 100 ms value is an explicit engineering parameter, not an evolved gene or a
measured biological constant. This profile currently applies to rear-wheel TCS.

Both bits are recorded in `physics_config_flags`; observations additionally label
`wheel_integration` and `tcs_recovery_ms`. Generation metadata names the profile.
The 128-byte intermediate stride has not grown. Two previously reserved words
now hold the acquired TCS gain and initialization marker, explicitly owned by
kernel A. Bulk reset and individual birth clear them. Runtime checkpoints v4
include `physics.mid`; old runtime snapshots are rejected, while DNA checkpoints
remain compatible. `drive_torque_out` remains applied wheel-side torque, also
used by existing energy accounting.

## Evidence and limits

The September 25 controlled GPU experiment exposed strong timestep sensitivity
and quarter-turn divergence in the original integration. An FMA-only orientation
change failed and was reverted. Implicit wheel integration alone improved step
sensitivity but did not pass the rotation tolerance; immediate TCS recovery still
amplified perturbations. The combined profile passed the declared tests:

- Identical native neural/action inputs in four rotated poses, 128 ticks,
  position/velocity/heading component differences below .002.
- Native acceleration, brake, passive steering, acquired-state reset and energy.
- Immediate torque cutoff after switching from acceleration to neural braking.
- One fixed two-second manoeuvre at 1/64, 1/128, 1/256 and 1/512 s: successive
  trajectories approach each other; coarse versus finest differs by under .1 m
  in position and .2 m/s in velocity.
- A non-slipping pedal request matches TCS-disabled drive torque.
- GPU checkpoint restoration reproduces continuation exactly, and birth resets
  only the selected slot’s acquired TCS state.
- Newborn profile inheritance preserves the parent execution flags and dt while
  retaining child morphology/capacity genes; dropping that inheritance fails the
  GPU birth assertion.
- Restoring explicit wheels or instantaneous TCS separately breaks the rotation
  test at the intended assertion. These negative controls are mandatory.

Run `scripts/check_native_flow_gpu.py` on the real GPU for the source-bound
receipt. Details and failed interventions live in
[the investigation](experiments/REPLAY-CAUSAL-AUDIT-20260925.md). These fixtures
are bounded numerical contracts, not proof of city driving, universal
convergence, tyre calibration or learned deliveries. Frozen-policy behavioral
comparisons and subsequent learning remain separate gates.
