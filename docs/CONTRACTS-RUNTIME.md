# Universe runtime contracts — 2026-09-22

## Implemented boundary

`brain_interface::MotorCommand` → `PhysicsStateGpu` → `PhysicsGpu::step` → kernels A/B/C → `PhysicsOutputGpu` → body state.

`src/contracts.rs` provides private validated quantities and explicit unit conversions. A declared `km/h` input cannot enter `SpeedMps::from_unit`; use `from_kmh`. A raw number mislabeled as m/s cannot be identified from its magnitude alone. Existing transport fields retain their ABI; not every producer is yet migrated to typed quantities.

Both `step` and `step_raw` validate applied motor ranges, finite body fields, positive mass/time/inertias/dimensions, temperatures and the stall flag before upload. Invalid commands are rejected, not clamped. Readback mapping failure is handled; all outputs are validated before any body receives the result. The training context propagates `step == false` rather than applying the returned physics state. No extra GPU readback was added; validation scans the existing host input/output.

Static Naga tests compare field names, order, offsets, sizes and stride in the three production shaders. Rust input is 360 bytes and the serialized WGSL stride is 368 bytes: padding is intentional, checked with two records. Intermediate buffers and other kernel families are not covered by this new ABI check.

## Reproduce

Metadata/domain tests only, no CPU simulation:

```sh
CARGO_BUILD_JOBS=1 cargo test --offline --lib contract -- --test-threads=1
```

Result: 9 passed. Evidence: `/tmp/universe-contract-tests.log`.

Small real GPU battery:

```sh
env -u DISPLAY VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json MESA_VK_VERSION_OVERRIDE=1.3 CARGO_BUILD_JOBS=1 cargo test --offline --lib straight_line_brake_and_contact_diagnostics -- --ignored --exact gpu::physics::physics_contact_tests::straight_line_brake_and_contact_diagnostics --nocapture --test-threads=1
```

Result: 1 passed, AMD Radeon RX 6750 XT (RADV NAVI22), 0.44 s test runtime. Evidence: `/tmp/universe-contract-gpu.log`. Checks invalid batch rejection without state mutation, continued valid dispatches, braking versus coasting, wall contact, impact energy and diagnostic on/off equivalence. The sandbox initially had no GPU device; the successful run used authorized host access. No CPU physics fallback.

## Remaining scope

This closes a runtime boundary, not the complete organism plan. Sensor/brain/genome/organ/replay contracts, all shader buffer layouts, full adoption of quantity types, training failure persistence and production integration gates still require implementation or audit. The GPU fixture proves braking mechanics, not learned braking, taxi readiness or evolutionary progress. Existing host/GPU transfers remain; this is not a GPU residency optimization.
