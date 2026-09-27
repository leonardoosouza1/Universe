# Vila Viva notebook CPU runtime

The owner authorized CPU execution for the Minecraft notebook migration on
2026-09-26. This scoped runtime is separate from the legacy GPU-only taxi gates.

`crates/vila-shadow-cpu/` is an independent Cargo workspace with no GPU, Bevy or
taxi dependency. It executes the approved **Vila Viva Shadow GRU**, not the legacy
Universe KAN/CfC/BitLinear brain. The two model formats are not interchangeable.

Flow: Paper recorded body signals -> 23 ordered world-clock/per-row features ->
Rust Linear/tanh + GRU64 -> activity probabilities, displacement, sleep -> panel.
The panel owns a separate hidden state per villager and resets on sequence gaps.
The vanilla Minecraft Brain remains the actuator; this is inference portability,
not a learned-controller claim.

The canonical implementation lives here. The standalone notebook release vendors
this crate as `cpu-shadow/`, together with a source manifest and approved weights:
https://github.com/leonardoosouza1/vila-viva-server (private).

Validation commands (one build job, one numerical CPU thread):

```bash
CARGO_BUILD_JOBS=1 cargo build --locked --release --manifest-path crates/vila-shadow-cpu/Cargo.toml
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ../mc-lab/vila-viva/brain/test_cpu_parity.py /path/cpu-model.json --binary crates/vila-shadow-cpu/target/release/vila-shadow-cpu
```

The migration measured 120 recurrent steps with independent Torch/Rust states,
three resets and six rejection cases on approved weights. Max absolute error
1.3709068298339844e-6, acceptance 2e-5. This proves bounded numerical parity;
it does not prove new behavior or notebook energy/thermal performance.

Preserved unrelated work: `src/gpu/school_tests.rs` and the portfolio document.
