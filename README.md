# Universe

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/Rust-1.85%2B%20(edition%202024)-orange.svg)](Cargo.toml)
[![GPU: wgpu / WGSL](https://img.shields.io/badge/GPU-wgpu%2024%20%C2%B7%20WGSL-brightgreen.svg)](src/gpu)

**A GPU-first neuroevolution simulator in Rust.** Populations of creatures, each with a
DNA genome and a hybrid recurrent brain, drive vehicles through procedurally generated
worlds. A genetic algorithm selects and mutates them between generations. The whole
per-tick loop (retina, perception, both brains, vehicle physics) runs as WGSL compute
kernels on one headless `wgpu` device. The project's focus is **measurement**: every run
records champion genomes, replays and activation statistics, so claims about what the
brain learned can be checked against the replay of a single creature on a single tick.

Status: research code, actively developed. The racing track is the mature scenario; the
taxi/city scenario and body co-evolution are experimental (see [Status](#status)).

---

## Highlights

All numbers below were measured on the hardware listed further down. Each one comes
with the baseline it was compared against.

- **Full tick on the GPU: 0.63 ms per tick vs 6.25 ms on the CPU** for 60 creatures
  (retina + perception + brain + physics, `gpu_loop` benchmark, 2026-08-03). Porting
  the whole loop, not single kernels, is what paid: the submit/readback boundary was the
  cost.
- **GPU kernels checked against the CPU reference.** Fast brain max abs error 1.79e-7
  (signal magnitude 3.94e-1), slow brain worst error 1.788e-7 over 9,600 comparisons,
  retina 99.9965% of pixels identical (max difference 1 level). `gpu_validate` runs
  these gates. Physics is tolerance-based, not bit identity.
- **Bit-exact determinism.** Two 30-generation runs × 48 creatures × 12,000 ticks
  produced byte-identical `driving.csv` files (md5). This is what makes paired A/B
  comparisons valid.
- **Driving learned from scratch.** From a random population (median 2.81 m, 100% of
  creatures stationary) to 4,086 m (≈2.5 laps) by generation 29, with no hand-written
  pilot. Braking emerged with training in 8/8 seeds: 4–10% of live ticks at gen 0 →
  19–66% at gen 29, including runs with no braking incentive.
- **Readback removed from the hot path.** Moving the retina digest onto the GPU cut a
  340 KB/tick download and took a generation from 35.1 s to 19.2 s (1.83×).
- **Cost model for scaling.** Tick time ≈ 0.86 ms + 0.032 ms × creatures (sweep at
  12/96/384/768 creatures). The fixed term is launch latency, so one process with many
  creatures beats many small processes (100 creatures in 1 process: 4.4 ms/tick vs
  10 × 10: 11.6 ms/tick).

---

## How it works

```
 genome (DNA, TLV format)
        │  decoded at birth
        ▼
 ┌──────────────────────────── one wgpu CommandEncoder, one submit per tick ─────────┐
 │ retina.wgsl        14 × 126 DDA raymarches per creature                           │
 │ perception*.wgsl   → 192-channel sensory wave                                     │
 │ retina_digest.wgsl → wall distance, edge rows (replaces a 340 KB/tick readback)   │
 │ brain.wgsl         KAN spline per channel → CfC → BitLinear 192→96 → FloatLinear  │
 │ slow_brain.wgsl    CfC(64) → Dense 64→16 → 16→3 = [throttle, steer, brake], /8 tk │
 │ physics_a/b/c      suspension + drivetrain → Pacejka tyres → collision/integrate  │
 └───────────────────────────────────────────────────────────────────────────────────┘
        │  ~2.6 kB/tick host↔device for 60 creatures (poses up, intent down)
        ▼
 evolution on the CPU, once per generation: tournament (k=3), Gaussian mutation,
 elitism, spectral rescaling of L1 at birth
```

Details worth knowing, all measured:

- **The retina is the hot spot, not the brain.** Over 2,440 PERF samples: retina 57.0%,
  physics and the rest 26.9%, brain 14.3%, perception 1.9% (`src/gpu/mod.rs` header).
- **Weights and recurrent state stay in VRAM** across ticks and come down only between
  generations (`src/gpu/core.rs`).
- **Physics is three kernels because one crashed the compiler.** The single 756-line
  physics shader crashed the RADV/ACO compiler on NAVI22. Bisection
  (`src/bin/fisica_bisect.rs`) isolated the Pacejka block, so the pipeline is split into
  A → B → C (`src/gpu/physics.rs`).
- **The brain is float, not ternary.** An early ternary cortex did not compose; the
  runtime cortex is F32. Ternary `BitLinear` remains only where it measured well.

---

## Results

| Experiment | Result | Baseline / control |
|---|---|---|
| Full tick on GPU (60 creatures) | 0.63 ms/tick | CPU 6.25 ms/tick |
| Retina kernel alone | 31.7× | CPU reference, same pixels (99.996%) |
| retina_digest on GPU | 19.2 s/generation | 35.1 s with readback |
| Determinism | byte-identical over 30 gens × 48 × 12k ticks | repeated runs, 2 seeds |
| Evolution from random population | 4,086 m (2.5 laps) at gen 29 | 2.81 m at gen 0 |
| Braking emerges (B1 battery) | 19–66% of live ticks at gen 29, 8/8 seeds | 4–10% at gen 0 |
| Single starting point (`--same-start`) | median 181 → 370 m, 4/4 seeds | per-creature start points (fitness measured the start, not the brain) |
| Track-width unit bug fixed | track 3× too wide found and fixed | `pista_check`: 14 m free per side after fix |
| Sensor audit | 49 of 192 channels dead, 13 with no source in any kernel | found by `--audit-vision`, then fixed |
| FlyWire connectome wiring (`--fly-brain`) | sugar→MN9 72.4 ± 5.0 Hz, 30/30 trials | shuffled targets/weights: 0 Hz, 0/30 |

Most of the reports behind these rows are in [`docs/`](docs/) (still largely in
Portuguese, being migrated). Good entry points:
[validation workflow](docs/VALIDATION-WORKFLOW.md),
[kernel timing](docs/KERNEL_TIMING.md), [math inventory](docs/MATH_INVENTORY.md),
[body co-evolution validation](docs/BODY-COEVOLUTION-VALIDATION.md),
[fly connectome results](docs/FLY-PROGRAM-RESULTS.md).

---

## What didn't work (and why)

Negative results are kept on purpose. Each one changed what was tested next.

- **No evolved population learned to complete a taxi delivery.** In the taxi "school"
  curriculum, approach and stopping improved, but every evolved population delivered
  zero passengers. The only deliveries came from a scripted witness pilot, which is
  capped at two deliveries by design. That pilot is a viability control, not a learned
  policy. The scenario was paused.
- **Sector memory (a "hippocampus") cost laps.** Runs with ≥2 laps: 32/28/34 of 40 with
  the memory organ vs 39/39/38 for the control. A paired ablation showed that *reading*
  the memory was the cost. The track is fully observable, so memory had nothing to add.
- **A neural gearbox broke a working population.** Turning on neural gear shifting in
  the same population dropped lap finishers from 39 to 8 (2 seeds). Higher speed outran
  the perception: the retina goes blind above ~28 m/s. The locked 2nd gear had been
  protecting it by accident.
- **Two AI-agent "proofs" were fabricated and caught.** A forced 0.3 throttle "panic
  escape" accounted for the whole 2.7× gain (1,744 m with the crutch vs 650 m without),
  and an injected initial speed gave six creatures identical distances to 4 decimals.
  Both were discarded. Every GPU claim since then has to pass `gpu_validate` and a
  replay check.
- **An ablation number had to be requalified.** Removing the cortex dropped performance
  by 94.7 ± 2.6% (5 reps), but this mixes the cortex's contribution with the disruption
  of the parts that co-adapted to it (sentinel vetoes went 2,583 → 15,171). Ablation is
  only clean when removing a part does not miscalibrate the rest.

---

## Hardware and stack

| | |
|---|---|
| GPU | AMD Radeon RX 6750 XT (12 GB, Navi22 / gfx1031), RADV Vulkan driver |
| CPU | AMD Ryzen 7 5800X3D |
| Language | Rust 1.85+, edition 2024 |
| GPU API | `wgpu` 24 + WGSL compute shaders (any Vulkan adapter should work; ROCm is not required) |
| Viewer | Bevy 0.16 (optional, graphical entry point only; compute runs headless) |

---

## Build and run

Requirements: Rust 1.85+, a GPU with a working Vulkan driver, Linux (tested). The
compute tools are headless and run under `env -u DISPLAY`. On some AMD/Mesa setups the
Vulkan loader picks the wrong ICD; export
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` if a tool cannot find an
adapter. Local dependency crates are vendored under [`vendor/`](vendor/).

```bash
cargo build --release
```

**Check that GPU kernels still match the CPU reference** (no flag = all five subsystems):

```bash
env -u DISPLAY ./target/release/gpu_validate            # retina + core + brain + physics + perception
env -u DISPLAY ./target/release/gpu_validate --fisica   # physics only
```

**Run an evolution** (`gpu_evolve` is the production path):

```bash
./target/release/gpu_evolve --help    # full flag table

env -u DISPLAY nice -n 5 ./target/release/gpu_evolve \
    --beings 48 --ticks 12000 --gens 30 \
    --same-start --seed 1 \
    --out runs/my-first-run
```

- `--beings` population size (default 60), `--ticks` ticks per generation, `--gens`
  number of generations.
- `--same-start` puts every creature on the same starting point. Without it, fitness
  partly measures where each creature was born.
- `--out` is where checkpoints and replays go. It defaults to `.`, so always set it.
- `--resume` continues from the checkpoint in `--out`; `--tracks 9001,9002` rotates
  training tracks; `--track-held-out <seed>` measures generalisation on a track that
  selection never sees. Older Portuguese flag names are still accepted as aliases.

**Tests:**

```bash
cargo test --release --bin gpu_evolve      # genome round-trip, track rotation, flags
cargo test --release --lib wgsl_static     # parses and type-checks all WGSL, no GPU needed
cargo test --release -p taxi-world         # taxi world crate
```

Shaders are embedded with `include_str!`, so `cargo build` never sees a WGSL syntax
error. Run `wgsl_static` after every shader change. CI (`.github/workflows/ci.yml`) runs
fmt, build, the test suites above and an English-only language check.

**Warning:** do not run `./target/release/universe` (the graphical binary) from the
repository root. It reads and overwrites `CHAMPION.dna` in the working directory. Use
`scripts/rodar.sh <label> ...`, which runs in an isolated directory.

---

## Repository map

| Path | What lives there |
|---|---|
| `src/gpu/` | compute core: WGSL kernels and their `wgpu` host code |
| `src/bin/` | command-line tools: `gpu_evolve`, `gpu_validate`, replay/genome/DNA inspectors, track tools |
| `src/evolution/` | genetic algorithm, DNA serialisation (`dna_genesis.rs` is the source of truth) |
| `src/math/`, `src/perception/`, `src/analysis/` | maths core, sensory channels, activation statistics |
| `crates/taxi-world/` | Rust port of the taxi/city world from [neuromod-lab](https://github.com/leonardoosouza1/neuromod-lab) |
| `vendor/` | vendored CfC, perception and helper crates |
| `tests/`, `benches/` | integration tests, Criterion benchmarks |
| `scripts/`, `tools/` | experiment drivers and analysis helpers |
| `docs/` | experiment reports, specs and validation records |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module map and invariants,
[CONTRIBUTING.md](CONTRIBUTING.md) and [docs/STYLE.md](docs/STYLE.md) for conventions.

---

## Status

- **Stable:** racing scenario, GPU tick, `gpu_validate` parity gates, deterministic
  evolution, replay and genome tooling.
- **Experimental:** body co-evolution (engine/brake genes selected with the brain;
  opt-in, validated as a first slice only), taxi/city scenario (`crates/taxi-world`,
  school curriculum; no learned delivery yet), FlyWire connectome as a fixed brain.
- **Ongoing:** translating `docs/` and remaining comments/flags to English.

Contribution rule that matters most here: claims come with the command, seeds and
artifact that produced them, and comments that record a measurement keep the number.

## License

MIT, see [LICENSE](LICENSE).
