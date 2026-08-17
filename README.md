# Universe

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/Rust-1.85%2B%20(edition%202024)-orange.svg)](Cargo.toml)
[![GPU: wgpu / WGSL](https://img.shields.io/badge/GPU-wgpu%2024%20%C2%B7%20WGSL-brightgreen.svg)](src/gpu)

**Universe is a GPU-first neuroevolution simulator written in Rust.** Populations of
creatures with a hybrid brain — a per-channel KAN pre-filter, a CfC recurrent stage, a
fast `BitLinear` layer and a float cortex layer — drive vehicles around procedurally
generated tracks. A genetic algorithm selects and mutates them between generations, and
every run emits heavy telemetry (per-generation champion genomes, replays, activation
statistics, ablation reports) because the point of the project is *measuring* what the
brain actually allocates, not just watching it drive.

---

## Why it is technically interesting

The whole per-tick loop — retina, perception, both brains, physics — runs in WGSL on a
**single `wgpu` device, with no window, no surface and no swapchain**. That means the
simulator runs headless under `env -u DISPLAY`, in cron, on a machine without a graphics
session (`src/gpu/mod.rs`, module header). Bevy is still a build dependency — `wgpu` is
pulled in through its tree, and `Cargo.toml` selects a trimmed feature set — but at runtime
it is only a viewer over the compute core, never something the core has to go through.

What the tick actually does, in order (`src/bin/gpu_evolve/context.rs`, `GpuCtx::tick`,
one `CommandEncoder`, one `submit`):

| Stage | Kernel(s) | What it computes |
|---|---|---|
| Retina | `retina.wgsl` | 14 rows x 126 columns = 1 764 DDA raymarches **per creature** (`src/config.rs:255-257`) |
| Perception | `perception.wgsl`, `perception_temporal.wgsl`, `perception_motion.wgsl` | packs the 192-channel `sensory_wave` (`src/config.rs:309`) |
| Retina digest | `retina_digest.wgsl` | perception-CfC state, `min_wall_ahead`, edge rows — replaces a 340 KB/tick readback |
| Fast brain | `brain.wgsl` | KAN spline per channel -> CfC -> `BitLinear` 192->96 -> tanh x127 -> i8 -> `FloatLinear` 96->48 -> `final_intent[96]` |
| Slow brain | `slow_gather.wgsl`, `slow_brain.wgsl` | CfC(64) -> Dense 64->16 -> Dense 16->3 = `[throttle, steer, brake]`, fires every 8 ticks |
| Physics | `physics_a/b/c.wgsl` | suspension + drivetrain, then the Pacejka tyre loop, then collision + integration |

Three details a newcomer will want to know, all of them measured rather than assumed:

- **The retina is the hot spot, not the brain.** Over 2 440 PERF samples the split was
  retina 57.0%, physics and the rest 26.9%, brain 14.3%, perception 1.9%
  (`src/gpu/mod.rs` header). That is why the retina was the first thing ported.
- **Host<->device traffic per tick is about 2.6 kB** for 60 creatures — poses up,
  motor intent down. Weights and the CfC recurrent state stay resident in VRAM across
  ticks and only come down between generations (`src/gpu/core.rs` header, byte budget).
- **Physics is split into three kernels because it had to be.** The single 756-line
  `physics.wgsl` crashed the RADV/ACO shader compiler on NAVI22; bisection
  (`src/bin/fisica_bisect.rs`) isolated the Pacejka block as the trigger, so the pipeline
  is A (suspension/power) -> B (tyres) -> C (collision/integration), each staying under
  ~50 simultaneous local f32 (`src/gpu/physics.rs` header).

The non-negotiable rule of the GPU port is that a kernel must produce the *same* result as
the CPU reference. `gpu_validate` locks that down for five subsystems, and every GPU claim
in `docs/` was produced by a run that passed it first.

Evolution itself stays on the CPU, deliberately: it runs once per N ticks, so it is not on
the hot path (`src/gpu/evolution.rs` header). Selection is tournament (k=3), mutation is
Gaussian noise per weight, elitism carries the best `n_elite` genomes through unmutated,
and L1 is spectrally re-scaled at birth so the sensory signal is not amplified before the
first mutation.

---

## Quickstart

### Requirements

- **Rust 1.85.0 or newer**, edition 2024 (`Cargo.toml`).
- **A GPU with a working Vulkan driver.** The project talks to the GPU through `wgpu` 24,
  so any Vulkan-capable adapter should work. ROCm is *not* required — it is simply what
  the author happens to have installed alongside (AMD RX 6750 XT / RADV).
- **Linux is the tested platform.** The graphical entry point needs X11; every GPU compute
  tool is headless and runs fine with `env -u DISPLAY`.
- On some AMD/Mesa setups the Vulkan loader picks the wrong ICD. Several binaries set
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` for you when it is unset
  (for example `src/bin/gpu_validate.rs:102-108`); export it yourself if a tool cannot
  find an adapter.

> **Known gap — a fresh clone does not build yet.** `Cargo.toml` declares two path
> dependencies outside this repository, `iara-perception` and `iara-cfc` under
> `../iara-runtime/`, and that directory is not tracked here and is not a submodule.
> TODO: vendor those crates, publish them, or add them as a submodule so external
> contributors can build. Until then the build only works on a machine that already has
> the sibling `iara-runtime` tree.

### Build

```bash
cargo build --release
```

### Check that the GPU kernels still match the CPU

With no arguments `gpu_validate` runs all five subsystems; each flag restricts it to one
(`src/bin/gpu_validate.rs:111-116`).

```bash
env -u DISPLAY ./target/release/gpu_validate              # retina + core + brain + physics + perception
env -u DISPLAY ./target/release/gpu_validate --fisica     # physics only
```

### Run an evolution

`gpu_evolve` is the production path: N ticks on the GPU, evaluate, select, mutate, repeat.

```bash
env -u DISPLAY nice -n 5 ./target/release/gpu_evolve \
    --seres 60 --ticks 8000 --gens 10 \
    --same-start --seed 1 \
    --out /path/to/run_dir
```

- `--seres` population size (default 60), `--ticks` ticks per generation (default 8 000,
  which is 125 s at 64 Hz), `--gens` number of generations (default 100).
- `--same-start` puts every creature on the same starting waypoint. Without it each
  creature is born somewhere different and fitness ends up measuring the *starting spot*
  rather than the brain.
- `--out` chooses where the run writes. **It defaults to `.`**, so pass it unless you want
  `evolve_best_genNNNN.bin` and `genNNNN.replay` landing in your working directory.
- `--retomar` resumes from the most recent `evolve_best_gen*.bin` checkpoint.
- Unknown flags are **silently ignored** (`src/bin/gpu_evolve/main.rs:743`, the `_ => {}`
  arm) and there is no `--help`. Check the `match` block at
  `src/bin/gpu_evolve/main.rs:541-744` for the full, authoritative list before inventing a
  flag name.

Useful extras:

| Flag | Effect |
|---|---|
| `--sigma <f>` / `--p-mutate <f>` | mutation magnitude (default 0.05) and per-weight mutation probability (default 0.02) |
| `--n-elite <n>` / `--tourney <k>` | elitism size and tournament size (both default 3) |
| `--fitness <mode>` | `checkpoint` (default), `prog`, `dist`, `centro` |
| `--pistas "9001,9002"` | rotate a set of training track seeds, deterministically by generation |
| `--pista-held-out <seed>` / `--held-out-cada <k>` | measure generalisation on a track selection never optimises against |
| `--ativacao` / `--ativacao-n <n>` / `--regioes` | L1/L2 activation statistics, sampled every n ticks, optionally split by track curvature |
| `--painel` / `--painel-porta <n>` | live HTTP dashboard (default port 3200); no thread is spawned without the flag |

### Measure track difficulty (CPU only, no GPU needed)

```bash
./target/release/pistas               # measured difficulty of each track seed
./target/release/pistas --curriculo   # build a difficulty ramp from the measurements
```

### Tests

```bash
cargo test --release --bin gpu_evolve       # genome round-trip, track rotation, pilot ordering
cargo test --release --test wgsl_validation # parses and type-checks the WGSL, no GPU required
```

`wgsl_validation` is the one worth running on every change to `src/gpu/*.wgsl`: shaders are
embedded with `include_str!`, so `cargo build` never sees a WGSL syntax error — without
this test the first sign of trouble is a runtime failure on a real GPU.

### One warning that has cost real training runs

**Never run `./target/release/universe` from the repository root.** It reads *and
overwrites* `CHAMPION.dna` in the working directory, so even a one-generation smoke test
crowns a weak champion over a trained one. Use the wrapper, which copies the champion and
the config into an isolated directory first:

```bash
scripts/rodar.sh <label> --headless --gens 1 --swarm 30
```

---

## Repository map

### Top level

| Path | What lives there |
|---|---|
| `src/` | all Rust sources (library + binaries) |
| `benches/` | Criterion micro-benchmarks (`bitlinear`) |
| `tests/` | integration tests: ablation controls, Trinity wiring, WGSL static validation |
| `scripts/` | shell and Python drivers for experiment batteries; `rodar.sh` is the safe launcher |
| `tools/` | Python analysis helpers for battery output |
| `docs/` | experiment reports, plans and archived crates (see the note below) |
| `.claude/` | configuration for the AI agents that work in this repo |
| `telemetry/`, `target/` | generated output, git-ignored |

### `src/`

| Path | What lives there |
|---|---|
| `src/gpu/` | the compute core: WGSL kernels plus their `wgpu` host code (retina, perception, both brains, physics, cortex sheet, genome I/O) |
| `src/bin/` | 29 command-line tools — evolution drivers, validators, DNA inspectors, replay analysers |
| `src/simulation/` | the per-tick engine used by the Bevy path (`motor.rs`) and the `Racer` component |
| `src/evolution/` | genetic algorithm pipeline, DNA serialisation (`dna_genesis.rs` is the source of truth), spawn, telemetry |
| `src/math/` | activations, linear algebra, statistics, quantisation, thresholds, and the measured "laws" the project relies on |
| `src/perception/` | the sensory-channel dictionary and the CPU-side retina |
| `src/analysis/` | activation, contextual and per-region statistics collected during runs |
| `src/brain/` | brain file format |
| `src/safety/` | the Sentinel emergency-brake layer |
| `src/assets/` | the standalone HTML replay player |

### Key root-level modules

`config.rs` is the constants hub and the source of truth for architecture sizes;
`layers.rs` (`BitLinear`), `float_layers.rs` (`FloatLinear`), `dense_layer.rs`,
`cfc.rs` (`CfCCell`) and `kan.rs` are the brain's building blocks; `engine.rs` assembles
them into `SovereignEngine` plus `SlowBrain`; `world.rs` and `pista_gen.rs` generate the
voxel world and the tracks; `physics.rs` is the CPU reference the GPU kernels are validated
against; `neuro.rs` holds the sensory-channel wiring and priming ablation.

`ARQUITETURA.md` (Portuguese, being replaced by `ARCHITECTURE.md`) carries a deeper module
map, the audited duplications, and the rules for not breaking the DNA format.

---

## Where the science lives

The interesting part of this repository is not the code, it is what was measured with it.
`docs/` holds the experiment reports, and `LEIA_PRIMEIRO.md` is the entry-point summary of
the neural-allocation battery (ablation over a trained champion: the gate hands 37% of its
weight to the cortex, which carries 94.7% of the performance — a 22x mismatch between what
the gate says and what actually matters).

Other reports worth opening: `docs/PLANO_EVOLUCAO.md` (the evolution plan and its measured
findings), `docs/REGIOES_CEREBRAIS.md` (brain regions), `docs/ESCADA_DIFICULDADE.md`
(difficulty curriculum), `docs/MATH_INVENTORY.md` (every equation in the codebase and where
it is used), `docs/CEREBRO_VIRTUAL.md` and `docs/DESTILABILIDADE.md`.

> **`docs/` is still written in Portuguese and is being migrated to English.** So are many
> in-code comments, some identifiers, and the `gpu_evolve` command-line flags themselves
> (`--seres`, `--ticks`, `--pistas`, ...). The flag names in this README are the ones that
> exist in the code today; they will be renamed as part of the migration, and this file
> will be updated with them.

---

## Contributing

The project standard is **English everywhere** — identifiers, comments, doc-comments, log
messages, test names and commit messages. This supersedes the older rule that kept comments
and output strings in Portuguese (revoked 2026-08-17).

- [CONTRIBUTING.md](CONTRIBUTING.md) — how to set up, what a good PR looks like, what to run
  before opening one.
- [ARCHITECTURE.md](ARCHITECTURE.md) — module map, sources of truth, and the invariants you
  must not break.
- [docs/STYLE.md](docs/STYLE.md) — naming, comment and file-size conventions.

Two habits this codebase cares about more than most:

1. **Comments carry measurements.** Many of them record a real bug that cost GPU hours —
   "the tank was sized per lap but refilled per sector", "76 of 96 neurons have period > 1".
   When you touch such a comment, keep the number. Translate the prose, never delete the
   evidence.
2. **Claims come with the command that produced them.** If you state that something got
   faster or drives further, say which run measured it and how many seeds it used.

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Leonardo Souza.
