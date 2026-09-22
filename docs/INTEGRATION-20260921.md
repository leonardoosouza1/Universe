# Executable training-to-panel integration — 2026-09-21

Authorized scope: close the real-run export/API/panel path before increasing experimental complexity.
Numerical simulations/checks remain real-GPU-only. No CPU brain forward or numerical test suite.

| Order | Producer | Device / data | Consumer | Frequency |
|---|---|---|---|---|
| 1 | gpu_evolve / DrivingDiagnostics::sample | GPU staged snapshots -> selected observation.jsonl, schema 2, lifetime f32[512] | trace producer | sampled ticks |
| 2 | genome_io::save_genome | host UGEN v2/v3/v4; exact observed genome_file per life | genome_export canonical loader | generation/birth |
| 3 | trace/producer.py::produce | host serialization, bytes + manifest dimensions/hashes | lab_api trace file endpoint | once per trace/life |
| 4 | lab-ui stores/run + MicroView | requested run/gen/being/profile | trace loader, panels | user selection |
| 5 | lifetime panel | recorded P/E/age/residual | inspection only, no inferred anatomy | scrub tick |

Confirmed breaks before editing: separate old exporter worktree cannot read new DNA;
MicroView calls openDefaultFixture regardless of requested run; producer omits resident
lifetime state and refuses multiple observed genomes. Preserve the refusal to mix weights:
represent individual lives separately. No fallback to another run on missing trace.

Validation: export actual existing GPU run (including two-life selected slot), check
manifest and byte/hash ownership, request through authenticated API, compile UI with one
job/core, and inspect real UI if browser tools permit. No claim of learned improvement.

## Delivered and checked

- Ported `genome_export`, `track_export`, `retina_map_export` to the canonical
  Universe checkout; UGEN v4 uses the same parser as training/replay.
- Exported motor spline ownership is selected from generation metadata, including
  inherited per-being heads. Checkpoint DNA and acquired lifetime weights remain distinct.
- `package_run` packages all recorded finished generations and separates multiple
  lives into `champion-lineage<ID>`. `genome.bin`, weights.json and stage observations
  share the same observed-genome SHA-256. No guessed champion substitution.
- Optional lifetime block is f32[T,512]. Layout pairs mean **offset/count**, not
  start/end. The panel reads only the selected row. Inactive lifetime mechanisms
  are labelled disabled; acquired-state buffers are not drawn as anatomical layers.
- Cache fingerprints include observation bytes, generation metadata and exporter /
  producer / schema versions. Changing observations no longer silently reuses old packs.
- The real `/run/:name/:gen?/:being?` route uses the API. It no longer opens a fixture.
  Missing indices resolve from a bounded registered-trace index; the selector exposes
  generation, recorded being and life. Async loads commit together only when still
  current, preventing a late response from replacing a newer selection.
- Added `--ui-dist` to serve the built UI and API on the same origin, retaining API
  auth and a loopback default. Current instance: `http://127.0.0.1:8131`.

Builds: release exporter binaries, `vue-tsc -b`, `vite build --configLoader runner`.
Each used low priority and one core; no CPU numerical simulator or brain oracle ran.
Schema, binary block sizes and SHA-256 checks are artifact/transport verification,
not a substitute for GPU numerical checks. CSS and JS returned correct content types
from the running service. Browser rendering itself was not instrumented this turn;
do not label the complete visual interaction verified solely from HTTP success.

## Real GPU reference, not a new learning claim

`/media/llm-data/universe-runs/neuromod-lab-taxi/integration-champion-20260921/manifest.json`
contains the exact command, executable/checkpoint SHA-256 and preregistered gate.
GPU: AMD Radeon RX 6750 XT (RADV NAVI22), current `gpu_evolve`; 12,000 ticks,
4 clones of `b1_nocost_s77.bin`, seed 77, world 8005, width 28 m, finite fuel,
no new coevolution flags. Wall/impact objective costs remain zero to reproduce
that checkpoint's reference setup; this is an evaluation, not an evolutionary battery.

Observed: each clone completed 3 laps, with lap ends 3712, 7346 and 11058;
the fourth lap was partial at the horizon. No wall-impact ticks in driving.csv;
4 hole-contact episodes in the run report. For the traced being, 57/187 sampled
commands had brake > .05, maximum .6814047; sampled fuel ranged .630996 to .786216.
Elapsed run time reported by the executable: 12.7 s. Clones are NOT independent
seeds, and these figures do not establish new learning or tougher-world generalization.

Open in the integrated panel:

- http://127.0.0.1:8131/run/integration-champion-20260921/0/0
- http://127.0.0.1:8131/run/integration-ecology-20260921/0/2?profile=champion-lineage5

The ecological source is the prior GPU experiment, copied with source provenance;
it is not presented as a new run. The newborn lineage 5 has samples 128..512 and its
own UGEN v4 genome. Its predecessor has one recorded sample at tick 64; that short
life trace is intentional, not the full driving replay. Only the recorded being
has neural detail; the selector never fabricates other beings' neural observations.

HTTP evidence: `artifacts/integration-20260921-http.json`. It verifies three real
trace profiles, consistent genome hashes, L1/L2, f32[512] block slices, world seed,
and HTML/CSS/JS delivery. Existing unrelated working-tree changes were preserved.

## Repeatable local workflow

From LLM (using an existing GPU run):

```bash
PYTHONPATH=neuromod-lab/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 19 taskset -c 0 .venv-rocm/bin/python -m lab_api.trace.package_run \
  /media/llm-data/universe-runs/neuromod-lab-taxi/RUN --db /media/llm-data/lab.db

PYTHONPATH=neuromod-lab/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  nice -n 19 taskset -c 0 .venv-rocm/bin/python -m lab_api --port 8131 \
  --ui-dist /home/leonardo/projects/LLM/lab-ui/dist
```

The packager records an explicit reconstructed runtime manifest if the original
launcher omitted one. It never invents a historical command or binary hash. Use
`UNIVERSE_S2_BIN_DIR` only for an intentionally rebuilt exporter deployment.
The running service must use the matching producer schema. Genome-only checkpoints
still do not resume an ecosystem or curriculum frontier. New coevolution mechanisms
remain experimental/off by default; the previous regression has not been erased.
