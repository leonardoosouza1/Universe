# Vila Viva Shadow CPU

Standalone, bounded CPU inference for the approved Vila Viva `Linear(23,64) → tanh → GRU(64) → activity/move/sleep` shadow model. It has no Universe workspace, GPU, or WGPU dependency, so a release pack can copy this directory as `cpu-shadow/`.

Export an approved run to a writable release-pack location:

```bash
VILA_BRAIN_ROOT=/media/llm-data/vila-viva/brain \
  /home/leonardo/projects/LLM/.venv-rocm/bin/python \
  mc-lab/vila-viva/brain/export_cpu.py shadow-20260926-220823 cpu-shadow/cpu-model.json
nice -n 10 env CARGO_BUILD_JOBS=1 cargo build --offline --manifest-path cpu-shadow/Cargo.toml
cpu-shadow/target/debug/vila-shadow-cpu cpu-shadow/cpu-model.json
```

The process reads one JSON request per stdin line and writes one JSON response per line. Input is explicit: `{"inputs":[23 floats],"hidden":[64 floats]}`; omitting `hidden` supplies zero state for that request only. Output has `embed`, `hidden`, activity probabilities, raw normalized `move` (the panel applies its existing `×4` display scale), and sleep probability. Invalid requests return `{"error":"..."}` and do not terminate a valid loaded process.

`cpu-model.json` pins the exact feature order, classes, dimensions, frame-20/world-clock/per-row contract, model and metrics SHA-256 provenance, and every finite tensor. The runtime rejects incompatible dimensions, duplicate classes, non-finite values, or contract drift before serving input.
