# Fly experiment tools

This directory owns orchestration and analysis for the paired Fly-controller
experiments. The simulator and numerical models remain in Rust/WGSL. Standard
library Python is sufficient for running, profiling and summarizing experiments.

From the repository root:

```sh
python3 scripts/fly/run.py --binary target/release/gpu_evolve \
  --maps /media/llm-data/datasets/flywire --output /media/llm-data/universe-runs/NEW-RUN
python3 scripts/fly/analyze.py /media/llm-data/universe-runs/NEW-RUN
```

`run.py` freezes the executable, hashes maps, records every command, serializes
runs and checks population/telemetry artifacts. Output directories must be new.
It locks other instances of this harness, not arbitrary GPU programs. Keep the
GPU exclusive when measuring performance. `--source-snapshot` identifies the
original source snapshot when reusing an older frozen executable.

`analyze.py` refuses incomplete batteries and nonfinite measurements. Replication
is by population seed. `profile.py --parent-pid PID --output NEW.csv` samples the
harness's running child without invoking GPU APIs. DRM counters are system-wide.

`plot.py PRIMARY_DIRECTORY FOLLOWUP_DIRECTORY` creates a PNG and SVG comparison
using Matplotlib. This optional plotting dependency is not needed by the runner.

See [the experiment protocol](../../docs/FLY-BATTERY-2026-09-14.md) for endpoints,
exclusions, stopping rules and scientific limitations.
