#!/usr/bin/env bash
# Run the bounded organism GPU continuation check on Radeon/Vulkan.
# This script never selects a CPU adapter and never runs a CPU simulation.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQUESTED_BASE="$1"
if [[ -e "$REQUESTED_BASE" ]]; then
  echo "refusing existing output path: $REQUESTED_BASE" >&2
  exit 2
fi
mkdir -p "$REQUESTED_BASE"
BASE="$(cd "$REQUESTED_BASE" && pwd)"
cd "$ROOT"

# The ICD override is intentionally explicit. A missing Radeon ICD is an error,
# not permission to fall back to a software adapter.
export WGPU_BACKEND="vulkan"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/radeon_icd.json}"
export MESA_VK_VERSION_OVERRIDE="${MESA_VK_VERSION_OVERRIDE:-1.3}"
export RAYON_NUM_THREADS=1
export CARGO_BUILD_JOBS=1
unset DISPLAY

cargo test --lib 'gpu::organism_' -- --ignored --nocapture --test-threads=1
cargo test --lib 'taxi_gpu' -- --ignored --nocapture --test-threads=1
cargo build --bin gpu_evolve

BIN="$ROOT/target/debug/gpu_evolve"
COMMON=(--scenario organism-taxi-v1 --beings 2 --gens 1 --snapshot-every 96
  --driving-diagnostics --vision-every 16 --no-spectral-init)

# First run: 192 GPU ticks, producing a checkpoint at tick 96.
"$BIN" "${COMMON[@]}" --ticks 192 --out "$BASE/full"
CHECKPOINT="$BASE/full/state-gen0000-tick00000096"
if [[ ! -d "$CHECKPOINT" ]]; then
  echo "missing expected checkpoint: $CHECKPOINT" >&2
  exit 1
fi

# Continuation run: restore tick 96 and execute through tick 192 elsewhere.
"$BIN" "${COMMON[@]}" --ticks 192 --resume-state "$CHECKPOINT" --out "$BASE/resumed"

# Compare the final observation and organism result. This checks continuation
# of the mechanical state; it is not a learning or fitness equivalence test.
python3 - "$BASE/full/observation.jsonl" "$BASE/resumed/observation.jsonl" <<'PY'
import json
import sys

def final_row(path):
    last = None
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    if not isinstance(last, dict) or not isinstance(last.get("body"), dict):
        raise SystemExit(f"missing final observation body: {path}")
    if "applied_motor" not in last["body"]:
        raise SystemExit(f"final body has no applied_motor: {path}")
    return last

def comparable(row):
    # Resume identity intentionally changes the provenance filename only.
    row = json.loads(json.dumps(row))
    row.get("body", {}).pop("genome_file", None)
    row.pop("genome_file", None)
    return row

if comparable(final_row(sys.argv[1])) != comparable(final_row(sys.argv[2])):
    raise SystemExit("final observation mismatch")
print("organism GPU final observation matches")
PY

python3 - "$BASE/full/organism_result_gen0000.json" "$BASE/resumed/organism_result_gen0000.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as left, open(sys.argv[2], encoding="utf-8") as right:
    if json.load(left) != json.load(right):
        raise SystemExit("organism result mismatch")
print("organism result matches")
PY

echo "GPU organism verification complete: $BASE"
