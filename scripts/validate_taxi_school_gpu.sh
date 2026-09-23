#!/usr/bin/env bash
# Reproducible Radeon/Vulkan validation for the taxi school. Host Python only
# reads JSON evidence; every simulated tick is dispatched by gpu_evolve to Vulkan.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${UNIVERSE_GPU_EVOLVE:-$ROOT/target/debug/gpu_evolve}"
OUT="${TAXI_SCHOOL_OUT:-$ROOT/artifacts/taxi-school-delivery-validation}"
MODE="${1:-witnesses}"
GPU_ENV=(env -u DISPLAY WGPU_BACKEND=vulkan
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
  MESA_VK_VERSION_OVERRIDE=1.3 RAYON_NUM_THREADS=1 CARGO_BUILD_JOBS=1)

mkdir -p "$OUT"
cd "$ROOT"

validate_school() {
  local run="$1" lesson="$2"
  python3 - "$run" "$lesson" <<'PY'
import json, math, pathlib, sys
root, expected = pathlib.Path(sys.argv[1]), sys.argv[2]
result = json.loads((root / "school_result_gen0000.json").read_text())
assert result["lesson"] == expected, (result["lesson"], expected)
assert result["successes"] == result["attempts"] == 1, result
assert result["failures"] == result["truncated"] == result["deaths"] == 0, result
row = result["beings"][0]
assert row["success"] and row["fail_code"] == 0, row
assert row["contact_ticks"] == 0 and row["contact_ticks_lifetime"] == 0, row
def finite(v):
    if isinstance(v, float): assert math.isfinite(v), v
    elif isinstance(v, dict):
        for child in v.values(): finite(child)
    elif isinstance(v, list):
        for child in v: finite(child)
finite(result)
overlap = 0
with (root / "observation.jsonl").open() as stream:
    for line in stream:
        sample = json.loads(line); finite(sample)
        command = sample.get("body",{}).get("motor_trace",{}).get("final_command")
        if command and command[0] > 1e-5 and command[2] > 1e-5: overlap += 1
assert overlap == 0, f"throttle/brake overlap in {overlap} samples"
print(json.dumps({"lesson": expected, "status": "pass", "outcome_tick": row["outcome_tick"], "score": row["score"]}))
PY
}

run_witness() {
  local lesson="$1" ticks="$2" offset="${3:-0}"
  local run="$OUT/witness-$lesson"
  rm -rf "$run"
  "${GPU_ENV[@]}" "$BIN" --scenario organism-taxi-v1 --taxi-reference-witness \
    --school "$lesson" --school-mode bench --school-horizon "$ticks" \
    --beings 1 --gens 1 --ticks "$ticks" --seed 7101 --gen-offset "$offset" \
    --school-shared-start rotate --driving-diagnostics --out "$run"
  validate_school "$run" "$lesson"
}

run_witnesses() {
  run_witness move 512
  run_witness stop 768
  run_witness turn_near 2048
  run_witness turn_far 3072
  run_witness approach_passenger 2048
  run_witness pickup 2048
  run_witness navigate_with_passenger 12288 3
  run_witness delivery 16000 3
}

run_production() {
  local run="$OUT/witness-production-two-deliveries"
  rm -rf "$run"
  "${GPU_ENV[@]}" "$BIN" --scenario organism-taxi-v1 --taxi-reference-witness \
    --beings 1 --gens 1 --ticks 30000 --seed 7101 --driving-diagnostics --out "$run"
  python3 scripts/validate_taxi_witness.py "$run" --deliveries 2 \
    --max-collision-j "${TAXI_WITNESS_MAX_COLLISION_J:-0}"
}

case "$MODE" in
  witnesses) run_witnesses ;;
  production) run_production ;;
  all) run_witnesses; run_production ;;
  *) echo "usage: $0 [witnesses|production|all]" >&2; exit 2 ;;
esac
