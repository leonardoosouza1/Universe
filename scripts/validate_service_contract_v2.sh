#!/usr/bin/env bash
# GPU integration evidence; this small battery cannot establish generalization.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?provide a new output directory on the run disk}"
PARENT="${2:?provide the frozen parent genome}"
mkdir -p "$OUT"
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
export MESA_VK_VERSION_OVERRIDE=1.3
export RAYON_NUM_THREADS=1
unset DISPLAY
BIN="$OUT/gpu_evolve"
test ! -e "$BIN" || { echo "output already contains a binary; use a fresh directory"; exit 2; }
cp "$ROOT/target/debug/gpu_evolve" "$BIN"
sha256sum "$BIN" "$PARENT" > "$OUT/hashes.txt"
git -C "$ROOT" diff > "$OUT/source.patch"
git -C "$ROOT" rev-parse HEAD > "$OUT/base-commit.txt"
run() {
  local name="$1"; shift
  local dir="$OUT/$name"
  mkdir "$dir"
  cp "$PARENT" "$dir/evolve_best_gen0000.bin"
  local args=(--scenario organism-taxi-v1 --school pickup --school-mode bench
    --beings 8 --gens 8 --ticks 4096 --school-horizon 4096 --seed 11
    --school-deadline-frac 1.0 --replay-every 7 --resume --resume-mix 1.0
    --out "$dir" "$@")
  printf '%s\n' "${args[@]}" > "$dir/command.txt"
  nice -n 10 "$BIN" "${args[@]}" > "$dir/stdout.log" 2>&1
}
# Paired ancestor, seed, geometry, time budget; only service contract differs.
run legacy --school-shared-start fixed --school-service-legacy
run v2 --school-shared-start fixed
# Oversized requested pool must clamp, deduplicate and rotate without panicking.
run coverage --school-eval-starts 3 --school-start-pool 24 --gens 4
python3 - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name in ('legacy', 'v2', 'coverage'):
    rows = [json.loads(p.read_text()) for p in sorted((root/name).glob('school_result_gen*.json'))]
    assert len(rows) == (4 if name == 'coverage' else 8)
    print(name, [(r['generation'], r['successes'], r['attempts'], r['score_max']) for r in rows])
    for row in rows:
        assert sum(row['failures_by_code'].values()) == row['failures']
manifests = [json.loads(p.read_text()) for p in sorted((root/'coverage').glob('organism_gen*.json'))]
assert all(m['school']['start_pool'] == 9 for m in manifests), 'canonical map has nine distinct pickup starts'
records = [json.loads(p.read_text()) for p in sorted((root/'coverage').glob('school_result_gen*.json'))]
seen = {b['start_index'] for r in records for b in r['beings']}
assert len(seen) == 9, seen
for r in records:
    if r['promotion_window_min_success_rate'] is not None:
        assert 0 <= r['promotion_window_min_success_rate'] <= 1
print('integration evidence passed; learned delivery is a separate gate')
PY
