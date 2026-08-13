#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
inicio=$(date +%s)
while pgrep -f gpu_evolve_grao >/dev/null 2>&1; do
  [ $(( $(date +%s) - inicio )) -gt 3600 ] && break
  sleep 30
done
echo "[$(date +%H:%M:%S)] grao terminou — disparando bateria do lento" >> "$ROOT/telemetry/chain_slow.log"
exec env SEEDS="1 2 3" JOBS=3 ./scripts/slow_authority_battery.sh >> "$ROOT/telemetry/slow_battery.log" 2>&1
