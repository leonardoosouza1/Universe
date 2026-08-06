#!/usr/bin/env bash
# BATERIA DA VIDA GANHA — a velocidade obedece a v* = 64/earn?
#
# # A previsão, registrada ANTES de medir
#
# O ser gasta 1 tick de vida por tick e ganha `earn` ticks por metro. O equilíbrio é
# `passo × earn = 1`, com passo = v/64 m por tick. Logo:
#
#     v* = 64 / earn
#
# earn 2 → 32 m/s · earn 3 → 21 m/s · earn 4 → 16 m/s · earn 8 → 8 m/s
#
# Já medido em earn=8: mediana 8,10 m/s. Se a lei valer, os outros seguem a curva. Se a
# velocidade saturar bem abaixo do previsto, o limite é do CORPO (motor, pneu, curva) e não
# da regra — e aí o número onde satura é a informação que interessa.
#
# Sem `--life-earn` a população converge para RASTEJAR: 3,64 m/s com 37,7 m/s disponíveis,
# mortalidade 0%. Pior, com mais tempo fica mais lenta (1,48 m/s em 24k contra 2,96 em 8k):
# distância em tempo fixo não cria pressa nenhuma.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0

SEEDS=${SEEDS:-"1 2 3"}
BEINGS=${BEINGS:-48}
TICKS=${TICKS:-12000}
GENERATIONS=${GENERATIONS:-14}
JOBS=${JOBS:-2}
EARNS=${EARNS:-"0 2 3 4 6 8"}

BIN="$ROOT/target/release/gpu_evolve_life"
nice -n 19 cargo build --release -j 2 --bin gpu_evolve 2>&1 | grep -E "^error" -A 4 && exit 1
cp -f "$ROOT/target/release/gpu_evolve" "$BIN"
echo "→ binário: $(md5sum "$BIN" | cut -c1-12)"

run_one() {
  local earn="$1" seed="$2"
  local outdir="$ROOT/telemetry/le_e${earn}_s${seed}"
  [ -f "$outdir/DONE" ] && return
  mkdir -p "$outdir"
  { echo "date: $(date -Is)"; echo "cond: life_earn=$earn seed=$seed ticks=$TICKS"
    echo "previsao: v* = 64/$earn m/s"; md5sum "$BIN"; } > "$outdir/provenance.txt"
  env -u DISPLAY nice -n 19 "$BIN" --seres "$BEINGS" --ticks "$TICKS" \
      --gens "$GENERATIONS" --seed "$seed" --same-start --life-earn "$earn" \
      --out "$outdir" > "$outdir/run.log" 2>&1
  local rc=$? got
  got=$(grep -cE "^gen " "$outdir/run.log" 2>/dev/null); got=${got:-0}
  if [ "$rc" -eq 0 ] && [ "$got" -eq "$GENERATIONS" ]; then
    touch "$outdir/DONE"; echo "  ✓ earn=$earn/s$seed"
  else
    echo "  ✗ earn=$earn/s$seed rc=$rc gens=$got/$GENERATIONS"
  fi
}

echo "→ [$(date +%H:%M:%S)] earns=[$EARNS] × sementes=[$SEEDS]"
for earn in $EARNS; do
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done
    run_one "$earn" "$seed" &
  done
done
wait
echo; echo "=== CONCLUÍDA [$(date +%H:%M:%S)] ==="
python3 "$ROOT/scripts/compare_life_earn.py"
