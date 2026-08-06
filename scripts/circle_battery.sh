#!/usr/bin/env bash
# O CÍRCULO É SELECIONADO PARA FORA quando o objetivo é PROGRESSO?
#
# Diagnóstico que motiva (replay_diag, 2026-08-06): o ser lê a pista (correlação volante ×
# curvatura à frente = +0,53) mas com ganho excessivo — fecha o raio até virar círculo, e
# trava o progresso em 265 m de 2.928. Circula DENTRO da pista, então não sai do traçado,
# não bate e não morre: some de toda métrica agregada.
#
# A aposta: sob HODÔMETRO circular era premiado (384 m de hodômetro contra 260 de progresso
# — 32% do movimento era giro e contava). Sob PROGRESSO, girar vale zero. Em 6 gerações os
# dois deram igual; a seleção precisa de mais tempo para desmontar um atrator.
#
# Por isso esta bateria é LONGA de propósito: 30 gerações, não 6.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0

SEEDS=${SEEDS:-"1 2 3"}
GENERATIONS=${GENERATIONS:-30}
JOBS=${JOBS:-3}

BIN="$ROOT/target/release/gpu_evolve_circle"
nice -n 19 cargo build --release -j 2 --bin gpu_evolve 2>&1 | grep -E "^error" -A 4 && exit 1
cp -f "$ROOT/target/release/gpu_evolve" "$BIN"
echo "→ binário: $(md5sum "$BIN" | cut -c1-12)"

run_one() {
  local fit="$1" seed="$2"
  local outdir="$ROOT/telemetry/cir_${fit}_s${seed}"
  [ -f "$outdir/DONE" ] && return
  mkdir -p "$outdir"
  { echo "date: $(date -Is)"; echo "cond: fitness=$fit seed=$seed gens=$GENERATIONS"; md5sum "$BIN"; } \
    > "$outdir/provenance.txt"
  env -u DISPLAY nice -n 19 "$BIN" --seres 48 --ticks 12000 --gens "$GENERATIONS" \
      --seed "$seed" --same-start --width 28 --fitness "$fit" --out "$outdir" \
      > "$outdir/run.log" 2>&1
  local rc=$? got
  got=$(grep -cE "^gen " "$outdir/run.log" 2>/dev/null); got=${got:-0}
  if [ "$rc" -eq 0 ] && [ "$got" -eq "$GENERATIONS" ]; then
    touch "$outdir/DONE"
    # O que importa não é a fitness reportada e sim o COMPORTAMENTO no replay.
    local f; f=$(ls -t "$outdir"/gen*.replay 2>/dev/null | head -1)
    if [ -n "$f" ]; then
      echo "  ✓ $fit/s$seed → $(./target/release/replay_diag --replay "$f" --width 28 2>/dev/null \
            | grep -E 'progresso alc|volante com sinal' | tr '\n' ' ')"
    else
      echo "  ✓ $fit/s$seed (sem replay)"
    fi
  else
    echo "  ✗ $fit/s$seed rc=$rc gens=$got/$GENERATIONS"
  fi
}

echo "→ [$(date +%H:%M:%S)] fitness dist vs prog · $GENERATIONS gerações · sementes [$SEEDS]"
for fit in dist prog; do
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done
    run_one "$fit" "$seed" &
  done
done
wait
echo "=== CONCLUÍDA [$(date +%H:%M:%S)] ==="
