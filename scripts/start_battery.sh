#!/usr/bin/env bash
# BATERIA DA LARGADA — a fitness mede o cérebro ou o lugar onde o ser nasceu?
#
# Achado que motivou (2026-08-05): cada ser nascia num waypoint diferente
# (`wi = i * n_wp / n_seres`). Como trechos de pista têm dificuldade diferente e a fitness
# é distância, o "campeão" era quem nascia na reta. Medido pelo elite, que passa SEM
# mutação e portanto é o mesmo genoma reavaliado:
#
#   largadas diferentes → 72% do resultado anterior (pior caso 54%)
#   mesma largada       → 99% (mediana 100%, 0/7 gerações perdendo metade)
#
# E a população inteira mudou de patamar: mediana 173 m → 365 m, mortalidade 2% → 0%.
#
# Esta bateria replica em várias sementes e testa se, com a seleção funcionando, mais
# tempo de vida finalmente quebra a parede dos ~380 m.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0

SEEDS=${SEEDS:-"1 2 3 4"}
BEINGS=${BEINGS:-48}
GENERATIONS=${GENERATIONS:-14}
JOBS=${JOBS:-3}

BIN="$ROOT/target/release/gpu_evolve_par_start"
nice -n 19 cargo build --release -j 2 --bin gpu_evolve 2>&1 | grep -E "^error" -A 4 && exit 1
cp -f "$ROOT/target/release/gpu_evolve" "$BIN"
echo "→ binário congelado: $(md5sum "$BIN" | cut -c1-12)"

# nome|ticks|flags
CONDITIONS=(
  "espalhado|8000|"
  "mesma|8000|--same-start"
  "mesma_longa|24000|--same-start"
  "mesma_vida|8000|--same-start --wall-life 16"
)

run_one() {
  local name="$1" ticks="$2" flags="$3" seed="$4"
  local outdir="$ROOT/telemetry/start_${name}_s${seed}"
  [ -f "$outdir/DONE" ] && { echo "  ↷ $name/s$seed já pronta"; return; }
  mkdir -p "$outdir"
  { echo "date: $(date -Is)"; echo "cond: $name ticks=$ticks flags=[$flags] seed=$seed"
    md5sum "$BIN"; } > "$outdir/provenance.txt"
  env -u DISPLAY nice -n 19 "$BIN" --seres "$BEINGS" --ticks "$ticks" \
      --gens "$GENERATIONS" --seed "$seed" $flags --out "$outdir" \
      > "$outdir/run.log" 2>&1
  local rc=$? got
  got=$(grep -cE "^gen " "$outdir/run.log" 2>/dev/null); got=${got:-0}
  if [ "$rc" -eq 0 ] && [ "$got" -eq "$GENERATIONS" ]; then
    touch "$outdir/DONE"; echo "  ✓ $name/s$seed"
  else
    echo "  ✗ $name/s$seed INCOMPLETA rc=$rc gens=$got/$GENERATIONS"
  fi
}

echo "→ [$(date +%H:%M:%S)] ${#CONDITIONS[@]} condições × $(echo $SEEDS | wc -w) sementes · $JOBS por vez"
for c in "${CONDITIONS[@]}"; do
  IFS='|' read -r name ticks flags <<< "$c"
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    run_one "$name" "$ticks" "$flags" "$seed" &
  done
done
wait
echo; echo "=== CONCLUÍDA [$(date +%H:%M:%S)] ==="
python3 "$ROOT/scripts/compare_start.py"
