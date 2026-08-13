#!/usr/bin/env bash
# BATERIA DE COMBUSTÍVEL — o mecanismo que as branches com voltas tinham e esta perdeu.
#
# # De onde veio
#
# `f1-world` e `state-of-art`, as duas branches em que os seres COMPLETAVAM VOLTAS, têm
# isto em `simulation.rs`:
#
#     racer.fuel -= (0.002 + pedal_intent.powi(2) * 0.004) * dt;
#     if racer.fuel <= 0.0 { racer.is_dead = true; }
#     if cruzou_a_linha && racer.distance > 400.0 {
#         racer.total_laps += 1;
#         if racer.total_laps % 3 == 0 { racer.fuel = 1.0; }   // REABASTECE
#     }
#
# Fechar volta reabastece; não fechar mata. A volta deixa de ser objetivo abstrato e vira
# sobrevivência. Na `race-world-v3` a palavra `fuel` aparecia UMA vez, como a constante 127
# no quadro de replay — o mecanismo tinha sido perdido inteiro.
#
# # O que esta bateria decide
#
# Com consumo de 0,002 a 0,006 por segundo, um tanque de 1,0 dura de 166 s a 500 s. Numa
# geração de 8.000 ticks (125 s) ele nunca acaba: o mecanismo existiria sem morder. Então
# o tanque é varrido junto com geração LONGA (24.000 ticks = 375 s), onde faltar é possível.
#
# tanque 0    = controle, sem combustível (o comportamento atual)
# tanque 0,15 = ~25-75 s: falta MUITO antes da primeira volta — pode matar antes de aprender
# tanque 0,4  = ~65-200 s
# tanque 1,0  = o valor original da f1-world
# tanque 2,0  = folgado, testa se o efeito é do incentivo ou só da pressa
#
# O que decide não é a distância: é `voltas reais` e `setores`. Combustível que reduz
# distância mas aumenta setores está funcionando — trocou vagar por avançar.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0

SEEDS=${SEEDS:-"1 2 3"}
BEINGS=${BEINGS:-48}
TICKS=${TICKS:-24000}
GENERATIONS=${GENERATIONS:-20}
WIDTH=${WIDTH:-28}
JOBS=${JOBS:-3}
TANQUES=${TANQUES:-"0 0.15 0.4 1.0 2.0"}

BIN="$ROOT/target/release/gpu_evolve_fuel"
nice -n 19 cargo build --release -j 2 --bin gpu_evolve 2>&1 | grep -E "^error" -A 4 && exit 1
cp -f "$ROOT/target/release/gpu_evolve" "$BIN"
echo "→ binário: $(md5sum "$BIN" | cut -c1-12)"

run_one() {
  local tanque="$1" seed="$2"
  local outdir="$ROOT/telemetry/fuel_t${tanque}_s${seed}"
  [ -f "$outdir/DONE" ] && return
  mkdir -p "$outdir"
  local arg=""
  [ "$tanque" != "0" ] && arg="--fuel-tank $tanque"
  { echo "date: $(date -Is)"
    echo "cond: tanque=$tanque seed=$seed ticks=$TICKS gens=$GENERATIONS width=$WIDTH"
    md5sum "$BIN"; } > "$outdir/provenance.txt"

  env -u DISPLAY nice -n 19 "$BIN" --seres "$BEINGS" --ticks "$TICKS" \
      --gens "$GENERATIONS" --seed "$seed" --same-start --width "$WIDTH" \
      --fitness checkpoint $arg --out "$outdir" > "$outdir/run.log" 2>&1
  local rc=$? got
  got=$(grep -cE "^gen " "$outdir/run.log" 2>/dev/null); got=${got:-0}
  if [ "$rc" -eq 0 ] && [ "$got" -eq "$GENERATIONS" ]; then
    touch "$outdir/DONE"
    local melhor; melhor=$(grep -oE "voltas reais: [0-9]+ seres ≥1" "$outdir/run.log" \
                  | grep -oE "[0-9]+ seres" | grep -oE "^[0-9]+" | sort -rn | head -1)
    echo "  ✓ tanque=$tanque/s$seed — melhor geração teve ${melhor:-0} seres com volta completa"
  else
    echo "  ✗ tanque=$tanque/s$seed rc=$rc gens=$got/$GENERATIONS"
  fi
}

echo "→ [$(date +%H:%M:%S)] tanques=[$TANQUES] × sementes=[$SEEDS] · $TICKS ticks · $GENERATIONS gens"
for t in $TANQUES; do
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done
    run_one "$t" "$seed" &
  done
done
wait
echo; echo "=== CONCLUÍDA [$(date +%H:%M:%S)] ==="
python3 "$ROOT/scripts/compare_fuel.py"
