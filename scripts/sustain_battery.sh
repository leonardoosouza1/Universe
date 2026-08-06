#!/usr/bin/env bash
# BATERIA DA SUSTENTABILIDADE — existe um `earn` em que o ser SOBREVIVE e anda mais?
#
# # A aritmética, agora que a vida é f64 e o ganho não trunca mais
#
# O ser gasta 1 tick por tick e ganha `earn × passo` por tick, com passo = v/64 m.
# Break-even: v* = 64/earn. Abaixo disso a vida escoa e ele morre; acima, acumula.
#
# Medido (pista 28 m, 6 gerações):
#   earn=6  → break-even 10,67 m/s · atingiu 8,00 (75%)  · vida 21%
#   earn=10 → break-even  6,40 m/s · atingiu 6,24 (98%)  · vida 28%
#
# A razão atingido/break-even SOBE conforme earn cresce. Se cruzar 1,0, o ser passa a viver
# a geração inteira andando rápido — e aí distância = v × tempo inteiro, que a 5 m/s por
# 187 s dá 935 m contra os 422 m do baseline que rasteja.
#
# Se NÃO cruzar, o que a curva mostrar é o teto real de velocidade sustentável deste corpo
# nesta pista — que é o número que importa para saber se a volta (2.928 m) é possível.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0

SEEDS=${SEEDS:-"1 2 3"}
BEINGS=${BEINGS:-48}
TICKS=${TICKS:-12000}
GENERATIONS=${GENERATIONS:-12}
WIDTH=${WIDTH:-28}
JOBS=${JOBS:-3}
EARNS=${EARNS:-"0 10 12 14 16 20"}

BIN="$ROOT/target/release/gpu_evolve_sustain"
nice -n 19 cargo build --release -j 2 --bin gpu_evolve 2>&1 | grep -E "^error" -A 4 && exit 1
cp -f "$ROOT/target/release/gpu_evolve" "$BIN"
echo "→ binário: $(md5sum "$BIN" | cut -c1-12)  (vida em f64 — sem o truncamento)"

run_one() {
  local earn="$1" seed="$2"
  local outdir="$ROOT/telemetry/sus_e${earn}_s${seed}"
  [ -f "$outdir/DONE" ] && return
  mkdir -p "$outdir"
  { echo "date: $(date -Is)"
    echo "cond: earn=$earn seed=$seed width=$WIDTH ticks=$TICKS"
    echo "break_even: $(python3 -c "print(f'{64/$earn:.2f}' if $earn>0 else '—')") m/s"
    md5sum "$BIN"; } > "$outdir/provenance.txt"
  env -u DISPLAY nice -n 19 "$BIN" --seres "$BEINGS" --ticks "$TICKS" \
      --gens "$GENERATIONS" --seed "$seed" --same-start --width "$WIDTH" \
      --life-earn "$earn" --out "$outdir" > "$outdir/run.log" 2>&1
  local rc=$? got
  got=$(grep -cE "^gen " "$outdir/run.log" 2>/dev/null); got=${got:-0}
  if [ "$rc" -eq 0 ] && [ "$got" -eq "$GENERATIONS" ]; then
    touch "$outdir/DONE"; echo "  ✓ earn=$earn/s$seed"
  else
    echo "  ✗ earn=$earn/s$seed rc=$rc gens=$got/$GENERATIONS"
  fi
}

echo "→ [$(date +%H:%M:%S)] earns=[$EARNS] × sementes=[$SEEDS] · pista $WIDTH m"
for earn in $EARNS; do
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done
    run_one "$earn" "$seed" &
  done
done
wait
echo; echo "=== CONCLUÍDA [$(date +%H:%M:%S)] ==="
python3 "$ROOT/scripts/compare_sustain.py"
