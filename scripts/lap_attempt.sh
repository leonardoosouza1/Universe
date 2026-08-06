#!/usr/bin/env bash
# TENTATIVA DE VOLTA COMPLETA — velocidade sustentável × tempo longo.
#
# # A aritmética que motiva
#
# Volta = 2.928 m. Teto do corpo medido: ~16 m/s (earn=2 pediu 32 e o corpo deu 16,04).
# A 12 m/s sustentados por 375 s (24.000 ticks) dá 4.500 m — sobra. A 8 m/s dá 3.000 m —
# passa raspando. Então a volta é alcançável; o que falta é SOBREVIVER na velocidade.
#
# `--life-earn e` fixa a velocidade de equilíbrio em v* = 64/e. Com e=2 (alvo 32 m/s) o ser
# vive só 5% da geração: não consegue pagar. O ponto útil é o maior alvo que ainda deixa o
# ser vivo, e é isso que esta bateria procura, cruzando com o orçamento de tempo.
#
# Não há regra nova de direção aqui. A única mudança é o mundo cobrar movimento.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0

SEEDS=${SEEDS:-"1 2 3"}
BEINGS=${BEINGS:-48}
GENERATIONS=${GENERATIONS:-20}
JOBS=${JOBS:-2}
# earn|ticks
# earn|ticks|largura
#
# Corrigido depois do sweep: earn 2/3/4 pedem 32/21/16 m/s e o corpo satura em ~16, então o
# ser NUNCA fica acima do break-even e a morte é garantida por construção (vida 5-12% da
# geração). A faixa útil é earn ALTO (alvo baixo mas sustentável), cruzada com pista larga,
# que é onde dá para carregar velocidade na curva.
COMBOS=${COMBOS:-"0|24000|28 8|24000|28 12|24000|28 16|24000|28 12|48000|28 12|24000|40"}

BIN="$ROOT/target/release/gpu_evolve_lap"
nice -n 19 cargo build --release -j 2 --bin gpu_evolve 2>&1 | grep -E "^error" -A 4 && exit 1
cp -f "$ROOT/target/release/gpu_evolve" "$BIN"
echo "→ binário: $(md5sum "$BIN" | cut -c1-12)"

run_one() {
  local earn="$1" ticks="$2" width="$3" seed="$4"
  local outdir="$ROOT/telemetry/lap_e${earn}_t${ticks}_w${width}_s${seed}"
  [ -f "$outdir/DONE" ] && return
  mkdir -p "$outdir"
  { echo "date: $(date -Is)"
    echo "cond: life_earn=$earn ticks=$ticks seed=$seed"
    echo "alvo: v* = $((64 / earn)) m/s · alcance = v*·ticks/64 m · volta = 2928 m"
    md5sum "$BIN"; } > "$outdir/provenance.txt"
  env -u DISPLAY nice -n 19 "$BIN" --seres "$BEINGS" --ticks "$ticks" \
      --gens "$GENERATIONS" --seed "$seed" --same-start --life-earn "$earn" --width "$width" \
      --out "$outdir" > "$outdir/run.log" 2>&1
  local rc=$? got
  got=$(grep -cE "^gen " "$outdir/run.log" 2>/dev/null); got=${got:-0}
  if [ "$rc" -eq 0 ] && [ "$got" -eq "$GENERATIONS" ]; then
    touch "$outdir/DONE"
    local v; v=$(grep -oE "≥1V=[0-9]+" "$outdir/run.log" | grep -oE "[0-9]+" | sort -rn | head -1)
    echo "  ✓ e$earn/t$ticks/w$width/s$seed — melhor ≥1V=${v:-0}"
  else
    echo "  ✗ e$earn/t$ticks/w$width/s$seed rc=$rc gens=$got/$GENERATIONS"
  fi
}

echo "→ [$(date +%H:%M:%S)] combos=[$COMBOS] × sementes=[$SEEDS]"
for combo in $COMBOS; do
  IFS='|' read -r earn ticks width <<< "$combo"
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done
    run_one "$earn" "$ticks" "$width" "$seed" &
  done
done
wait
echo; echo "=== CONCLUÍDA [$(date +%H:%M:%S)] ==="
grep -h "≥1V=[1-9]" "$ROOT"/telemetry/lap_e*/run.log 2>/dev/null | head -20 \
  || echo "nenhuma volta completa registrada"
