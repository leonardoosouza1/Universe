#!/usr/bin/env bash
# BATERIA DA METADE LENTA — ela aprende a soltar o volante quando tem autoridade E
# a seleção enxerga o gradiente?
#
# Contexto medido (2026-08-06): descartados por número — percepção (corr 1.000), física
# (raio de sobra), incentivo (planos), seleção cega (consertada 5→25 notas). Sobra o
# cérebro: o volante ELEITO crava em +0.63, correlação zero com a curva. Falta a memória
# de "já estou na curva" — função da metade lenta, hoje com autoridade ±0.20 e amplitude
# de saída ~0.05 (60% da capacidade, sinal fraco).
#
# Duas alavancas cruzadas:
#   auth  — quanto o lento manda no volante (0.2 histórico · 0.6 · 1.0)
#   ablação — SEM lento (--sem-lento) isola se o efeito é do órgão ou da autoridade
#
# Sucesso = passar de 3/24 setores E o volante do líder deixar de cravar (corr com a
# curva subindo). Se nem com autoridade máxima mudar, o teto é o cérebro pequeno e o
# próximo passo é CRESCÊ-LO com o critério de participação (brain_capacity.rs).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
BIN="$ROOT/target/release/gpu_evolve_slow"
SEEDS=${SEEDS:-"1 2 3"}; JOBS=${JOBS:-3}
echo "→ binário $(md5sum "$BIN" | cut -c1-12)"

run() {
  local nome="$1" auth="$2" extra="$3" seed="$4"
  local d="$ROOT/telemetry/slow_${nome}_s${seed}"
  [ -f "$d/DONE" ] && return
  mkdir -p "$d"
  env -u DISPLAY nice -n 19 "$BIN" --seres 48 --ticks 12000 --gens 25 \
      --seed "$seed" --same-start --width 28 --fitness checkpoint \
      --slow-steer "$auth" $extra --out "$d" > "$d/run.log" 2>&1
  local got; got=$(grep -cE "^gen " "$d/run.log" 2>/dev/null); got=${got:-0}
  [ "$got" -eq 25 ] && { touch "$d/DONE"
    local b; b=$(grep -oE "melhor [0-9]+/24" "$d/run.log" | grep -oE "[0-9]+/" | grep -oE "[0-9]+" | sort -rn | head -1)
    echo "  ✓ $nome/s$seed melhor ${b:-0}/24"; } || echo "  ✗ $nome/s$seed ($got/25)"
}
# nome|auth|extra
for c in "auth02|0.20|" "auth06|0.6|" "auth10|1.0|" "semlento|1.0|--sem-lento"; do
  IFS='|' read -r nome auth extra <<< "$c"
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done
    run "$nome" "$auth" "$extra" "$seed" &
  done
done
wait
echo "=== FILA CONCLUÍDA [$(date +%H:%M:%S)] ==="
