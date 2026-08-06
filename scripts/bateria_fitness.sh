#!/usr/bin/env bash
# BATERIA DE FITNESS — o que a seleção cobra muda o que eles fazem?
#
# # O achado que motivou
#
# Leonardo perguntou: "se andam tão devagar o motor não para pelo RPM? será que acharam um
# ponto de aceleração tão seguro?" Medido no observatório, ele estava certo:
#
#   acelerador MÁXIMO observado : 0,75   (nunca chega a 1,0)
#   acelerador acima de 0,9     : 0%     das amostras
#   acelerador abaixo de 0,3    : 42%
#   velocidade média            : 11,5 m/s (42 km/h) num GT3
#
# E a causa está na fitness: ela é `dist_acum`, soma de deslocamento ABSOLUTO. Não distingue
# avançar de andar em círculo, e não cobra velocidade. Rastejar a 11 m/s por 24.000 ticks
# acumula mais que correr a 30 m/s e bater na terceira curva — **a seleção premia exatamente
# o comportamento observado.** Não é limite de motor nem de cérebro.
#
# Três critérios, mesma semente, `dist` como CONTROLE:
#   dist      deslocamento absoluto (o que estava lá)
#   veloc     distância × (1 + v_média/30) — cobra andar rápido
#   progresso avanço de ARCO na pista com wrap; recuar SUBTRAI, círculo não conta
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
echo "→ recompilando…"
nice -n 19 cargo build --release -j 2 --bin cortex_evolve 2>&1 | grep -E "^error" -A 4 && exit 1

for crit in dist veloc progresso; do
  SAIDA="$RAIZ/telemetry/fit_$crit"
  [ -f "$SAIDA/PRONTO" ] && { echo "→ $crit já concluída"; continue; }
  mkdir -p "$SAIDA"
  echo "data: $(date -Is) · IARA_FITNESS=$crit · 60 seres · 24000 ticks · 30 gens" > "$SAIDA/proveniencia.txt"
  md5sum src/bin/cortex_evolve.rs src/gpu/cortex_sheet.wgsl >> "$SAIDA/proveniencia.txt"
  echo "→ [$(date +%H:%M:%S)] fitness=$crit"
  IARA_FITNESS=$crit env -u DISPLAY nice -n 19 ./target/release/cortex_evolve \
      --seres 60 --ticks 24000 --gens 30 --fat-beta 0.0 --div-sigma 0.0 --map-interval 30 \
      > "$SAIDA/run.log" 2>&1
  touch "$SAIDA/PRONTO"
  echo "→ [$(date +%H:%M:%S)] $crit concluída"
done
echo; echo "=== BATERIA DE FITNESS CONCLUÍDA ==="
python3 "$RAIZ/scripts/comparar_bateria.py"
