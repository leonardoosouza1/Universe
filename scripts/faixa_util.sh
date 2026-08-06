#!/usr/bin/env bash
# Onde a medição de ablação significa alguma coisa? A varredura de dificuldade.
#
# Leonardo apontou: "dependendo do tamanho da pista eles não conseguem mesmo, acho que é algo
# que limita ou ele não tem recursos suficientes". Isso é decisivo para o experimento de
# ablação, e é o mesmo par de armadilhas que a bancada de sobrevivência já tinha cobrado:
#
#   pista fácil demais  -> sobra folga; desligar qualquer via custa ~0 (BERÇÁRIO)
#   pista dura demais   -> tudo vai ao chão; o custo vira ruído/ruído (ARMADILHA)
#
# Medir ablação numa largura só, sem saber em qual dos dois se está, produz um número que
# parece resultado e não é. Esta varredura acha a FAIXA ÚTIL antes de gastar as repetições.
#
# A arquitetura do carro é FIXA (96 neurônios na L1, 48 na L2, sem genes de arquitetura), o
# que dá o segundo motivo para esta varredura: se o recurso é fixo e a pista aperta, a única
# saída é REALOCAR. É por isso que o custo de ablação deve crescer com a dificuldade — e é
# isso que a curva vai mostrar ou desmentir.
#
# Uso:  scripts/faixa_util.sh [SEMENTE_DA_PISTA] [GENS] [SWARM]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
SEED=${1:-42}; GENS=${2:-2}; SWARM=${3:-60}
BIN="$RAIZ/target/release/universe"
SAIDA="$RAIZ/telemetry/faixa"; rm -rf "$SAIDA"; mkdir -p "$SAIDA"
LARGURAS=(8 10 12 14 18 24)

echo "=== FAIXA ÚTIL: até onde o carro consegue? ==="
echo "campeão: $(./target/release/dump_dna CHAMPION.dna | sed -n 2p)"
echo "larguras: ${LARGURAS[*]} m · semente de pista fixa=$SEED · gens=$GENS · swarm=$SWARM"
echo

n=0
for w in "${LARGURAS[@]}"; do
  d="/tmp/faixa_w${w}"; rm -rf "$d"; mkdir -p "$d"
  cp "$RAIZ/CHAMPION.dna" "$d/"
  [ -f "$RAIZ/universe_config.json" ] && cp "$RAIZ/universe_config.json" "$d/"
  ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
      --seed "$SEED" --width "$w" > "$SAIDA/w${w}.log" 2>&1 ) &
  n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait
done
wait

echo "larg.  net_max  net_med  mortes/total   veredito"
echo "---------------------------------------------------------------"
for w in "${LARGURAS[@]}"; do
  L="$SAIDA/w${w}.log"
  mx=$(grep -o 'net_max=[0-9.]*' "$L" 2>/dev/null | cut -d= -f2 | sort -g | tail -1)
  md=$(grep -o 'net_med=[0-9.]*' "$L" 2>/dev/null | cut -d= -f2 | sort -g | tail -1)
  dt=$(grep -o 'Deaths [0-9]*/[0-9]*' "$L" 2>/dev/null | tail -1 | cut -d' ' -f2)
  mx=${mx:-0}; md=${md:-0}; dt=${dt:-?}
  v=$(python3 -c "
mx=float('${mx}' or 0); md=float('${md}' or 0)
print('CHÃO — ninguém anda, ablação insensível' if mx < 300 else
      ('TETO — folga demais, ablação insensível' if md > 0.7*mx and mx > 2000 else
       'FAIXA ÚTIL'))")
  printf "%4sm %8s %8s %13s   %s\n" "$w" "$mx" "$md" "$dt" "$v"
done
echo "---------------------------------------------------------------"
echo "Rodar a ablação SÓ nas larguras marcadas FAIXA ÚTIL."
