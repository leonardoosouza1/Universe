#!/usr/bin/env bash
# A/B: o limiar de nicho FIXO estava apagando a diversidade que existe?
#
# # O que motivou
#
# Consertado o fingerprint (que saturava e dava distância 0 entre quaisquer dois elites), a
# instrumentação nova mediu a distância REAL par a par no pool de elite:
#
#   gen 1-3:  dist_med 5,0-5,3 · dist_max 10
#   gen 4-7:  dist_med 2,4-2,6 · dist_max  8
#
# O limiar é 8. **Ele é 1,5-3× MAIOR que a distância típica** — engole tudo num cluster só, e
# o fitness-sharing (fitness ÷ tamanho do cluster) vira um fator uniforme que não separa
# ninguém. Daí `clusters=1` em toda geração de todo treino.
#
# E há diversidade a preservar: `dist_med` cai de 5,3 para 2,4 conforme a população converge,
# mas nunca chega a zero.
#
# A calibração histórica do limiar ("v14: tentou 10, piorou — tudo 1 cluster") é NULA: foi
# feita quando o fingerprint saturava e qualquer limiar dava 1 cluster.
#
# # O conserto e a hipótese, registrada ANTES
#
# Limiar ADAPTATIVO = mediana das distâncias do próprio pool (limitada por 8). Mesmo princípio
# que consertou o fingerprint: critério RELATIVO, não absoluto — a escala das distâncias muda
# ao longo da evolução e um número fixo não acompanha.
#
# PREDIÇÃO: com o adaptativo, `clusters` para de colapsar em 1; e SE a monocultura é o que
# limita a evolução, a mediana da população sobe mais rápido que os +0,051..0,060 voltas/gen
# medidos com limiar fixo.
#
# O risco declarado: pressão de diversidade demais promove indivíduo diferente-mas-ruim sobre
# bom-mas-parecido. Se a mediana PIORAR, é isso, e o limiar precisa de piso.
#
# Uso:  scripts/ab_nicho.sh [GENS] [SWARM] [ORCAMENTO_S]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
GENS=${1:-30}; SWARM=${2:-60}; ORC=${3:-220}
BIN="$RAIZ/target/release/universe"; SAIDA="$RAIZ/telemetry/ab_nicho"
[ -x "$BIN" ] || { echo "compile primeiro"; exit 1; }
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X"; exit 1; }
rm -rf "$SAIDA"; mkdir -p "$SAIDA"

echo "=== A/B: limiar de nicho fixo (8) contra adaptativo (mediana do pool) ==="
echo "gens=$GENS · swarm=$SWARM · orçamento=${ORC}s"
echo "medido: distância típica entre elites 2,4-5,3 · limiar fixo 8 = grande demais"
echo

for MODO in fixo adaptativo; do
  [ "$MODO" = adaptativo ] && FLAG=true || FLAG=false
  d="/tmp/an_${MODO}"; rm -rf "$d"; mkdir -p "$d"
  cp "$RAIZ/CHAMPION.dna" "$d/"
  python3 - "$RAIZ/universe_config.json" "$d/universe_config.json" "$ORC" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["generation_timeout_s"] = float(sys.argv[3])
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
  cat > "$d/params.json" <<EOF
{"experiment_name":"an_${MODO}","variant_id":0,"axiom_wall_avoidance":0.18,
"axiom_dark_center":0.45,"axiom_whisker":0.15,"axiom_racing_line":0.26,
"axiom_decay_divisor":50.0,"axiom_floor":0.35,"k_smooth":100.0,"k_speed":0.5,
"k_novel":50.0,"k_stab":200.0,"k_phi":300.0,"k_aesth":200.0,"lap_bonus":50000.0,
"survival_bonus":3000.0,"death_penalty":2000.0,"via_ablada":"","nascimento":"",
"fracao_exploradora":0.50,"lap_bonus_continuo":false,"nicho_adaptativo":$FLAG,
"speed_bleed_damage":15.0,"stagnation_secs":15.0,"wall_t_threshold":40}
EOF
  ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
      --lab-params "$d/params.json" > "$SAIDA/${MODO}.log" 2>&1 ) &
done
wait
echo "terminado."; echo
for MODO in fixo adaptativo; do
  echo "### $MODO — clusters por geração:"
  grep -h "NICHE" "$SAIDA/${MODO}.log" | awk '{print $4, $6, $7}' | tr '\n' ' '; echo; echo
  rm -rf "/tmp/n_$MODO"; mkdir -p "/tmp/n_$MODO"; cp "$SAIDA/${MODO}.log" "/tmp/n_$MODO/"
done
python3 "$RAIZ/scripts/tendencia.py" /tmp/n_fixo /tmp/n_adaptativo
