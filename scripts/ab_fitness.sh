#!/usr/bin/env bash
# A/B: o degrau DENTRO da fitness estava cegando a seleção?
#
# # O caminho até esta pergunta
#
# Consertado o relógio (120 s -> 220 s), a população passou a dar 3 voltas — mas a evolução
# ficou PLANA: 0,0–0,4σ em 26 gerações. Duas coisas apareceram ao procurar o motivo:
#
# 1. `[NICHE] clusters=1 largest=10` em TODA geração de TODO treino. O fingerprint
#    comportamental saturava e o fitness-sharing nunca separou nada. Consertado (critério por
#    posto); no campo virou `clusters=2 largest=6` — mas volta a 1 na geração 5.
#
# 2. **O degrau dentro da própria fitness.** `lap_bonus = total_laps as f32 * 50000` usa
#    contagem INTEIRA e domina `prog` (~5800 m) por ~9×:
#
#       3,01 voltas -> 155.000 · 3,99 voltas -> 156.600 (+1%) · 4,00 voltas -> 206.600 (+32%)
#
#    A população estava exatamente aí: melhor carro 3,72 de um teto físico de 4,36. **Ir de
#    3,0 para 3,99 voltas — a melhoria que se quer selecionar — valia 1%.**
#
# # A hipótese, registrada ANTES de rodar
#
# Se o degrau é o que cega a seleção, o bônus contínuo (voltas fracionárias) faz aparecer
# tendência positiva onde havia 0,0σ. Se continuar plano, o gargalo é outro — e a próxima
# suspeita passa a ser a monocultura do elite (clusters volta a 1 na gen 5) ou o teto físico.
#
# O controle importa tanto quanto o tratamento: sem ele eu credito ao contínuo uma melhora
# que pode ser só da geração ter sido boa.
#
# Uso:  scripts/ab_fitness.sh [GENS] [SWARM] [ORCAMENTO_S]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
GENS=${1:-30}; SWARM=${2:-60}; ORC=${3:-220}
BIN="$RAIZ/target/release/universe"; SAIDA="$RAIZ/telemetry/ab_fitness"
[ -x "$BIN" ] || { echo "compile primeiro"; exit 1; }
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X"; exit 1; }
rm -rf "$SAIDA"; mkdir -p "$SAIDA"

echo "=== A/B: fitness em degrau contra fitness contínua ==="
echo "gens=$GENS · swarm=$SWARM · orçamento=${ORC}s · fingerprint CORRIGIDO nos dois lados"
echo "hipótese: o degrau cega a seleção na faixa 3,0–4,0 voltas, onde a população vive"
echo

for MODO in degrau continuo; do
  [ "$MODO" = continuo ] && FLAG=true || FLAG=false
  d="/tmp/ab_${MODO}"; rm -rf "$d"; mkdir -p "$d"
  cp "$RAIZ/CHAMPION.dna" "$d/"
  python3 - "$RAIZ/universe_config.json" "$d/universe_config.json" "$ORC" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["generation_timeout_s"] = float(sys.argv[3])
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
  cat > "$d/params.json" <<EOF
{"experiment_name":"ab_${MODO}","variant_id":0,"axiom_wall_avoidance":0.18,
"axiom_dark_center":0.45,"axiom_whisker":0.15,"axiom_racing_line":0.26,
"axiom_decay_divisor":50.0,"axiom_floor":0.35,"k_smooth":100.0,"k_speed":0.5,
"k_novel":50.0,"k_stab":200.0,"k_phi":300.0,"k_aesth":200.0,"lap_bonus":50000.0,
"survival_bonus":3000.0,"death_penalty":2000.0,"via_ablada":"","nascimento":"",
"fracao_exploradora":0.50,"lap_bonus_continuo":$FLAG,
"speed_bleed_damage":15.0,"stagnation_secs":15.0,"wall_t_threshold":40}
EOF
  ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
      --lab-params "$d/params.json" > "$SAIDA/${MODO}.log" 2>&1 ) &
done
wait
echo "terminado."; echo
python3 "$RAIZ/scripts/comparar_treinos.py" \
  <(echo) <(echo) >/dev/null 2>&1 || true
for MODO in degrau continuo; do
  echo "########## $MODO ##########"
  mkdir -p "/tmp/rel_$MODO" && cp "$SAIDA/${MODO}.log" "/tmp/rel_$MODO/"
  python3 "$RAIZ/scripts/relatorio_voltas.py" "/tmp/rel_$MODO" | tail -14
done
