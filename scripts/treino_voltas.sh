#!/usr/bin/env bash
# TREINO COM O RELÓGIO CERTO — o que Leonardo pediu: voltas completas, e viver mais.
#
# POR QUE ESTE SCRIPT SUBSTITUI `treino_final.sh`
#
# `treino_final.sh` rodava com o orçamento padrão de 120 s e reportava `≥2 voltas = 0,0%` em
# TODA geração. Eu quase creditei isso ao cérebro. Era o relógio:
#
#   pista 1658 m · volta mais rápida FISICAMENTE possível 50,5 s  (scripts/teto_de_voltas.py)
#   -> 120 s cabem 2,38 voltas no teto OTIMISTA, e o carro real anda abaixo do teto
#
#   EXPERIMENTO CONTROLADO (scripts/teto_voltas_exp.sh), GERAÇÃO 1, mesmo campeão de partida,
#   mesma semente, mesma fração — só o relógio muda:
#
#     120 s :  0v=41  1v=19  2v=0   net_max 3194 m = 1,93 voltas
#     180 s :  0v=39  1v=4   2v=17  net_max 4796 m = 2,89 voltas
#
#   **17 de 60 carros dão duas voltas com o MESMO cérebro.** O `0,0%` nunca foi fracasso de
#   aprendizado; era contagem inteira avaliada em cima da fronteira física.
#
#   E o dano ia além da leitura: `lap_bonus = 50000` domina a fitness. Com a segunda volta
#   inalcançável, o gradiente que puxaria para ela estava CEPADO — a seleção também sofria.
#
# O ORÇAMENTO DESTE TREINO: 220 s, escolhido por conta e não por gosto.
#
#   O melhor carro a 180 s andou a 81% do teto físico (2,89 de 3,56 voltas). Mantendo essa
#   eficiência, 3 voltas exigem ~187 s. Com 220 s a terceira volta fica CONFORTÁVEL e a
#   quarta vira o alvo de esticada — de propósito, para a métrica não encostar no teto de
#   novo. Foi exatamente esse encosto que produziu o zero enganoso.
#
#   Custo: 1,83× de tempo de parede por geração (~3,5 min contra ~1,9 min).
#
# FRAÇÃO EXPLORADORA: fica em 0,50, o padrão. A hipótese de que baixá-la ajudava morreu na
# comparação pareada (−207 m, 1,1σ, sinal invertido) — ver `scripts/comparar_treinos.py`.
#
# Uso:  scripts/treino_voltas.sh [GENS] [SWARM] [REPS] [ORCAMENTO_S]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
GENS=${1:-60}; SWARM=${2:-60}; REPS=${3:-2}; ORC=${4:-220}
BIN="$RAIZ/target/release/universe"; SAIDA="$RAIZ/telemetry/treino_voltas"
[ -x "$BIN" ] || { echo "compile primeiro"; exit 1; }
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X"; exit 1; }
rm -rf "$SAIDA"; mkdir -p "$SAIDA"

echo "=== TREINO COM O RELÓGIO CERTO ==="
echo "gens=$GENS · swarm=$SWARM · reps=$REPS · orçamento=${ORC}s (padrão era 120s)"
echo "partida: $(./target/release/dump_dna CHAMPION.dna | sed -n 2p)"
echo "alvo: ≥3 voltas (só agora alcançável) · ≥2 voltas já sai de graça do relógio"
echo

for r in $(seq 1 "$REPS"); do
  d="/tmp/tv_r${r}"; rm -rf "$d"; mkdir -p "$d"
  cp "$RAIZ/CHAMPION.dna" "$d/"
  python3 - "$RAIZ/universe_config.json" "$d/universe_config.json" "$ORC" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["generation_timeout_s"] = float(sys.argv[3])
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
  cat > "$d/params.json" <<EOF
{"experiment_name":"tv_r${r}","variant_id":0,"axiom_wall_avoidance":0.18,
"axiom_dark_center":0.45,"axiom_whisker":0.15,"axiom_racing_line":0.26,
"axiom_decay_divisor":50.0,"axiom_floor":0.35,"k_smooth":100.0,"k_speed":0.5,
"k_novel":50.0,"k_stab":200.0,"k_phi":300.0,"k_aesth":200.0,"lap_bonus":50000.0,
"survival_bonus":3000.0,"death_penalty":2000.0,"via_ablada":"","nascimento":"",
"fracao_exploradora":0.50,
"speed_bleed_damage":15.0,"stagnation_secs":15.0,"wall_t_threshold":40}
EOF
  ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
      --lab-params "$d/params.json" > "$SAIDA/r${r}.log" 2>&1 ) &
done
wait
echo "treino terminado."; echo
python3 "$RAIZ/scripts/relatorio_voltas.py" "$SAIDA"
