#!/usr/bin/env bash
# EVOLUÇÃO LONGA — a corrida que nunca foi feita.
#
# # O número que motivou este script
#
# Leonardo: *"precisamos melhorar a simulação e espaço tempo, de forma que para os testes
# seja milhares de anos e não só alguns testes meia boca"*.
#
# Fiz a conta antes de tentar acelerar qualquer coisa, e ela é humilhante:
#
#   | escala                            | gerações | tempo de parede |
#   |---|---|---|
#   | o que eu venho rodando            |       30 | 1,8 horas       |
#   | raposas de Belyaev (raposa->dócil)|       60 | 3,5 horas       |
#   | estudo sério de Drosophila        |    1.000 | 2,4 dias        |
#   | Homo sapiens                      |   10.000 | 24 dias         |
#
# **Mil gerações custam dois dias e meio.** Nunca foi impossível — eu é que rodava 30 por vez,
# 0,3% de um estudo de verdade, e ficava concluindo coisas sobre "a evolução estagnou".
# Belyaev transformou raposa selvagem em cachorro dócil em 60 gerações; eu parei na metade.
#
# # O que precisou ser consertado para uma corrida de dias ser possível
#
# `--headless` criava janela real no X. As DUAS corridas do último A/B morreram no mesmo
# panic (`swap chain has been lost`) aos 27 e 28 de 30 gerações. Agora headless é headless de
# verdade: sem Winit, `backends: None`, roda com `env -u DISPLAY`. Nada usa GPU — não há um
# `.wgsl` no repositório e a retina é raycast DDA em CPU.
#
# Mesmo assim este script SUPERVISIONA: se o processo cair por qualquer motivo, ele retoma do
# `CHAMPION.dna` que a própria corrida gravou. Perder 2 dias de evolução por um panic no
# segundo dia seria o pior desperdício possível.
#
# Uso:  scripts/evoluir_longo.sh [GENS_TOTAL] [SWARM] [ORCAMENTO_S] [NOME]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
ALVO=${1:-1000}; SWARM=${2:-60}; ORC=${3:-220}; NOME=${4:-longo}
BIN="$RAIZ/target/release/universe"
SAIDA="$RAIZ/telemetry/evo_$NOME"; TRAB="/tmp/evo_$NOME"
[ -x "$BIN" ] || { echo "compile primeiro"; exit 1; }
mkdir -p "$SAIDA" "$TRAB"

# Só semeia o campeão na PRIMEIRA vez: retomada tem de continuar de onde parou, não voltar.
if [ ! -f "$TRAB/CHAMPION.dna" ]; then
  cp "$RAIZ/CHAMPION.dna" "$TRAB/"
  python3 - "$RAIZ/universe_config.json" "$TRAB/universe_config.json" "$ORC" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["generation_timeout_s"] = float(sys.argv[3])
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
  cat > "$TRAB/params.json" <<EOF
{"experiment_name":"evo_$NOME","variant_id":0,"axiom_wall_avoidance":0.18,
"axiom_dark_center":0.45,"axiom_whisker":0.15,"axiom_racing_line":0.26,
"axiom_decay_divisor":50.0,"axiom_floor":0.35,"k_smooth":100.0,"k_speed":0.5,
"k_novel":50.0,"k_stab":200.0,"k_phi":300.0,"k_aesth":200.0,"lap_bonus":50000.0,
"survival_bonus":3000.0,"death_penalty":2000.0,"via_ablada":"","nascimento":"",
"fracao_exploradora":0.50,"lap_bonus_continuo":false,"nicho_adaptativo":false,
"speed_bleed_damage":15.0,"stagnation_secs":15.0,"wall_t_threshold":40}
EOF
  echo "=== EVOLUÇÃO LONGA · alvo $ALVO gerações · swarm $SWARM · ${ORC}s por vida ==="
  echo "referência: Belyaev domesticou raposa em 60 gerações · Drosophila sério = 1000"
  echo
fi

feitas() { grep -hc "📊 GEN" "$SAIDA"/*.log 2>/dev/null | paste -sd+ | bc || echo 0; }

TENTATIVA=0
while [ "$(feitas)" -lt "$ALVO" ]; do
  TENTATIVA=$((TENTATIVA + 1))
  FALTAM=$(( ALVO - $(feitas) ))
  echo "[$(date +%H:%M:%S)] tentativa $TENTATIVA · $(feitas)/$ALVO feitas · faltam $FALTAM"
  ( cd "$TRAB" && env -u DISPLAY -u WAYLAND_DISPLAY "$BIN" --headless \
      --gens "$FALTAM" --swarm "$SWARM" --lab-params "$TRAB/params.json" \
      >> "$SAIDA/t${TENTATIVA}.log" 2>&1 )
  CODIGO=$?
  echo "[$(date +%H:%M:%S)] processo saiu com $CODIGO · $(feitas)/$ALVO"
  # Queda antes do alvo: retoma do CHAMPION.dna que a própria corrida gravou.
  if [ "$(feitas)" -lt "$ALVO" ]; then
    if [ "$TENTATIVA" -ge 200 ]; then
      echo "✗ 200 tentativas — algo está quebrado de verdade, parando."; exit 1
    fi
    echo "    retomando de $("$RAIZ/target/release/dump_dna" "$TRAB/CHAMPION.dna" 2>/dev/null | sed -n 2p)"
    sleep 5
  fi
done

echo; echo "=== $(feitas) gerações concluídas em $TENTATIVA tentativa(s) ==="
python3 "$RAIZ/scripts/tendencia.py" "$SAIDA"
