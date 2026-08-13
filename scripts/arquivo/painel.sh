#!/usr/bin/env bash
# PAINEL — sobe o replay do melhor ser e imprime o link.
#
# Um comando só: acha o replay mais recente (ou o que você passar), gera o HTML e serve.
# Fica na raiz do projeto de propósito, para o caminho ser curto de digitar.
#
#   ./painel.sh                          # o replay mais recente
#   ./painel.sh telemetry/sus_e12_s1     # o mais recente daquela corrida
#   ./painel.sh caminho/para/x.replay    # um arquivo específico
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
PORTA=${PORTA:-8127}
SRV=/tmp/replay_srv

alvo="${1:-}"
if [ -z "$alvo" ]; then
  # Mais recente de todo o telemetry, por data de modificação.
  alvo=$(find telemetry -name "*.replay" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
elif [ -d "$alvo" ]; then
  alvo=$(find "$alvo" -name "*.replay" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
fi

if [ -z "$alvo" ] || [ ! -f "$alvo" ]; then
  echo "✗ nenhum replay encontrado."
  echo "  Replays são gravados durante a evolução, em telemetry/<corrida>/gen*.replay."
  exit 1
fi

if [ ! -x target/release/replay_analyze ]; then
  echo "→ compilando o analisador (primeira vez)…"
  nice -n 19 cargo build --release -j 2 --bin replay_analyze 2>&1 | grep -E "^error" -A 4 && exit 1
fi

mkdir -p "$SRV"
echo "→ replay: $alvo"
./target/release/replay_analyze --replay "$alvo" --html "$SRV/index.html" 2>&1 \
  | grep -E "Arco total|Vel máx|Vel média|voltas" | sed 's/^/   /'

# Derruba um servidor antigo na mesma porta antes de subir (evita "address in use"
# silencioso, que deixaria você olhando um HTML velho sem perceber).
lsof -ti tcp:"$PORTA" 2>/dev/null | while read -r p; do kill "$p" 2>/dev/null; done
sleep 0.3
nohup python3 -m http.server "$PORTA" --directory "$SRV" >/dev/null 2>&1 &
sleep 0.7

if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORTA/" | grep -q 200; then
  echo
  echo "   ▶  http://localhost:$PORTA"
  echo
else
  echo "✗ o servidor não respondeu na porta $PORTA"
  exit 1
fi
