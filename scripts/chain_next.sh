#!/usr/bin/env bash
# ENCADEADOR — dispara a próxima bateria assim que a GPU liberar.
#
# Existe porque a máquina não pode ficar ociosa entre os meus turnos, e porque empilhar
# mais processos numa GPU já em 94% só faz todas as corridas ficarem lentas. Ele espera
# a fila atual drenar e emenda a seguinte, sozinho.
#
# Aguarda por PROCESSO, não por arquivo: as baterias escrevem DONE por corrida, mas o que
# interessa aqui é o device livre.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
LIMITE=${LIMITE:-2}          # dispara quando restarem <= LIMITE processos de GPU
ESPERA_MAX=${ESPERA_MAX:-14400}   # desiste depois de 4 h para não ficar preso para sempre
LOG="$ROOT/telemetry/chain.log"

inicio=$(date +%s)
while :; do
  n=$(pgrep -c -f "gpu_evolve_(queue|life|par_|bateria)" 2>/dev/null || echo 0)
  agora=$(date +%s)
  if [ "$n" -le "$LIMITE" ]; then
    echo "[$(date +%H:%M:%S)] GPU com $n processos — disparando próxima bateria" | tee -a "$LOG"
    break
  fi
  if [ $((agora - inicio)) -gt "$ESPERA_MAX" ]; then
    echo "[$(date +%H:%M:%S)] espera excedeu ${ESPERA_MAX}s com $n processos — desistindo" | tee -a "$LOG"
    exit 1
  fi
  sleep 60
done

exec env SEEDS="1 2 3" BEINGS=48 GENERATIONS=20 JOBS=2 \
     ./scripts/lap_attempt.sh >> "$ROOT/telemetry/lap_attempt.log" 2>&1
