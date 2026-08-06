#!/usr/bin/env bash
# WATCHDOG — o treino ainda está VIVO e PROGREDINDO?
#
# # Por que existe
#
# Leonardo, 2026-08-05: "às vezes você termina de falar, mas a GPU ainda está em uso,
# precisa validar o código, se não tem um bug que para a execução do treino".
#
# O modo de falha real neste projeto não é o processo morrer com erro — é ele CONTINUAR
# vivo sem produzir nada. Já aconteceu: `map_async` chamado antes do `poll` deixa o
# callback sem ser servido e o processo fica bloqueado para sempre, sem log, sem crash,
# sem código de saída. `ps` mostra o processo lá, a GPU mostra uso, e não sai geração
# nenhuma. Um `pgrep` responde "está rodando" e mente.
#
# Por isso este watchdog não pergunta "o processo existe", pergunta "o ARQUIVO cresceu".
#
# # Uso
#
#   ./scripts/watchdog.sh                    # varre telemetry/ uma vez
#   ./scripts/watchdog.sh --vigiar 300       # vigia por 300 s, avisa se travar
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

# Sem geração nova neste tempo = suspeito. Uma geração leva de 2 s (bateria pequena) a
# ~130 s (24.000 ticks), então 400 s cobre a mais lenta com folga de 3×.
STALL_S=${STALL_S:-400}
WATCH_S=0
[ "${1:-}" = "--vigiar" ] && WATCH_S=${2:-300}

verificar() {
  local achou=0 travados=0
  local agora; agora=$(date +%s)

  echo "── [$(date +%H:%M:%S)] processos de GPU ──"
  local procs; procs=$(pgrep -af "gpu_evolve|cortex_evolve|gpu_loop" 2>/dev/null | grep -v pgrep || true)
  if [ -z "$procs" ]; then
    echo "  (nenhum)"
  else
    echo "$procs" | sed 's/^/  /' | cut -c1-140
  fi

  echo "── logs em andamento ──"
  while IFS= read -r log; do
    [ -f "$log" ] || continue
    local mtime idade gens dir
    mtime=$(stat -c %Y "$log")
    idade=$(( agora - mtime ))
    gens=$(grep -cE "^gen " "$log" 2>/dev/null); gens=${gens:-0}
    dir=$(dirname "$log")
    # Concluída? Três sinais, porque nem todo caminho escreve marcador: o arquivo DONE,
    # o PRONTO (nome antigo) ou a linha final que o próprio binário imprime. Sem o
    # terceiro, uma corrida que TERMINOU aparece como travada — e um watchdog que grita
    # à toa é um watchdog que ninguém lê.
    if [ -f "$dir/DONE" ] || [ -f "$dir/PRONTO" ] || grep -qE "^=== (FIM|RESULTADO FINAL)" "$log" 2>/dev/null; then
      continue
    fi
    achou=1
    if [ "$idade" -gt "$STALL_S" ]; then
      printf "  ⚠ TRAVADO %5ss sem escrever | gens=%-3s %s\n" "$idade" "$gens" "$dir"
      travados=$((travados+1))
      # O que mais ajuda no diagnóstico é a ÚLTIMA linha antes de travar.
      tail -1 "$log" 2>/dev/null | sed 's/^/      última: /' | cut -c1-120
    else
      printf "  ✓ vivo  %5ss atrás      | gens=%-3s %s\n" "$idade" "$gens" "$dir"
    fi
  done < <(find telemetry -name "run.log" -newermt "-6 hours" 2>/dev/null)

  [ "$achou" -eq 0 ] && echo "  (nenhuma corrida em andamento)"
  return "$travados"
}

if [ "$WATCH_S" -gt 0 ]; then
  fim=$(( $(date +%s) + WATCH_S ))
  while [ "$(date +%s)" -lt "$fim" ]; do
    verificar || echo "  ⚠ $? corrida(s) travada(s) — investigar antes de confiar no resultado"
    echo
    sleep 60
  done
else
  verificar
fi
