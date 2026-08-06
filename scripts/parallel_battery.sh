#!/usr/bin/env bash
# BATERIA PARALELA — várias condições ao mesmo tempo na única GPU.
#
# # Por que paralelo, se antes eu insistia em série
#
# Medido em 2026-08-05: com UM processo a GPU fica em 68% de uso. Três processos
# simultâneos (dois sondas + uma bateria de 48 seres) completaram sem erro e sem derrubar
# nenhum. O gargalo de um processo pequeno é latência de lançamento de kernel, não
# throughput — é o mesmo diagnóstico já registrado no ecossistema para geração em lote.
#
# Então serializar estava DESPERDIÇANDO a máquina. O que não pode é recompilar no meio:
# por isso o binário é congelado antes de começar.
#
# # Por que baterias PEQUENAS
#
# Uma condição de 20 gerações × 12.000 ticks leva 16 min. Quatro condições em série são
# mais de uma hora para responder uma pergunta. Com 8 gerações × 6.000 ticks × 32 seres
# cada condição sai em ~1 min, e quatro em paralelo respondem em ~1 min. Isso muda o que
# dá para perguntar: vira exploração, não expedição.
#
# Sinal já validado nessa escala: a fração de ticks colado na parede sai de 1,2% para
# 10,8% entre a geração 0 e a 1. Não é preciso 20 gerações para ver o fenômeno.
#
# # Uso
#
#   ./scripts/parallel_battery.sh "0 20 60 150" "1 2 3"
#                                  \_ custos _/  \_ sementes _/
#
# Cada combinação custo×semente vira uma corrida. Rodam JOBS por vez.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

COSTS="${1:-0 20 60}"
SEEDS="${2:-1 2 3}"
BEINGS=${BEINGS:-32}
TICKS=${TICKS:-6000}
GENERATIONS=${GENERATIONS:-8}
JOBS=${JOBS:-3}
TAG=${TAG:-wall}
# Qual eixo esta bateria varre. `--wall-cost` cobra em DISTÂNCIA (mesma moeda da fitness);
# `--wall-life` cobra em TEMPO DE VIDA (moeda diferente). A Lei da Moeda Cruzada prevê que
# só a segunda funciona, e a primeira já foi medida: reduziu o encosto 5× destruindo 64%
# da distância. `--brain-skip` varre a velocidade de decisão.
FLAG=${FLAG:---wall-cost}

export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0

BIN="$ROOT/target/release/gpu_evolve_par_$TAG"
echo "→ congelando binário…"
nice -n 19 cargo build --release -j 2 --bin gpu_evolve 2>&1 | grep -E "^error" -A 4 && exit 1
cp -f "$ROOT/target/release/gpu_evolve" "$BIN"
echo "→ binário: $(md5sum "$BIN" | cut -c1-12)"

# Semáforo por descritor: no máximo JOBS processos de GPU ao mesmo tempo.
run_one() {
  local cost="$1" seed="$2"
  local outdir="$ROOT/telemetry/${TAG}_c${cost}_s${seed}"
  [ -f "$outdir/DONE" ] && { echo "  ↷ c$cost/s$seed já pronta"; return; }
  mkdir -p "$outdir"
  { echo "date: $(date -Is)"
    echo "cond: wall_cost=$cost seed=$seed beings=$BEINGS ticks=$TICKS gens=$GENERATIONS"
    md5sum "$BIN"
  } > "$outdir/provenance.txt"

  env -u DISPLAY nice -n 19 "$BIN" \
      --seres "$BEINGS" --ticks "$TICKS" --gens "$GENERATIONS" \
      --seed "$seed" $FLAG "$cost" --out "$outdir" \
      > "$outdir/run.log" 2>&1
  local rc=$?
  # DONE só é escrito se o processo saiu limpo E o log tem as gerações esperadas.
  # Sem isso, uma corrida morta no meio seria lida como concluída e a média dos
  # "últimos N" viria de uma janela diferente das outras — já aconteceu aqui.
  local got; got=$(grep -cE "^gen " "$outdir/run.log" 2>/dev/null || echo 0)
  if [ "$rc" -eq 0 ] && [ "$got" -eq "$GENERATIONS" ]; then
    touch "$outdir/DONE"
    echo "  ✓ c$cost/s$seed ($got gens)"
  else
    echo "  ✗ c$cost/s$seed INCOMPLETA rc=$rc gens=$got/$GENERATIONS"
  fi
}

echo "→ [$(date +%H:%M:%S)] custos=[$COSTS] sementes=[$SEEDS] · $JOBS por vez"
pids=()
for cost in $COSTS; do
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    run_one "$cost" "$seed" &
    pids+=($!)
  done
done
wait

echo; echo "=== BATERIA PARALELA CONCLUÍDA [$(date +%H:%M:%S)] ==="
python3 "$ROOT/scripts/compare_wall_cost.py" "$TAG"
