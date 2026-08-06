#!/usr/bin/env bash
# FILA AUTÔNOMA — mantém a GPU trabalhando enquanto o Leonardo está fora.
#
# Regra dele: "não pare até eu dizer que pare" e "não deixe a GPU queimando à toa".
# As duas juntas significam: fila cheia, mas só de experimentos VÁLIDOS.
#
# Todas as condições usam `--same-start`. Sem isso a fitness mede a largada, não o cérebro:
# o mesmo genoma reavaliado rendia 62% (pior caso 3%). Com largada única sobe para 98% e a
# mediana da população dobra (181 → 370 m). Qualquer bateria sem essa flag mede ruído.
#
# Ordem pensada para responder na sequência certa:
#   1. sigma      — agora que a seleção FUNCIONA, qual mutação faz a população progredir?
#                   (a monocultura apareceu: med/max = 0,97, ninguém explora)
#   2. pressao    — elitismo e torneio: quanto de pressão seletiva antes de colapsar
#   3. tempo      — 8k/24k/48k: onde satura o orçamento de vida
#   4. lento      — o cérebro lento remedido com seleção limpa (+27,3 m veio da lente ruidosa)
#   5. skip       — velocidade de decisão, o "até certo ponto ajudava" do Leonardo
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0

SEEDS=${SEEDS:-"1 2 3 4"}
BEINGS=${BEINGS:-48}
JOBS=${JOBS:-3}
LOG="$ROOT/telemetry/queue_overnight.log"

BIN="$ROOT/target/release/gpu_evolve_queue"
nice -n 19 cargo build --release -j 2 --bin gpu_evolve 2>&1 | grep -E "^error" -A 4 && exit 1
cp -f "$ROOT/target/release/gpu_evolve" "$BIN"
echo "[$(date +%H:%M:%S)] binário congelado: $(md5sum "$BIN" | cut -c1-12)" | tee -a "$LOG"

# grupo|nome|ticks|gens|flags
EXPERIMENTS=(
  "sigma|s005|12000|16|--same-start --sigma 0.05"
  "sigma|s015|12000|16|--same-start --sigma 0.15"
  "sigma|s030|12000|16|--same-start --sigma 0.30"
  "pressao|elite1|12000|16|--same-start --n-elite 1 --tourney 5"
  "pressao|elite8|12000|16|--same-start --n-elite 8 --tourney 2"
  "tempo|t48k|48000|10|--same-start"
  "lento|com|12000|16|--same-start"
  "lento|sem|12000|16|--same-start --sem-lento"
  "skip|k2|12000|16|--same-start --brain-skip 2"
  "skip|k4|12000|16|--same-start --brain-skip 4"
)

run_one() {
  local group="$1" name="$2" ticks="$3" gens="$4" flags="$5" seed="$6"
  local outdir="$ROOT/telemetry/q_${group}_${name}_s${seed}"
  [ -f "$outdir/DONE" ] && return
  mkdir -p "$outdir"
  { echo "date: $(date -Is)"
    echo "cond: $group/$name ticks=$ticks gens=$gens flags=[$flags] seed=$seed"
    md5sum "$BIN"; } > "$outdir/provenance.txt"
  env -u DISPLAY nice -n 19 "$BIN" --seres "$BEINGS" --ticks "$ticks" --gens "$gens" \
      --seed "$seed" $flags --out "$outdir" > "$outdir/run.log" 2>&1
  local rc=$? got
  got=$(grep -cE "^gen " "$outdir/run.log" 2>/dev/null); got=${got:-0}
  if [ "$rc" -eq 0 ] && [ "$got" -eq "$gens" ]; then
    touch "$outdir/DONE"; echo "[$(date +%H:%M:%S)] ✓ $group/$name/s$seed" | tee -a "$LOG"
  else
    echo "[$(date +%H:%M:%S)] ✗ $group/$name/s$seed rc=$rc gens=$got/$gens" | tee -a "$LOG"
  fi
}

for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r group name ticks gens flags <<< "$e"
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done
    run_one "$group" "$name" "$ticks" "$gens" "$flags" "$seed" &
  done
done
wait
echo "[$(date +%H:%M:%S)] === FILA CONCLUÍDA ===" | tee -a "$LOG"
python3 "$ROOT/scripts/compare_queue.py" | tee -a "$LOG"
