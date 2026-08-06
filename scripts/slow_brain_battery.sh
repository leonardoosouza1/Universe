#!/usr/bin/env bash
# ABLAÇÃO PAREADA DA METADE LENTA — o System 2 paga?
#
# # A pergunta
#
# O cérebro lento (CfC-64 → Dense → Dense, 1 tick a cada 8) existia no `motor.rs` e não
# existia em NENHUM caminho GPU. Agora existe, é herdável e roda no mesmo encoder do
# rápido. Falta o que nunca foi feito: medir se ele CONTRIBUI.
#
# # Por que ablação e não dois binários
#
# `--sem-lento` zera as três saídas ANTES do consumo, mantendo o órgão rodando: mesmo
# custo de GPU, mesmo estado recorrente, mesma mutação dos pesos no genoma. Só o SINAL
# é removido. Comparar contra um binário sem o órgão mudaria duas variáveis de uma vez.
#
# # Por que várias sementes
#
# Duas condições com uma semente cada não distinguem efeito de sorte. A regra que já me
# custou caro neste projeto: seguir o TAMANHO DO EFEITO conforme n cresce, não o σ de uma
# amostra pequena — um achado meu de "+264%, 4,3σ" virou -207 m com sinal invertido quando
# medido pareado. Aqui: mesmas sementes nas duas condições, comparação pareada por semente.
#
# # A métrica
#
# Média dos últimos 5 (não "melhor de todos", que CRESCE com o número de amostras mesmo
# sem melhora nenhuma — foi assim que uma varredura minha elegeu a condição PERDEDORA).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

BEINGS=${BEINGS:-48}
TICKS=${TICKS:-12000}
GENERATIONS=${GENERATIONS:-20}
SEEDS=${SEEDS:-"1 2 3"}

export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0

# BINÁRIO CONGELADO — não recompila aqui.
#
# A bateria roda seis vezes. Se o binário for recompilado no meio (por exemplo porque
# alguém está limpando o código em paralelo), as condições passam a usar builds
# diferentes e a comparação pareada morre EM SILÊNCIO: os números continuam saindo,
# só não significam mais o que dizem significar.
#
# `gpu_evolve_bateria` é uma cópia tirada antes do início. Congelada, dá para mexer no
# código enquanto o experimento roda.
BIN="$ROOT/target/release/gpu_evolve_bateria"
if [ ! -x "$BIN" ]; then
  echo "binário congelado ausente. Gere com:"
  echo "  nice -n 19 cargo build --release -j 2 --bin gpu_evolve"
  echo "  cp target/release/gpu_evolve target/release/gpu_evolve_bateria"
  exit 1
fi
echo "→ binário congelado: $(md5sum "$BIN" | cut -c1-12)"

for seed in $SEEDS; do
  for condition in com sem; do
    OUTDIR="$ROOT/telemetry/lento_${condition}_s${seed}"
    [ -f "$OUTDIR/PRONTO" ] && { echo "→ ${condition}/s${seed} já concluída"; continue; }
    mkdir -p "$OUTDIR"
    EXTRA_ARGS=""; [ "$condition" = "sem" ] && EXTRA_ARGS="--sem-lento"

    # Proveniência: já rodei binário velho uma vez e os números vieram idênticos até a
    # terceira casa. A impressão digital dos shaders detecta isso.
    { echo "data: $(date -Is)"
      echo "condition: $condition seres=$BEINGS ticks=$TICKS gens=$GENERATIONS seed=$seed"
      md5sum "$BIN" 2>/dev/null
    } > "$OUTDIR/proveniencia.txt"

    echo "→ [$(date +%H:%M:%S)] lento=$condition seed=$seed"
    env -u DISPLAY nice -n 19 "$BIN" \
        --seres "$BEINGS" --ticks "$TICKS" --gens "$GENERATIONS" --seed "$seed" $EXTRA_ARGS \
        --out "$OUTDIR" > "$OUTDIR/run.log" 2>&1
    touch "$OUTDIR/PRONTO"
  done
done

echo; echo "=== COMPARAÇÃO PAREADA ==="
python3 "$ROOT/scripts/compare_slow_brain.py"
