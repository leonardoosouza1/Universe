#!/usr/bin/env bash
# Roda o simulador SEMPRE num diretório isolado. Use isto, nunca o binário direto.
#
# POR QUE ESTE INVÓLUCRO EXISTE
#
# O simulador LÊ e SOBRESCREVE `CHAMPION.dna` no diretório de trabalho. Rodar da raiz do
# projeto — mesmo um teste de 1 geração — coroa um campeão fraco e apaga o treinado.
# Aconteceu DUAS vezes nesta sessão: a segunda foi um teste de fumaça da ablação de visão
# que deixou o campeão em 567 m no lugar de 3314, e contaminou uma bateria inteira que já
# estava rodando, porque ela copia o campeão da raiz.
#
# Só deu para desfazer porque o arquivo está versionado no git. Este invólucro remove a
# possibilidade: cria /tmp/<rótulo>, copia o campeão e a config para lá, e roda ali dentro.
#
# Uso:  scripts/rodar.sh <rótulo> [argumentos do universe...]
#   scripts/rodar.sh fumaca --headless --gens 1 --swarm 30
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
ROTULO="${1:?uso: rodar.sh <rótulo> [args]}"; shift

BIN="$RAIZ/target/release/universe"
[ -x "$BIN" ] || { echo "compile primeiro: cargo build --release"; exit 1; }
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X — o simulador precisa de GPU, xvfb não serve"; exit 1; }

DIR="/tmp/run_${ROTULO}"
rm -rf "$DIR"; mkdir -p "$DIR"
cp "$RAIZ/CHAMPION.dna" "$DIR/" 2>/dev/null
[ -f "$RAIZ/universe_config.json" ] && cp "$RAIZ/universe_config.json" "$DIR/"

cd "$DIR" && exec "$BIN" "$@"
