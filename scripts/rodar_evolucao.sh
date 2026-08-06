#!/usr/bin/env bash
# Lança a evolução SEMPRE recompilando antes.
#
# # Por que este script existe
#
# Mudei `brain.wgsl` (fadiga adaptativa), relancei `gpu_evolve` e os números vieram
# IDÊNTICOS aos de antes, geração a geração, até a terceira casa. Passei um tempo procurando
# bug na lógica da fadiga — e a lógica estava certa. **O binário é que não tinha sido
# recompilado.**
#
# A lição: num projeto onde o kernel é `include_str!` de um `.wgsl`, mudar o shader NÃO
# invalida o binário aos olhos de quem só olha o código. O rebuild tem de ser parte do
# lançamento, não um passo separado que se pode esquecer.
#
# E o sinal que denunciou: **valor idêntico entre condições que deveriam diferir.** É a mesma
# assinatura de prova fabricada que apareceu 4 vezes nesta sessão.
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
NOME=${1:-evolucao}
SAIDA="$RAIZ/telemetry/$NOME"

echo "→ recompilando (nice, -j2 para não travar a máquina)…"
nice -n 19 cargo build --release -j 2 --bin gpu_evolve 2>&1 | grep -E "^error" -A 4 && exit 1

mkdir -p "$SAIDA"
# Registra a impressão digital do que está rodando, para a corrida ser rastreável depois.
{ echo "data: $(date -Is)"
  echo "binário: $(stat -c %y target/release/gpu_evolve)"
  for f in src/gpu/brain.wgsl src/gpu/physics.wgsl src/gpu/perception.wgsl src/bin/gpu_evolve.rs; do
    echo "  $(md5sum "$f" 2>/dev/null)"
  done
} > "$SAIDA/proveniencia.txt"

nohup env -u DISPLAY ./target/release/gpu_evolve > "$SAIDA/run.log" 2>&1 &
echo "→ evolução rodando (PID $!) · log em $SAIDA/run.log"
echo "→ proveniência em $SAIDA/proveniencia.txt"
