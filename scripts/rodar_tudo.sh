#!/usr/bin/env bash
# Sequência autônoma: valida a evolução, mede os sentidos, mede o priming.
#
# A ORDEM IMPORTA e não é arbitrária:
#
# 1. VALIDAR A EVOLUÇÃO primeiro. Medir alocação em cima de treino quebrado seria repetir o
#    erro que o próprio conserto do currículo desfez. Se o colapso persistir, os passos
#    seguintes medem o colapso junto com o efeito.
#
# 2. SENTIDOS. Quanto cada grupo sensorial carrega — inclusive a visão, que ocupa 80 dos 223
#    canais. Necessário porque o mapa de PESOS não responde: o peso por canal é praticamente
#    idêntico entre os seis grupos (9,5–10,7) e a fatia da visão (35,4%) é exatamente sua
#    fração de canais. Alocação de pesos indistinguível de uniforme — só ablação separa.
#
# 3. PRIMING. A pergunta que a primeira tabela abriu e não fechou: o córtex carrega 94,7%,
#    mas os 3 neurônios motores nascem com 15 pesos ESCRITOS À MÃO, todos SATURADOS no
#    campeão treinado (ou seja, sobreviveram intactos à evolução). Se ablar só esses 15
#    custar quase o mesmo que ablar o córtex inteiro, o "cérebro" são as regras escritas à
#    mão e a evolução é decoração.
#
# 4. INDEPENDÊNCIA. Três modos de nascimento, do zero: com os canais do humano, com canais
#    sorteados (mesma esparsidade), e denso. Responde se a evolução acha sozinha o que foi
#    escrito à mão — e o `sinapses --priming` mostra quantas das 15 conexões ela reencontrou.
#
# Uso:  scripts/rodar_tudo.sh
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"

xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X — o simulador precisa de GPU, xvfb não serve"; exit 1; }

echo "############ 1/3 · VALIDAÇÃO DA EVOLUÇÃO $(date +%H:%M) ############"
./scripts/valida_evolucao.sh 4 10 60

echo; echo "############ 2/3 · SENTIDOS $(date +%H:%M) ############"
./scripts/alocacao.sh 4 3 60 "visao curvatura paredes pista corpo movimento"

echo; echo "############ 3/3 · PRIMING vs CÓRTEX $(date +%H:%M) ############"
./scripts/alocacao.sh 5 3 60 "priming cortex"

echo; echo "############ 4/4 · INDEPENDÊNCIA $(date +%H:%M) ############"
./scripts/independencia.sh 3 25 60

echo; echo "############ FIM $(date +%H:%M) ############"
