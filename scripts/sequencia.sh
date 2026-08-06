#!/usr/bin/env bash
# As baterias na ordem certa: as CURTAS primeiro, a longa por último.
#
# A independência precisa de 25 gerações × 9 corridas ≈ 4 h, e estava bloqueando duas
# baterias que respondem em ~20 min cada — inclusive a dos sentidos, que mede quanto a visão
# carrega. Resultado cedo vale mais que resultado em ordem bonita.
#
# 1. SENTIDOS — quanto cada grupo sensorial carrega. Necessário porque o mapa de PESOS não
#    responde: peso por canal 9,5–10,7 nos seis grupos, e a fatia da visão (35,4%) é
#    exatamente sua fração de canais. Alocação indistinguível de uniforme.
#
# 2. PRIMING vs CÓRTEX — o córtex carrega 94,7%, mas os 3 neurônios motores nascem com 15
#    pesos escritos à mão. Se ablar só esses 15 custar quase o mesmo, o "cérebro" são as
#    regras e a evolução é decoração.
#
# 3. INDEPENDÊNCIA — três modos de nascimento, do zero, 25 gerações. A evolução acha sozinha
#    os canais que o humano escolheu?
set -uo pipefail
cd "$(dirname "$0")/.."

echo "############ 1/3 SENTIDOS $(date +%H:%M) ############"
./scripts/alocacao.sh 4 3 60 "visao curvatura paredes pista corpo movimento"

echo; echo "############ 2/3 PRIMING vs CÓRTEX $(date +%H:%M) ############"
./scripts/alocacao.sh 5 3 60 "priming cortex"

echo; echo "############ 3/3 INDEPENDÊNCIA $(date +%H:%M) ############"
./scripts/independencia.sh 3 25 60

echo; echo "############ FIM $(date +%H:%M) ############"
