#!/usr/bin/env bash
# BATERIA DE VARIAÇÕES — roda condições em SÉRIE e compara pela média dos últimos 10.
#
# # Por que em série e por que essa métrica
#
# Em série porque só há uma GPU: duas corridas em paralelo disputam e envenenam o tempo de
# ambas. Pela média dos últimos 10 porque "melhor de todos" CRESCE com o número de amostras
# mesmo sem melhora nenhuma — foi assim que uma varredura minha de 15 gerações escolheu a
# condição PERDEDORA, e só as 100 gerações completas mostraram o erro (folha saturada
# max10=1270 m contra 792 m da que a varredura tinha eleito).
#
# Cada condição grava proveniência (md5 dos shaders + data), porque já rodei binário velho
# uma vez e os números vieram idênticos até a terceira casa.
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"

echo "→ recompilando…"
nice -n 19 cargo build --release -j 2 --bin cortex_evolve 2>&1 | grep -E "^error" -A 4 && exit 1

# nome|seres|ticks|gens|fat_beta|div_sigma|extra
# A pergunta desta bateria: qual PROPORÇÃO entre as partes serve ESTE corpo?
#
# A razão 1:4 (80% cerebelo) é o fato conservado em mamíferos — corpos com centenas de
# músculos. Nosso corpo tem TRÊS atuadores. O achado N1 do ecossistema ("o corpo dita a
# arquitetura", corr +0,972) prevê que a proporção ótima acompanha o corpo. E foi medido:
# bipartite 1:4 ficou ABAIXO da homogênea (med10 110 m contra 164 m).
#
# A faixa de `schedule_ext` foi aberta de 0,5 para 2,0 — antes a evolução não conseguia
# sequer TESTAR córtex-pesado.
CONDICOES=(
  "razao_livre|60|8000|80|0.0|0.0|"
  "vida_longa|60|24000|40|0.0|0.0|"
  # ACHADO: vida_longa (24000 ticks) deu 1,97 VOLTAS em 8 gerações — a primeira volta
  # completa da sessão. A variável foi TEMPO DE VIDA, não arquitetura. Parte do ganho é
  # mecânica (3x mais ticks = 3x mais distância possível), mas max10 subiu 76% contra a
  # folha de 100 gerações, mais do que 3x explicaria, e a MEDIANA subiu junto (232 m, a
  # melhor de todas) — outlier não move mediana.
  #
  # Estas duas isolam o efeito: vida ainda mais longa, e vida longa SEM partes.
  # MEDIDO em vida_longa (24k ticks): ≥1V em 7 de 24 gerações, e ≥2V=1 na gen 21 (6648 m).
  # A frequência DOBROU da 1a para a 2a metade (2/12 → 5/12) e a mediana subiu 254→320 m,
  # com mortalidade 0,3% — é aprendizado, não outlier.
  #
  # 48k ticks testa se mais tempo ainda paga ou se satura. Se a volta já virou rotina, o
  # gargalo passa a ser velocidade média, não sobrevivência.
  "vida_muito_longa|60|48000|30|0.0|0.0|"
  "vida_longa_homog|60|24000|30|0.0|0.0|--homogeneous"
  "pop_grande|120|8000|60|0.0|0.0|"
  "homogenea_ref|60|8000|80|0.0|0.0|--homogeneous"
)

for c in "${CONDICOES[@]}"; do
  IFS='|' read -r nome seres ticks gens fb ds extra <<< "$c"
  SAIDA="$RAIZ/telemetry/bat_$nome"
  [ -f "$SAIDA/PRONTO" ] && { echo "→ $nome já concluída, pulando"; continue; }
  mkdir -p "$SAIDA"
  { echo "data: $(date -Is)"
    echo "cond: seres=$seres ticks=$ticks gens=$gens fat_beta=$fb div_sigma=$ds $extra"
    for f in src/gpu/cortex_sheet.wgsl src/gpu/perception.wgsl src/bin/cortex_evolve.rs; do
      md5sum "$f" 2>/dev/null
    done
  } > "$SAIDA/proveniencia.txt"

  echo "→ [$(date +%H:%M:%S)] $nome: seres=$seres ticks=$ticks gens=$gens"
  env -u DISPLAY nice -n 19 ./target/release/cortex_evolve \
      --seres "$seres" --ticks "$ticks" --gens "$gens" \
      --fat-beta "$fb" --div-sigma "$ds" --map-interval 25 $extra \
      > "$SAIDA/run.log" 2>&1
  touch "$SAIDA/PRONTO"
  echo "→ [$(date +%H:%M:%S)] $nome concluída"
done

echo; echo "=== BATERIA CONCLUÍDA ==="
python3 "$RAIZ/scripts/comparar_bateria.py"
