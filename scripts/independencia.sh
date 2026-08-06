#!/usr/bin/env bash
# A evolução acha sozinha o que o humano escreveu? — três modos de nascimento, do zero.
#
# O priming de `spawn.rs` faz DUAS coisas de uma vez, e é preciso separá-las:
#
#   1. ESPARSIDADE — o neurônio motor conecta a poucos canais. Prior arquitetural, e é o que
#      resolve o SNR de 0,26 que o próprio comentário do código descreve ("~110 random inputs
#      drown the curvature signal").
#   2. QUAIS CANAIS — curvatura a 70 m → freio, a 120 m → esterço. Conhecimento humano.
#
# As três condições:
#
#   primado   esparso E com os canais escolhidos à mão      (o que existe hoje)
#   esparso   esparso com canais SORTEADOS                  (independente: herda a
#                                                            arquitetura, decide o conteúdo)
#   denso     aleatório sem esparsidade                     (piso: o sinal afogado)
#
# SEM campeão herdado — cada corrida nasce do zero, senão a comparação mediria o campeão e
# não o modo de nascimento.
#
# No fim, `sinapses --priming` mostra quantas das 15 conexões escritas à mão cada cérebro
# ACHOU sozinho, e quantas ele inventou por conta própria.
#
# Uso:  scripts/independencia.sh [REPS] [GENS] [SWARM]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
REPS=${1:-3}; GENS=${2:-25}; SWARM=${3:-60}
BIN="$RAIZ/target/release/universe"
SAIDA="$RAIZ/telemetry/independencia"

[ -x "$BIN" ] || { echo "compile primeiro"; exit 1; }
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X"; exit 1; }
rm -rf "$SAIDA"; mkdir -p "$SAIDA"

echo "=== A EVOLUÇÃO ACHA O QUE O HUMANO ESCREVEU? ==="
echo "reps=$REPS · gens=$GENS · swarm=$SWARM · SEM campeão herdado (nasce do zero)"
echo

n=0
for modo in primado esparso denso; do
  arg=$([ "$modo" = "primado" ] && echo "" || echo "$modo")
  for r in $(seq 1 "$REPS"); do
    d="/tmp/indep_${modo}_r${r}"; rm -rf "$d"; mkdir -p "$d"
    # NÃO copia CHAMPION.dna: cada corrida começa do zero
    [ -f "$RAIZ/universe_config.json" ] && cp "$RAIZ/universe_config.json" "$d/"
    cat > "$d/params.json" <<EOF
{"experiment_name":"indep_${modo}_r${r}","variant_id":0,"axiom_wall_avoidance":0.18,
"axiom_dark_center":0.45,"axiom_whisker":0.15,"axiom_racing_line":0.26,
"axiom_decay_divisor":50.0,"axiom_floor":0.35,"k_smooth":100.0,"k_speed":0.5,
"k_novel":50.0,"k_stab":200.0,"k_phi":300.0,"k_aesth":200.0,"lap_bonus":50000.0,
"survival_bonus":3000.0,"death_penalty":2000.0,"via_ablada":"","nascimento":"$arg",
"speed_bleed_damage":15.0,"stagnation_secs":15.0,"wall_t_threshold":40}
EOF
    ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
        --lab-params "$d/params.json" > "$SAIDA/${modo}_r${r}.log" 2>&1 ) &
    n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait
  done
done
wait
xdpyinfo >/dev/null 2>&1 || echo "⚠️  O X CAIU DURANTE A BATERIA."
echo "corridas terminadas ($n)."; echo

python3 - "$SAIDA" <<'PY'
import glob, os, re, statistics as st, sys
d = sys.argv[1]
rxg = re.compile(r"GEN (\d+) \| Laps: 0v=(\d+) 1v=(\d+) 2v=(\d+) 3\+v=(\d+) \| "
                 r"NET max=([\d.]+)m median=([\d.]+)m \| Deaths (\d+)/(\d+)")
por, mortas = {}, []
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    modo = os.path.basename(f).rsplit("_r", 1)[0]
    txt = open(f, errors="ignore").read()
    gs = [(int(m.group(1)), float(m.group(6)),
           (int(m.group(3))+int(m.group(4))+int(m.group(5)))/max(int(m.group(9)),1))
          for m in rxg.finditer(txt)]
    if not gs:
        mortas.append(os.path.basename(f)); continue
    por.setdefault(modo, []).append(gs)

if mortas:
    print(f"⚠️  {len(mortas)} corrida(s) sem dados: {', '.join(mortas[:5])}\n")
if not por:
    print("nenhum dado"); raise SystemExit(1)

print(f"{'modo':<10} {'net_max final':>14} {'±ep':>7} {'melhor de todas':>16} "
      f"{'≥1 volta no fim':>16} {'reps':>5}")
print("-" * 74)
for modo in ["primado", "esparso", "denso"]:
    if modo not in por: continue
    finais = [g[-1][1] for g in por[modo]]
    melhores = [max(x[1] for x in g) for g in por[modo]]
    voltas = [g[-1][2] for g in por[modo]]
    m = st.mean(finais)
    ep = st.stdev(finais)/len(finais)**0.5 if len(finais) > 1 else 0.0
    print(f"{modo:<10} {m:>13.0f}m {ep:>6.0f}m {st.mean(melhores):>15.0f}m "
          f"{st.mean(voltas):>15.0%} {len(finais):>5}")
print("-" * 74)
print("\nCOMO LER: 'primado' herda os canais do humano; 'esparso' herda só a arquitetura e")
print("sorteia os canais; 'denso' não herda nada. A distância entre 'esparso' e 'primado' é")
print("o que o conhecimento humano vale DEPOIS de a evolução ter chance de achar sozinha.")
PY

echo; echo "=== O QUE CADA CÉREBRO ACHOU SOZINHO ==="
echo "quantas das 15 conexões escritas à mão cada cérebro reencontrou, e quantas inventou"
echo
printf "%-10s %22s %26s %6s\n" "modo" "achou do humano (de 15)" "inventou nos motores" "reps"
echo "----------------------------------------------------------------------"
for modo in primado esparso denso; do
  achou=(); inventou=()
  for r in $(seq 1 "$REPS"); do
    dna="/tmp/indep_${modo}_r${r}/CHAMPION.dna"
    [ -f "$dna" ] || continue
    saida=$(./target/release/sinapses "$dna" --priming 2>/dev/null)
    a=$(echo "$saida" | grep -oE "^  [0-9]+/15" | grep -oE "^ +[0-9]+" | tr -d ' ')
    i=$(echo "$saida" | grep -oE "NÃO ESCREVEU: [0-9]+" | grep -oE "[0-9]+")
    [ -n "$a" ] && achou+=("$a")
    [ -n "$i" ] && inventou+=("$i")
  done
  if [ ${#achou[@]} -eq 0 ]; then
    printf "%-10s %22s %26s %6s\n" "$modo" "—" "—" "0"
    continue
  fi
  ma=$(printf '%s\n' "${achou[@]}" | awk '{s+=$1} END {printf "%.1f", s/NR}')
  mi=$(printf '%s\n' "${inventou[@]}" | awk '{s+=$1} END {printf "%.0f", s/NR}')
  printf "%-10s %22s %26s %6s\n" "$modo" "$ma" "$mi" "${#achou[@]}"
done
echo "----------------------------------------------------------------------"
echo
echo "COMO LER:"
echo "  'primado' nasce COM as 15 — se ele terminar com menos, a evolução as descartou."
echo "  'esparso' nasce com canais sorteados — o que ele tiver das 15 foi ACHADO sozinho."
echo "  Se 'esparso' achar poucas mas correr bem, existe mais de um caminho para pilotar,"
echo "  e o do humano é só um deles. Se achar poucas e correr mal, o conhecimento injetado"
echo "  era o cérebro."
