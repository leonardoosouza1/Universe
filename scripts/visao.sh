#!/usr/bin/env bash
# BATERIA DE VISÃO — varia alcance e abertura, que é o que o Leonardo apontou:
# *"o modelo não vê ângulos ou não vê o que está muito longe, ele tem um limite e com base
# na velocidade isso pode atrapalhar"*.
#
# A conta confirma. A retina tem 14 linhas com pitch de −0,012 a −0,060, olho a 0,4 m:
# a linha mais distante enxerga **33,3 m**. E a frenagem exigida até a curva mais fechada
# (raio 9,2 m → v_curva 11,4 m/s):
#
#   25 m/s → 25,2 m   cabe
#   28 m/s → 33,3 m   exatamente no limite
#   30 m/s → 39,2 m   NÃO cabe
#   40 m/s → 74,9 m   2,2× além
#
# Acima de ~28 m/s o carro fica CEGO para o ponto onde precisa começar a frear. As sondas de
# curvatura (20/40/70/120 m) são a compensação — e é por isso que ablá-las custa só 3%: o
# carro opera abaixo de 28 m/s, onde a retina basta.
#
# A pergunta desta bateria: dar mais alcance melhora? E mais ângulo? Ou o gargalo está
# noutro lugar (motor, decisão) e a visão sobra?
#
# ATENÇÃO À ASSIMETRIA DA FÍSICA: a velocidade cega cresce com a RAIZ do alcance.
#   33 m → 28,0 m/s     80 m → 40,9 m/s     120 m → 49,4 m/s
# Triplicar o alcance não dobra a velocidade útil. Se o ganho medido for proporcional ao
# alcance e não à raiz, o efeito não é de percepção — é de outra coisa.
#
# Uso:  scripts/visao.sh [REPS] [GENS] [SWARM]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
REPS=${1:-3}; GENS=${2:-3}; SWARM=${3:-60}
BIN="$RAIZ/target/release/universe"
SAIDA="$RAIZ/telemetry/visao"

[ -x "$BIN" ] || { echo "compile primeiro"; exit 1; }
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X"; exit 1; }
rm -rf "$SAIDA"; mkdir -p "$SAIDA"

# rótulo:alcance_m:fov_graus   (0 = padrão: 33,3 m e 120°)
COND=(
  "base:0:0"
  "perto20:20:0"
  "longe50:50:0"
  "longe80:80:0"
  "longe120:120:0"
  "estreito60:0:60"
  "estreito90:0:90"
  "largo150:0:150"
  "longe80_largo150:80:150"
)

echo "=== BATERIA DE VISÃO — alcance e abertura ==="
echo "reps=$REPS · gens=$GENS · swarm=$SWARM"
echo "padrão: alcance 33,3 m · abertura 120° · cego acima de 28,0 m/s"
echo

n=0
for entry in "${COND[@]}"; do
  IFS=: read -r rot alc fov <<< "$entry"
  for r in $(seq 1 "$REPS"); do
    d="/tmp/visao_${rot}_r${r}"; rm -rf "$d"; mkdir -p "$d"
    cp "$RAIZ/CHAMPION.dna" "$d/"
    [ -f "$RAIZ/universe_config.json" ] && cp "$RAIZ/universe_config.json" "$d/"
    cat > "$d/params.json" <<EOF
{"experiment_name":"visao_${rot}_r${r}","variant_id":0,"axiom_wall_avoidance":0.18,
"axiom_dark_center":0.45,"axiom_whisker":0.15,"axiom_racing_line":0.26,
"axiom_decay_divisor":50.0,"axiom_floor":0.35,"k_smooth":100.0,"k_speed":0.5,
"k_novel":50.0,"k_stab":200.0,"k_phi":300.0,"k_aesth":200.0,"lap_bonus":50000.0,
"survival_bonus":3000.0,"death_penalty":2000.0,"via_ablada":"","nascimento":"",
"visao_alcance_m":$alc,"visao_fov_deg":$fov,
"speed_bleed_damage":15.0,"stagnation_secs":15.0,"wall_t_threshold":40}
EOF
    ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
        --lab-params "$d/params.json" > "$SAIDA/${rot}_r${r}.log" 2>&1 ) &
    n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait   # 3 e não 4: divide CPU com a bateria em curso
  done
done
wait
xdpyinfo >/dev/null 2>&1 || echo "⚠️  O X CAIU DURANTE A BATERIA."
echo "corridas terminadas ($n)."; echo

python3 - "$SAIDA" <<'PY'
import glob, math, os, re, statistics as st, sys
d = sys.argv[1]
rx = re.compile(r"net_max=([\d.-]+) net_med=([\d.-]+)")
rxl = re.compile(r"Laps: 0v=(\d+) 1v=(\d+) 2v=(\d+) 3\+v=(\d+)")
rxv = re.compile(r"visão: alcance ([\d.]+) m · abertura ([\d.]+)° · cega acima de ([\d.]+)")
por, mortas = {}, []
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    rot = os.path.basename(f).rsplit("_r", 1)[0]
    txt = open(f, errors="ignore").read()
    v = [(float(m.group(1)), float(m.group(2))) for m in rx.finditer(txt)]
    if not v:
        mortas.append(os.path.basename(f)); continue
    laps = [(int(m.group(2))+int(m.group(3))+int(m.group(4))) for m in rxl.finditer(txt)]
    cfg = rxv.search(txt)
    por.setdefault(rot, []).append((
        max(x[0] for x in v), max(x[1] for x in v), max(laps) if laps else 0,
        (float(cfg.group(1)), float(cfg.group(2)), float(cfg.group(3))) if cfg else (0,0,0)))

if mortas:
    print(f"⚠️  {len(mortas)} corrida(s) sem dados: {', '.join(mortas[:5])}\n")
if "base" not in por:
    print("sem linha de base"); raise SystemExit(1)
b = [x[0] for x in por["base"]]
mb = st.mean(b); epb = st.stdev(b)/len(b)**0.5 if len(b) > 1 else 0.0

print(f"{'condição':<18} {'alcance':>8} {'abert.':>7} {'v cega':>8} {'net_max':>9} "
      f"{'±ep':>6} {'Δ vs base':>11} {'volta':>6} {'reps':>5}")
print("-" * 90)
for rot in [x.split(":")[0] for x in
            ["base","perto20","longe50","longe80","longe120",
             "estreito60","estreito90","largo150","longe80_largo150"]]:
    if rot not in por: continue
    v = por[rot]
    x = [y[0] for y in v]
    m = st.mean(x); ep = st.stdev(x)/len(x)**0.5 if len(x) > 1 else 0.0
    dl = (m - mb)/mb*100
    alc, fov, vc = v[0][3]
    print(f"{rot:<18} {alc:>7.0f}m {fov:>6.0f}° {vc:>7.1f} {m:>9.0f} {ep:>6.0f} "
          f"{dl:>+10.1f}% {st.mean([y[2] for y in v]):>5.1f} {len(x):>5}")
print("-" * 90)
print("\nCOMO LER: a velocidade cega cresce com a RAIZ do alcance (33m→28,0 · 80m→40,9 ·")
print("120m→49,4 m/s). Se o ganho medido acompanhar a raiz, é percepção. Se for plano,")
print("o gargalo está noutro lugar e a visão já sobrava.")
PY
