#!/usr/bin/env bash
# Qual fração da população vale sacrificar em exploração?
#
# Medido nos relatórios: com o padrão de 0,50 a MEDIANA da população é ZERO metros. Metade
# não sai do lugar, e `wall_mean = 0,00s` mostra que não é batida nem parede — é a mutação
# destruindo o genoma. A população EFETIVA é 30, não 60.
#
# Exploração de menos estagna; de mais desperdiça. Não há razão a priori para 0,50 ser o
# ponto. Isto mede.
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
GENS=${1:-12}; SWARM=${2:-60}; REPS=${3:-3}
BIN="$RAIZ/target/release/universe"; SAIDA="$RAIZ/telemetry/exploracao"
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X"; exit 1; }
rm -rf "$SAIDA"; mkdir -p "$SAIDA"
echo "=== FRAÇÃO EXPLORADORA — quanto sacrificar? ==="
echo "gens=$GENS swarm=$SWARM reps=$REPS · padrão = 0,50 (mediana zero)"; echo
n=0
for f in 0.10 0.50; do    # só as duas que decidem: o orçamento vai em REPETIÇÕES
  rot="f$(echo $f | tr -d '.')"
  for r in $(seq 1 $REPS); do
    d="/tmp/expl_${rot}_r${r}"; rm -rf "$d"; mkdir -p "$d"
    cp "$RAIZ/CHAMPION.dna" "$d/"
    [ -f "$RAIZ/universe_config.json" ] && cp "$RAIZ/universe_config.json" "$d/"
    cat > "$d/params.json" <<EOF
{"experiment_name":"expl_${rot}_r${r}","variant_id":0,"axiom_wall_avoidance":0.18,
"axiom_dark_center":0.45,"axiom_whisker":0.15,"axiom_racing_line":0.26,
"axiom_decay_divisor":50.0,"axiom_floor":0.35,"k_smooth":100.0,"k_speed":0.5,
"k_novel":50.0,"k_stab":200.0,"k_phi":300.0,"k_aesth":200.0,"lap_bonus":50000.0,
"survival_bonus":3000.0,"death_penalty":2000.0,"via_ablada":"","nascimento":"",
"fracao_exploradora":$f,
"speed_bleed_damage":15.0,"stagnation_secs":15.0,"wall_t_threshold":40}
EOF
    ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
        --lab-params "$d/params.json" > "$SAIDA/${rot}_r${r}.log" 2>&1 ) &
    n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait
  done
done
wait
echo "terminado ($n)."; echo
python3 - "$SAIDA" <<'PY'
import glob,os,re,statistics as st,sys
d=sys.argv[1]
rx=re.compile(r"NET max=([\d.]+)m median=([\d.]+)m \| Deaths (\d+)/(\d+)")
rxl=re.compile(r"Laps: 0v=(\d+) 1v=(\d+) 2v=(\d+) 3\+v=(\d+)")
por={}
for f in sorted(glob.glob(os.path.join(d,"*.log"))):
    rot=os.path.basename(f).rsplit("_r",1)[0]
    t=open(f,errors="ignore").read()
    g=[(float(m.group(1)),float(m.group(2)),int(m.group(3))/max(int(m.group(4)),1)) for m in rx.finditer(t)]
    l=[(int(m.group(2))+int(m.group(3))+int(m.group(4)))/60 for m in rxl.finditer(t)]
    if g: por.setdefault(rot,[]).append((g,l))
if not por: print("sem dados"); raise SystemExit
print(f"{'fração':>8} {'net_max':>9} {'MEDIANA':>9} {'≥1 volta':>10} {'mortal.':>9} {'reps':>5}")
print("-"*56)
for rot in ["f010","f050"]:
    if rot not in por: continue
    v=por[rot]; q=max(len(v[0][0])//3,1)
    mx=st.mean([st.mean([x[0] for x in g[-q:]]) for g,_ in v])
    md=st.mean([st.mean([x[1] for x in g[-q:]]) for g,_ in v])
    mo=st.mean([st.mean([x[2] for x in g[-q:]]) for g,_ in v])
    vo=st.mean([st.mean(l[-q:]) if l else 0 for _,l in v])
    print(f"{rot[1:]:>8} {mx:>8.0f}m {md:>8.0f}m {vo:>9.0%} {mo:>8.0%} {len(v):>5}")
print("-"*56)
print("\nA MEDIANA é a coluna que importa: ela mede quanto da população está viva de fato.")
PY
