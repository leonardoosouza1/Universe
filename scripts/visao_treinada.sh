#!/usr/bin/env bash
# MAIS VISÃO AJUDA? — a versão que a pergunta exige: TREINAR com cada visão, não testar.
#
# A bateria `visao.sh` deu o resultado errado pela razão certa: pegou o campeão treinado com
# alcance 33 m e mudou a visão dele. Medido: alcance 20 m dá −6,0% e alcance 50 m dá −5,4%.
# **Mudar em qualquer direção piora**, porque os pesos co-adaptaram àquela geometria exata.
#
# É a mesma armadilha do EXP10 em Python: ablar o modulador de um núcleo que co-adaptou a ele
# mede DESARRANJO, não contribuição. Aqui, mudar a visão de um cérebro que co-adaptou a ela
# mede desarranjo, não o valor da visão.
#
# A pergunta real — *"o carro enxerga longe o bastante para a velocidade que anda?"* — só se
# responde deixando cada condição EVOLUIR com a sua própria visão, do zero.
#
# A física, para poder ler o resultado:
#   alcance 20 m -> cego acima de 22,9 m/s
#   alcance 33 m -> cego acima de 28,0 m/s   (padrão)
#   alcance 80 m -> cego acima de 40,9 m/s
# A velocidade cega cresce com a RAIZ do alcance. Se o desempenho treinado acompanhar essa
# raiz, o gargalo é percepção. Se for plano, a visão já sobrava e o gargalo é outro.
#
# Uso:  scripts/visao_treinada.sh [GENS] [SWARM] [REPS]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
GENS=${1:-40}; SWARM=${2:-60}; REPS=${3:-2}
BIN="$RAIZ/target/release/universe"
SAIDA="$RAIZ/telemetry/visao_treinada"

[ -x "$BIN" ] || { echo "compile primeiro"; exit 1; }
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X"; exit 1; }
rm -rf "$SAIDA"; mkdir -p "$SAIDA"

echo "=== MAIS VISÃO AJUDA? (treinando cada condição do zero) ==="
echo "gens=$GENS · swarm=$SWARM · reps=$REPS · SEM campeão herdado"
echo

n=0
for entry in "perto20:20" "base33:0" "longe80:80"; do
  IFS=: read -r rot alc <<< "$entry"
  for r in $(seq 1 "$REPS"); do
    d="/tmp/vtr_${rot}_r${r}"; rm -rf "$d"; mkdir -p "$d"
    # SEM CHAMPION.dna: cada condição evolui com a SUA visão desde o começo
    [ -f "$RAIZ/universe_config.json" ] && cp "$RAIZ/universe_config.json" "$d/"
    cat > "$d/params.json" <<EOF
{"experiment_name":"vtr_${rot}_r${r}","variant_id":0,"axiom_wall_avoidance":0.18,
"axiom_dark_center":0.45,"axiom_whisker":0.15,"axiom_racing_line":0.26,
"axiom_decay_divisor":50.0,"axiom_floor":0.35,"k_smooth":100.0,"k_speed":0.5,
"k_novel":50.0,"k_stab":200.0,"k_phi":300.0,"k_aesth":200.0,"lap_bonus":50000.0,
"survival_bonus":3000.0,"death_penalty":2000.0,"via_ablada":"","nascimento":"",
"visao_alcance_m":$alc,"visao_fov_deg":0,
"speed_bleed_damage":15.0,"stagnation_secs":15.0,"wall_t_threshold":40}
EOF
    ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
        --lab-params "$d/params.json" > "$SAIDA/${rot}_r${r}.log" 2>&1 ) &
    n=$((n+1)); [ $((n % 3)) -eq 0 ] && wait
  done
done
wait
echo "corridas terminadas ($n)."; echo

python3 - "$SAIDA" <<'PY'
import glob, os, re, statistics as st, sys
d = sys.argv[1]
rxg = re.compile(r"GEN (\d+) \| Laps: 0v=(\d+) 1v=(\d+) 2v=(\d+) 3\+v=(\d+) \| "
                 r"NET max=([\d.]+)m median=([\d.]+)m \| Deaths (\d+)/(\d+)")
rxv = re.compile(r"cega acima de ([\d.]+)")
por = {}
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    rot = os.path.basename(f).rsplit("_r", 1)[0]
    txt = open(f, errors="ignore").read()
    gs = [(int(m.group(1)), float(m.group(6)), float(m.group(7)),
           (int(m.group(3))+int(m.group(4))+int(m.group(5)))/max(int(m.group(9)),1))
          for m in rxg.finditer(txt)]
    if not gs: continue
    vc = rxv.search(txt)
    por.setdefault(rot, []).append((gs, float(vc.group(1)) if vc else 0.0))

if not por: print("sem dados"); raise SystemExit(1)
print(f"{'condição':<10} {'v cega':>8} {'net_max fim':>12} {'net_med fim':>12} "
      f"{'≥1 volta fim':>13} {'reps':>5}")
print("-" * 66)
for rot in ["perto20", "base33", "longe80"]:
    if rot not in por: continue
    v = por[rot]
    q = max(len(v[0][0])//4, 1)
    fim_max = st.mean([st.mean([g[1] for g in gs[-q:]]) for gs, _ in v])
    fim_med = st.mean([st.mean([g[2] for g in gs[-q:]]) for gs, _ in v])
    fim_vol = st.mean([st.mean([g[3] for g in gs[-q:]]) for gs, _ in v])
    print(f"{rot:<10} {v[0][1]:>7.1f} {fim_max:>11.0f}m {fim_med:>11.0f}m "
          f"{fim_vol:>12.0%} {len(v):>5}")
print("-" * 66)
print("\nCOMO LER: 'fim' é a média do último quarto das gerações, não a última geração —")
print("uma geração isolada é sorte. A velocidade cega cresce com a RAIZ do alcance")
print("(20m→22,9 · 33m→28,0 · 80m→40,9 m/s): se o desempenho acompanhar essa curva, o")
print("gargalo é percepção; se for plano, a visão já sobrava.")
PY
