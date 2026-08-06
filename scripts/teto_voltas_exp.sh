#!/usr/bin/env bash
# ≥2 VOLTAS: o cérebro não consegue, ou o RELÓGIO não deixa?
#
# O QUE MOTIVOU (o número que eu quase deixei passar):
#
#   `≥2 voltas = 0,0%` em TODA geração, das DUAS condições do treino de fração. Zero cravado
#   em duas condições independentes não parece dificuldade, parece parede. E era.
#
#   `scripts/teto_de_voltas.py` mediu a pista pelo perfil de velocidade (teto de curva +
#   passada de frenagem + passada de aceleração, circuito fechado):
#
#     pista 1658 m · volta mais rápida FISICAMENTE possível 50,5 s · orçamento 120 s
#     -> teto de 2,38 voltas, mas com a folga realista de 35% sobre o teto: 1,76 voltas
#
#   E o melhor carro já faz `net_max = 3079 m` = **1,86 voltas**. Eles estão a 93% de duas
#   voltas. O 0,0% não é fracasso do cérebro — é uma métrica de DEGRAU (contagem inteira)
#   avaliada exatamente em cima da fronteira física.
#
#   É o mesmo erro de método que já me pegou duas vezes nesta sessão, na terceira roupa:
#   `max` escondia a população, janelas diferentes fabricaram um 4,3σ, e agora uma contagem
#   inteira esconde 1,86 voltas atrás de um zero.
#
# O EXPERIMENTO: mesma semente, mesma fração, só o orçamento muda.
#
#   120 s (atual)  1,76 voltas de folga realista -> ≥2 voltas deve ficar ~0
#   180 s          2,64 voltas de folga realista -> se o cérebro já sabe, ≥2 voltas APARECE
#
#   PREDIÇÃO REGISTRADA ANTES DE RODAR: se o gargalo é o relógio, 180 s destrava ≥2 voltas
#   sem nenhuma mudança no cérebro. Se ≥2 voltas continuar ~0 com 180 s, o gargalo é o
#   cérebro e aí sim vale evoluir contra ele.
#
#   O controle importa tanto quanto o tratamento: sem ele eu ia creditar à evolução um ganho
#   que é só do relógio.
#
# Uso:  scripts/teto_voltas_exp.sh [GENS] [SWARM]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
GENS=${1:-15}; SWARM=${2:-60}
BIN="$RAIZ/target/release/universe"; SAIDA="$RAIZ/telemetry/teto_voltas"
[ -x "$BIN" ] || { echo "compile primeiro"; exit 1; }
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X"; exit 1; }
rm -rf "$SAIDA"; mkdir -p "$SAIDA"

echo "=== ≥2 VOLTAS: relógio ou cérebro? ==="
echo "gens=$GENS · swarm=$SWARM · pista 1658 m · volta mais rápida possível 50,5 s"
echo "predição: se for o relógio, 180 s destrava ≥2 voltas SEM mudar o cérebro"
echo

for T in 120 180; do
  d="/tmp/tv_${T}"; rm -rf "$d"; mkdir -p "$d"
  cp "$RAIZ/CHAMPION.dna" "$d/"
  # o orçamento vive no config, não nos lab-params — por isso o config é reescrito por condição
  python3 - "$RAIZ/universe_config.json" "$d/universe_config.json" "$T" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["generation_timeout_s"] = float(sys.argv[3])
json.dump(c, open(sys.argv[2], "w"), indent=2)
PY
  cat > "$d/params.json" <<EOF
{"experiment_name":"tv_${T}","variant_id":0,"axiom_wall_avoidance":0.18,
"axiom_dark_center":0.45,"axiom_whisker":0.15,"axiom_racing_line":0.26,
"axiom_decay_divisor":50.0,"axiom_floor":0.35,"k_smooth":100.0,"k_speed":0.5,
"k_novel":50.0,"k_stab":200.0,"k_phi":300.0,"k_aesth":200.0,"lap_bonus":50000.0,
"survival_bonus":3000.0,"death_penalty":2000.0,"via_ablada":"","nascimento":"",
"fracao_exploradora":0.50,
"speed_bleed_damage":15.0,"stagnation_secs":15.0,"wall_t_threshold":40}
EOF
  ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
      --lab-params "$d/params.json" > "$SAIDA/t${T}.log" 2>&1 ) &
done
wait
echo "terminado."; echo

python3 - "$SAIDA" <<'PY'
import glob, os, re, statistics as st, sys
d = sys.argv[1]
PISTA = 1658.0
rx = re.compile(r"GEN (\d+) \| Laps: 0v=(\d+) 1v=(\d+) 2v=(\d+) 3\+v=(\d+) \| "
                r"NET max=([\d.]+)m median=([\d.]+)m \| Deaths (\d+)/(\d+)")
cond = {}
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    t = os.path.basename(f)[1:-4]
    for m in rx.finditer(open(f, errors="ignore").read()):
        tot = max(int(m.group(9)), 1)
        cond.setdefault(t, {})[int(m.group(1))] = (
            (int(m.group(3))+int(m.group(4))+int(m.group(5)))/tot,
            (int(m.group(4))+int(m.group(5)))/tot,
            float(m.group(6)), float(m.group(7)), int(m.group(8))/tot)
if len(cond) < 2:
    print("faltou condição:", list(cond)); raise SystemExit(1)

print(f"{'orçamento':>10} {'≥1 volta':>10} {'≥2 voltas':>11} {'net_max':>9} "
      f"{'= voltas':>9} {'MEDIANA':>9} {'mortal.':>8}")
print("-" * 72)
for t in sorted(cond, key=int):
    v = list(cond[t].values())
    nm = st.mean([x[2] for x in v])
    print(f"{t+' s':>10} {st.mean([x[0] for x in v]):>9.0%} {st.mean([x[1] for x in v]):>10.1%} "
          f"{nm:>8.0f}m {nm/PISTA:>8.2f}v {st.mean([x[3] for x in v]):>8.0f}m "
          f"{st.mean([x[4] for x in v]):>7.0%}")
print("-" * 72)

comuns = sorted(set(cond["120"]) & set(cond["180"]))
a = [cond["180"][g][1] for g in comuns]; b = [cond["120"][g][1] for g in comuns]
dif = [x - y for x, y in zip(a, b)]
md = st.mean(dif); ed = st.stdev(dif)/len(dif)**0.5 if len(dif) > 1 else 0.0
sig = abs(md)/ed if ed > 0 else 0.0
print(f"\n≥2 voltas, PAREADO em {len(comuns)} gerações: {md*100:+.1f}pp  ({sig:.1f}σ)")
print(f"  pico por geração: 120 s -> {max(cond['120'][g][1] for g in comuns):.1%} · "
      f"180 s -> {max(cond['180'][g][1] for g in comuns):.1%}")
print()
if md > 0.02 and sig > 2:
    print("  ERA O RELÓGIO. O cérebro JÁ sabia dar 2 voltas; a métrica de degrau escondia.")
    print("  Consequência: todo ≥2 voltas medido a 120 s estava medindo o orçamento.")
elif max(cond['180'][g][1] for g in comuns) == 0.0:
    print("  ERA O CÉREBRO. Nem com 50% a mais de relógio aparece uma segunda volta —")
    print("  a parede não é o orçamento. Aqui vale evoluir contra.")
else:
    print("  INCONCLUSIVO neste orçamento de gerações: aparece, mas não separa do ruído.")
PY
