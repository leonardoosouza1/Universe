#!/usr/bin/env bash
# TREINO LONGO — usar a evolução consertada para o que interessa: dar voltas.
#
# Até aqui só se mediu. A evolução foi consertada (currículo por população, com histerese e
# caminho de volta) e VALIDADA em 10 gerações: o colapso parou e a mediana da população foi
# de 694 m para 1346 m. Mas 10 gerações é validação, não treino.
#
# Estado a bater, medido:
#   ≥1 volta ......... 30-47% da população
#   ≥2 voltas ........ ~0%
#   ≥3 voltas ........ NUNCA aconteceu em corrida nenhuma
#   mortalidade ...... 60-90%
#
# O treino roda MUITAS gerações a partir do campeão atual e salva a linhagem
# (`linhagem/genNNNNN.dna`), para depois se poder perguntar com `sinapses --contra` o que
# nasceu, morreu e engrossou entre dois pontos do tempo.
#
# Duas repetições, para não confundir sorte com evolução — `spawn.rs` usa `rand::rng()`,
# não semeado.
#
# Uso:  scripts/treinar.sh [GENS] [SWARM] [REPS]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
GENS=${1:-120}; SWARM=${2:-80}; REPS=${3:-2}
BIN="$RAIZ/target/release/universe"
SAIDA="$RAIZ/telemetry/treino"

[ -x "$BIN" ] || { echo "compile primeiro"; exit 1; }
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X"; exit 1; }
mkdir -p "$SAIDA"

echo "=== TREINO LONGO ==="
echo "gens=$GENS · swarm=$SWARM · reps=$REPS"
echo "partida: $(./target/release/dump_dna CHAMPION.dna | sed -n 2p)"
echo "alvo: voltas completas e consistentes — hoje ≥2 voltas é ~0% e ≥3 nunca aconteceu"
echo

for r in $(seq 1 "$REPS"); do
  d="/tmp/treino_r${r}"; rm -rf "$d"; mkdir -p "$d"
  cp "$RAIZ/CHAMPION.dna" "$d/"
  [ -f "$RAIZ/universe_config.json" ] && cp "$RAIZ/universe_config.json" "$d/"
  ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
      > "$SAIDA/r${r}.log" 2>&1 ) &
done
wait
xdpyinfo >/dev/null 2>&1 || echo "⚠️  O X CAIU DURANTE O TREINO."
echo "treino terminado."; echo

python3 - "$SAIDA" <<'PY'
import glob, os, re, statistics as st, sys
d = sys.argv[1]
rxg = re.compile(r"GEN (\d+) \| Laps: 0v=(\d+) 1v=(\d+) 2v=(\d+) 3\+v=(\d+) \| "
                 r"NET max=([\d.]+)m median=([\d.]+)m \| Deaths (\d+)/(\d+)")
tab = {}
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    for m in rxg.finditer(open(f, errors="ignore").read()):
        g, tot = int(m.group(1)), int(m.group(9))
        tab.setdefault(g, []).append((
            (int(m.group(3))+int(m.group(4))+int(m.group(5)))/max(tot,1),  # ≥1 volta
            (int(m.group(4))+int(m.group(5)))/max(tot,1),                  # ≥2 voltas
            int(m.group(5))/max(tot,1),                                    # ≥3 voltas
            int(m.group(8))/max(tot,1),                                    # mortalidade
            float(m.group(6)), float(m.group(7))))                         # net max, med
if not tab:
    print("sem dados"); raise SystemExit(1)

gens = sorted(tab)
# resumo por blocos de 10 gerações — a trajetória importa mais que qualquer geração isolada
print(f"{'gens':>9} {'≥1 volta':>10} {'≥2':>7} {'≥3':>7} {'mortal.':>9} "
      f"{'net_max':>9} {'net_med':>9}")
print("-" * 68)
bloco = 10
for ini in range(0, len(gens), bloco):
    ch = gens[ini:ini+bloco]
    v = [x for g in ch for x in tab[g]]
    print(f"{ch[0]:>4}-{ch[-1]:<4} {st.mean([x[0] for x in v]):>9.0%} "
          f"{st.mean([x[1] for x in v]):>6.1%} {st.mean([x[2] for x in v]):>6.1%} "
          f"{st.mean([x[3] for x in v]):>8.0%} {st.mean([x[4] for x in v]):>8.0f}m "
          f"{st.mean([x[5] for x in v]):>8.0f}m")
print("-" * 68)

# EVOLUIU? compara o primeiro quarto com o último, que é a pergunta do Leonardo
q = max(len(gens)//4, 1)
ini = [x for g in gens[:q] for x in tab[g]]
fim = [x for g in gens[-q:] for x in tab[g]]
def med(v, i): return st.mean([x[i] for x in v])
print(f"\n{'':22} {'primeiro quarto':>16} {'último quarto':>15} {'Δ':>10}")
for i, nome in [(0,"≥1 volta"), (1,"≥2 voltas"), (2,"≥3 voltas"),
                (3,"mortalidade"), (5,"net_med")]:
    a, b = med(ini,i), med(fim,i)
    fmt = "{:>15.0f}m" if i == 5 else "{:>15.1%}"
    delta = f"{(b-a)*100:>+9.1f}pp" if i != 5 else f"{b-a:>+9.0f}m"
    print(f"  {nome:<20} " + fmt.format(a) + " " + fmt.format(b) + delta)

melhor2 = max(st.mean([x[1] for x in tab[g]]) for g in gens)
melhor3 = max(st.mean([x[2] for x in tab[g]]) for g in gens)
print(f"\n  melhor geração em ≥2 voltas: {melhor2:.1%}   ·   em ≥3 voltas: {melhor3:.1%}")
print("\n  " + ("EVOLUIU — a segunda metade é melhor que a primeira"
                if med(fim,0) > med(ini,0) + 0.03 else
                "NÃO evoluiu de forma clara neste orçamento"))
PY
