#!/usr/bin/env bash
# A evolução voltou a funcionar? — validação do currículo por população.
#
# O QUE ESTAVA QUEBRADO (medido em 14 corridas antes do conserto):
#
#   gen | ≥1 volta | mortalidade
#     1 |    34%   |    81%
#     2 |    10%   |    93%
#     3 |     1%   |   100%
#
# Causa: `level_for_champion` olhava só a distância do MELHOR carro. Um que fez 1,93 voltas
# no seco levava os 60 para o nível 3, onde a aderência cai a 0,33, a frenagem exigida vai a
# 130,9 m e o sensor mais longo do carro alcança 120 m — cegueira física, não dificuldade.
#
# O CONSERTO: `decide_next_env_level_pop` avança pela FRAÇÃO DA POPULAÇÃO que fecha volta
# (≥30% sobe, <5% recua), com histerese e no máximo um degrau por geração.
#
# O QUE ESTE SCRIPT MEDE: a mesma trajetória, agora com mais gerações. Se o conserto pegou,
# a mortalidade para de subir monotonicamente e a fração de voltas se sustenta.
#
# Uso:  scripts/valida_evolucao.sh [REPS] [GENS] [SWARM]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
REPS=${1:-4}; GENS=${2:-10}; SWARM=${3:-60}
BIN="$RAIZ/target/release/universe"
SAIDA="$RAIZ/telemetry/valida_evo"

[ -x "$BIN" ] || { echo "compile primeiro"; exit 1; }
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X (o simulador precisa de GPU; xvfb não serve)"; exit 1; }
rm -rf "$SAIDA"; mkdir -p "$SAIDA"

echo "=== A EVOLUÇÃO VOLTOU A FUNCIONAR? ==="
echo "reps=$REPS · gens=$GENS · swarm=$SWARM"
echo "antes do conserto: ≥1 volta caía 34% -> 10% -> 1%, mortalidade 81% -> 93% -> 100%"
echo

for r in $(seq 1 "$REPS"); do
  d="/tmp/vevo_r${r}"; rm -rf "$d"; mkdir -p "$d"
  cp "$RAIZ/CHAMPION.dna" "$d/"
  [ -f "$RAIZ/universe_config.json" ] && cp "$RAIZ/universe_config.json" "$d/"
  ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
      > "$SAIDA/r${r}.log" 2>&1 ) &
  [ $((r % 4)) -eq 0 ] && wait
done
wait
xdpyinfo >/dev/null 2>&1 || echo "⚠️  O X CAIU DURANTE A BATERIA."
echo "corridas terminadas."; echo

python3 - "$SAIDA" <<'PY'
import glob, os, re, statistics as st, sys
d = sys.argv[1]
rxg = re.compile(r"GEN (\d+) \| Laps: 0v=(\d+) 1v=(\d+) 2v=(\d+) 3\+v=(\d+) \| "
                 r"NET max=([\d.]+)m median=([\d.]+)m \| Deaths (\d+)/(\d+)")
rxw = re.compile(r"MAP DIAG track_width=([\d.]+)m")
tab, mortos = {}, 0
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    txt = open(f, errors="ignore").read()
    if "GEN " not in txt:
        mortos += 1; continue
    larg = None
    for ln in txt.splitlines():
        m = rxw.search(ln)
        if m: larg = float(m.group(1))
        m = rxg.search(ln)
        if m:
            g = int(m.group(1)); tot = int(m.group(9))
            volta = (int(m.group(3)) + int(m.group(4)) + int(m.group(5))) / max(tot, 1)
            tab.setdefault(g, []).append((larg, volta, int(m.group(8))/max(tot,1),
                                          float(m.group(6)), float(m.group(7))))
if mortos:
    print(f"⚠️  {mortos} corrida(s) sem dados — tabela incompleta.\n")
if not tab:
    print("nenhum dado"); raise SystemExit(1)

print(f"{'gen':>4} {'largura':>8} {'≥1 volta':>10} {'mortalidade':>12} "
      f"{'net_max':>9} {'net_med':>9} {'n':>3}")
print("-" * 62)
for g in sorted(tab):
    v = tab[g]; n = len(v)
    lg = [x[0] for x in v if x[0]]
    print(f"{g:>4} {(sum(lg)/len(lg) if lg else 0):>7.1f}m {sum(x[1] for x in v)/n:>9.0%} "
          f"{sum(x[2] for x in v)/n:>11.0%} {sum(x[3] for x in v)/n:>8.0f}m "
          f"{sum(x[4] for x in v)/n:>8.0f}m {n:>3}")
print("-" * 62)

gens = sorted(tab)
if len(gens) >= 3:
    volta = [sum(x[1] for x in tab[g])/len(tab[g]) for g in gens]
    mort  = [sum(x[2] for x in tab[g])/len(tab[g]) for g in gens]
    # o sintoma antigo: mortalidade sobe TODA geração até 100%
    monotona = all(mort[i] <= mort[i+1] + 1e-9 for i in range(len(mort)-1))
    chegou_100 = mort[-1] >= 0.995
    ultimas = volta[len(volta)//2:]
    sustenta = sum(ultimas)/len(ultimas) > 0.05
    print(f"\n  mortalidade monotônica crescente? {'SIM (sintoma antigo)' if monotona else 'não'}")
    print(f"  chegou a 100% de mortalidade?      {'SIM' if chegou_100 else 'não'}")
    print(f"  ≥1 volta na 2ª metade das gens:    {sum(ultimas)/len(ultimas):.1%}")
    print("\n  " + ("CONSERTO NÃO PEGOU — o colapso persiste"
                    if (monotona and chegou_100) or not sustenta else
                    "CONSERTO PEGOU — o colapso parou"))
PY
