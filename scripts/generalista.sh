#!/usr/bin/env bash
# O campeão é bom na MAIORIA das pistas, ou só na que treinou?
#
# O campeão atual foi treinado na semente 42, largura 14 m. A tabela de `./target/release/pistas`
# mostra que as pistas existentes variam muito em demanda de antecipação (de 0 a 35 m de
# frenagem exigida), então "ser bom" não é uma coisa só.
#
# Este script roda o MESMO campeão em cada pista e reporta a distância junto com a demanda
# medida daquela pista. Duas leituras saem daí:
#
#   - se ele vai bem nas fáceis e mal nas difíceis, é generalização que falta
#   - se ele vai mal até nas fáceis, é transferência de traçado: aprendeu a pista, não a pilotar
#
# A segunda é a hipótese que o número sozinho não distingue, e é por isso que a demanda
# medida entra na mesma tabela.
#
# Uso:  scripts/generalista.sh [REPS] [GENS] [SWARM]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
REPS=${1:-2}; GENS=${2:-2}; SWARM=${3:-60}
BIN="$RAIZ/target/release/universe"
SAIDA="$RAIZ/telemetry/generalista"

[ -x "$BIN" ] || { echo "compile primeiro: cargo build --release"; exit 1; }
if ! xdpyinfo >/dev/null 2>&1; then
  echo "✗ sem servidor X (o simulador precisa de GPU; xvfb não serve — sem compute shader)."
  exit 1
fi
rm -rf "$SAIDA"; mkdir -p "$SAIDA"

# semente:nome:largura — as pistas de traçado escrito à mão, mais as procedurais
PISTAS=(
  "123:dubai_oval:15"
  "64:interlagos:13"
  "768:nurburgring:12"
  "512:silverstone:12"
  "256:monaco:11"
  "1024:suzuka:10"
  "9001:proc_sigma:14"
  "42:treino_do_campeao:14"
)

echo "=== O CAMPEÃO GENERALIZA? ==="
echo "campeão: $(./target/release/dump_dna CHAMPION.dna | sed -n 2p)"
echo "reps=$REPS · gens=$GENS · swarm=$SWARM"; echo

n=0
for entry in "${PISTAS[@]}"; do
  IFS=: read -r seed nome larg <<< "$entry"
  for r in $(seq 1 "$REPS"); do
    d="/tmp/gen_${nome}_r${r}"; rm -rf "$d"; mkdir -p "$d"
    cp "$RAIZ/CHAMPION.dna" "$d/"
    [ -f "$RAIZ/universe_config.json" ] && cp "$RAIZ/universe_config.json" "$d/"
    ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
        --seed "$seed" --width "$larg" > "$SAIDA/${nome}_r${r}.log" 2>&1 ) &
    n=$((n+1)); [ $((n % 4)) -eq 0 ] && wait
  done
done
wait
xdpyinfo >/dev/null 2>&1 || echo "⚠️  O X CAIU DURANTE A BATERIA — corridas do fim sem dados."
echo "corridas terminadas ($n)."; echo

python3 - "$SAIDA" <<'PY'
import glob, os, re, statistics as st, sys
d = sys.argv[1]
rx = re.compile(r"net_max=([\d.-]+) net_med=([\d.-]+)")
mortes = re.compile(r"Deaths (\d+)/(\d+)")
por, mortas = {}, []
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    nome = os.path.basename(f).rsplit("_r", 1)[0]
    txt = open(f, errors="ignore").read()
    vals = [float(m.group(1)) for m in rx.finditer(txt)]
    if not vals:
        mortas.append(os.path.basename(f)); continue
    mm = mortes.findall(txt)
    frac = (int(mm[-1][0]) / int(mm[-1][1])) if mm else float("nan")
    por.setdefault(nome, []).append((max(vals), frac))

if mortas:
    print(f"⚠️  {len(mortas)} corrida(s) sem dados: {', '.join(mortas[:6])}")
    print("     a tabela abaixo está incompleta.\n")
if not por:
    print("nenhuma corrida produziu dados"); raise SystemExit(1)

print(f"{'pista':<20} {'net_max':>9} {'±ep':>7} {'mortes':>8} {'reps':>5}")
print("-" * 54)
linhas = []
for nome, v in por.items():
    dist = [x[0] for x in v]
    m = st.mean(dist); ep = st.stdev(dist)/len(dist)**0.5 if len(dist) > 1 else 0.0
    mf = st.mean([x[1] for x in v if x[1] == x[1]]) if any(x[1] == x[1] for x in v) else float("nan")
    linhas.append((m, f"{nome:<20} {m:>9.0f} {ep:>7.0f} {mf:>7.0%} {len(v):>5}"))
for _, l in sorted(linhas, reverse=True):
    print(l)
print("-" * 54)

vals = [m for m, _ in linhas]
if len(vals) > 1:
    razao = max(vals) / max(min(vals), 1.0)
    print(f"\nmelhor/pior = {razao:.1f}×")
    print("Cruzar com `./target/release/pistas` para saber se a queda acompanha a DEMANDA")
    print("medida da pista (falta generalizar) ou é independente dela (decorou o traçado).")
PY
