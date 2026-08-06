#!/usr/bin/env bash
# Alocação neural por ABLAÇÃO — quanto cada via do cérebro carrega.
#
# DUAS DECISÕES DE MÉTODO, cada uma de um erro já pago:
#
# 1. REPETIÇÕES. `spawn.rs` usa `rand::rng()`, thread-local e NÃO semeado: duas corridas com
#    a mesma configuração dão resultados diferentes. Comparar uma variante contra outra numa
#    rodada só compara ruído — o número mais bonito da tabela seria a rodada mais sortuda.
#
# 2. VERIFICAÇÃO DE DISPLAY ANTES DE COMEÇAR. Mesmo com `--headless` o Bevy cria uma janela
#    (invisível) e precisa de servidor X **com GPU**. Se a sessão gráfica cair no meio, as
#    corridas seguintes morrem com XOpenDisplayFailed — aconteceu aqui: 22 de 30 caíram e o
#    agregador imprimiu uma tabela com 8 corridas como se estivesse completa.
#    `xvfb-run` NÃO resolve: dá display mas o driver por software não tem compute shader
#    ("The selected version doesn't support Features(BUFFER_STORAGE | COMPUTE_SHADER | ...)").
#    Então: conferir antes, abortar se não houver, e conferir de novo no fim.
#
# 3. DIRETÓRIO ISOLADO por corrida. O simulador LÊ e SOBRESCREVE `CHAMPION.dna` no diretório
#    de trabalho. Rodando na raiz do projeto, um teste de 2 gerações coroa um campeão fraco e
#    apaga o treinado — aconteceu comigo aqui, e só deu para desfazer porque o arquivo estava
#    versionado no git. Cada corrida agora vive em /tmp com a SUA cópia.
#
# O que se mede: a ablação é aplicada a um cérebro TREINADO (o CHAMPION.dna copiado), então é
# comparação pareada dentro do mesmo genoma — a guarda que a bancada Python mostrou ser
# indispensável (estatística do genoma mede deriva junto com função).
#
# Uso:  scripts/alocacao.sh [REPS] [GENS] [SWARM]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ"

REPS=${1:-4}
GENS=${2:-3}
SWARM=${3:-150}
SAIDA="$RAIZ/telemetry/alocacao"
BIN="$RAIZ/target/release/universe"
PARALELO=4          # 12 núcleos, e cada processo do Bevy já usa vários

[ -x "$BIN" ] || { echo "compile primeiro: cargo build --release"; exit 1; }
[ -f "$RAIZ/CHAMPION.dna" ] || { echo "CHAMPION.dna não encontrado na raiz"; exit 1; }

# lista de vias: 4º argumento sobrescreve (ex.: "visao curvatura paredes"). A linha de base
# ("") é sempre incluída — sem ela não há contra o que medir custo.
# ATENÇÃO À ORDEM: o `mkdir` vem DEPOIS desta atribuição. Estava antes, e o diretório
# parametrizado nunca era criado — a bateria inteira falhou com "No such file or directory".
if [ $# -ge 4 ]; then
  read -r -a EXTRA <<< "$4"
  VIAS=("" "${EXTRA[@]}")
  SAIDA="$RAIZ/telemetry/alocacao_$(echo "$4" | tr ' ' '_' | cut -c1-30)"
else
  VIAS=("" "cortex" "reflexo" "lenta" "marcha" "sentinela")
fi
rm -rf "$SAIDA"; mkdir -p "$SAIDA"

ENVOLTORIO=""
if ! xdpyinfo >/dev/null 2>&1; then
  echo "✗ sem servidor X respondendo (DISPLAY=${DISPLAY:-vazio})."
  echo "  O simulador precisa de GPU: xvfb não serve (driver por software não tem compute shader)."
  echo "  Abra a sessão gráfica e rode de novo — melhor abortar que gerar tabela pela metade."
  exit 1
fi
echo "✓ X respondendo em ${DISPLAY}"

echo "=== ALOCAÇÃO NEURAL POR ABLAÇÃO ==="
echo "reps=$REPS · gens=$GENS · swarm=$SWARM · cada corrida em /tmp isolado"
echo "campeão de partida: $(./target/release/sinapses CHAMPION.dna --top 0 2>/dev/null | sed -n 2p)"
echo

n=0
for via in "${VIAS[@]}"; do
  rot=${via:-base}
  for r in $(seq 1 "$REPS"); do
    dir="/tmp/aloc_${rot}_r${r}"
    rm -rf "$dir"; mkdir -p "$dir"
    cp "$RAIZ/CHAMPION.dna" "$dir/"
    [ -f "$RAIZ/universe_config.json" ] && cp "$RAIZ/universe_config.json" "$dir/"
    cat > "$dir/params.json" <<EOF
{"experiment_name":"aloc_${rot}_r${r}","variant_id":0,"axiom_wall_avoidance":0.18,
"axiom_dark_center":0.45,"axiom_whisker":0.15,"axiom_racing_line":0.26,
"axiom_decay_divisor":50.0,"axiom_floor":0.35,"k_smooth":100.0,"k_speed":0.5,
"k_novel":50.0,"k_stab":200.0,"k_phi":300.0,"k_aesth":200.0,"lap_bonus":50000.0,
"survival_bonus":3000.0,"death_penalty":2000.0,"via_ablada":"$via",
"speed_bleed_damage":15.0,"stagnation_secs":15.0,"wall_t_threshold":40}
EOF
    ( cd "$dir" && $ENVOLTORIO "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
        --lab-params "$dir/params.json" > "$SAIDA/${rot}_r${r}.log" 2>&1 ) &
    n=$((n+1))
    [ $((n % PARALELO)) -eq 0 ] && wait
  done
done
wait
if ! xdpyinfo >/dev/null 2>&1; then
  echo "⚠️  O X CAIU DURANTE A BATERIA. As corridas depois da queda não têm dados."
fi
echo "corridas terminadas ($n). agregando..."; echo

python3 - "$SAIDA" <<'PY'
import glob, os, re, statistics as st, sys
d = sys.argv[1]
rx = re.compile(r"ALOC gen=(\d+) ablada=(\S+) uso_cortex=([\d.]+) uso_reflexo=([\d.]+) "
                r"uso_sentinela=([\d.]+) vetos=(\d+) n=(\d+) net_max=([\d.-]+) net_med=([\d.-]+)")
por, mortas = {}, {}
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    rot = os.path.basename(f).rsplit("_r", 1)[0]
    melhor, uso, vetos = 0.0, None, 0
    for ln in open(f, errors="ignore"):
        m = rx.search(ln)
        if m:
            melhor = max(melhor, float(m.group(8)))
            uso = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
            vetos += int(m.group(6))
    if uso:
        por.setdefault(rot, []).append((melhor, uso, vetos))
    else:
        mortas.setdefault(rot, []).append(os.path.basename(f))

# NUNCA imprimir tabela sobre dados parciais sem dizer. A versão anterior descartava as
# corridas mortas em silêncio e mostrava "reps 5 / reps 3" como se estivesse tudo bem — uma
# tabela com 27% dos dados que parecia resultado.
n_ok = sum(len(v) for v in por.values())
n_mortas = sum(len(v) for v in mortas.values())
if n_mortas:
    print(f"⚠️  {n_mortas} de {n_ok+n_mortas} corridas NÃO produziram dados:")
    for rot, fs in sorted(mortas.items()):
        motivo = ""
        for cand in fs[:1]:
            txt = open(os.path.join(d, cand), errors="ignore").read()
            if "XOpenDisplayFailed" in txt: motivo = "sem display X"
            elif "panicked" in txt: motivo = "panic"
            elif not txt.strip(): motivo = "log vazio"
        print(f"     {rot}: {len(fs)} corrida(s) — {motivo or 'motivo desconhecido'}")
    print()

if "base" not in por:
    print("sem linha de base — nada a comparar"); raise SystemExit(1)
FALTANDO = []
if FALTANDO:
    print(f"⚠️  CONDIÇÕES SEM NENHUM DADO: {', '.join(FALTANDO)}")
    print("     a tabela abaixo está INCOMPLETA — não vale como resultado.\n")
b = [x[0] for x in por["base"]]
mb, epb = st.mean(b), (st.stdev(b)/len(b)**0.5 if len(b) > 1 else 0.0)

print(f"{'via ablada':<12} {'net_max':>9} {'±ep':>7} {'reps':>5} {'CUSTO':>10} "
      f"{'|':>2} {'uso do portão cx/rf/sn':>24} {'vetos':>8}")
print("-" * 88)
for rot in ["base"] + [r for r in sorted(por) if r != "base"]:
    if rot not in por: continue
    v = [x[0] for x in por[rot]]
    m = st.mean(v); ep = st.stdev(v)/len(v)**0.5 if len(v) > 1 else 0.0
    custo = (mb - m)/mb*100 if mb else 0.0
    # erro padrão do custo, propagado: as duas médias são independentes
    ec = (100/mb)*((ep**2 + (m/mb*epb)**2)**0.5) if mb else 0.0
    u = por[rot][0][1]; vt = round(st.mean([x[2] for x in por[rot]]))
    cel = "—" if rot == "base" else f"{custo:>6.1f}±{ec:<3.1f}%"
    print(f"{rot:<12} {m:>9.0f} {ep:>7.0f} {len(v):>5} {cel:>10} {'|':>2} "
          f"{u[0]:>7.3f} {u[1]:>7.3f} {u[2]:>7.3f}  {vt:>8}")
print("-" * 88)
print("\nCOMO LER")
print("  'uso'   = peso médio que o portão dá à via — o quanto ela é USADA")
print("  'CUSTO' = quanto se perde ao desligá-la    — o quanto ela SERVE")
print("  A DISTÂNCIA entre os dois é o resultado. Uso alto com custo zero = uma via que")
print("  fala e ninguém escuta — exatamente o padrão dos genes moduladores da bancada")
print("  Python (deriva até b=-1.67 com ablação de +0.0%).")
PY
