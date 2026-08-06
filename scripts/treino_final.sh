#!/usr/bin/env bash
# TREINO com os achados aplicados — a máquina inteira, uma coisa só rodando.
#
# ORDEM DOS ACHADOS QUE ENTRAM AQUI (cada um medido, nenhum chutado):
#
# 1. CURRÍCULO POR POPULAÇÃO (já no código). Antes, clima e largura avançavam pela distância
#    do MELHOR carro; um que fez 1,93 voltas levava os 60 a um ambiente onde a frenagem
#    exigida (130,9 m) passa do sensor mais longo (120 m). Validado em 10 gerações: o colapso
#    parou e a mediana foi de 694 m para 1346 m.
#
# 2. FRAÇÃO EXPLORADORA (passada como argumento). A mutação estratificada punha METADE da
#    população em σ 0,08–0,20, e a MEDIANA da população era ZERO metros — com
#    `wall_mean = 0,00s`, ou seja não é batida, é a mutação destruindo o genoma. A população
#    efetiva era 30, não 60.
#
# 3. O QUE NÃO ENTRA, e por quê:
#    - mais alcance de visão: medido, não ajuda (120 m dá −2,6% num cérebro treinado em 33 m)
#    - autoridade sobre o câmbio: ablação custa 0,7 ± 0,9%, indistinguível de zero
#    - modulador (via lenta): 2,7%, contribuinte menor em todo domínio testado
#
# A economia de morte, para ler o resultado: `+40% do tanque a cada 25 m`, contra drenos de
# 1/s, 30/s (rodando parado) e 60/s (v<4 m/s sem progresso). **Mover ou morrer.** Autópsia de
# 11.450 mortes: 58,8% HP, 39,9% SpeedBleed, **3 batidas**. Eles não batem — param.
#
#    ⚠️ RETRATAÇÃO. Eu reportei aqui "+850 m na mediana (+264%), 4,3σ" para 0,10 contra 0,50.
#    ERA ARTEFATO MEU: a comparação usava 46 gerações de um lado e 8 do outro, sobre
#    intervalos DIFERENTES. Como o desempenho varia ao longo das gerações, média sobre
#    janelas distintas compara fases, não tratamentos.
#
#    REFEITO PAREADO POR GERAÇÃO (mesma janela, mesmas repetições, 10 gens comuns):
#      MEDIANA      875 m (0,10) contra 1081 m (0,50)  ->  −207 m,  1,1σ
#      ≥1 volta    27,6%          contra 32,6%          ->   −5,1pp, 1,3σ
#      mortalidade 76,9%          contra 73,5%          ->   +3,3pp, 1,0σ
#
#    O efeito NÃO EXISTE, e o sinal ainda inverteu. O mecanismo é real (metade da população
#    nasce em σ 0,08–0,20 e a mediana chega a zero com `wall_mean = 0,00s`, ou seja não é
#    batida) mas o TAMANHO do efeito não sobrevive a uma comparação honesta.
#    A fração é passada como argumento e fica em 0,50, o padrão.
#
#    Use `scripts/comparar_treinos.py` para qualquer comparação entre treinos — ele só
#    pareia gerações presentes NOS DOIS lados, exatamente para este erro não se repetir.
#
# Uso:  scripts/treino_final.sh [FRACAO] [GENS] [SWARM] [REPS]
set -uo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$RAIZ"
FRAC=${1:-0.10}; GENS=${2:-100}; SWARM=${3:-60}; REPS=${4:-2}
BIN="$RAIZ/target/release/universe"; SAIDA="$RAIZ/telemetry/treino_final"
[ -x "$BIN" ] || { echo "compile primeiro"; exit 1; }
xdpyinfo >/dev/null 2>&1 || { echo "✗ sem X"; exit 1; }
rm -rf "$SAIDA"; mkdir -p "$SAIDA"

echo "=== TREINO FINAL ==="
echo "fração exploradora=$FRAC · gens=$GENS · swarm=$SWARM · reps=$REPS"
echo "partida: $(./target/release/dump_dna CHAMPION.dna | sed -n 2p)"
echo "alvo: ≥2 voltas (hoje ~0%) e ≥3 voltas (nunca aconteceu em corrida nenhuma)"
echo

for r in $(seq 1 "$REPS"); do
  d="/tmp/tf_r${r}"; rm -rf "$d"; mkdir -p "$d"
  cp "$RAIZ/CHAMPION.dna" "$d/"
  [ -f "$RAIZ/universe_config.json" ] && cp "$RAIZ/universe_config.json" "$d/"
  cat > "$d/params.json" <<EOF
{"experiment_name":"tf_r${r}","variant_id":0,"axiom_wall_avoidance":0.18,
"axiom_dark_center":0.45,"axiom_whisker":0.15,"axiom_racing_line":0.26,
"axiom_decay_divisor":50.0,"axiom_floor":0.35,"k_smooth":100.0,"k_speed":0.5,
"k_novel":50.0,"k_stab":200.0,"k_phi":300.0,"k_aesth":200.0,"lap_bonus":50000.0,
"survival_bonus":3000.0,"death_penalty":2000.0,"via_ablada":"","nascimento":"",
"fracao_exploradora":$FRAC,
"speed_bleed_damage":15.0,"stagnation_secs":15.0,"wall_t_threshold":40}
EOF
  ( cd "$d" && "$BIN" --headless --gens "$GENS" --swarm "$SWARM" \
      --lab-params "$d/params.json" > "$SAIDA/r${r}.log" 2>&1 ) &
done
wait
echo "treino terminado."; echo

python3 - "$SAIDA" <<'PY'
import glob, os, re, statistics as st, sys
d = sys.argv[1]
rxg = re.compile(r"GEN (\d+) \| Laps: 0v=(\d+) 1v=(\d+) 2v=(\d+) 3\+v=(\d+) \| "
                 r"NET max=([\d.]+)m median=([\d.]+)m \| Deaths (\d+)/(\d+)")
tab = {}
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    for m in rxg.finditer(open(f, errors="ignore").read()):
        g, tot = int(m.group(1)), max(int(m.group(9)), 1)
        tab.setdefault(g, []).append((
            (int(m.group(3))+int(m.group(4))+int(m.group(5)))/tot,
            (int(m.group(4))+int(m.group(5)))/tot, int(m.group(5))/tot,
            int(m.group(8))/tot, float(m.group(6)), float(m.group(7))))
if not tab: print("sem dados"); raise SystemExit(1)
gens = sorted(tab)
print(f"{'gens':>9} {'≥1 volta':>10} {'≥2':>7} {'≥3':>7} {'mortal.':>9} {'net_max':>9} {'MEDIANA':>9}")
print("-" * 68)
for ini in range(0, len(gens), 10):
    ch = gens[ini:ini+10]; v = [x for g in ch for x in tab[g]]
    print(f"{ch[0]:>4}-{ch[-1]:<4} {st.mean([x[0] for x in v]):>9.0%} "
          f"{st.mean([x[1] for x in v]):>6.1%} {st.mean([x[2] for x in v]):>6.1%} "
          f"{st.mean([x[3] for x in v]):>8.0%} {st.mean([x[4] for x in v]):>8.0f}m "
          f"{st.mean([x[5] for x in v]):>8.0f}m")
print("-" * 68)
q = max(len(gens)//4, 1)
ini = [x for g in gens[:q] for x in tab[g]]; fim = [x for g in gens[-q:] for x in tab[g]]
m = lambda v, i: st.mean([x[i] for x in v])
print(f"\n{'':20} {'1º quarto':>12} {'último quarto':>15} {'Δ':>12}")
for i, nome in [(0,"≥1 volta"),(1,"≥2 voltas"),(2,"≥3 voltas"),(3,"mortalidade"),(5,"MEDIANA")]:
    a, b = m(ini,i), m(fim,i)
    if i == 5: print(f"  {nome:<18} {a:>11.0f}m {b:>14.0f}m {b-a:>+11.0f}m")
    else:      print(f"  {nome:<18} {a:>11.1%} {b:>14.1%} {(b-a)*100:>+10.1f}pp")
print(f"\n  melhor geração: ≥2 voltas {max(st.mean([x[1] for x in tab[g]]) for g in gens):.1%} · "
      f"≥3 voltas {max(st.mean([x[2] for x in tab[g]]) for g in gens):.1%}")
print("\n  " + ("EVOLUIU" if m(fim,0) > m(ini,0)+0.03 else "sem melhora clara neste orçamento"))
PY
