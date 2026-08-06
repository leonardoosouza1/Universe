#!/usr/bin/env python3
"""O carro não consegue, ou o CÉREBRO não consegue? — separando limite físico de limite de recurso.

Leonardo: *"dependendo do tamanho da pista eles não conseguem mesmo, acho que é algo que
limita ou ele não tem recursos suficientes"*. São duas hipóteses diferentes e elas são
distinguíveis por conta, não por impressão:

    LIMITE FÍSICO    a curva é impossível na aderência disponível. Cérebro nenhum resolve, e
                     medir alocação neural ali é medir ruído dividido por ruído.
    LIMITE DE RECURSO o carro poderia, mas o cérebro (arquitetura FIXA: 96 neurônios na L1,
                     48 na L2, sem genes de arquitetura) não dá conta. Aqui alocação importa.

O simulador já traz a física, no governador de curvatura (`motor.rs:1753`):

    v_safe = sqrt(μ · g · r)          μ = 1.0,  g = 9.81

e o log de cada corrida imprime a geometria por waypoint:

    WP[79] (290,456) seg_len=15.4m turn=0.9° straight

de onde sai o raio local pela aproximação de arco  r = seg_len / turn_rad.

A LARGURA entra porque a linha de corrida pode ser mais aberta que o eixo da pista: numa
curva de raio R com largura W, o raio da linha chega a ~R + W/2 (entrada por fora, ápice por
dentro, saída por fora). Então **a largura levanta o teto físico** — e é por isso que pista
estreita pode ser impossível enquanto a mesma curva, larga, é trivial.

Uso:  scripts/teto_fisico.py telemetry/faixa/w8.log [largura_m]
"""
from __future__ import annotations

import math
import re
import sys

G = 9.81
MU = 1.0

WP = re.compile(r"WP\[(\d+)\]\s+\([-\d.,]+\)\s+seg_len=([\d.]+)m\s+turn=([-\d.]+)°")


def curvas(caminho):
    """Devolve [(indice, comprimento, ângulo em graus, raio)] só dos trechos que curvam."""
    out = []
    for ln in open(caminho, errors="ignore"):
        m = WP.search(ln)
        if not m:
            continue
        i, seg, ang = int(m.group(1)), float(m.group(2)), abs(float(m.group(3)))
        if ang < 1.0:                     # reta: raio infinito, não restringe
            continue
        r = seg / math.radians(ang)
        out.append((i, seg, ang, r))
    # o log reimprime a pista a cada geração — sem dedupe a tabela mostra a mesma curva 3×
    vistos, unicos = set(), []
    for t in out:
        if t[0] in vistos:
            continue
        vistos.add(t[0]); unicos.append(t)
    return unicos


def velocidade_alcancada(caminho):
    """Maior velocidade vista no log, em m/s (o log dá km/h)."""
    v = [float(x) for x in re.findall(r"speed_kmh[=: ]+([\d.]+)", open(caminho, errors="ignore").read())]
    return max(v) / 3.6 if v else None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    log = sys.argv[1]
    largura = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0

    cs = curvas(log)
    if not cs:
        print(f"{log}: nenhuma geometria de waypoint no log "
              f"(a pista só é impressa em algumas corridas)")
        sys.exit(1)

    cs.sort(key=lambda x: x[3])
    print(f"=== {log} · largura {largura:.0f} m ===")
    print(f"{len(cs)} trechos com curvatura\n")
    print(f"{'WP':>5} {'compr.':>8} {'ângulo':>8} {'raio eixo':>10} {'raio linha':>11} "
          f"{'v_safe eixo':>12} {'v_safe linha':>13}")
    print("-" * 76)
    for i, seg, ang, r in cs[:8]:
        # a linha de corrida abre o raio em até W/2 (fora-dentro-fora)
        r_linha = r + largura / 2.0
        v_eixo = math.sqrt(MU * G * r)
        v_linha = math.sqrt(MU * G * r_linha)
        print(f"{i:>5} {seg:>7.1f}m {ang:>7.1f}° {r:>9.1f}m {r_linha:>10.1f}m "
              f"{v_eixo:>9.1f}m/s {v_linha:>10.1f}m/s")
    print("-" * 76)

    pior = cs[0]
    r_linha = pior[3] + largura / 2.0
    v_teto = math.sqrt(MU * G * r_linha)
    print(f"\nCURVA MAIS FECHADA: WP[{pior[0]}], raio de eixo {pior[3]:.1f} m")
    print(f"  teto físico na linha de corrida: {v_teto:.1f} m/s = {v_teto*3.6:.0f} km/h")

    v = velocidade_alcancada(log)
    if v:
        print(f"  maior velocidade vista no log:   {v:.1f} m/s = {v*3.6:.0f} km/h")
        if v > v_teto * 1.15:
            print("\n  -> o carro passa ACIMA do teto na curva mais fechada. Se ele bate ali,")
            print("     é FÍSICA: nenhuma alocação neural resolve, só frear antes.")
        else:
            print("\n  -> o carro fica ABAIXO do teto físico. O que falta não é aderência —")
            print("     é decidir a tempo. Aqui alocação de neurônios importa.")

    print(f"\nEFEITO DA LARGURA no teto da curva mais fechada:")
    for w in (8, 10, 12, 14, 18, 24):
        vt = math.sqrt(MU * G * (pior[3] + w / 2.0))
        print(f"  {w:>2} m -> {vt:>5.1f} m/s ({vt*3.6:>3.0f} km/h)")
    print("\n  A largura levanta o teto pela RAIZ: triplicar a largura muda o teto em ~25%.")
    print("  Numa curva de raio 9 m, alargar a pista quase não compra velocidade.")

    # ---- o gargalo de verdade: dá tempo de frear?
    print("\n" + "=" * 76)
    print("DÁ TEMPO DE FREAR? (aqui é que a coisa se decide)")
    print("=" * 76)
    print("  d = (v_entrada² − v_curva²) / (2·μ·g)     mesma aderência que segura a curva")
    print(f"\n  {'v de entrada':>13} {'distância de frenagem':>23}   {'sonda que precisa ver':>24}")
    SONDAS = [(20.0, "20 m"), (40.0, "40 m"), (70.0, "70 m"), (120.0, "120 m")]
    for v0 in (20.0, 25.0, 30.0, 35.0, 40.0):
        if v0 <= v_teto:
            continue
        d = (v0 ** 2 - v_teto ** 2) / (2 * MU * G)
        sonda = next((nome for dist, nome in SONDAS if dist >= d), "NENHUMA — cega")
        print(f"  {v0:>8.0f} m/s {d:>21.1f} m   {sonda:>24}")
    print("\n  As sondas de curvatura do carro estão a 20, 40, 70 e 120 m (canais 20, 78, 79, 80).")
    print("  A conta acima diz QUAL delas ainda chega a tempo para cada velocidade de entrada.")
    print("\n  -> Se a frenagem exigida passa de 40 m, reagir pela sonda de 40 m é tarde: o carro")
    print("     PRECISA da sonda de 70/120 m, que é justamente a informação mais ruidosa e mais")
    print("     antecipada. Isso não é falta de aderência nem de motor — é HORIZONTE DE")
    print("     PERCEPÇÃO, e é decidido por qual via do cérebro recebe aquele canal.")
