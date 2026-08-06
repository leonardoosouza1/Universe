#!/usr/bin/env python3
"""Tendência de treino COM teste de robustez de janela — a checagem que me pegou duas vezes.

# Por que este script existe

Duas vezes numa sessão só eu anunciei um resultado que era sorte de janela:

1. `+264%, 4,3σ` na fração exploradora — comparava 46 gerações de um lado e 8 do outro.
   Refeito pareado: 1,1σ, e o sinal ainda inverteu.
2. `+2,17pp, 2,2σ` na fitness contínua — a inclinação inteira era carregada pela ÚLTIMA
   geração. Tirando ela: 1,2σ. Tirando duas: 0,4σ.

O padrão é o mesmo: uma regressão sobre poucos pontos com variância alta é dominada pelos
pontos das pontas. Reportar só o σ da janela cheia esconde isso.

**Regra que virou código:** toda tendência sai com a tabela de janelas encolhidas ao lado. Se
o σ desaba ao tirar as 2 últimas gerações, não é tendência — é a última geração.

Uso:  scripts/tendencia.py <dir1> [dir2 ...] [--pista METROS]
"""
from __future__ import annotations

import glob
import os
import re
import statistics as st
import sys

RX = re.compile(
    r"GEN (\d+) \| Laps: 0v=(\d+) 1v=(\d+) 2v=(\d+) 3\+v=(\d+) \| "
    r"NET max=([\d.]+)m median=([\d.]+)m \| Deaths (\d+)/(\d+)")

METRICAS = [(0, "≥2 voltas", "pp"), (1, "≥3 voltas", "pp"),
            (2, "MEDIANA", "v"), (3, "melhor", "v"), (4, "mortalidade", "pp")]


def ler(diretorio, pista):
    tab = {}
    for f in sorted(glob.glob(os.path.join(diretorio, "*.log"))):
        for m in RX.finditer(open(f, errors="ignore").read()):
            tot = max(int(m.group(9)), 1)
            tab.setdefault(int(m.group(1)), []).append((
                (int(m.group(4)) + int(m.group(5))) / tot,
                int(m.group(5)) / tot,
                float(m.group(7)) / pista,
                float(m.group(6)) / pista,
                int(m.group(8)) / tot))
    return {g: tuple(st.mean([x[i] for x in v]) for i in range(5)) for g, v in tab.items()}


def reg(tab, gens, i):
    ys = [tab[g][i] for g in gens]
    n = len(gens)
    if n < 4:
        return 0.0, 0.0
    mx, my = st.mean(gens), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in gens)
    if sxx == 0:
        return 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(gens, ys)) / sxx
    res = [y - (my + b * (x - mx)) for x, y in zip(gens, ys)]
    se = (sum(r * r for r in res) / (n - 2) / sxx) ** 0.5
    return b, (abs(b / se) if se > 0 else 0.0)


def main():
    argv = sys.argv[1:]
    pista = 1658.0
    if "--pista" in argv:
        i = argv.index("--pista")
        pista = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if not argv:
        print(__doc__)
        sys.exit(2)

    for d in argv:
        tab = ler(d, pista)
        if len(tab) < 5:
            print(f"{d}: {len(tab)} gerações — pouco para tendência\n")
            continue
        gens = sorted(tab)
        print(f"=== {d} · gerações 1–{max(gens)} ===\n")
        print(f"{'métrica':<14} {'inclinação':>14} {'janela cheia':>13} "
              f"{'-1 ger':>8} {'-2 ger':>8} {'-3 ger':>8}  veredito")
        print("-" * 84)
        for i, nome, u in METRICAS:
            b, s0 = reg(tab, gens, i)
            f = 100 if u == "pp" else 1
            # ROBUSTEZ pela INCLINAÇÃO, não pelo σ. σ depende de n e oscila muito com poucos
            # pontos — tirar uma geração pode derrubar σ de 2,0 para 1,1 sem que o efeito
            # tenha mudado. O tamanho do efeito é que tem de ficar de pé.
            incl = [reg(tab, gens[:len(gens) - k], i)[0] * f for k in (1, 2, 3)]
            val = b * f
            mesma_direcao = all(x * val > 0 for x in incl) if val != 0 else False
            estavel = mesma_direcao and all(abs(x) > abs(val) * 0.5 for x in incl)
            if s0 > 2 and estavel:
                vd = "TENDÊNCIA REAL"
            elif s0 > 2:
                vd = "SORTE DE JANELA (efeito instável)"
            else:
                vd = "dentro do ruído"
            print(f"{nome:<14} {val:>+9.3f} {u}/gen {s0:>11.1f}σ "
                  f"{incl[0]:>+7.2f} {incl[1]:>+7.2f} {incl[2]:>+7.2f}  {vd}")
        print("-" * 84)
        print("  As 3 últimas colunas são a INCLINAÇÃO recalculada tirando 1, 2 e 3 gerações")
        print("  do fim. Se ela muda de sinal ou some, o σ da janela cheia era a última")
        print("  geração falando mais alto que o resto.\n")

    # ---- teste de INTERAÇÃO quando há exatamente duas condições
    if len(argv) == 2:
        a, b = ler(argv[0], pista), ler(argv[1], pista)
        com = sorted(set(a) & set(b))
        if len(com) >= 5:
            print(f"=== {os.path.basename(argv[1])} MENOS {os.path.basename(argv[0])} — "
                  f"teste de interação, {len(com)} gerações pareadas ===\n")
            print("  Comparar 'um deu 2,0σ e o outro 0,2σ' NÃO testa se as condições diferem —")
            print("  é a falácia da diferença de significâncias. O teste certo é regredir a")
            print("  DIFERENÇA pareada geração a geração.\n")
            print(f"  {'métrica':<14} {'incl. da diferença':>20} {'σ':>7}   veredito")
            print("  " + "-" * 64)
            difs = {g: tuple(b[g][i] - a[g][i] for i in range(5)) for g in com}
            for i, nome, u in METRICAS:
                bb, ss = reg(difs, com, i)
                f = 100 if u == "pp" else 1
                vd = "DIFEREM" if ss > 2 else "não dá para afirmar"
                print(f"  {nome:<14} {bb*f:>+15.3f} {u}/gen {ss:>6.1f}σ   {vd}")
            print("  " + "-" * 64 + "\n")


if __name__ == "__main__":
    main()
