#!/usr/bin/env python3
"""Compara dois treinos PAREADO POR GERAÇÃO — o jeito certo, aprendido errando.

# O erro que este script existe para não repetir

Comparei a fração exploradora 0,10 contra 0,50 tirando a média de **46 gerações de um lado e
8 do outro**, sobre intervalos diferentes. Deu "+264%, 4,3σ" — e era artefato. Refeito pareado
por geração, a mesma diferença virou **+336 m ± 178, ou 1,9σ**: não significativa.

Como o desempenho VARIA ao longo das gerações, média sobre janelas distintas compara coisas
distintas. Uma condição que rodou até a geração 46 inclui a fase madura; outra que parou na 8
só tem a fase inicial. A diferença entre elas mede a fase, não o tratamento.

**Comparação de série temporal tem de ser pareada por índice.** Este script só usa as
gerações presentes NOS DOIS lados, e reporta a diferença pareada com seu erro.

Uso:  scripts/comparar_treinos.py <dirA> <dirB> [rótuloA] [rótuloB]
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


def por_geracao(diretorio: str):
    """{geração: [(mediana, net_max, ≥1volta, ≥2voltas, mortalidade), ...]}"""
    tab: dict[int, list] = {}
    for f in sorted(glob.glob(os.path.join(diretorio, "*.log"))):
        for m in RX.finditer(open(f, errors="ignore").read()):
            g = int(m.group(1))
            tot = max(int(m.group(9)), 1)
            tab.setdefault(g, []).append((
                float(m.group(7)), float(m.group(6)),
                (int(m.group(3)) + int(m.group(4)) + int(m.group(5))) / tot,
                (int(m.group(4)) + int(m.group(5))) / tot,
                int(m.group(8)) / tot))
    return tab


def media_por_gen(tab, g, i):
    return st.mean([x[i] for x in tab[g]])


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    dA, dB = sys.argv[1], sys.argv[2]
    rA = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(dA)
    rB = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(dB)

    A, B = por_geracao(dA), por_geracao(dB)
    if not A or not B:
        print(f"sem dados: {rA}={len(A)} gens · {rB}={len(B)} gens")
        sys.exit(1)

    comuns = sorted(set(A) & set(B))
    if not comuns:
        print("nenhuma geração em comum — nada a parear")
        sys.exit(1)
    print(f"{rA}: gens 1–{max(A)}  ·  {rB}: gens 1–{max(B)}  ·  "
          f"PAREADO em {len(comuns)} gerações comuns\n")

    # trajetória em blocos, só nas gerações comuns
    print(f"{'gens':>9} {'MEDIANA A':>11} {'MEDIANA B':>11} {'Δ':>9} "
          f"{'≥1 volta A':>11} {'≥1 volta B':>11}")
    print("-" * 68)
    for ini in range(0, len(comuns), 10):
        ch = comuns[ini:ini + 10]
        ma = st.mean([media_por_gen(A, g, 0) for g in ch])
        mb = st.mean([media_por_gen(B, g, 0) for g in ch])
        va = st.mean([media_por_gen(A, g, 2) for g in ch])
        vb = st.mean([media_por_gen(B, g, 2) for g in ch])
        print(f"{ch[0]:>4}-{ch[-1]:<4} {ma:>10.0f}m {mb:>10.0f}m {ma-mb:>+8.0f}m "
              f"{va:>10.0%} {vb:>10.0%}")
    print("-" * 68)

    # diferença pareada: uma amostra por geração comum
    print(f"\n{'métrica':<14} {rA:>12} {rB:>12} {'Δ pareado':>14} {'σ':>7}  veredito")
    print("-" * 74)
    for i, nome, fmt in [(0, "MEDIANA", "m"), (1, "net_max", "m"),
                         (2, "≥1 volta", "%"), (3, "≥2 voltas", "%"),
                         (4, "mortalidade", "%")]:
        a = [media_por_gen(A, g, i) for g in comuns]
        b = [media_por_gen(B, g, i) for g in comuns]
        dif = [x - y for x, y in zip(a, b)]
        md = st.mean(dif)
        ed = st.stdev(dif) / len(dif) ** 0.5 if len(dif) > 1 else 0.0
        sig = abs(md) / ed if ed > 0 else 0.0
        if fmt == "%":
            cols = f"{st.mean(a):>11.1%} {st.mean(b):>11.1%} {md*100:>+12.1f}pp"
        else:
            cols = f"{st.mean(a):>11.0f}m {st.mean(b):>11.0f}m {md:>+12.0f}m"
        vd = "significativo" if sig > 2 else "dentro do ruído"
        print(f"{nome:<14} {cols} {sig:>6.1f}σ  {vd}")
    print("-" * 74)
    print("\nΔ pareado = média das diferenças GERAÇÃO A GERAÇÃO, não diferença das médias.")
    print("Corte em 2σ. Abaixo disso não se afirma nada, por mais bonita que seja a direção.")


if __name__ == "__main__":
    main()
