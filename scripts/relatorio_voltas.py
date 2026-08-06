#!/usr/bin/env python3
"""Relatório de treino em VOLTAS FRACIONÁRIAS — a métrica que não tem degrau.

# Por que fracionária

A contagem inteira de voltas escondeu, por várias gerações, carros que faziam **1,86 de 2
voltas**. `≥2 voltas = 0,0%` parecia fracasso do cérebro e era a métrica encostada na
fronteira física do orçamento de tempo. Volta fracionária (`net / comprimento_da_pista`) não
tem esse degrau: 1,86 aparece como 1,86.

Regra que ficou: **contagem inteira serve para comunicar, contínua serve para medir.** Este
relatório mostra as duas lado a lado, e a inteira sempre com a fracionária ao lado para o
degrau nunca mais passar despercebido.

Uso:  scripts/relatorio_voltas.py <dir com logs> [comprimento_pista_m]
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


def ler(diretorio):
    tab = {}
    for f in sorted(glob.glob(os.path.join(diretorio, "*.log"))):
        for m in RX.finditer(open(f, errors="ignore").read()):
            tot = max(int(m.group(9)), 1)
            tab.setdefault(int(m.group(1)), []).append((
                (int(m.group(3)) + int(m.group(4)) + int(m.group(5))) / tot,  # ≥1
                (int(m.group(4)) + int(m.group(5))) / tot,                    # ≥2
                int(m.group(5)) / tot,                                        # ≥3
                int(m.group(8)) / tot,                                        # mortalidade
                float(m.group(6)), float(m.group(7))))                        # net max, mediana
    return tab


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    d = sys.argv[1]
    pista = float(sys.argv[2]) if len(sys.argv) > 2 else 1658.0
    tab = ler(d)
    if not tab:
        print(f"sem dados em {d}")
        sys.exit(1)
    gens = sorted(tab)
    m = lambda v, i: st.mean([x[i] for x in v])

    print(f"=== {d} · {len(gens)} gerações · pista {pista:.0f} m ===\n")
    print(f"{'gens':>9} {'≥1v':>6} {'≥2v':>7} {'≥3v':>7} {'mortal.':>8} "
          f"{'MELHOR':>16} {'MEDIANA':>16}")
    print("-" * 76)
    for ini in range(0, len(gens), 10):
        ch = gens[ini:ini + 10]
        v = [x for g in ch for x in tab[g]]
        nm, md = m(v, 4), m(v, 5)
        print(f"{ch[0]:>4}-{ch[-1]:<4} {m(v,0):>5.0%} {m(v,1):>6.1%} {m(v,2):>6.1%} "
              f"{m(v,3):>7.0%} {nm:>8.0f}m {nm/pista:>5.2f}v {md:>8.0f}m {md/pista:>5.2f}v")
    print("-" * 76)

    q = max(len(gens) // 4, 1)
    ini = [x for g in gens[:q] for x in tab[g]]
    fim = [x for g in gens[-q:] for x in tab[g]]
    print(f"\n{'':22} {'1º quarto':>12} {'último quarto':>15} {'Δ':>12}")
    for i, nome in [(0, "≥1 volta"), (1, "≥2 voltas"), (2, "≥3 voltas"), (3, "mortalidade")]:
        a, b = m(ini, i), m(fim, i)
        print(f"  {nome:<20} {a:>11.1%} {b:>14.1%} {(b-a)*100:>+11.1f}pp")
    for i, nome in [(4, "MELHOR"), (5, "MEDIANA")]:
        a, b = m(ini, i) / pista, m(fim, i) / pista
        print(f"  {nome+' (voltas)':<20} {a:>11.2f}v {b:>14.2f}v {b-a:>+11.2f}v")

    pico2 = max(m(tab[g], 1) for g in gens)
    pico3 = max(m(tab[g], 2) for g in gens)
    picov = max(m(tab[g], 4) for g in gens) / pista
    print(f"\n  pico: ≥2 voltas {pico2:.1%} · ≥3 voltas {pico3:.1%} · "
          f"melhor carro {picov:.2f} voltas")

    # o aviso que eu queria ter tido antes de perder gerações medindo um zero
    teto = 220.0 / 50.5
    if picov > teto * 0.85:
        print(f"\n  ⚠️  o melhor carro está a {picov/teto:.0%} do teto físico do orçamento "
              f"({teto:.2f} voltas).")
        print(f"      A métrica está encostando no limite de novo — aumente o orçamento antes")
        print(f"      de creditar qualquer estagnação ao cérebro.")

    # VEREDITO POR REGRESSÃO, não por quartos.
    #
    # A primeira versão deste script comparava 1º quarto contra último quarto e escrevia
    # "EVOLUIU" com +7,5pp. A regressão sobre as MESMAS 10 gerações deu 0,6σ: ruído. Comparar
    # duas janelas ignora a variância entre gerações — e aqui ela é enorme (uma geração morre
    # inteira, a seguinte põe 41 de 60 carros em 3 voltas). Foi o mesmo erro das janelas
    # diferentes, dentro da ferramenta escrita para não repetir o erro das janelas diferentes.
    #
    # A tabela de quartos fica: ela mostra a trajetória. Só o VEREDITO passa a exigir σ.
    print("\n  TENDÊNCIA (regressão sobre todas as gerações, com erro):")
    veredito = []
    for i, nome, unidade in [(1, "≥2 voltas", "pp/gen"), (2, "≥3 voltas", "pp/gen"),
                             (5, "MEDIANA", "v/gen"), (3, "mortalidade", "pp/gen")]:
        xs = [g for g in gens for _ in tab[g]]
        ys = [(x[i] / pista if i == 5 else x[i]) for g in gens for x in tab[g]]
        n = len(xs)
        if n < 4:
            continue
        mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx == 0:
            continue
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        res = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
        se = (sum(r * r for r in res) / (n - 2) / sxx) ** 0.5 if n > 2 else 0.0
        sig = abs(b / se) if se > 0 else 0.0
        val = b if unidade == "v/gen" else b * 100
        ok = sig > 2
        veredito.append(ok)
        print(f"    {nome:<14} {val:>+7.3f} {unidade:<7} {sig:>5.1f}σ  "
              f"{'SIGNIFICATIVO' if ok else 'dentro do ruído'}")

    print("\n  " + ("EVOLUIU (tendência acima de 2σ)" if any(veredito) else
                    f"SEM EVOLUÇÃO COMPROVADA em {len(gens)} gerações — a variância entre "
                    f"gerações\n  ainda domina. Não é 'não evoluiu': é 'não dá para afirmar "
                    f"com estes dados'."))


if __name__ == "__main__":
    main()
