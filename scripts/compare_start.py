#!/usr/bin/env python3
"""Compara condições de largada: a seleção está medindo o cérebro ou o lugar de nascimento?

A coluna que decide é REPRODUTIBILIDADE DO ELITE. O elite passa de geração sem mutação,
então é o mesmo genoma reavaliado. Se ele rende ~100% do resultado anterior, a fitness é
propriedade do genoma e a seleção tem o que herdar. Se rende muito menos, a seleção premiou
sorte — e nenhuma mutação melhor conserta isso, porque não havia nada a transmitir.

A segunda coluna é a razão MEDIANA/MÁXIMO. Perto de 1 = a população inteira chegou ao nível
do campeão (o que o Leonardo chama de "cérebro populado"). Perto de 0 = um gênio sortudo e
59 seres perdidos.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAST_N = 3
GEN = re.compile(r"gen\s+(\d+)\s*\|\s*med=\s*([\d.]+)m.*?max=\s*([\d.]+)m.*?≥1V=(\d+).*?morto=\s*([\d.]+)%")
ELITE = re.compile(r"\[elite\] mesmo genoma: [\d.]+ m → [\d.]+ m \((-?[\d.]+)% do anterior\)")


def read_run(d: Path):
    log = d / "run.log"
    if not (d / "DONE").exists() or not log.exists():
        return None
    med, mx, laps, dead, rep = [], [], [], [], []
    for line in log.read_text(errors="replace").splitlines():
        g = GEN.search(line)
        if g:
            med.append(float(g.group(2))); mx.append(float(g.group(3)))
            laps.append(int(g.group(4))); dead.append(float(g.group(5)))
        e = ELITE.search(line)
        if e:
            rep.append(float(e.group(1)))
    if not med:
        return None
    tail = lambda v: sum(v[-LAST_N:]) / len(v[-LAST_N:]) if v else float("nan")
    return {"med": tail(med), "max": tail(mx), "dead": tail(dead),
            "laps": sum(laps), "rep": (sum(rep) / len(rep)) if rep else float("nan"),
            "rep_min": min(rep) if rep else float("nan")}


def main():
    conds = {}
    for d in sorted(ROOT.glob("telemetry/start_*_s*")):
        m = re.match(r"start_(.+)_s(\d+)$", d.name)
        if not m:
            continue
        r = read_run(d)
        if r:
            conds.setdefault(m.group(1), []).append(r)
    if not conds:
        print("nenhuma corrida concluída em telemetry/start_*")
        return 1

    print(f"{'condição':<14} | {'mediana':>8} {'máximo':>8} {'med/max':>8} | "
          f"{'elite%':>7} {'pior%':>7} | {'morte%':>7} {'voltas':>6} | {'n':>2}")
    print("-" * 88)
    for name in sorted(conds):
        rs = conds[name]
        agg = lambda k: sum(r[k] for r in rs) / len(rs)
        med, mx = agg("med"), agg("max")
        print(f"{name:<14} | {med:8.1f} {mx:8.1f} {med/max(mx,1e-9):8.2f} | "
              f"{agg('rep'):7.0f} {min(r['rep_min'] for r in rs):7.0f} | "
              f"{agg('dead'):7.1f} {sum(r['laps'] for r in rs):6d} | {len(rs):2d}")
    print("-" * 88)
    print("\nelite%  = o MESMO genoma reavaliado. ~100 = a fitness é do genoma.")
    print("med/max = quanto da população alcançou o campeão. ~1 = cérebro populado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
