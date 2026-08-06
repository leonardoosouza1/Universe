#!/usr/bin/env python3
"""Existe um `earn` em que o ser SOBREVIVE e ainda anda mais que rastejando?

A coluna que decide é `v/v*`: velocidade atingida sobre o break-even 64/earn. Abaixo de 1 a
vida escoa e ele morre; a partir de 1 ele se sustenta e a distância passa a ser
velocidade × geração inteira.

A segunda coluna que decide é a distância comparada ao baseline earn=0. Velocidade alta com
distância menor significa que ele corre e morre — trocou uma limitação por outra.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAST_N = 3
GEN = re.compile(r"gen\s+(\d+)\s*\|\s*med=\s*([\d.]+)m.*?max=\s*([\d.]+)m.*?≥1V=(\d+)")
VEL = re.compile(r"\[veloc\] mediana ([\d.]+) m/s .*?vida média ([\d.]+)% da geração")
ELITE = re.compile(r"\[elite\] mesmo genoma: [\d.]+ m → [\d.]+ m \((-?[\d.]+)% do anterior\)")


def read_run(d: Path):
    log = d / "run.log"
    if not (d / "DONE").exists() or not log.exists():
        return None
    med, mx, laps, vel, life, rep = [], [], [], [], [], []
    for line in log.read_text(errors="replace").splitlines():
        g = GEN.search(line)
        if g:
            med.append(float(g.group(2))); mx.append(float(g.group(3))); laps.append(int(g.group(4)))
        v = VEL.search(line)
        if v:
            vel.append(float(v.group(1))); life.append(float(v.group(2)))
        e = ELITE.search(line)
        if e:
            rep.append(float(e.group(1)))
    if not med:
        return None
    t = lambda a: sum(a[-LAST_N:]) / len(a[-LAST_N:]) if a else float("nan")
    return {"med": t(med), "max": t(mx), "vel": t(vel), "life": t(life),
            "laps": sum(laps), "rep": (sum(rep) / len(rep)) if rep else float("nan")}


def main():
    data = {}
    for d in sorted(ROOT.glob("telemetry/sus_e*_s*")):
        m = re.match(r"sus_e([\d.]+)_s(\d+)$", d.name)
        if m:
            r = read_run(d)
            if r:
                data.setdefault(m.group(1), []).append(r)
    if not data:
        print("nenhuma corrida concluída em telemetry/sus_e*")
        return 0

    base = data.get("0")
    base_med = (sum(r["med"] for r in base) / len(base)) if base else float("nan")

    print(f"{'earn':>5} | {'v*':>6} {'v':>6} {'v/v*':>6} | {'vida%':>6} | "
          f"{'mediana':>8} {'vs base':>8} | {'elite%':>6} {'voltas':>6} | {'n':>2}")
    print("-" * 82)
    for e in sorted(data, key=float):
        rs = data[e]
        a = lambda k: sum(r[k] for r in rs) / len(rs)
        ef = float(e)
        vstar = 64.0 / ef if ef > 0 else float("nan")
        v = a("vel")
        ratio = v / vstar if ef > 0 else float("nan")
        med = a("med")
        vs = med - base_med
        print(f"{e:>5} | {('—' if ef==0 else f'{vstar:.2f}'):>6} {v:6.2f} "
              f"{('—' if ef==0 else f'{ratio:.2f}'):>6} | {a('life'):6.0f} | "
              f"{med:8.1f} {vs:+8.1f} | {a('rep'):6.0f} {sum(r['laps'] for r in rs):6d} | {len(rs):2d}")
    print("-" * 82)
    print("\nv/v* ≥ 1 ⇒ o ser se sustenta: a vida deixa de ser o limite.")
    print("velocidade alta com distância MENOR ⇒ corre e morre; trocou uma limitação por outra.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
