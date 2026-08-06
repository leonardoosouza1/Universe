#!/usr/bin/env python3
"""Confronta a velocidade medida com a previsão v* = 64/earn.

A previsão saiu da aritmética da regra, não de ajuste: o ser gasta 1 tick de vida por tick
e ganha `earn` por metro, então o equilíbrio é passo×earn = 1, com passo = v/64.

Se a velocidade medida seguir a curva, a regra está governando o comportamento — e o valor
de `earn` vira um MOSTRADOR de velocidade-alvo, não um hiperparâmetro a tatear.

Se saturar bem abaixo do previsto, o limite é do CORPO (motor, aderência, raio de curva), e
o ponto onde satura é o dado que interessa: é a velocidade máxima que este corpo sustenta
nesta pista.
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
    tail = lambda a: sum(a[-LAST_N:]) / len(a[-LAST_N:]) if a else float("nan")
    return {"med": tail(med), "max": tail(mx), "vel": tail(vel), "life": tail(life),
            "laps": sum(laps), "rep": (sum(rep) / len(rep)) if rep else float("nan")}


def main():
    data = {}
    for d in sorted(ROOT.glob("telemetry/le_e*_s*")):
        m = re.match(r"le_e([\d.]+)_s(\d+)$", d.name)
        if not m:
            continue
        r = read_run(d)
        if r:
            data.setdefault(m.group(1), []).append(r)
    if not data:
        print("nenhuma corrida concluída em telemetry/le_e*")
        return 0

    print(f"{'earn':>5} | {'v previsto':>10} {'v medido':>9} {'razão':>6} | "
          f"{'mediana':>8} {'vida%':>6} {'elite%':>6} {'voltas':>6} | {'n':>2}")
    print("-" * 76)
    for e in sorted(data, key=float):
        rs = data[e]
        a = lambda k: sum(r[k] for r in rs) / len(rs)
        ef = float(e)
        prev = 64.0 / ef if ef > 0 else float("inf")
        vm = a("vel")
        razao = vm / prev if prev not in (0, float("inf")) else float("nan")
        prev_s = "—" if ef == 0 else f"{prev:.1f}"
        raz_s = "—" if ef == 0 else f"{razao:.2f}"
        print(f"{e:>5} | {prev_s:>10} {vm:9.2f} {raz_s:>6} | "
              f"{a('med'):8.1f} {a('life'):6.0f} {a('rep'):6.0f} "
              f"{sum(r['laps'] for r in rs):6d} | {len(rs):2d}")
    print("-" * 76)
    print("\nrazão ≈ 1 ⇒ a regra governa: `earn` é mostrador de velocidade-alvo.")
    print("razão < 1 que CAI conforme earn diminui ⇒ saturou no limite do corpo;")
    print("  a velocidade onde satura é o teto real desta pista.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
