#!/usr/bin/env python3
"""Lê a fila autônoma e compara cada grupo contra a sua própria referência.

Toda condição aqui usa `--same-start`, então o elite% deve ficar perto de 100. Se cair, a
comparação daquele grupo não vale: a fitness voltou a medir sorte e não genoma.

Duas colunas decidem, e elas se opõem:
  med/max  perto de 1 = população inteira no nível do campeão (o que queremos)
  ganho    quanto a mediana subiu ENTRE a primeira e a última geração (progresso)
Monocultura dá med/max alto com ganho baixo: todos iguais, ninguém avança.
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
    if len(med) < 2:
        return None
    tail = lambda v: sum(v[-LAST_N:]) / len(v[-LAST_N:])
    return {"med": tail(med), "max": tail(mx), "dead": tail(dead),
            "ganho": tail(med) - med[0], "laps": sum(laps),
            "rep": (sum(rep) / len(rep)) if rep else float("nan")}


def main():
    groups = {}
    for d in sorted(ROOT.glob("telemetry/q_*_s*")):
        m = re.match(r"q_([a-z]+)_([a-z0-9]+)_s(\d+)$", d.name)
        if not m:
            continue
        r = read_run(d)
        if r:
            groups.setdefault(m.group(1), {}).setdefault(m.group(2), []).append(r)
    if not groups:
        print("nenhuma corrida concluída em telemetry/q_*")
        return 0

    for g in sorted(groups):
        print(f"\n=== {g} ===")
        print(f"{'cond':<10} | {'mediana':>8} {'máximo':>8} {'med/max':>7} {'ganho':>8} | "
              f"{'elite%':>6} {'morte%':>6} {'voltas':>6} | {'n':>2}")
        print("-" * 78)
        for name in sorted(groups[g]):
            rs = groups[g][name]
            a = lambda k: sum(r[k] for r in rs) / len(rs)
            med, mx = a("med"), a("max")
            print(f"{name:<10} | {med:8.1f} {mx:8.1f} {med/max(mx,1e-9):7.2f} {a('ganho'):+8.1f} | "
                  f"{a('rep'):6.0f} {a('dead'):6.1f} {sum(r['laps'] for r in rs):6d} | {len(rs):2d}")

    print("\nelite% longe de 100 ⇒ aquele grupo voltou a medir sorte; ignore a comparação.")
    print("med/max alto com ganho baixo ⇒ monocultura: todos iguais, ninguém avança.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
