#!/usr/bin/env python3
"""Compara níveis de tanque. O que decide são VOLTAS e SETORES, não distância.

Um combustível que reduz distância mas aumenta setores cruzados está funcionando: trocou
vagar por avançar. Distância é a métrica que premiava girar em círculo — foi ela que
escondeu o problema por um dia inteiro.

Também mostra quantos ficaram SECOS: se ninguém seca, o tanque é grande demais e o
mecanismo existe sem morder; se todos secam na geração 0, é pequeno demais para aprender.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAST_N = 5
GEN = re.compile(r"gen\s+(\d+)\s*\|\s*med=\s*([\d.]+)m.*?max=\s*([\d.]+)m.*?morto=\s*([\d.]+)%")
SET = re.compile(r"\[setores\] mediana (\d+)/(\d+) · melhor (\d+)/\d+ · voltas reais: (\d+) seres ≥1, (\d+) ≥2")
FUEL = re.compile(r"\[combustível\] secos (\d+)/(\d+) · tanque restante mediano ([\d.]+)")


def read_run(d: Path):
    log = d / "run.log"
    if not (d / "DONE").exists() or not log.exists():
        return None
    med, mx, dead, cp_med, cp_max, v1, v2, secos, resto = [], [], [], [], [], [], [], [], []
    for line in log.read_text(errors="replace").splitlines():
        g = GEN.search(line)
        if g:
            med.append(float(g.group(2))); mx.append(float(g.group(3))); dead.append(float(g.group(4)))
        s = SET.search(line)
        if s:
            cp_med.append(int(s.group(1))); cp_max.append(int(s.group(3)))
            v1.append(int(s.group(4))); v2.append(int(s.group(5)))
        f = FUEL.search(line)
        if f:
            secos.append(100.0 * int(f.group(1)) / max(int(f.group(2)), 1))
            resto.append(float(f.group(3)))
    if not med:
        return None
    t = lambda a: sum(a[-LAST_N:]) / len(a[-LAST_N:]) if a else 0.0
    return {"med": t(med), "max": t(mx), "dead": t(dead),
            "cp_med": t(cp_med), "cp_max": t(cp_max),
            "v1_total": sum(v1), "v1_pico": max(v1) if v1 else 0, "v2_total": sum(v2),
            "secos": t(secos), "resto": t(resto)}


def main():
    data = {}
    for d in sorted(ROOT.glob("telemetry/fuel_t*_s*")):
        m = re.match(r"fuel_t([\d.]+)_s(\d+)$", d.name)
        if m:
            r = read_run(d)
            if r:
                data.setdefault(m.group(1), []).append(r)
    if not data:
        print("nenhuma corrida concluída em telemetry/fuel_t*")
        return 0

    print(f"{'tanque':>7} | {'setores':>8} {'melhor':>7} | {'≥1 volta':>9} {'pico':>5} {'≥2':>4} | "
          f"{'secos%':>7} {'resto':>6} | {'mediana':>8} {'morte%':>7} | {'n':>2}")
    print("-" * 92)
    for t in sorted(data, key=float):
        rs = data[t]
        a = lambda k: sum(r[k] for r in rs) / len(rs)
        print(f"{t:>7} | {a('cp_med'):8.1f} {a('cp_max'):7.1f} | "
              f"{sum(r['v1_total'] for r in rs):9d} {max(r['v1_pico'] for r in rs):5d} "
              f"{sum(r['v2_total'] for r in rs):4d} | "
              f"{a('secos'):7.0f} {a('resto'):6.2f} | {a('med'):8.0f} {a('dead'):7.1f} | {len(rs):2d}")
    print("-" * 92)
    print("\nsetores de 24 · '≥1 volta' soma seres-com-volta em todas as gerações · 'pico' é o melhor")
    print("secos% = quantos ficaram sem combustível. 0% ⇒ tanque grande demais, não morde.")
    print("100% já na geração 0 ⇒ pequeno demais, mata antes de aprender.")

    base = data.get("0")
    if base:
        b = sum(r["cp_med"] for r in base) / len(base)
        print(f"\nContra o controle sem combustível (setores medianos {b:.1f}):")
        for t in sorted(data, key=float):
            if t == "0":
                continue
            rs = data[t]
            d = sum(r["cp_med"] for r in rs) / len(rs) - b
            v = sum(r["v1_total"] for r in rs)
            print(f"  tanque {t:>5}: setores {d:+5.1f} · {v} seres completaram volta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
