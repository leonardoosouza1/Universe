#!/usr/bin/env python3
"""Compara custos de parede, pareado por semente.

A pergunta: dar CONSEQUÊNCIA a encostar na parede faz o ser aprender a evitá-la?

Não há regra nova dizendo para onde virar nem freio automático — só um custo de fitness
por segundo colado. Se a competência emergir, ela aparece como: menos tempo colado E mais
distância. Se o custo for alto demais, aparece o contrário — o ser para de andar para não
arriscar, e a distância cai junto com o tempo de parede. Por isso as DUAS colunas importam:
tempo colado sozinho é fácil de zerar ficando parado.

Só lê corridas com DONE. Uma corrida morta no meio tem "últimos N" vindos de outra janela,
e comparar janelas diferentes já produziu aqui um achado de "+264%, 4,3σ" que era artefato.
"""
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAST_N = 3

GEN_LINE = re.compile(r"gen\s+(\d+)\s*\|\s*med=\s*([\d.]+)m.*?max=\s*([\d.]+)m.*?≥1V=(\d+).*?morto=\s*([\d.]+)%")
WALL_LINE = re.compile(r"\[parede\]\s+([\d.]+)% dos ticks colado")


def read_run(path: Path):
    if not (path.parent / "DONE").exists():
        return None
    medians, maxima, laps, deaths, wall = [], [], [], [], []
    for line in path.read_text(errors="replace").splitlines():
        m = GEN_LINE.search(line)
        if m:
            medians.append(float(m.group(2)))
            maxima.append(float(m.group(3)))
            laps.append(int(m.group(4)))
            deaths.append(float(m.group(5)))
        w = WALL_LINE.search(line)
        if w:
            wall.append(float(w.group(1)))
    if not medians:
        return None
    tail = lambda v: sum(v[-LAST_N:]) / len(v[-LAST_N:]) if v else float("nan")
    return {
        "median": tail(medians), "max": tail(maxima),
        "wall": tail(wall), "death": tail(deaths),
        "laps": sum(laps), "gens": len(medians),
    }


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "wall"
    runs = sorted(ROOT.glob(f"telemetry/{tag}_c*_s*"))
    if not runs:
        print(f"nenhuma corrida em telemetry/{tag}_c*_s*")
        return 1

    data = {}
    for d in runs:
        m = re.match(rf"{tag}_c([\d.]+)_s(\d+)$", d.name)
        if not m:
            continue
        r = read_run(d / "run.log")
        if r:
            data.setdefault(m.group(1), {})[m.group(2)] = r

    if not data:
        print("nenhuma corrida CONCLUÍDA (falta DONE) — nada a comparar")
        return 0

    costs = sorted(data, key=float)
    print(f"{'custo':>7} | {'mediana':>9} {'máximo':>9} | {'colado%':>8} {'morte%':>8} | "
          f"{'voltas':>6} | {'n':>3}")
    print("-" * 70)
    resumo = {}
    for c in costs:
        seeds = data[c]
        agg = lambda k: sum(v[k] for v in seeds.values()) / len(seeds)
        resumo[c] = {k: agg(k) for k in ("median", "max", "wall", "death")}
        resumo[c]["laps"] = sum(v["laps"] for v in seeds.values())
        print(f"{c:>7} | {resumo[c]['median']:9.1f} {resumo[c]['max']:9.1f} | "
              f"{resumo[c]['wall']:8.1f} {resumo[c]['death']:8.1f} | "
              f"{resumo[c]['laps']:6d} | {len(seeds):3d}")
    print("-" * 70)

    base = costs[0]
    if float(base) != 0.0:
        print(f"\n(sem condição de custo 0 — não há baseline para comparar)")
        return 0

    print(f"\nContra o baseline (custo {base}):")
    for c in costs[1:]:
        seeds_comuns = set(data[base]) & set(data[c])
        if len(seeds_comuns) < 2:
            print(f"  custo {c}: menos de 2 sementes pareadas, pulando")
            continue
        d_med = [data[c][s]["median"] - data[base][s]["median"] for s in seeds_comuns]
        d_wall = [data[c][s]["wall"] - data[base][s]["wall"] for s in seeds_comuns]
        n = len(d_med)
        mean_med = sum(d_med) / n
        sd = math.sqrt(sum((x - mean_med) ** 2 for x in d_med) / (n - 1)) if n > 1 else 0.0
        positivos = sum(1 for x in d_med if x > 0)
        # Lei do Garimpo: piso de ruído ao escolher a melhor de M alternativas.
        threshold = sd * math.sqrt(2 * math.log(max(len(costs), 2)))
        print(f"  custo {c:>5}: distância {mean_med:+7.1f} m (σ={sd:5.1f}, {positivos}/{n} a favor, "
              f"piso de ruído {threshold:5.1f}) | tempo colado {sum(d_wall)/n:+6.1f} pp")

    # O que decide: menos parede E mais distância ao mesmo tempo.
    print("\nLeitura: o custo só PAGA se derrubar o tempo colado SEM derrubar a distância.")
    print("Tempo colado caindo junto com a distância = o ser parou de andar, não aprendeu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
