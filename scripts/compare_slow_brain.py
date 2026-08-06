"""Compara, pareado por semente, as condições COM e SEM a metade lenta do cérebro.

Por que pareado: duas condições com uma amostra cada não distinguem efeito de sorte.
Neste projeto um achado de "+264%, 4,3σ" virou −207 m com o SINAL INVERTIDO quando
finalmente foi medido pareado — o 4,3σ vinha de comparar janelas diferentes.

Por que média dos últimos N e não o máximo de todos: "melhor de todos" CRESCE com o
número de amostras mesmo sem melhora nenhuma. Uma varredura de 15 gerações elegeu por
esse critério a condição PERDEDORA, e só 100 gerações mostraram o erro.

Por que o limiar de ruído de busca (Lei do Garimpo): escolher a melhor de M variantes faz
o máximo de puro ruído valer σ·sqrt(2·ln M). Ganho abaixo disso é sorte, não achado.

Por que também o teste de SINAL: o sentido pode ser consistente enquanto a magnitude ainda
oscila. Com n pequeno, "todas as sementes apontam para o mesmo lado" carrega informação que
a média sozinha esconde.
"""
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAST_N = 5

GEN_LINE = re.compile(r"gen\s+(\d+)\s*\|\s*med=\s*([\d.]+)m.*?max=\s*([\d.]+)m.*?≥1V=(\d+)")
SLOW_LINE = re.compile(r"\[lento\] PR=([\d.]+)/3")


def read_run(path: Path):
    """Extrai (mediana, máximo, laps) por geração e a razão de participação média."""
    if not path.exists():
        return None
    medians, maxima, laps, ratios = [], [], [], []
    for line in path.read_text(errors="replace").splitlines():
        m = GEN_LINE.search(line)
        if m:
            medians.append(float(m.group(2)))
            maxima.append(float(m.group(3)))
            laps.append(int(m.group(4)))
        p = SLOW_LINE.search(line)
        if p:
            ratios.append(float(p.group(1)))
    if not medians:
        return None
    return {
        "med": sum(medians[-LAST_N:]) / len(medians[-LAST_N:]),
        "max": sum(maxima[-LAST_N:]) / len(maxima[-LAST_N:]),
        "laps": sum(laps),
        "gens": len(medians),
        "pr": (sum(ratios) / len(ratios)) if ratios else None,
    }


def main():
    seeds = sorted({
        d.name.rsplit("_s", 1)[1]
        for d in ROOT.glob("telemetry/lento_*_s*")
        if "_s" in d.name
    })
    if not seeds:
        print("nenhuma corrida encontrada em telemetry/lento_*")
        return 1

    print(f"{'semente':>8} | {'med COM':>9} {'med SEM':>9} {'Δ':>8} | "
          f"{'max COM':>9} {'max SEM':>9} | {'V COM':>6} {'V SEM':>6} | {'PR':>5}")
    print("-" * 84)

    deltas, ratios = [], []
    for s in seeds:
        with_slow = read_run(ROOT / f"telemetry/lento_com_s{s}/run.log")
        without_slow = read_run(ROOT / f"telemetry/lento_sem_s{s}/run.log")
        if not with_slow or not without_slow:
            print(f"{s:>8} | incompleta (with_slow={bool(with_slow)} without_slow={bool(without_slow)})")
            continue
        d = with_slow["med"] - without_slow["med"]
        deltas.append(d)
        if with_slow["pr"] is not None:
            ratios.append(with_slow["pr"])
        print(f"{s:>8} | {with_slow['med']:9.1f} {without_slow['med']:9.1f} {d:+8.1f} | "
              f"{with_slow['max']:9.1f} {without_slow['max']:9.1f} | {with_slow['laps']:6d} {without_slow['laps']:6d} | "
              f"{(with_slow['pr'] if with_slow['pr'] is not None else float('nan')):5.2f}")

    # Teste de sinal: com n pequeno, o SENTIDO consistente carrega informação que a
    # média sozinha esconde. Sob a hipótese nula (sem efeito) cada semente tem 50% de
    # chance de cair para cada lado, então k acertos em n valem (1/2)^n unilateral.
    positivos = sum(1 for d in deltas if d > 0)
    if deltas:
        p_sinal = 0.5 ** len(deltas)
        print(f"\nsentido: {positivos}/{len(deltas)} sementes a favor "
              f"(se todas concordam, p unilateral = {p_sinal:.4f})")

    if len(deltas) < 2:
        print("\npares insuficientes para um veredito — rode mais seeds")
        return 0

    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(var)
    stderr = sd / math.sqrt(n) if sd > 0 else 0.0

    print("-" * 84)
    print(f"\nefeito pareado: {mean:+.1f} m  (desvio {sd:.1f}, erro padrão {stderr:.1f}, n={n})")
    if stderr > 0:
        print(f"razão: {abs(mean) / stderr:.2f}σ")

    # Lei do Garimpo: o limiar que separa achado de sorte de busca.
    m_alternatives = 2  # with_slow e without_slow — duas alternativas comparadas
    threshold = sd * math.sqrt(2 * math.log(m_alternatives)) if sd > 0 else 0.0
    print(f"limiar de ruído de busca (σ·sqrt(2·ln {m_alternatives})): {threshold:.1f} m")

    print()
    if abs(mean) <= threshold:
        print("VEREDITO: dentro do ruído de busca — a metade lenta NÃO se paga com estes dados.")
    elif mean > 0:
        print("VEREDITO: a metade lenta CONTRIBUI acima do limiar de ruído de busca.")
    else:
        print("VEREDITO: a metade lenta PREJUDICA acima do limiar de ruído de busca.")

    if ratios:
        pr = sum(ratios) / len(ratios)
        print(f"\nrazão de participação média: {pr:.2f}/3 ({100 * pr / 3:.0f}% da capacidade)")
        if pr < 2.4:
            print("  Capacidade ociosa: crescer o órgão seria só espaço. "
                  "O caminho é dar-lhe canais melhores, não mais neurônios.")
        else:
            print("  Capacidade saturada: aqui crescer o órgão pode pagar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
