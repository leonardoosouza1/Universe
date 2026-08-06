#!/usr/bin/env python3
"""O ponto de troca automático (85% do RPM) é o ótimo? — o teto da via `marcha`.

Mesma disciplina do resto: medir o teto ANTES de acreditar no mecanismo. Se a regra
automática de `motor.rs:2099` já troca no ponto certo, o cérebro não tem o que acrescentar,
a ablação da via `marcha` vai dar ~0, e acrescentar mais detalhe de motor não dá ao modelo
mais o que decidir — dá mais o que a regra decide sozinha.

A física está toda em `physics.rs:380`:

    mult(r) = 1 − (r − 0.6)²·2        r = rpm / max_rpm     (pico em 60%)
    mult = 0                          se r > 1.0             (limitador)
    mult *= r/0.2                     se r < 0.2             (queda embaixo)

    torque na roda ∝ mult(r) · relação_da_marcha

Trocar de marcha derruba o RPM pela razão entre as relações. Então o ponto ótimo de troca é
onde o torque na roda da marcha SEGUINTE passa o da atual:

    mult(r) · g_atual  <  mult(r · g_prox/g_atual) · g_prox
"""
from __future__ import annotations

# relações reais de physics.rs:263 — índice 0=ré, 1=neutro, 2..6 = 1ª a 5ª
GEARS = [-3.5, 0.0, 3.8, 2.6, 1.9, 1.45, 1.15]
AUTO_UPSHIFT = 0.85   # motor.rs:2099
AUTO_DOWNSHIFT = 0.30  # motor.rs:2103


def mult(r: float) -> float:
    """Multiplicador de torque em função do RPM normalizado."""
    if r > 1.0:
        return 0.0
    m = 1.0 - (r - 0.6) ** 2 * 2.0
    if r < 0.2:
        m *= r / 0.2
    return max(0.0, min(1.0, m))


def ponto_otimo(g_atual: int) -> tuple[float, float, float]:
    """Onde trocar de `g_atual` para a próxima. Devolve (r_otimo, ganho%, r_apos_troca)."""
    a, b = GEARS[g_atual], GEARS[g_atual + 1]
    razao = b / a                      # < 1: a marcha seguinte derruba o RPM
    melhor_r, melhor_ganho = None, 0.0
    r = 0.20
    while r <= 1.0:
        atual = mult(r) * a
        prox = mult(r * razao) * b
        if prox > atual and melhor_r is None:
            melhor_r = r               # primeiro ponto em que trocar já compensa
        r += 0.005
    if melhor_r is None:
        return (1.0, 0.0, 1.0)
    # quanto se perde por trocar no ponto automático em vez do ótimo
    def roda(r_troca):
        # torque médio na roda percorrendo de r_troca até o limitador na marcha atual,
        # e do RPM correspondente em diante na seguinte
        passos = 0
        soma = 0.0
        r = 0.30
        while r < r_troca:
            soma += mult(r) * a; passos += 1; r += 0.01
        r2 = r_troca * razao
        while r2 < 1.0:
            soma += mult(r2) * b; passos += 1; r2 += 0.01
        return soma / max(passos, 1)

    perda = (roda(melhor_r) - roda(AUTO_UPSHIFT)) / max(roda(melhor_r), 1e-9) * 100
    return (melhor_r, perda, melhor_r * razao)


if __name__ == "__main__":
    print("PONTO DE TROCA: a regra automática de 85% é o ótimo?\n")
    print("torque na roda ∝ mult(rpm/max) · relação   ·   mult tem pico em 60% do RPM\n")
    print(f"{'marcha':>7} {'relação':>9} {'próxima':>9} {'RPM ótimo':>11} "
          f"{'RPM após troca':>15} {'perda do automático':>21}")
    print("-" * 80)
    perdas = []
    for g in range(2, 6):
        r_ot, perda, r_dep = ponto_otimo(g)
        perdas.append(perda)
        print(f"{g-1:>7} {GEARS[g]:>9.2f} {GEARS[g+1]:>9.2f} {r_ot*100:>10.0f}% "
              f"{r_dep*100:>14.0f}% {perda:>20.2f}%")
    print("-" * 80)
    media = sum(perdas) / len(perdas)
    print(f"\nperda média por trocar em {AUTO_UPSHIFT:.0%} em vez do ótimo: {media:.2f}%")
    print(f"\nEsse é o TETO da via `marcha`: o máximo que o cérebro pode ganhar decidindo")
    print(f"a troca melhor que a regra fixa. Se a ablação de `marcha` na bateria der um")
    print(f"custo muito maior que {media:.2f}%, é sinal de que ela está fazendo outra coisa")
    print(f"além de escolher o ponto de troca — provavelmente segurando marcha na curva.")

    print("\n" + "=" * 80)
    print("A FAIXA EM QUE O CÉREBRO CHEGA A DECIDIR")
    print("=" * 80)
    print(f"  A sobreposição neural (motor.rs:2107) é o ÚLTIMO else-if. Só é alcançada com")
    print(f"  embreagem colada e RPM entre {AUTO_DOWNSHIFT:.0%} e {AUTO_UPSHIFT:.0%} do máximo —")
    print(f"  ou seja {AUTO_UPSHIFT-AUTO_DOWNSHIFT:.0%} da faixa útil, que é onde se roda a maior parte do tempo.")
    print(f"  Então o cérebro TEM o piso; a pergunta é se ele usa melhor que a regra.")
