#!/usr/bin/env python3
"""≥2 VOLTAS é impossível ou só difícil? — o teto de voltas por conta, antes de gastar GPU.

Leonardo pediu voltas completas: *"conseguem completar voltas? eu quero isso também, o quanto
mais eles viverem melhor"*. Nos treinos, `≥1 volta` chega a 33% e **`≥2 voltas` fica em 0,0%
em TODAS as gerações, nas duas condições**. Zero cravado em duas condições independentes não
parece dificuldade — parece parede.

Antes de mexer no cérebro, a pergunta certa é a mesma de sempre: **o carro não consegue, ou o
CÉREBRO não consegue?** Aqui ela tem resposta fechada, porque o limitante candidato é o
orçamento de tempo:

    generation_timeout_s = 120 s   (config.rs:72)

Se a volta mais rápida FISICAMENTE possível for maior que 60 s, `≥2 voltas` é impossível por
construção — e todo neurônio, toda mutação e toda geração gasta ali mede ruído sobre uma
restrição que nenhuma delas alcança.

# Como se calcula a volta mais rápida possível

Perfil de velocidade clássico, três passadas sobre os waypoints:

  1. TETO DE CURVA por ponto, com a linha de corrida abrindo o raio em W/2:
         v_max[i] = sqrt(μ·g·(r_i + W/2))
  2. PASSADA PARA TRÁS (frenagem): não adianta chegar rápido no ponto i se o ponto i+1 é
     fechado. Propaga o limite para trás com  v² = v_next² + 2·a_fren·ds.
  3. PASSADA PARA FRENTE (aceleração): o carro não teleporta à velocidade da curva; sobe
     limitado por  v² = v_prev² + 2·a_acel·ds,  e pelo teto de motor.

O tempo sai integrando  dt = ds / v_médio  em cada segmento. Isso é um teto OTIMISTA: supõe
linha de corrida perfeita, aderência total o tempo todo e conhecimento futuro perfeito. Um
carro real (e um cérebro evoluído) fica ABAIXO. Se nem o teto otimista fecha 2 voltas, está
respondido.

Uso:  scripts/teto_de_voltas.py <log com WP[...]> [largura_m] [timeout_s]
"""
from __future__ import annotations

import math
import re
import sys

G = 9.81
MU = 1.0
A_ACEL = 8.0     # m/s² — aceleração de tração, generosa
A_FREN = MU * G  # m/s² — freio limitado pela mesma aderência que segura a curva
V_MOTOR = 85.0   # m/s teto de motor (~306 km/h), folgado de propósito

WP = re.compile(r"WP\[(\d+)\]\s+\(([-\d.]+),([-\d.]+)\)\s+seg_len=([\d.]+)m\s+turn=([-\d.]+)°")


def geometria(caminho):
    """[(seg_len, raio)] por waypoint, deduplicado (o log reimprime a pista a cada geração)."""
    vistos, pts = set(), []
    for ln in open(caminho, errors="ignore"):
        m = WP.search(ln)
        if not m:
            continue
        i = int(m.group(1))
        if i in vistos:
            continue
        vistos.add(i)
        seg, ang = float(m.group(4)), abs(float(m.group(5)))
        # reta -> raio infinito; senão r = arco / ângulo
        r = math.inf if ang < 0.05 else seg / math.radians(ang)
        pts.append((seg, r))
    return pts


def volta_mais_rapida(pts, largura):
    """Perfil de velocidade em circuito fechado. Devolve (tempo_s, v_media, v_min, v_max)."""
    n = len(pts)
    ds = [max(p[0], 0.1) for p in pts]
    # 1. teto de curva com a linha abrindo o raio
    v = [min(V_MOTOR, math.sqrt(MU * G * (p[1] + largura / 2.0)))
         if math.isfinite(p[1]) else V_MOTOR for p in pts]

    # 2/3. circuito é FECHADO: as passadas têm de dar a volta até estabilizar,
    # senão o começo da pista não sabe da curva que existe logo antes dele.
    for _ in range(4):
        for i in range(n - 1, -1, -1):                       # trás: frenagem
            prox = v[(i + 1) % n]
            v[i] = min(v[i], math.sqrt(prox ** 2 + 2 * A_FREN * ds[i]))
        for i in range(n):                                   # frente: aceleração
            ant = v[(i - 1) % n]
            v[i] = min(v[i], math.sqrt(ant ** 2 + 2 * A_ACEL * ds[i]))

    t = sum(d / max((v[i] + v[(i + 1) % n]) / 2.0, 0.5) for i, d in enumerate(ds))
    return t, sum(ds) / t, min(v), max(v)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    log = sys.argv[1]
    largura = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0

    pts = geometria(log)
    if not pts:
        print(f"{log}: sem geometria de waypoint")
        sys.exit(1)

    compr = sum(p[0] for p in pts)
    t, vmed, vmin, vmax = volta_mais_rapida(pts, largura)

    print(f"=== TETO DE VOLTAS · {log} ===")
    print(f"pista {compr:.0f} m · {len(pts)} waypoints · largura {largura:.0f} m · "
          f"orçamento {timeout:.0f} s\n")
    print(f"  volta mais rápida FISICAMENTE possível : {t:>6.1f} s")
    print(f"  velocidade média nessa volta           : {vmed:>6.1f} m/s = {vmed*3.6:.0f} km/h")
    print(f"  mais lento / mais rápido no traçado    : {vmin:>6.1f} / {vmax:.1f} m/s")
    print(f"\n  voltas que cabem em {timeout:.0f} s, no TETO : {timeout/t:>6.2f}")

    print("\n" + "-" * 70)
    cabe2 = timeout / t >= 2.0
    if not cabe2:
        falta = 2 * t - timeout
        print(f"  ≥2 VOLTAS É IMPOSSÍVEL NESTE ORÇAMENTO.")
        print(f"  Duas voltas custam {2*t:.0f} s no melhor caso físico; o orçamento dá {timeout:.0f} s.")
        print(f"  Faltam {falta:.0f} s ({falta/timeout*100:.0f}% a mais de tempo) para o teto sequer empatar.")
        print(f"\n  O 0,0% de ≥2 voltas em toda geração das duas condições NÃO é o cérebro")
        print(f"  falhando — é uma restrição que nenhuma mutação alcança. Continuar evoluindo")
        print(f"  contra ela é medir ruído.")
        print(f"\n  Para tornar ALCANÇÁVEL, uma das três (a primeira é a honesta):")
        print(f"    a) orçamento >= {2*t*1.35:.0f} s  (35% de folga sobre o teto — carro real não anda no teto)")
        print(f"    b) pista mais curta: <= {compr*timeout/(2*t)*0.74:.0f} m para 2 voltas com a mesma folga")
        print(f"    c) medir VOLTAS POR MINUTO em vez de contagem inteira — não tem degrau")
    else:
        print(f"  ≥2 voltas CABE no orçamento ({timeout/t:.2f} voltas no teto).")
        print(f"  Com folga realista de 35% sobre o teto físico: {timeout/(t*1.35):.2f} voltas.")
        if timeout / (t * 1.35) < 2.0:
            print(f"  -> cabe no teto mas NÃO com folga realista. Fica na fronteira: espere")
            print(f"     ≥2 voltas raro e ruidoso, não zero cravado.")
        print(f"  Então o 0,0% observado é do CÉREBRO, e vale evoluir contra.")
    print("-" * 70)

    print("\nSENSIBILIDADE — quanto orçamento cada nº de voltas exige (com 35% de folga):")
    for k in (1, 2, 3):
        print(f"  {k} volta{'s' if k > 1 else ' '}: {k*t*1.35:>6.0f} s de orçamento")
