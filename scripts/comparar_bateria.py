#!/usr/bin/env python3
"""Compara todas as corridas de telemetria pela MÉDIA DOS ÚLTIMOS 10.

Nunca "melhor de todos": essa métrica cresce com o número de amostras mesmo sem melhora, e
foi assim que uma varredura de 15 gerações elegeu a condição PERDEDORA neste projeto.
Toda inclinação sai com σ e com o teste de janela encolhida (tirando as últimas 1/2/3).
"""
import glob, os, re, statistics as st, sys

RX = re.compile(r"gen\s+(\d+) \| med=\s*(\d+)m .* max=\s*(\d+)m")

def carrega(p):
    med, mx = [], []
    for l in open(p, errors='ignore'):
        m = RX.match(l)
        if m: med.append(int(m.group(2))); mx.append(int(m.group(3)))
    return med, mx

def reg(y, fim=None):
    y = y[:fim] if fim else y
    x = list(range(len(y))); n = len(x)
    if n < 4: return 0.0, 0.0
    mx_, my = st.mean(x), st.mean(y)
    sxx = sum((a-mx_)**2 for a in x)
    if sxx == 0: return 0.0, 0.0
    b = sum((a-mx_)*(c-my) for a, c in zip(x, y))/sxx
    res = [c-(my+b*(a-mx_)) for a, c in zip(x, y)]
    se = (sum(r*r for r in res)/(n-2)/sxx)**0.5
    return b, (abs(b/se) if se > 0 else 0.0)

raiz = os.path.join(os.path.dirname(__file__), '..')
runs = []
for p in sorted(glob.glob(os.path.join(raiz, 'telemetry', '*', 'run.log'))):
    med, mx = carrega(p)
    if len(med) < 5: continue
    runs.append((os.path.basename(os.path.dirname(p)), med, mx))

if not runs:
    print("nenhuma corrida com dados"); sys.exit(0)

print(f"\n{'corrida':<26} {'gens':>5} {'med10':>7} {'max10':>7} {'incl.mediana':>15} {'incl.máximo':>15} {'voltas':>7}")
print("-"*92)
for nome, med, mx in sorted(runs, key=lambda r: -st.mean(r[2][-10:])):
    b1, s1 = reg(med); b2, s2 = reg(mx)
    voltas = max(mx)/2928.0
    print(f"{nome:<26} {len(med):>5} {st.mean(med[-10:]):>6.0f}m {st.mean(mx[-10:]):>6.0f}m "
          f"{b1:>+9.3f} {s1:>4.1f}σ {b2:>+9.3f} {s2:>4.1f}σ {voltas:>6.2f}v")
print("-"*92)

print("\nTESTE DE JANELA ENCOLHIDA (inclinação do MÁXIMO, tirando as últimas N)")
print(f"{'corrida':<26} {'cheia':>10} {'-1':>10} {'-2':>10} {'-3':>10}  veredito")
print("-"*80)
for nome, med, mx in sorted(runs, key=lambda r: -st.mean(r[2][-10:])):
    n = len(mx)
    vals = [reg(mx)[0]] + [reg(mx, n-k)[0] for k in (1, 2, 3)]
    sig = reg(mx)[1]
    estavel = all(v*vals[0] > 0 for v in vals[1:]) and all(abs(v) > abs(vals[0])*0.5 for v in vals[1:])
    vd = "TENDÊNCIA REAL" if (sig > 2 and estavel) else ("SORTE DE JANELA" if sig > 2 else "ruído")
    print(f"{nome:<26} " + " ".join(f"{v:>+9.2f}" for v in vals) + f"  {vd}")
print("-"*80)
print("\nmed10/max10 = média dos ÚLTIMOS 10 (não infla com n). volta = 2928 m.")
