#!/usr/bin/env python3
"""Analisador de bateria p/ Claude+agents (Leonardo 2026-08-15: "muito dado").

Uso: analisar-bateria.py <dir-do-run> [<dir2> ...]   (ou --todos)
Saída: veredito compacto por bateria + CANÁRIOS automáticos (as armadilhas que
já nos morderam, ver Universe/CLAUDE.md "Gaps recorrentes"):
  C1 setores×metros: linha oficial diz voltas mas [setores] diz 0 reais (pista quebrada)
  C2 clones: mediana == máximo por várias gens (população degenerada/determinística)
  C3 vida colapsada: vida média <60% por 3+ gens seguidas
  C4 pista presa: mesma pista por 15+ gens com completantes ≥ limiar (promoção travada)
  C5 t/gen inflado: última gen >1.5× a mediana (contenção de GPU ou vazamento)
  C6 pares idênticos: entre dois runs passados juntos, med idêntica em ≥5 gens (seed morto)
"""
import re, sys, os, json, glob

def parse(dirpath):
    p = os.path.join(dirpath, "run.log")
    if not os.path.exists(p): return None
    L = open(p, errors="replace").read()
    gens = [dict(g=int(a), med=int(b), mx=int(c), v1=int(d), v2=int(e))
            for a,b,c,d,e in re.findall(r"^gen\s+(\d+) \| med=\s*(\d+)m .*max=\s*(\d+)m .*≥1V=(\d+) ≥2V=(\d+)", L, re.M)]
    vidas = [int(x) for x in re.findall(r"vida média (\d+)%", L)]
    tg = [float(x) for x in re.findall(r"\| (\d+\.\d)s$", L, re.M)]
    pistas = re.findall(r"usa seed=(\d+)", L)
    promo = re.findall(r"pista (\d+) → (\d+)", L)
    reais = [tuple(map(int, m)) for m in re.findall(r"voltas reais: (\d+) seres ≥1, (\d+)", L)]
    picos = [float(x) for x in re.findall(r"pico (\d+\.\d) m/s", L)]
    fim = "=== FIM" in L
    return dict(run=os.path.basename(dirpath.rstrip("/")), gens=gens, vidas=vidas,
                tg=tg, pistas=pistas, promo=promo, reais=reais, picos=picos, fim=fim)

def canarios(d):
    alarmes = []
    gens = d["gens"]
    if not gens: return ["SEM GENS PARSEADAS"]
    # C1: oficial diz ≥1V alto mas setores dizem 0 reais na mesma janela
    if d["reais"]:
        ult = d["reais"][-min(5, len(d["reais"])):]
        v1_ult = [g["v1"] for g in gens[-len(ult):]]
        if max(v1_ult) >= 10 and max(r[0] for r in ult) == 0:
            alarmes.append("C1 setores×metros: ≥1V alto com 0 voltas REAIS — pista/setor quebrado")
    # C2: clones
    iguais = sum(1 for g in gens if g["med"] == g["mx"])
    if iguais >= 3: alarmes.append(f"C2 clones: mediana==máximo em {iguais} gens")
    # C3: vida colapsada
    seq = 0
    for v in d["vidas"]:
        seq = seq + 1 if v < 60 else 0
        if seq >= 3: alarmes.append("C3 vida média <60% por 3+ gens"); break
    # C4: pista presa com mérito
    if d["pistas"]:
        ult_pista = d["pistas"][-1]
        preso = sum(1 for p in d["pistas"][-15:] if p == ult_pista)
        v1_med = sorted(g["v1"] for g in gens[-10:])[len(gens[-10:])//2] if len(gens) >= 3 else 0
        if preso >= 15 and v1_med >= 12:
            alarmes.append(f"C4 pista {ult_pista} presa 15+ gens com v1 mediano {v1_med} (promoção travada?)")
    # C5: t/gen inflado
    if len(d["tg"]) >= 6:
        med_t = sorted(d["tg"])[len(d["tg"])//2]
        if d["tg"][-1] > 1.5 * med_t:
            alarmes.append(f"C5 t/gen {d['tg'][-1]:.0f}s vs mediana {med_t:.0f}s (contenção?)")
    return alarmes

def veredito(d):
    gens = d["gens"]
    ult5 = gens[-5:]
    pico = max((p for p in d["picos"]), default=0)
    mx_global = max((g["mx"] for g in gens), default=0)
    s = (f"{d['run']}: {'FIM' if d['fim'] else f'{len(gens)} gens'} | "
         f"máx {mx_global}m | últimas5 med {[g['med'] for g in ult5]} | "
         f"Σ≥1V {sum(g['v1'] for g in gens)} ({sum(g['v1'] for g in gens)/max(len(gens),1):.1f}/gen) "
         f"Σ≥2V {sum(g['v2'] for g in gens)} | pico {pico:.1f} m/s | "
         f"promoções {len(d['promo'])} | vida últ5 {d['vidas'][-5:]}")
    return s

def main():
    args = sys.argv[1:]
    if "--todos" in args:
        dirs = sorted(glob.glob("/media/llm-data/universe-runs/*/"))
    else:
        dirs = args
    pares = {}
    for dd in dirs:
        d = parse(dd)
        if not d: continue
        print(veredito(d))
        for a in canarios(d): print(f"  ⚠ {a}")
        # C6: agrupar por prefixo-sN p/ detectar pares idênticos
        base = re.sub(r"-s\d+$", "", d["run"])
        pares.setdefault(base, []).append(d)
    for base, ds in pares.items():
        if len(ds) >= 2:
            meds = [tuple(g["med"] for g in d["gens"]) for d in ds]
            n = min(len(m) for m in meds)
            if n >= 5 and len(set(m[:n] for m in meds)) == 1:
                print(f"  ⚠ C6 {base}: sementes com trajetória IDÊNTICA em {n} gens — seed morto")

if __name__ == "__main__":
    main()
