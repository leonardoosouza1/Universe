---
name: medir-cerebro
description: Radiografia completa do cérebro de um campeão — pesos, ativação, topologia e o teste de "evoluiu ou teve sorte". Use quando alguém pedir para analisar um genoma, comparar gerações, ou antes/depois de mudar arquitetura do cérebro.
---

# Medir o cérebro

Duas leituras que se cobrem, e nunca uma só:

- **pesos parados** (`genoma_view`) — o que o cérebro *é*, sem simular nada
- **ativação viva** (`gpu_evolve --ativacao`) — o que ele *faz* enquanto dirige

Peso grande que nunca dispara é capacidade desperdiçada; neurônio que dispara sempre é
viés, não sensor. Só o cruzamento das duas responde.

## 1. Anatomia dos pesos

```bash
./target/release/genoma_view <campeao.bin> --top 20
./target/release/genoma_view <campeao.bin> --html /media/llm-data/scratch/cerebro.html
```

Entrega distribuição por camada com histograma, orçamento de atenção por canal (com nome
real, via registro `Channel`), perfil por neurônio (surdos, mudos, influentes) e banco de τ.
O `--html` desenha o grafo L1→L2, o mapa de grupos e as barras de atenção.

## 2. Ativação durante a corrida

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
env -u DISPLAY ./target/release/gpu_evolve --seres 48 --gens 1 --ticks 20000 \
  --ativacao --ativacao-n 16 --painel --out /media/llm-data/universe-runs/<nome>
```

Bloco `[ativacao]` por camada: % mortos, % saturados, raio espectral, rank efetivo,
entropia da política, alarmes. Painel em `localhost:3200` mostra os neurônios disparando
ao vivo, com a média da geração embaixo de cada um.

## 3. O teste que separa evolução de sorte

**Compare o campeão com o genoma da geração 0 da mesma corrida.** Sem isso, "distribuição
gaussiana" parece "não aprendeu nada" — e gaussiana sobrevive a perturbação pequena, então
a aparência engana.

```python
import struct, statistics as st
def ler(p, n=18432):
    b = open(p,'rb').read(); off = 8 if b[:4]==b'UGEN' else 0
    return struct.unpack_from('<%df' % n, b, off)
a, z = ler('evolve_best_gen0000.bin'), ler('evolve_best_gen0029.bin')
cos = sum(x*y for x,y in zip(a,z)) / ((sum(x*x for x in a)**.5) * (sum(y*y for y in z)**.5))
print('cosseno', round(cos,4), '| mudaram >0.01:',
      sum(1 for x,y in zip(a,z) if abs(x-y)>0.01), '/', len(a))
```

Leitura: **cosseno ≥0,99 = a evolução só empurrou o sorteio inicial** (medido em
2026-08-13: 0,9925 depois de 30 gerações, 64% dos pesos intocados). Cosseno mais baixo com
desempenho estável é construção real.

## Números de referência (2026-08-13, campeão de 1 volta)

| | valor | limiar |
|---|---|---|
| cosseno gen0→gen29 | 0,9925 | <0,95 seria construção |
| raio espectral L1 | 6,98 | 1,0 |
| L2 saturada | 57,5% | 30% |
| entropia da política | 0,346 nats | 0,50 |
| rank efetivo L1 | 75/96 | — |
| atenção do canal top | 0,66% | uniforme = 0,52% |

## Armadilhas já pagas aqui

- **Ausência de medida disfarçada de medida.** L2 imprimia `raio_espectral=0.000
  rank=0/48`, que lê como colapso; era `w_rec = None`. Hoje imprime `n/a`. Ao ver zero
  exato, pergunte se foi medido.
- **`--retomar` mistura o campeão com genoma aleatório (70/30)** — a população recarregada
  rendeu 1.076 m contra 4.223 m do original. Não serve para reavaliar o campeão puro.
- **Variância entre seres no tick 0 é tautologia** com `--same-start`: todos estão no mesmo
  estado. Propriocepção só varia no tempo; a auditoria mede os dois eixos separados por isso.
