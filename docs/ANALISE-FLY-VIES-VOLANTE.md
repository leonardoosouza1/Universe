# A bateria da mosca: o embaralhado não dirige melhor, o mapa real dirige em círculo

Figura: `fly-vies-volante.png`. Fontes: `fly-battery-20260914/{real_plus,real_minus,shuffle_plus}_s{31,32,33}/`
(`population_gen0000.json`, `telemetry.csv`) e `fly-gain-check-20260914/real_plus_s{31,32,33}/`.
Protocolo: `FLY-BATTERY-2026-09-14.md`. Resultado relatado: `FLY-BATTERY-RESULTS-2026-09-14.md`.

**Nota de leitura minha:** comecei analisando os diretórios `f3_/f4_/f5_/f8_*`, que são a
escada exploratória anterior e todos usam `seed=31`. A bateria registrada é
`fly-battery-20260914/`, com as três sementes. O prefixo `f8_` colidir com a "F8 anterior"
citada no protocolo custou-me uma análise inteira; vale renomear as pastas exploratórias.

---

## 1. O número relatado confere

Recalculei o endpoint primário (`progress_m`, mediana da população de 40 seres) direto dos
`population_gen0000.json`:

| semente | mapa real, reflexo + | mapa real, reflexo − | embaralhado, reflexo + | real+ − embaralhado |
|---|---:|---:|---:|---:|
| 31 | 63.16 | 62.87 | 147.19 | **−84.03** |
| 32 | 38.85 | 46.40 | 201.16 | **−162.31** |
| 33 | 53.22 | 33.61 | 136.93 | **−83.71** |

Média **−110.02 m** — idêntico ao relatado. O mapa embaralhado avança mais que o mapa real
nas 3 sementes, por um fator que varia bastante entre elas: **2.33× (31), 5.18× (32),
2.57× (33)** — faixa 2.3–5.2×, média das razões 3.4×.

## 2. A decomposição: o embaralhado não anda mais, ele dirige melhor

A distância percorrida é praticamente a mesma nos dois braços (400 contra 392 m na semente
31; 399 contra 485; 414 contra 413). Então a diferença não é motor nem acelerador. É
conversão:

```
eficiência = avanço na pista / distância percorrida
  mapa real:      15.9 %   11.3 %   12.5 %
  embaralhado:    37.4 %   42.0 %   34.3 %
  razão:           2.35×    3.71×    2.75×      (faixa 2.3–3.7×, média 2.9×)
```

## 3. O mecanismo, medido: um viés de 4 % faz o carro circular

No sinal cru do volante, amostrado a cada 64 tiques com `tick ≥ 64`:

| | semente 31 | 32 | 33 |
|---|---:|---:|---:|
| média, mapa real | **−0.0370** | **−0.0407** | **−0.0400** |
| média, embaralhado | −0.0002 | +0.0032 | −0.0096 |
| desvio padrão, real | 0.988 | 0.987 | 0.987 |
| desvio padrão, embaralhado | 0.987 | 0.986 | 0.987 |

**Os desvios são idênticos. A única diferença entre os dois braços é um deslocamento de
base de ≈ −0.04 — 4 % do próprio desvio do sinal.** E um comando de volante constante
integra em curvatura:

```
voltas de rumo completadas na vida (mediana por ser)
  mapa real:     0.64   0.63   0.77
  embaralhado:   0.10   0.10   0.10
```

O ser do mapa real gira dois terços de uma volta completa ao longo de 400 m percorridos.
Ele dirige em círculo. O embaralhado vai quase reto. É isso, e só isso, que a eficiência
de 12 % contra 37 % está medindo.

## 4. Por que o controle embaralhado é enviesado A FAVOR do embaralhado

Isto não é azar de uma semente: é consequência de como o controle é construído. A leitura
do reflexo é uma soma ponderada sobre o mapa. Embaralhar o mapa transforma essa soma numa
soma de contribuições randomizadas, cujo valor esperado vai a **zero** pelo teorema central
do limite. **Embaralhar não destrói apenas a estrutura — destrói também o viés de base.**

Logo o contraste real-contra-embaralhado, como está desenhado, compara
*estrutura + viés* contra *nenhuma estrutura + nenhum viés*, e não consegue separar as duas
coisas. Não é um problema que mais sementes resolvam, nem mais mapas embaralhados: é do
desenho. E ele está sistematicamente contra o mapa real.

## 5. O follow-up de ganho que vocês já rodaram confirma o mecanismo — e mostra o conserto certo

Ganho 20 → 1, mesmo executável, mapa real:

| semente | voltas de rumo | eficiência | avanço | viés no volante aplicado |
|---|---|---|---|---|
| 31 | 0.64 → **0.39** | 15.9 → 16.9 % | 63 → 64 m | −0.0394 → −0.0390 |
| 32 | 0.63 → **0.43** | 11.3 → 20.5 % | 39 → 74 m | −0.0413 → −0.0273 |
| 33 | 0.77 → **0.47** | 12.5 → 18.9 % | 53 → 73 m | −0.0506 → −0.0398 |

Menos ganho, menos círculo, mais eficiência — **3 de 3 na mesma direção**, média −0.25
volta e +5.5 pp. O mecanismo está confirmado por um experimento que já existe.

**Mas o ganho fecha só metade do buraco**: com ganho 1 o mapa real ainda faz 0.43 volta e
18.8 % de eficiência, contra 0.10 e ~38 % do embaralhado. A razão está na última coluna —
**o viés no volante aplicado quase não muda**, porque a normalização adaptativa reescala
depois do ganho e preserva a proporção do deslocamento. Baixar o ganho reduz a amplitude;
não remove a base.

**O conserto é centrar, não amortecer:** subtrair a média corrente do sinal cru **nos dois
braços** antes do ganho, e igualar a escala (hoje o desvio do volante aplicado é 0.508 no
real e 0.599 no embaralhado — a normalização não iguala nem média nem escala).

(Observação de coluna: a saturação de 96.6 % relatada é do *input adaptado*, a montante;
no `applied_steer` do `telemetry.csv` a saturação é 0 % nos dois ganhos. São grandezas
diferentes, não um desacordo.)

## 6. Predição registrada, para antes de rodar

Com o sinal cru centrado e escalado igual nos dois braços, mesmo executável, mesmas três
sementes: **a diferença real-menos-embaralhado em avanço sai de −110 m para dentro de
±30 m.** Se ela continuar fortemente negativa depois de centrar e escalar, aí sim é um
achado sobre estrutura — e um achado forte, porque significa que a fiação real é pior que
o próprio embaralhado com as duas bases iguais.

## 7. O que este documento NÃO diz

Não diz que o conectoma da mosca dirige bem. Diz que **esta bateria não testou isso**: ela
mediu um deslocamento de base. Continuam valendo integralmente as ressalvas do próprio
protocolo — um único mapa embaralhado, uma geração (nenhuma inferência de aprendizado), o
defeito compartilhado dos bigodes diagonais no executável congelado, e o limiar de 1.55 m
como proxy de sobreposição e não impulso de colisão medido.

## 8. Uma recomendação de endpoint

Trocar o endpoint primário de **avanço** para **eficiência = avanço / distância**. Motivos:
separa "andou" de "andou na direção certa", que é a distinção que este resultado inteiro
dependeu; é adimensional, logo comparável entre mundos e entre pistas; e sai de graça dos
arquivos que já estão sendo escritos. O avanço continua como secundário.
