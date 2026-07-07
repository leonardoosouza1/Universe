# IARA-Vida — cérebro cognitivo de um "ser" que realmente pensa

> Branch `iara-vida` (a partir de race-world-v3). Objetivo do Leonardo: sair do bicho reativo que anda em
> círculo; construir um **ser com cognição real** — que sente necessidades, planeja, tem rotina, aprende a
> sobreviver — reusando TODA a base neural do Universe (BitLinear ternário, CfC, DenseLayer, evolução, DNA,
> neuromodulação). Dois alvos: **headless** (eu rodo/valido/evoluo) e **UI browser glass-box** (ver TUDO do cérebro).

## Por que o bicho antigo andava em círculo
Quimiotaxia reativa pura = seguir 1 gradiente = sem estado interno, sem conflito de metas → órbita.
A inteligência **emerge de conflito de necessidades + memória + rotina + previsão**. É isso que projetamos.

## O SER (glass-box)
Estado corporal: `energia`, `hidratação`, `vigor(stamina)`, `saúde`, `idade`.
Neuromoduladores (estilo `Metabolism` do Universe): `dopamina`(RPE), `cortisol`(medo), `curiosidade`(surpresa=erro de previsão), `adenosina`(fadiga).
Morre por: fome (energia→0), sede (hidratação→0), predador (saúde→0), velhice.

### Percepção (≈20 entradas i8, glass-box)
- Exterocepção: comida (esq/dir), água (esq/dir), predador (esq/dir + proximidade), ninho (dir+dist),
  vizinhos (densidade + alinhamento = **senso coletivo**).
- Interocepção (o ser sente a si mesmo): fome, sede, fadiga, medo, saúde.
- Contexto temporal: fase dia/noite (sin/cos), viés.

### Cérebro (recorrente + neuromodulado + preditivo — camadas REAIS do Universe)
```
senses(f32) → i8
  → CfC_perceptual (memória de percepção, dim-preserving)
  → BitLinear L1  (córtex instinto, i8→f32, tanh)         [fast brain ternário]
  → i8 → CfC_recorrente (MEMÓRIA DE TRABALHO — o estado que faz pensar)
  → DenseLayer L2 (associação, relu)
  → ganho neuromodulado (dopamina↑exploração, cortisol↑defesa)
  → cabeça de AÇÃO  (DenseLayer → [virar, acelerar, comer, beber, descansar, sinalizar])
  → cabeça de MUNDO (DenseLayer → prevê próxima percepção) → surpresa → curiosidade
```
Ação NÃO é só mover: o ser **escolhe** comer/beber/descansar/fugir/sinalizar via gates → precisa DECIDIR
prioridade (defeat "círculo"). Sinalizar = vocalização → vizinhos captam (inteligência coletiva).

### Genoma (DNA evoluível, quantizável i8 como o Universe)
Todos os pesos (L1/L2/ação/mundo) + decays dos CfC + setpoints de drive + morfologia. ~870 f32.

## MUNDO
Toroidal 2D. Recursos SEPARADOS em lugares diferentes (comida ≠ água ≠ ninho) → obriga alternar metas.
Predadores caçam. **Ciclo dia/noite**: à noite fora do ninho drena mais + predador mais forte + visão cai
→ **ROTINA emerge** (forragear/beber de dia, abrigar de noite). Currículo de dificuldade: escassez,
nº predadores, duração da noite. Obstáculos opcionais.

## EVOLUÇÃO (esquema REAL do Universe)
GA: tournament k=3 + crossover em blocos + gaussian σ=0.05 + decay 0.97^gen + elitismo. Salva campeão em `.dna`.
Fitness = w1·sobrevivência + w2·prole + w3·satisfação-média-de-necessidades + w4·rotina + w5·acerto-de-previsão − morte.

## MÉTRICAS DE INTELIGÊNCIA (não só "viveu")
- tempo de vida, prole, satisfação de necessidades.
- **repertório**: entropia da distribuição de ações (usa comer/beber/descansar/fugir na hora certa?).
- **adequação**: correlação necessidade↔ação (bebe quando com sede? = "sabe o que fazer").
- **rotina**: autocorrelação de visitas ao ninho no período do dia.
- **cognição**: acurácia do world-model (prevê o próximo passo).
- decodificador de "pensamento": clusteriza estados recorrentes → rotula pela ação seguinte → intenção atual.

## ENTREGAS
1. `src/bin/vida.rs` (headless): `--evolve`, `--record <dna>`, `--serve` (HTTP live), `--watch` (ASCII).
2. `vida/lab.html` (UI glass-box): mapa com playback, dashboard de necessidades+neuromod, **diagrama neural
   completo ao vivo**, heatmap da memória CfC no tempo, world-model (previsto×real+surpresa), ethograma de
   ações, curvas de evolução, sub-domínios (nichos por k-means), decodificador de intenção.
3. `--serve` liga a UI ao Rust real → **testar variações ao vivo** (perturbar cérebro, mudar mundo, spawnar).

## INOVAÇÃO (fora da curva)
Ser recorrente **neuromodulado + preditivo (curiosidade intrínseca)** com **interocepção** e **senso coletivo**,
evoluído, e **100% glass-box**: dá pra ver sensor→memória→decisão→previsão neurônio a neurônio, e ler a
"intenção" decodificando a memória. Headless valida números; UI mostra a mente por dentro.
