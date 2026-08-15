# Regiões cerebrais por evolução — tese e escada de experimentos

> Leonardo (2026-08-15): "evoluir com base na ideia do cérebro: achar algo que separe
> onde cada coisa precisa ir — neurônio de reflexo em tal parte, de memória em outra,
> de visão em outra — mas TODOS fazem algo. E o modelo precisa continuar dando as
> voltas facilmente."

## O princípio: separar por PRESSÃO, não por decreto

Não desenhamos o mapa funcional à mão (isso seria arquitetura, não evolução). Damos ao
substrato uma razão FÍSICA para que neurônios com funções parecidas fiquem juntos — e
medimos se o mapa emerge. É como o córtex real se organiza: fiação custa energia, e
minimizar fio agrupa quem conversa muito (Chklovskii; bate com a Lei do Garimpo e com
o achado "corpo dita arquitetura" do N1: corr(log J, k\*)=+0,972).

## Mecanismos candidatos (do mais emergente ao mais dirigido)

### M1 — Custo de fiação por distância (o principal)
Cada neurônio ganha POSIÇÃO fixa num anel/grade (L1: índice = posição; entradas: posição
pela família do canal — visão num arco, corpo noutro, derivados noutro). Custo metabólico
por tick ∝ λ·Σ|w|·dist(pré, pós), pago em VIDA (moeda cruzada — nunca na própria fitness).
Peso forte em conexão longa vira caro → a evolução puxa cada neurônio para "perto" (em
conectividade) do que ele escuta. Reflexo (visão→motor curto) e memória (τ longo, poucas
entradas) se separam sozinhos — SE a tese estiver certa.
- Predição M1: com λ>0, a modularidade Q da matriz L1 sobe vs λ=0, e [regioes] passa a
  achar folhas (>0,30) SEM cair a completagem de voltas.
- Armadilha conhecida: "pressão exige tensão" (neuromod exp00 vs exp02) — λ alto demais
  sem tarefa que exija capacidade colapsa em vez de organizar. Varrer λ em U.

### M2 — Regiões de τ semeadas
Já temos banco de τ por neurônio (P1.6). Semear blocos com priors de τ distintos
(bloco-reflexo: τ 0,1-0,5 s; bloco-memória: τ 2-15 s; bloco-visão: τ livre) e deixar a
mutação log-normal refinar. Mede se FUNÇÃO segue TEMPO (a tese do banco de τ: 40× de
seleção por TER escalas variadas).

### M3 — Máscara de conectividade herdável por bloco
L1 em 4-6 blocos; cada bloco tem máscara evoluível de QUAIS famílias de entrada enxerga.
Mais dirigido (nós definimos os blocos), mas dá o "onde cada coisa vai" imediato e é
barato de medir. Serve de teto de referência para M1 (o emergente tem que chegar perto
do dirigido para valer).

### M4 — Neuromodulação por região (enxerto doc-09)
O cérebro lento (hoje ocioso no treino) ganha canais que gateiam REGIÕES em vez de
neurônios: um canal de "alerta" module o bloco-reflexo, um de "contexto" o bloco-memória.
Entra depois que M1-M3 derem um mapa; é a ponte com a linha neuromod/EXP10.

## Escada (cada degrau com A/B ≥2 sementes, modelo não pode regredir de voltas)

1. **E-REG0 (métrica primeiro)**: calcular modularidade Q e mapa de comunidades da L1 do
   campeão ATUAL (código novo em src/analysis/regioes.rs) — baseline honesto do quanto
   de estrutura JÁ existe antes de qualquer pressão.
2. **E-REG1**: M1 com λ ∈ {0, baixo, médio} × 2 sementes × 40 gens, partindo do campeão.
   Aceite: Q sobe, [regioes:L1/L2] acham folhas, Σ≥1V/gen ≥ 80% do controle λ=0.
3. **E-REG2**: M2 sozinho vs M1+M2 (tempo ajuda o espaço?).
4. **E-REG3**: M3 como teto de referência; comparar mapas emergentes vs dirigidos.
5. **E-REG4**: M4 gateando o mapa vencedor.

## Regras herdadas que se aplicam

- Física nova SEMPRE atrás de flag (padrão off) — N baterias em paralelo comparando.
- Predição registrada ANTES de cada bateria; negativo documentado com mecanismo.
- Números idênticos entre sementes = alarme; validar com replay/mapa, não só log.
- Custos em moeda CRUZADA (vida/tempo), nunca desconto direto na fitness.

## E-REG0 — resultado (2026-08-15)

Modularidade da L1 do campeão 6951 m (k-means na assinatura de entrada + Q de
comunidade, 20 nulos por permutação de colunas): **Q ≈ −0,005 em k=2..8, idêntico
ao nulo (−0,0049 ± 0,0000)**. O cérebro atual NÃO tem regiões — cada neurônio
escuta mistura estatisticamente igual (consistente com PR 6,7/96 distribuído e
ablação 96→64 que derruba 576→19 m). Ponto de partida limpo para M1: qualquer
Q acima do nulo sob custo de fiação é emergência real. Ferramenta: o cálculo
vira função em src/analysis quando E-REG1 entrar (por ora, protocolo neste doc).

## E-REG1 — lançado (2026-08-15)

`--custo-fiacao λ` implementado: posição do neurônio j/96, canal c/192, dist
circular, custo = λ×média(|w|·d)/média(|w|)×2 ∈ [0,λ] vida/tick, recalculado por
geração. Bateria: λ=0 (controle pareado) vs λ=0,5, seed 31, 40 gens, partindo do
campeão rpm-evolucao (config idêntica fora o λ). PREDIÇÃO REGISTRADA: com λ=0,5
a modularidade Q da L1 (protocolo E-REG0) sobe acima do nulo (−0,005) no campeão
final, e Σ≥1V/gen fica ≥80% do controle. Se Q não subir: a pressão está fraca
(subir λ) ou a tese M1 falha aqui — medir antes de concluir.

## E-REG1 — VEREDITO: NEGATIVO com mecanismo (2026-08-15)

Q(λ0)=−0,0052±0,0007 · Q(λ0,5)=−0,0057±0,0001 — idênticos ao nulo. E a métrica
DIRETA da pressão não moveu: dist ponderada 0,2491 vs 0,2493 (acaso=0,25).
Mecanismo: o custo é média sobre 18.432 pesos → variância entre indivíduos ≈ 0 →
imposto CONSTANTE, que não seleciona (mesma lei do controle de Landauer:
imposto do mesmo tamanho mas constante não desloca o ótimo). Completagem: λ0,5
manteve 94% do controle — o custo era inócuo nos dois eixos.

**E-REG1b lançado**: `--fiacao-cauda 8` — só as 8 conexões mais FORTES de cada
neurônio pagam fio (a cauda é onde genomas diferem; Lei do Garimpo). 2 sementes,
mesma partida/config. Predição: dist ponderada da CAUDA cai vs 0,25 e Q sobe do
nulo; se a dist da cauda cair sem Q subir, a organização é local-sem-módulos
(informação nova de qualquer forma).

## E-REG1b — VEREDITO: NEGATIVO replicado (2026-08-15)

dist CAUDA top-8: s31 0,2489 (controle 0,2464) · s77 0,2471 (partida 0,2414) —
não caiu em nenhuma semente (se algo, subiu); Q segue no nulo. Mecanismo final
de M1: o diferencial de custo entre indivíduos (~0,01 vida/tick mesmo na cauda)
é ordens de grandeza menor que as diferenças de fitness por pilotagem — a
seleção não enxerga o fio. M1 (custo em VIDA) morto neste regime, 2 desenhos ×
2-3 braços. Alternativa não testada (fila): viés MUTACIONAL (mutação que
encurta fio é maior/mais provável — age na geração de variação, não na seleção).

## E-REG M3 — lançado: regiões DIRIGIDAS como teto de referência

`--mascara-regioes`: 4 blocos de 24 neurônios (visão 0-15+128-191 · parede/
estrada 16-63 · corpo 64-95 · associativo=tudo), peso fora da máscara = 0
reimposto pós-mutação e no load. Campeão mascarado PERDE ~2/3 das conexões na
largada — o experimento é a RECONSTRUÇÃO: 60 gens, 2 sementes. Predições:
(a) recupera ≥70% da completagem do controle até a gen 60; (b) [regioes:L1]
passa a achar folhas fortes por bloco; (c) se recuperar, região é viável e o
problema de M1 era só a moeda da pressão.

## Maratona-v3 (contexto) + M3 parcial + E-REG2 lançado (2026-08-15)

- Maratona-v3 FECHOU: perfis em produção sem canário (100 promoções/semente;
  s77 32,9 voltas-duplas/gen). Pistas curta/curvas/retão aprovadas.
- M3 aos 43/60: SEMENTES OPOSTAS — s31 mal reconstrói (med 1895, Σ≥1V=15);
  s77 reconstrói forte (26,7/gen, 520 ≥2V, máx 7980 m ≈ 70% do controle e
  subindo). Leitura parcial: cérebro regionalizado é VIÁVEL; a travessia do
  vale pós-amputação tem variância enorme entre sementes. Veredito na gen 60.
- E-REG2 lançado (`--tau-semeado`): τ por bloco (visão 0,5-4 s · REFLEXO
  parede/estrada 0,3-0,8 s · corpo 1-4 s · associativo 4-15 s), 2 sementes ×
  60 gens, mesma partida. Predições: (a) performance não regride >20% vs
  partida (τ re-otimiza rápido); (b) [regioes] mostra especialização por bloco
  maior que o histórico 0-1/48; (c) se blocos convergirem de volta ao τ
  uniforme ~2 s, "função segue tempo" é FALSA aqui.

## M3 — VEREDITO FINAL: POSITIVO (s77) com risco de travessia (2026-08-15)

s77: recuperação 73% do controle (29,2/gen, 842 ≥2V, máx 7980 m) e a predição-
chave confirmada — [regioes:L1] 36/96 folhas (histórico: 16/96), índices
AGRUPADOS nos blocos (25-39=parede/estrada, 48-63=corpo, 75-95=associativo).
Primeira demonstração de cérebro com regiões funcionais no Universe.
s31: presa no vale (0,9/gen, 3 promoções) — travessia pós-amputação é
vale-dependente; mitigação futura: máscara GRADUAL (atenuar fora-da-região ao
longo de N gens em vez de amputar na gen 0).

## E-REG2 parcial (41/60): função-segue-tempo VIVA

Performance ~90% do controle; estrutura temporal SOBREVIVE (espalha 10,6× vs
3× histórico; p10 0,38 s reflexo · p90 4,0 s) — não convergiu de volta.

## M2+M3 COMBINADO lançado (ereg-m23-completo)

Máscara de regiões + τ por bloco no MESMO cérebro — o desenho-alvo do Leonardo
("reflexo em tal parte, memória em outra, todos fazem algo"). 80 gens × 2
sementes. Predições: (a) recuperação ≥ M3-s77 (τ certo por região ajuda a
travessia); (b) folhas ≥ 36/96; (c) alarme se as duas sementes divergirem
tanto quanto no M3 — travessia continua sendo o gargalo do método.
