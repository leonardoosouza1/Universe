# Cérebro Virtual — plano-mestre para replicar a arquitetura do cérebro humano no Universe

Escrito em 2026-08-14. Caso de estudo real, não brincadeira: o objetivo de longo prazo é
um cérebro virtualizado com regiões funcionais, células de suporte e **economia
energética** — o ser aprende a se economizar porque pensar custa.

Este documento é o mapa de anos, não de uma sessão. Cada fase tem critério de entrada
(gatilho), critério de validação e critério de não-regressão. A regra que governa tudo:

> **A necessidade justifica a estrutura.** Região nova só entra quando a medição mostrar
> que a estrutura atual saturou. Construir córtex pré-frontal para um ser que ainda não
> precisa planejar é complexidade sem prova — e já medimos o custo disso (cortex_l3:
> 1.152 pesos mutados por 100+ gerações, zero efeito, puro ruído no genoma).

---

## Por que isso não é fantasia — o que JÁ medimos que sustenta o plano

O projeto tem seis resultados experimentais próprios que apontam exatamente para a
arquitetura do cérebro biológico. Não vamos copiar o cérebro por estética; vamos porque
os nossos números pedem as mesmas soluções que a evolução biológica encontrou:

1. **Custo energético desloca o ótimo cognitivo** (EXP Landauer, E4): quando lembrar
   custa energia, o ponto ótimo de memória cai de 1,27 para 0,25 bits — e o controle
   (imposto constante do mesmo tamanho) NÃO desloca. Cérebro caro vira lento E pobre.
   → É a base da Fase 4 (metabolismo): custo por spike muda o comportamento, não só o
   tamanho.
2. **Neuromodulação só paga com corpo acoplado** (Tese do Corpo, corr +0,970 com o teto
   analítico): sem estados internos lentos que mudam o ótimo, modulador é decoração.
   → O corpo (Fase 2 do plano de curto prazo) é pré-requisito do cérebro, não acessório.
3. **O corpo dita a arquitetura** (N1, corr +0,972): com corpo obrigatório, 83% do
   orçamento neural vai ao reflexo e o córtex encolhe — o elefante gasta 98% na tromba.
   → As proporções entre regiões não serão desenhadas à mão: emergem do orçamento.
4. **Mais energia compra cognição mais rica** (N3, corr +0,960): energia→bits com
   controle λ=0 travado. → O gasto energético virtual é o botão que regula quanta
   inteligência o sistema pode sustentar.
5. **Banco de escalas de tempo é selecionado** (40× vs 5,3× do controle): TER relógios
   variados paga; trocá-los com o regime não. → O τ log-espaçado já é nosso "tronco
   cerebral + córtex" em miniatura temporal.
6. **Moeda cruzada** (EXP10): ação protetora tem de custar em moeda DIFERENTE da que
   protege, senão política degenerada domina. → Orçamentos energéticos por região
   precisam de moedas separadas (glicose motora ≠ glicose cognitiva).

E a referência biológica que orienta as proporções (pesquisa consolidada em
`~/.claude/.../reference_neurociencia_cerebro.md`): 86 bi neurônios + 84 bi glia (1:1);
**cerebelo tem 80% dos neurônônios em 10% da massa** (coordenação fina é barata e massiva);
cognição correlaciona com neurônios corticais ABSOLUTOS (humano 16 bi > elefante 5,6 bi
apesar do cérebro 3× menor); o gargalo evolutivo é ENERGIA.

---

## Estado atual (o ponto de partida honesto)

- Cérebro: L1 96 CfC + L2 48 + lento (16→3) + KAN saída. ~24k pesos.
- Diagnóstico medido: campeão = 99,25% sorteio inicial (cosseno 0,9925); L2 saturada
  57,5%; entropia da política 0,346; raio espectral 7,0 (regime amplificador).
- O ser: dá voltas numa pista fácil sem buracos, 10,5 m/s, obediência 0,74.
- Infra: painel tempo-real, métricas de ativação, formato .brain (bloco ativável),
  auditoria 3-eixos, visao_check, genoma_view, docmcp.

---

## AS FASES

### Fase 0 — Fundação estável (em curso; P0-P1 do PLANO_EVOLUCAO.md)

Pré-requisitos de engenharia para qualquer coisa maior:

- [x] tick rápido (1,83× — meta 3×) sem readback por tick
- [ ] equivalência estatística pós-P0 (3 sementes, em validação agora)
- [ ] anti-decoreba: sementes rotativas + medida de generalização
- [ ] init espectral: nascer com |λ_max|≈1 (hoje 7,0) — matemática fechada, sem treino;
      candidato a maior ganho barato do projeto
- [ ] .brain ligado ao gpu_evolve (hoje o formato existe mas grava UGEN v2)

**Validação:** ser generaliza para pista nunca vista com razão ≥0,5; não-regressão ≥80%
na pista de treino.

### Fase 1 — Corpo com custo (P2 do plano de curto prazo)

Grip que satura, pneu com temperatura, motor/óleo, combustível. Cada estado interno
lento acoplado à decisão. É o que a Tese do Corpo exige ANTES de cérebro maior.

**Validação:** ablação por item; ganho da neuromodulação medido contra o teto analítico
G (forma fechada já derivada no projeto). Não-regressão ≥80% por item.

### Fase 2 — Metabolismo neural: pensar custa

A virtualização do gasto energético que você pediu, em três degraus:

1. **Custo por spike:** cada ativação |h|>limiar debita de um orçamento energético do
   ser (moeda SEPARADA da vida — lei da moeda cruzada). Esparsidade deixa de ser
   estética e vira sobrevivência. Já medimos MLPs byte com sparsity 95-99% — o teto
   biológico (~1-4% de neurônios ativos) é alcançável.
2. **Custo de manutenção por peso:** peso |w|>0 paga aluguel por geração (Lei do
   Garimpo/RIC como imposto, não só como critério de poda). O genoma encolhe sozinho
   quando a competência couber em menos — é o "inteligência enxuga" da visão IARA.
3. **Recompensa por economia:** fitness ganha termo de eficiência (metros por joule
   virtual), com peso pequeno para não dominar a corrida.

**Validação:** curva energia×desempenho reproduz o formato de N3 (mais orçamento → mais
bits → mais metros); ablação do custo → esparsidade desaba (controle). O ser deve
descobrir sozinho coasting/traçado econômico.

### Fase 3 — Regiões: o córtex se divide

Gatilho: L2 saturada + entropia baixa persistindo após Fases 1-2 (já há indício hoje).

Divisão por FUNÇÃO, com pressão seletiva própria e orçamento energético próprio
(proporções iniciais informadas por N1 — o orçamento decide, não o design):

| região virtual | análogo | papel no Universe |
|---|---|---|
| córtex visual | occipital | digestão da retina (o retina_digest já é o embrião) |
| cerebelo | cerebelo | micro-correção motora de alta frequência, barata e massiva (80% dos neurônios em 10% da massa — MUITOS neurônios pequenos e rasos) |
| córtex motor | M1 | volante/pedal a partir de intenção, não de pixels |
| pré-frontal | PFC | horizonte longo: gerir energia, escolher risco, planejar ultrapassagem (só entra na Fase 5 quando houver o que planejar) |
| tronco/reflexo | tronco | ré, anti-colisão, batimento — nunca desligável |

Formato `.brain` já suporta: um bloco por região, bit ATIVO por bloco, encoding por
bloco. Ablação de região vira flag, não cirurgia.

**Validação:** ablação diferencial (tirar cerebelo deve degradar micro-correção e NÃO a
escolha de traçado; tirar visual cega; etc.). Se ablação de uma região não tem efeito
dissociável, a região é decoração e sai.

### Fase 4 — Glia: as células que não disparam

Metade do cérebro real não é neurônio. Papéis virtualizáveis com mecanismo que já temos:

- **astrócito** = gestor de orçamento energético local (distribui joules por região,
  reforça sinapses muito usadas — nosso "água/choque" do IARA V2 é isso);
- **microglia** = poda ativa: remove peso morto usando os critérios já implementados
  (PR saturado + Garimpo) — hoje órfãos em brain_capacity.rs, aqui viram órgão;
- **mielina** = cache/latência: canal muito usado ganha τ mais rápido (o custo é pago em
  energia de manutenção — de novo moeda cruzada).

**Validação:** com glia ativa, mesma competência com menos joules/pesos que sem (pareado,
3 sementes). Se não economizar, glia é overhead e volta.

### Fase 5 — Ciclo completo: sono, consolidação, competição

- **Sono/replay:** entre gerações, replay hipocampal dos trechos de maior erro
  (dopamina). Já provamos internalização por replay no IARA Pesos Vivos (retenção 71%).
- **Pré-frontal ligado:** decisões de longo prazo (economizar agora para ultrapassar
  depois) — precisa de adversário, então:
- **Competição** (P5): outro ser na pista. Teoria dos jogos de verdade, com custo
  energético de cada manobra.

**Validação final da visão:** um ser que dá voltas em pista nunca vista, com derrapagem
e obstáculos, gastando MENOS energia por metro a cada geração — competência crescendo
enquanto o custo cai. Esse é o fenômeno "cérebro humano": não é ser grande, é ser
**eficiente sob orçamento**.

---

## Regras transversais (valem em toda fase)

1. Tudo mensurável: cada mecanismo novo nasce com sua métrica no painel e no [ativacao].
2. Não-regressão ≥80% ou a mudança volta e entra fatiada.
3. Ablação obrigatória: mecanismo sem efeito dissociável é removido.
4. ≥3 sementes pareadas, média das últimas 5 gens, estatística sobre vivos.
5. Duvidar de número bonito: canário de números idênticos, teto da métrica antes de rodar.
6. O orçamento decide as proporções entre regiões — não o design (lição N1).

## Vigília 24h

A skill `vigilia` (.claude/skills/vigilia) define o ciclo de auto-acionamento:
validar → decidir → agir → re-agendar. Enquanto houver fase incompleta, sempre há
próxima ação; a GPU não fica ociosa.
