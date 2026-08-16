# O Universe além do carro — brainstorm e plano de expansão

> Leonardo (2026-08-16): "essa é a ideia do Universe: é o universo, eu crio qualquer
> coisa lá e o modelo tem que se adaptar. Simular o quê mais? Outras frentes,
> competitividade de sobrevivência. Evoluir o projeto para algo grande, inovador e
> absurdo." + "como o cérebro separa as coisas hoje é meio codado; só vale investir
> em neuroevolução da separação se o treino passar por MAIS coisas."

## A tese unificadora

O que temos hoje NÃO é um simulador de corrida — é um **substrato**: mundo em grade
com física + percepção 192 canais + cérebro com regiões/relógios + evolução com
currículo por mérito + física com custos (RPM, vida, fio). O carro é o PRIMEIRO
corpo. A jogada grande: **manter o substrato e trocar o corpo/mundo/tarefa** — e a
medida de inteligência vira TRANSFERÊNCIA: o cérebro que aprendeu num mundo se
readapta mais rápido no próximo? (Já medimos a versão pequena disso: campeão
travado colapsa com caixa aberta; campeão regionalizado readapta.)

E a resposta ao "cérebro separado na mão": **a separação só emerge sozinha quando
o mundo EXIGE múltiplas competências** — foi exatamente o negativo do E-REG1 (uma
tarefa só → sem pressão pra modularidade; a seleção é surda) e é o consenso da
neuro (córtex se especializa porque corpo/mundo têm ESTRUTURA multi-tarefa). Ou
seja: o multi-mundo abaixo não é um extra — é O caminho para a máscara deixar de
ser codada. Ordem: mundos primeiro, emergência depois, medida sempre.

## Frentes de simulação (mesmo substrato, ordenadas por custo de entrada)

### F-A. Corrida-sobrevivência competitiva (custo: BAIXO — é o mundo atual + regras)
- **Energia finita** ("modelo vai ter X de energia"): FUEL_ON já implementado e
  DORMENTE (tanque nunca acaba em 8k ticks). Ligar com tanque apertado + reabastece
  por volta/setor → a volta vira sobrevivência literal (era assim nas branches que
  completavam voltas!). Primeira bateria da expansão — mecanismo pronto, só dose.
- **Recurso ESCASSO compartilhado**: N seres, M < N "postos" de energia por volta
  → chegar primeiro importa → competição SEM contato. Emergem estratégia de ritmo
  e bloqueio de linha.
- **Clima/μ variável**: atrito da pista muda por trecho/geração (chuva, gelo) —
  já existe slip/aderência na física; o pneu (ch70) vira sensor CRÍTICO.
- **Pistas com desafio real**: perfis novos (mista = retão+grampos; funil =
  largura variável; "ilhas" de grama no meio) — gerador paramétrico já aceita.

### F-B. Presa e predador (custo: MÉDIO — 2 populações no mesmo mundo)
- Mundo aberto (Living World Plan JÁ EXISTE em LIVING_WORLD_PLAN.md: modo --vida,
  comida/água/predadores!). Corrida de armamentos: presas evoluem fuga/atenção,
  predadores evoluem perseguição/emboscada. É o gerador de currículo AUTOMÁTICO
  perfeito — o adversário evolui junto, a dificuldade nunca satura (a pista
  "nunca vista" passa a ser o OUTRO).
- Métrica: gerações de coexistência, ciclos de Lotka-Volterra, complexidade de
  trajetória de fuga. As regiões do cérebro ganham papéis óbvios (visão=detecção,
  corpo=fôlego, associativo=antecipação).

### F-C. Forrageio com corpo interoceptivo (custo: MÉDIO — é o EXP10 do neuromod!)
- Fome/sede/temperatura/dano como variáveis LENTAS acopladas (a spec do corpo
  mínimo já está em ~/Downloads/neuromod/docs/07). Dia/noite: visão degrada à
  noite → memória espacial paga. É o mundo onde M4 (maestro+reserva) tem teto
  ALTO comprovado na pesquisa (G=25,6%).

### F-D. Multi-corpo / corpo evolutivo (custo: MÉDIO-ALTO)
- Sensores como genes JÁ existem (FASE SENSORES). Expandir: alcance de retina,
  nº de whiskers, massa/downforce como genes com CUSTO energético — o trade-off
  corpo×cérebro vira evolutivo (N1: corpo dita arquitetura, corr +0,972).
- Corpos distintos no MESMO mundo: carro rápido-frágil vs tanque lento-robusto —
  nichos emergem.

### F-E. Transferência entre mundos (custo: BAIXO de rodar, ALTO de valor)
- O protocolo: campeão do mundo A entra no mundo B (mesmo cérebro, corpo/tarefa
  novos) vs do-zero. Curva de readaptação = medida de GENERALIDADE do cérebro.
  Liga direto na tese-mãe IARA (inteligência = adaptação por sobrevivência, não
  skill fixa). Cada frente nova alimenta esta métrica.

### F-F. Neuroevolução da SEPARAÇÃO (o degrau final — só depois do multi-mundo)
- Trocar máscara FIXA por máscara HERDÁVEL (genes de conectividade por bloco) +
  neurônios com posição herdável. Predição registrável: no mundo multi-tarefa
  (A+B+C), máscaras herdáveis convergem pra separação; no mono-tarefa, não
  (replicaria E-REG1 como controle). Se confirmar: separação emergiu de pressão
  real — a resposta à tua pergunta "como o humano faz".

## Escada de execução proposta (cada degrau com A/B e predição antes)

1. **AGORA** (mecanismo pronto): energia finita — FUEL_ON com tanque apertado,
   refuel por setor/volta, 2 sementes × controle. [+ correção da regressão de
   parede dos regionalizados: braço com wall-drain.]
2. Pista MISTA (retão+grampo) + μ variável por trecho — o pneu/slip viram
   sensores que pagam.
3. Recurso escasso compartilhado (competição sem contato) — 1ª frente COMPETITIVA.
4. Forrageio interoceptivo (EXP10 real) — onde o maestro M4 tem teto alto.
5. Presa-predador (2 populações) — currículo automático infinito.
6. Transferência A↔B como métrica padrão de TODA bateria nova.
7. Máscara herdável (F-F) quando ≥3 mundos estiverem vivos.

## O que NÃO fazer (lições já pagas)
- Não abrir tudo de uma vez (teto6: corpo além da percepção colapsa; degrau a degrau).
- Não custo sem variância interindividual (E-REG1) nem pressão sem tensão (neuromod).
- Física nova SEMPRE atrás de flag com braço de controle.
- Cada mundo novo entra com: teto da métrica calculado, canários no analisador,
  pista_check-equivalente (validador de mundo), e replay/observabilidade DESDE o dia 1.
