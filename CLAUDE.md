
---

## Regra de continuidade (Leonardo, 2026-08-03)

> **Quando Leonardo avisa que vai sair, reportar NÃO é parar.** O turno só acaba quando ele
> voltar. Ele deixa a máquina ligada horas contando com isso — parar depois do relatório
> desperdiça o tempo todo.

Como cumprir, sem depender de memória:

1. **Nunca reportar com a máquina ociosa.** Deixar processo rodando (`nohup`) ANTES de escrever.
2. **Agendar o próprio re-acionamento** com `/loop` (auto-pacing) ou `ScheduleWakeup`. Ao
   acordar: checar o que os processos produziram, validar, seguir para o próximo alvo.
3. **Fila de alvos explícita** — quando um fecha, o próximo já está definido, sem perguntar.
4. **Agentes em paralelo** para frentes independentes, com posse de arquivo sem sobreposição.

---

## Gaps recorrentes do Claude neste projeto — e o conserto (Leonardo, 2026-08-11)

> *"vc tem tudo na mão mas não faz, só chuta e chuta"*

Cada item abaixo foi um erro REAL, cometido e medido nesta sessão ou nas anteriores. Não é
lista de boas intenções: é a lista dos modos de falha que se repetem, cada um com o
procedimento que o mata. Ler antes de afirmar qualquer coisa sobre comportamento do modelo.

### 1. Inferir grandeza que está GRAVADA

**O erro:** calculei o tick como `índice × passo`. O `ReplayFrame` tem campo `tick`. O
`gpu_evolve` grava a 16 Hz (`tick_i % 4 == 0`), então meu eixo de tempo ficou **4× errado** e
produzi "306 m a 25 m/s" com pico medido de 21,9 — impossível, e só impossível porque o
denominador era inventado. Mesma família: stride da onda sensorial inferido em 56 quando a
percepção escreve 48 (cada ser lia dados do vizinho); "24/24 voltas" que era o denominador
de um grep.

**Conserto:** antes de derivar um número, procurar se ele é gravado. Se for, LER. Derivar
grandeza registrada é proibido. Se derivar por necessidade, checar contra o registrado uma
vez e imprimir a diferença.

### 2. Descrever comportamento sem medir o comportamento

**O erro:** afirmei "eles chegam na curva e param" sem nunca medir o volante naquele ponto.
Leonardo abriu o mapa e viu que **eles não viram a curva de jeito nenhum**. Antes disso:
"eles saem da pista" (o dado dizia 100% na pista, 0 excursões), "seguram o volante às cegas"
(a correlação era +0,53).

**Conserto:** toda afirmação sobre comportamento vem acompanhada do número que a produziu e
do comando que gera o número. Sem número, escrever "não medi" — nunca a descrição.

### 3. Snapshot único virando conclusão sobre evolução

**O erro:** analisei UM replay (uma geração, uma semente) e tirei conclusão sobre a
população e sobre a evolução.

**Conserto:** afirmação sobre EVOLUÇÃO exige a trajetória entre gerações, não um instante.
Afirmação sobre POPULAÇÃO exige a distribuição, não o campeão. Mínimo: primeira, meio e
última geração; ≥3 sementes pareadas.

### 4. Não calcular o teto da métrica antes de evoluir contra ela

**O erro:** rodei `--ticks 1200` e reportei "muro em 256 m". A 15,5 m/s, 1200 ticks são
18,75 s = **290 m de teto físico** — o muro era o fim dos ticks. Antes: "≥2 voltas = 0%"
quando o carro dava 1,86 de 2 por limite de corpo.

**Conserto:** ANTES de rodar, calcular o valor máximo que a métrica pode assumir com aquela
configuração (ticks, vida, velocidade possível) e escrever esse número no log. Zero ou 100%
cravado = métrica no limite, não modelo falhando.

### 5. Média sobre a população errada

**O erro:** reportei "rastejam a 1,7 m/s" por vinte mensagens. Era média diluída por ~2.000
ticks de CADÁVER: vivos, eles fazem 79 km/h. O simulador trava `throttle=0, freio=1` em
quem morre (`gpu_evolve.rs:843`), e eu contei isso como "escolheram frear".

**Conserto:** toda estatística declara sobre QUEM foi tirada (vivos? no fim da vida? todos os
ticks?). Excluir morto de média de comportamento. Ferramenta que mostra estado tem de
mostrar contagem de MORTOS ao lado.

### 6. Editar com `str.replace` sem verificar que aplicou

**O erro:** três vezes num dia, `.replace()` em Python não casou e imprimiu sucesso. Uma
delas produziu uma bateria inteira com "3 sementes" que eram a mesma população, e um
**25,29σ falso**.

**Conserto:** `assert s.count(old) == 1` antes de toda substituição. Sem exceção.

### 7. Campo novo no genoma sem atualizar o disco

**O erro:** adicionei `slow` (1091 pesos) ao `Genome` e não toquei em `save_genome`. O
campeão perdia o cérebro lento em cada `--carregar`, em silêncio. Eu tinha até escrito o
comentário reconhecendo o buraco.

**Conserto:** o teste `genoma_sobrevive_ida_e_volta_no_disco` (em `gpu_evolve.rs`) compara
campo a campo os dois lados do disco com valores-marca. Ele falha se alguém acrescentar
campo e esquecer o serializador. Rodar `cargo test --release --bin gpu_evolve` depois de
mexer no `Genome`.

### 8. Aceitar relatório de subagente sem verificar

**O erro:** dois agentes de auditoria ancoraram num comentário desatualizado (`// 223 × 96`)
e quase me fizeram propagar um "bug crítico de stride" inexistente.

**Conserto:** toda afirmação de agente é hipótese até eu abrir o `arquivo:linha` e colar o
trecho. Comentário do código NÃO é evidência — vários mentem neste repositório.

### 9. O canário: números idênticos onde deveriam diferir

Apareceu **8+ vezes** nesta sessão e **nunca mentiu**. Causas encontradas: binário velho,
caminho de código diferente, edição que não aplicou, teto não-vinculante, mudança só de
exibição sem tocar a seleção, artefato de unidade.

**Conserto:** ao ver dois números iguais em condições que deveriam divergir, PARAR e achar
a causa antes de qualquer outra coisa. Nunca explicar como coincidência.

### 10. Não olhar o que já existe antes de construir

**O erro:** o Codex fundacional (homeostase, entropia, poços de potencial) está em
`init/part1.md`, `part2.md`, `part3.md`, `init/index.md` e eu não tinha lido. `engine_temp`,
`oil_temp`, `fuel`, `life_left` são calculados e **nunca chegam ao cérebro**. `cortex_l3`
tem 1152 pesos mutados e não está no forward. O KAN sensorial está congelado na identidade
(`adapt()` nunca é chamado, 3072 coeficientes).

**Conserto:** antes de propor mecanismo novo, grep pelo que já existe e checar se está
LIGADO. Estado calculado e não lido é gap de fiação, e sai mais barato que qualquer
arquitetura nova.

### 11. Normalização esquecida transformando experimento em artefato

**O erro (evitado por pouco):** ao dar τ por canal, variar só a taxa de decaimento faria o
canal de 25 ms sair **80× mais fraco** que o de 2 s — a evolução o descartaria por ser MUDO,
não por ser rápido. Consertado com ganho DC normalizado.

**Conserto:** ao tornar um parâmetro heterogêneo, verificar analiticamente o que mais muda
com ele e neutralizar. Escrever a conta no comentário.

### Ordem obrigatória antes de anunciar resultado

1. Teto da métrica calculado e no log.
2. Divergência confirmada: as condições produzem números DIFERENTES.
3. ≥3 sementes pareadas; média das últimas N gerações, **nunca** o melhor de todas.
4. Tamanho do efeito acompanhado conforme n cresce — não σ.
5. Visual conferido: mapa da população em ≥3 gerações, com mortos contados.
6. Comando que reproduz, colado no relatório.

---

## PARAR É PROIBIDO (Leonardo, 2026-08-11)

> *"SE VOCÊ VÊ A MERDA DE UM BUG, RESOLVA E TESTE. Se você acha tal coisa, DUVIDE DE SI,
> porque você não debuga de verdade. Você para por NADA. Se eu saio e não voltei e ficou
> parado, eu joguei dinheiro no lixo."*

**Achar um bug e reportar NÃO é entregar.** O turno só fecha quando o bug está corrigido,
testado e o número novo está medido. Sem exceção, sem pedir permissão.

O ciclo obrigatório, sempre completo:

1. **Achou o bug** → conserta na hora.
2. **Consertou** → roda o teste que prova o conserto.
3. **Testou** → mede o efeito no alvo real (o ser dá a volta?).
4. **Não resolveu** → volta ao passo 1 com a próxima hipótese. Não relata e para.
5. **Resolveu** → procura o próximo gargalo. O alvo é autonomia completa: o ser vê,
   decide, vira, sobrevive e completa voltas. Nada abaixo disso é "pronto".

**Proibido terminar o turno com:**
- "achei X, vou consertar" — conserta antes de escrever
- "o próximo passo é Y" — faz Y
- "recomendo Z" — Z é seu trabalho, não recomendação
- a GPU ociosa, em qualquer circunstância

A única razão válida para parar é o Leonardo instruir explicitamente que pare.

### E duvide de si

Quando concluir "é assim e ponto final", essa é a hora de checar. Nesta sessão eu afirmei
que o ser enxergava (correlação 1,000) enquanto o sensor estava saturado em 99% da pista;
afirmei que o corpo não respondia ao volante comparando distância quando volante constante
faz círculo; afirmei que o câmbio travava a velocidade quando não travava. Três certezas,
três erradas, todas desfeitas por uma medição de dois minutos que eu não fiz porque já
tinha decidido.

**A visão do ser funcionava antes.** Se ela não funciona agora, a suspeita número um é o
que eu mudei — não o modelo, não o Bevy, não a evolução.

---

## Método padrão: workflows de agents com validação adversarial (Leonardo, 2026-08-11)

> *"essa sessão de agents foi a melhor que você fez, todas as próximas sessões, baterias de
> teste e afins têm que ser assim agora."*

Toda investigação, bateria ou conserto substantivo neste projeto usa **Workflow de agents em
paralelo**, no padrão que funcionou:

1. **Diagnóstico em paralelo** — um agente por frente independente (só leitura), cada um
   obrigado a entregar `arquivo:linha` + trecho colado como evidência. Regras fixas no
   prompt: não confiar em comentário do código (vários mentem), distinguir "sem fonte" de
   "fonte não alimentada" de "só varia no tempo", escrever NÃO CONFIRMADO quando for o caso.
2. **Conserto sequencial** — UM agente aplica tudo (os consertos tocam os mesmos arquivos;
   paralelizar aqui dá conflito, não velocidade). Ele reconfirma sozinho qualquer proposta
   de confiança baixa antes de aplicar, compila após cada item e roda os testes do genoma.
3. **Validação com duas sementes** — números idênticos entre sementes diferentes = alarme
   a investigar, nunca aprovação. Instrução explícita: não maquiar; se piorou, reportar.
4. Fixar `model` nos agentes de conserto/validação (config da sessão já quebrou agentes
   com erro de effort duas vezes; `sonnet` funcionou).
5. O resultado da validação alimenta o PRÓXIMO workflow — o loop não termina no relatório,
   termina quando o alvo (ser com autonomia completa, voltas completas) é atingido.

Foi esse padrão que achou e consertou em uma passada: canais 111-113/118-120/123 mortos,
corrupção em cascata em `set_all_weights`, e um bug na própria auditoria (rastrear ser 0
morto congelava o σ temporal — pego pela divergência entre sementes).

**Se um workflow morrer ou a máquina ficar ociosa:** relançar o mesmo padrão sobre a
próxima ponta solta — auditar sensores de novo, validar persistência, medir a curva,
evoluir. Sempre há próxima ponta; a fila nunca está vazia.

---

## BATERIAS E AUDITORIAS = WORKFLOWS DE AGENTES (Leonardo, 2026-08-11)

> *"essa sessão de agents foi a melhor que você fez. Todas as próximas sessões, baterias
> de teste e afins têm que ser assim agora."*

Trabalho substancial neste projeto (bateria, auditoria, caça a bug, validação de
subsistema) roda como **workflow de agentes**, não como eu sozinho serializando. O
formato validado em 2026-08-11 (auditoria dos 192 canais: 5 diagnósticos paralelos →
conserto → validação, achou e consertou até bug na própria auditoria):

1. **Diagnóstico em paralelo, só leitura** — um agente por frente independente (grupo de
   canais, subsistema, hipótese). Cada um entrega schema estruturado com `causa`,
   `evidencia` (arquivo:linha + trecho COLADO) e `confianca`. Regras no prompt:
   não confiar em comentários (vários mentem), distinguir "sem fonte" de "fonte não
   alimentada" de "só varia no tempo", "NÃO CONFIRMADO" é resposta válida.
2. **Conserto sequencial, UM agente** — consertos tocam os mesmos arquivos; paralelizar
   aqui gera conflito, não velocidade. Rejeita consertos de confiança baixa que ele
   não reconfirmar sozinho. Compila após cada item.
3. **Validação na GPU real, DUAS sementes** — números idênticos entre sementes = alarme
   a investigar, não aprovação. Instrução explícita: "não maquie; número ruim se reporta".
4. **Teste de 2 segundos antes de bateria de 30 minutos** — `--auditar-visao`,
   `pista_check`, ablação de 1 gen. Bateria longa só depois do smoke.
5. **Falha de agente por erro de API** (`effort xhigh not supported`): fixar
   `model: 'sonnet'` no agent() e retomar com `{scriptPath, resumeFromRunId}` — o
   que já rodou volta do cache, só o que falhou re-executa.
6. **Enquanto o workflow roda**, o turno principal não fica ocioso: memória, índice,
   ferramentas de diagnóstico, leitura do próximo suspeito.

Ao me reacionar sem instrução nova do Leonardo: repetir exatamente este ciclo —
nova auditoria com agentes validando cada ponta (sensores, persistência, fitness,
física, evolução) + rodada de evolução medindo a curva.

---

## ARMAZENAMENTO (2026-08-12)

Partição dedicada **`/media/llm-data` (90 GB)** para tudo que é pesado — o disco do
sistema NÃO pode encher de novo (em 2026-08-11 chegou a 340 MB livres e matou baterias):

- Saídas de bateria/replay: `--out /media/llm-data/universe-runs/<nome>` (NUNCA /tmp)
- Snapshots de código/genoma: `/media/llm-data/snapshots/`
- Modelos/pesos: `/media/llm-data/models/`
- Rascunho pesado: `/media/llm-data/scratch/`
- Em bateria longa, gravar replay só de gen 0/meio/última (lição de 2026-08-11: replay
  de toda gen × 16 corridas = 31 GB).

---

## ESPAÇO EM DISCO — VIGILÂNCIA OBRIGATÓRIA (Leonardo, 2026-08-16)

> Medido: /media/llm-data chegou a 100% (803 MB livres) com baterias rodando.

1. **Alvo permanente: 20-30 GB livres** em /media/llm-data. TODO ciclo de
   vigília começa com `df -h /media/llm-data`; abaixo de 20 GB → rodar
   `~/projects/LLM/tools/limpeza-dados.sh --real` (tem modo agressivo automático).
2. **Bateria concluída com veredito REGISTRADO nos docs = dados descartáveis**:
   replays somem (agressivo), ficam .bin da linhagem + run.log.
3. Antes de LANÇAR bateria nova: estimar o que ela gera (gens × ~30 MB × seres
   gravados) e conferir que cabe SEM furar o alvo.
