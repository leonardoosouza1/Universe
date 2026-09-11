# Estudo — o cérebro do neuromod-lab dentro do Universe (11/09/2026)

Pergunta do dono: pegar a parte principal do neuromod-lab (tálamo, córtex, hipocampo, reflexo, camadas rápida/lenta), usar no Universe e comparar o cérebro atual da pista com o cérebro novo. Este documento diz o que já existe, o que é novo de verdade, e como comparar sem se enganar. Fatos com `arquivo:linha` (levantamento Opus, 11/09, sobre `race-world-v3`); o resto é decisão.

## 1. O cérebro atual da pista (o que se treina de verdade)

Quatro caminhos coexistem; **o que se treina é `BrainGpu`** (`src/gpu/brain/mod.rs:185`, kernel `src/gpu/brain.wgsl`), espelho do `SovereignEngine` (`src/engine.rs:195-260`) mais mecanismos só da GPU. Entrada: 192 canais i8 (`src/config.rs:314`, registro em `src/perception/channels.rs:159-400`). Saída: acelerador, direção, freio (`src/engine.rs:552-560`), viram física em `src/bin/gpu_evolve/context.rs:1862-1907` **no host** (KAN de saída, agressão/teto/viés do lento, alisamento).

| bloco | o que é | genes |
|---|---|---|
| L1 192→96, L2 96→48 (+ tanh, ReLU por neurônio) | rápida | 18 432 + 4 608 |
| CfC por canal com τ por neurônio (`Genome::tau`, `evolution.rs:117`) | banco de τ, herdável, log-mutado | 96 |
| slow brain CfC64→16→3 a cada 8 ticks (`engine.rs:43-50`) | lenta; modula agressão, teto, viés | 1 091 |
| M3 `--mascara-regioes`: 4 blocos de 24 (visão / parede-estrada / corpo / associativo) (`evolution.rs:731-757`) | regiões | máscara |
| M4 `--neuromod-regioes`: lento → 4 gates por bloco com reserva (`context.rs:1600-1624`, `brain.wgsl:871`) | modulação por região | 16 |
| gate softmax córtex/reflexo/sentinela (`src/gate.rs`), reflexo 3×(32→1) (`src/specialist.rs`), sentinela por regra | Trinity | 51 + esparso |
| memória/período/limiar/ativação por neurônio (`neuron_meta`, 480 f32/ser) | P1.7 | 480 |

Campeão do laboratório: **m23-s77** (M2+M3), 7 032 m mediana, 2,40 voltas, melhor avaliação pura do projeto (`docs/REGIOES_CEREBRAIS.md`).

**Diagnóstico do próprio Universe (`docs/ROADMAP.md:11-20`)**: o ser não sente o corpo (combustível, dano, temperatura calculados e fora dos 192 canais, salvo `ch117 OwnFuel` com body-block); a neuromodulação não tem o que reger (M4 empatou); um piloto por fórmula fechada bate o campeão (7 139 m vs 6 592 m). Treino longo proibido até liberação do dono.

## 2. Bloco a bloco: o que do neuromod-lab já existe, o que é novo

| bloco (neuromod-lab, `completo.py`) | no Universe | veredito |
|---|---|---|
| **Rápida** (4 ReLU por opção) | L1/L2 densas, ReLU por neurônio | **existe, mais forte**. Não portar |
| **Lenta** (3 EMA, τ ~5/25/100) | CfC64 lento + banco de τ por neurônio (M2: τ por bloco sobrevive 60 gens, espalha 8,8×) | **existe, mais forte**. Não portar |
| **Reflexo** (espécie lenta de combustível + portão) | `SpecialistBrain` só no caminho CPU/Bevy; sentinela por regra; não há reflexo no `BrainGpu` | existe no cérebro errado; sem combustível que acabe não há o que refletir |
| **Córtex por distrito** (Δ de 4 pesos por contexto) | contexto discreto pronto (24 setores, `statics.rs:128`; curvatura classificada ch 78-82/101/111-113), mas o setor **nunca entra no forward** | parcialmente novo; a L2+KAN já condicionam por curvatura vista — o ganho seria só onde a visão não alcança |
| **Hipocampo** (memória por contexto, escrita por evento, EMA com τ, lida bilinear) | eventos existem (setor `main.rs:3055`, volta `:3085`, grama `context.rs:937`, dano `:1949`, seco `:1727`); **memória indexada por setor não existe** | **NOVO** |
| **Tálamo** (portão por GRUPO de sentidos × necessidade) | M4 gateia blocos de **neurônios** pelo lento; famílias de canais definidas (`neuro.rs:171-182`); necessidade não entra em gate nenhum | **NOVO na forma** (entrada × necessidade); sem estado lento no corpo, empata — já medido dos dois lados |
| **Custo por gene / por ativação, `dof`, `metab`** | custo por sensor e por combustível; custo de fiação (M1) reprovado 2× (`evolution.rs:731-733`); nada por ativação | **NOVO** (método) |
| **Ablação por bloco** | `Via` enum, madura, só CPU/Bevy (`neuro.rs:70-124`); GPU só `slow_ablated` | portar a ablação para o `BrainGpu` |
| **Portão de identidade (genes 0 ⇒ igual)** | existe como contrato de init (`evolution.rs:975-982`) + `--sem-*` | existe; falta o **portão de sensibilidade** (`neuromod-lab/src/testes_sentidos.py`: cada entrada nova consegue mudar a saída?) |

Resumo: **não se porta o `SerCompleto`**. Dois blocos são novos (hipocampo por setor, tálamo por necessidade) e um método é novo (custo/dof, sensibilidade, pareado por semente com MDE, ablação "memória apagada na avaliação"). O Astra estava certo em separar mecanismo de adaptação; o que ele não tinha visto é quanto já existe.

## 3. Por que o hipocampo por setor pode pagar aqui (e onde eu estava errado)

Eu disse que "memória na pista já deu nulo". Errado em parte: os 5 mecanismos que falharam eram memória de pesos de LLM (`iara-semente`), e o M4 mediu **modulação**, não memória episódica. O que nunca foi testado: memória **escrita numa volta e lida na seguinte**.

Mecanismo que a justifica: a percepção não alcança a curva em alta velocidade (horizonte de percepção: acima de ~35 m/s nenhuma sonda chega a tempo; `docs`/memória "d_freio"). Um ser que faz ≥ 2 voltas (mediana 1,8 em 20 000 ticks, máx 2,6) tem, na volta 2, informação que os sensores não dão: **o que aconteceu neste setor e no próximo na volta anterior** (velocidade média, saiu na grama, bateu, secou). É informação fora do MDP observável — o teorema "memória inútil se observável" não se aplica.

Predição falsificável: com hipocampo por setor, a distância da volta 2 em relação à volta 1 melhora mais que no controle; apagar a memória na avaliação (`--hipocampo-apagar`) devolve o ser ao controle. Se o campeão com memória não perde nada ao apagá-la, a memória é decoração — e isso também é resultado.

## 4. Por que o tálamo por necessidade NÃO deve ser o primeiro

Os dois laboratórios chegaram ao mesmo veredito por caminhos independentes: sem estado lento no corpo que mude o ótimo (combustível que acaba, dano oculto), modulação empata (Universe M4 16/08; neuromod-lab preds 165/179/180; EXP10). O tálamo só ganha alvo com `--interocepcao` (M1 do ROADMAP: ch114-117/98/124) **e** energia escassa (rodada 3, nunca rodada: `--fuel-tank 0,25` + refuel por setor). Ordem: hipocampo primeiro (não depende do corpo), tálamo depois do corpo chegar ao cérebro.

## 5. Como comparar "cérebro atual × cérebro novo" sem se enganar

1. **Mesma partida**: os dois braços partem do campeão m23-s77 (`--retomar`, com a correção `5c1ceab` — o reload que carregava população aleatória foi pago em 25/08).
2. **Pareado por semente**: 3 sementes × 2 braços, mesma pista por geração (rotação `[8005, 8200, 9001]`, `pistas.rs:22-28`), 20 000 ticks (para haver volta 2), 60 gerações. Diferença por semente, t, MDE — como no lab (`docs/REVISAO-CHECKLIST.md`).
3. **Nascimento neutro**: genes novos em zero ⇒ o braço novo na geração 0 é bit a bit o controle (contrato `evolution.rs:975-982`; teste obrigatório).
4. **Sensibilidade**: antes da GPU, perturbar só os genes do hipocampo e provar que a saída muda (`testes_sentidos` do lab, portado).
5. **Ablação na avaliação**: campeão novo avaliado com e sem memória (`--hipocampo-apagar`), 3 pistas, 48 cópias.
6. **Métrica principal**: distância por volta (nova telemetria `tempo_por_volta`), completagem ≥1V/≥2V por geração, `net_max`; secundária: `dof` ligado, contribuição do bloco no comando (|bias| médio), mortes.
7. **Portão de mundo** (lição do táxi): o prêmio para memória só existe onde há volta 2. Se a mediana ficar < 1,5 voltas na configuração, subir ticks antes de concluir.

## 6. Decisões do dono

1. Liberar treino na GPU para a bateria pareada (hoje proibido, `ROADMAP.md:90-92`). A GPU é compartilhada com a fila `nmd` do neuromod-lab (regra: um processo na GPU) — a bateria entra pela fila, via wrapper.
2. Aceitar a ordem: C0 telemetria/ablação → C1 hipocampo por setor → bateria → só depois C2 interocepção + tálamo (+ energia rodada 3).
3. O `SerCompleto` inteiro não é portado; o táxi (PRD do `taxi-world`) segue como o mundo onde ele é testado.
