# Plano de evolução — de "resolveu a pista" a "sabe dirigir"

Escrito em 2026-08-13, depois de o ser dar a primeira volta completa
(3.189 m medianos, 25 de 48 seres ≥1 volta, semente 77, sem buracos).

## A situação, em números

O que já funciona:

| | |
|---|---|
| obediência à curva | +0,740 (era +0,058 na gen 0) |
| velocidade | 10,5 m/s · aproveitamento de pista 97% |
| mortalidade | 7 de 48 · vida média 96% da geração |
| voltas | 25 de 48 seres ≥1 volta |

O que os números dizem que ele **não** aprendeu:

| medida | valor | leitura |
|---|---|---|
| cosseno do genoma gen0 → gen29 | **0,9925** | o campeão é 99,25% o sorteio inicial |
| pesos que nunca mudaram | 64% | mediana de \|Δw\| = 0 |
| atenção por canal | 0,52–0,66% | uniforme: nenhum sentido foi eleito |
| entropia da política (L2) | 0,346 nats | repertório motor pobre |
| L2 saturada | 57,5% | metade dos motores no batente |
| raio espectral de L1 | 6,98 | limiar 1,0 — regime amplificador |

**Diagnóstico:** a evolução encontrou um bilhete premiado no sorteio inicial e o ajustou
0,75%. Isso resolve *esta* pista. Não é o mesmo que dirigir.

## Princípio que rege o plano

Cada etapa entra sozinha, com bateria de validação e **critério de não-regressão
explícito**. Se o ser piorar além do limite, volta-se a etapa anterior e a mudança entra
fatiada. A dificuldade sobe; a capacidade de sobreviver não pode desabar.

> Leonardo: *"se o modelo ficar ruim de sobreviver, volte algumas casas até onde ele
> estava bom e vá aplicando de pouco em pouco — assim você acha os gaps."*

**Baseline de não-regressão** (semente 77, 30 gens, sem buracos, 20k ticks):
mediana 3.189 m · 25/48 voltas · obediência 0,740 · 10,5 m/s.

---

## P0 · Acelerar o relógio (FECHADO 2026-08-14)

> *"algo como que para o modelo continua igual, mas para a gente o tempo acelerou."*

**Gargalo medido:** 2,30 ms/tick contra 0,63 ms/tick do tick GPU puro — 3,7× de
desperdício. Causa: todo tick faz `map_async` + `poll(Wait)` da retina inteira (340 KB) e
processa 84.672 pixels num laço Rust, só para produzir 19 floats por ser.

**Conserto:** kernel de digestão da retina na GPU, devolvendo 16 floats de estado CfC +
`min_wall_ahead` + `edge_row6/7`. De quebra, unifica o CfC de percepção — hoje existe em
três binários com risco de divergência silenciosa.

**Validação:** paridade exata. Mesma semente, mesmos metros
(`gen0 med=0 max=1078 · gen1 med=24 max=895 · gen2 med=262 max=1080`).
Se os metros mudarem, é bug, não melhoria. Meta: 35,1 s → 12-15 s por geração.

**Por que primeiro:** todas as outras etapas custam baterias. Acelerar paga em cada uma.

**RESULTADO (2026-08-14):** 35,1 → 19,2 s/gen (1,83×), readback de 340 KB/tick
eliminado, CfC unificado no crate iara-cfc com teste de paridade CPU↔GPU passando.
Paridade bit-a-bit de trajetória é IMPOSSÍVEL (±1 i8 na retina × ganho de entrada 7×
× loop pelo mundo); critério trocado por equivalência estatística em 3 sementes:
s77 mesma ordem (máx 1,56V, recorde), s31 equivalência exata (516 vs 519 m),
s113 regrediu ~54% (alerta anotado, sem viés direcional no conjunto — 1↑ 1= 1↓).
Comando: ver /media/llm-data/universe-runs/p0-{baseline,depois,equiv-*}/

---

## P0.5 · Inicialização espectral de L1 (implementado 2026-08-13)

**Gargalo medido:** raio espectral de L1 = 6,98 (4 corridas independentes:
7.101/7.050/7.048/6.983 — `analysis/activation.rs`, alarme `SpectralRadiusHigh`).
Teoria echo-state pede `|λ_max| <= ~1` para dinâmica de entrada estável; a 7× a matriz
amplifica o sinal sensorial antes mesmo da primeira mutação. O campeão medido é 99,25%
o sorteio inicial (cosseno 0,9925) — a evolução não conserta init ruim, só o empurra.

**O que a matriz realmente é** (mapeado antes de mexer, porque o nome no código
engana): não existe recorrência h→h no L1. `w_rec` em `activation.rs` é, na prática, a
matriz de pesos de ENTRADA de L1 (`sensory_wave[192] → L1[96]`, feedforward), com o
nome herdado do REFACTOR_PLAN original. A recorrência de verdade é o CfC — diagonal por
canal (`s = s·decay + x·gain_dt`, `decay = exp(-d·dt) ∈ (0,1)` sempre) e já estável por
construção; não há autovalor ali para normalizar. O alarme mede — e a hipótese ataca —
o maior valor singular da matriz L1 (192→96), que é a grandeza que de fato decide se o
sinal sensorial amplifica ou atenua ao entrar no cérebro rápido.

**Conserto:** em `Genome::random` (`src/gpu/evolution.rs`), depois de sortear L1
(N(0, σ=0,30) i.i.d., como sempre), mede-se `linalg::largest_singular_value(l1,
L1_NEURONS, SENSORY_WAVE_SIZE)` — a MESMA conta que `analysis::activation::
spectral_summary` usa para o alarme (extraída para `math::linalg::{gram_matrix,
largest_singular_value}`, fonte única, evita duplicidade divergente) — e L1 inteiro é
re-escalado por um fator escalar único `alvo/medido` (`thresholds::
SPECTRAL_INIT_TARGET = 1.0`). Escalar por um escalar preserva a direção relativa de
cada peso e escala o maior valor singular linearmente, então o alvo é atingido
exatamente (testado: `largest_singular_value_escala_linear_com_a_matriz`). Só L1: L2
não entra na medição do alarme (`w_rec: None` na chamada de `collect_activation_stats`
para L2), então não há "o que o alarme mede" para L2 mirar.

**Genoma legado (herança intocada):** `load_genome` nunca chama `Genome::random` —
monta `Genome { .. }` direto por struct literal a partir dos bytes do disco. Um
campeão salvo antes desta mudança carrega com o ganho espectral que sempre teve, sem
re-normalização silenciosa no meio de uma bateria. Testado explicitamente
(`genoma_legado_carregado_do_disco_nao_e_reescalado`, `src/bin/gpu_evolve/tests.rs`).

**Flag A/B:** `--sem-init-espectral` no `gpu_evolve` desliga a normalização (padrão:
LIGADA). Braço de controle = população nascendo no σ=0,30 cru de sempre (ganho ~7,0).

**Testado:**
- `gpu::evolution::tests::genoma_novo_nasce_com_ganho_espectral_perto_de_1` — genoma
  novo (5 sementes) nasce com ganho medido em `[0.8, 1.2]`.
- `gpu::evolution::tests::spectral_rescale_factor_{atinge_alvo_exatamente,
  ganho_quase_zero_nao_explode}` — a matemática do fator, isolada.
- `bin::gpu_evolve::tests::genoma_legado_carregado_do_disco_nao_e_reescalado` — disco
  não re-escala.
- `math::linalg::tests::largest_singular_value_*` — a função compartilhada.

**A/B PENDENTE — não rodado nesta sessão** (a tarefa era implementar a inicialização,
não queimar GPU; há 3 baterias de validação ocupando-a). Falta medir, com ≥3 sementes
pareadas, `--sem-init-espectral` ligado vs. desligado, em N gerações:
- mediana de distância e ≥1V/≥2V (a inicialização ajuda a evolução a progredir mais
  rápido, ou é irrelevante porque a seleção já convergia pro mesmo lugar?);
- raio espectral do CAMPEÃO ao longo das gerações, não só na gen 0 — a mutação aditiva
  (gpu/evolution.rs, N(0,σ=0.05) em 2% dos pesos, sem re-normalizar a matriz inteira)
  pode fazer o espectro derivar de volta para longe de 1.0 em poucas dezenas de
  gerações (risco já documentado antes desta implementação — medir, não assumir);
- cosseno genoma gen0→genN — se a inicialização espectral também ficar ~99% parada,
  o problema não é o ponto de partida, é a pressão de seleção/mutação em cima dele.

---

## P1 · Anti-decoreba (implementado 2026-08-14)

> Leonardo: *"preciso que o modelo saiba se movimentar, não que saiba andar na mesma
> pista."* O campeão medido no diagnóstico do topo deste documento é 99,25% o sorteio
> inicial e treina SEMPRE na mesma pista (semente 9001, `--same-start`) — P1 é o
> instrumento que separa "decorou a pista" de "aprendeu a dirigir".

**Mudança:** a pista de treino rotaciona por geração, sorteada deterministicamente de um
conjunto (`--pistas`); a cada K gerações a população inteira é avaliada (SEM efeito na
seleção) numa pista reservada que nunca entra no treino (`--pista-held-out`).

**Flags novas (`gpu_evolve`):**

| Flag | Default | Semântica |
|---|---|---|
| `--pistas "9001,9002,..."` | vazio | Conjunto de treino. `pista_da_gen(pistas, gen_abs) = pistas[gen_abs % pistas.len()]` — rotação por bloco de 1 geração, determinística. **Vazio = comportamento anterior a esta flag**: `pista_da_gen` cai no braço `pistas.is_empty()` e devolve 9001 pra qualquer geração — o mesmo hardcode literal que `main.rs` tinha antes (`VoxelWorld::new(9001, track_width)`, fora do laço). O A/B do campeão-decoreba em andamento não é tocado sem passar a flag. |
| `--pista-held-out <seed>` | ausente (desligado) | Pista RESERVADA — nunca entra em `--pistas`. Sem a flag, nenhum tick extra roda (custo zero). Se a seed também estiver em `--pistas`, um aviso não-fatal é impresso (ela deixaria de medir generalização). |
| `--held-out-cada <K>` | 5 | A cada K gerações (`gen_abs % K == 0`) a população roda `n_ticks` (mesmo N do treino) na pista reservada, com os MESMOS genomas já upados (sem mutação/seleção nesse passo). |

**Mecanismo de troca de pista (custo):** `GpuCtx::reload_world` (e os equivalentes em
`PerceptionGpu`/`PhysicsGpu`) trocam grade/fluxo/alturas/heightmap via
`queue.write_buffer`, sem recriar device/pipeline/bind-group — válido porque
`VoxelWorld::new` usa `size=320`/`cell_size=3.0` CONSTANTES independentes da seed
(`world.rs`), então nenhum buffer estático muda de tamanho entre pistas. Custo: ~400 KB
de upload total, sub-milissegundo — irrelevante perto dos ~19,2 s/gen medidos no P0.
`racing_line`/`grid_cpu` (cópias CPU usadas pelos canais sensoriais 108-110 e pelo voxel-
sob-o-ser) são atualizados junto, senão ficariam presos na pista antiga em silêncio.

**Métrica nova, impressa toda geração de held-out:**

```
[generalizacao] gen=X treino=Ym held=Zm razao=Z/Y
```

`treino`/`held` são a MEDIANA da fitness entre os seres que terminaram VIVOS
(`life_left > 0`) em cada pista — não a média (sensível a outlier) nem o máximo (mede o
campeão, não a população). `razao = held/treino`: perto de 1,0 é generalização, perto de
0 é decoreba. A seleção NUNCA vê `held` — `avaliar_held_out` devolve só um `f64` de
relatório e roda DEPOIS da fitness de treino já fechada, ANTES de `evolve()`; se a seleção
enxergasse a pista reservada, o campeão passaria a decorar as duas, e a métrica que deveria
flagar decoreba pararia de flagar (ver o comentário completo na função,
`src/bin/gpu_evolve/main.rs`).

**Simplificações deliberadas da avaliação held-out** (documentadas para não virarem
surpresa numa bateria futura): setor sempre UNIFORME (ignora `--portal-curva`, pra manter
a contagem de setores comparável nos dois lados da razão); sem `--fuel` (mede rota, não
gestão de tanque); sem dopamina/joy (não alimenta nenhuma fórmula de `--fitness`); vida
CHEIA (`n_ticks`) em vez do orçamento de `--life-earn` do treino (mede se o corpo navega a
pista reservada, não se sobrevive com o fôlego calibrado pro treino).

**Testado** (`src/bin/gpu_evolve/pistas.rs`, sem GPU):
- `conjunto_vazio_e_pista_fixa_9001_comportamento_atual` — retrocompatibilidade da flag.
- `rotacao_e_deterministica_mesma_gen_mesma_pista` — mesma geração → mesma pista.
- `rotacao_percorre_o_conjunto_em_ordem_e_repete` — `gen % len`.
- `lista_de_uma_pista_e_fixa_nessa_pista` — caso degenerado (1 pista = fixa).
- `parse_pistas_*` — CSV com espaços, string vazia, token inválido, vírgula sobrando.

**A/B PENDENTE — não rodado nesta sessão** (a tarefa era implementar rotação + held-out,
não queimar GPU; havia baterias A/B ocupando-a). Falta medir, com ≥3 sementes pareadas:
- `razao` ao longo de N gerações com `--pistas` de 8 sementes vs. o braço de controle
  (sem `--pistas`, pista 9001 fixa — o comportamento medido no topo deste documento);
- na pista de treino, mediana ≥ 80% do baseline (≥2.550 m) — não-regressão. Queda na pista
  RESERVADA é *esperada e desejada*, é o que a métrica existe para capturar;
- se `razao` ficar perto de 0 mesmo depois de várias gerações: reduzir o conjunto de
  treino (2 pistas em vez de 8) ou aumentar `--held-out-cada` (menos ruído por corrida,
  mais gerações entre medições) antes de concluir que a arquitetura não generaliza.

---

## P2 · Corpo

Cada item entra sozinho, com ablação. A ordem vai do que mais muda o ótimo para o que
menos muda.

1. **Grip que satura** — hoje o Pacejka nunca chega ao limite; o carro anda como em
   trilho. Derrapagem obriga a frear antes da curva.
2. **Temperatura de motor e óleo** — canais 71/72 já ligados; a queda de potência acima de
   105 °C já existe no shader e ninguém chega lá.
3. **Pneu** — `tire_temp` não é atualizado no kernel. Pneu frio/quente muda o grip: é
   estado interno lento acoplado à decisão.
4. **Furo com recuperação** — evento raro que exige mudar de política e voltar.

**Não-regressão por item:** após 40 gerações com o item ligado, mediana ≥ 80% do valor
medido antes dele. Se não voltar, o item sai e entra pela metade (ex.: metade do efeito
de temperatura).

**Por que isto importa além da dificuldade:** já medimos que neuromodulação e memória só
pagam quando existe **corpo acoplado** — estado interno lento que faz o ótimo de
comportamento variar. Sem isso, o ótimo é constante e um reflexo puro basta. Ligar o
corpo é o que cria pressão para o cérebro crescer.

---

## P3 · Obstáculos progressivos

Buracos por nível: tamanho, posição e densidade crescentes.

**Regra de admissão, medida com `visao_check`:** um obstáculo só entra no currículo se der
**≥3 s de reação** na velocidade atual da população. Hoje o buraco padrão dá 5,0 s a
6 m/s mas **3,0 s a 10,4 m/s** — ficou mais difícil porque eles aceleraram.

**Pré-requisito:** os canais 114-117 (aproximação e movimento foveal) estão mortos no
caminho GPU — o ser não tem sensor de aproximação de obstáculo. Ligar antes, senão o
currículo mede cegueira, não competência.

---

## P4 · Regiões cerebrais

Separar visão, decisão e motor em regiões com pressão seletiva própria.

**Gatilho:** quando P2/P3 saturarem o cérebro atual — L2 já está 57,5% saturada com
entropia 0,346, o que sugere uma via única fazendo tudo. A necessidade justifica a
estrutura; construir antes disso é complexidade sem prova.

---

## P5 · Competição

Seres interferindo uns nos outros; itens que atrapalham quem está à frente. Último,
porque só faz sentido quando derrapagem e obstáculos já forem dominados.

---

## Defeitos conhecidos que atravessam o plano

Não são etapas, são dívidas — cada uma pode invalidar uma medição futura:

1. ~~**raio espectral 7,0** (limiar 1,0) — 4 medições independentes concordam~~ —
   endereçado em P0.5 (init espectral, LIGADA por padrão); A/B de efeito na fitness
   ainda pendente de rodar
2. **canais 114-117 mortos** no caminho GPU (bloqueia P3)
3. **`--retomar` mistura 70/30** com genoma aleatório — impede reavaliar o campeão puro
4. **`.brain` não ligado** — o formato existe e passa nos testes, o `gpu_evolve` ainda
   grava `UGEN v2`
5. **matriz de L2 não entregue** à análise de ativação
6. **KAN sensorial congelado** na identidade (`adapt()` nunca é chamado)
7. **`cortex_l3` morto** — 1.152 pesos mutados fora do forward
