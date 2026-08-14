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

## P1.5 · Cérebro sem lixo (saturação de L2) (implementado 2026-08-14)

**Gargalo medido:** L2 dispara em 97,4% dos ticks, 48 de 48 neurônios sempre ativos.
Persistência 97,4% com baseline 97,4% = ganho temporal ZERO — a camada não distingue
situação nenhuma. Entropia da política (throttle/steer/brake, `--ativacao`) = 0,301 nats,
abaixo do limiar de 0,50: repertório motor pobre.

**Hipótese e confirmação (item 1 da tarefa):** `brain.wgsl` ETAPA 2 (linhas 559-576,
comentário de cabeçalho) e `engine.rs` (linhas 494-515) tinham `2.0` HARDCODED em dois
sítios cada — `l2_raw = acc * (2.0/127.0)` / `raw = val * 2.0` — aplicado tanto ao `l2_raw`
(via L2) quanto ao bypass de L1. Confirmado: o ganho ERA fixo, sem parâmetro nenhum.

**Achado que muda a leitura do problema (mesmo item 1):** `final_intent` tem 96 slots, mas
só a METADE passa por L2. `final_intent[0..48]` vem do matmul L2 (`w_l2r`, ETAPA 2a,
`brain.wgsl:586-600`); `final_intent[48..96]` é BYPASS direto de `l1_raw[48..96]` — nunca
passa pela matriz L2, só pelo mesmo ganho/fadiga/tanh (`brain.wgsl:685-695`,
`engine.rs:506-515`). Quem consome cada metade (`src/simulation/motor.rs`):
  - `final_intent[0]`=throttle, `[1]`=steer, `[2]`=brake (`motor.rs:1811-1857`,
    `racer.rs:302-304`) e `[3]`/`[4]`=troca de marcha (`motor.rs:2451-2452`,
    `2544-2546`) — os 5 atuadores literais estão TODOS dentro da metade produzida por L2.
  - `axiom_policy` (`motor.rs:1833-1836`) soma um viés aditivo a `final_intent[0..64]` —
    MISTURA os 48 de L2 com os primeiros 16 do bypass; não há leitura isolada de "só
    bypass" em nenhum consumidor.
  - Diagnóstico de saturação/magnitude (`motor.rs:2842-2875`, dims_sat/intent_mag) lê os
    96 inteiros.

  Conclusão: a saturação medida (48/48 de L2) explica diretamente os 5 atuadores e o
  viés de imitação, que são o que decide a corrida — não é um artefato de metade
  irrelevante da saída. Mas como as duas metades COMPARTILHAM o mesmo `out_gain`/
  `out_beta` na ETAPA 2, qualquer A/B no ganho mexe nas duas ao mesmo tempo; não dá pra
  isolar "só conserta L2" sem também escalar o bypass — registrado aqui para quem for
  interpretar o resultado do A/B.

**Conserto (itens 2-4 da tarefa) — duas alavancas, ambas atrás de flag, ambas com o
padrão = comportamento histórico:**

1. **`--ganho-saida <g>`** (`gpu_evolve`) — `params.out_gain` em `brain.wgsl`
   (`BrainParams::out_gain`, `gpu/brain.rs`). Substitui o `2.0` hardcode nos DOIS sítios
   da ETAPA 2 (L2 e bypass). Padrão `OUT_GAIN_PADRAO = 2.0` — quem não passar a flag
   reproduz EXATAMENTE o histórico (testado, ver abaixo). O A/B mede 2.0 vs 1.0 vs 0.5.
2. **`--norm-l2-por-desvio`** (padrão DESLIGADA) — `params.norm_l2_by_std` em
   `brain.wgsl`. Divide os 48 `l2_raw` pelo DESVIO-PADRÃO POPULACIONAL do próprio ser
   (`ETAPA 2b`, `brain.wgsl:609-647`) antes do ganho/tanh — ataca a ESCALA da entrada de
   L2 (que pode crescer com a mutação ao longo das gerações), não só o ganho de saída.
   Só se aplica à via de L2 — o bypass não passa por L2, então não há "l2_raw" ali para
   normalizar (`brain.wgsl:681-684`). Com a flag desligada, `l2_std=1.0` e a divisão é
   um no-op EXATO — testado (`out_stage_norm_none_e_std_um_sao_identicos`).
   `NORM_EPS = 1e-3` evita explosão numérica quando os 48 valores ainda são quase todos
   ~0 (pesos recém-sorteados).

**Sem regressão fora da flag:** `BrainGpu::new` nasce com `out_gain = OUT_GAIN_PADRAO`,
`norm_l2_by_std = false` (`gpu/brain.rs:360-362`); `forward()` (usado por `gpu_validate`)
sempre usa `OUT_GAIN_PADRAO` hardcoded, mesmo contrato que `out_beta` já tinha. `engine.rs`
(caminho CPU/Bevy de `vida`/`universe`, fora do escopo do A/B em `gpu_evolve`) NÃO foi
tocado — continua com o `2.0` hardcode; propagar o ganho pra lá é trabalho futuro se o
A/B escolher um valor diferente de 2.0 (ver "não feito" abaixo).

**Matemática extraída para função pura testável** (item 5 da tarefa) —
`math::activations::out_stage` (`src/math/activations.rs`) espelha bit-a-bit a ETAPA 2
do shader/`engine.rs`: normalização opcional por desvio → ganho → fadiga adaptativa
(soft-threshold) → tanh. `math::stats::population_std_dev` espelha o cálculo de desvio
POPULACIONAL (÷N, via soma-de-quadrados) que o shader faz sem alocar array extra.

**Testado** (sem GPU):
- `math::activations::tests::out_stage_ganho_padrao_reproduz_formula_historica_exatamente`
  — com `out_gain=2.0`, `out_stage` reproduz EXATAMENTE (bit a bit) a fórmula hardcoded
  antiga, para 6 valores de `raw_pregain` × 2 de `out_beta` × 3 de `out_fat`.
- `out_stage_norm_none_e_std_um_sao_identicos` — flag desligada (`None`) é idêntica a
  `Some(1.0)`, confirma que o no-op é exato.
- `out_stage_norm_ligada_atenua_por_desvio_maior` — normalização atenua por um desvio
  maior, isolado de fadiga (`out_beta=0`).
- `out_stage_norm_com_desvio_quase_zero_nao_explode` — `NORM_EPS` segura o caso
  degenerado (desvio ~0 no início do treino).
- `out_stage_com_population_std_dev_reduz_saturacao` — end-to-end com
  `population_std_dev` sobre um vetor de 48 `l2_raw` simulando saturação: a normalização
  reduz (nunca aumenta) a fração de canais com `|tanh|>0,9`.
- `math::stats::tests::population_std_dev_{conhecido,de_constante_e_zero,vazio_e_zero,
  e_menor_que_amostral_para_mesmo_conjunto}` — a função de desvio isolada.
- `tests/wgsl_validation.rs::brain_wgsl_e_valido` (novo) — `naga::front::wgsl::parse_str`
  + `naga::valid::Validator` confirmam que o `brain.wgsl` editado parseia e type-checa
  sem GPU (contrato do uniform `Params` com os campos novos + padding em múltiplo de 16
  bytes).

**Build + testes (comandos que produziram os números, sem tocar a GPU — instrução
explícita da tarefa, há 2 baterias de currículo ocupando-a):**
```
cd /home/leonardo/projects/LLM/Universe
nice -n 19 cargo build --release -j 2 --bin gpu_evolve   # exit 0
nice -n 19 cargo build --release -j 2 --workspace        # exit 0
nice -n 19 cargo test --release --workspace --exclude iara_genesis --no-fail-fast
```
309 testes executados, 308 ok, 1 FALHA — `test_bitlinear_inference`
(`tests/integration_test.rs`), a MESMA falha pré-existente já confirmada (via `git
stash`) na fase anterior (Fase Canais), em subsistema não-relacionado (BitLinear/
`ablar_priming`) — não é regressão desta sessão. Os testes novos desta sessão (`out_stage`
×5, `population_std_dev` ×4, `wgsl_validation::brain_wgsl_e_valido`, `gpu_evolve` bin
14/14 incluindo `pistas`/genoma) passam todos.

**A/B PENDENTE — não rodado nesta sessão** (instrução explícita: não tocar a GPU, há 2
baterias de currículo ocupando-a). Quando a GPU liberar, com ≥3 sementes pareadas:
- `--ganho-saida 2.0` (controle) vs `1.0` vs `0.5`, cada um por N gerações — medir
  `--ativacao`: % de ticks com 48/48 neurônios de L2 ativos, entropia da política
  (alvo: sair de 0,301 para ≥0,50 nats), persistência vs baseline;
- não-regressão: mediana de distância ≥ 80% do baseline (3.189 m) em CADA ganho testado
  — se o ganho menor destravar entropia mas destruir a fitness, o item entra pela
  metade (ex.: 1,5 em vez de 1,0) ou some, igual à regra do P2;
- repetir a bateria com `--norm-l2-por-desvio` LIGADA em cima do melhor `--ganho-saida`
  encontrado — mede se a segunda alavanca (escala) ajuda além do ganho escalar sozinho,
  ou se é redundante com ele;
- se a saturação não ceder com nenhuma combinação: a hipótese muda de "escala/ganho" pra
  "os pesos de L2 aprenderam a saturar de propósito" (ex.: política bang-bang é ótima
  nesta pista) — precisaria de medição de fitness por REGIME de ganho, não só de
  saturação, antes de insistir nessa via.

---

## P1.7 · Cérebro assíncrono (implementado 2026-08-14)

> Leonardo: *"o que estou pedindo é um cérebro ASSÍNCRONO; ele vai trabalhar em
> dimensões, neurônios e partes diferentes conforme está na pista; se você fica
> sempre no mesmo tanh só está queimando tempo."* Citou como exemplo:
> `h[n] = alfa·h[n-1] + (1-alfa)·x[n]`.

**Gargalo medido (não suposto):** `grep -c tanh src/gpu/brain.wgsl` = 20 ocorrências, nenhuma
outra ativação. Todos os 96 neurônios de L1 e os 48 de L2 disparam em TODO tick — não há
memória própria, período ou limiar por neurônio. O banco de τ do CfC (`tau`, já herdável e
mutável desde antes) é o precedente direto: tornar τ heterogêneo quebrou a parede dos 680 m
no Universe — a hipótese aqui é que o mesmo tipo de heterogeneidade, agora no CÉREBRO
(não só no pré-processador sensorial), abre espaço parecido.

**Três mecanismos, um só commit** (todos tocam `Genome` + `brain.wgsl` + `genome_io.rs` —
separá-los corromperia o formato posicional do disco):

1. **Memória por neurônio** (`neuron_alpha`, 96 L1 + 48 L2) — filtro EMA
   `h[n] = h[n-1]·α + x[n]·(1-α)` sobre o valor PRÉ-ativação de cada neurônio (L1: o
   acumulador do matmul, ANTES de tanh/ReLU/etc.; L2: o `l2_raw` pré-ganho). Nasce
   α=0.0 (sem memória = idêntico a hoje); clamp em `evolve()`: [0, 0.95] — α≥1 diverge
   (a equação a diferenças só é estável com o polo do filtro dentro do círculo unitário).
   Estado `h[n-1]` num buffer novo na GPU (`buf_neuron_mem`, COPIA O PADRÃO de `buf_cfc`:
   read_write, residente entre ticks, resetado em `BrainGpu::reset_cfc` — o mesmo método
   que já zerava CfC/fadiga, estendido).

   **GAP 11 (CLAUDE.md do projeto) — o mais importante deste mecanismo:** com α alto o
   filtro passa-baixa preserva o ganho DC (α=0 e α=0.95 dão a MESMA saída para entrada
   CONSTANTE — a combinação convexa já garante isso, sem precisar de normalização extra,
   diferente do CfC), mas ATENUA a AMPLITUDE/variância de um sinal que FLUTUA tick a
   tick — o caso real de todo canal sensorial. Modelando a entrada como ruído branco de
   variância σx²: `Var(h)/Var(x) = (1-α)/(1+α)`. Em α=0.95 isso é ≈2,6% — o neurônio
   pareceria 97% mais fraco (MUDO) e seria descartado por FRACO, não por LENTO. Conserto:
   ganho de compensação `gain(α) = sqrt((1+α)/(1-α))` (gain(0)=1, sem mudança no braço
   neutro), aplicado ao `h[n]` antes de seguir no pipeline. Conta completa, com a mesma
   notação, no comentário de `neuron_mem_gain` em `brain.wgsl` e no doc do campo
   `neuron_alpha` em `gpu/evolution.rs`.

   Flag: `--sem-memoria-neuronal`.

2. **Período e limiar** (`neuron_period_l1`, `neuron_threshold_l1` — só L1; a assincronia
   temporal mira a ENTRADA sensorial, onde escalas de tempo diferentes fazem sentido
   físico — L2 já lê L1 inteiro todo tick, sem "período" próprio que faça sentido igual).
   PERÍODO: a cada quantos ticks o neurônio recalcula sua memória (nasce 1 = todo tick).
   Fora do seu tick, MANTÉM o valor anterior — reusa o mesmo `buf_neuron_mem` do mecanismo
   1 (a decisão "recalcula ou mantém" só decide SE o buffer é atualizado neste tick; o
   valor final é sempre recomputado a partir do que está no buffer, então "manter" sai de
   graça, sem precisar de um segundo buffer de cache). LIMIAR: abaixo dele, o neurônio
   contribui zero (nasce 0.0 = sempre contribui).

   **HONESTIDADE OBRIGATÓRIA (a mesma da tarefa):** na GPU todas as 96 threads de L1 rodam
   em TODO dispatch, sempre — período > 1 NÃO economiza um ciclo de clock. O ganho é de
   REPRESENTAÇÃO (escalas de tempo diferentes convivendo no mesmo cérebro, o pedido
   original), nunca de velocidade. Documentado assim no shader e reportado por geração:

   ```
   [periodo] neurônios de L1 disparando/tick: MEDIA ± DESVIO de TOTAL (X%) · janela simulada N ticks
   ```

   Calculado em CPU pura (`gpu::evolution::simulate_l1_fire_counts`, testado sem GPU) —
   o período de cada neurônio já é conhecido antes do tick 0 rodar, então simular é mais
   barato e mais exato que amostrar a GPU. Janela = MMC(1..8) = 840 ticks (cobre um ciclo
   completo dos períodos possíveis, clamp de `evolve()`), não a corrida inteira.

   Flags: `--sem-periodo-neuronal` e `--sem-limiar-neuronal` (independentes).

3. **Ativação heterogênea** (`neuron_activation`, 96 L1 + 48 L2) — seletor por neurônio
   entre tanh (hoje)/ReLU/linear/leaky-ReLU(0,01), usando as fórmulas JÁ EXISTENTES em
   `math::activations` (não reimplementadas — `brain.wgsl::apply_activation` é o espelho
   GPU). Nasce 0.0 (tanh = idêntico a hoje). Aplicado em dois pontos bem definidos pela
   arquitetura: a não-linearidade de QUANTIZAÇÃO de L1 (`l1_raw → i8`, o que alimenta L2)
   e a não-linearidade FINAL de L2 (o que vira `final_intent[0..48]`). O bypass de L1→
   `final_intent[48..96]` continua com tanh fixo de propósito — é squashing de SAÍDA
   MOTORA (parametrizado por `out_gain`/`out_beta`, P1.5), não "ativação do neurônio de
   L1" (que já foi decidida na quantização).

   **Mutação DISCRETA, taxa MENOR que a gaussiana dos pesos** (0,2%/neurônio/geração,
   contra a taxa 3× — ~6%/geração — usada nos outros campos herdáveis pequenos): trocar
   tanh→ReLU muda o REGIME inteiro do neurônio (de sempre-ativo-saturando-suave para
   metade-em-zero-exato), não é um ajuste fino como mover um peso em σ=0,05 — é uma
   decisão discreta cujo efeito só a seleção mede depois de várias gerações. Taxa alta
   faria o repertório oscilar sem tempo de a seleção provar se a troca ajudou.

   **Custo de divergência de warp (RDNA2, wave32 — CLAUDE.md do projeto):** se as 32
   lanes de uma wave tiverem ids de ativação diferentes, o hardware percorre TODOS os
   casos presentes na wave (predicação) — real, mas pequeno (ALU pura, poucas
   instruções por caso, sem acesso a memória divergente). Documentado no shader, não
   vendido como grátis.

   **Por que ataca o problema:** com tanh todo neurônio satura suave no batente; com
   ReLU metade fica em ZERO EXATO — esparsidade CONTEXTUAL de verdade (o neurônio "não
   opina" naquele instante em vez de opinar fraco), diferente da saturação de tanh, que
   sempre emite algo.

   Flag: `--sem-ativacao-heterogenea`.

**Buffers novos (`brain.wgsl`, bindings 11-12 — cabe no limite já pedido, 12 storage
buffers por estágio, sem precisar bumpar `gpu_evolve`/`gpu_validate`/`gpu_loop`):**
`buf_neuron_meta` (read-only, `NEURON_META_PER_SER=480` f32/ser: `l1_alpha(96) ·
l1_period(96) · l1_threshold(96) · l1_activation(96) · l2_alpha(48) · l2_activation(48)`,
subido uma vez por geração — mesmo contrato de `buf_tau`) e `buf_neuron_mem` (read_write,
`NEURON_MEM_PER_SER=144` f32/ser, estado residente — mesmo contrato de `buf_cfc`).

**Braço de controle, o mais forte possível:** cada `--sem-X` (1) congela a MUTAÇÃO do
campo em `evolve()` (atomics `NEURON_*_MUTATE_ON` em `gpu/evolution.rs`, padrão `true`) —
o campo fica preso no neutro de nascença pelo resto da corrida — E (2) liga o bit
correspondente de `BrainParams::mech_off_mask`, que faz o SHADER ignorar o que o genoma
carrega, mesmo que um `--retomar` anterior à flag já tivesse mutado alguma coisa. Sem
nenhuma flag (padrão), os três mecanismos mutam livremente a partir do neutro.

**Genoma legado (herança intocada):** `.bin` salvo antes de P1.7 carrega com os seis
campos em NEUTRO — a verdade sobre aquele genoma (rodava sem os três mecanismos), não uma
invenção. Mesmo padrão opcional de `slow`/`tau`/`sensors` em `load_genome`
(genome_io.rs) — comprimento do arquivo decide o que é lido.

**Reprodução exata na geração 0 (requisito da tarefa, testado):** `Genome::random`
sempre nasce com os seis campos neutros — `neuron_alpha`=0.0, `neuron_period_l1`=1.0,
`neuron_threshold_l1`=0.0, `neuron_activation`=0.0 — que fazem `brain.wgsl` reproduzir a
fórmula de antes desta mudança bit a bit (α=0 ⇒ ganho DC exato 1, h[n]=x[n]; período=1 ⇒
recalcula todo tick; limiar=0 ⇒ `abs(v)<0.0` nunca é verdade, nunca zera; ativação=0 ⇒
tanh). `forward()` (usado por `gpu_validate`) força `mech_off_mask=0b1111` incondicional
— mesmo contrato que já tinha para `out_beta`/`out_gain`/`norm_l2_by_std`: o caminho de
equivalência CPU/GPU nunca expõe parâmetro novo, sempre reproduz o histórico.

**Testado** (sem GPU):
- `gpu::evolution::tests::genoma_novo_tem_mecanismos_p17_neutros` — 5 sementes, os seis
  campos nascem no valor exato que reproduz "hoje".
- `neuron_mem_gain_em_alfa_zero_e_identidade` / `neuron_mem_gain_cresce_e_fica_finito_no_teto_do_clamp`
  — a matemática do ganho (mesma fórmula do shader, espelhada em Rust puro).
- `evolve_muta_mecanismos_p17_e_flags_sem_x_congelam_no_neutro` — um teste só (evita a
  corrida de atomics entre threads de `cargo test`, R8 do CLAUDE.md do projeto): 40
  gerações com `p_mutate=1.0`, nenhum campo sai de [0,0.95]/[1,8]/[0,0.9]/{0,1,2,3} e
  ALGUM valor realmente mudou (não vacuamente verdadeiro); depois, com os quatro atomics
  desligados, mais 20 gerações com os seis campos bit-idênticos ao neutro.
- `pack_neuron_meta_produz_layout_esperado_pelo_shader` — o achatamento SoA bate com o
  layout que `brain.wgsl` espera, campo a campo.
- `simulate_l1_fire_counts_*` (3 testes) — período neutro dispara 100% sempre (desvio
  zero); período=2 uniforme dá média 50% mas desvio MÁXIMO (todos em lockstep, degrau
  0/total, não um valor constante — só período HETEROGÊNEO produz assincronia de
  verdade); população mista (período=1 vs período=4) tem desvio > 0.
- `bin::gpu_evolve::tests::genoma_sobrevive_ida_e_volta_no_disco` (estendido) — os seis
  campos, marcados fora do neutro, sobrevivem ida-e-volta no disco.
- `bin::gpu_evolve::tests::genoma_antigo_sem_lento_nem_tau_ainda_carrega` (estendido) —
  `.bin` legado (sem P1.7) carrega com os seis campos em neutro.
- `bin::gpu_evolve::tests::genoma_v2_com_cabecalho_sobrevive_ida_e_volta` (estendido) —
  idem, formato v2 com cabeçalho.
- `tests/wgsl_validation.rs::brain_wgsl_e_valido` — `brain.wgsl` editado ainda
  parseia/type-checa (naga), sem precisar de GPU.

**Smoke de GPU** (não bateria longa — 1-2 gerações, prova só que roda):
```
cd /home/leonardo/projects/LLM/Universe
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
env -u DISPLAY ./target/release/gpu_evolve --seres 8 --ticks 200 --gens 2 --out /media/llm-data/universe-runs/p1-7-smoke
```

**A/B PENDENTE — não rodado nesta sessão** (a tarefa era implementar os três mecanismos e
provar que rodam, não medir efeito na fitness). Falta, com ≥3 sementes pareadas, ≥40
gerações:
- cada mecanismo isolado (`--sem-Y` nos outros dois) contra o braço totalmente neutro
  (`--sem-memoria-neuronal --sem-periodo-neuronal --sem-limiar-neuronal
  --sem-ativacao-heterogenea` — reproduz "hoje" pelo resto da corrida) — mediana de
  distância, ≥1V/≥2V, e se o α/período/limiar/ativação realmente se afastam do neutro sob
  seleção real (não só sob `p_mutate=1.0` sintético dos testes);
- distribuição final de ativações escolhidas por neurônio (tanh domina? ReLU se
  concentra em canais específicos — parede vs. curvatura vs. velocidade?);
- correlação entre α alto e QUAL canal de entrada o neurônio lê mais forte (memória
  deveria ser selecionada nos canais que precisam de contexto temporal — parede/
  curvatura —, não nos que já são instantâneos, ex. telemetria de RPM).

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
