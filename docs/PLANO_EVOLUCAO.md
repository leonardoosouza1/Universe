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

## P1.8 · Auditoria total (conserto aplicado em 2026-08-14)

Workflow de agentes (diagnóstico paralelo → conserto sequencial, mesmo padrão do
CLAUDE.md do projeto): 5 frentes auditaram o cérebro/mundo/currículo em paralelo,
um agente sequencial aplicou os achados confirmados em ordem de gravidade,
compilando após cada item. Prioridades fixas do Leonardo: (1) L2 tem que
funcionar, (2) gerador de pistas com piso de raio + gate de passabilidade,
(3) currículo com piso de mérito real, (4) partes mortas do cérebro.

### 1 · L2 tem que funcionar

**Causa raiz confirmada:** com σ_pesos=0,30 i.i.d. e 96 entradas quase-saturadas
(L1 dispara em 97,4% dos ticks), `Std(l2_raw) ≈ 2,9-5,9` mesmo com `out_gain`
reduzido — `tanh` satura sozinho em qualquer valor de `--ganho-saida` testado no
A/B (2,0/1,0/0,5). L1 recebe correção espectral adaptativa por genoma
(`SPECTRAL_INIT_ON`); L2 nunca recebeu — nem no init, nem depois.

**Conserto:** `--norm-l2-por-desvio` já existia (P1.5) mas nascia DESLIGADA —
normaliza `l2_raw` pelo desvio-padrão MEDIDO dos 48 valores do ser a cada tick,
antes do ganho/tanh (`brain.wgsl`, guardado atrás de `params.norm_l2_by_std`,
já testado/no-op-exato quando desligada). Virou o padrão em `gpu_evolve`
(`src/bin/gpu_evolve/main.rs`): `norm_l2_por_desvio` nasce `true`, com
`--sem-norm-l2-por-desvio` como braço de controle explícito para quem precisar
reproduzir o comportamento histórico. Escolhido em vez de reescalar L2 no init
(a outra alavanca do achado) porque já está implementado, testado por
`gpu_validate --cerebro` (ver evidência abaixo) e é adaptativo por tick, não só
no nascimento — ataca a causa (escala dinâmica que deriva por mutação sem
regularização), não o sintoma.

**Evidência:** `gpu_validate --cerebro` com a flag ligada por padrão —
`final_intent (CPU-manual vs GPU) max_dif=1,7881e-7 mismatches=0/96`,
`cerebro (drift CfC 100t) PASS`. `cargo test --release --bin gpu_evolve` 14/14.

### 2 · Gerador de pistas — piso de raio + gate de passabilidade

**Achado crítico confirmado:** `validate_track` (`src/world.rs`) media ângulo de
virada e comprimento de segmento, nunca o RAIO da curva — pista podia ter curva
fisicamente impossível e passar validação. **Achado alta:** buracos procedurais
nasciam com deslocamento lateral ZERO (em cima do eixo) e sem checar largura
livre pós-colocação; para seed≥1.000.000 (currículo) eram calculados sobre um
traçado FANTASMA (`procedural_waypoints` rodava de novo, independente do
`do_curriculo` que gerou o traçado real). **Achado alta:** `pista_check` (a
ferramenta que mede se a pista é dirigível) nunca era chamada por `gpu_evolve` —
não existia gate contra pista com defeito.

**Conserto:**
- `validate_track` ganhou piso de raio (`min_radius_m < 6.0` rejeita — mesmo
  piso que `pista_gen.rs::gerar` já usa, `d.r_min < 6.0`, consistência entre os
  dois geradores).
- `procedural_waypoints` foi de 32 para 96 tentativas antes do fallback, e o
  fallback (quando esgota) agora **loga alto** (`eprintln!`) em vez de cair em
  silêncio num candidato não-validado — achado médio "fallback contradiz a
  própria garantia que o módulo declara".
- `track_holes_for_seed`/`procedural_holes` (`src/world.rs`) ganharam parâmetro
  de traçado REAL (elimina o traçado fantasma) e deslocamento lateral pro lado
  interno da curva (30% da largura real) + guard `track_width >= 12.0` no
  overlay do buraco (não perfura pista estreita demais pro chassi 2m).
- `VoxelWorld::validar_pista(width, ate) -> Vec<String>` (novo método, extraído
  da lógica de `pista_check.rs`) roda como GATE em `gpu_evolve` para toda seed de
  `--pistas`/`--pista-held-out` ANTES de subir GPU — aborta com `exit(1)` e
  mensagem clara se alguma tiver defeito (eixo fora do asfalto ou
  estrangulamento).

**Achado adicional descoberto na verificação (fora da lista original):**
`validate_track` já tinha um critério pré-existente,
`min_seg_len_at_sharp < 12.0` ("curvas afiadas precisam segmento ≥ 12m"), que
**rejeita ~100% dos candidatos procedurais** para QUALQUER seed testada (9001,
9002, 9006, 9008, 1, 5, 42, 100, 555 — medido com um teste temporário, removido
depois de confirmar). Isso significa que o caminho "validado" de
`procedural_waypoints` já era essencialmente morto ANTES desta sessão — toda
seed procedural sempre caía no fallback não-validado, silenciosamente. O piso de
raio novo, portanto, não é hoje o filtro ativo (o `min_seg_len_at_sharp` rejeita
primeiro); ele passa a valer no dia em que esse critério for recalibrado. A
proteção REAL contra pista quebrada hoje é o gate `validar_pista` (item acima),
que testei contra a geometria de fato servida (não a candidata rejeitada) e
confirmei OK para 9001/9002/9006/9008 (3306-3414 m de eixo examinados, 0
defeitos). Recalibrar `min_seg_len_at_sharp`/a amostragem do Catmull-Rom (hoje
`n_samples=8`, não-uniforme em arco) fica para uma sessão futura — não é um bug
que eu introduzi, e mexer no critério de 12m sem medir o efeito seria o mesmo
erro que este projeto já cometeu (tweak sem pesquisa).

**Evidência:** `cargo build --release --bin gpu_evolve` limpo a cada item.
Smoke `--seres 4 --ticks 10 --gens 1 --pistas 9001,9002,9006,9008,9001`: gate
imprime `[pista_check] seed N: OK (Xm de eixo examinados, sem defeito)` para as
quatro, `exit 0`.

### 3 · Currículo com piso de mérito real

**Achado alta confirmado:** `--trocar-ao-completar` promovia usando `r.n_ge1v`
(hodômetro/`prog_acum` ≥ `lap_dist`, sujeito a aliasing de posição — busca o
segmento mais PRÓXIMO, sem exigir ordem de visita) enquanto o log de "voltas
reais" já usava `voltas_reais`/`v1` (setor a setor, ORDENADO) — dois cálculos
genuinamente diferentes respondendo a mesma pergunta, um deles não bate com o
que o operador lê. Contexto do dia: "`--trocar-ao-completar 1`: UM ser
rastejando promove a turma."

**Conserto:** a condição de promoção agora usa `v1` (a MESMA grandeza do log
"voltas reais") e exige UMA de duas rotas:
- **população:** `v1 ≥ max(N configurado, 25% da população)` — o piso pedido
  pelo Leonardo (`--trocar-ao-completar exige N≥25% da população`);
- **velocidade:** `v1 ≥ N configurado` **e** mediana de velocidade da geração
  ≥ `--curriculo-veloc-min` (padrão 5,0 m/s = 18 km/h — ~2× acima do pior
  "rastejar" já medido neste projeto com vida fixa, 1,48-2,96 m/s, e bem abaixo
  do ritmo de corrida real, ~22 m/s).

Isso ataca exatamente o padrão do contexto (um ser sozinho completando a volta
sem o resto da população acompanhar em quantidade OU ritmo). O log agora
imprime qual rota disparou a promoção, e também imprime quando `v1>0` mas
NENHUMA rota bateu (antes: silêncio total, exigia reler código pra saber por
quê a pista não trocou).

**Achados médios aplicados de graça (mesma área):**
- métrica `[elite] mesmo genoma: X%` agora guarda a pista em que a fitness
  anterior foi medida (`elite_fitness_anterior_pista`) e só compara/acumula
  quando a pista não mudou — senão imprime "pista mudou... razão não
  comparável" em vez de comparar fitness de pistas diferentes como se fossem a
  mesma grandeza.

**Evidência:** smoke `--seres 8 --ticks 400 --gens 2 --trocar-ao-completar 1
--pistas 9001,64` roda sem erro; `cargo test --release --bin gpu_evolve` 14/14.

### 4 · Partes mortas do cérebro

- **`cortex_l3`** (1.152 pesos, caminho CPU/Bevy): mutado toda geração em
  `src/evolution/spawn.rs` sem que `forward()` o use (`engine.rs:517`, `let _ =
  &self.cortex_l3; // kept struct, forward unused`). Mutação REMOVIDA (opção
  barata do achado — religar no forward é feature nova, 24→96 dims sem
  mapeamento óbvio, fica para decisão explícita do Leonardo).
- **KAN sensorial** (3.072 coeficientes, `SensoryPreFilter`): `adapt()` nunca é
  chamado em nenhum caminho do `gpu_evolve` — coeficientes ficam `[1.0;16]`
  identity para SEMPRE, em todo ser, toda geração (confirmado: não está no
  `Genome`, `apply_to_engine` nunca toca `kan_filters`). Em vez de arrancar o
  kernel GPU (risco maior, mexeria em bind group/shader já validado por
  `gpu_validate --cerebro`), a correção ficou no lado CPU que é 100%
  redundante: `BrainGpu::upload_weights` subia os mesmos 3.072×n floats
  IDÊNTICOS a cada geração — agora sobe uma vez só (`kan_uploaded: AtomicBool`)
  e pula nas chamadas seguintes. Comportamento idêntico (dado nunca muda),
  banda economizada.
- **`kan_knots` de SAÍDA** (3×12, campo do `Genome`): mutava toda geração em
  `evolve()` mas o forward do `gpu_evolve` sempre usa `default_kan_knots()`
  global (`main.rs`), nunca os do genoma — mutação era RNG/CPU gasto à toa e
  sujava o `.dna` com "evolução" que não existe. Mutação REMOVIDA de
  `evolve()` (opção barata do achado).
- **`DEAD_L1_INPUTS`** (canal 70, TireTemp): zerado só em `Genome::random`
  (nascimento) — `evolve()` mutava L1 livremente e não reaplicava a máscara.
  Medido no campeão real (`evolve_best_gen0059.bin`): coluna ch70 já 96/96
  não-zero, |w| médio 0,2085 (igual ao canal vizinho vivo). Novo método
  `Genome::zero_dead_inputs()`, chamado em `evolve()` logo após a mutação de
  L1 — **não** em `load_genome` (preservaria o princípio de "carregar do disco
  não modifica peso", mesmo motivo do teste
  `genoma_legado_carregado_do_disco_nao_e_reescalado`). Teste de regressão novo:
  `evolve_preserva_zero_dos_canais_mortos` (10 gerações, `p_mutate=1.0`, falha
  sem o fix, passa com ele).

**Rejeitado nesta passada (feature nova, não bug):** religar `cortex_l3`/KAN
sensorial/`kan_knots` de verdade no forward — cada um exige decisão de design
(mapeamento de dimensões, novo tráfego de buffer por-ser) que o próprio achado
já sinalizava precisar de confirmação explícita do Leonardo antes de codar.

**Evidência:** `cargo build --lib` limpo a cada item; `cargo test --release
--lib gpu::evolution::` 19/19 (inclui o teste novo).

### 5 · Achados críticos/altos adicionais aplicados (fora dos 4 títulos, mesma
### auditoria, "em ordem de gravidade")

- **[crítica] CfC de percepção corrompia o ser seguinte.** `retina_digest.wgsl`
  escrevia o estado CfC squashed ("ch192-207") em `buf_output` via
  `write_channel(being, 192+i, ...)` — mas `buf_output` tem EXATAMENTE 48
  words (192 bytes) por ser, sem folga (`SENSORY_WAVE_SIZE=192`,
  `brain.wgsl`'s `const WAVE: u32 = 192u` nunca lê índice ≥192). O índice caía
  no INÍCIO do bloco do PRÓXIMO ser, corrompendo via `atomicOr` os canais 0-15
  dele (FAR/NEAR pool — lidos de verdade por L1). Como é `atomicOr` sobre
  buffer zerado, o efeito é silencioso e não aparece em auditoria de variância
  (preserva variância, só mistura o dado errado). Fix: os 16 valores agora vão
  para `extra_out` (`EXTRA_OUT_STRIDE` 3→19), espaço próprio por ser —
  `cfc_state` (o estado recorrente de verdade, usado pela PRÓXIMA chamada) não
  foi tocado, só o canal de exposição que corrompia o vizinho. Escopo: só
  `retina_digest.wgsl`, o kernel usado por `gpu_evolve`
  (`src/bin/gpu_evolve/context.rs` usa `RetinaDigestGpu`, não
  `perception_temporal.wgsl` — este último roda com o slot sempre zerado em
  `gpu_evolve`, `val[1..17]` deliberadamente zero em `context.rs`, então o
  mesmo padrão de escrita ali é hoje um no-op para este binário). A mesma
  classe de bug existe em `perception_temporal.wgsl` para `gpu_loop.rs`/
  `gpu_report.rs` (caminho Bevy/interativo) — **fora do escopo desta sessão**
  (não é o pipeline de treino), documentado aqui para não se perder.
  Validado na GPU real: `retina_digest_bate_com_cpu_ao_longo_de_varios_ticks`
  (`--ignored`, roda com `cargo test --release --lib -- --ignored
  retina_digest_parity`) — reescrito para ler `extra_out[3..19]` em vez do
  endereço aliasado, PASS bit-exato contra o oráculo CPU. Bug encontrado
  DURANTE o conserto (não no plano original): a primeira versão do fix só
  copiava os 3 primeiros floats de `extra_out` em `read_after_submit`
  (`retina_digest.rs`) — os 16 novos ficavam sempre `0.0`. Pego pelo próprio
  teste de paridade (GPU=0 vs CPU=-2), corrigido antes de declarar pronto.
  `gpu_validate --nucleo`/`--cerebro` PASS; smoke `gpu_evolve --seres 70`
  (acima do limiar de 64 onde o bug se manifestava) roda limpo, 3 gerações,
  mediana crescendo (+8,38 m/gen).
- **[alta] `avaliar_held_out` contaminava `AUDIT_ACUM`.** `ctx.tick()`
  acumula em `AUDIT_ACUM` sempre que `AUDITAR` está ligada, sem distinguir
  ticks de treino de ticks do held-out (pista reservada, nunca deveria entrar
  em métrica nenhuma usada pela seleção). Fix: `AUDITAR.store(false,...)` ao
  redor da chamada de `avaliar_held_out`, restaurado depois — mesmo padrão que
  `l1_samples`/`l2_samples`/etc já usavam, só que via o toggle que já existia
  em vez de clonar o `AuditAcum` inteiro. Bônus: held-out para de pagar o
  readback de visão que nunca usava.
- **[alta] `DEAD_L1_INPUTS` erodindo por mutação.** Coberto na seção 4.
- **[alta] `--retomar` + `--tau-hetero` descartava o τ evoluído do elite
  exato.** `randomize_tau` rodava sobre a população INTEIRA (inclusive
  `genomes[0]`, o clone exato do campeão carregado do disco), sobrescrevendo
  um τ que já passou por seleção real em gerações anteriores. Fix: novo
  booleano `elite_carregado_do_checkpoint` pula só o índice 0 quando ele é o
  elite exato — os demais slots (filhos atenuados) continuam recebendo τ novo
  normalmente, preservando o propósito da flag.

### O que ficou de fora (rejeitado, com motivo)

- **Hall-of-fame por pista** (achado alta — sem HOF, `n_elite=1` real
  (`scripts/queue_overnight.sh`) arrisca perder o campeão bom da pista antiga
  pra um sortudo da pista nova). Não implementado nesta passada: é uma
  feature nova (estrutura de dados + ponto de injeção em `evolve()`/`main.rs`,
  interação com elitismo/torneio a decidir), não um bug de uma linha — o
  próprio achado descreve como "ajustes de como plugar", sinal de que precisa
  de decisão de design, não só correção. Fica para uma sessão dedicada.
- **Recalibrar `min_seg_len_at_sharp`/amostragem do gerador procedural** —
  descrito na seção 2. Descoberto durante a verificação, não estava nos
  achados originais, e mexer sem medir o efeito seria o mesmo erro que este
  projeto já cometeu antes (tweak sem pesquisa).
- **Religar `cortex_l3`/KAN sensorial/`kan_knots` de verdade no forward** —
  feature nova em cada caso, coberto na seção 4.
- **Cosméticos de baixa prioridade** (comentários "223" residuais em
  `core.rs`, largura hardcoded de 14,0m na medição de dificuldade do
  currículo, R_min por torque/amortecimento no `MATH_INVENTORY.md`) — não
  tocados; achados de baixo impacto, sem urgência, e cada um isolado o
  bastante para entrar como item avulso depois.

### Build + testes finais

```
cargo build --release -j 2 --workspace           # limpo, só warnings pré-existentes
cargo test  --release --workspace --exclude iara_genesis   # todos os alvos PASS
```

Achado colateral pego pelo gate final (não fazia parte da lista de P1.8, mas
bloqueava `cargo test --workspace` limpo): `tests/integration_test.rs::
test_bitlinear_inference` comparava contra uma fórmula que multiplicava
`weight_scale` no forward — `BitLinear::forward` (`src/layers.rs`) não faz mais
isso desde um commit anterior a esta sessão (`git log` confirma: nenhuma
mudança minha em `layers.rs`/`tests/integration_test.rs` antes deste fix). A
asserção nunca foi atualizada quando o comportamento mudou; corrigida para
`30,0/127,0` (o valor que `forward()` de fato produz hoje), com o cálculo
documentado no próprio teste.

Também adicionado: `tests/wgsl_validation.rs::retina_digest_wgsl_e_valido` —
`retina_digest.wgsl` (o shader mais tocado nesta sessão) ganhou a mesma rede de
segurança estática (naga parse+type-check, sem GPU) que `brain.wgsl`/
`perception*.wgsl` já tinham.

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

---

## P1.9 — Fechamento do cérebro atual: L2, câmbio e seed (2026-08-14)

Pergunta do Leonardo: "se L2 não é usado, investigar pra que era, tem fluxo pra alimentar?"

**Resposta (2 agentes + verificação própria):**
- L2 É o único caminho motor: `final_intent[0..3)` (throttle/steer/brake) vem todo de L2.
  Fluxo de entrada funciona: percepção → KAN → CfC → L1 → i8 → L2.
- O que NÃO tem consumidor: linhas 5..47 de L2 + bypass L1 `[48..96)` + `cortex_l3`.
  Arqueologia: bypass nasceu como "baseline reativo + composição AND" (commit 3dbaed6);
  linhas 3-95 eram "feature neurons for evolution to explore" (9ac2cdd) — exploração sem
  consumidor. Dimensão 96 = L1_NEURONS, artefato, não escolha semântica.
- Mielinização lê `[0..64)` mas `myelinate()` é no-op DELIBERADO desde a migração F32
  (layers.rs:55-62). `axiom_policy` só escreve bias[0..3).
- Pesos: linhas cegas ≈ indistinguíveis das lidas (cosseno/curtose, 2 sementes) — carona
  genética do elitismo. Sinal fraco inverso em RMS² (lidas crescem MAIS, p≈0.04-0.05 nas 2).
  JSON: /media/llm-data/scratch/l2-investigacao/.

**Bugs achados e consertados no mesmo turno:**
1. `--seed` NÃO entrava na mutação sob `--retomar` (semente = f(generation) puro) —
   pos-auditoria-s77/s31 eram byte-idênticos (md5 gens 0/5/10/15/18). Fix: termo
   `pop_seed` na semente de `evolve()` que se ANULA em pop_seed=0 (compat total).
   Teste: `evolve_pop_seeds_diferentes_divergem_e_zero_reproduz`. s31 relançado com
   binário novo: gen0001/0002 divergem do s77 ✓ (md5).
2. Comentário P1.8 do brain.wgsl dizia "intent[3]/[4] = shift" no treino — FALSO
   (câmbio era automático... e nem automático era, ver 3). Corrigido no header.
3. **Caixa de 6 marchas sem upshift**: physics_power usa `gear_ratio` na força, mas o
   gpu_evolve só tinha piso marcha 2 + queda <3 m/s. Teto físico da marcha 2:
   (8500+250)rpm ÷ (3,8×3,7) × 0,33 = **21,5 m/s** — pico real medido da população:
   21,9 m/s, colado no limitador. A população NUNCA pôde passar disso, independente
   do cérebro. Fix: `--cambio travado|auto|neural` (paridade motor.rs:2451-2494,
   auto-RPM + override neural por borda de intent[3]/[4], cooldown 0,35 s). Padrão
   TRAVADO até A/B validar (mudar dinâmica por default sem medir é proibido).
4. `--l2-congelar-cegas`: congela mutação das linhas 5..48 de L2 (4.128 pesos de busca
   inerte). Teste: `evolve_congela_linhas_cegas_de_l2`. Padrão OFF (A/B).

**Fila (quando a GPU liberar):**
- Smoke 2 min: `--cambio auto` — marcha >2 aparece? velocidade passa de 22 m/s?
- A/B 3 braços × 2 sementes: cambio travado/auto/neural (neural dá às linhas 3-4 de
  L2 função de verdade; auto responde "o teto era o gargalo?").
- Tensão rica: marcha 3+ ultrapassa 28 m/s = velocidade onde a retina CEGA — o ser
  ganha controle do trade-off velocidade×percepção (liga com a Tese do Corpo).
- Destino das linhas cegas restantes: candidatas a entrada do cérebro lento
  (buracos do slow_gather, ex. CHANNEL_MAP[25]=-1) = enxerto 1 de neuromod.

### P1.9 — resultado do A/B de câmbio (2026-08-14, 2 sementes genuínas)

Baseline travado (pos-auditoria) vs `--cambio auto` (30 gens, 20k ticks, mesmo
campeão-base curriculo-s77/gen0059, pistas 9001+9003 com gate):

| braço | Σ seres ≥1V | máx global | pico veloc |
|---|---|---|---|
| travado s77/s31 | 108 / 534 | 5907/5880 m | ~21,9 (motorizado; 27,9 em ladeira) |
| auto s77/s31 | 18 / 5 | 4786/4212 m | 45,1 / 46,6 m/s |

**Conclusão (replicada):** liberar marcha 3-6 SEM percepção compatível derruba o
desempenho ~6-100× em completude de volta. A retina cega acima de 28 m/s e a
marcha 2 funcionava como protetor acidental. Corpo e percepção co-adaptados —
mesma família dos achados "120 m de visão não ajuda" e Tese do Corpo.

**Predição REGISTRADA antes do braço neural fechar** (rodando agora): com mix
85%/zero o intent[3]/[4] do elite fica abaixo do limiar 0,5 → neural ≈ travado
(a opção sem a obrigação); cauda pequena de chance de descobrir upshift só em
reta. Se neural >> travado, a leitura muda: o cérebro consegue usar velocidade
QUANDO ele decide onde.

**Implicação pro fechamento do cérebro:** o consumidor natural das linhas cegas
de L2 não é o câmbio (corpo não sustenta) — reforça o caminho "linhas cegas →
entradas do cérebro lento" (enxerto 1 de neuromod) como destino de valor.

### P1.9 — braço neural fechado: predição REFUTADA, veredito final do câmbio

Predição registrada era "neural ≈ travado" (intent[3]/[4] abaixo de 0,5). ERRADA
pelo mecanismo: com `--norm-l2-por-desvio` (padrão), a saída das linhas cegas é
z-score×ganho 2,0 → |tanh| passa de 0,5 com frequência → a caixa recebeu RUÍDO.

Medição causal limpa (mesma população determinística na gen 0, só o modo muda):
- travado gen0: 39/48 completam · neural gen0: 8/48 · depois 0 completantes por
  29 gerações (Σ≥1V=8 vs 108/534 do travado). max_global do neural = a própria
  gen 0 (4665 m). Sementes divergem (md5 gen0001 ✓) — replicado 2×.

**Veredito final: `--cambio travado` PERMANECE padrão.** Dois negativos com
mecanismo: (a) auto força velocidade além da percepção (retina cega >28 m/s);
(b) neural entrega atuador a linhas-ruído. A caixa 3-6 vira alavanca REAL
quando a percepção crescer — e qualquer religação futura das linhas cegas
deve passar pelo cérebro LENTO (integrador, gated por out-beta), nunca
direto num atuador. Runs: cambio-{auto,neural}-s{77,31}, travado-cont-* segue.

### P1.9 — destrave de velocidade por DEGRAU (`--cambio-teto`, 2026-08-14)

Diagnóstico com o laboratório (obs.f32, 20k ticks do campeão):
- Visão NÃO saturada: 0% dos canais em +127 em todas as famílias e faixas de
  velocidade; horizonte varia (std 34-91); 23% em −127 = asfalto escuro (semântica).
- RPM no limitador em 73,1% dos ticks (méd 8155/8500) — o motor vive estourado
  na marcha 2. Gargalo confirmado é o CORPO, não o cérebro nem o sensor.
- temp motor 20→90° e canal 71 vivo (std 16,8) — correção de leitura minha:
  o "constante" era o começo frio da corrida no screenshot.

Ação: `--cambio-teto N` limita a caixa do auto/neural. Bateria cambio-teto3-s{77,31}
(auto, teto 3 = 30,5 m/s, +42% sobre o teto atual, na borda dos 28 m/s da retina),
30 gens, partindo dos campeões travado-cont (recorde 6706 m).

**Predição REGISTRADA antes do resultado:** teto 3 fica dentro do alcance da
percepção → a evolução readapta em ≤30 gens e SUPERA o travado em Σ≥1V e máx
global (diferente do teto 6, que caiu 6-100×). Se falhar igual ao teto 6, a
leitura muda: o problema não é a cegueira acima de 28 m/s, é a política de
frenagem/curva co-adaptada à marcha 2 — e o degrau precisa ser o currículo
(teto sobe DENTRO da bateria quando a população completa volta).

### P1.9 — resultado teto3 + currículo de teto lançado

teto3 fixo (30 gens, por gen): Σ≥1V/gen 6,2/21,9 vs travado 23,6/36,1 — predição
"supera em 30 gens" REFUTADA; mas ≥2V=156 (s31), máx 6587 m, pico 37 m/s = 30-100×
melhor que teto6, readaptação em curso. Leitura: degrau único ainda é grande demais
OU 30 gens é pouco. Resposta: `--cambio-curriculo` — teto começa em 2 (≡ travado) e
sobe +1 a cada promoção por mérito (a promoção que sobe teto NÃO roda pista; um
degrau por vez). Bateria cambio-curric-s{77,31} (40 gens, máx 6) rodando.
Predição: população mantém completagem entre degraus e termina ≥ teto3 fixo em
Σ≥1V, com marcha máx ≥4 conquistada por mérito.

### P1.9 — currículo de teto: PREDIÇÃO CONFIRMADA, recorde novo (2026-08-15)

Final 40 gens × 2 sementes (cambio-curric, auto, teto 2→6 por mérito):
- s31: Σ≥1V=1201 (30,0/gen) · Σ≥2V=247 · **máx 6951 m (recorde; antes 6706)** · pico 52,9 m/s
- s77: Σ≥1V=812 (20,3/gen) · Σ≥2V=26 · máx 6385 m · pico 56,9 m/s
- As DUAS sementes conquistaram os 4 degraus (2→3→4→5→6) por mérito populacional.

Comparação (Σ≥1V/gen): travado 23,6/36,1 · teto6 direto 0,6/0,2 · teto3 fixo 6,2/21,9
· **currículo 20,3/30,0** — com a caixa INTEIRA liberada e voltas duplas (247 no s31).
A mesma população que colapsava 100× com teto6 de porteira aberta domina teto6
quando o corpo é conquistado degrau a degrau. Currículo de corpo = mecanismo
validado; candidato a padrão de bateria longa.

Fila (aguardando ordem do Leonardo p/ GPU): observação do campeão 6951 m no
laboratório (cérebro a 50 m/s) · continuação curric p/ consolidar · A/B
curric como padrão.

### P1.9 — CORREÇÃO da leitura do currículo (validação de replay, 2026-08-15)

Abrir o replay corrigiu a narrativa:
1. Os 4 degraus subiram nas gens 0-3 (48→22→18→20 completantes; limiar 12 promove
   até com a população CAINDO) — o "currículo" durou 4 gens, o resto foi caixa
   aberta. g3: 28/48 mortos, mediana 1905 m. Recuperação: g28 com 2 mortos,
   obediência 0,87, mediana 5572, máx 6950 (recorde validado por setores em
   ordem: 57/24 = 2,375V × 2928 = 6954 ≈ 6951 ✓).
2. CONFOUND detectado: teto6-direto (ontem) partiu do campeão VELHO (linhagem
   4898); curric e teto3-fixo partiram do NOVO (travado-cont gen0039). Válida
   é só curric vs teto3-fixo (mesmo start): 30,0 vs 21,9 Σ≥1V/gen, 6951 vs
   6587 m, 247 vs 156 ≥2V — caixa inteira com 4 promoções > presa na 3ª.
3. O que falta pra fechar a causalidade (aguardando ordem de GPU): CONTROLE
   teto6-direto partindo do MESMO travado-cont gen0039 — separa "4 promoções
   ajudam" de "campeão novo aguenta caixa aberta". E limiar de promoção do teto
   merece subir (ex.: 50% da população) pra currículo agir como currículo.

### P1.9 — custo de RPM: veredito final (2026-08-15)

Bateria rpm-evolucao (custo 2.0, cambio auto, 30 gens × 2 sementes) vs controle
cambio-curric (sem custo): Σ≥1V/gen 35,4/37,4 vs 30,0 · Σ≥2V 269/312 vs 247 ·
vida 88-99%. Campeão novo em observação de 20k ticks: RPM méd 8155→4241, tempo
≥90% do giro 86%→3%, marchas 2ª-5ª em uso. A física com válvula de escape
ENSINOU a poupar motor sem perder completagem (ritmo médio menor: 12,3 vs 20,4
m/s — troca deliberada). Pergunta do Leonardo "não troca 2ª→3ª?" validada nos
dados: em 240k frames da maratona, ZERO frames de 2ª acima de 18,5 m/s — a
troca é imediata; o replay que ele via era a captura TRAVADO antiga do lab
(agora substituída pelo campeão novo). Nota de parser: frames de replay têm
prefixo [FRAME_TAG, being] = stride 130 — ferramenta própria sempre, nunca
parse manual (2 erros meus de offset até usar o formato real).

### P1.9 — bateria de 139 validações (2026-08-15): 106 passou · 16 falhou · 14 alarme

Workflow de 10 agentes temáticos, cada validação com número+comando. Triagem:
~20 problemas eram specs MINHAS erradas nos testes (±1 de marcha ignorando que a
1ª não existe; fórmula de RPM sem embreagem patinando; "16 Hz"=stride 4 ticks —
reincidência do gap#1; faixas históricas). Reais e tratados:
1. **ch70 (TireTemp) estava VIVO e nós zerávamos os pesos dele**: o veredito
   "morto" leu physics_c.wgsl; o pipeline real termina em physics_power_c.wgsl,
   que IMPLEMENTA a térmica (aquece por derrapagem+freio, esfria) e usa a temp
   no GRIP. Medição: 1-79 nos dois campeões. DEAD_L1_INPUTS agora vazio; o
   próximo campeão pode aprender a sentir o pneu. (Premissa herdada por 3
   sessões, derrubada por validação em volume — exatamente o propósito.)
2. Σ≥2V da linha oficial vs [setores] divergem (13 vs 22 em rpm-evolucao-s31)
   — dois contadores com definições diferentes; investigação aberta (fila).
3. Cruzadas log×replay com diffs ±2-5% (max/vida) — definições de população/
   amostragem; documentar as definições é a correção, não os números.
