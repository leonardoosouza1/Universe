# PRD — `taxi-world` v1: o mundo do táxi completo em Rust

Guia de desenvolvimento para agentes em paralelo (Opus/Sonnet/Astra). Cada parte é independente o bastante para um agente pegá-la sozinho, com a fonte da verdade em linhas do Python, os arquivos que cria, a interface que expõe e o teste que a aceita. Nada aqui é opinião: quando o Python e este documento discordarem, **o Python manda** e o agente anota a discrepância no relatório final.

Repositório: `Universe`, branch `race-world-v3`, crate `crates/taxi-world` (T1–T7 mergeados em 10/09, 94 testes). Lab de origem: `~/projects/LLM/neuromod-lab` (`src/campo.py`, `src/completo.py`, `src/e43_campo.py`). Spec v0: `neuromod-lab/docs/71-spec-mundo-taxi.md`. Plano: `Universe/docs/plano-porta-v1-2026-09-11.md`.

---

## 0. Por que v1 e o que "pronto" significa

O v0 reproduz o Python bit a bit no tique básico (P0–P3 verdes: 29 sentidos com erro 2.4e-7, 100 % das ações, `t_vida`/`rec` iguais) e faz **1 geração P128 R4 T7200 em 0,108 s na CPU** (torch na GPU: ~20 s). Mas ele roda **outro mundo**: sem retina, sem custo por ativação, sem `SerCompleto`, e o CLI ignora as leis do cenário (`corpo.*`, `distritos`, `precos`, `passageiros.*`). Por isso a pred 174 falhou (+9.6 % fit, +31 % oficina com o mesmo genoma) e o Rust ainda não pode entrar em análise.

**v1 pronto =**
1. P0–P4 verdes com o **cfg efetivo do cenário** (não o `Config::default()`).
2. Retina, custo por ativação/basal, distritos+preços, respawn/ponto: cada um com portão "desligado ⇒ bit a bit igual ao v0" e "ligado ⇒ distribuições a ≤ 3 % do Python".
3. `SerCompleto` (NP 765) e ablações com portão "genes 0 ⇒ Linear" e portão de traço contra um campeão COMPLETO do Python.
4. Replay em arquivo no esquema do `retrato_vida.json` + `replay.html`; métricas no esquema do `e43_*.json` + §4 do plano.
5. Determinismo: mesma semente ⇒ JSON byte a byte igual, 4 e 12 threads.
6. Primeiro teste de verdade (V9) rodado com predição registrada antes.

---

## 1. Regras para todo agente (não negociáveis)

1. **Worktree** nasce do HEAD de `race-world-v3` com `cwd = Universe` (worktree do Agent que nasce de outro cwd cai em branch morta). Branch `taxi/V<n>-<slug>`. Antes de começar: `git merge --ff-only race-world-v3`.
2. **Sem GPU.** Nada de torch, nada de `python` que importe `campo.py` (inicializa CUDA). Parte que precisa do Python roda **pela fila `nmd`**: escrever o pedido JSON em `/media/llm-data/universe-runs/neuromod-lab-taxi/_nmd/pedidos/<id>.json` e esperar `feitos/<id>.json` + `log/<id>.log`. Nunca `pgrep -f e43_campo`.
3. **CPU com folga**: `nice -n 10 cargo ...`, `CARGO_BUILD_JOBS=4`, testes com `--test-threads=4`. Bateria torch na GPU é limitada por CPU de lançamento; não roubar.
4. **Saídas de dados** só em `/media/llm-data/universe-runs/neuromod-lab-taxi/` (Rust: subpasta `rust/`). Nunca `/tmp`, nunca o disco do sistema. O disco está a 90 %: declarar quanto vai gravar; replay > 100 MB é bug.
5. **Um cérebro, um portão.** Nenhuma parte é aceita sem o teste que a compara ao Python (bit a bit onde o v0 é bit a bit; distribuição a 3 % onde há RNG). Teste verde = exit 0 **e** a última linha diz `ok`.
6. **Não inventar.** Onde o Python tem uma constante, o Rust tem a mesma, com o comentário `// campo.py:<linha>`. Onde a ordem de operações importa (SPEC §3), a ordem é a do Python.
7. **Sem dependência nova** no `Cargo.toml` sem justificativa no relatório (v0 usa `serde`, `serde_json`, `thiserror`; threads via `std::thread::scope`). `rayon` só na V8, se o teste de determinismo passar.
8. **Relatório final** (obrigatório, no PR/commit e no chat): o que implementou, arquivos, contagem de testes antes/depois, os números que os testes imprimiram, o que ficou fora e por quê, discrepâncias Python × PRD. Sem número = não está pronto.
9. Modelo: nunca Fable em subagente. Sonnet para implementação; Opus para spec e síntese (Leonardo pode escolher Opus para tudo — decisão dele).

---

## 2. Mapa do crate (o que já existe; não reescrever)

| módulo | responsabilidade | API pública que as partes usam |
|---|---|---|
| `constants.rs` | `NF = 29`, `S = 10`, `G = 9`, `NP = 40`, `DIRS`, `Config` (campos com `// campo.py:<linha>`) | `Config`, `NF`, `NP`, `S`, `G` |
| `genome.rs` | `Genome(Vec<f32>)` NP floats na ordem `NOMES_F`; arquivo TAXG | `Genome` |
| `trace.rs` | carrega `traco_ref.json` (mapa, genoma, estado inicial, trajetória) | `Trace::load`, `Trace::to_world` |
| `world.rs` | mapa imutável: custo, distritos, serviços, `pmul/aliq/alug` | `World::{terrain_cost_at, district_at, nearest_services}` |
| `beings.rs` | estado SoA por ser (`campo.py:169-187`) | `Beings::from_trace` |
| `physics.rs` | as 35 operações do `passo` (`campo.py:400-483`) | usado por `tick` |
| `senses.rs` | `F` (B,G,NF) (`campo.py:285-330`, obs rápido) | `observe`, `compute_target` |
| `decision.rs` | nota, centragem G1, máscara de serviços, argmax (`campo.py:485-497`) | `decide` |
| `services.rs` | `serv_ab`/`serv_need`/`custos` (`campo.py:372-398`) | usado por `tick` |
| `passengers.rs` | embarque, entrega, tarifa, respawn com `TaxiRng` (`campo.py:443-473`) | `TaxiRng` |
| `tick.rs` | `Sim` + `step()` na ordem do `um_passo`; `StepRecord` | `Sim::{new, from_trace}`, `step` |
| `gate.rs` | portões P0–P3 contra o traço | `run_gate` |
| `evolution.rs` | GA (elite 15 %, σ_mut, corte, janela), `Simulador` trait, `Champion` | `evolve`, `GaParams` |
| `judge.rs` | 128 cópias, `LifeSample`, `Measures`, JSON no esquema do lab | `judge`, `measures_to_json` |
| `sim_cpu.rs` | `SimCpu`: `Simulador` + `LifeSampler` sobre `tick`, 4 threads | `SimCpu` |
| `taxi_evolve_cli.rs` | `taxi_evolve --cenario --mapa --campeao [--saida --tag --seeds --P --G --R --sigma-mut --paciencia]` | `run` |

Testes: `tests/identity.rs` (P0–P3), `tests/physics.rs`, `tests/gate_tick_zero.rs`, `tests/evolve_smoke.rs` (+ medição `#[ignore]`), fixtures em `tests/fixtures/tiny_trace.json`.

---

## 3. As partes

Grafo: **V1 → V2 → V3 (portão que libera o resto)**; depois **V4, V5, V6, V7 em paralelo**; V7b depois de V7; **V8** depois de V3 (pode andar junto com V4–V7); **V9** por último. Spec do cérebro (`docs/72-spec-cerebro.md`, Opus) pode ser escrita já, em paralelo com V1–V3.

### V1 — Exportador de mundo e traços novos (Python, `neuromod-lab`)

**Objetivo.** O Rust não gera mapa (`gerar_mapa` = `torch.rand` + `conv2d`, SPEC §2.2). Hoje só existe um mapa exportado (L 200, default). Precisamos dos mapas e do cfg **de cada cenário**, e de traços com respawn ligado e com cérebro COMPLETO.

**Fonte da verdade.** `src/traco_referencia.py` (25 linhas, exporta mapa + genoma + estado + trajetória); `e43_campo.py:43-64` `aplica_cenario`, `:66-` `cfg_de_flags`; `politica_util.mundo()` (`politica_util.py:37-49`: cfg do cenário + 3 mapas de teste `gerar_mapa(1000+s)`); `proveniencia.manifesto` (cfg efetivo serializado, o mesmo do `manifesto.cfg` dos JSONs de bateria).

**Entrega.**
- `src/exporta_mundo.py --cenario X.json --out DIR [--R 4]`: grava `DIR/mundo.json` com:
  - `cfg`: `vars(cfg)` normalizado por `proveniencia.normaliza` (TODAS as chaves de `CfgCampo`, `campo.py:36-63`, mais `retina`, `regime`, `seres` como no manifesto);
  - `ambiente`: `NM_RISCO`, `NM_PONTO`, `NM_RETINA`;
  - `nomes_f`, `nomes_op`, `nomes_corpo`, `acoes`, `tipos`, `KR`, `retina_padrao`;
  - `mapas_evolucao[R]` e `mapas_teste[3]`: cada um com `custo` (tex×tex), `opac` e `tipo_g` (as grades que `_retina` lê, `campo.py:357-371`), `centros`, `serv{tipo: [[x,y]]}`, `serv_msk`, `pmul`, `aliq`, `alug`, `demanda`, `perfis`, `seed_mapa`, `L`, `K`;
  - `cenario_hash`, `config_hash`.
- `src/traco_referencia.py` ganha `--cenario` (usa `mundo()`), `--braco` (rótulo de `BRACOS`, via `politica_util.braco`; grava `W` do braço e `ser`), `--respawn` (float; default 1e9 = identidade; 20 = P4), `--T`, `--seed`. Com `--braco COMPLETO...` grava também por tique: `h` (B,G,H), `z` (B,K), `MEM` (B,DMAX,NM), `gate` (B,NG), `c` (B) do ser — o que o portão do cérebro compara.
- Pedidos na fila: `v1a_mundo_A`, `v1b_mundo_C`, `v1c_mundo_D`, `v1d_traco_C_linear_respawn20` (T 7200, B 128), `v1e_traco_C_completo` (respawn 1e9, B 8, T 2400, campeão COMPLETO do `e79/e43_mundoC5.json` semente 0).

**Aceite.** `src/testes_exporta.py` (sem GPU: `NM_COMPILE=0`, roda pela fila como portão): `mundo.json` carrega; `cfg` tem as mesmas chaves e valores do `manifesto.cfg` do `e79/e43_mundoC5.json` braço linear (`proveniencia.diferencas` vazio); `mapas_teste` tem 3 itens com `serv` não vazio para os 5 tipos; `allow_nan=False`.

**Fora.** Gerar mapa em Rust.

### V2 — `Config` completo a partir do cfg efetivo

**Objetivo.** Matar a causa provável do +10 %: o CLI só lê `mundo.dias/horizonte/T_max` e `protocolo`. Todo campo de `CfgCampo` que o tique usa passa a vir do `mundo.json` (V1) — e o mapa também.

**Fonte da verdade.** `campo.py:36-63` (`CfgCampo`); `constants.rs` `Config` (campos já existentes com linha do Python); `e43_campo.py:66-` (`cfg_de_flags`).

**Entrega.**
- `src/config_lab.rs`: `pub fn config_from_lab(cfg: &serde_json::Value) -> Result<Config, ConfigError>`; `#[deny(missing_docs)]`; para cada campo de `Config` uma chave do cfg; chave ausente = erro (não default silencioso); chave presente no cfg e **sem** campo em `Config` = erro `Unmapped(key)` **se** estiver na lista `USED_BY_TICK` (todas as que `campo.py:400-483`, `285-330`, `372-398`, `443-473` leem) — as demais vão para `ignored: Vec<String>` no retorno, impressas pelo CLI.
- `Config` ganha os campos que faltam (mínimo): `GUINCHO`, `FUEL_POS_PANE`, `KCAL_PANE`, `D_PANE`, `dinh0`, `fome_passo`, `amp_preco`, `periodo_preco`, `tarifa_base`, `tarifa_km`, `aluguel`, `LIMITE_DIVIDA`, `respawn`, `ponto`, `pax_cidade`, `raio_px`, `NPX`, `w_trab`, `fitness`, `horizonte`, `T_max`, `corte_vivos`, `loja_no_posto`, `custo_gene`, `gene_min`, `custo_suave`, `custo_basal`, `custo_modo`, `amostra_ativ`, `ativ_ref`, `limiar_disparo`, `ativ_centrada`, `custo_canais`, `retina: Option<Retina>`, `k_cego`, `nevoa`, `margem_risco`, `talamo_norm`, `cortex_amp`, `hip_esc`, `temp_decisao` (este último: se > 0, erro `Unsupported` — fica fora do v1).
- `trace.rs`/`world.rs`: `World::from_lab_map(&Value)` lê o formato do `mundo.json` (inclui `opac`, `tipo_g`, `demanda`, `perfis`).
- `taxi_evolve_cli.rs`: `--mundo DIR/mundo.json` substitui `--mapa` (mantém `--mapa` para o traço v0); imprime as chaves ignoradas.

**Aceite.** `tests/config_lab.rs`: (a) `config_from_lab(cfg_A)` difere de `Config::default()` exatamente nos campos que o cenário A muda (lista fixa no teste, ≥ 9); (b) `config_from_lab(cfg_do_manifesto_C5)` sem erro e `ignored` só com chaves de protocolo; (c) chave usada pelo tique faltando ⇒ `Err`.

**Fora.** `temp_decisao`, `regime`, `reproducao`, `seres` (lote misto), `jardim_comum` — erro explícito se ligados.

### V3 — Portão P4 (respawn ligado, distribuições)

**Objetivo.** O único portão que falta para o Rust valer como motor: com RNG próprio e cfg do cenário, as distribuições batem com o Python a ≤ 3 %.

**Fonte da verdade.** `plano-mundo-taxi.md` "10/09 16:20"; `campo.py:628-650` `medidas` (128 cópias, `jardim_comum=False`, `corte_vivos=0`, `medir_ativ=True`); `campo.py:604-620` `_fit` (`fitness='vida'`: `t_vida/240 + w_trab*rec/300`).

**Entrega.**
- `src/gate.rs`: `pub fn run_p4(mundo: &LabWorld, genome: &Genome, python: &P4Reference, tol: f32) -> P4Report` — roda 128 cópias × os 3 `mapas_teste` × sementes `7777..=7781`, T 7200, respawn do cfg; compara **média** e **KS** (implementação própria, 40 linhas) de `fit, t_vida, rec, n_come, n_posto, litros_por_abast, oficina, panes, dinh_fim` com o JSON do Python (`v1d` do V1 dá o traço; a referência de distribuição vem de um pedido `julga_politica`-like: pedido `v3_ref_C_linear` = `e43_campo.py --so linear --G 0 --P 1 ...` ou o `linear_campeao` de `e85/julga_parada_C.json`, que já é exatamente isso: 3 mapas × 128 cópias × 5 sementes, `mede()`).
- `tests/p4.rs`: `#[ignore]`d (release, ~1 min), imprime a tabela; verde se todas as médias ≤ 3 % e KS p > .05.
- Se vermelho: `run_gate` P3 sobre `v1d` (traço com respawn 20) aponta o **primeiro tique** de divergência e o campo (`pd_`, `px_ok`, `dinh`...). Relatório obrigatório com esse tique. Orçamento: 1 dia; se não fechar, para e reporta — não "ajustar até bater".

**Aceite.** P4 verde no C com o cfg do cenário. **V3 vermelho bloqueia V9** (V4–V7 podem continuar, mas não são aceitas até V3 verde).

### V4 — Retina (8 raios)

**Objetivo.** Todas as baterias A/B/C/D rodaram com `NM_RETINA=1`. Sem retina o Rust vê outro mundo (onisciente).

**Fonte da verdade.** `campo.py:26` `KR = 8`, `:28` `RETINA_PADRAO`, `:357-371` `_retina` (raios amostrados a cada `passo` até `R_dia/R_noite`; célula opaca ou borda bloqueia; devolve `ret_d`, `ret_t`, `livre`, `Rd`); `:292-301` (obs rápido) e `:329-341` (obs lento): substitui `_dist_v` pelo que o raio viu, `cego = k_cego*R`, `Fs` por opção clampado em `cego`; `:303` coluna `livre/100` por opção (ficar = 0); `:310` 16 colunas de corpo `ret_d/100` e `ret_t/5`. NF passa de 29 para **45** (`nomes_f` do JSON do lab: `ret_d_0..7`, `ret_t_0..7` + a coluna `livre` por opção entra em `NOMES_OP`, conferir contra `mundo.json.nomes_f`).

**Entrega.**
- `src/retina.rs`: `pub fn retina(world, cfg, pos: &[[f32;2]], night: &[bool], out: &mut RetinaOut)` com `RetinaOut{ret_d: [f32;8], ret_t: [u8;8], livre: [f32;8], r: f32}` por ser.
- `senses.rs`: `NF` deixa de ser constante fixa: `pub fn nf(cfg) -> usize` (29 sem retina, 45 com); `observe` recebe `Option<&RetinaOut>`; `genome.rs` valida NP contra `nf(cfg)`.
- Grades `opac`/`tipo_g` do `mundo.json` (V1) em `World`.

**Aceite.** `tests/retina.rs`: (a) `retina: None` ⇒ `F` bit a bit igual ao v0 nos 29 sentidos (roda `identity` P0–P3 inalterado); (b) traço `v1f_traco_C_linear_retina` (pedido novo, B 8, T 2400, respawn 1e9, retina do C) ⇒ P0–P3 verdes com NF 45; (c) P4 com retina no C (mesma referência `e85/julga_parada_C.json`, que já tem retina).

### V5 — Custo por ativação, basal e `dof`

**Objetivo.** A lei central da tese ("a parte só paga quando o mundo exige") é `custo_modo='ativacao'` + `custo_basal`. Sem ela o cérebro em Rust não paga nada.

**Fonte da verdade.** `campo.py:552-575` (`rodar`: `basal`, `metab` inicial, `ativ_soma` por bloco a cada `amostra_ativ` tiques a partir de t > 0, `metab = 1 + basal + custo_gene*Σ n_b[b]*ativ_soma[b]/ativ_n/100`); `:581-588` `dof_de` (genes com |w| > `gene_min` além de `Linear.NP`; `custo_suave` opcional); `:589-603` `BLOCOS`, `dof_por_bloco`; `:420-422` onde `metab` entra (calor, água se em `custo_canais`, fome sempre); `Cerebro._mag_blk` (`:812`: crua ou centrada nas 9 opções se `ativ_centrada`), `ativ_blocos` (`:817`); `completo.py:260-264` `_blk` por bloco.

**Entrega.**
- `src/metabolism.rs`: `Metab{basal, gene_cost, mode: Existence|Activation|Firing, sample_every, ref, threshold, centered}`; `pub fn update(&mut self, t, blk: &BlockOutputs, n_genes_per_block)`; `pub fn dof(genome, cfg) -> f32`; `pub fn dof_per_block`.
- `physics.rs`: `metab` multiplica `fome_passo` sempre, `agua_passo` e `calor_sobe` conforme `custo_canais` (`campo.py:420-422`).
- `judge.rs`: `Measures` ganha `dof`, `metab`, `dof_bloco`, `ativacao_bloco` (`campo.py:634-646`).

**Aceite.** `tests/metabolism.rs`: (a) `custo_gene = 0` ⇒ tique bit a bit igual ao v0; (b) com o genoma COMPLETO da semente 0 do `e79/e43_mundoC5.json`: `dof` = `dof_por_semente[0]` do JSON (175.0) exato; `metab` médio do juiz a ≤ 1 % de `metab_por_semente[0]` (1.48) — depende da V6 para o `_blk`; até lá, teste com `BlockOutputs` sintético igual ao do traço `v1e`.

### V6 — `Cerebro` + `SerCompleto` + ablações

**Objetivo.** O objeto da tese. Sem ele o Rust só evolui a espinha.

**Fonte da verdade.** `campo.py:784-828` (`Cerebro`: rápida H=4 relu sobre F por opção; lenta K=3 relu sobre o corpo com `z` EMA em banco `tau` log-espaçado; `ESC=.01`, `ESC_S=.1`; offsets `O_U..O_P`, NP 344); `completo.py` inteiro (152 linhas): grupos do tálamo por nome (`GRUPOS`, `_IG`), `IDX_CORTEX=[0,1,6,7]`, `DMAX=12`, `NM=3`, offsets `O_REF/O_TAL/O_COR/O_HIP`, NP 765 (com NF 45; com NF 29 é outro NP — o Rust calcula dos offsets, não hardcoda); `reset` (`:210-235`, inclusive `liga` das ablações e `talamo_norm` teto/fixo); `_gate` (`:238-244`); `_mem` (`:246-248`); `extra` (`:249-267`, ordem exata das somas em `Wtot`, `_blk` por bloco); `extra_serv` (`:268-273`); `atualizar` (`:274-286`: `z`, `c` do reflexo, `MEM` com `oh_o`/`oh_n`, `gate_soma`, `ref_soma`); `ativacao` (`:287-294`); `nomes_genes_*` (`:304-321`); `um_passo` (`campo.py:485-497`: `extra` entra na nota antes da centragem G1; `extra_serv` soma em `ss` antes da máscara). **Spec Opus** `neuromod-lab/docs/72-spec-cerebro.md` com uma linha por operação, shapes e ordem — escrever antes de codar, no formato da docs/71.

**Entrega.**
- `src/brain/mod.rs`: `pub trait Brain { fn reset(&mut self, cfg, genome_row: &[f32], being: usize); fn extra(&mut self, f: &[f32] /*G×NF*/, body: &BodyView, out: &mut [f32; G]); fn extra_serv(&self, out: &mut [f32; S]); fn update(&mut self, body: &BodyView, world_events: &Events); fn blocks(&self) -> Option<&BlockOutputs>; fn np(cfg) -> usize; fn gene_names(cfg) -> Vec<String>; }` — `Linear` implementa com `extra = 0`.
- `src/brain/free.rs` (`Cerebro`), `src/brain/complete.rs` (`SerCompleto`), `src/brain/parts.rs` (`Liga{talamo, cortex, hipocampo, reflexo, livre, rapida, lenta}` + construtores `sem_cortex()`...; overrides `talamo_norm`, `cortex_amp`, `hip_esc` do cfg).
- `tick.rs`: `Sim` genérico em `B: Brain`; `StepRecord` ganha `blk`, `gates`, `mem_abs`, `mem_boxes`, `h`, `z` (opcionais).
- `genome.rs`: TAXG v2 com `NP` variável e `brain: String` no cabeçalho; carregador de `W` do JSON do lab para qualquer braço (`bracos[<rotulo>].W[<seed>]`, alinhamento por `nomes_f` como `alinha_semente`, `e43_campo.py:29-37`).
- `taxi_evolve_cli.rs`: `--braco <rotulo>` (mesmos rótulos de `BRACOS`, `e43_campo.py:39`), `--so` lista, JSON de saída com `ser{nome, cerebro}` por braço.

**Aceite.** `tests/brain.rs`: (a) genoma COMPLETO com todos os genes além de `Linear.NP` em zero ⇒ nota bit a bit igual ao `Linear` (regra 17); (b) traço `v1e` (campeão COMPLETO, respawn 1e9, B 8, T 2400): P0–P3 verdes **e** `h`, `z`, `MEM`, `gate`, `c` a ≤ 1e-4 por tique; (c) cada ablação com `liga` desligada ⇒ igual ao SerCompleto com os genes daquele bloco zerados; (d) P4 do COMPLETO no C (referência: `e85/julga_parada_COMPLETO_C.json`, `linear_campeao` = campeão COMPLETO, 3 mapas × 128 × 5).

### V7 — Replay

**Objetivo.** Olhar a vida. Sem visual Bevy (decisão do dono, 10/09): arquivo no **mesmo esquema** do `retrato_vida.json` do lab, para que `resumo.py retrato` e o `replay.html` (V7b) leiam os dois.

**Fonte da verdade.** `neuromod-lab/src/retrato_vida.py:60-120` (o que grava e como). Esquema:
```
{ "meta": {"bateria","mapa","L","dias","cenario","passo","nomes_g","nomes_op","grupos","motor":"rust","versao"},
  "mapa": {"custo":[[..]],"centros":[[x,y]],"serv":{"posto":[[x,y]],...},"pmul":[..],"aliq":[..],"alug":[..]},
  "seres": [ {"arm","semente","NP","t_vida","ent","panes","dinh",
              "pos":[[x,y]], "acao":[int], "corpo":[[fuel,fome,agua,dinh,vida,calor,fad]],
              "blk":[{"rap":[dirige,desloca],"len":..,"tal":..,"cor":..,"hip":..,"ref":..}],
              "gates":[[g0..g4]], "mem":[[abs,caixas]], "ev":[[idx,"posto"|"come"|"ent"|"pane"|"oficina"]],
              "detalhe": {"h":[[..]],"z":[[..]]} | null } ] }
```
`dirige` = média sobre as 9 opções de |saída − média| (o que o argmax vê), `desloca` = média (retrato_vida.py:96).

**Entrega.**
- `src/replay.rs`: `Recorder::new(every: u32, detail_being: Option<usize>)`, `record(&Sim, &StepRecord)`, `finish() -> ReplayDoc`, `write(path)`.
- `taxi_evolve_cli.rs`: `--replay k` grava o campeão de cada braço/semente após o juiz (uma vida por ser, `mapas_teste[0]`, T = 240·dias) em `<saida>/replay_<tag>.json`; `--replay-seres n` limita.
- Novo binário `taxi_replay --mundo --braco --W <json> --seed --dias --passo --out` para gravar uma vida sem evoluir.

**Aceite.** `tests/replay.rs`: (a) `resumo.py retrato` (quando existir) ou um leitor Rust de teste carrega o arquivo e confere que `len(pos) == len(acao) == len(corpo)` e que os eventos batem com os contadores finais (`ent`, `panes`, `n_come`, `n_posto`); (b) tamanho ≤ 1 MB por ser-vida com `k = 2`, T 1920; (c) mesma semente ⇒ arquivo idêntico.

### V7b — `replay.html`

**Objetivo.** Leonardo pensa visual. Página estática (sem servidor, sem build) que carrega um `replay_*.json` ou `retrato_vida*.json` por `<input type=file>`.

**Entrega.** `crates/taxi-world/replay.html` (um arquivo, JS puro, canvas): mapa (custo em cinza, distritos por `pmul` em cor, serviços por tipo), caminho de cada ser com cor por ação (mover/ficar/serviço) e scrubber de tempo; painel do corpo (7 curvas); eventos como marcas; barras de `blk` DIRIGE/DESLOCA por bloco no tique atual; `gates` do tálamo; caixas de memória tocadas por dia; tabela final (vida, entregas, panes, dinheiro, fração visitada, distritos). Tema claro/escuro.

**Aceite.** Abre `e72/retrato_vida_mundoC5.json` (Python) e um replay Rust sem mudar código; nada externo carregado.

### V8 — Métricas completas, threads, determinismo

**Fonte da verdade.** `e43_campo.py` JSON de saída (chaves por braço: `fit, mortos, ent, panes, fome_fim, agua_fim, dinh, res, litros_por_abast, n_posto, n_come, oficina, vida_vivos, dof, metab, t_vida{media,mediana,p10,p90,frac_teto}, rec, NP, NF, *_por_semente, W{seed}, segundos, ger_efetiva, seeds, sigma_mut, crn, campeao, G, P, R, T_evo, T_aval, tempos{evolucao_s, avaliacao_s, total_local_s}, manifesto, config_hash, ser`; topo: `nomes_f, cenario, cenario_json, cenario_hash, argv, predicao, L, fitness, NF, S, bracos, tempos{preparacao_s, evolucao_conjunta_s, total_s}`); `proveniencia.py` (`manifesto`, `hash_config`, `identidade_codigo`).

**Entrega.**
- `judge.rs`/`measures_to_json`: todas as chaves acima + por ser-vida (no replay e agregadas no juiz): `causa_morte{fome,sede,divida,dano}` (`politica_util.mede`, `:63-64`), `frac_acoes{mover,ficar,servico}`, `frac_visitada` (células de 10 m), `distritos_tocados`, `caixas_mem`, `dinh_min/max`, `divida_max`, `custo_metab_acumulado`, `blk_medio{dirige,desloca}`, `gates_media`, `marcha_lenta_frac`, `dist_percorrida`.
- `manifesto` no JSON: `cfg` (o `Config` de volta ao formato do lab), `protocolo`, `mapas` (seeds), `flags`, `codigo` (`git rev-parse HEAD` do Universe + sha256 dos `.rs` do crate), `ambiente`, `motor: "rust-cpu"`; `config_hash` = sha256 do manifesto normalizado — **mesma função** do Python (`proveniencia.hash_config`), testada contra um manifesto do lab.
- `--threads N` (default 4) em `SimCpu`; partição por genoma; RNG por ser semeado de `(seed, índice)` — já é a regra do T7.
- `tests/determinism.rs`: mesma semente, 4 e 12 threads ⇒ JSON byte a byte igual (release, `#[ignore]`).

**Aceite.** `scripts/analise_campo.py` lê o JSON Rust sem `--legado`; determinismo verde.

### V9 — Primeiro teste de verdade

**Protocolo.** `mundoC_cidade_da_vida.json` exato: P 128, G 30, paciência 15, R 4, sementes 0–4, σ_mut .1, corte .05, `obs_rapido`, 5 seres (linear, COMPLETO, sem hipocampo, sem córtex, sem reflexo), evolução **por ser** (o Rust não tem lote misto; anotar).

**Predições (registrar em `PREDICOES.md` ≥ 186 ANTES de rodar):**
- (a) tempo total ≤ 330 s (C5 em torch: 6574 s).
- (b) cada campeão Rust, julgado em **torch** (`julga_politica --braco <rotulo> --campeao <json rust>`, pela fila `nmd`), fica a ≤ 5 % do campeão torch do C5 do mesmo ser.
- (c) juiz Rust e juiz torch sobre o **mesmo** campeão diferem ≤ 3 %.
- (d) ordem preservada: linear > sem córtex > {COMPLETO, sem hip, sem reflexo}.

**Aceite.** (b) e (c) ✓ ⇒ o lab migra as baterias para Rust e a bateria final da tese roda com n = 8. ✗ ⇒ o motor volta ao portão; nenhuma conclusão científica sai do Rust.

---

## 4. Ordem de lançamento dos agentes

| onda | agentes | pré |
|---|---|---|
| 1 | **V1** (Python, fila) · **Opus spec `docs/72-spec-cerebro.md`** | — |
| 2 | **V2** · **V3** (mesmo agente, sequência) | V1 |
| 3 | **V4** · **V5** · **V6** · **V7** (4 worktrees) | V2 mergeada (V4–V7 não esperam V3, mas só são aceitas com V3 verde) |
| 4 | **V7b** · **V8** | V7 / V3 |
| 5 | **V9** (eu + fila) | tudo |

Cada agente entrega: branch, testes verdes (`cargo test -p taxi-world` e os `#[ignore]` que a parte pede, em release), relatório (§1.8). Merge em `race-world-v3` só com portão verde; conflito se resolve **para o lado do Python**.

## 5. O que fica fora do v1 (e onde entra)

`temp_decisao`, `regime`, `reproducao`, `prefeito`, `seres` (lote misto), `jardim_comum`, geração de mapa em Rust, `varredura='grade'`, WGSL. Cada um é uma parte própria depois do V9, com o mesmo formato: fonte da verdade em linha, portão desligado ⇒ idêntico, ligado ⇒ ≤ 3 %.
