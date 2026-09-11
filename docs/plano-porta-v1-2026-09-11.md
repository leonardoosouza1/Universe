# Porta do táxi para Rust — plano v1 (11/09/2026)

Revisão do `plano-mundo-taxi.md` depois do v0 (T1–T7) e da pred 174 ✗. Objetivo: motor Rust **100 % do mundo que o lab usa hoje**, com portões, replay e métricas, pronto para o primeiro teste de verdade (bateria do protocolo C5 evoluída em Rust e julgada em Rust **e** em torch).

## 0. Vale a pena agora? Sim — mas não é "portar", é "terminar"

O que existe (medido 10/09, `race-world-v3`, 94 testes):

| item | estado |
|---|---|
| tique completo na CPU (`tick.rs`, 4 threads) | pronto; P0–P3 verdes (erro 2.4e-7 nos sentidos, 100 % das ações, `t_vida`/`rec` bit a bit) |
| GA + juiz + CLI `taxi_evolve` com JSON no esquema do lab | pronto; `analise_campo.py` lê a saída |
| velocidade | **1 geração P128 R4 T7200 em 0,108 s** (torch: ~20 s por ser-geração; ≈ 200–450×) |
| P4 (respawn ligado, distribuições) | **falhou**: +9.6 % fit, +12 % rec, +16 % n_come, +31 % oficina, panes iguais |
| cobertura do mundo do lab | **parcial**: NF 29 (sem retina), só `Linear` (NP 40), cenário lido só em `mundo.dias/horizonte/T_max` e `protocolo` |
| replay | não existe (`StepRecord` guarda F/nota/escolha por tique só para o portão) |
| WGSL v1 | não começou |

Por que vale: a régua do lab é n. Com n = 5, DP .47 e MDE .78 — metade dos efeitos medidos esta semana fica abaixo disso. Rust na CPU já dá n = 20 pelo preço de n = 5, sem tocar na GPU (que continua sendo o gargalo de lançamento do torch). **WGSL não é pré-requisito**: fica para depois do primeiro teste, só se o tique com cérebro em CPU ficar abaixo de 20× o torch.

Por que ainda não está pronto: (1) P4 vermelho — o motor não pode entrar em análise com +10 % sistemático; (2) sem retina, sem custo por ativação, sem distritos/preços do cenário, sem `SerCompleto`, ele roda **outro mundo**, não o A/B/C/D; (3) sem replay não há como olhar.

**Suspeito nº 1 do +10 %** (a auditar antes de qualquer outra coisa): o CLI ignora `cenario.corpo` (`guincho`, `kcal_pane`, `fuel_pos_pane`, `dinh0`, `custo_*`), `distritos`, `precos`, `passageiros.*`. Se o Python rodou com o cfg do cenário e o Rust com os defaults do `Config`, +31 % de oficina e +16 % de comida são exatamente o que uma pane mais barata produz. Conferir com `proveniencia.diferencas(cfg_manifesto, Config)` antes de caçar RNG.

Regra de CPU (17/08, "CPU banida para experimentos"): o espírito era velocidade; Rust na CPU é 200× o torch na GPU. Proposta: liberar Rust-CPU com `nice -n 5` e `--threads` ≤ 80 % dos núcleos livres (12 de 16 sem bateria torch; 4 com). **Decisão do Leonardo.**

## 1. Definição de pronto do v1

1. Portões P0–P4 verdes contra o Python, com o cfg efetivo do cenário (não o default).
2. Cada lei do lab em Rust com portão "desligada ⇒ idêntico ao v0" e "ligada ⇒ distribuição a ≤ 3 % do Python": retina, custo por ativação/basal, distritos + preços, passageiros com respawn/ponto.
3. `SerCompleto` e as ablações (`sem córtex`, `sem hipocampo`, `sem reflexo`, `talamo fixo`, `cortex 0.1`, `hip 0.1`) com portão "genes 0 ⇒ Linear" e portão de traço contra um campeão COMPLETO do Python.
4. Replay em arquivo no esquema do `retrato_vida.json` + visualizador HTML; métricas no esquema do `e43_*.json` **mais** as do §4.
5. Determinismo: mesma semente ⇒ mesmo JSON, byte a byte, 4 e 12 threads.
6. Primeiro teste de verdade (§5) executado, com predição registrada antes.

## 2. Tarefas (Sonnet em worktrees paralelas, como T1–T7; Opus só na spec do cérebro)

| id | o quê | depende | portão |
|---|---|---|---|
| **V1** (Python, lab) | `exporta_mundo.py --cenario X --out dir/`: os R mapas de evolução + 3 de teste com custo_tex, centros, serviços, pmul/aliq/alug por distrito, demanda, perfis; e o **cfg efetivo** (o mesmo `manifesto.cfg` do `proveniencia`). `traco_referencia.py` ganha `--cenario`, `--braco`, `--respawn` (P4) | — | JSON validado por `testes_proveniencia`; roda pela fila `nmd` |
| **V2** | `Config::from_cfg(manifesto.cfg)`: TODO campo de `CfgCampo` que o tique usa mapeado, com `#[deny(unused)]` de campo; erro se faltar chave | V1 | teste: `Config::from_cfg(cfg_A) != Config::default()` nos 9 campos que o cenário A muda; `diferencas()` vazio contra o manifesto |
| **V3** | P4: 128 cópias × 3 mapas × sementes 7777–7781, respawn 20, T 7200, cfg do cenário; distribuições de fit, t_vida, rec, n_come, n_posto, litros/abast, oficina, panes; tolerância 3 % na média, KS p > .05 | V2 | vermelho bloqueia V4–V9 na fila |
| **V4** | retina: 8 raios, `R_dia/R_noite/passo/k_cego`, sobrescreve `_dist_v` (campo.py 292–341, 61 linhas); NF 29 → 45 | V2 | retina desligada ⇒ bit a bit com v0; ligada ⇒ P4 com retina |
| **V5** | custo: `custo_modo` existência/ativação/disparo, `custo_basal`, `custo_gene`, `ativ_centrada`, `amostra_ativ`; `metab` nos canais fome/água/calor; `dof`, `dof_bloco`, `ativacao_bloco` | V2 | custo 0 ⇒ idêntico; `metab` e `dof` do campeão COMPLETO do C5 a ≤ 1 % do JSON Python |
| **V6** | `Cerebro` + `SerCompleto` (campo.py 784–828 + completo.py, ~200 linhas torch): tálamo (portões NG), córtex (rápida h / lenta z), hipocampo (MEM por distrito, K caixas), reflexo, livre; ablações por `cerebro` overrides; genoma NP 765 com `nomes_genes` | V5 + spec Opus (`docs/72-spec-cerebro.md`, 1 ciclo) | genes 0 ⇒ Linear; traço de um campeão COMPLETO (V1 `--braco`) com 100 % das ações; P4 do COMPLETO ≤ 3 % |
| **V7** | replay: `--replay k` grava a cada k tiques pos, ação, corpo (fuel/fome/água/dinh/vida/calor/fad), eventos (posto/come/entrega/pane/oficina), contribuição por bloco (DIRIGE/DESLOCA), portões do tálamo, \|MEM\| e caixas tocadas — **mesmo esquema do `retrato_vida.json`** para reusar `resumo.py retrato` | V6 (campos de cérebro opcionais) | `resumo.py retrato` lê sem erro; tamanho ≤ 1 MB por ser-vida em k = 2 |
| **V7b** | `replay.html`: página estática que carrega o JSON: mapa com custo/serviços/distritos, caminho por ser com cor por ação, curvas do corpo, eventos, barras de contribuição por bloco, caixas de memória por dia | V7 | abre o retrato C5 (Python) e um replay Rust sem mudar código |
| **V8** | métricas do §4 no JSON do juiz + `--threads N` (`std::thread::scope`, N = flag, default 4) + teste de determinismo | V3 | JSON byte a byte igual em 4 e 12 threads |
| **V9** | **primeiro teste de verdade** (§5) | V3–V8 | vereditos no `PREDICOES.md` |
| WGSL v1 | tique em wgpu compute, padrão `gpu_evolve` | V9 | só se ms/tique com cérebro na CPU < 20× o torch |

Paralelismo: V1 (eu, Python, fila) → V2+V3 (um Sonnet) → V4, V5, V6, V7/V7b em quatro worktrees → V8 → V9. Regras dos agentes: worktree nasce do HEAD de `race-world-v3` com cwd = Universe; `nice -n 10`, `CARGO_BUILD_JOBS=4`; nunca Fable; cada tarefa entrega testes + relatório final com os números que imprimiu.

## 3. Ordem de risco (o que pode matar)

1. **P4 continua vermelho depois do V2.** Aí é RNG (ordem dos sorteios de respawn/passageiro), acumulação f32 vs f32-torch, ou `serv_need` com `paga`. O traço v2 com respawn ligado (V1) pega em que tique diverge. Orçamento: 1 dia; se passar, para e reporta.
2. **Cérebro em torch tem semântica escondida** (broadcast, `clamp`, `tanh`, ordem de atualização h/z, MEM por distrito). Por isso a spec Opus antes do V6, com linha do Python por operação, como a docs/71.
3. **Mapa**: o Rust não gera mapa (torch.rand + conv2d). V1 exporta; o portão do gerador Rust (estatístico) fica fora do v1.
4. **Determinismo com threads**: partição por genoma, RNG por ser semeado de (semente, índice) — já é a regra do T7; V8 só testa.

## 4. Métricas que o Rust passa a dar (além do esquema `e43_*.json`)

Por ser-vida (no juiz e no replay): dias vividos, causa da morte (fome/sede/dívida/dano), entregas, receita, panes, oficina, litros por abastecimento, dinheiro mín/máx/fim, dívida máxima, fração do tempo em cada ação (mover/ficar/serviço), fração do mapa visitada (células de 10 m), distritos tocados, caixas de memória tocadas, custo metabólico acumulado por bloco, contribuição média por bloco em DIRIGE e DESLOCA, gates do tálamo (média e entropia), tempo em marcha lenta, distância percorrida.

Por bateria: tudo por semente (nunca só média), `ger_efetiva`, curva de melhor por geração, diversidade da elite, `config_hash`, `manifesto` (mesmo formato do lab), tempos separados (evolução/juiz), threads usadas, versão do crate.

Por comparação (`resumo.py pareado`, Python): Δ pareado por semente, t, p, MDE, `ger_efetiva` igual, diff de cfg.

## 5. O primeiro teste de verdade (V9)

Protocolo C5 exato (`mundoC_cidade_da_vida.json`: juntos, obs-rápido, corte .05, P 128, G 30, pac 15, R 4, 5 sementes), 5 seres (linear, COMPLETO, sem hip, sem córtex, sem reflexo), evoluído em Rust.

Predições a registrar antes (≥ 186):
- (a) tempo total ≤ 1/20 do torch (C5: 6574 s) → ≤ 330 s.
- (b) o campeão Rust de cada ser, julgado em **torch** (`julga_politica --braco`, fila `nmd`), fica a ≤ 5 % do campeão torch do C5 por ser.
- (c) o juiz Rust e o juiz torch, sobre o MESMO campeão, diferem ≤ 3 % (é o P4 aplicado ao produto).
- (d) ordem dos seres preservada: linear > sem córtex > resto.

Se (b) e (c) ✓: o lab migra as baterias para Rust; n = 8 na bateria final da tese. Se ✗: o motor volta ao portão, o lab continua em torch. Nada de conclusão científica sai do Rust antes disso.

## 6. Estimativa

| etapa | quem | tempo |
|---|---|---|
| V1 | eu (Python, fila) | 0,5 dia |
| V2 + V3 | 1 Sonnet | 1 dia (P4 pode custar +1) |
| spec cérebro | Opus | 0,5 dia |
| V4, V5, V6, V7/V7b | 4 Sonnets em paralelo | 1–2 dias |
| V8 + V9 | 1 Sonnet + eu | 1 dia |
| **total** | | **4–5 dias**, P4 sendo o único item sem teto garantido |
