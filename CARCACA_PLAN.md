# Universe × IARA — carcaça da IARA nos seres do Universe (plano de desenvolvimento + execução)

> Levar a ARQUITETURA da IARA (não os pesos) pros seres do Universe. Aproveitar o mundo/evolução/neuromod
> já prontos (branch iara-vida) e ADICIONAR os órgãos IARA que faltam: a FLORESTA (memória associativa em
> grafo) + a PLASTICIDADE CAUSAL dentro-da-vida (dopamina/cortisol gravando sinapse) + o ROTEADOR System1/2.
> Headless-first: valida por CLI (--validate) e JSON (--record → trace.json). Faz→testa rápido→segue.

## O que JÁ EXISTE (reaproveitar — iara-vida)
- Mundo multi-necessidade (comida≠água≠ninho, dia/noite, predadores) — `src/survival.rs`, `src/bin/vida.rs`.
- Neuromoduladores `dopa/corti/curio/adeno` no `Being`.
- Órgão PREDITIVO (world-model `wm` + `pred` + erro na fitness) — já pontua o "andar move a cena".
- CfC recorrente (memória), evolução (GA + DNA save/load), perception/metabolism/memory.
- Headless: `--evolve` (CLI métricas), `--record` (JSON trace por tick), `--validate`, `--watch` (ASCII).

## O que FALTA (a carcaça IARA — validada em Python `iara_carcaca.py`)
1. **FLORESTA associativa** (`src/carcaca.rs`): grafo situação→ação com FORÇA de sinapse; cresce/poda dentro da vida.
2. **Plasticidade CAUSAL** (`_plast`/`_reinforce`): dopamina abre, fadiga fecha; bater→cortisol→grava evitar. Aprende VIVENDO.
3. **Roteador System 1/2 + reflexo:** reflexo rápido vs deliberação preditiva, gated por neuromod.

## Fases (headless, cada uma valida rápido)
- **F0 — esqueleto** *(agora)*: branch + `src/carcaca.rs` (órgãos IARA em Rust puro) + `src/bin/carcaca.rs` (ser num mundo mínimo) → compila e roda, saída CLI/JSON.
- **F1 — aprender vivendo**: valida que o ser aprende DENTRO da vida (evita parede/satisfaz necessidade melhora numa vida só) via a floresta+plasticidade. Métrica CLI + curva JSON.
- **F2 — população + evolução**: milhares de carcaças no mundo real do vida.rs (reusa GA/DNA); valida escala + tese.
- **F3 — órgãos completos**: roteador System1/2, verificador (sabe/abstém), reflexo-byte, floresta como conhecimento navegável.
- **F4 — visual** (depois): o lab Bevy (`--lab`) pra VER; headless primeiro.

## Validação
CLI (fitness, curva de aprendizado-dentro-da-vida, --validate) + JSON (`--record`: floresta crescendo, neuromod, decisão por tick).
Números honestos sempre; a prova é o ser DESCOBRIR o laço ação↔percepção sozinho, agora dentro do Universe.
