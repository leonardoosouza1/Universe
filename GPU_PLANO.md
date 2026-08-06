# Núcleo na GPU — contrato de implementação

Leonardo: *"tudo gpu sem problemas"* · *"bevy seria só um frontend que exibe o que está
acontecendo"*.

Este arquivo é o CONTRATO. Cada agente implementa um pedaço sem tocar no arquivo do outro.

---

## O que já está feito

| item | estado |
|---|---|
| headless sem janela (`env -u DISPLAY`, `backends: None` no Bevy) | ✅ |
| retina em WGSL compute (`src/gpu/retina.wgsl`) | ✅ **31,7×**, equivalência 99,996% |
| retina devolve FÍSICA (distância em metros + material) | ✅ |
| recurso `RetinaGpuRes` persistente entre ticks | ✅ |

## Onde o tempo está agora (2440 amostras de PERF, medido)

| componente | antes | depois da retina |
|---|---|---|
| retina | 57,0% | ~15% |
| física | 26,9% | ~27% |
| cérebro | 14,3% | ~14% |
| percepção | 1,9% | ~2% |

---

## Fatos verificados no código (não confiar em estimativa de agente)

- **Pesos são POR INDIVÍDUO.** `SovereignEngine` (engine.rs:179) é componente de cada ser:
  `instinct_layer` BitLinear 192→96, `cortex_l2` 96→48, `cortex_l3` 48→24, `slow_brain`,
  `cfc_state`, `kan_filters`.
  → Kernel do cérebro = **1 workgroup por ser, lendo sua própria fatia de pesos**. É batched
  matmul com pesos distintos por lote; não é impedimento.
- **Dimensões:** `L1_NEURONS=96`, `L2_NEURONS=48`, `L3_NEURONS=24`, `RETINA_ROWS=14`,
  `RETINA_COLS=126`, `SENSORY_WAVE_SIZE=223`.
- **Mundo:** grade 320×320 (102.400 células), `cell_size` 3 m, lado 960 m. Estático por pista.
- **Física não tem dependência carro-carro** em `physics.rs` — nenhum carro escreve no corpo
  de outro. Só a PERCEPÇÃO tem (drafting ch 72, estresse competitivo ch 125, via hash
  espacial).

### Correções a afirmações erradas dos mapeamentos

- ❌ *"Trinity/RNN não é paralelizável, fica na CPU"* — a recorrência é sequencial no TEMPO,
  mas os 60 seres são independentes. Um workgroup por ser resolve.
- ❌ *"2000 carros"*, *"66× de speedup"* — não existem. Rodamos 60 seres; a retina mediu
  **31,7×** no kernel isolado contra CPU de uma thread.
- ⚠️ Estimativas de ganho dos agentes (12-50×, 22×) são especulação. **Só entra no relatório
  o que for medido pelo PERF da simulação inteira.**

---

## Regra inegociável, em todos os pedaços

Cada kernel tem de dar o MESMO resultado que a CPU, dentro de tolerância declarada. Um caminho
GPU que diverge em silêncio invalida todo experimento já medido. Cada agente entrega o kernel
**e** o teste de equivalência contra a implementação CPU existente.

Tolerâncias:
- inteiros/quantizados (retina, sensory_wave i8): diferença **≤ 1**
- floats de uma passada (força, intenção): **≤ 2 ULP** ou erro relativo ≤ 1e-5
- estado acumulado por N ticks: declarar a deriva medida, não assumir

---

## Divisão de trabalho — SEM sobreposição de arquivos

| agente | arquivos que PODE criar/editar | entrega |
|---|---|---|
| **A · física** | `src/gpu/fisica.wgsl`, `src/gpu/fisica.rs` | kernel de suspensão+pneu+colisão+integração e host |
| **B · cérebro** | `src/gpu/cerebro.wgsl`, `src/gpu/cerebro.rs` | kernel do forward completo (KAN→CfC→L1→L2→gate→mix) e host |
| **C · percepção** | `src/gpu/percepcao.wgsl`, `src/gpu/percepcao.rs` | canais 0-63, 76-110, 128-191 a partir da retina e do mundo |
| **D · núcleo** | `src/gpu/nucleo.rs`, `src/bin/gpu_valida.rs` | buffers residentes, orquestração de dispatch, arnês de validação |

**Ninguém edita** `src/gpu/mod.rs`, `src/gpu/retina.wgsl`, `Cargo.toml`, `src/lib.rs`,
`src/simulation/motor.rs` — a integração final é minha, em um único passo, para haver uma
única compilação.

---

## Restrição operacional — CPU

A máquina do Leonardo travou com builds e simulações comendo todos os núcleos. Portanto:

- **NENHUM agente roda `cargo build`, `cargo test`, `cargo check` ou o simulador.**
- Agentes só leem arquivos e escrevem código.
- A compilação é única, no fim, com `nice -n 19 cargo build -j 4`.

---

## Fronteira alvo

O objetivo final é **estado residente na GPU** por toda a geração: sobe pesos e mundo uma vez,
e por tick só trafega o mínimo. Só volta à CPU o que a evolução precisa no fim da geração
(fitness, voltas, causa de morte) e os efeitos colaterais do Bevy (esconder entidade morta,
cor do material), que são pós-tick e não bloqueiam.

Fica na CPU, por ora e com motivo:
- hash espacial entre seres (drafting ch 72, estresse ch 125)
- pipeline de evolução (seleção, mutação, DNA em disco)
- telemetria e relatório

---

## Ordem de validação (fase D, minha)

1. compilar uma vez (`nice -n 19 cargo build -j 4 --release`)
2. `gpu_valida` — equivalência de cada kernel contra a CPU, isoladamente
3. corrida curta ponta a ponta com `env -u DISPLAY`, comparando PERF antes/depois
4. corrida de 2 gerações comparando resultado (voltas, mortes) com o caminho CPU
5. só então corrida longa
