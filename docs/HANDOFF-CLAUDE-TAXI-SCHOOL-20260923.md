# Handoff para o Claude — fechar a escola e a entrega do táxi

Data: 2026-09-23  
Branch: `organism/school-20260922`  
Último commit concluído: `c43521e` (`feat(taxi): report service phases instead of race laps`)

## Estado real

As Tasks 1–6 do plano [`docs/superpowers/plans/2026-09-23-taxi-school-delivery.md`](superpowers/plans/2026-09-23-taxi-school-delivery.md) estão commitadas. Elas entregam:

- modos explícitos do táxi (`Production`, `MotorBench`, `ServiceBench`);
- oito lições com pares de estações e folds sem vazamento;
- transições de pickup/dropoff compartilhadas com o mundo de produção;
- juiz GPU das quatro lições de serviço;
- avaliação de cada genoma em `K` largadas antes da seleção;
- eventos causais de serviço e telemetria própria do táxi.

O projeto **ainda não está funcional de ponta a ponta**. A Task 7 está incompleta porque a evolução passa em `move`, mas não aprende `stop`. Sem isso, não há base científica para afirmar que `turn`, `pickup`, `delivery`, o currículo adaptativo ou a transferência para produção funcionam com cérebros evoluídos.

Não há processo `gpu_evolve` em execução no momento deste handoff.

## Evidência GPU já obtida

Todas as simulações abaixo foram executadas em Radeon/Vulkan, sem fallback de simulação na CPU.

| Experimento | Geração final | Resultado |
|---|---:|---|
| `artifacts/taxi-school-delivery-task7-evolve-move-seed7-k3-v2` | 15 | **48/48** sucessos em `move` |
| `artifacts/taxi-school-delivery-task7-evolve-move-seed11-k3` | 15 | **40/48** sucessos em `move` |
| `artifacts/taxi-school-delivery-task7-evolve-stop-seed7-k3-baseline` | 15 | **0/48** em `stop`; progresso mediano 0,569 |
| `artifacts/taxi-school-delivery-task7-evolve-stop-from-move-seed7-k3` | 15 | **0/48** em `stop`; progresso mediano 0,972: chega e ultrapassa |
| `artifacts/taxi-school-delivery-task7-evolve-stop-explore-s19` | 7 | **0/72**; mutação motora forte degradou locomoção |
| `artifacts/taxi-school-delivery-task7-evolve-stop-reflex-v3-s7-smoke` | 5 | **0/48**; prior experimental aumentou contatos e não resolveu parada |

O witness diagnóstico concluiu as oito lições individualmente. Em produção, o witness fez duas entregas em `artifacts/taxi-school-delivery-task7-production-two`, mas teve uma colisão de aproximadamente 12 kJ e uma violação de sinal. Isso prova viabilidade física parcial; **não prova aprendizado**, e ainda não satisfaz o gate limpo do validador.

## Diagnóstico do bloqueio

O problema imediato é a ponte entre locomover e parar:

1. `move` já é evoluível em duas sementes com três largadas por genoma.
2. O campeão de `move` chega perto do alvo de `stop`, mas passa por ele em velocidade.
3. A pontuação de `stop` dá sinal forte para aproximação e dwell já parado, porém oferece pouco gradiente físico durante a frenagem.
4. Aumentar indiscriminadamente mutação de motor destrói o que já foi aprendido.
5. Acrescentar diretamente um neurônio de distância da tarefa para freio não pagou no smoke e causou contatos. Esse experimento não deve ser confundido com correção validada.

Portanto, não aumente gerações antes de corrigir o sinal de aprendizado. Mais gerações na mesma paisagem repetem o platô.

## Alterações locais ainda sem commit

Revise o diff antes de tocar no código. Há mudanças de mais de um autor.

### Alterações desta Task 7 que podem ser aproveitadas

- `src/gpu/taxi_witness.rs` e `src/gpu/taxi_witness.wgsl`: witness v2 por lição, estado de hold na GPU e piloto de duas entregas.
- `src/bin/gpu_evolve/main.rs`: passa o ID da lição ao witness e usa a capacidade correta de largadas K.
- `src/bin/gpu_evolve/school_batch.rs`: `required_start_count`; corrige o erro que exigia `genomas × K` geometrias distintas quando todos os genomas devem compartilhar as mesmas K largadas.
- `src/bin/gpu_evolve/driving_diagnostics.rs` e `src/bin/gpu_evolve/tests.rs`: preservam a cadeia causal quando uma entrega e o pickup do passageiro seguinte ocorrem no mesmo tick.
- `scripts/validate_taxi_witness.py`: valida finitude, identidade, ausência de pedal simultâneo e cadeia completa de eventos.
- `scripts/validate_taxi_school_gpu.sh`: esqueleto da bateria reproduzível; ainda não é o gate final.

### Experimento que não deve entrar sem nova prova

`src/gpu/evolution.rs` contém o prior experimental `REFLEX_TASK_BRAKE_NEURON`. O A/B terminou em 0/48 e mais contatos. Reverta somente esse hunk ou mantenha-o atrás de experimento isolado; não o promova como default e não o misture à correção de shaping.

### Arquivos que já estavam modificados antes desta Task 7

Não restaure nem inclua automaticamente estes arquivos em um commit da escola:

- `docs/SCHOOL-SHARED-START-20260922.md`
- `docs/SCHOOL-STOP-SHAPING-20260922.md`
- `docs/TAXI-FEASIBILITY-AND-SELECTION.md`
- `src/gpu/brain/mod.rs`
- `src/gpu/organism_action.rs`
- `src/gpu/organism_action.wgsl`
- `src/gpu/retina_digest.rs`
- `.claude/worktrees/`
- `artifacts/`

Remova apenas o cache local `scripts/__pycache__/` se for preparar um commit; artefatos científicos devem continuar fora do commit, salvo decisão explícita do repositório.

## Próxima correção: shaping físico para `stop`

Reconstrua primeiro um único ser/tick de `stop` pelo fluxo:

`school geometry → retina/wave → L1 → L2/CfC → motor raw → arbitration → applied command → physics → school.wgsl → record → aggregate K → selection`

O judge estrito deve continuar exigindo parada real. Acrescente somente um gradiente intermediário medido na física, sem premiar o valor bruto do comando de freio.

Use uma curva de velocidade admissível em função da distância restante:

```text
v_allow(d) = sqrt(v_stop² + 2 * a_ref * max(d - r_goal, 0))
speed_excess = max(v - v_allow(d), 0)
```

Sugestão inicial de constantes, depois calibrada com a dinâmica real do carro:

- `v_stop = 0.3 m/s`, igual ao critério estrito;
- `r_goal =` raio já usado pela lição, sem duplicar uma constante divergente;
- `a_ref` obtido de uma frenagem GPU medida, não inventado no shader.

Registre em slots novos do record, sem alterar offsets antigos:

- velocidade ao entrar na zona de frenagem;
- menor velocidade atingida enquanto a distância diminui;
- integral limitada de excesso sobre `v_allow`;
- ticks em conformidade com a curva;
- distância no instante da menor velocidade.

O shaping pode recompensar redução física de velocidade e conformidade, mas deve ser limitado e menor que o bônus de sucesso. Não recompense simplesmente `brake > 0`, pois isso seleciona freio parado ou pedal simultâneo. Não marque sucesso por shaping; sucesso continua sendo posição, velocidade e dwell estritos.

Antes de uma run evolutiva:

1. crie fixture GPU com aproximação que desacelera e deve pontuar mais;
2. crie fixture que freia longe e fica parada fora do alvo; ela não pode vencer;
3. crie fixture que passa rápido pelo alvo; deve ter progresso, mas não sucesso;
4. crie fixture com throttle+brake; não pode obter vantagem;
5. confirme bit a bit que shaping desligado mantém o score histórico.

## Ordem obrigatória de execução

1. Separar os hunks seguros da Task 7 do prior experimental não comprovado.
2. Medir a desaceleração real do carro na GPU e implementar o shaping físico de `stop`.
3. Rodar `stop` em duas sementes, `K=3`, com horizonte suficiente. Exigir sucesso estrito em largadas distintas.
4. Congelar os campeões e avaliá-los sem mutação em todos os candidatos de validação/teste.
5. Só então avançar, em ordem, por `turn_near`, `turn_far`, `approach`, `pickup`, `navigate` e `delivery`.
6. Nas lições de serviço, se intents aleatórios encerram episódios antes de haver chance de aprender, transforme pedidos prematuros em recusa/penalidade durante a pré-escola. Mantenha o predicado de sucesso idêntico ao da produção. Isso precisa de fixture e decisão documentada; não esconda intents nem injete ação correta.
7. Rodar o currículo adaptativo completo. Promoção depende de sucesso estrito, nunca do score de shaping.
8. Usar dois graduados evoluídos independentes, congelados, em tarefas de produção não vistas. Cada um precisa completar ao menos uma cadeia `issued → pickup → dropoff → delivery`.
9. Executar a matriz legada da Task 9: corrida, lições antigas, currículo antigo, observation v2/v3, checkpoint e replay.

Pare no primeiro gate que falhar, corrija a causa e repita esse gate. Não continue gerando resultados posteriores sobre uma competência anterior ausente.

## Comando-base para simulação

Use GPU Vulkan/Radeon e limite os processos auxiliares:

```bash
env -u DISPLAY \
  WGPU_BACKEND=vulkan \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  MESA_VK_VERSION_OVERRIDE=1.3 \
  RAYON_NUM_THREADS=1 \
  CARGO_BUILD_JOBS=1 \
  cargo run --release --bin gpu_evolve -- <argumentos>
```

Não valide a simulação com backend CPU. Build, testes puros e validação de JSON podem usar CPU com uma thread.

## Critério para dizer “finalizado”

O trabalho só está finalizado quando todos os itens abaixo tiverem artefatos verificáveis:

- testemunhas das oito lições passam sem shortcut, não finitos ou pedal simultâneo;
- duas sementes aprendem cada lição com `K >= 3`;
- campeões congelados generalizam aos folds não vistos sem mutação;
- currículo adaptativo promove pelas oito lições usando sucesso estrito;
- dois graduados independentes fazem entrega em produção;
- cada entrega possui cadeia causal completa e ID único;
- não há regressão no mundo de corrida, escola antiga, checkpoint, observation ou replay;
- o relatório final contém comandos, seeds, gerações, ticks, contagens, hashes dos genomas e caminhos dos artefatos;
- uma revisão fresca confere diff, fórmulas, contratos e artefatos antes da declaração final.

Até esses gates passarem, a formulação correta é: **a infraestrutura da escola está montada, `move` evolui, e `stop` ainda bloqueia a progressão**.
