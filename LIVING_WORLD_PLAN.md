# Mundo Vivo — Plano por Fases (Leonardo 2026-07-08)

Transformar o engine do race num **mundo aberto de sobrevivência** com seres vivos (neuromodulação real), tudo é semente, conhecimento=sobrevivência, e validar o que o GRUPO aprendeu. Reusa o engine (não recria); implementado como **modo sobrevivência** (o race continua intacto). Regra do Leonardo: só dizer "pronto" quando TUDO estiver finalizado e rodando.

## Varredura (o que já existe — reuso)
- Retina DDA genérica (`world.rs:render_binocular_retina`); sensory_wave 223ch (`motor.rs`), canais de racing-line são guardados por `if !racing_line.is_empty()`.
- Cérebro agnóstico `SovereignEngine` (223→96→48→24→final_intent) `engine.rs`.
- Física move livre no plano (reusável pro forrageio); spatial-hash 20m já existe.
- Evolução geracional: fitness (`pipeline.rs:calc_fitness`), elite pool + niching, crossover block-wise + mutação Gaussian + **epigenética** (`spawn.rs`, `seed_epigenetic`).
- `metabolism.rs`: dopamina=RPE (+`learning_modulation`), cortisol=medo, curiosidade=novidade, memória episódica, neurônios-espelho, herança hormonal 10%.
- `knowledge.rs` (feito): árvores de conhecimento germinadas do mundo-semente + HUD.

## Decisões de escopo (honestas, pra entregar funcionando)
1. Neuromoduladores **modulam** comportamento/aprendizado/reprodução (NÃO arbiter de drives completo — seria semanas).
2. **Reusar física de carro** pro movimento (corpo-animal dedicado fica pra depois).
3. **Re-evoluir** os seres com o novo objetivo (fitness de sobrevivência) — o campeão de corrida não forrageia.
4. Tudo em **modo sobrevivência** (flag), sem quebrar o race.

## Fases (cada uma entregue RODANDO e validada)
- **F1 — Mundo aberto + forrageio + sobreviver:** mundo sem muro/racing-line (só terreno); sementes espalhadas; o ser se move (cérebro) e come/germina sementes; fitness = sobreviver (tempo vivo + sementes colhidas). Validar: seres se movem no aberto, forrageiam, sobrevivem; roda e renderiza.
- **F2 — Percepção do mundo + ver os outros:** retina/sensores reportam CLASSE (semente/comida/água/predador/outro-ser) via camada de objetos/voxel; seres se enxergam. Validar: canais novos entram no cérebro; seres reagem a objetos e uns aos outros.
- **F3 — Sistema vivo (neuromodulação causal) + conhecimento=vida:** ligar hormônios ao comportamento (cortisol→cauto, dopamina→vigor+plasticidade, curiosidade→explora) + serotonina/NE/ACh(foco)/ocitocina/testosterona como moduladores; saber o tipo da semente = viver (veneno mata quem não sabe); dopamina gruda em gatilho aprendido. Validar: hormônios mudam comportamento; quem sabe sobrevive mais.
- **F4 — Comunicação em bytes + reprodução + cultura:** fala em bytes compartilha fato; reprodução in-life (reusa crossover/epigenética); filho herda genoma + baselines + conhecimento (cultura); lifecycle/idade. Validar: saber se espalha; nascem filhos; cultura passa.
- **F5 — Validação coletiva + inspeção:** métricas do que o grupo aprendeu (HUD/gráfico por geração); clicar no ser → hormônios+drives+cérebro+saber. Validar: dá pra ver e medir o saber coletivo emergindo.

## Estado
- [~] F1 (em progresso)  [ ] F2  [ ] F3  [ ] F4  [ ] F5

### F1 — log
- **Mundo aberto** (`world.rs` flag `SURVIVAL`): road=0 wall=0 grass=102400. Flag `--vida` em `main.rs`. ✓
- **Fix do "parece carro" + travamento** (feedback Leonardo): em `--vida`, DESLIGADO o motor de corrida (`update_racers` via `run_if(racing_mode)` em `simulation/mod.rs`) E a física de carro (`PhysicsPlugin` via `run_if(!SURVIVAL)`). Criado `survival.rs`: controlador LEVE de criatura — movimento cinemático (rumo+velocidade, sem marcha/freio), sensação barata (posição das sementes), energia (pasta perto de semente, fome→morte), e **LOD por proximidade da câmera** (só perto computa forrageio; longe vagueia barato). `survival_step` roda no Update (1×/frame — NÃO multiplica com time_scale, ao contrário do motor de corrida). Seres ESPALHADOS pelo mapa (tira o oval). ✓ Validado: **200 seres rodando ~1.2 núcleos, sem travar, movendo, sem panic**.
- **Polish visual/UX (feedback Leonardo 2026-07-08):**
  - ✅ **Dia claro**: `AtmosphereModel::new(Nishita{sun_position alto})` + `AmbientLight brightness 1600` no modo vida (`main.rs setup_scene`). O chão custom não tem normais → só aparecia sob ambient forte; sol a pino tira o "entardecer".
  - ✅ **UI de carro REMOVIDA**: `update_hud`/`handle_ui_interactions` gateados por `not_survival` (`ui.rs`). Sem DEV CTRL/leaderboard/CÓRTEX Porsche/RPM/VOLANTE no `--vida`.
  - ✅ **Comida real e visível**: 90 plantas (caule+arbusto+baga amarela emissiva) espalhadas por TODO o mapa (`survival.rs spawn_food`).
  - ✅ **Forrageio consertado**: comida no mapa todo + TODOS os seres buscam com fome (removido o gate de sensação por LOD-câmera que deixava os longe só vagando).
  - ✅ **Água** (14 lagos azuis reflexivos) + **Predadores** (8 criaturas vermelhas: corpo+cabeça+focinho+olhos brilhantes) espalhados (`survival.rs spawn_food`).
  - ✅ **Dinâmica de medo/caça**: `predator_step` (predador caça presa mais próxima <120m); `survival_step` prioriza MEDO(foge <70m, corre 11 vs 8) > FOME(come planta) > vaguear; mordida <5m tira HP; fome tira HP; morte hp<=0.
- **Pendente:** (a) **cluster-LOD** pra milhares (render é o teto — N×7 sub-meshes; esconder/simplificar distantes); (b) **cluster de conhecimento** (validação do saber coletivo) na HUD; (c) sede/beber água (thirst); (d) cérebro dirigindo (F3 — neuromodulação); (e) reprodução/cultura (F4); (f) câmera de sobrevivência melhor.
