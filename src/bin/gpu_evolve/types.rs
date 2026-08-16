//! Tipos de estado por-ser (CPU-side) do binário `gpu_evolve`.
//!
//! `BeingState` é DISTINTO do `BeingState` de `gpu_loop.rs` (20 campos aqui contra 5 lá) —
//! não unificar (ver risco R4 do REFACTOR_PLAN.md). RetinaParamsGpu/PoseRaw/uniform_entry/
//! storage_entry/pos_of/make_physics_state NÃO ficam aqui: foram deduplicados para
//! `universe::gpu::retina_types`, `universe::gpu::core` e `universe::gpu::physics` — os
//! três bins (`gpu_evolve`, `gpu_loop`, `cortex_evolve`) importam de lá diretamente.

use universe::gpu::physics::PhysicsStateGpu;

/// Estado por ser mantido na CPU entre ticks. Espelha (mas não compartilha) o
/// `BeingState` de `gpu_loop.rs`.
pub struct BeingState {
    pub phys:               PhysicsStateGpu,
    pub prev_steer:         f32,
    pub stagnation_ticks:   u32,
    /// Ticks com parede dentro de `WALL_NEAR_M` à frente. Vira custo de fitness quando
    /// `--wall-cost` > 0; com 0 é só telemetria, e serve de baseline pareado.
    pub wall_ticks:         u32,
    /// Dopamina: avanço ponderado por superfície neste tick (motor.rs:1673).
    /// joy_tick = progress_speed * surface_factor; range [-1, 1].
    pub joy_tick:           f32,
    /// Dopamina: segundos de flow contínuo (motor.rs:1675-1677). Acumula quando
    /// joy_tick > 0.5, decai 2× mais rápido quando não há progresso. Teto 5 s.
    pub joy_streak:         f32,
    /// Edge density da linha 6 da retina (motor.rs:923), normalizado [0,1].
    /// Calculado no readback da retina; entregue ao kernel no próximo tick.
    pub edge_row6:          f32,
    /// Edge density da linha 7 da retina (motor.rs:923), normalizado [0,1].
    pub edge_row7:          f32,
    /// Avanço ponderado por CENTRALIZAÇÃO: soma de (avanço_no_tick × proximidade do
    /// eixo da pista). Cravar o volante deriva o ser para a borda → proximidade baixa
    /// → esse avanço vale menos. Dá gradiente denso para MODULAR o volante e seguir o
    /// traçado, atravessando o vale da fitness enganosa (cravar rende 3/24, modular mal
    /// rende menos, modular bem rende muito — o meio precisa de recompensa contínua).
    pub center_prog:        f64,
    /// Saídas do piloto sensorial (`--piloto-sensor`). Calculadas no laço da retina, onde
    /// `voxel_at` está vivo, e consumidas no laço de comandos.
    pub sensor_steer:       f32,
    pub sensor_thr:         f32,
    pub sensor_brk:         f32,
    /// Distância mínima de fan-ray (7 direções) ao redor do ser, calculada onde `voxel_at`
    /// está disponível e usada pelo piloto fantasma para freio antecipado de curva.
    pub phantom_d_min:      f32,
    /// Distâncias às paredes laterais e frontal, calculadas em tick() a partir de tick_dists.
    /// Gravadas no replay para diagnóstico: sem elas é impossível saber o que o cérebro via
    /// ao parar. Os campos existem no ReplayFrame mas chegavam sempre zerados (..Default::default()).
    /// Sintoma medido: dist_wall_l/r/f = 0.0 em TODOS os 5000 frames do replay gen0029.
    pub dist_wall_l:        f32,
    pub dist_wall_r:        f32,
    pub dist_wall_f:        f32,
    /// Índice do waypoint mais próximo, cacheado (avança monotônico, busca local).
    pub wp_near:            usize,
    /// Ticks em que o ser esteve VIVO. Sem isto a velocidade média fica errada: dividir a
    /// distância pelo tempo total da geração subestima quem morreu cedo, que é exatamente
    /// quem o experimento de vida-ganha precisa medir.
    pub ticks_alive:        u32,
    /// Ticks de vida restantes. Encostar na parede CONSOME vida, não pontos.
    ///
    /// Lei da Moeda Cruzada (validada no ecossistema): toda ação protetora tem de custar
    /// em moeda DIFERENTE do que protege, senão a política degenerada domina. Medido aqui:
    /// cobrando a parede em DISTÂNCIA (mesma moeda da fitness), o ser aprendeu a não andar
    /// — tempo colado caiu de 36% para 7,7%, mas a mediana caiu de 126,7 m para 46,0 m e a
    /// mortalidade triplicou. Cobrando em TEMPO, evitar a parede COMPRA oportunidade de
    /// andar em vez de disputar com ela.
    /// Vida restante em ticks, como f64.
    ///
    /// Era i64 e o ganho `(passo * earn) as i64` TRUNCAVA: a 10,14 m/s o passo é
    /// 0,158 m/tick e ×6 dá 0,95 → vira 0. Com earn=e o ser só ganhava vida andando mais
    /// de 1/e metros por tick (acima de 10,7 m/s para e=6); abaixo disso ganhava NADA.
    pub life_left:          f64,
    /// Combustível, 1,0 no nascimento. Restaurado das branches `f1-world` e
    /// `state-of-art`, onde os seres COMPLETAVAM VOLTAS.
    ///
    /// Lá o consumo é `(0,002 + acelerador²·0,004)·dt` por segundo, o ser MORRE em zero e
    /// **fechar volta reabastece**. Isso é o "por quê" que a race-world-v3 perdeu: a volta
    /// deixou de ser objetivo abstrato e virou sobrevivência. No caminho GPU atual a
    /// palavra `fuel` aparecia uma única vez — como constante 127 no quadro de replay.
    pub fuel:               f64,
    pub reverse_timer:      f32,
    pub post_reverse_lockout: f32,
    /// Trava de troca de marcha (segundos). ANÁLOGA ao `shift_timer` do Bevy
    /// (physics.rs:292 usa 0,15 s lá); aqui 0,35 s — valor deliberadamente maior
    /// porque sem embreagem simulada no caminho gpu_evolve o limiar de RPM oscila
    /// e a caixa bateria 2↔3 todo tick. NÃO é paridade numérica.
    pub shift_cooldown:     f32,
    /// intent[3]/[4] do tick anterior — detecção de BORDA do pedido neural de marcha
    /// (paridade com `racer.memory[3]/[4]` em motor.rs:2484-2494: troca só em subida
    /// 0→1 do sinal, não enquanto segurado).
    pub shift_up_prev:      f32,
    pub shift_down_prev:    f32,
    /// M4: reserva de neuromodulador por REGIÃO (4 blocos), ∈[0,1]. Começa cheia;
    /// usar gate ≠ 1,0 drena (a·|g−1|·R), repõe (1−R)/τ. Ver REGIOES_CEREBRAIS.md.
    pub nm_reserva:         [f32; 4],
    /// Anti-encalhe escalonado (2026-08-16): nº de rés re-disparadas em sequência
    /// curta. Medido no campeão m23: ré→sai cedo→lockout→engancha de novo, loop de
    /// 10k ticks no MESMO canto (obs tick 9574+). Cada falha alonga a ré e alterna
    /// o lado do volante (quebra a simetria do canto).
    pub re_falhas:          u8,
    /// Segundos desde a última SAÍDA de ré (para detectar re-encalhe rápido).
    pub re_desde_saida:     f32,
    /// D4: dano acumulado ∈[0,1] — envelhecimento do corpo dentro da vida.
    /// Acumula por RPM alto e parede; degrada acelerador/direção; SEM sensor
    /// direto (oculto). Zera no respawn. Ver ESCADA_DIFICULDADE.md.
    pub dano:               f32,
}
