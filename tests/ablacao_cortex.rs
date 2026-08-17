//! Does pathway ablation do what it promises? (treatment)
//!
//! `neuro::definir_ablacao` uses `OnceLock` — it can only be set ONCE per process. Since
//! each file in `tests/` becomes its own binary, the treatment/control pair lives in two
//! files: here the cortex pathway is switched off; in `ablacao_controle.rs` there is no
//! ablation at all and the same check must give the OPPOSITE result.
//!
//! Without the control, this test would also pass if `decide()` simply ignored the
//! cortex's intent always — which would be a bug, not ablation working.

use universe::{
    chaos::ChaosInput,
    engine::SovereignEngine,
    neuro::{Via, definir_ablacao},
    safety::SafetyState,
};

fn calmo() -> SafetyState {
    SafetyState {
        speed_kmh: 30.0,
        yaw_rate: 0.0,
        slip_avg: 0.0,
        ttc_forward: 10.0,
        wall_left: 5.0,
        wall_right: 5.0,
        stall_timer: 0.0,
        pinado_timer: 0.0,
        rain: 0.0,
        curv25: 0.0,
        current_thr: 0.5,
    }
}

fn sem_caos() -> ChaosInput {
    ChaosInput {
        yaw_rate: 0.0,
        slip_avg: 0.0,
        retina_temporal_var: 0.0,
    }
}

#[test]
fn cortex_ablado_faz_a_intencao_do_cortex_nao_importar() {
    definir_ablacao(Via::Cortex);
    assert_eq!(universe::neuro::ablada(), Via::Cortex);

    let entrada = [0_i8; 32];
    // The SAME world state in both, changing ONLY what the cortex proposes. New engines on
    // each call because the engine keeps internal state (gate, oscillators, counters).
    let a = SovereignEngine::new().decide([1.0, 1.0, 0.0], &entrada, &calmo(), &sem_caos());
    let b = SovereignEngine::new().decide([-1.0, -1.0, 0.0], &entrada, &calmo(), &sem_caos());

    for i in 0..3 {
        assert!(
            (a[i] - b[i]).abs() < 1e-6,
            "with the cortex ablated the output must not depend on the cortex's intent, \
             but axis {i} gave {} against {}",
            a[i],
            b[i]
        );
    }
}

#[test]
fn ablar_o_cortex_nao_desliga_o_veto_duro() {
    definir_ablacao(Via::Cortex);
    // The hard veto is the safety net. Switching it off along with the soft pathway would make
    // the car crash in seconds, and the measured cost would become "died early" instead of the
    // pathway's contribution.
    let mut s = calmo();
    s.ttc_forward = 0.1;
    s.speed_kmh = 50.0;
    let mut e = SovereignEngine::new();
    let out = e.decide([1.0, 0.0, 0.0], &[0_i8; 32], &s, &sem_caos());
    assert_eq!(out[0], 0.0, "throttle had to be vetoed");
    assert_eq!(out[2], 1.0, "brake had to be forced");
    assert!(e.hard_veto_count >= 1);
}
