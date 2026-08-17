//! The control for the ablation test: WITHOUT ablation, the cortex intent has to matter.
//!
//! Pair of `ablacao_cortex.rs`. This file never calls `definir_ablacao`, so
//! `neuro::ablada()` returns `Via::Nenhuma` and the behaviour is the simulator's normal one.
//!
//! It is this test that gives the other one meaning. On its own, "the output does not change
//! when the cortex changes" would pass just the same if `decide()` ignored the cortex due to
//! a bug — here we verify that, without ablation, it DOES change.

use universe::{
    chaos::ChaosInput,
    engine::SovereignEngine,
    neuro::{Via, ablada},
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
fn sem_ablacao_a_intencao_do_cortex_importa() {
    assert_eq!(
        ablada(),
        Via::Nenhuma,
        "this binary must not have an ablation set"
    );

    let entrada = [0_i8; 32];
    let a = SovereignEngine::new().decide([1.0, 1.0, 0.0], &entrada, &calmo(), &sem_caos());
    let b = SovereignEngine::new().decide([-1.0, -1.0, 0.0], &entrada, &calmo(), &sem_caos());

    let dif: f32 = (0..3).map(|i| (a[i] - b[i]).abs()).sum();
    assert!(
        dif > 1e-3,
        "without ablation the output MUST respond to the cortex intent; total difference was {dif}"
    );
}
