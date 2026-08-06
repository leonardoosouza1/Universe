//! O controle do teste de ablação: SEM ablação, a intenção do córtex tem de importar.
//!
//! Par de `ablacao_cortex.rs`. Este arquivo nunca chama `definir_ablacao`, então
//! `neuro::ablada()` devolve `Via::Nenhuma` e o comportamento é o normal do simulador.
//!
//! É este teste que dá sentido ao outro. Sozinho, "a saída não muda quando o córtex muda"
//! passaria igual se `decide()` ignorasse o córtex por um bug — aqui se verifica que, sem
//! ablação, ela MUDA.

use universe::{
    chaos::ChaosInput,
    engine::SovereignEngine,
    neuro::{ablada, Via},
    safety::SafetyState,
};

fn calmo() -> SafetyState {
    SafetyState {
        speed_kmh: 30.0, yaw_rate: 0.0, slip_avg: 0.0, ttc_forward: 10.0,
        wall_left: 5.0, wall_right: 5.0, stall_timer: 0.0, pinado_timer: 0.0,
        rain: 0.0, curv25: 0.0, current_thr: 0.5,
    }
}

fn sem_caos() -> ChaosInput {
    ChaosInput { yaw_rate: 0.0, slip_avg: 0.0, retina_temporal_var: 0.0 }
}

#[test]
fn sem_ablacao_a_intencao_do_cortex_importa() {
    assert_eq!(ablada(), Via::Nenhuma, "este binário não pode ter ablação definida");

    let entrada = [0_i8; 32];
    let a = SovereignEngine::new().decide([1.0, 1.0, 0.0], &entrada, &calmo(), &sem_caos());
    let b = SovereignEngine::new().decide([-1.0, -1.0, 0.0], &entrada, &calmo(), &sem_caos());

    let dif: f32 = (0..3).map(|i| (a[i] - b[i]).abs()).sum();
    assert!(
        dif > 1e-3,
        "sem ablação a saída TEM de responder à intenção do córtex; diferença total foi {dif}"
    );
}
