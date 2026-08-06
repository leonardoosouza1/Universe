//! A ablação de via faz o que promete? (tratamento)
//!
//! `neuro::definir_ablacao` usa `OnceLock` — só pode ser fixada UMA vez por processo. Como
//! cada arquivo em `tests/` vira um binário próprio, o par tratamento/controle mora em dois
//! arquivos: aqui a via do córtex está desligada; em `ablacao_controle.rs` não há ablação
//! nenhuma e a mesma verificação tem de dar o resultado OPOSTO.
//!
//! Sem o controle, este teste passaria também se `decide()` simplesmente ignorasse a
//! intenção do córtex sempre — o que seria um bug, não a ablação funcionando.

use universe::{
    chaos::ChaosInput,
    engine::SovereignEngine,
    neuro::{definir_ablacao, Via},
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
fn cortex_ablado_faz_a_intencao_do_cortex_nao_importar() {
    definir_ablacao(Via::Cortex);
    assert_eq!(universe::neuro::ablada(), Via::Cortex);

    let entrada = [0_i8; 32];
    // MESMO estado de mundo nos dois, mudando SÓ o que o córtex propõe. Motores novos em
    // cada chamada porque o engine guarda estado interno (portão, osciladores, contadores).
    let a = SovereignEngine::new().decide([1.0, 1.0, 0.0], &entrada, &calmo(), &sem_caos());
    let b = SovereignEngine::new().decide([-1.0, -1.0, 0.0], &entrada, &calmo(), &sem_caos());

    for i in 0..3 {
        assert!(
            (a[i] - b[i]).abs() < 1e-6,
            "com o córtex ablado a saída não pode depender da intenção do córtex, \
             mas o eixo {i} deu {} contra {}",
            a[i], b[i]
        );
    }
}

#[test]
fn ablar_o_cortex_nao_desliga_o_veto_duro() {
    definir_ablacao(Via::Cortex);
    // O veto duro é a rede de segurança. Desligá-lo junto com a via suave faria o carro bater
    // em segundos, e o custo medido viraria "morreu cedo" em vez da contribuição da via.
    let mut s = calmo();
    s.ttc_forward = 0.1;
    s.speed_kmh = 50.0;
    let mut e = SovereignEngine::new();
    let out = e.decide([1.0, 0.0, 0.0], &[0_i8; 32], &s, &sem_caos());
    assert_eq!(out[0], 0.0, "acelerador tinha de estar vetado");
    assert_eq!(out[2], 1.0, "freio tinha de estar forçado");
    assert!(e.hard_veto_count >= 1);
}
