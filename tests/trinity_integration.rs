use universe::{chaos::ChaosInput, engine::SovereignEngine, safety::SafetyState};

fn calm_safety() -> SafetyState {
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

fn calm_chaos() -> ChaosInput {
    ChaosInput {
        yaw_rate: 0.0,
        slip_avg: 0.0,
        retina_temporal_var: 0.0,
    }
}

#[test]
fn decide_returns_clamped_outputs() {
    let mut engine = SovereignEngine::new();
    let input = [0_i8; 32];
    let out = engine.decide([0.5, 0.0, 0.0], &input, &calm_safety(), &calm_chaos());
    for v in out {
        assert!(v >= -1.0 && v <= 1.0);
    }
}

#[test]
fn hard_veto_on_imminent_collision() {
    let mut engine = SovereignEngine::new();
    let mut s = calm_safety();
    s.ttc_forward = 0.1;
    s.speed_kmh = 50.0;
    let input = [0_i8; 32];
    let out = engine.decide([1.0, 0.0, 0.0], &input, &s, &calm_chaos());
    assert_eq!(out[0], 0.0, "throttle must be vetoed");
    assert_eq!(out[2], 1.0, "brake must be forced");
    assert!(engine.hard_veto_count >= 1);
}

#[test]
fn anti_stall_engages_reverse() {
    let mut engine = SovereignEngine::new();
    let mut s = calm_safety();
    s.speed_kmh = 1.0;
    s.stall_timer = 0.5;
    s.wall_left = 0.4;
    let input = [0_i8; 32];
    let out = engine.decide([1.0, 0.0, 0.0], &input, &s, &calm_chaos());
    assert!(
        out[0] < 0.0,
        "anti-stall must reverse throttle, got {}",
        out[0]
    );
}
