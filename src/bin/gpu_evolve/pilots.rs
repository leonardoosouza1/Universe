//! Pilotos de diagnóstico: controladores puramente geométricos/sensoriais que
//! SUBSTITUEM o cérebro por flag, para separar culpa do CORPO da culpa do CÉREBRO.
//!
//! - **Piloto sensorial** (`--piloto-sensor`): dirige só por raios de parede — mede o
//!   teto da informação sensorial disponível.
//! - **Piloto fantasma** (`--piloto`): dirige pela racing line (oráculo que o ser não
//!   possui) — mede se o CORPO consegue completar a pista, isolando a competência do
//!   cérebro.
//!
//! Nenhuma das duas funções aqui é usada em produção (fitness normal roda com ganho
//! 0.0, ver `statics::PILOTO_GANHO`/`statics::PILOTO_SENSOR`) — são instrumento de
//! diagnóstico, chamadas de dentro de `context.rs::GpuCtx::tick()`.
//!
//! `voxel_at` chega como closure (`impl Fn`) em vez de acessar `GpuCtx` diretamente
//! porque essas funções são puras — testáveis sem GPU — e porque `GpuCtx::voxel_at`
//! precisa de `&self.grid_cpu` (não pode virar `fn` livre sem o contexto do ser
//! chamador, ver nota R do REFACTOR_PLAN.md sobre as closures `voxel_at` triplicadas).

/// Raio até a parede numa direção `(dx, dz)` a partir de `(px, pz)`, em metros
/// (passo de 1 m, teto `cap`). Usado por ambos os pilotos.
#[inline]
fn raio(px: f32, pz: f32, dx: f32, dz: f32, cap: f32, voxel_at: &impl Fn(f32, f32) -> u32) -> f32 {
    let mut d = cap;
    let mut t = 1.0_f32;
    while t <= cap {
        let v = voxel_at(px + dx * t, pz + dz * t);
        if v == 2 || v == 3 { d = t; break; }
        t += 1.0;
    }
    d
}

/// Comando do piloto sensorial: controlador de "vinte linhas" que dirige só por
/// raios de parede (laterais + diagonais ±45°), tentando centralizar e antecipar
/// curvas. Devolve `(steer, throttle, brake)`.
///
/// # Por quê
///
/// Leonardo: *"roda uma bateria onde não tem cérebro, o carro deve sobreviver por
/// sensores, ver distância das paredes em ambos os lados, sempre tentando centralizar
/// no centro da pista e fazer as curvas de forma correta."* O piloto fantasma anterior
/// usava a racing line — um oráculo que o ser NÃO possui, então media o corpo, não a
/// informação. Este usa só raios de parede, exatamente o que a retina do ser enxerga.
pub fn piloto_sensorial(
    ganho: f32,
    px: f32, pz: f32,
    fx: f32, fz: f32,
    perp_x: f32, perp_z: f32,
    speed: f32,
    fwd_dist: f32,
    voxel_at: &impl Fn(f32, f32) -> u32,
) -> (f32, f32, f32) {
    // Laterais puras: centralização.
    let esq = raio(px, pz, perp_x, perp_z, 25.0, voxel_at);
    let dir = raio(px, pz, -perp_x, -perp_z, 25.0, voxel_at);
    // Diagonais a ±45°: antecipação. A curva aparece nelas ANTES de aparecer
    // nas laterais — é o que permite virar antes de raspar a parede.
    let k = std::f32::consts::FRAC_1_SQRT_2;
    let diag_e = raio(px, pz, fx * k + perp_x * k, fz * k + perp_z * k, 25.0, voxel_at);
    let diag_d = raio(px, pz, fx * k - perp_x * k, fz * k - perp_z * k, 25.0, voxel_at);

    // Erro de centralização, normalizado: +1 = colado num lado.
    let centro = (dir - esq) / (dir + esq).max(1.0);
    // Erro de antecipação: para que lado a pista abre à frente.
    let abertura = (diag_d - diag_e) / (diag_d + diag_e).max(1.0);

    // A antecipação pesa mais que a centralização: corrigir só pelo lateral é reagir
    // depois de já estar errado, e numa pista de 28 m isso chega tarde na curva de
    // raio 51 m.
    let steer = (ganho * (centro * 0.4 + abertura * 1.0)).clamp(-1.0, 1.0);

    // Velocidade pelo espaço crítico: usa o MÍNIMO entre distância frontal e
    // diagonais estreitas. As diagonais detectam a parede lateral da curva ANTES do
    // raio frontal — isso é o que permite frear antes de bater.
    let critico = fwd_dist.min(diag_e).min(diag_d);
    let assimetria = (diag_e - diag_d).abs() / (diag_e + diag_d).max(1.0);
    // critico cai quando há parede perto; assimetria sobe quando há curva.
    let sinal_curva = (1.0 - assimetria).clamp(0.0, 1.0);
    let livre = (critico / 25.0).clamp(0.0, 1.0) * sinal_curva;
    let alvo = 5.0 + livre * 13.0; // 5 m/s na curva fechada, 18 na reta
    let thr = if speed < alvo { 0.85 } else { 0.0 };
    let brk = if speed > alvo * 1.05 { 0.7 } else { 0.0 };
    (steer, thr, brk)
}

/// Distância mínima de fan-ray (7 direções: frente + ±15°/30°/45°) ao redor do ser.
/// Alimenta o freio antecipado do piloto fantasma. A curva aparece como parede
/// diagonal ANTES de aparecer frontalmente — o fan detecta a curva antes do raio puro.
pub fn fan_ray_d_min(
    px: f32, pz: f32,
    fx: f32, fz: f32,
    voxel_at: &impl Fn(f32, f32) -> u32,
) -> f32 {
    let (pxp, pzp) = (-fz, fx); // perpendicular esquerda
    let k15 = 0.9659_f32; let s15 = 0.2588_f32;
    let k30 = 0.8660_f32; let s30 = 0.5000_f32;
    let k45 = 0.7071_f32; let s45 = 0.7071_f32;
    let d0   = raio(px, pz, fx, fz, 30.0, voxel_at);
    let dl15 = raio(px, pz, fx*k15 + pxp*s15, fz*k15 + pzp*s15, 30.0, voxel_at);
    let dr15 = raio(px, pz, fx*k15 - pxp*s15, fz*k15 - pzp*s15, 30.0, voxel_at);
    let dl30 = raio(px, pz, fx*k30 + pxp*s30, fz*k30 + pzp*s30, 30.0, voxel_at);
    let dr30 = raio(px, pz, fx*k30 - pxp*s30, fz*k30 - pzp*s30, 30.0, voxel_at);
    let dl45 = raio(px, pz, fx*k45 + pxp*s45, fz*k45 + pzp*s45, 25.0, voxel_at);
    let dr45 = raio(px, pz, fx*k45 - pxp*s45, fz*k45 - pzp*s45, 25.0, voxel_at);
    d0.min(dl15).min(dr15).min(dl30).min(dr30).min(dl45).min(dr45)
}

/// Comando pure-pursuit do piloto fantasma: aponta para um ponto `L` metros à
/// frente na racing line e corrige pelo heading (não pelo desvio lateral — evita o
/// bug de `lookahead_offset` que mistura erro longitudinal com lateral quando o
/// carro está atrás do ponto de antecipação). `L` adaptativo: mais distância em alta
/// velocidade para suavizar. `d_min` = saída de `fan_ray_d_min`, usada para o freio
/// antecipado de curva.
///
/// Devolve `(steer, throttle, brake)`.
#[allow(clippy::too_many_arguments)]
pub fn piloto_fantasma_comando(
    ganho: f32,
    thr_base: f32,
    pos_x: f32, pos_z: f32,
    fwd_x: f32, fwd_z: f32,
    tgt_x: f32, tgt_z: f32,
    speed_now: f32,
    d_min: f32,
) -> (f32, f32, f32) {
    let dx = tgt_x - pos_x;
    let dz = tgt_z - pos_z;
    let dist_tgt = (dx * dx + dz * dz).sqrt().max(0.1);
    // Cross product (fwd × dir_to_target): positivo = alvo à esquerda.
    let cross = fwd_x * (dz / dist_tgt) - fwd_z * (dx / dist_tgt);
    // Alvo à esquerda (cross > 0) → virar esquerda (negativo no jogo).
    // Ganho proporcional ao ângulo; cap ±0.7 evita derrapagem.
    let steer = (ganho * (-cross * 1.2)).clamp(-0.7, 0.7);

    // v_max física para R=16m, mu=1.6: 15.9 m/s. Alvo 14 m/s com margem.
    let livre_geo = (d_min / 30.0).clamp(0.0, 1.0);
    let v_alvo = 6.0 + livre_geo * 10.0; // 6 m/s perto de parede, 16 m/s livre
    let (thr, brk) = if speed_now > v_alvo + 0.5 {
        (0.0, ((speed_now - v_alvo) / 8.0).clamp(0.15, 0.7))
    } else if speed_now < v_alvo - 1.5 {
        (thr_base, 0.0)
    } else {
        (thr_base * 0.4, 0.0)
    };
    (steer, thr, brk)
}
