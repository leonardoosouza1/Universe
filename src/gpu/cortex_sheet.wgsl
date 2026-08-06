// ── FOLHA CORTICAL EQUIPOTENTE — KERNEL A: AFERÊNCIA + ONDAS ESPONTÂNEAS ────
//
// Projeta os 223 canais sensoriais para a folha 32×32=1024 unidades.
// Cada canal k tem um alvo (tx_k, ty_k) determinístico (hash do índice k),
// e contribui para a unidade i com peso P ∝ exp(−‖pos_i − alvo_k‖² / σ²).
//
// Ondas espontâneas: nos primeiros SPONT_TICKS, injeta atividade rítmica
// endógena antes do sinal sensorial chegar — necessário para a segregação
// hebbiana ter o que correlacionar (doc §5.5).
//
// MECANISMOS ANTI-SATURAÇÃO (adicionados 2026-08-04):
//
//   1. Fadiga adaptativa por unidade (spike-frequency adaptation):
//      Cada unidade acumula fat += SHEET_FAT_ALPHA * (|act| - fat) * dt,
//      e o input efetivo é reduzido por fat: net_eff = net_input - SHEET_FAT_BETA * fat.
//      Quando a unidade dispara em ponto fixo (|act| alto por muitos ticks), fat cresce
//      e o input efetivo vira negativo — quebrando o ponto fixo sem if/clamp.
//      Mecanismo: análogo à adaptação de neurônio piramidal cortical (after-hyperpolarization).
//
//   2. Normalização divisiva por vizinhança local (inibição lateral shunting):
//      Antes de escrever act, divide pelo campo médio local de atividade dos 4 vizinhos
//      ortogonais (tick t-1, lido de buf_act antes da atualização):
//        act_norm = tanh(h) / (1.0 + SHEET_DIV_SIGMA * mean_neigh_abs)
//      Quando muitas unidades vizinhas estão ativas, o ganho de cada uma é reduzido.
//      Implementa inibição lateral sem loop sobre vizinhos (divisor = campo local).
//      Mecanismo: análogo a interneurônios GABAérgicos de basket cells no neocórtex.
//
// PROIBIDO: nenhum `if canal_é_visão {}`, nenhum clamp forçado, nenhum reset.

// ── Constantes de layout ─────────────────────────────────────────────────────
const SHEET_W:   u32 = 32u;   // largura da folha (colunas)
const SHEET_H:   u32 = 32u;   // altura da folha (linhas)
const SHEET_N:   u32 = 1024u; // = SHEET_W × SHEET_H
const N_INPUTS:  u32 = 223u;  // canais sensoriais (SENSORY_WAVE_SIZE)

// σ² para a gaussiana de aferência: σ=8 células → cobre ¼ da folha.
// Valor grande → modalidades se sobrepõem inicialmente (mais plasticidade).
const SIGMA_SQ:  f32 = 64.0;  // σ=8 → σ²=64

// Limiar de contribuição: exp(−d²/σ²) < EPS → ignora canal.
// exp(−3²) = exp(−9/64) ≈ 0.87; este EPS de 0.05 captura raio ≈ 2.8σ.
const AFFERENCE_EPS: f32 = 0.05;

// Ticks de atividade espontânea antes do sinal sensorial (doc §5.5).
const SPONT_TICKS: u32 = 200u;

// ── Constantes de fadiga adaptativa (mecanismo 1) ────────────────────────────
// SHEET_FAT_ALPHA: velocidade de crescimento da fadiga — FIXO (não varia na varredura).
//   τ_fat ≈ 1/alpha = 2.0s → com dt=1/64, cresce 0.0078/tick por unit de |act|.
//   Com |act|≈1.0, atinge 0.63 em ~128 ticks (2s) — antes do ponto fixo se consolidar.
const SHEET_FAT_ALPHA: f32 = 0.5;

// SHEET_FAT_BETA e SHEET_DIV_SIGMA agora vêm do uniform (campo params.fat_beta e params.div_sigma).
// Isso permite varredura sem recompilar o shader.
//
// ⚠️ MEDIDO (2026-08-04): inibição FORTE (BETA=1.2, SIGMA=2.0) eliminou saturação (89.8%→0%)
// mas custou o TETO (1055m→440m). A mediana ficou igual (~140m) — não é a rede morrendo,
// é especificamente o teto. Varredura em 4 pontos para achar o ótimo.

// ── Params uniform ────────────────────────────────────────────────────────────
struct Params {
    n_beings:   u32,
    tick:       u32,     // tick global (0..∞)
    // fat_beta: força de subtração da fadiga adaptativa (mecanismo 1).
    //   0.0 = desligado; 1.2 = valor original forte.
    //   Varredura: 0.0, 0.4, 0.8, 1.2
    fat_beta:   f32,
    // div_sigma: ganho da normalização divisiva por vizinhança (mecanismo 2).
    //   0.0 = desligado; 2.0 = valor original forte.
    //   Varredura: 0.0, 0.5, 1.0, 2.0
    div_sigma:  f32,
}

// ── Bindings ──────────────────────────────────────────────────────────────────
@group(0) @binding(0) var<uniform>             params:      Params;
// sensory_wave: i32 sign-extended. Layout: ser * N_INPUTS + canal.
@group(0) @binding(1) var<storage, read>       buf_wave:    array<i32>;
// Pesos de aferência: f32 por (unidade, canal). Layout: ser*SHEET_N*N_INPUTS + unidade*N_INPUTS + canal.
// Inicializado por zeros; atualizado pelo kernel B (Hebb).
@group(0) @binding(2) var<storage, read_write> buf_afw:     array<f32>;
// Estado EMA da folha: f32 por unidade por ser. Layout: ser*SHEET_N + unidade.
@group(0) @binding(3) var<storage, read_write> buf_state:   array<f32>;
// τ por unidade por ser (constante de integração). Layout: ser*SHEET_N + unidade.
@group(0) @binding(4) var<storage, read>       buf_tau:     array<f32>;
// Ativação de saída da folha (pós-EMA): f32 por unidade por ser.
// Escrito aqui, lido pelo kernel B (Hebb) e pelo kernel D (motor).
// LIDO no início (vizinhos do tick t-1) e ESCRITO no final (tick t).
@group(0) @binding(5) var<storage, read_write> buf_act:     array<f32>;
// Fadiga adaptativa por unidade por ser: f32. Layout: ser*SHEET_N + unidade.
// Inicializado em 0.0; cresce quando a unidade dispara sustentadamente.
// Mecanismo: spike-frequency adaptation (after-hyperpolarization cortical).
@group(0) @binding(6) var<storage, read_write> buf_fat:     array<f32>;

// ── Função auxiliar: alvo espacial de canal k (hash → grade) ─────────────────
// Mapeia índice de canal k → (tx, ty) ∈ [0, SHEET_W-1] × [0, SHEET_H-1].
// Usa dois hashes distintos de Knuth para tx e ty, garantindo distribuição
// pseudo-aleatória REPRODUTÍVEL. NÃO é rótulo de região — é ponto alvo.
fn channel_target(k: u32) -> vec2<f32> {
    // Hash de Knuth: multiplier 2654435761 (número de Fibonacci dourado)
    let hx: u32 = (k * 2654435761u) ^ (k >> 5u);
    let hy: u32 = (k * 2246822519u) ^ (k >> 3u);
    // Mapeia para [0, SHEET_W-1] e [0, SHEET_H-1]
    let tx: f32 = f32(hx % SHEET_W);
    let ty: f32 = f32(hy % SHEET_H);
    return vec2<f32>(tx, ty);
}

// ── Função auxiliar: onda espontânea rítmica ────────────────────────────────
// Gera atividade endógena correlacionada espacialmente, modulada por tick.
// Amplitude decresce com o tick (período crítico fecha).
// Frequência: ~1 ciclo por 32 ticks (theta-band ~2 Hz @ 64 Hz).
fn spontaneous_wave(unit_x: f32, unit_y: f32, tick: u32) -> f32 {
    // Fator de amplitude: decresce de 1 → 0 nos primeiros SPONT_TICKS.
    if tick >= SPONT_TICKS { return 0.0; }
    let amp: f32 = 1.0 - f32(tick) / f32(SPONT_TICKS);

    // Onda viajante no plano da folha: cos(kx*x + ky*y - ω*t).
    // kx, ky: comprimento de onda ≈ folha/4 (8 células).
    let kwave: f32 = 6.283185307 / 8.0;  // 2π/8
    let omega: f32 = 6.283185307 / 32.0; // 2π/32 ticks por ciclo
    let t: f32     = f32(tick);

    // Dois modos ortogonais para maximizar a estrutura de correlação 2D.
    let mode_a: f32 = cos(kwave * unit_x - omega * t);
    let mode_b: f32 = cos(kwave * unit_y - omega * t * 1.618);  // φ = razão áurea

    return amp * 0.5 * (mode_a + mode_b);
}

// ── Função auxiliar: ativação da vizinhança (toroidal, borda refletida) ─────
// Lê buf_act do tick anterior para o vizinho ortogonal deslocado de (dx, dy).
// Borda refletida: evita artefatos de vizinhança nula nas bordas da folha.
fn neigh_act(being: u32, ux: u32, uy: u32, dx: i32, dy: i32) -> f32 {
    // Calcula coordenadas com reflexão nas bordas
    let nx_i: i32 = clamp(i32(ux) + dx, 0, i32(SHEET_W) - 1);
    let ny_i: i32 = clamp(i32(uy) + dy, 0, i32(SHEET_H) - 1);
    let nu: u32 = u32(ny_i) * SHEET_W + u32(nx_i);
    return buf_act[being * SHEET_N + nu];
}

// ── Kernel A: 1 thread por unidade por ser ───────────────────────────────────
// workgroup_size(32, 1, 1): 32 threads por WG → 1024/32 = 32 WGs por ser.
// dispatch: (SHEET_N/32, n_beings, 1) = (32, n, 1).

@compute @workgroup_size(32, 1, 1)
fn afference(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let unit: u32  = gid.x;  // índice na folha [0, SHEET_N)
    let being: u32 = gid.y;  // índice do ser

    if unit >= SHEET_N || being >= params.n_beings { return; }

    // Posição 2D desta unidade na grade
    let ux: u32 = unit % SHEET_W;
    let uy: u32 = unit / SHEET_W;
    let ux_f: f32 = f32(ux);
    let uy_f: f32 = f32(uy);

    // ── Mecanismo 2: normalização divisiva — lê vizinhos DO TICK ANTERIOR ─────
    // Lê buf_act antes de qualquer escrita neste tick (garantido: threads de outros
    // workgroups do mesmo dispatch também só leram buf_act até este ponto,
    // pois ainda não houve barreira de memória global, mas as ESCRITAS deste tick
    // só ocorrem no final — WGSL garante que storage reads veem apenas valores
    // escritos por dispatches anteriores, não pelo dispatch em andamento).
    let an: f32 = abs(neigh_act(being, ux, uy,  0, -1));  // norte
    let as_: f32 = abs(neigh_act(being, ux, uy,  0,  1)); // sul
    let ae: f32 = abs(neigh_act(being, ux, uy,  1,  0));  // leste
    let aw: f32 = abs(neigh_act(being, ux, uy, -1,  0));  // oeste
    // Média dos 4 vizinhos ortogonais (atividade do campo local)
    let mean_neigh: f32 = (an + as_ + ae + aw) * 0.25;
    // Denominador da normalização divisiva: 1.0 = sem inibição, > 1.0 = inibido
    // div_sigma vem do uniform — configurável em tempo de execução sem recompilar.
    let div_denom: f32 = 1.0 + params.div_sigma * mean_neigh;

    // Acumula contribuição de cada canal sensorial com viés espacial gaussiano
    var total_input: f32 = 0.0;
    let wave_base:  u32  = being * N_INPUTS;
    let afw_base:   u32  = being * SHEET_N * N_INPUTS + unit * N_INPUTS;

    for (var k: u32 = 0u; k < N_INPUTS; k = k + 1u) {
        // Alvo espacial deste canal (determinístico, sem rótulo de região)
        let tgt: vec2<f32> = channel_target(k);
        let dx: f32 = ux_f - tgt.x;
        let dy: f32 = uy_f - tgt.y;
        let d_sq: f32 = dx * dx + dy * dy;

        // Peso gaussiano de aferência (calculado em tempo real — salvo apenas o
        // peso hebbiano aprendido em buf_afw). A gaussiana é o prior FRACO
        // (análogo a FGF8/Emx2); o Hebb em buf_afw refina depois.
        let gauss: f32 = exp(-d_sq / SIGMA_SQ);
        if gauss < AFFERENCE_EPS { continue; }

        // Sinal sensorial normalizado ∈ [-1, 1]
        let sig: f32 = f32(buf_wave[wave_base + k]) / 127.0;

        // Contribuição = gaussiana × peso hebbiano × sinal
        // buf_afw inicializado em 1.0 (sem aprendizado ainda = só prior gaussiano)
        let hebb_w: f32 = buf_afw[afw_base + k];
        total_input = total_input + gauss * hebb_w * sig;
    }

    // Onda espontânea endógena (fase inicial, doc §5.5)
    let spont: f32 = spontaneous_wave(ux_f, uy_f, params.tick);

    // Entrada total para esta unidade (pré-fadiga)
    let net_input: f32 = total_input + spont;

    // ── Mecanismo 1: fadiga adaptativa — lê estado lento do tick anterior ────
    let fat_idx: u32 = being * SHEET_N + unit;
    var fat: f32 = buf_fat[fat_idx];

    // Input efetivo: subtrai contribuição da fadiga.
    // Quando fat → max (unidade saturada), net_eff inverte sinal → quebra ponto fixo.
    // fat_beta vem do uniform — configurável em tempo de execução sem recompilar.
    let net_eff: f32 = net_input - params.fat_beta * fat;

    // EMA: h = α·h + (1−α)·entrada, onde α = exp(−dt/τ)
    // dt = 1/64 s ≈ 0.015625 s. τ por unidade (banco log-espaçado, gerado no host).
    let tau: f32    = buf_tau[being * SHEET_N + unit];
    let dt: f32     = 1.0 / 64.0;
    let alpha: f32  = exp(-dt / tau);

    var h: f32 = buf_state[being * SHEET_N + unit];
    h = alpha * h + (1.0 - alpha) * net_eff;

    // Ativação pós-EMA: tanh ∈ (-1, 1)
    let act_raw: f32 = tanh(h);

    // ── Mecanismo 2: normalização divisiva pela atividade local ───────────────
    // act_norm = act_raw / div_denom
    // Efeito: quando muitos vizinhos estão ativos (mean_neigh alto),
    // o ganho desta unidade cai → inibição lateral emergente.
    // Não é clamp nem reset — é redução de ganho por campo divisor.
    let act: f32 = act_raw / div_denom;

    // ── Mecanismo 1: atualiza fadiga ─────────────────────────────────────────
    // fat += alpha_fat * (|act_normalizado| - fat) * dt
    // Usa |act| normalizado (pós-divisão) — rastreia a saída real da unidade.
    fat = fat + SHEET_FAT_ALPHA * (abs(act) - fat) * dt;

    // Escreve de volta
    buf_state[being * SHEET_N + unit] = h;
    buf_act[being * SHEET_N + unit]   = act;
    buf_fat[fat_idx]                  = fat;
}
