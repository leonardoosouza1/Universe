// ── CÉREBRO LENTO NA GPU ──────────────────────────────────────────────────────
//
// Porta exata de SlowBrain (engine.rs) + CfCCell (cfc.rs) para WGSL.
//
// # Pipeline por ser (1 workgroup de 64 threads)
//
//   input[48] i8 → padding zeros → cfc_input[64]
//   → CfC(64, estado residente) → cfc_out[64] i8 = tanh(state)*127
//   → f32: cfc_f32[i] = cfc_out[i] / 127.0
//   → Dense1(64→16) com pesos f32 → relu → d1[16]
//   → Dense2(16→3) com pesos f32 → tanh → output[3]
//
//   Roda 1 tick a cada 8 (tick_interval = 8). Nos ticks pulados, output PERSISTE.
//
// # Layout de buffers (SoA)
//
//   buf_weights:  dense1.weights(1024) | dense1.bias(16) | dense2.weights(48) | dense2.bias(3)
//                 = 1091 f32 por ser. Ordem bate com SlowBrain::all_weights().
//
//   buf_state:    per ser: [cfc_state[64], cfc_fatigue[64], counter_u32_bitcast_f32,
//                           tick_counter_u32_bitcast_f32, output[3]]
//                 = 64 + 64 + 1 + 1 + 3 = 133 f32 por ser
//
//   buf_input:    [48 i32 por ser]  (sign-extended de i8 pelo host)
//
//   buf_output:   [3 f32 por ser]   (throttle, steer, brake) — resultado por tick
//
// # Kernel PEQUENO (RADV/ACO)
//
// Dois kernels encadeados pelo host no mesmo CommandEncoder:
//   Kernel A (entry point: run_cfc) — etapa CfC: input→state→cfc_out
//   Kernel B (entry point: run_dense) — etapa Dense1+Dense2→output
//
// Dividir em dois kernels foi a escolha que mantém cada shader < 100 linhas de
// corpo, dentro do limite que o RADV/ACO compilou sem travar neste projeto.
//
// # CfC — idêntico a cfc.rs
//
//   decay = exp(-0.5 * dt)
//   gain_dt = 1.0 * dt
//   x_raw = input_i32 / 127.0
//   limiar = FATIGUE_BETA * fatigue[i]
//   x = sign(x_raw) * max(0, |x_raw| - limiar)    ← encolhimento suave
//   s = s * decay + x * gain_dt
//   if h % 7 == 0: s += noise  (hash de Knuth, ~14% dos canais)
//   s = clamp(s, -2.0, 2.0)
//   fatigue[i] += FATIGUE_ALPHA * (|s| - fatigue[i]) * dt
//   cfc_out[i] = i32(clamp(tanh(s) * 127, -127, 127))
//
// FATIGUE_ALPHA = 1.0, FATIGUE_BETA = 0.4 (mesmos valores de cfc.rs)

const CFC_SIZE:      u32 = 64u;
const INPUT_SIZE:    u32 = 48u;   // input real; 48..63 são zeros
const D1_IN:         u32 = 64u;   // Dense1 entrada = CFC_SIZE
const D1_OUT:        u32 = 16u;   // Dense1 saída
const D2_IN:         u32 = 16u;   // Dense2 entrada = D1_OUT
const D2_OUT:        u32 = 3u;    // Dense2 saída (throttle, steer, brake)

// Slots do buf_state por ser (offsets em f32):
//   [0..64)   cfc_state
//   [64..128) cfc_fatigue
//   [128]     counter (u32 bitcast)
//   [129]     tick_counter (u32 bitcast)
//   [130..133) output[3]  (persiste entre ticks)
const STATE_PER_SER:    u32 = 133u;
const OFF_CFC_STATE:    u32 = 0u;
const OFF_CFC_FATIGUE:  u32 = 64u;
const OFF_COUNTER:      u32 = 128u;
const OFF_TICK_COUNTER: u32 = 129u;
const OFF_OUTPUT:       u32 = 130u;

// Pesos por ser em buf_weights:
//   [0..1024)    dense1.weights  (64×16, row-major: weights[j*64+i])
//   [1024..1040) dense1.bias     (16)
//   [1040..1088) dense2.weights  (16×3, row-major: weights[j*16+i])
//   [1088..1091) dense2.bias     (3)
const WEIGHTS_PER_SER:  u32 = 1091u;
const OFF_D1W:          u32 = 0u;
const OFF_D1B:          u32 = 1024u;   // 64*16 = 1024
const OFF_D2W:          u32 = 1040u;   // 1024 + 16
const OFF_D2B:          u32 = 1088u;   // 1040 + 48

// Parâmetros CfC (mesmos de cfc.rs e brain.wgsl)
const CFC_DECAY_RATE:  f32 = 0.5;
const CFC_INPUT_GAIN:  f32 = 1.0;
const CFC_CLAMP:       f32 = 2.0;
const CFC_DENOISE_MAG: f32 = 0.10;
const FATIGUE_ALPHA:   f32 = 1.0;
const FATIGUE_BETA:    f32 = 0.4;

// tick_interval: roda 1× a cada 8 ticks (engine.rs: tick_interval = 8)
const TICK_INTERVAL:   u32 = 8u;

// ── UNIFORM ───────────────────────────────────────────────────────────────────

struct Params {
    n_beings: u32,
    dt:       f32,
    _pad0:    u32,
    _pad1:    u32,
}

// ── BINDINGS (mesmos para os dois kernels) ────────────────────────────────────

@group(0) @binding(0) var<uniform>             params:      Params;

// input: 48 i32 por ser (sign-extended de i8 pelo host)
// Layout: ser * INPUT_SIZE + canal
@group(0) @binding(1) var<storage, read>       buf_input:   array<i32>;

// pesos Dense1 + Dense2 (imutáveis durante um tick; sobem por geração/spawn)
// Layout: ser * WEIGHTS_PER_SER + offset_local
@group(0) @binding(2) var<storage, read>       buf_weights: array<f32>;

// estado residente (cfc_state, cfc_fatigue, counter, tick_counter, output)
// Layout: ser * STATE_PER_SER + offset_local
@group(0) @binding(3) var<storage, read_write> buf_state:   array<f32>;

// output final [3 f32 por ser] para leitura rápida pelo host
// Escrito pelo kernel B (run_dense) apenas nos ticks que rodam.
// Nos ticks pulados o host lê buf_state[OFF_OUTPUT..] diretamente,
// mas manter este buffer separado permite copy_buffer_to_buffer sem aritmética.
// Layout: ser * 3 + {0,1,2}
@group(0) @binding(4) var<storage, read_write> buf_output:  array<f32>;

// ── MEMÓRIA DE WORKGROUP (compartilhada entre os 64 threads do mesmo ser) ─────

// Saída do CfC: tanh(state)*127 como i32, disponível para o Dense1.
// Gerada pelo kernel A, consumida pelo kernel B (via buffer de estado, não LDS,
// porque os dois kernels são dispatches separados — LDS não persiste entre eles).
// Dentro de cada kernel, usamos vars de workgroup para comunicação intra-kernel.

// Kernel A: w_cfc_out guarda o resultado do CfC para que o host possa
// lê-lo via buf_state se precisar, mas aqui serve para o kernel B.
var<workgroup> w_cfc_out: array<i32, 64>;

// Kernel B: w_d1_raw guarda a saída linear do Dense1 (antes do relu).
var<workgroup> w_d1_raw: array<f32, 64>;  // 64 slots, só D1_OUT=16 usados

// ── KERNEL A: CfC ─────────────────────────────────────────────────────────────
//
// 64 threads, uma por canal CfC.
// Lê buf_input[0..48] (zeros em 48..63), atualiza estado e fatigue,
// escreve cfc_out em buf_state[OFF_CFC_STATE..] como f32/127 (para ser lido
// pelo kernel B como f32 em vez de re-fazer a divisão).
//
// IMPORTANTE: também atualiza tick_counter. Se o tick for pulado, escreve
// o counter incrementado e retorna SEM tocar o output.
// O kernel B verifica o tick_counter para decidir se roda.

@compute @workgroup_size(64)
fn run_cfc(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let being = wgid.x;
    let tid   = lid.x;   // [0, 63]

    if being >= params.n_beings { return; }

    let state_base:   u32 = being * STATE_PER_SER;
    let input_base:   u32 = being * INPUT_SIZE;

    // Lê e incrementa tick_counter (apenas tid==0 escreve, todos lêem o valor atual).
    let tc_old: u32 = bitcast<u32>(buf_state[state_base + OFF_TICK_COUNTER]);
    let tc_new: u32 = tc_old + 1u;
    if tid == 0u {
        buf_state[state_base + OFF_TICK_COUNTER] = bitcast<f32>(tc_new);
    }

    // Se este tick não é um tick de córtex, não há nada a fazer.
    // O output anterior persiste em buf_state[OFF_OUTPUT] e buf_output.
    if (tc_new % TICK_INTERVAL) != 0u { return; }

    // ── Atualiza counter do denoising (independente do tick_counter) ──────────
    let cnt_old: u32 = bitcast<u32>(buf_state[state_base + OFF_COUNTER]);
    let cnt_new: u32 = cnt_old + 1u;
    if tid == 0u {
        buf_state[state_base + OFF_COUNTER] = bitcast<f32>(cnt_new);
    }

    // ── Lê estado do canal tid ────────────────────────────────────────────────
    let i: u32 = tid;
    var s:   f32 = buf_state[state_base + OFF_CFC_STATE   + i];
    var fat: f32 = buf_state[state_base + OFF_CFC_FATIGUE + i];

    let dt: f32     = params.dt;
    let decay: f32  = exp(-CFC_DECAY_RATE * dt);
    let gain_dt:f32 = CFC_INPUT_GAIN * dt;

    // Input do canal i: canais 0..47 vêm do buf_input; canais 48..63 são zeros.
    var x_raw: f32;
    if i < INPUT_SIZE {
        x_raw = f32(buf_input[input_base + i]) / 127.0;
    } else {
        x_raw = 0.0;
    }

    // Encolhimento suave — espelha cfc.rs exatamente
    let threshold: f32 = FATIGUE_BETA * fat;
    let x: f32 = sign(x_raw) * max(0.0, abs(x_raw) - threshold);

    // Equação CfC
    s = s * decay + x * gain_dt;

    // Ruído determinístico por hash de Knuth (~14% dos canais)
    let h: u32 = cnt_new * 2654435761u + i;
    if (h % 7u) == 0u {
        let noise: f32 = (f32(h >> 8u) / 16777216.0 - 0.5) * 2.0 * CFC_DENOISE_MAG;
        s = s + noise;
    }

    s = clamp(s, -CFC_CLAMP, CFC_CLAMP);

    // Atualiza fadiga
    fat = fat + FATIGUE_ALPHA * (abs(s) - fat) * dt;

    // Salva estado atualizado
    buf_state[state_base + OFF_CFC_STATE   + i] = s;
    buf_state[state_base + OFF_CFC_FATIGUE + i] = fat;

    // Salva cfc_out como f32 normalizado [-1,1] no estado temporário
    // (usamos OFF_OUTPUT+3..67 seria errado — guardamos em w_cfc_out via workgroup)
    // Como kernel B é um dispatch separado, comunicamos via buf_state.
    // Reservamos os slots OFF_CFC_STATE para state, mas o kernel B precisa de cfc_f32.
    // Solução: o kernel B refaz tanh(s)/1 — ou seja lê s de buf_state e aplica tanh.
    // Isso evita um buffer extra e mantém o código pequeno.
    // (cfc_out = tanh(s), que o kernel B recalculará lendo s de buf_state)
}

// ── KERNEL B: Dense1 + Dense2 ─────────────────────────────────────────────────
//
// 64 threads. Apenas D1_OUT=16 threads ativas para Dense1; D2_OUT=3 para Dense2.
// Threads 16..63 ficam ociosas — ok para RDNA2 (2 waves de 32, uma parcial).
//
// Fluxo:
//   1. Verifica tick_counter: se não é tick de córtex, copia buf_state→buf_output e retorna.
//   2. Recalcula cfc_f32[i] = tanh(buf_state[CFC_STATE+i]) para i em [0,64).
//   3. Dense1: 16 threads calculam w_d1_raw[j] = Σ weights[j*64+i] * cfc_f32[i] + bias[j]
//   4. ReLU inline: d1[j] = max(0, w_d1_raw[j])
//   5. Dense2: 3 threads calculam output[k] = tanh(Σ weights[k*16+j] * d1[j] + bias[k])
//   6. Salva output em buf_state[OFF_OUTPUT..] e buf_output[being*3..].

var<workgroup> w_cfc_f32: array<f32, 64>;
var<workgroup> w_d1_relu: array<f32, 16>;

@compute @workgroup_size(64)
fn run_dense(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let being = wgid.x;
    let tid   = lid.x;

    if being >= params.n_beings { return; }

    let state_base:   u32 = being * STATE_PER_SER;
    let weight_base:  u32 = being * WEIGHTS_PER_SER;
    let out_base:     u32 = being * D2_OUT;

    // Verifica se é tick de córtex (tick_counter já foi incrementado pelo kernel A)
    let tc: u32 = bitcast<u32>(buf_state[state_base + OFF_TICK_COUNTER]);
    if (tc % TICK_INTERVAL) != 0u {
        // Tick pulado: copia o output persistido do estado para buf_output.
        // Apenas 3 threads necessárias.
        if tid < D2_OUT {
            buf_output[out_base + tid] = buf_state[state_base + OFF_OUTPUT + tid];
        }
        return;
    }

    // ── Etapa 1: recalcula cfc_f32 (tanh do estado atualizado pelo kernel A) ──
    if tid < CFC_SIZE {
        let s: f32 = buf_state[state_base + OFF_CFC_STATE + tid];
        // A CPU faz uma ida-e-volta por i8 entre o CfC e o Dense1:
        //   output[i] = (tanh(s) * 127).clamp(-127,127) as i8   // `as i8` TRUNCA
        //   cfc_f32[i] = output[i] as f32 / 127
        // Isso quantiza em passos de 1/127. Usar `tanh(s)` direto parece inofensivo
        // (erro ≤ 0,004 por canal) mas são 64 canais somados com pesos ~0,35 — medido:
        // erro médio 2,9e-3 e pior 1,2e-2 na saída, cem vezes a tolerância.
        // A quantização faz parte da equação, não é detalhe de armazenamento.
        let q: f32 = clamp(tanh(s) * 127.0, -127.0, 127.0);
        w_cfc_f32[tid] = trunc(q) / 127.0;
    }

    workgroupBarrier();

    // ── Etapa 2: Dense1 (64→16) ───────────────────────────────────────────────
    // Thread tid ∈ [0,15] calcula neurônio j de saída do Dense1.
    if tid < D1_OUT {
        let j: u32 = tid;
        let w_base: u32 = weight_base + OFF_D1W + j * D1_IN;
        var acc: f32 = 0.0;
        for (var i: u32 = 0u; i < D1_IN; i = i + 1u) {
            // anti-FMA: produto separado antes de somar (consistência com CPU)
            let prod: f32 = buf_weights[w_base + i] * w_cfc_f32[i];
            acc = acc + prod;
        }
        acc = acc + buf_weights[weight_base + OFF_D1B + j];  // bias
        // ReLU
        w_d1_relu[j] = max(0.0, acc);
    }

    workgroupBarrier();

    // ── Etapa 3: Dense2 (16→3) → tanh ────────────────────────────────────────
    // Thread tid ∈ [0,2] calcula saída k.
    if tid < D2_OUT {
        let k: u32 = tid;
        let w_base: u32 = weight_base + OFF_D2W + k * D2_IN;
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j < D2_IN; j = j + 1u) {
            let prod: f32 = buf_weights[w_base + j] * w_d1_relu[j];
            acc = acc + prod;
        }
        acc = acc + buf_weights[weight_base + OFF_D2B + k];  // bias
        let out: f32 = tanh(acc);

        // Salva em dois lugares:
        //   buf_state: persiste para ticks futuros (output hold)
        //   buf_output: lido pelo host após readback
        buf_state[state_base + OFF_OUTPUT + k] = out;
        buf_output[out_base  + k]              = out;
    }
}
