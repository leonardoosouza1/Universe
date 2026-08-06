// ── FOLHA CORTICAL EQUIPOTENTE — KERNEL C: PROJEÇÃO MOTORA ───────────────────
//
// Projeta as ativações da folha para [throttle, steer, brake].
//
// PRINCÍPIO: quais unidades projetam para o motor NÃO é escolhido à mão.
// Usamos os pesos buf_mw (motor weights), que começam em zeros distribuídos
// aleatoriamente e são aprendidos/evoluídos. As unidades com maior |peso
// acumulado| dominam automaticamente a saída motora.
//
// buf_mw: f32[SHEET_N × 3] por ser. Layout: ser*SHEET_N*3 + unidade*3 + motor_idx.
// motor_idx: 0=throttle, 1=steer, 2=brake.
//
// Saída: buf_motor[ser*3 + idx] = tanh(Σ_i mw[i,idx] * act[i])

const SHEET_N:   u32 = 1024u;
const N_MOTOR:   u32 = 3u;  // throttle=0, steer=1, brake=2

struct Params {
    n_beings: u32,
    tick:     u32,
    _pad0:    u32,
    _pad1:    u32,
}

@group(0) @binding(0) var<uniform>             params:    Params;
// Ativações da folha: f32 por unidade por ser. Layout: ser*SHEET_N + unidade.
@group(0) @binding(1) var<storage, read>       buf_act:   array<f32>;
// Pesos de projeção motora: f32. Layout: ser*SHEET_N*3 + unidade*3 + motor_idx.
@group(0) @binding(2) var<storage, read>       buf_mw:    array<f32>;
// Saída motora: f32[3] por ser. Layout: ser*3 + motor_idx.
@group(0) @binding(3) var<storage, read_write> buf_motor: array<f32>;

// LDS para redução paralela de 3 motores
var<workgroup> wg_motor: array<f32, 3>;

// ── Kernel C: 1 workgroup por ser, 64 threads ─────────────────────────────────
// Cada thread acumula SHEET_N/64 unidades para os 3 motores.

@compute @workgroup_size(64, 1, 1)
fn motor(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let being: u32 = wgid.x;
    let tid:   u32 = lid.x;

    if being >= params.n_beings { return; }

    // Inicializa acumuladores no LDS (tid 0)
    if tid == 0u {
        wg_motor[0] = 0.0;
        wg_motor[1] = 0.0;
        wg_motor[2] = 0.0;
    }
    workgroupBarrier();

    // Acumulação local por thread (SHEET_N/64 = 16 unidades)
    var acc_t: f32 = 0.0;  // throttle
    var acc_s: f32 = 0.0;  // steer
    var acc_b: f32 = 0.0;  // brake

    let act_base: u32 = being * SHEET_N;
    let mw_base:  u32 = being * SHEET_N * N_MOTOR;

    for (var i: u32 = tid * 16u; i < (tid + 1u) * 16u; i = i + 1u) {
        if i >= SHEET_N { break; }
        let a: f32 = buf_act[act_base + i];
        let row: u32 = mw_base + i * N_MOTOR;
        acc_t = acc_t + buf_mw[row + 0u] * a;
        acc_s = acc_s + buf_mw[row + 1u] * a;
        acc_b = acc_b + buf_mw[row + 2u] * a;
    }

    // Redução via thread 0 (suficientemente simples — 64 acumulações)
    // Para o RDNA2 / WGSL, sem atomicAdd<f32> nativo, usamos tid 0 como master.
    if tid == 0u {
        var sum_t: f32 = 0.0;
        var sum_s: f32 = 0.0;
        var sum_b: f32 = 0.0;
        for (var j: u32 = 0u; j < SHEET_N; j = j + 1u) {
            let a: f32  = buf_act[act_base + j];
            let row: u32 = mw_base + j * N_MOTOR;
            sum_t = sum_t + buf_mw[row + 0u] * a;
            sum_s = sum_s + buf_mw[row + 1u] * a;
            sum_b = sum_b + buf_mw[row + 2u] * a;
        }
        // tanh para saturar saída em (-1, 1)
        let motor_base: u32 = being * N_MOTOR;
        buf_motor[motor_base + 0u] = tanh(sum_t);
        buf_motor[motor_base + 1u] = tanh(sum_s);
        buf_motor[motor_base + 2u] = tanh(sum_b);
    }
}
