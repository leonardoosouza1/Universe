// ── CÉREBRO NA GPU ────────────────────────────────────────────────────────────
//
// # Papel no sistema
//
// Este kernel substitui a parte NEURAL de `SovereignEngine::execute_cycle` para
// todos os seres em um único dispatch. A CPU continua dona do pipeline de evolução,
// dos timers de fadiga/metabolism, da Trinity (Gate + Reflex + Sentinel) e de tudo
// que tem branches pesados sobre state não-tensor.
//
// O que o kernel produz: `final_intent[0..96]` f32 por ser, equivalente ao campo
// `SovereignEngine::final_intent` ANTES da Trinity e do axiom_policy.
//
// # Pipeline por ser (1 workgroup de 128 threads)
//
//   sensory_wave[WAVE=192] ──► KAN por-canal (spline 16-knots)
//                       ──► CfC pré-processador (canais 0..96, state residente)
//                       ──► BitLinear L1  (192→96, pesos f32, input i8)
//                       ──► tanh × 127 → i8  (quantiza pra L2)
//                       ──► FloatLinear L2 (96→48, pesos f32)
//                       ──► [opcional, params.norm_l2_by_std] / desvio-padrão dos 48
//                       ──► final_intent[0..48]  = tanh(l2_raw × params.out_gain)
//                       + l1_raw[48..96] → tanh(× params.out_gain) → final_intent[48..96]
//                       (cortex_l3 NÃO é usado na CPU — omitido aqui)
//
//   `params.out_gain` (padrão 2.0, idêntico ao hardcode histórico) e
//   `params.norm_l2_by_std` (padrão 1/LIGADO desde P1.8 — ver docs/PLANO_EVOLUCAO.md
//   seção "P1.8", achado "L2 tem que funcionar") são os dois parâmetros do A/B
//   de P1.5 contra a saturação de L2 (48/48 neurônios ativos em 97,4% dos ticks,
//   entropia 0,301 nats). Ver `BrainParams` no host (`brain.rs`) e a ETAPA 2 abaixo.
//
//   # P1.8 (revisado 2026-08-14) — o que cada fatia de final_intent alimenta
//
//   TREINO (`gpu_evolve`): só [0..3) — throttle/steer/brake via `kan_eval`
//   (context.rs:954-957). Câmbio é automático por RPM; [3]/[4] NUNCA são lidos
//   nesse pipeline (a versão anterior deste comentário dizia "0-4 via shift" —
//   isso vale só no Bevy). ENGINE BEVY: [0..3) atuadores (motor.rs:2026-2097),
//   [3]/[4] câmbio (motor.rs:2451-2452), [0..64) entram em mielinização/coerência
//   MAS `myelinate()` é no-op deliberado desde a migração F32 (layers.rs:55-62)
//   e `axiom_policy` só escreve bias[0..3). RESUMO: [5..48) de L2 e o bypass
//   [48..96) não alimentam decisão em NENHUM pipeline — são "feature neurons
//   for evolution to explore" (commit 9ac2cdd) que nunca ganharam consumidor.
//   Evidência completa: docs/PLANO_EVOLUCAO.md "P1.8" + investigação L2
//   2026-08-14 (linhas cegas ≈ linhas lidas por carona genética do elitismo).
// # Layout de memória (SoA — Structure of Arrays)
//
// Por que SoA e não AoS?
// Com 60 seres e 223×96 = 21.408 pesos em L1 por ser, o total de L1 é ~1.3 MB.
// Matmul de L1: thread `j` (output neuron j) lê a linha inteira
// weights[j*WAVE .. (j+1)*WAVE] da memória do seu ser.
// Com SoA (ser0 inteiro → ser1 inteiro → ...) os 96 threads do mesmo workgroup
// lêem linhas distintas do MESMO ser — burst de leitura sequencial na VRAM.
// Com AoS (ser0_w0, ser1_w0, ..., ser0_w1, ser1_w1, ...) os acessos seriam
// intercalados entre seres → sem coalescência. SoA é a escolha correta.
//
// # Estado recorrente (CfC state) residente na GPU
//
// `buf_cfc` (binding 6) persiste entre ticks. O kernel lê o state, aplica a
// equação CfC e escreve de volta. A CPU só acessa este buffer em:
//   - spawn: inicializa zeros via `reset_cfc(ser_idx)`
//   - death: pode baixar para log (operação rara, não bloqueia tick)
//
// # Memória de workgroup
//
// Um workgroup por ser, 128 threads. LDS usada:
//   w_kan[223]  i32 pós-KAN         : 223×4 =  892 B
//   w_cfc[96]   i32 pós-CfC         :  96×4 =  384 B
//   w_l1r[96]   f32 acumuladores L1 :  96×4 =  384 B
//   w_l1a[96]   i32 l1 quantizado   :  96×4 =  384 B
//   w_l2r[48]   f32 l2_raw pré-ganho:  48×4 =  192 B  (P1.5, norm_l2_by_std)
//   Total: ~2,2 KB por workgroup. RDNA2 garante ≥ 32 KB LDS por workgroup.
//
// # Divergência GPU×CPU esperada
//
// 1. tanh/exp: WGSL IEEE 754-like, ≤ 1 ULP para |x| < 4. A saturação em ±1
//    ocorre nos dois lados antes de qualquer quantização. Diferença ≤ 1 LSB pós-i8.
//
// 2. Matmul L1/L2: acumulação f32 sequencial dentro de cada thread, mesma ordem
//    que a CPU (in_idx de 0 a N). Divergência esperada < 1e-5 por neurônio.
//
// 3. CfC — ruído determinístico: hash u32 com wrapping — identico à CPU
//    (Rust wrapping_mul = WGSL u32 operação que wraps por padrão).
//    ATENÇÃO: `denoise_counter` é salvo junto com o state em buf_cfc.
//
// 4. KAN axiom 1 (|x| < 0.001 → 0): f32 sem FTZ — identico à CPU.
//
// # O que ficou na CPU e por quê
//
// - Trinity (Gate + Reflex + Sentinel): lê SafetyState com campos opcionais do
//   Bevy (opt_body), acessa estado global de ablação (`crate::neuro::ablada()`).
//   Custo: ~2% do total. Branches pesados e divergentes.
//
// - axiom_policy: 25+ branches sobre flags booleanas por ser. Altamente
//   divergente numa GPU — as lanes se serializam. rayon na CPU paraleliza bem.
//
// - Mielinização (sort top-K por resonance_coherence): evento raro, sort para
//   N=96 seria possível mas não vale o código.
//
// - predictive coding, mirror neurons, metabolism: dependem de state Bevy não
//   tensorizado. Custo < 0.5% do total.
//
// - SlowBrain (CfC-64 → Dense → Dense, roda 1/8 ticks): custo ~1.7% do tick,
//   não vale portar o estado DenseLayer separadamente.

// ── CONSTANTES ────────────────────────────────────────────────────────────────

// Tamanho da sensory_wave (config.rs: SENSORY_WAVE_SIZE = 223)
const WAVE:      u32 = 192u;

// Palavras u32 da onda empacotada (4 canais i8 por u32). DEVE bater com o
// WAVE_U32S dos shaders de percepção (perception*.wgsl = 48). Era 56u hardcoded aqui
// (de quando a onda tinha 223 canais); ao reduzir para 192 esqueci este sítio, e o
// cérebro passou a ler a percepção com stride 56 enquanto ela é escrita com 48 —
// cada ser lia dados do vizinho. Bug medido e corrigido nesta sessão.
const WAVE_U32S: u32 = 48u;   // ceil(192/4)

// L1: BitLinear WAVE→96 (config.rs: L1_NEURONS = 96)
const L1_OUT:    u32 = 96u;

// L2: FloatLinear 96→48 (config.rs: L2_NEURONS = 48)
const L2_OUT:    u32 = 48u;

// KAN: 16 knots por canal (kan.rs: spline_coeffs [f32; 16])
const KAN_KNOTS: u32 = 16u;

// Parâmetros CfC.
//
// τ DEIXOU DE SER CONSTANTE. Aqui havia `CFC_DECAY_RATE = 0.5` e `CFC_INPUT_GAIN = 1.0`:
// um τ de 2,0 s idêntico nos 96 canais, inclusive nos de parede, curvatura e velocidade.
// Agora a taxa vem de `buf_tau` — um valor por canal por ser, herdado do genoma
// (espelha `cfc.rs: decay_rates`).
//
// `CFC_DC_GAIN` é o ganho em regime permanente, IGUAL para todo τ. O par antigo dava
// input_gain/decay = 1,0/0,5 = 2,0, então 2,0 mantém a escala do sinal. Sem essa
// normalização um canal rápido sairia ~80× mais fraco e seria descartado por ser mudo,
// não por ser rápido (ver `cfc.rs: dc_gain`).
const CFC_DC_GAIN:     f32 = 2.0;
const CFC_CLAMP:       f32 = 2.0;
const CFC_DENOISE_MAG: f32 = 0.10;  // ~14% dos canais perturbados por tick

// Parâmetros de fadiga adaptativa (spike-frequency adaptation)
//
// Mecanismo: neurônio que dispara forte por muito tempo reduz seu próprio ganho.
// É o que o cérebro real faz para evitar ponto fixo e saturação irreversível.
//
// Equação por canal i (a cada tick):
//   fatigue[i] += FATIGUE_ALPHA * (|state[i]| - fatigue[i]) * dt
//   x_efetivo[i] = sign(x[i]) * max(0, |x[i]| - FATIGUE_BETA * fatigue[i])
//
// FATIGUE_ALPHA = 1.0: filtro passa-baixa com τ ≈ 1s sobre |state|.
//   Com dt=1/64, alpha*dt ≈ 0.0156 → sobe em ~64 ticks (1s) para 63% do valor alvo.
//   Estado saturado em 2.0 → fatigue atinge ~1.8 em ~2s (128 ticks).
//   Mais rápido que a versão 0.3: garante que a fadiga sobe antes do ponto fixo
//   se consolidar (que ocorria em ~1000-1200 ticks com alpha=0.3).
//
// FATIGUE_BETA = 0.4: limiar aplicado sobre a MAGNITUDE do input.
//   Com fatigue=1.8 e beta=0.4 o limiar é 0.72: o canal PERDE ganho enquanto está
//   saturado sem emudecer, e recupera assim que a fadiga cede. O ponto fixo de entrada
//   máxima sustentada é s = 2/(1+2β) = 1.111 (saída 102/127) — o canal continua
//   informando, que é o contrato testado em `cfc.rs`. Com β=0.8 seria 0.769 (saída 82).
//   NÃO inverte o sinal: a versão anterior subtraía direto e uma entrada positiva
//   sustentada saía NEGATIVA, o que quebra os contratos testados em cfc.rs.
//   Driving normal: ser oscila entre aceleração/esterço → |state| varia → fat varia →
//   fadiga não sobe ao máximo (só sob disparo sustentado, i.e., ponto fixo).
const FATIGUE_ALPHA: f32 = 1.0;
const FATIGUE_BETA:  f32 = 0.4;  // ver cfc.rs: 2/(1+2β) preserva resposta sustentada

// Fadiga adaptativa na SAÍDA (output fatigue, slots [96..192] do buf_fatigue por ser)
//
// Problema diagnosticado: intent[0] (throttle) saturou em +1.0 desde o tick 10
// e nunca soltou — 46.249 alertas de |v|>0.9. A fadiga do CfC (input) quebra ponto
// fixo no estado recorrente, mas não alcança a saída L2/bypass diretamente.
//
// Mecanismo: mesmo filtro passa-baixa aplicado na saída final_intent.
//   out_fat[i] += OUT_ALPHA * (|intent_raw[i]| - out_fat[i]) * dt
//   intent_final[i] = tanh(sign(raw) * max(0, |raw| - params.out_beta * out_fat[i]))
//
// OUT_ALPHA = 4.0: τ ≈ 0.25s (sobe em ~16 ticks para 63% do alvo).
//   Resposta mais rápida: o throttle trava em 10 ticks, então a fadiga precisa agir
//   antes de 16 ticks para quebrar o ponto fixo de saída.
//   (v2: 2.0→4.0 para convergência mais rápida — com 2.0 levava 32 ticks para 63%
//    do alvo, lento demais quando l2_raw é grande e a subtração marginal não bastava.)
//
// OUT_BETA = 0.85: encolher a magnitude do raw em 0.85 × fat antes do tanh.
//   Simétrico por construção — `out_fat` rastreia |raw|, então subtrair com sinal
//   amplificava o lado negativo e atenuava o positivo (viés para baixo em toda a
//   população). O encolhimento sobre a magnitude trata os dois lados igual.
//   Em regime permanente (out_fat → |l2_raw|), intent → tanh(|l2_raw| × (1-0.85))
//                                                         = tanh(|l2_raw| × 0.15)
//   Com |l2_raw| = 3.8 (típico saturado): tanh(0.57) ≈ 0.54 → abaixo de 0.90 ✓
//   Com |l2_raw| = 4.5 (caso extremo):    tanh(0.675) ≈ 0.59 → abaixo de 0.90 ✓
//   Sinal variado (l2_raw oscilando): out_fat não converge para |l2_raw_max|,
//   fica em nível intermediário, sem penalidade excessiva sobre sinal legítimo.
//   (v2: 0.6→0.85 — com 0.6, em regime permanente ficava tanh(l2_raw*0.4) que para
//    l2_raw=3.8 ainda dava tanh(1.52)≈0.91, marginalmente acima do limiar de 0.90.)
const OUT_ALPHA: f32 = 4.0;
// OUT_BETA saiu daqui: agora vem de `params.out_beta`, por ser.

// ── STRUCTS ───────────────────────────────────────────────────────────────────

struct Params {
    n_beings:          u32,
    dt:                f32,
    // Quando 1: ignora buf_wave (binding 1) e desempacota o sensory_wave direto
    // de buf_percept (binding 8), que contém a saída de percepcao.wgsl (56 u32
    // por ser, i8 empacotados em little-endian). Elimina o readback percepção→CPU
    // e o write_buffer CPU→GPU que existia antes. O gpu_validate passa 0 aqui.
    use_percept_direct: u32,
    // Força da fadiga de saída. Era `const OUT_BETA = 0.85`; virou parâmetro para
    // poder ser medida em teste pareado (ver `BrainParams::out_beta` no host).
    out_beta:          f32,
    // P1.5: ganho aplicado a l2_raw e ao bypass L1 antes da fadiga/tanh na ETAPA 2.
    // Era `l2_raw = acc * (2.0/127.0)` e `raw_bp = w_l1r[tid] * 2.0` — dois `2.0`
    // hardcoded. Padrão 2.0 (`BrainParams::OUT_GAIN_PADRAO`) = comportamento
    // idêntico ao histórico; o A/B testa 2.0 vs 1.0 vs 0.5 (docs/PLANO_EVOLUCAO.md
    // P1.5) contra a saturação de L2 (48/48 neurônios ativos em 97,4% dos ticks).
    out_gain:          f32,
    // P1.5, segunda alavanca (padrão DESLIGADA=0): 1 = normaliza l2_raw pelo
    // desvio-padrão dos 48 l2_raw do MESMO ser antes do ganho/tanh — ataca a
    // ESCALA da entrada de L2 (que pode crescer com a mutação ao longo das
    // gerações) em vez de só o ganho de saída. Ver ETAPA 2 abaixo.
    norm_l2_by_std:    u32,
    // P1.7 (docs/PLANO_EVOLUCAO.md) — braço de controle dos três mecanismos
    // assíncronos, um bit por mecanismo. Era `_pad0` (padding morto).
    // bit0 = memória (mecanismo 1) desligada: alfa tratado como 0.0 no shader,
    //        INDEPENDENTE do que o genoma carrega (mesmo contrato de `--sem-lento`:
    //        o dado pode existir e ter mutado, o bit é que decide se ele é LIDO).
    // bit1 = período (mecanismo 2) desligado: período tratado como 1 (recalcula
    //        todo tick, igual hoje).
    // bit2 = limiar (mecanismo 2) desligado: limiar tratado como 0.0 (nunca zera
    //        contribuição por fraqueza).
    // bit3 = ativação heterogênea (mecanismo 3) desligada: id tratado como 0
    //        (tanh, a não-linearidade única de hoje).
    // 0 (padrão) = todos os mecanismos LEEM o genoma normalmente. Um genoma
    // recém-nascido (`Genome::random`) tem os quatro campos em valor neutro
    // (0.0/1.0/0.0/0), então mesmo com a máscara em 0 a saída da geração 0 é
    // idêntica à fórmula antiga — testado em
    // `gpu::evolution::tests::genoma_novo_tem_mecanismos_p17_neutros`.
    mech_off_mask:     u32,
    _pad1:             u32,
}

// ── BINDINGS ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform>            params:     Params;

// sensory_wave como i32 (sign-extended de i8 pelo host, evita bit-tricks no shader)
// Layout: ser * WAVE + canal
@group(0) @binding(1) var<storage, read>      buf_wave:   array<i32>;

// KAN: spline_coeffs[16] por canal por ser
// Layout: ser * WAVE * KAN_KNOTS + canal * KAN_KNOTS + knot_idx
@group(0) @binding(2) var<storage, read>      buf_kan:    array<f32>;

// L1 pesos (BitLinear WAVE→96, f32 runtime)
// Layout SoA: ser * (L1_OUT * WAVE) + neurônio_saída * WAVE + canal_entrada
@group(0) @binding(3) var<storage, read>      buf_l1w:    array<f32>;

// L2 pesos (FloatLinear 96→48, f32)
// Layout SoA: ser * (L2_OUT * L1_OUT) + neurônio_saída * L1_OUT + canal_entrada
@group(0) @binding(4) var<storage, read>      buf_l2w:    array<f32>;

// Saída: final_intent[96] f32 por ser
// Layout: ser * 96 + neurônio
@group(0) @binding(5) var<storage, read_write> buf_intent: array<f32>;

// Estado CfC residente entre ticks (96 floats de state + 1 u32 counter)
// Layout por ser: [state[0], ..., state[95], counter_u32_bitcast_f32]  = 97 × f32
// Offset por ser: ser * 97
@group(0) @binding(6) var<storage, read_write> buf_cfc:    array<f32>;

// DEBUG TEMPORÁRIO: exporta l1_raw (96 f32) + inputs para ser 0 (96+127 i32)
// Layout: [l1_raw[0..95], input_cfc[0..95], input_kan[96..222]]
// Tamanho: 96*4 + 96*4 + 127*4 = 1276 bytes
@group(0) @binding(7) var<storage, read_write> buf_debug:  array<f32>;

// Saída de percepcao.wgsl: 56 u32 por ser (223 i8 empacotados em little-endian).
// Layout: buf_percept[ser * 56 + ch/4], byte (ch % 4), interpretado como i8.
// Usado APENAS quando params.use_percept_direct == 1 (gpu_loop); ignorado pelo
// gpu_validate que usa buf_wave (binding 1) com i32 sign-extended da CPU.
// O tipo é u32 e não atomic<u32> porque aqui apenas LEMOS (percepcao.wgsl já terminou).
@group(0) @binding(8) var<storage, read>       buf_percept: array<u32>;

// Fadiga adaptativa por neurônio CfC — estado lento residente entre ticks.
// Layout: ser * L1_OUT + canal  (96 f32 por ser, inicializados em 0.0)
// Não salva counter separado: usa o mesmo counter do buf_cfc.
// Cresce quando |state[i]| é alto; subtrai do input CfC quebrando ponto fixo.
@group(0) @binding(9) var<storage, read_write> buf_fatigue: array<f32>;

// Constantes de tempo do CfC, uma por canal por ser — o "banco de τ".
// Layout: ser * L1_OUT + canal  (96 f32 por ser). Guarda a TAXA DE DECAIMENTO em s⁻¹
// (d = 1/τ), não τ, porque é a forma que entra direto no exponencial.
// Somente leitura: τ é herdado do genoma e fixo na vida do ser. Modular τ em tempo real
// foi medido e não paga (+0,11%); TER τ variado é que é selecionado (40× vs 5,3×).
@group(0) @binding(10) var<storage, read>      buf_tau:     array<f32>;

// ── P1.7 — CÉREBRO ASSÍNCRONO (docs/PLANO_EVOLUCAO.md §P1.7) ──────────────────
//
// Três mecanismos herdáveis, todos por-neurônio, todos nascendo NEUTROS (idênticos
// ao cérebro de hoje) — ver `gpu::evolution::Genome` para os campos e `evolve()`
// para a mutação. Crítica do Leonardo que motivou isto: *"o que estou pedindo é um
// cérebro ASSÍNCRONO; ele vai trabalhar em dimensões, neurônios e partes diferentes
// conforme está na pista; se você fica sempre no mesmo tanh só está queimando
// tempo."* — hoje os 20 usos de tanh neste arquivo são a ÚNICA não-linearidade, e
// os 96+48 neurônios de L1/L2 disparam em TODO tick (medido, não suposto).
//
// Metadados por neurônio — READ-ONLY, subido uma vez por geração (mesmo padrão de
// `buf_tau`: o dado é herdado do genoma e fixo na vida do ser).
// Layout por ser, NEURON_META_PER_SER=480 f32, contíguo em blocos:
//   [0..96)    l1_alpha       (mecanismo 1 — memória L1, ∈[0,0.95])
//   [96..192)  l1_period      (mecanismo 2 — período de recálculo L1, inteiro ≥1
//                              armazenado como f32, mesma convenção de disco/genoma
//                              de todo campo — só f32, ver genome_io.rs)
//   [192..288) l1_threshold   (mecanismo 2 — limiar de contribuição L1, ∈[0,~1))
//   [288..384) l1_activation  (mecanismo 3 — id de ativação L1: 0=tanh 1=relu
//                              2=linear 3=leaky-relu(0.01), armazenado como f32)
//   [384..432) l2_alpha       (mecanismo 1 — memória L2, ∈[0,0.95])
//   [432..480) l2_activation  (mecanismo 3 — id de ativação L2)
const NEURON_META_PER_SER: u32 = 480u;
// Offsets dos blocos dentro de cada bloco-de-ser de `buf_neuron_meta` (ver layout
// acima). `l1_alpha` fica no offset 0 (implícito, sem constante própria).
const NEURON_META_L1_PERIOD_OFF:     u32 = 96u;
const NEURON_META_L1_THRESHOLD_OFF:  u32 = 192u;
const NEURON_META_L1_ACTIVATION_OFF: u32 = 288u;
const NEURON_META_L2_ALPHA_OFF:      u32 = 384u;
const NEURON_META_L2_ACTIVATION_OFF: u32 = 432u;
@group(0) @binding(11) var<storage, read>      buf_neuron_meta: array<f32>;

// Estado persistente da memória EMA por neurônio (mecanismo 1) — COPIA O PADRÃO de
// `buf_cfc`: reside na GPU entre ticks, read_write, resetado em spawn/entre gerações
// (`BrainGpu::reset_cfc` agora zera este buffer junto com buf_cfc/buf_fatigue — ver
// host), NUNCA baixado por tick. Guarda h[n-1] — ver a conta do ganho de amplitude no
// comentário de `apply_neuron_memory` abaixo.
// Layout por ser, NEURON_MEM_PER_SER=144 f32: [0..96) memória L1, [96..144) memória L2.
const NEURON_MEM_PER_SER: u32 = 144u;
@group(0) @binding(12) var<storage, read_write> buf_neuron_mem: array<f32>;

// ── MEMÓRIA DE WORKGROUP ──────────────────────────────────────────────────────

var<workgroup> w_kan: array<i32, 223>;  // pós-KAN, pré-CfC: todos os canais
var<workgroup> w_cfc: array<i32, 96>;   // saída CfC (tanh(state) × 127, como i32)
var<workgroup> w_l1r: array<f32, 96>;   // acumuladores raw do L1 (pré-tanh)
var<workgroup> w_l1a: array<i32, 96>;   // L1 quantizado para entrada no L2 (i8 × 127)
var<workgroup> w_l2r: array<f32, 48>;   // l2_raw PRÉ-ganho (acc/127, sem params.out_gain)
                                          // — usado só quando params.norm_l2_by_std==1

// ── KAN filter_pulse (equivalente a KanAxiomFilter::filter_pulse, kan.rs:18-29) ─

// Variante que recebe raw_i32 (sinal inteiro em [-127,127]) e devolve o valor
// quantizado após KAN, já em escala de i32 (equivalente a i8).
//
// CORREÇÃO NUMÉRICA: a versão anterior fazia raw_f32 = raw_i32/127.0, depois
// filtered = raw_f32 * coeff, depois w_kan = i32(filtered * 127.0).
// Isso equivale a raw_i32 / 127.0 * coeff * 127.0. O round-trip /127 *127
// introduzia 1 ULP de diferença em relação à CPU, causando boundary crossing
// em alguns valores (ex: -18/127*1.0*127 != -18 exato em f32 no GPU).
// Evitar o round-trip: usar raw_i32 * coeff diretamente, sem dividir/multiplicar
// por 127. O resultado é clamp(raw_i32 * coeff, -127, 127).
fn kan_filter_int(raw_i32: i32, knots_base: u32) -> i32 {
    let raw_f32: f32 = f32(raw_i32);

    // Axiom 1: sinal zero não carrega direção (kan.rs:20-21)
    // Em escala de i32: |raw_i32| = 0 (identicamente zero) → 0.
    // Threshold original: |raw_f32/127| < 0.001 → |raw_f32| < 0.127 → |raw_i32| < 1 em inteiro.
    // Como raw_i32 é inteiro: apenas raw_i32 == 0 satisfaz estritamente |raw_i32| < 0.127.
    // NOTA: valor 1 em i8 → raw_f32_original = 1/127 ≈ 0.00787, que é > 0.001 → não zerado.
    // Logo o threshold correto em i32 é: raw_i32 == 0.
    if raw_i32 == 0 { return 0; }

    // Índice do spline: mapeado de raw_f32_normalized ∈ [-1,1] para [0,15].
    // raw_f32_normalized = raw_i32 / 127.0 (mas podemos calcular o idx sem dividir)
    // idx = (raw_i32/127.0 + 1.0) * 7.5 = (raw_i32 + 127.0) * 7.5 / 127.0
    //     = (f32(raw_i32) + 127.0) * (7.5 / 127.0)
    let idx_f: f32 = (raw_f32 + 127.0) * (7.5 / 127.0);
    let idx: u32   = u32(clamp(idx_f, 0.0, 15.0));
    let c: f32     = buf_kan[knots_base + idx];

    // Resultado sem round-trip: raw_i32 * coeff (em escala de i8 / 127)
    // Equivalente a: (raw_i32/127 * coeff) * 127 = raw_i32 * coeff
    return i32(clamp(raw_f32 * c, -127.0, 127.0));
}

// ── P1.7: ativação heterogênea por neurônio (mecanismo 3) ─────────────────────
//
// ids: 0=tanh (padrão, hoje) · 1=ReLU · 2=linear · 3=leaky-ReLU(0.01).
// Mesmas fórmulas de `math::activations` (src/math/activations.rs) — NÃO
// reimplementa lógica nova, só espelha o que já existe lá (regra da tarefa: "as
// funções JÁ EXISTEM, use de lá"). O clamp para a faixa i8 acontece no CHAMADOR
// (`clamp(activated * 127.0, -127.0, 127.0)`), igual ao que já existia para tanh —
// linear/ReLU sem essa saturação natural do tanh dependem inteiramente desse
// clamp externo para não estourar a quantização.
//
// CUSTO DE DIVERGÊNCIA DE WARP (RDNA2, wave32): as 32 lanes de uma wave executam
// em LOCKSTEP (SIMD). Se os neurônios mapeados nas lanes de uma MESMA wave tiverem
// ids diferentes, o hardware precisa percorrer os branches que QUALQUER lane da
// wave tomou — cada lane só grava o resultado do seu próprio caso (predicação),
// mas o tempo pago é o de executar todos os casos distintos presentes na wave, não
// só o seu. Com 96 neurônios de L1 = 3 waves de 32 (RDNA2 wave size, ver
// CLAUDE.md do projeto) e mutação de ativação numa taxa BAIXA (ver `evolve()` em
// `gpu/evolution.rs`), a fração de waves heterogêneas cresce devagar ao longo das
// gerações — o custo é real (não é grátis) mas pequeno: os 4 casos são ALU pura
// (sem acesso a memória divergente), poucas instruções cada.
fn apply_activation(id: u32, x: f32) -> f32 {
    if id == 1u { return max(x, 0.0); }                     // ReLU
    if id == 2u { return x; }                                // linear (satura no clamp do chamador)
    if id == 3u { return select(x, x * 0.01, x < 0.0); }      // leaky-ReLU(0.01)
    return tanh(x);                                           // 0 = tanh (padrão, idêntico a hoje)
}

// ── P1.7: memória por neurônio (mecanismo 1) ───────────────────────────────────
//
// h[n] = h[n-1]·alfa + x[n]·(1-alfa) — combinação convexa, coeficientes somam 1.
// Para ENTRADA CONSTANTE isso já tem ganho DC=1 em QUALQUER alfa (resolve h*=h*·alfa
// + x·(1-alfa) ⇒ h*=x), diferente do CfC (`buf_cfc` acima), que precisa do fator
// `CFC_DC_GAIN·(1-decay)` porque ali o "ganho" e o "decaimento" são parâmetros
// livres independentes. Aqui não há esse artefato — mas há OUTRO: a AMPLITUDE
// (variância) de um sinal que FLUTUA tick a tick (o caso real dos canais
// sensoriais — parede/curvatura/velocidade nunca são constantes) É atenuada por um
// filtro passa-baixa, mesmo com ganho DC exato em 1. Modelando x[n] como
// aproximadamente ruído branco de variância σx² (resposta ao impulso do filtro:
// h[n] = (1-alfa)·Σ_{k=0}^{n} alfa^k·x[n-k]):
//
//   Var(h) = (1-alfa)²·σx²·Σ alfa^{2k} = (1-alfa)²/(1-alfa²)·σx² = (1-alfa)/(1+alfa)·σx²
//
// Em alfa=0.95, Var(h)/Var(x) ≈ 0,026 — o neurônio pareceria 97% mais fraco (MUDO)
// e seria descartado por fraco, não por lento (GAP 11 do CLAUDE.md do projeto).
// O ganho que RESTAURA Var(saída)=Var(x) é o inverso da raiz desse fator:
//
//   gain(alfa) = sqrt((1+alfa)/(1-alfa))         [gain(0)=1 — sem mudança em alfa=0]
//
// Em alfa=0.95, gain≈6,24 — grande, mas finito porque alfa é clampado em 0.95 em
// `evolve()` (alfa≥1 diverge, condição de estabilidade da equação a diferenças:
// o polo do filtro fica em |alfa|<1). `max(1.0-alfa, 1e-3)` é só uma segunda trava
// numérica (o clamp de `evolve()` já impede alfa=1, isto é defesa em profundidade
// contra um genoma corrompido/legado fora de faixa).
const NEURON_MEM_GAIN_EPS: f32 = 1e-3;
fn neuron_mem_gain(alfa: f32) -> f32 {
    return sqrt((1.0 + alfa) / max(1.0 - alfa, NEURON_MEM_GAIN_EPS));
}

// ── KERNEL PRINCIPAL ──────────────────────────────────────────────────────────

@compute @workgroup_size(128)
fn main(
    @builtin(global_invocation_id) gid:  vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
    @builtin(workgroup_id)         wgid: vec3<u32>,
) {
    let being = wgid.x;    // um workgroup por ser
    let tid = lid.x;       // thread local [0, 127]

    // Mais workgroups do que seres vivos: guard padrão (sem dispatch arredondado)
    if being >= params.n_beings { return; }

    let wave_base: u32 = being * WAVE;
    let kan_base:  u32 = being * WAVE * KAN_KNOTS;

    // ─────────────────────────────────────────────────────────────────────────
    // ETAPA 0A: KAN per-channel (SensoryPreFilter::apply, kan.rs:72-79)
    //
    // WAVE = 223: precisa de 2 passes de 128 threads.
    //   Pass 1: tid ∈ [0,127] → canal tid       (todos os 128 threads ativam)
    //   Pass 2: tid ∈ [0, 94] → canal tid+128   (canais 128..222)
    // As escritas em w_kan são em posições disjuntas — sem conflito.
    //
    // FONTE DO WAVE: dois caminhos controlados por params.use_percept_direct.
    //   0 = buf_wave (binding 1): i32 sign-extended, enviado pela CPU via write_buffer.
    //       Usado pelo gpu_validate, que constrói o wave na CPU e passa para a GPU.
    //   1 = buf_percept (binding 8): 56 u32 por ser, saída de percepcao.wgsl.
    //       Layout: byte ch = (buf_percept[ser*56 + ch/4] >> ((ch%4)*8)) & 0xFF,
    //       interpretado como i8 two's-complement.
    //       Usado pelo gpu_loop: percepção → cérebro DENTRO da VRAM, zero readback.
    // ─────────────────────────────────────────────────────────────────────────

    // Desempacota o canal `ch` do sensory_wave de acordo com o modo de leitura.
    // Inline nos dois passes abaixo para evitar sobrecarga de chamada de função.

    // Pass 1: canais 0..127
    {
        let ch: u32 = tid;
        // ch < 128 sempre verdadeiro aqui; WAVE=223 > 128, sem guard necessário
        var raw_i32: i32;
        if params.use_percept_direct == 1u {
            // Desempacota i8 two's-complement de buf_percept (saída de percepcao.wgsl).
            // 56 u32 por ser: byte ch está em word (ch/4), shift ((ch%4)*8).
            let percept_base: u32 = being * WAVE_U32S;
            let word: u32   = buf_percept[percept_base + ch / 4u];
            let byte_u: u32 = (word >> ((ch % 4u) * 8u)) & 0xFFu;
            // Extensão de sinal u8 → i32 two's-complement:
            // se bit 7 estiver ligado (byte >= 128), o valor é byte - 256.
            raw_i32 = select(i32(byte_u), i32(byte_u) - 256, byte_u >= 128u);
        } else {
            raw_i32 = buf_wave[wave_base + ch];
        }
        w_kan[ch] = kan_filter_int(raw_i32, kan_base + ch * KAN_KNOTS);
    }
    // Pass 2: canais 128..222 (95 canais, 128-127=1 canal a mais que threads 0..94)
    {
        let ch: u32 = tid + 128u;
        if ch < WAVE {
            var raw_i32: i32;
            if params.use_percept_direct == 1u {
                let percept_base: u32 = being * WAVE_U32S;
                let word: u32   = buf_percept[percept_base + ch / 4u];
                let byte_u: u32 = (word >> ((ch % 4u) * 8u)) & 0xFFu;
                raw_i32 = select(i32(byte_u), i32(byte_u) - 256, byte_u >= 128u);
            } else {
                raw_i32 = buf_wave[wave_base + ch];
            }
            w_kan[ch] = kan_filter_int(raw_i32, kan_base + ch * KAN_KNOTS);
        }
    }

    workgroupBarrier();

    // ─────────────────────────────────────────────────────────────────────────
    // ETAPA 0B: CfC pré-processador (CfCCell::update, cfc.rs:59-86)
    //
    // Atualiza state[0..96] e escreve filtered_retina em w_cfc[0..96].
    // Apenas 96 threads (tid < 96) trabalham — 32 ficam ociosas.
    //
    // State layout em buf_cfc (por ser):
    //   [state[0], ..., state[95], counter_u32_bitcast_f32] = 97 × f32
    //   offset por ser: ser * 97
    //
    // Counter: incrementado ANTES do loop de canais (cfc.rs:68: wrapping_add(1)).
    // Todos os threads lêem o mesmo counter_old e calculam counter_new idêntico.
    // Apenas tid==0 escreve counter_new de volta (um write, sem race).
    // ─────────────────────────────────────────────────────────────────────────

    if tid < L1_OUT {  // L1_OUT = 96 = tamanho do CfC
        let i: u32        = tid;
        let cfc_base: u32 = being * 97u;

        // Estado atual do canal i
        var s: f32 = buf_cfc[cfc_base + i];

        // Fadiga adaptativa: lê estado lento de buf_fatigue.
        // Layout: buf_fatigue tem 192 slots por ser (FATIGUE_PER_SER = L1_OUT * 2).
        //   slots [0..96]   por ser = fadiga de INPUT do CfC (este bloco)
        //   slots [96..192] por ser = fadiga de SAÍDA (L2 + bypass, bloco etapa 2)
        //
        // CRÍTICO: usar being * (L1_OUT * 2u) — NOT being * L1_OUT.
        //   sendo * L1_OUT (= being * 96) faz ser N>0 acessar slots da fadiga de saída
        //   do ser anterior, causando cross-contamination e drift > 1.0 em 100 ticks.
        //   O fator 2u garante pulo de 192 slots por ser (tamanho real do bloco).
        let fat_base: u32 = being * (L1_OUT * 2u);
        var fat: f32 = buf_fatigue[fat_base + i];

        // Counter: lido por todos, incrementado por um (cfc.rs:68)
        let counter_old: u32  = bitcast<u32>(buf_cfc[cfc_base + 96u]);
        let counter_new: u32  = counter_old + 1u;  // wrapping em u32 por padrão em WGSL

        let dt: f32     = params.dt;
        // τ POR CANAL: taxa de decaimento herdada do genoma, em s⁻¹.
        let d: f32       = buf_tau[being * L1_OUT + i];
        let decay: f32   = exp(-d * dt);
        // Ganho DC normalizado — reaproveita o `decay` já calculado, sem exp extra.
        // Estacionário: s* = x·G·(1-decay)/(1-decay) = x·G, para qualquer τ.
        let gain_dt: f32 = CFC_DC_GAIN * (1.0 - decay);

        // x = input normalizado do canal, com fadiga aplicada como LIMIAR SUAVE:
        //   x_efetivo = sign(x) * max(0, |x| - beta * fat)
        // Espelha `cfc.rs` exatamente. A forma anterior (`x - beta*fat`, subtração crua)
        // INVERTIA o sinal: com fat≈1,8 e beta=0,8 ela tira 1,44 de um x∈[-1,1], então
        // entrada positiva sustentada saía negativa, e sem entrada alguma o estado era
        // empurrado para longe de zero. Os testes `update_stabilizes_on_constant_input` e
        // `state_decays_to_zero_without_input` em cfc.rs cobrem os dois casos.
        let x_raw: f32   = f32(w_kan[i]) / 127.0;
        let threshold: f32  = FATIGUE_BETA * fat;
        let x: f32       = sign(x_raw) * max(0.0, abs(x_raw) - threshold);

        // Equação CfC (cfc.rs:72): state = state * decay + x * gain_dt
        s = s * decay + x * gain_dt;

        // Ruído determinístico por hash (cfc.rs:74-80)
        // ~14% dos canais atingidos por tick (condição h % 7 == 0)
        // Não usando `if CFC_DENOISE_MAG > 0.0` como bloco externo porque WGSL
        // avalia constantes em compile-time — o branch será eliminado pelo compilador.
        let h: u32 = counter_new * 2654435761u + i;  // hash de Knuth (cfc.rs:75)
        if (h % 7u) == 0u {
            // Noise em [-CFC_DENOISE_MAG, +CFC_DENOISE_MAG] (cfc.rs:77-78)
            let noise: f32 = (f32(h >> 8u) / 16777216.0 - 0.5) * 2.0 * CFC_DENOISE_MAG;
            s = s + noise;
        }

        // Clamp (cfc.rs:82)
        s = clamp(s, -CFC_CLAMP, CFC_CLAMP);

        // Atualiza fadiga: filtro passa-baixa sobre |state| (spike-frequency adaptation).
        // fat += alpha * (|s| - fat) * dt
        // Com alpha=0.3, dt=1/64: fat cresce ~0.0047 por tick quando |s|=2.0.
        // Atinge 90% do valor de saturação em ~τ = 1/alpha ≈ 3.3 s ≈ 210 ticks.
        fat = fat + FATIGUE_ALPHA * (abs(s) - fat) * dt;
        buf_fatigue[fat_base + i] = fat;

        // Salva state atualizado (residente — lido no próximo tick)
        buf_cfc[cfc_base + i] = s;

        // filtered = tanh(state) → i8 (cfc.rs:84-85)
        w_cfc[i] = i32(clamp(tanh(s) * 127.0, -127.0, 127.0));

        // Apenas tid==0 atualiza o counter (evita race condition de escrita)
        if i == 0u {
            buf_cfc[cfc_base + 96u] = bitcast<f32>(counter_new);
        }
    }

    workgroupBarrier();

    // ─────────────────────────────────────────────────────────────────────────
    // ETAPA 1: BitLinear L1 — WAVE→96 (layers.rs:40-51)
    //
    //   l1_raw[j] = Σ_{i=0}^{WAVE-1} weights[j*WAVE+i] * input[i] / 127
    //
    // Input composto: w_cfc[0..96] (pós-CfC) ++ w_kan[96..WAVE] (pós-KAN)
    // 96 threads ativas (tid < 96), uma por neurônio de saída.
    // Threads 96..127 ficam ociosas — ok, RDNA2 fica 3 waves com 1 wave ociosa.
    //
    // Ordem de acumulação: i de 0 a WAVE-1, sequencial, igual à CPU.
    // Divergência de matmul esperada: < 1e-5 por neurônio (223 FMA f32).
    //
    // Layout buf_l1w: ser*(L1_OUT*WAVE) + j*WAVE + i  (row-major por ser)
    // ─────────────────────────────────────────────────────────────────────────

    if tid < L1_OUT {
        let j: u32 = tid;
        let l1_base: u32 = being * (L1_OUT * WAVE) + j * WAVE;
        // CORREÇÃO: acumula SEM normalizar a cada passo — multiplica por norm
        // UMA VEZ no final, idêntico à CPU (layers.rs:47-49):
        //   acc += w[i] * input[i]  (sem norm)
        //   output[j] = acc * norm_factor  (norm no final)
        // Aplicar norm a cada passo introduz erro de FP de ~5e-4 por acumulação,
        // que ao cruzar um limiar i8 amplifica para ~1.5e-2 na saída L2.
        var acc: f32     = 0.0;

        // Parte CfC: canais 0..96 (w_cfc contém filtered_retina)
        // ANTI-FMA: armazena produto em var antes de somar, evitando fusão em FMA
        // (GPU pode fundir a*b+c em FMA, gerando resultado diferente da CPU que faz
        //  separado: product = a*b; acc += product. Forçar a mesma ordem garante bit-a-bit.)
        for (var i: u32 = 0u; i < L1_OUT; i = i + 1u) {
            let prod: f32 = buf_l1w[l1_base + i] * f32(w_cfc[i]);
            acc = acc + prod;
        }
        // Parte KAN-only: canais 96..222 (passaram pelo KAN mas não pelo CfC)
        for (var i: u32 = L1_OUT; i < WAVE; i = i + 1u) {
            let prod: f32 = buf_l1w[l1_base + i] * f32(w_kan[i]);
            acc = acc + prod;
        }

        // Normaliza por 127 uma vez só — idêntico à CPU (layers.rs:49)
        w_l1r[j] = acc * (1.0 / 127.0);

        // DEBUG (ser 0 apenas): exporta l1_raw + inputs CfC/KAN para diagnóstico
        // Layout buf_debug: [l1_raw[0..95], cfc_in[0..95], kan_in[96..222]]
        if being == 0u {
            // l1_raw[j]
            buf_debug[j] = w_l1r[j];
            // inputs CfC (canais 0..95)
            if j < L1_OUT {
                buf_debug[96u + j] = f32(w_cfc[j]);
            }
            // inputs KAN (canais 96..222, 127 valores)
            // Cada thread j escreve 127/96 ≈ 1-2 entradas do KAN
            // Mais simples: só tid=0 escreve tudo (96 threads disponíveis, só 127 valores)
            // Alternativa: cada thread j escreve kan_in[j+96] se j < 127
            if j < 127u {
                buf_debug[192u + j] = f32(w_kan[j + 96u]);
            }
        }
    }

    workgroupBarrier();

    // Quantização l1_raw → l1_act (engine.rs:461-463), estendida com P1.7:
    //   l1_act[i] = clamp(ATIVAÇÃO(MEMÓRIA(PERÍODO/LIMIAR(l1_raw[i]))) × 127, -127, 127)
    //
    // Os quatro passos (mecanismo 2 → 1 → 3) vivem no MESMO bloco `if tid < L1_OUT`
    // que já existia (não abrem barreira nova): cada thread só lê/escreve o SEU
    // próprio índice em `w_l1r`/`buf_neuron_mem` — sem dependência cruzada entre
    // neurônios — então cabe dentro do bloco de quantização sem risco de race.
    // `w_l1r[tid]` é SOBRESCRITO com o valor pós-mecanismos: o bypass de L1
    // (ETAPA 2, threads 48..95, mais abaixo) lê o MESMO array e portanto herda o
    // efeito automaticamente — correto, porque memória/período/limiar são
    // propriedades do NEURÔNIO, não de "para onde o sinal vai depois".
    if tid < L1_OUT {
        let j: u32 = tid;
        let meta_base: u32 = being * NEURON_META_PER_SER;
        let mem_base:  u32 = being * NEURON_MEM_PER_SER;
        let off = params.mech_off_mask;

        // ── Mecanismo 2 (período) — decide se ESTE tick recalcula a memória ────
        // HONESTIDADE (CLAUDE.md/tarefa): na GPU todas as 96 threads rodam de
        // qualquer forma a cada dispatch — período > 1 NÃO poupa tempo de GPU, só
        // decide SE a memória é atualizada com o x[n] deste tick ou se mantém o
        // valor anterior. O ganho é de REPRESENTAÇÃO (escalas de tempo
        // convivendo — o pedido original do Leonardo), não de velocidade.
        let period_raw: f32 = buf_neuron_meta[meta_base + NEURON_META_L1_PERIOD_OFF + j];
        let period_off: bool = (off & 2u) != 0u;
        // +0.5 antes do truncamento u32(): arredonda em vez de truncar pra baixo
        // (período salvo como 2.999999 por erro de ponto flutuante não viraria 2).
        // max(...,1u): período 0 (buffer não inicializado, ex. `forward()`/
        // `gpu_validate`, que nunca sobe este buffer) cai em "recalcula todo tick",
        // igual ao comportamento de hoje — nunca "nunca recalcula".
        let period: u32 = select(max(u32(period_raw + 0.5), 1u), 1u, period_off);
        // counter_new já foi escrito em buf_cfc por tid==0 na ETAPA 0B, antes da
        // primeira workgroupBarrier() — visível aqui para todas as 96 threads.
        let tick_counter: u32 = bitcast<u32>(buf_cfc[being * 97u + 96u]);
        let fires: bool = period <= 1u || (tick_counter % period) == 0u;

        // ── Mecanismo 1 (memória) — EMA com ganho de amplitude ──────────────────
        // Ver a conta completa em `neuron_mem_gain` acima do kernel.
        let alfa_raw: f32 = buf_neuron_meta[meta_base + j];
        let mem_off: bool = (off & 1u) != 0u;
        let alfa: f32 = select(alfa_raw, 0.0, mem_off);
        var h: f32 = buf_neuron_mem[mem_base + j];
        if fires {
            // x[n] = l1_raw deste tick (matmul da ETAPA 1, ainda intocado aqui).
            h = h * alfa + w_l1r[j] * (1.0 - alfa);
            buf_neuron_mem[mem_base + j] = h;
        }
        // fora do tick do neurônio (!fires): `h` já é o valor guardado da última
        // vez que ele disparou — "mantém o valor anterior", sem tocar o buffer.
        var v_eff: f32 = h * neuron_mem_gain(alfa);

        // ── Mecanismo 2 (limiar) — abaixo dele, contribui zero ──────────────────
        let thr_raw: f32 = buf_neuron_meta[meta_base + NEURON_META_L1_THRESHOLD_OFF + j];
        let threshold_off: bool = (off & 4u) != 0u;
        let threshold: f32 = select(thr_raw, 0.0, threshold_off);
        if abs(v_eff) < threshold { v_eff = 0.0; }

        // Bypass (ETAPA 2, tid 48..95) e a quantização abaixo leem o valor final.
        w_l1r[j] = v_eff;

        // ── Mecanismo 3 (ativação heterogênea) ──────────────────────────────────
        let act_raw: f32 = buf_neuron_meta[meta_base + NEURON_META_L1_ACTIVATION_OFF + j];
        let activation_off: bool = (off & 8u) != 0u;
        let act_id: u32 = select(u32(act_raw + 0.5), 0u, activation_off);
        let activated: f32 = apply_activation(act_id, v_eff);
        w_l1a[tid] = i32(clamp(activated * 127.0, -127.0, 127.0));
    }

    workgroupBarrier();

    // ─────────────────────────────────────────────────────────────────────────
    // ETAPA 2: FloatLinear L2 — 96→48 (float_layers.rs:23-34)
    //          + bypass L1→final_intent[48..96]
    //
    // ENGINE (engine.rs:465-472):
    //   final_intent[0..48]  = tanh(l2_raw × out_gain)       — via L2
    //   final_intent[48..96] = tanh(l1_raw[48..96] × out_gain) — bypass de L2
    //   (`out_gain` era a constante `2.0` hardcoded; ver `params.out_gain` — P1.5)
    //
    // Por que bypass? L2 tem 48 outputs mas final_intent tem 96 slots.
    // Os slots 48..95 são preenchidos diretamente com l1_raw (pré-L2).
    //
    // Threads 0..47:  calculam L2 e escrevem final_intent[0..47]
    // Threads 48..95: escrevem bypass final_intent[48..95]
    // Threads 96..127: ociosas
    //
    // Layout buf_l2w: ser*(L2_OUT*L1_OUT) + j*L1_OUT + i
    // ─────────────────────────────────────────────────────────────────────────

    let intent_base: u32 = being * 96u;

    // ── Sub-etapa 2a: matmul L2, guarda l2_raw PRÉ-ganho em LDS ────────────────
    //
    // Separado do resto (fadiga+tanh) porque a normalização por desvio (item 3
    // de P1.5, `params.norm_l2_by_std`) precisa dos 48 valores de l2_raw do MESMO
    // ser prontos ao mesmo tempo — cada thread só calcula o SEU neurônio j, mas o
    // desvio-padrão é sobre os 48. `w_l2r` guarda a mesma grandeza que
    // `float_layers.rs::forward` devolve na CPU (acc/127, SEM ganho nenhum).
    if tid < L2_OUT {
        let j: u32 = tid;
        let l2_base: u32 = being * (L2_OUT * L1_OUT) + j * L1_OUT;
        // CORREÇÃO: acumula sem norm por passo — aplica 1/127 uma vez no final
        // (float_layers.rs:29-34: acc += w[i] * input[i]; output = acc * norm_factor)
        var acc: f32     = 0.0;

        // ANTI-FMA: same pattern as L1 — store product to var before adding
        for (var i: u32 = 0u; i < L1_OUT; i = i + 1u) {
            let prod: f32 = buf_l2w[l2_base + i] * f32(w_l1a[i]);
            acc = acc + prod;
        }

        let raw_l2: f32 = acc * (1.0 / 127.0);

        // ── P1.7, mecanismo 1 (memória) para L2 — sem período/limiar: a tarefa
        // pede período/limiar só para L1 ("a assincronia de verdade" mira a
        // ENTRADA sensorial, onde escalas de tempo diferentes fazem sentido físico;
        // L2 já lê L1 inteiro todo tick). L2 sempre "dispara" — `fires` implícito.
        // Mesma conta de ganho de `neuron_mem_gain` acima do kernel.
        let l2_meta_base: u32 = being * NEURON_META_PER_SER + NEURON_META_L2_ALPHA_OFF;
        let l2_mem_base:  u32 = being * NEURON_MEM_PER_SER + L1_OUT;  // slots [96..144) = L2
        let alfa_l2_raw: f32 = buf_neuron_meta[l2_meta_base + j];
        let mem_off_l2: bool = (params.mech_off_mask & 1u) != 0u;
        let alfa_l2: f32 = select(alfa_l2_raw, 0.0, mem_off_l2);
        var h_l2: f32 = buf_neuron_mem[l2_mem_base + j];
        h_l2 = h_l2 * alfa_l2 + raw_l2 * (1.0 - alfa_l2);
        buf_neuron_mem[l2_mem_base + j] = h_l2;

        w_l2r[j] = h_l2 * neuron_mem_gain(alfa_l2);
    }

    // Barrier em CONTROLE UNIFORME — TODAS as 128 threads passam por aqui, mesmo
    // as que não escreveram w_l2r (tid >= 48), porque WGSL exige que
    // workgroupBarrier() não fique dentro de um `if` que só parte das threads
    // executa (senão as que não entram no `if` nunca alcançam a barreira e o
    // driver trava ou diverge — a barreira DEVE estar em código não-divergente).
    workgroupBarrier();

    // ── Sub-etapa 2b: desvio-padrão populacional dos 48 l2_raw deste ser ───────
    //
    // Só entra em jogo quando `params.norm_l2_by_std == 1`; com a flag OFF
    // (padrão) `l2_std` fica em 1.0 e a divisão abaixo é um no-op exato — não
    // muda um bit do resultado histórico. Toda thread recalcula o mesmo valor
    // (redundante, mas 48 somas é desprezível ao lado do matmul 96×48 acima) —
    // mais barato que mais um workgroup array + mais uma barreira só para
    // broadcast de um escalar.
    //
    // Desvio POPULACIONAL (divisor N, não N-1): aqui a finalidade é normalizar
    // amplitude, não inferir a variância de uma amostra — mesmo espírito de
    // layer norm. NÃO subtrai a média: a hipótese em teste é que a SATURAÇÃO
    // vem da ESCALA de l2_raw crescendo com a mutação ao longo das gerações
    // (docs/PLANO_EVOLUCAO.md P1.5), não de um viés direcional — subtrair a
    // média removeria também um eventual sinal direcional genuíno (ex.: "vira
    // mais pra direita que pra esquerda nesta pista"), o que não é o que se
    // quer atacar aqui.
    var l2_std: f32 = 1.0;
    if params.norm_l2_by_std == 1u {
        var sum: f32 = 0.0;
        var sum_sq: f32 = 0.0;
        for (var k: u32 = 0u; k < L2_OUT; k = k + 1u) {
            let v: f32 = w_l2r[k];
            sum = sum + v;
            sum_sq = sum_sq + v * v;
        }
        let mean: f32 = sum / f32(L2_OUT);
        // max(0.0, ...): variância nunca é negativa por definição, mas o cálculo
        // via soma-de-quadrados (E[x²] - E[x]²) pode dar um epsilon negativo por
        // cancelamento de ponto flutuante quando os 48 valores são quase iguais.
        let variance: f32 = max(0.0, sum_sq / f32(L2_OUT) - mean * mean);
        // NORM_EPS evita explosão no início do treino, quando os 48 l2_raw ainda
        // são quase todos ~0 (pesos recém-sorteados, σ=0.30 diluído por 96
        // entradas) — dividir por um desvio ~0 amplificaria RUÍDO numérico, não
        // sinal. 1e-3 está bem abaixo de qualquer desvio genuíno observado
        // (l2_raw típico saturado ~3.8, ver comentário de OUT_BETA acima).
        const NORM_EPS: f32 = 1e-3;
        l2_std = max(sqrt(variance), NORM_EPS);
    }

    if tid < L2_OUT {
        // Thread tid ∈ [0,47]: neurônio j de L2
        let j: u32 = tid;

        // Normaliza pela ESCALA (l2_std=1.0 e a divisão é no-op se a flag estiver
        // desligada), depois aplica o ganho de saída — mesma ordem de grandeza
        // que `acc * (2.0/127.0)` tinha antes quando out_gain=2.0 e a normalização
        // está OFF: w_l2r[j] * out_gain == acc*(1/127)*2.0 == acc*(2.0/127.0).
        let l2_scaled: f32 = w_l2r[j] / l2_std;
        let l2_raw: f32 = l2_scaled * params.out_gain;

        // Fadiga de saída: lê slot [96+j] do buf_fatigue (segunda metade do buffer expandido).
        // Rastreia |intent| para canal j; subtrai params.out_beta * out_fat do raw antes do tanh.
        // Quebra o ponto fixo de saída sem afetar canais que variam normalmente.
        let out_fat_base: u32 = being * L1_OUT * 2u + L1_OUT;  // slots [96..192] por ser
        var out_fat: f32 = buf_fatigue[out_fat_base + j];
        out_fat = out_fat + OUT_ALPHA * (abs(l2_raw) - out_fat) * params.dt;
        buf_fatigue[out_fat_base + j] = out_fat;

        // Fadiga de saída como LIMIAR SUAVE sobre a MAGNITUDE:
        //   intent = tanh(sign(raw) * max(0, |raw| - params.out_beta * out_fat))
        // `out_fat` rastreia |raw|, então a forma anterior (`raw - OUT_BETA*out_fat`)
        // era ASSIMÉTRICA: com raw sustentado negativo ela subtraía um termo positivo
        // do já-negativo e empurrava para a saturação (-0,999), enquanto com raw
        // positivo atenuava para 0,15·raw. A fadiga enviesava toda a população para
        // baixo. Agora encolhe a magnitude e preserva o sinal, nos dois lados.
        let adapted: f32 = sign(l2_raw) * max(0.0, abs(l2_raw) - params.out_beta * out_fat);

        // P1.7, mecanismo 3 — ativação heterogênea de L2 (a não-linearidade FINAL
        // de L2, antes hardcoded como tanh). O bypass de L1 (abaixo) continua com
        // tanh fixo — squashing de SAÍDA MOTORA, não "ativação do neurônio de L1"
        // (que já foi decidida na quantização, ETAPA 1).
        let l2_act_base: u32 = being * NEURON_META_PER_SER + NEURON_META_L2_ACTIVATION_OFF;
        let act_raw_l2: f32 = buf_neuron_meta[l2_act_base + j];
        let activation_off_l2: bool = (params.mech_off_mask & 8u) != 0u;
        let act_id_l2: u32 = select(u32(act_raw_l2 + 0.5), 0u, activation_off_l2);
        // clamp: o comentário de apply_activation prometia "satura no clamp do
        // chamador", mas nenhum chamador clampava — ativação linear (id=2) vazava
        // valor CRU (medido no laboratório: slots 16/31/45 do campeão a até 3,19)
        // e inflava toda métrica que assume intent ∈ [-1,1] (saturação, dims_sat,
        // intent_mag). Motor (0..3) nunca foi afetado: kan_eval/context clampam.
        buf_intent[intent_base + j] = clamp(apply_activation(act_id_l2, adapted), -1.0, 1.0);
    }

    // Threads 48..95: bypass de L1 → final_intent[48..95]
    // (engine.rs:469-471): final_intent[i] = tanh(l1_raw[i] × out_gain) para i ∈ [48,95]
    // A normalização por desvio (params.norm_l2_by_std) NÃO se aplica aqui — o
    // nome é "norm_l2_by_std" porque mede e corrige a escala de l2_raw
    // especificamente; o bypass não passa pela matmul de L2, então não há
    // "l2_raw" aqui para normalizar.
    if tid >= L2_OUT && tid < L1_OUT {
        // Mesmo mecanismo de fadiga de saída para o bypass.
        let raw_bp: f32 = w_l1r[tid] * params.out_gain;
        let out_fat_base_bp: u32 = being * L1_OUT * 2u + L1_OUT;
        var out_fat_bp: f32 = buf_fatigue[out_fat_base_bp + tid];
        out_fat_bp = out_fat_bp + OUT_ALPHA * (abs(raw_bp) - out_fat_bp) * params.dt;
        buf_fatigue[out_fat_base_bp + tid] = out_fat_bp;

        let adapted_bp: f32 = sign(raw_bp) * max(0.0, abs(raw_bp) - params.out_beta * out_fat_bp);
        buf_intent[intent_base + tid] = tanh(adapted_bp);
    }

    // Nota: não há barreira necessária no final — cada thread escreve em posição
    // disjunta de buf_intent e não há leitura posterior neste dispatch.
}
