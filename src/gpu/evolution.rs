//! Seleção e mutação de genomas entre gerações — roda na CPU.
//!
//! O CUSTO está nos ticks (GPU). A evolução acontece entre gerações,
//! uma vez por N ticks. Portanto pode rodar na CPU sem penalidade significativa.
//!
//! # Genoma
//!
//! Um `Genome` carrega os pesos de L1 (BitLinear f32) e L2 (FloatLinear f32) do
//! `SovereignEngine`, além dos knots KAN de saída. São os mesmos campos que o
//! `EliteDna` do motor principal, mas sem os metadados de serialização.
//!
//! # Algoritmo evolutivo
//!
//! - **Seleção**: torneio (k=3) — seleciona o melhor entre 3 candidatos aleatórios.
//!   Menos propenso à deriva genética que a seleção proporcional.
//! - **Mutação gaussiana**: ruído N(0, σ) por peso, com σ=0.05 por padrão.
//!   Cada peso tem probabilidade `p_mutate` de ser perturbado.
//! - **Elitismo**: os melhores `n_elite` genomas passam sem mutação.
//!   Preserva o progresso; σ cobre a exploração.
//!
//! # Inicialização aleatória
//!
//! `Genome::random()` usa LCG com seed derivada do índice do ser. Magnitude σ=0.3
//! pra L1/L2. Pesos zerados (BitLinear::new()) → rede nula → nenhum movimento;
//! σ=0.3 coloca pesos na escala que o motor.rs usa (~0.1–0.5 após treinamento).
//!
//! # Inicialização espectral de L1 (P0.5)
//!
//! O sorteio N(0, σ=0.30) i.i.d. acima, medido com `analysis::activation` em 4
//! corridas independentes, produz ganho espectral de L1 (maior valor singular da
//! matriz feedforward 192→96, a mesma grandeza que `AlarmKind::SpectralRadiusHigh`
//! audita) em ~7.0 — bem acima do `|λ_max| <= ~1` que a teoria de echo-state pede
//! para o sinal sensorial não ser amplificado antes da primeira mutação. Depois de
//! sortear `l1`, `Genome::random` mede esse ganho com
//! `math::linalg::largest_singular_value` e re-escala TODOS os pesos de L1 por um
//! único fator escalar (`thresholds::SPECTRAL_INIT_TARGET / medido`) — um escalar
//! preserva a direção relativa de cada peso, só ajusta a magnitude, e como a medida é
//! LINEAR na matriz (`largest_singular_value(c·W) = c · largest_singular_value(W)`,
//! ver teste `largest_singular_value_escala_linear_com_a_matriz` em
//! `math/linalg.rs`), o fator atinge o alvo exatamente, não por aproximação.
//!
//! Controlado por [`SPECTRAL_INIT_ON`] (padrão: ligado) — o braço de controle do A/B
//! é `--sem-init-espectral` no `gpu_evolve`. Só toca `l1`: L2 não entra no alarme
//! (`collect_activation_stats("L2", ..., w_rec: None, ...)` em `gpu_evolve/main.rs`),
//! então não há "o que o alarme mede" para L2 mirar. `load_genome` (genoma vindo do
//! disco) NUNCA passa por `Genome::random` — constrói o `Genome` direto por struct
//! literal — então um campeão salvo antes desta mudança carrega com o ganho espectral
//! que ele sempre teve, intocado.

use crate::config::iara::brain::{SENSORY_WAVE_SIZE, L1_NEURONS, L2_NEURONS};
use crate::kan::KAN_OUT_KNOTS;
use crate::engine::SovereignEngine;
use crate::math::{linalg, thresholds};
use std::sync::atomic::{AtomicBool, Ordering};

/// Liga/desliga a inicialização espectral de L1 (P0.5) em `Genome::random`.
///
/// Padrão `true` (LIGADA) — nascer com ganho espectral ≈ `thresholds::SPECTRAL_INIT_TARGET`
/// é o comportamento recomendado, não um experimento à parte. `gpu_evolve
/// --sem-init-espectral` desliga (`store(false, ...)`) para rodar o braço de controle
/// do A/B: população nascendo com o σ=0.30 i.i.d. cru, sem correção — o mesmo cérebro
/// de antes da hipótese P0.5.
///
/// É um `AtomicBool` global (não um parâmetro de `Genome::random`) pelo mesmo motivo de
/// `world::NO_HOLES`/`world::EASY_TRACK`: `Genome::random(seed)` é chamado em várias
/// dezenas de lugares (população inicial, retomada de checkpoint, testes) e mudar a
/// assinatura obrigaria a tocar todos eles só para propagar uma flag de CLI que é lida
/// uma vez no início do processo.
pub static SPECTRAL_INIT_ON: AtomicBool = AtomicBool::new(true);

/// Fator escalar que leva um ganho `medido` para `alvo` (== `alvo / medido`).
///
/// Função pura, sem estado global — extraída de `Genome::random` para poder ser
/// testada sem depender de [`SPECTRAL_INIT_ON`] (que é global e mutável; um teste que
/// o alternasse arriscaria interferir com qualquer outro teste que chame
/// `Genome::random` em paralelo, ver R8 do CLAUDE.md do projeto — "canário: números
/// idênticos onde deveriam divergir").
///
/// `medido <= 1e-6` (L1 inteiro ~0, nunca observado com σ=0.30 e 18.432 pesos) devolve
/// `1.0` — nada para escalar, e `0.0 × 1.0 = 0.0` continua nulo, não uma divisão por
/// quase-zero explodindo o peso.
fn spectral_rescale_factor(medido: f32, alvo: f32) -> f32 {
    if medido > 1e-6 { alvo / medido } else { 1.0 }
}

// ── Dimensões derivadas ────────────────────────────────────────────────────────
pub const N_L1_WEIGHTS: usize = SENSORY_WAVE_SIZE * L1_NEURONS; // 192 × 96 = 18.432
pub const N_L2_WEIGHTS: usize = L1_NEURONS * L2_NEURONS;         // 96  × 48 =  4.608
pub const N_KAN_KNOTS:  usize = 3 * KAN_OUT_KNOTS;               // 3   × 12 =     36

/// Genoma completo de um ser: L1 + L2 + KAN de saída.
#[derive(Clone)]
pub struct Genome {
    /// Pesos L1 (instinct_layer): SENSORY_WAVE_SIZE × L1_NEURONS f32.
    /// Magnitude típica pós-treinamento: ~0.1–0.5. Init aleatório com σ=0.3.
    pub l1: Vec<f32>,
    /// Pesos L2 (cortex_l2): L1_NEURONS × L2_NEURONS f32.
    pub l2: Vec<f32>,
    /// Knots KAN de saída: 3 outputs × KAN_OUT_KNOTS f32.
    /// Init com KanOutputHead::new() (tanh ramp — equivale a identidade suave).
    pub kan_knots: [[f32; KAN_OUT_KNOTS]; 3],
    /// Pesos do cérebro LENTO na ordem de `SlowBrain::all_weights()`:
    /// dense1.weights(1024) | dense1.bias(16) | dense2.weights(48) | dense2.bias(3).
    ///
    /// Sem isto o lento não é um órgão, é uma constante: `SlowBrain::new()` usa
    /// sementes FIXAS, então todos os seres nasceriam com os mesmos pesos lentos e a
    /// modulação seria idêntica para a população inteira — a evolução não teria como
    /// ajustá-la nem como se livrar dela. Herdável e mutável, ela pode ser selecionada
    /// ou descartada pelo próprio processo.
    pub slow: Vec<f32>,
    /// Banco de τ: taxa de decaimento do CfC por canal (s⁻¹), `N_CFC_CHANNELS` valores.
    ///
    /// Nasce uniforme em `DECAY_DEFAULT` (τ = 2,0 s, o valor único de antes) e só vira
    /// heterogêneo por `randomize_tau`. Assim a condição de controle é literalmente o
    /// cérebro histórico rodando pelo código novo — mesmo shader, mesma fórmula, só o
    /// vetor difere.
    pub tau: Vec<f32>,
    /// Banco de sensores herdáveis (FASE SENSORES, degrau 1) — `N_SENSOR_GENES` valores
    /// na ordem de `sensor_idx`: FOV do sweep de parede/curvatura + alcance de parede +
    /// alcance lateral + 3 lookaheads de curvatura. Todos vivem em `perception.wgsl`
    /// (pipeline espacial) — nenhum muda o TAMANHO de um buffer GPU, só o VALOR que já
    /// era lido de uma constante de compilação.
    ///
    /// Nasce em `SENSOR_DEFAULTS` (os hardcodes de hoje, byte-idênticos aos literais do
    /// shader) e só vira heterogêneo quando `SENSORS_EVOLVE_ON` está ligado — mesmo
    /// precedente do banco de τ, que é o molde desta mudança (ver CLAUDE.md do projeto,
    /// pedido do Leonardo: "copie o padrão dele"): o braço de controle do experimento é
    /// literalmente os olhos de hoje, não um terceiro comportamento.
    pub sensors: Vec<f32>,

    // ── P1.7 — cérebro assíncrono (docs/PLANO_EVOLUCAO.md §P1.7) ────────────────
    //
    // Crítica do Leonardo que motivou os três campos abaixo: *"o que estou pedindo
    // é um cérebro ASSÍNCRONO; ele vai trabalhar em dimensões, neurônios e partes
    // diferentes conforme está na pista; se você fica sempre no mesmo tanh só está
    // queimando tempo."* Medido (não suposto): 20 usos de tanh em `brain.wgsl` e
    // nenhuma outra ativação; os 96 neurônios de L1 e 48 de L2 disparam em TODO
    // tick. O banco de τ do CfC (`tau`, acima) já é herdável/mutável e é o
    // PRECEDENTE copiado aqui — quando ele virou heterogêneo, quebrou a parede dos
    // 680 m do Universe.
    //
    // Todos os seis nascem NEUTROS (idênticos ao cérebro de hoje) em
    // `Genome::random`/`from_engine` — só a MUTAÇÃO em `evolve()` os afasta do
    // neutro, e só quando os atomics `NEURON_*_MUTATE_ON` (padrão `true`, abaixo)
    // permitem. `gpu_evolve --sem-X` desliga o atomic correspondente (mutação
    // congela no neutro pelo resto da corrida) E força `BrainGpu::*_off = true`
    // (o shader ignora o que quer que o genoma carregue, mesmo herdado de um
    // `--retomar` anterior à flag) — o braço de controle mais forte possível.
    /// Mecanismo 1 (memória por neurônio): alfa do filtro EMA
    /// `h[n] = h[n-1]·alfa + x[n]·(1-alfa)`, um por neurônio de L1(96)+L2(48).
    /// Nasce 0.0 (sem memória = idêntico a hoje). Clamp em `evolve()`: [0, 0.95] —
    /// alfa≥1 diverge (condição de estabilidade da equação a diferenças, o polo do
    /// filtro tem que ficar dentro do círculo unitário). Layout: `[0..96)`=L1,
    /// `[96..144)`=L2 — mesma ordem que `BrainGpu::NEURON_MEM_PER_SER` no host.
    pub neuron_alpha: Vec<f32>,
    /// Mecanismo 2 (período): a cada quantos ticks o neurônio de L1 recalcula sua
    /// memória (fora do seu tick, mantém o valor anterior — reusa o buffer do
    /// mecanismo 1). Nasce 1.0 (recalcula todo tick = idêntico a hoje). Inteiro
    /// armazenado como f32 (mesma convenção de todo campo do genoma — disco é
    /// posicional, só f32, ver `genome_io.rs`). Só L1 — só ali a assincronia
    /// temporal faz sentido físico (entrada sensorial, ver doc do mecanismo em
    /// `brain.wgsl`); L2 já lê L1 inteiro todo tick.
    pub neuron_period_l1: Vec<f32>,
    /// Mecanismo 2 (limiar): abaixo dele, o neurônio de L1 contribui zero. Nasce
    /// 0.0 (sempre contribui = idêntico a hoje, já que a magnitude pós-ativação
    /// nunca é negativa). Só L1, mesmo motivo do período.
    pub neuron_threshold_l1: Vec<f32>,
    /// Mecanismo 3 (ativação heterogênea): id de ativação por neurônio de
    /// L1(96)+L2(48). 0=tanh 1=ReLU 2=linear 3=leaky-ReLU(0.01) — ver
    /// `math::activations` para as fórmulas e `brain.wgsl::apply_activation` para
    /// o espelho GPU. Nasce 0.0 (tanh = idêntico a hoje). Layout: `[0..96)`=L1,
    /// `[96..144)`=L2.
    pub neuron_activation: Vec<f32>,
    /// M4 (REGIOES_CEREBRAIS.md): pesos do maestro — 4 gates de região × (3 saídas
    /// do cérebro lento + bias) = 16 f32. gate_b = 1 + 0,5·tanh(w·slow_out + b).
    /// Herdável e mutado como o lento. Bloco OPCIONAL no disco (genome_io).
    pub gates_w: Vec<f32>,
}

/// Neurônios de L1 e L2 herdáveis pelos mecanismos de P1.7 — mesmos números de
/// `config::iara::brain::{L1_NEURONS, L2_NEURONS}`, reexportados aqui para
/// `genome_io.rs` dimensionar os blocos do disco sem duplicar o import.
pub const N_L1_NEURONS: usize = L1_NEURONS;
pub const N_L2_NEURONS: usize = L2_NEURONS;

/// P1.7 — braço de controle da MUTAÇÃO do mecanismo 1 (memória). Padrão `true`:
/// `evolve()` muta `neuron_alpha` como qualquer outro campo herdável (mesmo
/// precedente do banco de τ, mutado incondicionalmente). `gpu_evolve
/// --sem-memoria-neuronal` desliga — o campo fica CONGELADO em 0.0 (o valor de
/// nascença) pelo resto da corrida, e `BrainGpu::mem_off` é ligado junto (o shader
/// ignora o campo mesmo que um `--retomar` anterior já tivesse mutado algo).
pub static NEURON_MEM_MUTATE_ON: AtomicBool = AtomicBool::new(true);
/// P1.7 — idem para o mecanismo 2, sub-parte período (`--sem-periodo-neuronal`).
pub static NEURON_PERIOD_MUTATE_ON: AtomicBool = AtomicBool::new(true);
/// P1.7 — idem para o mecanismo 2, sub-parte limiar (`--sem-limiar-neuronal`).
pub static NEURON_THRESHOLD_MUTATE_ON: AtomicBool = AtomicBool::new(true);
/// P1.7 — idem para o mecanismo 3, ativação heterogênea (`--sem-ativacao-heterogenea`).
pub static NEURON_ACTIVATION_MUTATE_ON: AtomicBool = AtomicBool::new(true);

/// Número de parâmetros sensoriais herdáveis (FASE SENSORES, degrau 1).
///
/// Cobre só os sensores BARATOS: mudar o VALOR não muda o tamanho de nenhum buffer GPU,
/// só a fórmula que já existe em `perception.wgsl` (pipeline espacial — `buf_sensors`,
/// binding 7 de `layout_spatial`). Ordem fixa, usada em `Genome::random`,
/// `pack_sensors`, `save_genome`/`load_genome` e no shader (`perception.wgsl`, a mesma
/// constante `N_SENSOR_GENES` do lado GPU) — os dois lados têm que concordar na ordem:
/// não há nome de campo no disco nem no buffer, só posição (mesma regra do genoma
/// inteiro, ver `genome_io.rs`).
///
/// FORA DE ESCOPO nesta fase (mapa de exploração que motivou esta mudança):
/// - **Resolução da retina / nº de neurônios**: mudam o TAMANHO de um buffer GPU (o
///   próprio genoma, no caso de L1/L2) — não é "sensor", é a FORMA do genoma. Formato
///   `.bin` é posicional; mudar isso invalida todo `.bin` salvo.
/// - **Alcance da retina** (`visao::Params`, `ALCANCE_PADRAO`): o mecanismo já existe
///   (uniform + testes próprios) mas `gpu_evolve` NUNCA chama `visao::definir()` — só o
///   binário Bevy/UI (`main.rs`) chama. Herdar isso agora criaria um gene MORTO no
///   binário de treino (gap 10 do CLAUDE.md: "estado calculado e não lido"). Conectar
///   `gpu_evolve` a `visao::Params` é pré-requisito de um degrau futuro, não desta.
/// - **Alcance do whisker** (`WHISKER_RANGE`): duplicado em `motor.rs` (CPU/Bevy, fora
///   do caminho de `gpu_evolve`) e `perception.wgsl` (GPU) sem constante compartilhada.
///   Herdar só o lado GPU faria o binário de treino divergir do de visualização sem
///   nenhum aviso — adiado até existir uma fonte única entre os dois caminhos.
/// - **Curvatura 200/260/275/350 m** (`perception_temporal.wgsl`, ch101/111-113, e
///   `motor.rs::curv_at(260.0)` no caminho CPU): vivem num PIPELINE diferente
///   (`layout_temporal`, não `layout_spatial`) — entram como o PRÓXIMO gene sensorial,
///   mesma receita, buffer novo naquele layout. Não incluídos aqui para manter o
///   primeiro corte pequeno o bastante para revisar de verdade.
pub const N_SENSOR_GENES: usize = 6;

/// Índices dos genes sensoriais — mesma ordem em `SENSOR_DEFAULTS`, `SENSOR_RANGES`,
/// `perception.wgsl` (`N_SENSOR_GENES`/`sensors[being * N_SENSOR_GENES + i]`) e no
/// arquivo `.bin` no disco.
pub mod sensor_idx {
    pub const FOV_HALF_RAD: usize = 0;
    pub const WALL_RANGE_M: usize = 1;
    pub const LAT_RANGE_M:  usize = 2;
    pub const CURV_NEAR_M:  usize = 3;
    pub const CURV_MID_M:   usize = 4;
    pub const CURV_FAR_M:   usize = 5;
}

/// Valor de nascença de cada gene sensorial — os hardcodes de HOJE, byte-idênticos aos
/// literais em `perception.wgsl` (antigo `FOV_HALF_RAD`, `WALL_MAX`, `LAT_SAMPLES[9]` e
/// os 3 `dists` do BLOCO K). Usado quando `SENSORS_EVOLVE_ON` está desligado (padrão) e
/// como fallback de `load_genome` para um `.bin` gravado antes deste campo existir — nos
/// dois casos o objetivo é reproduzir o comportamento de hoje, não inventar um.
pub const SENSOR_DEFAULTS: [f32; N_SENSOR_GENES] = [
    0.872664625997, // fov_half_rad: 50° — literal idêntico ao antigo perception.wgsl
    60.0,           // wall_range_m: WALL_MAX de hoje
    40.0,           // lat_range_m: LAT_SAMPLES[9] de hoje (o alcance máximo da amostra)
    40.0,           // curv_near_m: BLOCO K dists[0]
    70.0,           // curv_mid_m:  BLOCO K dists[1]
    120.0,          // curv_far_m:  BLOCO K dists[2]
];

/// Faixas FISICAMENTE válidas de cada gene — usadas no sorteio inicial (quando a
/// evolução está ligada, `random_sensors_within_ranges`) e como CLAMP depois de toda
/// mutação (`mutate_sensors`), para nunca produzir um sensor impossível.
///
/// # De onde vem cada faixa
///
/// - **fov_half_rad** (½-ângulo 30°..85°, abertura total 60°..170°): a faixa de
///   exemplo pedida na tarefa. Piso 60° de abertura: abaixo disso as 8 amostras fixas
///   do sweep (BLOCO C/D/E de `perception.wgsl`) ficam espremidas numa fatia <7,5° à
///   frente do carro — perde visão periférica que o BUG 1 (histórico, ver comentário de
///   `LAT_SAMPLES` no shader) já mostrou custar caro. Teto 170°: acima disso o
///   ½-ângulo passa de 85°, quase olhando para trás; o sweep tem 8 ângulos FIXOS (não
///   adaptativos), então gastaria metade da resolução atrás do carro, onde nenhum
///   canal de saída usa o dado.
/// - **wall_range_m** (20..120 m): piso 20 m ≈ 0,71 s de antecipação a 28 m/s
///   (velocidade máxima observada — mesmo número que já justifica `LAT_SAMPLES` neste
///   arquivo), abaixo do qual o sweep de parede reage tarde demais para o corpo (física
///   em `physics_a.wgsl`) frear a tempo. Teto 120 m: mesma ordem de grandeza do bloco
///   vizinho no MESMO kernel (road_ahead FAR já cobre 45..150 m) — um wall_range muito
///   maior que o road_ahead faria o sensor "ver" parede antes de qualquer canal de
///   pista correspondente, produzindo um alarme sem contexto.
/// - **lat_range_m** (15..100 m): piso 15 m ≈ 0,54 s a 28 m/s — mesma lógica do BUG 1
///   já documentado no shader (27 m / 0,96 s foi medido "marginal"; 15 m fica abaixo só
///   porque o sweep lateral tem 10 amostras densas perto do corpo, ao contrário do
///   frontal, que tem 1 canal só). Teto 100 m: mesma ordem de grandeza do wall_range,
///   sem ultrapassar uma fração pequena da grade (320 células × 3 m = 960 m; 100 m é
///   <11% disso — continua um sensor "local", não um radar de mundo inteiro).
/// - **curv_near/mid/far_m** (15..80 / 40..150 / 80..250 m): os três lookaheads de
///   curvatura do BLOCO K, hoje fixos em 40/70/120 m. As faixas dão folga de ~2× para
///   cada lado do valor histórico (que fica perto do meio de cada faixa, não na borda),
///   coerente com o teto de detecção de curva já usado no resto do arquivo
///   (`perception_temporal.wgsl` sonda curvatura até 350 m). NÃO forçamos
///   near < mid < far depois da mutação: os nomes são só a intenção de NASCENÇA — a
///   função `curv_at_d` (equivalente WGSL) não distingue "qual" distância é qual, só lê
///   o valor recebido, então a evolução pode redistribuir os três livremente.
pub const SENSOR_RANGES: [(f32, f32); N_SENSOR_GENES] = [
    // 30°..85° in radians. The lower bound is exactly FRAC_PI_6, so use the constant
    // rather than a rounded literal; the upper bound has no named constant.
    (std::f32::consts::FRAC_PI_6, 1.483529864), // fov_half_rad
    (20.0, 120.0),               // wall_range_m
    (15.0, 100.0),                // lat_range_m
    (15.0, 80.0),                 // curv_near_m
    (40.0, 150.0),                // curv_mid_m
    (80.0, 250.0),                // curv_far_m
];

/// Fração da FAIXA de cada gene usada como σ da mutação gaussiana (ver `mutate_sensors`).
/// 8%: rápido o bastante para explorar a faixa inteira em dezenas de gerações — mesma
/// ordem de grandeza da escolha já feita para `slow`/`tau` acima (poucos genes contra
/// dezenas de milhares de pesos, então taxa e magnitude maiores que L1/L2).
pub const SENSOR_MUTATE_FRAC: f32 = 0.08;

/// Liga/desliga a evolução do banco de sensores (FASE SENSORES, degrau 1: FOV do sweep
/// de parede/curvatura, alcance de parede, alcance lateral, 3 lookaheads de curvatura —
/// ver `N_SENSOR_GENES`).
///
/// Padrão `false` (DESLIGADA). Regra do CLAUDE.md deste projeto: *"toda mudança atrás
/// de flag com padrão = comportamento de hoje, para não invalidar bateria em curso"* —
/// e ao contrário de `SPECTRAL_INIT_ON` (que já tinha 4 corridas medindo o ganho ~7.0
/// antes de ganhar default `true`), esta mudança não tem NENHUMA medição prévia. Então:
///
/// - Desligado (padrão): `Genome::random` sorteia `sensors = SENSOR_DEFAULTS` sempre
///   (todo genoma nasce idêntico — os hardcodes de hoje) e `evolve()` NUNCA muta o
///   campo. O comportamento de qualquer bateria que não passe uma flag nova é
///   bit-idêntico ao de antes desta mudança existir.
/// - Ligado (`--sensores-evolutivos`): `Genome::random` sorteia dentro de
///   `SENSOR_RANGES` (heterogêneo desde o nascimento — diferente do precedente do τ,
///   que só varia por `randomize_tau` opt-in; aqui a própria tarefa pediu sorteio
///   direto em `Genome::random`) e `evolve()` muta o campo a cada geração.
///
/// `--sensores-fixos` força de volta para `false` — é o nome literal pedido pela
/// tarefa. Mesmo sendo o padrão já hoje, existir como flag permite documentar a
/// intenção explicitamente num comando (e vence se aparecer DEPOIS de
/// `--sensores-evolutivos` na linha de comando), do mesmo jeito que `--sem-lento` e
/// `--sem-init-espectral` são os braços de controle explícitos de outros experimentos.
///
/// # O que NÃO herda deste campo (paridade CPU/GPU documentada, não escondida)
///
/// `apply_to_engine`/`from_engine` (o caminho CPU/Bevy, via `SovereignEngine`) não
/// tocam `sensors` — o `motor.rs` (Bevy) nunca leu `perception.wgsl`, só o binário de
/// treino (`gpu_evolve`) e o `gpu_validate` leem. Um campeão evoluído com sensores
/// heterogêneos e depois carregado no visualizador Bevy vai VER com o FOV/alcances
/// hardcoded de sempre — divergência conhecida, mesma categoria já aceita para
/// `out_gain`/P1.5 (`docs/PLANO_EVOLUCAO.md`), não um bug silencioso.
pub static SENSORS_EVOLVE_ON: AtomicBool = AtomicBool::new(false);

/// Abstração mínima para os dois números que toda mutação de genoma neste arquivo
/// consome: "sorteou dentro da taxa de mutação?" (`chance`, uniforme `[0,1)`) e "quanto
/// perturbar" (`gauss`, N(0,1)).
///
/// Existe só para `mutate_sensors` poder receber UM gerador em vez de dois closures
/// `FnMut` fechando sobre a mesma variável — o `Rng` local de `evolve()` é `&mut`
/// emprestado por CADA closure, e dois `FnMut` vivos ao mesmo tempo sobre a mesma
/// variável são dois empréstimos mutáveis simultâneos, que o borrow checker rejeita.
/// Uma referência `&mut dyn GenoRng` só empresta uma vez.
pub trait GenoRng {
    fn chance(&mut self) -> f32;
    fn gauss(&mut self) -> f32;
}

impl GenoRng for crate::rng::Lcg {
    fn chance(&mut self) -> f32 { self.uniform() }
    fn gauss(&mut self) -> f32 { self.gaussian() }
}

/// Sorteia os `N_SENSOR_GENES` genes uniformemente dentro de `SENSOR_RANGES` — a
/// condição HETEROGÊNEA (usada só quando `SENSORS_EVOLVE_ON` está ligado).
///
/// Função PURA (recebe a seed, não lê nenhum atomic) para poder ser testada sem o
/// cuidado do gap 8/R8 do CLAUDE.md deste projeto: alternar um atomic global dentro de
/// um `#[test]` arrisca interferir com outro teste que rode em paralelo no mesmo
/// processo (mesmo motivo por trás de `spectral_rescale_factor` ter sido extraída como
/// função pura em vez de testada via `SPECTRAL_INIT_ON`).
pub fn random_sensors_within_ranges(seed: u64) -> Vec<f32> {
    let mut rng = crate::rng::Lcg::new(seed);
    SENSOR_RANGES.iter().map(|&(lo, hi)| lo + rng.uniform() * (hi - lo)).collect()
}

/// Muta o banco de sensores IN-PLACE — ruído gaussiano PROPORCIONAL À FAIXA de cada
/// gene (`SENSOR_MUTATE_FRAC × (hi - lo)`), nunca um σ absoluto fixo sobre todos.
///
/// Por quê: `fov_half_rad` está em radianos (faixa ~1,0 de largura) e os alcances em
/// metros (faixas de 65 a 170 de largura) — um σ fixo em metros atropelaria o FOV (a
/// faixa inteira do ângulo caberia dentro de UMA perturbação típica) ou, se calibrado
/// pro FOV, deixaria os alcances quase congelados (a mesma perturbação em radianos é
/// ruído nos metros). É exatamente o gap 11 do CLAUDE.md do projeto ("normalização
/// esquecida transformando experimento em artefato"), agora entre ângulo e distância em
/// vez de τ curto/longo — o mesmo problema, resolvido com a mesma ideia: escalar pela
/// grandeza própria de cada canal antes de comparar.
///
/// Clamp em `SENSOR_RANGES` depois de cada mutação — nunca produz sensor fisicamente
/// impossível (FOV negativo, alcance de zero ou menor).
///
/// Função PURA (recebe os geradores de fora, não lê `SENSORS_EVOLVE_ON`) pelo mesmo
/// motivo de `random_sensors_within_ranges` — ver o comentário lá.
pub fn mutate_sensors(sensors: &mut [f32], p_mutate: f32, rng: &mut impl GenoRng) {
    debug_assert_eq!(sensors.len(), N_SENSOR_GENES, "banco de sensores tem que ter N_SENSOR_GENES genes");
    for i in 0..N_SENSOR_GENES {
        if rng.chance() < p_mutate {
            let (lo, hi) = SENSOR_RANGES[i];
            let sigma = (hi - lo) * SENSOR_MUTATE_FRAC;
            sensors[i] = (sensors[i] + rng.gauss() * sigma).clamp(lo, hi);
        }
    }
}

/// Canais do CfC — o filtro cobre os primeiros 96 da onda sensorial.
pub const N_CFC_CHANNELS: usize = 96;

impl Genome {
    /// Zera a coluna de L1 dos canais em `perception::channels::DEAD_L1_INPUTS`
    /// (hoje só TireTemp, ch70 — estruturalmente constante, ver a doc completa em
    /// `Genome::random`). Extraída para método porque `Genome::random` zerava só
    /// no NASCIMENTO — `evolve()` muta L1 toda geração sem reaplicar a máscara, e
    /// medido no campeão real (`evolve_best_gen0059.bin`, coluna ch70) os 96
    /// pesos já estavam 100% não-zero (|w| médio 0,2085, igual à magnitude do
    /// canal vizinho vivo ch71) — a proteção erode geração a geração (achado
    /// P1.8 "DEAD_L1_INPUTS zera só no nascimento").
    ///
    /// Chamar depois de qualquer mutação de L1. NÃO chamar em `load_genome` —
    /// isso reescreveria em silêncio pesos reais que um campeão já treinou
    /// (mesmo princípio do teste `genoma_legado_carregado_do_disco_nao_e_reescalado`:
    /// carregar do disco não modifica peso).
    pub fn zero_dead_inputs(&mut self) {
        for &(ch, _motivo) in crate::perception::channels::DEAD_L1_INPUTS {
            for neuron in 0..L1_NEURONS {
                self.l1[neuron * SENSORY_WAVE_SIZE + ch] = 0.0;
            }
        }
    }

    /// Gera genoma com pesos aleatórios (N(0, σ_l1)).
    ///
    /// σ_l1 = 0.30 para L1/L2 — colocada na escala de pesos reais observados
    /// no CHAMPION.dna (|w| médio ≈ 0.15, max ≈ 0.7). Pesos zerados = rede nula.
    /// σ = 0.3 garante pelo menos ~30% dos pesos não-nulos com ação não-trivial.
    ///
    /// Depois de sortear L1, e se [`SPECTRAL_INIT_ON`] estiver ligado (padrão), L1 é
    /// re-escalado para nascer com ganho espectral ≈ `thresholds::SPECTRAL_INIT_TARGET`
    /// — ver o módulo "Inicialização espectral de L1 (P0.5)" no topo do arquivo.
    ///
    /// KAN knots = default (tanh ramp) — funciona como identidade suave e não
    /// precisa ser evoluído nos primeiros estágios.
    pub fn random(seed: u64) -> Self {
        // LCG de 64 bits (Knuth MMIX) — determinístico por seed.
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x6C62272E07BB0142);
        let mut rng_f32 = move || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f32) / (1u64 << 53) as f32;
            u * 2.0 - 1.0  // uniforme [-1, 1]
        };

        // Box-Muller para N(0, σ).
        let sigma_l1 = 0.30_f32;
        let gauss = |rng: &mut dyn FnMut() -> f32| -> f32 {
            let u1 = (rng() * 0.5 + 0.5).max(1e-7_f32);
            let u2 = rng() * 0.5 + 0.5;
            let g = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
            (g * sigma_l1).clamp(-3.0 * sigma_l1, 3.0 * sigma_l1)  // clamp em 3σ
        };

        let mut l1: Vec<f32> = (0..N_L1_WEIGHTS).map(|_| gauss(&mut rng_f32)).collect();
        let l2: Vec<f32> = (0..N_L2_WEIGHTS).map(|_| gauss(&mut rng_f32)).collect();

        // P0.5 — inicialização espectral: mede o ganho que o sorteio acima acabou de
        // produzir (mesma conta que `analysis::activation` usa pro alarme,
        // `SpectralRadiusHigh`) e re-escala L1 por um único fator escalar para que o
        // ganho medido bata em `thresholds::SPECTRAL_INIT_TARGET`. Um fator escalar
        // sobre a matriz inteira escala o maior valor singular pelo mesmo fator
        // (linear — ver teste em `math/linalg.rs`), então o alvo é atingido
        // exatamente, não por tentativa. Só toca L1: é a única matriz que o alarme
        // audita (L2 usa `w_rec: None` na chamada de `collect_activation_stats`).
        //
        // `SPECTRAL_INIT_ON` é o braço de controle do A/B (`--sem-init-espectral`,
        // padrão LIGADO). Genoma vindo do disco (`load_genome`) nunca passa por aqui.
        if SPECTRAL_INIT_ON.load(Ordering::Relaxed) {
            let medido = linalg::largest_singular_value(&l1, L1_NEURONS, SENSORY_WAVE_SIZE);
            let fator = spectral_rescale_factor(medido, thresholds::SPECTRAL_INIT_TARGET);
            for w in l1.iter_mut() { *w *= fator; }
        }

        // FASE CANAIS (2026-08-14) — zera a coluna de L1 dos canais em
        // `perception::channels::DEAD_L1_INPUTS` (hoje só TireTemp, ch70): canais
        // cujo kernel escreve mas o valor é estruturalmente constante (sem dado real
        // por trás — ver justificativa no registro). O cérebro não nasce gastando
        // peso de entrada com estática. Roda DEPOIS da re-escala espectral de
        // propósito: mede/normaliza o ganho da matriz i.i.d. inteira primeiro, e só
        // então zera as ~96 entradas (de 18.432) dos canais mortos — perturbação
        // pequena o bastante (<1% dos pesos) para não tirar o ganho medido de
        // [0.8, 1.2] (ver teste `genoma_novo_nasce_com_ganho_espectral_perto_de_1`).
        // NÃO é poda de neurônio — a ablação 96→64 (PLANO_EVOLUCAO.md) provou que
        // nenhum neurônio de L1 é descartável; isto só impede um peso de ENTRADA de
        // ler um canal que nunca varia.
        for &(ch, _motivo) in crate::perception::channels::DEAD_L1_INPUTS {
            for neuron in 0..L1_NEURONS {
                l1[neuron * SENSORY_WAVE_SIZE + ch] = 0.0;
            }
        }
        // (não chama `Genome::zero_dead_inputs` aqui porque `self` ainda não existe
        // — `l1` é uma Vec local que só vira campo do Genome no `Self { .. }` abaixo.
        // Mesma lógica, ver a doc do método para por que `evolve()` PRECISA
        // reaplicar isto e `load_genome` não pode.)

        let kan_knots = crate::evolution::dna_genesis::default_kan_knots();

        // Lento: parte da inicialização padrão e passa a divergir por mutação. Ele nasce
        // igual em todos os seres de propósito — a diferença tem que ser SELECIONADA,
        // não sorteada, senão a primeira geração já mistura dois efeitos.
        let slow = crate::engine::SlowBrain::new().all_weights();
        // τ uniforme = comportamento herdado. Heterogêneo só por `randomize_tau`, para que
        // o braço de controle do experimento seja o cérebro antigo, não um terceiro caso.
        let tau = vec![crate::cfc::DECAY_DEFAULT; N_CFC_CHANNELS];
        // FASE SENSORES (degrau 1): desligado (padrão), sensors = SENSOR_DEFAULTS sempre —
        // os hardcodes de hoje, idênticos para toda a população. Ligado
        // (`SENSORS_EVOLVE_ON`), sorteia dentro de `SENSOR_RANGES` — heterogêneo desde o
        // nascimento (diferente do τ: a própria tarefa pediu sorteio direto aqui, não um
        // `randomize_sensors` opt-in separado). Seed com offset próprio para não repetir a
        // sequência que já gerou l1/l2/tau para o mesmo `seed`.
        let sensors = if SENSORS_EVOLVE_ON.load(Ordering::Relaxed) {
            random_sensors_within_ranges(seed.wrapping_add(0x5E45_0000_5E45_0000))
        } else {
            SENSOR_DEFAULTS.to_vec()
        };
        // P1.7 — os três mecanismos SEMPRE nascem neutros, sem gate por atomic (ao
        // contrário de `sensors`/`tau`): não existe pedido de "sortear heterogêneo
        // desde o nascimento" para estes campos — só mutação, geração após geração,
        // divergindo do neutro devagar. Ver a doc dos campos em `Genome` para a
        // conta de por que cada default reproduz o cérebro de hoje exatamente.
        let neuron_alpha = vec![0.0; N_L1_NEURONS + N_L2_NEURONS];
        let neuron_period_l1 = vec![1.0; N_L1_NEURONS];
        let neuron_threshold_l1 = vec![0.0; N_L1_NEURONS];
        let neuron_activation = vec![0.0; N_L1_NEURONS + N_L2_NEURONS];
        Self {
            l1, l2, kan_knots, slow, tau, sensors,
            neuron_alpha, neuron_period_l1, neuron_threshold_l1, neuron_activation,
            gates_w: vec![0.0; 16],
        }
    }

    /// Sorteia o banco de τ log-uniforme em [25 ms, 4 s] — a condição heterogênea.
    pub fn randomize_tau(&mut self, seed: u64) {
        self.tau = crate::cfc::log_uniform_decays(seed, N_CFC_CHANNELS);
    }

    /// Achata o banco de τ de vários genomas no layout que `BrainGpu::upload_tau` espera.
    pub fn pack_taus(genomes: &[Genome]) -> Vec<f32> {
        let mut out = Vec::with_capacity(genomes.len() * N_CFC_CHANNELS);
        for g in genomes { out.extend_from_slice(&g.tau); }
        out
    }

    /// Achata o banco de sensores de vários genomas no layout que
    /// `PerceptionGpu::upload_sensors` espera — mesmo padrão de `pack_taus`.
    pub fn pack_sensors(genomes: &[Genome]) -> Vec<f32> {
        let mut out = Vec::with_capacity(genomes.len() * N_SENSOR_GENES);
        for g in genomes { out.extend_from_slice(&g.sensors); }
        out
    }

    /// P1.7 — achata os metadados por neurônio (mecanismos 1/2/3) de vários
    /// genomas no layout SoA que `BrainGpu::upload_neuron_meta` espera — mesmo
    /// padrão de `pack_taus`/`pack_sensors`. Por genoma, 480 floats na MESMA
    /// ordem de `NEURON_META_PER_SER` em `brain.wgsl`:
    ///   `l1_alpha(96) ++ l1_period(96) ++ l1_threshold(96) ++ l1_activation(96)
    ///    ++ l2_alpha(48) ++ l2_activation(48)`.
    /// `neuron_alpha`/`neuron_activation` guardam L1+L2 juntos (`[0..96)`=L1,
    /// `[96..144)`=L2, ver doc dos campos) — esta função é quem separa os dois na
    /// hora de montar o buffer, porque o shader quer os blocos NÃO intercalados
    /// (todo l1_* antes de todo l2_*, para os offsets `NEURON_META_L1_*_OFF`
    /// serem constantes fixas em vez de dependerem de `n_beings`).
    pub fn pack_neuron_meta(genomes: &[Genome]) -> Vec<f32> {
        let per_ser = N_L1_NEURONS * 4 + N_L2_NEURONS * 2; // 480, espelha NEURON_META_PER_SER
        let mut out = Vec::with_capacity(genomes.len() * per_ser);
        for g in genomes {
            debug_assert_eq!(g.neuron_alpha.len(), N_L1_NEURONS + N_L2_NEURONS);
            debug_assert_eq!(g.neuron_period_l1.len(), N_L1_NEURONS);
            debug_assert_eq!(g.neuron_threshold_l1.len(), N_L1_NEURONS);
            debug_assert_eq!(g.neuron_activation.len(), N_L1_NEURONS + N_L2_NEURONS);
            out.extend_from_slice(&g.neuron_alpha[..N_L1_NEURONS]);       // l1_alpha
            out.extend_from_slice(&g.neuron_period_l1);                   // l1_period
            out.extend_from_slice(&g.neuron_threshold_l1);                // l1_threshold
            out.extend_from_slice(&g.neuron_activation[..N_L1_NEURONS]);  // l1_activation
            out.extend_from_slice(&g.neuron_alpha[N_L1_NEURONS..]);       // l2_alpha
            out.extend_from_slice(&g.neuron_activation[N_L1_NEURONS..]);  // l2_activation
        }
        out
    }

    /// Carrega genoma a partir de um `SovereignEngine` existente.
    pub fn from_engine(eng: &SovereignEngine) -> Self {
        let l1 = eng.instinct_layer.weights.clone();
        let l2 = eng.cortex_l2.weights.clone();
        // KAN knots residem em engine.kan_filters (per-canal), mas os knots de
        // SAÍDA (throttle/steer/brake) não ficam no engine — usamos o default.
        let kan_knots = crate::evolution::dna_genesis::default_kan_knots();
        let slow = eng.slow_brain.all_weights();
        let tau = vec![crate::cfc::DECAY_DEFAULT; N_CFC_CHANNELS];
        // `SovereignEngine` (caminho CPU/Bevy) não tem conceito de sensores herdáveis —
        // ver "O que NÃO herda deste campo" na doc de `SENSORS_EVOLVE_ON`.
        let sensors = SENSOR_DEFAULTS.to_vec();
        // P1.7 — mesmo motivo de `sensors`: o caminho CPU/Bevy (`SovereignEngine`)
        // não tem conceito de cérebro assíncrono, só o binário de treino GPU lê
        // estes campos. Nascem neutros, igual a `Genome::random`.
        let neuron_alpha = vec![0.0; N_L1_NEURONS + N_L2_NEURONS];
        let neuron_period_l1 = vec![1.0; N_L1_NEURONS];
        let neuron_threshold_l1 = vec![0.0; N_L1_NEURONS];
        let neuron_activation = vec![0.0; N_L1_NEURONS + N_L2_NEURONS];
        Self {
            l1, l2, kan_knots, slow, tau, sensors,
            neuron_alpha, neuron_period_l1, neuron_threshold_l1, neuron_activation,
            gates_w: vec![0.0; 16],
        }
    }

    /// Escreve genoma em um `SovereignEngine` (substitui pesos).
    ///
    /// Falha explicitamente se L1/L2 divergirem — truncamento silencioso já causou
    /// pesos perdidos sem aviso, resultado: seres nasciam com genoma parcialmente zerado
    /// enquanto métricas de seleção pareciam normais.
    /// Quantos neurônios de L1 ficam VIVOS. `0` = todos (comportamento normal).
    ///
    /// Leonardo: *"podemos avaliar o quanto de espaço eles realmente precisam, e se o
    /// modelo ficar com pouco espaço a gente vai aumentando."*
    ///
    /// A medida de capacidade diz que L1 usa 6,7 das 96 dimensões (7%) — mas métrica não
    /// é prova. Zerar os pesos de saída dos neurônios acima do corte e medir os metros
    /// transforma a estimativa em experimento: se cortar 96→16 não custar desempenho, os
    /// 93% ociosos estão demonstrados, não inferidos.
    ///
    /// Zera-se a SAÍDA (a coluna de L2 que lê aquele neurônio) e não a entrada, porque é
    /// isso que remove a contribuição dele para o comportamento; zerar a entrada deixaria
    /// o neurônio emitindo o seu viés.
    pub fn ablar_l1_acima_de(&mut self, vivos: usize) {
        use crate::config::iara::brain::{L1_NEURONS, L2_NEURONS};
        if vivos == 0 || vivos >= L1_NEURONS { return; }
        for j in 0..L2_NEURONS {
            for n in vivos..L1_NEURONS {
                let idx = j * L1_NEURONS + n;
                if idx < self.l2.len() { self.l2[idx] = 0.0; }
            }
        }
    }

    pub fn apply_to_engine(&self, eng: &mut SovereignEngine) {
        assert_eq!(
            self.l1.len(), eng.instinct_layer.weights.len(),
            "L1 tem {} pesos, engine espera {} — divergência de SENSORY_WAVE_SIZE ou L1_NEURONS",
            self.l1.len(), eng.instinct_layer.weights.len()
        );
        eng.instinct_layer.weights.copy_from_slice(&self.l1);
        assert_eq!(
            self.l2.len(), eng.cortex_l2.weights.len(),
            "L2 tem {} pesos, engine espera {} — divergência de L1_NEURONS ou L2_NEURONS",
            self.l2.len(), eng.cortex_l2.weights.len()
        );
        eng.cortex_l2.weights.copy_from_slice(&self.l2);
        if !self.slow.is_empty() { eng.slow_brain.set_all_weights(&self.slow); }
        // KAN knots são usados pelo ciclo de evolução diretamente (não no engine).
    }
}

/// Parâmetros do algoritmo evolutivo.
pub struct EvoParams {
    /// Taxa de mutação por peso: probabilidade de cada peso ser perturbado.
    pub p_mutate: f32,
    /// Desvio-padrão do ruído gaussiano adicionado a cada peso mutado.
    pub sigma: f32,
    /// Número de genomas elite que passam sem mutação.
    pub n_elite: usize,
    /// Tamanho do torneio para seleção de pais.
    pub tournament_k: usize,
    /// Congela a mutação das linhas CEGAS de L2 (5..48). Investigação 2026-08-14:
    /// nenhum pipeline lê final_intent[5..48) — são 4.128 pesos de espaço de busca
    /// inerte, carregados por carona genética do elitismo (linhas cegas ficaram
    /// estatisticamente indistinguíveis das lidas em cosseno/curtose nas 2 sementes).
    /// Linhas 0..l2_linhas_lidas seguem mutáveis.
    pub congela_l2_cegas: bool,
    /// Quantas linhas de L2 têm consumidor real NO MODO ATUAL: 3 (throttle/steer/
    /// brake) com câmbio travado/auto, 5 com `--cambio neural` (intent[3]/[4]).
    /// Achado da bateria P1.9: congelar com o valor fixo 5 deixava as linhas 3-4
    /// mutáveis mesmo sem leitor no modo travado (padrão).
    pub l2_linhas_lidas: usize,
    /// M3: reimpõe a máscara de regiões após cada mutação (ver mascara_permite).
    pub mascara_regioes: bool,
}

/// Linhas de L2 com consumidor no pipeline Bevy (compat p/ testes e default).
pub const L2_LINHAS_LIDAS: usize = 5;

/// E-REG (M3, docs/REGIOES_CEREBRAIS.md): máscara DIRIGIDA de regiões — teto de
/// referência depois que M1 (custo em vida) falhou 2× (variância interindividual
/// do custo ≈ 0; seleção não enxerga fio). 4 blocos de 24 neurônios de L1:
///   bloco 0 (j 0-23):  VISÃO       — canais 0-15 (retina) + 128-191 (fóvea/perif/horizonte)
///   bloco 1 (j 24-47): PAREDE/ESTRADA — canais 16-63
///   bloco 2 (j 48-71): CORPO/PROX  — canais 64-95
///   bloco 3 (j 72-95): ASSOCIATIVO — todos os 192 (hub integrador)
/// Peso fora da máscara = 0, reimposto pós-mutação e ao carregar (mesma máquina
/// do antigo DEAD_L1_INPUTS, agora por região).
pub fn mascara_permite(j: usize, c: usize) -> bool {
    match j / 24 {
        0 => c < 16 || c >= 128,
        1 => (16..64).contains(&c),
        2 => (64..96).contains(&c),
        _ => true,
    }
}

/// Zera os pesos fora da máscara de regiões (chamar pós-mutação e pós-load).
pub fn aplicar_mascara_regioes(g: &mut Genome) {
    for j in 0..L1_NEURONS {
        for c in 0..SENSORY_WAVE_SIZE {
            if !mascara_permite(j, c) {
                g.l1[j * SENSORY_WAVE_SIZE + c] = 0.0;
            }
        }
    }
}

impl Default for EvoParams {
    fn default() -> Self {
        Self {
            p_mutate: 0.02,    // 2% dos pesos por geração
            sigma: 0.05,       // perturbação gaussiana σ=0.05
            n_elite: 3,        // top 3 passam sem mutação
            tournament_k: 3,   // torneio de 3
            congela_l2_cegas: false,
            l2_linhas_lidas: L2_LINHAS_LIDAS,
            mascara_regioes: false,
        }
    }
}

/// Resultado de uma geração.
pub struct GenResult {
    pub generation: usize,
    pub dist_max:  f64,  // metros, melhor indivíduo
    pub dist_med:  f64,  // metros, mediana
    pub n_ge1v:    usize, // seres com ≥1 volta completa
    pub n_ge2v:    usize, // seres com ≥2 voltas completas
    pub mortality: f32,  // fração de seres que não andaram (dist < 10m)
}

/// Executa um ciclo evolutivo: seleciona + muta.
///
/// `fitnesses[i]` = distância acumulada em metros do ser i na geração anterior.
/// Retorna nova geração de genomas na mesma ordem (0..pop_size).
///
/// `pop_seed` (o `--seed` da CLI) ENTRA na semente de mutação. Antes ele não
/// entrava: a semente era f(generation) puro, e sob `--retomar` (população
/// derivada do campeão, sem `Genome::random`) duas execuções com `--seed 77` e
/// `--seed 31` produziam genomas byte-idênticos em toda geração (medido por
/// md5sum em pos-auditoria-s{77,31}, gens 0/5/10/15/18/19). `pop_seed=0`
/// reproduz o stream antigo exatamente (o termo se anula) — baterias antigas
/// continuam reproduzíveis.
pub fn evolve(
    genomes:    &[Genome],
    fitnesses:  &[f64],
    params:     &EvoParams,
    generation: usize,   // índice da geração (para seed determinística)
    pop_seed:   u64,     // `--seed` da CLI; 0 = stream histórico
) -> Vec<Genome> {
    let n = genomes.len();
    assert_eq!(fitnesses.len(), n, "fitnesses e genomes devem ter mesmo tamanho");

    // ── Ordena por fitness decrescente ─────────────────────────────────────────
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut nova: Vec<Genome> = Vec::with_capacity(n);

    // ── Elitismo: os melhores passam inalterados ────────────────────────────────
    for &idx in order.iter().take(params.n_elite) {
        nova.push(genomes[idx].clone());
    }

    // ── Seleção por torneio + mutação ───────────────────────────────────────────
    // Struct local para encapsular o RNG sem aliasing de closures.
    struct Rng { state: u64, spare: Option<f32> }
    impl Rng {
        fn new(seed: u64) -> Self { Self { state: seed, spare: None } }
        fn u64(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }
        fn f32(&mut self) -> f32 { ((self.u64() >> 11) as f32) / (1u64 << 53) as f32 }
        // Box-Muller N(0,1).
        fn gauss(&mut self) -> f32 {
            if let Some(v) = self.spare.take() { return v; }
            // No rejection loop here: `u1` is clamped away from zero, so the first
            // draw is always valid. The `loop` this used to sit in never iterated
            // twice and clippy reported it as dead structure.
            let u1 = self.f32().max(1e-7_f32);
            let u2 = self.f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f32::consts::TAU * u2;
            self.spare = Some(r * theta.sin());
            r * theta.cos()
        }
    }
    // `Rng` implementa `GenoRng` para `mutate_sensors` poder receber UMA referência
    // (`&mut rng`) em vez de dois closures — ver a doc de `GenoRng` no topo do arquivo.
    impl GenoRng for Rng {
        fn chance(&mut self) -> f32 { self.f32() }
        fn gauss(&mut self) -> f32 { Rng::gauss(self) }
    }

    let seed = (generation as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0xCAFE_BABE_DEAD_BEEF)
        .wrapping_add(pop_seed.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    let mut rng = Rng::new(seed);

    while nova.len() < n {
        // Torneio: seleciona melhor entre k candidatos aleatórios.
        let parent_idx = {
            let mut best = rng.u64() as usize % n;
            for _ in 1..params.tournament_k {
                let cand = rng.u64() as usize % n;
                if fitnesses[cand] > fitnesses[best] { best = cand; }
            }
            best
        };

        let pai = &genomes[parent_idx];
        let mut filho = pai.clone();

        // Mutação gaussiana em L1 (2% dos pesos por geração, σ configurável).
        for w in filho.l1.iter_mut() {
            if rng.f32() < params.p_mutate {
                *w += rng.gauss() * params.sigma;
                *w = w.clamp(-2.0, 2.0);
            }
        }
        // P1.8, achado "DEAD_L1_INPUTS zera só no nascimento" — reaplica a máscara
        // dos canais mortos (ch70/TireTemp) depois da mutação de L1, pra ela não
        // erodir geração a geração (medida no campeão real: 96/96 pesos já
        // ressuscitados). Seguro chamar aqui: `evolve()` é quem PRODUZ genomas
        // novos a cada geração, então isto é parte do processo normal de gerar
        // filhos — nenhum teste assume que `evolve()` preserva o valor cru da
        // mutação nesses canais.
        filho.zero_dead_inputs();
        if params.mascara_regioes { aplicar_mascara_regioes(&mut filho); }
        // Mutação gaussiana em L2. Com `congela_l2_cegas`, só as linhas com
        // consumidor real (0..L2_LINHAS_LIDAS) mutam — as cegas ficam como estão
        // (nem random walk: o que o campeão tinha, fica congelado nele).
        let l2_limite = if params.congela_l2_cegas {
            params.l2_linhas_lidas.min(L2_NEURONS) * L1_NEURONS
        } else {
            filho.l2.len()
        };
        for w in filho.l2[..l2_limite].iter_mut() {
            if rng.f32() < params.p_mutate {
                *w += rng.gauss() * params.sigma;
                *w = w.clamp(-2.0, 2.0);
            }
        }
        // Mutação do lento. σ maior porque são mil pesos contra vinte e seis mil do
        // rápido — com a mesma taxa o lento receberia perturbação praticamente nula.
        // M4: gates do maestro mutam como o lento (poucos pesos → σ do lento).
        for w in filho.gates_w.iter_mut() {
            if rng.f32() < params.p_mutate * 4.0 {
                *w += rng.gauss() * params.sigma * 2.0;
                *w = w.clamp(-2.0, 2.0);
            }
        }
        for w in filho.slow.iter_mut() {
            if rng.f32() < params.p_mutate * 3.0 {
                *w += rng.gauss() * params.sigma * 1.5;
                *w = w.clamp(-2.0, 2.0);
            }
        }
        // Mutação do banco de τ: MULTIPLICATIVA, em log.
        //
        // Aditiva seria errada aqui. A faixa vai de 0,25 a 40 s⁻¹ — um σ que mexe de
        // verdade num canal de 0,25 atropela um de 40, e vice-versa. O que tem escala
        // constante é a razão: ×1,15 significa a mesma coisa em qualquer ponto da faixa.
        //
        // Taxa 3× a dos pesos pelo mesmo motivo do lento: são 96 valores contra 26 mil,
        // e com a taxa base o banco de τ mal seria tocado numa vida de 30 gerações.
        let d_lo = 1.0 / crate::cfc::TAU_MAX_S;
        let d_hi = 1.0 / crate::cfc::TAU_MIN_S;
        for d in filho.tau.iter_mut() {
            if rng.f32() < params.p_mutate * 3.0 {
                *d = (*d * (rng.gauss() * 0.20).exp()).clamp(d_lo, d_hi);
            }
        }
        // Mutação do banco de sensores (FASE SENSORES) — só quando a evolução dos
        // sensores está ligada (`SENSORS_EVOLVE_ON`, padrão DESLIGADA). Desligado, o
        // campo fica bit-idêntico a `SENSOR_DEFAULTS` a vida inteira da bateria — ver a
        // doc de `SENSORS_EVOLVE_ON` para por que este gate é mais forte que o do τ
        // (τ muta incondicionalmente; sensores só mutam com confirmação explícita).
        // Mesma taxa 3× do lento/τ pelo mesmo motivo: 6 genes contra dezenas de milhares
        // de pesos — a taxa base mal tocaria o banco numa vida inteira de gerações.
        if SENSORS_EVOLVE_ON.load(Ordering::Relaxed) {
            mutate_sensors(&mut filho.sensors, params.p_mutate * 3.0, &mut rng);
        }
        // kan_knots (SAÍDA, 3×12): NÃO mutado — achado P1.8 "kan_knots é mutado em
        // toda geração mas nunca é lido de volta". `main.rs` passa `&kan_knots_global`
        // (`default_kan_knots()`, fixo) para `ctx.tick()` em TODOS os call sites reais
        // (laço principal e held-out) — `genomes[i].kan_knots` nunca é lido fora de
        // save/load. Mutar aqui era RNG/CPU gasto à toa por indivíduo/geração, E
        // sujava o `.dna` com "evolução" que não existe (o campo muda no disco, o
        // comportamento não). Se/quando alguém religar os knots por-ser em
        // `ctx.tick()` (ver doc do campo, achado completo em docs/PLANO_EVOLUCAO.md
        // P1.8), a mutação volta aqui.

        // ── P1.7 — cérebro assíncrono (docs/PLANO_EVOLUCAO.md §P1.7) ────────────
        // Os quatro blocos abaixo são SEMPRE os ÚLTIMOS a consumir `rng` neste
        // laço — l1/l2/slow/tau/sensores (acima) mutam exatamente como mutavam
        // antes de P1.7 existir, byte a byte, porque nenhum deles teve sequer uma
        // chamada de `rng` inserida ANTES de si. Um genoma que nasce neutro (todo
        // `Genome::random`) e nunca muta estes campos (os quatro atomics
        // desligados via `--sem-*`) reproduz a saída de hoje geração após geração
        // — é o braço de controle mais forte que dá pra escrever. (`kan_knots`
        // deixou de mutar em P1.8 — a garantia acima é sobre l1/l2/slow/tau/sensores,
        // não muda com essa remoção.)

        // Mecanismo 1 (memória): alfa muta ADITIVO-clampado, não multiplicativo
        // como τ — τ é uma taxa de decaimento em s⁻¹ (grandeza com escala própria,
        // por isso log-uniforme); alfa é um peso de mistura já normalizado em
        // [0,1), então um passo gaussiano pequeno é a escolha direta, mesmo
        // espírito do resto dos campos "poucos genes contra dezenas de milhares de
        // pesos" (taxa 3×, σ maior que a dos pesos).
        if NEURON_MEM_MUTATE_ON.load(Ordering::Relaxed) {
            for a in filho.neuron_alpha.iter_mut() {
                if rng.f32() < params.p_mutate * 3.0 {
                    *a = (*a + rng.gauss() * 0.10).clamp(0.0, 0.95);
                }
            }
        }
        // Mecanismo 2, período: INTEIRO — passo discreto ±1, não gaussiano
        // contínuo (1.3→1.31 não muda o comportamento do neurônio nenhum tick;
        // 1→2 muda). Teto 8: acima disso o neurônio já fica quase sempre "mudo"
        // (mantendo o valor antigo 7 de cada 8 ticks) sem ganho adicional de
        // representação — mesma lógica de faixa física do resto do arquivo, só
        // que aqui o "físico" é "ainda reage rápido o bastante pra pista".
        if NEURON_PERIOD_MUTATE_ON.load(Ordering::Relaxed) {
            for p in filho.neuron_period_l1.iter_mut() {
                if rng.f32() < params.p_mutate * 3.0 {
                    let passo = if rng.f32() < 0.5 { -1.0 } else { 1.0 };
                    *p = (*p + passo).round().clamp(1.0, 8.0);
                }
            }
        }
        // Mecanismo 2, limiar: gaussiano-clampado, mesma receita do alfa. Teto
        // 0.9: acima disso o neurônio raramente ultrapassaria o limiar (a faixa
        // pós-ativação é [-1,1] ou [0,∞) pro ReLU/linear, mas o clamp de
        // quantização em brain.wgsl já satura em [-1,1] antes do ×127) e ficaria
        // permanentemente mudo — não é "assincronia", é desligar o neurônio de
        // vez, que já existe como caminho separado (ablação de L1).
        if NEURON_THRESHOLD_MUTATE_ON.load(Ordering::Relaxed) {
            for t in filho.neuron_threshold_l1.iter_mut() {
                if rng.f32() < params.p_mutate * 3.0 {
                    *t = (*t + rng.gauss() * 0.05).clamp(0.0, 0.9);
                }
            }
        }
        // Mecanismo 3 (ativação heterogênea): mutação DISCRETA — troca de
        // categoria, não perturbação. Taxa MENOR que a gaussiana dos pesos (e
        // menor que a taxa 3× usada acima): trocar tanh→ReLU muda o REGIME
        // inteiro do neurônio (de saturação suave, sempre ativo, para
        // esparsidade dura, metade em zero exato) — não é um ajuste fino como
        // mover um peso em σ=0.05, é uma decisão discreta cujo efeito só a
        // seleção mede depois de várias gerações. Uma taxa alta faria o
        // repertório de ativações oscilar geração a geração sem tempo de a
        // seleção provar se a troca ajudou — mesma família de raciocínio do KAN
        // knots (0.5%) logo acima, mas ainda mais conservador porque aqui o
        // "custo" de errar é maior (muda o sinal que chega inteiro em L2).
        if NEURON_ACTIVATION_MUTATE_ON.load(Ordering::Relaxed) {
            const ACT_MUTATE_RATE: f32 = 0.002; // 0.2% por neurônio por geração
            for act in filho.neuron_activation.iter_mut() {
                if rng.f32() < ACT_MUTATE_RATE {
                    *act = (rng.f32() * 4.0).floor().min(3.0);
                }
            }
        }

        nova.push(filho);
    }

    nova
}

/// P1.7, mecanismo 2 — quantos neurônios de L1 (população inteira) EFETIVAMENTE
/// recalculam a memória em cada tick, dado o banco de períodos já upado na GPU.
///
/// Espelha EXATAMENTE a condição `fires` de `brain.wgsl`
/// (`period <= 1 || tick_counter % period == 0`, contador começando em 1 — o
/// shader incrementa o counter ANTES de qualquer leitura, `counter_old=0` no
/// primeiro tick vira `counter_new=1`) — simulado aqui em Rust puro, SEM GPU,
/// porque o período de cada neurônio é conhecido de antemão (não depende do
/// dado sensorial daquele tick, só do índice do tick). HONESTIDADE (CLAUDE.md/
/// tarefa): isto mede REPRESENTAÇÃO (quantos neurônios estão "acordados"
/// naquele instante), não tempo de GPU — todas as threads rodam de qualquer
/// jeito a cada dispatch, período>1 não poupa um único ciclo de clock.
///
/// `n_ticks` idealmente cobre pelo menos um ciclo completo de MMC dos períodos
/// em uso (MMC(1..8)=840) para a média/desvio não dependerem de ONDE a janela
/// começou a amostrar — o chamador passa `n_ticks.min(840)` ou mais.
///
/// Retorna `(media, desvio_padrao)` da contagem de neurônios disparando por
/// tick, somada sobre TODOS os seres (não por ser).
pub fn simulate_l1_fire_counts(periods_all: &[f32], n_beings: usize, n_ticks: usize) -> (f64, f64) {
    if n_beings == 0 || n_ticks == 0 { return (0.0, 0.0); }
    debug_assert_eq!(periods_all.len(), n_beings * N_L1_NEURONS,
        "periods_all deveria ter n_beings*N_L1_NEURONS valores");
    let mut contagens: Vec<f64> = Vec::with_capacity(n_ticks);
    for tick in 1..=n_ticks {
        let mut count: u64 = 0;
        for &p_raw in periods_all.iter() {
            let period = (p_raw.round() as i64).max(1) as usize;
            if period <= 1 || tick % period == 0 { count += 1; }
        }
        contagens.push(count as f64);
    }
    let media = contagens.iter().sum::<f64>() / n_ticks as f64;
    let var = contagens.iter().map(|&c| (c - media).powi(2)).sum::<f64>() / n_ticks as f64;
    (media, var.sqrt())
}

/// Calcula estatísticas de uma geração a partir das distâncias acumuladas.
pub fn gen_stats(generation: usize, dist_acum: &[f64], lap_dist: f64) -> GenResult {
    let n = dist_acum.len();
    let dist_max = dist_acum.iter().cloned().fold(0.0_f64, f64::max);
    let dist_med = {
        let mut v = dist_acum.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let n_ge1v  = dist_acum.iter().filter(|&&d| d >= lap_dist).count();
    let n_ge2v  = dist_acum.iter().filter(|&&d| d >= lap_dist * 2.0).count();
    let n_morto = dist_acum.iter().filter(|&&d| d < 10.0).count();
    let mortality = n_morto as f32 / n as f32;
    GenResult { generation, dist_max, dist_med, n_ge1v, n_ge2v, mortality }
}

/// Calcula inclinação linear (OLS) e seu desvio-padrão (erro do estimador).
///
/// `xs[i]` = geração i (float), `ys[i]` = mediana da geração i.
/// Retorna `(slope, slope_stderr)`. Se n < 2 retorna (0.0, 0.0).
pub fn linear_slope(ys: &[f64]) -> (f64, f64) {
    let n = ys.len();
    if n < 2 { return (0.0, 0.0); }
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mean_x = xs.iter().sum::<f64>() / n as f64;
    let mean_y = ys.iter().sum::<f64>() / n as f64;
    let sxx: f64 = xs.iter().map(|&x| (x - mean_x).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| (x - mean_x) * (y - mean_y)).sum();
    if sxx < 1e-10 { return (0.0, 0.0); }
    let slope = sxy / sxx;
    // SSR = variância residual
    let ssr: f64 = xs.iter().zip(ys.iter())
        .map(|(&x, &y)| (y - (mean_y + slope * (x - mean_x))).powi(2))
        .sum();
    let var_resid = ssr / (n as f64 - 2.0).max(1.0);
    let slope_stderr = (var_resid / sxx).sqrt();
    (slope, slope_stderr)
}

// ---------------------------------------------------------------------------
// Testes unitários — P0.5 (inicialização espectral de L1).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `spectral_rescale_factor` é pura function matemática — o teste central da
    /// inicialização espectral que não precisa tocar `SPECTRAL_INIT_ON` nem sortear
    /// pesos: dado o ganho ~7.0 medido empiricamente (CLAUDE.md do projeto, 4 corridas
    /// independentes), o fator devolvido tem que levar exatamente a 1.0.
    #[test]
    fn spectral_rescale_factor_atinge_alvo_exatamente() {
        let fator = spectral_rescale_factor(7.0, 1.0);
        assert!((fator - 1.0 / 7.0).abs() < 1e-6, "fator = {fator}, esperado 1/7");
        assert!((7.0 * fator - 1.0).abs() < 1e-6, "aplicar o fator deveria levar a 1.0");
    }

    /// Ganho ~0 (L1 nulo, nunca observado na prática) não deve gerar divisão explosiva
    /// — o fator cai para 1.0 (identidade: nada para escalar).
    #[test]
    fn spectral_rescale_factor_ganho_quase_zero_nao_explode() {
        assert_eq!(spectral_rescale_factor(0.0, 1.0), 1.0);
        assert_eq!(spectral_rescale_factor(1e-9, 1.0), 1.0);
    }

    /// P0.5, requisito 4: genoma novo (flag padrão LIGADA) nasce com ganho espectral
    /// de L1 medido em [0.8, 1.2] — não exatamente 1.0 porque a normalização mede o
    /// GANHO DA MATRIZ INTEIRA sorteada (deve bater ~1.0 dentro do erro numérico do
    /// power-iteration, ~100 iterações, mais o efeito do clamp em 3σ na cauda da
    /// gaussiana), mas longe o bastante do ~7.0 sem correção para provar que a
    /// re-escala está de fato acontecendo.
    ///
    /// Não mexe em `SPECTRAL_INIT_ON` (fica no padrão `true` do processo de teste) —
    /// ver doc de `spectral_rescale_factor` sobre por que evitar alternar o atomic
    /// global dentro de testes.
    #[test]
    fn genoma_novo_nasce_com_ganho_espectral_perto_de_1() {
        assert!(
            SPECTRAL_INIT_ON.load(Ordering::Relaxed),
            "pré-condição do teste: a flag deveria estar no padrão (ligada) neste processo"
        );
        for seed in [1u64, 2, 3, 42, 999_999] {
            let g = Genome::random(seed);
            let ganho = linalg::largest_singular_value(&g.l1, L1_NEURONS, SENSORY_WAVE_SIZE);
            assert!(
                (0.8..=1.2).contains(&ganho),
                "seed {seed}: ganho espectral de L1 = {ganho}, esperado em [0.8, 1.2]"
            );
        }
    }

    /// FASE CANAIS (2026-08-14): genoma novo nasce com a coluna de L1 dos canais
    /// `DEAD_L1_INPUTS` (hoje só TireTemp, ch70) inteiramente zerada — nenhum dos
    /// `L1_NEURONS` pesos que leem esse canal escapa do zeramento, em nenhuma seed.
    #[test]
    fn genoma_novo_zera_l1_dos_canais_mortos() {
        use crate::perception::channels::DEAD_L1_INPUTS;
        // Lista vazia desde 2026-08-15 (ch70 refutado por medição — canal estava
        // VIVO; térmica implementada em physics_power_c.wgsl). Teste vira no-op
        // até existir canal genuinamente morto; a máquina segue no forward.
        if DEAD_L1_INPUTS.is_empty() { return; }
        for seed in [1u64, 7, 999] {
            let g = Genome::random(seed);
            for &(ch, _motivo) in DEAD_L1_INPUTS {
                for neuron in 0..L1_NEURONS {
                    let w = g.l1[neuron * SENSORY_WAVE_SIZE + ch];
                    assert_eq!(w, 0.0, "seed {seed}: l1[neuron={neuron}, ch={ch}] = {w}, esperado 0.0");
                }
            }
        }
    }

    /// FASE CANAIS: zerar os canais mortos não deve mudar canais VIZINHOS — sanidade
    /// contra um erro de índice (off-by-one na coluna) que zerasse o canal errado.
    /// Não mexe em `SPECTRAL_INIT_ON` (mesmo cuidado de
    /// `genoma_novo_nasce_com_ganho_espectral_perto_de_1` — ver doc de
    /// `spectral_rescale_factor` sobre por que evitar alternar o atomic global
    /// dentro de testes, R8 do CLAUDE.md do projeto). Um vizinho ficar em 0.0
    /// exatamente após reescala espectral tem probabilidade praticamente nula
    /// (gaussiana contínua × fator escalar não-nulo); múltiplas seeds reduzem
    /// ainda mais o risco de falso-negativo.
    #[test]
    fn zerar_canal_morto_nao_afeta_canal_vizinho_vivo() {
        use crate::perception::channels::DEAD_L1_INPUTS;
        for seed in [1u64, 7, 999] {
            let g = Genome::random(seed);
            for &(ch, _) in DEAD_L1_INPUTS {
                if ch + 1 < SENSORY_WAVE_SIZE {
                    // `0 * SENSORY_WAVE_SIZE` is written out on purpose: it names the
                    // L1 neuron being indexed, and reads in parallel with the other
                    // rows. Clippy calls it `erasing_op`; collapsing it would hide
                    // which neuron this is.
                    #[allow(clippy::erasing_op)]
                    let vizinho = g.l1[0 * SENSORY_WAVE_SIZE + (ch + 1)];
                    assert_ne!(vizinho, 0.0, "seed {seed}: canal vizinho ch{} não deveria ter sido zerado", ch + 1);
                }
            }
        }
    }

    /// P1.8, achado "DEAD_L1_INPUTS zera só no nascimento, mutação de evolve() não
    /// reaplica a máscara": com `p_mutate=1.0` (todo peso muta toda geração), a
    /// coluna do canal morto tinha 100% de chance de sair de 0.0 já na geração 1
    /// SEM o fix — este teste falha sem `filho.zero_dead_inputs()` depois da
    /// mutação de L1 (`evolve()`, gpu/evolution.rs) e passa com ele.
    /// Bug pos-auditoria-s{77,31} (2026-08-14): a semente de mutação era
    /// f(generation) puro — dois processos com `--seed` diferentes, retomando o
    /// MESMO campeão, produziam genomas byte-idênticos em toda geração (md5sum
    /// igual em gens 0/5/10/15/18). Este teste falha sem o termo `pop_seed` na
    /// semente e passa com ele. Também prova a compat: pop_seed=0 é determinístico.
    #[test]
    fn evolve_pop_seeds_diferentes_divergem_e_zero_reproduz() {
        let n = 10;
        let genomes: Vec<Genome> = (0..n).map(|i| Genome::random(i as u64)).collect();
        let params = EvoParams { p_mutate: 1.0, sigma: 0.05, n_elite: 1, tournament_k: 3, congela_l2_cegas: false, l2_linhas_lidas: L2_LINHAS_LIDAS, mascara_regioes: false };
        let fitnesses: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let a  = evolve(&genomes, &fitnesses, &params, 1, 77);
        let b  = evolve(&genomes, &fitnesses, &params, 1, 31);
        let a2 = evolve(&genomes, &fitnesses, &params, 1, 77);

        // Elite (índice 0) passa clonada nos dois — compara um filho MUTADO.
        assert_ne!(a[1].l1, b[1].l1, "seeds 77 e 31 produziram o mesmo filho: pop_seed não entrou na mutação");
        assert_eq!(a[1].l1, a2[1].l1, "mesma seed deixou de ser determinística");
        assert_eq!(a[0].l1, b[0].l1, "elite não pode depender de seed (passa sem mutação)");
    }

    /// `--l2-congelar-cegas`: linhas 5..48 de L2 não mutam; 0..5 mutam normal.
    #[test]
    fn evolve_congela_linhas_cegas_de_l2() {
        let n = 10;
        let genomes: Vec<Genome> = (0..n).map(|i| Genome::random(i as u64)).collect();
        let params = EvoParams { p_mutate: 1.0, sigma: 0.05, n_elite: 1, tournament_k: 3, congela_l2_cegas: true, l2_linhas_lidas: L2_LINHAS_LIDAS, mascara_regioes: false };
        let fitnesses: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let nova = evolve(&genomes, &fitnesses, &params, 1, 0);
        // Filho mutado (índice 1): acha o pai comparando as linhas cegas — elas têm
        // de ser idênticas às de ALGUM genoma original (nunca mutaram).
        let corte = L2_LINHAS_LIDAS * L1_NEURONS;
        let filho = &nova[1];
        let pai = genomes.iter().find(|g| g.l2[corte..] == filho.l2[corte..])
            .expect("linhas cegas do filho não batem com nenhum pai: mutaram apesar do congelamento");
        // E as linhas lidas TÊM de ter mutado (p_mutate=1.0 → 480 pesos, todos perturbados).
        assert_ne!(pai.l2[..corte], filho.l2[..corte],
            "linhas lidas não mutaram — congelamento vazou pro range errado");
    }

    #[test]
    fn evolve_preserva_zero_dos_canais_mortos() {
        use crate::perception::channels::DEAD_L1_INPUTS;
        // Lista vazia desde 2026-08-15 (ch70 refutado por medição — canal estava
        // VIVO; térmica implementada em physics_power_c.wgsl). Teste vira no-op
        // até existir canal genuinamente morto; a máquina segue no forward.
        if DEAD_L1_INPUTS.is_empty() { return; }

        let n = 10;
        let mut genomes: Vec<Genome> = (0..n).map(|i| Genome::random(i as u64)).collect();
        // p_mutate=1.0: garante que TODO peso de L1 (inclusive os dos canais mortos,
        // se a máscara não fosse reaplicada) sofre tentativa de mutação toda geração.
        let params = EvoParams { p_mutate: 1.0, sigma: 0.05, n_elite: 1, tournament_k: 3, congela_l2_cegas: false, l2_linhas_lidas: L2_LINHAS_LIDAS, mascara_regioes: false };
        let fitnesses: Vec<f64> = (0..n).map(|i| i as f64).collect();

        for geracao in 0..10 {
            genomes = evolve(&genomes, &fitnesses, &params, geracao, 0);
            for (gi, g) in genomes.iter().enumerate() {
                for &(ch, _motivo) in DEAD_L1_INPUTS {
                    for neuron in 0..L1_NEURONS {
                        let w = g.l1[neuron * SENSORY_WAVE_SIZE + ch];
                        assert_eq!(
                            w, 0.0,
                            "geração {geracao}, genoma {gi}: l1[neuron={neuron}, ch={ch}] = {w}, \
                             esperado 0.0 — canal morto ressuscitou pela mutação"
                        );
                    }
                }
            }
        }
    }

    // P0.5, requisito 4 (parte 2, "herança intocada"): o teste que carrega um genoma
    // legado de verdade do disco e confirma que `load_genome` não re-escala mora em
    // `src/bin/gpu_evolve/tests.rs::genoma_legado_carregado_do_disco_nao_e_reescalado`
    // — precisa de `load_genome`/`save_genome`, que vivem no binário `gpu_evolve`, não
    // nesta crate. A garantia estrutural é simples de conferir por leitura: `load_genome`
    // monta `Genome { .. }` por struct literal direto (genome_io.rs, campos listados
    // na doc do próprio struct), nunca chamando `Genome::random` nem `spectral_rescale_factor`.

    // ─────────────────────────────────────────────────────────────────────────
    // Testes — FASE SENSORES (degrau 1).
    // ─────────────────────────────────────────────────────────────────────────

    /// `SENSOR_DEFAULTS` tem que caber dentro da própria `SENSOR_RANGES` — senão o
    /// valor de nascença (usado com a evolução desligada, e como fallback de genoma
    /// legado) já nasceria fora da faixa que a mutação/clamp respeitam quando ligada.
    /// Checagem de consistência entre as duas tabelas, não comportamento de runtime.
    #[test]
    fn sensor_defaults_estao_dentro_das_proprias_faixas() {
        for i in 0..N_SENSOR_GENES {
            let (lo, hi) = SENSOR_RANGES[i];
            let v = SENSOR_DEFAULTS[i];
            assert!(v >= lo && v <= hi, "gene {i}: default {v} fora da faixa [{lo}, {hi}]");
        }
    }

    /// `SENSORS_EVOLVE_ON` desligado (padrão do processo de teste — não tocamos o
    /// atomic aqui, mesmo cuidado de `genoma_novo_nasce_com_ganho_espectral_perto_de_1`
    /// sobre não alternar atomics globais dentro de `#[test]`): `Genome::random` produz
    /// `sensors` byte-idêntico a `SENSOR_DEFAULTS`, em qualquer seed — é o comportamento
    /// que protege a bateria em curso (regra do CLAUDE.md do projeto).
    #[test]
    fn genoma_novo_nasce_com_sensores_no_padrao_quando_evolucao_desligada() {
        assert!(
            !SENSORS_EVOLVE_ON.load(Ordering::Relaxed),
            "pré-condição do teste: SENSORS_EVOLVE_ON deveria estar no padrão (desligado) neste processo"
        );
        for seed in [1u64, 2, 3, 42, 999_999] {
            let g = Genome::random(seed);
            assert_eq!(g.sensors, SENSOR_DEFAULTS.to_vec(), "seed {seed}: sensores deveriam ser SENSOR_DEFAULTS");
        }
    }

    /// `random_sensors_within_ranges` é função pura — testável sem tocar o atomic.
    /// Todo gene sorteado, em toda seed, cai dentro da sua própria faixa física.
    #[test]
    fn sensores_sorteados_ficam_dentro_das_faixas_fisicas() {
        for seed in [1u64, 2, 3, 42, 999_999, 0xDEAD_BEEF] {
            let sensores = random_sensors_within_ranges(seed);
            assert_eq!(sensores.len(), N_SENSOR_GENES);
            for i in 0..N_SENSOR_GENES {
                let (lo, hi) = SENSOR_RANGES[i];
                let v = sensores[i];
                assert!(v >= lo && v <= hi, "seed {seed}, gene {i}: sorteado {v} fora da faixa [{lo}, {hi}]");
            }
        }
    }

    /// `mutate_sensors` é função pura — testável sem tocar `SENSORS_EVOLVE_ON`.
    /// Estressa com `p_mutate = 1.0` (muta todo gene toda chamada) por centenas de
    /// "gerações" simuladas: mesmo acumulando ruído gaussiano sem limite, o clamp em
    /// `SENSOR_RANGES` garante que nenhum gene escapa — é o requisito 4 da tarefa
    /// ("mutação não sai das faixas").
    #[test]
    fn mutacao_de_sensores_nunca_sai_das_faixas() {
        for seed in [1u64, 2, 3, 42, 999_999] {
            let mut sensores = SENSOR_DEFAULTS.to_vec();
            let mut rng = crate::rng::Lcg::new(seed);
            for _geracao in 0..500 {
                mutate_sensors(&mut sensores, 1.0, &mut rng);
            }
            for i in 0..N_SENSOR_GENES {
                let (lo, hi) = SENSOR_RANGES[i];
                let v = sensores[i];
                assert!(v >= lo && v <= hi, "seed {seed}, gene {i}: {v} fora da faixa [{lo}, {hi}] após 500 mutações");
            }
        }
    }

    /// Com `p_mutate = 0.0`, `mutate_sensors` não toca em NADA — sanity check contra um
    /// bug de sinal na comparação `rng.chance() < p_mutate` (o mesmo tipo de bug de
    /// sinal já visto neste projeto no LCG do `slow_probe`, ver `src/rng.rs`).
    #[test]
    fn mutacao_com_p_mutate_zero_nao_muda_sensores() {
        let original = SENSOR_DEFAULTS.to_vec();
        let mut sensores = original.clone();
        let mut rng = crate::rng::Lcg::new(123);
        for _ in 0..50 {
            mutate_sensors(&mut sensores, 0.0, &mut rng);
        }
        assert_eq!(sensores, original);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Testes — P1.7 (cérebro assíncrono: memória, período/limiar, ativação).
    // ─────────────────────────────────────────────────────────────────────────

    /// Requisito central da tarefa ("cada flag no padrão reproduz a saída atual
    /// EXATAMENTE"): um genoma recém-nascido tem os seis campos de P1.7 em valor
    /// NEUTRO — o valor que faz `brain.wgsl` reproduzir a fórmula de antes desta
    /// mudança bit a bit (alfa=0 ⇒ h[n]=x[n] com ganho 1; período=1 ⇒ recalcula
    /// todo tick; limiar=0 ⇒ nunca zera, pois `abs(v) < 0.0` nunca é verdade;
    /// ativação=0 ⇒ tanh). Referenciado no comentário de
    /// `BrainParams::mech_off_mask` em `brain.wgsl`.
    #[test]
    fn genoma_novo_tem_mecanismos_p17_neutros() {
        for seed in [1u64, 2, 3, 42, 999_999] {
            let g = Genome::random(seed);
            assert_eq!(g.neuron_alpha.len(), N_L1_NEURONS + N_L2_NEURONS, "seed {seed}");
            assert!(g.neuron_alpha.iter().all(|&a| a == 0.0), "seed {seed}: alfa deveria nascer 0.0");
            assert_eq!(g.neuron_period_l1.len(), N_L1_NEURONS, "seed {seed}");
            assert!(g.neuron_period_l1.iter().all(|&p| p == 1.0), "seed {seed}: período deveria nascer 1.0");
            assert_eq!(g.neuron_threshold_l1.len(), N_L1_NEURONS, "seed {seed}");
            assert!(g.neuron_threshold_l1.iter().all(|&t| t == 0.0), "seed {seed}: limiar deveria nascer 0.0");
            assert_eq!(g.neuron_activation.len(), N_L1_NEURONS + N_L2_NEURONS, "seed {seed}");
            assert!(g.neuron_activation.iter().all(|&a| a == 0.0), "seed {seed}: ativação deveria nascer 0.0 (tanh)");
        }
    }

    /// A matemática do ganho de amplitude do mecanismo 1 (`neuron_mem_gain` em
    /// `brain.wgsl`), espelhada aqui em Rust puro para poder testar sem GPU — o
    /// mesmo padrão de `out_stage`/`population_std_dev` em `math/activations.rs`.
    /// `gain(0) == 1` é o que garante que alfa=0 não muda NADA da amplitude.
    fn neuron_mem_gain(alfa: f32) -> f32 {
        ((1.0 + alfa) / (1.0 - alfa).max(1e-3)).sqrt()
    }

    #[test]
    fn neuron_mem_gain_em_alfa_zero_e_identidade() {
        assert_eq!(neuron_mem_gain(0.0), 1.0);
    }

    /// O ganho cresce com alfa (compensa a atenuação de amplitude — GAP 11 do
    /// CLAUDE.md do projeto) e fica FINITO no teto de clamp de `evolve()` (0.95).
    #[test]
    fn neuron_mem_gain_cresce_e_fica_finito_no_teto_do_clamp() {
        let g_baixo = neuron_mem_gain(0.2);
        let g_alto  = neuron_mem_gain(0.95);
        assert!(g_alto > g_baixo, "ganho deveria crescer com alfa: {g_baixo} vs {g_alto}");
        assert!(g_alto.is_finite(), "ganho no teto do clamp (0.95) deveria ser finito, obteve {g_alto}");
        // sqrt((1+0.95)/(1-0.95)) = sqrt(1.95/0.05) = sqrt(39) ≈ 6.245
        assert!((g_alto - 39.0f32.sqrt()).abs() < 1e-4, "ganho em alfa=0.95: {g_alto}, esperado ≈6.245");
    }

    /// `evolve()`, com os quatro atomics de P1.7 no padrão (`true` — mutação
    /// LIGADA), muta os seis campos ao longo de várias gerações mas NUNCA sai das
    /// faixas fisicamente válidas: alfa∈[0,0.95], período∈[1,8] (inteiro),
    /// limiar∈[0,0.9], ativação∈{0,1,2,3} — E, no MESMO teste (não em dois
    /// separados), o braço de controle: com os quatro atomics DESLIGADOS,
    /// `evolve()` NUNCA toca os seis campos de P1.7, ficam bit-idênticos ao
    /// neutro de nascença por quantas gerações rodar.
    ///
    /// FUNDIDO num único `#[test]` de propósito: os quatro atomics
    /// (`NEURON_*_MUTATE_ON`) são globais do processo, e `cargo test` roda
    /// testes de um mesmo binário em THREADS do mesmo processo — dois `#[test]`
    /// separados, um lendo "deveria estar true" e outro alternando para `false`
    /// e de volta, correm risco real de se cruzar (medido: a primeira versão
    /// deste teste, separada, falhava intermitentemente por essa razão — R8 do
    /// CLAUDE.md do projeto). Um teste só, numa thread só, elimina a corrida:
    /// primeiro mede o padrão (mutando), depois liga o braço de controle
    /// (congela), sem nenhuma outra thread podendo observar um estado
    /// intermediário destes atomics.
    #[test]
    fn evolve_muta_mecanismos_p17_e_flags_sem_x_congelam_no_neutro() {
        assert!(NEURON_MEM_MUTATE_ON.load(Ordering::Relaxed), "pré-condição: padrão deveria ser true");
        assert!(NEURON_PERIOD_MUTATE_ON.load(Ordering::Relaxed), "pré-condição: padrão deveria ser true");
        assert!(NEURON_THRESHOLD_MUTATE_ON.load(Ordering::Relaxed), "pré-condição: padrão deveria ser true");
        assert!(NEURON_ACTIVATION_MUTATE_ON.load(Ordering::Relaxed), "pré-condição: padrão deveria ser true");

        // ── Parte 1: padrão (mutação LIGADA) — faixas respeitadas + algo mudou ──
        let n = 12;
        let mut genomes: Vec<Genome> = (0..n).map(|i| Genome::random(i as u64)).collect();
        let params = EvoParams { p_mutate: 1.0, sigma: 0.05, n_elite: 1, tournament_k: 3, congela_l2_cegas: false, l2_linhas_lidas: L2_LINHAS_LIDAS, mascara_regioes: false };
        let fitnesses: Vec<f64> = (0..n).map(|i| i as f64).collect();

        for geracao in 0..40 {
            genomes = evolve(&genomes, &fitnesses, &params, geracao, 0);
            for g in &genomes {
                for &a in &g.neuron_alpha {
                    assert!((0.0..=0.95).contains(&a), "geração {geracao}: alfa {a} fora de [0, 0.95]");
                }
                for &p in &g.neuron_period_l1 {
                    assert!((1.0..=8.0).contains(&p), "geração {geracao}: período {p} fora de [1, 8]");
                    assert_eq!(p, p.round(), "geração {geracao}: período {p} deveria ser inteiro");
                }
                for &t in &g.neuron_threshold_l1 {
                    assert!((0.0..=0.9).contains(&t), "geração {geracao}: limiar {t} fora de [0, 0.9]");
                }
                for &act in &g.neuron_activation {
                    assert!((0.0..=3.0).contains(&act) && act == act.floor(),
                        "geração {geracao}: id de ativação {act} deveria ser inteiro em [0,3]");
                }
            }
        }

        // Depois de 40 gerações com p_mutate=1.0 (toda mutação sempre dispara), pelo
        // menos ALGUM valor deveria ter saído do neutro — senão o teste acima
        // estaria vacuamente verdadeiro (tudo continua 0.0/1.0/0.0/0.0 e "nunca sai
        // da faixa" seria trivial, não uma prova de que a mutação funciona).
        let algum_alfa_mudou = genomes.iter().any(|g| g.neuron_alpha.iter().any(|&a| a != 0.0));
        let algum_periodo_mudou = genomes.iter().any(|g| g.neuron_period_l1.iter().any(|&p| p != 1.0));
        let algum_limiar_mudou = genomes.iter().any(|g| g.neuron_threshold_l1.iter().any(|&t| t != 0.0));
        let alguma_ativacao_mudou = genomes.iter().any(|g| g.neuron_activation.iter().any(|&a| a != 0.0));
        assert!(algum_alfa_mudou, "40 gerações com p_mutate=1.0 e nenhum alfa mudou — mutação não está rodando");
        assert!(algum_periodo_mudou, "40 gerações com p_mutate=1.0 e nenhum período mudou — mutação não está rodando");
        assert!(algum_limiar_mudou, "40 gerações com p_mutate=1.0 e nenhum limiar mudou — mutação não está rodando");
        assert!(alguma_ativacao_mudou, "40 gerações e nenhuma ativação mudou — mutação discreta não está rodando");

        // ── Parte 2: braço de controle (`--sem-*`) — congela no neutro ──────────
        // `let _guard` devolve os atomics ao padrão mesmo se o resto do teste
        // falhar no meio, para não vazar `false` para testes que rodem DEPOIS
        // deste no mesmo processo (mesmo que hoje nenhum outro toque estes
        // atomics — defesa contra o próximo teste que vier a tocar).
        struct RestauraAtomics;
        impl Drop for RestauraAtomics {
            fn drop(&mut self) {
                NEURON_MEM_MUTATE_ON.store(true, Ordering::Relaxed);
                NEURON_PERIOD_MUTATE_ON.store(true, Ordering::Relaxed);
                NEURON_THRESHOLD_MUTATE_ON.store(true, Ordering::Relaxed);
                NEURON_ACTIVATION_MUTATE_ON.store(true, Ordering::Relaxed);
            }
        }
        let _guard = RestauraAtomics;
        NEURON_MEM_MUTATE_ON.store(false, Ordering::Relaxed);
        NEURON_PERIOD_MUTATE_ON.store(false, Ordering::Relaxed);
        NEURON_THRESHOLD_MUTATE_ON.store(false, Ordering::Relaxed);
        NEURON_ACTIVATION_MUTATE_ON.store(false, Ordering::Relaxed);

        let n2 = 10;
        let mut genomes2: Vec<Genome> = (0..n2).map(|i| Genome::random(i as u64 + 500)).collect();
        let fitnesses2: Vec<f64> = (0..n2).map(|i| i as f64).collect();

        for geracao in 0..20 {
            genomes2 = evolve(&genomes2, &fitnesses2, &params, geracao, 0);
        }

        for g in &genomes2 {
            assert!(g.neuron_alpha.iter().all(|&a| a == 0.0), "--sem-memoria-neuronal deveria congelar alfa em 0.0");
            assert!(g.neuron_period_l1.iter().all(|&p| p == 1.0), "--sem-periodo-neuronal deveria congelar período em 1.0");
            assert!(g.neuron_threshold_l1.iter().all(|&t| t == 0.0), "--sem-limiar-neuronal deveria congelar limiar em 0.0");
            assert!(g.neuron_activation.iter().all(|&a| a == 0.0), "--sem-ativacao-heterogenea deveria congelar ativação em 0.0");
        }
    }

    /// Braço de controle do mecanismo 2: com TODOS os períodos em 1.0 (o neutro
    /// de nascença), o simulador tem que devolver TODOS os neurônios disparando
    /// em TODO tick — desvio ZERO (nada varia), média = n_beings×96. Prova que a
    /// função reproduz "hoje" exatamente quando o período está no default.
    #[test]
    fn simulate_l1_fire_counts_com_periodo_neutro_dispara_todo_neuronio_todo_tick() {
        let n_beings = 5;
        let periods = vec![1.0f32; n_beings * N_L1_NEURONS];
        let (media, desvio) = simulate_l1_fire_counts(&periods, n_beings, 100);
        assert_eq!(media, (n_beings * N_L1_NEURONS) as f64, "período=1 deveria disparar 100% sempre");
        assert_eq!(desvio, 0.0, "sem variação possível com período uniforme=1");
    }

    /// Com período=2 uniforme, a MÉDIA ao longo do tempo é 50% da população —
    /// mas o desvio NÃO é zero: como todos os neurônios têm a MESMA fase
    /// (contador compartilhado), eles disparam em LOCKSTEP — tick par = TODOS
    /// disparam, tick ímpar = NENHUM dispara. É um degrau (0 ou N), não um valor
    /// constante — a assincronia de verdade só aparece quando os PERÍODOS
    /// diferem entre neurônios (ver o teste de população mista, abaixo), não
    /// quando todos compartilham o mesmo período uniforme.
    #[test]
    fn simulate_l1_fire_counts_com_periodo_dois_alterna_entre_zero_e_total() {
        let n_beings = 4;
        let total = (n_beings * N_L1_NEURONS) as f64; // 384
        let periods = vec![2.0f32; n_beings * N_L1_NEURONS];
        let (media, desvio) = simulate_l1_fire_counts(&periods, n_beings, 200);
        assert_eq!(media, total / 2.0, "período=2 deveria disparar 50% em média");
        // Distribuição de dois pontos (0 ou total, 50/50): desvio = total/2 exatamente.
        assert_eq!(desvio, total / 2.0, "degrau 0/total deveria ter desvio = total/2");
    }

    /// População MISTA (metade período=1, metade período=4): a média fica entre
    /// os dois extremos e o desvio passa a ser NÃO-ZERO (o total por tick varia
    /// conforme os múltiplos de 4 se alinham ou não) — sanity check de que a
    /// função realmente soma contribuições heterogêneas, não só replica um caso.
    #[test]
    fn simulate_l1_fire_counts_populacao_mista_tem_desvio_no_zero() {
        let n_beings = 2;
        let mut periods = vec![1.0f32; N_L1_NEURONS]; // ser 0: todos período 1
        periods.extend(vec![4.0f32; N_L1_NEURONS]);   // ser 1: todos período 4
        let (media, desvio) = simulate_l1_fire_counts(&periods, n_beings, 400);
        // Ser 0 contribui 96 sempre; ser 1 contribui 96 a cada 4 ticks, 0 nos outros —
        // média esperada = 96 + 96*0.25 = 120.
        assert!((media - 120.0).abs() < 1.0, "média esperada ≈120, obteve {media}");
        assert!(desvio > 0.0, "população mista deveria ter desvio > 0, obteve {desvio}");
    }

    /// `pack_neuron_meta` produz o layout EXATO que `brain.wgsl` espera: por
    /// genoma, 480 floats nos seis blocos na ordem l1_alpha·l1_period·
    /// l1_threshold·l1_activation·l2_alpha·l2_activation — não intercalados por
    /// ser (ao contrário de `pack_taus`/`pack_sensors`, aqui L1 e L2 vêm de DOIS
    /// vetores do genoma que precisam ser costurados na ordem certa).
    #[test]
    fn pack_neuron_meta_produz_layout_esperado_pelo_shader() {
        let mut g = Genome::random(7);
        // Marca cada bloco com um valor único e identificável, fora do neutro.
        for (i, a) in g.neuron_alpha.iter_mut().enumerate() { *a = 1000.0 + i as f32; }
        for (i, p) in g.neuron_period_l1.iter_mut().enumerate() { *p = 2000.0 + i as f32; }
        for (i, t) in g.neuron_threshold_l1.iter_mut().enumerate() { *t = 3000.0 + i as f32; }
        for (i, act) in g.neuron_activation.iter_mut().enumerate() { *act = 4000.0 + i as f32; }

        let packed = Genome::pack_neuron_meta(&[g.clone()]);
        assert_eq!(packed.len(), N_L1_NEURONS * 4 + N_L2_NEURONS * 2);

        // l1_alpha = neuron_alpha[0..96] = 1000..1096
        for i in 0..N_L1_NEURONS {
            assert_eq!(packed[i], 1000.0 + i as f32, "l1_alpha[{i}]");
        }
        // l1_period = neuron_period_l1[0..96] = 2000..2096
        for i in 0..N_L1_NEURONS {
            assert_eq!(packed[96 + i], 2000.0 + i as f32, "l1_period[{i}]");
        }
        // l1_threshold = neuron_threshold_l1[0..96] = 3000..3096
        for i in 0..N_L1_NEURONS {
            assert_eq!(packed[192 + i], 3000.0 + i as f32, "l1_threshold[{i}]");
        }
        // l1_activation = neuron_activation[0..96] = 4000..4096
        for i in 0..N_L1_NEURONS {
            assert_eq!(packed[288 + i], 4000.0 + i as f32, "l1_activation[{i}]");
        }
        // l2_alpha = neuron_alpha[96..144] = 1096..1144
        for i in 0..N_L2_NEURONS {
            assert_eq!(packed[384 + i], 1000.0 + (N_L1_NEURONS + i) as f32, "l2_alpha[{i}]");
        }
        // l2_activation = neuron_activation[96..144] = 4096..4144
        for i in 0..N_L2_NEURONS {
            assert_eq!(packed[432 + i], 4000.0 + (N_L1_NEURONS + i) as f32, "l2_activation[{i}]");
        }
    }
}
