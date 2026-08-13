// ── COLHEITA DA ENTRADA DO CÉREBRO LENTO ─────────────────────────────────────
//
// # Por que este kernel existe
//
// No `gpu_evolve` a onda sensorial de 223 canais vive INTEIRA na GPU: a percepção
// escreve em `buf_wave` e o kernel do cérebro rápido lê de lá, sem passar pela CPU.
// O cérebro lento precisa de 48 desses canais. Trazer a onda de volta ao host só
// para escolher 48 valores custaria um readback por tick — seria voltar para a CPU
// exatamente no caminho que foi portado para não passar por ela.
//
// Então a seleção acontece aqui, na GPU: 48 threads, cada uma copia um canal.
//
// # O mapa de canais é DECLARADO, não adivinhado
//
// `CHANNEL_MAP[k]` diz de qual canal da onda vem a entrada `k` do lento. Um valor
// negativo significa "esta entrada não tem fonte neste caminho" e o kernel escreve
// zero. Isso é deliberado: em `motor.rs` o lento recebe grandezas do corpo do ser
// (HP, combustível, temperatura de pneu, tempo em parede/grama, memória episódica)
// que simplesmente NÃO EXISTEM no `gpu_evolve`, cujo estado por ser é só física +
// temporizadores. Preencher esses slots com um canal parecido seria inventar sinal;
// zero é honesto e a razão de participação vai mostrar quantas dimensões sobraram.
//
// A onda chega empacotada: 4 canais i8 por u32 (WAVE_U32S = ceil(192/4) = 48 u32
// por ser), que é o mesmo empacotamento que o kernel do cérebro rápido consome.

const INPUT_SIZE: u32 = 48u;
const WAVE_U32S:  u32 = 48u;   // ceil(192/4). Era 56u (onda de 223) — o lento lia
                               // a percepção com stride errado, entrada embaralhada. Corrigido.

struct GatherParams {
    n_beings: u32,
    _pad0:    u32,
    _pad1:    u32,
    _pad2:    u32,
}

@group(0) @binding(0) var<uniform>             gparams:    GatherParams;
@group(0) @binding(1) var<storage, read>       buf_wave:   array<u32>;
@group(0) @binding(2) var<storage, read_write> buf_input:  array<i32>;

// Mapa canal-da-onda → entrada do lento.
//
// As fontes seguem os blocos documentados em `perception.wgsl`. Os índices 0..13 e
// 22..26 ficam sem fonte porque correspondem a estado corporal e memória episódica
// que o `gpu_evolve` não modela.
const CHANNEL_MAP: array<i32, 48> = array<i32, 48>(
    // 0..3   joy_tick, joy_streak, hp, combustível — sem fonte neste caminho
    -1, -1, -1, -1,
    // 4      velocidade
    33,
    // 5..13  temperatura de pneu, potência, alinhamento, 2 reservados, tempo em
    //        parede/grama, estagnação, noite — sem fonte
    -1, -1, 32, -1, -1, -1, -1, -1, -1,
    // 14..17 parede próxima por região (4 linhas da retina, 40/25/15/8 m)
    36, 37, 38, 39,
    // 18..21 parede distante por região
    40, 41, 42, 43,
    // 22..26 memória episódica, intent[3], saída anterior do lento — sem fonte
    -1, -1, -1, -1, -1,
    // 27..28 curvatura a 30 e 60 m
    78, 79,
    // 29     janela de graça — sem fonte
    -1,
    // 30     curvatura a 90 m
    80,
    // 31     distância da parede à frente
    82,
    // 32..47 bins de visão em alta definição (fóvea)
    128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 141, 142, 143,
);

@compute @workgroup_size(48)
fn gather(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let being: u32 = wg.x;
    let k:     u32 = lid.x;
    if being >= gparams.n_beings || k >= INPUT_SIZE { return; }

    let src: i32 = CHANNEL_MAP[k];
    var v: i32 = 0;
    if src >= 0 {
        let ch: u32 = u32(src);
        // Desempacota o i8 do canal `ch`: 4 canais por u32, byte `ch % 4`.
        let word: u32 = buf_wave[being * WAVE_U32S + ch / 4u];
        let byte: u32 = (word >> ((ch % 4u) * 8u)) & 0xFFu;
        // Extensão de sinal de 8 para 32 bits.
        if byte >= 128u { v = i32(byte) - 256; } else { v = i32(byte); }
    }
    buf_input[being * INPUT_SIZE + k] = v;
}
