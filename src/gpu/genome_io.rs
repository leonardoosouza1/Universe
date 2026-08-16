//! MOVIDO do binário gpu_evolve para a lib (2026-08-14, P1.9): o genoma_view
//! mantinha um parser DUPLICADO que quebrou em silêncio quando os blocos P1.6/P1.7
//! entraram no formato (lia neuron_activation achando que era τ — 'espalhamento 0,0×'
//! num campeão com 3,2× real). Parser canônico vive AQUI; binários importam.
//! Persistência de genomas em disco: `save_genome` / `load_genome`.
//!
//! # Formato v2 (com cabeçalho) — E3 (REFACTOR_PLAN.md §2.3, item 3)
//!
//! ```text
//! magic:   4 bytes  = b"UGEN"
//! versao:  4 bytes  = u32 LE = 2
//! L1:      N_L1_WEIGHTS  × f32 LE
//! L2:      N_L2_WEIGHTS  × f32 LE
//! KAN:     3*KAN_OUT_KNOTS × f32 LE
//! slow:    N_SLOW_WEIGHTS × f32 LE   (opcional — lido só se o arquivo for grande o bastante)
//! tau:     N_CFC_CHANNELS × f32 LE   (opcional — idem)
//! sensors: N_SENSOR_GENES × f32 LE   (opcional — idem; FASE SENSORES, degrau 1)
//! neuron_alpha:       (L1+L2) × f32 LE  (opcional — idem; P1.7, mecanismo 1)
//! neuron_period_l1:   L1      × f32 LE  (opcional — idem; P1.7, mecanismo 2)
//! neuron_threshold_l1: L1     × f32 LE  (opcional — idem; P1.7, mecanismo 2)
//! neuron_activation:  (L1+L2) × f32 LE  (opcional — idem; P1.7, mecanismo 3)
//! ```
//!
//! # Compatibilidade com o formato legado (SEM cabeçalho)
//!
//! Antes desta versão o arquivo começava direto nos bytes de L1 — sem magic, sem
//! versão. `load_genome` faz *sniff* dos primeiros 4 bytes: se forem exatamente
//! `b"UGEN"`, lê a versão e pula o cabeçalho (8 bytes); caso contrário, assume
//! formato legado e lê os blocos a partir do byte 0, exatamente como antes.
//!
//! Isso preserva a garantia original: um `.bin` legado gravado antes do cérebro
//! lento e do banco de τ existirem ainda carrega (caindo no default documentado em
//! `load_genome`), e agora um `.bin` v2 também é reconhecido sem ambiguidade — o
//! sniff nunca confunde os dois porque 4 bytes de peso f32 aleatório colidirem com
//! exatamente `b"UGEN"` tem probabilidade ~2⁻³² por arquivo.
//!
//! **NÃO reordenar os campos de `Genome` nem a ordem de escrita abaixo** — o disco
//! não tem nomes de campo, só posição. Reordenar corrompe `.bin` existentes em
//! silêncio (regra R5 do REFACTOR_PLAN.md). Todo campo novo em `Genome` exige
//! atualizar `save_genome` E `load_genome` juntos — os testes em `tests.rs` pegam o
//! esquecimento.

use crate::gpu::evolution::Genome;
use crate::kan::KAN_OUT_KNOTS;
use crate::evolution::dna_genesis::default_kan_knots;

/// Magic do cabeçalho v2 — 4 bytes ASCII, "Universe GENome".
const MAGIC: &[u8; 4] = b"UGEN";
/// Versão atual do formato. Bump aqui SEMPRE que o layout dos blocos mudar
/// (campo novo, ordem nova) — `load_genome` decide o que ler a partir dela.
const FORMAT_VERSION: u32 = 2;

/// Salva o melhor genoma: cabeçalho v2 + L1 | L2 | KAN | lento | banco de τ, tudo
/// f32 little-endian.
///
/// O lento estava faltando historicamente. Salvar só L1/L2/KAN significava que o
/// campeão perdia os 1091 pesos do cérebro lento ao ser recarregado por
/// `--carregar`: voltava com a inicialização padrão, idêntica para todo mundo.
/// Gerações de seleção sobre a modulação eram descartadas em silêncio a cada
/// retomada.
pub fn save_genome(genome: &Genome, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&FORMAT_VERSION.to_le_bytes())?;
    for w in &genome.l1 { f.write_all(&w.to_le_bytes())?; }
    for w in &genome.l2 { f.write_all(&w.to_le_bytes())?; }
    for row in &genome.kan_knots {
        for k in row { f.write_all(&k.to_le_bytes())?; }
    }
    for w in &genome.slow { f.write_all(&w.to_le_bytes())?; }
    for d in &genome.tau  { f.write_all(&d.to_le_bytes())?; }
    for s in &genome.sensors { f.write_all(&s.to_le_bytes())?; }
    // P1.7 (docs/PLANO_EVOLUCAO.md) — cérebro assíncrono: memória, período/limiar,
    // ativação heterogênea. Mesma regra dos blocos acima: campo novo aqui SEM
    // atualizar `load_genome` junto é o gap 7 do CLAUDE.md do projeto (perdemos
    // 1091 pesos em silêncio uma vez por esquecer isto) — `tests.rs` cobre.
    for a in &genome.neuron_alpha { f.write_all(&a.to_le_bytes())?; }
    for p in &genome.neuron_period_l1 { f.write_all(&p.to_le_bytes())?; }
    for t in &genome.neuron_threshold_l1 { f.write_all(&t.to_le_bytes())?; }
    for a in &genome.neuron_activation { f.write_all(&a.to_le_bytes())?; }
    // M4: gates do maestro (16 f32) — bloco opcional, sempre por último.
    for a in &genome.gates_w { f.write_all(&a.to_le_bytes())?; }
    Ok(())
}

/// Tenta carregar genoma de um arquivo salvo por `save_genome` — v2 (com cabeçalho
/// `UGEN`) OU legado (sem cabeçalho, bytes de L1 direto no início do arquivo).
pub fn load_genome(path: &str) -> Option<Genome> {
    let bytes = std::fs::read(path).ok()?;

    // Sniff dos primeiros 4 bytes: distingue v2 de legado sem ambiguidade prática.
    let base_off = if bytes.len() >= 8 && &bytes[0..4] == MAGIC {
        let versao = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        // Só existe a v2 até agora; versão desconhecida (maior) ainda tenta ler
        // com o layout atual — arquivo de versão MENOR não deveria ter magic.
        debug_assert!(versao <= FORMAT_VERSION, "genoma salvo por versão futura ({versao})");
        8usize
    } else {
        0usize
    };

    let n_l1 = crate::gpu::evolution::N_L1_WEIGHTS;
    let n_l2 = crate::gpu::evolution::N_L2_WEIGHTS;
    let n_kan = 3 * KAN_OUT_KNOTS;
    let expected = base_off + (n_l1 + n_l2 + n_kan) * 4;
    if bytes.len() < expected { return None; }

    // Leitura por deslocamento explícito. A versão com closure mutando `off` não passa
    // no borrow-check assim que alguém precisa consultar `off` de fora dela — e precisar
    // é inevitável aqui, porque os blocos finais são opcionais e a decisão de lê-los
    // depende de comparar `off` com o tamanho do arquivo.
    fn read_at(bytes: &[u8], off: usize, n: usize) -> Vec<f32> {
        bytes[off..off + n * 4].chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    let mut off = base_off;
    let l1 = read_at(&bytes, off, n_l1);   off += n_l1 * 4;
    let l2 = read_at(&bytes, off, n_l2);   off += n_l2 * 4;
    let kan_flat = read_at(&bytes, off, n_kan); off += n_kan * 4;
    let mut kan_knots = default_kan_knots();
    for o in 0..3 {
        for k in 0..KAN_OUT_KNOTS {
            kan_knots[o][k] = kan_flat[o * KAN_OUT_KNOTS + k];
        }
    }
    // Blocos opcionais, lidos só se o arquivo os contiver. Arquivo gravado antes destes
    // campos cai no padrão — que é a verdade, não uma invenção: aquele genoma realmente
    // não carregava o dado.
    let n_slow = crate::engine::SlowBrain::new().all_weights().len();
    let n_tau  = crate::gpu::evolution::N_CFC_CHANNELS;

    let slow = if bytes.len() >= off + n_slow * 4 {
        let v = read_at(&bytes, off, n_slow); off += n_slow * 4; v
    } else {
        crate::engine::SlowBrain::new().all_weights()
    };
    let tau = if bytes.len() >= off + n_tau * 4 {
        let v = read_at(&bytes, off, n_tau); off += n_tau * 4; v
    } else {
        vec![crate::cfc::DECAY_DEFAULT; n_tau]
    };
    // FASE SENSORES (degrau 1) — mesmo padrão opcional de slow/tau: arquivo gravado
    // antes deste campo existir cai no default, que é a verdade (aquele genoma
    // realmente rodava com os hardcodes de hoje, não com um valor inventado).
    let n_sensors = crate::gpu::evolution::N_SENSOR_GENES;
    let sensors = if bytes.len() >= off + n_sensors * 4 {
        let v = read_at(&bytes, off, n_sensors); off += n_sensors * 4; v
    } else {
        crate::gpu::evolution::SENSOR_DEFAULTS.to_vec()
    };

    // P1.7 — mesmo padrão opcional de slow/tau/sensors: arquivo gravado antes
    // destes campos existirem cai no NEUTRO (idêntico ao cérebro de hoje — ver
    // `Genome::random`), que é a verdade, não uma invenção: aquele genoma
    // realmente rodava sem memória/período/limiar/ativação heterogênea.
    let n_l1 = crate::gpu::evolution::N_L1_NEURONS;
    let n_l2 = crate::gpu::evolution::N_L2_NEURONS;
    let neuron_alpha = if bytes.len() >= off + (n_l1 + n_l2) * 4 {
        let v = read_at(&bytes, off, n_l1 + n_l2); off += (n_l1 + n_l2) * 4; v
    } else {
        vec![0.0; n_l1 + n_l2]
    };
    let neuron_period_l1 = if bytes.len() >= off + n_l1 * 4 {
        let v = read_at(&bytes, off, n_l1); off += n_l1 * 4; v
    } else {
        vec![1.0; n_l1]
    };
    let neuron_threshold_l1 = if bytes.len() >= off + n_l1 * 4 {
        let v = read_at(&bytes, off, n_l1); off += n_l1 * 4; v
    } else {
        vec![0.0; n_l1]
    };
    let neuron_activation = if bytes.len() >= off + (n_l1 + n_l2) * 4 {
        let v = read_at(&bytes, off, n_l1 + n_l2); off += (n_l1 + n_l2) * 4; v
    } else {
        vec![0.0; n_l1 + n_l2]
    };
    // M4: gates do maestro (opcional; genoma antigo → zeros = gates neutros 1,0).
    let gates_w = if bytes.len() >= off + 16 * 4 {
        read_at(&bytes, off, 16)
    } else {
        vec![0.0; 16]
    };

    Some(Genome {
        l1, l2, kan_knots, slow, tau, sensors,
        neuron_alpha, neuron_period_l1, neuron_threshold_l1, neuron_activation,
        gates_w,
    })
}

/// Semente do genoma inicial do ser `i` sob a semente de população `pop_seed`.
///
/// Existia em TRÊS ramos distintos (checkpoint falhou / sem checkpoint / do zero) e eu
/// semeei apenas um, sem conferir qual roda. Resultado: uma bateria inteira de 6 corridas
/// com "três sementes" que eram a MESMA população — `gen 0` idêntico até o tempo de parede
/// nas três, e um efeito pareado de 25σ que era uma amostra repetida três vezes.
///
/// Com a função única, semear é impossível de esquecer num ramo.
pub fn population_seed_for(pop_seed: u64, index: usize) -> u64 {
    // Multiplicador primo grande separa as famílias de sementes: populações de `pop_seed`
    // diferentes não compartilham nenhum genoma inicial.
    pop_seed.wrapping_mul(1_000_003).wrapping_add(index as u64)
}
