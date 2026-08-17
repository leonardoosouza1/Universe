//! AUDITORIA DA VISÃO (3 eixos) — ativada por `--auditar-visao`.
//!
//! Canal APROVADO se e somente se:
//!   (a) varia no TEMPO (σ_temporal > 2.0 ao longo dos ticks)
//!   (b) varia ENTRE SERES no último tick (σ_entre > 2.0 com seres espalhados)
//!   (c) tem FONTE documentada (algum kernel WGSL escreve aquele índice)
//!
//! A versão anterior media (b) uma única vez no tick 0, quando os seres ainda estão
//! em posições idênticas — produzia falsos mortos para canais proprioceptivos corretos.
//!
//! Estrutura de acumulação: por tick, grava a onda do ser 0 (variância temporal) e a
//! variância entre seres do tick atual (último tick = população espalhada = melhor medida).

use universe::perception::channels::source_for as channel_source_for;

pub const N_CH: usize = universe::config::iara::brain::SENSORY_WAVE_SIZE;

/// Tabela estática: quais canais cada kernel WGSL escreve.
///
/// Delegado para `perception::channels::source_for()` — fonte de verdade única.
/// Mantida como wrapper local para não quebrar as chamadas internas desta função.
#[inline]
fn canal_tem_fonte(ch: usize) -> &'static str {
    channel_source_for(ch)
}

/// Acumulador de auditoria — protegido por Mutex para acesso do loop de tick.
struct AuditAcum {
    /// Histórico temporal do ser 0: n_ticks × N_CH valores i8.
    hist_ser0: Vec<[i8; N_CH]>,
    /// Snapshot do último tick com ≥4 seres VIVOS: N_CH × n_seres.
    ///
    /// Critério (b) original: "último tick da corrida". Problema: se todos morrem, o
    /// último tick tem seres congelados em estado idêntico (freio=1, throttle=0) →
    /// σ_entre≈0 → canal reprova injustamente. Substituído por "último tick com ≥4 vivos",
    /// que garante que a população ainda está se movendo e diferenciada.
    ultimo_tick_4vivos: Vec<[i8; N_CH]>,
    n_ticks: usize,
}

impl AuditAcum {
    fn new() -> Self {
        Self {
            hist_ser0: Vec::new(),
            ultimo_tick_4vivos: Vec::new(),
            n_ticks: 0,
        }
    }
}

static AUDIT_ACUM: std::sync::LazyLock<std::sync::Mutex<AuditAcum>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(AuditAcum::new()));

/// Limita a memória do acumulador (risco R6 do REFACTOR_PLAN.md): sem isto,
/// `--auditar-visao` numa bateria de 100 gerações × 8.000 ticks acumula
/// 800.000 entradas × 208 bytes ≈ 166 MB em `hist_ser0`, sem nunca liberar.
///
/// Chamado uma vez no INÍCIO de cada geração (main.rs). Efeito colateral: o
/// relatório final passa a refletir a ÚLTIMA geração rodada, não a soma de
/// todas — o que é o comportamento útil (população já evoluída, não diluída
/// pelos primeiros indivíduos aleatórios).
pub fn nova_geracao() {
    let mut g = AUDIT_ACUM.lock().unwrap();
    g.hist_ser0.clear();
    g.ultimo_tick_4vivos.clear();
    g.n_ticks = 0;
}

/// Chamado a cada tick quando `--auditar-visao` está ativo.
/// Grava onda do ser 0 (variância temporal) e snapshot de todos (variância entre seres).
/// `n_vivos` = quantos seres têm life_left > 0 neste tick.
pub fn auditoria_acumular(ondas: &[[i8; N_CH]], n: usize, ser_vivo: usize, n_vivos: usize) {
    let mut g = AUDIT_ACUM.lock().unwrap();
    if n == 0 {
        return;
    }
    // Onda do primeiro ser VIVO — trajetória temporal estável (ser morto fica constante).
    g.hist_ser0.push(ondas[ser_vivo.min(n - 1)]);
    // Snapshot do tick atual apenas se ≥4 seres estão vivos.
    // Critério (b) novo: "último tick com ≥4 vivos" em vez de "último tick da corrida".
    // Se todos morrem, o último tick tem seres congelados (freio=1) → σ_entre≈0 → falso reprovado.
    // Com ≥4 a população ainda está diferenciada e a variância mede o canal, não o mecanismo de morte.
    if n_vivos >= 4 {
        g.ultimo_tick_4vivos = ondas[..n].to_vec();
    }
    g.n_ticks += 1;
}

/// Imprime o relatório final depois de todos os ticks.
pub fn auditoria_relatorio() {
    let g = AUDIT_ACUM.lock().unwrap();
    let n_ticks = g.n_ticks;
    if n_ticks == 0 {
        println!("[auditoria] nenhum tick acumulado — ative --auditar-visao");
        return;
    }
    let n_seres = g.ultimo_tick_4vivos.len();
    println!();
    println!(
        "╔══ AUDITORIA DA VISÃO — RELATÓRIO FINAL ({n_ticks} ticks, {n_seres} seres, {N_CH} canais) ══"
    );

    // ── (a) Variância temporal: desvio-padrão ao longo dos ticks, ser 0 ──────
    let mut sigma_temporal = [0.0f32; N_CH];
    for ch in 0..N_CH {
        if g.hist_ser0.is_empty() {
            break;
        }
        let vals: Vec<f32> = g.hist_ser0.iter().map(|f| f[ch] as f32).collect();
        let m = vals.iter().sum::<f32>() / vals.len() as f32;
        let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f32>() / vals.len() as f32;
        sigma_temporal[ch] = var.sqrt();
    }

    // ── (b) Variância entre seres: desvio-padrão no último tick com ≥4 vivos ─
    //
    // Critério anterior: "último tick da corrida". Se todos morrem, o último tick
    // tem seres congelados em (freio=1, throttle=0) → σ_entre≈0 → falso reprovado.
    // Critério novo: "último tick com ≥4 seres vivos", onde a população ainda está
    // diferenciada e a variância mede o canal, não o mecanismo de morte.
    let mut sigma_entre = [0.0f32; N_CH];
    if !g.ultimo_tick_4vivos.is_empty() {
        let n = g.ultimo_tick_4vivos.len() as f32;
        for ch in 0..N_CH {
            let vals: Vec<f32> = g.ultimo_tick_4vivos.iter().map(|f| f[ch] as f32).collect();
            let m = vals.iter().sum::<f32>() / n;
            let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f32>() / n;
            sigma_entre[ch] = var.sqrt();
        }
    }

    // ── (c) Fonte ─────────────────────────────────────────────────────────────
    // Classifica por motivo de falha.
    let thr = 2.0_f32; // σ mínimo para considerar "varia"
    let mut aprovados: Vec<usize> = Vec::new();
    let mut falha_temporal: Vec<usize> = Vec::new();
    let mut falha_entre: Vec<usize> = Vec::new();
    let mut falha_fonte: Vec<usize> = Vec::new();
    let mut falha_multi: Vec<usize> = Vec::new(); // múltiplas falhas

    for ch in 0..N_CH {
        let ok_t = sigma_temporal[ch] >= thr;
        let ok_e = sigma_entre[ch] >= thr;
        let ok_f = canal_tem_fonte(ch) != "sem_fonte";
        match (ok_t, ok_e, ok_f) {
            (true, true, true) => aprovados.push(ch),
            (false, true, true) => falha_temporal.push(ch),
            (true, false, true) => falha_entre.push(ch),
            (true, true, false) => falha_fonte.push(ch),
            _ if !ok_f => falha_fonte.push(ch),
            _ => falha_multi.push(ch),
        }
    }

    println!("║");
    println!("║ Critérios de aprovação:");
    println!("║   (a) σ_temporal ≥ {thr:.0} (varia ao longo dos ticks, ser 0)");
    println!("║   (b) σ_entre    ≥ {thr:.0} (varia entre seres — último tick com ≥4 vivos)");
    println!("║   (c) fonte      = kernel WGSL documenta escrita no canal");
    println!("║");
    println!("║ APROVADOS: {}/{N_CH}", aprovados.len());

    // Imprime faixas contíguas de aprovados
    if !aprovados.is_empty() {
        let faixas = faixas_contiguas(&aprovados);
        let faixas_str: Vec<String> = faixas
            .iter()
            .map(|(a, b)| {
                if a == b {
                    format!("{a}")
                } else {
                    format!("{a}-{b}")
                }
            })
            .collect();
        println!("║   {}", faixas_str.join(" "));
    }

    println!("║");
    println!("║ REPROVADOS por tipo de falha:");

    if !falha_temporal.is_empty() {
        println!(
            "║   [sem variância temporal — constante no tempo] {} canais:",
            falha_temporal.len()
        );
        let faixas = faixas_contiguas(&falha_temporal);
        for (a, b) in &faixas {
            if a == b {
                println!(
                    "║     ch{a:3}  σ_t={:.1} σ_e={:.1}  fonte={}",
                    sigma_temporal[*a],
                    sigma_entre[*a],
                    canal_tem_fonte(*a)
                );
            } else {
                println!(
                    "║     ch{a:3}..{b:3}  (ex: σ_t={:.1}  σ_e={:.1})",
                    sigma_temporal[*a], sigma_entre[*a]
                );
            }
        }
    }

    if !falha_entre.is_empty() {
        println!(
            "║   [sem variância entre seres no último tick] {} canais:",
            falha_entre.len()
        );
        let faixas = faixas_contiguas(&falha_entre);
        for (a, b) in &faixas {
            if a == b {
                println!(
                    "║     ch{a:3}  σ_t={:.1} σ_e={:.1}  fonte={}",
                    sigma_temporal[*a],
                    sigma_entre[*a],
                    canal_tem_fonte(*a)
                );
            } else {
                println!(
                    "║     ch{a:3}..{b:3}  (ex: σ_t={:.1}  σ_e={:.1})",
                    sigma_temporal[*a], sigma_entre[*a]
                );
            }
        }
    }

    if !falha_fonte.is_empty() {
        println!("║   [sem fonte documentada] {} canais:", falha_fonte.len());
        for &ch in &falha_fonte {
            println!(
                "║     ch{ch:3}  σ_t={:.1} σ_e={:.1}  fonte={}",
                sigma_temporal[ch],
                sigma_entre[ch],
                canal_tem_fonte(ch)
            );
        }
    }

    if !falha_multi.is_empty() {
        println!("║   [múltiplas falhas] {} canais:", falha_multi.len());
        let faixas = faixas_contiguas(&falha_multi);
        for (a, b) in &faixas {
            if a == b {
                println!(
                    "║     ch{a:3}  σ_t={:.1} σ_e={:.1}  fonte={}",
                    sigma_temporal[*a],
                    sigma_entre[*a],
                    canal_tem_fonte(*a)
                );
            } else {
                println!(
                    "║     ch{a:3}..{b:3}  (ex: σ_t={:.1}  σ_e={:.1}  fonte={})",
                    sigma_temporal[*a],
                    sigma_entre[*a],
                    canal_tem_fonte(*a)
                );
            }
        }
    }

    // Canais mais informativos (maior σ_temporal)
    let mut ranking_t: Vec<(usize, f32)> = (0..N_CH).map(|ch| (ch, sigma_temporal[ch])).collect();
    ranking_t.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let topo: Vec<String> = ranking_t
        .iter()
        .take(10)
        .map(|(ch, s)| format!("ch{ch}(σ_t{s:.0})"))
        .collect();
    println!("║");
    println!("║ Top-10 σ_temporal: {}", topo.join(" "));

    println!("╚══");
}

/// Só existe para `auditoria_relatorio()` decidir se há dados acumulados mesmo
/// quando a flag `--auditar-visao` já foi consultada e limpa em outro ponto.
pub fn n_ticks_acumulados() -> usize {
    AUDIT_ACUM.lock().unwrap().n_ticks
}

fn faixas_contiguas(idxs: &[usize]) -> Vec<(usize, usize)> {
    if idxs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut ini = idxs[0];
    let mut ant = idxs[0];
    for &c in idxs.iter().skip(1) {
        if c != ant + 1 {
            out.push((ini, ant));
            ini = c;
        }
        ant = c;
    }
    out.push((ini, ant));
    out
}
