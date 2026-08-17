//! GENOME VIEWER — detailed anatomy of the saved brain (`evolve_best_gen*.bin`).
//!
//! # Why it exists
//!
//! Leonardo: *"when I ask you to show me some technical data, it has to be detailed —
//! a chart, what the clustering looks like, what the distribution of those neurons looks
//! like."*
//!
//! The original `dna_view` reads the `.dna` format (ternary TLV, 128×64) of the Bevy
//! path. The GPU evolution path saves another format — `UGEN` v2, with a 192×96 L1, a
//! 96×48 L2, KAN knots, slow brain and τ bank. This binary reads THAT format and answers
//! what the three-line report of `--ativacao` does not: **where** the competence lives
//! inside the brain.
//!
//! The activation report measures the brain RUNNING (needs a GPU and a world). This one
//! measures the brain AT REST — only the weights, simulating nothing. The two readings
//! cover each other: a neuron can have a large weight and never fire, and that difference
//! is what matters.
//!
//! # What it shows
//!
//! · weight distribution per layer, with an ASCII histogram
//! · **grouping by sensory channel** — which group of senses feeds the brain the most,
//!   using the real names from the `Channel` registry (not loose indices)
//! · **per-neuron profile** — which are dead, which dominate, how much input they receive
//! · L1 and L2 heatmap
//! · τ bank: distribution of the time scales (the brain's "bank of clocks")
//!
//! # Usage
//!
//!   genoma_view <path.bin> [--top 20] [--heatmap]

use universe::config::iara::brain::{L1_NEURONS, L2_NEURONS, SENSORY_WAVE_SIZE};
// `name_for` is a free fn of the module: covers all 192 indices, while the `Channel`
// enum only has variants for the individually named channels.
use universe::math::stats;
use universe::perception::channels::name_for;

const MAGIC: &[u8; 4] = b"UGEN";

/// Genome for display. Filled by the CANONICAL parser (`universe::gpu::genome_io`).
///
/// The read replica that used to live here broke silently when the P1.6/P1.7
/// blocks entered the format: "τ = last 96 floats of the file" started reading
/// `neuron_activation` — and the real champion (τ spreads 3.2× in the training log)
/// showed up as "monotemporal 0.0×". Never a duplicated parser again.
struct Genoma {
    versao: u32,
    l1: Vec<f32>,
    l2: Vec<f32>,
    kan: Vec<f32>,
    slow: Vec<f32>,
    tau: Vec<f32>,
}

fn carregar(path: &str) -> Option<Genoma> {
    // Sniff only to DISPLAY the version; the actual read is done by the canonical parser.
    let versao = std::fs::read(path)
        .ok()
        .map(|b| {
            if b.len() > 8 && &b[0..4] == MAGIC {
                u32::from_le_bytes([b[4], b[5], b[6], b[7]])
            } else {
                1
            }
        })
        .unwrap_or(0);
    let g = universe::gpu::genome_io::load_genome(path)?;
    // `Genome.tau` stores the rate d (s⁻¹); display converts to τ = 1/d further on,
    // identical to what it was before (the field already came as d in the format).
    let kan: Vec<f32> = g.kan_knots.iter().flat_map(|r| r.iter().copied()).collect();
    Some(Genoma {
        versao,
        l1: g.l1,
        l2: g.l2,
        kan,
        slow: g.slow,
        tau: g.tau,
    })
}

/// ASCII histogram — the "shape" of the distribution, which mean and deviation hide.
fn histograma(vals: &[f32], bins: usize, largura: usize) {
    if vals.is_empty() {
        println!("    (empty)");
        return;
    }
    let h = stats::histogram(vals, bins);
    let lo = vals.iter().cloned().fold(f32::MAX, f32::min);
    let hi = vals.iter().cloned().fold(f32::MIN, f32::max);
    let pico = *h.iter().max().unwrap_or(&1) as f32;
    for (i, &c) in h.iter().enumerate() {
        let a = lo + (hi - lo) * i as f32 / bins as f32;
        let n = ((c as f32 / pico) * largura as f32).round() as usize;
        println!(
            "    {a:+7.3} │{}{} {c}",
            "█".repeat(n),
            " ".repeat(largura - n)
        );
    }
}

fn resumo(nome: &str, w: &[f32]) {
    if w.is_empty() {
        println!("  {nome}: absent");
        return;
    }
    // `percentile` sorts in place, so work on a mutable copy.
    let mut abs: Vec<f32> = w.iter().map(|x| x.abs()).collect();
    let media = stats::mean(w);
    let sigma = stats::std_dev(w);
    let mortos = abs.iter().filter(|&&x| x < 0.01).count();
    let fortes = abs.iter().filter(|&&x| x > 1.0).count();
    println!(
        "  {nome}: {} weights · mean {media:+.4} · σ {sigma:.4} · |w|max {:.3}",
        w.len(),
        abs.iter().cloned().fold(0.0, f32::max)
    );
    println!(
        "    near-zero (|w|<0.01): {mortos} ({:.1}%) · strong (|w|>1.0): {fortes} ({:.1}%)",
        mortos as f32 * 100.0 / w.len() as f32,
        fortes as f32 * 100.0 / w.len() as f32
    );
    println!(
        "    |w| percentiles: p50 {:.3} · p90 {:.3} · p99 {:.3}",
        stats::percentile(&mut abs, 50.0),
        stats::percentile(&mut abs, 90.0),
        stats::percentile(&mut abs, 99.0)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: genoma_view <path.bin> [--top N] [--heatmap]");
        std::process::exit(1);
    });
    let mut top = 20usize;
    let mut heatmap = false;
    let mut html_out: Option<String> = None;
    let mut json_out: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--top" => {
                i += 1;
                top = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(20);
            }
            "--heatmap" => heatmap = true,
            // Leonardo: "I really want to be able to see, like, who talks to whom, what
            // the groups are" — text does not show topology. `--html` writes a
            // self-contained page with the brain graph.
            "--html" => {
                i += 1;
                html_out = args.get(i).cloned();
            }
            "--json" => {
                i += 1;
                json_out = args.get(i).cloned();
            }
            _ => {}
        }
        i += 1;
    }

    let Some(g) = carregar(&path) else {
        eprintln!("failed to read {path}");
        std::process::exit(1);
    };

    println!("═══════════════════════════════════════════════════════════════════");
    println!("🧬 GENOME — {path}");
    println!("═══════════════════════════════════════════════════════════════════");
    println!(
        "  format UGEN v{} · L1 {}×{} · L2 {}×{} · KAN {} · slow {} · τ {}",
        g.versao,
        SENSORY_WAVE_SIZE,
        L1_NEURONS,
        L1_NEURONS,
        L2_NEURONS,
        g.kan.len(),
        g.slow.len(),
        g.tau.len()
    );
    println!();

    println!("─── WEIGHT DISTRIBUTION ───────────────────────────────────────────");
    resumo("L1 (sensory → fast)", &g.l1);
    resumo("L2 (fast → motor)", &g.l2);
    resumo("slow (modulation)", &g.slow);
    println!("\n  shape of L1:");
    histograma(&g.l1, 16, 40);

    // ── GROUPING BY SENSE ──────────────────────────────────────────────────
    //
    // The question that matters is not "which weight is larger", it is "which SENSE the
    // brain depends on". Summing |w| of all neurons per input channel yields the brain's
    // attention budget — and a channel with sum ~0 is a sense it learned to ignore,
    // either because it is useless or because it is broken (that is how the dead channels
    // showed up in the 2026-08-11 audit).
    println!("\n─── ATTENTION BUDGET PER SENSORY CHANNEL ──────────────────────────");
    let mut por_canal: Vec<(usize, f32)> = (0..SENSORY_WAVE_SIZE)
        .map(|ch| {
            let soma: f32 = (0..L1_NEURONS)
                .map(|n| {
                    g.l1.get(n * SENSORY_WAVE_SIZE + ch)
                        .map_or(0.0, |w| w.abs())
                })
                .sum();
            (ch, soma)
        })
        .collect();
    let total: f32 = por_canal.iter().map(|(_, s)| s).sum::<f32>().max(1e-6);
    por_canal.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  the {top} best-served channels:");
    for (ch, soma) in por_canal.iter().take(top) {
        let pct = soma * 100.0 / total;
        let barra = "▁▂▃▄▅▆▇█"
            .chars()
            .nth(((pct / 2.0).min(7.0)) as usize)
            .unwrap_or('▁');
        println!("    ch{ch:3} {barra} {pct:5.2}%  {}", name_for(*ch));
    }
    let ignorados: Vec<usize> = por_canal
        .iter()
        .filter(|(_, s)| *s * 100.0 / total < 0.05)
        .map(|(c, _)| *c)
        .collect();
    println!(
        "\n  practically ignored channels (<0.05% of attention): {}",
        ignorados.len()
    );
    if !ignorados.is_empty() {
        let amostra: Vec<String> = ignorados.iter().take(24).map(|c| format!("{c}")).collect();
        println!("    {}", amostra.join(" "));
    }

    // ── PER-NEURON PROFILE ─────────────────────────────────────────────────
    println!("\n─── PROFILE OF THE {L1_NEURONS} L1 NEURONS ───────────────────────────────");
    let mut perfil: Vec<(usize, f32, f32)> = (0..L1_NEURONS)
        .map(|n| {
            let ini = n * SENSORY_WAVE_SIZE;
            let fim = (ini + SENSORY_WAVE_SIZE).min(g.l1.len());
            let fatia = &g.l1[ini.min(g.l1.len())..fim];
            let soma: f32 = fatia.iter().map(|w| w.abs()).sum();
            // Output: how much this neuron weighs in L2 — measures whether anyone listens to it.
            let saida: f32 = (0..L2_NEURONS)
                .map(|j| g.l2.get(j * L1_NEURONS + n).map_or(0.0, |w| w.abs()))
                .sum();
            (n, soma, saida)
        })
        .collect();
    let entrada_media = perfil.iter().map(|(_, e, _)| e).sum::<f32>() / L1_NEURONS as f32;
    let mudos = perfil
        .iter()
        .filter(|(_, _, s)| *s < 0.05 * entrada_media)
        .count();
    let surdos = perfil
        .iter()
        .filter(|(_, e, _)| *e < 0.05 * entrada_media)
        .count();
    println!("  mean input per neuron: {entrada_media:.2}");
    println!("  DEAF (receive almost no input): {surdos}");
    println!("  MUTE (L2 barely listens to them): {mudos}");
    perfil.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    println!("\n  the 10 most influential neurons (by output weight in L2):");
    for (n, ent, sai) in perfil.iter().take(10) {
        // Which channel each depends on the most — gives the neuron's "role".
        let dom = (0..SENSORY_WAVE_SIZE)
            .max_by(|&a, &b| {
                let wa = g.l1.get(n * SENSORY_WAVE_SIZE + a).map_or(0.0, |w| w.abs());
                let wb = g.l1.get(n * SENSORY_WAVE_SIZE + b).map_or(0.0, |w| w.abs());
                wa.partial_cmp(&wb).unwrap()
            })
            .unwrap_or(0);
        println!(
            "    N{n:02}  input {ent:6.2}  output {sai:6.2}  ← listens mostly to ch{dom} ({})",
            name_for(dom)
        );
    }

    // ── τ BANK ─────────────────────────────────────────────────────────────
    if !g.tau.is_empty() {
        println!("\n─── τ BANK (CfC time scales) ──────────────────────────────────────");
        let mut t = g.tau.clone();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {} channels · rate d [{:.3}, {:.3}] s⁻¹ · τ = 1/d [{:.3}, {:.3}] s",
            t.len(),
            t[0],
            t[t.len() - 1],
            1.0 / t[t.len() - 1].max(1e-6),
            1.0 / t[0].max(1e-6)
        );
        let mut t90 = t.clone();
        let mut t10 = t.clone();
        println!(
            "  p90/p10 spread: {:.1}× (1.0 = monotemporal population)",
            stats::percentile(&mut t90, 90.0) / stats::percentile(&mut t10, 10.0).max(1e-6)
        );
        println!("\n  shape:");
        histograma(&t, 12, 34);
    }

    if heatmap {
        println!("\n─── L2 HEATMAP (48 motors × 96 neurons) ───────────────────────────");
        println!("  + strong positive · - strong negative · · weak · . null");
        for j in 0..L2_NEURONS.min(48) {
            let linha: String = (0..L1_NEURONS)
                .map(
                    |n| match g.l2.get(j * L1_NEURONS + n).copied().unwrap_or(0.0) {
                        w if w > 0.3 => '+',
                        w if w < -0.3 => '-',
                        w if w.abs() > 0.05 => '·',
                        _ => '.',
                    },
                )
                .collect();
            println!("  M{j:02} │{linha}");
        }
    }

    // `--json`: dump completo pro LABORATÓRIO DE OBSERVAÇÃO (pesos + nomes de canal
    // + P1.7 por neurônio), via parser canônico — nunca duplicar leitura de .bin.
    if let Some(dest) = &json_out {
        let gen_full =
            universe::gpu::genome_io::load_genome(std::env::args().nth(1).unwrap().as_str())
                .unwrap();
        let nomes: Vec<String> = (0..SENSORY_WAVE_SIZE)
            .map(|c| format!("\"{}\"", name_for(c)))
            .collect();
        let docs: Vec<String> = (0..SENSORY_WAVE_SIZE)
            .map(|c| format!("{:?}", universe::perception::channels::doc_for(c)))
            .collect();
        let arr = |v: &[f32]| {
            v.iter()
                .map(|x| format!("{x:.4}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let js = format!(
            "{{\"l1\":[{}],\"l2\":[{}],\"tau\":[{}],\"act\":[{}],\"alpha\":[{}],\"nomes\":[{}],\"docs\":[{}]}}",
            arr(&gen_full.l1),
            arr(&gen_full.l2),
            arr(&gen_full.tau),
            arr(&gen_full.neuron_activation),
            arr(&gen_full.neuron_alpha),
            nomes.join(","),
            docs.join(",")
        );
        std::fs::write(dest, js).expect("gravar json");
        println!("  → json do genoma em {dest}");
    }
    if let Some(dest) = &html_out {
        match escrever_html(&g, dest) {
            Ok(_) => println!("\n  → mapa visual do cérebro em {dest}"),
            Err(e) => eprintln!("\n  falha ao escrever {dest}: {e}"),
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  pesos parados ≠ neurônio ativo. Cruze com [ativacao] do gpu_evolve:");
    println!("  peso grande que nunca dispara é capacidade desperdiçada.");
}

// ── MAPA VISUAL DO CÉREBRO ────────────────────────────────────────────────────
//
// Leonardo: *"eu não quero só que você me mande em texto — eu quero poder ver, tipo,
// aqui tem neurônio x, neurônio y, eles fazem comunicação entre eles. Quem conversa com
// quem, quem ativa, quais grupos são."*
//
// Escreve uma página HTML autocontida (nenhum recurso externo, abre com duplo clique) com
// quatro leituras que o texto não dá:
//
//   1. GRAFO  — 96 neurônios de L1 em anel, ligados aos 48 motores de L2. A espessura e a
//      cor de cada aresta são o peso; só as ligações fortes são desenhadas, senão viram
//      4.608 linhas de espaguete.
//   2. GRUPOS — neurônios agrupados por SEMELHANÇA DE ENTRADA (correlação dos vetores de
//      peso). Dois neurônios que escutam os mesmos canais fazem o mesmo trabalho: se a
//      evolução tivesse especializado o cérebro, apareceriam blocos; se ainda é sorteio,
//      aparece ruído uniforme. É o teste visual de "evoluiu ou teve sorte".
//   3. MATRIZ — L2×L1 completa em cores, para achar motor que ignora o cérebro inteiro.
//   4. ATENÇÃO — orçamento por canal sensorial, barras ordenadas.
fn escrever_html(g: &Genoma, dest: &str) -> std::io::Result<()> {
    use std::fmt::Write as _;

    // Peso de saída de cada neurônio de L1 (o quanto L2 o escuta) — vira o raio do nó.
    let saida: Vec<f32> = (0..L1_NEURONS)
        .map(|n| {
            (0..L2_NEURONS)
                .map(|j| g.l2.get(j * L1_NEURONS + n).map_or(0.0, |w| w.abs()))
                .sum()
        })
        .collect();
    let saida_max = saida.iter().cloned().fold(1e-6, f32::max);

    // Correlação entre vetores de entrada de dois neurônios: mede se fazem o mesmo trabalho.
    let vetor = |n: usize| -> &[f32] {
        let a = (n * SENSORY_WAVE_SIZE).min(g.l1.len());
        let b = (a + SENSORY_WAVE_SIZE).min(g.l1.len());
        &g.l1[a..b]
    };
    let mut sim = vec![0.0f32; L1_NEURONS * L1_NEURONS];
    for a in 0..L1_NEURONS {
        let va = vetor(a);
        let na: f32 = va.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for b in 0..L1_NEURONS {
            let vb = vetor(b);
            let nb: f32 = vb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            let dot: f32 = va.iter().zip(vb).map(|(x, y)| x * y).sum();
            sim[a * L1_NEURONS + b] = dot / (na * nb);
        }
    }
    // Fora da diagonal: se a evolução agrupou funções, existe par com |cos| alto.
    let mut fora: Vec<f32> = Vec::new();
    for a in 0..L1_NEURONS {
        for b in 0..L1_NEURONS {
            if a != b {
                fora.push(sim[a * L1_NEURONS + b].abs());
            }
        }
    }
    fora.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let sim_max = fora.first().copied().unwrap_or(0.0);
    let sim_med = fora.iter().sum::<f32>() / fora.len().max(1) as f32;

    // Atenção por canal.
    let mut aten: Vec<(usize, f32)> = (0..SENSORY_WAVE_SIZE)
        .map(|ch| {
            (
                ch,
                (0..L1_NEURONS)
                    .map(|n| {
                        g.l1.get(n * SENSORY_WAVE_SIZE + ch)
                            .map_or(0.0, |w| w.abs())
                    })
                    .sum(),
            )
        })
        .collect();
    let aten_tot: f32 = aten.iter().map(|(_, s)| s).sum::<f32>().max(1e-6);
    aten.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut h = String::new();
    write!(h, r#"<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Cérebro — mapa visual</title><style>
:root{{--bg:#0d1117;--fg:#e6edf3;--dim:#8b949e;--pos:#3fb950;--neg:#f85149;--card:#161b22}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,sans-serif;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:28px 0 8px;color:var(--fg)}}
.sub{{color:var(--dim);margin-bottom:20px;font-size:13px}}
.card{{background:var(--card);border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}
.kpi{{font-size:26px;font-weight:600}} .kpi small{{font-size:12px;color:var(--dim);font-weight:400;display:block}}
canvas{{max-width:100%;height:auto;display:block}}
.leg{{color:var(--dim);font-size:12px;margin-top:8px}}
.bar{{height:13px;background:linear-gradient(90deg,#1f6feb,#58a6ff);border-radius:3px}}
table{{border-collapse:collapse;width:100%;font-size:12px}} td{{padding:2px 6px;border-bottom:1px solid #21262d}}
td.n{{color:var(--dim);width:74px}} td.v{{width:56px;text-align:right;color:var(--dim)}}
</style></head><body>
<h1>Cérebro — mapa visual</h1>
<div class="sub">L1 {L1_NEURONS} neurônios · L2 {L2_NEURONS} motores · {SENSORY_WAVE_SIZE} canais sensoriais · UGEN v{}</div>
<div class="grid">
 <div class="card"><div class="kpi">{:.3}<small>maior semelhança entre dois neurônios (1,0 = clones)</small></div></div>
 <div class="card"><div class="kpi">{:.3}<small>semelhança média — perto de 0 significa que cada um faz coisa diferente por acaso, não por especialização</small></div></div>
 <div class="card"><div class="kpi">{:.2}%<small>atenção do canal mais atendido (uniforme seria {:.2}%)</small></div></div>
</div>
"#, g.versao, sim_max, sim_med, aten[0].1 * 100.0 / aten_tot, 100.0 / SENSORY_WAVE_SIZE as f32).ok();

    // ── Grafo L1 → L2 ────────────────────────────────────────────────────────
    write!(h, r#"<h2>Quem conversa com quem</h2><div class="card"><canvas id="g" width="1100" height="620"></canvas>
<div class="leg">Anel externo: os {L1_NEURONS} neurônios de L1, raio proporcional a quanto L2 os escuta. Centro: os {L2_NEURONS} motores.
Verde = ligação excitatória, vermelho = inibitória. Só ligações |w| &gt; 0,45 são desenhadas.</div></div>
<script>
const L1={L1_NEURONS}, L2={L2_NEURONS};
const saida={:?};
const arestas={};
const c=document.getElementById('g'), x=c.getContext('2d');
const cx=550, cy=310, R=270, r=120;
function posL1(i){{const a=i/L1*2*Math.PI-Math.PI/2;return [cx+R*Math.cos(a), cy+R*Math.sin(a)];}}
function posL2(j){{const a=j/L2*2*Math.PI-Math.PI/2;return [cx+r*Math.cos(a), cy+r*Math.sin(a)];}}
for(const [n,j,w] of arestas){{
  const [x1,y1]=posL1(n), [x2,y2]=posL2(j);
  x.strokeStyle = w>0 ? `rgba(63,185,80,${{Math.min(Math.abs(w),1)*0.5}})` : `rgba(248,81,73,${{Math.min(Math.abs(w),1)*0.5}})`;
  x.lineWidth=Math.min(Math.abs(w)*2.4,2.4); x.beginPath(); x.moveTo(x1,y1);
  x.quadraticCurveTo(cx,cy,x2,y2); x.stroke();
}}
for(let j=0;j<L2;j++){{const [px,py]=posL2(j);x.fillStyle='#58a6ff';x.beginPath();x.arc(px,py,4,0,7);x.fill();}}
for(let n=0;n<L1;n++){{const [px,py]=posL1(n);const s=saida[n];
  x.fillStyle='#e3b341';x.beginPath();x.arc(px,py,3+s*7,0,7);x.fill();
  if(n%8===0){{const a=n/L1*2*Math.PI-Math.PI/2;x.fillStyle='#8b949e';x.font='10px sans-serif';
    x.fillText('N'+n, cx+(R+18)*Math.cos(a)-8, cy+(R+18)*Math.sin(a)+3);}}
}}
</script>"#,
        saida.iter().map(|s| (s / saida_max * 100.0).round() / 100.0).collect::<Vec<f32>>(),
        {
            // Serializa a mao como array de arrays. `{:?}` de tupla Rust produz
            // "(0, 1, 0.5)", que em JS e o operador virgula — o array chegava como lixo e
            // o grafo saia em branco.
            let mut e = String::from("[");
            for j in 0..L2_NEURONS { for n in 0..L1_NEURONS {
                let w = g.l2.get(j * L1_NEURONS + n).copied().unwrap_or(0.0);
                if w.abs() > 0.45 {
                    if e.len() > 1 { e.push(','); }
                    let _ = write!(e, "[{n},{j},{:.2}]", w);
                }
            }}
            e.push(']');
            e
        }).ok();

    // ── Mapa de semelhança ───────────────────────────────────────────────────
    write!(h, r#"<h2>Grupos: quem faz o mesmo trabalho</h2><div class="card"><canvas id="s" width="580" height="580"></canvas>
<div class="leg">Cada célula é a semelhança entre os vetores de entrada de dois neurônios. <b>Blocos</b> = a evolução formou
grupos especializados. <b>Ruído uniforme</b> = cada neurônio ficou diferente por sorteio, não por função — o cérebro
ainda não se organizou.</div></div>
<script>
const S={:?};
const cs=document.getElementById('s'), sx=cs.getContext('2d'), px=580/{L1_NEURONS};
for(let a=0;a<{L1_NEURONS};a++)for(let b=0;b<{L1_NEURONS};b++){{
  const v=S[a*{L1_NEURONS}+b];
  sx.fillStyle = v>0 ? `rgba(63,185,80,${{Math.min(v*3,1)}})` : `rgba(248,81,73,${{Math.min(-v*3,1)}})`;
  sx.fillRect(b*px,a*px,px+0.5,px+0.5);
}}
</script>"#, sim.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<f32>>()).ok();

    // ── Atenção por canal ────────────────────────────────────────────────────
    write!(
        h,
        r#"<h2>Orçamento de atenção por sentido</h2><div class="card"><table>"#
    )
    .ok();
    for (ch, s) in aten.iter().take(28) {
        let pct = s * 100.0 / aten_tot;
        write!(h, r#"<tr><td class="n">ch{ch}</td><td>{}</td><td style="width:55%"><div class="bar" style="width:{:.1}%"></div></td><td class="v">{pct:.2}%</td></tr>"#,
               name_for(*ch), pct / aten[0].1 * aten_tot * 100.0).ok();
    }
    write!(h, r#"</table><div class="leg">Uniforme seria {:.2}% por canal. Quanto mais plano, menos o cérebro escolheu no que prestar atenção.</div></div>
</body></html>"#, 100.0 / SENSORY_WAVE_SIZE as f32).ok();

    std::fs::write(dest, h)
}
