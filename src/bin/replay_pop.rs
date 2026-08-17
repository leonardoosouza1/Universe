//! OBSERVATÓRIO DA POPULAÇÃO — todos os seres, quadro a quadro, contra a pista real.
//!
//! # Por que existe
//!
//! Leonardo: *"quando eu pedir para observar era para OBSERVAR MESMO, cada detalhe, até um
//! PEIDO se o modelo pudesse soltar, para sempre ter em mãos DADOS REAIS para comparar e
//! validar."*
//!
//! O `replay_diag` olha UM ser. Isso responde "o que o campeão faz", nunca "por que só 2 de
//! 60 passam". As duas perguntas são diferentes e a segunda é a que importa: se todos
//! falham no MESMO ponto, é o mundo; se falham em pontos diferentes, é o cérebro.
//!
//! Este binário lê o replay inteiro e cruza cada ser com a geometria do circuito,
//! reconstruída da `world_seed`. Sem renderizar, sem interpretar figura, sem achismo.
//!
//! # Uso
//!
//!   replay_pop --replay telemetry/<corrida>/gen0029.replay [--width 28] [--csv saida.csv]

use bevy::math::Vec2;
use std::collections::BTreeMap;
use std::path::PathBuf;
use universe::replay::ReplayReader;
use universe::world::{VoxelType, VoxelWorld};

/// Tudo que dá para saber de um ser a partir do replay + geometria.
struct Perfil {
    id: usize,
    progresso: f32,
    setor_final: usize,
    parado_pct: f32,
    re_pct: f32,
    fora_pct: f32,
    volante_medio: f32,
    volante_desvio: f32,
    veloc_media: f32,
    veloc_max: f32,
    /// Onde o progresso parou de subir — o ponto de desistência.
    travou_em: f32,
    corr_curva: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path: Option<PathBuf> = None;
    let mut width = 18.0f32;
    let mut csv: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--replay" => {
                i += 1;
                if i < args.len() {
                    path = Some(PathBuf::from(&args[i]));
                }
            }
            "--width" => {
                i += 1;
                if i < args.len() {
                    width = args[i].parse().unwrap_or(18.0);
                }
            }
            "--csv" => {
                i += 1;
                if i < args.len() {
                    csv = Some(PathBuf::from(&args[i]));
                }
            }
            _ => {}
        }
        i += 1;
    }
    let Some(path) = path else {
        eprintln!("uso: replay_pop --replay <arquivo.replay> [--width 28] [--csv saida.csv]");
        std::process::exit(2);
    };

    let mut reader = match ReplayReader::open(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("não abriu {}: {e}", path.display());
            std::process::exit(1);
        }
    };
    let seed = reader.header.world_seed;
    let todos = reader.all_frames();
    let world = {
        let _q = universe::world::silence_map_diag();
        VoxelWorld::new(seed, width)
    };
    let lap = world.lap_distance;

    let vivos: Vec<usize> = (0..todos.len()).filter(|&i| !todos[i].is_empty()).collect();
    println!("=== POPULAÇÃO: {} ===", path.display());
    println!(
        "{} seres gravados · volta {lap:.0} m · pista {width:.0} m\n",
        vivos.len()
    );
    if vivos.is_empty() {
        println!("Nenhum ser no replay. Se a corrida é antiga, ela gravava só o líder —");
        println!("regrave com o binário atual para ter a população inteira.");
        return;
    }

    let mut perfis: Vec<Perfil> = Vec::new();
    for &id in &vivos {
        let fr = &todos[id];
        let n = fr.len() as f32;
        let mut parado = 0usize;
        let mut re = 0usize;
        let mut fora = 0usize;
        let (mut vsum, mut vmax) = (0.0f64, 0.0f32);
        let mut str_v: Vec<f32> = Vec::with_capacity(fr.len());
        let mut curv_v: Vec<f32> = Vec::with_capacity(fr.len());
        let mut prog_max = f32::MIN;
        let mut prog_min = f32::MAX;
        let mut travou = 0.0f32;
        let mut ultimo_avanco = 0.0f32;

        for f in fr {
            let p = Vec2::new(f.pos[0], f.pos[2]);
            if !matches!(world.material_at(p), VoxelType::Road) {
                fora += 1;
            }
            if f.speed < 0.5 {
                parado += 1;
            }
            if f.gear == 0 {
                re += 1;
            }
            vsum += f.speed as f64;
            vmax = vmax.max(f.speed);
            str_v.push(f.final_str);
            curv_v.push(curvatura(&world, p, 25.0));
            let pr = world.get_track_progress(p.x, p.y);
            if pr > prog_max {
                prog_max = pr;
                ultimo_avanco = pr;
            }
            prog_min = prog_min.min(pr);
            travou = ultimo_avanco;
        }
        let m = str_v.iter().sum::<f32>() / n;
        let sd = (str_v.iter().map(|v| (v - m).powi(2)).sum::<f32>() / n).sqrt();
        let pares: Vec<(f32, f32)> = curv_v
            .iter()
            .zip(&str_v)
            .filter(|(c, _)| c.is_finite())
            .map(|(c, s)| (*c, *s))
            .collect();
        let corr = if pares.len() > 30 {
            let (a, b): (Vec<f32>, Vec<f32>) = pares.into_iter().unzip();
            correlacao(&a, &b)
        } else {
            f32::NAN
        };

        let span = (prog_max - prog_min).max(0.0);
        perfis.push(Perfil {
            id,
            progresso: span,
            setor_final: ((travou / lap) * 24.0) as usize,
            parado_pct: 100.0 * parado as f32 / n,
            re_pct: 100.0 * re as f32 / n,
            fora_pct: 100.0 * fora as f32 / n,
            volante_medio: m,
            volante_desvio: sd,
            veloc_media: (vsum / fr.len() as f64) as f32,
            veloc_max: vmax,
            travou_em: travou,
            corr_curva: corr,
        });
    }

    perfis.sort_by(|a, b| b.progresso.partial_cmp(&a.progresso).unwrap());

    println!(
        "{:>4} {:>9} {:>6} {:>7} {:>6} {:>6} {:>8} {:>7} {:>7} {:>7}",
        "ser", "progr(m)", "%volta", "travou", "parado", "ré", "volante", "±", "v.méd", "corr"
    );
    println!("{:-<80}", "");
    for p in perfis.iter().take(12) {
        println!(
            "{:>4} {:>9.0} {:>5.1}% {:>7.0} {:>5.0}% {:>5.0}% {:>+8.2} {:>7.2} {:>7.2} {:>+7.2}",
            p.id,
            p.progresso,
            100.0 * p.progresso / lap,
            p.travou_em,
            p.parado_pct,
            p.re_pct,
            p.volante_medio,
            p.volante_desvio,
            p.veloc_media,
            p.corr_curva
        );
    }
    if perfis.len() > 12 {
        println!("{:>4} … e mais {}", "", perfis.len() - 12);
    }

    // ── Onde a população trava ──────────────────────────────────────────────
    // A pergunta que só a população responde: todos falham no MESMO lugar (o mundo tem um
    // obstáculo) ou em lugares diferentes (é competência individual)?
    let mut histo: BTreeMap<usize, usize> = BTreeMap::new();
    for p in &perfis {
        *histo.entry(p.setor_final).or_insert(0) += 1;
    }
    println!("\n── ONDE CADA UM PAROU (setor de 24) ──");
    let total = perfis.len();
    for (setor, n) in &histo {
        let metros = *setor as f32 * lap / 24.0;
        let barra: String = std::iter::repeat('█')
            .take((*n * 40 / total).max(1))
            .collect();
        println!(
            "  setor {:>2} (~{:>5.0} m) {:>3} seres  {}",
            setor, metros, n, barra
        );
    }
    let (setor_moda, n_moda) = histo
        .iter()
        .max_by_key(|(_, n)| **n)
        .map(|(s, n)| (*s, *n))
        .unwrap();
    let frac = 100.0 * n_moda as f32 / total as f32;
    println!(
        "\n  {:.0}% da população para no setor {} (~{:.0} m).",
        frac,
        setor_moda,
        setor_moda as f32 * lap / 24.0
    );
    if frac > 60.0 {
        println!("  ⇒ MESMO ponto para quase todos: é o MUNDO que barra ali, não competência.");
        descrever(&world, setor_moda as f32 * lap / 24.0);
    } else {
        println!("  ⇒ pontos espalhados: a diferença é de COMPETÊNCIA entre indivíduos.");
    }

    // ── Agregados da população ──────────────────────────────────────────────
    let med = |f: &dyn Fn(&Perfil) -> f32| -> f32 {
        let mut v: Vec<f32> = perfis.iter().map(f).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!("\n── MEDIANAS DA POPULAÇÃO ({} seres) ──", total);
    println!(
        "  progresso {:.0} m ({:.1}% da volta) · parado {:.0}% · ré {:.0}% · fora da pista {:.0}%",
        med(&|p| p.progresso),
        100.0 * med(&|p| p.progresso) / lap,
        med(&|p| p.parado_pct),
        med(&|p| p.re_pct),
        med(&|p| p.fora_pct)
    );
    println!(
        "  volante {:+.2} (±{:.2}) · velocidade {:.2} m/s · correlação com a curva {:+.2}",
        med(&|p| p.volante_medio),
        med(&|p| p.volante_desvio),
        med(&|p| p.veloc_media),
        med(&|p| p.corr_curva)
    );

    if let Some(cp) = csv {
        let mut linhas = String::from(
            "ser,progresso_m,pct_volta,travou_em_m,setor,parado_pct,re_pct,fora_pct,volante_medio,volante_desvio,veloc_media,veloc_max,corr_curva\n",
        );
        for p in &perfis {
            linhas.push_str(&format!(
                "{},{:.1},{:.2},{:.1},{},{:.1},{:.1},{:.1},{:.3},{:.3},{:.2},{:.2},{:.3}\n",
                p.id,
                p.progresso,
                100.0 * p.progresso / lap,
                p.travou_em,
                p.setor_final,
                p.parado_pct,
                p.re_pct,
                p.fora_pct,
                p.volante_medio,
                p.volante_desvio,
                p.veloc_media,
                p.veloc_max,
                p.corr_curva
            ));
        }
        match std::fs::write(&cp, linhas) {
            Ok(_) => println!("\nCSV: {}", cp.display()),
            Err(e) => eprintln!("\nfalhou ao escrever CSV: {e}"),
        }
    }
}

fn descrever(world: &VoxelWorld, at: f32) {
    let wps = &world.waypoints_world;
    if wps.len() < 3 {
        return;
    }
    let mut acc = 0.0f32;
    let mut idx = 0usize;
    for i in 0..wps.len() - 1 {
        let seg = wps[i].distance(wps[i + 1]);
        if acc + seg >= at {
            idx = i;
            break;
        }
        acc += seg;
    }
    let a = wps[idx.saturating_sub(1)];
    let b = wps[idx];
    let c = wps[(idx + 1).min(wps.len() - 1)];
    let v1 = (b - a).normalize_or_zero();
    let v2 = (c - b).normalize_or_zero();
    let ang = v1.dot(v2).clamp(-1.0, 1.0).acos().to_degrees();
    let lado = if v1.x * v2.y - v1.y * v2.x > 0.0 {
        "direita"
    } else {
        "esquerda"
    };
    println!("  Geometria ali: {ang:.0}° para a {lado} (waypoint {idx}).");
}

fn curvatura(world: &VoxelWorld, p: Vec2, ahead: f32) -> f32 {
    let wps = &world.waypoints_world;
    if wps.len() < 4 {
        return f32::NAN;
    }
    let mut i0 = 0usize;
    let mut best = f32::MAX;
    for (i, w) in wps.iter().enumerate() {
        let d = w.distance(p);
        if d < best {
            best = d;
            i0 = i;
        }
    }
    let mut acc = 0.0;
    let mut i = i0;
    while acc < ahead && i + 1 < wps.len() {
        acc += wps[i].distance(wps[i + 1]);
        i += 1;
    }
    if i + 1 >= wps.len() {
        return f32::NAN;
    }
    let a = wps[i.saturating_sub(1)];
    let b = wps[i];
    let c = wps[i + 1];
    let v1 = (b - a).normalize_or_zero();
    let v2 = (c - b).normalize_or_zero();
    v1.x * v2.y - v1.y * v2.x
}

fn correlacao(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let cov: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let sa: f32 = a.iter().map(|x| (x - ma).powi(2)).sum::<f32>().sqrt();
    let sb: f32 = b.iter().map(|y| (y - mb).powi(2)).sum::<f32>().sqrt();
    if sa * sb < 1e-12 {
        0.0
    } else {
        cov / (sa * sb)
    }
}
