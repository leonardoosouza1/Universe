//! FULL OBSERVATORY — every detail of every being, against the real track.
//!
//! # Why it exists
//!
//! Leonardo: *"I want more data to observe, even for you. Put the models into groups,
//! selection, progress, which models use more throttle, which crashed the most and why.
//! Literally observe as much as you can to see every detail of everyone."*
//!
//! `replay_diag` looks at one being; `replay_pop` summarizes the population. This one goes
//! deep: measures 20 quantities per being, groups the beings by BEHAVIOUR (not by
//! performance), and tells where and how each group fails.
//!
//! Everything here is derived from recorded data or from the geometry reconstructed from
//! the `world_seed`. Nothing is estimated.
//!
//! # Usage
//!
//!   replay_obs --replay <file.replay> [--width 28] [--grupos 4] [--csv obs.csv]
//!   replay_obs --replay <a> --contra <b>     # compares two generations (what progressed)

use bevy::math::Vec2;
use std::collections::BTreeMap;
use std::path::PathBuf;
use universe::replay::{ReplayFrame, ReplayReader};
use universe::world::{VoxelType, VoxelWorld};

/// Full portrait of one being. Every field comes from recorded data or from geometry.
#[derive(Clone)]
struct Retrato {
    id: usize,
    // Result
    progresso: f32,
    setor_final: usize,
    voltas: u32,
    // Control usage
    thr_medio: f32,
    thr_cheio_pct: f32, // throttle > 0.9
    thr_zero_pct: f32,  // throttle < 0.1
    brk_pct: f32,       // brake applied
    str_medio: f32,     // WITH sign: steering bias
    str_desvio: f32,
    str_invers_s: f32, // steering reversals per second — jitter
    // Movement
    vel_media: f32,
    vel_max: f32,
    vel_alta_pct: f32, // above 10 m/s
    parado_pct: f32,
    re_pct: f32,
    // Failures
    batidas: usize, // sharp speed drops next to the wall
    fora_pct: f32,
    episodios_parado: usize,
    // Relation to the track
    corr_curva: f32,
    desvio_eixo: f32, // mean distance to the track centerline
    grupo: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path: Option<PathBuf> = None;
    let mut contra: Option<PathBuf> = None;
    let mut width = 18.0f32;
    let mut k_grupos = 4usize;
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
            "--contra" => {
                i += 1;
                if i < args.len() {
                    contra = Some(PathBuf::from(&args[i]));
                }
            }
            "--width" => {
                i += 1;
                if i < args.len() {
                    width = args[i].parse().unwrap_or(18.0);
                }
            }
            "--grupos" => {
                i += 1;
                if i < args.len() {
                    k_grupos = args[i].parse().unwrap_or(4);
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
        eprintln!(
            "usage: replay_obs --replay <file> [--width 28] [--grupos 4] [--csv obs.csv] [--contra other.replay]"
        );
        std::process::exit(2);
    };

    let (mut retratos, world, lap) = match carregar(&path, width) {
        Some(t) => t,
        None => {
            eprintln!("replay empty or unreadable: {}", path.display());
            std::process::exit(1);
        }
    };

    agrupar(&mut retratos, k_grupos);

    println!("=== OBSERVATORY: {} ===", path.display());
    println!(
        "{} beings · lap {lap:.0} m · track {width:.0} m\n",
        retratos.len()
    );

    tabela_individual(&retratos, lap);
    resumo_grupos(&retratos, lap, k_grupos);
    onde_param(&retratos, &world, lap);
    quem_bate(&retratos);
    quem_acelera(&retratos);

    if let Some(cp) = contra {
        if let Some((antes, _, _)) = carregar(&cp, width) {
            comparar_geracoes(&antes, &retratos, lap, &cp, &path);
        }
    }

    if let Some(cp) = csv {
        escrever_csv(&retratos, lap, &cp);
    }
}

// ── COLLECTION ────────────────────────────────────────────────────────────────

fn carregar(path: &PathBuf, width: f32) -> Option<(Vec<Retrato>, VoxelWorld, f32)> {
    let mut reader = ReplayReader::open(path).ok()?;
    let seed = reader.header.world_seed;
    let todos = reader.all_frames();
    let world = {
        let _q = universe::world::silence_map_diag();
        VoxelWorld::new(seed, width)
    };
    let lap = world.lap_distance;

    let mut out = Vec::new();
    for (id, fr) in todos.iter().enumerate() {
        if fr.is_empty() {
            continue;
        }
        out.push(retratar(id, fr, &world, lap, width));
    }
    if out.is_empty() {
        return None;
    }
    Some((out, world, lap))
}

fn retratar(id: usize, fr: &[ReplayFrame], world: &VoxelWorld, lap: f32, width: f32) -> Retrato {
    let n = fr.len() as f32;
    let (mut thr, mut str_s, mut vel) = (0.0f32, 0.0f32, 0.0f32);
    let (mut thr_cheio, mut thr_zero, mut brk, mut alta, mut parado, mut re, mut fora) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut vmax = 0.0f32;
    let mut str_v: Vec<f32> = Vec::with_capacity(fr.len());
    let mut curv_v: Vec<f32> = Vec::with_capacity(fr.len());
    // NET FORWARD progress, accumulated with line-crossing handling.
    //
    // The first version used `max - min` of visited positions, which is AMPLITUDE, not
    // progress: a being that backs up across the line the wrong way goes from 14 m to 2900 m
    // and scored "2914 m — 99.5% of the lap". Four beings appeared like that, and with them
    // the clustering elected as leader the group with throttle 0.04 and 80% stopped.
    let mut prog_liq = 0.0f32;
    let mut prog_ant = world.get_track_progress(fr[0].pos[0], fr[0].pos[2]);
    let mut prog_pico = 0.0f32;
    let mut inversoes = 0usize;
    let mut batidas = 0usize;
    let mut episodios = 0usize;
    let mut estava_parado = false;
    let mut desvio_soma = 0.0f32;
    let mut ultima_vel = fr[0].speed;

    for f in fr {
        let p = Vec2::new(f.pos[0], f.pos[2]);
        let mat = world.material_at(p);
        thr += f.final_thr;
        str_s += f.final_str;
        vel += f.speed;
        vmax = vmax.max(f.speed);
        if f.final_thr > 0.9 {
            thr_cheio += 1;
        }
        if f.final_thr < 0.1 {
            thr_zero += 1;
        }
        if f.final_brk > 0.1 {
            brk += 1;
        }
        if f.speed > 10.0 {
            alta += 1;
        }
        if f.speed < 0.5 {
            parado += 1;
            if !estava_parado {
                episodios += 1;
                estava_parado = true;
            }
        } else {
            estava_parado = false;
        }
        if f.gear == 0 {
            re += 1;
        }
        if !matches!(mat, VoxelType::Road) {
            fora += 1;
        }

        // Crash: sharp speed loss (>6 m/s between 1/16 s frames) — only possible
        // against geometry, not by braking. It is the honest proxy the recorded data
        // allows; there is no collision flag in the frame.
        if ultima_vel - f.speed > 6.0 {
            batidas += 1;
        }
        ultima_vel = f.speed;

        str_v.push(f.final_str);
        curv_v.push(curvatura(world, p, 25.0));
        let pr = world.get_track_progress(p.x, p.y);
        let mut d = pr - prog_ant;
        if d < -lap * 0.5 {
            d += lap;
        } else if d > lap * 0.5 {
            d -= lap;
        }
        prog_liq += d.clamp(-lap * 0.1, lap * 0.1);
        prog_pico = prog_pico.max(prog_liq);
        prog_ant = pr;
        desvio_soma += dist_eixo(world, p).min(width * 2.0);
    }

    for w in str_v.windows(2) {
        if w[0].signum() != w[1].signum() && w[0].abs() > 0.1 && w[1].abs() > 0.1 {
            inversoes += 1;
        }
    }

    let m = str_s / n;
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
        0.0
    };

    // Frames at 16 Hz (1 every 4 ticks of 64 Hz).
    let segundos = n / 16.0;
    // The PEAK of net progress: how far it actually got forward, no credit for backing up.
    let span = prog_pico.max(0.0);

    Retrato {
        id,
        progresso: span,
        setor_final: ((prog_pico.max(0.0) / lap) * 24.0) as usize,
        voltas: fr.last().map(|f| f.laps as u32).unwrap_or(0),
        thr_medio: thr / n,
        thr_cheio_pct: 100.0 * thr_cheio as f32 / n,
        thr_zero_pct: 100.0 * thr_zero as f32 / n,
        brk_pct: 100.0 * brk as f32 / n,
        str_medio: m,
        str_desvio: sd,
        str_invers_s: inversoes as f32 / segundos.max(1.0),
        vel_media: vel / n,
        vel_max: vmax,
        vel_alta_pct: 100.0 * alta as f32 / n,
        parado_pct: 100.0 * parado as f32 / n,
        re_pct: 100.0 * re as f32 / n,
        batidas,
        fora_pct: 100.0 * fora as f32 / n,
        episodios_parado: episodios,
        corr_curva: corr,
        desvio_eixo: desvio_soma / n,
        grupo: 0,
    }
}

// ── GROUPING BY BEHAVIOUR ─────────────────────────────────────────────────────

/// k-means over the driving STYLE, not the result.
///
/// Grouping by distance covered only reorders the ranking. Grouping by HOW the being drives
/// (throttle, steering, brake, jitter) reveals whether the population has distinct strategies
/// or is a monoculture with noise — which is the question `med/max = 0.97` raised.
fn agrupar(r: &mut [Retrato], k: usize) {
    if r.len() < k || k < 2 {
        return;
    }
    let feats = |x: &Retrato| {
        [
            x.thr_medio,
            x.str_medio,
            x.str_desvio,
            x.vel_media / 10.0,
            x.re_pct / 100.0,
            x.str_invers_s / 5.0,
        ]
    };
    const D: usize = 6;

    // Normalizes each dimension so the largest-scale one does not dominate the distance.
    let mut mn = [f32::MAX; D];
    let mut mx = [f32::MIN; D];
    for x in r.iter() {
        let f = feats(x);
        for d in 0..D {
            mn[d] = mn[d].min(f[d]);
            mx[d] = mx[d].max(f[d]);
        }
    }
    let norm = |x: &Retrato| {
        let f = feats(x);
        let mut o = [0.0f32; D];
        for d in 0..D {
            let r = (mx[d] - mn[d]).max(1e-6);
            o[d] = (f[d] - mn[d]) / r;
        }
        o
    };

    // Seeds spread by index — deterministic, no RNG.
    let mut centros: Vec<[f32; D]> = (0..k).map(|c| norm(&r[c * r.len() / k])).collect();
    for _ in 0..25 {
        let mut soma = vec![[0.0f32; D]; k];
        let mut cont = vec![0usize; k];
        for x in r.iter_mut() {
            let p = norm(x);
            let mut melhor = 0;
            let mut dm = f32::MAX;
            for (c, ce) in centros.iter().enumerate() {
                let d: f32 = (0..D).map(|i| (p[i] - ce[i]).powi(2)).sum();
                if d < dm {
                    dm = d;
                    melhor = c;
                }
            }
            x.grupo = melhor;
            for i in 0..D {
                soma[melhor][i] += p[i];
            }
            cont[melhor] += 1;
        }
        for c in 0..k {
            if cont[c] > 0 {
                for i in 0..D {
                    centros[c][i] = soma[c][i] / cont[c] as f32;
                }
            }
        }
    }
}

// ── REPORTS ───────────────────────────────────────────────────────────────────

fn tabela_individual(r: &[Retrato], lap: f32) {
    let mut v: Vec<&Retrato> = r.iter().collect();
    v.sort_by(|a, b| b.progresso.partial_cmp(&a.progresso).unwrap());
    println!("── EACH BEING (sorted by progress) ──");
    println!(
        "{:>4} {:>3} {:>8} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6} {:>6} {:>5} {:>6}",
        "being",
        "gr",
        "progr",
        "%lap",
        "thr",
        "thr100",
        "brake",
        "steer",
        "±",
        "v.avg",
        "stop",
        "crsh",
        "corr"
    );
    println!("{:-<95}", "");
    for x in v.iter().take(16) {
        println!(
            "{:>4} {:>3} {:>8.0} {:>5.1}% {:>6.2} {:>5.0}% {:>5.0}% {:>+7.2} {:>6.2} {:>6.2} {:>5.0}% {:>5} {:>+6.2}",
            x.id,
            x.grupo,
            x.progresso,
            100.0 * x.progresso / lap,
            x.thr_medio,
            x.thr_cheio_pct,
            x.brk_pct,
            x.str_medio,
            x.str_desvio,
            x.vel_media,
            x.parado_pct,
            x.batidas,
            x.corr_curva
        );
    }
    if v.len() > 16 {
        println!("{:>4} … and {} more", "", v.len() - 16);
    }
}

fn resumo_grupos(r: &[Retrato], lap: f32, k: usize) {
    println!("\n── BEHAVIOUR GROUPS ──");
    println!("(grouped by HOW they drive, not by how far they get)");
    println!(
        "{:>3} {:>4} {:>8} {:>6} {:>7} {:>6} {:>6} {:>6} {:>5} {:>6}",
        "gr", "n", "progr", "thr", "steer", "±", "v.avg", "stop", "crsh", "rev/s"
    );
    println!("{:-<80}", "");
    for g in 0..k {
        let m: Vec<&Retrato> = r.iter().filter(|x| x.grupo == g).collect();
        if m.is_empty() {
            continue;
        }
        let a = |f: &dyn Fn(&Retrato) -> f32| m.iter().map(|x| f(x)).sum::<f32>() / m.len() as f32;
        println!(
            "{:>3} {:>4} {:>8.0} {:>6.2} {:>+7.2} {:>6.2} {:>6.2} {:>5.0}% {:>5.1} {:>6.2}",
            g,
            m.len(),
            a(&|x| x.progresso),
            a(&|x| x.thr_medio),
            a(&|x| x.str_medio),
            a(&|x| x.str_desvio),
            a(&|x| x.vel_media),
            a(&|x| x.parado_pct),
            m.iter().map(|x| x.batidas).sum::<usize>() as f32 / m.len() as f32,
            a(&|x| x.str_invers_s)
        );
    }
    // The group that progresses most is the one selection will copy — worth knowing what it
    // does differently, because that is what the whole next generation will imitate.
    let melhor = (0..k)
        .filter(|g| r.iter().any(|x| x.grupo == *g))
        .max_by(|a, b| {
            let ma = r
                .iter()
                .filter(|x| x.grupo == *a)
                .map(|x| x.progresso)
                .sum::<f32>();
            let mb = r
                .iter()
                .filter(|x| x.grupo == *b)
                .map(|x| x.progresso)
                .sum::<f32>();
            ma.partial_cmp(&mb).unwrap()
        });
    if let Some(g) = melhor {
        let m: Vec<&Retrato> = r.iter().filter(|x| x.grupo == g).collect();
        let a = |f: &dyn Fn(&Retrato) -> f32| m.iter().map(|x| f(x)).sum::<f32>() / m.len() as f32;
        println!(
            "\n  Leading group: {} ({} beings, {:.0} m mean)",
            g,
            m.len(),
            a(&|x| x.progresso)
        );
        println!(
            "  Style: throttle {:.2} · steering {:+.2} · {:.1} reversals/s · stopped {:.0}%",
            a(&|x| x.thr_medio),
            a(&|x| x.str_medio),
            a(&|x| x.str_invers_s),
            a(&|x| x.parado_pct)
        );
        println!("  The next generation will be a copy of this — this is the style selection rewards.");
    }
    let _ = lap;
}

fn onde_param(r: &[Retrato], world: &VoxelWorld, lap: f32) {
    let mut h: BTreeMap<usize, usize> = BTreeMap::new();
    for x in r {
        *h.entry(x.setor_final).or_insert(0) += 1;
    }
    println!("\n── WHERE EACH ONE STOPPED (24 sectors) ──");
    let total = r.len();
    for (s, n) in &h {
        let barra: String = std::iter::repeat('█')
            .take((*n * 40 / total).max(1))
            .collect();
        println!(
            "  sector {:>2} (~{:>5.0} m) {:>3}  {}",
            s,
            *s as f32 * lap / 24.0,
            n,
            barra
        );
    }
    if let Some((sm, nm)) = h.iter().max_by_key(|(_, n)| **n).map(|(s, n)| (*s, *n)) {
        let frac = 100.0 * nm as f32 / total as f32;
        println!(
            "\n  {:.0}% stop in sector {} (~{:.0} m).",
            frac,
            sm,
            sm as f32 * lap / 24.0
        );
        if frac > 60.0 {
            println!("  ⇒ the SAME point for almost everyone: a WORLD barrier, not competence.");
            descrever(world, sm as f32 * lap / 24.0);
        } else {
            println!("  ⇒ spread out: the difference is COMPETENCE between individuals.");
        }
    }
}

fn quem_bate(r: &[Retrato]) {
    let mut v: Vec<&Retrato> = r.iter().filter(|x| x.batidas > 0).collect();
    v.sort_by_key(|x| std::cmp::Reverse(x.batidas));
    println!("\n── WHO CRASHES MOST (speed drop > 6 m/s between frames) ──");
    if v.is_empty() {
        println!("  nobody: no sharp speed loss.");
        return;
    }
    for x in v.iter().take(6) {
        println!(
            "  being {:>3}: {:>3} crashes · max spd {:>5.1} · steering {:+.2}(±{:.2}) · off track {:.0}% · progress {:.0} m",
            x.id, x.batidas, x.vel_max, x.str_medio, x.str_desvio, x.fora_pct, x.progresso
        );
    }
    // Does crashing correlate with racing, or with driving badly? The answer changes what to do.
    let vel: Vec<f32> = r.iter().map(|x| x.vel_media).collect();
    let bat: Vec<f32> = r.iter().map(|x| x.batidas as f32).collect();
    println!(
        "\n  correlation crashes × mean speed: {:+.2}",
        correlacao(&vel, &bat)
    );
    let prog: Vec<f32> = r.iter().map(|x| x.progresso).collect();
    println!(
        "  correlation crashes × progress:         {:+.2}",
        correlacao(&prog, &bat)
    );
    println!("  (positive with progress = who tries more, crashes more — crashing is the cost of trying)");
}

fn quem_acelera(r: &[Retrato]) {
    let mut v: Vec<&Retrato> = r.iter().collect();
    v.sort_by(|a, b| b.thr_medio.partial_cmp(&a.thr_medio).unwrap());
    println!("\n── THROTTLE ──");
    println!(
        "  heaviest foot: being {} ({:.2} mean, {:.0}% floored, {:.0} m)",
        v[0].id, v[0].thr_medio, v[0].thr_cheio_pct, v[0].progresso
    );
    let u = v[v.len() - 1];
    println!(
        "  lightest foot: being {} ({:.2} mean, {:.0}% released, {:.0} m)",
        u.id, u.thr_medio, u.thr_zero_pct, u.progresso
    );
    let thr: Vec<f32> = r.iter().map(|x| x.thr_medio).collect();
    let prog: Vec<f32> = r.iter().map(|x| x.progresso).collect();
    let vel: Vec<f32> = r.iter().map(|x| x.vel_media).collect();
    println!(
        "  correlation throttle × progress:  {:+.2}",
        correlacao(&thr, &prog)
    );
    println!(
        "  correlation throttle × speed: {:+.2}",
        correlacao(&thr, &vel)
    );
    println!("  (high throttle without high speed = pedal floored against something)");
}

fn comparar_geracoes(antes: &[Retrato], agora: &[Retrato], lap: f32, pa: &PathBuf, pb: &PathBuf) {
    println!("\n── PROGRESS BETWEEN GENERATIONS ──");
    println!("  {}  →  {}", pa.display(), pb.display());
    let m = |v: &[Retrato], f: &dyn Fn(&Retrato) -> f32| {
        let mut a: Vec<f32> = v.iter().map(f).collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        a[a.len() / 2]
    };
    let linhas: [(&str, &dyn Fn(&Retrato) -> f32); 6] = [
        ("progress (m)", &|x: &Retrato| x.progresso),
        ("speed", &|x: &Retrato| x.vel_media),
        ("throttle", &|x: &Retrato| x.thr_medio),
        ("mean steering", &|x: &Retrato| x.str_medio),
        ("stopped %", &|x: &Retrato| x.parado_pct),
        ("crashes", &|x: &Retrato| x.batidas as f32),
    ];
    println!(
        "  {:<16} {:>9} {:>9} {:>9}",
        "metric", "before", "now", "Δ"
    );
    for (nome, f) in linhas {
        let a = m(antes, f);
        let b = m(agora, f);
        println!("  {:<16} {:>9.2} {:>9.2} {:>+9.2}", nome, a, b, b - a);
    }
    let _ = lap;
}

fn escrever_csv(r: &[Retrato], lap: f32, cp: &PathBuf) {
    let mut s = String::from(
        "ser,grupo,progresso_m,pct_volta,setor,voltas,thr_medio,thr_cheio_pct,thr_zero_pct,\
brk_pct,str_medio,str_desvio,str_inversoes_s,vel_media,vel_max,vel_alta_pct,parado_pct,re_pct,\
batidas,fora_pct,episodios_parado,corr_curva,desvio_eixo\n",
    );
    for x in r {
        s.push_str(&format!(
            "{},{},{:.1},{:.2},{},{},{:.3},{:.1},{:.1},{:.1},{:.3},{:.3},{:.2},\
{:.2},{:.2},{:.1},{:.1},{:.1},{},{:.1},{},{:.3},{:.2}\n",
            x.id,
            x.grupo,
            x.progresso,
            100.0 * x.progresso / lap,
            x.setor_final,
            x.voltas,
            x.thr_medio,
            x.thr_cheio_pct,
            x.thr_zero_pct,
            x.brk_pct,
            x.str_medio,
            x.str_desvio,
            x.str_invers_s,
            x.vel_media,
            x.vel_max,
            x.vel_alta_pct,
            x.parado_pct,
            x.re_pct,
            x.batidas,
            x.fora_pct,
            x.episodios_parado,
            x.corr_curva,
            x.desvio_eixo
        ));
    }
    match std::fs::write(cp, s) {
        Ok(_) => println!("\nCSV with 23 columns per being: {}", cp.display()),
        Err(e) => eprintln!("\nfailed to write CSV: {e}"),
    }
}

// ── GEOMETRY ──────────────────────────────────────────────────────────────────

fn dist_eixo(world: &VoxelWorld, p: Vec2) -> f32 {
    world
        .waypoints_world
        .iter()
        .map(|w| w.distance(p))
        .fold(f32::MAX, f32::min)
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
        "right"
    } else {
        "left"
    };
    println!("  Geometry there: {ang:.0}° to the {lado} (waypoint {idx}).");
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
    if n < 3.0 {
        return 0.0;
    }
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
