//! OBSERVATÓRIO COMPLETO — cada detalhe de cada ser, contra a pista real.
//!
//! # Por que existe
//!
//! Leonardo: *"quero mais dados para observar, até para você. Colocar os modelos em grupos,
//! seleção, avanços, quais modelos usam mais acelerador, qual bateu mais e por quê.
//! Literalmente observar o máximo que puder para ver cada detalhe de todo mundo."*
//!
//! O `replay_diag` olha um ser; o `replay_pop` resume a população. Este vai ao fundo: mede
//! 20 grandezas por ser, agrupa os seres por COMPORTAMENTO (não por desempenho), e conta
//! onde e como cada grupo falha.
//!
//! Tudo aqui é derivado de dado gravado ou da geometria reconstruída da `world_seed`.
//! Nada é estimado.
//!
//! # Uso
//!
//!   replay_obs --replay <arquivo.replay> [--width 28] [--grupos 4] [--csv obs.csv]
//!   replay_obs --replay <a> --contra <b>     # compara duas gerações (o que avançou)

use std::collections::BTreeMap;
use std::path::PathBuf;
use universe::replay::{ReplayReader, ReplayFrame};
use universe::world::{VoxelWorld, VoxelType};
use bevy::math::Vec2;

/// Retrato completo de um ser. Cada campo sai de dado gravado ou de geometria.
#[derive(Clone)]
struct Retrato {
    id: usize,
    // Resultado
    progresso: f32,
    setor_final: usize,
    voltas: u32,
    // Uso dos controles
    thr_medio: f32,
    thr_cheio_pct: f32,   // acelerador > 0,9
    thr_zero_pct: f32,    // acelerador < 0,1
    brk_pct: f32,         // freio acionado
    str_medio: f32,       // COM sinal: viés de direção
    str_desvio: f32,
    str_invers_s: f32,    // inversões de volante por segundo — jitter
    // Movimento
    vel_media: f32,
    vel_max: f32,
    vel_alta_pct: f32,    // acima de 10 m/s
    parado_pct: f32,
    re_pct: f32,
    // Falhas
    batidas: usize,       // quedas bruscas de velocidade junto à parede
    fora_pct: f32,
    episodios_parado: usize,
    // Relação com a pista
    corr_curva: f32,
    desvio_eixo: f32,     // distância média ao eixo da pista
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
            "--replay" => { i += 1; if i < args.len() { path = Some(PathBuf::from(&args[i])); } }
            "--contra" => { i += 1; if i < args.len() { contra = Some(PathBuf::from(&args[i])); } }
            "--width"  => { i += 1; if i < args.len() { width = args[i].parse().unwrap_or(18.0); } }
            "--grupos" => { i += 1; if i < args.len() { k_grupos = args[i].parse().unwrap_or(4); } }
            "--csv"    => { i += 1; if i < args.len() { csv = Some(PathBuf::from(&args[i])); } }
            _ => {}
        }
        i += 1;
    }
    let Some(path) = path else {
        eprintln!("uso: replay_obs --replay <arquivo> [--width 28] [--grupos 4] [--csv obs.csv] [--contra outro.replay]");
        std::process::exit(2);
    };

    let (mut retratos, world, lap) = match carregar(&path, width) {
        Some(t) => t,
        None => { eprintln!("replay vazio ou ilegível: {}", path.display()); std::process::exit(1); }
    };

    agrupar(&mut retratos, k_grupos);

    println!("=== OBSERVATÓRIO: {} ===", path.display());
    println!("{} seres · volta {lap:.0} m · pista {width:.0} m\n", retratos.len());

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

    if let Some(cp) = csv { escrever_csv(&retratos, lap, &cp); }
}

// ── COLETA ────────────────────────────────────────────────────────────────────

fn carregar(path: &PathBuf, width: f32) -> Option<(Vec<Retrato>, VoxelWorld, f32)> {
    let mut reader = ReplayReader::open(path).ok()?;
    let seed = reader.header.world_seed;
    let todos = reader.all_frames();
    let world = { let _q = universe::world::silence_map_diag(); VoxelWorld::new(seed, width) };
    let lap = world.lap_distance;

    let mut out = Vec::new();
    for (id, fr) in todos.iter().enumerate() {
        if fr.is_empty() { continue; }
        out.push(retratar(id, fr, &world, lap, width));
    }
    if out.is_empty() { return None; }
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
    // Progresso LÍQUIDO PARA FRENTE, acumulado com tratamento do cruzamento de linha.
    //
    // A primeira versão usava `max - min` das posições visitadas, o que é AMPLITUDE, não
    // avanço: um ser que recua cruzando a linha ao contrário vai de 14 m para 2900 m e
    // marcava "2914 m — 99,5% da volta". Quatro seres apareceram assim, e com eles o
    // agrupamento elegeu como líder o grupo de acelerador 0,04 e 80% parado.
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
        thr += f.final_thr; str_s += f.final_str; vel += f.speed;
        vmax = vmax.max(f.speed);
        if f.final_thr > 0.9 { thr_cheio += 1; }
        if f.final_thr < 0.1 { thr_zero += 1; }
        if f.final_brk > 0.1 { brk += 1; }
        if f.speed > 10.0 { alta += 1; }
        if f.speed < 0.5 { parado += 1; if !estava_parado { episodios += 1; estava_parado = true; } }
        else { estava_parado = false; }
        if f.gear == 0 { re += 1; }
        if !matches!(mat, VoxelType::Road) { fora += 1; }

        // Batida: perda brusca de velocidade (>6 m/s entre quadros de 1/16 s) — só é
        // possível contra geometria, não por frenagem. É o proxy honesto que o dado
        // gravado permite; não há flag de colisão no quadro.
        if ultima_vel - f.speed > 6.0 { batidas += 1; }
        ultima_vel = f.speed;

        str_v.push(f.final_str);
        curv_v.push(curvatura(world, p, 25.0));
        let pr = world.get_track_progress(p.x, p.y);
        let mut d = pr - prog_ant;
        if d < -lap * 0.5 { d += lap; } else if d > lap * 0.5 { d -= lap; }
        prog_liq += d.clamp(-lap * 0.1, lap * 0.1);
        prog_pico = prog_pico.max(prog_liq);
        prog_ant = pr;
        desvio_soma += dist_eixo(world, p).min(width * 2.0);
    }

    for w in str_v.windows(2) {
        if w[0].signum() != w[1].signum() && w[0].abs() > 0.1 && w[1].abs() > 0.1 { inversoes += 1; }
    }

    let m = str_s / n;
    let sd = (str_v.iter().map(|v| (v - m).powi(2)).sum::<f32>() / n).sqrt();
    let pares: Vec<(f32, f32)> = curv_v.iter().zip(&str_v)
        .filter(|(c, _)| c.is_finite()).map(|(c, s)| (*c, *s)).collect();
    let corr = if pares.len() > 30 {
        let (a, b): (Vec<f32>, Vec<f32>) = pares.into_iter().unzip();
        correlacao(&a, &b)
    } else { 0.0 };

    // Quadros a 16 Hz (1 a cada 4 ticks de 64 Hz).
    let segundos = n / 16.0;
    // O PICO do avanço líquido: o quanto ele chegou a andar para frente, sem crédito por recuar.
    let span = prog_pico.max(0.0);

    Retrato {
        id, progresso: span,
        setor_final: ((prog_pico.max(0.0) / lap) * 24.0) as usize,
        voltas: fr.last().map(|f| f.laps as u32).unwrap_or(0),
        thr_medio: thr / n,
        thr_cheio_pct: 100.0 * thr_cheio as f32 / n,
        thr_zero_pct: 100.0 * thr_zero as f32 / n,
        brk_pct: 100.0 * brk as f32 / n,
        str_medio: m, str_desvio: sd,
        str_invers_s: inversoes as f32 / segundos.max(1.0),
        vel_media: vel / n, vel_max: vmax,
        vel_alta_pct: 100.0 * alta as f32 / n,
        parado_pct: 100.0 * parado as f32 / n,
        re_pct: 100.0 * re as f32 / n,
        batidas, fora_pct: 100.0 * fora as f32 / n,
        episodios_parado: episodios,
        corr_curva: corr,
        desvio_eixo: desvio_soma / n,
        grupo: 0,
    }
}

// ── AGRUPAMENTO POR COMPORTAMENTO ─────────────────────────────────────────────

/// k-médias sobre o ESTILO de pilotagem, não sobre o resultado.
///
/// Agrupar por distância percorrida só reordena o ranking. Agrupar por COMO o ser dirige
/// (acelerador, volante, freio, jitter) revela se a população tem estratégias distintas ou
/// se é uma monocultura com ruído — que é a pergunta que o `med/max = 0,97` levantou.
fn agrupar(r: &mut [Retrato], k: usize) {
    if r.len() < k || k < 2 { return; }
    let feats = |x: &Retrato| [x.thr_medio, x.str_medio, x.str_desvio,
                               x.vel_media / 10.0, x.re_pct / 100.0, x.str_invers_s / 5.0];
    const D: usize = 6;

    // Normaliza cada dimensão para não deixar a de maior escala dominar a distância.
    let mut mn = [f32::MAX; D]; let mut mx = [f32::MIN; D];
    for x in r.iter() {
        let f = feats(x);
        for d in 0..D { mn[d] = mn[d].min(f[d]); mx[d] = mx[d].max(f[d]); }
    }
    let norm = |x: &Retrato| {
        let f = feats(x);
        let mut o = [0.0f32; D];
        for d in 0..D { let r = (mx[d] - mn[d]).max(1e-6); o[d] = (f[d] - mn[d]) / r; }
        o
    };

    // Sementes espalhadas pelo índice — determinístico, sem RNG.
    let mut centros: Vec<[f32; D]> = (0..k).map(|c| norm(&r[c * r.len() / k])).collect();
    for _ in 0..25 {
        let mut soma = vec![[0.0f32; D]; k];
        let mut cont = vec![0usize; k];
        for x in r.iter_mut() {
            let p = norm(x);
            let mut melhor = 0; let mut dm = f32::MAX;
            for (c, ce) in centros.iter().enumerate() {
                let d: f32 = (0..D).map(|i| (p[i] - ce[i]).powi(2)).sum();
                if d < dm { dm = d; melhor = c; }
            }
            x.grupo = melhor;
            for i in 0..D { soma[melhor][i] += p[i]; }
            cont[melhor] += 1;
        }
        for c in 0..k {
            if cont[c] > 0 { for i in 0..D { centros[c][i] = soma[c][i] / cont[c] as f32; } }
        }
    }
}

// ── RELATÓRIOS ────────────────────────────────────────────────────────────────

fn tabela_individual(r: &[Retrato], lap: f32) {
    let mut v: Vec<&Retrato> = r.iter().collect();
    v.sort_by(|a, b| b.progresso.partial_cmp(&a.progresso).unwrap());
    println!("── CADA SER (ordenado por progresso) ──");
    println!("{:>4} {:>3} {:>8} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6} {:>6} {:>5} {:>6}",
             "ser","gr","progr","%vlt","thr","thr100","freio","volante","±","v.méd","parado","bat","corr");
    println!("{:-<95}", "");
    for x in v.iter().take(16) {
        println!("{:>4} {:>3} {:>8.0} {:>5.1}% {:>6.2} {:>5.0}% {:>5.0}% {:>+7.2} {:>6.2} {:>6.2} {:>5.0}% {:>5} {:>+6.2}",
                 x.id, x.grupo, x.progresso, 100.0*x.progresso/lap, x.thr_medio,
                 x.thr_cheio_pct, x.brk_pct, x.str_medio, x.str_desvio,
                 x.vel_media, x.parado_pct, x.batidas, x.corr_curva);
    }
    if v.len() > 16 { println!("{:>4} … e mais {}", "", v.len()-16); }
}

fn resumo_grupos(r: &[Retrato], lap: f32, k: usize) {
    println!("\n── GRUPOS DE COMPORTAMENTO ──");
    println!("(agrupados por COMO dirigem, não por quanto avançam)");
    println!("{:>3} {:>4} {:>8} {:>6} {:>7} {:>6} {:>6} {:>6} {:>5} {:>6}",
             "gr","n","progr","thr","volante","±","v.méd","parado","bat","invers/s");
    println!("{:-<80}", "");
    for g in 0..k {
        let m: Vec<&Retrato> = r.iter().filter(|x| x.grupo == g).collect();
        if m.is_empty() { continue; }
        let a = |f: &dyn Fn(&Retrato) -> f32| m.iter().map(|x| f(x)).sum::<f32>() / m.len() as f32;
        println!("{:>3} {:>4} {:>8.0} {:>6.2} {:>+7.2} {:>6.2} {:>6.2} {:>5.0}% {:>5.1} {:>6.2}",
                 g, m.len(), a(&|x| x.progresso), a(&|x| x.thr_medio), a(&|x| x.str_medio),
                 a(&|x| x.str_desvio), a(&|x| x.vel_media), a(&|x| x.parado_pct),
                 m.iter().map(|x| x.batidas).sum::<usize>() as f32 / m.len() as f32,
                 a(&|x| x.str_invers_s));
    }
    // O grupo que mais avança é o que a seleção vai copiar — vale saber o que ele faz
    // de diferente, porque é isso que a próxima geração inteira vai imitar.
    let melhor = (0..k).filter(|g| r.iter().any(|x| x.grupo == *g))
        .max_by(|a, b| {
            let ma = r.iter().filter(|x| x.grupo == *a).map(|x| x.progresso).sum::<f32>();
            let mb = r.iter().filter(|x| x.grupo == *b).map(|x| x.progresso).sum::<f32>();
            ma.partial_cmp(&mb).unwrap()
        });
    if let Some(g) = melhor {
        let m: Vec<&Retrato> = r.iter().filter(|x| x.grupo == g).collect();
        let a = |f: &dyn Fn(&Retrato) -> f32| m.iter().map(|x| f(x)).sum::<f32>() / m.len() as f32;
        println!("\n  Grupo líder: {} ({} seres, {:.0} m médios)", g, m.len(), a(&|x| x.progresso));
        println!("  Estilo: acelerador {:.2} · volante {:+.2} · {:.1} inversões/s · parado {:.0}%",
                 a(&|x| x.thr_medio), a(&|x| x.str_medio), a(&|x| x.str_invers_s), a(&|x| x.parado_pct));
        println!("  A próxima geração vai ser cópia disto — é este estilo que a seleção premia.");
    }
    let _ = lap;
}

fn onde_param(r: &[Retrato], world: &VoxelWorld, lap: f32) {
    let mut h: BTreeMap<usize, usize> = BTreeMap::new();
    for x in r { *h.entry(x.setor_final).or_insert(0) += 1; }
    println!("\n── ONDE CADA UM PAROU (24 setores) ──");
    let total = r.len();
    for (s, n) in &h {
        let barra: String = std::iter::repeat('█').take((*n * 40 / total).max(1)).collect();
        println!("  setor {:>2} (~{:>5.0} m) {:>3}  {}", s, *s as f32 * lap / 24.0, n, barra);
    }
    if let Some((sm, nm)) = h.iter().max_by_key(|(_, n)| **n).map(|(s, n)| (*s, *n)) {
        let frac = 100.0 * nm as f32 / total as f32;
        println!("\n  {:.0}% para no setor {} (~{:.0} m).", frac, sm, sm as f32 * lap / 24.0);
        if frac > 60.0 {
            println!("  ⇒ MESMO ponto para quase todos: barreira do MUNDO, não competência.");
            descrever(world, sm as f32 * lap / 24.0);
        } else {
            println!("  ⇒ espalhado: a diferença é de COMPETÊNCIA entre indivíduos.");
        }
    }
}

fn quem_bate(r: &[Retrato]) {
    let mut v: Vec<&Retrato> = r.iter().filter(|x| x.batidas > 0).collect();
    v.sort_by_key(|x| std::cmp::Reverse(x.batidas));
    println!("\n── QUEM MAIS BATE (queda de velocidade > 6 m/s entre quadros) ──");
    if v.is_empty() { println!("  ninguém: nenhuma perda brusca de velocidade."); return; }
    for x in v.iter().take(6) {
        println!("  ser {:>3}: {:>3} batidas · vel máx {:>5.1} · volante {:+.2}(±{:.2}) · fora da pista {:.0}% · progresso {:.0} m",
                 x.id, x.batidas, x.vel_max, x.str_medio, x.str_desvio, x.fora_pct, x.progresso);
    }
    // Bater se correlaciona com correr, ou com dirigir mal? A resposta muda o que fazer.
    let vel: Vec<f32> = r.iter().map(|x| x.vel_media).collect();
    let bat: Vec<f32> = r.iter().map(|x| x.batidas as f32).collect();
    println!("\n  correlação batidas × velocidade média: {:+.2}", correlacao(&vel, &bat));
    let prog: Vec<f32> = r.iter().map(|x| x.progresso).collect();
    println!("  correlação batidas × progresso:         {:+.2}", correlacao(&prog, &bat));
    println!("  (positiva com progresso = quem tenta mais, bate mais — bater é custo de tentar)");
}

fn quem_acelera(r: &[Retrato]) {
    let mut v: Vec<&Retrato> = r.iter().collect();
    v.sort_by(|a, b| b.thr_medio.partial_cmp(&a.thr_medio).unwrap());
    println!("\n── ACELERADOR ──");
    println!("  mais fundo: ser {} ({:.2} médio, {:.0}% no fundo, {:.0} m)",
             v[0].id, v[0].thr_medio, v[0].thr_cheio_pct, v[0].progresso);
    let u = v[v.len()-1];
    println!("  mais leve:  ser {} ({:.2} médio, {:.0}% solto, {:.0} m)",
             u.id, u.thr_medio, u.thr_zero_pct, u.progresso);
    let thr: Vec<f32> = r.iter().map(|x| x.thr_medio).collect();
    let prog: Vec<f32> = r.iter().map(|x| x.progresso).collect();
    let vel: Vec<f32> = r.iter().map(|x| x.vel_media).collect();
    println!("  correlação acelerador × progresso:  {:+.2}", correlacao(&thr, &prog));
    println!("  correlação acelerador × velocidade: {:+.2}", correlacao(&thr, &vel));
    println!("  (acelerador alto sem velocidade alta = pedal no fundo contra alguma coisa)");
}

fn comparar_geracoes(antes: &[Retrato], agora: &[Retrato], lap: f32, pa: &PathBuf, pb: &PathBuf) {
    println!("\n── AVANÇO ENTRE GERAÇÕES ──");
    println!("  {}  →  {}", pa.display(), pb.display());
    let m = |v: &[Retrato], f: &dyn Fn(&Retrato) -> f32| {
        let mut a: Vec<f32> = v.iter().map(f).collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        a[a.len()/2]
    };
    let linhas: [(&str, &dyn Fn(&Retrato) -> f32); 6] = [
        ("progresso (m)",  &|x: &Retrato| x.progresso),
        ("velocidade",     &|x: &Retrato| x.vel_media),
        ("acelerador",     &|x: &Retrato| x.thr_medio),
        ("volante médio",  &|x: &Retrato| x.str_medio),
        ("parado %",       &|x: &Retrato| x.parado_pct),
        ("batidas",        &|x: &Retrato| x.batidas as f32),
    ];
    println!("  {:<16} {:>9} {:>9} {:>9}", "métrica", "antes", "agora", "Δ");
    for (nome, f) in linhas {
        let a = m(antes, f); let b = m(agora, f);
        println!("  {:<16} {:>9.2} {:>9.2} {:>+9.2}", nome, a, b, b - a);
    }
    let _ = lap;
}

fn escrever_csv(r: &[Retrato], lap: f32, cp: &PathBuf) {
    let mut s = String::from("ser,grupo,progresso_m,pct_volta,setor,voltas,thr_medio,thr_cheio_pct,thr_zero_pct,\
brk_pct,str_medio,str_desvio,str_inversoes_s,vel_media,vel_max,vel_alta_pct,parado_pct,re_pct,\
batidas,fora_pct,episodios_parado,corr_curva,desvio_eixo\n");
    for x in r {
        s.push_str(&format!("{},{},{:.1},{:.2},{},{},{:.3},{:.1},{:.1},{:.1},{:.3},{:.3},{:.2},\
{:.2},{:.2},{:.1},{:.1},{:.1},{},{:.1},{},{:.3},{:.2}\n",
            x.id, x.grupo, x.progresso, 100.0*x.progresso/lap, x.setor_final, x.voltas,
            x.thr_medio, x.thr_cheio_pct, x.thr_zero_pct, x.brk_pct, x.str_medio, x.str_desvio,
            x.str_invers_s, x.vel_media, x.vel_max, x.vel_alta_pct, x.parado_pct, x.re_pct,
            x.batidas, x.fora_pct, x.episodios_parado, x.corr_curva, x.desvio_eixo));
    }
    match std::fs::write(cp, s) {
        Ok(_) => println!("\nCSV com 23 colunas por ser: {}", cp.display()),
        Err(e) => eprintln!("\nfalhou ao escrever CSV: {e}"),
    }
}

// ── GEOMETRIA ─────────────────────────────────────────────────────────────────

fn dist_eixo(world: &VoxelWorld, p: Vec2) -> f32 {
    world.waypoints_world.iter().map(|w| w.distance(p)).fold(f32::MAX, f32::min)
}

fn descrever(world: &VoxelWorld, at: f32) {
    let wps = &world.waypoints_world;
    if wps.len() < 3 { return; }
    let mut acc = 0.0f32; let mut idx = 0usize;
    for i in 0..wps.len()-1 {
        let seg = wps[i].distance(wps[i+1]);
        if acc + seg >= at { idx = i; break; }
        acc += seg;
    }
    let a = wps[idx.saturating_sub(1)]; let b = wps[idx]; let c = wps[(idx+1).min(wps.len()-1)];
    let v1 = (b-a).normalize_or_zero(); let v2 = (c-b).normalize_or_zero();
    let ang = v1.dot(v2).clamp(-1.0,1.0).acos().to_degrees();
    let lado = if v1.x*v2.y - v1.y*v2.x > 0.0 { "direita" } else { "esquerda" };
    println!("  Geometria ali: {ang:.0}° para a {lado} (waypoint {idx}).");
}

fn curvatura(world: &VoxelWorld, p: Vec2, ahead: f32) -> f32 {
    let wps = &world.waypoints_world;
    if wps.len() < 4 { return f32::NAN; }
    let mut i0 = 0usize; let mut best = f32::MAX;
    for (i, w) in wps.iter().enumerate() {
        let d = w.distance(p);
        if d < best { best = d; i0 = i; }
    }
    let mut acc = 0.0; let mut i = i0;
    while acc < ahead && i+1 < wps.len() { acc += wps[i].distance(wps[i+1]); i += 1; }
    if i+1 >= wps.len() { return f32::NAN; }
    let a = wps[i.saturating_sub(1)]; let b = wps[i]; let c = wps[i+1];
    let v1 = (b-a).normalize_or_zero(); let v2 = (c-b).normalize_or_zero();
    v1.x*v2.y - v1.y*v2.x
}

fn correlacao(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    if n < 3.0 { return 0.0; }
    let ma = a.iter().sum::<f32>()/n; let mb = b.iter().sum::<f32>()/n;
    let cov: f32 = a.iter().zip(b).map(|(x,y)| (x-ma)*(y-mb)).sum();
    let sa: f32 = a.iter().map(|x| (x-ma).powi(2)).sum::<f32>().sqrt();
    let sb: f32 = b.iter().map(|y| (y-mb).powi(2)).sum::<f32>().sqrt();
    if sa*sb < 1e-12 { 0.0 } else { cov/(sa*sb) }
}
