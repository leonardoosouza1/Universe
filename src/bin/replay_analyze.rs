//! Analisador de replays quadro a quadro.
//!
//! # Uso
//!
//!   # Perfil de velocidade + eventos críticos de UM replay:
//!   ./replay_analyze --replay champion_motor.replay
//!
//!   # Diff entre dois caminhos (motor.rs vs cortex_evolve):
//!   ./replay_analyze --replay-a champion_motor.replay --replay-b champion_cortex.replay
//!
//!   # Gerar HTML player (pista + player interativo):
//!   ./replay_analyze --replay champion_motor.replay --html replay.html
//!
//!   # Mostrar todos os eventos de um campeão (being_idx=0):
//!   ./replay_analyze --replay champion_motor.replay --events
//!
//! # Saídas
//!
//! Sem --html: texto puro (Markdown-ish) para análise direta no terminal.
//! Com --html: arquivo HTML autocontido que Leonardo abre no browser.

use std::path::PathBuf;
use universe::replay::{
    ReplayReader, ReplayWriter,
    section_profile, detect_events, diff_replays,
    generate_html_player,
    SOURCE_MOTOR, SOURCE_CORTEX,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // --- Parse simples de args ---
    let mut replay_path:   Option<PathBuf> = None;
    let mut replay_a:      Option<PathBuf> = None;
    let mut replay_b:      Option<PathBuf> = None;
    let mut html_output:   Option<PathBuf> = None;
    // Largura da pista, só para desenho — não muda o traçado dos waypoints.
    let mut track_width: f32 = 18.0;
    let mut show_events = false;
    let mut n_sections:    usize = 20;
    let mut being_idx:     u8 = 0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--replay"     => { i += 1; replay_path = Some(PathBuf::from(&args[i])); }
            "--replay-a"   => { i += 1; replay_a    = Some(PathBuf::from(&args[i])); }
            "--replay-b"   => { i += 1; replay_b    = Some(PathBuf::from(&args[i])); }
            "--html"       => { i += 1; html_output  = Some(PathBuf::from(&args[i])); }
            "--width"      => { i += 1; if i < args.len() { track_width = args[i].parse().unwrap_or(18.0); } }
            "--events"     => { show_events = true; }
            "--sections"   => { i += 1; n_sections  = args[i].parse().unwrap_or(20); }
            "--being"      => { i += 1; being_idx   = args[i].parse().unwrap_or(0); }
            _ => {}
        }
        i += 1;
    }

    // --- Modo diff A vs B ---
    if let (Some(pa), Some(pb)) = (replay_a, replay_b) {
        run_diff(&pa, &pb, html_output.as_deref(), n_sections);
        return;
    }

    // --- Modo single replay ---
    let path = match replay_path {
        Some(p) => p,
        None => {
            eprintln!("USO: replay_analyze --replay <arquivo.replay> [--html saida.html] [--events]");
            eprintln!("     replay_analyze --replay-a <A.replay> --replay-b <B.replay> [--html diff.html]");
            std::process::exit(1);
        }
    };

    run_single(&path, being_idx, show_events, n_sections, html_output.as_deref(), track_width);
}

// ── Análise de replay único ──────────────────────────────────────────────────

fn run_single(path: &std::path::Path, being_idx: u8, show_events: bool, n_sections: usize, html_out: Option<&std::path::Path>, track_width: f32) {
    let mut reader = match ReplayReader::open(path) {
        Ok(r) => r,
        Err(e) => { eprintln!("ERRO ao abrir {}: {}", path.display(), e); std::process::exit(1); }
    };

    let hdr = reader.header;
    let source_name = if hdr.source == SOURCE_MOTOR { "motor.rs (caminho original)" } else { "cortex_evolve (folha cortical)" };

    println!("=== REPLAY: {} ===", path.display());
    println!("Fonte:        {}", source_name);
    println!("Seres:        {}", hdr.n_beings);
    println!("Tick rate:    {} Hz", hdr.tick_rate);
    println!("World seed:   {}", hdr.world_seed);
    println!();

    let frames = reader.frames_of(being_idx);
    if frames.is_empty() {
        eprintln!("NENHUM frame para being_idx={}. Verifique o arquivo.", being_idx);
        std::process::exit(1);
    }

    println!("--- Being {} ({} frames) ---", being_idx, frames.len());

    // Estatísticas gerais
    let arc_max = frames.iter().map(|f| f.arc_prog).fold(f32::NEG_INFINITY, f32::max);
    let speed_max = frames.iter().map(|f| f.speed).fold(f32::NEG_INFINITY, f32::max);
    let speed_avg: f32 = frames.iter().map(|f| f.speed).sum::<f32>() / frames.len() as f32;
    let laps = frames.last().map(|f| f.laps).unwrap_or(0);
    let ticks_total = frames.last().map(|f| f.tick).unwrap_or(0);
    let tempo_virtual = ticks_total as f32 / hdr.tick_rate;

    // Quanto o governor interveio
    let gov_count = frames.iter().filter(|f| (f.intent_thr - f.final_thr).abs() > 0.05).count();
    let sentinel_count = frames.iter().filter(|f| f.safety_flags & 0x02 != 0).count();
    let ecas_count = frames.iter().filter(|f| f.safety_flags & 0x04 != 0).count();
    let rev_count  = frames.iter().filter(|f| f.safety_flags & 0x08 != 0).count();

    println!("Arco total:   {:.1} m ({} voltas)", arc_max, laps);
    println!("Tempo virtual:{:.1} s ({} ticks)", tempo_virtual, ticks_total);
    println!("Vel máx:      {:.1} m/s ({:.0} km/h)", speed_max, speed_max * 3.6);
    println!("Vel média:    {:.1} m/s ({:.0} km/h)", speed_avg, speed_avg * 3.6);
    println!();
    println!("=== INTERVENÇÃO EXTERNA ===");
    println!("Governor (|thr_raw - thr_final| > 0.05): {} frames = {:.1}%",
        gov_count, gov_count as f32 / frames.len() as f32 * 100.0);
    println!("Sentinel hard-veto:  {} frames = {:.1}%",
        sentinel_count, sentinel_count as f32 / frames.len() as f32 * 100.0);
    println!("ECAS (steer push):   {} frames = {:.1}%",
        ecas_count, ecas_count as f32 / frames.len() as f32 * 100.0);
    println!("Reversing:           {} frames = {:.1}%",
        rev_count, rev_count as f32 / frames.len() as f32 * 100.0);
    println!();

    // --- Perfil de velocidade por seção ---
    if let Some(profiles) = section_profile(&frames, n_sections) {
        println!("=== PERFIL DE VELOCIDADE ({} seções) ===", n_sections);
        println!("{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "Arco", "AvgSpd", "MaxSpd", "MinSpd", "TmEst", "Gov%", "GOV_N");
        println!("{}", "-".repeat(65));

        // Ordena cópia por tempo estimado (decrescente = onde perde mais)
        let mut worst: Vec<usize> = (0..profiles.len()).collect();
        worst.sort_by(|&a, &b| profiles[b].time_est.partial_cmp(&profiles[a].time_est).unwrap_or(std::cmp::Ordering::Equal));

        for p in &profiles {
            let gov_pct = if p.total_count > 0 {
                p.gov_count as f32 / p.total_count as f32 * 100.0
            } else { 0.0 };
            println!("{:>6.0}m  {:>6.1}   {:>6.1}   {:>6.1}   {:>6.2}s  {:>5.1}%  {:>5}",
                p.arc_start, p.avg_speed, p.max_speed, p.min_speed,
                p.time_est, gov_pct, p.gov_count);
        }

        println!();
        println!("=== TOP-5 SEÇÕES MAIS LENTAS (onde perde mais tempo) ===");
        for &wi in worst.iter().take(5) {
            let p = &profiles[wi];
            println!("  Arco {:.0}m–{:.0}m: avg {:.1} m/s, gov {:.0}%, est {:.2}s perdidos",
                p.arc_start, p.arc_end, p.avg_speed,
                p.gov_count as f32 / p.total_count.max(1) as f32 * 100.0,
                p.time_est);
        }
        println!();
    }

    // --- Eventos críticos ---
    if show_events {
        let events = detect_events(&frames);
        println!("=== EVENTOS CRÍTICOS ({} total) ===", events.len());
        for ev in events.iter().take(50) { // limita a 50 pra não inundar
            print!("  tick={:>5} arc={:>7.1}m spd={:>5.1}  ", ev.tick, ev.arc, ev.speed);
            match &ev.kind {
                universe::replay::EventKind::GovernorCut { thr_before, thr_after } =>
                    println!("GOVERNOR_CUT thr {:.2}→{:.2} (delta {:.2})",
                        thr_before, thr_after, thr_before - thr_after),
                universe::replay::EventKind::SentinelVeto =>
                    println!("SENTINEL_HARD_VETO steer_intent={:.2}", ev.intent_str),
                universe::replay::EventKind::EcasActive =>
                    println!("ECAS steer_push str_final={:.2}", ev.final_str),
                universe::replay::EventKind::Reversing =>
                    println!("REVERSING"),
                universe::replay::EventKind::SpeedDrop { delta } =>
                    println!("SPEED_DROP -{:.1} m/s em 10 frames", delta),
                universe::replay::EventKind::NeuralSaturation { dims } =>
                    println!("NEURAL_SATURACAO {} dims", dims),
            }
        }
        if events.len() > 50 {
            println!("  ... e mais {} eventos", events.len() - 50);
        }
        println!();
    }

    // --- Gerar HTML ---
    if let Some(html_path) = html_out {
        // Pista REAL, reconstruída da semente do mundo guardada no cabeçalho.
        //
        // Antes desenhávamos o próprio caminho do ser como se fosse a pista. Com o ser
        // fazendo 357 m de uma volta de 2.928 m, o "mapa" era uma diagonal — e não era
        // mapa nenhum. O `world_seed` sempre esteve no arquivo; bastava usá-lo.
        //
        // A largura não é gravada (só a semente), e ela não altera o traçado dos
        // waypoints — só a espessura da faixa dirigível. Por isso `--width` aqui é
        // cosmético e o traçado sai correto com o padrão.
        let track: Vec<[f32; 2]> = {
            let mundo = universe::world::VoxelWorld::new(reader.header.world_seed, track_width);
            mundo.waypoints_world.iter().map(|p| [p.x, p.y]).collect()
        };

        let all_frames = reader.all_frames();
        let labels: Vec<&str> = (0..all_frames.len())
            .map(|i| if i == 0 { "Campeão" } else { "Ser" })
            .collect();

        match generate_html_player(&track, &all_frames, &labels, html_path) {
            Ok(_)  => println!("HTML gerado: {}", html_path.display()),
            Err(e) => eprintln!("ERRO ao gerar HTML: {}", e),
        }
    }
}

// ── Diff entre dois replays ──────────────────────────────────────────────────

fn run_diff(path_a: &std::path::Path, path_b: &std::path::Path, html_out: Option<&std::path::Path>, n_sections: usize) {
    let mut ra = match ReplayReader::open(path_a) {
        Ok(r) => r, Err(e) => { eprintln!("ERRO A: {}", e); std::process::exit(1); }
    };
    let mut rb = match ReplayReader::open(path_b) {
        Ok(r) => r, Err(e) => { eprintln!("ERRO B: {}", e); std::process::exit(1); }
    };

    let src_a = if ra.header.source == SOURCE_MOTOR { "motor.rs" } else { "cortex_evolve" };
    let src_b = if rb.header.source == SOURCE_MOTOR { "motor.rs" } else { "cortex_evolve" };

    println!("=== DIFF: {} (A) vs {} (B) ===", src_a, src_b);
    println!("A: {}", path_a.display());
    println!("B: {}", path_b.display());
    println!();

    let frames_a = ra.frames_of(0);
    let frames_b = rb.frames_of(0);

    println!("Frames A: {}", frames_a.len());
    println!("Frames B: {}", frames_b.len());

    let arc_a = frames_a.iter().map(|f| f.arc_prog).fold(f32::NEG_INFINITY, f32::max);
    let arc_b = frames_b.iter().map(|f| f.arc_prog).fold(f32::NEG_INFINITY, f32::max);
    let laps_a = frames_a.last().map(|f| f.laps).unwrap_or(0);
    let laps_b = frames_b.last().map(|f| f.laps).unwrap_or(0);

    println!("Arco A:   {:.1}m ({} voltas)", arc_a, laps_a);
    println!("Arco B:   {:.1}m ({} voltas)", arc_b, laps_b);
    println!("Vantagem: {:+.1}m ({:+} voltas) em favor de {}",
        arc_a - arc_b, laps_a as i32 - laps_b as i32,
        if arc_a >= arc_b { "A" } else { "B" });
    println!();

    // Intervenção em cada caminho
    let gov_a = frames_a.iter().filter(|f| (f.intent_thr - f.final_thr).abs() > 0.05).count();
    let gov_b = frames_b.iter().filter(|f| (f.intent_thr - f.final_thr).abs() > 0.05).count();
    println!("=== INTERVENÇÃO DO GOVERNOR ===");
    println!("A ({:>14}): {} frames = {:.1}%", src_a, gov_a, gov_a as f32 / frames_a.len().max(1) as f32 * 100.0);
    println!("B ({:>14}): {} frames = {:.1}%", src_b, gov_b, gov_b as f32 / frames_b.len().max(1) as f32 * 100.0);
    println!();

    // Diff frame a frame
    let diffs = diff_replays(&frames_a, &frames_b);
    if diffs.is_empty() {
        println!("Nenhum diff calculado (frames vazios).");
    } else {
        // Divergência média e máxima de posição
        let pos_avg: f32 = diffs.iter().map(|d| d.pos_diverge).sum::<f32>() / diffs.len() as f32;
        let pos_max = diffs.iter().map(|d| d.pos_diverge).fold(f32::NEG_INFINITY, f32::max);
        let pos_max_arc = diffs.iter().max_by(|a, b| a.pos_diverge.partial_cmp(&b.pos_diverge).unwrap_or(std::cmp::Ordering::Equal)).map(|d| d.arc).unwrap_or(0.0);

        let spd_avg_a: f32 = diffs.iter().map(|d| d.speed_a).sum::<f32>() / diffs.len() as f32;
        let spd_avg_b: f32 = diffs.iter().map(|d| d.speed_b).sum::<f32>() / diffs.len() as f32;
        let thr_diff_avg: f32 = diffs.iter().map(|d| d.thr_diff).sum::<f32>() / diffs.len() as f32;
        let str_diff_avg: f32 = diffs.iter().map(|d| d.steer_diff.abs()).sum::<f32>() / diffs.len() as f32;

        println!("=== DIVERGÊNCIA DE TRAJETÓRIA ===");
        println!("Distância média A vs B:  {:.1}m", pos_avg);
        println!("Distância máxima:        {:.1}m (em arco {:.0}m)", pos_max, pos_max_arc);
        println!();
        println!("=== DIFERENÇA DE COMANDOS (A - B) ===");
        println!("Vel média A: {:.1} m/s  |  B: {:.1} m/s  |  delta: {:+.1} m/s",
            spd_avg_a, spd_avg_b, spd_avg_a - spd_avg_b);
        println!("Throttle avg diff:  {:+.3}", thr_diff_avg);
        println!("Steer abs avg diff: {:.3}", str_diff_avg);
        println!();

        // Top-10 frames com maior divergência de posição
        let mut sorted: Vec<usize> = (0..diffs.len()).collect();
        sorted.sort_by(|&a, &b| diffs[b].pos_diverge.partial_cmp(&diffs[a].pos_diverge).unwrap_or(std::cmp::Ordering::Equal));

        println!("=== TOP-10 PONTOS DE MAIOR DIVERGÊNCIA ===");
        println!("{:>8} {:>10} {:>8} {:>8} {:>8} {:>8}",
            "Arco", "PosDiv(m)", "SpdA", "SpdB", "ThrDiff", "StrDiff");
        println!("{}", "-".repeat(55));
        for &si in sorted.iter().take(10) {
            let d = &diffs[si];
            println!("{:>6.0}m  {:>8.1}m  {:>6.1}  {:>6.1}  {:>+7.3}  {:>+7.3}",
                d.arc, d.pos_diverge, d.speed_a, d.speed_b, d.thr_diff, d.steer_diff);
        }
        println!();

        // Perfil de velocidade por seção para ambos
        println!("=== PERFIL COMPARATIVO POR SEÇÃO ===");
        let prof_a = section_profile(&frames_a, n_sections);
        let prof_b = section_profile(&frames_b, n_sections);
        if let (Some(pa), Some(pb)) = (prof_a, prof_b) {
            println!("{:>8} {:>10} {:>10} {:>10}",
                "Arco", "SpdA(m/s)", "SpdB(m/s)", "Delta");
            println!("{}", "-".repeat(45));
            let n = pa.len().min(pb.len());
            for i in 0..n {
                let delta = pa[i].avg_speed - pb[i].avg_speed;
                let mark = if delta.abs() > 2.0 { " <--" } else { "" };
                println!("{:>6.0}m  {:>8.1}   {:>8.1}   {:>+8.1}{}",
                    pa[i].arc_start, pa[i].avg_speed, pb[i].avg_speed, delta, mark);
            }
        }
    }

    // --- HTML diff ---
    if let Some(html_path) = html_out {
        let track: Vec<[f32; 2]> = frames_a.iter()
            .step_by(5)
            .map(|f| [f.pos[0], f.pos[2]])
            .collect();

        let all_a = ra.all_frames();
        let all_b = rb.all_frames();

        // Intercala seres: A primeiro, B segundo (índice 0 = campeão A, índice n = campeão B)
        let mut combined = all_a.clone();
        for frames in all_b {
            combined.push(frames);
        }

        let n_a = all_a.len();
        let n_b = combined.len() - n_a;
        let mut labels: Vec<String> = Vec::new();
        for i in 0..n_a { labels.push(format!("A-{} ({})", i, src_a)); }
        for i in 0..n_b { labels.push(format!("B-{} ({})", i, src_b)); }
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        match generate_html_player(&track, &combined, &label_refs, html_path) {
            Ok(_)  => println!("HTML diff gerado: {}", html_path.display()),
            Err(e) => eprintln!("ERRO ao gerar HTML: {}", e),
        }
    }
}
