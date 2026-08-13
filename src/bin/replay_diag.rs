//! DIAGNÓSTICO DE COMPORTAMENTO a partir do replay — sem renderizar nada.
//!
//! # Por que existe
//!
//! Leonardo, olhando o mapa recém-consertado: *"agora que você tem o html, dá até para
//! debugar, dá para usar os dados que alimentam o html, nem precisa renderizar e validar,
//! só ler e comparar: se sai da pista, se bate, se dá ré."*
//!
//! Está certo, e muda o ciclo. Até aqui eu julgava o comportamento por médias de geração
//! (mediana, máximo, mortalidade), que dizem QUANTO mas nunca ONDE nem POR QUÊ. O replay
//! guarda posição e velocidade por quadro, e a geometria da pista é reconstruível da
//! `world_seed` que já está no cabeçalho. Cruzando os dois sai a verdade por quadro:
//! em que ponto do traçado ele saiu, quanto tempo passou na grama, onde parou.
//!
//! Foi assim que os dois últimos bugs apareceram — o mapa que era o rastro do próprio ser,
//! e a fitness de hodômetro que não exigia seguir a pista. Nenhum dos dois aparecia em
//! média alguma.
//!
//! # Uso
//!
//!   replay_diag --replay telemetry/<corrida>/gen0011.replay [--width 28]

use std::path::PathBuf;
use universe::replay::ReplayReader;
use universe::world::{VoxelWorld, VoxelType};
use bevy::math::Vec2;   // mesmo tipo que `world.rs` usa (via bevy::prelude)

/// Um episódio contíguo fora da pista.
struct Excursion {
    start_tick: u32,
    end_tick: u32,
    at_progress: f32,
    max_depth: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path: Option<PathBuf> = None;
    let mut width = 18.0f32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--replay" => { i += 1; if i < args.len() { path = Some(PathBuf::from(&args[i])); } }
            "--width"  => { i += 1; if i < args.len() { width = args[i].parse().unwrap_or(18.0); } }
            _ => {}
        }
        i += 1;
    }
    let Some(path) = path else {
        eprintln!("uso: replay_diag --replay <arquivo.replay> [--width 28]");
        std::process::exit(2);
    };

    let mut reader = match ReplayReader::open(&path) {
        Ok(r) => r,
        Err(e) => { eprintln!("não abriu {}: {e}", path.display()); std::process::exit(1); }
    };
    let seed = reader.header.world_seed;
    let frames = reader.all_frames();
    let Some(fr) = frames.first() else {
        eprintln!("replay sem quadros"); std::process::exit(1);
    };
    if fr.is_empty() { eprintln!("replay sem quadros"); std::process::exit(1); }

    // O construtor do mundo despeja 129 waypoints no stdout. Útil uma vez, ruído em
    // toda chamada — silencia enquanto constrói.
    let world = {
        let _mudo = universe::world::silence_map_diag();
        VoxelWorld::new(seed, width)
    };
    let lap = world.lap_distance;

    println!("=== DIAGNÓSTICO: {} ===", path.display());
    println!("mundo semente {seed} · pista {width:.0} m · volta {lap:.0} m · {} quadros\n", fr.len());

    // ── Varredura por quadro ────────────────────────────────────────────────
    let mut on_road = 0usize;
    let mut on_grass = 0usize;
    let mut on_wall = 0usize;
    let mut reversing = 0usize;
    let mut stopped = 0usize;
    let mut excursions: Vec<Excursion> = Vec::new();
    let mut cur: Option<Excursion> = None;
    // Avanço LÍQUIDO monotônico, não amplitude. `max - min` conta um ser que recua
    // cruzando a linha (14m→2900m) como '99,5% da volta' — falso. Mesmo bug já
    // corrigido no replay_obs; o replay_diag ainda tinha a versão antiga.
    let mut prog_liq = 0.0f32;
    let mut prog_pico = 0.0f32;
    let mut prog_ant = world.get_track_progress(fr[0].pos[0], fr[0].pos[2]);
    let mut speed_sum = 0.0f64;
    let mut speed_max = 0.0f32;

    for f in fr {
        let p = Vec2::new(f.pos[0], f.pos[2]);
        let mat = world.material_at(p);
        let prog = world.get_track_progress(p.x, p.y);
        let mut d = prog - prog_ant;
        if d < -lap * 0.5 { d += lap; } else if d > lap * 0.5 { d -= lap; }
        prog_liq += d.clamp(-lap * 0.1, lap * 0.1);
        prog_pico = prog_pico.max(prog_liq);
        prog_ant = prog;
        speed_sum += f.speed as f64;
        speed_max = speed_max.max(f.speed);

        // Marcha à ré: velocidade projetada no sentido de avanço é negativa.
        // `fwd` no quadro é (x, z) já normalizado.
        if f.gear == 0 { reversing += 1; }
        if f.speed < 0.5 { stopped += 1; }

        let fora = match mat {
            VoxelType::Road => { on_road += 1; false }
            VoxelType::Wall => { on_wall += 1; true }
            _ => { on_grass += 1; true }
        };

        // Profundidade da saída: distância até o eixo da pista menos a meia-largura.
        let depth = distance_to_centerline(&world, p) - width / 2.0;

        if fora {
            match cur.as_mut() {
                Some(e) => { e.end_tick = f.tick; e.max_depth = e.max_depth.max(depth); }
                None => cur = Some(Excursion {
                    start_tick: f.tick, end_tick: f.tick, at_progress: prog, max_depth: depth,
                }),
            }
        } else if let Some(e) = cur.take() {
            excursions.push(e);
        }
    }
    if let Some(e) = cur.take() { excursions.push(e); }

    let n = fr.len() as f32;
    let pct = |c: usize| 100.0 * c as f32 / n;

    println!("── ONDE O SER PASSOU O TEMPO ──");
    println!("  na pista:   {:5.1}%  ({} quadros)", pct(on_road), on_road);
    println!("  na grama:   {:5.1}%  ({} quadros)", pct(on_grass), on_grass);
    println!("  em parede:  {:5.1}%  ({} quadros)", pct(on_wall), on_wall);
    println!("  parado (<0,5 m/s): {:5.1}%", pct(stopped));
    println!("  marcha à ré:       {:5.1}%", pct(reversing));

    println!("\n── AVANÇO NO CIRCUITO ──");
    let span = prog_pico.max(0.0);
    println!("  progresso alcançado: {:.0} m de {:.0} m ({:.1}% da volta)",
             span, lap, 100.0 * span / lap);
    println!("  velocidade média {:.2} m/s ({:.1} km/h) · máxima {:.2} m/s ({:.1} km/h)",
             speed_sum / fr.len() as f64, speed_sum / fr.len() as f64 * 3.6,
             speed_max, speed_max * 3.6);

    // ── O ser TENTA virar? ─────────────────────────────────────────────────
    // Todos param no mesmo ponto do traçado (~265 m, onde a curva passa de 10° para 18°).
    // Isto separa as duas explicações: volante em zero = não manda virar; volante no
    // batente sem o carro girar = manda e o corpo não obedece.
    {
        let mut str_abs: Vec<f32> = fr.iter().map(|f| f.final_str.abs()).collect();
        str_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = str_abs[str_abs.len() / 2];
        let p90 = str_abs[(str_abs.len() * 9 / 10).min(str_abs.len() - 1)];
        let saturado = str_abs.iter().filter(|&&v| v > 0.9).count();
        let quieto = str_abs.iter().filter(|&&v| v < 0.05).count();
        let mut thr: Vec<f32> = fr.iter().map(|f| f.final_thr).collect();
        thr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("\n── COMANDOS ──");
        println!("  volante |x|: mediana {:.2} · p90 {:.2} · no batente {:.0}% · quase zero {:.0}%",
                 med, p90, pct(saturado), pct(quieto));
        println!("  acelerador: mediana {:.2}", thr[thr.len() / 2]);
        // O SINAL decide entre "corrige para os dois lados" e "vive torto".
        // Volante cravado num lado + acelerador constante = círculo, e círculo dentro de
        // uma pista larga não conta como sair da pista nem como bater: some de todas as
        // métricas agregadas e só aparece no progresso que para de subir.
        let assinado: Vec<f32> = fr.iter().map(|f| f.final_str).collect();
        let media_ass = assinado.iter().sum::<f32>() / assinado.len() as f32;
        let esq = assinado.iter().filter(|&&v| v < -0.05).count();
        let dir = assinado.iter().filter(|&&v| v >  0.05).count();
        println!("  volante com sinal: média {:+.2} · esquerda {:.0}% · direita {:.0}%",
                 media_ass, pct(esq), pct(dir));
        if media_ass.abs() > 0.3 && (pct(esq) > 80.0 || pct(dir) > 80.0) {
            println!("  ⇒ VOLANTE CRAVADO num lado só: o ser anda em CÍRCULO, não em curva.");
            println!("    Círculo dentro de pista larga não sai do traçado nem bate —");
            println!("    some de toda métrica agregada e só aparece no progresso parado.");
        }

        if med < 0.05 && p90 < 0.2 {
            println!("  ⇒ o volante fica PARADO: o cérebro não está mandando virar.");
        } else if pct(saturado) > 30.0 {
            println!("  ⇒ volante no batente {:.0}% do tempo: manda virar e não resolve.", pct(saturado));
        }
    }

    // ── O volante RESPONDE à pista? ────────────────────────────────────────
    //
    // Tirar o alisamento (`--steer-alpha 1.0`) não mudou nada: 265 m e +0,40 de média nos
    // dois casos. Então o comando constante vem do cérebro, não do filtro. Resta separar
    // "não enxerga a curva" de "enxerga e não usa": se o volante não correlaciona com a
    // curvatura do traçado à frente, ele não está dirigindo — está segurando.
    {
        let mut cur: Vec<f32> = Vec::with_capacity(fr.len());
        let mut st:  Vec<f32> = Vec::with_capacity(fr.len());
        for f in fr {
            let p = Vec2::new(f.pos[0], f.pos[2]);
            let k = curvatura_a_frente(&world, p, 25.0);
            if k.is_finite() {
                cur.push(k);
                st.push(f.final_str);
            }
        }
        if cur.len() > 30 {
            let r = correlacao(&cur, &st);
            let var_st = variancia(&st);
            println!("\n── O VOLANTE SEGUE A PISTA? ──");
            println!("  correlação volante × curvatura 25 m à frente: {:+.3}", r);
            println!("  variância do volante: {:.4}", var_st);
            if r.abs() < 0.15 {
                println!("  ⇒ NÃO segue: o comando é praticamente independente do traçado.");
                println!("    O ser não está dirigindo, está segurando o volante numa posição.");
            } else if r > 0.3 {
                println!("  ⇒ segue a curvatura — o problema é de amplitude ou de timing, não de leitura.");
            }
        }
    }

    println!("\n── SAÍDAS DE PISTA ({}) ──", excursions.len());
    if excursions.is_empty() {
        println!("  nenhuma — ficou no traçado o tempo todo.");
    } else {
        // A PRIMEIRA saída é a que importa: é onde a competência acaba. As seguintes já
        // acontecem com o ser perdido, então descrevem a deriva, não a falha.
        for (k, e) in excursions.iter().take(6).enumerate() {
            let dur = (e.end_tick - e.start_tick) as f32 / 64.0;
            println!("  {}. tick {:>6}–{:<6} ({:5.1}s) · aos {:.0} m da volta ({:.0}%) · até {:.1} m fora",
                     k + 1, e.start_tick, e.end_tick, dur,
                     e.at_progress, 100.0 * e.at_progress / lap, e.max_depth.max(0.0));
        }
        if excursions.len() > 6 { println!("  … e mais {}", excursions.len() - 6); }

        let e = &excursions[0];
        println!("\n  A PRIMEIRA saída é o diagnóstico: o ser dirigiu bem até {:.0} m", e.at_progress);
        println!("  ({:.0}% da volta) e ali deixou o traçado.", 100.0 * e.at_progress / lap);
        descrever_curva(&world, e.at_progress);
    }

    // Veredito curto, para não precisar interpretar as tabelas toda vez.
    println!("\n── VEREDITO ──");
    if pct(on_road) < 50.0 {
        println!("  Passa mais tempo FORA da pista do que nela. O objetivo não está exigindo");
        println!("  seguir o traçado, ou ele não consegue — olhe a primeira saída acima.");
    } else if pct(stopped) > 30.0 {
        println!("  Fica parado {:.0}% do tempo. O gargalo é ANDAR, não navegar.", pct(stopped));
    } else if span < lap * 0.5 {
        println!("  Fica na pista mas só cobre {:.0}% da volta: trava num ponto específico.",
                 100.0 * span / lap);
    } else {
        println!("  Segue o traçado e cobre boa parte da volta.");
    }
}

/// Distância até o eixo da pista, pelo waypoint mais próximo.
fn distance_to_centerline(world: &VoxelWorld, p: Vec2) -> f32 {
    world.waypoints_world.iter()
        .map(|w| w.distance(p))
        .fold(f32::MAX, f32::min)
}

/// Descreve a curvatura no ponto onde o ser saiu — reta e curva fechada falham por
/// motivos diferentes, e a tabela sozinha não distingue.
fn descrever_curva(world: &VoxelWorld, at_progress: f32) {
    let wps = &world.waypoints_world;
    if wps.len() < 3 { return; }
    // Acha o waypoint cujo progresso acumulado está mais perto do ponto de saída.
    let mut acc = 0.0f32;
    let mut idx = 0usize;
    for i in 0..wps.len() - 1 {
        let seg = wps[i].distance(wps[i + 1]);
        if acc + seg >= at_progress { idx = i; break; }
        acc += seg;
    }
    let a = wps[idx.saturating_sub(1)];
    let b = wps[idx];
    let c = wps[(idx + 1).min(wps.len() - 1)];
    let v1 = (b - a).normalize_or_zero();
    let v2 = (c - b).normalize_or_zero();
    let ang = v1.dot(v2).clamp(-1.0, 1.0).acos().to_degrees();
    let tipo = if ang > 45.0 { "CURVA FECHADA" }
               else if ang > 15.0 { "curva" }
               else { "reta" };
    println!("  Geometria ali: {tipo} ({ang:.0}° entre os segmentos, waypoint {idx}).");
}


/// Curvatura assinada do traçado `ahead` metros à frente do ponto `p`.
///
/// Positiva = a pista vira à direita. Serve de "resposta certa" contra a qual comparar o
/// comando do volante, sem depender de nenhum canal sensorial do cérebro.
fn curvatura_a_frente(world: &VoxelWorld, p: Vec2, ahead: f32) -> f32 {
    let wps = &world.waypoints_world;
    if wps.len() < 4 { return f32::NAN; }
    let mut i0 = 0usize;
    let mut best = f32::MAX;
    for (i, w) in wps.iter().enumerate() {
        let d = w.distance(p);
        if d < best { best = d; i0 = i; }
    }
    // Anda pela lista até acumular `ahead` metros.
    let mut acc = 0.0;
    let mut i = i0;
    while acc < ahead && i + 1 < wps.len() {
        acc += wps[i].distance(wps[i + 1]);
        i += 1;
    }
    if i + 1 >= wps.len() { return f32::NAN; }
    let a = wps[i.saturating_sub(1)];
    let b = wps[i];
    let c = wps[i + 1];
    let v1 = (b - a).normalize_or_zero();
    let v2 = (c - b).normalize_or_zero();
    // Produto vetorial em 2D dá o SENTIDO da curva.
    v1.x * v2.y - v1.y * v2.x
}

fn variancia(v: &[f32]) -> f32 {
    let m = v.iter().sum::<f32>() / v.len() as f32;
    v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32
}

fn correlacao(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let cov: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let sa: f32 = a.iter().map(|x| (x - ma).powi(2)).sum::<f32>().sqrt();
    let sb: f32 = b.iter().map(|y| (y - mb).powi(2)).sum::<f32>().sqrt();
    if sa * sb < 1e-12 { 0.0 } else { cov / (sa * sb) }
}
