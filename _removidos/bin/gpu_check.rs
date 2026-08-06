//! Prova de equivalência e medida de ganho da retina em GPU.
//!
//! Roda com `env -u DISPLAY` — é o ponto do módulo `gpu`: compute sem parte gráfica.
//!
//! # Por que este binário existe
//!
//! A retina custa 57% de todo o tempo de simulação (2440 amostras de PERF), contra 14% do
//! cérebro. Passar isso para a GPU é o maior ganho disponível. Mas um caminho GPU que responde
//! DIFERENTE em silêncio invalidaria todo experimento já medido — então a equivalência com a
//! CPU vem antes do ganho, e a CPU aqui roda só como oráculo, em algumas dezenas de posturas.

use bevy::math::{Vec2, Vec3};
use universe::config::iara::vision::RETINA_TOTAL;
use universe::gpu::{Pose, RetinaGpu};
use universe::world::VoxelWorld;

fn main() {
    let world = VoxelWorld::new(9001, 18.0);
    println!("mundo: {}x{} células · {:.0} m de lado · volta {:.0} m",
             world.size, world.size, world.size as f32 * world.cell_size, world.lap_distance);

    // Posturas espalhadas pela pista, em ângulos variados — não adianta validar num ponto só.
    let mut poses = Vec::new();
    let n_wp = world.waypoints_world.len();
    for k in 0..48usize {
        let wp = world.waypoints_world[(k * n_wp / 48).min(n_wp - 1)];
        let ang = k as f32 * 0.37;
        poses.push(Pose {
            pos: Vec3::new(wp.x, 0.0, wp.y),
            forward: Vec2::new(ang.cos(), ang.sin()),
            right: Vec2::new(-ang.sin(), ang.cos()),
            gaze_yaw: 0.0,
        });
    }

    let gpu = match RetinaGpu::new(&world, poses.len().max(64)) {
        Some(g) => g,
        None => { println!("✗ nenhuma GPU utilizável"); std::process::exit(1); }
    };
    println!("GPU: {}\n", gpu.adapter);

    // ---- EQUIVALÊNCIA
    let gpu_output = gpu.render_legacy(&poses);
    let mut worst: Vec<(usize, i32)> = Vec::new();
    let mut identical = 0usize;
    let mut total = 0usize;
    let mut max_dif = 0i32;
    for (i, p) in poses.iter().enumerate() {
        let cpu = world.render_binocular_retina(
            p.pos, Vec3::new(p.forward.x, 0.0, p.forward.y),
            Vec3::new(p.right.x, 0.0, p.right.y), p.gaze_yaw);
        let mut worst_diff = 0i32;
        for px in 0..RETINA_TOTAL {
            let d = (cpu[px] as i32 - gpu_output[i][px] as i32).abs();
            if d == 0 { identical += 1; }
            if d > worst_diff { worst_diff = d; }
            if d > max_dif { max_dif = d; }
            total += 1;
        }
        if worst_diff > 0 { worst.push((i, worst_diff)); }
    }
    println!("EQUIVALÊNCIA GPU × CPU");
    println!("  pixels comparados : {total}");
    println!("  idênticos         : {identical} ({:.4}%)", identical as f32 / total as f32 * 100.0);
    println!("  maior diferença   : {max_dif}");
    if !worst.is_empty() {
        println!("  posturas com diferença: {} de {}", worst.len(), poses.len());
    }
    let ok = max_dif <= 1;
    println!("  -> {}\n", if max_dif == 0 { "IDÊNTICO" }
                          else if ok { "diferença ≤1 (arredondamento de float)" }
                          else { "✗ DIVERGE — não usar" });

    // ---- GANHO
    let repetitions = 40;
    let t0 = std::time::Instant::now();
    for _ in 0..repetitions { let _ = gpu.render(&poses); }
    let t_gpu = t0.elapsed().as_secs_f64() / repetitions as f64;

    let t1 = std::time::Instant::now();
    for _ in 0..repetitions {
        for p in &poses {
            let _ = world.render_binocular_retina(
                p.pos, Vec3::new(p.forward.x, 0.0, p.forward.y),
                Vec3::new(p.right.x, 0.0, p.right.y), p.gaze_yaw);
        }
    }
    let t_cpu = t1.elapsed().as_secs_f64() / repetitions as f64;

    println!("GANHO ({} seres, {} raios por despacho)", poses.len(), poses.len() * RETINA_TOTAL);
    println!("  CPU (1 thread) : {:>8.2} ms", t_cpu * 1000.0);
    println!("  GPU            : {:>8.2} ms", t_gpu * 1000.0);
    println!("  aceleração     : {:>8.1}×", t_cpu / t_gpu);
    println!("\n  A CPU real usa rayon em todos os núcleos, então o ganho de ponta a ponta");
    println!("  é menor que este número — a medida honesta é o PERF da simulação inteira.");

    // ---- O QUE A REDE PASSA A RECEBER
    //
    // O ponto da mudança não é velocidade, é INFORMAÇÃO. A codificação antiga espremia tipo
    // e distância num único i8 com uma curva de salência escrita à mão; a física entrega os
    // dois separados. Aqui se mede quanto disso chega, por material.
    let fis = gpu.render(&poses);
    let nomes = ["Grass", "Road", "Wall", "Hole"];
    println!("\n\nINFORMAÇÃO ENTREGUE, POR MATERIAL");
    println!("  {:<7} {:>9} {:>26} {:>26}", "material", "pixels", "CODIFICADA (níveis i8)", "FÍSICA (distâncias)");
    println!("  {}", "-".repeat(72));
    for m in 0..4u8 {
        let mut niveis_cod = std::collections::HashSet::new();
        let mut niveis_fis = std::collections::HashSet::new();
        let mut n = 0usize;
        let (mut dmin, mut dmax) = (f32::MAX, 0.0f32);
        for r in &fis {
            for px in r.iter().filter(|p| p.material == m) {
                n += 1;
                niveis_cod.insert(universe::gpu::legacy_encoding(*px));
                niveis_fis.insert((px.dist_m * 100.0) as i32); // resolução de 1 cm
                if px.dist_m < dmin { dmin = px.dist_m; }
                if px.dist_m > dmax { dmax = px.dist_m; }
            }
        }
        if n == 0 { continue; }
        println!("  {:<7} {:>9} {:>26} {:>26}", nomes[m as usize], n,
                 format!("{}", niveis_cod.len()),
                 format!("{} ({:.1}–{:.1} m)", niveis_fis.len(), dmin, dmax));
    }
    println!("  {}", "-".repeat(72));
    println!("\n  A pista deixa de ser 4 números e vira distância de verdade.");
    println!("  E o material sai num canal PRÓPRIO, então deixa de disputar escala com ela —");
    println!("  era isso que fazia o max-pool jogar a pista fora sempre que havia parede.");

    if !ok { std::process::exit(1); }
}
