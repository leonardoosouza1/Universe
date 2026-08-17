//! VALIDAÇÃO DA PISTA — a pista é dirigível, ou o mapa está quebrado?
//!
//! # Por que existe
//!
//! Leonardo: *"ou vc errou, o mapa é bugado para o modelo. Sério, tem algo de errado. O
//! modelo consegue fazer tudo — vira, vê, pensa — mas não vira a curva."*
//!
//! Todo controlador testado para entre 255 e 271 m: o cérebro evoluído, um piloto
//! geométrico com a racing line, um piloto sensorial de raios de parede, e o mesmo piloto
//! com o sinal invertido. Quatro controladores independentes parando no mesmo ponto não é
//! coincidência de controle — é propriedade do mundo.
//!
//! Este binário não simula nada. Ele CAMINHA pelo eixo da pista metro a metro e pergunta
//! ao mesmo `VoxelWorld` que o simulador usa: o que tem aqui? A pista continua? Qual a
//! largura livre? Se houver parede atravessada no meio da estrada em 270 m, acabou a
//! investigação.
//!
//! # Uso
//!
//!   pista_check [--width 28] [--ate 600] [--seed 9001]

use bevy::math::Vec2;
use universe::world::{VoxelType, VoxelWorld};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut width = 28.0f32;
    let mut ate = 600.0f32;
    let mut seed = 9001u64;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--width" => {
                i += 1;
                if i < args.len() {
                    width = args[i].parse().unwrap_or(28.0);
                }
            }
            "--ate" => {
                i += 1;
                if i < args.len() {
                    ate = args[i].parse().unwrap_or(600.0);
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse().unwrap_or(9001);
                }
            }
            _ => {}
        }
        i += 1;
    }

    let world = {
        let _q = universe::world::silence_map_diag();
        VoxelWorld::new(seed, width)
    };
    let wps = &world.waypoints_world;
    let n = wps.len();

    println!(
        "PISTA semente {seed} · largura declarada {width:.0} m · volta {:.0} m · {n} waypoints\n",
        world.lap_distance
    );

    // Caminha pelo EIXO, interpolando entre waypoints, e mede em cada metro:
    //   · o material sob o eixo (tem de ser Road; qualquer outra coisa é buraco no traçado)
    //   · a largura livre para cada lado até encontrar parede
    //   · o raio da curva local
    let mut arco = 0.0f32;
    let mut wp = 0usize;
    let mut t = 0.0f32;
    let mut problemas = 0usize;

    println!(
        "{:>6} {:>8} {:>7} {:>7} {:>7}  {}",
        "arco", "material", "esq", "dir", "raio", ""
    );
    println!("{}", "-".repeat(62));

    while arco < ate && wp < n - 1 {
        let a = wps[wp];
        let b = wps[wp + 1];
        let seg = (b - a).length().max(0.001);
        let p = a + (b - a) * (t / seg);

        // Normal ao eixo, para medir a largura livre dos dois lados.
        let dir_seg = (b - a) / seg;
        let nrm = Vec2::new(-dir_seg.y, dir_seg.x);

        let mat = world.material_at(p);
        let livre = |sinal: f32| -> f32 {
            let mut d = 0.0f32;
            while d < 300.0 {
                let q = p + nrm * (d * sinal);
                if !matches!(world.material_at(q), VoxelType::Road) {
                    return d;
                }
                d += 0.5;
            }
            300.0
        };
        let esq = livre(1.0);
        let dir = livre(-1.0);

        // Raio local pelo ângulo entre os segmentos vizinhos.
        let raio = if wp > 0 && wp + 1 < n {
            let v1 = wps[wp] - wps[wp - 1];
            let v2 = wps[wp + 1] - wps[wp];
            let ang = (v1.x * v2.y - v1.y * v2.x).atan2(v1.dot(v2)).abs();
            if ang > 1e-4 {
                (0.5 * (v1.length() + v2.length())) / ang
            } else {
                99999.0
            }
        } else {
            99999.0
        };

        // Só imprime o que interessa: a cada 10 m, e SEMPRE quando há defeito.
        let defeito = !matches!(mat, VoxelType::Road) || esq + dir < width * 0.5;
        if defeito {
            problemas += 1;
        }
        if arco as u32 % 10 == 0 || defeito {
            let marca = if !matches!(mat, VoxelType::Road) {
                "  ← EIXO FORA DA PISTA"
            } else if esq + dir < width * 0.5 {
                "  ← ESTRANGULAMENTO"
            } else {
                ""
            };
            println!(
                "{:6.0} {:>8} {:6.1}m {:6.1}m {:6.0}m{}",
                arco,
                format!("{:?}", mat),
                esq,
                dir,
                raio.min(99999.0),
                marca
            );
        }

        arco += 1.0;
        t += 1.0;
        if t >= seg {
            t -= seg;
            wp += 1;
        }
    }

    // ── O QUE O SER REALMENTE PERCEBE ─────────────────────────────────────
    //
    // Leonardo: *"ele tem que perceber a curva, a visão dele tem que ver literalmente."*
    //
    // Fidelidade não basta. Um sensor perfeito de um valor CONSTANTE entrega zero bit de
    // informação, e foi assim que eu confundi correlação 1,000 com "eles enxergam". Aqui
    // reproduzo o raio frontal exatamente como `gpu_evolve.rs` o calcula — sondas de 1 a
    // 10 m, padrão 30,0 quando nada é encontrado — e reporto quantas vezes ele SATURA.
    {
        let mut arco = 0.0f32;
        let mut wp = 0usize;
        let mut t = 0.0f32;
        let mut sat = 0usize;
        let mut tot = 0usize;
        let mut vals: Vec<f32> = Vec::new();
        while arco < ate && wp < n - 1 {
            let a = wps[wp];
            let b = wps[wp + 1];
            let seg = (b - a).length().max(0.001);
            let p = a + (b - a) * (t / seg);
            let f = (b - a) / seg;
            // O SENSOR CERTO. A versão anterior media as sondas de 1 a 10 m, que são o
            // gatilho da MARCHA À RÉ em `gpu_evolve.rs:573` — não a visão. A visão do ser
            // está em `perception.wgsl:373`: oito amostras até 60 m. Medir o sensor errado
            // e concluir "o ser é cego" é exatamente o erro que este arquivo existe para
            // não deixar acontecer.
            let mut fwd = 60.0f32;
            for sd in [3.0f32, 8.0, 15.0, 25.0, 35.0, 45.0, 52.0, 60.0] {
                if !matches!(world.material_at(p + f * sd), VoxelType::Road) {
                    fwd = sd;
                    break;
                }
            }
            if fwd >= 60.0 {
                sat += 1;
            }
            vals.push(fwd);
            tot += 1;
            arco += 1.0;
            t += 1.0;
            if t >= seg {
                t -= seg;
                wp += 1;
            }
        }
        let media = vals.iter().sum::<f32>() / tot.max(1) as f32;
        let dp = (vals.iter().map(|v| (v - media).powi(2)).sum::<f32>() / tot.max(1) as f32).sqrt();
        println!("\n── O QUE A VISÃO FRONTAL ENTREGA (carro no eixo, perception.wgsl) ──");
        println!(
            "  saturado em 60,0 m: {:.1}% dos pontos ({sat} de {tot})",
            100.0 * sat as f32 / tot.max(1) as f32
        );
        println!("  média {media:.1} m · desvio padrão {dp:.2} m");
        if dp < 0.5 {
            println!(
                "  → VARIÂNCIA NULA: o canal é constante. O ser é cego para a curva por\n                      \x20    este sensor, por mais fiel que ele seja."
            );
        }
    }

    println!("\n{}", "-".repeat(62));
    if problemas == 0 {
        println!(
            "Nenhum defeito: o eixo está sobre asfalto e a largura livre é coerente\n\
                  em todo o trecho examinado. A pista é dirigível — o problema não é o mapa."
        );
    } else {
        println!(
            "{problemas} pontos com defeito no trecho examinado.\n\
                  Eixo fora da pista ou estrangulamento significa que NENHUM controlador\n\
                  passaria — a investigação de controle estava perseguindo a coisa errada."
        );
    }
}
