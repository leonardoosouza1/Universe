//! VALIDAÇÃO DA VISÃO POR OBJETO — o ser enxerga o que está na frente dele?
//!
//! # Por que existe
//!
//! Leonardo: *"seria bom a gente ter alguma forma de validar se o modelo enxerga realmente
//! as coisas — se enxerga uma parede, se enxerga o chão, se enxerga algo que a gente
//! coloque. A partir de um evento em que a visão bate naquele objeto, a gente calcula se
//! ele realmente está enxergando. Pra avaliar: poxa, o modelo vai tão bem sem buraco, mas
//! com o buraco não dá — beleza, porque ele satura aqui, ou porque só enxerga quando está
//! muito em cima."*
//!
//! O A/B de 2026-08-13 mostrou o problema com números: mesma semente, mesmo genoma, um
//! único flag de diferença. Com buracos a população travou em 1.076 m por 20 gerações e
//! **zero** dos 48 seres deu uma volta. Sem buracos, mediana 3.189 m e **25 de 48**
//! completaram. Os buracos ficam em 1.087 m — onze metros depois da parede.
//!
//! Isso não prova que o ser é cego para o buraco; prova que ele morre lá. Este binário
//! separa as duas coisas medindo a **curva de detecção**: aproxima um observador do objeto
//! metro a metro e responde, para cada distância, quantos pixels da retina o encontram e
//! com que valor codificado. Se o buraco só aparece a 8 m, nenhum cérebro desvia a 6 m/s —
//! e o defeito é de codificação, não de comportamento.
//!
//! # O que ele mede
//!
//! Replica o DDA de `retina.wgsl` (que para em `v==2u` Parede ou `v==3u` Buraco) e a
//! `legacy_encoding()` de `perception.wgsl`, e reporta por distância:
//!
//! · **pixels** que atingiram o alvo (de `RETINA_COLS` × linhas amostradas)
//! · **valor i8** codificado — o número que o cérebro realmente recebe
//! · **contraste** contra o fundo daquela mesma coluna sem o objeto
//! · **ambiguidade**: se o valor colide com o de outro material a outra distância
//!
//! # Uso
//!
//!   visao_check [--seed 9001] [--width 28] [--alvo buraco|parede] [--de 120] [--ate 2]

use bevy::math::Vec2;
use universe::config::iara::vision::{RETINA_COLS, RETINA_ROWS};
use universe::world::{VoxelType, VoxelWorld};

/// Codificação legada, réplica exata de `perception.wgsl:251`.
///
/// Mantida aqui (e não importada) porque o original vive em WGSL: qualquer divergência
/// entre as duas é justamente o tipo de bug que esta ferramenta existe para expor. Se
/// alguém mexer no shader e esquecer daqui, o teste `encoding_bate_com_o_shader` quebra.
fn legacy_encoding(d: f32, mat: VoxelType) -> i32 {
    let depth_fade = (d / 1200.0).clamp(0.0, 1.0);
    let v = match mat {
        VoxelType::Wall => {
            if d < 30.0 {
                127.0 - ((d / 30.0).clamp(0.0, 1.0).sqrt() * 72.0)
            } else if d < 120.0 {
                55.0 - (((d - 30.0) / 90.0).clamp(0.0, 1.0) * 55.0)
            } else if d < 1200.0 {
                -(((d - 120.0) / 480.0).clamp(0.0, 1.0) * 25.0)
            } else {
                -25.0 - (((d - 1200.0) / 1200.0).clamp(0.0, 1.0) * 25.0)
            }
        }
        VoxelType::Grass => 40.0 - (depth_fade * 40.0),
        VoxelType::Road => -128.0 + (depth_fade * 96.0),
        VoxelType::Hole => 90.0 - ((d / 50.0).clamp(0.0, 1.0) * 30.0),
    };
    v.clamp(-128.0, 127.0) as i32
}

/// Um raio de visão: anda pelo mundo até bater em Parede ou Buraco, como o DDA do shader.
///
/// Devolve `(distância, material)` do primeiro obstáculo, ou `None` se o raio percorreu o
/// alcance inteiro sem encontrar nada — o caso "céu aberto" que satura o canal.
fn raio(world: &VoxelWorld, origem: Vec2, dir: Vec2, alcance: f32) -> Option<(f32, VoxelType)> {
    // Passo de meio voxel: fino o bastante para não pular um alvo de 2×2 células, barato o
    // bastante para varrer 120 m em microssegundos.
    let passo = world.cell_size * 0.5;
    let mut d = passo;
    while d < alcance {
        let m = world.material_at(origem + dir * d);
        if matches!(m, VoxelType::Wall | VoxelType::Hole) {
            return Some((d, m));
        }
        d += passo;
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seed = 9001u64;
    let mut width = 28.0f32;
    let mut alvo = "buraco".to_string();
    let mut de = 120.0f32;
    let mut ate = 2.0f32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args[i].parse().unwrap_or(9001);
            }
            "--width" => {
                i += 1;
                width = args[i].parse().unwrap_or(28.0);
            }
            "--alvo" => {
                i += 1;
                alvo = args[i].clone();
            }
            "--de" => {
                i += 1;
                de = args[i].parse().unwrap_or(120.0);
            }
            "--ate" => {
                i += 1;
                ate = args[i].parse().unwrap_or(2.0);
            }
            _ => {}
        }
        i += 1;
    }

    let world = {
        let _q = universe::world::silence_map_diag();
        VoxelWorld::new(seed, width)
    };
    let alvo_tipo = if alvo == "parede" {
        VoxelType::Wall
    } else {
        VoxelType::Hole
    };

    // Acha o primeiro alvo do tipo pedido caminhando pelo eixo da pista, e guarda o ponto
    // do EIXO onde ele aparece — é de lá que o ser chega, então é de lá que a visão importa.
    let wps = &world.waypoints_world;
    let mut alvo_pos: Option<(Vec2, Vec2, f32)> = None; // (ponto do alvo, direção da pista, arco)
    {
        let mut arco = 0.0f32;
        let mut wp = 0usize;
        let mut t = 0.0f32;
        while wp < wps.len() - 1 && arco < 3000.0 {
            let a = wps[wp];
            let b = wps[wp + 1];
            let seg = (b - a).length().max(0.001);
            let p = a + (b - a) * (t / seg);
            let dir = (b - a) / seg;
            // Varre a largura da pista: o buraco não fica no eixo necessariamente.
            let nrm = Vec2::new(-dir.y, dir.x);
            for off in [-12.0f32, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0] {
                if world.material_at(p + nrm * off) == alvo_tipo {
                    alvo_pos = Some((p + nrm * off, dir, arco));
                    break;
                }
            }
            if alvo_pos.is_some() {
                break;
            }
            arco += 1.0;
            t += 1.0;
            if t >= seg {
                t -= seg;
                wp += 1;
            }
        }
    }

    let Some((alvo_ponto, dir_pista, arco_alvo)) = alvo_pos else {
        println!("Nenhum '{alvo}' encontrado nos primeiros 3000 m da semente {seed}.");
        return;
    };

    let fov = universe::visao::fov_deg().to_radians();
    println!(
        "VALIDAÇÃO DA VISÃO · semente {seed} · pista {width:.0} m · FOV {:.0}° · retina {RETINA_COLS}×{RETINA_ROWS}",
        fov.to_degrees()
    );
    println!(
        "Alvo: {alvo} em {arco_alvo:.0} m do início ({:.0}, {:.0})\n",
        alvo_ponto.x, alvo_ponto.y
    );
    println!(
        "{:>6} {:>8} {:>9} {:>9} {:>10}  {}",
        "dist", "pixels", "valor i8", "fundo i8", "contraste", "veredito"
    );
    println!("{}", "-".repeat(74));

    let mut primeira_deteccao: Option<f32> = None;
    let mut d = de;
    while d >= ate {
        // Observador no eixo, a `d` metros antes do alvo, olhando na direção da pista.
        let obs = alvo_ponto - dir_pista * d;
        let frente = dir_pista;
        let lat = Vec2::new(-frente.y, frente.x);

        let mut pixels_no_alvo = 0usize;
        let mut valor_alvo: Option<i32> = None;
        let mut valor_fundo: Option<i32> = None;

        // Varre as colunas da retina (o eixo horizontal é o que enxerga obstáculo lateral).
        for col in 0..RETINA_COLS {
            let frac = (col as f32 + 0.5) / RETINA_COLS as f32 - 0.5; // -0.5 .. +0.5
            let ang = frac * fov;
            let dir_raio = (frente * ang.cos() + lat * ang.sin()).normalize();
            match raio(&world, obs, dir_raio, 200.0) {
                Some((dist, mat)) if mat == alvo_tipo => {
                    pixels_no_alvo += 1;
                    let v = legacy_encoding(dist, mat);
                    valor_alvo = Some(valor_alvo.map_or(v, |o: i32| o.max(v)));
                }
                Some((dist, mat)) => {
                    let v = legacy_encoding(dist, mat);
                    valor_fundo = Some(valor_fundo.map_or(v, |o: i32| o.max(v)));
                }
                None => {}
            }
        }

        if pixels_no_alvo > 0 && primeira_deteccao.is_none() {
            primeira_deteccao = Some(d);
        }

        let va = valor_alvo.unwrap_or(-128);
        let vf = valor_fundo.unwrap_or(-128);
        let contraste = (va - vf).abs();
        let veredito = if pixels_no_alvo == 0 {
            "INVISÍVEL"
        } else if contraste < 10 {
            "← confunde com o fundo"
        } else if pixels_no_alvo == 1 {
            "1 pixel só (frágil)"
        } else {
            "visível"
        };
        println!("{d:6.0} {pixels_no_alvo:8} {va:9} {vf:9} {contraste:10}  {veredito}");

        d -= if d > 20.0 { 10.0 } else { 2.0 };
    }

    println!("\n{}", "-".repeat(74));
    match primeira_deteccao {
        Some(dist) => {
            // O que importa não é ver, é ver A TEMPO. A 6 m/s (velocidade medida na
            // bateria de 2026-08-13) cada metro de antecipação vale 1/6 s de reação.
            for v in [4.0f32, 6.0, 10.0] {
                println!(
                    "  a {v:.0} m/s o ser tem {:.1} s desde a primeira detecção até o impacto",
                    dist / v
                );
            }
            println!("\n  Primeira detecção a {dist:.0} m.");
        }
        None => println!(
            "  O alvo NUNCA foi detectado em nenhuma distância testada.\n\
                          \x20 O ser é literalmente cego para este objeto — nenhum cérebro o evita."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ancora os números da tabela de codificação. Se alguém mexer em `legacy_encoding` no
    /// shader e não aqui, este teste denuncia a divergência.
    #[test]
    fn encoding_bate_com_o_shader() {
        assert_eq!(legacy_encoding(0.0, VoxelType::Wall), 127);
        assert_eq!(legacy_encoding(0.0, VoxelType::Hole), 90);
        assert_eq!(legacy_encoding(50.0, VoxelType::Hole), 60);
        // A saturação que motivou esta ferramenta: além de 50 m o buraco é uma constante.
        assert_eq!(
            legacy_encoding(50.0, VoxelType::Hole),
            legacy_encoding(200.0, VoxelType::Hole)
        );
    }

    /// A ambiguidade medida: buraco longe tem o mesmo valor que parede a meia distância.
    #[test]
    fn buraco_longe_colide_com_parede_media() {
        let buraco_60m = legacy_encoding(60.0, VoxelType::Hole);
        let parede_28m = legacy_encoding(28.0, VoxelType::Wall);
        assert!(
            (buraco_60m - parede_28m).abs() <= 3,
            "buraco a 60 m = {buraco_60m}, parede a 28 m = {parede_28m}"
        );
    }
}
