//! Geração de pistas por DIFICULDADE MEDIDA — o currículo que não existia.
//!
//! # Por que isto existe
//!
//! As pistas de `world.rs` são listas de waypoints escritas à mão, rotuladas nos comentários
//! como EASY / HARD / EXTREME, e o campo `complexity` do `TrackProfile` **nunca é lido**.
//! Medindo a dificuldade de verdade (`src/bin/pistas.rs`), o quadro é outro:
//!
//! | pista | rótulo humano | demanda medida |
//! |---|---|---|
//! | Nürburgring | EXTREME | **0 m** — nem precisa frear |
//! | Silverstone | VERY HARD | 16 m |
//! | Procedural Sigma | (sem rótulo) | 33 m |
//!
//! O rótulo estava invertido, e as seis pistas à mão vivem todas numa faixa em que a sonda
//! de curvatura de 20 m já basta. Depois há um penhasco direto para as procedurais. Não é um
//! currículo — é um platô e um abismo.
//!
//! # Qual é o eixo certo de dificuldade
//!
//! Não é a largura. Medido: alargar de 8 para 24 m muda a distância percorrida em 27% e a
//! MORTALIDADE de 58/60 para 33/60 — largura compra margem de erro, não velocidade.
//!
//! Também não é só o raio da curva. Uma curva fechada logo depois de outra curva é fácil:
//! chega-se devagar nela de qualquer jeito. O que exige do cérebro é a **distância de
//! frenagem** na pior combinação da pista:
//!
//! ```text
//! v_curva = sqrt(μ·g·(r + largura/2))
//! d_freio = (v_entrada² − v_curva²) / (2·μ·g)
//! ```
//!
//! e `d_freio` diz **qual sonda de curvatura ainda chega a tempo** — o carro tem quatro, a
//! 20, 40, 70 e 120 m. Essa é a demanda cognitiva: quanto mais longe é preciso enxergar,
//! mais ruidosa é a informação com que se decide.
//!
//! # A consequência que decide o projeto
//!
//! Resolvendo `d_freio > 40 m` para o raio:
//!
//! ```text
//! a 30 m/s  ->  precisa de curva com r < 4.7 m   (irreal)
//! a 40 m/s  ->  basta curva com r < 76 m         (qualquer uma das atuais)
//! ```
//!
//! **Fechar mais as curvas quase não aumenta a demanda; deixar o carro correr na reta
//! aumenta muito.** Reta longa entrando em curva média é o caso duro. É por isso que o
//! gerador tem `reta_alvo` como primeiro parâmetro e não o raio.
//!
//! # Como se garante que a dificuldade é a pedida
//!
//! Por amostragem com rejeição: gera, **mede**, aceita se ficou na faixa. A dificuldade
//! reportada é sempre a medida, nunca a pedida — mesma disciplina do resto desta pesquisa
//! (medir o teto antes de acreditar no mecanismo).

use bevy::math::Vec2;

pub const CELULA: f32 = 3.0; // metros por célula (world.rs:57)
pub const G: f32 = 9.81;
pub const MU: f32 = 1.0;
/// As quatro sondas de curvatura do carro (canais 20, 78, 79, 80).
pub const SONDAS: [f32; 4] = [20.0, 40.0, 70.0, 120.0];

/// O que uma pista exige, medido — não rotulado.
#[derive(Debug, Clone, Copy)]
pub struct Dificuldade {
    pub comprimento: f32,
    pub r_min: f32,
    pub n_curvas: usize,
    /// Distância de frenagem exigida na pior combinação reta→curva da volta.
    pub d_freio: f32,
    /// Raio da curva que exige mais (não necessariamente a mais fechada).
    pub r_pior: f32,
    /// Velocidade de chegada naquela curva.
    pub v_pior: f32,
}

impl Dificuldade {
    /// A menor sonda que ainda chega a tempo. `None` = nenhuma alcança.
    pub fn sonda(&self) -> Option<f32> {
        if self.d_freio <= 0.0 {
            return Some(0.0);
        }
        SONDAS.iter().copied().find(|&x| x >= self.d_freio)
    }

    pub fn rotulo(&self) -> &'static str {
        match self.sonda() {
            Some(x) if x == 0.0 => "trivial (nem freia)",
            Some(x) if x <= 20.0 => "fácil (sonda de 20 m basta)",
            Some(x) if x <= 40.0 => "média (precisa da de 40 m)",
            Some(x) if x <= 70.0 => "difícil (precisa da de 70 m)",
            Some(_) => "extrema (precisa da de 120 m)",
            None => "impossível (nem a sonda mais longa alcança)",
        }
    }
}

/// Mede o que a pista exige, percorrendo-a com a mesma aderência que segura as curvas.
///
/// Duas voltas: a primeira acerta a velocidade de entrada de cada trecho (o começo é
/// otimista e a primeira curva corrige), a segunda mede.
pub fn medir(wps_celulas: &[Vec2], largura: f32, v_teto: f32) -> Dificuldade {
    let p: Vec<Vec2> = wps_celulas.iter().map(|w| *w * CELULA).collect();
    let n = p.len();
    let mut d = Dificuldade {
        comprimento: 0.0, r_min: f32::INFINITY, n_curvas: 0,
        d_freio: 0.0, r_pior: 0.0, v_pior: 0.0,
    };
    if n < 3 {
        return d;
    }
    for i in 0..n - 1 {
        d.comprimento += (p[i + 1] - p[i]).length();
    }

    let mut v = v_teto;
    for volta in 0..2 {
        for i in 1..n - 1 {
            let seg_ant = (p[i] - p[i - 1]).length();
            v = (v * v + 2.0 * MU * G * seg_ant).sqrt().min(v_teto);
            let a = (p[i] - p[i - 1]).normalize_or_zero();
            let b = (p[i + 1] - p[i]).normalize_or_zero();
            let ang = a.dot(b).clamp(-1.0, 1.0).acos();
            if ang.to_degrees() < 1.0 {
                continue;
            }
            let r = seg_ant.min((p[i + 1] - p[i]).length()) / ang;
            if volta == 1 {
                d.n_curvas += 1;
                d.r_min = d.r_min.min(r);
            }
            let v_curva = (MU * G * (r + largura / 2.0)).sqrt();
            if v > v_curva {
                let df = (v * v - v_curva * v_curva) / (2.0 * MU * G);
                if volta == 1 && df > d.d_freio {
                    d.d_freio = df;
                    d.r_pior = r;
                    d.v_pior = v;
                }
                v = v_curva;
            }
        }
    }
    if !d.r_min.is_finite() {
        d.r_min = 0.0;
    }
    d
}

/// Gerador simples e determinístico. Não uso `rand` para a pista ser reprodutível a partir
/// da semente — o `spawn.rs` já sofre de `rand::rng()` não semeado e isso custou repetições
/// caras; não vale repetir o erro no mundo.
struct Rng(u64);

impl Rng {
    fn novo(s: u64) -> Self {
        Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1)
    }
    fn proximo(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn faixa(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.proximo()
    }
}

/// Gera um traçado fechado com `n_vertices`, com `n_retas` trechos longos.
///
/// Os vértices ficam em torno de um centro, com raios e ângulos sorteados; ligar em ordem
/// angular fecha o circuito de graça. As "retas" saem de deixar um vão angular grande entre
/// dois vértices consecutivos; as "curvas", de agrupar vértices.
fn tracado(semente: u64, n_vertices: usize, n_retas: usize, raio_base: f32) -> Vec<Vec2> {
    let mut r = Rng::novo(semente);
    let centro = Vec2::new(120.0, 120.0); // grade 240×240 (world.rs)

    // fatias angulares: `n_retas` delas ganham peso extra e viram os trechos longos
    let mut pesos: Vec<f32> = (0..n_vertices).map(|_| r.faixa(0.6, 1.4)).collect();
    for k in 0..n_retas.min(n_vertices) {
        let i = (r.proximo() * n_vertices as f32) as usize % n_vertices;
        pesos[i] += 2.5 + 2.0 * (k as f32 / n_retas.max(1) as f32);
    }
    let soma: f32 = pesos.iter().sum();

    let mut pontos = Vec::with_capacity(n_vertices + 1);
    let mut acc = 0.0f32;
    for i in 0..n_vertices {
        acc += pesos[i] / soma * std::f32::consts::TAU;
        let raio = raio_base * r.faixa(0.78, 1.15);
        pontos.push(centro + Vec2::new(acc.cos(), acc.sin()) * raio);
    }
    pontos.push(pontos[0]); // fecha
    pontos
}

/// Gera uma pista cuja demanda MEDIDA cai na faixa pedida, por amostragem com rejeição.
///
/// Devolve `None` se não achou em `tentativas` — melhor devolver nada que devolver uma pista
/// e chamá-la de difícil sem ela ser.
pub fn gerar(
    semente: u64,
    d_freio_min: f32,
    d_freio_max: f32,
    largura: f32,
    v_teto: f32,
    tentativas: u32,
) -> Option<(Vec<Vec2>, Dificuldade)> {
    let mut melhor: Option<(Vec<Vec2>, Dificuldade)> = None;
    for t in 0..tentativas {
        let s = semente.wrapping_mul(2654435761).wrapping_add(t as u64);
        let mut r = Rng::novo(s);
        let n_vert = 6 + (r.proximo() * 10.0) as usize;
        let n_retas = 1 + (r.proximo() * 3.0) as usize;
        let raio = r.faixa(45.0, 85.0);
        let wps = tracado(s, n_vert, n_retas, raio);
        let d = medir(&wps, largura, v_teto);
        if d.comprimento < 600.0 || d.r_min < 6.0 {
            continue; // pista curta demais ou com curva fisicamente absurda
        }
        if d.d_freio >= d_freio_min && d.d_freio <= d_freio_max {
            return Some((wps, d));
        }
        // guarda a mais próxima do alvo, só para diagnóstico de por que falhou
        let alvo = (d_freio_min + d_freio_max) * 0.5;
        if melhor.as_ref().map_or(true, |(_, m)| {
            (d.d_freio - alvo).abs() < (m.d_freio - alvo).abs()
        }) {
            melhor = Some((wps, d));
        }
    }
    let _ = melhor; // a mais próxima NÃO é devolvida: entregar quase-difícil como difícil
                    // é exatamente o tipo de mentira silenciosa que este projeto evita
    None
}

/// Perfis ESTRUTURAIS de pista (pedido do Leonardo 2026-08-15: "pistas realmente
/// diferentes, não só uma rotina da outra — curta, só curva, com retão").
/// Cada perfil muda os parâmetros do `tracado`, não só a semente:
///   curta:  raio_base 25-40 m, 6-9 vértices  → volta ~1200-2000 m
///   curvas: raio_base 35-55 m, 12-16 vértices, 0-1 retas → curva atrás de curva
///   retao:  raio_base 75-110 m, 5-7 vértices, 3-4 retas  → retas longas, poucas curvas
/// Guardas de sempre: r_min ≥ 6 m (físico) e comprimento ≥ 600 m.
pub fn gerar_perfil(semente: u64, perfil: &str, largura: f32) -> Option<Vec<Vec2>> {
    for t in 0..200u64 {
        let s = semente.wrapping_mul(2654435761).wrapping_add(t);
        let mut r = Rng::novo(s);
        let (n_vert, n_retas, raio) = match perfil {
            "curta"  => (6 + (r.proximo() * 4.0) as usize, 1, r.faixa(25.0, 40.0)),
            "curvas" => (12 + (r.proximo() * 5.0) as usize, (r.proximo() * 2.0) as usize, r.faixa(35.0, 55.0)),
            "retao"  => (5 + (r.proximo() * 3.0) as usize, 3 + (r.proximo() * 2.0) as usize, r.faixa(75.0, 110.0)),
            _ => return None,
        };
        let wps = tracado(s, n_vert, n_retas, raio);
        let d = medir(&wps, largura, 40.0);
        if d.comprimento < 600.0 || d.r_min < 6.0 { continue; }
        // curta de verdade: rejeita volta acima de 2200 m
        if perfil == "curta" && d.comprimento > 2200.0 { continue; }
        return Some(wps);
    }
    None
}

/// Faixas de `d_freio` de cada degrau do currículo, escolhidas para que cada uma exija uma
/// sonda de curvatura mais longa que a anterior (o carro tem sondas a 20, 40, 70 e 120 m).
pub const DEGRAUS: [(f32, f32); 4] = [(2.0, 18.0), (22.0, 38.0), (45.0, 68.0), (75.0, 115.0)];

/// Pista do currículo: degrau 1..4, variante para ter várias no mesmo nível.
///
/// Devolve `None` quando não existe geometria naquela faixa com o teto de velocidade
/// suposto — o que é informação, não falha. Pela conta em `medir`, `d_freio` cresce com o
/// QUADRADO da velocidade de reta e só linearmente com o raio, então os degraus altos podem
/// ser inalcançáveis enquanto o carro não subir de marcha. Entregar uma pista mais fácil e
/// chamá-la de difícil seria a mentira silenciosa que este projeto evita.
pub fn do_curriculo(degrau: usize, variante: u64) -> Option<Vec<Vec2>> {
    let (lo, hi) = DEGRAUS[degrau.clamp(1, 4) - 1];
    // teto de velocidade suposto: 40 m/s, que é o que a 3ª marcha alcança (physics.rs)
    for tentativa in 0..40u64 {
        let s = (variante + 1)
            .wrapping_mul(7919)
            .wrapping_add(tentativa.wrapping_mul(104_729))
            .wrapping_add(degrau as u64 * 15_485_863);
        if let Some((wps, _)) = gerar(s, lo, hi, 14.0, 40.0, 300) {
            return Some(wps);
        }
    }
    None
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn medir_reconhece_pista_facil_e_dificil() {
        // quadrado grande: curvas abertas, o carro nem precisa frear
        let facil = vec![
            Vec2::new(40.0, 40.0), Vec2::new(200.0, 40.0),
            Vec2::new(200.0, 200.0), Vec2::new(40.0, 200.0),
            Vec2::new(40.0, 40.0),
        ];
        let df = medir(&facil, 14.0, 30.0);
        assert!(df.comprimento > 1000.0, "compr={}", df.comprimento);

        // zigue-zague apertado: mesma extensão, curvas muito mais fechadas
        let dificil: Vec<Vec2> = (0..20)
            .map(|i| Vec2::new(40.0 + i as f32 * 8.0, if i % 2 == 0 { 60.0 } else { 90.0 }))
            .collect();
        let dd = medir(&dificil, 14.0, 30.0);
        assert!(dd.r_min < df.r_min,
                "o zigue-zague tem de ter raio menor: {} vs {}", dd.r_min, df.r_min);
        assert!(dd.d_freio > df.d_freio,
                "e exigir mais frenagem: {} vs {}", dd.d_freio, df.d_freio);
    }

    #[test]
    fn reta_longa_exige_mais_que_curva_fechada_sozinha() {
        // a mesma curva de 90°, uma logo depois de outra curva, outra no fim de uma reta
        let sem_reta = vec![
            Vec2::new(100.0, 100.0), Vec2::new(115.0, 100.0),
            Vec2::new(115.0, 115.0), Vec2::new(100.0, 115.0),
            Vec2::new(100.0, 100.0),
        ];
        let com_reta = vec![
            Vec2::new(20.0, 100.0), Vec2::new(200.0, 100.0),
            Vec2::new(200.0, 115.0), Vec2::new(20.0, 115.0),
            Vec2::new(20.0, 100.0),
        ];
        let a = medir(&sem_reta, 14.0, 40.0);
        let b = medir(&com_reta, 14.0, 40.0);
        assert!(b.d_freio > a.d_freio,
                "reta longa tem de exigir mais frenagem: {} vs {}", b.d_freio, a.d_freio);
    }

    #[test]
    fn gerar_devolve_so_o_que_esta_na_faixa_pedida() {
        // se achar, a dificuldade MEDIDA tem de estar dentro do pedido — nunca "quase"
        if let Some((_, d)) = gerar(7, 10.0, 30.0, 14.0, 35.0, 400) {
            assert!(d.d_freio >= 10.0 && d.d_freio <= 30.0,
                    "devolveu fora da faixa: {}", d.d_freio);
        }
    }
}

#[cfg(test)]
mod testes_curriculo {
    use super::*;

    #[test]
    fn cada_degrau_achado_esta_na_faixa_que_promete() {
        for degrau in 1..=4usize {
            if let Some(wps) = do_curriculo(degrau, 0) {
                let d = medir(&wps, 14.0, 40.0);
                let (lo, hi) = DEGRAUS[degrau - 1];
                assert!(
                    d.d_freio >= lo && d.d_freio <= hi,
                    "degrau {degrau} devolveu pista com d_freio={:.1}, fora de [{lo}, {hi}]",
                    d.d_freio
                );
            }
        }
    }

    #[test]
    fn degraus_sao_deterministicos() {
        // duas chamadas iguais têm de dar a MESMA pista: o mundo não pode variar entre
        // corridas do mesmo experimento (o `spawn.rs` já sofre disso com rand::rng())
        let a = do_curriculo(2, 3);
        let b = do_curriculo(2, 3);
        match (a, b) {
            (Some(x), Some(y)) => {
                assert_eq!(x.len(), y.len());
                for (p, q) in x.iter().zip(y.iter()) {
                    assert!((p.x - q.x).abs() < 1e-6 && (p.y - q.y).abs() < 1e-6);
                }
            }
            (None, None) => {}
            _ => panic!("mesma entrada deu resultados diferentes"),
        }
    }

    #[test]
    fn variantes_do_mesmo_degrau_sao_pistas_diferentes() {
        if let (Some(a), Some(b)) = (do_curriculo(2, 0), do_curriculo(2, 1)) {
            let iguais = a.len() == b.len()
                && a.iter().zip(b.iter()).all(|(p, q)| (p.x - q.x).abs() < 1e-6);
            assert!(!iguais, "variantes 0 e 1 do degrau 2 deram a mesma pista");
        }
    }
}
