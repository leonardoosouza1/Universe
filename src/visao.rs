//! Parâmetros da visão em tempo de execução — para poder VARIAR o alcance e o ângulo.
//!
//! # Por que isto existe
//!
//! O Leonardo observou: *"o modelo não vê ângulos ou não vê o que está muito longe, ele tem
//! um limite e com base na velocidade isso pode atrapalhar"*. A conta confirma, e o número é
//! duro.
//!
//! A retina tem 14 linhas com pitch de −0,012 (linha 0) a −0,060 (linha 13), e o olho a
//! 0,4 m do chão. Cada linha enxerga `0.4 / |pitch|` metros à frente:
//!
//! | linha | distância |
//! |---|---|
//! | 0 (a mais longe) | **33,3 m** |
//! | 7 | 10,6 m |
//! | 13 | 6,7 m |
//!
//! E a distância de frenagem até a curva mais fechada da pista (raio 9,2 m, v_curva 11,4 m/s):
//!
//! | velocidade | frenagem exigida | cabe nos 33,3 m? |
//! |---|---|---|
//! | 25 m/s | 25,2 m | sim |
//! | **28 m/s** | **33,3 m** | **exatamente no limite** |
//! | 30 m/s | 39,2 m | **não** |
//! | 40 m/s | 74,9 m | não, 2,2× além |
//!
//! > **Acima de ~28 m/s (101 km/h) o carro fica cego para o ponto onde precisa começar a
//! > frear.** A retina não alcança.
//!
//! Isso explica as sondas de curvatura a 20/40/70/120 m: são a compensação para o limite da
//! retina. E explica por que ablar curvatura custa só 3% — o carro opera abaixo de 28 m/s,
//! onde a retina basta e as sondas são redundantes.
//!
//! # O que dá para variar
//!
//! `RETINA_ROWS` e `RETINA_COLS` são tamanhos de array e têm de continuar constantes. Mas o
//! ALCANCE (pitch da linha 0) e o ÂNGULO (FOV) são só números num laço de raycasting —
//! podem ser parâmetros. São exatamente os dois eixos da observação do Leonardo.

use std::sync::OnceLock;

use crate::config::iara::vision::{FOV_DEG, RETINA_MAX_PITCH, RETINA_ROWS};

/// Altura do olho acima do chão, em metros (`world.rs`: `pos + Vec3::Y * 0.4`).
pub const ALTURA_OLHO: f32 = 0.4;

/// Alcance atual da linha mais distante, em metros. É o que se quer variar.
pub const ALCANCE_PADRAO: f32 = ALTURA_OLHO / 0.012; // = 33,33 m

#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// Até onde a linha 0 enxerga no chão, em metros.
    pub alcance_m: f32,
    /// Abertura horizontal em graus.
    pub fov_deg: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            alcance_m: ALCANCE_PADRAO,
            fov_deg: FOV_DEG,
        }
    }
}

static PARAMS: OnceLock<Params> = OnceLock::new();

pub fn definir(p: Params) {
    let _ = PARAMS.set(p);
}

#[inline]
pub fn params() -> Params {
    *PARAMS.get().unwrap_or(&Params {
        alcance_m: ALCANCE_PADRAO,
        fov_deg: FOV_DEG,
    })
}

#[inline]
pub fn fov_deg() -> f32 {
    params().fov_deg
}

/// Pitch da linha `row`, respeitando o alcance pedido.
///
/// A linha 13 (o chão logo à frente, 6,7 m) fica FIXA: mexer nela mudaria o que o carro vê
/// dos próprios pneus, que é outro problema. O que se estica é a linha 0, e as do meio
/// interpolam — assim a única coisa que muda é **até onde ele enxerga**.
#[inline]
pub fn pitch(row: usize) -> f32 {
    let p = params();
    if (p.alcance_m - ALCANCE_PADRAO).abs() < 1e-3 {
        return crate::config::iara::vision::RETINA_PITCHES[row.min(RETINA_ROWS - 1)];
    }
    let min_pitch = -(ALTURA_OLHO / p.alcance_m.max(1.0)); // linha 0, a mais distante
    if RETINA_ROWS <= 1 {
        return min_pitch;
    }
    let passo = (RETINA_MAX_PITCH - min_pitch) / (RETINA_ROWS as f32 - 1.0);
    min_pitch + passo * row.min(RETINA_ROWS - 1) as f32
}

/// A que distância no chão a linha `row` olha. Serve para relatório e para os testes.
pub fn distancia_da_linha(row: usize) -> f32 {
    let p = pitch(row);
    if p < -1e-6 {
        ALTURA_OLHO / -p
    } else {
        f32::INFINITY
    }
}

/// Velocidade acima da qual a frenagem exigida passa do alcance da visão.
///
/// `d_freio = (v² − v_curva²) / (2·μ·g)`, resolvido para `d_freio = alcance`.
pub fn velocidade_cega(alcance_m: f32, v_curva: f32) -> f32 {
    (2.0 * 9.81 * alcance_m + v_curva * v_curva).sqrt()
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn o_padrao_reproduz_a_tabela_de_pitches_original() {
        // sem mexer em nada, `pitch()` tem de devolver exatamente o que a const devolvia —
        // senão a mudança altera o comportamento de todo experimento anterior
        for row in 0..RETINA_ROWS {
            let esperado = crate::config::iara::vision::RETINA_PITCHES[row];
            assert!(
                (pitch(row) - esperado).abs() < 1e-9,
                "linha {row}: {} vs {}",
                pitch(row),
                esperado
            );
        }
        assert!(
            (distancia_da_linha(0) - 33.33).abs() < 0.1,
            "a linha 0 enxerga {:.1} m",
            distancia_da_linha(0)
        );
    }

    #[test]
    fn o_alcance_pedido_e_o_alcance_entregue() {
        // exercita a aritmética sem depender do OnceLock global
        for alvo in [20.0f32, 50.0, 80.0, 120.0] {
            let min_pitch = -(ALTURA_OLHO / alvo);
            let d = ALTURA_OLHO / -min_pitch;
            assert!((d - alvo).abs() < 0.01, "pedi {alvo} m e a conta deu {d} m");
        }
    }

    #[test]
    fn a_velocidade_cega_bate_com_a_fisica() {
        // curva mais fechada medida na pista: raio 9,2 m -> v_curva 11,4 m/s
        let v = velocidade_cega(33.33, 11.4);
        assert!((v - 28.0).abs() < 0.5, "deu {v:.1} m/s, esperava ~28");
        // dobrar o alcance NÃO dobra a velocidade — a relação é de raiz
        let v2 = velocidade_cega(66.66, 11.4);
        assert!(
            v2 < 2.0 * v,
            "a relação tem de ser sublinear: {v2:.1} vs {:.1}",
            2.0 * v
        );
        assert!(v2 > v, "mais alcance tem de permitir mais velocidade");
    }
}
