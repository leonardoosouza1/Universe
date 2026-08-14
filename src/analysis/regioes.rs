//! Regiões da pista × ativação — o cérebro usa neurônios diferentes em reta e em curva?
//!
//! # Por que existe
//!
//! Leonardo: *"o cérebro vai trabalhar em dimensões, neurônios e partes diferentes
//! conforme ele está na pista, e isso precisa ser medido."*
//!
//! Nunca cruzamos os dois lados que já existiam separados:
//!   - amostras de ativação (`--ativacao`), coletadas em `gpu_evolve/context.rs`
//!     (`l1_samples`/`l2_samples`) — *o que* o cérebro faz;
//!   - geometria da pista, na mesma definição discreta de curvatura κ que
//!     `src/bin/replay_mapa.rs::Pista::medir` usa para `ω_exigido = v·κ`
//!     (ângulo com sinal entre segmentos consecutivos ÷ comprimento médio) — *onde*
//!     o ser está. Reimplementada aqui (não importada de lá) porque `replay_mapa` é
//!     um binário, não uma lib — a fórmula é pequena o bastante para não justificar
//!     um crate-ponte só para isto.
//!
//! Este módulo só enxerga números: quem gera o par (amostra de ativação, contexto
//! físico) alinhado por índice é o caller (`GpuCtx::tick`, que empilha os dois na
//! MESMA iteração — ver `regiao_l1`/`regiao_l2` em `context.rs`).
//!
//! A pergunta de coesão ("os neurônios especializados em curva disparam JUNTOS?")
//! reusa `contextual::coativacao` — é exatamente a mesma pergunta de E3
//! (excesso de co-ativação sobre independência), só que restrita aos ticks de curva
//! e aos neurônios que passaram no limiar de especialização. Não reimplementada.

use crate::analysis::contextual;
use crate::math::stats;

/// Curvatura κ (1/m) por waypoint do eixo da pista, mais busca do ponto mais próximo
/// de uma posição arbitrária. É a MESMA definição discreta de `replay_mapa.rs::Pista`:
/// κ = ângulo com sinal entre os dois segmentos que tocam o waypoint, dividido pelo
/// comprimento médio dos dois — "nada estimado" (comentário original).
#[derive(Debug, Clone)]
pub struct CurvaturaPista {
    pts: Vec<(f32, f32)>,
    /// κ COM SINAL por waypoint — o sinal não importa para reta-vs-curva (usamos
    /// `.abs()` em `curvatura_em`), mas fica guardado bruto para eventual uso futuro
    /// (ex.: distinguir esquerda/direita).
    kappa: Vec<f32>,
}

impl CurvaturaPista {
    /// `waypoints_xz`: eixo da pista em coordenadas de mundo (x, z), fechado ou não
    /// (funciona igual — o traçado é tratado como ciclo via `% n`, que é seguro mesmo
    /// em lista aberta porque o ponto de fechamento tem o menor peso na curvatura
    /// de tudo mais, e nenhum ser passa perto dele de qualquer forma se a pista for
    /// aberta). Menos de 3 pontos = pista degenerada, todas as curvaturas ficam 0.
    pub fn medir(waypoints_xz: &[(f32, f32)]) -> Self {
        let n = waypoints_xz.len();
        if n < 3 {
            return Self { pts: waypoints_xz.to_vec(), kappa: vec![0.0; n] };
        }
        let mut kappa = vec![0.0f32; n];
        for i in 0..n {
            let a = waypoints_xz[(i + n - 1) % n];
            let b = waypoints_xz[i];
            let c = waypoints_xz[(i + 1) % n];
            let v1 = (b.0 - a.0, b.1 - a.1);
            let v2 = (c.0 - b.0, c.1 - b.1);
            let l1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt().max(1e-3);
            let l2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt().max(1e-3);
            let cross = v1.0 * v2.1 - v1.1 * v2.0;
            let dot = v1.0 * v2.0 + v1.1 * v2.1;
            let ang = cross.atan2(dot);
            kappa[i] = ang / (0.5 * (l1 + l2));
        }
        Self { pts: waypoints_xz.to_vec(), kappa }
    }

    /// |κ| (1/m) do waypoint mais próximo de `(x, z)`. Busca linear — só é chamada nos
    /// ticks amostrados (stride de `--ativacao-n`, tipicamente 1 em 64), nunca no laço
    /// quente de física; com algumas centenas de waypoints o custo é irrelevante.
    pub fn curvatura_em(&self, x: f32, z: f32) -> f32 {
        if self.pts.is_empty() {
            return 0.0;
        }
        let mut melhor = 0usize;
        let mut best_d = f32::MAX;
        for (i, p) in self.pts.iter().enumerate() {
            let d = (p.0 - x).powi(2) + (p.1 - z).powi(2);
            if d < best_d {
                best_d = d;
                melhor = i;
            }
        }
        self.kappa[melhor].abs()
    }
}

/// Contexto físico de UMA amostra de ativação — mesmo índice do vetor de amostras
/// (`l1_samples[i]` ↔ `regiao_l1[i]`). O alinhamento é responsabilidade do caller:
/// este tipo não carrega tick nem ser, só os três números que o protocolo pediu.
#[derive(Debug, Clone, Copy, Default)]
pub struct ContextoAmostra {
    /// |κ| local da pista (1/m) na posição do ser neste tick. 0 = reta perfeita.
    pub curvatura: f32,
    /// Velocidade escalar (m/s).
    pub velocidade: f32,
    /// Menor distância a qualquer parede visível (m) — min(esquerda, direita, frente).
    pub dist_parede: f32,
}

/// Métricas de UM neurônio: quanto ele muda entre reta e curva.
#[derive(Debug, Clone)]
pub struct EspecializacaoNeuronio {
    pub indice: usize,
    /// Média de |ativação| nos ticks classificados como RETA.
    pub media_reta: f32,
    /// Média de |ativação| nos ticks classificados como CURVA.
    pub media_curva: f32,
    /// |media_curva - media_reta| / (media_curva + media_reta + eps). ~1 = só dispara
    /// num dos dois regimes; ~0 = dispara igual nos dois.
    pub indice_especializacao: f32,
    /// Pearson entre |ativação| do neurônio e curvatura local, sobre TODAS as amostras.
    pub correlacao_curvatura: f32,
    /// Idem, contra velocidade.
    pub correlacao_velocidade: f32,
    /// Idem, contra distância à parede.
    pub correlacao_dist_parede: f32,
}

/// Relatório completo de uma camada (L1 ou L2) para uma geração.
#[derive(Debug, Clone)]
pub struct RelatorioRegioes {
    pub camada: String,
    /// |κ| acima disso = "em curva". SEMPRE reportado — é uma escolha, e escolha que
    /// muda o resultado tem de aparecer no relatório (mesma armadilha nº1 do
    /// protocolo de `contextual.rs`: nada aqui zera exatamente).
    pub limiar_curva: f32,
    pub n_reta: usize,
    pub n_curva: usize,
    pub por_neuronio: Vec<EspecializacaoNeuronio>,
    /// Limiar de `indice_especializacao` acima do qual um neurônio conta como
    /// "especializado". SEMPRE reportado, mesmo motivo do `limiar_curva`.
    pub limiar_especializacao: f32,
    pub indices_especializados: Vec<usize>,
    /// Co-ativação (E3 de `contextual.rs`) restrita aos ticks de CURVA e às colunas
    /// especializadas. `None` quando há menos de 2 especializados ou menos de 2
    /// amostras de curva — coativação entre 0 ou 1 neurônio não é uma pergunta válida.
    pub coesao: Option<contextual::CoAtivacao>,
}

const EPS: f32 = 1e-6;

/// Pearson entre `xs` e `ys` (mesmo comprimento, truncado ao menor se divergir).
/// Genérico o bastante para não ter nome de domínio — mesma convenção de
/// `crate::math::stats` (funções puras, sem semântica de cérebro).
fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let mx = stats::mean(&xs[..n]);
    let my = stats::mean(&ys[..n]);
    let mut cov = 0.0f64;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    for i in 0..n {
        let dx = (xs[i] - mx) as f64;
        let dy = (ys[i] - my) as f64;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 1e-12 || vy <= 1e-12 {
        return 0.0; // uma das duas séries é constante — correlação indefinida, não zero-por-acaso
    }
    (cov / (vx.sqrt() * vy.sqrt())) as f32
}

/// Cruza amostras de ativação de UMA camada com o contexto físico alinhado.
///
/// `amostras[t]` e `contexto[t]` têm de vir do MESMO tick/ser — é o caller
/// (`GpuCtx::tick`) que garante isso empilhando os dois juntos. `None` se os
/// comprimentos não baterem (desalinhamento é bug de coleta, não dado a analisar
/// às cegas) ou se não houver amostra nenhuma.
pub fn analisar(
    camada: &str,
    amostras: &[Vec<f32>],
    contexto: &[ContextoAmostra],
    limiar_curva: f32,
    limiar_especializacao: f32,
    limiar_ativacao_coesao: f32,
) -> Option<RelatorioRegioes> {
    if amostras.is_empty() || amostras.len() != contexto.len() {
        return None;
    }
    let n_neur = amostras[0].len();
    if n_neur == 0 {
        return None;
    }

    let mut reta_idx = Vec::new();
    let mut curva_idx = Vec::new();
    for (t, c) in contexto.iter().enumerate() {
        if c.curvatura >= limiar_curva {
            curva_idx.push(t);
        } else {
            reta_idx.push(t);
        }
    }

    let media_abs = |idxs: &[usize], k: usize| -> f32 {
        if idxs.is_empty() {
            return 0.0;
        }
        idxs.iter().map(|&t| amostras[t][k].abs()).sum::<f32>() / idxs.len() as f32
    };

    let curvaturas: Vec<f32> = contexto.iter().map(|c| c.curvatura).collect();
    let velocidades: Vec<f32> = contexto.iter().map(|c| c.velocidade).collect();
    let dist_paredes: Vec<f32> = contexto.iter().map(|c| c.dist_parede).collect();

    let mut por_neuronio = Vec::with_capacity(n_neur);
    for k in 0..n_neur {
        let media_reta = media_abs(&reta_idx, k);
        let media_curva = media_abs(&curva_idx, k);
        let indice_especializacao = (media_curva - media_reta).abs() / (media_curva + media_reta + EPS);
        let ativ_abs: Vec<f32> = amostras.iter().map(|a| a[k].abs()).collect();
        por_neuronio.push(EspecializacaoNeuronio {
            indice: k,
            media_reta,
            media_curva,
            indice_especializacao,
            correlacao_curvatura: pearson(&ativ_abs, &curvaturas),
            correlacao_velocidade: pearson(&ativ_abs, &velocidades),
            correlacao_dist_parede: pearson(&ativ_abs, &dist_paredes),
        });
    }

    let indices_especializados: Vec<usize> = por_neuronio.iter()
        .filter(|e| e.indice_especializacao >= limiar_especializacao)
        .map(|e| e.indice)
        .collect();

    // Coesão: os especializados disparam JUNTOS durante a curva, ou cada um decide
    // por conta própria? Restringe amostras às linhas de CURVA e colunas
    // especializadas, e reusa `contextual::coativacao` — mesma pergunta de E3,
    // domínio menor.
    let coesao = if indices_especializados.len() >= 2 && curva_idx.len() >= 2 {
        let submatriz: Vec<Vec<f32>> = curva_idx.iter()
            .map(|&t| indices_especializados.iter().map(|&k| amostras[t][k]).collect())
            .collect();
        Some(contextual::coativacao(camada, &submatriz, limiar_ativacao_coesao, 5))
    } else {
        None
    };

    Some(RelatorioRegioes {
        camada: camada.to_string(),
        limiar_curva,
        n_reta: reta_idx.len(),
        n_curva: curva_idx.len(),
        por_neuronio,
        limiar_especializacao,
        indices_especializados,
        coesao,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CurvaturaPista ───────────────────────────────────────────────────────

    #[test]
    fn reta_tem_curvatura_proxima_de_zero() {
        // Pontos colineares — nenhum ângulo entre segmentos consecutivos.
        let pts: Vec<(f32, f32)> = (0..10).map(|i| (i as f32, 0.0)).collect();
        let c = CurvaturaPista::medir(&pts);
        assert!(c.curvatura_em(5.0, 0.0) < 1e-4, "reta deveria ter kappa ~0");
    }

    #[test]
    fn curva_fechada_tem_curvatura_bem_maior_que_reta() {
        // Um "L": segmento reto de (0,0) a (10,0), depois vira 90° reto para (10,10).
        // O vértice (10,0) tem uma quina de 90°; um ponto no meio da reta não tem.
        let mut pts: Vec<(f32, f32)> = (0..=10).map(|i| (i as f32, 0.0)).collect();
        pts.extend((1..=10).map(|i| (10.0, i as f32)));
        let c = CurvaturaPista::medir(&pts);
        let k_reta = c.curvatura_em(5.0, 0.0);
        let k_curva = c.curvatura_em(10.0, 0.0); // a quina
        assert!(k_curva > k_reta * 10.0,
            "quina deveria ser bem mais curva que a reta: reta={k_reta} curva={k_curva}");
    }

    #[test]
    fn pista_vazia_nao_quebra() {
        let c = CurvaturaPista::medir(&[]);
        assert_eq!(c.curvatura_em(0.0, 0.0), 0.0);
    }

    // ── analisar() ───────────────────────────────────────────────────────────

    /// Constrói `contexto` alternando reta/curva por índice: pares = reta (curvatura
    /// 0.0), ímpares = curva (curvatura 1.0). Mesma convenção nos dois testes centrais
    /// do protocolo (item 2 da tarefa).
    fn contexto_alternado(n: usize) -> Vec<ContextoAmostra> {
        (0..n).map(|t| ContextoAmostra {
            curvatura: if t % 2 == 0 { 0.0 } else { 1.0 },
            velocidade: 10.0,
            dist_parede: 5.0,
        }).collect()
    }

    #[test]
    fn neuronio_so_dispara_em_curva_tem_indice_perto_de_1() {
        let n = 20;
        let contexto = contexto_alternado(n);
        // Neurônio 0: 0.0 na reta, 1.0 na curva. Neurônio 1: constante 0.3 (controle).
        let amostras: Vec<Vec<f32>> = (0..n).map(|t| {
            let em_curva = t % 2 == 1;
            vec![if em_curva { 1.0 } else { 0.0 }, 0.3]
        }).collect();
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.5, 0.05).unwrap();
        assert!(r.por_neuronio[0].indice_especializacao > 0.95,
            "esperava ~1, veio {}", r.por_neuronio[0].indice_especializacao);
    }

    #[test]
    fn neuronio_dispara_igual_nos_dois_tem_indice_perto_de_0() {
        let n = 20;
        let contexto = contexto_alternado(n);
        let amostras: Vec<Vec<f32>> = (0..n).map(|_| vec![0.7, 0.5]).collect();
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.5, 0.05).unwrap();
        assert!(r.por_neuronio[0].indice_especializacao < 0.05,
            "esperava ~0, veio {}", r.por_neuronio[0].indice_especializacao);
    }

    #[test]
    fn conta_especializados_acima_do_limiar_reportado() {
        let n = 20;
        let contexto = contexto_alternado(n);
        // 3 neurônios: 0 especializado em curva, 1 especializado em reta, 2 neutro.
        let amostras: Vec<Vec<f32>> = (0..n).map(|t| {
            let em_curva = t % 2 == 1;
            vec![
                if em_curva { 1.0 } else { 0.0 },
                if em_curva { 0.0 } else { 1.0 },
                0.5,
            ]
        }).collect();
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.8, 0.05).unwrap();
        assert_eq!(r.limiar_especializacao, 0.8);
        assert_eq!(r.indices_especializados.len(), 2);
        assert!(r.indices_especializados.contains(&0));
        assert!(r.indices_especializados.contains(&1));
        assert!(!r.indices_especializados.contains(&2));
    }

    #[test]
    fn correlacao_curvatura_positiva_para_neuronio_alinhado_com_curva() {
        let n = 40;
        let contexto = contexto_alternado(n);
        let amostras: Vec<Vec<f32>> = (0..n).map(|t| {
            vec![contexto[t].curvatura] // ativação = a própria curvatura → corr = 1
        }).collect();
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.5, 0.05).unwrap();
        assert!(r.por_neuronio[0].correlacao_curvatura > 0.99,
            "correlacao deveria ser ~1, veio {}", r.por_neuronio[0].correlacao_curvatura);
    }

    #[test]
    fn coesao_none_quando_menos_de_2_especializados() {
        let n = 20;
        let contexto = contexto_alternado(n);
        let amostras: Vec<Vec<f32>> = (0..n).map(|_| vec![0.5, 0.5]).collect();
        // limiar altíssimo — ninguém passa.
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.99, 0.05).unwrap();
        assert!(r.indices_especializados.is_empty());
        assert!(r.coesao.is_none());
    }

    #[test]
    fn coesao_detecta_especializados_que_disparam_juntos_na_curva() {
        let n = 80;
        let contexto = contexto_alternado(n);
        // Neurônios 0 e 1: especializados em curva E acoplados — sempre JUNTOS quando
        // disparam, mas não em TODA curva (senão ambos ficam com marginal 1.0 dentro
        // do subconjunto de curva e a co-ativação vira idêntica à independência —
        // 1.0×1.0=1.0 —, exatamente o caso que `ganho_temporal_zero_quando_tudo_sempre_ativo`
        // de contextual.rs já cobre para persistência; aqui precisa de VARIAÇÃO dentro
        // da curva para a coesão ter algo a detectar, igual
        // `coativacao_detecta_par_acoplado`). A cada 4 ticks de curva, 1 fica "frio".
        let mut cnt_curva = 0usize;
        let amostras: Vec<Vec<f32>> = (0..n).map(|t| {
            let em_curva = t % 2 == 1;
            if em_curva {
                let ligado = cnt_curva % 4 != 0;
                cnt_curva += 1;
                let v = if ligado { 1.0 } else { 0.0 };
                vec![v, v, 0.1]
            } else {
                vec![0.0, 0.0, 0.1]
            }
        }).collect();
        let r = analisar("teste", &amostras, &contexto, 0.5, 0.5, 0.5).unwrap();
        assert_eq!(r.indices_especializados.len(), 2);
        let co = r.coesao.expect("deveria ter coesao com 2 especializados");
        assert!(co.excesso > 0.0, "excesso deveria ser positivo, veio {}", co.excesso);
    }

    #[test]
    fn amostras_e_contexto_desalinhados_retorna_none() {
        let amostras = vec![vec![0.0; 2]; 5];
        let contexto = contexto_alternado(3);
        assert!(analisar("teste", &amostras, &contexto, 0.5, 0.5, 0.05).is_none());
    }
}
