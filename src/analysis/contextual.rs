//! Esparsidade contextual — *quem* dispara para *esta* situação.
//!
//! # Por que existe
//!
//! Leonardo, entregando o protocolo de esparsidade contextual: *"você está tentando achar
//! algo que calcule o valor do peso do neurônio, né? Por que não tenta de outras formas, e
//! para cada tipo de parte do cérebro?"*
//!
//! A pergunta que eu vinha perseguindo — *qual neurônio importa* — está **refutada** no
//! documento dele (§5): com os 8 neurônios mais "importantes" de 128 o desempenho fica
//! PIOR que o baseline; é preciso ~50% deles para chegar perto do total. A informação é
//! distribuída. Selecionar subconjunto fixo não funciona; o que funciona é projetar.
//!
//! Os nossos próprios números já diziam isso e eu não li assim: atenção uniforme
//! (0,52-0,66% por canal), rank efetivo 75/96, zero neurônios surdos ou mudos.
//!
//! A distinção que salva a análise, e que dá nome a este módulo:
//!
//! · **esparsidade estática** — quais neurônios são globalmente importantes → NÃO existe
//! · **esparsidade contextual** — quais disparam *para este tick* → existe e é explorável
//!
//! # A régua muda com a aritmética
//!
//! O cérebro tem seis partes com naturezas numéricas distintas, e medir todas com a mesma
//! régua foi o erro anterior (valor singular aplicado ao cérebro inteiro):
//!
//! | parte | aritmética | o que perguntar |
//! |---|---|---|
//! | KAN sensorial | inteira (i32) | quantos knots saíram da identidade |
//! | CfC | inteira + `decay = exp(-d·dt)` diagonal | persistência por τ, zona morta |
//! | L1 | pesos f32 × entrada i8 ÷127 | ganho σ_max, esparsidade contextual |
//! | L2 | f32 puro | esparsidade, saturação |
//! | bypass L1→saída | f32, pula L2 | metade da saída não passa por L2 |
//! | banco de τ | f32 com semântica de TEMPO | distribuição log, correlação com persistência |
//!
//! # Os quatro experimentos
//!
//! Traduzidos do protocolo (E1-E4), na ordem de execução que ele define — E2 primeiro
//! porque, se a esparsidade real for baixa, o resto perde sentido.
//!
//! **Armadilha nº 1 do protocolo, respeitada aqui:** a não-linearidade é `tanh`, que não
//! zera exatamente. O limiar é parâmetro explícito e **sempre reportado** junto do
//! resultado; `sensibilidade_ao_limiar` mede quanto a conclusão depende dele.

use crate::math::stats;

/// Fração de neurônios ativos por amostra, e a distribuição dessa fração.
///
/// E2 do protocolo. Responde: o cérebro é esparso de verdade?
#[derive(Debug, Clone)]
pub struct Esparsidade {
    pub camada: String,
    /// Limiar usado — SEMPRE reportado junto, porque muda todos os números.
    pub limiar: f32,
    /// Média da fração ativa por tick (0..1).
    pub fracao_ativa_media: f32,
    /// Desvio da fração ativa entre ticks — variação indica que o conjunto ativo responde
    /// ao contexto; constante indica que dispara sempre o mesmo tanto.
    pub fracao_ativa_sigma: f32,
    pub p5: f32,
    pub p95: f32,
    /// Fração de amostras em que a camada estava saturada (|h| > 0,99) — o oposto de
    /// esparsa. L2 mediu 57,5% aqui, o que é defeito, não esparsidade.
    pub saturacao_media: f32,
}

/// Quanto o conjunto ativo persiste entre ticks consecutivos.
///
/// E1 do protocolo — a ideia potencialmente nova. No nosso caso há previsão teórica: o
/// CfC é literalmente um estado com inércia (`s = s·decay + x·gain`), então canais de τ
/// longo devem persistir mais que os de τ curto. Se isso aparecer, é a primeira validação
/// empírica de que o banco de τ faz o que promete.
#[derive(Debug, Clone)]
pub struct Persistencia {
    pub camada: String,
    pub limiar: f32,
    /// |A[t] ∩ A[t-1]| / |A[t]| — média sobre os ticks.
    pub persistencia: f32,
    /// Mesmo cálculo contra uma amostra ALEATÓRIA (não adjacente). É o baseline
    /// obrigatório: sem ele, persistência alta pode ser só "a camada dispara sempre".
    pub baseline_aleatorio: f32,
    /// `persistencia - baseline`. Critério de refutação do protocolo: abaixo de 10 pontos,
    /// a ideia do preditor incremental morre.
    pub ganho_temporal: f32,
    /// Fração média do conjunto ativo que MUDA por tick — o que um preditor incremental
    /// precisaria adivinhar.
    pub churn: f32,
}

/// Excesso de co-ativação sobre a independência estatística.
///
/// E3 do protocolo. É a base empírica para dividir o cérebro em regiões: os grupos devem
/// SAIR daqui (clusters de neurônios que disparam juntos), não de desenho a priori.
#[derive(Debug, Clone)]
pub struct CoAtivacao {
    pub camada: String,
    pub limiar: f32,
    /// P(i e j ativos) média entre pares.
    pub coativacao_media: f32,
    /// P(i)·P(j) média — o que se esperaria se cada um decidisse sozinho.
    pub esperado_independente: f32,
    /// Diferença. Positivo = existe estrutura de grupo.
    pub excesso: f32,
    /// Os pares mais acoplados: (i, j, excesso).
    pub pares_fortes: Vec<(usize, usize, f32)>,
}

/// Distribuição de frequência de ativação por neurônio.
///
/// E4 do protocolo. Cauda longa = poucos neurônios quentes carregam quase tudo.
#[derive(Debug, Clone)]
pub struct QuenteFrio {
    pub camada: String,
    pub limiar: f32,
    /// Frequência de ativação de cada neurônio (0..1), ordem original.
    pub frequencia: Vec<f32>,
    /// Quantos neurônios cobrem 80% de todas as ativações.
    pub neuronios_para_80pct: usize,
    /// Neurônios que nunca dispararam.
    pub nunca_ativos: usize,
    /// Neurônios que dispararam em mais de 95% dos ticks — sempre ligados são viés, não
    /// sensor: não carregam informação sobre o contexto.
    pub sempre_ativos: usize,
}

/// Máscara binária de atividade: `|h| > limiar`.
fn mascara(amostra: &[f32], limiar: f32) -> Vec<bool> {
    amostra.iter().map(|v| v.abs() > limiar).collect()
}

/// E2 — esparsidade real por camada.
pub fn esparsidade(camada: &str, amostras: &[Vec<f32>], limiar: f32) -> Esparsidade {
    let mut fracoes: Vec<f32> = Vec::with_capacity(amostras.len());
    let mut saturacoes: Vec<f32> = Vec::with_capacity(amostras.len());
    for a in amostras {
        if a.is_empty() { continue; }
        let ativos = a.iter().filter(|v| v.abs() > limiar).count();
        let sat = a.iter().filter(|v| v.abs() > 0.99).count();
        fracoes.push(ativos as f32 / a.len() as f32);
        saturacoes.push(sat as f32 / a.len() as f32);
    }
    let mut ord = fracoes.clone();
    Esparsidade {
        camada: camada.to_string(),
        limiar,
        fracao_ativa_media: stats::mean(&fracoes),
        fracao_ativa_sigma: stats::std_dev(&fracoes),
        p5: stats::percentile(&mut ord, 5.0),
        p95: stats::percentile(&mut ord.clone(), 95.0),
        saturacao_media: stats::mean(&saturacoes),
    }
}

/// E1 — persistência temporal do conjunto ativo.
///
/// `passo_aleatorio` define o deslocamento usado como baseline: comparar A[t] com
/// A[t + passo] em vez de A[t-1]. Um passo grande o bastante para não haver correlação
/// temporal, mas dentro da mesma corrida (mesma distribuição).
pub fn persistencia(camada: &str, amostras: &[Vec<f32>], limiar: f32, passo_aleatorio: usize) -> Persistencia {
    if amostras.len() < 2 {
        return Persistencia { camada: camada.to_string(), limiar,
            persistencia: 0.0, baseline_aleatorio: 0.0, ganho_temporal: 0.0, churn: 0.0 };
    }
    let mascaras: Vec<Vec<bool>> = amostras.iter().map(|a| mascara(a, limiar)).collect();

    let sobreposicao = |a: &[bool], b: &[bool]| -> Option<f32> {
        let n_a = a.iter().filter(|x| **x).count();
        if n_a == 0 { return None; }   // sem conjunto ativo, a razão é indefinida
        let inter = a.iter().zip(b).filter(|(x, y)| **x && **y).count();
        Some(inter as f32 / n_a as f32)
    };

    let mut adj: Vec<f32> = Vec::new();
    let mut ale: Vec<f32> = Vec::new();
    let passo = passo_aleatorio.max(2);
    for t in 1..mascaras.len() {
        if let Some(v) = sobreposicao(&mascaras[t], &mascaras[t - 1]) { adj.push(v); }
        let outro = (t + passo) % mascaras.len();
        if outro != t && outro != t - 1 {
            if let Some(v) = sobreposicao(&mascaras[t], &mascaras[outro]) { ale.push(v); }
        }
    }
    let p = stats::mean(&adj);
    let b = stats::mean(&ale);
    Persistencia {
        camada: camada.to_string(), limiar,
        persistencia: p,
        baseline_aleatorio: b,
        ganho_temporal: p - b,
        churn: 1.0 - p,
    }
}

/// E3 — co-ativação acima da independência.
///
/// Custo O(n² · amostras); com n=96 e milhares de amostras é barato o bastante para rodar
/// offline. `max_pares` limita só o relatório dos pares fortes.
pub fn coativacao(camada: &str, amostras: &[Vec<f32>], limiar: f32, max_pares: usize) -> CoAtivacao {
    let n = amostras.first().map_or(0, |a| a.len());
    if n == 0 || amostras.is_empty() {
        return CoAtivacao { camada: camada.to_string(), limiar, coativacao_media: 0.0,
            esperado_independente: 0.0, excesso: 0.0, pares_fortes: Vec::new() };
    }
    let mascaras: Vec<Vec<bool>> = amostras.iter().map(|a| mascara(a, limiar)).collect();
    let total = mascaras.len() as f32;

    // Marginais: P(i ativo).
    let marginal: Vec<f32> = (0..n)
        .map(|i| mascaras.iter().filter(|m| m.get(i).copied().unwrap_or(false)).count() as f32 / total)
        .collect();

    let mut soma_conj = 0.0f64;
    let mut soma_indep = 0.0f64;
    let mut pares: Vec<(usize, usize, f32)> = Vec::new();
    let mut n_pares = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let conj = mascaras.iter()
                .filter(|m| m.get(i).copied().unwrap_or(false) && m.get(j).copied().unwrap_or(false))
                .count() as f32 / total;
            let indep = marginal[i] * marginal[j];
            soma_conj += conj as f64;
            soma_indep += indep as f64;
            n_pares += 1;
            let exc = conj - indep;
            if exc.abs() > 0.05 { pares.push((i, j, exc)); }
        }
    }
    pares.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
    pares.truncate(max_pares);

    let d = n_pares.max(1) as f64;
    let cm = (soma_conj / d) as f32;
    let ei = (soma_indep / d) as f32;
    CoAtivacao {
        camada: camada.to_string(), limiar,
        coativacao_media: cm, esperado_independente: ei, excesso: cm - ei,
        pares_fortes: pares,
    }
}

/// E4 — neurônios quentes e frios.
pub fn quente_frio(camada: &str, amostras: &[Vec<f32>], limiar: f32) -> QuenteFrio {
    let n = amostras.first().map_or(0, |a| a.len());
    let total = amostras.len().max(1) as f32;
    let frequencia: Vec<f32> = (0..n)
        .map(|i| amostras.iter().filter(|a| a.get(i).map_or(false, |v| v.abs() > limiar)).count() as f32 / total)
        .collect();

    // Quantos neurônios cobrem 80% de TODAS as ativações (não 80% dos neurônios).
    let mut ord: Vec<f32> = frequencia.clone();
    ord.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let soma: f32 = ord.iter().sum();
    let alvo = soma * 0.8;
    let mut acc = 0.0f32;
    let mut k = 0usize;
    for f in &ord {
        acc += f; k += 1;
        if acc >= alvo { break; }
    }

    QuenteFrio {
        camada: camada.to_string(), limiar,
        nunca_ativos: frequencia.iter().filter(|f| **f == 0.0).count(),
        sempre_ativos: frequencia.iter().filter(|f| **f > 0.95).count(),
        neuronios_para_80pct: k,
        frequencia,
    }
}

/// Quanto a conclusão depende do limiar escolhido.
///
/// Armadilha nº 1 do protocolo: com `tanh` nada zera exatamente, então o limiar é uma
/// escolha, e escolha que muda o resultado precisa ser exposta. Devolve a fração ativa
/// para cada limiar testado.
pub fn sensibilidade_ao_limiar(amostras: &[Vec<f32>], limiares: &[f32]) -> Vec<(f32, f32)> {
    limiares.iter()
        .map(|&l| (l, esparsidade("_", amostras, l).fracao_ativa_media))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn amostras_fixas(padrao: &[&[f32]]) -> Vec<Vec<f32>> {
        padrao.iter().map(|p| p.to_vec()).collect()
    }

    #[test]
    fn esparsidade_conta_acima_do_limiar() {
        let a = amostras_fixas(&[&[1.0, 0.0, 0.0, 0.0], &[1.0, 1.0, 0.0, 0.0]]);
        let e = esparsidade("t", &a, 0.5);
        // 1/4 e 2/4 → média 0,375.
        assert!((e.fracao_ativa_media - 0.375).abs() < 1e-5);
    }

    #[test]
    fn persistencia_maxima_quando_conjunto_nao_muda() {
        let a = amostras_fixas(&[&[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0]]);
        let p = persistencia("t", &a, 0.5, 2);
        assert!((p.persistencia - 1.0).abs() < 1e-5);
        assert!((p.churn - 0.0).abs() < 1e-5);
    }

    #[test]
    fn persistencia_zero_quando_conjunto_alterna() {
        let a = amostras_fixas(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[0.0, 1.0]]);
        let p = persistencia("t", &a, 0.5, 2);
        assert!(p.persistencia < 1e-5, "persistência foi {}", p.persistencia);
    }

    /// O baseline é o que separa "persiste" de "dispara sempre". Uma camada com TODOS os
    /// neurônios sempre ativos tem persistência 1,0 — e baseline 1,0 também, então ganho
    /// temporal zero. Sem o baseline, essa camada pareceria o caso ideal.
    #[test]
    fn ganho_temporal_zero_quando_tudo_sempre_ativo() {
        let a = amostras_fixas(&[&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0]]);
        let p = persistencia("t", &a, 0.5, 2);
        assert!((p.persistencia - 1.0).abs() < 1e-5);
        assert!(p.ganho_temporal.abs() < 1e-5, "ganho foi {}", p.ganho_temporal);
    }

    #[test]
    fn coativacao_detecta_par_acoplado() {
        // 0 e 1 sempre juntos; 2 independente.
        let a = amostras_fixas(&[&[1.0, 1.0, 0.0], &[0.0, 0.0, 1.0], &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0]]);
        let c = coativacao("t", &a, 0.5, 5);
        assert!(c.excesso > 0.0, "excesso foi {}", c.excesso);
        assert!(c.pares_fortes.iter().any(|(i, j, _)| *i == 0 && *j == 1));
    }

    #[test]
    fn quente_frio_separa_sempre_de_nunca() {
        let a = amostras_fixas(&[&[1.0, 0.0, 0.5], &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.9]]);
        let q = quente_frio("t", &a, 0.4);
        assert_eq!(q.sempre_ativos, 1);   // neurônio 0
        assert_eq!(q.nunca_ativos, 1);    // neurônio 1
    }

    #[test]
    fn sensibilidade_cai_com_limiar_maior() {
        let a = amostras_fixas(&[&[0.9, 0.5, 0.1], &[0.8, 0.4, 0.05]]);
        let s = sensibilidade_ao_limiar(&a, &[0.01, 0.45, 0.95]);
        assert!(s[0].1 >= s[1].1 && s[1].1 >= s[2].1, "fração deveria cair: {s:?}");
    }
}
