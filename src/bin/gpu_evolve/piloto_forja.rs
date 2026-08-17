//! PILOTO v2 — runtime da BASE FORJADA (não-linear), lendo `piloto_v2.json`.
//!
//! O v1 (`piloto_linear.rs`) resolve um controlador LINEAR por ridge sobre os pares
//! (sentiu→fez) do replay do campeão. O v2 troca a BASE: em vez de alimentar o ridge
//! com `[x, EMA_a, EMA_b]` (576 features), alimenta com uma PIRÂMIDE DE GAUSSIANAS —
//! 512 centros "grossos" (`Cg`) e 512 "finos" (`Cf`), cada conjunto lido por DUAS
//! larguras, dando 2048 features. Erro no held-out temporal (ticks 17000..20000):
//! **0,2620 → 0,1794 (−31,5%)**, por ação acel 0,109 · volante 0,137 · freio 0,292
//! (`piloto_v2.json:err_holdout_temporal` e `err_holdout_por_acao`).
//!
//! ## Matemática exata (FORMATO.md, `"formato": "piloto-v2/1"`)
//! Com `x` = os MESMOS 192 canais que o L1 recebe (pós-KAN+CfC, idênticos ao v1):
//! 1. `z[j] = (x[j] - mu[j]) / sd[j]`
//! 2. `dg[a] = Σ_j (z[j] - Cg[a][j])²` · `df[b] = Σ_j (z[j] - Cf[b][j])²`
//! 3. `phi = [ e^(-dg/2sg0²) ‖ e^(-dg/2sg1²) ‖ e^(-df/2sf0²) ‖ e^(-df/2sf1²) ]`
//!    — 2048 valores NESTA ORDEM (é a ordem das linhas de `W`; trocar a ordem dos
//!    quatro blocos é a armadilha da spec e o teste `ordem_dos_quatro_blocos_de_phi`
//!    existe só para pegá-la)
//! 4. `pre[k] = (Σ_i phi[i]·W[i][k]) · y_sd[k] + y_mu[k]`
//! 5. `intent[k] = tanh(pre[k])` → `[acelerador, volante, freio]`
//!
//! O `tanh` do passo 5 NÃO é enfeite: o ajuste foi feito em `arctanh(clip(y, ±0,999))`,
//! logo `W` prevê a PRÉ-ativação. Sem ele o erro sobe de 0,1794 para 0,2306 (medido
//! pela forja, FORMATO.md:16). `clip_arctanh` está no JSON como registro do ajuste;
//! o runtime não usa clip nenhum. A saída é o intent CRU, o MESMO ponto de entrega do
//! v1 (pré `kan_eval`/clamps) — quem chama segue com o pipeline normal.
//!
//! ## Custo por tick (pirâmide 2×512 centros sobre 192 canais)
//! - distâncias: `(512+512) × 192 = 196 608` mult (mesmo número de sub e add)
//! - `2048` exponenciais
//! - leitura de W: `2048 × 3 = 6 144` MAC
//! - **total ≈ 202 752 mult/tick** (`mult_por_tick()` devolve esse número, conferido
//!   no teste). A 64 Hz = **12,98 M mult/s** (o v1 gastava 576×3 = 1 728 mult/tick,
//!   117× menos, mas errava 46% mais).
//! - MEDIDO, não estimado: **143,8 µs/tick** (5 000 ticks em 718,8 ms, 6 956 ticks/s,
//!   `rustc -O` sem `target-cpu=native`, Ryzen 7 5800X3D, 1 thread). Isso é **0,92% do
//!   orçamento de um tick a 64 Hz** (15 625 µs) para UM ser — é a conta que autoriza
//!   chamar de desprezível, e vale para o uso do v1 (o piloto dirige o ser 0, o resto
//!   segue o cérebro). Dirigir a POPULAÇÃO inteira NÃO é de graça: 200 seres × 143,8 µs
//!   = 28,8 ms/tick, quase o dobro do orçamento. Para isso, a base tem de ir para a GPU.
//! - `carregar` (parse do JSON de 4,1 MB): **14,1 ms**, uma vez.
//! - memória residente: `(512+512)×192×4 B = 786 KB` de centros + 24 KB de W.
//! - SEM ESTADO: `agir` é função pura de `x` (o v1 carregava 2 EMAs). Isso é o que
//!   permite chamá-lo fora de ordem, repetir um tick, ou testá-lo linha a linha.
//!
//! ## Fragilidade medida, não suposta
//! `sd` tem 6 canais degenerados (98, 114, 115, 116, 117, 124) com `sd = 1e-12`: são
//! canais CONSTANTES em zero nos 20 001 ticks do dump (medido: `np.unique(X[:,j]) == [0]`).
//! Dentro do dump `z = 0` neles e nada acontece. Numa pista NOVA, se um deles sair de
//! zero, `z ≈ 5e11`, `dg` estoura e TODO o `phi` vai a zero — a saída congela em
//! `tanh(y_mu)`, constante. Por isso:
//! - `z` é saturado em ±`Z_MAX` (1e3) só para não gerar `inf`/`NaN` na física.
//!   Não muda número nenhum em distribuição: o maior `|z|` do dump inteiro é **19,07**
//!   e o maior `|centro|` é **17,63** — 1e3 é 50× acima de tudo que existe.
//! - `soma_phi` é telemetria de "estou dentro do que vi": no dump inteiro ela vai de
//!   **287,9 a 936,7** (p1 = 375,9; tick 17000 = 402,4). `agir_com_portao` abstém
//!   quando ela cai abaixo de `thr_phi` — desligado por padrão (`thr_phi = 0`), porque
//!   ligar sem medir mudaria o comportamento validado.

use std::io::Read;

/// Saturação de `z`. Puramente anti-`NaN` — ver o bloco "Fragilidade medida" acima.
const Z_MAX: f32 = 1.0e3;

/// Um conjunto de centros compartilhando um banco de larguras (lentes).
/// A pirâmide tem 2 grupos (`Cg` com `sg`, `Cf` com `sf`); o RBF tem 1.
struct Grupo {
    /// `[n × c]` ACHATADO row-major: o centro `a`, canal `j`, é `centros[a * c + j]`.
    /// O parser JSON daqui varre números e ignora os colchetes internos, então o
    /// achatamento é assumido, não lido — por isso `carregar` CONFERE `len == n * c`
    /// e recusa o arquivo se não bater (é a única defesa contra matriz transposta).
    centros: Vec<f32>,
    n: usize,
    /// `1/(2σ²)` por lente, na ordem em que as lentes aparecem em `phi`.
    inv2s2: Vec<f32>,
}

pub struct PilotoForja {
    base: String,
    c: usize,
    mu: Vec<f32>,
    sd: Vec<f32>,
    grupos: Vec<Grupo>,
    /// `[n_feat × 3]` ACHATADO row-major: `w[i * 3 + k]` = peso da feature `i` na ação `k`.
    w: Vec<f32>,
    y_mu: [f32; 3],
    y_sd: [f32; 3],
    n_feat: usize,
    // buffers reusados entre ticks (zero alocação no laço quente)
    z: Vec<f32>,
    d2: Vec<f32>,
    phi: Vec<f32>,
    // ── telemetria ──────────────────────────────────────────────────────────
    /// Σφ do último tick. Faixa medida no dump: 287,9 … 936,7.
    pub ultima_soma_phi: f32,
    /// Quantas vezes `z` bateu em `Z_MAX` (≠0 = canal degenerado saiu de zero).
    pub saturacoes_z: u64,
    pub abstencoes: u64,
    pub decisoes: u64,
    /// Limiar de Σφ abaixo do qual `agir_com_portao` abstém. 0 = portão desligado.
    pub thr_phi: f32,
    /// Índices com `sd` degenerado (< 1e-6): canais que o solve viu constantes.
    pub canais_degenerados: Vec<usize>,
}

impl PilotoForja {
    /// Carrega `piloto_v2.json`. Parser dependência-zero, mesma estratégia do
    /// `piloto_linear.rs`: acha `"chave"`, pega o `[` seguinte, casa colchetes e
    /// varre TODOS os números do intervalo. Consequência: matriz `[[..],[..]]` sai
    /// ACHATADA row-major (a varredura ignora os colchetes internos). As dimensões
    /// vêm dos escalares `n_entrada`/`ng`/`nf`/`n_feat` do próprio JSON e são
    /// conferidas contra o comprimento lido — arquivo que não bate é recusado.
    ///
    /// Bases: `"piramide"` (a vencedora, obrigatória) e `"rbf"` atrás do mesmo match.
    /// `"linear"` NÃO é aceita aqui de propósito: quem faz linear é `piloto_linear.rs`,
    /// que aplica `clamp` em vez de `tanh` — aceitar aqui daria número silenciosamente
    /// diferente do v1.
    pub fn carregar(path: &str) -> Option<PilotoForja> {
        let mut s = String::new();
        std::fs::File::open(path)
            .ok()?
            .read_to_string(&mut s)
            .ok()?;

        // ── extratores ──────────────────────────────────────────────────────
        let arr = |chave: &str| -> Option<Vec<f32>> {
            let i = s.find(&format!("\"{chave}\""))?;
            let a = s[i..].find('[')? + i;
            let mut nivel = 0i32;
            let mut fim = a;
            for (k, ch) in s[a..].char_indices() {
                match ch {
                    '[' => nivel += 1,
                    ']' => {
                        nivel -= 1;
                        if nivel == 0 {
                            fim = a + k;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let nums: Vec<f32> = s[a..=fim]
                .split(|c: char| {
                    !(c.is_ascii_digit()
                        || c == '.'
                        || c == '-'
                        || c == 'e'
                        || c == 'E'
                        || c == '+')
                })
                .filter(|t| !t.is_empty() && t.chars().any(|c| c.is_ascii_digit()))
                .filter_map(|t| t.parse().ok())
                .collect();
            Some(nums)
        };
        // Escalar (não-array). NÃO reaproveitar `arr` aqui: `arr` procura o próximo
        // '[', que num campo escalar pertence a OUTRA chave — foi assim que o limiar
        // do portão do v1 virou INFINITO em silêncio (piloto_linear.rs:88-91).
        let bruto = |chave: &str| -> Option<String> {
            let i = s.find(&format!("\"{chave}\""))?;
            let resto = &s[i + chave.len() + 2..];
            let j = resto.find(':')? + 1;
            let fim = resto[j..]
                .find(|c: char| c == ',' || c == '}')
                .unwrap_or(resto.len() - j)
                + j;
            Some(resto[j..fim].trim().trim_matches('"').to_string())
        };
        let escalar = |chave: &str| -> Option<f32> { bruto(chave)?.parse().ok() };
        let inteiro = |chave: &str| -> Option<usize> { bruto(chave)?.parse().ok() };

        let formato = bruto("formato").unwrap_or_default();
        let base = bruto("base").unwrap_or_default();
        if !formato.starts_with("piloto-v2/") {
            eprintln!(
                "[piloto-forja] formato desconhecido {formato:?} em {path} — esperava piloto-v2/*"
            );
            return None;
        }

        let c = inteiro("n_entrada")?;
        let mu = arr("mu")?;
        let sd = arr("sd")?;
        let w = arr("W")?;
        let y_mu_v = arr("y_mu")?;
        let y_sd_v = arr("y_sd")?;
        if mu.len() != c || sd.len() != c || y_mu_v.len() != 3 || y_sd_v.len() != 3 {
            eprintln!(
                "[piloto-forja] dimensões inesperadas: n_entrada={c} mu={} sd={} y_mu={} y_sd={}",
                mu.len(),
                sd.len(),
                y_mu_v.len(),
                y_sd_v.len()
            );
            return None;
        }

        // ── bases ───────────────────────────────────────────────────────────
        // A ORDEM desta lista é a ordem dos blocos de `phi`, e portanto a ordem das
        // linhas de `W`. Mexer aqui sem mexer no solve troca as ações de lugar.
        let mut grupos: Vec<Grupo> = Vec::new();
        let monta = |chave_c: &str, n: usize, sig: Vec<f32>| -> Option<Grupo> {
            let centros = arr(chave_c)?;
            if centros.len() != n * c {
                eprintln!(
                    "[piloto-forja] {chave_c} tem {} números, esperava {n}×{c}={} \
                           (matriz transposta ou dimensão errada no JSON)",
                    centros.len(),
                    n * c
                );
                return None;
            }
            Some(Grupo {
                centros,
                n,
                inv2s2: sig.iter().map(|s| 1.0 / (2.0 * s * s)).collect(),
            })
        };
        match base.as_str() {
            "piramide" => {
                let ng = inteiro("ng")?;
                let nf = inteiro("nf")?;
                grupos.push(monta("Cg", ng, arr("sg")?)?);
                grupos.push(monta("Cf", nf, arr("sf")?)?);
            }
            "rbf" => {
                // Não há `piloto_v2.json` com base "rbf" nesta máquina, logo este
                // ramo NÃO ESTÁ VALIDADO contra dado real — só contra os testes
                // sintéticos daqui. Está aberto porque a matemática é a pirâmide
                // com um grupo só.
                let ng = inteiro("ng")?;
                grupos.push(monta("Cg", ng, arr("sg")?)?);
            }
            outra => {
                eprintln!(
                    "[piloto-forja] base {outra:?} não suportada por este runtime \
                           (linear é do piloto_linear.rs, que usa clamp em vez de tanh)"
                );
                return None;
            }
        }

        let n_feat: usize = grupos.iter().map(|g| g.n * g.inv2s2.len()).sum();
        if let Some(declarado) = inteiro("n_feat") {
            if declarado != n_feat {
                eprintln!("[piloto-forja] n_feat declarado {declarado} ≠ montado {n_feat}");
                return None;
            }
        }
        if w.len() != n_feat * 3 {
            eprintln!(
                "[piloto-forja] W tem {} números, esperava {n_feat}×3={}",
                w.len(),
                n_feat * 3
            );
            return None;
        }

        let canais_degenerados: Vec<usize> = (0..c).filter(|&j| sd[j].abs() < 1e-6).collect();
        let maior_n = grupos.iter().map(|g| g.n).max().unwrap_or(0);
        let mult = grupos.iter().map(|g| g.n * c).sum::<usize>() + n_feat * 3;
        println!(
            "  [piloto-forja] base={base} {c}→{n_feat}→3 · {} grupos · {mult} mult/tick \
                  · err_holdout {} · degenerados {:?}",
            grupos.len(),
            escalar("err_holdout_temporal").unwrap_or(f32::NAN),
            canais_degenerados
        );

        Some(PilotoForja {
            base,
            c,
            mu,
            sd,
            grupos,
            w,
            y_mu: [y_mu_v[0], y_mu_v[1], y_mu_v[2]],
            y_sd: [y_sd_v[0], y_sd_v[1], y_sd_v[2]],
            n_feat,
            z: vec![0.0; c],
            d2: vec![0.0; maior_n],
            phi: vec![0.0; n_feat],
            ultima_soma_phi: 0.0,
            saturacoes_z: 0,
            abstencoes: 0,
            decisoes: 0,
            thr_phi: escalar("thr_phi").unwrap_or(0.0),
            canais_degenerados,
        })
    }

    pub fn n_entradas(&self) -> usize {
        self.c
    }
    pub fn n_features(&self) -> usize {
        self.n_feat
    }
    pub fn base(&self) -> &str {
        &self.base
    }

    /// Multiplicações por tick desta configuração (distâncias + leitura de W).
    /// Pirâmide 2×512 sobre 192 canais = 202 752; a 64 Hz, 12,98 M mult/s.
    pub fn mult_por_tick(&self) -> usize {
        self.grupos.iter().map(|g| g.n * self.c).sum::<usize>() + self.n_feat * 3
    }

    /// Um tick. `x` = os 192 canais COMO O L1 RECEBE (pós-KAN+CfC, o mesmo X do solve).
    /// Devolve `[acelerador, volante, freio]` na MESMA convenção do intent do cérebro
    /// (pós-`tanh`, pré `kan_eval`/clamps) — mesma assinatura e mesmo ponto de entrega
    /// do `PilotoLinear::agir`.
    pub fn agir(&mut self, x: &[f32]) -> [f32; 3] {
        let c = self.c;
        let n_feat = self.n_feat;
        debug_assert_eq!(
            x.len(),
            c,
            "piloto-forja: X com {} canais, esperava {c}",
            x.len()
        );
        // Desestruturação para emprestar `grupos` (só leitura) e `z`/`d2`/`phi`
        // (escrita) ao mesmo tempo — `for g in &self.grupos { self.d2[..] = .. }`
        // não compila, os dois empréstimos seriam do mesmo `self`.
        let PilotoForja {
            mu,
            sd,
            grupos,
            w,
            z,
            d2,
            phi,
            y_mu,
            y_sd,
            ultima_soma_phi,
            saturacoes_z,
            ..
        } = self;

        // 1. padroniza (com saturação anti-NaN — ver doc do módulo)
        let mut saturou = false;
        for j in 0..c {
            let v = (x[j] - mu[j]) / sd[j];
            let v = if v.is_finite() { v } else { 0.0 };
            if v > Z_MAX || v < -Z_MAX {
                saturou = true;
            }
            z[j] = v.clamp(-Z_MAX, Z_MAX);
        }
        if saturou {
            *saturacoes_z += 1;
        }

        // 2+3. distâncias e lentes. O laço externo é o GRUPO e o interno é a LENTE:
        //      é isso que produz [dg/sg0 ‖ dg/sg1 ‖ df/sf0 ‖ df/sf1], a ordem das
        //      linhas de W. Cada d² é calculado UMA vez e lido por todas as lentes
        //      do grupo (é daí que a pirâmide sai barata: 1024 distâncias, 2048 features).
        let mut p = 0usize;
        let mut soma = 0.0f32;
        for g in grupos.iter() {
            for a in 0..g.n {
                let base = a * c;
                let mut d = 0.0f32;
                for j in 0..c {
                    let dd = z[j] - g.centros[base + j];
                    d += dd * dd;
                }
                d2[a] = d;
            }
            for &inv in &g.inv2s2 {
                for a in 0..g.n {
                    let v = (-d2[a] * inv).exp();
                    phi[p] = v;
                    soma += v;
                    p += 1;
                }
            }
        }
        debug_assert_eq!(p, n_feat);
        *ultima_soma_phi = soma;

        // 4. phi @ W
        let mut pre = [0.0f32; 3];
        for i in 0..n_feat {
            let f = phi[i];
            pre[0] += f * w[i * 3];
            pre[1] += f * w[i * 3 + 1];
            pre[2] += f * w[i * 3 + 2];
        }

        // 5. desnormaliza e tanh (W prevê a PRÉ-ativação — ver doc do módulo)
        [
            (pre[0] * y_sd[0] + y_mu[0]).tanh(),
            (pre[1] * y_sd[1] + y_mu[1]).tanh(),
            (pre[2] * y_sd[2] + y_mu[2]).tanh(),
        ]
    }

    /// Como `agir`, mas com PORTÃO por Σφ: devolve `None` quando o estado não acendeu
    /// lente nenhuma (`Σφ < thr_phi`), e aí quem decide o tick é o cérebro — mesmo
    /// contrato do `PilotoLinear::agir_com_portao`.
    ///
    /// Σφ medido no dump inteiro (20 001 ticks): mín **287,9**, p1 **375,9**,
    /// mediana 724,7, máx 936,7 (tick 17000 = 402,4). `thr_phi` vem do JSON se
    /// existir, senão **0 = portão desligado**: ligar por conta própria mudaria o
    /// comportamento que rendeu 0,1794, e isso é decisão de quem calibrar, não do
    /// runtime. Um valor conservador seria ~150 (metade do mínimo observado).
    pub fn agir_com_portao(&mut self, x: &[f32]) -> Option<[f32; 3]> {
        let y = self.agir(x);
        if self.ultima_soma_phi >= self.thr_phi {
            self.decisoes += 1;
            Some(y)
        } else {
            self.abstencoes += 1;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V2: &str = "/media/llm-data/universe-runs/piloto-formula/v2/piloto_v2.json";
    const SANIDADE: &str = "/media/llm-data/universe-runs/piloto-formula/v2/sanidade3.txt";

    /// JSON sintético mínimo no formato piloto-v2/1.
    /// `cg`/`cf`: linhas de centros já em ordem row-major.
    fn json_piramide(
        c: usize,
        cg: &[Vec<f32>],
        cf: &[Vec<f32>],
        sg: [f32; 2],
        sf: [f32; 2],
        w: &[[f32; 3]],
        mu: &[f32],
        sd: &[f32],
        y_mu: [f32; 3],
        y_sd: [f32; 3],
    ) -> String {
        let mat = |m: &[Vec<f32>]| -> String {
            m.iter()
                .map(|r| {
                    format!(
                        "[{}]",
                        r.iter()
                            .map(|v| format!("{v:?}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        let wm: String = w
            .iter()
            .map(|r| format!("[{:?},{:?},{:?}]", r[0], r[1], r[2]))
            .collect::<Vec<_>>()
            .join(",");
        let v = |x: &[f32]| {
            x.iter()
                .map(|v| format!("{v:?}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "{{\"formato\": \"piloto-v2/1\", \"base\": \"piramide\", \"n_entrada\": {c}, \"n_saida\": 3, \
             \"ng\": {}, \"nf\": {}, \"n_feat\": {}, \"sg\": [{:?},{:?}], \"sf\": [{:?},{:?}], \
             \"mu\": [{}], \"sd\": [{}], \"y_mu\": [{:?},{:?},{:?}], \"y_sd\": [{:?},{:?},{:?}], \
             \"Cg\": [{}], \"Cf\": [{}], \"W\": [{}]}}",
            cg.len(),
            cf.len(),
            (cg.len() + cf.len()) * 2,
            sg[0],
            sg[1],
            sf[0],
            sf[1],
            v(mu),
            v(sd),
            y_mu[0],
            y_mu[1],
            y_mu[2],
            y_sd[0],
            y_sd[1],
            y_sd[2],
            mat(cg),
            mat(cf),
            wm
        )
    }

    fn escreve(nome: &str, conteudo: &str) -> String {
        let p = std::env::temp_dir().join(nome);
        std::fs::write(&p, conteudo).unwrap();
        p.to_str().unwrap().to_string()
    }

    /// Os 5 passos do FORMATO.md num caso que dá para conferir na mão.
    /// 1 centro grosso em 0, 1 fino em 0, x = [1,0] → z = [1,0] → dg = df = 1.
    #[test]
    fn cinco_passos_conferidos_na_mao() {
        let c = 2;
        let j = json_piramide(
            c,
            &[vec![0.0, 0.0]],
            &[vec![0.0, 0.0]],
            [1.0, 2.0],
            [0.5, 4.0],
            // W: só a feature 0 (dg com sg[0]) pesa, na ação 0.
            &[
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            &[0.0, 0.0],
            &[1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        let mut p = PilotoForja::carregar(&escreve("pf_5passos.json", &j)).expect("carregar");
        assert_eq!((p.n_entradas(), p.n_features()), (2, 4));
        assert_eq!(p.base(), "piramide");
        let y = p.agir(&[1.0, 0.0]);
        // phi[0] = exp(-1/(2·1²)) = 0.60653066 ; pre[0] = 0.60653066 ; y0 = tanh(pre)
        let esperado = (-0.5f32).exp().tanh();
        assert!(
            (y[0] - esperado).abs() < 1e-6,
            "y0={} esperado={esperado}",
            y[0]
        );
        assert_eq!((y[1], y[2]), (0.0, 0.0));
        // Σφ = e^-0.5 + e^-0.125 + e^-2 + e^-0.03125
        let soma: f32 = (-0.5f32).exp() + (-0.125f32).exp() + (-2.0f32).exp() + (-0.03125f32).exp();
        assert!(
            (p.ultima_soma_phi - soma).abs() < 1e-6,
            "Σφ={} esperado={soma}",
            p.ultima_soma_phi
        );
        // 1+1 centros × 2 canais + 4 features × 3 = 4 + 12 = 16 mult/tick
        assert_eq!(p.mult_por_tick(), 16);
    }

    /// A ARMADILHA DA SPEC: `phi` é `[dg/sg0 ‖ dg/sg1 ‖ df/sf0 ‖ df/sf1]`.
    /// Aqui cada um dos 4 blocos tem um σ diferente e um centro diferente, e W liga
    /// UM bloco por ação — se a ordem for trocada, dois valores caem no lugar errado.
    #[test]
    fn ordem_dos_quatro_blocos_de_phi() {
        let c = 1;
        // Cg = [[0]], Cf = [[3]]; x = 1 → dg = 1, df = 4.
        let j = json_piramide(
            c,
            &[vec![0.0]],
            &[vec![3.0]],
            [1.0, 2.0],
            [4.0, 8.0],
            // feature 0 → ação 0 · feature 1 → ação 1 · feature 2 → ação 2 · feature 3 → ação 2 (peso 10)
            &[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 10.0],
            ],
            &[0.0],
            &[1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        let mut p = PilotoForja::carregar(&escreve("pf_ordem.json", &j)).unwrap();
        let y = p.agir(&[1.0]);
        let f0 = (-1.0f32 / 2.0).exp(); // dg=1, sg0=1 → exp(-1/2)
        let f1 = (-1.0f32 / 8.0).exp(); // dg=1, sg1=2 → exp(-1/8)
        let f2 = (-4.0f32 / 32.0).exp(); // df=4, sf0=4 → exp(-4/32)
        let f3 = (-4.0f32 / 128.0).exp(); // df=4, sf1=8 → exp(-4/128)
        assert!(
            (y[0] - f0.tanh()).abs() < 1e-6,
            "bloco 0 (Cg,sg0) fora de lugar: {}",
            y[0]
        );
        assert!(
            (y[1] - f1.tanh()).abs() < 1e-6,
            "bloco 1 (Cg,sg1) fora de lugar: {}",
            y[1]
        );
        assert!(
            (y[2] - (f2 + 10.0 * f3).tanh()).abs() < 1e-6,
            "blocos 2/3 (Cf) fora de lugar: {}",
            y[2]
        );
        // f1 > f0 e f3 > f2: se os blocos trocassem, y[0] e y[1] trocariam de valor.
        assert!(f1 > f0 && f3 > f2);
    }

    /// `W` e `Cg`/`Cf` chegam ACHATADOS do parser (ele varre números e ignora os
    /// colchetes internos). Este teste crava o row-major: `w[i*3+k]` e `centros[a*c+j]`.
    /// Com 2 centros de 3 canais, um centro casa exatamente com x e o outro não.
    #[test]
    fn achatamento_row_major_de_matriz() {
        let c = 3;
        let j = json_piramide(
            c,
            &[vec![1.0, 2.0, 3.0], vec![-9.0, -9.0, -9.0]], // Cg: centro 0 casa com x
            &[vec![0.0, 0.0, 0.0]],
            [1.0, 1.0],
            [1.0, 1.0],
            // n_feat = (2+1)*2 = 6. Ação 1 lê SÓ a feature 0 (centro 0, sg0).
            &[
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        let mut p = PilotoForja::carregar(&escreve("pf_flat.json", &j)).unwrap();
        assert_eq!(p.n_features(), 6);
        let y = p.agir(&[1.0, 2.0, 3.0]);
        // d² do centro 0 = 0 → phi[0] = 1 → pre[1] = 5 → y1 = tanh(5)
        assert!(
            (y[1] - 5.0f32.tanh()).abs() < 1e-6,
            "y1={} (esperava tanh(5)); \
                 se a matriz fosse lida transposta o centro 0 não casaria",
            y[1]
        );
        // Matriz com dimensão errada tem de ser RECUSADA, não lida torto.
        let ruim = j.replace("\"ng\": 2", "\"ng\": 3");
        assert!(
            PilotoForja::carregar(&escreve("pf_flat_ruim.json", &ruim)).is_none(),
            "aceitou Cg com número de linhas errado"
        );
    }

    /// Portão por Σφ e saturação de z (canal degenerado saindo de zero).
    #[test]
    fn portao_por_soma_phi_e_saturacao_de_z() {
        let c = 1;
        let mut j = json_piramide(
            c,
            &[vec![0.0]],
            &[vec![0.0]],
            [1.0, 1.0],
            [1.0, 1.0],
            &[[0.0, 0.0, 0.0]; 4],
            &[0.0],
            &[1.0e-12],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        j = j.replace("\"n_saida\": 3", "\"n_saida\": 3, \"thr_phi\": 1.0");
        assert_eq!(j.matches("thr_phi").count(), 1);
        let mut p = PilotoForja::carregar(&escreve("pf_portao.json", &j)).unwrap();
        assert!(
            (p.thr_phi - 1.0).abs() < 1e-9,
            "limiar lido = {}",
            p.thr_phi
        );
        assert_eq!(
            p.canais_degenerados,
            vec![0],
            "sd=1e-12 tem de ser marcado degenerado"
        );
        // x = mu = 0 → z = 0 → phi todo 1 → Σφ = 4 ≥ 1 → decide
        assert!(p.agir_com_portao(&[0.0]).is_some());
        assert_eq!(p.saturacoes_z, 0);
        // x = 1 com sd = 1e-12 → z = 1e12 → satura em Z_MAX, φ→0 → Σφ < 1 → abstém
        assert!(p.agir_com_portao(&[1.0]).is_none());
        assert_eq!(p.saturacoes_z, 1, "saturação de z não foi contada");
        assert!(
            p.ultima_soma_phi.is_finite(),
            "Σφ virou NaN/inf: {}",
            p.ultima_soma_phi
        );
        assert_eq!((p.decisoes, p.abstencoes), (1, 1));
    }

    /// SANIDADE CONTRA O ARQUIVO REAL. Carrega `piloto_v2.json` (o mesmo que a forja
    /// gerou) e compara a saída com a referência gerada em Python a partir do MESMO
    /// arquivo (`sanidade3.txt`, produzido por `gerar_sanidade3.py` executando os 5
    /// passos do FORMATO.md em float64). Ticks 0, 1, 2 do `X.npy` + o tick 17000, que
    /// é o valor publicado no FORMATO.md (`[0.91202, -0.99821, -0.99239]`).
    /// Tolerância 1e-5: o runtime é f32, a referência é f64 (diferença medida ≤ 2,8e-6).
    #[test]
    fn sanidade_contra_o_piloto_v2_real() {
        if !std::path::Path::new(V2).exists() || !std::path::Path::new(SANIDADE).exists() {
            eprintln!(
                "\n[ATENÇÃO] TESTE NÃO EXECUTADO — falta {V2} ou {SANIDADE}.\n\
                       Este é o único teste que prova o runtime contra o piloto real;\n\
                       sem /media/llm-data montado ele não roda. NÃO é um PASS.\n"
            );
            return;
        }
        let mut ref_txt = String::new();
        std::fs::File::open(SANIDADE)
            .unwrap()
            .read_to_string(&mut ref_txt)
            .unwrap();
        let mut p = PilotoForja::carregar(V2).expect("carregar piloto_v2.json real");
        assert_eq!(p.base(), "piramide");
        assert_eq!((p.n_entradas(), p.n_features()), (192, 2048));
        assert_eq!(p.mult_por_tick(), 1024 * 192 + 2048 * 3);

        let mut tol = 1e-5f32;
        let mut x: Vec<f32> = Vec::new();
        let mut tick = usize::MAX;
        let mut conferidos = 0usize;
        for linha in ref_txt.lines() {
            let mut it = linha.split_whitespace();
            match it.next() {
                Some("TOL") => tol = it.next().unwrap().parse().unwrap(),
                Some("TICK") => tick = it.next().unwrap().parse().unwrap(),
                Some("X") => x = it.map(|t| t.parse().unwrap()).collect(),
                Some("Y") => {
                    let esperado: Vec<f32> = it.map(|t| t.parse().unwrap()).collect();
                    assert_eq!(
                        x.len(),
                        p.n_entradas(),
                        "tick {tick}: X com {} canais",
                        x.len()
                    );
                    assert_eq!(esperado.len(), 3);
                    let y = p.agir(&x);
                    let mut pior = 0.0f32;
                    for k in 0..3 {
                        let dif = (y[k] - esperado[k]).abs();
                        assert!(
                            dif <= tol,
                            "tick {tick} ação {k}: rust {} vs python {} (dif {dif:.3e} > tol {tol:.0e})",
                            y[k],
                            esperado[k]
                        );
                        if dif > pior {
                            pior = dif;
                        }
                    }
                    // Σφ medida no dump: 287,9 … 936,7. Fora disso, algo mudou.
                    assert!(
                        p.ultima_soma_phi > 280.0 && p.ultima_soma_phi < 940.0,
                        "tick {tick}: Σφ = {} fora da faixa medida [287,9 ; 936,7]",
                        p.ultima_soma_phi
                    );
                    assert_eq!(
                        p.saturacoes_z, 0,
                        "tick {tick}: z saturou em dado do próprio dump"
                    );
                    conferidos += 1;
                    println!(
                        "  tick {tick}: rust [{:.7}, {:.7}, {:.7}]  Σφ={:.2}  \
                              maior dif vs python(f64) = {pior:.2e}  (tol {tol:.0e}) ✓",
                        y[0], y[1], y[2], p.ultima_soma_phi
                    );
                }
                _ => {}
            }
        }
        assert_eq!(
            conferidos, 4,
            "esperava 4 ticks de referência, conferi {conferidos}"
        );
        assert_eq!(
            p.canais_degenerados,
            vec![98, 114, 115, 116, 117, 124],
            "canais degenerados mudaram — o dump ou o solve não é o mesmo"
        );
    }
}
