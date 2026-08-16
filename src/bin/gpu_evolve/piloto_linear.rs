//! PILOTO POR FÓRMULA (PILOTO_POR_FORMULA.md, handoff da sessão-forja).
//!
//! Controlador RESOLVIDO por ridge (sem gradiente) sobre os pares (sentiu→fez)
//! do replay do campeão. Substitui a DECISÃO (L1→L2→motor) mantendo o
//! pré-processamento vivo (percepção+KAN+CfC — o X do solve é pós-CfC de
//! propósito, ver meta.json do dump). Runtime: 2 EMAs de estado + 1 matvec
//! [576×3] por tick — desprezível.
//!
//! Uso: `--piloto-formula <piloto.json>` → o ser 0 é dirigido pela fórmula
//! (via download_debug, mesmo canal do --observar); os demais seguem o cérebro.
//! A validação compara metros do ser 0 contra o campeão original e o mediano.

use std::io::Read;

pub struct PilotoLinear {
    w: Vec<f32>,      // [576 × 3] row-major (linha = feature, coluna = ação)
    mu: Vec<f32>,     // [576]
    sd: Vec<f32>,     // [576]
    y_mu: [f32; 3],
    y_sd: [f32; 3],
    alphas: [f32; 2],
    h_a: Vec<f32>,    // estado EMA alpha[0], [192]
    h_b: Vec<f32>,    // estado EMA alpha[1], [192]
    iniciado: bool,
    // ── PORTÃO DE ABSTENÇÃO (v2) ────────────────────────────────────────────
    // Medido em E-PIL1 (2026-08-16): na única pista NOVA em que os dois seres
    // ficaram na pista (9007), o clone caiu para 10% do campeão — freio médio
    // 0,56 contra 0,00 do campeão, volante 0,10 contra 0,59, velocidade por
    // quarto de corrida 8,8 → 0,3 → 0,1 → 0,1 m/s. Mecanismo: fora do que viu,
    // o linear EXTRAPOLA sem limite justo no eixo que ele ajusta pior (freio,
    // erro 0,45); freio alto → carro para → estado ainda mais estranho → trava
    // para sempre. Não é ruído: é o modo de falha de extrapolação linear.
    //
    // Conserto: guardar 512 linhas de referência do treino; se a distância² ao
    // vizinho mais próximo passar do quantil 0,99 do treino, o piloto ABSTÉM-SE
    // e quem decide naquele tick é o CÉREBRO (que continua rodando de qualquer
    // forma). Fallback melhor que o "frear/reduzir marcha" da spec — frear é
    // exatamente o que trava. Vira córtex + reflexo forjado com portão.
    refs: Vec<f32>,       // [n_ref × 192] achatado, no espaço padronizado
    n_ref: usize,
    thr_abster: f32,      // limiar em distância AO QUADRADO
    pub abstencoes: u64,  // telemetria: quantos ticks o cérebro reassumiu
    pub decisoes: u64,
    /// Controle: probabilidade FIXA de abstenção (flag --portao-aleatorio).
    pub prob_aleatoria: f32,
    /// Semente do sorteio do controle (--portao-semente).
    pub semente: u64,
    contador: u64,
    /// Comprimento das SEQUÊNCIAS contíguas de abstenção. É o que separa "o portão
    /// escolhe os ticks certos" de "o cérebro só precisa respirar": um tick solto de
    /// cérebro não tira um carro parado do lugar, um bloco tira. Predição registrada
    /// antes do controle: mediana ≥ 4 no portão contra ≈ 1 no aleatório.
    pub seqs: Vec<u32>,
    seq_atual: u32,
}

impl PilotoLinear {
    /// Carrega o piloto.json gravado pelo solve (formato do PILOTO_POR_FORMULA.md).
    /// Parser JSON mínimo e dependência-zero: extrai os arrays por chave. O arquivo
    /// é gerado por nós (numpy → json.dump), formato estável.
    pub fn carregar(path: &str) -> Option<PilotoLinear> {
        let mut s = String::new();
        std::fs::File::open(path).ok()?.read_to_string(&mut s).ok()?;
        let arr = |chave: &str| -> Option<Vec<f32>> {
            let i = s.find(&format!("\"{chave}\""))?;
            let a = s[i..].find('[')? + i;
            // W é [[...],[...],...] — pega até o colchete que FECHA o nível de abertura.
            let mut nivel = 0i32;
            let mut fim = a;
            for (k, c) in s[a..].char_indices() {
                match c {
                    '[' => nivel += 1,
                    ']' => {
                        nivel -= 1;
                        if nivel == 0 { fim = a + k; break; }
                    }
                    _ => {}
                }
            }
            let nums: Vec<f32> = s[a..=fim]
                .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+'))
                .filter(|t| !t.is_empty() && t.chars().any(|c| c.is_ascii_digit()))
                .filter_map(|t| t.parse().ok())
                .collect();
            Some(nums)
        };
        let w = arr("W")?;
        let mu = arr("mu")?;
        let sd = arr("sd")?;
        let y_mu_v = arr("y_mu")?;
        let y_sd_v = arr("y_sd")?;
        let alphas_v = arr("alphas")?;
        if w.len() != mu.len() * 3 || mu.len() != sd.len() || y_mu_v.len() != 3 || alphas_v.len() != 2 {
            eprintln!("[piloto-formula] formato inesperado: W={} mu={} y_mu={} alphas={}",
                      w.len(), mu.len(), y_mu_v.len(), alphas_v.len());
            return None;
        }
        // Escalar (não-array). `arr` procura o próximo '[', que num campo escalar
        // pertence a OUTRA chave — devolveria número errado ou nada, e o limiar
        // caía em INFINITO: o portão existia e nunca disparava, em silêncio.
        // Pego em 2026-08-16 (o log dizia "portão ativo" com "limiar inf" e
        // 0,0% de abstenção). Erro silencioso é o pior modo de falha aqui.
        let escalar = |chave: &str| -> Option<f32> {
            let i = s.find(&format!("\"{chave}\""))?;
            let resto = &s[i + chave.len() + 2..];
            let j = resto.find(':')? + 1;
            let fim = resto[j..].find(|c| c == ',' || c == '}').unwrap_or(resto.len() - j) + j;
            resto[j..fim].trim().trim_matches('"').parse().ok()
        };
        // Blocos do portão: opcionais (piloto.json antigo continua carregando).
        let refs = arr("refs").unwrap_or_default();
        let c = mu.len() / 3;
        let n_ref = if c > 0 { refs.len() / c } else { 0 };
        let thr_abster = escalar("thr_abster").unwrap_or(f32::INFINITY);
        if n_ref > 0 {
            if !thr_abster.is_finite() {
                eprintln!("[ERRO] piloto com {n_ref} referências mas SEM limiar legível \
                           (thr_abster) — o portão nunca abstém. Corrija o JSON.");
            }
            println!("  [piloto] portão de abstenção ativo: {n_ref} referências, limiar dist² {thr_abster:.1}");
        }
        Some(PilotoLinear {
            w, mu, sd,
            y_mu: [y_mu_v[0], y_mu_v[1], y_mu_v[2]],
            y_sd: [y_sd_v[0], y_sd_v[1], y_sd_v[2]],
            alphas: [alphas_v[0], alphas_v[1]],
            h_a: Vec::new(), h_b: Vec::new(), iniciado: false,
            refs, n_ref, thr_abster, abstencoes: 0, decisoes: 0,
            prob_aleatoria: 0.0, semente: 0, contador: 0,
            seqs: Vec::new(), seq_atual: 0,
        })
    }

    /// O estado atual está DENTRO do que o piloto viu? Distância² ao vizinho mais
    /// próximo das referências, no espaço padronizado do bloco cru. Sem referências
    /// (piloto.json v1) devolve sempre `true` — comportamento antigo preservado.
    fn conhece(&self, x: &[f32]) -> bool {
        if self.n_ref == 0 { return true; }
        let c = self.n_entradas();
        let mut melhor = f32::MAX;
        for r in 0..self.n_ref {
            let base = r * c;
            let mut d2 = 0.0f32;
            for i in 0..c {
                let z = (x[i] - self.mu[i]) / self.sd[i];
                let dd = z - self.refs[base + i];
                d2 += dd * dd;
                if d2 >= melhor { break; }   // poda: já perdeu para o melhor até aqui
            }
            if d2 < melhor { melhor = d2; }
        }
        melhor <= self.thr_abster
    }

    pub fn n_entradas(&self) -> usize { self.mu.len() / 3 }

    /// Um tick: x = os 192 canais COMO O L1 RECEBE (pós-KAN+CfC — mesmo X do solve).
    /// Devolve [acelerador, volante, freio] na MESMA convenção do intent do cérebro
    /// (pós-tanh, pré kan_eval) — o chamador aplica o resto do pipeline normal.
    /// Como `agir`, mas com PORTÃO: devolve `None` quando o estado está fora do
    /// que o piloto viu (o chamador deixa o cérebro decidir aquele tick). Os EMAs
    /// continuam sendo atualizados mesmo na abstenção — o piloto segue "assistindo"
    /// para não voltar com memória velha quando o estado retornar ao conhecido.
    pub fn agir_com_portao(&mut self, x: &[f32]) -> Option<[f32; 3]> {
        // CONTROLE DO PORTÃO (--portao-aleatorio p): abstém com probabilidade p
        // FIXA, ignorando o estado. Sem este controle não dá para saber se o portão
        // escolhe os ticks CERTOS ou se qualquer intervenção do cérebro na mesma
        // dose já salvaria a corrida — a diferença entre "sabe que não sabe" e
        // "tomar fôlego de vez em quando". PRNG por contador: determinístico.
        if self.prob_aleatoria > 0.0 {
            self.contador = self.contador.wrapping_add(1);
            // Semente própria (--portao-semente): sorteios INDEPENDENTES na mesma dose,
            // para medir a variância corrida-a-corrida. Sem ela, todo run repete o mesmo
            // sorteio e três "réplicas" seriam a mesma coisa (o erro das 3 sementes que
            // eram a mesma população, CLAUDE.md gap 6).
            let h = (self.contador ^ self.semente.wrapping_mul(0xD6E8_FEB8_6659_FD93))
                .wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
            let u = ((h >> 40) as f32) / 16_777_216.0;   // [0,1)
            let y = self.agir(x);
            return if u >= self.prob_aleatoria {
                self.decisoes += 1; self.fecha_seq(); Some(y)
            } else {
                self.abstencoes += 1; self.seq_atual += 1; None
            };
        }
        let dentro = self.conhece(x);
        let y = self.agir(x);
        if dentro {
            self.decisoes += 1; self.fecha_seq(); Some(y)
        } else {
            self.abstencoes += 1; self.seq_atual += 1; None
        }
    }

    fn fecha_seq(&mut self) {
        if self.seq_atual > 0 { self.seqs.push(self.seq_atual); self.seq_atual = 0; }
    }

    /// (mediana, máxima, quantidade) das sequências contíguas de abstenção.
    pub fn estat_seqs(&self) -> (f32, u32, usize) {
        if self.seqs.is_empty() { return (0.0, 0, 0); }
        let mut v = self.seqs.clone(); v.sort_unstable();
        (v[v.len() / 2] as f32, *v.last().unwrap(), v.len())
    }

    pub fn agir(&mut self, x: &[f32]) -> [f32; 3] {
        let c = self.n_entradas();
        debug_assert_eq!(x.len(), c, "piloto: X com {} canais, esperava {c}", x.len());
        if !self.iniciado {
            self.h_a = x.to_vec();
            self.h_b = x.to_vec();
            self.iniciado = true;
        }
        let (a0, a1) = (self.alphas[0], self.alphas[1]);
        for i in 0..c {
            self.h_a[i] = a0 * self.h_a[i] + (1.0 - a0) * x[i];
            self.h_b[i] = a1 * self.h_b[i] + (1.0 - a1) * x[i];
        }
        let mut y = [0.0f32; 3];
        // feats = [x, h_a, h_b] normalizado; W row-major [feature][acao]
        for (bloco, vals) in [(0usize, x), (1, self.h_a.as_slice()), (2, self.h_b.as_slice())] {
            for i in 0..c {
                let f = bloco * c + i;
                let z = (vals[i] - self.mu[f]) / self.sd[f];
                y[0] += z * self.w[f * 3];
                y[1] += z * self.w[f * 3 + 1];
                y[2] += z * self.w[f * 3 + 2];
            }
        }
        [
            (y[0] * self.y_sd[0] + self.y_mu[0]).clamp(-1.0, 1.0),
            (y[1] * self.y_sd[1] + self.y_mu[1]).clamp(-1.0, 1.0),
            (y[2] * self.y_sd[2] + self.y_mu[2]).clamp(-1.0, 1.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json_teste(c: usize) -> String {
        // W identidade-ish: só o bloco cru (features 0..c) com peso 1 na ação 0.
        let mut w = vec![vec![0.0f32; 3]; c * 3];
        w[0][0] = 1.0;
        let linhas: Vec<String> = w.iter().map(|r| format!("[{},{},{}]", r[0], r[1], r[2])).collect();
        format!(
            "{{\"W\": [{}], \"mu\": [{}], \"sd\": [{}], \"y_mu\": [0,0,0], \"y_sd\": [1,1,1], \"alphas\": [0.8, 0.97], \"lam\": 1.0}}",
            linhas.join(","),
            vec!["0"; c * 3].join(","),
            vec!["1"; c * 3].join(","),
        )
    }

    #[test]
    fn carrega_e_age() {
        let p = std::env::temp_dir().join("piloto_teste.json");
        std::fs::write(&p, json_teste(4)).unwrap();
        let mut pl = PilotoLinear::carregar(p.to_str().unwrap()).expect("carregar");
        assert_eq!(pl.n_entradas(), 4);
        let y = pl.agir(&[0.5, 0.0, 0.0, 0.0]);
        // feature 0 → ação 0 com peso 1: y0 = 0.5
        assert!((y[0] - 0.5).abs() < 1e-5, "y0={}", y[0]);
        assert_eq!(y[1], 0.0);
    }

    #[test]
    fn ema_persiste_entre_ticks() {
        let p = std::env::temp_dir().join("piloto_teste2.json");
        std::fs::write(&p, json_teste(2)).unwrap();
        let mut pl = PilotoLinear::carregar(p.to_str().unwrap()).unwrap();
        pl.agir(&[1.0, 0.0]);
        pl.agir(&[0.0, 0.0]);
        // h_a após [1,0]: inicia 1.0; tick 2: 0.8*1.0+0.2*0.0=0.8
        assert!((pl.h_a[0] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn portao_le_limiar_escalar_e_abstem() {
        // Regressão do bug do limiar INFINITO: `thr_abster` é escalar, e o parser
        // de array pegava o '[' de outra chave. Com o limiar lido, um estado longe
        // das referências tem de ABSTER (None) e um estado em cima delas, decidir.
        let p = std::env::temp_dir().join("piloto_portao.json");
        let mut j = json_teste(2);
        j = j.replace("\"lam\": 1.0", "\"refs\": [[0,0]], \"thr_abster\": 1.0, \"lam\": 1.0");
        assert_eq!(j.matches("thr_abster").count(), 1);
        std::fs::write(&p, &j).unwrap();
        let mut pl = PilotoLinear::carregar(p.to_str().unwrap()).unwrap();
        assert_eq!(pl.n_ref, 1);
        assert!((pl.thr_abster - 1.0).abs() < 1e-6, "limiar lido = {}", pl.thr_abster);
        assert!(pl.agir_com_portao(&[0.2, 0.0]).is_some(), "perto da referência: decide");
        assert!(pl.agir_com_portao(&[9.0, 9.0]).is_none(), "longe: abstém");
        assert_eq!((pl.decisoes, pl.abstencoes), (1, 1));
    }

    #[test]
    fn clamp_nas_acoes() {
        let p = std::env::temp_dir().join("piloto_teste3.json");
        let mut j = json_teste(1);
        j = j.replace("\"y_sd\": [1,1,1]", "\"y_sd\": [100,1,1]");
        std::fs::write(&p, &j).unwrap();
        let mut pl = PilotoLinear::carregar(p.to_str().unwrap()).unwrap();
        let y = pl.agir(&[5.0]);
        assert!(y[0] <= 1.0 && y[0] >= -1.0);
    }
}
