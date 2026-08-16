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
        Some(PilotoLinear {
            w, mu, sd,
            y_mu: [y_mu_v[0], y_mu_v[1], y_mu_v[2]],
            y_sd: [y_sd_v[0], y_sd_v[1], y_sd_v[2]],
            alphas: [alphas_v[0], alphas_v[1]],
            h_a: Vec::new(), h_b: Vec::new(), iniciado: false,
        })
    }

    pub fn n_entradas(&self) -> usize { self.mu.len() / 3 }

    /// Um tick: x = os 192 canais COMO O L1 RECEBE (pós-KAN+CfC — mesmo X do solve).
    /// Devolve [acelerador, volante, freio] na MESMA convenção do intent do cérebro
    /// (pós-tanh, pré kan_eval) — o chamador aplica o resto do pipeline normal.
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
