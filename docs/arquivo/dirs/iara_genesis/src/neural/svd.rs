use bytemuck;
use faer::{Mat, prelude::*};
use half::bf16;
use std::time::Instant;

/// Resultado da Fatoração SVD Low-Rank
/// W ≈ A × B
#[derive(Debug)]
pub struct LowRankTensors {
    pub a_bytes: Vec<u8>, // Vetor flat [rows x rank] em bf16
    pub b_bytes: Vec<u8>, // Vetor flat [rank x cols] em bf16
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
}

impl LowRankTensors {
    /// Inferência Líquida Direta.
    /// Em vez de reconstruir a matriz gigante de 27MB na RAM para processar M = W * X,
    /// a GPU (ou CPU) realiza Y = A * (B * X).
    /// Complexidade: Multiplicações caem de (8960 * 1536 = 13.7 milhões) para apenas ~600k ops,
    /// sem precisar de descompressão, mexendo diretamente nos bits minúsculos!
    pub fn forward_pass(&self, input_x: &[f32]) -> Vec<f32> {
        assert_eq!(
            input_x.len(),
            self.cols,
            "Tensor de input deve bater com in_features"
        );
        let a_bf16: &[half::bf16] = bytemuck::cast_slice(&self.a_bytes);
        let b_bf16: &[half::bf16] = bytemuck::cast_slice(&self.b_bytes);

        // 1. Estágio Intermediário: H = B * X [rank, cols] * [cols] -> [rank]
        let mut h = vec![0.0f32; self.rank];
        for r in 0..self.rank {
            let mut sum = 0.0f32;
            for c in 0..self.cols {
                let w = b_bf16[r * self.cols + c].to_f32();
                sum += w * input_x[c];
            }
            h[r] = sum;
        }

        // 2. Estágio Final: Y = A * H -> [rows, rank] * [rank] -> [rows]
        let mut y = vec![0.0f32; self.rows];
        for r in 0..self.rows {
            let mut sum = 0.0f32;
            for c in 0..self.rank {
                let w = a_bf16[r * self.rank + c].to_f32();
                sum += w * h[c];
            }
            y[r] = sum;
        }

        y
    }

    /// Descomprime para a representação maciça original (Restabelecimento do Tensor BF16 Lossy)
    pub fn decode_full_tensor(&self) -> Vec<u8> {
        let a_bf16: &[half::bf16] = bytemuck::cast_slice(&self.a_bytes);
        let b_bf16: &[half::bf16] = bytemuck::cast_slice(&self.b_bytes);

        let mut out_bf16 = vec![half::bf16::ZERO; self.rows * self.cols];

        for r in 0..self.rows {
            for c in 0..self.cols {
                let mut sum = 0.0_f32;
                for k in 0..self.rank {
                    let a_val = a_bf16[r * self.rank + k].to_f32();
                    let b_val = b_bf16[k * self.cols + c].to_f32();
                    sum += a_val * b_val;
                }
                out_bf16[r * self.cols + c] = half::bf16::from_f32(sum);
            }
        }

        bytemuck::cast_slice(&out_bf16).to_vec()
    }
}

/// Fatora um tensor gigante formatado em BF16 em duas pequenas matrizes (A e B)
/// usando Thin SVD e low-rank approximation (Top-K).
pub fn factorize_bf16_tensor(
    tensor_bytes: &[u8],
    rows: usize,
    cols: usize,
    rank: usize,
) -> LowRankTensors {
    let t0 = Instant::now();

    // 1. Casting the bytes directly into BF16 slices
    let bf16_data: &[bf16] = bytemuck::cast_slice(tensor_bytes);
    assert_eq!(
        bf16_data.len(),
        rows * cols,
        "Tamanho dos bytes não bate com a dimensão informada"
    );

    // 2. Preenchendo a matriz faer com f32 (pois a lib solver trabalha com normais fp32)
    // faer guarda matriz por colunas (Col-Major) por padrão se usarmos Mat::from_fn,
    // mas safetensors do pytorch são Row-Major. Portanto vamos mapear com iteradores manuais.
    let mut mat = Mat::<f32>::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            mat.write(i, j, bf16_data[i * cols + j].to_f32());
        }
    }
    println!(
        "Matriz {}x{} inflada na memória f32 em {:.2?}",
        rows,
        cols,
        t0.elapsed()
    );

    let t1 = Instant::now();
    // 3. O compute real do Thin SVD
    let svd = mat.thin_svd();
    println!("Thin SVD Concluído via faer em {:.2?}", t1.elapsed());

    let u = svd.u(); // [rows, min(rows,cols)]
    let s = svd.s_diagonal(); // [min(rows,cols)]
    let v_t = svd.v().adjoint(); // [min(rows,cols), cols] adjoint/transpose

    // 4. Truncate pelo rank K
    let k = rank.min(rows).min(cols);

    // Compute A = U[:, :k] * S[:k]
    // Compute B = V_T[:k, :]

    let mut a_data = Vec::with_capacity(rows * k);
    let mut b_data = Vec::with_capacity(k * cols);

    // B matrix = shape [k, cols] in row-major continuous memory
    for r in 0..k {
        for c in 0..cols {
            b_data.push(bf16::from_f32(v_t.read(r, c)));
        }
    }

    // A matrix = U * S. Shape [rows, k] in row-major
    for r in 0..rows {
        for c in 0..k {
            let val = u.read(r, c) * s.read(c);
            a_data.push(bf16::from_f32(val));
        }
    }

    LowRankTensors {
        a_bytes: bytemuck::cast_slice(&a_data).to_vec(),
        b_bytes: bytemuck::cast_slice(&b_data).to_vec(),
        rows,
        cols,
        rank: k,
    }
}

/// Calcula o Root Mean Square Error (RMSE) da reconstrução com o tensor original.
pub fn compute_rmse_bf16(original_bytes: &[u8], low_rank: &LowRankTensors) -> f32 {
    let bf16_data: &[bf16] = bytemuck::cast_slice(original_bytes);

    let a_bf16: &[bf16] = bytemuck::cast_slice(&low_rank.a_bytes);
    let b_bf16: &[bf16] = bytemuck::cast_slice(&low_rank.b_bytes);
    let rank = low_rank.rank;
    let cols = low_rank.cols;
    let rows = low_rank.rows;

    let mut error_sq = 0.0_f64;

    for i in 0..rows {
        for j in 0..cols {
            let orig = bf16_data[i * cols + j].to_f32() as f64;

            // Reconstroi W[i, j] = inner_product(A[i, :], B[:, j])
            let mut synth = 0.0_f64;
            for r in 0..rank {
                let a_val = a_bf16[i * rank + r].to_f32() as f64;
                let b_val = b_bf16[r * cols + j].to_f32() as f64;
                synth += a_val * b_val;
            }

            let diff = orig - synth;
            error_sq += diff * diff;
        }
    }

    (error_sq / (rows as f64 * cols as f64)).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_mock() {
        let rows = 128;
        let cols = 64;
        let mut sim_bytes = Vec::new();
        // Gerando bloco com rank falso de ~ 1
        for i in 0..rows {
            for j in 0..cols {
                let v = ((i + j) as f32 * 0.01).sin();
                let b = bf16::from_f32(v);
                let bytes = bytemuck::bytes_of(&b);
                sim_bytes.extend_from_slice(bytes);
            }
        }

        // Tensor de [128, 64] => rank normal = 64
        // Pede rank 8 => Compressao brutal
        let lora = factorize_bf16_tensor(&sim_bytes, rows, cols, 8);

        let rmse = compute_rmse_bf16(&sim_bytes, &lora);

        println!("RMSE de matriz Mockada: {}", rmse);
        assert!(rmse < 0.1, "RMSE deveria ser baixo por ter simulação suave");
    }
}
