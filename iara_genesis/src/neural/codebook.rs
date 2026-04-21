//! # Codebook — Serialização dos pesos do autoencoder
//!
//! O codebook contém os pesos do autoencoder e é compartilhado
//! pela população inteira. Um codebook ~4KB permite descomprimir
//! qualquer organismo que foi codificado com ele.
//!
//! ## Formato serializado
//!
//! ```text
//! magic[4] = b"CDBK"
//! version: u16
//! input_dim: u32
//! hidden_dim: u32
//! latent_dim: u32
//! epochs_trained: u32
//! num_params: u32
//! params: f32 × num_params  (enc_w1, enc_b1, enc_w2, enc_b2, dec_w1, dec_b1, dec_w2, dec_b2)
//! ```

use super::autoencoder::TinyAutoencoder;

pub const CODEBOOK_MAGIC: &[u8; 4] = b"CDBK";
pub const CODEBOOK_VERSION: u16 = 1;

/// Serializa o codebook (pesos do autoencoder) para bytes.
pub fn serialize(ae: &TinyAutoencoder) -> Vec<u8> {
    let num_params = ae.num_params();
    let header_size = 4 + 2 + 4 + 4 + 4 + 4 + 4; // 22 bytes
    let data_size = num_params * 4;
    let mut buf = Vec::with_capacity(header_size + data_size);

    buf.extend_from_slice(CODEBOOK_MAGIC);
    buf.extend_from_slice(&CODEBOOK_VERSION.to_le_bytes());
    buf.extend_from_slice(&(ae.input_dim as u32).to_le_bytes());
    buf.extend_from_slice(&(ae.hidden_dim as u32).to_le_bytes());
    buf.extend_from_slice(&(ae.latent_dim as u32).to_le_bytes());
    buf.extend_from_slice(&ae.epochs_trained.to_le_bytes());
    buf.extend_from_slice(&(num_params as u32).to_le_bytes());

    // Serializa todos os parâmetros em ordem
    for params in [
        &ae.enc_w1, &ae.enc_b1, &ae.enc_w2, &ae.enc_b2,
        &ae.dec_w1, &ae.dec_b1, &ae.dec_w2, &ae.dec_b2,
    ] {
        for &p in params.iter() {
            buf.extend_from_slice(&p.to_le_bytes());
        }
    }

    buf
}

/// Desserializa o codebook de bytes → TinyAutoencoder.
pub fn deserialize(buf: &[u8]) -> Option<TinyAutoencoder> {
    if buf.len() < 22 { return None; }

    if &buf[0..4] != CODEBOOK_MAGIC { return None; }

    let _version = u16::from_le_bytes(buf[4..6].try_into().ok()?);
    let input_dim = u32::from_le_bytes(buf[6..10].try_into().ok()?) as usize;
    let hidden_dim = u32::from_le_bytes(buf[10..14].try_into().ok()?) as usize;
    let latent_dim = u32::from_le_bytes(buf[14..18].try_into().ok()?) as usize;
    let epochs_trained = u32::from_le_bytes(buf[18..22].try_into().ok()?);
    let num_params = u32::from_le_bytes(buf[22..26].try_into().ok()?) as usize;

    if buf.len() < 26 + num_params * 4 { return None; }

    // Ler todos os floats
    let mut all_params = Vec::with_capacity(num_params);
    for i in 0..num_params {
        let off = 26 + i * 4;
        let v = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        all_params.push(if v.is_finite() { v } else { 0.0 });
    }

    // Reconstruir as dimensões das matrizes
    let sizes = [
        input_dim * hidden_dim,  // enc_w1
        hidden_dim,              // enc_b1
        hidden_dim * latent_dim, // enc_w2
        latent_dim,              // enc_b2
        latent_dim * hidden_dim, // dec_w1
        hidden_dim,              // dec_b1
        hidden_dim * input_dim,  // dec_w2
        input_dim,               // dec_b2
    ];

    let expected_total: usize = sizes.iter().sum();
    if num_params != expected_total { return None; }

    let mut offset = 0;
    let mut take = |size: usize| -> Vec<f32> {
        let slice = all_params[offset..offset + size].to_vec();
        offset += size;
        slice
    };

    let enc_w1 = take(sizes[0]);
    let enc_b1 = take(sizes[1]);
    let enc_w2 = take(sizes[2]);
    let enc_b2 = take(sizes[3]);
    let dec_w1 = take(sizes[4]);
    let dec_b1 = take(sizes[5]);
    let dec_w2 = take(sizes[6]);
    let dec_b2 = take(sizes[7]);

    Some(TinyAutoencoder {
        enc_w1, enc_b1, enc_w2, enc_b2,
        dec_w1, dec_b1, dec_w2, dec_b2,
        input_dim, hidden_dim, latent_dim,
        learning_rate: super::autoencoder::DEFAULT_LEARNING_RATE,
        l1_lambda: super::autoencoder::DEFAULT_L1_LAMBDA,
        epochs_trained,
    })
}

/// Tamanho do codebook em bytes para dimensões dadas.
pub fn codebook_size(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> usize {
    let num_params = input_dim * hidden_dim + hidden_dim
        + hidden_dim * latent_dim + latent_dim
        + latent_dim * hidden_dim + hidden_dim
        + hidden_dim * input_dim + input_dim;
    26 + num_params * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_deserialize_round_trip() {
        let mut ae = TinyAutoencoder::new(64, 16, 8);
        ae.epochs_trained = 42;

        let bytes = serialize(&ae);
        let restored = deserialize(&bytes).expect("codebook deser must work");

        assert_eq!(restored.input_dim, 64);
        assert_eq!(restored.hidden_dim, 16);
        assert_eq!(restored.latent_dim, 8);
        assert_eq!(restored.epochs_trained, 42);
        assert_eq!(restored.enc_w1.len(), ae.enc_w1.len());
        assert_eq!(restored.dec_w2.len(), ae.dec_w2.len());

        // Pesos devem ser idênticos
        for (a, b) in ae.enc_w1.iter().zip(restored.enc_w1.iter()) {
            assert!((a - b).abs() < 1e-7, "Weight mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn codebook_size_calculation() {
        let ae = TinyAutoencoder::new(64, 16, 8);
        let bytes = serialize(&ae);
        let expected = codebook_size(64, 16, 8);
        assert_eq!(bytes.len(), expected);
    }

    #[test]
    fn invalid_magic_returns_none() {
        let mut bytes = serialize(&TinyAutoencoder::new(64, 16, 8));
        bytes[0] = b'X';
        assert!(deserialize(&bytes).is_none());
    }

    #[test]
    fn truncated_data_returns_none() {
        let bytes = serialize(&TinyAutoencoder::new(64, 16, 8));
        assert!(deserialize(&bytes[..bytes.len() / 2]).is_none());
    }
}
