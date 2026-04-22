//! # iara_genesis — Compressão Inteligente de DNA
//!
//! Este crate reimagina como o DNA de organismos IARA é armazenado.
//! Em vez de salvar arrays brutos de pesos ternários (8KB+), usamos
//! **matemática generativa**: fórmulas que *produzem* os pesos.
//!
//! ## Filosofia: Solo e Árvore
//!
//! - **Solo** (`Soil`): o campo de energia onde o organismo evoluiu.
//!   Fourier 2D comprimido — captura o fitness landscape em ~128 bytes.
//!
//! - **Árvore** (`Tree`): o organismo como estrutura generativa.
//!   A semente (polynomial seed) gera o tronco (backbone de instinto),
//!   que ramifica em galhos (especializações), cobertos de folhas
//!   (sparse corrections). Raízes conectam ao Solo.
//!
//! ## Codecs disponíveis
//!
//! | Codec | Ratio típico | Fidelidade | Uso |
//! |-------|-------------|------------|-----|
//! | `Fourier` | 32:1 | ~92-96% | Pesos com padrões periódicos |
//! | `Chebyshev` | 128:1 | ~85-90% | Pesos suaves, gradientes |
//! | `SparseRLE` | 2-50:1 | 100% lossless | Pesos muito esparsos |
//! | `Neural` | 32:1 | ~94-98% | Populações treinadas |
//! | `Hybrid` | 20-40:1 | ~97%+ | Auto-seleciona o melhor |
//!
//! ## Exemplo
//!
//! ```rust
//! use iara_genesis::codec::{Codec, HybridCodec};
//!
//! let weights: Vec<i8> = vec![0; 8192]; // pesos ternários
//! let codec = HybridCodec::new();
//!
//! let compressed = codec.encode(&weights);
//! let decompressed = codec.decode(&compressed).unwrap();
//!
//! // Medir fidelidade
//! let accuracy = iara_genesis::codec::ternary_accuracy(&weights, &decompressed);
//! println!("Fidelity: {:.1}%, Ratio: {:.0}:1",
//!     accuracy * 100.0,
//!     weights.len() as f64 / compressed.data.len() as f64);
//! ```

pub mod math;
pub mod codec;
pub mod neural;
pub mod model;
pub mod tlv;
pub mod universal;
