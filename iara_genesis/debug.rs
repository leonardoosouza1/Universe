use iara_genesis::codec::{HybridCodec, Codec};

fn main() {
    let codec = HybridCodec::new();
    let mut sparse = vec![0i8; 4096];
    for i in (0..4096).step_by(100) { sparse[i] = 1; }

    let compressed = codec.encode(&sparse);
    println!("Codec picked: {}", String::from_utf8_lossy(&compressed.codec_tag));
    println!("Compressed Size: {}", compressed.data.len());
    println!("Ratio: {}", compressed.ratio());
}
