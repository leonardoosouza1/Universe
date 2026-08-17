use iara_genesis::tlv::{Chunk, GenesisPayload};
use iara_genesis::universal::{
    self, DEFAULT_BLOCK_SIZE, Motor, analyze_block, compress_universal, decompress_universal,
    deserialize_payload, serialize_payload,
};
use std::env;
use std::fs;
use std::time::Instant;

const DNA_MAGIC: &[u8; 4] = b"IARA";

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("═══════════════════════════════════════════════════");
        println!("  🧬 DNA V4 Universal Compressor");
        println!("═══════════════════════════════════════════════════");
        println!();
        println!("Usage:");
        println!("  dna_compress compress <input> <output.dna>    Compresses any file");
        println!("  dna_compress decompress <input.dna> <output>  Rebuilds the original file");
        println!("  dna_compress analyze <input>                  Analyzes without compressing");
        println!(
            "  dna_compress tensor-svd <input.bin> <rows> <cols> <rank> <out.dna> Decomposes a massive tensor"
        );
        println!(
            "  dna_compress tensor-svd-restore <input.dna> <out.bin>        Restores the original size (lossy)"
        );
        return;
    }

    match args[1].as_str() {
        "compress" => {
            if args.len() < 4 {
                eprintln!("Missing: <input> <output.dna>");
                return;
            }
            cmd_compress(&args[2], &args[3]);
        }
        "decompress" => {
            if args.len() < 4 {
                eprintln!("Missing: <input.dna> <output>");
                return;
            }
            cmd_decompress(&args[2], &args[3]);
        }
        "analyze" => {
            cmd_analyze(&args[2]);
        }
        "tensor-svd" => {
            if args.len() < 7 {
                eprintln!("Missing: <input.bin> <rows> <cols> <rank> <out.dna>");
                return;
            }
            let rows: usize = args[3].parse().unwrap();
            let cols: usize = args[4].parse().unwrap();
            let rank: usize = args[5].parse().unwrap();
            cmd_tensor_svd(&args[2], rows, cols, rank, &args[6]);
        }
        "tensor-svd-restore" => {
            if args.len() < 4 {
                eprintln!("Missing: <input.dna> <out.bin>");
                return;
            }
            cmd_tensor_svd_restore(&args[2], &args[3]);
        }
        _ => eprintln!("Unknown command: {}", args[1]),
    }
}

fn cmd_tensor_svd(input: &str, rows: usize, cols: usize, rank: usize, output: &str) {
    let t0 = Instant::now();
    let data = fs::read(input).expect("Failed to read file");
    let read_time = t0.elapsed();

    println!("═══════════════════════════════════════════════════════════");
    println!("  🧠 DNA V4 SVD Tensor Compressor");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Input:    {} ({:.1} MB)", input, data.len() as f64 / 1e6);
    println!("  Shape:    [{} x {}] BF16", rows, cols);
    println!("  SVD Rank: {}", rank);

    let t1 = Instant::now();
    let ltr = iara_genesis::neural::svd::factorize_bf16_tensor(&data, rows, cols, rank);
    let factorize_time = t1.elapsed();

    println!(
        "  SVD factorization completed in {:.1}ms",
        factorize_time.as_secs_f64() * 1000.0
    );

    let t_rmse = Instant::now();
    let rmse = iara_genesis::neural::svd::compute_rmse_bf16(&data, &ltr);
    println!(
        "  Validation RMSE: {:.5} (val time: {:.1}ms)",
        rmse,
        t_rmse.elapsed().as_secs_f64() * 1000.0
    );

    println!("───────────────────────────────────────────────────────────");
    println!("  ⚡ Direct Inference (Liquid Weights)");
    let dummy_input = vec![1.0_f32; cols];
    let t_inf = Instant::now();
    let _y = ltr.forward_pass(&dummy_input);
    let inf_time = t_inf.elapsed();
    let orig_flops = rows * cols * 2;
    let new_flops = (rank * cols * 2) + (rows * rank * 2);
    println!(
        "  -> Completed in {:.2}ms without decompression!",
        inf_time.as_secs_f64() * 1000.0
    );
    println!(
        "  -> Complexity: Dropped from {} FLOPs to {} FLOPs ({:.1}% saving)",
        orig_flops,
        new_flops,
        (1.0 - (new_flops as f64 / orig_flops as f64)) * 100.0
    );

    // Save as DNA V4
    let mut genesis = GenesisPayload::new();

    // Encapsulating matrix A [rows * rank] and matrix B [rank * cols]
    let mut payload = Vec::new();
    payload.extend_from_slice(&(rows as u32).to_le_bytes());
    payload.extend_from_slice(&(cols as u32).to_le_bytes());
    payload.extend_from_slice(&(rank as u32).to_le_bytes());

    payload.extend_from_slice(&ltr.a_bytes);
    payload.extend_from_slice(&ltr.b_bytes);

    genesis.add_chunk(Chunk::new(*b"TSVD", payload));

    let mut dna_bytes = Vec::new();
    dna_bytes.extend_from_slice(DNA_MAGIC);
    dna_bytes.extend_from_slice(&4u16.to_le_bytes());
    dna_bytes.extend_from_slice(&genesis.to_bytes());

    let t2 = Instant::now();
    fs::write(output, &dna_bytes).expect("Failed to write DNA");
    let write_time = t2.elapsed();

    let ratio = data.len() as f64 / dna_bytes.len() as f64;
    let saving = (1.0 - dna_bytes.len() as f64 / data.len() as f64) * 100.0;

    println!("───────────────────────────────────────────────────────────");
    println!(
        "  Output:   {} ({:.2} MB)",
        output,
        dna_bytes.len() as f64 / 1e6
    );
    println!("  Ratio:    {:.2}×", ratio);
    println!("  Saving:   {:.1}% of VRAM", saving);
    println!("───────────────────────────────────────────────────────────");
}

fn cmd_tensor_svd_restore(input: &str, output: &str) {
    let t0 = Instant::now();
    let dna_bytes = fs::read(input).expect("Failed to read DNA file");

    // Parse DNA V4 format to get TSVD chunk
    let genesis = GenesisPayload::from_bytes(&dna_bytes[6..]).expect("Genesis parse failure");
    let chunk = genesis
        .find_chunk(b"TSVD")
        .expect("Not a Tensor SVD DNA");

    let mut payload = chunk.data.as_slice();
    let rows = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    let rank = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;

    let offset_b = 12 + (rows * rank * 2);
    let a_bytes = &payload[12..offset_b];
    let b_bytes = &payload[offset_b..];

    let ltr = iara_genesis::neural::svd::LowRankTensors {
        a_bytes: a_bytes.to_vec(),
        b_bytes: b_bytes.to_vec(),
        rows,
        cols,
        rank,
    };

    println!("═══════════════════════════════════════════════════════════");
    println!("  🧠 DNA V4 SVD Tensor Extractor (Back to the Original)");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Shape:    [{} x {}] via Rank {}", rows, cols, rank);
    println!(
        "  Mass:     {:.1} MB recovered from the DNA",
        (rows * cols * 2) as f64 / 1e6
    );

    let t_inf = Instant::now();
    // Reconstruction: M = A * B.
    let out_bytes = ltr.decode_full_tensor();

    fs::write(output, &out_bytes).expect("Error writing restored matrix");
    println!(
        "  -> Completed in {:.1}ms",
        t_inf.elapsed().as_secs_f64() * 1000.0
    );
    println!(
        "  Output written to: {} ({:.1} MB)",
        output,
        fs::metadata(output).unwrap().len() as f64 / 1e6
    );
}

fn cmd_analyze(input: &str) {
    let data = fs::read(input).expect("Failed to read file");
    let total = data.len();
    let block_size = DEFAULT_BLOCK_SIZE;
    let num_blocks = (total + block_size - 1) / block_size;

    println!("═══════════════════════════════════════════════════════════");
    println!("  🔬 Spectral Analysis: {}", input);
    println!("  Size:    {} bytes ({:.1} MB)", total, total as f64 / 1e6);
    println!("  Blocks:  {} × {} bytes", num_blocks, block_size);
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!(
        "  {:>6}  {:>7}  {:>6}  {:>6}  {:>6}  {:<10}",
        "Block", "Entropy", "Zeros%", "Rough", "Period", "Motor"
    );
    println!("  ──────  ───────  ──────  ──────  ──────  ──────────");

    let mut motor_counts = [0usize; 5];

    for (i, chunk) in data.chunks(block_size).enumerate() {
        let profile = analyze_block(chunk);
        let motor_name = match profile.best_motor {
            Motor::Zero => {
                motor_counts[0] += 1;
                "ZERO"
            }
            Motor::ByteRle => {
                motor_counts[1] += 1;
                "BRLE"
            }
            Motor::Delta => {
                motor_counts[2] += 1;
                "DELT"
            }
            Motor::Fourier { num_harmonics } => {
                motor_counts[3] += 1;
                // Can't return formatted string from match, print inline
                if i < 20 || i >= num_blocks - 5 {
                    print!(
                        "  {:>6}  {:>7.2}  {:>5.1}%  {:>6.1}  {:>6.3}  DFTS-{}\n",
                        i,
                        profile.entropy,
                        profile.zero_ratio * 100.0,
                        profile.roughness,
                        profile.periodicity,
                        num_harmonics
                    );
                }
                continue;
            }
            Motor::Literal => {
                motor_counts[4] += 1;
                "LITL"
            }
        };

        if i < 20 || i >= num_blocks - 5 {
            println!(
                "  {:>6}  {:>7.2}  {:>5.1}%  {:>6.1}  {:>6.3}  {}",
                i,
                profile.entropy,
                profile.zero_ratio * 100.0,
                profile.roughness,
                profile.periodicity,
                motor_name
            );
        } else if i == 20 {
            println!("  ... ({} blocks omitted) ...", num_blocks - 25);
        }
    }

    println!();
    println!("  Motor summary:");
    println!("    ZERO (cost 0):        {:>5} blocks", motor_counts[0]);
    println!("    BRLE (RLE bytes):     {:>5} blocks", motor_counts[1]);
    println!("    DELT (delta coding):  {:>5} blocks", motor_counts[2]);
    println!("    DFTS (Fourier):       {:>5} blocks", motor_counts[3]);
    println!("    LITL (literal):       {:>5} blocks", motor_counts[4]);
}

fn cmd_compress(input: &str, output: &str) {
    let t0 = Instant::now();
    let data = fs::read(input).expect("Failed to read file");
    let read_time = t0.elapsed();

    println!("═══════════════════════════════════════════════════════════");
    println!("  🧬 DNA V4 Universal Compressor");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Input:  {} ({:.1} MB)", input, data.len() as f64 / 1e6);

    let t1 = Instant::now();
    let result = compress_universal(&data, DEFAULT_BLOCK_SIZE);
    let compress_time = t1.elapsed();

    // Pack into the DNA V4 format
    let payload_data = serialize_payload(&result, DEFAULT_BLOCK_SIZE as u32);
    let mut genesis = GenesisPayload::new();
    genesis.add_chunk(Chunk::new(*b"UCMP", payload_data));

    let mut dna_bytes = Vec::new();
    dna_bytes.extend_from_slice(DNA_MAGIC);
    dna_bytes.extend_from_slice(&4u16.to_le_bytes());
    dna_bytes.extend_from_slice(&genesis.to_bytes());

    let t2 = Instant::now();
    fs::write(output, &dna_bytes).expect("Failed to write DNA");
    let write_time = t2.elapsed();

    let ratio = data.len() as f64 / dna_bytes.len() as f64;
    let saving = (1.0 - dna_bytes.len() as f64 / data.len() as f64) * 100.0;

    println!(
        "  Output: {} ({:.1} MB)",
        output,
        dna_bytes.len() as f64 / 1e6
    );
    println!("───────────────────────────────────────────────────────────");
    println!("  Blocks:     {}", result.block_count);
    println!("  Ratio:      {:.2}×", ratio);
    println!("  Saving:     {:.1}%", saving);
    println!("───────────────────────────────────────────────────────────");
    println!("  Motors used:");
    println!(
        "    ZERO: {:>5} blocks (zero cost)",
        result.motor_stats.zero_blocks
    );
    println!(
        "    BRLE: {:>5} blocks (RLE)",
        result.motor_stats.rle_blocks
    );
    println!(
        "    DELT: {:>5} blocks (delta)",
        result.motor_stats.delta_blocks
    );
    println!(
        "    DFTS: {:>5} blocks (Fourier)",
        result.motor_stats.fourier_blocks
    );
    println!(
        "    LITL: {:>5} blocks (literal)",
        result.motor_stats.literal_blocks
    );
    println!("───────────────────────────────────────────────────────────");
    println!(
        "  Read time:        {:>8.1}ms",
        read_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Compression time: {:>8.1}ms",
        compress_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Write time:       {:>8.1}ms",
        write_time.as_secs_f64() * 1000.0
    );
    println!("═══════════════════════════════════════════════════════════");
}

fn cmd_decompress(input: &str, output: &str) {
    let t0 = Instant::now();
    let dna_bytes = fs::read(input).expect("Failed to read DNA");

    if dna_bytes.len() < 6 || &dna_bytes[0..4] != DNA_MAGIC {
        eprintln!("❌ Not a valid DNA V4 file");
        return;
    }

    let version = u16::from_le_bytes(dna_bytes[4..6].try_into().unwrap());
    if version != 4 {
        eprintln!("❌ Unsupported DNA version: v{}", version);
        return;
    }

    let genesis = GenesisPayload::from_bytes(&dna_bytes[6..]).expect("Failed to read GenesisPayload");

    let ucmp = genesis
        .find_chunk(b"UCMP")
        .expect("UCMP chunk not found — not a universally compressed DNA");

    let blocks = deserialize_payload(&ucmp.data).expect("Failed to deserialize compressed blocks");

    let t1 = Instant::now();
    let restored = decompress_universal(&blocks);
    let decompress_time = t1.elapsed();

    fs::write(output, &restored).expect("Failed to write file");

    println!("═══════════════════════════════════════════════════════════");
    println!("  🧬 DNA V4 Universal Decompressor");
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  Input:  {} ({:.1} MB)",
        input,
        dna_bytes.len() as f64 / 1e6
    );
    println!(
        "  Output: {} ({:.1} MB)",
        output,
        restored.len() as f64 / 1e6
    );
    println!("  Decompressed blocks: {}", blocks.len());
    println!(
        "  Total time: {:.1}ms (read) + {:.1}ms (decompression)",
        t0.elapsed().as_secs_f64() * 1000.0 - decompress_time.as_secs_f64() * 1000.0,
        decompress_time.as_secs_f64() * 1000.0
    );
    println!("═══════════════════════════════════════════════════════════");
}
