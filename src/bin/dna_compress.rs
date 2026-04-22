use std::env;
use std::fs;
use std::time::Instant;
use iara_genesis::universal::{
    self, analyze_block, compress_universal, decompress_universal,
    serialize_payload, deserialize_payload, DEFAULT_BLOCK_SIZE, Motor,
};
use iara_genesis::tlv::{GenesisPayload, Chunk};

const DNA_MAGIC: &[u8; 4] = b"IARA";

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("═══════════════════════════════════════════════════");
        println!("  🧬 DNA V4 Universal Compressor");
        println!("═══════════════════════════════════════════════════");
        println!();
        println!("Uso:");
        println!("  dna_compress compress <input> <output.dna>    Comprime qualquer arquivo");
        println!("  dna_compress decompress <input.dna> <output>  Reconstrói arquivo original");
        println!("  dna_compress analyze <input>                  Analisa sem comprimir");
        println!("  dna_compress tensor-svd <input.bin> <rows> <cols> <rank> <out.dna> Decompõe tensor massivo");
        println!("  dna_compress tensor-svd-restore <input.dna> <out.bin>        Restabelece tamanho original (lossy)");
        return;
    }

    match args[1].as_str() {
        "compress" => {
            if args.len() < 4 { eprintln!("Faltando: <input> <output.dna>"); return; }
            cmd_compress(&args[2], &args[3]);
        }
        "decompress" => {
            if args.len() < 4 { eprintln!("Faltando: <input.dna> <output>"); return; }
            cmd_decompress(&args[2], &args[3]);
        }
        "analyze" => {
            cmd_analyze(&args[2]);
        }
        "tensor-svd" => {
            if args.len() < 7 { eprintln!("Faltando: <input.bin> <rows> <cols> <rank> <out.dna>"); return; }
            let rows: usize = args[3].parse().unwrap();
            let cols: usize = args[4].parse().unwrap();
            let rank: usize = args[5].parse().unwrap();
            cmd_tensor_svd(&args[2], rows, cols, rank, &args[6]);
        }
        "tensor-svd-restore" => {
            if args.len() < 4 { eprintln!("Faltando: <input.dna> <out.bin>"); return; }
            cmd_tensor_svd_restore(&args[2], &args[3]);
        }
        _ => eprintln!("Comando desconhecido: {}", args[1]),
    }
}

fn cmd_tensor_svd(input: &str, rows: usize, cols: usize, rank: usize, output: &str) {
    let t0 = Instant::now();
    let data = fs::read(input).expect("Falha ao ler arquivo");
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

    println!("  Fatoração SVD concluída em {:.1}ms", factorize_time.as_secs_f64() * 1000.0);

    let t_rmse = Instant::now();
    let rmse = iara_genesis::neural::svd::compute_rmse_bf16(&data, &ltr);
    println!("  Validation RMSE: {:.5} (tempo val: {:.1}ms)", rmse, t_rmse.elapsed().as_secs_f64() * 1000.0);

    println!("───────────────────────────────────────────────────────────");
    println!("  ⚡ Inferência Direta (Liquid Weights)");
    let dummy_input = vec![1.0_f32; cols];
    let t_inf = Instant::now();
    let _y = ltr.forward_pass(&dummy_input);
    let inf_time = t_inf.elapsed();
    let orig_flops = rows * cols * 2;
    let new_flops = (rank * cols * 2) + (rows * rank * 2);
    println!("  -> Completada em {:.2}ms sem descompressão!", inf_time.as_secs_f64() * 1000.0);
    println!("  -> Complexidade: Caiu de {} FLOPs para {} FLOPs ({:.1}% economia)", 
        orig_flops, new_flops, (1.0 - (new_flops as f64 / orig_flops as f64)) * 100.0);

    // Save as DNA V4
    let mut genesis = GenesisPayload::new();
    
    // Encapsulando matriz A [rows * rank] e matriz B [rank * cols]
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
    fs::write(output, &dna_bytes).expect("Falha ao escrever DNA");
    let write_time = t2.elapsed();

    let ratio = data.len() as f64 / dna_bytes.len() as f64;
    let saving = (1.0 - dna_bytes.len() as f64 / data.len() as f64) * 100.0;

    println!("───────────────────────────────────────────────────────────");
    println!("  Output:   {} ({:.2} MB)", output, dna_bytes.len() as f64 / 1e6);
    println!("  Ratio:    {:.2}×", ratio);
    println!("  Economia: {:.1}% da VRAM", saving);
    println!("───────────────────────────────────────────────────────────");
}

fn cmd_tensor_svd_restore(input: &str, output: &str) {
    let t0 = Instant::now();
    let dna_bytes = fs::read(input).expect("Falha ao ler arquivo DNA");
    
    // Parse DNA V4 format to get TSVD chunk
    let genesis = GenesisPayload::from_bytes(&dna_bytes[6..]).expect("Falha Parse Genesis");
    let chunk = genesis.find_chunk(b"TSVD").expect("Não é um Tensor SVD DNA");
    
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
    println!("  🧠 DNA V4 SVD Tensor Extractor (Volta ao Original)");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Shape:    [{} x {}] via Rank {}", rows, cols, rank);
    println!("  Massa:    {:.1} MB recuperada do DNA", (rows * cols * 2) as f64 / 1e6);
    
    let t_inf = Instant::now();
    // Reconstrução: M = A * B.
    let out_bytes = ltr.decode_full_tensor();

    fs::write(output, &out_bytes).expect("Erro ao escrever Matrix Restabelecida");
    println!("  -> Completado em {:.1}ms", t_inf.elapsed().as_secs_f64() * 1000.0);
    println!("  Output escrito em: {} ({:.1} MB)", output, fs::metadata(output).unwrap().len() as f64 / 1e6);
}

fn cmd_analyze(input: &str) {
    let data = fs::read(input).expect("Falha ao ler arquivo");
    let total = data.len();
    let block_size = DEFAULT_BLOCK_SIZE;
    let num_blocks = (total + block_size - 1) / block_size;

    println!("═══════════════════════════════════════════════════════════");
    println!("  🔬 Análise Espectral: {}", input);
    println!("  Tamanho: {} bytes ({:.1} MB)", total, total as f64 / 1e6);
    println!("  Blocos:  {} × {} bytes", num_blocks, block_size);
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("  {:>6}  {:>7}  {:>6}  {:>6}  {:>6}  {:<10}",
        "Bloco", "Entropy", "Zeros%", "Rough", "Period", "Motor");
    println!("  ──────  ───────  ──────  ──────  ──────  ──────────");

    let mut motor_counts = [0usize; 5];

    for (i, chunk) in data.chunks(block_size).enumerate() {
        let profile = analyze_block(chunk);
        let motor_name = match profile.best_motor {
            Motor::Zero => { motor_counts[0] += 1; "ZERO" }
            Motor::ByteRle => { motor_counts[1] += 1; "BRLE" }
            Motor::Delta => { motor_counts[2] += 1; "DELT" }
            Motor::Fourier { num_harmonics } => {
                motor_counts[3] += 1;
                // Can't return formatted string from match, print inline
                if i < 20 || i >= num_blocks - 5 {
                    print!("  {:>6}  {:>7.2}  {:>5.1}%  {:>6.1}  {:>6.3}  DFTS-{}\n",
                        i, profile.entropy, profile.zero_ratio * 100.0,
                        profile.roughness, profile.periodicity, num_harmonics);
                }
                continue;
            }
            Motor::Literal => { motor_counts[4] += 1; "LITL" }
        };

        if i < 20 || i >= num_blocks - 5 {
            println!("  {:>6}  {:>7.2}  {:>5.1}%  {:>6.1}  {:>6.3}  {}",
                i, profile.entropy, profile.zero_ratio * 100.0,
                profile.roughness, profile.periodicity, motor_name);
        } else if i == 20 {
            println!("  ... ({} blocos omitidos) ...", num_blocks - 25);
        }
    }

    println!();
    println!("  Resumo de motores:");
    println!("    ZERO (custo 0):       {:>5} blocos", motor_counts[0]);
    println!("    BRLE (RLE bytes):     {:>5} blocos", motor_counts[1]);
    println!("    DELT (delta coding):  {:>5} blocos", motor_counts[2]);
    println!("    DFTS (Fourier):       {:>5} blocos", motor_counts[3]);
    println!("    LITL (literal):       {:>5} blocos", motor_counts[4]);
}

fn cmd_compress(input: &str, output: &str) {
    let t0 = Instant::now();
    let data = fs::read(input).expect("Falha ao ler arquivo");
    let read_time = t0.elapsed();

    println!("═══════════════════════════════════════════════════════════");
    println!("  🧬 DNA V4 Universal Compressor");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Input:  {} ({:.1} MB)", input, data.len() as f64 / 1e6);

    let t1 = Instant::now();
    let result = compress_universal(&data, DEFAULT_BLOCK_SIZE);
    let compress_time = t1.elapsed();

    // Empacotar no formato DNA V4
    let payload_data = serialize_payload(&result, DEFAULT_BLOCK_SIZE as u32);
    let mut genesis = GenesisPayload::new();
    genesis.add_chunk(Chunk::new(*b"UCMP", payload_data));

    let mut dna_bytes = Vec::new();
    dna_bytes.extend_from_slice(DNA_MAGIC);
    dna_bytes.extend_from_slice(&4u16.to_le_bytes());
    dna_bytes.extend_from_slice(&genesis.to_bytes());

    let t2 = Instant::now();
    fs::write(output, &dna_bytes).expect("Falha ao escrever DNA");
    let write_time = t2.elapsed();

    let ratio = data.len() as f64 / dna_bytes.len() as f64;
    let saving = (1.0 - dna_bytes.len() as f64 / data.len() as f64) * 100.0;

    println!("  Output: {} ({:.1} MB)", output, dna_bytes.len() as f64 / 1e6);
    println!("───────────────────────────────────────────────────────────");
    println!("  Blocos:     {}", result.block_count);
    println!("  Ratio:      {:.2}×", ratio);
    println!("  Economia:   {:.1}%", saving);
    println!("───────────────────────────────────────────────────────────");
    println!("  Motores usados:");
    println!("    ZERO: {:>5} blocos (custo zero)", result.motor_stats.zero_blocks);
    println!("    BRLE: {:>5} blocos (RLE)", result.motor_stats.rle_blocks);
    println!("    DELT: {:>5} blocos (delta)", result.motor_stats.delta_blocks);
    println!("    DFTS: {:>5} blocos (Fourier)", result.motor_stats.fourier_blocks);
    println!("    LITL: {:>5} blocos (literal)", result.motor_stats.literal_blocks);
    println!("───────────────────────────────────────────────────────────");
    println!("  Tempo leitura:    {:>8.1}ms", read_time.as_secs_f64() * 1000.0);
    println!("  Tempo compressão: {:>8.1}ms", compress_time.as_secs_f64() * 1000.0);
    println!("  Tempo escrita:    {:>8.1}ms", write_time.as_secs_f64() * 1000.0);
    println!("═══════════════════════════════════════════════════════════");
}

fn cmd_decompress(input: &str, output: &str) {
    let t0 = Instant::now();
    let dna_bytes = fs::read(input).expect("Falha ao ler DNA");

    if dna_bytes.len() < 6 || &dna_bytes[0..4] != DNA_MAGIC {
        eprintln!("❌ Não é um arquivo DNA V4 válido");
        return;
    }

    let version = u16::from_le_bytes(dna_bytes[4..6].try_into().unwrap());
    if version != 4 {
        eprintln!("❌ Versão DNA não suportada: v{}", version);
        return;
    }

    let genesis = GenesisPayload::from_bytes(&dna_bytes[6..])
        .expect("Falha ao ler GenesisPayload");

    let ucmp = genesis.find_chunk(b"UCMP")
        .expect("Chunk UCMP não encontrado — não é um DNA comprimido universal");

    let blocks = deserialize_payload(&ucmp.data)
        .expect("Falha ao deserializar blocos comprimidos");

    let t1 = Instant::now();
    let restored = decompress_universal(&blocks);
    let decompress_time = t1.elapsed();

    fs::write(output, &restored).expect("Falha ao escrever arquivo");

    println!("═══════════════════════════════════════════════════════════");
    println!("  🧬 DNA V4 Universal Decompressor");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Input:  {} ({:.1} MB)", input, dna_bytes.len() as f64 / 1e6);
    println!("  Output: {} ({:.1} MB)", output, restored.len() as f64 / 1e6);
    println!("  Blocos descomprimidos: {}", blocks.len());
    println!("  Tempo total: {:.1}ms (leitura) + {:.1}ms (descompressão)",
        t0.elapsed().as_secs_f64() * 1000.0 - decompress_time.as_secs_f64() * 1000.0,
        decompress_time.as_secs_f64() * 1000.0);
    println!("═══════════════════════════════════════════════════════════");
}
