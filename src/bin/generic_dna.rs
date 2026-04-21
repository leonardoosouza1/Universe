use std::env;
use std::fs;
use universe::evolution::dna_genesis::DNA_MAGIC;
use iara_genesis::tlv::{GenesisPayload, Chunk};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        println!("Uso: cargo run --bin generic_dna encode|decode <input_file> <output_file>");
        return;
    }

    let mode = args[1].as_str();
    let in_file = args[2].as_str();
    let out_file = args[3].as_str();

    match mode {
        "encode" => {
            let data = fs::read(in_file).unwrap_or_else(|_| panic!("Falha ao ler {}", in_file));
            let mut payload = GenesisPayload::new();
            
            // Adiciona o data_chunk
            payload.add_chunk(Chunk::new(*b"DATA", data));

            let mut out = Vec::new();
            out.extend_from_slice(DNA_MAGIC); // IARA
            out.extend_from_slice(&4u16.to_le_bytes()); // V4
            
            // O payload do Genesis gera a assinatura GNSS, metadata etc...
            out.extend_from_slice(&payload.to_bytes());
            
            fs::write(out_file, out).unwrap_or_else(|_| panic!("Falha ao escrever {}", out_file));
            println!("Encoded '{}' -> '{}'", in_file, out_file);
        }
        "decode" => {
            let buf = fs::read(in_file).unwrap_or_else(|_| panic!("Falha ao ler {}", in_file));
            
            if &buf[0..4] != DNA_MAGIC || u16::from_le_bytes(buf[4..6].try_into().unwrap()) != 4 {
                panic!("Não é DNA V4 válido!");
            }

            let payload = GenesisPayload::from_bytes(&buf[6..]).expect("Falha ao ler blocos GNSS TLV");
            let chunk = payload.find_chunk(b"DATA").expect("Faltando chunk com tag DATA");

            fs::write(out_file, &chunk.data).unwrap_or_else(|_| panic!("Falha ao escrever {}", out_file));
            println!("Decoded '{}' -> '{}'", in_file, out_file);
        }
        _ => println!("Use 'encode' ou 'decode'."),
    }
}
