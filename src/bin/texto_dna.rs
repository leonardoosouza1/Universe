use std::env;
use std::fs;
use universe::evolution::dna::DNA_MAGIC;
use iara_genesis::tlv::{GenesisPayload, Chunk};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Uso: cargo run --bin texto_dna encode|decode");
        return;
    }

    match args[1].as_str() {
        "encode" => {
            let txt = fs::read_to_string("texto_teste.txt").unwrap();
            let mut payload = GenesisPayload::new();
            
            // Adiciona o chunk de texto
            payload.add_chunk(Chunk::new(*b"TEXT", txt.into_bytes()));

            let mut out = Vec::new();
            out.extend_from_slice(universe::evolution::dna_genesis::DNA_MAGIC); // IARA
            out.extend_from_slice(&4u16.to_le_bytes()); // V4
            
            // O payload do Genesis gera a assinatura própria GNSS...
            out.extend_from_slice(&payload.to_bytes());
            
            fs::write("texto_teste.dna", out).unwrap();
            println!("Sucesso: texto_teste.txt convertido para texto_teste.dna!");
        }
        "decode" => {
            let buf = fs::read("texto_teste.dna").unwrap();
            
            // Validar header IARA V4
            if &buf[0..4] != universe::evolution::dna_genesis::DNA_MAGIC || u16::from_le_bytes(buf[4..6].try_into().unwrap()) != 4 {
                panic!("Não é DNA V4!");
            }

            let payload = GenesisPayload::from_bytes(&buf[6..]).unwrap();
            let chunk = payload.find_chunk(b"TEXT").unwrap();

            fs::write("texto_teste_restaurado.txt", &chunk.data).unwrap();
            println!("Sucesso: texto_teste.dna convertido de volta para texto_teste_restaurado.txt!");
        }
        _ => println!("Comando inválido."),
    }
}
