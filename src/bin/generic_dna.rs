use iara_genesis::tlv::{Chunk, GenesisPayload};
use std::env;
use std::fs;
use universe::evolution::dna_genesis::DNA_MAGIC;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        println!("Usage: cargo run --bin generic_dna encode|decode <input_file> <output_file>");
        return;
    }

    let mode = args[1].as_str();
    let in_file = args[2].as_str();
    let out_file = args[3].as_str();

    match mode {
        "encode" => {
            let data = fs::read(in_file).unwrap_or_else(|_| panic!("Failed to read {}", in_file));
            let mut payload = GenesisPayload::new();

            // Adds the data_chunk
            payload.add_chunk(Chunk::new(*b"DATA", data));

            let mut out = Vec::new();
            out.extend_from_slice(DNA_MAGIC); // IARA
            out.extend_from_slice(&4u16.to_le_bytes()); // V4

            // The Genesis payload generates the GNSS signature, metadata, etc...
            out.extend_from_slice(&payload.to_bytes());

            fs::write(out_file, out).unwrap_or_else(|_| panic!("Failed to write {}", out_file));
            println!("Encoded '{}' -> '{}'", in_file, out_file);
        }
        "decode" => {
            let buf = fs::read(in_file).unwrap_or_else(|_| panic!("Failed to read {}", in_file));

            if &buf[0..4] != DNA_MAGIC || u16::from_le_bytes(buf[4..6].try_into().unwrap()) != 4 {
                panic!("Not a valid DNA V4!");
            }

            let payload =
                GenesisPayload::from_bytes(&buf[6..]).expect("Failed to read GNSS TLV blocks");
            let chunk = payload
                .find_chunk(b"DATA")
                .expect("Missing chunk with DATA tag");

            fs::write(out_file, &chunk.data)
                .unwrap_or_else(|_| panic!("Failed to write {}", out_file));
            println!("Decoded '{}' -> '{}'", in_file, out_file);
        }
        _ => println!("Use 'encode' or 'decode'."),
    }
}
