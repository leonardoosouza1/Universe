import sys

def fix_dna_payload_def():
    path = "src/evolution/dna_genesis.rs"
    with open(path, "r") as f:
        content = f.read()
    # Remove the redundant f32 layers from DnaPayload
    content = content.replace("pub cortex_l2: Vec<f32>,\n    pub cortex_l3: Vec<f32>,", "")
    # Add cortex_l3_raw if missing
    if "pub cortex_l3_raw: Option<Vec<i8>>," not in content:
        content = content.replace("pub cortex_l2_raw: Option<Vec<i8>>,", "pub cortex_l2_raw: Option<Vec<i8>>,\n    pub cortex_l3_raw: Option<Vec<i8>>,")
    with open(path, "w") as f:
        f.write(content)

def fix_dna_payload_literals():
    path = "src/evolution/dna_genesis.rs"
    with open(path, "r") as f:
        content = f.read()
    # DnaPayload literals shouldn't have car_model_idx twice (from previous script)
    # And should have cortex_l3_raw: None
    content = content.replace("cortex_l2_raw,\n                            });", "cortex_l2_raw,\n                                cortex_l3_raw: None,\n                            });")
    content = content.replace("cortex_l2_raw: None,\n        });", "cortex_l2_raw: None,\n            cortex_l3_raw: None,\n        });")
    with open(path, "w") as f:
        f.write(content)

fix_dna_payload_def()
fix_dna_payload_literals()
