import sys

def fix_pipeline():
    path = "src/evolution/pipeline.rs"
    with open(path, "r") as f:
        content = f.read()
    # 1. Quantize slow_weights for save_champion_dna_v7 call
    if "let slow_weights_i8: Vec<i8> =" not in content:
        content = content.replace(
            "save_champion_dna_v7(\n                    &weights,\n                    scale,\n                    &slow_weights,",
            "let slow_weights_i8: Vec<i8> = slow_weights.iter().map(|&w| (w * 63.5).round().clamp(-127.0, 127.0) as i8).collect();\n                save_champion_dna_v7(\n                    &weights,\n                    scale,\n                    &slow_weights_i8,"
        )
    with open(path, "w") as f:
        f.write(content)

def fix_spawn():
    path = "src/evolution/spawn.rs"
    with open(path, "r") as f:
        content = f.read()
    # 1. Convert i8 DNA to f32 current_best_slow_w
    content = content.replace(
        "current_best_slow_w = Some(sw);",
        "current_best_slow_w = Some(sw.iter().map(|&w| w as f32 / 63.5).collect());"
    )
    with open(path, "w") as f:
        f.write(content)

fix_pipeline()
fix_spawn()
print("Fixed type mismatches")
