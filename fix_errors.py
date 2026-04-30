import sys

def fix_dna_genesis():
    path = "src/evolution/dna_genesis.rs"
    with open(path, "r") as f:
        content = f.read()
    # DnaPayload needs car_model_idx: 0 (default for old DNA)
    content = content.replace("generation: 0, seed: 0, track_width: 0.0, fitness: 0.0, version,", "generation: 0, seed: 0, track_width: 0.0, fitness: 0.0, car_model_idx: 0, version,")
    # Fix the three return Some(DnaPayload { ... })
    # We need to find where they are.
    import re
    # Match return Some(DnaPayload { followed by some fields and ending with })
    # We will just inject car_model_idx: 0 into the struct literal
    content = content.replace("return Some(DnaPayload {", "return Some(DnaPayload {\n                car_model_idx: 0,")
    with open(path, "w") as f:
        f.write(content)

def fix_pipeline():
    path = "src/evolution/pipeline.rs"
    with open(path, "r") as f:
        content = f.read()
    # EliteDna missing car_model_idx
    content = content.replace("morpho: r.morpho.clone(),", "morpho: r.morpho.clone(),\n                car_model_idx: r.car_model_idx,")
    with open(path, "w") as f:
        f.write(content)

def fix_spawn():
    path = "src/evolution/spawn.rs"
    with open(path, "r") as f:
        content = f.read()
    # config.elite_clones -> race.elite_pool.len()
    content = content.replace("let elite_range_end = 1 + config.elite_clones;", "let elite_range_end = 1 + race.elite_pool.len();")
    with open(path, "w") as f:
        f.write(content)

fix_dna_genesis()
fix_pipeline()
fix_spawn()
print("Fixed remaining compilation errors")
