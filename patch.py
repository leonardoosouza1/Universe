import sys

def patch_racer():
    path = "src/simulation/racer.rs"
    with open(path, "r") as f:
        content = f.read()
    if "pub car_model_idx: u8" not in content:
        content = content.replace("pub car_model: String,", "pub car_model: String,\n    pub car_model_idx: u8,")
        with open(path, "w") as f:
            f.write(content)

def patch_settings():
    path = "src/evolution/settings.rs"
    with open(path, "r") as f:
        content = f.read()
    if "pub best_car_model_idx: Option<u8>" not in content:
        content = content.replace("pub best_weights: Option<Vec<f32>>,", "pub best_weights: Option<Vec<f32>>,\n    pub best_car_model_idx: Option<u8>,")
        with open(path, "w") as f:
            f.write(content)

def patch_dna():
    path = "src/evolution/dna_genesis.rs"
    with open(path, "r") as f:
        content = f.read()
    if "pub car_model_idx: u8," not in content:
        content = content.replace("pub morpho: MorphoGenes,", "pub morpho: MorphoGenes,\n    pub car_model_idx: u8,")
        with open(path, "w") as f:
            f.write(content)

def patch_pipeline():
    path = "src/evolution/pipeline.rs"
    with open(path, "r") as f:
        content = f.read()
    if "car_model_idx: r.car_model_idx" not in content:
        content = content.replace("morpho: r.morpho,\n            }", "morpho: r.morpho,\n                car_model_idx: r.car_model_idx,\n            }")
    if "race.best_car_model_idx = Some(best.car_model_idx)" not in content:
        content = content.replace("race.best_weights = Some(weights.clone());", "race.best_weights = Some(weights.clone());\n                race.best_car_model_idx = Some(best.car_model_idx);")
    if "race.best_car_model_idx = None" not in content:
        content = content.replace("race.best_weights = None;", "race.best_weights = None;\n            race.best_car_model_idx = None;")
    with open(path, "w") as f:
        f.write(content)

patch_racer()
patch_settings()
patch_dna()
patch_pipeline()
print("Patched basic structs")
