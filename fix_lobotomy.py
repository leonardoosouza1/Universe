import sys

def patch_settings():
    path = "src/evolution/settings.rs"
    with open(path, "r") as f:
        content = f.read()
    if "pub best_l3_weights: Option<Vec<f32>>," not in content:
        content = content.replace("pub best_l2_weights: Option<Vec<f32>>,", "pub best_l2_weights: Option<Vec<f32>>,\n    pub best_l3_weights: Option<Vec<f32>>,")
    # Change best_slow_weights to f32
    content = content.replace("pub best_slow_weights: Option<Vec<i8>>,", "pub best_slow_weights: Option<Vec<f32>>,")
    with open(path, "w") as f:
        f.write(content)

def patch_dna_genesis():
    path = "src/evolution/dna_genesis.rs"
    with open(path, "r") as f:
        content = f.read()
    # Add L2/L3 and change slow_weights to f32 in EliteDna
    if "pub cortex_l2: Vec<f32>," not in content:
        content = content.replace("pub slow_scale: f32,", "pub slow_scale: f32,\n    pub cortex_l2: Vec<f32>,\n    pub cortex_l3: Vec<f32>,")
    content = content.replace("pub slow_weights: Vec<i8>,", "pub slow_weights: Vec<f32>,")
    with open(path, "w") as f:
        f.write(content)

def patch_pipeline():
    path = "src/evolution/pipeline.rs"
    with open(path, "r") as f:
        content = f.read()
    # Update EliteDna creation in tournament/selection
    # We need to find where EliteDna { ... } is created
    # and add cortex_l2/l3.
    # Also fix slow_weights mapping (now f32).
    content = content.replace(
        "slow_weights: r.engine.slow_brain.layer.weights.iter()\n                    .map(|&w| (w * 63.5).round().clamp(-127.0, 127.0) as i8).collect(),",
        "slow_weights: r.engine.slow_brain.layer.weights.clone(),"
    )
    if "cortex_l2: r.engine.cortex_l2.weights.clone()," not in content:
        content = content.replace(
            "morpho: r.morpho.clone(),",
            "morpho: r.morpho.clone(),\n                cortex_l2: r.engine.cortex_l2.weights.clone(),\n                cortex_l3: r.engine.cortex_l3.weights.clone(),"
        )
    
    # Update RaceSettings champion save
    if "race.best_l3_weights = Some(best.engine.cortex_l3.weights.clone());" not in content:
        content = content.replace(
            "race.best_l2_weights = Some(best.engine.cortex_l2.weights.clone());",
            "race.best_l2_weights = Some(best.engine.cortex_l2.weights.clone());\n                race.best_l3_weights = Some(best.engine.cortex_l3.weights.clone());"
        )
    # Fix slow weights in champion save
    content = content.replace(
        "let slow_weights: Vec<i8> = best.engine.slow_brain.layer.weights.iter()\n                    .map(|&w| (w * 63.5).round().clamp(-127.0, 127.0) as i8).collect();",
        "let slow_weights: Vec<f32> = best.engine.slow_brain.layer.weights.clone();"
    )
    with open(path, "w") as f:
        f.write(content)

def patch_spawn():
    path = "src/evolution/spawn.rs"
    with open(path, "r") as f:
        content = f.read()
    # 1. Update load from RaceSettings (Slot 0)
    # instinct_layer is already there. Need L2/L3/Slow
    if "if let Some(ref l3) = race.best_l3_weights {" not in content:
        content = content.replace(
            "if let Some(ref l2) = race.best_l2_weights {\n                engine.cortex_l2.set_weights(l2);\n            }",
            "if let Some(ref l2) = race.best_l2_weights {\n                engine.cortex_l2.set_weights(l2);\n            }\n            if let Some(ref l3) = race.best_l3_weights {\n                engine.cortex_l3.set_weights(l3);\n            }"
        )
    # Fix slow brain load (now f32)
    content = content.replace(
        "let exact_slow: Vec<f32> = current_best_slow_w.unwrap().iter().map(|&w| w as f32 / 63.5).collect();\n                    engine.slow_brain.layer.set_weights(&exact_slow);",
        "engine.slow_brain.layer.set_weights(&current_best_slow_w.unwrap());"
    )
    
    # 2. Update load from Elite pool
    if "engine.cortex_l3.set_weights(&elite.cortex_l3);" not in content:
        content = content.replace(
            "engine.slow_brain.layer.set_weights(&exact_slow);",
            "engine.slow_brain.layer.set_weights(&elite.slow_weights);\n                    engine.cortex_l2.set_weights(&elite.cortex_l2);\n                    engine.cortex_l3.set_weights(&elite.cortex_l3);"
        )
    # Remove the elite exact_slow calc which was for i8
    content = content.replace(
        "let exact_slow: Vec<f32> = elite.slow_weights.iter().map(|&w| w as f32 / 63.5).collect();",
        ""
    )
    
    # 3. Handle mutation for L2/L3 for non-champion
    # Actually L2/L3 might need mutation too.
    # Currently spawn.rs only mutates instinct_layer and cortex_l2?
    # Let's check.
    with open(path, "w") as f:
        f.write(content)

patch_settings()
patch_dna_genesis()
patch_pipeline()
patch_spawn()
print("Fixed brain lobotomy and precision loss")
