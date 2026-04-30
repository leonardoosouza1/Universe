import sys

def fix_pipeline():
    path = "src/evolution/pipeline.rs"
    with open(path, "r") as f:
        content = f.read()
    if "car_model_idx: r.car_model_idx," not in content:
        content = content.replace("morpho: r.morpho,", "morpho: r.morpho,\n                car_model_idx: r.car_model_idx,")
    with open(path, "w") as f:
        f.write(content)

def fix_dna_genesis():
    path = "src/evolution/dna_genesis.rs"
    with open(path, "r") as f:
        content = f.read()
    # DnaPayload struct literal should have car_model_idx exactly once
    # Line 805 has it twice
    content = content.replace("fitness: 0.0, car_model_idx: 0, version,", "fitness: 0.0, version,")
    # Also some redundant car_model_idx: 0 at the start of literals from previous failed script
    # I will just clean up DnaPayload literals
    import re
    content = re.sub(r"DnaPayload \{\n\s*car_model_idx: 0,\n\s*car_model_idx: 0,", "DnaPayload {\n                car_model_idx: 0,", content)
    # And fix the ones that don't have it yet if any
    # Actually I will use a more robust way: find all DnaPayload { ... } and make sure car_model_idx is there
    # But for now let is just fix the known errors
    with open(path, "w") as f:
        f.write(content)

fix_pipeline()
fix_dna_genesis()
