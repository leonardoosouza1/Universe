import sys

def fix_pipeline():
    path = "src/evolution/pipeline.rs"
    with open(path, "r") as f:
        content = f.read()
    # EliteDna literal missing l2/l3
    if "cortex_l2: r.engine.cortex_l2.weights.clone()," not in content:
        content = content.replace(
            "morpho: r.morpho,",
            "morpho: r.morpho,\n                cortex_l2: r.engine.cortex_l2.weights.clone(),\n                cortex_l3: r.engine.cortex_l3.weights.clone(),"
        )
    with open(path, "w") as f:
        f.write(content)

fix_pipeline()
print("Fixed missing fields in EliteDna literal")
