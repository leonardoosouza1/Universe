import sys

path = "src/simulation/motor.rs"
with open(path, "r") as f:
    content = f.read()

# 1. Add seeded RNG initialization
# We want to do this at the start of execute_cycle (which is the main loop).
# Let's find execute_cycle.
if "let mut rng = rand::rng();" in content:
    # Seed based on stable_seed + grace_ticks (frame counter)
    content = content.replace(
        "let mut rng = rand::rng();",
        "let mut rng = rand::rngs::StdRng::seed_from_u64(racer.stable_seed + racer.grace_ticks as u64);"
    )

# 2. Replace rand::random::<f32>() with rng.random::<f32>()
# BUT we need to make sure rng is available.
# Actually, I should just search and replace rand::random with rng.random
content = content.replace("rand::random::<f32>()", "rng.random::<f32>()")

# 3. Handle the second rand::rng() at 1336
content = content.replace(
    "let mut rng = rand::rng();",
    "let mut rng = rand::rngs::StdRng::seed_from_u64(racer.stable_seed + racer.grace_ticks as u64 + 1000);" # offset to avoid correlation
)

with open(path, "w") as f:
    f.write(content)

print("Patched motor.rs for determinism")
