import sys

path = "src/evolution/spawn.rs"
with open(path, "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if "let (racer_entity, car_model) = match i % 5 {" in line:
        new_lines.extend([
            "        let elite_range_end = 1 + config.elite_clones;\n",
            "        let mut spawn_model_idx = i % 5;\n",
            "        if slot_is_champion {\n",
            "            if let Some(idx) = race.best_car_model_idx {\n",
            "                spawn_model_idx = idx as usize;\n",
            "            }\n",
            "        } else if i >= 1 && i < elite_range_end {\n",
            "            if i - 1 < race.elite_pool.len() {\n",
            "                let elite = &race.elite_pool[i - 1];\n",
            "                spawn_model_idx = elite.car_model_idx as usize;\n",
            "            }\n",
            "        }\n",
            "        let (racer_entity, car_model) = match spawn_model_idx % 5 {\n"
        ])
    elif "car_model: car_model.to_string()," in line:
        new_lines.append(line)
        new_lines.append("                car_model_idx: (spawn_model_idx % 5) as u8,\n")
    else:
        new_lines.append(line)

with open(path, "w") as f:
    f.writelines(new_lines)

print("Patched spawn.rs")
