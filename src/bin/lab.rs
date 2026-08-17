use serde_json::json;
use std::fs;
use std::process::{Child, Command};
use std::thread;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse --spec argument
    let spec_path = args
        .iter()
        .position(|a| a == "--spec")
        .and_then(|i| args.get(i + 1))
        .expect("--spec <path> argument required");

    // Parse --gens argument, default to "20"
    let gens = args
        .iter()
        .position(|a| a == "--gens")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("20");

    // Read and parse spec JSON
    let spec_json = fs::read_to_string(spec_path).expect("failed to read spec file");
    let spec: serde_json::Value =
        serde_json::from_str(&spec_json).expect("invalid JSON in spec file");

    let variants = spec["variants"]
        .as_array()
        .expect("spec must contain 'variants' array");

    let lab_name = spec["name"].as_str().unwrap_or("unnamed_lab");

    println!("=== Lab Runner: {} ===", lab_name);
    println!("Generations: {}", gens);
    println!("Variants: {}\n", variants.len());

    let mut handles = vec![];

    // Spawn a thread for each variant
    for (idx, variant) in variants.iter().enumerate() {
        let name = variant["name"]
            .as_str()
            .expect("each variant must have 'name'");

        let config_overrides = variant.get("config");
        let lab_params_spec = variant.get("params");

        // Build config JSON
        let mut config = json!({"selected_track_index": 0});
        if let Some(overrides) = config_overrides {
            if let Some(obj) = overrides.as_object() {
                for (k, v) in obj {
                    config[k] = v.clone();
                }
            }
        }

        // Build lab_params JSON with all defaults, then merge overrides
        let mut params = json!({
            "experiment_name": name.to_string(),
            "variant_id": idx,
            "axiom_wall_avoidance": 0.18,
            "axiom_dark_center": 0.45,
            "axiom_whisker": 0.15,
            "axiom_racing_line": 0.26,
            "axiom_decay_divisor": 50.0,
            "axiom_floor": 0.35,
            "k_smooth": 100.0,
            "k_speed": 0.5,
            "k_novel": 50.0,
            "k_stab": 200.0,
            "k_phi": 300.0,
            "k_aesth": 200.0,
            "lap_bonus": 50000.0,
            "survival_bonus": 3000.0,
            "death_penalty": 2000.0,
            "speed_bleed_damage": 15.0,
            "stagnation_secs": 15.0,
            "wall_t_threshold": 40
        });
        if let Some(lab_params) = lab_params_spec {
            if let Some(obj) = lab_params.as_object() {
                for (k, v) in obj {
                    params[k] = v.clone();
                }
            }
        }

        // Write temp files
        let config_path = format!("/tmp/lab_{}_config.json", name);
        let params_path = format!("/tmp/lab_{}_params.json", name);

        fs::write(
            &config_path,
            serde_json::to_string_pretty(&config).expect("failed to serialize config"),
        )
        .expect("failed to write config file");

        fs::write(
            &params_path,
            serde_json::to_string_pretty(&params).expect("failed to serialize params"),
        )
        .expect("failed to write params file");

        // Spawn runner thread
        let gens_copy = gens.to_string();
        let name_copy = name.to_string();
        let config_path_copy = config_path.clone();
        let params_path_copy = params_path.clone();

        let handle = thread::spawn(move || {
            println!("[{}] Starting universe simulation...", name_copy);

            let output = Command::new("./target/release/universe")
                .args(&[
                    "--headless",
                    "--gens",
                    &gens_copy,
                    "--lab-params",
                    &params_path_copy,
                    "--config",
                    &config_path_copy,
                ])
                .output()
                .expect("failed to spawn universe process");

            let success = output.status.success();
            if success {
                println!("[{}] Complete — OK", name_copy);
            } else {
                println!(
                    "[{}] Complete — FAILED (exit code: {})",
                    name_copy,
                    output.status.code().unwrap_or(-1)
                );
                if !output.stderr.is_empty() {
                    eprintln!(
                        "[{}] stderr: {}",
                        name_copy,
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
            }

            (name_copy, success)
        });

        handles.push(handle);
    }

    // Wait for all threads and collect results
    println!("\n=== Waiting for all variants to complete ===\n");

    let mut results = vec![];
    for handle in handles {
        let (name, ok) = handle.join().expect("thread panicked");
        results.push((name, ok));
    }

    // Print summary
    println!("\n=== Results ===");
    let mut passed = 0;
    let mut failed = 0;
    for (name, ok) in &results {
        if *ok {
            println!("{}: OK", name);
            passed += 1;
        } else {
            println!("{}: FAILED", name);
            failed += 1;
        }
    }

    println!(
        "\nSummary: {}/{} variants completed successfully",
        passed,
        passed + failed
    );

    if failed == 0 {
        println!("\nAll variants completed! Results are in telemetry/ directories.");
        println!("Run: python analysis/lab_compare.py --spec {}", spec_path);
    }
}
