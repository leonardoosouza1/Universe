//! Birth-time body expression and generation-level provenance, shared by train/holdout.

use std::{io, path::Path};
use universe::gpu::{
    evolution::Genome,
    physics::{PhysicsStateGpu, make_physics_state},
};

pub fn at_birth(
    genome: &Genome,
    enabled: bool,
    position: [f32; 3],
    forward: [f32; 3],
) -> PhysicsStateGpu {
    let mut state = make_physics_state(position, forward);
    if enabled {
        genome.body.express(&mut state);
        genome.organs.express(&mut state);
        state.config_flags |= 1 << 7;
    }
    state
}

pub fn record(out: &str, generation: usize, genomes: &[Genome], enabled: bool) -> io::Result<()> {
    record_with_control(out, generation, genomes, enabled, None)
}

pub fn record_with_control(
    out: &str,
    generation: usize,
    genomes: &[Genome],
    enabled: bool,
    control: Option<&str>,
) -> io::Result<()> {
    let population: Vec<_> = genomes
        .iter()
        .enumerate()
        .map(|(being, genome)| {
            let p = at_birth(genome, enabled, [0.0; 3], [1.0, 0.0, 0.0]);
            serde_json::json!({"being":being,"genes":genome.body.0,
            "organ_genes":genome.organs.0,"lifetime_genes":genome.lifetime.0,
            // The organism-taxi metabolism block (efficiency, durability, repair;
            // src/gpu/organism.wgsl) mutates and is uploaded every generation even with a
            // fixed body, and was invisible in this record until 2026-09-24.
            "organism_genes":genome.organism,
            "tank_scale":if enabled {genome.organs.tank_scale()} else {1.0},
            "cooling_scale":1.0+p._pad_before_wind,
            "mass_kg":p.mass,"engine_max_torque_nm":p.max_torque,
            "brake_max_torque_nm":p.brake_torque_max,
            "fuel_load_scale":universe::gpu::body::fuel_load_scale(&p)})
        })
        .collect();
    let mutation = match control {
        Some("fixed") => {
            serde_json::json!({"distribution":"none", "birth_reset":"neutral_morphology"})
        }
        Some("random") => {
            serde_json::json!({"distribution":"independent_uniform_redraw", "bounds":[-1,1], "frequency":"generation"})
        }
        _ if enabled => {
            serde_json::json!({"distribution":"uniform_additive","half_width":0.1,"bounds":[-1,1]})
        }
        _ => serde_json::json!({"distribution":"none", "birth_reset":null}),
    };
    let path = Path::new(out).join(format!("body_gen{generation:04}.json"));
    let file = std::fs::File::create(path)?;
    serde_json::to_writer(
        file,
        &serde_json::json!({"schema_version":3,
        "generation":generation,"body_enabled":enabled,"population":population,
        "mutation":mutation,"research_control":control,"organism_metabolism_neutralized":control.is_some(),
        "model":"capacity_mass_tradeoff_v1"}),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use universe::gpu::body::BodyGenes;

    #[test]
    fn fixed_body_ignores_inherited_capacity_without_erasing_dna() {
        let mut genome = Genome::random(4);
        genome.body = BodyGenes([1.0, -1.0]);
        let neutral = make_physics_state([0.0; 3], [1.0, 0.0, 0.0]);
        let fixed = at_birth(&genome, false, [0.0; 3], [1.0, 0.0, 0.0]);
        assert_eq!(bytemuck::bytes_of(&neutral), bytemuck::bytes_of(&fixed));
        let evolved = at_birth(&genome, true, [0.0; 3], [1.0, 0.0, 0.0]);
        assert!(evolved.max_torque > fixed.max_torque);
        assert!(evolved.brake_torque_max < fixed.brake_torque_max);
        assert_eq!(genome.body.0, [1.0, -1.0]);
    }

    /// The organism metabolism block evolves even under `--fixed-body`; the record must
    /// show it so a school run can tell whether that block drifted.
    #[test]
    fn record_carries_the_organism_block_with_a_fixed_body() {
        let mut genome = Genome::random(4);
        genome.organism[12] = 0.5;
        let dir = std::env::temp_dir().join(format!("body-record-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        record(
            dir.to_str().unwrap(),
            3,
            std::slice::from_ref(&genome),
            false,
        )
        .unwrap();
        let text = std::fs::read_to_string(dir.join("body_gen0003.json")).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let being = &json["population"][0];
        assert_eq!(being["organism_genes"][12], 0.5);
        assert_eq!(being["organism_genes"].as_array().unwrap().len(), 16);
        assert_eq!(json["body_enabled"], false);
        assert_eq!(being["mass_kg"], 1350.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// Per-being capacity; disabled bodies retain the historical global tank.
pub fn tank_capacity(genome: &Genome, enabled: bool, base: f32) -> f64 {
    f64::from(base)
        * if enabled {
            f64::from(genome.organs.tank_scale())
        } else {
            1.0
        }
}

/// Morphology comparison: neutral metabolism in every arm; random morphology is
/// redrawn independently of selection, with identical bodies across K replicas.
/// This is generation setup only and adds no per-tick device transfers.
pub fn apply_research_control(
    genomes: &mut [Genome],
    k: usize,
    mode: &str,
    generation: usize,
    seed: u64,
) {
    assert!(k > 0 && genomes.len() % k == 0);
    assert!(matches!(mode, "fixed" | "random" | "evolved"));
    for (index, replicas) in genomes.chunks_exact_mut(k).enumerate() {
        let mut body = Default::default();
        let mut organs = Default::default();
        if mode == "random" {
            let mut model = replicas[0].clone();
            let mut state = seed
                ^ (generation as u64).wrapping_mul(0x9e3779b97f4a7c15)
                ^ (index as u64).wrapping_mul(0xbf58476d1ce4e5b9)
                ^ 0x52414e444f4d;
            for value in model.body.0.iter_mut().chain(model.organs.0.iter_mut()) {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^= z >> 31;
                *value = 2.0 * ((z >> 40) as f32 / 16777216.0) - 1.0;
            }
            body = model.body;
            organs = model.organs;
        }
        for genome in replicas {
            genome.organism = [0.0; 16];
            if mode != "evolved" {
                genome.body = body;
                genome.organs = organs;
            }
        }
    }
}
