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
    let population: Vec<_> = genomes
        .iter()
        .enumerate()
        .map(|(being, genome)| {
            let p = at_birth(genome, enabled, [0.0; 3], [1.0, 0.0, 0.0]);
            serde_json::json!({"being":being,"genes":genome.body.0,
            "organ_genes":genome.organs.0,"lifetime_genes":genome.lifetime.0,
            "tank_scale":if enabled {genome.organs.tank_scale()} else {1.0},
            "cooling_scale":1.0+p._pad_before_wind,
            "mass_kg":p.mass,"engine_max_torque_nm":p.max_torque,
            "brake_max_torque_nm":p.brake_torque_max,
            "fuel_load_scale":universe::gpu::body::fuel_load_scale(&p)})
        })
        .collect();
    let path = Path::new(out).join(format!("body_gen{generation:04}.json"));
    let file = std::fs::File::create(path)?;
    serde_json::to_writer(
        file,
        &serde_json::json!({"schema_version":2,
        "generation":generation,"body_enabled":enabled,"population":population,
        "mutation":{"distribution":"uniform_additive","half_width":0.1,"bounds":[-1,1]},
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
