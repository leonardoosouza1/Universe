//! Cheap generation-boundary diagnostics that prove the population actually changes.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use universe::config::iara::brain::L1_NEURONS;
use universe::gpu::evolution::Genome;

#[derive(Debug, PartialEq)]
pub struct GenomeDiversity {
    pub distinct_fast_hashes: usize,
    pub fast_rms_from_slot0: f64,
    pub distinct_brake_row_hashes: usize,
    pub brake_row_rms_from_slot0: f64,
}

fn hash_floats(values: impl Iterator<Item = f32>) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for value in values {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn rms_from(reference: impl Iterator<Item = f32>, sample: impl Iterator<Item = f32>) -> f64 {
    let mut squared = 0.0_f64;
    let mut count = 0_u64;
    for (a, b) in reference.zip(sample) {
        squared += f64::from(a - b).powi(2);
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        (squared / count as f64).sqrt()
    }
}

/// Measures fast-brain diversity, including the L2 row that directly drives braking.
pub fn measure(genomes: &[Genome]) -> GenomeDiversity {
    let Some(reference) = genomes.first() else {
        return GenomeDiversity {
            distinct_fast_hashes: 0,
            fast_rms_from_slot0: 0.0,
            distinct_brake_row_hashes: 0,
            brake_row_rms_from_slot0: 0.0,
        };
    };
    let brake_start = 2 * L1_NEURONS;
    let brake_end = brake_start + L1_NEURONS;
    let reference_fast = || reference.l1.iter().chain(reference.l2.iter()).copied();
    let reference_brake = || reference.l2[brake_start..brake_end].iter().copied();
    let mut fast_hashes = HashSet::with_capacity(genomes.len());
    let mut brake_hashes = HashSet::with_capacity(genomes.len());
    let mut fast_rms = 0.0;
    let mut brake_rms = 0.0;
    for genome in genomes {
        let fast = || genome.l1.iter().chain(genome.l2.iter()).copied();
        let brake = || genome.l2[brake_start..brake_end].iter().copied();
        fast_hashes.insert(hash_floats(fast()));
        brake_hashes.insert(hash_floats(brake()));
        fast_rms += rms_from(reference_fast(), fast());
        brake_rms += rms_from(reference_brake(), brake());
    }
    let n = genomes.len() as f64;
    GenomeDiversity {
        distinct_fast_hashes: fast_hashes.len(),
        fast_rms_from_slot0: fast_rms / n,
        distinct_brake_row_hashes: brake_hashes.len(),
        brake_row_rms_from_slot0: brake_rms / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_are_reported_as_one_fast_and_brake_genome() {
        let genome = Genome::random(7);
        let measured = measure(&[genome.clone(), genome]);
        assert_eq!(measured.distinct_fast_hashes, 1);
        assert_eq!(measured.distinct_brake_row_hashes, 1);
        assert_eq!(measured.fast_rms_from_slot0, 0.0);
        assert_eq!(measured.brake_row_rms_from_slot0, 0.0);
    }

    #[test]
    fn a_brake_row_mutation_is_visible_in_both_metrics() {
        let original = Genome::random(11);
        let mut child = original.clone();
        child.l2[2 * L1_NEURONS + 3] += 0.5;
        let measured = measure(&[original, child]);
        assert_eq!(measured.distinct_fast_hashes, 2);
        assert_eq!(measured.distinct_brake_row_hashes, 2);
        assert!(measured.fast_rms_from_slot0 > 0.0);
        assert!(measured.brake_row_rms_from_slot0 > 0.0);
    }
}
