use universe::gpu::school::{R_FAIL_CODE, R_SCORE, R_STATUS, STATUS_FAILED, STATUS_SUCCESS};

#[derive(Clone, Debug, PartialEq)]
pub struct SchoolAggregate {
    pub genome: usize,
    pub score: f64,
    pub mean_score: f64,
    pub worst_score: f64,
    pub worst_failure_penalty: f64,
    pub successes: usize,
    pub failures: usize,
    pub deaths: usize,
    pub starts: usize,
}

pub fn aggregate_school_batch(
    replica_records: &[Vec<f32>],
    genome_count: usize,
    starts_per_genome: usize,
) -> Vec<SchoolAggregate> {
    assert!(genome_count > 0, "genome count must be positive");
    assert!(starts_per_genome > 0, "starts per genome must be positive");
    assert_eq!(
        replica_records.len(),
        genome_count * starts_per_genome,
        "record count must equal genome_count * starts_per_genome"
    );
    replica_records
        .chunks_exact(starts_per_genome)
        .enumerate()
        .map(|(genome, batch)| {
            assert!(
                batch.iter().all(|row| row.len() > R_FAIL_CODE),
                "school record is shorter than its schema"
            );
            let mean_score =
                batch.iter().map(|row| row[R_SCORE] as f64).sum::<f64>() / starts_per_genome as f64;
            let worst_score = batch
                .iter()
                .map(|row| row[R_SCORE] as f64)
                .fold(f64::INFINITY, f64::min);
            let successes = batch
                .iter()
                .filter(|row| row[R_STATUS] as u32 == STATUS_SUCCESS)
                .count();
            let failures = batch
                .iter()
                .filter(|row| row[R_STATUS] as u32 == STATUS_FAILED)
                .count();
            let deaths = batch
                .iter()
                .filter(|row| row[R_FAIL_CODE] as u32 == 2)
                .count();
            let worst_failure_penalty = batch
                .iter()
                .map(|row| {
                    if row[R_STATUS] as u32 != STATUS_FAILED {
                        0.0
                    } else if row[R_FAIL_CODE] as u32 == 2 {
                        300.0
                    } else {
                        200.0
                    }
                })
                .fold(0.0_f64, f64::max);
            // K=1 is the compatibility arm: its score is byte-for-byte the old
            // selection input. Robustness pressure only exists when batching is asked for.
            let score = if starts_per_genome == 1 {
                mean_score
            } else {
                mean_score - worst_failure_penalty
            };
            SchoolAggregate {
                genome,
                score,
                mean_score,
                worst_score,
                worst_failure_penalty,
                successes,
                failures,
                deaths,
                starts: starts_per_genome,
            }
        })
        .collect()
}

pub fn expand_replicas<T: Clone>(genomes: &[T], starts_per_genome: usize) -> Vec<T> {
    assert!(starts_per_genome > 0, "starts per genome must be positive");
    genomes
        .iter()
        .flat_map(|genome| std::iter::repeat_n(genome.clone(), starts_per_genome))
        .collect()
}

pub fn representatives<T: Clone>(replicas: &[T], starts_per_genome: usize) -> Vec<T> {
    assert!(starts_per_genome > 0, "starts per genome must be positive");
    assert_eq!(
        replicas.len() % starts_per_genome,
        0,
        "replica count must form complete batches"
    );
    replicas
        .chunks_exact(starts_per_genome)
        .map(|batch| batch[0].clone())
        .collect()
}

/// Number of distinct lesson geometries the city must provide. Batched replicas
/// deliberately share the same K starts across genomes; asking for every physical
/// slot would turn `genomes * K` into a false city-capacity requirement.
pub fn required_start_count(physical_slots: usize, starts_per_genome: usize) -> usize {
    assert!(physical_slots > 0, "physical slot count must be positive");
    assert!(starts_per_genome > 0, "starts per genome must be positive");
    if starts_per_genome == 1 {
        physical_slots
    } else {
        starts_per_genome
    }
}

/// Every genome sees the same ordered start batch. A generation rotation changes
/// that batch deterministically without assigning an easier start to one lineage.
pub fn start_index(
    replica_slot: usize,
    starts_per_genome: usize,
    candidate_count: usize,
    rotation: usize,
) -> usize {
    assert!(starts_per_genome > 0 && candidate_count > 0);
    (replica_slot % starts_per_genome + rotation) % candidate_count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(score: f32, status: u32, fail_code: u32) -> Vec<f32> {
        let mut row = vec![0.0; universe::gpu::school::SCHOOL_STRIDE_F32];
        row[R_SCORE] = score;
        row[R_STATUS] = status as f32;
        row[R_FAIL_CODE] = fail_code as f32;
        row
    }

    #[test]
    fn aggregates_contiguous_replicas_without_crossing_genomes() {
        let rows = vec![
            record(10.0, 1, 0),
            record(20.0, 1, 0),
            record(30.0, 2, 1),
            record(100.0, 1, 0),
            record(110.0, 1, 0),
            record(120.0, 1, 0),
        ];
        let aggregate = aggregate_school_batch(&rows, 2, 3);
        assert_eq!(aggregate.len(), 2);
        assert_eq!(aggregate[0].mean_score, 20.0);
        assert_eq!(aggregate[0].score, -180.0);
        assert_eq!((aggregate[0].successes, aggregate[0].failures), (2, 1));
        assert_eq!(aggregate[1].mean_score, 110.0);
        assert_eq!(aggregate[1].score, 110.0);
    }

    #[test]
    fn death_is_counted_and_is_the_worst_failure_penalty() {
        let rows = vec![record(50.0, 2, 1), record(60.0, 2, 2), record(70.0, 3, 0)];
        let aggregate = aggregate_school_batch(&rows, 1, 3);
        assert_eq!(aggregate[0].deaths, 1);
        assert_eq!(aggregate[0].worst_failure_penalty, 300.0);
        assert_eq!(aggregate[0].score, -240.0);
    }

    #[test]
    fn one_start_is_bit_compatible_with_historical_score() {
        let rows = vec![record(-42.5, 2, 2), record(91.25, 1, 0)];
        let aggregate = aggregate_school_batch(&rows, 2, 1);
        assert_eq!(aggregate[0].score, -42.5);
        assert_eq!(aggregate[1].score, 91.25);
    }

    #[test]
    fn expansion_assigns_three_identical_genomes_to_each_batch() {
        let expanded = expand_replicas(&[11_u64, 29_u64], 3);
        assert_eq!(expanded, vec![11, 11, 11, 29, 29, 29]);
        assert_eq!(expanded.len(), 6);
        assert_eq!(representatives(&expanded, 3), vec![11, 29]);
        let starts: Vec<_> = (0..6).map(|slot| start_index(slot, 3, 8, 2)).collect();
        assert_eq!(starts, vec![2, 3, 4, 2, 3, 4]);
        assert_eq!(required_start_count(6, 3), 3);
        assert_eq!(required_start_count(6, 1), 6);
    }

    #[test]
    #[should_panic(expected = "record count")]
    fn rejects_incomplete_batches() {
        let _ = aggregate_school_batch(&[record(1.0, 1, 0)], 2, 3);
    }
}
