use universe::gpu::school::{R_FAIL_CODE, R_SCORE, R_STATUS, STATUS_FAILED, STATUS_SUCCESS};

/// How the K per-start scores of one genome become its selection score
/// (`--school-select`). `Mean` is the historical rule: mean minus the worst failure
/// penalty. `Worst` selects on the lowest start score, so a genome cannot pay for two
/// bad starts with one good one (docs/GAP-SWEEP-20260924.md P0-3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchoolSelection {
    Mean,
    Worst,
}

impl SchoolSelection {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "mean" => Some(Self::Mean),
            "worst" => Some(Self::Worst),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Worst => "worst",
        }
    }
}

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

/// A life-mode school observer must not replace the production objective.
/// CLI validation permits replicated starts only in bench mode.
pub fn selection_scores(
    production: &[f64],
    school: Option<&[SchoolAggregate]>,
    bench: bool,
) -> Vec<f64> {
    if bench {
        if let Some(aggregates) = school {
            return aggregates.iter().map(|aggregate| aggregate.score).collect();
        }
    }
    production.to_vec()
}

pub fn aggregate_school_batch(
    replica_records: &[Vec<f32>],
    genome_count: usize,
    starts_per_genome: usize,
    selection: SchoolSelection,
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
                match selection {
                    SchoolSelection::Mean => mean_score - worst_failure_penalty,
                    // The failed replica already carries its -200 inside R_SCORE.
                    SchoolSelection::Worst => worst_score,
                }
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
/// With K = 1 every slot has its own start.
/// With K > 1 the pool defaults to exactly K, which makes the per-generation rotation
/// in [`start_index`] a no-op (`gen * K mod K == 0`): the same K starts every
/// generation. `--school-start-pool N` (N >= K) loads N candidates so the batch of K
/// rotates through them (docs/GAP-SWEEP-20260924.md P0-2).
pub fn required_start_pool(physical_slots: usize, starts_per_genome: usize, pool: usize) -> usize {
    assert!(physical_slots > 0, "physical slot count must be positive");
    assert!(starts_per_genome > 0, "starts per genome must be positive");
    if starts_per_genome == 1 {
        physical_slots
    } else {
        pool.max(starts_per_genome)
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

    #[test]
    fn life_observation_cannot_replace_the_production_ranking() {
        // Deliberately opposite rankings: a lesson specialist is not necessarily
        // the best taxi. Only an explicit bench may select by the lesson.
        let school = aggregate_school_batch(
            &[
                record(1000.0, STATUS_SUCCESS, 0),
                record(-200.0, STATUS_FAILED, 1),
            ],
            2,
            1,
            SchoolSelection::Mean,
        );
        let production = [-40.0, 1200.0];
        assert_eq!(
            selection_scores(&production, Some(&school), false),
            production
        );
        assert_eq!(
            selection_scores(&production, Some(&school), true),
            [1000.0, -200.0]
        );
        assert_eq!(selection_scores(&production, None, false), production);
        assert_eq!(selection_scores(&production, None, true), production);
    }

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
        let aggregate = aggregate_school_batch(&rows, 2, 3, SchoolSelection::Mean);
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
        let aggregate = aggregate_school_batch(&rows, 1, 3, SchoolSelection::Mean);
        assert_eq!(aggregate[0].deaths, 1);
        assert_eq!(aggregate[0].worst_failure_penalty, 300.0);
        assert_eq!(aggregate[0].score, -240.0);
    }

    #[test]
    fn one_start_is_bit_compatible_with_historical_score() {
        let rows = vec![record(-42.5, 2, 2), record(91.25, 1, 0)];
        let aggregate = aggregate_school_batch(&rows, 2, 1, SchoolSelection::Worst);
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
        assert_eq!(required_start_pool(6, 3, 0), 3);
        assert_eq!(required_start_pool(6, 1, 0), 6);
    }

    #[test]
    #[should_panic(expected = "record count")]
    fn rejects_incomplete_batches() {
        let _ = aggregate_school_batch(&[record(1.0, 1, 0)], 2, 3, SchoolSelection::Mean);
    }

    /// The 2026-09-23 seed-11 champion: 2102 on one start, 176 and 264 truncated on the
    /// other two. Under the mean it beats an all-round overshooter; under `worst` the
    /// overshooter (300 - 200 each start, none truncated) wins.
    #[test]
    fn worst_selection_denies_a_single_start_specialist() {
        let specialist = vec![
            record(2102.0, 1, 0),
            record(176.0, 3, 0),
            record(264.0, 3, 0),
        ];
        let overshooter = vec![
            record(100.0, 2, 5),
            record(100.0, 2, 5),
            record(100.0, 2, 5),
        ];
        let rows: Vec<Vec<f32>> = specialist.into_iter().chain(overshooter).collect();
        let mean = aggregate_school_batch(&rows, 2, 3, SchoolSelection::Mean);
        assert!((mean[0].score - 847.333).abs() < 0.01, "{:?}", mean[0]);
        assert_eq!(mean[1].score, 100.0 - 200.0);
        assert!(mean[0].score > mean[1].score);
        let worst = aggregate_school_batch(&rows, 2, 3, SchoolSelection::Worst);
        assert_eq!(worst[0].score, 176.0);
        assert_eq!(worst[1].score, 100.0);
        assert!(
            worst[0].score > worst[1].score,
            "a truncated crawler still beats a failed mover on score alone; the deadline flag is what turns truncation into failure"
        );
        let crawler_under_deadline = vec![
            record(2102.0, 1, 0),
            record(-24.0, 2, 16),
            record(64.0, 2, 16),
        ];
        let worst = aggregate_school_batch(&crawler_under_deadline, 1, 3, SchoolSelection::Worst);
        assert_eq!(worst[0].score, -24.0);
        assert_eq!(worst[0].failures, 2);
    }

    #[test]
    fn selection_mode_round_trips_by_name() {
        for mode in [SchoolSelection::Mean, SchoolSelection::Worst] {
            assert_eq!(SchoolSelection::parse(mode.name()), Some(mode));
        }
        assert_eq!(SchoolSelection::parse("median"), None);
    }

    /// With pool == K the rotation is a no-op: the same three candidates every
    /// generation. With a pool of 9 the batch walks the pool and returns after 3.
    #[test]
    fn start_pool_makes_the_generation_rotation_real() {
        let fixed: Vec<Vec<usize>> = (0..4)
            .map(|generation| {
                (0..3)
                    .map(|slot| start_index(slot, 3, 3, generation * 3))
                    .collect()
            })
            .collect();
        assert!(fixed.iter().all(|g| *g == vec![0, 1, 2]), "{fixed:?}");
        let rotating: Vec<Vec<usize>> = (0..4)
            .map(|generation| {
                (0..3)
                    .map(|slot| start_index(slot, 3, 9, generation * 3))
                    .collect()
            })
            .collect();
        assert_eq!(
            rotating,
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![0, 1, 2]]
        );
        assert_eq!(required_start_pool(6, 3, 0), 3);
        assert_eq!(required_start_pool(6, 3, 9), 9);
        assert_eq!(
            required_start_pool(6, 3, 2),
            3,
            "a pool below K is raised to K"
        );
        assert_eq!(required_start_pool(6, 1, 9), 6, "K = 1 ignores the pool");
    }
}

/// Population evidence over a complete pool window, not a frozen-policy test.
/// A specialist cannot promote a lesson while any observed start has zero success.
#[derive(Default)]
pub struct CoverageWindow {
    counts: Vec<(usize, usize)>,
}
impl CoverageWindow {
    pub fn observe(&mut self, pool: usize, rows: &[(usize, bool)]) -> Option<f64> {
        if self.counts.len() != pool {
            self.counts = vec![(0, 0); pool];
        }
        for &(start, success) in rows {
            let count = self
                .counts
                .get_mut(start)
                .expect("start index outside pool");
            count.0 += usize::from(success);
            count.1 += 1;
        }
        if pool == 0 || self.counts.iter().any(|(_, total)| *total == 0) {
            return None;
        }
        let minimum = self
            .counts
            .iter()
            .map(|(success, total)| *success as f64 / *total as f64)
            .fold(1.0, f64::min);
        self.counts.fill((0, 0));
        Some(minimum)
    }
    pub fn clear(&mut self) {
        self.counts.clear();
    }
}
