//! Bounded retention of complete, successful GPU-evaluated competencies.
//!
//! This archive is a retention aid for the evolution loop. It is deliberately
//! not a MAP-Elites implementation and it does not route behaviour at runtime:
//! an entry is retained only after the caller has established complete success
//! for every declared start of one lesson contract. The caller owns checkpoint
//! persistence (`genome_io::save_genome`).
//!
//! Fitness is comparable only inside one `(context, lesson)` bucket. The
//! `context` string must therefore identify the start/world contract as well as
//! any other evaluation conditions that affect the score. The archive never
//! draws RNG state; reinjection is a deterministic clone operation and the
//! caller may mutate the clones afterwards if that is explicitly desired.

use std::{cmp::Ordering, fmt};

use universe::gpu::evolution::{
    Genome, N_CFC_CHANNELS, N_L1_NEURONS, N_L1_WEIGHTS, N_L2_NEURONS, N_L2_WEIGHTS, N_SENSOR_GENES,
};

/// Hard upper bound across all contexts and lessons.
///
/// The cap is intentionally independent of `capacity_per_context`, so a caller
/// cannot turn an untrusted stream of context labels into an unbounded clone
/// store. When it is reached, the oldest entry is evicted. Eviction never
/// compares scores across contexts or lessons.
pub const MAX_TOTAL_ENTRIES: usize = 64;

const N_SLOW_WEIGHTS: usize = 1_091;
const N_GATES_WEIGHTS: usize = 16;
const MAX_CONTEXT_LEN: usize = 256;
const MAX_LESSON_LEN: usize = 128;

/// Errors returned before an archive mutation occurs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArchiveError {
    /// A per-context capacity of zero cannot retain a skill.
    InvalidCapacity,
    /// The caller supplied an empty or oversized contract label.
    InvalidLabel { field: &'static str },
    /// Candidate and metric arrays describe different populations.
    LengthMismatch {
        candidates: usize,
        scores: usize,
        successes: usize,
    },
    /// A success count must be measured over at least one start.
    InvalidStarts,
    /// A candidate reported more successes than starts.
    InvalidSuccessCount {
        index: usize,
        successes: usize,
        starts: usize,
    },
    /// A score cannot order archive entries when it is NaN or infinite.
    NonFiniteScore { index: usize },
    /// A complete genome clone must retain the current GPU upload dimensions.
    GenomeDimension {
        index: usize,
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A genome containing NaN or infinity cannot become a retained skill.
    NonFiniteGenome {
        index: usize,
        field: &'static str,
        offset: usize,
    },
    /// The elite boundary cannot lie outside the offspring slice.
    InvalidEliteSlots {
        elite_slots: usize,
        offspring: usize,
    },
}

impl fmt::Display for ArchiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCapacity => write!(f, "competence archive capacity must be positive"),
            Self::InvalidLabel { field } => {
                write!(
                    f,
                    "competence archive {field} must be non-empty and within its size limit"
                )
            }
            Self::LengthMismatch {
                candidates,
                scores,
                successes,
            } => write!(
                f,
                "competence archive lengths differ: candidates={candidates}, scores={scores}, successes={successes}"
            ),
            Self::InvalidStarts => write!(f, "competence archive starts must be positive"),
            Self::InvalidSuccessCount {
                index,
                successes,
                starts,
            } => write!(
                f,
                "competence archive candidate {index} reports {successes} successes over {starts} starts"
            ),
            Self::NonFiniteScore { index } => {
                write!(
                    f,
                    "competence archive candidate {index} has a non-finite score"
                )
            }
            Self::GenomeDimension {
                index,
                field,
                expected,
                actual,
            } => write!(
                f,
                "competence archive candidate {index} has {field} dimension {actual}, expected {expected}"
            ),
            Self::NonFiniteGenome {
                index,
                field,
                offset,
            } => write!(
                f,
                "competence archive candidate {index} has a non-finite {field} value at {offset}"
            ),
            Self::InvalidEliteSlots {
                elite_slots,
                offspring,
            } => write!(
                f,
                "competence archive elite boundary {elite_slots} exceeds offspring length {offspring}"
            ),
        }
    }
}

impl std::error::Error for ArchiveError {}

/// Selects the source contracts eligible for deterministic reinjection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceFilter<'a> {
    /// Use entries from one exact context and lesson contract.
    SameContract { context: &'a str, lesson: &'a str },
    /// Use entries from one context, ranking each lesson independently before
    /// merging the results by their recorded score. The caller should use this
    /// only when the context explicitly declares a compatible lesson family.
    Context { context: &'a str },
    /// Explicit opt-in to all retained contracts.
    Any,
}

/// A retained full genome and the evidence that admitted it.
#[derive(Clone)]
pub struct ArchiveEntry {
    context: String,
    lesson: String,
    genome: Genome,
    score: f64,
    successes: usize,
    starts: usize,
    generation: usize,
    sequence: u64,
}

impl ArchiveEntry {
    pub fn context(&self) -> &str {
        &self.context
    }

    pub fn lesson(&self) -> &str {
        &self.lesson
    }

    pub fn genome(&self) -> &Genome {
        &self.genome
    }

    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn successes(&self) -> usize {
        self.successes
    }

    pub fn starts(&self) -> usize {
        self.starts
    }

    pub fn generation(&self) -> usize {
        self.generation
    }
}

/// Provenance returned for each offspring replaced by [`Archive::reinject_best`].
#[derive(Clone, Debug, PartialEq)]
pub struct ReinjectionProvenance {
    pub destination_index: usize,
    pub source_context: String,
    pub source_lesson: String,
    pub source_score: f64,
    pub source_successes: usize,
    pub source_starts: usize,
    pub source_generation: usize,
}

/// Bounded archive of complete skills observed under explicit contracts.
pub struct Archive {
    capacity_per_context: usize,
    entries: Vec<ArchiveEntry>,
    next_sequence: u64,
}

impl Archive {
    /// Create an archive with at most `capacity_per_context` entries for each
    /// context root and at most [`MAX_TOTAL_ENTRIES`] entries overall.
    pub fn new(capacity_per_context: usize) -> Result<Self, ArchiveError> {
        if capacity_per_context == 0 {
            return Err(ArchiveError::InvalidCapacity);
        }
        Ok(Self {
            capacity_per_context,
            entries: Vec::new(),
            next_sequence: 0,
        })
    }

    pub fn capacity_per_context(&self) -> usize {
        self.capacity_per_context
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries are kept in deterministic key/rank order for inspection and
    /// checkpoint manifests. The genome itself is a full clone of the input.
    pub fn entries(&self) -> impl Iterator<Item = &ArchiveEntry> {
        self.entries.iter()
    }

    /// Observe one population evaluated under one lesson contract.
    ///
    /// Only candidates with `successes[i] == starts` enter the archive. A
    /// candidate that merely has a good shaping score, or that succeeded on
    /// only part of a start batch, is intentionally not a retained competence.
    /// All candidates are validated before any retained entry is inserted.
    pub fn observe(
        &mut self,
        context: &str,
        lesson: &str,
        candidates: &[Genome],
        scores: &[f64],
        successes: &[usize],
        starts: usize,
        generation: usize,
    ) -> Result<(), ArchiveError> {
        validate_label("context", context, MAX_CONTEXT_LEN)?;
        validate_label("lesson", lesson, MAX_LESSON_LEN)?;
        if starts == 0 {
            return Err(ArchiveError::InvalidStarts);
        }
        if candidates.len() != scores.len() || candidates.len() != successes.len() {
            return Err(ArchiveError::LengthMismatch {
                candidates: candidates.len(),
                scores: scores.len(),
                successes: successes.len(),
            });
        }

        for (index, candidate) in candidates.iter().enumerate() {
            validate_genome(candidate, index)?;
            if !scores[index].is_finite() {
                return Err(ArchiveError::NonFiniteScore { index });
            }
            if successes[index] > starts {
                return Err(ArchiveError::InvalidSuccessCount {
                    index,
                    successes: successes[index],
                    starts,
                });
            }
        }

        for (index, candidate) in candidates.iter().enumerate() {
            if successes[index] != starts {
                continue;
            }
            let score = scores[index];
            if let Some(existing) = self.entries.iter_mut().find(|entry| {
                entry.context == context
                    && entry.lesson == lesson
                    && genomes_equal(&entry.genome, candidate)
            }) {
                // Re-observing the same clone cannot consume another slot. Keep
                // the strongest evidence and preserve sequence for deterministic
                // ties.
                if score > existing.score {
                    existing.score = score;
                    existing.successes = successes[index];
                    existing.starts = starts;
                    existing.generation = generation;
                }
                continue;
            }
            let sequence = self.next_sequence;
            self.next_sequence = self.next_sequence.wrapping_add(1);
            self.entries.push(ArchiveEntry {
                context: context.to_owned(),
                lesson: lesson.to_owned(),
                genome: candidate.clone(),
                score,
                successes: successes[index],
                starts,
                generation,
                sequence,
            });
        }

        self.enforce_limits(context, lesson);
        Ok(())
    }

    /// Clone the best eligible retained skills into the offspring tail.
    ///
    /// Indices `[0, elite_slots)` are never touched. At most one copy of each
    /// eligible archive entry is injected, in deterministic rank order; any
    /// remaining tail slots keep their selected offspring. This method consumes
    /// no RNG and returns provenance for every replacement.
    pub fn reinject_best(
        &self,
        offspring: &mut [Genome],
        elite_slots: usize,
        source: SourceFilter<'_>,
    ) -> Result<Vec<ReinjectionProvenance>, ArchiveError> {
        if elite_slots > offspring.len() {
            return Err(ArchiveError::InvalidEliteSlots {
                elite_slots,
                offspring: offspring.len(),
            });
        }
        validate_source(source)?;
        for (index, genome) in offspring.iter().enumerate() {
            validate_genome(genome, index)?;
        }

        let mut eligible: Vec<&ArchiveEntry> = self
            .entries
            .iter()
            .filter(|entry| matches_source(entry, source))
            .collect();
        // A score order is valid only inside one `(context, lesson)` bucket.
        // For a context-wide or global source, contract order is the primary
        // order and score is only the tie-breaker within that contract.
        eligible.sort_by(|a, b| match source {
            SourceFilter::SameContract { .. } => compare_best(a, b),
            SourceFilter::Context { .. } | SourceFilter::Any => compare_key_order(a, b),
        });

        let mut provenance = Vec::with_capacity(eligible.len().min(offspring.len() - elite_slots));
        for (destination_index, source_entry) in (elite_slots..offspring.len()).zip(eligible) {
            offspring[destination_index] = source_entry.genome.clone();
            provenance.push(ReinjectionProvenance {
                destination_index,
                source_context: source_entry.context.clone(),
                source_lesson: source_entry.lesson.clone(),
                source_score: source_entry.score,
                source_successes: source_entry.successes,
                source_starts: source_entry.starts,
                source_generation: source_entry.generation,
            });
        }
        Ok(provenance)
    }

    fn enforce_limits(&mut self, context: &str, lesson: &str) {
        while self
            .entries
            .iter()
            .filter(|entry| entry.context == context && entry.lesson == lesson)
            .count()
            > self.capacity_per_context
        {
            let index = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.context == context && entry.lesson == lesson)
                .min_by(|(_, a), (_, b)| compare_best(a, b).reverse())
                .map(|(index, _)| index)
                .expect("bucket count proved non-empty");
            self.entries.remove(index);
        }

        while self
            .entries
            .iter()
            .filter(|entry| entry.context == context)
            .count()
            > self.capacity_per_context
        {
            let index = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.context == context)
                .min_by(|(_, a), (_, b)| compare_oldest(a, b))
                .map(|(index, _)| index)
                .expect("context count proved non-empty");
            self.entries.remove(index);
        }

        while self.entries.len() > MAX_TOTAL_ENTRIES {
            let index = self
                .entries
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| compare_oldest(a, b))
                .map(|(index, _)| index)
                .expect("global count proved non-empty");
            self.entries.remove(index);
        }
        self.entries.sort_by(compare_key_order);
    }
}

fn validate_label(field: &'static str, value: &str, max_len: usize) -> Result<(), ArchiveError> {
    if value.is_empty() || value.len() > max_len {
        return Err(ArchiveError::InvalidLabel { field });
    }
    Ok(())
}

fn validate_source(source: SourceFilter<'_>) -> Result<(), ArchiveError> {
    match source {
        SourceFilter::SameContract { context, lesson } => {
            validate_label("context", context, MAX_CONTEXT_LEN)?;
            validate_label("lesson", lesson, MAX_LESSON_LEN)?;
        }
        SourceFilter::Context { context } => validate_label("context", context, MAX_CONTEXT_LEN)?,
        SourceFilter::Any => {}
    }
    Ok(())
}

fn matches_source(entry: &ArchiveEntry, source: SourceFilter<'_>) -> bool {
    match source {
        SourceFilter::SameContract { context, lesson } => {
            entry.context == context && entry.lesson == lesson
        }
        SourceFilter::Context { context } => entry.context == context,
        SourceFilter::Any => true,
    }
}

fn validate_genome(genome: &Genome, index: usize) -> Result<(), ArchiveError> {
    let dimensions = [
        ("l1", genome.l1.len(), N_L1_WEIGHTS),
        ("l2", genome.l2.len(), N_L2_WEIGHTS),
        ("slow", genome.slow.len(), N_SLOW_WEIGHTS),
        ("tau", genome.tau.len(), N_CFC_CHANNELS),
        ("sensors", genome.sensors.len(), N_SENSOR_GENES),
        (
            "neuron_alpha",
            genome.neuron_alpha.len(),
            N_L1_NEURONS + N_L2_NEURONS,
        ),
        (
            "neuron_period_l1",
            genome.neuron_period_l1.len(),
            N_L1_NEURONS,
        ),
        (
            "neuron_threshold_l1",
            genome.neuron_threshold_l1.len(),
            N_L1_NEURONS,
        ),
        (
            "neuron_activation",
            genome.neuron_activation.len(),
            N_L1_NEURONS + N_L2_NEURONS,
        ),
        ("gates_w", genome.gates_w.len(), N_GATES_WEIGHTS),
        (
            "hip",
            genome.hip.len(),
            universe::gpu::hippocampus::N_HIP_GENES,
        ),
    ];
    for &(field, actual, expected) in &dimensions {
        if actual != expected {
            return Err(ArchiveError::GenomeDimension {
                index,
                field,
                expected,
                actual,
            });
        }
    }

    for (offset, value) in genome.l1.iter().enumerate() {
        ensure_finite(index, "l1", offset, *value)?;
    }
    for (offset, value) in genome.l2.iter().enumerate() {
        ensure_finite(index, "l2", offset, *value)?;
    }
    for (row, knots) in genome.kan_knots.iter().enumerate() {
        for (column, value) in knots.iter().enumerate() {
            ensure_finite(index, "kan_knots", row * knots.len() + column, *value)?;
        }
    }
    for (field, values) in [
        ("slow", genome.slow.as_slice()),
        ("tau", genome.tau.as_slice()),
        ("sensors", genome.sensors.as_slice()),
        ("neuron_alpha", genome.neuron_alpha.as_slice()),
        ("neuron_period_l1", genome.neuron_period_l1.as_slice()),
        ("neuron_threshold_l1", genome.neuron_threshold_l1.as_slice()),
        ("neuron_activation", genome.neuron_activation.as_slice()),
        ("gates_w", genome.gates_w.as_slice()),
        ("hip", genome.hip.as_slice()),
        ("fly", genome.fly.as_slice()),
    ] {
        for (offset, value) in values.iter().enumerate() {
            ensure_finite(index, field, offset, *value)?;
        }
    }
    for (offset, value) in genome.body.0.iter().enumerate() {
        ensure_finite(index, "body", offset, *value)?;
    }
    for (offset, value) in genome.organs.0.iter().enumerate() {
        ensure_finite(index, "organs", offset, *value)?;
    }
    for (offset, value) in genome.lifetime.0.iter().enumerate() {
        ensure_finite(index, "lifetime", offset, *value)?;
    }
    for (offset, value) in genome.organism.iter().enumerate() {
        ensure_finite(index, "organism", offset, *value)?;
    }
    Ok(())
}

fn ensure_finite(
    index: usize,
    field: &'static str,
    offset: usize,
    value: f32,
) -> Result<(), ArchiveError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ArchiveError::NonFiniteGenome {
            index,
            field,
            offset,
        })
    }
}

fn genomes_equal(a: &Genome, b: &Genome) -> bool {
    a.l1.iter()
        .zip(&b.l1)
        .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.l2
            .iter()
            .zip(&b.l2)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.kan_knots.iter().zip(&b.kan_knots).all(|(row_a, row_b)| {
            row_a
                .iter()
                .zip(row_b)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        })
        && a.slow
            .iter()
            .zip(&b.slow)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.tau
            .iter()
            .zip(&b.tau)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.sensors
            .iter()
            .zip(&b.sensors)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.neuron_alpha
            .iter()
            .zip(&b.neuron_alpha)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.neuron_period_l1
            .iter()
            .zip(&b.neuron_period_l1)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.neuron_threshold_l1
            .iter()
            .zip(&b.neuron_threshold_l1)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.neuron_activation
            .iter()
            .zip(&b.neuron_activation)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.gates_w
            .iter()
            .zip(&b.gates_w)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.hip
            .iter()
            .zip(&b.hip)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.fly
            .iter()
            .zip(&b.fly)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.body
            .0
            .iter()
            .zip(&b.body.0)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.organs
            .0
            .iter()
            .zip(&b.organs.0)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.lifetime
            .0
            .iter()
            .zip(&b.lifetime.0)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.organism
            .iter()
            .zip(&b.organism)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn compare_best(a: &ArchiveEntry, b: &ArchiveEntry) -> Ordering {
    b.score
        .partial_cmp(&a.score)
        .expect("archive scores are finite")
        .then_with(|| a.generation.cmp(&b.generation))
        .then_with(|| a.context.cmp(&b.context))
        .then_with(|| a.lesson.cmp(&b.lesson))
        .then_with(|| a.sequence.cmp(&b.sequence))
}

fn compare_oldest(a: &ArchiveEntry, b: &ArchiveEntry) -> Ordering {
    a.generation
        .cmp(&b.generation)
        .then_with(|| a.sequence.cmp(&b.sequence))
        .then_with(|| a.context.cmp(&b.context))
        .then_with(|| a.lesson.cmp(&b.lesson))
}

fn compare_key_order(a: &ArchiveEntry, b: &ArchiveEntry) -> Ordering {
    a.context
        .cmp(&b.context)
        .then_with(|| a.lesson.cmp(&b.lesson))
        .then_with(|| compare_best(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(seed: u64, first_gene: f32) -> Genome {
        let mut genome = Genome::random(seed);
        genome.l1[0] = first_gene;
        genome
    }

    #[test]
    fn rejects_dimension_mismatch_before_retention() {
        let mut genome = candidate(1, 0.1);
        genome.l1.pop();
        let mut archive = Archive::new(2).unwrap();
        let error = archive
            .observe("track-a-starts-3", "move", &[genome], &[1.0], &[3], 3, 0)
            .unwrap_err();
        assert!(matches!(
            error,
            ArchiveError::GenomeDimension { field: "l1", .. }
        ));
        assert!(archive.is_empty());
    }

    #[test]
    fn rejects_non_finite_genome_before_retention() {
        let mut genome = candidate(2, 0.2);
        genome.l2[0] = f32::NAN;
        let mut archive = Archive::new(2).unwrap();
        let error = archive
            .observe("track-a-starts-3", "move", &[genome], &[1.0], &[3], 3, 0)
            .unwrap_err();
        assert!(matches!(
            error,
            ArchiveError::NonFiniteGenome { field: "l2", .. }
        ));
        assert!(archive.is_empty());
    }

    #[test]
    fn retention_and_reinjection_are_deterministic() {
        let genomes = vec![candidate(3, 0.3), candidate(4, 0.4), candidate(5, 0.5)];
        let scores = [4.0, 9.0, 7.0];
        let successes = [1, 1, 1];
        let mut first = Archive::new(2).unwrap();
        let mut second = Archive::new(2).unwrap();
        first
            .observe(
                "track-a-starts-1",
                "move",
                &genomes,
                &scores,
                &successes,
                1,
                8,
            )
            .unwrap();
        second
            .observe(
                "track-a-starts-1",
                "move",
                &genomes,
                &scores,
                &successes,
                1,
                8,
            )
            .unwrap();
        let first_scores: Vec<_> = first.entries().map(ArchiveEntry::score).collect();
        let second_scores: Vec<_> = second.entries().map(ArchiveEntry::score).collect();
        assert_eq!(first_scores, second_scores);
        assert_eq!(first_scores, vec![9.0, 7.0]);

        let mut offspring = vec![candidate(10, 1.0), candidate(11, 1.1), candidate(12, 1.2)];
        let elite_before = offspring[0].l1[0];
        let provenance = first
            .reinject_best(
                &mut offspring,
                1,
                SourceFilter::SameContract {
                    context: "track-a-starts-1",
                    lesson: "move",
                },
            )
            .unwrap();
        assert_eq!(provenance.len(), 2);
        assert_eq!(offspring[0].l1[0], elite_before);
        assert_eq!(offspring[1].l1[0], 0.4);
        assert_eq!(offspring[2].l1[0], 0.5);
    }

    #[test]
    fn partial_success_is_not_a_retained_skill() {
        let mut archive = Archive::new(2).unwrap();
        archive
            .observe(
                "track-a-starts-3",
                "move",
                &[candidate(13, 1.3)],
                &[999.0],
                &[2],
                3,
                0,
            )
            .unwrap();
        assert!(archive.is_empty());
    }
}
