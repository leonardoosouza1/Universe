//! Deterministic finite training frontier with explicit holdout protection.
//!
//! Scores are completed-lap (or success) fractions.  This module does not
//! compute fitness and never samples from a global random generator.

use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurriculumConfig {
    pub mastery_threshold: f64,
    pub mastery_streak: usize,
    pub regression_threshold: f64,
    /// Every Nth episode is an archive probe when an archive exists. Zero disables probes.
    pub archive_probe_period: usize,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            mastery_threshold: 1.0,
            mastery_streak: 2,
            regression_threshold: 0.8,
            archive_probe_period: 5,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EpisodeKind {
    Frontier,
    ArchiveProbe,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CurriculumEventKind {
    Promotion,
    Regression,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurriculumEvent {
    pub episode: u64,
    pub seed: u64,
    pub kind: CurriculumEventKind,
    pub score: f64,
    pub frontier_index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CurriculumError {
    EmptyTrainingSet,
    DuplicateTrainingSeed(u64),
    DuplicateHoldoutSeed(u64),
    TrainingHoldoutOverlap(u64),
    InvalidConfig,
    NoActiveEpisode,
    InvalidScore,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActiveKind {
    Frontier,
    ArchiveProbe,
}

/// Finite curriculum state. Training order is preserved exactly as provided.
#[derive(Debug, PartialEq)]
pub struct AdaptiveCurriculum {
    training: Vec<u64>,
    holdout: HashSet<u64>,
    config: CurriculumConfig,
    frontier_index: usize,
    mastered: Vec<u64>,
    mastery_streak: usize,
    episode: u64,
    archive_probe_index: usize,
    active: Option<(u64, ActiveKind)>,
}

impl AdaptiveCurriculum {
    pub fn new<I, H>(
        training: I,
        holdout: H,
        config: CurriculumConfig,
    ) -> Result<Self, CurriculumError>
    where
        I: IntoIterator<Item = u64>,
        H: IntoIterator<Item = u64>,
    {
        let training: Vec<u64> = training.into_iter().collect();
        if training.is_empty() {
            return Err(CurriculumError::EmptyTrainingSet);
        }
        let mut seen = HashSet::new();
        for &seed in &training {
            if !seen.insert(seed) {
                return Err(CurriculumError::DuplicateTrainingSeed(seed));
            }
        }
        let holdout_vec: Vec<u64> = holdout.into_iter().collect();
        let mut holdout_set = HashSet::new();
        for seed in holdout_vec {
            if !holdout_set.insert(seed) {
                return Err(CurriculumError::DuplicateHoldoutSeed(seed));
            }
            if seen.contains(&seed) {
                return Err(CurriculumError::TrainingHoldoutOverlap(seed));
            }
        }
        if !config.mastery_threshold.is_finite()
            || !config.regression_threshold.is_finite()
            || !(0.0..=1.0).contains(&config.mastery_threshold)
            || !(0.0..=1.0).contains(&config.regression_threshold)
            || config.mastery_streak == 0
        {
            return Err(CurriculumError::InvalidConfig);
        }
        Ok(Self {
            training,
            holdout: holdout_set,
            config,
            frontier_index: 0,
            mastered: Vec::new(),
            mastery_streak: 0,
            episode: 0,
            archive_probe_index: 0,
            active: None,
        })
    }

    /// Returns the seed for the current episode, or starts the next one.
    pub fn seed_for_episode(&mut self) -> Option<u64> {
        if let Some((seed, _)) = self.active {
            return Some(seed);
        }
        let (seed, kind) = if self.should_probe() {
            let seed = self.mastered[self.archive_probe_index % self.mastered.len()];
            self.archive_probe_index = self.archive_probe_index.wrapping_add(1);
            (seed, ActiveKind::ArchiveProbe)
        } else {
            let seed = *self.training.get(self.frontier_index)?;
            (seed, ActiveKind::Frontier)
        };
        debug_assert!(!self.holdout.contains(&seed));
        self.active = Some((seed, kind));
        Some(seed)
    }

    /// Record a normalized success fraction and update the frontier.
    pub fn observe(&mut self, score: f64) -> Result<Option<CurriculumEvent>, CurriculumError> {
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Err(CurriculumError::InvalidScore);
        }
        let (seed, kind) = self.active.take().ok_or(CurriculumError::NoActiveEpisode)?;
        let episode = self.episode;
        self.episode = self.episode.saturating_add(1);
        match kind {
            ActiveKind::ArchiveProbe => {
                if score < self.config.regression_threshold {
                    self.frontier_index = self
                        .training
                        .iter()
                        .position(|s| *s == seed)
                        .expect("archive seed belongs to training");
                    self.mastered.truncate(self.frontier_index);
                    self.mastery_streak = 0;
                    return Ok(Some(CurriculumEvent {
                        episode,
                        seed,
                        kind: CurriculumEventKind::Regression,
                        score,
                        frontier_index: self.frontier_index,
                    }));
                }
            }
            ActiveKind::Frontier => {
                if score >= self.config.mastery_threshold {
                    self.mastery_streak += 1;
                    if self.mastery_streak >= self.config.mastery_streak {
                        self.mastery_streak = 0;
                        if !self.mastered.contains(&seed) {
                            self.mastered.push(seed);
                        }
                        self.frontier_index = (self.frontier_index + 1).min(self.training.len());
                        return Ok(Some(CurriculumEvent {
                            episode,
                            seed,
                            kind: CurriculumEventKind::Promotion,
                            score,
                            frontier_index: self.frontier_index,
                        }));
                    }
                } else {
                    self.mastery_streak = 0;
                }
            }
        }
        Ok(None)
    }

    fn should_probe(&self) -> bool {
        !self.mastered.is_empty()
            && (self.is_complete()
                || (self.config.archive_probe_period != 0
                    && self.episode != 0
                    && self.episode % self.config.archive_probe_period as u64 == 0))
    }

    pub fn frontier_index(&self) -> usize {
        self.frontier_index
    }
    pub fn mastered(&self) -> &[u64] {
        &self.mastered
    }
    pub fn is_complete(&self) -> bool {
        self.frontier_index == self.training.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CurriculumConfig {
        CurriculumConfig {
            mastery_threshold: 0.9,
            mastery_streak: 2,
            regression_threshold: 0.7,
            archive_probe_period: 3,
        }
    }

    #[test]
    fn rejects_duplicate_and_holdout_overlap() {
        assert_eq!(
            AdaptiveCurriculum::new([1, 1], [], cfg()),
            Err(CurriculumError::DuplicateTrainingSeed(1))
        );
        assert_eq!(
            AdaptiveCurriculum::new([1], [1], cfg()),
            Err(CurriculumError::TrainingHoldoutOverlap(1))
        );
        assert_eq!(
            AdaptiveCurriculum::new([1], [2, 2], cfg()),
            Err(CurriculumError::DuplicateHoldoutSeed(2))
        );
    }

    #[test]
    fn mastery_advances_in_order_and_never_enters_holdout() {
        let mut c = AdaptiveCurriculum::new([11, 12], [99], cfg()).unwrap();
        assert_eq!(c.seed_for_episode(), Some(11));
        assert_eq!(c.observe(1.0).unwrap(), None);
        assert_eq!(c.seed_for_episode(), Some(11));
        assert_eq!(
            c.observe(1.0).unwrap().unwrap().kind,
            CurriculumEventKind::Promotion
        );
        assert_eq!(c.seed_for_episode(), Some(12));
        let seed = c.seed_for_episode().unwrap();
        assert!(!c.holdout.contains(&seed));
    }

    #[test]
    fn archive_failure_regresses_and_freezes_progress() {
        let mut c = AdaptiveCurriculum::new(
            [1, 2],
            [],
            CurriculumConfig {
                archive_probe_period: 2,
                ..cfg()
            },
        )
        .unwrap();
        c.seed_for_episode();
        c.observe(1.0).unwrap();
        c.seed_for_episode();
        assert!(c.observe(1.0).unwrap().is_some());
        c.seed_for_episode();
        assert_eq!(
            c.observe(0.0).unwrap().unwrap().kind,
            CurriculumEventKind::Regression
        );
        assert_eq!(c.frontier_index(), 0);
    }

    #[test]
    fn invalid_scores_and_replay_are_deterministic() {
        let mut a = AdaptiveCurriculum::new([4, 5], [], cfg()).unwrap();
        let mut b = AdaptiveCurriculum::new([4, 5], [], cfg()).unwrap();
        assert_eq!(a.seed_for_episode(), b.seed_for_episode());
        assert_eq!(a.observe(f64::NAN), Err(CurriculumError::InvalidScore));
        assert_eq!(a.observe(1.1), Err(CurriculumError::InvalidScore));
        assert_eq!(a.observe(1.0), b.observe(1.0));
    }
}
