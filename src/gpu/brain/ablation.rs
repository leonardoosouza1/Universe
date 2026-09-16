//! Counterfactual neuron-output masks for deterministic GPU replays.

use crate::config::iara::brain::{L1_NEURONS, L2_NEURONS};
use std::fmt;

const CFC_NEURONS: usize = L1_NEURONS;
const CFC_WORD_OFFSET: usize = 0;
const L1_WORD_OFFSET: usize = 3;
const L2_WORD_OFFSET: usize = 6;

/// Packed `u32` words reserved for one being's CfC, L1, and L2 masks.
pub const ABLATION_WORDS_PER_BEING: usize = 8;

/// A neural stage whose emitted value can be removed from a replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AblationStage {
    CfC,
    L1,
    L2,
}

impl AblationStage {
    fn parse(value: &str) -> Result<Self, AblationError> {
        match value {
            "cfc" => Ok(Self::CfC),
            "l1" => Ok(Self::L1),
            "l2" => Ok(Self::L2),
            _ => Err(AblationError(format!(
                "unknown ablation stage {value:?}; expected cfc, l1, or l2"
            ))),
        }
    }

    fn width(self) -> usize {
        match self {
            Self::CfC => CFC_NEURONS,
            Self::L1 => L1_NEURONS,
            Self::L2 => L2_NEURONS,
        }
    }

    fn word_offset(self) -> usize {
        match self {
            Self::CfC => CFC_WORD_OFFSET,
            Self::L1 => L1_WORD_OFFSET,
            Self::L2 => L2_WORD_OFFSET,
        }
    }
}

/// One output intervention in a counterfactual replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AblationTarget {
    pub stage: AblationStage,
    pub index: usize,
}

/// Validated counterfactual intervention applied to one population slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AblationPlan {
    being: usize,
    from_tick: u32,
    targets: Vec<AblationTarget>,
}

impl AblationPlan {
    /// Parses comma-separated targets such as `cfc:4,l1:17,l2:2`.
    pub fn parse(
        spec: &str,
        being: usize,
        from_tick: u32,
        population: usize,
    ) -> Result<Self, AblationError> {
        if being >= population {
            return Err(AblationError(format!(
                "ablation being {being} is outside population 0..{population}"
            )));
        }
        if spec.trim().is_empty() {
            return Err(AblationError("ablation target list is empty".into()));
        }

        let mut targets = Vec::new();
        for raw in spec.split(',') {
            let (stage, index) = raw.trim().split_once(':').ok_or_else(|| {
                AblationError(format!(
                    "invalid ablation target {raw:?}; expected stage:index"
                ))
            })?;
            let stage = AblationStage::parse(stage)?;
            let index = index.parse::<usize>().map_err(|_| {
                AblationError(format!("invalid neuron index {index:?} in target {raw:?}"))
            })?;
            if index >= stage.width() {
                return Err(AblationError(format!(
                    "{} index {index} is outside 0..{}",
                    stage,
                    stage.width()
                )));
            }
            let target = AblationTarget { stage, index };
            if !targets.contains(&target) {
                targets.push(target);
            }
        }

        Ok(Self {
            being,
            from_tick,
            targets,
        })
    }

    pub fn being(&self) -> usize {
        self.being
    }

    pub fn from_tick(&self) -> u32 {
        self.from_tick
    }

    pub fn targets(&self) -> &[AblationTarget] {
        &self.targets
    }

    pub fn active_at(&self, world_tick: u32) -> bool {
        world_tick >= self.from_tick
    }

    /// Builds the packed mask uploaded once before the replay begins.
    pub fn mask_words(&self, population: usize) -> Vec<u32> {
        let mut words = vec![0; population * ABLATION_WORDS_PER_BEING];
        let base = self.being * ABLATION_WORDS_PER_BEING;
        for target in &self.targets {
            let word = base + target.stage.word_offset() + target.index / 32;
            words[word] |= 1u32 << (target.index % 32);
        }
        words
    }

    /// Returns a stable CLI representation for manifests and replay metadata.
    pub fn canonical_spec(&self) -> String {
        self.targets
            .iter()
            .map(|target| format!("{}:{}", target.stage, target.index))
            .collect::<Vec<_>>()
            .join(",")
    }
}

impl fmt::Display for AblationStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::CfC => "cfc",
            Self::L1 => "l1",
            Self::L2 => "l2",
        })
    }
}

/// Invalid counterfactual-mask configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AblationError(String);

impl fmt::Display for AblationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AblationError {}
