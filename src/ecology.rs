//! Deterministic, bounded population turnover for the open-ecology slice (E6).
//!
//! This module only decides bounded births and records events. The GPU runtime adapter
//! owns genome mutation, physics reset, and the actual replacement of a slot.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Invalid runtime input is rejected before it can mutate scheduler state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EcologyError {
    InvalidConfig(&'static str),
    SnapshotCountMismatch { expected: usize, actual: usize },
    SnapshotSlotOutOfBounds { slot: usize, capacity: usize },
    DuplicateSnapshotSlot { slot: usize },
    NonFiniteSnapshot { slot: usize },
    NegativeEnergy { slot: usize },
    InvalidParentLineage { slot: usize },
    DuplicateLiveLineage { lineage_id: u64 },
}

/// The state supplied by the runtime for one fixed population slot.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LifeSnapshot {
    pub slot: usize,
    pub lineage_id: u64,
    pub alive: bool,
    pub age_ticks: u64,
    pub energy: f64,
    pub progress: f64,
}

/// Bounds and resource rules for [`Ecology`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EcologyConfig {
    pub capacity: usize,
    pub maturity_ticks: u64,
    pub reproduction_cooldown_ticks: u64,
    /// Energy removed from the parent and dissipated by reproduction.
    pub reproduction_cost: f64,
    /// Energy transferred from the parent to the newborn.
    pub child_energy: f64,
    pub parent_energy_reserve: f64,
    pub minimum_progress: f64,
    pub seed: u64,
    pub max_events: usize,
}

impl Default for EcologyConfig {
    fn default() -> Self {
        Self {
            capacity: 0,
            maturity_ticks: 0,
            reproduction_cooldown_ticks: 0,
            reproduction_cost: 0.0,
            child_energy: 0.0,
            parent_energy_reserve: 0.0,
            minimum_progress: 0.0,
            seed: 0,
            max_events: 256,
        }
    }
}

/// A replacement the caller should apply to `slot` after resetting its body.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BirthDecision {
    pub tick: u64,
    pub slot: usize,
    pub parent_slot: usize,
    pub parent_lineage_id: u64,
    pub child_lineage_id: u64,
    pub child_energy: f64,
    pub parent_energy_after: f64,
    pub reproduction_cost: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EventKind {
    Birth,
    Death,
}

/// Bounded audit event. `energy` is transferred energy for births and the
/// parent's observed energy at death; no fitness bonus is recorded or applied.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EcologyEvent {
    pub tick: u64,
    pub kind: EventKind,
    pub slot: usize,
    pub parent_lineage_id: Option<u64>,
    pub child_lineage_id: Option<u64>,
    pub energy: f64,
}

/// Stateful scheduler; all randomness is a local hash of `(seed, tick, slot)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ecology {
    config: EcologyConfig,
    tick: u64,
    next_lineage_id: u64,
    last_alive: Vec<bool>,
    last_lineage: Vec<Option<u64>>,
    cooldown_until: HashMap<u64, u64>,
    events: Vec<EcologyEvent>,
    extinct: bool,
}

impl Ecology {
    pub fn new(config: EcologyConfig) -> Self {
        Self::try_new(config).expect("invalid ecology configuration")
    }

    pub fn try_new(config: EcologyConfig) -> Result<Self, EcologyError> {
        validate_config(config)?;
        Ok(Self {
            last_alive: vec![false; config.capacity],
            last_lineage: vec![None; config.capacity],
            next_lineage_id: 1,
            cooldown_until: HashMap::new(),
            events: Vec::new(),
            extinct: false,
            config,
            tick: 0,
        })
    }

    pub fn config(&self) -> EcologyConfig {
        self.config
    }
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Observe one tick and return deterministic birth decisions. Dead slots
    /// are eligible for replacement only when a currently live parent can pay
    /// both the dissipated cost and the child's transferred energy.
    /// Validate and advance one bounded scheduler tick.
    ///
    /// Validation runs before any death event, cooldown, lineage counter, or extinction
    /// state is changed. Callers therefore get an atomic rejection for malformed input.
    pub fn tick(&mut self, snapshots: &[LifeSnapshot]) -> Result<Vec<BirthDecision>, EcologyError> {
        self.tick_filtered(snapshots, &vec![true; self.config.capacity], usize::MAX)
    }

    /// Speculate on one bounded GPU transaction; install this clone only on acceptance.
    pub fn propose_at(
        &self,
        tick: u64,
        snapshots: &[LifeSnapshot],
        eligible: &[bool],
    ) -> Result<(Self, Vec<BirthDecision>), EcologyError> {
        if tick < self.tick || eligible.len() != self.config.capacity {
            return Err(EcologyError::InvalidConfig(
                "nonmonotonic tick or eligibility shape",
            ));
        }
        let mut proposed = self.clone();
        proposed.tick = tick;
        let births = proposed.tick_filtered(snapshots, eligible, 1)?;
        Ok((proposed, births))
    }

    pub fn validate_checkpoint(&self) -> Result<(), EcologyError> {
        validate_config(self.config)?;
        if self.last_alive.len() != self.config.capacity
            || self.last_lineage.len() != self.config.capacity
            || self.events.len() > self.config.max_events
            || self.next_lineage_id == 0
        {
            return Err(EcologyError::InvalidConfig("invalid scheduler checkpoint"));
        }
        Ok(())
    }

    fn tick_filtered(
        &mut self,
        snapshots: &[LifeSnapshot],
        allowed: &[bool],
        limit: usize,
    ) -> Result<Vec<BirthDecision>, EcologyError> {
        validate_snapshots(snapshots, self.config.capacity)?;
        let mut states = snapshots.to_vec();
        states.sort_by_key(|s| s.slot);
        let mut max_lineage = self.next_lineage_id;
        for state in &states {
            max_lineage = max_lineage.max(state.lineage_id.saturating_add(1));
            if state.slot >= self.config.capacity {
                continue;
            }
            if self.last_alive[state.slot] && !state.alive {
                self.push_event(EcologyEvent {
                    tick: self.tick,
                    kind: EventKind::Death,
                    slot: state.slot,
                    parent_lineage_id: None,
                    child_lineage_id: self.last_lineage[state.slot],
                    energy: state.energy.max(0.0),
                });
            }
            self.last_alive[state.slot] = state.alive;
            self.last_lineage[state.slot] = state.alive.then_some(state.lineage_id);
        }
        self.next_lineage_id = max_lineage;

        if !states.iter().any(|state| state.alive) {
            self.extinct = true;
        }
        if self.extinct {
            self.tick = self.tick.saturating_add(1);
            return Ok(Vec::new());
        }

        let mut available_energy: HashMap<u64, f64> = states
            .iter()
            .filter(|s| s.alive)
            .map(|s| (s.lineage_id, s.energy.max(0.0)))
            .collect();
        self.cooldown_until
            .retain(|lineage, _| available_energy.contains_key(lineage));
        let mut births = Vec::new();
        let targets: Vec<LifeSnapshot> = states
            .iter()
            .copied()
            .filter(|s| !s.alive && s.slot < self.config.capacity)
            .collect();
        for target in targets {
            if births.len() == limit {
                break;
            }
            let required = self.config.reproduction_cost
                + self.config.child_energy
                + self.config.parent_energy_reserve;
            if !required.is_finite() || required < 0.0 {
                continue;
            }
            let mut eligible: Vec<&LifeSnapshot> = states
                .iter()
                .filter(|parent| {
                    parent.alive
                        && allowed[parent.slot]
                        && parent.slot != target.slot
                        && parent.age_ticks >= self.config.maturity_ticks
                        && parent.progress >= self.config.minimum_progress
                        && self
                            .cooldown_until
                            .get(&parent.lineage_id)
                            .copied()
                            .unwrap_or(0)
                            <= self.tick
                        && available_energy
                            .get(&parent.lineage_id)
                            .copied()
                            .unwrap_or(0.0)
                            >= required
                })
                .collect();
            if eligible.is_empty() {
                continue;
            }
            let index =
                (mix(self.config.seed ^ self.tick ^ target.slot as u64) as usize) % eligible.len();
            let parent = eligible.swap_remove(index);
            let parent_energy = available_energy
                .get_mut(&parent.lineage_id)
                .expect("eligible parent has energy");
            *parent_energy -= self.config.reproduction_cost + self.config.child_energy;
            self.cooldown_until.insert(
                parent.lineage_id,
                self.tick
                    .saturating_add(self.config.reproduction_cooldown_ticks),
            );
            let child = self.next_lineage_id;
            self.next_lineage_id = self.next_lineage_id.saturating_add(1).max(1);
            let decision = BirthDecision {
                tick: self.tick,
                slot: target.slot,
                parent_slot: parent.slot,
                parent_lineage_id: parent.lineage_id,
                child_lineage_id: child,
                child_energy: self.config.child_energy,
                parent_energy_after: *parent_energy,
                reproduction_cost: self.config.reproduction_cost,
            };
            self.push_event(EcologyEvent {
                tick: self.tick,
                kind: EventKind::Birth,
                slot: target.slot,
                parent_lineage_id: Some(parent.lineage_id),
                child_lineage_id: Some(child),
                energy: self.config.child_energy,
            });
            births.push(decision);
        }
        self.tick = self.tick.saturating_add(1);
        Ok(births)
    }

    pub fn drain_events(&mut self) -> Vec<EcologyEvent> {
        std::mem::take(&mut self.events)
    }
    pub fn events(&self) -> &[EcologyEvent] {
        &self.events
    }

    fn push_event(&mut self, event: EcologyEvent) {
        if self.config.max_events == 0 {
            return;
        }
        if self.events.len() == self.config.max_events {
            self.events.remove(0);
        }
        self.events.push(event);
    }
}

fn validate_config(config: EcologyConfig) -> Result<(), EcologyError> {
    if config.capacity == 0 {
        return Err(EcologyError::InvalidConfig("capacity must be positive"));
    }
    if !config.reproduction_cost.is_finite()
        || !config.child_energy.is_finite()
        || !config.parent_energy_reserve.is_finite()
        || !config.minimum_progress.is_finite()
        || config.reproduction_cost < 0.0
        || config.child_energy < 0.0
        || config.parent_energy_reserve < 0.0
        || config.minimum_progress < 0.0
    {
        return Err(EcologyError::InvalidConfig(
            "resource and progress bounds must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_snapshots(snapshots: &[LifeSnapshot], capacity: usize) -> Result<(), EcologyError> {
    if snapshots.len() != capacity {
        return Err(EcologyError::SnapshotCountMismatch {
            expected: capacity,
            actual: snapshots.len(),
        });
    }
    let mut slots = std::collections::HashSet::with_capacity(snapshots.len());
    let mut live_lineages = std::collections::HashSet::new();
    for state in snapshots {
        if state.slot >= capacity {
            return Err(EcologyError::SnapshotSlotOutOfBounds {
                slot: state.slot,
                capacity,
            });
        }
        if !slots.insert(state.slot) {
            return Err(EcologyError::DuplicateSnapshotSlot { slot: state.slot });
        }
        if !state.energy.is_finite() || !state.progress.is_finite() {
            return Err(EcologyError::NonFiniteSnapshot { slot: state.slot });
        }
        if state.energy < 0.0 {
            return Err(EcologyError::NegativeEnergy { slot: state.slot });
        }
        // A live state is a possible parent and must carry a real lineage identity.
        // Dead target slots may retain an old identity (for death accounting), but zero
        // is never accepted as a parent identity.
        if state.alive && state.lineage_id == 0 {
            return Err(EcologyError::InvalidParentLineage { slot: state.slot });
        }
        if state.alive && !live_lineages.insert(state.lineage_id) {
            return Err(EcologyError::DuplicateLiveLineage {
                lineage_id: state.lineage_id,
            });
        }
    }
    Ok(())
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> EcologyConfig {
        EcologyConfig {
            capacity: 2,
            maturity_ticks: 2,
            reproduction_cooldown_ticks: 3,
            reproduction_cost: 1.0,
            child_energy: 2.0,
            parent_energy_reserve: 1.0,
            minimum_progress: 1.0,
            seed: 7,
            max_events: 16,
        }
    }

    fn parent() -> LifeSnapshot {
        LifeSnapshot {
            slot: 0,
            lineage_id: 10,
            alive: true,
            age_ticks: 2,
            energy: 10.0,
            progress: 1.0,
        }
    }
    fn dead() -> LifeSnapshot {
        LifeSnapshot {
            slot: 1,
            lineage_id: 20,
            alive: false,
            age_ticks: 0,
            energy: 0.0,
            progress: 0.0,
        }
    }

    #[test]
    fn maturity_and_conservation_are_explicit() {
        let mut eco = Ecology::new(config());
        assert_eq!(
            eco.tick(&[
                LifeSnapshot {
                    age_ticks: 1,
                    ..parent()
                },
                dead()
            ])
            .unwrap(),
            Vec::new()
        );
        let births = eco.tick(&[parent(), dead()]).unwrap();
        assert_eq!(births.len(), 1);
        assert_eq!(births[0].parent_energy_after, 7.0);
    }

    #[test]
    fn cooldown_capacity_and_lineage_are_bounded() {
        let mut eco = Ecology::new(EcologyConfig {
            capacity: 3,
            reproduction_cooldown_ticks: 2,
            ..config()
        });
        let p = parent();
        let dead_a = dead();
        assert_eq!(
            eco.tick(&[p, dead_a, LifeSnapshot { slot: 2, ..dead_a }])
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            eco.tick(&[p, dead_a, LifeSnapshot { slot: 2, ..dead_a }])
                .unwrap()
                .len(),
            0
        );
        assert_eq!(
            eco.tick(&[p, dead_a, LifeSnapshot { slot: 2, ..dead_a }])
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn extinction_has_no_automatic_resurrection_and_seed_replays() {
        let mut a = Ecology::new(config());
        let mut b = Ecology::new(config());
        let extinct = [LifeSnapshot { slot: 0, ..dead() }, dead()];
        assert!(a.tick(&extinct).unwrap().is_empty());
        assert!(a.tick(&[parent(), dead()]).unwrap().is_empty());
        let mut c = Ecology::new(config());
        assert_eq!(
            b.tick(&[parent(), dead()]).unwrap(),
            c.tick(&[parent(), dead()]).unwrap()
        );
        assert_eq!(b.drain_events(), c.drain_events());
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    #[test]
    fn absolute_boundary_proposals_and_checkpoint_are_transactional() {
        let config = EcologyConfig {
            capacity: 2,
            maturity_ticks: 64,
            reproduction_cooldown_ticks: 128,
            reproduction_cost: 10.0,
            child_energy: 20.0,
            parent_energy_reserve: 5.0,
            ..Default::default()
        };
        let scheduler = Ecology::new(config);
        let snapshots = [
            LifeSnapshot {
                slot: 0,
                lineage_id: 1,
                alive: true,
                age_ticks: 64,
                energy: 100.0,
                progress: 0.0,
            },
            LifeSnapshot {
                slot: 1,
                lineage_id: 2,
                alive: false,
                age_ticks: 0,
                energy: 0.0,
                progress: 0.0,
            },
        ];
        let (accepted, births) = scheduler
            .propose_at(64, &snapshots, &[true, false])
            .unwrap();
        assert_eq!(births.len(), 1);
        assert_eq!(births[0].tick, 64);
        assert_eq!(scheduler.current_tick(), 0);
        let encoded = serde_json::to_vec(&accepted).unwrap();
        let restored: Ecology = serde_json::from_slice(&encoded).unwrap();
        restored.validate_checkpoint().unwrap();
        assert!(restored
            .propose_at(128, &snapshots, &[true, false])
            .unwrap()
            .1
            .is_empty());
        assert_eq!(
            restored
                .propose_at(192, &snapshots, &[true, false])
                .unwrap()
                .1
                .len(),
            1
        );
        assert!(restored.propose_at(64, &snapshots, &[true, false]).is_err());
    }
}
