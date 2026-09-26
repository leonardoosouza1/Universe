//! Bounded within-episode ecology integration for `gpu_evolve`.
//!
//! A dead slot is reused in place. The scheduler owns lineage and resource decisions;
//! this adapter owns the concrete replacement: hereditary genome mutation, actual fuel
//! transfer, a fresh physics state, and slot-scoped recurrent-state resets. It never
//! calls a population-wide reset while another being is alive.

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use universe::engine::SovereignEngine;
use universe::gpu::evolution::{EvoParams, Genome};
use universe::gpu::physics::PhysicsStateGpu;

use crate::context::GpuCtx;
use crate::types::BeingState;
use universe::ecology::{BirthDecision, Ecology, EcologyConfig, EcologyError, LifeSnapshot};

/// A durable birth record. The genotype path is included in the line record so an
/// observer can reconstruct exactly which hereditary payload was installed in the slot.
#[derive(Clone, Debug)]
pub struct BirthRecord {
    pub tick: u64,
    pub slot: usize,
    pub parent_slot: usize,
    pub parent_lineage_id: u64,
    pub child_lineage_id: u64,
    pub genotype_path: PathBuf,
    pub child_energy: f64,
    pub parent_energy_after: f64,
    pub reproduction_cost: f64,
    pub gpu_receipt: Option<universe::gpu::organism_reproduction::BirthReceipt>,
}

/// Concrete newborn state supplied by `main.rs`, including the expressed tank capacity.
#[derive(Clone, Copy, Debug)]
pub struct SpawnState {
    pub physics: PhysicsStateGpu,
    pub fuel_capacity: f64,
    pub life_ticks: f64,
    pub wp_near: usize,
}

/// Observer seam for lineage and genotype provenance. I/O failures abort the run.
/// Organism birth records follow accepted GPU receipts; a failed write must not be retried as a birth.
pub trait BirthObserver {
    fn record_birth(&mut self, record: &mut BirthRecord, genome: &Genome) -> io::Result<()>;
    fn record_death(&mut self, event: &universe::ecology::EcologyEvent) -> io::Result<()>;
}

/// JSONL observer plus one `.bin` genotype file per birth.
pub struct GenotypeBirthObserver {
    events: BufWriter<std::fs::File>,
    genotype_dir: PathBuf,
}

impl GenotypeBirthObserver {
    /// Continue an existing lineage journal without erasing prior accepted births.
    pub fn resume(
        events_path: impl AsRef<Path>,
        genotype_dir: impl AsRef<Path>,
    ) -> io::Result<Self> {
        Ok(Self {
            events: BufWriter::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(events_path)?,
            ),
            genotype_dir: genotype_dir.as_ref().to_path_buf(),
        })
    }

    pub fn create(
        events_path: impl AsRef<Path>,
        genotype_dir: impl AsRef<Path>,
    ) -> io::Result<Self> {
        Ok(Self {
            events: BufWriter::new(std::fs::File::create(events_path)?),
            genotype_dir: genotype_dir.as_ref().to_path_buf(),
        })
    }
}

impl BirthObserver for GenotypeBirthObserver {
    fn record_death(&mut self, event: &universe::ecology::EcologyEvent) -> io::Result<()> {
        serde_json::to_writer(
            &mut self.events,
            &serde_json::json!({
                "schema_version":1,"event":"death","tick":event.tick,
                "slot":event.slot,"lineage_id":event.child_lineage_id,"residual_fuel":event.energy,
            }),
        )?;
        self.events.write_all(b"\n")?;
        self.events.flush()
    }

    fn record_birth(&mut self, record: &mut BirthRecord, genome: &Genome) -> io::Result<()> {
        std::fs::create_dir_all(&self.genotype_dir)?;
        let name = format!(
            "birth_tick{:012}_slot{:04}_lineage{:020}.bin",
            record.tick, record.slot, record.child_lineage_id
        );
        let path = self.genotype_dir.join(name);
        let path_text = path.to_string_lossy().into_owned();
        record.genotype_path = path.clone();
        universe::gpu::genome_io::save_genome(genome, &path_text)?;
        let line = serde_json::json!({
            "schema_version": 1,"event":"birth",
            "tick": record.tick,
            "slot": record.slot,
            "parent_slot": record.parent_slot,
            "parent_lineage_id": record.parent_lineage_id,
            "child_lineage_id": record.child_lineage_id,
            "genotype_path": path_text,
            "child_energy":record.child_energy,"parent_energy_after":record.parent_energy_after,
            "parent_energy_before":record.parent_energy_after+record.child_energy+record.reproduction_cost,
            "reproduction_cost":record.reproduction_cost,
            "gpu_receipt":record.gpu_receipt,
        });
        serde_json::to_writer(&mut self.events, &line)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        self.events.write_all(b"\n")?;
        self.events.flush()
    }
}

#[derive(Debug)]
pub enum EcologyRuntimeError {
    InvalidShape(&'static str),
    Scheduler(EcologyError),
    Observer(io::Error),
    InvalidFuel { slot: usize },
    UnsupportedState(&'static str),
}

impl From<EcologyError> for EcologyRuntimeError {
    fn from(error: EcologyError) -> Self {
        Self::Scheduler(error)
    }
}

impl From<io::Error> for EcologyRuntimeError {
    fn from(error: io::Error) -> Self {
        Self::Observer(error)
    }
}

/// Per-episode lineage and age state. The vectors are fixed at construction and are
/// never resized by a birth, which keeps slot identity and GPU strides stable.
pub struct EcologyRuntime {
    scheduler: Ecology,
    lineage_ids: Vec<u64>,
    age_ticks: Vec<u64>,
    observer: Option<Box<dyn BirthObserver>>,
    last_world_tick: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EcologyCheckpoint {
    pub scheduler: Ecology,
    pub lineage_ids: Vec<u64>,
    pub age_ticks: Vec<u64>,
    pub last_world_tick: u64,
}

impl EcologyRuntime {
    pub fn new(config: EcologyConfig, population: usize) -> Result<Self, EcologyRuntimeError> {
        if population == 0 || population != config.capacity {
            return Err(EcologyRuntimeError::InvalidShape(
                "population must equal positive ecology capacity",
            ));
        }
        let scheduler = Ecology::try_new(config)?;
        Ok(Self {
            scheduler,
            lineage_ids: (1..=population as u64).collect(),
            age_ticks: vec![0; population],
            observer: None,
            last_world_tick: 0,
        })
    }

    pub fn with_observer(mut self, observer: Box<dyn BirthObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    pub fn scheduler(&self) -> &Ecology {
        &self.scheduler
    }
    pub fn lineage_ids(&self) -> &[u64] {
        &self.lineage_ids
    }
    pub fn age_ticks(&self) -> &[u64] {
        &self.age_ticks
    }

    pub fn checkpoint(&self) -> EcologyCheckpoint {
        EcologyCheckpoint {
            scheduler: self.scheduler.clone(),
            lineage_ids: self.lineage_ids.clone(),
            age_ticks: self.age_ticks.clone(),
            last_world_tick: self.last_world_tick,
        }
    }
    pub fn restore_checkpoint(
        &mut self,
        saved: EcologyCheckpoint,
    ) -> Result<(), EcologyRuntimeError> {
        saved.scheduler.validate_checkpoint()?;
        let n = self.lineage_ids.len();
        if saved.scheduler.config() != self.scheduler.config()
            || saved.lineage_ids.len() != n
            || saved.age_ticks.len() != n
            || saved.lineage_ids.iter().any(|id| *id == 0)
            || saved
                .lineage_ids
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != n
            || (saved.last_world_tick > 0
                && saved.scheduler.current_tick() != saved.last_world_tick.saturating_add(1))
        {
            return Err(EcologyRuntimeError::InvalidShape(
                "ecology checkpoint population",
            ));
        }
        self.scheduler = saved.scheduler;
        self.lineage_ids = saved.lineage_ids;
        self.age_ticks = saved.age_ticks;
        self.last_world_tick = saved.last_world_tick;
        Ok(())
    }

    /// Call at absolute world-tick multiples of 64, after physics. GPU receipts authorize lineage commits.
    pub fn step_organism<F>(
        &mut self,
        ctx: &mut GpuCtx,
        states: &mut [BeingState],
        genomes: &mut [Genome],
        engines: &mut [SovereignEngine],
        progress: &[f64],
        evo_params: &EvoParams,
        evolving_body: bool,
        lifetime_mutation_flags: u32,
        world_tick: u64,
        mut spawn_state: F,
    ) -> Result<Vec<BirthDecision>, EcologyRuntimeError>
    where
        F: FnMut(usize, usize, &Genome) -> SpawnState,
    {
        if world_tick % 64 != 0 || world_tick <= self.last_world_tick {
            return Err(EcologyRuntimeError::UnsupportedState(
                "ecology boundary must advance at multiples of 64",
            ));
        }
        let n = states.len();
        if n != self.lineage_ids.len()
            || genomes.len() != n
            || engines.len() != n
            || progress.len() != n
        {
            return Err(EcologyRuntimeError::InvalidShape(
                "organism ecology population",
            ));
        }
        ctx.sample_organism(states)
            .map_err(EcologyRuntimeError::UnsupportedState)?;
        let elapsed = world_tick - self.last_world_tick;
        let mut snapshots = Vec::with_capacity(n);
        let mut eligible = Vec::with_capacity(n);
        for (slot, body) in ctx.organism_samples.iter().take(n).enumerate() {
            let v = &body.values;
            if v[29] == 1.0 {
                self.age_ticks[slot] = self.age_ticks[slot].saturating_add(elapsed);
            }
            snapshots.push(LifeSnapshot {
                slot,
                lineage_id: self.lineage_ids[slot],
                alive: v[29] == 1.0,
                age_ticks: self.age_ticks[slot],
                energy: v[0] as f64,
                progress: progress[slot],
            });
            // Conservative eligibility: maximum child water cap12.5kg *25% +1kg parent reserve.
            // The shader rechecks/debits the exact mutated child capacities.
            eligible.push(v[4] >= 0.5 && v[1] >= 4.125 && v[2] >= 25.0);
        }
        let (proposed, decisions) = self
            .scheduler
            .propose_at(world_tick, &snapshots, &eligible)?;
        self.last_world_tick = world_tick;
        if let Some(decision) = decisions.first() {
            let slot = decision.slot;
            let parent = decision.parent_slot;
            let seed = mutation_seed(world_tick, slot, decision.child_lineage_id);
            let mut child = mutate_genome(
                &genomes[parent],
                seed,
                evo_params,
                evolving_body,
                lifetime_mutation_flags,
            );
            universe::gpu::body::mutate_organism_population(
                std::slice::from_mut(&mut child),
                0,
                world_tick as usize,
                seed,
            );
            let template = universe::gpu::organism::birth_state(
                &child.organism,
                universe::gpu::organism::CONFIG_DISABLE_OLD_THERMAL,
            )
            .map_err(EcologyRuntimeError::UnsupportedState)?;
            let mut spawned = spawn_state(slot, parent, &child);
            // Initial resident setup is not rerun after a birth. Preserve its
            // numerical/contact profile while expressing the child's own body.
            universe::gpu::physics::inherit_resident_profile(
                &states[parent].phys,
                &mut spawned.physics,
            );
            if spawned.physics.validate_contract().is_err()
                || ctx.organism_action.is_none()
                || ctx.organism_sensors.is_none()
            {
                return Err(EcologyRuntimeError::UnsupportedState(
                    "invalid resident birth configuration",
                ));
            }
            let gpu = universe::gpu::organism_reproduction::ReproductionGpu::new(&ctx.device);
            let body = ctx
                .organism
                .as_ref()
                .ok_or(EcologyRuntimeError::UnsupportedState(
                    "missing organism GPU",
                ))?;
            let receipt = gpu
                .transact(
                    &ctx.device,
                    &ctx.queue,
                    &body.buf_state,
                    n,
                    parent,
                    slot,
                    &template,
                    decision.child_energy as f32,
                    decision.reproduction_cost as f32,
                    self.scheduler.config().parent_energy_reserve as f32,
                )
                .map_err(EcologyRuntimeError::UnsupportedState)?;
            if !receipt.accepted() {
                // Observe deaths/time without applying speculative births or cooldowns.
                self.scheduler = self
                    .scheduler
                    .propose_at(world_tick, &snapshots, &vec![false; n])?
                    .0;
                self.flush_deaths()?;
                return Ok(Vec::new());
            }
            ctx.queue.write_buffer(
                &body.buf_genes,
                (slot * 64) as u64,
                bytemuck::cast_slice(&child.organism),
            );
            self.scheduler = proposed;
            genomes[slot] = child;
            genomes[slot].apply_to_engine(&mut engines[slot]);
            states[slot].phys = spawned.physics;
            reset_cpu_slot(&mut states[slot]);
            states[slot].life_left = spawned.life_ticks;
            states[slot].wp_near = spawned.wp_near;
            reset_gpu_slot(ctx, slot);
            ctx.brain.upload_lifetime_slot(slot, genomes[slot].lifetime);
            self.lineage_ids[slot] = decision.child_lineage_id;
            self.age_ticks[slot] = 0;
            sync_hereditary_banks(ctx, genomes, engines);
            // Hereditary uploads do not clear survivor recurrent/controller state.
            ctx.phys_gpu
                .upload_resident_slot(slot, &states[slot].phys)
                .map_err(EcologyRuntimeError::UnsupportedState)?;
            let action = ctx.organism_action.as_ref().unwrap();
            action
                .reset_action_slot(&ctx.queue, slot, states[slot].phys.current_gear)
                .map_err(EcologyRuntimeError::UnsupportedState)?;
            action
                .upload_heads(&ctx.queue, &ctx.per_being_heads)
                .map_err(EcologyRuntimeError::UnsupportedState)?;
            ctx.organism_sensors
                .as_ref()
                .unwrap()
                .upload_horizon_slot(&ctx.queue, slot, ctx.sensor_curve_ranges[slot])
                .map_err(EcologyRuntimeError::UnsupportedState)?;

            ctx.sample_organism(states)
                .map_err(EcologyRuntimeError::UnsupportedState)?;
            let mut record = BirthRecord {
                tick: world_tick,
                slot,
                parent_slot: parent,
                parent_lineage_id: decision.parent_lineage_id,
                child_lineage_id: decision.child_lineage_id,
                genotype_path: PathBuf::new(),
                child_energy: receipt.values[7] as f64,
                parent_energy_after: receipt.values[4] as f64,
                reproduction_cost: receipt.values[11] as f64,
                gpu_receipt: Some(receipt),
            };
            if let Some(observer) = self.observer.as_mut() {
                observer.record_birth(&mut record, &genomes[slot])?;
            }
        } else {
            self.scheduler = proposed;
        }
        self.flush_deaths()?;
        Ok(decisions)
    }
    fn flush_deaths(&mut self) -> Result<(), EcologyRuntimeError> {
        for event in self.scheduler.drain_events() {
            if event.kind == universe::ecology::EventKind::Death {
                if let Some(observer) = self.observer.as_mut() {
                    observer.record_death(&event)?;
                }
            }
        }
        Ok(())
    }

    /// Observe one post-physics tick and apply all bounded birth decisions.
    ///
    /// `spawn_state` is supplied by the caller because starting formation and body/organ
    /// expression are owned by `main.rs`. It must return a zero-velocity newborn state for
    /// the target slot, including the genome-expressed fuel capacity.
    pub fn step<F>(
        &mut self,
        ctx: &mut GpuCtx,
        states: &mut [BeingState],
        genomes: &mut [Genome],
        engines: &mut [SovereignEngine],
        progress: &[f64],
        evo_params: &EvoParams,
        evolving_body: bool,
        lifetime_mutation_flags: u32,
        mut spawn_state: F,
    ) -> Result<Vec<BirthDecision>, EcologyRuntimeError>
    where
        F: FnMut(usize, &Genome) -> SpawnState,
    {
        let n = states.len();
        if n == 0
            || n != genomes.len()
            || n != engines.len()
            || n != progress.len()
            || n != self.lineage_ids.len()
            || n > ctx.capacity
        {
            return Err(EcologyRuntimeError::InvalidShape(
                "state, genome, engine, progress, and runtime capacities must match",
            ));
        }

        let next_age: Vec<u64> = states
            .iter()
            .zip(self.age_ticks.iter())
            .map(|(state, age)| {
                if state.life_left > 0.0 {
                    age.saturating_add(1)
                } else {
                    *age
                }
            })
            .collect();
        let snapshots: Vec<LifeSnapshot> = states
            .iter()
            .enumerate()
            .map(|(slot, state)| LifeSnapshot {
                slot,
                lineage_id: self.lineage_ids[slot],
                alive: state.life_left > 0.0,
                age_ticks: next_age[slot],
                energy: if state.fuel.is_finite() {
                    state.fuel.max(0.0)
                } else {
                    state.fuel
                },
                progress: progress[slot],
            })
            .collect();
        let decisions = self.scheduler.tick(&snapshots)?;
        self.age_ticks = next_age;
        for event in self.scheduler.drain_events() {
            if event.kind == universe::ecology::EventKind::Death {
                if let Some(observer) = self.observer.as_mut() {
                    observer.record_death(&event)?;
                }
            }
        }

        if decisions.is_empty() {
            return Ok(decisions);
        }

        for decision in &decisions {
            let slot = decision.slot;
            if slot >= n || !decision.child_energy.is_finite() || decision.child_energy < 0.0 {
                return Err(EcologyRuntimeError::InvalidFuel { slot });
            }
            let parent = decision.parent_slot;
            if parent >= n
                || !states[parent].fuel.is_finite()
                || states[parent].fuel < decision.reproduction_cost + decision.child_energy
            {
                return Err(EcologyRuntimeError::InvalidFuel { slot });
            }

            let seed = mutation_seed(
                self.scheduler.current_tick(),
                slot,
                decision.child_lineage_id,
            );
            let child = mutate_genome(
                &genomes[parent],
                seed,
                evo_params,
                evolving_body,
                lifetime_mutation_flags,
            );
            let spawned = spawn_state(slot, &child);
            if !spawned.fuel_capacity.is_finite() || spawned.fuel_capacity < decision.child_energy {
                return Err(EcologyRuntimeError::InvalidFuel { slot });
            }
            let mut genotype_record = BirthRecord {
                tick: decision.tick,
                slot,
                parent_slot: parent,
                parent_lineage_id: decision.parent_lineage_id,
                child_lineage_id: decision.child_lineage_id,
                genotype_path: PathBuf::new(),
                child_energy: decision.child_energy,
                parent_energy_after: decision.parent_energy_after,
                reproduction_cost: decision.reproduction_cost,
                gpu_receipt: None,
            };
            if let Some(observer) = self.observer.as_mut() {
                observer.record_birth(&mut genotype_record, &child)?;
            }

            // Resource accounting uses the actual parent fuel observed this tick. The
            // scheduler's reserve is a guard, while reproduction_cost is the dissipated
            // amount and child_energy is the transferred amount.
            states[parent].fuel -= decision.reproduction_cost + decision.child_energy;
            genomes[slot] = child;
            genomes[slot].apply_to_engine(&mut engines[slot]);
            states[slot].phys = spawned.physics;
            states[slot].fuel_capacity = spawned.fuel_capacity;
            states[slot].fuel = decision.child_energy;
            reset_cpu_slot(&mut states[slot]);
            states[slot].life_left = spawned.life_ticks;
            states[slot].wp_near = spawned.wp_near;
            reset_gpu_slot(ctx, slot);
            ctx.brain.upload_lifetime_slot(slot, genomes[slot].lifetime);
            self.lineage_ids[slot] = decision.child_lineage_id;
            self.age_ticks[slot] = 0;
        }

        // Upload complete hereditary banks after all slot replacements. This rewrites
        // weights/genes for every slot but never touches resident recurrent state; the
        // per-slot reset above is the only state reset performed here.
        sync_hereditary_banks(ctx, genomes, engines);
        Ok(decisions)
    }
}

fn reset_cpu_slot(state: &mut BeingState) {
    state.prev_steer = 0.0;
    state.stagnation_ticks = 0;
    state.wall_ticks = 0;
    state.collision_energy_j = 0.0;
    state.impact_count = 0;
    state.hole_hits = 0;
    state.hole_cooldown_ticks = 0;
    state.joy_tick = 0.0;
    state.joy_streak = 0.0;
    state.learning_reward = 0.0;
    state.edge_row6 = 0.0;
    state.edge_row7 = 0.0;
    state.center_prog = 0.0;
    state.sensor_steer = 0.0;
    state.sensor_thr = 0.0;
    state.sensor_brk = 0.0;
    state.phantom_d_min = 30.0;
    state.dist_wall_l = 40.0;
    state.dist_wall_r = 40.0;
    state.dist_wall_f = 50.0;
    state.wp_near = 0;
    state.ticks_alive = 0;
    state.reverse_timer = 0.0;
    state.post_reverse_lockout = 0.0;
    state.shift_cooldown = 0.0;
    state.shift_up_prev = 0.0;
    state.shift_down_prev = 0.0;
    state.nm_reserva = [1.0; 4];
    state.re_falhas = 0;
    state.re_desde_saida = 999.0;
    state.tire_temp = universe::tire_thermal::DEFAULT_AMBIENT;
    state.dano = 0.0;
}

fn reset_gpu_slot(ctx: &mut GpuCtx, slot: usize) {
    if let Some(traffic) = &ctx.traffic {
        traffic.reset_slot(&ctx.queue, slot);
    }
    ctx.clear_physical_retina_slot(slot);
    ctx.retina_digest.reset_slot(&ctx.queue, slot);
    ctx.percept
        .reset_temporal_slot(&ctx.device, &ctx.queue, slot);
    ctx.brain.reset_cfc(slot);
    // Taxi consumes resident intents even on skipped neural ticks. The newborn
    // must not execute the previous occupant's service/pickup/drop requests.
    ctx.queue.write_buffer(
        ctx.brain.buf_intent(),
        (slot * 96 * 4) as u64,
        bytemuck::cast_slice(&[0.0f32; 96]),
    );
    ctx.slow.reset_being(slot);
    if let Some(memory) = ctx.hip_mem.get_mut(slot) {
        memory.reset();
    }
    if let Some(fly) = ctx.fly.as_ref() {
        fly.reset_being(slot);
    }
    // `GpuCtx::fly_spiking` is intentionally private and its delayed bit ring needs a
    // dedicated slot-clear kernel. The parent wiring must reject ecology while spiking
    // is active (or add `FlySpikingGpu::reset_being` in the owning context); silently
    // calling its population reset here would erase living beings.
    if let Some(intent) = ctx.last_intents.get_mut(slot) {
        *intent = [0.0; 96];
    }
    if let Some(value) = ctx.lap_grass.get_mut(slot) {
        *value = false;
    }
    if let Some(value) = ctx.lap_damage.get_mut(slot) {
        *value = 0.0;
    }
    if let Some(value) = ctx.lap_dry.get_mut(slot) {
        *value = false;
    }
    if let Some(value) = ctx.lifetime_reward_sum.get_mut(slot) {
        *value = 0.0;
    }
}

fn sync_hereditary_banks(ctx: &mut GpuCtx, genomes: &[Genome], engines: &[SovereignEngine]) {
    ctx.upload_engines(engines);
    ctx.per_being_heads = genomes.iter().map(|g| g.kan_knots).collect();
    ctx.brain
        .upload_tau(&Genome::pack_taus(genomes), genomes.len());
    ctx.upload_sensors(&Genome::pack_sensors(genomes), genomes.len());
    ctx.brain
        .upload_neuron_meta(&Genome::pack_neuron_meta(genomes), genomes.len());
    if !ctx.gates_w.is_empty() {
        ctx.gates_w = genomes
            .iter()
            .flat_map(|g| g.gates_w.iter().copied())
            .collect();
    }
    if !ctx.hip_genes.is_empty() {
        ctx.hip_genes = genomes.iter().flat_map(|g| g.hip.iter().copied()).collect();
    }
}

fn mutation_seed(tick: u64, slot: usize, lineage: u64) -> u64 {
    tick.wrapping_mul(0x9e3779b97f4a7c15) ^ (slot as u64).wrapping_mul(0xbf58476d1ce4e5b9) ^ lineage
}

/// Reuse the canonical operator so frozen mechanisms and directed wiring masks
/// retain the same meaning in generational evolution and ecological births.
fn mutate_genome(
    parent: &Genome,
    seed: u64,
    params: &EvoParams,
    body: bool,
    life_flags: u32,
) -> Genome {
    let mut params = params.clone();
    params.n_elite = 0;
    let mut children =
        universe::gpu::evolution::evolve(&[parent.clone()], &[1.0], &params, 0, seed);
    if body {
        universe::gpu::body::mutate_population(&mut children, 0, 0, seed);
    }
    let mut child = children.pop().expect("one ecological child");
    child.lifetime = child.lifetime.mutated(seed ^ 0x4c494645, life_flags);
    child
}
