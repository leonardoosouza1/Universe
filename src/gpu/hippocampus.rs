//! Sector-indexed episodic memory — the "hippocampus" of C1
//! (`docs/PRD-CEREBRO-NEUROMOD.md` §C1).
//!
//! # What it is
//!
//! One small memory slot per circuit sector (24 of them, `N_SECTORS`). While a being
//! traverses a sector the host accumulates what happened there; when the being crosses
//! into the next sector, the accumulated summary is written into that sector's slot with
//! an EMA. Every tick the being READS the slot of the sector it is in plus the slot of the
//! NEXT sector, and a 3×9 weight matrix turns those 9 numbers into a bias on
//! throttle / steer / brake.
//!
//! # Why it exists
//!
//! `docs/ESTUDO-CEREBRO-NEUROMOD.md` §3: above ~35 m/s no perception probe reaches the
//! corner in time, so on lap 2 the being holds information the sensors cannot give it —
//! what happened in THIS sector and in the NEXT one on the previous lap. That is
//! information outside the observable MDP, so the "memory is useless when the state is
//! observable" theorem does not apply here. Memory of weights failed five times in this
//! ecosystem; memory written on one lap and read on the next has never been tested.
//!
//! # Why this module is pure
//!
//! No GPU, no I/O, no globals — the same discipline as `src/predator.rs`: the mechanism
//! is written and tested on its own, and stays "NOT wired" until `context.rs` calls it.
//! The CLI flags (`--hipocampo-setor`, `--hipocampo-apagar`, `--sem-hipocampo`) live in  // check-language: allow -- quoting the flag names verbatim
//! the binary (`src/bin/gpu_evolve/statics.rs`) and reach this module as plain booleans,
//! so every behaviour they select is reachable from a test with no process-global state.
//!
//! # The identity contract
//!
//! All 29 genes are born at 0.0 and [`bias`] then returns `[0.0, 0.0, 0.0]` bit for bit,
//! for ANY memory content: every row of `W` is zero, so the dot product is `+0.0`,
//! `tanh(+0.0) == +0.0`, the gain `tanh(genes[28]) == +0.0`, and `+0.0 * +0.0 == +0.0`.
//! That is what lets the new arm reproduce the old brain exactly on generation 0
//! (the birth-neutral guarantee stated in the P1.7 block of `evolve()`, `gpu::evolution`;
//! the PRD cites it as `evolution.rs:975-982`, which is that comment as of 11/09/2026).

use crate::math::activations::sigmoid;

/// Circuit sectors. Mirrors `N_CHECKPOINTS` in `src/bin/gpu_evolve/statics.rs` (24) and
/// `N_SETORES` in `fantasma.rs` — the library cannot import a binary's constant, so the
/// three have to agree. `sector_count_matches_the_checkpoint_grid` in
/// `src/bin/gpu_evolve/tests.rs` is the assertion that keeps them in sync.
pub const N_SECTORS: usize = 24;

/// Numbers stored per sector: `[speed_mean/40, grass_frac, damage_norm, dt_improvement]`.
pub const MEM_PER_SECTOR: usize = 4;

/// Length of one read: this sector's slot, the next sector's slot, and a constant 1.0.
///
/// The trailing 1.0 is a bias column, not data: it is what lets the 3×9 matrix express a
/// RESTING command offset that does not depend on any memory at all. It is also why
/// `--hipocampo-apagar` is an ablation of the MEMORY and not of the organ — see
/// [`SectorMemory::read_gated`].
pub const HIP_READ_LEN: usize = 2 * MEM_PER_SECTOR + 1;

/// Outputs: throttle, steer, brake — in that order, the same order as `kan_eval`.
pub const HIP_OUTPUTS: usize = 3;

/// Genes: 27 for `W` (3×9) + 1 for the write rate + 1 for the read gain.
pub const N_HIP_GENES: usize = HIP_OUTPUTS * HIP_READ_LEN + 2;

/// Index of the gene that sets the EMA write rate, through `sigmoid`.
/// Gene 0.0 ⇒ α = 0.5 exactly (`sigmoid(0) = 1/(1+1)`), the PRD's documented default.
pub const GENE_WRITE_RATE: usize = HIP_OUTPUTS * HIP_READ_LEN;

/// Index of the gene that sets the read gain, through `tanh`. Gene 0.0 ⇒ gain 0.0 exactly.
pub const GENE_READ_GAIN: usize = GENE_WRITE_RATE + 1;

/// Authority of the hippocampus over each command, applied by the caller in
/// `context.rs`: `throttle += HIP_AUTHORITY * bias[0]`, and likewise for steer and brake.
///
/// 0.5 and not 1.0 because `bias` is already saturated by `tanh` into [-1, 1] and the
/// commands themselves live in [0, 1] / [-1, 1]: at authority 1.0 a single saturated gene
/// row could pin a command at its clamp for a whole lap, and the hippocampus would stop
/// being a bias and become the driver. Same magnitude family as the slow brain's own
/// authority over the same three commands (0.5 aggression / 0.8 cap / 0.20 steer,
/// `context.rs`), which is the contract this block is sitting next to.
pub const HIP_AUTHORITY: f32 = 0.5;

/// The read a being gets with `--hipocampo-apagar`: memory zeroed, bias column intact.
pub const ERASED_READ: [f32; HIP_READ_LEN] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

/// Speed used to normalise `speed_mean` into roughly [0, 1], in m/s.
///
/// 40 m/s is above the measured population peak (21.9 m/s locked on the rev limiter with
/// the gearbox frozen, 30.5 m/s at gear ceiling 3) and above the ~35 m/s at which no
/// perception probe reaches the corner in time — the regime this mechanism exists for. A
/// being that goes faster than this saturates the channel rather than clipping it, because
/// the value is NOT clamped: losing the ordering above 40 m/s would erase exactly the
/// difference the experiment is looking for.
pub const SPEED_NORM_MPS: f32 = 40.0;

/// Per-being hippocampus: the 24 slots plus the accumulator for the sector being
/// traversed right now.
///
/// Reset at birth (once per generation), like the recurrent brain state: memory that
/// survives a generation boundary would make the champion's re-evaluation
/// non-deterministic, which is the same reason `main.rs` rebuilds the hostile world every
/// generation.
#[derive(Clone, Debug)]
pub struct SectorMemory {
    /// `mem[s]` = what happened in sector `s` the last time the being went through it.
    mem: [[f32; MEM_PER_SECTOR]; N_SECTORS],
    /// Sector whose traversal is currently being accumulated (checkpoint-ordered).
    open_sector: usize,
    /// Sector used for the READ — geometric, `(track_progress / lap_dist) * 24`.
    ///
    /// Deliberately NOT the same as `open_sector`. The write follows the checkpoint
    /// ratchet (`main.rs`'s `checkpoint[i]`, which only advances on the NEXT sector in
    /// sequence), because that is what defines a traversal; the read follows where the
    /// car physically is, because that is what "what happened here last lap" means. They
    /// differ exactly when the being drives backwards or skips a sector, which is the
    /// case where using the ratchet for the read would hand it the wrong sector's memory.
    read_sector: usize,
    /// Sum of speed over the ticks accumulated so far, m/s.
    acc_speed: f32,
    /// Ticks of those spent with the being on grass.
    acc_grass: u32,
    /// Damage level at the moment the current sector was opened — the subtraction at
    /// close is what makes the stored number "damage taken IN this sector".
    damage_at_open: f32,
    /// Latest damage level seen while accumulating.
    damage_now: f32,
    /// Ticks accumulated for the open sector.
    acc_ticks: u32,
    /// Ticks the PREVIOUS traversal of each sector took. 0 = never traversed, which is
    /// what makes `dt_improvement` 0 on the first passage instead of a division by zero.
    prev_ticks: [u32; N_SECTORS],
}

impl Default for SectorMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl SectorMemory {
    /// A newborn hippocampus: every slot zero, no sector ever traversed.
    pub fn new() -> Self {
        Self {
            mem: [[0.0; MEM_PER_SECTOR]; N_SECTORS],
            open_sector: 0,
            read_sector: 0,
            acc_speed: 0.0,
            acc_grass: 0,
            damage_at_open: 0.0,
            damage_now: 0.0,
            acc_ticks: 0,
            prev_ticks: [0; N_SECTORS],
        }
    }

    /// Raw slot access, for tests and for the `hip_gen<N>.f32` observation rows.
    pub fn slot(&self, sector: usize) -> [f32; MEM_PER_SECTOR] {
        self.mem[sector % N_SECTORS]
    }

    /// Sector the read uses this tick.
    pub fn read_sector(&self) -> usize {
        self.read_sector
    }

    /// Sets the sector the read uses. Called once per tick by the host with the geometric
    /// sector; see the field's doc for why it is not the checkpoint-ordered one.
    pub fn set_read_sector(&mut self, sector: u32) {
        self.read_sector = sector as usize % N_SECTORS;
    }

    /// EMA write: `mem[s] += α·(x − mem[s])`.
    ///
    /// α = 1 overwrites, α = 0 does nothing, α = 0.5 (the birth value, `sigmoid(0)`)
    /// halves the distance to the new observation on every passage. The EMA is what makes
    /// the memory a running summary of the sector rather than the last lap alone, which
    /// matters because a single lap through a sector can be an accident (a spin, a
    /// collision) and the being should not rewrite its whole idea of the corner from it.
    pub fn write(&mut self, sector: usize, x: &[f32; MEM_PER_SECTOR], alpha: f32) {
        let slot = &mut self.mem[sector % N_SECTORS];
        for k in 0..MEM_PER_SECTOR {
            slot[k] += alpha * (x[k] - slot[k]);
        }
    }

    /// The 9 numbers the being reads: this sector, the next sector, and the bias column.
    pub fn read(&self, sector: usize) -> [f32; HIP_READ_LEN] {
        let here = self.mem[sector % N_SECTORS];
        let next = self.mem[(sector + 1) % N_SECTORS];
        let mut out = [0.0; HIP_READ_LEN];
        out[..MEM_PER_SECTOR].copy_from_slice(&here);
        out[MEM_PER_SECTOR..2 * MEM_PER_SECTOR].copy_from_slice(&next);
        out[HIP_READ_LEN - 1] = 1.0;
        out
    }

    /// The read under `--hipocampo-apagar` (`erased == true`): [`ERASED_READ`].
    ///
    /// **This is NOT a full ablation, and the difference is the whole point.** With the
    /// memory zeroed the bias column still carries 1.0, so the being keeps the RESTING
    /// bias `gain · tanh(W[:, 8])` it evolved — what is removed is only the episodic part.
    /// That is the ablation the experiment wants: it answers "does the CONTENT of the
    /// memory pay?", not "does having the organ pay?".
    ///
    /// To remove the organ instead, use `--sem-hipocampo`, which forces `bias` to  // check-language: allow -- quoting the flag name verbatim
    /// `[0, 0, 0]` and freezes the genes — the control arm carrying the same genome
    /// layout. The two readings are different questions and the PRD asks for both.
    pub fn read_gated(&self, sector: usize, erased: bool) -> [f32; HIP_READ_LEN] {
        if erased {
            ERASED_READ
        } else {
            self.read(sector)
        }
    }

    /// One tick of evidence about the sector currently being traversed.
    ///
    /// `damage` is the being's cumulative hidden damage (`BeingState::dano`, already
    /// normalised to [0, 1]); what gets stored is the DIFFERENCE across the sector, so the
    /// slot says "this corner hurt" and not "I am an old car".
    pub fn accumulate(&mut self, speed_mps: f32, on_grass: bool, damage: f32) {
        if self.acc_ticks == 0 {
            self.damage_at_open = damage;
        }
        self.acc_speed += speed_mps;
        self.acc_grass += u32::from(on_grass);
        self.damage_now = damage;
        self.acc_ticks += 1;
    }

    /// Closes the open sector and opens `entered_sector`. Returns what was written, for
    /// telemetry; `None` when there was nothing to write (no tick was accumulated).
    ///
    /// Called at the sector crossing in `main.rs`. The sector WRITTEN is the one just
    /// left, never the one just entered: the accumulated evidence describes the stretch
    /// the car has driven, and `read` expects `mem[s]` to be "what happened while I was in
    /// s".
    pub fn cross(
        &mut self,
        entered_sector: u32,
        alpha: f32,
    ) -> Option<(usize, [f32; MEM_PER_SECTOR])> {
        let closed = self.open_sector;
        let written = if self.acc_ticks > 0 {
            let t_now = self.acc_ticks;
            let t_prev = self.prev_ticks[closed];
            // `dt_improvement`: positive = this passage was FASTER than the previous one.
            // 0 on the first passage — there is no previous time, and inventing one
            // (assuming the being was slower, say) would hand the mechanism a signal the
            // being never earned.
            let dt_improvement = if t_prev == 0 {
                0.0
            } else {
                ((t_prev as f32 - t_now as f32) / t_prev as f32).clamp(-1.0, 1.0)
            };
            let ticks = t_now as f32;
            let x = [
                (self.acc_speed / ticks) / SPEED_NORM_MPS,
                self.acc_grass as f32 / ticks,
                (self.damage_now - self.damage_at_open).clamp(0.0, 1.0),
                dt_improvement,
            ];
            self.write(closed, &x, alpha);
            self.prev_ticks[closed] = t_now;
            Some((closed, x))
        } else {
            None
        };
        self.open_sector = entered_sector as usize % N_SECTORS;
        self.acc_speed = 0.0;
        self.acc_grass = 0;
        self.acc_ticks = 0;
        self.damage_at_open = self.damage_now;
        written
    }
}

/// EMA write rate from the genes: `sigmoid(genes[27])`. Gene 0.0 ⇒ exactly 0.5.
#[inline]
pub fn write_rate(genes: &[f32]) -> f32 {
    sigmoid(genes[GENE_WRITE_RATE])
}

/// Read gain from the genes: `tanh(genes[28])`. Gene 0.0 ⇒ exactly 0.0, which is half of
/// why an all-zero genome produces an all-zero bias (the other half is `W` itself).
#[inline]
pub fn read_gain(genes: &[f32]) -> f32 {
    genes[GENE_READ_GAIN].tanh()
}

/// The command bias: `gain · tanh(W · x)`, `W` = `genes[0..27]` as 3 rows of 9.
///
/// `genes` must have at least [`N_HIP_GENES`] entries; `x` is a [`SectorMemory::read`].
/// Output order is throttle, steer, brake.
///
/// The double saturation is deliberate. The inner `tanh` bounds the bias no matter how
/// large the weights grow, so a runaway gene cannot pin a command; the outer `gain`, also
/// a `tanh`, is the single knob selection can use to turn the whole organ down to zero
/// without having to drive 27 weights back to zero one mutation at a time. That is the
/// shape `neuromod-lab/completo.py:246-248` uses for the same job, and the reason the PRD
/// cites it: an organ that cannot be switched off by one gene cannot be DISCARDED by
/// selection, only tolerated.
pub fn bias(genes: &[f32], x: &[f32; HIP_READ_LEN]) -> [f32; HIP_OUTPUTS] {
    debug_assert!(
        genes.len() >= N_HIP_GENES,
        "hippocampus genes: got {}, need {N_HIP_GENES}",
        genes.len()
    );
    let gain = read_gain(genes);
    let mut out = [0.0f32; HIP_OUTPUTS];
    for (o, slot) in out.iter_mut().enumerate() {
        let row = &genes[o * HIP_READ_LEN..(o + 1) * HIP_READ_LEN];
        let mut sum = 0.0f32;
        for k in 0..HIP_READ_LEN {
            sum += row[k] * x[k];
        }
        *slot = gain * sum.tanh();
    }
    out
}

/// Columns of one `hip_gen<N>.f32` row: `[tick, sector, mem[s](4), mem[s+1](4), bias(3)]`.
///
/// **Discrepancy with `docs/PRD-CEREBRO-NEUROMOD.md` §C1, recorded rather than
/// silently resolved:** the PRD lists this exact layout and then calls it "14 f32".
/// 1 + 1 + 4 + 4 + 3 = 13. The layout is the contract (a reader has to know which column
/// is which); the count is arithmetic, and it is wrong. 13 columns are written. Adding a
/// fourteenth column to match the number would mean inventing a field the PRD never
/// names, which is worse than contradicting it.
pub const HIP_OBS_COLS: usize = 2 + 2 * MEM_PER_SECTOR + HIP_OUTPUTS;

#[cfg(test)]
mod tests {
    use super::*;

    /// The two genes that gate the organ have to produce their documented defaults
    /// EXACTLY, not approximately — the birth-neutral contract is a bit-for-bit claim.
    #[test]
    fn zero_genes_give_alpha_one_half_and_gain_zero() {
        let genes = [0.0f32; N_HIP_GENES];
        assert_eq!(write_rate(&genes), 0.5, "sigmoid(0) must be exactly 0.5");
        assert_eq!(
            read_gain(&genes).to_bits(),
            0,
            "tanh(0) must be exactly +0.0"
        );
    }

    /// The constants have to agree with each other; a stale `N_HIP_GENES` would slice
    /// `W` wrong and only show up as a mysteriously weak organ.
    #[test]
    fn gene_layout_is_self_consistent() {
        assert_eq!(HIP_READ_LEN, 9);
        assert_eq!(N_HIP_GENES, 29);
        assert_eq!(GENE_WRITE_RATE, 27);
        assert_eq!(GENE_READ_GAIN, 28);
        assert_eq!(HIP_OBS_COLS, 13);
    }
}
