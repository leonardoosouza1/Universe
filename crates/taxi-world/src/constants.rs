//! Fixed dimensions and the `CfgCampo` behaviour constants (SPEC §0, §3.2).
//!
//! SPEC = `/home/leonardo/projects/LLM/neuromod-lab/docs/71-spec-mundo-taxi.md`.
//! Everything here is a literal transcription of a SPEC table; a constant that
//! differs from the SPEC value is a bug, not a tuning choice.

/// Movement options: 8 headings + "stay" (SPEC §0).
pub const G: usize = 9;
/// District grid resolution, `GRID × GRID` cells (SPEC §0 `GR`, §2.1 `campo_dist`).
/// Independent of the world's side length `L` — always 128 in v0.
pub const GRID: usize = 128;
/// Terrain-cost texture resolution, `TEX × TEX` cells (SPEC §2.1 `custo`, §3.2 `tex`).
/// Independent of `L` — always 64 in v0. A DIFFERENT grid than `GRID` on purpose
/// (SPEC §2.3, armadilha 1): do not unify the two.
pub const TEX: usize = 64;
/// Service types: posto (fuel), comida (food), agua (water), descanso (rest),
/// oficina (repair) — this fixed order (SPEC §0 `TIPOS`) is load-bearing everywhere
/// a `[_; N_TYPES]` array is indexed by type.
pub const N_TYPES: usize = 5;
/// Service actions (SPEC §0 `ACOES`, §6): 5 fuel quantities + 2 water quantities +
/// food + rest + repair.
pub const N_SERVICES: usize = 10;
/// Passenger slots per being (SPEC §3.2 `NPX`). Declared here — not in a T2 file —
/// because it is a fixed cross-task dimension exactly like `G`/`GRID`/`TEX`, and T1
/// is built before `beings.rs` exists.
pub const NPX: usize = 12;

/// Width of the sense vector `F` per movement option: `NOMES_OP (10) + NOMES_CORPO
/// (19)` with `NM_RISCO=1, NM_PONTO=1` (SPEC §0). **This is the crate's single
/// source of truth for `NF`**, used by `genome.rs` to validate a loaded genome's
/// header. The plan's own T3 interface block (`senses.rs`) redeclares `pub const NF
/// = N_OP + N_BODY`; both must reduce to 29 — T3 should reuse `taxi_world::NF`
/// rather than keep a second definition (this crate exists before `senses.rs` does,
/// so T1 has to own the number somewhere).
pub const NF: usize = 29;
/// Service-note weight count — same value as `N_SERVICES`, kept as a separate name
/// because the SPEC calls it `S` (SPEC §0) and the genome layout (§4.3) refers to it
/// by that name.
pub const S: usize = N_SERVICES;
/// Genome length: `NF` movement/body weights + `S` service weights + 1 bias
/// (SPEC §0 `NP`, §4.3 `NOMES_F`).
pub const NP: usize = NF + S + 1;

/// The 9 movement directions, normalized, index 8 = "stay" (SPEC §0). The diagonal
/// components are `FRAC_1_SQRT_2` **computed in f32**, not a truncated decimal
/// literal — SPEC §0 is explicit that this must be
/// `std::f32::consts::FRAC_1_SQRT_2`, because the reference trajectory was produced
/// with that exact rounding.
pub const DIRS: [[f32; 2]; G] = {
    let d = std::f32::consts::FRAC_1_SQRT_2;
    [
        [1.0, 0.0],
        [d, d],
        [0.0, 1.0],
        [-d, d],
        [-1.0, 0.0],
        [-d, -d],
        [0.0, -1.0],
        [d, -d],
        [0.0, 0.0],
    ]
};

/// `CfgCampo` behaviour constants (SPEC §3.2, `src/campo.py:36-63`).
///
/// `Default` mirrors `CfgCampo`'s OWN defaults, i.e. what `campo.py` would build
/// with no arguments — **not** the reference trace's overrides. Per
/// `traco_referencia.py:12`, the trace generator only overrides `respawn` (→ 1e9),
/// `ponto` (→ 0.0, already the default), `grava_acoes` (→ true) and `horizonte`
/// (→ 'fixo'); every other field here is the plain default. A caller that needs the
/// trace-exact `respawn`/`ponto` (T5's identity gate, SPEC §11.2) reads
/// `Trace::cfg` directly rather than expecting `Config::default()` to already carry
/// it — keeping the two separate means a stale trace can never make the defaults
/// silently drift.
///
/// Every field is `f32`, including `shop_at_station` (Python `loja_no_posto`,
/// naturally a bool): this crate does no GPU work in T1, so the choice is not about
/// buffer layout — it is about NOT having two different scalar types for what is,
/// in every formula in the SPEC, a 0.0/1.0 multiplier (`campo.py:364`,
/// `s.PERM_F`-style tables). One type also means one deserialization path if a
/// later task ever wants to build a `Config` from `Trace::cfg` field-by-field.
///
/// Left out on purpose: `fitness` (`'vida'`, a string selector) and `w_trab` belong
/// to the fitness formula in T6's `judge.rs` (SPEC §8), not to the physics constants
/// T1 owns; `dia`/`vel`/`vel_noite`/`raio`/`raio_px`/`dinh0` etc. below ARE included
/// because T2's `physics.rs` and T3/T4 need them and nothing else in T1..T6 owns
/// this table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Ticks per simulated day (`dia`, `campo.py:38`).
    pub day_ticks: f32,
    /// Daytime speed multiplier (`vel`, `campo.py:38`).
    pub day_speed: f32,
    /// Night-time speed multiplier (`vel_noite`, `campo.py:38`).
    pub night_speed: f32,
    /// Radius for using a service or boarding/dropping a passenger, metres
    /// (`raio`, `campo.py:39`).
    pub service_radius_m: f32,
    /// Radius of the ring passengers are born in around a being, metres
    /// (`raio_px`, `campo.py:39`). Not used by v0's physics (respawn is off,
    /// SPEC §7.3), kept for completeness of the SPEC §3.2 table.
    pub passenger_spawn_radius_m: f32,
    /// Starting debt, currency units (`dinh0`, `campo.py:39,158`).
    pub initial_debt: f32,
    /// Whether the food/water shop is co-located with the fuel station
    /// (`loja_no_posto`, `campo.py:39`) — 1.0 = true, the only value used in v0.
    pub shop_at_station: f32,
    /// Fuel burned per tick per unit of (cost × weight) while driving
    /// (`fuel_passo`, `campo.py:40`).
    pub fuel_per_tick: f32,
    /// Water lost per tick while alive (`agua_passo`, `campo.py:40`).
    pub water_per_tick: f32,
    /// Extra water lost per tick while alive AND overheated (`agua_calor`,
    /// `campo.py:40`).
    pub water_heat_per_tick: f32,
    /// Hunger gained per tick, scaled by `metabolism` (`fome_passo`, `campo.py:40`).
    pub hunger_per_tick: f32,
    /// Fatigue gained per tick while moving (`fad_passo`, `campo.py:40`).
    pub fatigue_per_tick: f32,
    /// Heat gained per tick while moving, scaled by terrain cost (`calor_sobe`,
    /// `campo.py:41`).
    pub heat_rise: f32,
    /// Heat lost per tick while not moving (`calor_desce`, `campo.py:41`).
    pub heat_fall: f32,
    /// Heat threshold that triggers overheating (`CALOR_MAX`, `campo.py:42`).
    pub heat_max: f32,
    /// Damage from overheating, per event (`D_QUENTE`, `campo.py:42`).
    pub damage_hot: f32,
    /// Damage from breaking down, per event, ADDED to the fuel-shortfall term
    /// (SPEC §8) (`D_PANE`, `campo.py:42`).
    pub damage_breakdown: f32,
    /// Damage per tick spent driving while hot (`D_RODAR`, `campo.py:42`).
    pub damage_driving_hot: f32,
    /// Death-rate coefficient from hunger (`fome_mata`, `campo.py:43`).
    pub hunger_death_rate: f32,
    /// Hunger value above which the death rate is non-zero (`FOME_CRIT`,
    /// `campo.py:43`).
    pub hunger_critical: f32,
    /// Death-rate coefficient from thirst (`agua_mata`, `campo.py:43`).
    pub thirst_death_rate: f32,
    /// Water value below which the death rate is non-zero (`AGUA_CRIT`,
    /// `campo.py:43`).
    pub thirst_critical: f32,
    /// Reserve fuel spent (capped) on a breakdown tow, litres-equivalent
    /// (`KCAL_PANE`, `campo.py:43`).
    pub breakdown_kcal: f32,
    /// Per-gene activation cost — always 0.0 in v0, the `Linear` brain has no
    /// degrees of freedom to charge for (`custo_gene`, `campo.py:44`, SPEC §4.3 E58).
    pub gene_cost: f32,
    /// Safety margin added to the fuel-risk sense column (`margem_risco`,
    /// `campo.py:44`).
    pub risk_margin: f32,
    /// Currency cost of a tow — 0.0 in v0 (`GUINCHO`, `campo.py:44`).
    pub tow_cost: f32,
    /// Fuel granted after a breakdown tow (`FUEL_POS_PANE`, `campo.py:44`).
    pub breakdown_fuel_refill: f32,
    /// Fog distance cap on perceived distances — `+inf` in v0 (the world is
    /// omniscient, SPEC §3.2) (`nevoa`, `campo.py:44`).
    pub fog_distance: f32,
    /// Blindness factor — inert while `fog_distance` is infinite (`k_cego`,
    /// `campo.py:44`).
    pub blind_factor: f32,
    /// Passenger-slot respawn denominator (`respawn`, `campo.py:46`). The
    /// reference trace overrides this to `1e9` (effectively off) — see this
    /// struct's own doc comment.
    pub respawn_denominator: f32,
    /// Taxi-stand mechanic toggle — 0.0 (off) in v0; the whole block is skipped
    /// when 0 (SPEC §7.5, not implemented at all in T1..T6 — it consumes RNG and
    /// would break the identity gate) (`ponto`, `campo.py:58`).
    pub taxi_stand: f32,
    /// Amplitude of the daily price sine wave (`amp_preco`, `campo.py:59`).
    pub day_price_amplitude: f32,
    /// Period of the daily price sine wave, in simulated days (`periodo_preco`,
    /// `campo.py:59`).
    pub day_price_period: f32,
    /// Fuel price per litre, before district multiplier (`preco_L`, `campo.py:60`).
    pub fuel_price_per_l: f32,
    /// Water price per litre, before district multiplier (`preco_agua`,
    /// `campo.py:60`).
    pub water_price_per_l: f32,
    /// Food price, before district multiplier (`preco_comida`, `campo.py:60`).
    pub food_price: f32,
    /// Rest-stop price away from home, before district multiplier (`HOTEL`,
    /// `campo.py:60`).
    pub rest_price: f32,
    /// Repair price, before district multiplier (`preco_oficina`, `campo.py:60`).
    pub repair_price: f32,
    /// Life restored by a repair (`REPARO`, `campo.py:60`).
    pub repair_heal: f32,
    /// Fixed component of the fare (`tarifa_base`, `campo.py:61`).
    pub fare_base: f32,
    /// Distance component of the fare, per world unit of `‖dest - emb‖`
    /// (`tarifa_km`, `campo.py:61`).
    pub fare_per_distance: f32,
    /// Rent charged once per simulated day (`aluguel`, `campo.py:61`).
    pub rent: f32,
    /// Debt floor; also the denominator of the debt-damage term (SPEC §8)
    /// (`LIMITE_DIVIDA`, `campo.py:61`).
    pub debt_limit: f32,
    /// Fraction of overflow fuel lost when it spills into the reserve tank
    /// (`PERDA_ARMAZ`, `campo.py:61`).
    pub reserve_storage_loss_frac: f32,
    /// Reserve tank capacity (`RES_MAX`, `campo.py:61`).
    pub reserve_max: f32,
    /// Reserve fuel lost per tick, passively (`perda_res = 0.02/240`,
    /// `campo.py:61`).
    pub reserve_decay_per_tick: f32,
    /// Hunger relief from a meal (`REFEICAO`, `campo.py:62`).
    pub meal_relief: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            day_ticks: 240.0,
            day_speed: 1.0,
            night_speed: 0.7,
            service_radius_m: 6.0,
            passenger_spawn_radius_m: 45.0,
            initial_debt: 20.0,
            shop_at_station: 1.0,
            fuel_per_tick: 0.15,
            water_per_tick: 0.08,
            water_heat_per_tick: 0.12,
            hunger_per_tick: 0.32,
            fatigue_per_tick: 0.15,
            heat_rise: 0.15,
            heat_fall: 0.6,
            heat_max: 92.0,
            damage_hot: 6.0,
            damage_breakdown: 6.0,
            damage_driving_hot: 0.15,
            hunger_death_rate: 0.15,
            hunger_critical: 90.0,
            thirst_death_rate: 0.15,
            thirst_critical: 10.0,
            breakdown_kcal: 20.0,
            gene_cost: 0.0,
            risk_margin: 30.0,
            tow_cost: 0.0,
            breakdown_fuel_refill: 15.0,
            fog_distance: f32::INFINITY,
            blind_factor: 2.0,
            respawn_denominator: 20.0,
            taxi_stand: 0.0,
            day_price_amplitude: 0.5,
            day_price_period: 3.0,
            fuel_price_per_l: 0.3,
            water_price_per_l: 0.05,
            food_price: 2.0,
            rest_price: 8.0,
            repair_price: 10.0,
            repair_heal: 25.0,
            fare_base: 3.0,
            fare_per_distance: 0.25,
            rent: 4.0,
            debt_limit: 60.0,
            reserve_storage_loss_frac: 0.15,
            reserve_max: 120.0,
            reserve_decay_per_tick: 0.02 / 240.0,
            meal_relief: 90.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirs_diagonals_use_the_f32_frac_1_sqrt_2_constant() {
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert_eq!(DIRS[1], [expected, expected]);
        assert_eq!(DIRS[3], [-expected, expected]);
        assert_eq!(DIRS[5], [-expected, -expected]);
        assert_eq!(DIRS[7], [expected, -expected]);
        assert_ne!(
            expected, 0.7071,
            "must not be a truncated literal (SPEC §0)"
        );
    }

    #[test]
    fn dirs_index_eight_is_stay_and_the_cardinals_are_axis_aligned() {
        assert_eq!(DIRS[8], [0.0, 0.0]);
        assert_eq!(DIRS[0], [1.0, 0.0]);
        assert_eq!(DIRS[2], [0.0, 1.0]);
        assert_eq!(DIRS[4], [-1.0, 0.0]);
        assert_eq!(DIRS[6], [0.0, -1.0]);
    }

    #[test]
    fn np_equals_forty_as_the_genome_layout_requires() {
        assert_eq!(NF, 29);
        assert_eq!(S, 10);
        assert_eq!(NP, 40, "SPEC §0: NP = NF + S + 1 = 29 + 10 + 1 = 40");
    }

    #[test]
    fn config_default_matches_the_spec_three_two_table() {
        let cfg = Config::default();
        assert_eq!(cfg.day_ticks, 240.0);
        assert_eq!(cfg.fuel_per_tick, 0.15);
        assert_eq!(cfg.debt_limit, 60.0);
        assert!(cfg.fog_distance.is_infinite() && cfg.fog_distance > 0.0);
        assert!((cfg.reserve_decay_per_tick - 0.02 / 240.0).abs() < 1e-12);
    }
}
