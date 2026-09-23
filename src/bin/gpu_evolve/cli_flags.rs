//! Canonical CLI flag registry of `gpu_evolve` — one table, every spelling.
//!
//! The argument loop in `main.rs` is 112 `match` arms deep, and until this table
//! existed each arm carried its flag's name as a bare literal. That made a rename a
//! 112-site edit with no way to check it: a queued battery passing the old spelling
//! would fall through to the `_ => {}` arm and run with the default instead, silently
//! — the class of failure `docs/MIGRATION.md` records for `--pista-held-out`.
//!
//! So the names live here, once. Every arm in `main.rs` matches the CANONICAL
//! (English) name; [`resolve`] maps a deprecated spelling onto it before the match and
//! warns once per alias, so nothing that is already in the queue breaks. Adding a flag
//! is one row here plus one arm there, and `cli_flags_tests.rs` fails if the two drift
//! apart in either direction.
//!
//! Renaming a flag is therefore: change `name`, move the old spelling into `aliases`.
//! Nothing is ever deleted from `aliases` — a run request written months ago is still
//! a valid input.

use std::collections::BTreeSet;
use std::sync::{Mutex, OnceLock};

/// One canonical flag of `gpu_evolve`, plus every spelling it used to answer to.
#[derive(Debug)]
pub struct Flag {
    /// Canonical English name, spelled exactly as the `match` arm in `main.rs`.
    pub name: &'static str,
    /// Deprecated spellings, still accepted with a warning. Never shrinks.
    pub aliases: &'static [&'static str],
    /// `true` when the argument that follows is this flag's value.
    ///
    /// The parser in `main.rs` does its own `i += 1`; this field is the declaration
    /// that a test checks against it, not a second implementation of it. Hence the
    /// `allow`: nothing on the RUN path reads it, and that is the point — a field the
    /// parser consulted would be a behaviour change, and this pass is a rename.
    #[allow(dead_code)]
    pub takes_value: bool,
    /// One line: what the flag does. Present tense, no leading capital.
    ///
    /// Printed by `gpu_evolve --help`. Unknown arguments fail before GPU startup
    /// so misspelled experiment settings cannot silently select a different arm.
    #[allow(dead_code)]
    pub help: &'static str,
}

/// `takes_value: true`, named so the table below reads as a table.
const VALUE: bool = true;
/// `takes_value: false` — a switch that consumes no argument.
const SWITCH: bool = false;

/// Row constructor, so one flag fits on one line of [`FLAGS`].
const fn flag(
    name: &'static str,
    aliases: &'static [&'static str],
    takes_value: bool,
    help: &'static str,
) -> Flag {
    Flag {
        name,
        aliases,
        takes_value,
        help,
    }
}

/// Every flag `gpu_evolve` accepts, in the order `main.rs` matches them.
///
/// Kept one row per line (`rustfmt::skip`) because it is a table: a reader compares
/// columns down the page, which rustfmt's per-field wrapping would destroy.
#[rustfmt::skip]
pub const FLAGS: &[Flag] = &[
    flag("--driving-diagnostics", &[], SWITCH, "record contacts and selected sensory/brain snapshots"),
    flag("--slow-kinematics", &[], VALUE, "slow-brain input mapping: canonical (default) | legacy"),
    flag("--wall-probes", &[], VALUE, "CPU wall-distance sampling: continuous (default) | legacy"),
    flag("--vision-being", &[], VALUE, "being index for sensory/brain snapshots (default 0)"),
    flag("--vision-every", &[], VALUE, "snapshot cadence in ticks (default 16)"),
    flag("--ablate", &[], VALUE, "counterfactual output mask: cfc:N,l1:N,l2:N"),
    flag("--ablate-being", &[], VALUE, "population slot targeted by --ablate (default 0)"),
    flag("--ablate-from-tick", &[], VALUE, "first world tick eligible for --ablate"),
    flag("--erase-fast-memory", &[], SWITCH, "erase native sensory CfC state before each decision"),
    flag("--beings",                      &["--seres"],                    VALUE,  "beings per generation (default 60)"),
    flag("--ticks",                       &[],                             VALUE,  "ticks per generation (default 20000)"),
    flag("--gens",                        &[],                             VALUE,  "generations to run (default 100)"),
    flag("--sigma",                       &[],                             VALUE,  "mutation sigma (default 0.05)"),
    flag("--motor-p-mutate",              &[],                             VALUE,  "mutation probability for throttle/steer/brake L2 rows"),
    flag("--motor-sigma",                 &[],                             VALUE,  "mutation sigma for throttle/steer/brake L2 rows"),
    flag("--p-mutate",                    &[],                             VALUE,  "per-gene mutation probability (default 0.02)"),
    flag("--hip-p",                       &[],                             VALUE,  "per-gene mutation probability of the hippocampus block"),
    flag("--hip-sigma",                   &[],                             VALUE,  "mutation sigma of the hippocampus block"),
    flag("--no-slow",                     &["--sem-lento"],                SWITCH, "ablate the slow brain (control arm)"),  // check-language: allow
    flag("--l1-alive",                    &["--l1-vivos"],                 VALUE,  "keep only the N most alive L1 neurons"),
    flag("--resume-mix",                  &["--retomar-mix"],              VALUE,  "L1/L2 weight multiplier for resumed children; 1 preserves checkpoint weights"),
    flag("--switch-on-complete",          &["--trocar-ao-completar"],      VALUE,  "laps to complete before the track is switched"),
    flag("--curriculum-min-speed",        &["--curriculo-veloc-min"],      VALUE,  "minimum speed the curriculum demands, m/s"),
    flag("--tau-hetero",                  &[],                             SWITCH, "seed a heterogeneous tau bank"),
    flag("--tau-bank",                    &["--banco-tau"],                VALUE,  "tau bank parametrisation: free | log-fixed | spectral"),
    flag("--no-spectral-init",            &["--sem-init-espectral"],       SWITCH, "L1 is born raw, without the spectral gain (control arm)"),  // check-language: allow
    flag("--driving-reflex-birth",        &[],                             SWITCH, "experimentally seed sparse evolvable body reflexes at birth"),
    flag("--no-driving-reflex-birth",     &[],                             SWITCH, "use the default fully random motor pathways at birth"),
    flag("--evolving-sensors",            &["--sensores-evolutivos"],      SWITCH, "let FOV, ranges and lookaheads mutate"),
    flag("--snapshot-every", &[], VALUE, "save full organism runtime every N ticks; zero disables"),
    flag("--resume-state", &[], VALUE, "resume an organism runtime checkpoint directory"),
    flag("--scenario", &[], VALUE, "race-v1 or organism-taxi-v1; explicit body/world interface version"),
    flag("--ecology",                     &[], SWITCH, "bounded in-episode births paid from parent fuel; native brain"),
    flag("--ecology-founders",            &[], VALUE, "initial living beings; other slots are vacant"),
    flag("--ecology-maturity-ticks",      &[], VALUE, "minimum parent age and cooldown (default 640 ticks)"),
    flag("--ecology-min-progress",        &[], VALUE, "minimum signed track progress for reproduction (default 20 metres)"),
    flag("--plasticity",                  &[], SWITCH, "resident bounded motor plasticity; cleared at birth"),
    flag("--evolve-learning",             &[], SWITCH, "mutate inherited learning-rule parameters; enables plasticity"),
    flag("--development",                 &[], SWITCH, "age/energy-dependent neural development"),
    flag("--evolving-topology",            &[], SWITCH, "evolve bounded neuron/connection capacity masks"),
    flag("--adaptive-curriculum",          &[], SWITCH, "training-only mastery frontier with archive regression probes"),
    flag("--evolving-body",               &[],                             SWITCH, "evolve inherited engine/brake capacities with physical costs; requires --fuel"),
    flag("--fixed-body",                  &[],                             SWITCH, "express the historical car even when DNA carries body genes (default)"),
    flag("--fixed-sensors",               &["--sensores-fixos"],           SWITCH, "hold every sensor at SENSOR_DEFAULTS (default, control arm)"),
    flag("--no-neuron-memory",            &["--sem-memoria-neuronal"],     SWITCH, "freeze per-neuron memory at alpha = 0 (control arm)"),  // check-language: allow
    flag("--no-neuron-period",            &["--sem-periodo-neuronal"],     SWITCH, "freeze the per-neuron period of L1 (control arm)"),  // check-language: allow
    flag("--no-neuron-threshold",         &["--sem-limiar-neuronal"],      SWITCH, "freeze the per-neuron threshold of L1 (control arm)"),  // check-language: allow
    flag("--no-heterogeneous-activation", &["--sem-ativacao-heterogenea"], SWITCH, "freeze the per-neuron activation choice (control arm)"),  // check-language: allow
    flag("--hippocampus-sector",          &["--hipocampo-setor"],          SWITCH, "turn the sector hippocampus on, and its mutation with it"),
    flag("--hippocampus-erase",           &["--hipocampo-apagar"],         SWITCH, "erase the sector memory between laps"),
    flag("--no-hippocampus",              &["--sem-hipocampo"],            SWITCH, "force the hippocampus off (control arm; wins if it comes last)"),  // check-language: allow
    flag("--symmetric-center",            &["--centro-simetrico"],         SWITCH, "charge backwards progress in the `centro` fitness"),
    flag("--out-beta",                    &[],                             VALUE,  "output fatigue strength; 0 disables it"),
    flag("--output-gain",                 &["--ganho-saida"],              VALUE,  "gain applied to l2_raw/bypass before the tanh"),
    flag("--norm-l2-by-deviation",        &["--norm-l2-por-desvio"],       SWITCH, "normalise l2_raw by its own standard deviation (default)"),
    flag("--no-norm-l2-by-deviation",     &["--sem-norm-l2-por-desvio"],   SWITCH, "historical L2 path, unnormalised (control arm)"),  // check-language: allow
    flag("--pilot",                       &["--piloto"],                   VALUE,  "ghost-pilot gain; diagnostic, not production"),
    flag("--pilot-thr",                   &["--piloto-thr"],               VALUE,  "throttle the ghost pilot holds"),
    flag("--steer-fixed",                 &["--steer-fixo"],               VALUE,  "override steering with a constant"),
    flag("--steer-scale",                 &["--steer-escala"],             VALUE,  "scale the steering command"),
    flag("--pilot-sensor",                &["--piloto-sensor"],            VALUE,  "sensor gain of the ghost pilot"),
    flag("--thr-fixed",                   &["--thr-fixo"],                 VALUE,  "override throttle with a constant"),
    flag("--seed",                        &[],                             VALUE,  "population seed"),
    flag("--wall-cost",                   &[],                             VALUE,  "fitness metres charged per second near a wall (default 2)"),
    flag("--impact-cost",                 &[],                             VALUE,  "fitness metres charged per kilojoule dissipated in impacts (default 1)"),
    flag("--cost-gate-progress",          &[],                             VALUE,  "hold wall/impact cost at zero until the population median fitness reaches this many metres"),
    flag("--brain-skip",                  &[],                             VALUE,  "run the brain once every N ticks"),
    flag("--kernel-profile",              &["--perfil-kernel"],            VALUE,  "kernel profiling level"),
    flag("--wall-life",                   &[],                             VALUE,  "life lost per wall contact"),
    flag("--wiring-tail",                 &["--fiacao-cauda"],             VALUE,  "how many tail weights the wiring cost looks at"),
    flag("--wiring-cost",                 &["--custo-fiacao"],             VALUE,  "fitness cost per unit of wiring"),
    flag("--rpm-cost",                    &["--custo-rpm"],                VALUE,  "fitness cost per unit of RPM"),
    flag("--stag-life-drain",             &[],                             VALUE,  "life drained while stagnant"),
    flag("--same-start",                  &[],                             SWITCH, "every being starts at the same waypoint"),
    flag("--fixed-spawn-slots",           &[],                             SWITCH, "organism city: keep the being->spawn slot map fixed across generations (default rotates by hash(generation, seed))"),
    flag("--navigation-observation",      &[],                             VALUE,  "organism city taxi channels: v2 (dx/dz per 100 m, default) or v3 (body-frame direction + compressed distance)"),
    flag("--taxi-reference-witness",      &[],                             SWITCH, "organism-taxi-v1 scripted reference controller; one being, one generation, never an evolution arm"),
    flag("--school",                      &[],                             VALUE,  "GPU-judged organism lesson: move, stop, turn_near, turn_far, approach_passenger, pickup, navigate_with_passenger or delivery"),
    flag("--school-mode",                 &[],                             VALUE,  "bench (passengers and taxi-taxi contacts off, fitness = lesson score) or life (judge only, taxi fitness)"),
    flag("--school-horizon",              &[],                             VALUE,  "judged ticks per lesson attempt (default: the generation tick budget)"),
    flag("--school-curriculum",           &[],                             VALUE,  "lesson schedule over generations: adaptive (advance on competence) or uniform (cycle the list)"),
    flag("--school-lessons",              &[],                             VALUE,  "comma-separated curriculum lessons (default: all eight motor and taxi-service lessons)"),
    flag("--school-advance-p",            &[],                             VALUE,  "adaptive curriculum: competence p_hat needed to count a generation toward advancing (default 0.5)"),
    flag("--school-advance-gens",         &[],                             VALUE,  "adaptive curriculum: consecutive competent generations before the next lesson (default 2)"),
    flag("--school-start-fold",           &[],                             VALUE,  "lesson starts taken from all, train, validation or test (candidate index mod 3)"),
    flag("--school-shared-start",         &[],                             VALUE,  "all beings of a generation judged on one lesson start of the fold: rotate (candidate = generation mod count) or fixed (candidate 0); default one start per slot, rotated by hash (docs/SCHOOL-SHARED-START-20260922.md)"),
    flag("--school-stop-shaping",         &[],                             SWITCH, "stop lesson: score also pays 300*(1 - slowest speed inside / v_min) + 200*(longest near-stop dwell / hold_ticks); strict success is unchanged (docs/SCHOOL-STOP-SHAPING-20260922.md)"),
    flag("--feedback-mode",               &[],                             VALUE,  "online reward composition: legacy (asymmetric energy term, default), symmetric (energy both ways, paid money as cost) or progress (symmetric plus same-key task progress)"),
    flag("--plasticity-freeze-after",     &[],                             VALUE,  "tick from which the plastic residual is applied but no longer updated; needs --plasticity"),
    flag("--help",                        &[],                             SWITCH, "print this flag table and exit"),
    flag("--social-sensors",              &[],                             SWITCH, "enable population-relative proximity inputs for multi-being worlds"),
    flag("--random-start",                &["--largada-aleatoria"],        SWITCH, "one random start waypoint, shared by the whole population"),
    flag("--damage",                      &["--dano"],                     VALUE,  "damage scale; 0 disables it"),
    flag("--sensory-noise",               &["--ruido-sensorial"],          VALUE,  "sigma of the gaussian noise added to the sensory wave"),
    flag("--dropout-noise",               &["--ruido-dropout"],            VALUE,  "probability a sensory channel is dropped"),
    flag("--weather",                     &["--clima"],                    VALUE,  "grip multiplier (1.0 = dry)"),
    flag("--weather-freq",                &["--clima-freq"],               VALUE,  "frequency the grip multiplier oscillates at, Hz"),
    flag("--wind",                        &["--vento"],                    VALUE,  "wind vector; takes TWO values, x and z"),
    flag("--ghost",                       &["--fantasma"],                 SWITCH, "life bonus for running ahead of the previous champion"),
    flag("--ladder",                      &["--escada"],                   SWITCH, "automatic difficulty ladder (see ladder module)"),
    flag("--interoception",               &["--interocepcao"],             SWITCH, "publish the eight body values into the sensory wave"),
    flag("--resource-scarce",             &["--recurso-escasso"],          VALUE,  "number of energy stations, fewer than the beings"),
    flag("--resource-load",               &["--recurso-carga"],            VALUE,  "energy a station holds"),
    flag("--resource-regen",              &["--recurso-regen"],            VALUE,  "station regeneration per second"),
    flag("--resource-radius",             &["--recurso-raio"],             VALUE,  "station radius, metres"),
    flag("--resource-transfer",           &["--recurso-transfer"],         VALUE,  "station transfer rate"),
    flag("--predator",                    &["--predador"],                 VALUE,  "predator speed as a fraction of the being's"),
    flag("--predator-drain",              &["--predador-dreno"],           VALUE,  "life the predator drains per second"),
    flag("--predator-radius",             &["--predador-raio"],            VALUE,  "predator reach, metres"),
    flag("--predator-perception",         &["--predador-percep"],          VALUE,  "distance at which the being perceives the predator, metres"),
    flag("--predator-grace",              &["--predador-gracia"],          VALUE,  "grace ticks before the predator starts hunting"),
    flag("--predator-spawn",              &["--predador-spawn"],           VALUE,  "how far behind the being the predator spawns, metres"),
    flag("--tire-thermal",                &["--pneu-termico"],             VALUE,  "thermal tire model gain; 0 disables it"),
    flag("--sensor-cost",                 &["--custo-sensor"],             VALUE,  "fitness cost per unit of sensor reach"),
    flag("--sensor-cost-k",               &["--custo-sensor-k"],           VALUE,  "exponent of the sensor cost"),
    flag("--sensor-cost-cap",             &["--custo-sensor-cap"],         VALUE,  "ceiling of the sensor cost"),
    flag("--pilot-formula",               &["--piloto-formula"],           VALUE,  "path to a closed-form pilot"),
    flag("--pilot-forge",                 &["--piloto-forja"],             VALUE,  "path to a forged pilot"),
    flag("--random-gate",                 &["--portao-aleatorio"],         VALUE,  "probability a region gate is randomised"),
    flag("--gate-seed",                   &["--portao-semente"],           VALUE,  "seed of the random region gate"),
    flag("--life-earn",                   &[],                             VALUE,  "life earned per metre"),
    flag("--width",                       &[],                             VALUE,  "track width, metres (default 18)"),
    flag("--fitness",                     &[],                             VALUE,  "fitness mode"),
    flag("--steer-alpha",                 &[],                             VALUE,  "steering low-pass alpha"),
    flag("--motor-head",                  &[],                             VALUE,  "native motor transfer: spline or linear"),
    flag("--throttle-arbitration",        &[],                             VALUE,  "native throttle merge: legacy or signed"),
    flag("--reverse-controller-v2",       &[],                             SWITCH, "use the corrected bounded reverse controller (default)"),
    flag("--legacy-reverse-controller",   &[],                             SWITCH, "use the historical reverse controller as a control arm"),
    flag("--fuel-auto",                   &[],                             VALUE,  "size the tank from the refuelling interval with factor k; enables fuel"),
    flag("--fuel",                        &[],                             SWITCH, "enable the fuel model"),
    flag("--no-fuel",                     &[],                             SWITCH, "disable fuel explicitly (fuel is off by default)"),
    flag("--fuel-tank",                   &[],                             VALUE,  "tank size, tank units; enables fuel"),
    flag("--refuel-lap-only",             &[],                             SWITCH, "refuel on lap close only, not on every sector"),
    flag("--audit-vision",                &["--auditar-visao"],            SWITCH, "audit what the retina actually sees"),
    flag("--retina-road-view",            &[], VALUE, "opt-in physical retina with log ground sampling 4..N metres (20..300); other sensors unchanged"),
    flag("--retina-range",                &[],                             VALUE,  "how far the retina and every metric sensor reach, metres (default 33.3)"),
    flag("--retina-beings",               &[],                             SWITCH, "the retina also occludes on other beings' bodies (docs/RETINA-BEINGS.md)"),
    flag("--start-formation",             &[],                             SWITCH, "start beings in a non-overlapping formation at one waypoint"),
    flag("--no-holes",                    &["--sem-buracos"],              SWITCH, "remove the intentional holes from the track"),  // check-language: allow
    flag("--gearbox-curriculum",          &["--cambio-curriculo"],         SWITCH, "raise the gear ceiling along the curriculum"),
    flag("--gearbox-ceiling",             &["--cambio-teto"],              VALUE,  "highest gear allowed, 2..6"),
    flag("--gearbox",                     &["--cambio"],                   VALUE,  "gearbox mode: travado | auto | neural"),
    flag("--l2-freeze-blind",             &["--l2-congelar-cegas"],        SWITCH, "freeze the mutation of the L2 rows nothing reads (default)"),
    flag("--no-l2-freeze-blind",          &[],                             SWITCH, "allow unused L2 rows to mutate (control arm)"),
    flag("--region-mask",                 &["--mascara-regioes"],          SWITCH, "directed regions: the reference ceiling of M3 (default)"),
    flag("--no-region-mask",              &[],                             SWITCH, "disable the default directed region mask"),
    flag("--tau-seeded",                  &["--tau-semeado"],              SWITCH, "seed tau per region block instead of uniformly (default)"),
    flag("--no-tau-seeded",               &[],                             SWITCH, "disable the default region-specific tau seed"),
    flag("--real-peripheral-vision",      &["--visao-periferica-real"],    SWITCH, "ch144-159 look at the retina edges instead of copying the fovea"),
    flag("--neuromod-regions",            &["--neuromod-regioes"],         SWITCH, "M4 region conductor: the slow brain gates regions"),
    flag("--no-neuromod-regions",         &[],                             SWITCH, "disable the slow-brain region conductor (default)"),
    flag("--observe",                     &["--observar"],                 VALUE,  "record the neural chain of being 0 per tick into this path"),
    flag("--record-wave",                 &["--gravar-onda"],              VALUE,  "record the raw 192-channel wave of being 0 into this UWAV path"),
    flag("--curve-portal",                &["--portal-curva"],             SWITCH, "place the checkpoints on the curves instead of uniformly"),
    flag("--slow-steer",                  &[],                             VALUE,  "steering authority granted to the slow brain"),
    flag("--refuel-laps",                 &[],                             VALUE,  "refuel every N laps"),
    flag("--life-cap",                    &[],                             VALUE,  "ceiling on accumulated life"),
    flag("--n-elite",                     &[],                             VALUE,  "elite kept per generation (default 3)"),
    flag("--tourney",                     &[],                             VALUE,  "tournament size (default 3)"),
    flag("--gen-offset",                  &[],                             VALUE,  "generation number the run starts counting from"),
    flag("--activation",                  &["--ativacao"],                 SWITCH, "collect L1/L2 activation metrics and runtime guards"),
    flag("--activation-n",                &["--ativacao-n"],               VALUE,  "sample the activation metrics every N ticks (default 64)"),
    flag("--regions",                     &["--regioes"],                  SWITCH, "cross the activation samples with the local track curvature"),
    flag("--panel",                       &["--painel"],                   SWITCH, "serve the real-time HTTP panel"),
    flag("--panel-port",                  &["--painel-porta"],             VALUE,  "port of the HTTP panel (default 3200); enables the panel"),
    flag("--tracks",                      &["--pistas"],                   VALUE,  "comma-separated training track seeds, rotated per generation"),
    flag("--world",                       &[],                             VALUE,  "parametric WorldSpec (seed=N,radius=..,curves=..,width=..,holes=..,length=..), repeatable, rotated like --tracks"),
    flag("--world-file",                  &[],                             VALUE,  "JSON array of WorldSpec objects, a world pool"),
    flag("--track-held-out",              &["--pista-held-out"],           VALUE,  "track seed reserved for measuring generalisation"),
    flag("--held-out-every",              &["--held-out-cada"],            VALUE,  "measure the held-out track every K generations (default 5)"),  // check-language: allow
    flag("--replay-every",                &["--replay-cada"],              VALUE,  "write a replay every N generations"),  // check-language: allow
    flag("--fly-map",                     &[],                             VALUE,  "path to a flymap v1 file; required by every other --fly-* flag"),
    flag("--fly-brain",                   &[],                             SWITCH, "the fly map's three commands replace the fast brain's intents"),
    flag("--brain-control",               &[],                             VALUE,  "direct: Fly drives the body; legacy: preserve native and Fly composition"),
    flag("--fly-reflex-alpha",            &[],                             VALUE,  "steering baseline adaptation rate per tick (0 = off)"),
    flag("--fly-reflex-gain",             &[],                             VALUE,  "gain applied to the steering residual after adaptation"),
    flag("--fly-readout-norm",            &[],                             SWITCH, "readout emits the normalised asymmetry index instead of the raw group difference"),
    flag("--fly-spiking",                 &[],                             SWITCH, "GPU-resident delayed LIF fly map drives the three commands (requires v2 pre-CSR)"),
    flag("--fly-tau",                     &[],                             VALUE,  "time constant of the fly rate model, in ticks (default 4)"),
    flag("--fly-p-bind",                  &[],                             VALUE,  "probability a binding gene is redrawn (default 0.05)"),
    flag("--fly-p-off",                   &[],                             VALUE,  "probability a live cell type is pruned (default 0.05)"),
    flag("--fly-p-on",                    &[],                             VALUE,  "probability a pruned cell type comes back (default 0.015)"),
    flag("--fly-sigma",                   &[],                             VALUE,  "sigma of the log-normal step on the fly gain and scale (default 0.1)"),
    flag("--fly-log-every",               &[],                             VALUE,  "print the elite's fly selection every N generations"),
    flag("--fly-spiking-trace",           &[],                             VALUE,  "write compact per-neuron spiking trace for the selected being"),
    flag("--fly-spiking-trace-every",     &[],                             VALUE,  "sample the spiking trace every N ticks (default 64)"),
    flag("--fly-spiking-trace-being",     &[],                             VALUE,  "being index recorded by the spiking trace (default 0)"),
    flag("--fly-eye",                     &[],                             SWITCH, "specific neurons are driven by specific retina pixels (needs a v2 map with the pixel-binding block)"),
    flag("--fly-readout",                 &[],                             VALUE,  "readout source for the three commands: genome | reflex (default genome; reflex needs a v3 map with the reflex-readout block)"),
    flag("--fly-cruise-throttle",         &[],                             VALUE,  "constant throttle used when --fly-readout reflex's throttle slot is NONE (default 0.6)"),
    flag("--fly-reflex-sign",             &[],                             VALUE,  "sign multiplied into the steer command under --fly-readout reflex: 1 or -1 (default 1)"),
    flag("--resume",                      &["--retomar"],                  SWITCH, "resume from the checkpoint in --out"),
    flag("--out",                         &[],                             VALUE,  "output directory"),
];

/// The flag `arg` names, by canonical name or by any deprecated alias. Pure lookup.
///
/// Linear over 112 rows, once per command-line argument — a map would be measurably
/// slower to build than this is to scan.
pub fn canonical(arg: &str) -> Option<&'static Flag> {
    FLAGS
        .iter()
        .find(|f| f.name == arg || f.aliases.contains(&arg))
}

/// [`canonical`], plus a one-time warning on stderr when `arg` is a deprecated alias.
///
/// This is what the argument loop calls. The warning is emitted once per alias per
/// process, so a flag repeated on the command line does not repeat the notice.
pub fn resolve(arg: &str) -> Option<&'static Flag> {
    let found = canonical(arg)?;
    if let Some(alias) = found.aliases.iter().copied().find(|a| *a == arg) {
        warn_once(
            alias,
            &format!("flag `{alias}` is deprecated, use `{}`", found.name),
        );
    }
    Some(found)
}

/// A deprecated spelling of a flag's VALUE, for the flags whose values are an enum.
#[derive(Debug)]
pub struct ValueAlias {
    /// Canonical name of the flag this value belongs to; must appear in [`FLAGS`].
    pub flag: &'static str,
    /// The deprecated value.
    pub alias: &'static str,
    /// The canonical value it maps to.
    pub name: &'static str,
}

/// Row constructor for [`VALUE_ALIASES`].
const fn value_alias(flag: &'static str, alias: &'static str, name: &'static str) -> ValueAlias {
    ValueAlias { flag, alias, name }
}

/// Deprecated values, per flag. Same contract as [`Flag::aliases`]: never shrinks.
#[rustfmt::skip]
pub const VALUE_ALIASES: &[ValueAlias] = &[
    value_alias("--tau-bank", "livre",    "free"),     // check-language: allow
    value_alias("--tau-bank", "log-fixo", "log-fixed"),
    value_alias("--tau-bank", "espectral", "spectral"),
];

/// The canonical spelling of `value` for `flag`, or `None` when it is already
/// canonical (or unknown — the caller still owns validation).
///
/// The pure half of [`resolve_value`], which is what the parser calls; this one exists
/// so a test can assert the mapping without emitting a warning, hence `allow`.
#[allow(dead_code)]
pub fn canonical_value(flag: &str, value: &str) -> Option<&'static str> {
    VALUE_ALIASES
        .iter()
        .find(|v| v.flag == flag && v.alias == value)
        .map(|v| v.name)
}

/// [`canonical_value`], plus a one-time warning naming the flag and both spellings.
pub fn resolve_value(flag: &str, value: &str) -> Option<&'static str> {
    let found = VALUE_ALIASES
        .iter()
        .find(|v| v.flag == flag && v.alias == value)?;
    warn_once(
        found.alias,
        &format!(
            "value `{}` of {} is deprecated, use `{}`",
            found.alias, found.flag, found.name
        ),
    );
    Some(found.name)
}

/// Prints `message` the first time `key` is seen in this process.
///
/// A `Mutex<BTreeSet>` and not a per-row `AtomicBool`: the set is touched at most 112
/// times in the life of a run, and the alternative is a second table to keep in step
/// with the first.
fn warn_once(key: &'static str, message: &str) {
    static WARNED: OnceLock<Mutex<BTreeSet<&'static str>>> = OnceLock::new();
    let seen = WARNED.get_or_init(|| Mutex::new(BTreeSet::new()));
    // A poisoned lock here means another thread panicked mid-warning. Losing a
    // deprecation notice is not worth aborting a run that is otherwise fine.
    if let Ok(mut set) = seen.lock() {
        if set.insert(key) {
            eprintln!("{message}");
        }
    }
}
