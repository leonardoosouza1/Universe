//! Canonical CLI flag names of `hip_check`, cut down from the pattern in
//! `src/bin/gpu_evolve/cli_flags.rs` to the three flags this binary renamed to English
//! on 2026-09-12 (`--hip`, `--memory` and `--help` never had a Portuguese spelling, so
//! they need no row here).
//!
//! `main()` runs every argument through [`resolve`] before matching on it, so the
//! `match` in `main()` only ever sees a canonical name and a deprecated spelling keeps
//! working with a one-time warning on stderr — never delete a row, an old command line
//! still has to run.

/// One canonical flag, plus the single deprecated spelling it used to be.
struct Flag {
    /// Canonical English name, spelled exactly as the `match` arm in `main()`.
    name: &'static str,
    /// The Portuguese spelling this flag answered to before the 2026-09-12 rename.
    alias: &'static str,
}

/// Renamed flags of `hip_check`. Never shrinks: a run request written before the
/// rename must still parse.
const FLAGS: &[Flag] = &[
    Flag {
        name: "--genome",
        alias: "--genoma",
    },
    Flag {
        name: "--tolerance",
        alias: "--tolerancia",
    },
    Flag {
        name: "--sectors",
        alias: "--setores",
    },
];

/// The canonical name for `arg`, whether `arg` already is one or is the deprecated
/// alias. `None` when `arg` names neither of the three renamed flags, so `--hip`,
/// `--memory`, `--help` and an unknown flag all fall through untouched.
///
/// The pure half of [`resolve`], so a test can check the mapping without a warning
/// landing on stderr.
///
/// `allow(dead_code)`: nothing on the run path calls this outside `tests.rs` — `main()`
/// always wants the warning too, so it calls [`resolve`] directly.
#[allow(dead_code)]
pub fn canonical(arg: &str) -> Option<&'static str> {
    FLAGS
        .iter()
        .find(|f| f.name == arg || f.alias == arg)
        .map(|f| f.name)
}

/// [`canonical`], plus a warning on stderr the one time `arg` is the deprecated
/// spelling. What `main()`'s argument loop calls.
pub fn resolve(arg: &str) -> Option<&'static str> {
    let found = FLAGS.iter().find(|f| f.name == arg || f.alias == arg)?;
    if found.alias == arg {
        eprintln!("flag `{arg}` is deprecated, use `{}`", found.name);
    }
    Some(found.name)
}
