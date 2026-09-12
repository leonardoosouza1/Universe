//! Acceptance tests for the flag registry: the table and the parser cannot drift.
//!
//! A separate file, like `laps_tests.rs`, so `cli_flags.rs` stays a table a reader can
//! hold in one pass (`docs/STYLE.md` §2).
//!
//! The tests that matter here read `main.rs` ITSELF with `include_str!` and compare its
//! `match` arms against [`FLAGS`]. That is deliberate: a registry that merely *claims*
//! to describe the parser is worth nothing, because the failure it exists to prevent —
//! a flag renamed in one place and not the other, falling through to `_ => {}` and
//! running with the default — leaves no trace in the log. Comparing the two in BOTH
//! directions is the only check that fails at compile-test time instead of after a
//! six-hour battery.

use crate::cli_flags::{FLAGS, VALUE_ALIASES, canonical, canonical_value, resolve, resolve_value};

/// The parser's own source, read at test-compile time.
const MAIN_RS: &str = include_str!("main.rs");

/// Every `"--flag" => {` arm of the argument loop, with `true` when the arm's body
/// opens by consuming the next argument (`i += 1;`) — i.e. when the flag takes a value.
///
/// The scan is intentionally literal-minded: it only accepts a line that is exactly a
/// flag pattern opening a block, which is how all 112 arms are written. A new arm in
/// some other shape is not silently ignored — it is simply not found, and the
/// round-trip test below then fails on the registry row that has no arm.
fn parser_arms() -> Vec<(&'static str, bool)> {
    let lines: Vec<&'static str> = MAIN_RS.lines().collect();
    let mut arms = Vec::new();
    for (n, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if !trimmed.starts_with("\"--") || !trimmed.ends_with("\" => {") {
            continue;
        }
        let name = trimmed.trim_end_matches(" => {").trim_matches('"');
        // First line of the body that is neither blank nor a comment.
        let first = lines[n + 1..]
            .iter()
            .map(|l| l.trim())
            .find(|l| !l.is_empty() && !l.starts_with("//"))
            .unwrap_or("");
        arms.push((name, first == "i += 1;"));
    }
    arms
}

#[test]
fn the_scan_of_main_rs_finds_the_whole_argument_loop() {
    // A guard on the guard: if `parser_arms` silently stopped matching (a reformat, a
    // moved loop), every test below would pass over an empty list and prove nothing.
    let arms = parser_arms();
    assert!(
        arms.len() >= 100,
        "only {} arms found in main.rs — the scan is broken, not the registry",
        arms.len()
    );
    assert_eq!(arms.len(), FLAGS.len(), "one arm per registry row, exactly");
}

#[test]
fn every_flag_the_parser_matches_is_in_the_registry() {
    for (name, _) in parser_arms() {
        let found = canonical(name).unwrap_or_else(|| {
            panic!("main.rs matches `{name}`, which is not in cli_flags::FLAGS")
        });
        assert_eq!(
            found.name, name,
            "main.rs must match the CANONICAL name, not an alias"
        );
    }
}

#[test]
fn every_registry_row_is_matched_by_the_parser() {
    let arms = parser_arms();
    for flag in FLAGS {
        assert!(
            arms.iter().any(|(name, _)| *name == flag.name),
            "`{}` is in the registry but no arm in main.rs matches it",
            flag.name
        );
    }
}

#[test]
fn takes_value_matches_what_the_parser_consumes() {
    for (name, consumes) in parser_arms() {
        let flag = canonical(name).expect("covered by the test above");
        assert_eq!(
            flag.takes_value,
            consumes,
            "`{name}`: registry says takes_value={}, main.rs {} the next argument",
            flag.takes_value,
            if consumes {
                "consumes"
            } else {
                "does not consume"
            }
        );
    }
}

#[test]
fn aliases_resolve_to_their_canonical_flag() {
    for flag in FLAGS {
        assert_eq!(
            canonical(flag.name).map(|f| f.name),
            Some(flag.name),
            "a canonical name must resolve to itself"
        );
        for alias in flag.aliases {
            assert_eq!(
                resolve(alias).map(|f| f.name),
                Some(flag.name),
                "alias `{alias}` must resolve to `{}`",
                flag.name
            );
        }
    }
}

#[test]
fn no_two_registry_entries_collide() {
    let mut seen: Vec<&str> = Vec::new();
    for flag in FLAGS {
        for spelling in std::iter::once(&flag.name).chain(flag.aliases) {
            assert!(
                !seen.contains(spelling),
                "`{spelling}` appears twice in the registry (row `{}`)",
                flag.name
            );
            seen.push(spelling);
        }
    }
}

#[test]
fn every_row_carries_a_help_line() {
    for flag in FLAGS {
        assert!(!flag.help.is_empty(), "`{}` has no help text", flag.name);
        assert!(
            !flag.help.ends_with('.'),
            "`{}`: help is one line, no full stop",
            flag.name
        );
    }
}

/// The renames of 2026-09-12, pinned one by one.
///
/// These eleven are the flags the queued battery requests in
/// `/media/llm-data/universe-runs/neuromod-lab-taxi/_nmd/` were written against. The
/// table above could satisfy every structural test while dropping one of them; this
/// test is the one that notices.
#[test]
fn the_portuguese_spellings_of_the_queue_still_resolve() {
    // check-language: off -- the left column quotes the deprecated Portuguese flags.
    let pairs = [
        ("--hipocampo-setor", "--hippocampus-sector"),
        ("--hipocampo-apagar", "--hippocampus-erase"),
        ("--sem-hipocampo", "--no-hippocampus"),
        ("--banco-tau", "--tau-bank"),
        ("--replay-cada", "--replay-every"),
        ("--tau-semeado", "--tau-seeded"),
        ("--mascara-regioes", "--region-mask"),
        ("--retomar", "--resume"),
        ("--seres", "--beings"),
        ("--pistas", "--tracks"),
        ("--largada-aleatoria", "--random-start"),
    ];
    // check-language: on
    for (old, new) in pairs {
        assert_eq!(
            canonical(old).map(|f| f.name),
            Some(new),
            "`{old}` must still resolve to `{new}`"
        );
    }
}

#[test]
fn value_aliases_name_a_flag_in_the_registry() {
    for value in VALUE_ALIASES {
        let flag = canonical(value.flag)
            .unwrap_or_else(|| panic!("value alias names unknown flag `{}`", value.flag));
        assert_eq!(
            flag.name, value.flag,
            "value aliases key on canonical names"
        );
        assert!(flag.takes_value, "`{}` takes no value", value.flag);
        assert_ne!(
            value.alias, value.name,
            "an alias must differ from its name"
        );
    }
}

#[test]
fn the_tau_bank_values_of_the_queue_still_resolve() {
    // check-language: off -- deprecated Portuguese values of `--tau-bank`.
    let pairs = [
        ("livre", "free"),
        ("log-fixo", "log-fixed"),
        ("espectral", "spectral"),
    ];
    // check-language: on
    for (old, new) in pairs {
        assert_eq!(canonical_value("--tau-bank", old), Some(new));
        assert_eq!(resolve_value("--tau-bank", old), Some(new));
        // A canonical value is not an alias of itself: the caller keeps it as it is.
        assert_eq!(canonical_value("--tau-bank", new), None);
        // And the table is keyed on the flag, not on the value alone.
        assert_eq!(canonical_value("--tracks", old), None);
    }
}
