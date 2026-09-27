# Experimental Villager motor policy

This standalone crate is the canonical source for the server's
`motor-policy/` mirror. It reads Paper `vila-survival-observation/v3`
and emits one bounded `nudge` for an explicitly leased, unaware Villager.
The current candidate choice is a **fixed baseline**, not a learned policy.
No approved weights or bed-policy state are read or changed.

Build with `cargo build --manifest-path crates/vila-motor-policy-cpu/Cargo.toml`
and run `--decide-once OBS LIVE EPISODE DECISION PILOT SEQ` in an isolated
Paper copy. See `docs/experiments/VILA-POLICY-PILOT-20260927.md` for the
preregistration, result, and remaining task-level gates.
