# Experimental Villager motor policy

This standalone crate is the canonical source for the server's
`motor-policy/` mirror. It reads Paper `vila-survival-observation/v4`
and emits one bounded `nudge` for an explicitly leased, unaware Villager.
The candidate choice uses a **fixed goal-alignment baseline**, not a learned
policy. It abstains when HOME requires a vertical route or no safe pulse
points meaningfully toward it.
No approved weights or bed-policy state are read or changed.

Build with `cargo build --manifest-path crates/vila-motor-policy-cpu/Cargo.toml`
and run `--decide-once OBS LIVE EPISODE DECISION PILOT SEQ` in an isolated
Paper copy. See `docs/experiments/VILA-POLICY-PILOT-20260927.md` for the
preregistration, result, and remaining task-level gates.
