# Vila bed policy CPU pilot

This standalone Rust process is a **bounded goal selector**, separate from the
approved predictive `vila-shadow-cpu` model. It reads one JSON request from
stdin and writes one JSON response. The caller owns versioned state, constructs
bed candidates from verified Paper observations, applies the selected waypoint
through the opt-in Java `ControlPilot`, and returns a matched outcome. JSON
requests on stdin do not modify files. The explicit `--dispatch-once` mode
writes one bounded opt-in command and decision record; it never edits a world.

`{"op":"init"}` returns a `vila-goal-policy/v2` state. `decide` takes that state,
an observation (`episode`, `villager_uuid`, `tick`, `is_night`, `has_home`) and
at most 128 candidates. Each candidate supplies a stable bed/approach `id`,
`x/y/z`, distance in blocks, path point count, crowding count, and the boolean
Paper gates `path_complete` and `bed_available`. Invalid or unreachable
candidates are excluded. The policy emits no action by day or for a villager
with a home. Before feedback, zero weights break ties toward the nearest bed;
this prior is **not learned skill**.

`decide_paper` accepts a complete `vila-bed-candidates/v3` Paper snapshot with
required `observation.sleeping`. Missing sleep state fails closed, and a
sleeping villager receives no action. For
an unhomed villager, it uses the verified bed candidates. For a villager that
already owns HOME, it returns a separate `return_home`/`walk_to` decision only
when `home_route_status=reachable` and `home_approach` is valid. The output
names its `goal`, `action`, episode, UUID, decision tick, candidate ID and
target coordinates; it is still a proposal, not an applied command. The
initial HOME proposal has zero learned updates and has not improved the task.

`learn` requires the exact decision identity and update count plus outcome:
decision tick, actual action tick within 40 ticks, arrival, home acquisition,
sleep, progress ratio, lost health and rescue. It
updates a five-feature linear reward estimator and a bounded per-bed running
mean. Replaying feedback against an updated state fails. Rewards prioritize
home and sleep (5 each), then arrival (1) and progress (up to 0.5), while
penalizing damage, rescue and time. These coefficients define a research
objective, not a certified behavior. State must be saved atomically by the
caller and frozen for controlled evaluation; learning on held-out runs would
invalidate the comparison.

Build/test with `CARGO_BUILD_JOBS=1 cargo test --offline --locked`. This crate
must be copied byte-identically to `vila-viva-server/cpu-policy` before use.
Opt-in Paper candidate production exists in `BedCandidates`, and the opt-in
Java `ControlPilot` executes bounded commands. The one-shot Rust dispatcher
below now converts a policy decision to a Java command in an isolated copy.
Continuous invocation, matched vanilla evaluation and deployment gates are
**not implemented yet**. The separate one-shot feedback ingestor below now
rejects incomplete Paper episodes and calls this Rust binary's `learn` API;
its enriched Java outcome has now run in one isolated Paper episode, with a
verified Rust `learn` update. This does not establish benefit over vanilla.

The original `decide` operation still handles only free-bed selection and
returns `null` for a HOME owner. Use `decide_paper` for the full Paper envelope;
that operation proposes a HOME route only after the Paper producer verifies a
complete route and at least 160 contiguous stationary ticks while awake, at
night and on ground. This is a conservative interference gate, not proof that
vanilla failed or that the policy is superior. The
older `pilot_outcome` event lacks fields required by `learn`; it fails closed
in the ingestor. The newer opt-in Java source adds episode/UUID, action tick
and per-tick sleep observation, but is not active in production. See
`../docs/AI-FLOW-CONTRACT-AUDIT.md` before extending
the policy. A changed Rust contract must be mirrored byte-for-byte in
`Universe/crates/vila-policy-cpu/` and checked against a real Paper payload.

The opt-in CLI supports one bounded dispatch:

```text
vila-policy-cpu --dispatch-once OBS LIVE EPISODE STATE DECISION PILOT SEQ
```

It requires Paper inputs modified within 10 seconds, live tick within 20 ticks
of observation, matching opt-in episode/UUID, and the selected villager alive
and awake in both inputs. `STATE` is read if present or initialized to v2.
When the policy proposes an action, the CLI writes a
`vila-pilot-dispatch/v1` record with the state/decision, then `pilot.json`.
Synchronized temporary files are published without overwriting; an existing
dispatch fails closed. `SEQ` is caller-supplied and must be monotonic. This is
**one action**, not a daemon, learner or proof of benefit.

In the isolated `policy-dispatch-1351` episode, seq1 came from Paper
observation tick15060/live tick15069 and was accepted by Java at tick15080.
The path had 12 points, but by tick15280 the villager had not arrived;
progress was -0.0015 block and health 5→5. State updates remained zero.

For a **closed** episode recorded with the enriched opt-in Java contract:

```text
python3 ingest_feedback.py DECISION PILOT EPISODE_DIR POLICY_BIN STATE_OUT RECEIPT_OUT
```

`STATE_OUT` must be `state.json` and `RECEIPT_OUT` must be `feedback.json`
beside `decision.json` and `pilot.json`. The ingestor checks the opt-in episode
header, close marker, zero dropped lines, gzip event streams, single accepted
seq, episode/UUID/villager index, exact action/target/tick, health, progress,
per-tick sleep observation and rescues for the same villager. It then sends
the verified `Outcome` to Rust `learn`, atomically publishes new state and a
receipt, and refuses to overwrite them. The old `policy-dispatch-1351` Paper
episode is rejected because its events lack identity and `slept_during`; no
reward or weight update may be inferred from sampled keyframes. In the later
`policy-feedback-home-1434` Paper episode, seq1 was accepted at tick 13820,
`pilot_outcome` observed sleep at tick 13887 with 1.5767 blocks of progress,
health 20→20 and no rescue, and the closed zero-drop episode yielded one
experimental Rust update (reward 5.1877). Replay was rejected. A matched
vanilla baseline is pending; this is a research update only, never automatic
promotion to the approved model.
