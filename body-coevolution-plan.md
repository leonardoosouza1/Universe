# Body–brain coevolution: first executable stage

Status: E1a implemented and integration-tested on 2026-09-20. The broader roadmap
remains in `../META/EVOLUCAO-AUTOMATICA-CORPO-CEREBRO.md`.

Review: `Genome` already evolves neural weights and recurrent parameters;
`brain_interface.rs` already separates sensing and motor commands. Car capabilities
are constants in `make_physics_state`. The variable-length Fly genome tail prevents
blindly appending positional genes. Held-out creates its own bodies and must express
the same genomes. A real baseline trace was recorded for being 0, generation 0,
tick 64: neural output → motor mapper → GPU physics → body observation.

Implement E1a, a bounded experimental slice, not the entire E1–E6 research roadmap:

- [x] Add two bounded body genes: engine and brake capacity. Greater capacity adds
  mass; greater engine capacity increases throttle-dependent fuel use. Coefficients
  are explicit experimental assumptions, not calibrated vehicle engineering.
- [x] Persist genes in a versioned header extension; preserve exact v2 output for
  neutral bodies, legacy loading, and the variable-length Fly tail. Reject unknown
  versions and malformed new headers.
- [x] Add opt-in `--evolving-body` and `--fixed-body`. Preserve the existing brain
  RNG stream. Select brain and body together; keep elite bodies unchanged.
- [x] Express bodies at birth in training and held-out using existing GPU physics
  fields. No new tick buffer, readback, kernel, or neural action override.
- [x] Record all population body genes/expressed parameters per generation and
  expose the selected body's physical parameters in observation samples.
- [x] Verify bounds, inheritance, persistence/Fly compatibility, neutral physical
  equivalence, GPU response, and before/after control runs; run short paired smokes.

Not implemented in this stage: variable topology, developmental growth, additional
sensor organs, learned plasticity, thermal redesign, continuous reproduction, taxi
integration. Each changes the hypothesis and requires a separate controlled test.
Small smoke runs prove wiring and reproducibility, not superior intelligence.
Results and remaining replay provenance gap: `docs/BODY-COEVOLUTION-VALIDATION.md`.
