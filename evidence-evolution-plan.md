# Evidence-driven evolution implementation

Goal: make the research roadmap executable in Universe, with opt-in experiments, truthful provenance and GPU-only simulation. Research requiring a new learner (PPO/HER/Dreamer), faithful snapshot restoration and full traffic generation remains conditional on the service-learning gates; do not label approximations as those algorithms.

- [ ] P1: Add a strict experiment contract and prepared service starts (near pickup/dropoff, then greater distance); preserve default geometry and event rules. Verify GPU accepted service from declared starting conditions.
- [ ] P2: Add bounded competence retention with context-specific scores and checkpoint provenance; inject archived genomes after mutation without overwriting elites. Verify no cross-context score ranking or implicit resume.
- [ ] P3: Add opt-in service-head mutation and frozen-backbone profile; preserve the legacy RNG path. Verify emitted genotype changes and GPU service-output reachability.
- [ ] P4: Provide a small learned-controller comparison using the same observation/action contract, explicitly distinguishing it from the scripted witness. Verify active producer and trace identity.
- [ ] P5: Add deterministic city variation, coherent spawn/lesson geometry and train/test scenario identities. Verify city contracts and short GPU rollouts, not generalization claims.
- [ ] P6: Build a GPU experiment runner with paired seeds, fixed/random/evolved body comparisons, staged curriculum, bounded budget and service-funnel report. Reject malformed or incompatible runs.
- [ ] P7: Validate negative contracts, GPU fixtures and a small integrated battery; document observed outcomes, failed gates and remaining work. Freeze source/binary/config hashes for every battery.

Done means mechanisms are connected, defaults preserved, incompatible combinations fail, artifacts are inspectable, and measured results are reported without claiming autonomous delivery from witnesses or prepared starts.

## Executed flow before edits

| Phase | Producer | Data / device | Consumer / frequency |
|---|---|---|---|
| Generation setup | gpu_evolve::main, organism_school::lesson_start_pool | LessonGeometry, f32 metres/headings on host; full Genome | body_evolution::at_birth, SchoolGpu::reset; once/generation |
| Perception | organism_sensors/taxi encode + native perception | sensory wave 192 channels GPU, goal/body channels | native recurrent brain each decision |
| Brain/action | native brain -> organism_action | intent96 f32 -> spline, slow modulation, clamps -> actuator | GPU physics; taxi service requests read native intent directly |
| Task/judge | taxi post -> school::judge | body128 + school56 f32, schema4 | sample/readback at generation boundary |
| Selection | aggregate_school_batch -> evolution::evolve | per-genome score over K replicas, CPU generation boundary | mutated genomes uploaded next generation |
| Evidence | main writes school_result/organism/genome files | JSON plus genome binary | experiment summarizer / existing replay consumers |

Previously verified trace: graduate-adaptive-life-s11 being0 tick11008 had raw brake .322675, motor-head .275588 and applied .275588, speed .036736m/s. This demonstrates a working actuator in that sample, not useful timing. Last native paired service-v2 check had zero pickup requests in both arms. New evidence must distinguish action reachability, prepared-state learning and full autonomous service.

New controls may not insert per-tick host inference or readback. New JSON metadata is additive; existing binary genome and school record layouts stay unchanged unless explicitly versioned. Mutation/head ablations are inference-time experimental conditions, not automatically new saved architectures. Default-off features must not consume extra RNG draws.
