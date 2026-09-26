# Research runtime contract v1

These controls are opt-in. Genome binary layout and school schema4/56-f32 layout are unchanged. Existing consumers may ignore additive `research` metadata; they must not describe `linear-wave-v2` as a native anatomical trace.

| Control | Producer and runtime meaning | Preconditions / observable evidence |
|---|---|---|
| `--school-start-distance-m M` | Host `prepare_starts`: static swept-clearance check, lane-aligned start M metres before station; body starts at rest | Finite M in [0,120], service-v2 bench, fixed pickup/delivery lesson. Delivery starts onboard; accepted pickup remains absent |
| `--school-start-curriculum 0,6,18,36,native` | Host generation boundary: advance after complete start-coverage windows meet `--school-advance-p` for `--school-advance-gens` windows | Fresh run, distinct starts, no lesson curriculum or fixed-distance override. Final `native` stage uses ordinary starts; no fabricated graduation |
| `--school-retain-skills N` | Archive admits full success across the evaluated K starts; saves genomes and origin. Deterministic context rotation reinjects at most N into offspring tail after mutation | Fresh school bench, K>=2 common starts; no head-only/random-body combination; requires space beyond elites. Capacity N/context and 64 total. No implicit archive restoration |
| `--mutation-profile head-only` | Selection plus Gaussian perturbation of native L2 rows5–7 only; no body/lifetime mutations | Native episodic taxi, explicit `--research-body fixed`, no lifetime evolution. Metadata contains dose. Not SM-G; genotype change does not prove behavior change |
| `--research-controller linear` | GPU `brain.wgsl` early branch reads same packed wave192 and writes intent96 using first9 L1 rows | Requires `--no-region-mask`, `--no-slow`, no lifetime/hip/fly/native neural traces. Other intent rows zero. Motor mapper remains unchanged |
| `--city-family F --city-seed S` | Static city builder shares generated centers across road, flow, stations, lights, spawn and school geometry | Explicit family grid/compact/wide and u64 environment seed. No legacy track-held-out. Grid spacing variation, not full topology/block grammar |
| `--research-body fixed/random/evolved` | Fixed neutral morphology, generation-wise independent random morphology, or inherited mutated morphology | Episodic taxi only; organism metabolism DNA neutral in all arms. Random body uses independent RNG; K replicas share morphology |

## Equations and boundaries

Linear fast controller:

`intent[j] = tanh(sum_c(l1[j,c] * signed_wave[c]/127))`, j=0..8.

All other intents are zero. No CfC, L2, plasticity or anatomical intermediate trace is claimed. Motor spline, gearbox and physical actuator semantics remain those of the scenario. This experiment needs its controller flag to replay its genome correctly; the genome file alone does not encode the override. Run and replay metadata record the identity.

Prepared-distance geometry is in metres. The existing legal taxi event remains authoritative. Delivery preparation is flags bit4 in the school uniform only; no new record offsets. Initial onboard state does not create a successful pickup tick. Default bit4=0 preserves ordinary initialization.

Curriculum promotion measures population performance across coverage windows, not one frozen policy's competence. A completed stage does not prove generalization. Changing distance clears coverage. Archive scores are ranked only within an exact context including lesson, start batch, geometry and preparation; cross-context injection rotates contexts without comparing score scales. Reassessment still occurs in the new environment.

The morphology experiment neutralizes organism metabolism genes so fixed-body is not secretly evolving a separate metabolism. `random` is redrawn each generation independent of selection, not an inherited random founder that selection could evolve. Capacity/mass/energy equations remain in `gpu/body.rs` and `gpu/organs.rs`.

## Evidence and rejection

`organism_gen*.json` holds the runtime controls, geometry and body expression. `school_result_gen*.json` holds stage, observed task events and coverage. `skills/archive_gen*.json` links retained DNA to its admitting batch. `skill_reinjection_gen*.json` identifies overwritten offspring slots and source evidence. `run_research_battery.py` snapshots source/binary and writes commands and per-seed first/middle/last summaries; missing events cause failure, not zero imputation.

No new per-tick host inference or readback is introduced. Static geometry, reproduction and reporting remain host-side generation operations. Tests of inference/physics run on real Vulkan GPU; adapter fallback to CPU is forbidden.

## Deliberate implementation limits

This is a staged implementation of the immediate roadmap. Not implemented: faithful Go-Explore snapshots, MAP-Elites behavioral tessellation, sensitivity-gradient mutation, PPO learner/interop, ACCEL regret estimation, curved block grammar, IDM/MOBIL traffic, HER or Dreamer. The corresponding research directions require passing basic service gates and separate implementation/validation; the controls here must not be marketed under those algorithm names. The near-start curriculum, skill archive and grid-family controls are bounded adaptations.

## Integration corrections found by validation

- Prepared tasks initialize once on the first resident GPU dispatch, before the first sensor/brain read. This is a reset latch, not a numeric tick assumption; the taxi loop starts at tick1. Uniform bit5 invokes initialization without elapsed ticks, requests, or accepted events. Prepared delivery starts in onboard phase2.
- `linear-wave-v1` divided an already spectrally scaled L1 matrix by sqrt(192) again. `linear-wave-v2` removes the extra gain reduction. The native regional mask is incompatible with using the first9 L1 rows as complete action rows: it removes body/task inputs. The linear arm therefore explicitly requires `--no-region-mask`. Its comparison is an architecture intervention, not a single-layer ablation.
- `head-only` freezes hereditary blocks per ancestry, not motor behavior. Shared L2 normalization couples service rows and motor outputs. Each generation emits immutable SHA-256 ancestry evidence; a population slot can inherit a different parent.
- Runtime checkpoints bind to the SHA-256 of the actual executable, including all kernels. Archive entries and reinjections carry genome hashes and file evidence. Archive plus dynamic difficulty ladder is rejected.
- Body provenance schema3 describes independent redraw, neutral reset, or inherited mutation accurately. Research fixed resets morphology and organism metabolism; legacy fixed body still preserves dormant DNA.
- An evaluation fold is only an audit unless the training manifest proves exclusion. One-generation evaluation verifies the exact exported genome, checkpoint-loaded flag, and mix1.0; it does not infer held-out status from a directory name.

## Navigation and operational endpoints

`--navigation-observation v4` is an explicit privileged map-route observation:
channels178/179 are the cardinal descent direction of the existing BFS road field
in the body's frame; channel183 is remaining road distance/(distance+100m).
Within4m, direction points to the exact station. Missing/off-road/unreachable
routes encode (0,0,1). Other v3 channels and traffic ownership remain unchanged.
It never writes actions. It does not prove learned mapping or visual-only control,
and the point-road graph does not certify lane legality, vehicle clearance or
turning feasibility. The option remains off by default.

Service-v2 stall accounting resets on new task progress or accepted service,
not merely reaching a station. The limit is max(20s,min(180s,half the episode
budget)); 20s exceeds this generator's 16s red-light half-cycle. An open service-v2
horizon is failure, not free truncation. Delivery readiness resets after pickup.
The runner reports contact-free successes and full contact-free deliveries
(accepted pickup AND dropoff) separately from prepared deliveries. Final physical
integrity is still a separate field; none of these small tests certifies safe
urban driving.

## Short complete service course

`--school-service-course M` (12..96 metres in 3m cells) is a distinct training
intervention, not a reinterpretation of prepared-onboard distance stages. Four
fixed straight tasks occupy real city roads. Each starts 6m before its passenger,
in the legal lane, seeking pickup; delivery requires the authoritative pickup and
dropoff transitions. All four directions are evaluated per genome (K=4). The
station bank is replaced explicitly for this bench; production city defaults
remain unchanged. No actuator or success override is introduced.

The course cannot combine with prepared dropoffs, changing city families, lesson
curricula, ecology or runtime-state resume. Population DNA may be evaluated with
the same declared controller/body/navigation settings. This course is a bridge
to complete service, not evidence of city-wide navigation or traffic safety.

| Phase | Producer | Consumer | Device / frequency |
|---|---|---|---|
| Course geometry | service_course::build, static clearance and road routes | station bank, LessonGeometry | host, setup |
| Task initialization | SchoolGpu::initialize before first sensory read | task/body observation | GPU, birth |
| Observation | taxi senses -> optional route_senses -> traffic senses | packed wave192 -> brain | GPU, tick |
| Action and events | neural intent -> organism_action -> physics -> taxi post | school judge | GPU, tick |
| Selection | frozen judged records, K=4 common starts | inherited brain/body mutations | host, generation |

Judge revision `service-v2-banked-pickup-frozen-evidence-v1` freezes the entire
school record at closure. Requests and accepted events after that tick are not
part of the attempt. Readiness resets after a full task's pickup, but the score
banks 200 points for that accepted pickup; otherwise resetting up to 200 points
would reward refusing pickup. This is once per assigned passenger, not per
request/tick. Prepared onboard episodes have no accepted pickup and earn none.
The unchanged terminal gate still requires actual delivery; the extra credit
does not turn pickup into success. Coefficients remain engineering hypotheses.

The runner verifies an embedded build-source SHA against its captured source,
and checks event spans and exact checkpoint bytes. Source identity is separate
from scientific effectiveness. Old binaries/runs without this evidence contract
remain historical and must not be silently reported as current validations.
