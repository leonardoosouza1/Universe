# Answer to Astra's spiking-kernel delivery (14/09/2026, 17:05)

Received as commit `61ec419` (committed unchanged, as the project's convention requires):
`src/fly_map_spiking.rs` (704 lines, CPU reference), `src/gpu/fly_map/spiking.rs` (1 077),
`spiking.wgsl` (277), `spiking_parity_test.rs` (165), the `fly_map_view` binary (286) and four
flags. Request it answers: [`KERNELS-FLY-SPIKING.md`](KERNELS-FLY-SPIKING.md).

This is a partial audit. What is checked below is checked by reading the code and running the
suites; what is not checked is named as not checked, and the one acceptance criterion that
cannot be exercised with the delivered API is stated as a concrete request, not a complaint.

---

## 1. What checks out

**Constants: exact.** `SpikingParams::default()` (`fly_map_spiking.rs:261-278`) matches the
request term for term — `dt 0.1 ms`, `v0 = v_rst = −52 mV`, `v_th = −45 mV`, `t_mbr 20 ms`,
`tau_syn 5 ms`, refractory `2.2 ms`, delay `1.8 ms`, `w_syn 0.275 mV`, Poisson factor 250 — and
the derived counts come out right: `delay_steps() = 18`, `refractory_steps() = 22`. The readout
time constant is 50 ms, the value K3 asked for.

**The gate is off by default and nothing moved.** With `--fly-spiking` absent the rate path is
byte-for-byte what it was: 438 lib tests, 77 `gpu_evolve`, 7 WGSL and 6 GPU parity tests pass on
the merged tree, including the three that guard the rate model (`fly_map_parity_*`). That is the
fifth acceptance criterion and it holds.

**Parity exists and runs on this GPU.** `fly_spiking_parity_gpu_matches_cpu_delay_and_readout`
passes on the RX 6750 XT, covering the delay ring and the ± readout on a two-neuron map.

**The honesty is right.** The delivery states plainly that the full-map sugar→MN9 gate and the
≤ 12 ms/tick budget are not measured, and draws no biological conclusion from the smoke runs.
That is the correct posture and it is why this answer is an audit rather than a rebuttal.

## 2. What is NOT checked yet, and why

**Acceptance criterion 2 — the sugar→MN9 gate — cannot be run through the delivered API.** The
criterion is: drive the 20 sugar receptor neurons at 100 Hz for 1 s on the real map and read
`72 ± 10 Hz` at MN9 with `400 ± 100` neurons active (our torch implementation measures 72.4 ± 5.0
and 420, `neuromod-lab/src/fly_gate.py`). The delivered external drive is the retina only:
`EyeMode::{Photoreceptor, Motion}` (`fly_map_spiking.rs:236-242`) converts pixels into Poisson
input for the neurons the map's `pixel_of_neuron` block binds. There is no entry point that
drives an arbitrary neuron set, so the published computation the request names as the pass/fail
criterion cannot be exercised at all.

**Request, small and concrete:** an injection hook that takes a list of neuron indices and a rate
— the same shape `LifBrain::set_poisson` has in the Python reference — reachable from a test.
With that, the gate is a 30-second test on this machine and the kernel earns the criterion
instead of inheriting it. Until then the spiking path is validated as *dynamics* (delay,
refractory, readout on a tiny map) and not as *this connectome's behaviour*.

**Criterion 3 — the tick budget — MEASURED 14/09 18:00, and it is the number that matters most.**
`gpu_evolve --fly-spiking --fly-eye`, 8 beings, 2 000 ticks, `flymap_783_motion_v2.bin`,
`--kernel-profile 250`: the spiking kernel costs **108.5 ms per tick** (114.5 ms tick wall). The
rate path on the same GPU costs 6.8 ms at 8 beings and 7.9 ms at 40 — it is flat in beings after
today's rewrite. So the spiking path is **16× more expensive already at 8 beings**, against a
target of ≤ 12 ms at 40. Full record: `PREDICOES-CEREBRO.md` F7v (prediction registered before
the run).

**The arithmetic points at a specific diagnostic, and it is one log line.** 108.5 ms over 156
substeps is **0.70 ms per substep**. If the active fraction were the ~0.3 % this document's own
cost argument assumes, a substep would touch ~45 000 edges, and 0.70 ms for that is ~64 M
edge-operations per second — two orders of magnitude below what this GPU demonstrably does (the
rate kernel moves all 15.1 M edges in 7.9 ms, ~1.9 G/s). **One of the two premises is therefore
wrong: either the active fraction is far above 0.3 % on this connectome, or the kernel walks more
than the active rows.** Please print, per substep, the number of neurons that spiked and the
number of edges actually traversed (the `--fly-spiking-trace` already collects spikes; what is
missing is the aggregate). Until that number exists, neither of us can say whether the path is
optimisable or whether the event-driven premise does not hold here — and that distinction decides
whether this is a kernel problem or a modelling one.

## 3. One correction to the request itself, from a measurement made after it was written

The request's K2 asked for photoreceptor drive. **Do not use photoreceptors for vision in a
spiking model** — the optic lobe does not propagate there, and this is measured, not argued
(`neuromod-lab/PREDICOES.md` 189v): driving all 7 932 R1-6 cells lights the lamina (L1/L2 reach
26 Hz at 250 Hz input) but the next stage is net inhibitory (Mi1 receives −3 214 synapse·Hz, T5
−1 997, HS −219) and nothing past the first hop ever fires, at any drive level. The reason is
biological: in the animal those cells are graded and do not spike, so inhibition modulates a
tonic release that a threshold model does not have. Driving T4/T5 instead reaches the outputs
(LPLC2 24 Hz, HS 250 Hz, VS 241 Hz, DNp01 95.6 Hz in 100 % of trials).

`EyeMode::Motion` is the right target, and the delivery's own comment is candid that it computes
*motion energy*, not direction selectivity. That matters for what can be claimed from it: T4 and
T5 are direction-selective in four subtypes each (`T4a/b/c/d`), and an unsigned frame difference
does not reproduce that. The map file already carries what a polarity-aware binding would need
(`flymap_783_motion_v2.bin`, all 12 246 T4/T5 bound by angle); what is missing is a per-subtype
preferred-direction table. Not a defect in the delivery — a named next step.

## 4. Where this sits in the programme

The steering reflex the spiking path was built to serve is already measured and reproduces in
the rate model too (`191v`, `193v`): motion on one side swings the HS/VS asymmetry index by
+2.000 (spiking) and +1.74/+1.92 (rate), and the swing vanishes on a degree-preserving shuffled
map. Closing the loop on the track exposed a defect that was mine, not the kernel's: the readout
shipped as a raw group difference, which carries the two groups' standing imbalance into the
wheel, while the measured quantity is a normalised index (`F3v` in
[`PREDICOES-CEREBRO.md`](PREDICOES-CEREBRO.md); fixed by `--fly-readout-norm`). So the spiking
path is not blocked by the reflex work, and the reflex work is not blocked by the spiking path —
they meet at `out_pair`, which both command sources write.

**In order, what would make the spiking delivery land:** (1) the per-substep counters — spiking
neurons and edges traversed — because until that number exists nobody can say whether 108.5 ms is
a kernel problem or a modelling one, and everything else depends on the answer; (2) the injection
hook and the sugar gate; (3) the T4/T5 polarity table, if direction selectivity is wanted rather
than motion energy.

## 5. Context you should have before optimising anything

The programme this kernel serves reached a negative result today, and it changes what "landing"
means. The connectome's steering reflex is real (`191v`, `193v`) but it **cannot drive a car**,
and the reason is measured rather than guessed (`F3v`→`F6v`): the readout swings by +1.92 between
a localised object on the left and on the right — specific to the real wiring, −0.15 on a shuffled
map — but only 0.027 for the diffuse left/right imbalance a bare track offers. The circuit needs a
**discrete object** in the visual field, and the Universe retina renders none: it marches through
the static voxel grid only, so the forty beings racing together are invisible to each other
(`retina.wgsl:63`; the design to fix it is [`RETINA-BEINGS.md`](RETINA-BEINGS.md)).

Two consequences for your side. First, the spiking path is not on the critical path for the
reflex any more — the rate model reproduces the same asymmetry (+1.74/+1.92 against +2.000), so
whatever the spiking kernel buys, it is not that. Second, the thing it could uniquely buy is the
computation the rate model provably cannot represent: the sugar→MN9 pathway, which is 5 synapses
deep with 1.8 ms delays and therefore lives entirely inside one 15.6 ms tick. That is exactly
what criterion 2 tests, which is why the injection hook matters more than it looked.
