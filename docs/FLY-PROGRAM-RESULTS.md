# The fly-connectome programme — what was measured, 14/09/2026

One day, nine registered predictions, every number reproduced from raw data. This is the synthesis;
the per-experiment records with their predictions-before-the-run are in
[`PREDICOES-CEREBRO.md`](PREDICOES-CEREBRO.md) (entries F1–F7) and `neuromod-lab/PREDICOES.md`
(186–193). The chronological log is `neuromod-lab/docs/48-relatorio-2026-09-08.md`.

## The question

Leonardo, 14/09 08:48: *use the fly's map, do not generate anything on it — start from the idea of
selecting, then the rest.* The material is the FlyWire v783 connectome: **138 639 neurons,
15 091 983 signed connections, 54.5 M synapses, 40 % inhibitory edges**, as packaged by Shiu et
al. 2024. The programme asks two things in order: does that wiring carry computation, and can we
use it.

## 1. The wiring carries computation — and it is the targets AND the weights

The published gate first: 20 sugar receptor neurons driven at 100 Hz for one second make the
proboscis motor neuron MN9 fire at **72.4 ± 5.0 Hz** with **420 neurons active**, reproducing the
paper (186v). Then the same stimulus on three controls that each preserve a different statistic:

| map | MN9 | trials firing | active | spikes |
|---|---|---|---|---|
| real | **72.4 ± 5.0 Hz** | 30/30 | 420 | 340 k |
| shuffled targets (degrees and signs preserved) | **0.0 Hz** | 0/30 | 175 | 76 k |
| shuffled weights (graph and signs preserved) | **0.0 Hz** | 0/30 | 96 | 98 k |
| shuffled signs (graph and weights preserved) | 29.4 ± 24.2 Hz | 28/30 | **26 022** | **14.0 M** |

Permuting the targets erases the pathway. Permuting only the synapse *counts* also erases it —
7 % of the edges carry 43 % of the synapses, so the pathway is a narrow ridge of heavy edges and
redistributing magnitudes dilutes exactly that ridge. Permuting the signs does not erase it but
replaces it with a seizure: 62× more neurons active. **The connectome is not a generic reservoir**
(190v). That result is what licensed everything after it, and it also sets a warning for anyone
approximating the map: pruning, quantisation that flattens the weight tail, or a threshold on edge
weight can break the model in ways a microbenchmark will not show.

## 2. Two things that do not work, each with its mechanism

**Selection over the map, open loop, is indistinguishable from a random graph** (187, 188). Four
arms of 15 generations — bindings only, bindings plus pruning, bindings plus a fixed random mask,
and pruning on a shuffled map — land at 0.443 to 0.485 with the shuffled arm at the top. The
pruning operator that beats free mutation by a wide margin on a 44-parameter taxi policy (192v,
+7.8 held-out deliveries, p = 1.2e-6) adds **nothing** when it masks 8 866 cell types of a fixed
graph. The operator earns its keep when the mask chooses among few interpretable inputs, not when
it switches off coarse blocks of a large map.

**The optic lobe does not propagate in a spiking model** (189v). Driving all 7 932 photoreceptors
lights the lamina (L1/L2 reach 26 Hz at 250 Hz input) but the next stage is net inhibitory —
Mi1 receives −3 214 synapse·Hz, T5 −1 997, HS −219 — and nothing past the first hop ever fires, at
any drive level; raising the drive only deepens the inhibition. The reason is biological: in the
animal, photoreceptors and lamina/medulla cells are **graded, they do not spike**, so inhibition
modulates a tonic release that a threshold model does not have. Driving the motion detectors
(T4/T5) instead reaches the outputs: LPLC2 24 Hz, HS 250 Hz, VS 241 Hz, **DNp01 (the giant fiber)
95.6 Hz in 100 % of trials**. Consequence taken: the eye binds to T4/T5, and the rate model — which
represents graded stages correctly — is the right model for the optic lobe.

## 3. The reflex is real, and it still cannot drive a car

Driving 2 000 motion cells on one side of the visual field swings the HS/VS asymmetry index from
−1.000 to +1.000 (**swing +2.000**) in the spiking model and **+1.74/+1.92** in the rate model the
Universe kernel runs; across ~1 200 descending neurons the swing is +0.88. On a shuffled map the
swing vanishes and the rates collapse from 28.9 Hz to 0.1 Hz (191v, 193v). A steering reflex,
read rather than evolved.

On the track it fails, and the failure took three arms to explain properly:

| | steering bias | median metres | dead |
|---|---|---|---|
| F3 raw difference | −0.295 | 25 (shuffled 99) | 40 % |
| F4 normalised index | −0.249 | 21 (shuffled 48) | 38 % |
| F5 high-pass + gain | **−0.008** | 13 (shuffled 93) | 48 % |

F3's bias was my own design error: the laboratory measured a *normalised asymmetry index* and the
simulator shipped a *raw difference*, which carries the two groups' standing imbalance straight to
the wheel. F4 showed normalisation removes scale, not imbalance. F5's high-pass removed the bias
completely — and the command still had **zero correlation with the scene** (median 0.003, largest
0.069 across 20 beings). Geometry was verified perfect first: 100 % of left-side cells bound to
the left half of the retina, correlation 1.000 between pixel column and azimuth.

The transfer curve says why. Sweeping the left/right drive ratio, the HS+VS index runs from +0.890
to +0.072 — but around balance it is nearly flat: **a 20 % imbalance moves it 0.027 against 0.817
of full excursion**, and at exact balance it sits at +0.133 rather than 0. The brake channel
confirms it from another direction: it correlates **+0.33 with the distance to the wall ahead**
(the opposite of an approach detector) and the shuffled map produces the same numbers, so that
channel is stimulus statistics, not wiring.

## 4. The refinement that matters — and the simulator gap it exposed

The conclusion is not "the circuit needs large contrast" (F5v) but something sharper. With a
**localised** disc of motion cells, the LPLC2 asymmetry runs from −0.971 (object left) to +0.951
(object right) at the same radius: **swing +1.922 on the real map against −0.149 on the shuffled
one** (F6v). Seventy times the diffuse response, and specific to the real wiring. Pure expansion,
by contrast, is *not* detected: the 37× gain with radius appears equally on the shuffled map,
because growing a disc grows the number of driven cells and any graph responds to that.

So the circuit localises discrete objects. And the track offers none — which is a fact about the
simulator, verified in code: the retina marches a ray through the static voxel grid only
(`retina.wgsl:63`), and the per-being poses set each ray's origin without ever occluding it. **The
forty beings racing the same track cannot see each other.** That predates this work and affects
every brain in the project, not one experiment. The design to close it is
[`RETINA-BEINGS.md`](RETINA-BEINGS.md); the experiment it unlocks is F3 repeated on a populated
track, with the shuffled map alongside as it has been throughout.

## 5. Cost, for whoever plans the next battery

| path | 8 beings | 40 beings |
|---|---|---|
| rate model (`fly_step`, after the lanes-are-beings rewrite) | 6.8 ms/tick | **7.9 ms/tick** (flat in beings) |
| spiking model (`--fly-spiking`, 156 substeps) | **108.5 ms/tick** | not measured |

The rate kernel went from 98 ms to 7.9 ms at 40 beings by making lanes beings so the gathers
coalesce, with byte-identical output. The spiking path is 16× more expensive at 8 beings against a
target of 12 ms at 40, and its own arithmetic is inconsistent: 0.70 ms per substep implies ~64 M
edge-operations per second where the rate kernel demonstrates ~1.9 G/s, so either the active
fraction is far above the assumed 0.3 % or the kernel walks more than the active rows. That
diagnostic is the first item in [`RESPOSTA-KERNELS-ASTRA.md`](RESPOSTA-KERNELS-ASTRA.md).

## 6. What would change the verdict

Nothing here says the connectome is useless for control; it says a bare oval track is the wrong
test bed for a circuit that evolved to detect discrete objects. Three things would move it:

1. **Discrete objects in the retina** — the beings, or an obstacle. The prediction is pre-registered
   in F6v and the measurement that decides it is the same one F5 failed: correlation between the
   steering command and the scene, with the shuffled map as the control.
2. **The sugar gate on the spiking path** — the one computation the rate model provably cannot
   represent, because it is five synapses deep with 1.8 ms delays and lives inside a single tick.
   Blocked on an injection hook (audit item 2).
3. **Direction selectivity** — `EyeMode::Motion` computes motion energy, not direction; T4 and T5
   are direction-selective in four subtypes each, and the map file already binds all 12 246 of them
   by angle. What is missing is a per-subtype preferred-direction table.

## 7. Method note

Every experiment above registered its prediction, its ruler and its refutation criterion *before*
running, and several of them refuted me: the pruning control (I predicted the advantage would
vanish; it doubled), F3 (I predicted the map would beat the shuffle; the shuffle won 4×), F6 (I
predicted expansion detection; got object localisation instead). Three defects were found by tests
that existed only because the discipline required them — a brightness mapping that made the retina
blind to the road, a kernel that did not truncate like the CPU cast it mirrors, and a test file
whose `__main__` block sat in the middle so half its tests never ran. The negative results here
cost hours; without the pre-registration they would have cost days and produced a confident wrong
answer.
