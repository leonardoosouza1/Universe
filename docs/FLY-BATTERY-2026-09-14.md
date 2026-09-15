# Fly control battery, 2026-09-14

## Registered before execution

This is an exploratory correction of the earlier F8 experiment. The previous
same-start condition overlapped bodies and its reported mortality meant less than
10 m of track progress. It did not measure physical death. The current retinal
ground lookup also uses floor rather than the previous round convention; earlier
scores are not a controlled comparison with this executable.

Run three paired population seeds (31, 32, 33), one generation, 40 beings, 15,000
ticks, track 8005, width 28. Each seed runs real map/positive reflex, real map/negative
reflex, and shuffled map/positive reflex. All use the validated starting formation,
visible bodies, normalized reflex readout, alpha 0.02, gain 20, cruise 0.6. Freeze
one executable and hash both maps. Execute sequentially. Stop for process failure
or nonfinite measurements; do not stop or add seeds based on favorable scores.

Primary descriptive endpoint: the entire population's signed track progress,
summarized by median within each run and paired seed differences. Secondary:
distance, life-budget exhaustion, fraction below 10 m, coverage by visible bodies,
left/right asymmetry, raw and applied steering, nearest-body distance and sampled
overlaps. Life exhaustion at the fixed horizon is not premature death. CPU/GPU
timings include sampled diagnostics and are not clean kernel microbenchmarks.

Exclude initialization/warmup episodes and ticks below 64 from telemetry summaries.
Report coverage including zero-coverage frames. Coverage outside 0.5–15% is a
diagnostic flag, not an exclusion chosen after seeing behavior. Report correlations
only with nonzero variance and identify repeated observations as dependent. The
independent replication unit is the paired population seed (n=3), not 120 beings.
Use descriptive ranges; three seeds do not justify a strong significance claim.

A favorable progress score alone cannot establish collision avoidance. The sampled
1.55 m center-distance threshold is a retinal body-overlap proxy, not a measured
collision impulse, and sampling every 64 ticks can miss events. A retinal-off arm
and multiple shuffled maps are necessary follow-ups before claiming connectome
specificity. No learning or evolutionary improvement can be inferred from one
generation. The native brain and downstream motor transforms remain part of the
legacy hybrid controller in this battery.

The companion visual demonstration uses actual spiking state, a selected being and
sampled frames. It is a visualization validation, not evidence that the spiking
controller solves driving or reproduces a biological fly.

## Diagnostic follow-up registered during the primary battery

The first two completed seed-31 arms show approximately 96.6% saturation of the
adapted steering input at gain 20. After all nine primary runs finish, repeat the
real/positive arm with gain 1 for all three seeds. Reuse the exact frozen primary
executable and all other parameters. This is a post-observation diagnostic, not
an independent confirmation. Compare saturation, progress and sampled overlaps
against the corresponding gain-20 runs. Complete all three even if progress
worsens. Do not promote the gain to a new default from this small battery alone.

## Validation finding after the primary battery

The full validator exposed a second sensor-frame problem: diagonal whisker
channels 94–97 used `PosePerception.phase`, which the headless runtime sets to
zero. The sensors therefore stayed world-aligned while the vehicle turned. The
kernel now rotates the resident forward vector instead. The validator's CPU
reference also had the wrong channel order (`-60,-30,+30,+60` instead of the
documented `+30,-30,+60,-60`). A separate physical wall fixture checks that a body
turn moves the same obstacle between the expected channels.

Both the primary and gain follow-up batteries intentionally use the frozen
pre-fix executable. Their paired comparisons retain that shared defect; they do
not establish performance of the corrected sensor frame. The direct-mode visual
demonstration and final kernel validation use the corrected executable.
