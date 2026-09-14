# Rendering beings into the retina (`--retina-beings`)

Status: design, 14/09/2026. Not implemented. Written after the measurement that made it the
next step (`docs/PREDICOES-CEREBRO.md` F5v/F6v).

## 1. Why — the measurement, not a wish

Forty beings race the same track and **none of them can see the others**. Verified in code:
the retina kernel marches a ray through the static voxel grid and stops at Wall or Hole
(`src/gpu/retina.wgsl:63` samples `grid[y*n + x]`, the DDA loop at `:127-136`); the per-being
poses (`buf_poses`) only set each ray's ORIGIN, never occlude it. The CPU reference has the
same shape (`World::render_binocular_retina`, `src/world.rs:966`).

That is a hole in the simulator that predates the fly work and affects every brain, not one
experiment: a population racing shoulder to shoulder is, perceptually, forty solitary cars.

The fly work is what measured the cost of it. The connectome's steering readout swings by
**+1.92** between a localised object on the left and the same object on the right, and that
swing is specific to the real wiring (**−0.15** on a degree-preserving shuffled map). But the
same readout moves only **0.027** for the kind of diffuse left/right imbalance a bare track
offers, which is why the reflex could not steer a car (F5v: bias removed, gain applied, and
still zero correlation with the scene). The circuit needs a discrete object in the visual
field. The track has none — and the most natural discrete objects, the other beings, are
already present in the physics and simply absent from perception.

## 2. What to build

**A fifth material.** `VoxelType` today is `Grass = 0, Road = 1, Wall = 2, Hole = 3`
(`src/world.rs:30-35`). Add `Being = 4` as a *material code produced by the retina*, not a
voxel the world stores — nothing writes it into the grid. Two places must learn it explicitly:

- `legacy_encoding` (`src/gpu/mod.rs:150`) — today its `_ =>` arm catches everything unknown
  and encodes it as Hole, so a new code would be silently mis-encoded rather than rejected.
  Give `Being` its own branch and pick a value that is distinguishable from all four existing
  materials at the same distance (the four current ranges are measured in
  `FLY-MAP-BRAIN.md` "Eye binding"; a near Wall is +127 and Road is around −128).
- the WGSL mirror of that function inside `fly_map.wgsl::eye_pixel_drive`, which is the
  duplicated-implementation risk the parity test already guards (`fly_map_parity_with_the_eye_on`).

**The occlusion test.** After the DDA produces `hit_dist`, intersect the ray with every OTHER
being's body, treated as a vertical cylinder of radius `BEING_RADIUS` (one constant, defined
once, in metres — read the car's actual half-width from the physics rather than inventing a
number). If any intersection is nearer than `hit_dist`, it wins: `hit_dist` becomes that
distance and the material becomes `Being`. Ray-circle in 2D (x, z) is the whole test; the eye
is at 0.4 m and the bodies are low, so a height test can be skipped in v1 — say so in a
comment rather than leaving it implicit.

Positions come from `poses[]`, which the kernel already binds — **no new buffer**. Skip
`being == self` by index.

**Cost.** 1 764 rays × 40 beings = 70 560 rays, each testing 39 circles: ~2.8 M ray-circle
tests per tick, all register math, against a DDA that already does up to 400 grid steps per
ray. Expect it to disappear into the noise; measure it with `--kernel-profile` rather than
assuming, and report the retina kernel's before/after milliseconds.

**The flag.** `--retina-beings`, **off by default**, because this changes the input
distribution of every existing champion (`CLAUDE.md` rule 3: new perception enters behind a
flag with a paired A/B, and an old champion needs the A/B, not faith).

## 3. How it must be proven

1. **CPU and GPU agree.** `World::render_binocular_retina` gains the same test, and
   `gpu_validate --retina` must pass with beings on: the existing tolerance is i8 diff ≤ 1
   (`gpu_validate.rs:17-19`). The CPU path is the reference; the kernel is what has to match it.
2. **Off is bit-identical.** With the flag off, every retina pixel, every downstream channel
   and `gpu_validate`'s existing numbers are unchanged — the same guarantee the fly blocks
   carry, and the reason the flag exists.
3. **The geometry is right, not just plausible.** A unit test with two beings placed at known
   positions: one directly ahead at 10 m must appear in the central columns with distance
   10 ± (cell size), one at 90° must not appear at all inside a 120° field of view, and a being
   BEHIND a wall must stay invisible (the DDA hit must win when it is nearer).
4. **A 2-second smoke before any battery**, per the repo's rule, printing the fraction of
   retina pixels whose material is `Being` in a populated generation. If that fraction is zero,
   the feature is not doing anything and no battery should follow.

## 4. The experiment it unlocks (prediction goes in `PREDICOES-CEREBRO.md` before running)

Repeat F3 on a populated track with `--retina-beings`: the fly reflex bound to the wheel, real
map against the degree-preserving shuffled map, same seeds. The measurement that decides it is
the one F5v used and failed: **the correlation between the steering command and the scene** —
now with a discrete object present. F6v predicts it should be measurable where it was 0.069
before; the shuffled arm is what says whether any of it is the wiring.

Second, cheaper use, independent of the fly: the population's own brains get a sense they never
had. Whether that changes anything is its own paired A/B, and it is the kind of change that
should be measured on the existing champion before anyone claims it helps.
