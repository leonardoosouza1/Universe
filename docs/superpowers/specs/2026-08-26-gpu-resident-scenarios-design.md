# Design — GPU-resident simulation and fast scenario screening

Owner request (2026-08-26): *"testar vários cenários em minutos... não deveria ter nada
na CPU mais, só GPU, se falta faça kernel"* and *"alocar blocos da GPU e rodar os seres
ali, estados globais, para nada ter que ir para a CPU"*.

Everything below is ordered by **measured** payoff. Where a number is a guess it says so.

> **⚠ Correction, 2026-08-27 — section 4 (Phase 2, open-loop screening) is substantially
> wrong and has a correction block inside it.** Its speed claim ("three orders of
> magnitude per candidate") was 46× when measured; its rejection threshold (0.26) rejects
> brains that drive *better* than the recording champion; the score does not rank
> candidates by quality (Spearman −0.05 over 85 candidates); and **both** of the numbers
> §4 borrows from `docs/DESTILABILIDADE.md` were taken across a gap that was never
> checked — the 0.26/0.18 pair and the "7156 m against 6592 m" claim, the second of which
> that document files under *HISTÓRICO — a versão anterior deste veredito (fiação com
> bug)* (`:140-147`). <!-- check-language: allow --> Evidence, commands and the re-scoped
> proposal: **`docs/SCREEN_CALIBRATION.md`**.
>
> **Scope of what was audited, stated precisely so this banner does not certify more than
> it checked.** Task 6 measured §4 and only §4. §1-3 and §5 were re-read in the review of
> 2026-08-27 for the *specific* failure mode found here — a constant imported from another
> document without a check — and cite no external-document constants; their own numbers
> (the cost model, the ablation table, the 60 ms world swap, PR = 4.68) were measured in
> this repository on 2026-08-26 and carry their own commands. They are **unaudited by
> measurement**, not verified.

## 1. The measured baseline

All figures from 2026-08-26, RX 6750 XT, champion `ereg-m4-m4-s77` gen 43 reloaded,
one generation, evaluation only.

**Cost model, fitted from a scaling sweep at 12/48/192/768 beings:**

```
ms per tick = 0.88 + 0.0333 × n_beings
              ^^^^   ^^^^^^
              fixed  marginal
```

| beings | ms/tick | per generation (20k ticks) |
|---|---|---|
| 32 | 1.95 | 17.6 s waiting + 21.3 s working |
| 864 | 29.65 | 17.6 s waiting + 575 s working |
| 3200 | 107.44 | 17.6 s waiting + 2131 s working |

**Where the marginal cost is NOT.** Ablation at 768 beings, 1500 ticks:

| configuration | ms/tick | share |
|---|---|---|
| complete (SUPERSEDED — see note below) | 33.61 | — |
| `--sem-lento` (no slow brain) | 29.92 | slow brain = 11% |
| `--brain-skip 1000` (no fast brain) | 28.41 | fast brain = 15% |
| both off | 26.53 | **both brains = 21%** |

**Note added during Task 3 (kernel-timing instrument, 2026-08-26):** the `complete`
row's 33.61 ms/tick disagrees with the fitted cost model above (`0.88 + 0.0333 × 768 =
26.45`) and with two later, independent direct measurements at the same 768-being
config — `time` around a 1500-tick evaluation (~27.6-28.1 ms/tick, 4 runs) and a
scaling sweep re-run during Task 3's review round (~27.8-27.9 ms/tick). The `complete`
row was not reproduced; treat 33.61 as wrong rather than as a real regression the
brains introduce, and use the fitted model or a fresh measurement instead.

**79% of the cost is perception and physics.** With no brain at all a being still costs
34.5 µs per tick. Optimising the brain is optimising a fifth of the problem.

**The GPU is not the constraint.** At 768 beings: **24 W of ~200 W**, utilisation 66%
(peak 89%). Utilisation here means "work was queued", not "shaders were busy". A being
tick is roughly 25k MACs for the brain; 33 µs for that is ~0.7 GFLOP/s against a card
that does ~13 TFLOP/s. The work is latency-bound, most plausibly the retina's voxel
raycast (uncoalesced VRAM reads), which is consistent with both the low power and the
time scaling with population.

**The per-tick CPU round trip.** `src/bin/gpu_evolve/context.rs:1302-1303`:

```rust
self.queue.submit(Some(enc.finish()));
self.device.poll(wgpu::Maintain::Wait);   // full pipeline flush, every tick
```

The tick is: GPU perception+brain → CPU reads intents → CPU assembles `phys_states` →
GPU physics → CPU reads back. `BeingState` (`types.rs:13`) lives on the CPU, as do
life/fuel accounting and the four hostile mechanisms added earlier today (tyre thermal,
predator, stations, sensor tax).

**Honest bound on removing it:** the flush is the `0.88 ms` fixed term. It is half the
time at 32 beings and **3% at 864**. Eliminating CPU involvement is necessary to batch
ticks, but on its own it does not deliver "minutes" at matrix scale. The upside beyond
that is real but **unmeasured**: 24 W with 66% queued occupancy suggests bubbles between
dispatches, and those inflate the marginal term too. Phase 0 settles it.

**No per-kernel timing exists.** `rocprofv3` produced an empty trace: wgpu runs on
Vulkan/RADV and rocprof instruments HIP/HSA. The wgpu timestamp path is unused —
`timestamp_writes: None` at `core.rs:503`, `cortex_sheet.rs:521`, `retina_digest.rs:249`.
Every optimisation proposed without this instrument is a guess, including the one above.

## 2. Phase 0 — the instrument (blocks Phase 1, not Phase 2)

Enable `wgpu::Features::TIMESTAMP_QUERY` and populate the `timestamp_writes` fields that
already exist on every compute pass. Emit, once per N ticks, a table of milliseconds per
kernel and the gaps between them.

Deliverable: for a given `--seres`, a breakdown of the 33.6 ms/tick across
retina / perception / brain / slow brain / physics_a,b,c / **idle gaps**.

Acceptance: the sum of the parts plus gaps reconstructs the wall-clock ms/tick within
10%. If it does not, the instrument is wrong and nothing built on it can be trusted.

This is small — the fields are present and set to `None` — and it converts every later
decision from opinion into measurement. It also answers, at no extra cost, whether the
CPU-side hostile mechanisms are expensive when switched on, since the ablation above ran
with them off.

### Ordering note

Phase 0 and Phase 1 make the engine faster. **Phase 2 makes the question cheaper, and it
depends on neither.** It is the phase that actually delivers "many scenarios in minutes",
because it removes the physics from the loop instead of accelerating it. Phases 0 and 2
can therefore start in either order or together; Phase 1 needs Phase 0 first, and Phase 3
wants whatever speed exists by then.

If only one phase is ever built, it should be Phase 2.

## 3. Phase 1 — GPU-resident state and multi-tick dispatch

The owner's architecture, adopted: all per-being state lives in GPU buffers; the CPU
issues work and does not participate in the tick.

**State moves to the GPU.** `BeingState`'s CPU fields (`prev_steer`, `stagnation_ticks`,
`wall_ticks`, `joy_tick`, `joy_streak`, `edge_row6/7`) join the existing physics buffer.
Life and fuel accounting become a kernel. The four hostile mechanisms are ported to WGSL
— this is the "se falta faça kernel" instruction, and Phase 0 says whether it is urgent
or merely tidy.

**K ticks per submit.** With no CPU dependency inside the tick, the encoder chains K
ticks before a single `submit` + `poll`. The fixed term falls by ~K. K is bounded by how
often telemetry has to be observed, not by the GPU: any per-tick readback forces K = 1,
which is why the telemetry constraint below is part of this phase and not an afterthought.

**Then attack whatever Phase 0 names.** The prior is the retina, and `retina_digest.wgsl`
plus its parity test suggest that path is already partly cut. The target is chosen by the
measurement, not here.

**Constraints that do not bend:** telemetry that today reads back per tick either moves
to a GPU-side accumulator or becomes opt-in and off by default. `--observar` stays
available for analysis runs; it simply forces K = 1.

**Acceptance:** with the feature off, `obs.f32` is byte-identical to the current binary
(same sha256). This check has caught three real problems in this repository today; it is
the gate, not a formality.

## 4. Phase 2 — open-loop screening

This is what actually delivers "many scenarios in minutes", and it does not need the
engine to be fast.

Distilling a controller from a recorded champion is already proven here: a closed-form
ridge pilot fitted to the champion's own replay drove ~~**7156 m against the champion's
6592 m**~~ (`docs/DESTILABILIDADE.md`). [**Superseded — see the correction below; the
premise survives, this number does not.**] The same document measures where a clone stops
working: per-tick error **0.26 compounds** in closed loop and the clone stalls, **0.18
does not** and it drives.

That threshold is a calibrated rejection filter. Scoring a candidate brain against
recorded states is a forward pass with no physics: ~500 MFLOP over 20k states, versus
~18 s for one closed-loop evaluation. **Three orders of magnitude per candidate.**

**Two limits, stated up front because they bound what this can ever do:**

* **Necessary, not sufficient.** Above 0.26 the candidate is rejected with confidence.
  Passing does not prove it drives — 0.26 to 0.18 is a grey band. The funnel saves time
  on rejects, not on survivors.
* **Blind to genuine novelty.** The screen only recognises behaviour resembling the
  champion that produced the recording. A brain that is different *and better* scores as
  divergent and is rejected. That directly limits the later goals (different brain
  shapes, self-modifying weights): the funnel accelerates search near what exists and
  cannot see what is new. Any use of it for novel architectures needs a separate,
  closed-loop arm.

> ### ⚠ Correction, 2026-08-27 — the SPEED, the THRESHOLD and BOTH borrowed constants are wrong
>
> Measured in Task 6 against closed-loop ground truth, 85 candidates, one `--gens 1`
> evaluation each. Full record and commands: **`docs/SCREEN_CALIBRATION.md`**. The
> original text is left in place because it is the only evidence of how the error was
> possible; every claim it makes that the measurement contradicts is listed here.
>
> **1. "Three orders of magnitude per candidate" — no. Measured 46×, best case.**
> The `~18 s` closed-loop figure was never measured; it is 0.97-1.06 s per genome
> (`/usr/bin/time` around 48 beings × 20 000 ticks: 46.37 s and 51.07 s, two repeats).
> The screen costs 208.6 ms/genome at the full window with `--batch 48`, and 21.8 ms at
> the 2 000-tick window this calibration recommends — **4.8× and 46× cheaper**
> respectively. Unbatched (`--batch 1`) it is 8 404 ms/genome, i.e. **8.4× MORE expensive
> than simulating the whole world**. The FLOP count is what produced the wrong estimate:
> both sides are launch-latency bound, so arithmetic was the one thing neither side was
> limited by.
>
> **2. "Above 0.26 the candidate is rejected with confidence" — no. 6 of 7 rejects drive.**
> Of 85 candidates, the screen rejects 7. Six of them drive at ≥75% of the recording
> champion and two drive *better* than it: `noi050_s2` (score 0.4287, 123% of champion)
> and `gen30` (score 0.2618, **162%**, 41/48 beings completing a lap against the
> champion's 22). `gen30` is not a synthetic perturbation — it is
> `ereg-m4-m4-s77/evolve_best_gen0030.bin`, a champion this lineage's own selection
> produced. Only `noi200_s1` (a brain destroyed by σ = 2.0 noise, 71 m) is a true reject.
> **The 0.26 threshold is unsafe for pruning and must not be used to discard candidates.**
> It was transplanted from `docs/DESTILABILIDADE.md`, where it was measured for a
> *distilled pilot* replacing the brain — not for genomes of the same architecture — and
> the transplant was never checked until now.
>
> **3. The screen does not rank.** Spearman of the score against closed-loop driving
> distance is **−0.05** over the 85 candidates. The sign is the one a rejection filter
> wants (more divergent ↔ drives worse); the magnitude is the problem, and it is near zero
> because two subgroups cancel: ρ = **−0.618** among the 27 candidates that drive worse
> than the reference, ρ = **+0.382**, *inverted*, among the 56 that drive better. A
> distance from one point cannot tell the two directions apart, and most of this
> population sits in the half where the screen is backwards. It is not monotone in genome
> distance either (13.5% of strictly-ordered pairs inverted; three perturbation directions
> of identical magnitude score 3.2× apart).
>
> **4. The "7156 m against 6592 m" that justifies Phase 2 existing at all is a
> superseded number from a buggy run.** `docs/DESTILABILIDADE.md:140-147` files it under
> *"HISTÓRICO — a versão anterior deste veredito (fiação com bug)"*
> <!-- check-language: allow --> — the pilot's override never reached the motor, so 7156 m
> was **the champion itself** running with one contaminated memory channel (ch120), not a
> distilled controller driving. The corrected figure for the linear ridge pilot the
> sentence describes is **1027 m, 15.6% of the champion** (`:350`).
>
> The *premise* of Phase 2 nevertheless survives, and it is worth being exact about what
> was salvaged: the same corrected table shows a **non-linear** closed-form pilot (pyramid
> basis + arctanh, still solved in closed form) at **7485 m, 113.5%** on the training
> track and a mixed-notebook version at **102.4% on a track it had never seen**. So
> "distilling a controller from a recorded champion is proven here" is true — for the v2
> non-linear pilot, at different numbers, and not for the linear ridge the sentence names.
> Two constants were imported from that document into this section and neither was checked
> against its own corrections; Task 6 audited the 0.26/0.18 pair and this one was found in
> the review of that work.
>
> **5. "Blind to genuine novelty" was right, and is worse than stated.** The bullet
> treats this as a limit on *novel architectures*. It is not: `gen30` is the same
> architecture, from the same lineage, twelve generations from the recording genome, and
> it is rejected. The blindness bites inside the existing search, not only outside it.
>
> **What survives.** The screen is bit-exact (the recording genome scores 0.0000 at all
> 20 000 ticks) and it reliably separates a *destroyed* brain from a functioning one:
> 0.9066 for `noi200_s1` against ≤0.4287 for everything that could still drive. Phase 2
> is therefore re-scoped from a rejection funnel to a **tripwire at ≈0.60**, at 21.8 ms
> per candidate — with the caveat that 0.60 is itself n = 1 per side (one destroyed brain,
> one highest survivor) and is a hypothesis to be calibrated, not a number to start pruning
> with. Anything more than that needs a reference that is not one champion — see
> `docs/SCREEN_CALIBRATION.md` §6.

## 5. Phase 3 — the world matrix (sketch)

Approved earlier in this conversation and unchanged in intent; it rides on whatever speed
Phases 0-2 produce.

Unit is the **cell**: `{world_seed, own population, own elite, own telemetry}`. Selection
never crosses cells. Generations are grouped by world so cells sharing a world enter one
dispatch; a world swap costs **60 ms**, measured (7.11 s vs 5.37 s over 29 extra swaps),
which is 0.5% of a multi-hour run.

Per cell: participation ratio globally and per region block (visão / parede / corpo /
associativo — four blocks of 24, `main.rs:2134`), using `brain_capacity::participation_ratio`.
Today's baseline is **PR = 4.68 of 96**: the current world occupies 5% of the brain.

Replication is not optional. The same contrast measured today gave −810 m at n=1 and
−481 m at n=9. Recommended shape: 3 worlds × 3 population variants × 3 replicates.

Cross-evaluation grid `T[i][j]` = champion of cell i evaluated in world j, normalised
**per column** (worlds have laps of 1639/2377/2928 m; raw metres compare track length).
The scientific content is the interaction term: if `T` is rank-1, `T[i][j] ≈ a_i · b_j`,
nothing specialised — there are merely good champions and easy worlds. Measured by the
share of variance in the first singular value.

## 6. What this design deliberately does not do

* **No new brain architectures in these phases.** A different brain shape changes the
  genome layout, the buffer sizes and the mutation operator. Mixing it in is what turns
  weeks into months. It gets its own spec once the matrix runs.
* **No self-modifying weights yet.** It is only *evaluable* once the matrix can show what
  an expensive geometry looks like. Without that there is nothing to compare against.
* **No persistent-kernel rewrite.** K-ticks-per-submit gets most of the benefit of a
  fully persistent kernel without fighting wgpu's execution model.

## 7. Verification rules for every phase

1. Feature off ⇒ byte-identical `obs.f32` (sha256), against a binary built before the
   change.
2. Determinism: same configuration twice ⇒ identical output.
3. Any performance claim carries the command that produced it and the hardware state
   (power and utilisation), because 66% "utilisation" at 24 W already fooled this
   analysis once.
4. No GPU training run without the owner's explicit release.
