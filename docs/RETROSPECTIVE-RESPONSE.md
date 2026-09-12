# Response to "Retrospectiva 08–12/09" (docs/RETROSPECTIVA-NEUROMOD-08-12.md)

Fable, 12 Sep 15:50. The retrospective was written before `u3`, `u10b` and `u11` existed; this note maps its six
points onto what has been measured since, with the handbook's labels.

| retrospective | status now | evidence |
|---|---|---|
| §4.1 the hip arm carried mutational load the control did not pay (p 1.0 / σ 0.2) | **agreed, fixed and measured** | `u10b` (p 0.3 / σ 0.1): completion regression gone in 3/3 replicas — verified, PREDICOES U10b |
| §4.2 / §5 U1 was a non-measurement (effect +0.16 % vs MDE 2.3 %) | **agreed; the ruler is now 0.45 %** | `u10b` paired sd fell 5× (0.0070 → 0.0014) once the hip arm stopped carrying noise: Δ +0.12 %, MDE 0.45 %, still null — verified |
| §5 the track world has no "prize floor" for `lap2/lap1` and `≥2V` | **agreed, and the floor of the primary metric is already known** | r = t2/t1 = 1.00 ± 0.01 for the CONTROL in every generation of u1v/u10b: the second pass is not faster for anyone, so the prize available to memory on track 8005 at 9 m/s is ~0 — the "observable world" finding, PREDICOES U10b/U11. For lap counts the floor of a do-nothing policy is 0 laps vs 3 for the champion (trivially above noise); a fixed-command floor (straight, fixed steer) is registered as the next cheap check (U12 below) |
| §6.2 declare the MDE before the full U1 | **adopted as a rule** | every new row in PREDICOES carries the ruler as a pre-specified effect size; the MDE is reported with the result, not used as a threshold (external review, 12/09) |
| §6.3 run the 60-generation U1 only after 1 and 2 | **agreed, and gated further** | `u11` (running): a world only qualifies for a memory arm if the control's second pass can be faster (r ≤ 0.97); otherwise the 60-generation battery would measure nothing whatever the brain does |
| §6.5 identity gates for Rust and HIP | open | Rust gap 9–12 % noted in the lab diary; the taxi-world crate is not in this session's scope |

## U12 — prize floor of the track world (registered, not run)

Three trivial policies evaluated with the same judge as the champion, one generation, 40 copies, track 8005,
30 000 ticks: (a) all-zero genome (born-neutral commands), (b) constant throttle with zero steer, (c) constant
throttle with a fixed steer offset. Prediction: (a) and (c) complete 0 laps, (b) < 0.5 laps median; the champion's 3
laps sit far above the floor, so `≥2V` and median laps are legitimate rewards on 8005. If any trivial policy completes
≥ 1 lap, the lap metrics on this track are inflated by the geometry and the whole U-series needs a harder track.
Requires a way to evaluate a hand-made genome through `gpu_evolve` (a `--genome-file` flag or a written checkpoint);
not available today — CPU work after the rename merge.
