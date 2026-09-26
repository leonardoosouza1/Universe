#!/usr/bin/env python3
"""Decompose the school SCORE formula for a cross-city learning run, from
recorded files only, to explain how selection/fitness can rise while task
progress falls, and to surface degenerate strategies (parking, wall-hugging,
service-request spam) that the score formula does not currently penalize.

This is host-side file aggregation only: it reads JSON/CSV artifacts already
written to disk by a finished `gpu_evolve --school delivery` run. It does not
touch the GPU, does not run any simulation, and does not change any reward
weight. Item 4 of its output is a read-only PROPOSAL, not an implemented
change.

Score formula reference (read-only, quoted for traceability):
  src/gpu/school.wgsl:107-124 (service_v2 branch, lesson >= delivery):
    school[k+SCORE] = school[k+PROGRESS] + 1000.0*accepted_delivery
                     + 200.0*(school[k+SERVICE_READINESS] + completed_pickup)
                     + 500.0*success
                     - 200.0*failed - 100.0*(1.0-integrity) - 100.0*(1.0-alive)
                     - seconds - physical_penalty
                     - 5.0*school[k+ENGINE_OFF_EVENTS] - school[k+STARTER_PAID_J]/1500.0
  where (school.wgsl:110-120):
    seconds = budget*DT if not success else school[k+ELAPSED]*DT   (DT = 1/64)
    physical_penalty = min(school[k+COLLISION_J]/1000.0, 200.0)
    accepted_delivery = success * clamp(school[k+DELIVERY_DELTA], 0.0, 1.0)
    completed_pickup = 1.0 if (lesson>=7.5 and accepted_pickup_tick set) else 0.0
  SERVICE_READINESS is a one-time ratchet (school.wgsl:227-234):
    proximity = clamp(1.0-max(goal_distance-area_radius,0.0)/15.0, 0.0, 1.0)
    slowing = clamp(1.0-speed/max(v_min,0.001), 0.0, 1.0)
    school[k+SERVICE_READINESS] = max(prev, proximity*slowing)   (only once PROPELLED>=propelled_min)
  Selection: src/bin/gpu_evolve/school_batch.rs:79-119 -- with `--school-select worst`
  and starts_per_genome > 1, the per-genome selection score is the MINIMUM
  R_SCORE among that genome's starts (worst_score), not the mean:
    SchoolSelection::Worst => worst_score
  fail_code_name mapping: src/gpu/school.rs:175-198 (18 = blocked_without_progress,
  16 = open_at_deadline, 2 = death).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DT = 1.0 / 64.0
REVISIT_RADIUS_M = 5.0


# ---------------------------------------------------------------------------
# Command line parsing helpers
# ---------------------------------------------------------------------------


def parse_argv_json(path: Path) -> dict:
    """Parse a gpu_evolve command.json (a flat argv array) into a flag->value
    dict. A flag followed by another "--flag" or by the end of the array is
    treated as a boolean switch."""
    tokens = json.loads(path.read_text())
    args: dict[str, object] = {}
    i = 1  # token 0 is the binary path
    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str) and tok.startswith("--"):
            key = tok[2:]
            has_value = i + 1 < len(tokens) and not (
                isinstance(tokens[i + 1], str) and tokens[i + 1].startswith("--")
            )
            if has_value:
                args[key] = tokens[i + 1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1
    return args


# ---------------------------------------------------------------------------
# Score decomposition
# ---------------------------------------------------------------------------


def score_components(being: dict, budget_ticks: float) -> tuple[dict, float, float]:
    """Reconstruct the school.wgsl SCORE for one being's recorded outcome,
    split into named components. Returns (components, reconstructed_score,
    residual = recorded_score - reconstructed_score).

    Known, documented residual source (not a bug in this script): the school
    record freezes PROGRESS/DELIVERY_DELTA/etc. at the tick close() fires, but
    the exported "alive"/"integrity" fields can reflect a LATER (end of the
    generation) body state for a being that died after its school outcome was
    already judged. See the README note this script prints when residual is
    large for a specific being.
    """
    status = being.get("status")
    success = 1.0 if being.get("success") else 0.0
    failed = 1.0 if status == "failed" else 0.0
    alive = 1.0 if being.get("alive") else 0.0
    integrity = float(being.get("integrity", 1.0))
    progress = float(being.get("progress", 0.0))
    service_readiness = float(being.get("service_readiness", 0.0))
    lesson = float(being.get("lesson", 0.0))
    accepted_pickup_tick = being.get("accepted_pickup_tick")
    completed_pickup = 1.0 if (lesson >= 7.5 and accepted_pickup_tick is not None) else 0.0
    delivery_delta = float(being.get("delivery_delta", 0.0))
    accepted_delivery = success * max(0.0, min(1.0, delivery_delta))
    judged_ticks = float(being.get("judged_ticks", 0.0))
    seconds = (budget_ticks * DT) if not success else judged_ticks * DT
    collision_j = float(being.get("collision_energy_j", 0.0))
    physical_penalty = min(collision_j / 1000.0, 200.0)
    engine_off = float(being.get("engine_off_events", 0.0))
    starter_paid = float(being.get("starter_paid_j", 0.0))

    components = {
        "progress": progress,
        "delivery_bonus": 1000.0 * accepted_delivery,
        "readiness_bonus": 200.0 * (service_readiness + completed_pickup),
        "success_bonus": 500.0 * success,
        "failed_penalty": -200.0 * failed,
        "integrity_penalty": -100.0 * (1.0 - integrity),
        "alive_penalty": -100.0 * (1.0 - alive),
        "time_penalty": -seconds,
        "physical_penalty": -physical_penalty,
        "engine_off_penalty": -5.0 * engine_off,
        "starter_penalty": -starter_paid / 1500.0,
    }
    reconstructed = sum(components.values())
    residual = float(being.get("score", 0.0)) - reconstructed
    return components, reconstructed, residual


def budget_ticks_for(command_args: dict, horizon_ticks: float) -> float:
    """The judging budget used inside `seconds` (school.wgsl:111):
    select(horizon, min(horizon, deadline), deadline > 0.5)."""
    frac_raw = command_args.get("school-deadline-frac")
    try:
        frac = float(frac_raw) if frac_raw is not None else 0.0
    except (TypeError, ValueError):
        frac = 0.0
    if frac and frac > 0.0:
        deadline_ticks = frac * horizon_ticks
        return min(horizon_ticks, deadline_ticks)
    return horizon_ticks


# ---------------------------------------------------------------------------
# Loading recorded files
# ---------------------------------------------------------------------------


def discover_generations(run_dir: Path) -> list[int]:
    gens = []
    for p in sorted(run_dir.glob("school_result_gen*.json")):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            gens.append(int(digits))
    return sorted(set(gens))


def load_school_gen(run_dir: Path, gen: int) -> dict:
    path = run_dir / f"school_result_gen{gen:04d}.json"
    return json.loads(path.read_text())


def load_selection_gen(run_dir: Path, gen: int) -> dict | None:
    path = run_dir / f"selection_gen{gen:04d}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_driving_csv(run_dir: Path) -> dict:
    """Aggregate driving.csv per (generation, being). driving.csv rows are
    64-tick WINDOW deltas, reset every flush -- confirmed by
    src/bin/gpu_evolve/driving_diagnostics.rs:367,532,893
    (`self.windows.fill(Window::default())`) and by the raw data itself
    (stationary_ticks goes 53 -> 29 -> 0 across the first three rows of
    being 0/generation 0 in this run, which is impossible for a monotonic
    cumulative counter). Position (x, z) is the absolute pose at the end of
    each window, not a delta.

    Returns:
      sums_full[(gen,being)]  -> summed counters over the WHOLE recorded episode
      sums_judged[(gen,being)]-> summed counters restricted to rows whose
                                 last_tick <= that being's school outcome_tick
      tracks[(gen,being)]     -> ordered list of (x, z) window-end positions
                                 (full episode, used for the revisit proxy)
    """
    path = run_dir / "driving.csv"
    sums_full: dict = defaultdict(lambda: defaultdict(float))
    sums_judged: dict = defaultdict(lambda: defaultdict(float))
    tracks: dict = defaultdict(list)
    counter_fields = [
        "ticks",
        "stationary_ticks",
        "reverse_ticks",
        "near_wall_ticks",
        "contact_ticks",
        "distance_m",
        "collision_energy_j",
    ]
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gen = int(row["generation"])
            being = int(row["being"])
            key = (gen, being)
            for field in counter_fields:
                sums_full[key][field] += float(row[field])
            tracks[key].append((float(row["x"]), float(row["z"])))
    return sums_full, sums_judged, tracks, counter_fields


def restrict_to_judged_span(
    run_dir: Path,
    sums_judged: dict,
    counter_fields: list[str],
    outcome_tick_by_key: dict,
) -> None:
    """Second pass over driving.csv, filling sums_judged in place, restricted
    to rows whose last_tick does not exceed the being's recorded outcome_tick.
    Kept as a separate pass so the caller can build outcome_tick_by_key from
    school_result first."""
    path = run_dir / "driving.csv"
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gen = int(row["generation"])
            being = int(row["being"])
            key = (gen, being)
            last_tick = float(row["last_tick"])
            outcome = outcome_tick_by_key.get(key)
            if outcome is None or last_tick <= outcome:
                for field in counter_fields:
                    sums_judged[key][field] += float(row[field])


def spatial_revisit_fraction(points: list[tuple[float, float]], radius: float = REVISIT_RADIUS_M) -> float | None:
    """Fraction of window-end samples (excluding the first) that land within
    `radius` metres of an EARLIER sample of the same being. A high fraction is
    a parking/loop/back-and-forth proxy: the being keeps returning to ground
    it already covered instead of covering new road."""
    n = len(points)
    if n < 2:
        return None
    xy = np.asarray(points, dtype=np.float64)
    revisited = 0
    # O(n^2) but n is ~256 samples per being per generation (16384 ticks /
    # 64-tick window); cheap enough for 64 beings x <=4 generations.
    for i in range(1, n):
        earlier = xy[:i]
        d2 = np.sum((earlier - xy[i]) ** 2, axis=1)
        if np.any(d2 <= radius * radius):
            revisited += 1
    return revisited / (n - 1)


# ---------------------------------------------------------------------------
# Rank correlation (Spearman), stdlib-only
# ---------------------------------------------------------------------------


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and not (isinstance(a, float) and math.isnan(a)) and not (isinstance(b, float) and math.isnan(b))]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx = _ranks(list(xs))
    ry = _ranks(list(ys))
    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze(run_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    command_path = run_dir / "command.json"
    command_args = parse_argv_json(command_path) if command_path.exists() else {}

    gens = discover_generations(run_dir)
    if not gens:
        raise SystemExit(f"no school_result_gen*.json found under {run_dir}")

    schema_notes: list[str] = []

    gen_data = {}
    outcome_tick_by_key: dict = {}
    for gen in gens:
        school = load_school_gen(run_dir, gen)
        selection = load_selection_gen(run_dir, gen)
        beings = school.get("beings")
        if beings is None:
            schema_notes.append(f"gen{gen}: school_result has no 'beings' key")
            beings = []
        horizon_ticks = float(school.get("horizon_ticks", 0.0))
        budget = budget_ticks_for(command_args, horizon_ticks)
        for b in beings:
            key = (gen, int(b.get("slot", -1)))
            outcome_tick_by_key[key] = float(b.get("outcome_tick", horizon_ticks))
        gen_data[gen] = {
            "school": school,
            "selection": selection,
            "beings": beings,
            "horizon_ticks": horizon_ticks,
            "budget_ticks": budget,
        }

    driving_path = run_dir / "driving.csv"
    if driving_path.exists():
        sums_full, sums_judged, tracks, counter_fields = load_driving_csv(run_dir)
        restrict_to_judged_span(run_dir, sums_judged, counter_fields, outcome_tick_by_key)
    else:
        schema_notes.append("driving.csv missing: degenerate-indicator section skipped")
        sums_full, sums_judged, tracks, counter_fields = {}, {}, {}, []

    # -----------------------------------------------------------------
    # Item 1: per generation, per genome, selection score decomposition
    # -----------------------------------------------------------------
    selection_report = {"generations": []}
    population_component_means = {}  # gen -> {component: mean over all beings}
    population_progress_mean = {}  # gen -> mean progress over all beings

    for gen in gens:
        data = gen_data[gen]
        beings = data["beings"]
        budget = data["budget_ticks"]
        selection_json = data["selection"]
        aggregates = {a["genome"]: a for a in data["school"].get("aggregates", [])}

        by_genome: dict = defaultdict(list)
        for b in beings:
            by_genome[int(b.get("genome", -1))].append(b)

        all_components = []
        genome_rows = []
        for genome_id in sorted(by_genome):
            starts = by_genome[genome_id]
            comps_per_start = []
            for b in starts:
                comps, recon, residual = score_components(b, budget)
                comps_per_start.append((b, comps, recon, residual))
                all_components.append(comps)
            # worst start = min recorded score, this is the selection score
            # under --school-select worst (src/bin/gpu_evolve/school_batch.rs:117)
            worst_entry = min(comps_per_start, key=lambda t: t[0].get("score", float("inf")))
            worst_being, worst_comps, worst_recon, worst_residual = worst_entry
            mean_components = {
                k: statistics.mean(c[k] for _, c, _, _ in comps_per_start) for k in worst_comps
            }
            mean_progress = statistics.mean(b.get("progress", 0.0) for b in starts)
            selection_score = None
            if selection_json and 0 <= genome_id < len(selection_json.get("scores", [])):
                selection_score = selection_json["scores"][genome_id]
            aggregate = aggregates.get(genome_id, {})

            genome_rows.append(
                {
                    "genome": genome_id,
                    "selection_score": selection_score,
                    "aggregate_mean_score": aggregate.get("mean_score"),
                    "aggregate_worst_score": aggregate.get("worst_score"),
                    "aggregate_worst_failure_penalty": aggregate.get("worst_failure_penalty"),
                    "mean_progress": mean_progress,
                    "worst_start_index": worst_being.get("start_index"),
                    "worst_start_fail": worst_being.get("fail"),
                    "worst_start_score_reconstructed": worst_recon,
                    "worst_start_score_recorded": worst_being.get("score"),
                    "worst_start_residual": worst_residual,
                    "worst_start_components": worst_comps,
                    "mean_components_across_starts": mean_components,
                }
            )
        selection_report["generations"].append(
            {
                "generation": gen,
                "horizon_ticks": data["horizon_ticks"],
                "budget_ticks": budget,
                "genome_count": len(genome_rows),
                "genomes": genome_rows,
            }
        )
        population_component_means[gen] = {
            k: statistics.mean(c[k] for c in all_components) for k in all_components[0]
        } if all_components else {}
        population_progress_mean[gen] = (
            statistics.mean(b.get("progress", 0.0) for b in beings) if beings else None
        )

    # Explicit first-vs-last-generation delta table (population mean, all beings)
    first_gen, last_gen = gens[0], gens[-1]
    component_delta = {}
    if population_component_means.get(first_gen) and population_component_means.get(last_gen):
        for k in population_component_means[first_gen]:
            first_v = population_component_means[first_gen][k]
            last_v = population_component_means[last_gen][k]
            component_delta[k] = {
                "first_gen": first_v,
                "last_gen": last_v,
                "delta": last_v - first_v,
            }
    component_delta_sorted = sorted(
        component_delta.items(), key=lambda kv: abs(kv[1]["delta"]), reverse=True
    )

    # Same delta table, but using the SELECTION-defining worst-start component
    # (this is what --school-select worst actually reproduces on), meaned
    # over the 16 genomes.
    def worst_component_population_mean(gen: int) -> dict:
        rows = selection_report["generations"][gens.index(gen)]["genomes"]
        if not rows:
            return {}
        keys = rows[0]["worst_start_components"].keys()
        return {k: statistics.mean(r["worst_start_components"][k] for r in rows) for k in keys}

    worst_component_first = worst_component_population_mean(first_gen)
    worst_component_last = worst_component_population_mean(last_gen)
    worst_component_delta = {
        k: {
            "first_gen": worst_component_first[k],
            "last_gen": worst_component_last[k],
            "delta": worst_component_last[k] - worst_component_first[k],
        }
        for k in worst_component_first
    } if worst_component_first and worst_component_last else {}
    worst_component_delta_sorted = sorted(
        worst_component_delta.items(), key=lambda kv: abs(kv[1]["delta"]), reverse=True
    )

    selection_report["population_mean_component_by_gen"] = population_component_means
    selection_report["population_mean_progress_by_gen"] = population_progress_mean
    selection_report["component_delta_first_vs_last_all_beings"] = dict(component_delta_sorted)
    selection_report["component_delta_first_vs_last_worst_start_only"] = dict(
        worst_component_delta_sorted
    )
    selection_report["note"] = (
        "worst_start_components are the components of the single start whose "
        "recorded score is minimum among that genome's starts; under "
        "--school-select worst (school_batch.rs:117) this IS the per-genome "
        "selection score when starts_per_genome > 1, so its component deltas "
        "explain the selection trend directly. all_beings deltas are the "
        "unweighted population mean across every recorded start and are shown "
        "for context."
    )

    # -----------------------------------------------------------------
    # Item 2: per-start matrix, first vs last generation
    # -----------------------------------------------------------------
    def build_matrix(gen: int) -> dict:
        beings = gen_data[gen]["beings"]
        genome_ids = sorted({int(b.get("genome", -1)) for b in beings})
        start_ids = sorted({int(b.get("start_index", -1)) for b in beings})
        progress = {g: {s: None for s in start_ids} for g in genome_ids}
        fail = {g: {s: None for s in start_ids} for g in genome_ids}
        score = {g: {s: None for s in start_ids} for g in genome_ids}
        for b in beings:
            g = int(b.get("genome", -1))
            s = int(b.get("start_index", -1))
            progress[g][s] = b.get("progress")
            fail[g][s] = b.get("fail")
            score[g][s] = b.get("score")
        return {
            "generation": gen,
            "genome_ids": genome_ids,
            "start_ids": start_ids,
            "progress": progress,
            "fail_code": fail,
            "score": score,
        }

    per_start_matrix = {
        "first_generation": build_matrix(first_gen),
        "last_generation": build_matrix(last_gen),
    }

    # -----------------------------------------------------------------
    # Item 3: degenerate-strategy indicators per being + correlation
    # -----------------------------------------------------------------
    being_indicators = []
    for gen in gens:
        beings = gen_data[gen]["beings"]
        for b in beings:
            slot = int(b.get("slot", -1))
            key = (gen, slot)
            judged = sums_judged.get(key, {})
            full = sums_full.get(key, {})
            judged_ticks_sum = judged.get("ticks", 0.0)
            full_ticks_sum = full.get("ticks", 0.0)

            def frac(d, num_field, denom):
                return (d.get(num_field, 0.0) / denom) if denom else None

            requests_denominator = b.get("judged_ticks") or None
            pickup_per_1000 = (
                (b.get("pickup_requests", 0.0) / requests_denominator * 1000.0)
                if requests_denominator
                else None
            )
            dropoff_per_1000 = (
                (b.get("dropoff_requests", 0.0) / requests_denominator * 1000.0)
                if requests_denominator
                else None
            )
            revisit = None
            if key in tracks:
                revisit = spatial_revisit_fraction(tracks[key])

            being_indicators.append(
                {
                    "generation": gen,
                    "genome": int(b.get("genome", -1)),
                    "start_index": int(b.get("start_index", -1)),
                    "slot": slot,
                    "selection_relevant_score": b.get("score"),
                    "progress": b.get("progress"),
                    "fail": b.get("fail"),
                    "judged_ticks": b.get("judged_ticks"),
                    "stationary_fraction_judged": frac(judged, "stationary_ticks", judged_ticks_sum),
                    "reverse_fraction_judged": frac(judged, "reverse_ticks", judged_ticks_sum),
                    "near_wall_fraction_judged": frac(judged, "near_wall_ticks", judged_ticks_sum),
                    "contact_ticks_judged": judged.get("contact_ticks"),
                    "stationary_fraction_full_episode": frac(full, "stationary_ticks", full_ticks_sum),
                    "near_wall_fraction_full_episode": frac(full, "near_wall_ticks", full_ticks_sum),
                    "contact_ticks_full_episode": full.get("contact_ticks"),
                    "distance_m_full_episode": full.get("distance_m"),
                    "pickup_requests_per_1000_ticks": pickup_per_1000,
                    "dropoff_requests_per_1000_ticks": dropoff_per_1000,
                    "route_revisit_fraction_5m": revisit,
                }
            )

    # Genome-level aggregation (mean of the 4 starts) paired with that
    # genome's selection score, pooled across generations and per generation.
    indicator_fields = [
        "stationary_fraction_judged",
        "reverse_fraction_judged",
        "near_wall_fraction_judged",
        "contact_ticks_judged",
        "distance_m_full_episode",
        "pickup_requests_per_1000_ticks",
        "dropoff_requests_per_1000_ticks",
        "route_revisit_fraction_5m",
    ]

    genome_level = defaultdict(lambda: defaultdict(list))  # (gen,genome) -> field -> [values]
    genome_selection_score = {}
    for gen in gens:
        rows = selection_report["generations"][gens.index(gen)]["genomes"]
        for r in rows:
            genome_selection_score[(gen, r["genome"])] = r["selection_score"]
    for row in being_indicators:
        key = (row["generation"], row["genome"])
        for f in indicator_fields:
            if row[f] is not None:
                genome_level[key][f].append(row[f])

    genome_mean_indicator = {}
    for key, fields in genome_level.items():
        genome_mean_indicator[key] = {f: statistics.mean(v) for f, v in fields.items() if v}

    correlations_pooled = {}
    correlations_by_gen = {gen: {} for gen in gens}
    for f in indicator_fields:
        pooled_x, pooled_y = [], []
        for gen in gens:
            gen_x, gen_y = [], []
            for genome_id in sorted({g for (gg, g) in genome_mean_indicator if gg == gen}):
                key = (gen, genome_id)
                sel = genome_selection_score.get(key)
                val = genome_mean_indicator.get(key, {}).get(f)
                if sel is not None and val is not None:
                    gen_x.append(val)
                    gen_y.append(sel)
                    pooled_x.append(val)
                    pooled_y.append(sel)
            correlations_by_gen[gen][f] = spearman(gen_x, gen_y)
        correlations_pooled[f] = spearman(pooled_x, pooled_y)

    # "Rewarded by selection" = indicator whose value tracks a HIGHER (less
    # negative) selection score across genomes, i.e. positive Spearman rho.
    rewarded_ranked = sorted(
        ((f, rho) for f, rho in correlations_pooled.items() if rho is not None),
        key=lambda kv: kv[1],
        reverse=True,
    )

    degenerate_report = {
        "note": (
            "stationary/reverse/near_wall/contact fractions are computed from "
            "driving.csv 64-tick WINDOW deltas (src/bin/gpu_evolve/"
            "driving_diagnostics.rs:711-746, reset every flush at "
            "driving_diagnostics.rs:367/532/893), summed per being and divided "
            "by summed ticks. '_judged' restricts the sum to rows whose "
            "last_tick <= that being's recorded school outcome_tick "
            "(the physics episode keeps running to the generation horizon "
            "after the school judge already closed the attempt, so an "
            "unrestricted full-episode fraction mixes judged and post-outcome "
            "behaviour). pickup/dropoff request rates come from the school "
            "record's own pickup_requests/dropoff_requests counters (school.wgsl:"
            "178-179), which the driving.csv schema does not carry. "
            "route_revisit_fraction_5m is a parking/loop proxy: fraction of a "
            "being's 64-tick window-end (x,z) positions that lie within 5 m of "
            "an earlier position of the SAME being in the SAME generation; it "
            "is a heuristic over sparse samples (~256 points per full episode), "
            "not a ground-truth path integral."
        ),
        "per_being": being_indicators,
        "genome_mean_indicator_by_gen_genome": {
            f"{gen}:{genome}": v for (gen, genome), v in genome_mean_indicator.items()
        },
        "spearman_pooled_across_generations": correlations_pooled,
        "spearman_by_generation": correlations_by_gen,
        "rewarded_by_selection_ranked_high_to_low": rewarded_ranked,
    }

    # -----------------------------------------------------------------
    # Item 4: discriminating comparison proposal (read-only, no weight change)
    # -----------------------------------------------------------------
    def task_only_score(being: dict) -> float:
        """Progress + terminal-outcome terms only: no time cost, no collision
        penalty, no failed/integrity/alive penalty. Read-only diagnostic,
        NOT a proposed replacement weighting -- see the note below."""
        success = 1.0 if being.get("success") else 0.0
        lesson = float(being.get("lesson", 0.0))
        accepted_pickup_tick = being.get("accepted_pickup_tick")
        completed_pickup = 1.0 if (lesson >= 7.5 and accepted_pickup_tick is not None) else 0.0
        delivery_delta = float(being.get("delivery_delta", 0.0))
        accepted_delivery = success * max(0.0, min(1.0, delivery_delta))
        progress = float(being.get("progress", 0.0))
        return progress + 1000.0 * accepted_delivery + 200.0 * completed_pickup + 500.0 * success

    task_only_by_gen = {
        gen: statistics.mean(task_only_score(b) for b in gen_data[gen]["beings"])
        for gen in gens
        if gen_data[gen]["beings"]
    }
    full_score_by_gen = {
        gen: statistics.mean(b.get("score", 0.0) for b in gen_data[gen]["beings"])
        for gen in gens
        if gen_data[gen]["beings"]
    }
    delta_task_only = task_only_by_gen[last_gen] - task_only_by_gen[first_gen]
    delta_full_score = full_score_by_gen[last_gen] - full_score_by_gen[first_gen]
    discriminating_margin = delta_full_score - delta_task_only

    discriminating_comparison = {
        "proposal": (
            "Compare the change in a TASK-ONLY score (progress + delivery/"
            "pickup/success terms, with no time cost and no physical-contact "
            "penalty) against the change in the full operational SCORE, "
            "generation-over-generation. No reward weight is changed anywhere "
            "in this proposal; task_only_score is a read-only diagnostic "
            "computed here from already-recorded fields, never written back "
            "into selection."
        ),
        "pass_condition": (
            "'Learned navigation' PASSES only if delta_task_only_score >= 0, "
            "i.e. the population's mean task-only score does not fall from "
            "the first to the last recorded generation."
        ),
        "fail_condition": (
            "FAILS (supports a degenerate, non-navigation explanation for any "
            "rise in the full SCORE) if delta_task_only_score < 0 while "
            "delta_full_score > 0. The gap "
            "(discriminating_margin = delta_full_score - delta_task_only_score) "
            "is the amount of the full SCORE's rise that is attributable to "
            "non-task terms (time/physical/integrity/alive penalties shrinking) "
            "rather than to task competence."
        ),
        "measured_numbers": {
            "first_generation": first_gen,
            "last_generation": last_gen,
            "task_only_score_mean_first_gen": task_only_by_gen[first_gen],
            "task_only_score_mean_last_gen": task_only_by_gen[last_gen],
            "delta_task_only_score": delta_task_only,
            "full_score_mean_first_gen": full_score_by_gen[first_gen],
            "full_score_mean_last_gen": full_score_by_gen[last_gen],
            "delta_full_score": delta_full_score,
            "discriminating_margin": discriminating_margin,
        },
        "verdict_this_run": (
            "FAIL: task-only score fell while the full score rose"
            if delta_task_only < 0 and delta_full_score > 0
            else "did not meet the stated FAIL condition; re-check the numbers above"
        ),
    }

    # -----------------------------------------------------------------
    # Assemble + write
    # -----------------------------------------------------------------
    result = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "generations_found": gens,
        "command_args_parsed": command_args,
        "schema_notes": schema_notes,
        "selection_decomposition": selection_report,
        "per_start_matrix": per_start_matrix,
        "degenerate_indicators": degenerate_report,
        "discriminating_comparison": discriminating_comparison,
    }

    (out_dir / "selection_decomposition.json").write_text(
        json.dumps({"schema_version": 1, "run_dir": str(run_dir), **selection_report}, indent=2)
    )
    (out_dir / "per_start_matrix.json").write_text(json.dumps(per_start_matrix, indent=2))
    (out_dir / "degenerate_indicators.json").write_text(json.dumps(degenerate_report, indent=2))
    (out_dir / "discriminating_comparison.json").write_text(
        json.dumps(discriminating_comparison, indent=2)
    )
    (out_dir / "full_report.json").write_text(json.dumps(result, indent=2))

    write_markdown_summary(out_dir, result)

    return result


def write_markdown_summary(out_dir: Path, result: dict) -> None:
    gens = result["generations_found"]
    first_gen, last_gen = gens[0], gens[-1]
    sel = result["selection_decomposition"]
    prog = sel["population_mean_progress_by_gen"]
    worst_delta = sel["component_delta_first_vs_last_worst_start_only"]
    all_delta = sel["component_delta_first_vs_last_all_beings"]
    disc = result["discriminating_comparison"]
    deg = result["degenerate_indicators"]

    lines = []
    lines.append("# Cross-city forward-learning: selection vs. task-progress decomposition")
    lines.append("")
    lines.append(f"Run: `{result['run_dir']}`  |  generations analyzed: {gens}")
    lines.append("")
    lines.append("Scope: file aggregation only, from recorded GPU output. No simulation.")
    lines.append("")
    lines.append("## 1. Population mean progress vs. score, first vs. last recorded generation")
    lines.append("")
    lines.append(f"- mean `progress` (school PROGRESS field, 0..200): "
                  f"gen{first_gen}={prog.get(first_gen):.2f} -> gen{last_gen}={prog.get(last_gen):.2f}")
    lines.append(f"- mean full `score` (all beings): see `component_delta_first_vs_last_all_beings`")
    lines.append("")
    lines.append("### Component deltas that explain the SELECTION score "
                  "(worst-of-4-starts per genome, meaned over genomes -- this is "
                  "the quantity `--school-select worst` actually reproduces on)")
    lines.append("")
    lines.append("| component | gen{} mean | gen{} mean | delta |".format(first_gen, last_gen))
    lines.append("|---|---|---|---|")
    for k, v in worst_delta.items():
        lines.append(f"| {k} | {v['first_gen']:.3f} | {v['last_gen']:.3f} | {v['delta']:+.3f} |")
    lines.append("")
    lines.append("### Same table, unweighted mean over ALL recorded starts (context only)")
    lines.append("")
    lines.append("| component | gen{} mean | gen{} mean | delta |".format(first_gen, last_gen))
    lines.append("|---|---|---|---|")
    for k, v in all_delta.items():
        lines.append(f"| {k} | {v['first_gen']:.3f} | {v['last_gen']:.3f} | {v['delta']:+.3f} |")
    lines.append("")
    lines.append("## 2. Per-start matrix")
    lines.append("")
    lines.append("See `per_start_matrix.json` for the full DNA x start_index "
                  "progress/fail_code/score grids for the first and last generation.")
    lines.append("")
    lines.append("## 3. Degenerate-strategy indicators rewarded by selection "
                  "(Spearman rho, pooled across generations, genome-level means)")
    lines.append("")
    lines.append("| indicator | rho vs. selection score |")
    lines.append("|---|---|")
    for f, rho in deg["rewarded_by_selection_ranked_high_to_low"]:
        lines.append(f"| {f} | {rho:+.3f} |")
    lines.append("")
    lines.append("Positive rho: the indicator is higher for genomes selection currently "
                  "favours (higher/less-negative selection score). This is a cross-sectional "
                  "association across DNA within/pooled across generations, not a causal claim.")
    lines.append("")
    lines.append("## 4. Discriminating comparison proposal (not implemented, weights unchanged)")
    lines.append("")
    lines.append(disc["proposal"])
    lines.append("")
    m = disc["measured_numbers"]
    lines.append(f"- task-only score mean: gen{first_gen}={m['task_only_score_mean_first_gen']:.2f} "
                  f"-> gen{last_gen}={m['task_only_score_mean_last_gen']:.2f} "
                  f"(delta={m['delta_task_only_score']:+.2f})")
    lines.append(f"- full score mean: gen{first_gen}={m['full_score_mean_first_gen']:.2f} "
                  f"-> gen{last_gen}={m['full_score_mean_last_gen']:.2f} "
                  f"(delta={m['delta_full_score']:+.2f})")
    lines.append(f"- discriminating margin (full delta - task-only delta): "
                  f"{m['discriminating_margin']:+.2f}")
    lines.append(f"- verdict on this run: **{disc['verdict_this_run']}**")
    lines.append("")
    if result["schema_notes"]:
        lines.append("## Schema notes / surprises")
        lines.append("")
        for note in result["schema_notes"]:
            lines.append(f"- {note}")
        lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def print_stdout_table(result: dict) -> None:
    gens = result["generations_found"]
    first_gen, last_gen = gens[0], gens[-1]
    sel = result["selection_decomposition"]
    prog = sel["population_mean_progress_by_gen"]
    worst_delta = sel["component_delta_first_vs_last_worst_start_only"]
    disc = result["discriminating_comparison"]
    deg = result["degenerate_indicators"]

    print(f"\n=== analyze_city_failure_segments: {result['run_dir']} ===")
    print(f"generations: {gens}")
    print(f"population mean progress: gen{first_gen}={prog.get(first_gen):.2f} "
          f"-> gen{last_gen}={prog.get(last_gen):.2f}")
    print("\nSELECTION-defining (worst-start) component deltas, first -> last generation:")
    print(f"{'component':<20}{'gen'+str(first_gen):>12}{'gen'+str(last_gen):>12}{'delta':>12}")
    for k, v in worst_delta.items():
        print(f"{k:<20}{v['first_gen']:>12.2f}{v['last_gen']:>12.2f}{v['delta']:>+12.2f}")
    print("\nTop indicators rewarded by selection (Spearman rho, pooled):")
    for f, rho in deg["rewarded_by_selection_ranked_high_to_low"][:5]:
        print(f"  {f:<32} rho={rho:+.3f}")
    print("\nDiscriminating comparison verdict:", disc["verdict_this_run"])
    m = disc["measured_numbers"]
    print(f"  delta_task_only_score={m['delta_task_only_score']:+.2f}  "
          f"delta_full_score={m['delta_full_score']:+.2f}  "
          f"discriminating_margin={m['discriminating_margin']:+.2f}")
    if result["schema_notes"]:
        print("\nSchema notes:")
        for note in result["schema_notes"]:
            print(f"  - {note}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="run's learning/ directory")
    parser.add_argument("--out", required=True, type=Path, help="output directory (created if missing)")
    args = parser.parse_args()

    run_dir = args.run.resolve()
    out_dir = args.out.resolve()
    if not run_dir.is_dir():
        print(f"error: --run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    result = analyze(run_dir, out_dir)
    print_stdout_table(result)
    print(f"Wrote: {out_dir}/selection_decomposition.json")
    print(f"Wrote: {out_dir}/per_start_matrix.json")
    print(f"Wrote: {out_dir}/degenerate_indicators.json")
    print(f"Wrote: {out_dir}/discriminating_comparison.json")
    print(f"Wrote: {out_dir}/full_report.json")
    print(f"Wrote: {out_dir}/SUMMARY.md")


if __name__ == "__main__":
    main()
