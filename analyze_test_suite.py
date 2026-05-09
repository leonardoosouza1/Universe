#!/usr/bin/env python3
"""Analyze test suite telemetry: DNA validation, performance curves, learning."""

import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

def load_telemetry():
    """Load all telemetry runs."""
    runs = {}
    telemetry_dir = Path("telemetry")

    if not telemetry_dir.exists():
        print("❌ No telemetry/ directory found")
        return runs

    for run_dir in telemetry_dir.iterdir():
        if not run_dir.is_dir():
            continue

        gen_file = run_dir / "generations.ndjson"
        meta_file = run_dir / "run_meta.json"

        if not gen_file.exists():
            continue

        run_id = run_dir.name
        meta = {}
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

        gens = []
        with open(gen_file) as f:
            for line in f:
                try:
                    gens.append(json.loads(line))
                except:
                    pass

        if gens:
            runs[run_id] = {"meta": meta, "gens": gens}

    return runs

def analyze_run(run_id, data):
    """Analyze single run metrics."""
    gens = data["gens"]
    meta = data["meta"]

    dists = [g.get("champion_dist_m", 0) for g in gens]
    p50s = [g.get("p50_dist_m", 0) for g in gens]
    bleeds = [g.get("speed_bleeds", 0) for g in gens]
    crashes = [g.get("crashes", 0) for g in gens]
    laps = [g.get("lap_count", 0) for g in gens]
    survivors = [g.get("survivors", 0) for g in gens]

    return {
        "run_id": run_id,
        "experiment": meta.get("experiment_name", "unknown"),
        "gens_count": len(gens),
        "max_champion": max(dists) if dists else 0,
        "final_champion": dists[-1] if dists else 0,
        "max_p50": max(p50s) if p50s else 0,
        "final_p50": p50s[-1] if p50s else 0,
        "total_bleeds": sum(bleeds),
        "total_crashes": sum(crashes),
        "total_laps": sum(laps),
        "avg_survivors": statistics.mean(survivors) if survivors else 0,
        "learning_curve": dists,  # full curve
    }

def main():
    runs = load_telemetry()

    if not runs:
        print("❌ No telemetry data found. Check if tests completed.")
        return

    print(f"\n📊 TELEMETRY ANALYSIS ({len(runs)} runs)\n")

    analyses = []
    for run_id in sorted(runs.keys()):
        analysis = analyze_run(run_id, runs[run_id])
        analyses.append(analysis)

    # Group by experiment
    by_exp = defaultdict(list)
    for a in analyses:
        by_exp[a["experiment"]].append(a)

    # Report per group
    for exp_name in sorted(by_exp.keys()):
        exp_runs = by_exp[exp_name]
        print(f"📌 {exp_name} ({len(exp_runs)} runs)")
        print("─" * 80)

        for a in exp_runs:
            print(f"  {a['run_id']}")
            print(f"    Gens: {a['gens_count']} | Champion: max={a['max_champion']:.0f}m final={a['final_champion']:.0f}m")
            print(f"    P50: max={a['max_p50']:.0f}m final={a['final_p50']:.0f}m")
            print(f"    Survival: {a['total_crashes']} crashes, {a['total_bleeds']} bleeds, {a['total_laps']} laps")
            print(f"    Avg survivors/gen: {a['avg_survivors']:.1f}")

            # Learning curve
            curve = a["learning_curve"]
            if len(curve) > 1:
                growth = ((curve[-1] - curve[0]) / max(curve[0], 1)) * 100
                print(f"    Learning: gen0={curve[0]:.0f}m → gen{len(curve)-1}={curve[-1]:.0f}m ({growth:+.0f}%)")
            print()

    # Consolidated summary
    print("\n✅ CONSOLIDATED SUMMARY")
    print("─" * 80)
    if analyses:
        avg_final = statistics.mean(a["final_champion"] for a in analyses)
        max_overall = max(a["max_champion"] for a in analyses)
        total_crashes = sum(a["total_crashes"] for a in analyses)
        total_laps = sum(a["total_laps"] for a in analyses)

        print(f"Final champion distance (avg): {avg_final:.0f}m")
        print(f"Best champion distance (max): {max_overall:.0f}m")
        print(f"Total crashes across all runs: {total_crashes}")
        print(f"Total laps completed: {total_laps}")
        print(f"Health: Zone-based pooling active, SlowBrain frozen (size mismatch pending fix)")

if __name__ == "__main__":
    main()
