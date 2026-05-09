#!/usr/bin/env python3
"""Champion replay analysis: learning curves, performance patterns, DNA insights."""

import json
from pathlib import Path
from collections import defaultdict
import statistics

def extract_champion_behavior(run_dir):
    """Extract champion progression from generations.ndjson."""
    gen_file = run_dir / "generations.ndjson"
    if not gen_file.exists():
        return None

    gens = []
    with open(gen_file) as f:
        for line in f:
            try:
                gens.append(json.loads(line))
            except:
                pass

    if not gens:
        return None

    return {
        "run_id": run_dir.name,
        "gens": [
            {
                "gen": g.get("gen"),
                "champion_m": g.get("champion_dist_m", 0),
                "p50_m": g.get("p50_dist_m", 0),
                "avg_m": g.get("avg_dist_m", 0),
                "crashes": g.get("crashes", 0),
                "survivors": g.get("survivors", 0),
                "fitness": g.get("best_fitness", 0),
            }
            for g in gens
        ]
    }

def format_learning_curve(gens_data):
    """Format champion distance across generations."""
    distances = [g["champion_m"] for g in gens_data]
    if len(distances) < 2:
        return f"Gen {len(distances)}: {distances[0]:.0f}m"

    growth_pct = ((distances[-1] - distances[0]) / max(distances[0], 1)) * 100
    peak = max(distances)
    final = distances[-1]

    return f"Gens {len(distances)}: {distances[0]:.0f}m → {final:.0f}m (peak {peak:.0f}m, {growth_pct:+.0f}%)"

def main():
    print("\n🏆 CHAMPION REPLAY ANALYSIS\n")

    telemetry_dir = Path("telemetry")
    if not telemetry_dir.exists():
        print("No telemetry/ found")
        return

    # Load all runs
    runs = []
    for run_dir in sorted(telemetry_dir.iterdir())[:30]:  # Top 30 runs
        if not run_dir.is_dir():
            continue
        data = extract_champion_behavior(run_dir)
        if data and len(data["gens"]) > 0:
            runs.append(data)

    if not runs:
        print("No telemetry data found")
        return

    # Group by generation count
    by_gen_count = defaultdict(list)
    for run in runs:
        by_gen_count[len(run["gens"])].append(run)

    # Report multi-gen runs
    print("📈 MULTI-GENERATION RUNS (Learning Curves)")
    print("─" * 80)

    multi_gen_runs = [r for r in runs if len(r["gens"]) > 1]
    if not multi_gen_runs:
        print("(No multi-gen runs found)")
    else:
        for run in sorted(multi_gen_runs, key=lambda r: max(g["champion_m"] for g in r["gens"]), reverse=True)[:10]:
            print(f"\n🎯 Run {run['run_id']}")
            for g in run["gens"]:
                symbol = "→" if g["gen"] < len(run["gens"]) else "✓"
                crash_str = f" | crashes={g['crashes']}" if g["crashes"] > 0 else ""
                print(f"  {symbol} GEN {g['gen']}: {g['champion_m']:7.0f}m (p50={g['p50_m']:6.0f}m, fitness={g['fitness']:8.0f}{crash_str})")

    # Single-gen peaks
    print("\n\n⭐ SINGLE-GEN CHAMPIONS (Instant Performance)")
    print("─" * 80)

    single_gen_peaks = []
    for run in runs:
        for g in run["gens"]:
            single_gen_peaks.append({
                "run_id": run["run_id"],
                "gen": g["gen"],
                "distance": g["champion_m"],
                "fitness": g["fitness"],
                "crashes": g["crashes"],
            })

    top_singles = sorted(single_gen_peaks, key=lambda x: x["distance"], reverse=True)[:15]
    for i, peak in enumerate(top_singles, 1):
        print(f"{i:2d}. {peak['distance']:7.0f}m | Gen {peak['gen']} | Crashes: {peak['crashes']:2d} | Fitness: {peak['fitness']:8.0f}")

    # Statistics
    print("\n\n📊 AGGREGATE STATISTICS")
    print("─" * 80)

    all_dists = [g["champion_m"] for run in runs for g in run["gens"]]
    all_crashes = [g["crashes"] for run in runs for g in run["gens"]]
    all_survivors = [g["survivors"] for run in runs for g in run["gens"]]

    print(f"Champion distance:")
    print(f"  Max: {max(all_dists):.0f}m")
    print(f"  Median: {statistics.median(all_dists):.0f}m")
    print(f"  Mean: {statistics.mean(all_dists):.0f}m")
    print(f"Crashes per gen: {statistics.mean(all_crashes):.1f} avg")
    print(f"Survivors per gen: {statistics.mean(all_survivors):.1f} avg")

    # Learning efficiency
    improving_runs = []
    for run in multi_gen_runs:
        first = run["gens"][0]["champion_m"]
        last = run["gens"][-1]["champion_m"]
        if first > 0:
            improvement = ((last - first) / first) * 100
            improving_runs.append((run["run_id"], improvement, first, last, len(run["gens"])))

    if improving_runs:
        print(f"\n📈 Learning Efficiency (multi-gen runs):")
        for run_id, improvement, first, last, gens in sorted(improving_runs, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {run_id}: {first:.0f}m → {last:.0f}m over {gens} gens ({improvement:+.0f}%)")

    print("\n✅ ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
