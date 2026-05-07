#!/usr/bin/env python3
"""
Evaluate vision + heading axiom improvements vs baseline.
Analyzes: swing, crashes, speed_bleeds, elite pool size.
"""
import json
import glob
import numpy as np
from pathlib import Path

def analyze_run(run_dir):
    """Analyze a single telemetry run."""
    gen_file = Path(run_dir) / "generations.ndjson"
    if not gen_file.exists():
        return None

    data = []
    with open(gen_file) as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass

    if not data:
        return None

    dists = [d['champion_dist_m'] for d in data]
    swings = [abs(dists[i] - dists[i-1]) for i in range(1, len(dists))]
    crashes = sum(1 for d in dists if d <= 0)
    speed_bleeds = [d.get('speed_bleeds', 0) for d in data]
    elite_sizes = [d.get('elite_pool_size', 0) for d in data]

    return {
        'gens': len(data),
        'final_dist': dists[-1] if dists else 0,
        'max_dist': max(dists) if dists else 0,
        'avg_dist': np.mean(dists),
        'min_dist': min(dists) if dists else 0,
        'avg_swing': np.mean(swings) if swings else 0,
        'max_swing': max(swings) if swings else 0,
        'crashes': crashes,
        'avg_speed_bleeds': np.mean(speed_bleeds) if speed_bleeds else 0,
        'max_speed_bleeds': max(speed_bleeds) if speed_bleeds else 0,
        'avg_elite_size': np.mean(elite_sizes) if elite_sizes else 0,
        'dists': dists,
    }

# Find latest run
runs = sorted(glob.glob("/home/leonardo/projects/LLM/Universe/telemetry/*/"))
if not runs:
    print("No telemetry data found")
    exit(1)

latest = runs[-1]
print(f"Analyzing: {latest}")
result = analyze_run(latest)

if result:
    print(f"\n📊 20-GEN BENCHMARK RESULTS")
    print(f"{'─' * 50}")
    print(f"Final distance:     {result['final_dist']:.0f}m")
    print(f"Max distance:       {result['max_dist']:.0f}m")
    print(f"Avg distance:       {result['avg_dist']:.0f}m")
    print(f"Avg swing:          {result['avg_swing']:.0f}m")
    print(f"Max swing:          {result['max_swing']:.0f}m")
    print(f"Crashes (dist≤0):   {result['crashes']}")
    print(f"Avg speed_bleeds:   {result['avg_speed_bleeds']:.1f}/20")
    print(f"Max speed_bleeds:   {result['max_speed_bleeds']:.0f}/20")
    print(f"Avg elite pool:     {result['avg_elite_size']:.1f}")
    print(f"{'─' * 50}")

    # Targets from plan
    targets = {
        'avg_swing': 800,
        'crashes': 0,
        'avg_speed_bleeds': 10,
        'elite_size': 3,  # should maintain some diversity
    }

    print(f"\n🎯 TARGET COMPARISON")
    print(f"{'─' * 50}")
    print(f"Avg swing:    {result['avg_swing']:.0f}m  {'✅' if result['avg_swing'] < targets['avg_swing'] else '❌'} < {targets['avg_swing']}m")
    print(f"Crashes:      {result['crashes']}       {'✅' if result['crashes'] == 0 else '❌'} = 0")
    print(f"Speed bleeds: {result['avg_speed_bleeds']:.1f}/20 {'✅' if result['avg_speed_bleeds'] < targets['avg_speed_bleeds'] else '❌'} < {targets['avg_speed_bleeds']}/20")
    print(f"Elite pool:   {result['avg_elite_size']:.1f}  {'✅' if result['avg_elite_size'] >= targets['elite_size'] else '⚠️'} >= {targets['elite_size']} (diversity)")

    # Distance progression
    print(f"\n📈 DISTANCE PROGRESSION")
    print(f"{'─' * 50}")
    for i, d in enumerate(result['dists'], 1):
        bar = '█' * int(d / 100) if d > 0 else '🔥'
        print(f"Gen {i:2d}: {d:6.0f}m {bar}")
else:
    print("No valid telemetry data")
