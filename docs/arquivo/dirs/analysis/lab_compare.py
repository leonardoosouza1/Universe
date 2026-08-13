#!/usr/bin/env python3
"""
Lab experiment comparison tool.

Usage:
    python analysis/lab_compare.py --spec lab_specs/narrow_widths.json

Reads experiment results from telemetry/ directories and generates comparison
table and visualization.
"""

import os
import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Parse --spec argument
    spec_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--spec":
            if i + 1 < len(sys.argv):
                spec_path = sys.argv[i + 1]
            break
    
    if not spec_path:
        print("Usage: python lab_compare.py --spec <path_to_lab_spec.json>")
        print("Example: python analysis/lab_compare.py --spec lab_specs/narrow_widths.json")
        sys.exit(1)
    
    if not Path(spec_path).exists():
        print(f"Error: spec file not found: {spec_path}")
        sys.exit(1)
    
    # Load spec
    with open(spec_path) as f:
        spec = json.load(f)
    
    spec_name = spec.get("name", "unnamed_lab")
    variant_names = [v["name"] for v in spec.get("variants", [])]
    
    if not variant_names:
        print("Error: spec contains no variants")
        sys.exit(1)
    
    print(f"\n=== {spec_name} ===")
    print(f"Expected variants: {', '.join(variant_names)}\n")
    
    # Find and load results from telemetry directories
    telemetry_root = Path("telemetry")
    if not telemetry_root.exists():
        print(f"Warning: telemetry/ directory not found")
        return
    
    results = {}
    
    for run_dir in sorted(telemetry_root.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Try to load run metadata
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except json.JSONDecodeError:
            continue
        
        exp_name = meta.get("experiment_name", "")
        if exp_name not in variant_names:
            continue
        
        # Load generations.ndjson
        gen_path = run_dir / "generations.ndjson"
        if not gen_path.exists():
            print(f"Warning: {run_dir.name} has no generations.ndjson")
            continue
        
        try:
            gen_text = gen_path.read_text().strip()
            if not gen_text:
                print(f"Warning: {run_dir.name} generations.ndjson is empty")
                continue
            
            lines = [json.loads(line) for line in gen_text.split("\n") if line.strip()]
            if not lines:
                continue
            
            df = pd.DataFrame(lines)
            results[exp_name] = {
                'run_dir': run_dir,
                'df': df
            }
            print(f"Loaded: {exp_name} ({len(df)} generations)")
        except Exception as e:
            print(f"Warning: failed to load {run_dir.name}: {e}")
            continue
    
    if not results:
        print("No experiment results found in telemetry/")
        return
    
    # Print comparison table
    print(f"\n=== Comparison Table ===")
    print(f"{'Variant':<20} {'Max Distance':<15} {'Final P50':<15} {'Avg Speed':<15}")
    print("-" * 65)
    
    for name in variant_names:
        if name not in results:
            print(f"{name:<20} {'NO DATA':<15} {'':<15} {'':<15}")
            continue
        
        df = results[name]['df']
        
        # Extract metrics
        max_dist = df["champion_dist_m"].max() if "champion_dist_m" in df else 0
        final_p50 = df["p50_dist_m"].iloc[-1] if "p50_dist_m" in df and len(df) > 0 else 0
        avg_speed = df["avg_speed_m_per_s"].mean() if "avg_speed_m_per_s" in df else 0
        
        print(f"{name:<20} {max_dist:<15.2f} {final_p50:<15.2f} {avg_speed:<15.2f}")
    
    # Generate progression plot
    print(f"\n=== Generating plots ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{spec_name} — Experiment Comparison", fontsize=16, fontweight='bold')
    
    # Plot 1: Champion Distance Progression
    ax = axes[0, 0]
    colors = plt.cm.viridis([(i+1)/len(variant_names) for i in range(len(variant_names))])
    for (name, data), color in zip(results.items(), colors):
        df = data['df']
        if "gen" in df and "champion_dist_m" in df:
            ax.plot(df["gen"], df["champion_dist_m"], label=name, color=color, marker='o', linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Champion Distance (m)")
    ax.set_title("Champion Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: P50 Distance Progression
    ax = axes[0, 1]
    for (name, data), color in zip(results.items(), colors):
        df = data['df']
        if "gen" in df and "p50_dist_m" in df:
            ax.plot(df["gen"], df["p50_dist_m"], label=name, color=color, marker='s', linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("P50 Distance (m)")
    ax.set_title("Population Median Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Speed Bleeds
    ax = axes[1, 0]
    for (name, data), color in zip(results.items(), colors):
        df = data['df']
        if "gen" in df and "speed_bleeds" in df:
            ax.plot(df["gen"], df["speed_bleeds"], label=name, color=color, marker='^', linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Speed Bleeds")
    ax.set_title("Speed Bleeds Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final metrics bar chart
    ax = axes[1, 1]
    variant_list = [v for v in variant_names if v in results]
    max_dists = [results[v]['df']["champion_dist_m"].max() for v in variant_list]
    x_pos = range(len(variant_list))
    bars = ax.bar(x_pos, max_dists, color=colors[:len(variant_list)])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variant_list, rotation=45, ha='right')
    ax.set_ylabel("Max Champion Distance (m)")
    ax.set_title("Final Performance Comparison")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, dist in zip(bars, max_dists):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{dist:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    plot_filename = f"lab_results_{spec_name}.png"
    plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
    print(f"Plot saved: {plot_filename}")
    
    plt.close()


if __name__ == "__main__":
    main()
