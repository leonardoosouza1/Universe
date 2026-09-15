#!/usr/bin/env python3
"""Summarize complete paired fly runs without treating beings as independent seeds."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as stats


def correlation(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return stats.correlation(xs, ys)


def summarize(path, ticks):
    population = json.loads((path / "population_gen0000.json").read_text())
    with (path / "telemetry.csv").open() as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if any(not math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError(f"nonfinite telemetry: {path}")
    if any(not math.isfinite(p[key]) for p in population for key in
           ["progress_m", "distance_m", "fitness", "life_remaining_ticks", "x", "z"]):
        raise ValueError(f"nonfinite population: {path}")
    episode = max(row["episode"] for row in rows)
    rows = [r for r in rows if r["episode"] == episode and r["tick"] >= 64 and r["alive"]]
    if not rows:
        raise ValueError(f"no post-warmup living measurements: {path}")
    steering_correlations = []
    for being in range(len(population)):
        samples = [r for r in rows if r["being"] == being]
        corr = correlation([r["asymmetry"] for r in samples], [r["applied_steer"] for r in samples])
        if corr is not None:
            steering_correlations.append(corr)
    return {
        "population": len(population), "samples": len(rows),
        "signed_progress_median_m": stats.median(p["progress_m"] for p in population),
        "clipped_progress_median_m": stats.median(max(p["progress_m"], 0) for p in population),
        "distance_median_m": stats.median(p["distance_m"] for p in population),
        "low_progress_fraction": stats.mean(p["low_progress"] for p in population),
        "premature_exhaustion_fraction": stats.mean(p["life_exhausted"] and p["alive_ticks"] < ticks for p in population),
        "horizon_exhaustion_fraction": stats.mean(p["life_exhausted"] for p in population),
        "coverage_mean": stats.mean(r["coverage"] for r in rows),
        "coverage_median": stats.median(r["coverage"] for r in rows),
        "visible_fraction": stats.mean(r["coverage"] > 0 for r in rows),
        "overlap_sample_fraction": stats.mean(r["overlap_count"] > 0 for r in rows),
        "nearest_body_median_m": stats.median(r["body_distance"] for r in rows),
        "steer_saturation_fraction": stats.mean(abs(r["raw_steer"]) >= 0.99 for r in rows),
        "within_being_asymmetry_applied_steer_correlation_median": stats.median(steering_correlations) if steering_correlations else None,
        "beings_with_defined_correlation": len(steering_correlations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "battery.json").read_text())
    arms = manifest.get("arms", ["real_plus", "real_minus", "shuffle_plus"])
    planned = len(manifest["seeds"]) * len(arms)
    if len(manifest["runs"]) != planned or any(r["status"] != "complete" for r in manifest["runs"]):
        raise SystemExit("Battery is incomplete: refusing a final summary")
    result = {"runs": {}, "paired_differences_m": {}, "independent_seeds": len(manifest["seeds"])}
    for run in manifest["runs"]:
        result["runs"][run["name"]] = summarize(args.directory / run["name"], manifest["ticks"])
    for other in ["real_minus", "shuffle_plus"]:
        if "real_plus" not in arms or other not in arms:
            continue
        result["paired_differences_m"][f"real_plus_minus_{other}"] = [
            result["runs"][f"real_plus_s{s}"]["signed_progress_median_m"]
            - result["runs"][f"{other}_s{s}"]["signed_progress_median_m"] for s in manifest["seeds"]]
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Measured fly-control battery", "", f"One generation per run; {len(manifest['seeds'])} independent population seeds.", "",
             "| Run | Median signed progress (m) | Low progress | Premature exhaustion | Mean coverage | Visible samples | Sampled overlap |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in result["runs"].items():
        lines.append(f"| {name} | {r['signed_progress_median_m']:.2f} | {r['low_progress_fraction']:.1%} | "
                     f"{r['premature_exhaustion_fraction']:.1%} | {r['coverage_mean']:.2%} | "
                     f"{r['visible_fraction']:.1%} | {r['overlap_sample_fraction']:.2%} |")
    if result["paired_differences_m"]:
        lines += ["", "Paired differences in population median signed progress (metres):", ""]
    for name, differences in result["paired_differences_m"].items():
        lines.append(f"- {name}: {', '.join(f'{d:.2f}' for d in differences)}; mean {stats.mean(differences):.2f} m.")
    lines += ["", "Life exhaustion at the fixed horizon is expected, not premature mortality. Overlaps are sampled retinal body proxies, not collision events. Zero visibility is retained. Repeated ticks and beings share a world and are not independent replicates. No learning, evolutionary gain or connectome-specific avoidance is established by these comparisons.", ""]
    (args.directory / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
