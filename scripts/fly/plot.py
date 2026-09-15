#!/usr/bin/env python3
"""Plot population-seed comparisons; requires matplotlib, never an active GPU."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("followup", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = json.loads((args.primary / "summary.json").read_text())["runs"]
    followup = json.loads((args.followup / "summary.json").read_text())["runs"]
    seeds = json.loads((args.primary / "battery.json").read_text())["seeds"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    colors = ["#2274A5", "#E58606", "#8A4F7D"]
    for seed, color in zip(seeds, colors):
        axes[0, 0].plot([0, 1, 2], [primary[f"{arm}_s{seed}"]["signed_progress_median_m"]
                                    for arm in ["real_plus", "real_minus", "shuffle_plus"]],
                        "o-", color=color, label=f"Seed {seed}")
        base = primary[f"real_plus_s{seed}"]
        probe = followup[f"real_plus_s{seed}"]
        for axis, field, scale in [
            (axes[0, 1], "steer_saturation_fraction", 100),
            (axes[1, 0], "signed_progress_median_m", 1),
            (axes[1, 1], "overlap_sample_fraction", 100),
        ]:
            axis.plot([0, 1], [base[field] * scale, probe[field] * scale], "o-", color=color)
    axes[0, 0].set(title="Primary: signed population-median progress", ylabel="Metres",
                   xticks=[0, 1, 2], xticklabels=["Real +", "Real −", "Shuffled +"])
    axes[0, 0].legend(frameon=False)
    for axis in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        axis.set(xticks=[0, 1], xticklabels=["Gain 20", "Gain 1"])
    axes[0, 1].set(title="Follow-up: steering-input saturation", ylabel="Living samples (%)")
    axes[1, 0].set(title="Follow-up: signed population-median progress", ylabel="Metres")
    axes[1, 1].set(title="Follow-up: sampled retinal-body overlap", ylabel="Living samples (%)")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fly controller diagnostic battery · 3 paired seeds · 40 beings · 15,000 ticks", fontsize=13)
    fig.text(0.5, -0.035, "One point = one population seed. Frozen legacy hybrid executable; pre-fix whisker frame.\n"
             "One generation cannot establish learning. Sampled overlap is not a collision-event measurement.",
             ha="center", fontsize=9)
    for extension in ["png", "svg"]:
        fig.savefig(args.primary / f"comparison.{extension}", dpi=160, bbox_inches="tight")
    print(args.primary / "comparison.png")


if __name__ == "__main__":
    main()
