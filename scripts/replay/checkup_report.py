#!/usr/bin/env python3
"""Audit control intent and sensor information in an existing driving recording.

Population metrics use every recorded physics window. Neural metrics apply only
to sampled, living beings; they cannot establish a pre-impact causal response.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from driving_report import make_report


# Names verified against perception/channels.rs and the actual kernel writes.
CHANNELS = {
    64: "alignment", 65: "forward_speed", 70: "tire_temperature",
    71: "engine_temperature", 82: "retina_forward_proximity",
    102: "approach_left", 103: "approach_right", 104: "approach_front",
    105: "minimum_ttc_urgency", 106: "approach_left_fine",
    107: "approach_right_fine", 118: "forward_distance_normalized",
    122: "previous_brake", 123: "minimum_lateral_clearance_normalized",
}


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def motor_summary(samples):
    usable = [s for s in samples if all(
        isinstance(s.get("body", {}).get(key), list)
        and len(s["body"][key]) >= 3
        and all(isinstance(v, (float, int)) and math.isfinite(v)
                for v in s["body"][key][:3])
        for key in ("raw_motor", "applied_motor"))]
    n = len(usable)
    traces = [s["body"]["motor_trace"] for s in usable
              if isinstance(s["body"].get("motor_trace"), dict)]
    return {
        "samples": len(samples), "usable_motor_samples": n,
        "raw_brake_positive_count": sum(s["body"]["raw_motor"][2] > 0 for s in usable),
        "raw_brake_over_0_05_count": sum(s["body"]["raw_motor"][2] > .05 for s in usable),
        "applied_brake_over_0_05_count": sum(s["body"]["applied_motor"][2] > .05 for s in usable),
        "positive_raw_brake_but_zero_applied_count": sum(
            s["body"]["raw_motor"][2] > .05 and s["body"]["applied_motor"][2] == 0
            for s in usable),
        "raw_brake_mean": fraction(sum(s["body"]["raw_motor"][2] for s in usable), n),
        "raw_throttle_mean": fraction(sum(s["body"]["raw_motor"][0] for s in usable), n),
        "applied_throttle_mean": fraction(sum(s["body"]["applied_motor"][0] for s in usable), n),
        "negative_raw_throttle_but_applied_over_0_1_count": sum(
            s["body"]["raw_motor"][0] < 0 and s["body"]["applied_motor"][0] > .1 for s in usable),
        "slow_output_mean": [fraction(sum(s["body"]["slow"][i] for s in usable
                                       if len(s["body"].get("slow", [])) >= 3),
                                      sum(len(s["body"].get("slow", [])) >= 3 for s in usable))
                             for i in range(3)],
        "recorded_brake_suppression": dict(Counter(t.get("brake_suppression", "unspecified") for t in traces)),
        "recorded_override_cause": dict(Counter(t.get("override_cause", "unspecified") for t in traces)),
        "reason_attribution": "Requires motor_trace; post-physics speed cannot identify the suppressing gate.",
    }


def body_summary(samples):
    fields = (
        "damage", "engine_temperature_c", "oil_temperature_c", "rpm_fraction",
        "tyre_temperature_native", "tyre_temperature_model", "terrain_grip",
        "environment_grip", "slip_ratio_peak", "slip_angle_peak_rad", "fuel", "life",
    )
    result = {}
    for field in fields:
        values = [float(s["body"][field]) for s in samples
                  if isinstance(s.get("body", {}).get(field), (int, float))
                  and math.isfinite(float(s["body"][field]))]
        if values:
            result[field] = {
                "samples": len(values), "min": min(values), "mean": sum(values) / len(values),
                "max": max(values), "final": values[-1],
            }
    return result


def summarize(report):
    alive_samples = [s for s in report["observations"] if s.get("body", {}).get("alive") is True]
    covered = sorted({int(s["being"]) for s in report["observations"]})
    windows = report["windows"]
    alive_ticks = sum(w["alive_ticks"] for w in windows)
    distance_m = sum(w["distance_m"] for w in windows)
    totals = {name: sum(w[name] for w in windows) for name in (
        "brake_ticks", "contact_ticks", "impact_ticks", "stationary_ticks",
        "reverse_ticks", "collision_energy_j")}
    per_being = {}
    for being in covered:
        samples = [s for s in alive_samples if int(s["being"]) == being]
        fast = [s for s in samples if abs(s["body"].get("speed", 0)) > 10]
        sensors = {}
        for index, name in CHANNELS.items():
            values = [s["wave"][index] for s in samples if len(s.get("wave", [])) > index]
            counts = Counter(values)
            sensors[str(index)] = {
                "name": name, "samples": len(values), "unique_values": len(counts),
                "min": min(values) if values else None, "max": max(values) if values else None,
                "constant_in_recording": len(counts) == 1,
                "zero_fraction": fraction(counts[0], len(values)),
                "endpoint_fraction": fraction(counts[-127] + counts[127], len(values)),
                "nonzero_values_only_endpoints": bool(values) and any(v != 0 for v in values)
                    and all(v in (-127, 0, 127) for v in values),
            }
        per_being[str(being)] = {
            "alive": motor_summary(samples),
            "post_physics_speed_abs_over_10_mps": motor_summary(fast),
            "body": body_summary(samples),
            "channels": sensors,
            "motor_trace_samples": sum(isinstance(s["body"].get("motor_trace"), dict) for s in samples),
        }
    return {
        "schema_version": 1, "run": report["run"], "generation": report["generation"],
        "input_sha256": report["input_sha256"],
        "population": {
            "recorded_beings": len(report["beings"]), "alive_ticks": alive_ticks,
            "distance_m": distance_m, **totals,
            "brake_fraction_alive": fraction(totals["brake_ticks"], alive_ticks),
            "contact_fraction_alive": fraction(totals["contact_ticks"], alive_ticks),
            "stationary_fraction_alive": fraction(totals["stationary_ticks"], alive_ticks),
            "impact_ticks_per_km": fraction(1000 * totals["impact_ticks"], distance_m),
        },
        "neural_coverage": {
            "configured_selected_being": report["meta"].get("selected_being"),
            "observed_beings": covered, "sample_every_ticks": report["meta"].get("sample_every"),
            "sample_phase": report["meta"].get("sample_phase"),
            "selection_is_best": False,
            "selection_note": "The configured index is recorded; no ranking or automatic champion selection.",
        },
        "sampled_beings": per_being,
        "limitations": [
            "Frozen replay analysis, not a new training or hardware experiment.",
            "Sparse neural samples are not population-wide evidence or exact impact timing.",
            "fast_state is sensory CfC state, not dense L1/L2 neuron activation.",
            "A constant recorded channel may have a valid source; constancy does not prove missing wiring.",
            "Channel 118 is front clearance (larger is clearer); channel 123 is lateral clearance.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--generation", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(make_report(args.run, args.generation))
    manifest = args.run / "manifest.json"
    if manifest.exists():
        provenance = json.loads(manifest.read_text())
        result["provenance"] = {k: provenance.get(k) for k in (
            "checkpoint_sha256", "purpose", "population", "controller", "command")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "population": result["population"],
                      "neural_coverage": result["neural_coverage"]}))


if __name__ == "__main__":
    main()
