#!/usr/bin/env python3
"""Build a self-contained static report from gpu_evolve driving diagnostics."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def _num(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"non-numeric measurement: {value!r}")
    if not math.isfinite(result):
        raise ValueError(f"non-finite measurement: {value!r}")
    return result


def load_run(run):
    meta = json.loads((run / "driving.meta.json").read_text(encoding="utf-8"))
    with (run / "driving.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"generation", "first_tick", "last_tick", "being", "ticks", "alive_ticks",
                "contact_ticks", "impact_ticks", "brake_ticks", "speed_sum", "distance_m",
                "collision_energy_j", "throttle_sum", "brake_sum"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("driving.csv is missing required diagnostic columns")
    seen_windows = set()
    previous_window = {}
    for row in rows:
        key = (int(row["generation"]), int(row["first_tick"]), int(row["last_tick"]), int(row["being"]))
        if key in seen_windows:
            raise ValueError(f"duplicate CSV window: {key}")
        seen_windows.add(key)
        if int(row["last_tick"]) < int(row["first_tick"]) or int(row["ticks"]) <= 0:
            raise ValueError(f"invalid CSV window: {key}")
        stream = (key[0], key[3])
        if stream in previous_window and key[1] <= previous_window[stream]:
            raise ValueError(f"CSV windows out of order: {key}")
        previous_window[stream] = key[2]
        for name, value in row.items():
            if name not in {"generation", "first_tick", "last_tick", "being", "ticks", "alive_ticks",
                            "contact_ticks", "impact_ticks", "contact_events", "brake_ticks",
                            "overlap_ticks", "stationary_ticks", "reverse_ticks", "near_wall_ticks"}:
                _num(value)
    observations = []
    sample_path = run / "observation.jsonl"
    if sample_path.exists():
        seen_observations = set()
        previous_observation = None
        for line_no, line in enumerate(sample_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                observations.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid observation.jsonl line {line_no}: {exc}") from exc
            key = (observations[-1].get("generation", 0), observations[-1].get("tick"), observations[-1].get("being"))
            if key in seen_observations:
                raise ValueError(f"duplicate observation: {key}")
            if previous_observation is not None and key[:2] < previous_observation[:2]:
                raise ValueError(f"observations out of order: {key}")
            seen_observations.add(key)
            previous_observation = key
            for field in ("tick", "being"):
                if not isinstance(observations[-1].get(field), (int, float)) or not math.isfinite(float(observations[-1][field])):
                    raise ValueError(f"invalid observation {field} at line {line_no}")
            for field in ("retina", "wave", "fast_state"):
                values = observations[-1].get(field)
                if values is not None and (not isinstance(values, list) or any(not math.isfinite(float(v)) for v in values)):
                    raise ValueError(f"invalid observation {field} at line {line_no}")
    return meta, rows, observations


def make_report(run, generation=None):
    meta, raw_rows, observations = load_run(run)
    generations = {int(row["generation"]) for row in raw_rows}
    if generation is None:
        if len(generations) != 1:
            raise ValueError("multiple generations recorded: select one with --generation")
        generation = next(iter(generations))
    if generation not in generations:
        raise ValueError("requested generation was not recorded")
    raw_rows = [row for row in raw_rows if int(row["generation"]) == generation]
    observations = [row for row in observations if row["generation"] == generation]
    beings = {}
    windows = []
    for row in raw_rows:
        being = int(row["being"])
        ticks = int(row["ticks"])
        alive = int(row["alive_ticks"])
        record = {
            "generation": int(row["generation"]), "first_tick": int(row["first_tick"]),
            "last_tick": int(row["last_tick"]), "being": being, "ticks": ticks,
            "alive_ticks": alive, "alive_fraction": alive / ticks if ticks else 0.0,
            "contact_ticks": int(row["contact_ticks"]),
            "contact_fraction_alive": int(row["contact_ticks"]) / alive if alive else 0.0,
            "impact_ticks": int(row["impact_ticks"]), "contact_events": int(row["contact_events"]),
            "brake_ticks": int(row["brake_ticks"]),
            "brake_fraction_alive": int(row["brake_ticks"]) / alive if alive else 0.0,
            "overlap_ticks": int(row["overlap_ticks"]), "stationary_ticks": int(row["stationary_ticks"]),
            "reverse_ticks": int(row["reverse_ticks"]), "near_wall_ticks": int(row["near_wall_ticks"]),
            "mean_speed_mps": _num(row["speed_sum"]) / alive if alive else 0.0,
            "distance_m": _num(row["distance_m"]), "collision_energy_j": _num(row["collision_energy_j"]),
            "impulse_ns": _num(row["impulse_ns"]), "impact_peak_mps": _num(row["impact_peak_mps"]),
            "slip_peak": _num(row["slip_peak"]), "slip_angle_peak_rad": _num(row["slip_angle_peak_rad"]),
            "mean_throttle": _num(row["throttle_sum"]) / alive if alive else 0.0,
            "mean_brake": _num(row["brake_sum"]) / alive if alive else 0.0,
            "position": [_num(row["x"]), _num(row["z"])], "speed_mps": _num(row["speed"]),
            "throttle": _num(row["throttle"]), "steer": _num(row["steer"]), "brake": _num(row["brake"]),
            "wall_left_m": _num(row["wall_left"]), "wall_front_m": _num(row["wall_front"]),
            "wall_right_m": _num(row["wall_right"]), "life": _num(row["life"]), "fuel": _num(row["fuel"]),
        }
        beings.setdefault(str(being), []).append(record)
        windows.append(record)
    for values in beings.values():
        values.sort(key=lambda item: (item["generation"], item["last_tick"]))
    observations.sort(key=lambda item: (item.get("generation", 0), item.get("tick", 0)))
    summary_by_being = {}
    for being, values in beings.items():
        alive_ticks = sum(v["alive_ticks"] for v in values)
        ticks = sum(v["ticks"] for v in values)
        summary_by_being[being] = {
            "alive_ticks": alive_ticks, "ticks": ticks,
            "alive_fraction": alive_ticks / ticks if ticks else 0.0,
            "contact_ticks_alive": sum(v["contact_ticks"] for v in values),
            "impact_ticks": sum(v["impact_ticks"] for v in values),
            "stationary_ticks_alive": sum(v["stationary_ticks"] for v in values),
            "brake_ticks_alive": sum(v["brake_ticks"] for v in values),
            "mean_speed_mps_alive": (sum(v["mean_speed_mps"] * v["alive_ticks"] for v in values) / alive_ticks
                                     if alive_ticks else 0.0),
            "collision_energy_j": sum(v["collision_energy_j"] for v in values),
        }
    raw = [o.get("body", {}).get("raw_motor") for o in observations
           if isinstance(o.get("body", {}).get("raw_motor"), list)]
    raw_summary = None
    if raw:
        raw_summary = {name: {"mean": sum(v[i] for v in raw) / len(raw),
                              "positive_fraction": sum(v[i] > 0.05 for v in raw) / len(raw),
                              "min": min(v[i] for v in raw), "max": max(v[i] for v in raw)}
                       for i, name in enumerate(("throttle", "steer", "brake"))}
    report = {
        "schema_version": 1, "generation": generation, "meta": meta, "run": run.name,
        "windows": windows, "beings": beings, "observations": observations,
        "summary_by_being": summary_by_being, "selected_raw_motor_summary": raw_summary,
        "input_sha256": {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in ("driving.meta.json", "driving.csv", "observation.jsonl") if (run / name).exists()
        },
    }
    return report


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    # The existing replay player consumes this payload; no parallel viewer is generated.
    safe = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    safe = safe.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    (output / "driving_data.js").write_text(
        "window.DRIVING_DATA=" + safe + ";\n", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation", type=int)
    args = parser.parse_args(argv)
    try:
        report = make_report(args.run, args.generation)
        write_report(report, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(args.output), "beings": len(report["beings"]),
                      "windows": len(report["windows"]), "observations": len(report["observations"])}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
