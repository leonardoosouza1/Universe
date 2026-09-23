#!/usr/bin/env python3
"""Validate a recorded GPU taxi witness without simulating anything on the host."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        assert math.isfinite(value), f"non-finite value at {path}: {value}"
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")


def validate(root: Path, required_deliveries: int, max_collision_j: float) -> dict[str, Any]:
    scenario = json.loads((root / "organism_gen0000.json").read_text())
    controller = scenario.get("controller", "")
    assert controller in {
        "organism-taxi-reference-witness-v1",
        "organism-taxi-reference-witness-v2",
    }, f"reference identity missing: {controller!r}"
    assert scenario.get("learning_evidence") is False, "witness must never be labelled learned"

    results = json.loads((root / "organism_result_gen0000.json").read_text())
    assert len(results) == 1, "witness requires exactly one being"
    outcome = results[0]
    finite_tree(outcome, "outcome")
    assert outcome["deliveries"] >= required_deliveries, (
        f"wanted {required_deliveries} deliveries, got {outcome['deliveries']}"
    )
    assert outcome["alive"] == 1, "body did not survive the witness"
    assert outcome["collision_energy_j"] <= max_collision_j, (
        f"collision energy {outcome['collision_energy_j']} J exceeds {max_collision_j} J"
    )
    assert outcome["traffic"][8] == 0, "red-light violation during witness"
    assert outcome["energy_j"] > 0 and outcome["water_kg"] > 0, "depleted resources"

    events: list[dict[str, Any]] = []
    samples = 0
    pedal_overlap = 0
    with (root / "observation.jsonl").open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            finite_tree(row, f"observation[{line_number}]")
            assert row["generation"] == 0 and row["being"] == 0
            command = row.get("body", {}).get("motor_trace", {}).get("final_command")
            if isinstance(command, list) and len(command) >= 3:
                pedal_overlap += int(command[0] > 1e-5 and command[2] > 1e-5)
            events.extend(row.get("service_events") or [])
            samples += 1

    assert samples > 0, "empty observation trace"
    assert pedal_overlap == 0, f"throttle/brake overlap in {pedal_overlap} samples"

    delivery_ids: list[int] = []
    state: dict[int, int] = {}
    phase_order = {
        "passenger_issued": 1,
        "pickup_accepted": 2,
        "dropoff_accepted": 3,
        "delivery_incremented": 4,
    }
    for event in events:
        phase = event.get("kind") or event.get("phase") or event.get("event")
        if phase not in phase_order:
            continue
        passenger = int(event.get("passenger_id", 0))
        assert passenger > 0, f"event without passenger identity: {event}"
        expected = state.get(passenger, 0) + 1
        assert phase_order[phase] == expected, (
            f"passenger {passenger} event order is invalid: expected phase {expected}, got {phase}"
        )
        state[passenger] = expected
        if phase == "delivery_incremented":
            assert passenger not in delivery_ids, f"duplicate delivery for passenger {passenger}"
            delivery_ids.append(passenger)

    assert len(delivery_ids) >= required_deliveries, (
        f"trace contains {len(delivery_ids)} complete event chains, wanted {required_deliveries}"
    )
    return {
        "controller": controller,
        "learning_evidence": False,
        "samples": samples,
        "deliveries": delivery_ids,
        "pedal_overlap_samples": pedal_overlap,
        "outcome": outcome,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--deliveries", type=int, default=1)
    parser.add_argument("--max-collision-j", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(validate(args.run, args.deliveries, args.max_collision_j), indent=2))
