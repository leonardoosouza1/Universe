#!/usr/bin/env python3
"""Reconstruct recorded fast-brain stages from a canonical genoma_dump export."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path


def f32(value: float) -> float:
    return ctypes.c_float(value).value


def read_matrix(path: Path, rows: int, columns: int) -> list[list[float]]:
    values = [f32(float(token)) for token in path.read_text().split()]
    expected = rows * columns
    if len(values) != expected:
        raise ValueError(f"{path} contains {len(values)} values; expected {expected}")
    return [values[start : start + columns] for start in range(0, expected, columns)]


def ordered_matmul_row(weights: list[float], inputs: list[int]) -> float:
    accumulator = f32(0.0)
    for weight, value in zip(weights, inputs, strict=True):
        product = f32(weight * f32(value))
        accumulator = f32(accumulator + product)
    return f32(accumulator * f32(1.0 / 127.0))


def activation(identifier: float, value: float) -> float:
    kind = int(f32(identifier + f32(0.5)))
    if kind == 1:
        return max(value, 0.0)
    if kind == 2:
        return value
    if kind == 3:
        return value if value >= 0.0 else f32(value * f32(0.01))
    return f32(math.tanh(value))


def maximum_error(left: list[float], right: list[float]) -> float:
    return max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)


def validate(observation_path: Path, dump_path: Path, sample_limit: int) -> dict[str, float | int]:
    l1 = read_matrix(dump_path / "l1.txt", 96, 192)
    l2 = read_matrix(dump_path / "l2.txt", 48, 96)
    activation_ids = [f32(float(value)) for value in (dump_path / "neuron_activation.txt").read_text().split()]
    if len(activation_ids) != 144:
        raise ValueError("neuron_activation.txt must contain 144 values")

    rows = []
    with observation_path.open() as source:
        for line in source:
            row = json.loads(line)
            if row.get("neural_dispatch") and row.get("stages") is not None:
                rows.append(row)
                if len(rows) == sample_limit:
                    break
    if len(rows) < sample_limit:
        raise ValueError(f"found {len(rows)} fresh neural samples; requested {sample_limit}")

    result: dict[str, float | int] = {
        "samples": len(rows),
        "max_l1_matmul_error": 0.0,
        "max_l1_quantisation_error": 0,
        "max_l2_matmul_error": 0.0,
        "max_l2_std_error": 0.0,
        "max_final_intent_error": 0.0,
    }
    for row in rows:
        stages = row["stages"]
        cfc_input = [
            int(max(-127.0, min(127.0, f32(f32(math.tanh(f32(value))) * f32(127.0)))))
            for value in stages["cfc_state"]
        ]
        wave = [int(value) for value in row["wave"]]
        l1_input = cfc_input + wave[96:]
        reconstructed_l1 = [ordered_matmul_row(weights, l1_input) for weights in l1]
        result["max_l1_matmul_error"] = max(
            float(result["max_l1_matmul_error"]),
            maximum_error(reconstructed_l1, stages["l1_matmul"]),
        )

        reconstructed_quantised = [
            int(max(-127.0, min(127.0, f32(activation(activation_ids[index], f32(value)) * f32(127.0)))))
            for index, value in enumerate(stages["l1_effective"])
        ]
        result["max_l1_quantisation_error"] = max(
            int(result["max_l1_quantisation_error"]),
            max(
                abs(a - b)
                for a, b in zip(reconstructed_quantised, stages["l1_to_l2"], strict=True)
            ),
        )

        l2_input = [int(value) for value in stages["l1_to_l2"]]
        reconstructed_l2 = [ordered_matmul_row(weights, l2_input) for weights in l2]
        result["max_l2_matmul_error"] = max(
            float(result["max_l2_matmul_error"]),
            maximum_error(reconstructed_l2, stages["l2_matmul"]),
        )

        total = f32(0.0)
        total_squared = f32(0.0)
        for value in map(f32, stages["l2_effective"]):
            total = f32(total + value)
            total_squared = f32(total_squared + f32(value * value))
        mean = f32(total / f32(48.0))
        variance = max(0.0, f32(f32(total_squared / f32(48.0)) - f32(mean * mean)))
        reconstructed_std = max(f32(math.sqrt(variance)), f32(0.001))
        result["max_l2_std_error"] = max(
            float(result["max_l2_std_error"]),
            abs(reconstructed_std - stages["l2_std"]),
        )

        reconstructed_intent = [
            max(-1.0, min(1.0, activation(activation_ids[96 + index], f32(value))))
            for index, value in enumerate(stages["l2_adapted"])
        ]
        reconstructed_intent.extend(f32(math.tanh(f32(value))) for value in stages["bypass_adapted"])
        result["max_final_intent_error"] = max(
            float(result["max_final_intent_error"]),
            maximum_error(reconstructed_intent, stages["final_intent"]),
        )

        if stages["final_intent"][:3] != row["body"]["raw_motor"]:
            raise ValueError(f"tick {row['tick']} has a motor/trace phase mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--genome-dump", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")

    result = validate(args.observation, args.genome_dump, args.samples)
    print(json.dumps(result, indent=2, sort_keys=True))
    errors = [
        float(result["max_l1_matmul_error"]),
        float(result["max_l2_matmul_error"]),
        float(result["max_l2_std_error"]),
        float(result["max_final_intent_error"]),
    ]
    if int(result["max_l1_quantisation_error"]) != 0 or max(errors) > args.tolerance:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
