#!/usr/bin/env python3
"""Build a compact native-brain replay from the verified calibrated run."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    "/media/llm-data/universe-runs/navigation-improvement-20260914/calibrated120-8005"
)
HEADER_SIZE = 64
ENTRY_SIZE = 130
FRAME_TICK_OFFSET = 2 + 40


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def compact_replay(source: Path, output: Path, max_tick: int) -> tuple[int, int, int]:
    data = source.read_bytes()
    if len(data) < HEADER_SIZE or data[:8] != b"IARARPLY":
        raise ValueError(f"not an IARA replay: {source}")
    n_beings = data[11]
    body = data[HEADER_SIZE:]
    if len(body) % ENTRY_SIZE:
        raise ValueError(f"truncated replay entry stream: {source}")
    kept = bytearray(data[:HEADER_SIZE])
    ticks: list[int] = []
    beings: set[int] = set()
    for start in range(0, len(body), ENTRY_SIZE):
        entry = body[start : start + ENTRY_SIZE]
        if entry[0] != 0xFE or entry[1] >= n_beings:
            raise ValueError(f"invalid replay entry at byte {HEADER_SIZE + start}")
        tick = struct.unpack_from("<I", entry, FRAME_TICK_OFFSET)[0]
        if tick <= max_tick:
            kept.extend(entry)
            ticks.append(tick)
            beings.add(entry[1])
    if not ticks or len(beings) != n_beings:
        raise ValueError("tick limit did not retain at least one frame for every being")
    output.write_bytes(kept)
    return len(ticks), min(ticks), max(ticks)


def compact_diagnostics(source: Path, output: Path, generation: int, max_tick: int) -> int:
    full = json.loads(source.read_text())
    observations = [
        row
        for row in full.get("observations", [])
        if row.get("generation") == generation and row.get("tick", max_tick + 1) <= max_tick
    ]
    windows = [
        row for row in full.get("windows", []) if row.get("last_tick", max_tick + 1) <= max_tick
    ]
    if not observations:
        raise ValueError("no diagnostic observations align with the compact replay")
    compact = {
        "schema_version": full.get("schema_version"),
        "generation": generation,
        "meta": full["meta"],
        "run": {
            "fixture": "prefix of a recorded run; full-run summaries intentionally omitted",
            "source_report_sha256": digest(source),
            "max_tick": max_tick,
        },
        "windows": windows,
        "beings": [],
        "observations": observations,
        "summary_by_being": {},
        "selected_raw_motor_summary": None,
    }
    output.write_text(json.dumps(compact, separators=(",", ":")))
    return len(observations)


def run(source: Path, output: Path, viewer: Path, max_tick: int) -> None:
    metadata_path = source / "gen0000.meta.json"
    replay_path = source / "gen0000.replay"
    report_path = source / "report" / "report.json"
    manifest_path = source / "manifest.json"
    for path in [metadata_path, replay_path, report_path, manifest_path, viewer]:
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    checkpoint = Path(metadata["checkpoint_source"])
    expected = metadata["checkpoint_sha256"]
    if manifest.get("population") != "exact clones; different formation positions":
        raise ValueError("manifest does not declare an exact checkpoint-clone population")
    if manifest.get("checkpoint_sha256") != expected or digest(checkpoint) != expected:
        raise ValueError("checkpoint identity does not match metadata and manifest")
    if metadata.get("checkpoint_mix") != 1.0 or metadata.get("tau_reseeded") is not False:
        raise ValueError("recording is not an unmodified checkpoint-clone population")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="universe-native-preview-") as temporary:
        temporary = Path(temporary)
        compact_path = temporary / "native-preview.replay"
        compact_meta = temporary / "native-preview.meta.json"
        compact_report = temporary / "report.json"
        frames, first_tick, last_tick = compact_replay(replay_path, compact_path, max_tick)
        fixture_meta = dict(metadata)
        fixture_meta.update(
            fixture_source_replay_sha256=digest(replay_path),
            fixture_tick_limit=max_tick,
            fixture_semantics="recorded prefix; no frame or neural value synthesized",
        )
        compact_meta.write_text(json.dumps(fixture_meta, indent=2) + "\n")
        observations = compact_diagnostics(
            report_path, compact_report, int(metadata["generation"]), max_tick
        )
        command = [
            str(viewer),
            "--replay",
            str(compact_path),
            "--html",
            str(output),
            "--diagnostics",
            str(compact_report),
            "--native-genome",
            str(checkpoint),
            "--native-being",
            str(full_selected_being(report_path)),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    print(
        f"{output} · {output.stat().st_size} bytes · {frames} recorded frames "
        f"across ticks {first_tick}..{last_tick} · {observations} recorded observations"
    )


def full_selected_being(report: Path) -> int:
    data = json.loads(report.read_text())
    value = data.get("meta", {}).get("selected_being")
    if not isinstance(value, int):
        raise ValueError("diagnostic report has no integer selected_being")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "target" / "checkup-20260915" / "native-preview.html",
    )
    parser.add_argument(
        "--viewer", type=Path, default=ROOT / "target" / "release" / "replay_view"
    )
    parser.add_argument("--max-tick", type=int, default=256)
    args = parser.parse_args()
    if args.max_tick <= 0:
        parser.error("--max-tick must be positive")
    run(args.source, args.output, args.viewer, args.max_tick)
