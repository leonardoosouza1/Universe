#!/usr/bin/env python3
"""Record and validate a native champion replay in a new artifact directory.

The checkpoint's time constants are preserved. No random population, pilot,
or training generation is substituted when a checkpoint fails to load.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def control_arguments(args):
    """Keep each experimental intervention independently selectable and recorded."""
    result = []
    for attribute, flag in (("motor_head", "--motor-head"),
                            ("throttle_arbitration", "--throttle-arbitration"),
                            ("slow_kinematics", "--slow-kinematics"),
                            ("wall_probes", "--wall-probes")):
        value = getattr(args, attribute, None)
        if value is not None:
            result.extend([flag, value])
    if getattr(args, "reverse_controller_v2", False):
        result.append("--reverse-controller-v2")
    return result


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    checkpoint = args.checkpoint.resolve(strict=True)
    binary = ROOT / "target/release/gpu_evolve"
    viewer = ROOT / "target/release/replay_view"
    if not binary.is_file() or not viewer.is_file():
        raise ValueError("build first: cargo build --release --bin gpu_evolve --bin replay_view")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    shutil.copy2(checkpoint, inputs / "champion.bin")
    shutil.copy2(binary, inputs / "gpu_evolve")
    shutil.copy2(viewer, inputs / "replay_view")
    # The existing resume loader discovers checkpoints by filename. Keep the
    # original separately because evolution saves its output under this name.
    shutil.copy2(checkpoint, output / "evolve_best_gen0000.bin")
    command = [
        str(inputs / "gpu_evolve"), "--gens", "1", "--ticks", str(args.ticks),
        "--beings", str(args.beings), "--width", str(args.width), "--seed", str(args.seed),
        "--resume", "--resume-mix", "1", "--same-start", "--replay-every", "1",
        "--tracks", str(args.track), "--no-hippocampus", "--fuel", "--out", str(output),
    ]
    if args.driving_diagnostics:
        command += ["--driving-diagnostics", "--vision-being", str(args.vision_being), "--vision-every", str(args.vision_every)]
    if args.retina_road_view is not None:
        command += ["--retina-road-view", str(args.retina_road_view)]
    if args.erase_fast_memory:
        command.append("--erase-fast-memory")
    command += control_arguments(args)
    manifest = {
        "schema_version": 1, "purpose": "native champion navigation evaluation",
        "checkpoint_source": str(checkpoint), "checkpoint_sha256": digest(inputs / "champion.bin"),
        "binary_sha256": digest(inputs / "gpu_evolve"), "viewer_sha256": digest(inputs / "replay_view"),
        "controller": "native_neural", "population": "exact clones; identical centreline start",
        "time_constants": "loaded; no reseeding", "command": command,
        "acceptance": {"min_laps": args.min_laps, "min_fraction": args.min_fraction},
    }
    manifest_path = output / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    start = time.monotonic()
    print(f"Recording {args.beings} beings, {args.ticks / 64:.2f} simulated seconds on track {args.track}.", flush=True)
    with (output / "run.log").open("w") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(simulation_exit=process.returncode, wall_seconds=time.monotonic() - start)
    save()
    if process.returncode:
        raise ValueError(f"simulation failed: {output / 'run.log'}")
    metadata = json.loads((output / "gen0000.meta.json").read_text())
    if not metadata.get("checkpoint_loaded") or metadata.get("tau_reseeded"):
        raise ValueError("simulation did not confirm the checkpoint identity/time-constant contract")
    metadata.update(checkpoint_sha256=manifest["checkpoint_sha256"], checkpoint_source=str(checkpoint),
                    binary_sha256=manifest["binary_sha256"], viewer_sha256=manifest["viewer_sha256"])
    (output / "gen0000.meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    export = [str(inputs / "replay_view"), "--replay", str(output / "gen0000.replay"),
              "--html", str(output / "replay-unvalidated.html"),
              "--native-genome", str(inputs / "champion.bin"),
              "--native-being", str(args.vision_being)]
    if args.driving_diagnostics:
        report = output / "report"
        subprocess.run([sys.executable, str(ROOT / "scripts/replay/driving_report.py"), "--run", str(output), "--output", str(report)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/replay/checkup_report.py"),
                        "--run", str(output), "--output", str(output / "control-checkup.json")], cwd=ROOT, check=True)
        export += ["--diagnostics", str(report / "report.json")]
    manifest["export_command"] = export
    save()
    subprocess.run(export, cwd=ROOT, check=True)
    validate = [sys.executable, str(ROOT / "scripts/fly/validate_replay.py"),
                str(output / "replay-unvalidated.html"), "--min-laps", str(args.min_laps),
                "--min-fraction", str(args.min_fraction), "--annotate-html", str(output / "replay.html")]
    with (output / "behavior-validation.json").open("w") as report:
        validation = subprocess.run(validate, cwd=ROOT, stdout=report)
    manifest.update(validation_command=validate, validation_exit=validation.returncode)
    # The annotated HTML retains all embedded measurements, including failed gates.
    # Keep one canonical export; a second population-sized HTML wastes disk space.
    if (output / "replay.html").is_file():
        (output / "replay-unvalidated.html").unlink()
        manifest["canonical_replay_html"] = "replay.html"
        manifest["retired_intermediates"] = ["replay-unvalidated.html"]
    save()
    if validation.returncode:
        raise ValueError(f"behavioral gate FAILED; inspect {output / 'behavior-validation.json'}")
    result = json.loads((output / "behavior-validation.json").read_text())
    print(f"LAP GATE PASS: {result['passed']}/{result['total']} beings completed at least {args.min_laps} laps. This does not establish braking or wall anticipation.")
    print(output / "replay.html")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--track", type=int, default=8005)
    parser.add_argument("--ticks", type=int, default=30000)
    parser.add_argument("--beings", type=int, default=16)
    parser.add_argument("--width", type=float, default=28)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--min-laps", type=int, default=2)
    parser.add_argument("--min-fraction", type=float, default=0.8)
    parser.add_argument("--driving-diagnostics", action="store_true")
    parser.add_argument("--vision-being", type=int, default=0)
    parser.add_argument("--vision-every", type=int, default=16)
    parser.add_argument("--erase-fast-memory", action="store_true")
    parser.add_argument("--retina-road-view", type=float, choices=None)
    parser.add_argument("--motor-head", choices=("spline", "linear"))
    parser.add_argument("--throttle-arbitration", choices=("legacy", "signed"))
    parser.add_argument("--slow-kinematics", choices=("legacy", "canonical"))
    parser.add_argument("--wall-probes", choices=("legacy", "continuous"))
    parser.add_argument("--reverse-controller-v2", action="store_true")
    args = parser.parse_args()
    if args.retina_road_view is not None and not 20 <= args.retina_road_view <= 300:
        parser.error("road-view range must be in [20, 300] metres")
    if not 0 <= args.vision_being < args.beings or args.vision_every < 1:
        parser.error("invalid observation being or cadence")
    if args.ticks < 8 or not 1 <= args.beings <= 255 or not 0 < args.width < 1000:
        parser.error("ticks >= 8, beings in [1,255], and width in (0,1000) are required")
    if args.min_laps < 1 or not 0 < args.min_fraction <= 1 or args.track < 0 or args.seed < 0:
        parser.error("invalid acceptance threshold or seed")
    try:
        return run(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"run_champion: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
