#!/usr/bin/env python3
"""Run a sequential, provenance-preserving, paired fly-control battery."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=15000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33])
    parser.add_argument("--arms", nargs="+", choices=["real_plus", "real_minus", "shuffle_plus"],
                        default=["real_plus", "real_minus", "shuffle_plus"])
    parser.add_argument("--gain", type=float, default=20.0)
    parser.add_argument("--source-snapshot", type=Path, help="existing source snapshot for a reused frozen binary")
    args = parser.parse_args()
    if args.ticks < 64 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("use at least 64 ticks and distinct seeds")
    if not math.isfinite(args.gain) or args.gain < 0 or len(set(args.arms)) != len(args.arms):
        parser.error("use a finite nonnegative gain and distinct arms")
    if any(seed < 0 or seed >= 2**64 for seed in args.seeds):
        parser.error("seeds must fit an unsigned 64-bit integer")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output.resolve() / "gpu_evolve.frozen"
    shutil.copy2(args.binary, frozen)
    maps = {"real": args.maps / "flymap_783_reflex_v3.bin",
            "shuffle": args.maps / "flymap_783_reflex_shuf0_v3.bin"}
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "binary_sha256": digest(frozen),
        "maps": {name: {"path": str(path), "sha256": digest(path)} for name, path in maps.items()},
        "seeds": args.seeds, "ticks": args.ticks, "beings": 40, "arms": args.arms, "gain": args.gain,
        "design": "exploratory paired fixed-genome control; no learning claim",
        "sampling": "every 64 ticks; exclude warmup episodes and tick < 64 from summaries",
        "stopping": "all planned runs, unless process failure or nonfinite measurements",
        "runs": [],
    }
    if args.source_snapshot:
        original = json.loads((args.source_snapshot / "battery.json").read_text())
        if original["binary_sha256"] != manifest["binary_sha256"]:
            raise ValueError("source snapshot belongs to a different executable")
        manifest["git_head"] = original["git_head"]
        for name in ["source.patch", "source-files.json"]:
            shutil.copy2(args.source_snapshot / name, args.output / name)
        shutil.copytree(args.source_snapshot / "source", args.output / "source")
    else:
        (args.output / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
        manifest["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        changed = subprocess.check_output(["git", "ls-files", "-m", "-o", "--exclude-standard"], text=True).splitlines()
        source_hashes = {}
        for name in changed:
            source = Path(name)
            if name.startswith(".claude/") or source.suffix not in {".rs", ".wgsl", ".py", ".md", ".html", ".js", ".toml"}:
                continue
            if not source.is_file():
                continue
            target = args.output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[name] = digest(target)
        (args.output / "source-files.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    manifest["source_snapshot"] = str(args.source_snapshot.resolve()) if args.source_snapshot else None
    # The lock serializes invocations of this harness, not unrelated GPU programs.
    with open(args.output.parent / ".fly-battery.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in args.seeds:
            for arm, map_name, sign in [("real_plus", "real", 1),
                                        ("real_minus", "real", -1),
                                        ("shuffle_plus", "shuffle", 1)]:
                if arm not in args.arms:
                    continue
                name = f"{arm}_s{seed}"
                run_dir = args.output.resolve() / name
                run_dir.mkdir()
                telemetry = run_dir / "telemetry.csv"
                command = [str(frozen), "--gens", "1", "--ticks", str(args.ticks),
                           "--beings", "40", "--width", "28", "--seed", str(seed),
                           "--start-formation", "--retina-beings", "--replay-every", "1",
                           "--fly-map", str(maps[map_name]), "--fly-brain", "--fly-eye",
                           "--fly-readout", "reflex", "--fly-readout-norm",
                           "--fly-reflex-alpha", "0.02", "--fly-reflex-gain", str(args.gain),
                           "--fly-cruise-throttle", "0.6", "--fly-reflex-sign", str(sign),
                           "--tracks", "8005", "--out", str(run_dir)]
                env = os.environ.copy()
                env["UNIVERSE_RETINA_TELEMETRY"] = str(telemetry)
                record = {"name": name, "seed": seed, "arm": arm, "command": command,
                          "telemetry": str(telemetry), "status": "running"}
                manifest["runs"].append(record)
                manifest_path = args.output / "battery.json"
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"START {name}", flush=True)
                start = time.monotonic()
                with open(run_dir / "run.log", "w") as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                record.update(returncode=result.returncode, seconds=time.monotonic() - start,
                              status="complete" if result.returncode == 0 else "failed")
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"END {name}: exit={result.returncode} seconds={record['seconds']:.1f}", flush=True)
                if result.returncode:
                    raise SystemExit(result.returncode)
                population = json.loads((run_dir / "population_gen0000.json").read_text())
                if len(population) != 40:
                    raise RuntimeError("missing population measurements")
                from analyze import summarize
                summarize(run_dir, args.ticks)
    print("COMPLETE: every scheduled run exited successfully", flush=True)


if __name__ == "__main__":
    main()
