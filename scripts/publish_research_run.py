#!/usr/bin/env python3
"""Publish an immutable research-run copy for the existing Lab UI catalog.

This converts metadata only. It neither simulates nor fabricates observations.
Trace packaging/registration remains owned by the existing lab_api producer.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from run_research_battery import RUNS_ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("name", help="new, unique catalog directory name")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_relative_to(RUNS_ROOT) or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("source must be a run below universe-runs; name must be one directory component")
    report = json.loads((source.parent / "report.json").read_text())
    matches = [r for r in report["runs"] if r["name"] == source.name]
    if report["status"] != "completed" or len(matches) != 1:
        parser.error("publication requires a uniquely identified completed research run")
    result = matches[0]
    provenance = json.loads((source.parent / "manifest.json").read_text())
    # A run only reaches the catalog through the GPU contract gate: the battery
    # runner records the receipt it executed under, and the receipt must still be
    # the same file (a hand-built manifest has no receipt, a moved one no longer hashes).
    gate = provenance.get("gpu_contract_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("path"), str):
        parser.error("manifest.json records no gpu_contract_gate; only gate-executed runs are published")
    receipt_path = Path(gate["path"])
    if not receipt_path.is_file() or sha(receipt_path) != gate.get("sha256"):
        parser.error(f"gate receipt missing or changed since execution: {receipt_path}")
    command = json.loads((source / "command.json").read_text())
    # Match the wrapper's CLI dictionary without silently inventing defaults.
    values = {}
    index = 1
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            parser.error(f"unexpected positional command argument: {token}")
        value = True
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            index += 1
            value = command[index]
        values[token[2:]] = value
        index += 1
    for key in ("gens", "ticks", "beings", "seed"):
        values[key] = int(values[key])
    populations = sorted(source.glob("population_gen*.json"))
    if len(populations) != values["gens"]:
        parser.error("population evidence does not cover the requested generations")
    destination = RUNS_ROOT / args.name
    destination.mkdir(exist_ok=False)
    copied = {}
    # Recursive: skills/reinjected-*.bin and diagnostic_candidates/ are evidence too,
    # and "byte_identical_catalog_copy" must mean the whole directory.
    for path in sorted(source.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            digest = sha(path)
            if sha(target) != digest:
                raise ValueError(f"publication copy differs: {relative}")
            copied[str(relative)] = digest
    manifest = dict(id=args.name, written=datetime.now(timezone.utc).isoformat(),
                    args=values, command=command, simulation_exit=0,
                    universe_commit=provenance["base_commit"],
                    source_sha256=provenance["source_sha256"],
                    binary_sha256=provenance["binary_sha256"],
                    gpu_contract_gate=gate,
                    publication=dict(source=str(source), files_sha256=copied,
                                     kind="byte_identical_catalog_copy",
                                     scope="short full-service course; not general city competence"),
                    research_result=result)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
