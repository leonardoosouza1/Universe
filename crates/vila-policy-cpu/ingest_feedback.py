#!/usr/bin/env python3
"""Turn one closed Paper pilot episode into a matched Rust policy update.

This is an opt-in, one-shot research bridge. It never reads an active episode
or replaces an approved model/state.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def fail(message):
    raise ValueError(message)


def read_json(path):
    if path.stat().st_size > 1_000_000:
        fail(f"oversize JSON: {path}")
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        fail(message)


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"invalid {name}")
    return float(value)


def events_for_window(episode_dir, start_tick, end_tick, episode_start, day_ticks):
    first = (start_tick - episode_start) // day_ticks
    last = (end_tick - episode_start) // day_ticks
    require(0 <= first <= last <= first + 1, "unsupported event window")
    events = []
    for day in range(first, last + 1):
        path = episode_dir / f"day_{day:04d}" / "events.jsonl.gz"
        require(path.is_file(), f"missing event stream: {path}")
        require(path.stat().st_size <= 10_000_000, "oversize event stream")
        with gzip.open(path, "rt") as stream:
            for line in stream:
                require(len(line) <= 100_000 and len(events) < 100_000,
                        "event stream limit exceeded")
                event = json.loads(line)
                require(type(event.get("tick")) is int, "event missing tick")
                if start_tick <= event["tick"] <= end_tick:
                    events.append(event)
    return events


def matched_outcome(record, pilot, episode_dir):
    header = read_json(episode_dir / "episode.json")
    closed = read_json(episode_dir / "episode_end.json")
    decision = record["decision"]
    require(record.get("schema") == "vila-pilot-dispatch/v1", "dispatch schema mismatch")
    require(header.get("id") == decision["episode"] == pilot.get("episode") == episode_dir.name,
            "episode mismatch")
    require(header.get("ai_pilot_enabled") is True and header.get("recorder_enabled") is True,
            "pilot or recorder disabled")
    require(closed.get("lines_dropped") == 0, "incomplete episode: dropped lines")
    require(type(closed.get("end_tick")) is int and type(header.get("start_tick")) is int,
            "invalid episode ticks")
    require(type(header.get("day_ticks")) is int and header["day_ticks"] > 0,
            "invalid day length")
    require(record.get("seq") == pilot.get("seq") and type(pilot.get("seq")) is int,
            "sequence mismatch")
    require(decision.get("villager_uuid") == pilot.get("villager_uuid"), "villager mismatch")
    require(decision.get("action") == pilot.get("action") == "walk_to", "action mismatch")
    for axis in "xyz":
        require(abs(number(decision[axis], axis) - number(pilot[axis], axis)) <= 1e-6,
                "target mismatch")
    require(type(decision.get("tick")) is int and type(pilot.get("issued_tick")) is int
            and type(pilot.get("expires_tick")) is int,
            "invalid command ticks")
    require(header["start_tick"] <= decision["tick"] <= pilot["issued_tick"]
            <= pilot["expires_tick"] < closed["end_tick"], "command outside episode")
    events = events_for_window(episode_dir, pilot["issued_tick"], closed["end_tick"],
                               header["start_tick"], header["day_ticks"])
    sequence = pilot["seq"]
    actions = [e for e in events if e.get("type") == "pilot_action" and e.get("seq") == sequence]
    outcomes = [e for e in events if e.get("type") == "pilot_outcome" and e.get("seq") == sequence]
    rejects = [e for e in events if e.get("type") == "pilot_reject" and e.get("seq") == sequence]
    require(len(actions) == len(outcomes) == 1 and not rejects, "missing/duplicate/rejected action")
    action, result = actions[0], outcomes[0]
    for event in (action, result):
        require(event.get("episode") == header["id"]
                and event.get("villager_uuid") == decision["villager_uuid"]
                and type(event.get("v")) is int and event["v"] >= 0,
                "event identity missing or mismatched")
    require(action["v"] == result["v"], "villager index mismatch")
    require(action.get("action") == "walk_to" and action.get("path_accepted") is True,
            "action not accepted")
    require(type(result.get("action_tick")) is int
            and result["action_tick"] == action["tick"]
            and pilot["issued_tick"] <= action["tick"] <= pilot["expires_tick"]
            and decision["tick"] <= action["tick"] <= decision["tick"] + 40
            and action["tick"] < result["tick"] <= action["tick"] + 400,
            "action/outcome tick mismatch")
    require(result.get("reason") in ("deadline", "sleeping")
            and result.get("villager_available") is True, "unusable outcome")
    require(type(result.get("slept_during")) is bool
            and result["slept_during"] == (result["reason"] == "sleeping"),
            "sleep outcome missing or inconsistent")
    for axis in "xyz":
        require(abs(number(action[f"target_{axis}"], f"target_{axis}")
                    - number(decision[axis], axis)) <= 1e-6, "applied target mismatch")
    start_distance = number(result.get("start_distance"), "start_distance")
    remaining = number(result.get("remaining_distance"), "remaining_distance")
    progress = number(result.get("progress"), "progress")
    require(start_distance > 0 and remaining >= 0
            and abs(start_distance - number(action.get("start_distance"), "action start_distance")) <= 1e-6
            and abs(start_distance - remaining - progress) <= 1e-5,
            "distance/progress mismatch")
    health_before = number(result.get("health_before"), "health_before")
    health_after = number(result.get("health_after"), "health_after")
    require(0 <= health_before <= 20 and 0 <= health_after <= 20,
            "invalid health")
    require(type(result.get("arrived")) is bool
            and type(result.get("had_home_before")) is bool
            and type(result.get("has_home_after")) is bool,
            "missing task outcome")
    rescued = any(e.get("type") == "rescue" and e.get("v") == action["v"]
                  and action["tick"] < e["tick"] <= result["tick"] for e in events)
    outcome = dict(episode=header["id"], villager_uuid=decision["villager_uuid"],
                   decision_tick=decision["tick"], action_tick=action["tick"],
                   end_tick=result["tick"], candidate_id=decision["candidate_id"],
                   arrived=result["arrived"],
                   home_acquired=not result["had_home_before"] and result["has_home_after"],
                   slept=result["slept_during"],
                   progress_ratio=max(-1.0, min(1.0, progress / start_distance)),
                   health_lost=max(0.0, health_before - health_after), rescued=rescued)
    return outcome, action, result


def publish_new(path, payload):
    require(not path.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def ingest(decision_path, pilot_path, episode_dir, policy_bin, state_path, receipt_path):
    require(state_path.parent == receipt_path.parent == decision_path.parent == pilot_path.parent
            and state_path.name == "state.json" and receipt_path.name == "feedback.json",
            "feedback outputs must be fixed beside the dispatch record")
    require(not state_path.exists() and not receipt_path.exists(), "feedback already published")
    record, pilot = read_json(decision_path), read_json(pilot_path)
    outcome, action, result = matched_outcome(record, pilot, episode_dir)
    request = dict(op="learn", state=record["state"],
                   decision=record["decision"], outcome=outcome)
    completed = subprocess.run([str(policy_bin)], input=json.dumps(request),
                               text=True, capture_output=True, timeout=10, check=False)
    require(completed.returncode == 0, f"Rust learn rejected feedback: {completed.stderr.strip()}")
    response = json.loads(completed.stdout)
    require(response.get("op") == "learn"
            and response["state"]["updates"] == record["state"]["updates"] + 1
            and math.isfinite(response["reward"]), "invalid Rust learn response")
    receipt = dict(schema="vila-pilot-feedback/v1", episode=outcome["episode"],
                   seq=pilot["seq"], outcome=outcome, reward=response["reward"],
                   action_tick=action["tick"], end_tick=result["tick"],
                   decision_sha256=hashlib.sha256(decision_path.read_bytes()).hexdigest(),
                   pilot_sha256=hashlib.sha256(pilot_path.read_bytes()).hexdigest())
    publish_new(state_path, response["state"])
    publish_new(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("decision", "pilot", "episode_dir", "policy_bin", "state_out", "receipt_out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    try:
        receipt = ingest(args.decision, args.pilot, args.episode_dir, args.policy_bin,
                         args.state_out, args.receipt_out)
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired,
            gzip.BadGzipFile) as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
