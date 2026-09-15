#!/usr/bin/env python3
"""Validate real track traversals in an exported replay_view HTML file."""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

PROJECTION_JUMP_TOLERANCE_M = 20.0
CORRIDOR_MARGIN_M = 3.0
INTEGRITY_ERRORS = {"incomplete_series", "nonfinite_or_malformed_frame",
                    "non_increasing_ticks", "truncated_series", "invalid_sample_interval"}


def _value(text, name):
    match = re.search(r"window\.REPLAY_%s\s*=\s*" % name, text)
    if not match:
        raise ValueError(f"missing window.REPLAY_{name}")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(text[match.end():])
    return value


def _project(point, track):
    x, y = point
    best = None
    for i, (a, b) in enumerate(zip(track, track[1:])):
        ax, ay = a; bx, by = b
        dx, dy = bx - ax, by - ay
        den = dx * dx + dy * dy
        if den <= 1e-12:
            continue
        q = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / den))
        px, py = ax + q * dx, ay + q * dy
        d2 = (x - px) ** 2 + (y - py) ** 2
        if best is None or d2 < best[0]:
            best = (d2, i, q)
    if best is None:
        raise ValueError("track has no non-degenerate segment")
    return best


def validate(data, min_laps=1, min_fraction=0.8, sectors=8, max_speed=100.0):
    track = data.get("track")
    beings = data.get("beings")
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    if not isinstance(track, list) or len(track) < 3 or not isinstance(beings, list) or not beings:
        raise ValueError("HTML must contain a non-empty TRACK and BEINGS")
    ids = [being.get("id", index) if isinstance(being, dict) else index for index, being in enumerate(beings)]
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in ids):
        raise ValueError("being IDs must be integers or strings")
    if len(set(map(str, ids))) != len(ids):
        raise ValueError("being IDs must be unique")
    if not all(isinstance(p, list) and len(p) >= 2 and all(math.isfinite(float(v)) for v in p[:2]) for p in track):
        raise ValueError("TRACK contains non-finite or malformed points")
    points = [(float(p[0]), float(p[1])) for p in track]
    if points[0] != points[-1]:
        points.append(points[0])
    lengths = []
    for a, b in zip(points, points[1:]):
        lengths.append(math.hypot(b[0] - a[0], b[1] - a[1]))
    lap_length = sum(lengths)
    if lap_length <= 0 or sectors < 1:
        raise ValueError("track has no measurable lap length")
    tick_rate = float(meta.get("tick_rate", 64.0))
    sample_every = meta.get("sample_every")
    track_width = meta.get("track_width")
    if not math.isfinite(tick_rate) or tick_rate <= 0:
        raise ValueError("META tick_rate must be positive")
    if sample_every is not None and (not math.isfinite(float(sample_every)) or float(sample_every) <= 0 or not float(sample_every).is_integer()):
        raise ValueError("META sample_every must be a positive integer")
    if track_width is not None and (not math.isfinite(float(track_width)) or float(track_width) <= 0):
        raise ValueError("META track_width must be positive")
    corridor = None if track_width is None else float(track_width) / 2.0 + CORRIDOR_MARGIN_M
    planned_ticks = meta.get("planned_ticks")
    if planned_ticks is not None and (not isinstance(planned_ticks, int) or planned_ticks < 1):
        raise ValueError("META planned_ticks must be a positive integer")
    results = []
    series_ends = []
    for index, being in enumerate(beings):
        being_id = being.get("id", index) if isinstance(being, dict) else index
        frames = being.get("frames") if isinstance(being, dict) else None
        reason = None
        if not isinstance(frames, list) or len(frames) < 2:
            series_ends.append(None)
            results.append({"being": index, "id": being_id, "valid": False, "laps": 0,
                            "progress_m": 0.0, "lap_completion_ticks": [],
                            "reason": "incomplete_series"})
            continue
        samples = []
        for frame in frames:
            try:
                # Replay exports use x/y world coordinates; z is accepted for
                # older fixtures that model the ground plane as x/z.
                x = float(frame["x"])
                y = float(frame["z"] if "z" in frame else frame["y"])
                tick = float(frame.get("t", frame.get("tick")))
                if not all(math.isfinite(v) for v in (x, y, tick)) or tick < 0 or not tick.is_integer():
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                reason = "nonfinite_or_malformed_frame"; break
            d2, segment, q = _project((x, y), points)
            if corridor is not None and math.sqrt(d2) > corridor:
                reason = "out_of_corridor"; break
            arc = sum(lengths[:segment]) + q * lengths[segment]
            samples.append((tick, arc, x, y))
        if reason:
            series_ends.append(None)
            results.append({"being": index, "id": being_id, "valid": False, "laps": 0,
                            "progress_m": 0.0, "lap_completion_ticks": [],
                            "reason": reason}); continue
        series_ends.append(samples[-1][0])
        if planned_ticks is not None and sample_every is not None:
            expected_end = (planned_ticks // int(sample_every)) * int(sample_every)
            if samples[-1][0] != expected_end or (
                    meta.get("source") == "gpu_evolve" and samples[0][0] != sample_every):
                reason = "truncated_series"
        start = samples[0][1]
        unwrapped = 0.0
        max_progress = 0.0
        previous = start
        initial_sector = math.floor(start / lap_length * sectors)
        max_sector = initial_sector
        moved = False
        lap_completion_ticks = []
        completed_laps = 0
        for (tick0, _, x0, y0), (tick1, arc, x1, y1) in zip(samples, samples[1:]):
            if reason:
                break
            dt = tick1 - tick0
            if dt <= 0:
                reason = "non_increasing_ticks"; break
            if sample_every is not None and abs(dt - float(sample_every)) > 1e-6:
                reason = "invalid_sample_interval"; break
            if math.hypot(x1 - x0, y1 - y0) > max(5.0, max_speed * dt / tick_rate + 5.0):
                reason = "implausible_jump"; break
            delta = arc - previous
            if delta > lap_length / 2:
                delta -= lap_length
            elif delta < -lap_length / 2:
                delta += lap_length
            # At a sharp corner, a point inside the road can be equally close to both
            # centreline segments. Switching between those projections can change arc
            # length by up to roughly twice the road corridor while the physical point
            # barely moves. The Euclidean check above remains the hard speed gate; this
            # bound only absorbs that known projection ambiguity.
            projection_bound = PROJECTION_JUMP_TOLERANCE_M
            if corridor is not None:
                projection_bound = max(projection_bound, 2.0 * corridor)
            if abs(delta) > max(projection_bound, max_speed * dt / tick_rate + 5.0):
                reason = "implausible_jump"; break
            old_boundary = math.floor((start + unwrapped) / lap_length * sectors)
            unwrapped += delta
            max_progress = max(max_progress, unwrapped)
            new_boundary = math.floor((start + unwrapped) / lap_length * sectors)
            if new_boundary > old_boundary + 1:
                reason = "sector_jump"; break
            if new_boundary > max_sector:
                max_sector = new_boundary
                moved = True
            while unwrapped >= (completed_laps + 1) * lap_length:
                completed_laps += 1
                lap_completion_ticks.append(tick1)
            previous = arc
            moved = moved or abs(delta) > 1e-6
        if reason is None and not moved:
            reason = "stationary"
        elif reason is None and max_progress <= 0.0:
            reason = "no_forward_progress"
        laps = max(0, int(math.floor(max_progress / lap_length))) if reason is None else len(lap_completion_ticks)
        visited = max(0, max_sector - initial_sector)
        valid = reason is None and laps >= min_laps and visited >= min_laps * sectors
        if reason is None and not valid:
            reason = "insufficient_sequential_progress"
        results.append({"being": index, "valid": valid, "laps": laps, "progress_m": max_progress,
                        "sectors_visited": visited, "lap_completion_ticks": lap_completion_ticks,
                        "reason": reason, "id": being_id})
    complete_ends = [end for end in series_ends if end is not None]
    if complete_ends and max(complete_ends) != min(complete_ends):
        for result, end in zip(results, series_ends):
            if end is not None and end < max(complete_ends):
                result["valid"] = False
                result["reason"] = "truncated_series"
    passed = sum(1 for result in results if result["valid"])
    fraction = passed / len(results)
    integrity_valid = all(result["reason"] not in INTEGRITY_ERRORS for result in results)
    return {"schema_version": 1, "valid": integrity_valid and fraction >= min_fraction,
            "integrity_valid": integrity_valid, "passed": passed, "total": len(results),
            "fraction": fraction, "min_laps": min_laps, "min_fraction": min_fraction,
            "lap_length_m": lap_length, "beings": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--min-laps", type=int, default=1)
    parser.add_argument("--min-fraction", type=float, default=0.8)
    parser.add_argument("--sectors", type=int, default=8)
    parser.add_argument("--max-speed", type=float, default=100.0)
    parser.add_argument("--annotate-html", type=Path, metavar="OUT")
    args = parser.parse_args(argv)
    if args.min_laps < 1 or not 0.0 <= args.min_fraction <= 1.0 or args.sectors < 1 or not math.isfinite(args.max_speed) or args.max_speed <= 0:
        parser.error("thresholds must be positive; min-fraction must be in [0,1]")
    try:
        raw_html = args.html.read_bytes()
        text = raw_html.decode("utf-8")
        data = {"track": _value(text, "TRACK"), "beings": _value(text, "BEINGS"), "meta": _value(text, "META")}
        report = validate(data, args.min_laps, args.min_fraction, args.sectors, args.max_speed)
        report["input_sha256"] = hashlib.sha256(raw_html).hexdigest()
        report["thresholds"] = {"min_laps": args.min_laps, "min_fraction": args.min_fraction,
                                 "sectors": args.sectors, "max_speed": args.max_speed,
                                 "projection_jump_tolerance_m": PROJECTION_JUMP_TOLERANCE_M,
                                 "corridor_margin_m": CORRIDOR_MARGIN_M}
        if args.annotate_html:
            marker = "</head>"
            if marker not in text.lower():
                raise ValueError("HTML has no </head> for annotation")
            safe = json.dumps(report, separators=(",", ":"), sort_keys=True)
            safe = safe.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            script = f"<script>window.REPLAY_VALIDATION={safe};</script>"
            text = re.sub(r"<script>window\.REPLAY_VALIDATION=.*?;</script>", "", text, flags=re.DOTALL)
            head_at = text.lower().index(marker)
            annotated = text[:head_at] + script + text[head_at:]
            args.annotate_html.write_text(annotated, encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"valid": False, "error": str(error)}), file=sys.stdout)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report["valid"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
