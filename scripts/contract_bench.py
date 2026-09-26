#!/usr/bin/env python3
"""Contract bench: drop frozen genomes into pre-built school scenarios and judge
their inputs (sensory channels) and outputs (motion, judge verdict) against fixed
contracts, without a city trip. One GPU process at a time.

    contract_bench.py run    --root <dir> [--binary <gpu_evolve>] [--only genome,lesson]
    contract_bench.py report --root <dir>

Outputs land under <root>/<genome>/<lesson>/ (school_result, observation.jsonl,
driving.csv, run.log, exit.json); the report writes <root>/contracts.json and
<root>/CONTRACTS.md.
"""
import argparse, csv, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

RUNS = "/media/llm-data/universe-runs"
GENOMES = {
    "witness": None,  # scripted reference controller, positive control
    "founder": f"{RUNS}/cross-city-forward-learning-20260925/founder/genome_input.bin",
    "champion48": f"{RUNS}/stable-centered-backbone-thaw-20260925/evolve_best_gen0011.bin",
    "candidate": f"{RUNS}/cross-city-forward-learning-20260925/candidate/evolve_best_gen0000.bin",
}
# lesson -> (ticks, extra flags). Motor lessons use observation v3: v4/v5 route
# channels are station-indexed and encode a free lesson sub-goal as direction 0 /
# distance 1 (taxi.wgsl route_senses). Service lessons keep v4 as trained.
# "move_v4" is the negative control that must show that channel degeneracy.
LESSONS = {
    "move": (1280, ["--navigation-observation", "v3"]),
    "move_v4": (1280, ["--navigation-observation", "v4"]),
    "stop": (1280, ["--navigation-observation", "v3"]),
    # Turn horizon 2560 (40 s): a red arm lasts 896+128 ticks = 16 s and the
    # scripted witness waiting at it timed out at 1280 (see *_1280 dirs).
    "turn_near": (2560, ["--navigation-observation", "v3"]),
    "turn_far": (2560, ["--navigation-observation", "v3"]),
    "approach_passenger": (2048, ["--navigation-observation", "v4"]),
    "pickup": (2048, ["--navigation-observation", "v4"]),
    "delivery48": (4096, ["--navigation-observation", "v4", "--school-service-course", "48"]),
}
LESSON_FLAG = {"delivery48": "delivery", "move_v4": "move"}
ENV = {
    "WGPU_BACKEND": "vulkan",
    "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.json",
    "MESA_VK_VERSION_OVERRIDE": "1.3",
    "RAYON_NUM_THREADS": "1",
}


def command(binary, genome, lesson, out):
    ticks, extra = LESSONS[lesson]
    cmd = [binary, "--scenario", "organism-taxi-v1", "--school", LESSON_FLAG.get(lesson, lesson),
           "--school-mode", "bench", "--school-start-fold", "all", *extra,
           "--school-eval-starts", "4", "--school-start-pool", "4", "--school-select", "worst",
           "--beings", "1", "--gens", "1", "--ticks", str(ticks), "--school-horizon", str(ticks),
           "--school-deadline-frac", "1", "--seed", "11"]
    if genome == "witness":
        cmd += ["--taxi-reference-witness"]
    else:
        cmd += ["--resume", "--resume-mix", "1"]
    cmd += ["--no-slow", "--fixed-body", "--out-beta", "0.85",
            "--replay-every", "1", "--driving-diagnostics", "--out", str(out),
            "--research-body", "fixed", "--retina-centered-grid", "--organism-stable-wheels",
            "--vision-being", "0"]
    return cmd


def run(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    binary = root / "gpu_evolve"
    if not binary.exists():
        try:
            os.link(args.binary, binary)
        except OSError:
            shutil.copyfile(args.binary, binary)
            os.chmod(binary, 0o755)
    only = set(args.only.split(",")) if args.only else None
    log = open(root / "bench.log", "a")
    for genome, src in GENOMES.items():
        for lesson in LESSONS:
            if only and genome not in only and lesson not in only:
                continue
            if lesson == "move_v4" and genome != "champion48":
                continue
            out = root / genome / lesson
            if (out / "exit.json").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            if src:
                shutil.copyfile(src, out / "evolve_best_gen0000.bin")
            cmd = command(str(binary), genome, lesson, out)
            (out / "command.json").write_text(json.dumps(cmd, indent=1))
            env = {**os.environ, **ENV}
            env.pop("DISPLAY", None)
            t0 = time.time()
            with open(out / "run.log", "w") as rl:
                rc = subprocess.call(["nice", "-n", "10", *cmd], stdout=rl, stderr=subprocess.STDOUT, env=env)
            dt = time.time() - t0
            (out / "exit.json").write_text(json.dumps({"exit_code": rc, "seconds": dt}))
            log.write(f"{time.strftime('%H:%M:%S')} {genome}/{lesson} exit={rc} {dt:.1f}s\n")
            log.flush()
    log.write("DONE\n")


# ---------------------------------------------------------------- contracts
def load_observation(path):
    recs = []
    for line in open(path):
        recs.append(json.loads(line))
    return recs


def input_contracts(recs, lesson):
    """Sensory contracts on the observed being (vision-being 0), records every 16 ticks."""
    c = {}
    if not recs:
        return {"no_observation": {"pass": False, "value": None}}
    first = recs[0]
    w0 = first["wave"]
    c["I1_has_task_ch180"] = {"value": w0[180], "pass": w0[180] >= 120}
    c["I2_route_ahead_at_start_ch178"] = {"value": w0[178], "pass": w0[178] >= 100}
    # Wall rays: perception.wgsl writes ch16-23, but perception_temporal.wgsl:367
    # overwrites ch20 with fuel and perception_motion.wgsl:231 overwrites ch21-27
    # with per-row motion, so only the ch16-19 half of the fan survives.
    c["I3_near_wall_ray_ch16_within_15m"] = {"value": w0[16], "pass": w0[16] >= 95}
    c["I4_centre_ray_ch19_open_at_start"] = {"value": w0[19], "pass": w0[19] <= 40}
    fuel = first["body"]["fuel"] / max(first["body"]["fuel_capacity"], 1.0)
    c["I8_ch20_is_fuel_not_wall"] = {"value": [w0[20], int(fuel * 127)], "pass": abs(w0[20] - int(fuel * 127)) <= 2}
    # distance channel vs geometry, v4: 183 = d/(d+100) * 127
    err_d, err_v, n = 0.0, 0.0, 0
    for r in recs:
        if r["wave"][180] < 120:  # no task: 183 encodes distance 1 by design
            continue
        t = r["taxi"]
        d = math.hypot(t["target_dx_m"], t["target_dz_m"])
        expect = 127.0 * d / (d + 100.0)
        err_d = max(err_d, abs(r["wave"][183] - expect))
        v = r["body"]["speed"]
        if v < 60:
            err_v = max(err_v, abs(r["wave"][65] / 2.0 - v))
        n += 1
    c["I5_distance_ch183_tracks_geometry_max_err_i8"] = {"value": round(err_d, 1), "pass": err_d <= 8}
    c["I6_speed_ch65_tracks_body_max_err_mps"] = {"value": round(err_v, 2), "pass": err_v <= 1.0}
    if lesson.startswith("turn"):
        peak = max(abs(r["wave"][179]) for r in recs)
        c["I7_turn_route_lateral_peak_ch179"] = {"value": peak, "pass": peak >= 60}
    return c


def output_contracts(res, driving, lesson):
    beings = res["beings"]
    c = {}
    n = len(beings)
    succ = sum(1 for b in beings if b["success"])
    c["O1_judge_success"] = {"value": f"{succ}/{n}", "pass": succ == n}
    vmax = [b["max_speed_mps"] for b in beings]
    c["O2_reaches_3mps"] = {"value": [round(v, 2) for v in vmax], "pass": min(vmax) >= 3.0}
    contact = [b["contact_ticks"] for b in beings]
    c["O3_no_contact"] = {"value": contact, "pass": max(contact) == 0}
    # stationary fraction over the judged span, from driving.csv windows
    frac = []
    for idx, b in enumerate(beings):
        end = b["outcome_tick"] or b["judged_ticks"]
        rows = [r for r in driving if int(r["being"]) == idx and int(r["first_tick"]) <= end]
        ticks = sum(int(r["ticks"]) for r in rows) or 1
        still = sum(int(r["stationary_ticks"]) for r in rows)
        frac.append(round(still / ticks, 2))
    c["O4_stationary_fraction_lt_0.5"] = {"value": frac, "pass": max(frac) < 0.5}
    prog = [round(b["progress"], 2) for b in beings]
    motor = lesson in ("move", "move_v4", "stop", "turn_near", "turn_far")  # progress in [0,1] only here
    c["O5_progress"] = {"value": prog, "pass": (min(prog) >= 0.5 if motor else succ == n)}
    codes = sorted({b["fail"] for b in beings})
    c["fail_codes"] = {"value": codes, "pass": codes == ["none"]}
    dwell = [b.get("max_dwell_ticks") for b in beings]
    c["stop_hold_ticks"] = {"value": dwell, "pass": True}
    return c


def report(args):
    root = Path(args.root)
    table = {}
    for genome in GENOMES:
        for lesson in LESSONS:
            out = root / genome / lesson
            rp = out / "school_result_gen0000.json"
            entry = {"exit": None, "input": {}, "output": {}}
            if (out / "exit.json").exists():
                entry["exit"] = json.loads((out / "exit.json").read_text())
            if rp.exists():
                res = json.loads(rp.read_text())
                driving = list(csv.DictReader(open(out / "driving.csv"))) if (out / "driving.csv").exists() else []
                entry["output"] = output_contracts(res, driving, lesson)
                obs = out / "observation.jsonl"
                entry["input"] = input_contracts(load_observation(obs) if obs.exists() else [], lesson)
            table[f"{genome}/{lesson}"] = entry
    (root / "contracts.json").write_text(json.dumps(table, indent=1))
    lines = ["# Contract bench — " + time.strftime("%Y-%m-%d %H:%M"), "",
             "PASS/FAIL per contract; value in brackets. `-` = run missing or crashed.", ""]
    keys_in = sorted({k for e in table.values() for k in e["input"]})
    keys_out = sorted({k for e in table.values() for k in e["output"]})
    for section, keys in (("Inputs (sensory channels of being 0)", keys_in), ("Outputs (judge + motion)", keys_out)):
        lines += [f"## {section}", "", "| run | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for name, e in table.items():
            src = e["input"] if section.startswith("Inputs") else e["output"]
            cells = []
            for k in keys:
                v = src.get(k)
                cells.append("-" if v is None else ("PASS" if v["pass"] else "FAIL") + f" [{v['value']}]")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    (root / "CONTRACTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--root", required=True)
    r.add_argument("--binary", default="/media/llm-data/universe-runs/navigation-preview-paired-20260925/gpu_evolve",
                   help="same partition as --root so the run dir can hard-link it")
    r.add_argument("--only", default="")
    p = sub.add_parser("report"); p.add_argument("--root", required=True)
    a = ap.parse_args()
    (run if a.cmd == "run" else report)(a)
