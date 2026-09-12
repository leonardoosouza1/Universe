#!/usr/bin/env python3
"""u1 battery analysis — hippocampus per sector against control (PRD C5).

Run it with this machine's python:
    /home/leonardo/projects/LLM/.venv-rocm/bin/python tools/laps.py <run> [<run> ...]
    /home/leonardo/projects/LLM/.venv-rocm/bin/python tools/laps.py \
        --pair u1_s031_ctrl:u1_s031_hip u1_s077_ctrl:u1_s077_hip u1_s113_ctrl:u1_s113_hip \
        --gens last 10
    /home/leonardo/projects/LLM/.venv-rocm/bin/python tools/laps.py --self-test

INPUT, READ ONLY: `<run>/laps_gen<NNNN>.jsonl`, one line per being:
`{"being":i,"seed":s,"laps":[{lap,tick_start,tick_end,dist,sector_ticks[24],
grass_ticks,damage,dry_events,partial}, ...]}` (`src/bin/gpu_evolve/laps.rs`).

`partial` (since 2026-09-12) marks the lap that was still OPEN when the
generation ended — always a being's last record, one per being. THREE ledger
vintages exist and all three read the same here: written before the flag (no
key at all, every record a closed lap), written with the key spelled `parcial`
(the flag's first day), and written with `partial` (the English pass of
2026-09-12). `is_partial` accepts both spellings and defaults to False, so the
old u1 files give exactly the numbers they always gave.

A partial lap has NO comparable time or distance (its `tick_end - tick_start`
and `dist` cover a fragment), so it is excluded from every lap COUNT (med
laps, >=1V, >=2V) and from `lap2/lap1`. What it does carry honestly is
`sector_ticks` FOR THE SECTORS IT FINISHED — not the one it died in (or the one
the generation ended in): that sector has no comparable time either, for the
same reason the whole lap does not. Found by `interrupted_sector()`: sectors
are driven in increasing index order between crossings
(`LapTracker::on_sector`, only advances on the next expected checkpoint,
`main.rs:3241-3250`), so the interrupted one is the highest index reached,
EXCLUDING `CLOSE_SLOT` (23) — see its definition below for why. BUG FIXED
2026-09-12 (external review): before this exclusion, a being that entered a
sector and died 1 tick later contributed r = 1/200 = 0.005, a fake 99.5 %
speed-up, because any `sector_ticks[s] > 0` counted as a "2nd pass" with no
regard for whether the being was still alive when it left that sector.

PRIMARY METRIC (U6 amendment, `docs/PREDICOES-CEREBRO.md`): per-sector time on
the 2nd pass against the 1st, in ticks. The 1st lap must be CLOSED (lap 1 of the
being); the 2nd pass is lap 2 if it closed, ELSE the partial lap 2. For every
sector s with `t1 = lap1.sector_ticks[s] > 0` and `t2 = lap2.sector_ticks[s] >
0`, EXCLUDING the interrupted sector when the 2nd pass is partial: r = t2/t1 ->
median per being -> population median. r < 1 means the 2nd pass was faster,
which is what the hippocampus is supposed to buy. Sectors the partial never
reached have `t2 = 0` and drop out by the same rule that already drops sector
23; the sector it DID reach but never finished driving is dropped on purpose.

TWO SECTOR SLOTS ARE A LEDGER ARTEFACT, not driving (checked on
`u0c_valida_hip`, 2026-09-12):
  * sector 23 (`CLOSE_SLOT`) — the lap closes when the being ENTERS sector 23
    (`main.rs:3263-3267`, `checkpoint % 24 == 0`), so `lap1.sector_ticks[23]` is
    always 0 (27/27 and 21/21 of the >=2-lap beings in gens 0 and 3) on ANY
    closed lap, not just lap 1 — the `t1 > 0` rule drops it on its own. The lap
    that follows a close TILES from that same sector (`LapTracker::on_lap`,
    `laps.rs:113-132`, "opens the next at the same tick, odometer and sector"),
    so on the NEXT lap 23 is the FIRST sector driven, a full and legitimate
    pass — not the last, and `interrupted_sector()` must not pick it as the
    interrupted one, or it would exclude a complete sector and let the real
    interrupted one (a lower index) slip back into the metric.
  * sector 0 — `LapTracker::new` opens lap 1 in slot 0 at tick 0, so that slot
    holds "everything before the first crossing" plus the real pass (median
    3 760 ticks on lap 1 against 238 on lap 2, gen 0). Including it injects a
    fake r ~ 0.06. It is EXCLUDED by default; `--with-start-sector` puts it back
    for inspection. 22 sectors per being remain.

COVERAGE, TWO NUMBERS: `beings r` is how many beings entered the primary metric,
split into `r 2V` (a CLOSED lap 2) and `r par` (only a partial 2nd pass). The
second column is the one the partial lap bought: on a ledger written before
2026-09-12 it is always 0, and `u0c` gen 0 measured 27 of 60 beings while 54 of
60 closed lap 1. The partial pass is a WEAKER measurement — fewer sectors per
being, and the being died in it — so the two are reported apart instead of
averaged into one coverage figure.

Secondary: `lap2/lap1` in TICKS (not metres — per-lap distance moves with the
racing line, U6 amendment), computed over the SAME clean sectors as the primary
metric: sum(lap2.sector_ticks[clean]) / sum(lap1.sector_ticks[clean]), normally
sectors 1..22. The `tick_end - tick_start` version is kept beside it as
`raw`: lap 1 starts at tick 0, so its raw duration carries the same start-up
that contaminates sector 0, and `raw` is optimistic for lap 2 for that reason
(u0c gen 0: 0.83 raw against 1.01 clean).

NEVER writes into /media/llm-data or into a run directory: the only outputs are
the `--out` JSON and the optional `--png`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import re
import shutil
import sys
import tempfile

import numpy as np

try:  # scipy is only needed for the p-value and the exact MDE.
    from scipy import stats as _st
except Exception:  # pragma: no cover - environment without scipy
    _st = None

DEFAULT_BASE = "/media/llm-data/universe-runs/neuromod-lab-taxi"
DEFAULT_OUT = (
    "/tmp/claude-1000/-home-leonardo-projects-LLM/"
    "a39cc9a1-b1be-48b6-9c7c-89e12aefeb1c/scratchpad/laps.json"
)
N_SECTORS = 24  # statics.rs:128 N_CHECKPOINTS
START_SLOT = 0  # slot LapTracker::new opens at tick 0 (contaminated on lap 1)
CLOSE_SLOT = N_SECTORS - 1  # 23: every lap closes on ENTERING this sector
# (main.rs:3241-3311, `checkpoint % N_CHECKPOINTS == 0` fires on that crossing), so it is
# ALWAYS 0 on a closed lap and, on the lap that follows, the FIRST sector driven (laps
# tile, laps.rs:113-132) — never the last. `interrupted_sector()` excludes it from its
# search for exactly that reason.
ALPHA = 0.05
POWER = 0.80

MDE_FORMULA = (
    "MDE = (t_{1-alpha/2, n-1} + t_{power, n-1}) * sd / sqrt(n); "
    "n=3, alpha=0.05, power=0.80 => (4.3027+1.0607)*sd/sqrt(3) = 3.0965*sd. "
    "The 'exact' one solves power = P(|T'(df=n-1, ncp=Delta*sqrt(n)/sd)| > t_crit) = 0.80."
)


# ------------------------------------------------------------------ input ----


def run_gens(run_dir: str) -> list[int]:
    """Generations with a lap ledger in this run, ascending."""
    gens = []
    for path in glob.glob(os.path.join(run_dir, "laps_gen*.jsonl")):
        m = re.search(r"laps_gen(\d+)\.jsonl$", path)
        if m:
            gens.append(int(m.group(1)))
    return sorted(gens)


def load_gen(run_dir: str, gen: int) -> list[dict]:
    """One generation: list of beings, each with `laps` (closed laps + 1 partial)."""
    path = os.path.join(run_dir, f"laps_gen{gen:04d}.jsonl")
    with open(path, "r", errors="replace") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    beings = []
    for i, line in enumerate(lines):
        try:
            beings.append(json.loads(line))
        except json.JSONDecodeError:
            # Only the last line may be half-written: the run is still going and
            # this generation is being flushed right now. Anything else is a
            # corrupt ledger and must not be swallowed.
            if i == len(lines) - 1:
                print(f"  warning: last line of {path} is incomplete (run in progress)",
                      file=sys.stderr)
                break
            raise
    return beings


def resolve(name: str, base: str) -> str:
    """Run name under `base`, or an absolute path taken as is."""
    return name if os.path.isabs(name) else os.path.join(base, name)


# ----------------------------------------------------------------- metric ----


def is_partial(lap: dict) -> bool:
    """The lap was still open when the generation ended.

    Reads BOTH wire spellings: `partial` (current) and `parcial` (the flag's
    first day, 2026-09-12). A ledger written before the flag existed has
    neither key, and every record in it is a closed lap — hence the `False`
    default, which is what keeps the old u1 numbers identical.
    """
    if "partial" in lap:
        return bool(lap["partial"])
    return bool(lap.get("parcial", False))


def closed_laps(being: dict) -> list[dict]:
    """The being's CLOSED laps, in order. This is what counts as a lap."""
    return [lap for lap in (being.get("laps") or []) if not is_partial(lap)]


def second_pass(being: dict) -> tuple[dict, dict, bool] | None:
    """(lap1, lap2, lap2_is_partial) for the primary metric, or None.

    Lap 1 must be closed. Lap 2 is the closed 2nd lap when there is one, else the
    partial lap — which is a 2nd pass exactly for the sectors it reached.
    """
    laps = being.get("laps") or []
    done = closed_laps(being)
    if not done:
        return None
    if len(done) >= 2:
        return done[0], done[1], False
    tail = laps[-1] if laps and is_partial(laps[-1]) else None
    if tail is None or tail.get("lap") != done[0].get("lap", 0) + 1:
        return None  # the partial is not the lap right after lap 1
    return done[0], tail, True


def interrupted_sector(sector_ticks) -> int | None:
    """The sector a PARTIAL lap died in (or the generation ended in) — not a completed
    pass, so it must not enter the primary metric or its coverage count.

    Sectors are driven in increasing index order between crossings
    (`LapTracker::on_sector` only advances on the next expected checkpoint,
    `main.rs:3241-3250`), so within a partial lap the reached sectors are a contiguous
    run and the interrupted one is the HIGHEST index among them — EXCLUDING `CLOSE_SLOT`
    (23) from the search. `CLOSE_SLOT` is always the first sector of the lap that follows
    a close (see its definition above), never the last, and `lap1.sector_ticks[CLOSE_SLOT]`
    is always 0 so it can never form a pair anyway; picking it as "interrupted" by raw
    index would exclude a complete sector and leave the REAL interrupted one (a lower
    index) in the metric — the exact bug this function exists to prevent.

    Returns None when the partial reached nothing at all (dies within the same tick the
    previous lap closed): nothing to exclude, `sector_pairs` already yields no pairs.
    """
    reached = [s for s in range(CLOSE_SLOT) if sector_ticks[s] > 0]  # 0..22
    return max(reached) if reached else None


def sector_pairs(being: dict, with_start_slot: bool) -> list[tuple[int, int, int]]:
    """(sector, t1, t2) for every sector driven on lap 1 AND on the 2nd pass.

    On a PARTIAL 2nd pass the sector the being died in (`interrupted_sector`) is excluded
    too: it has no comparable time, same as the lap it belongs to (see module doc).
    """
    pas = second_pass(being)
    if pas is None:
        return []
    lap1, lap2, is_partial = pas
    t1 = lap1["sector_ticks"]
    t2 = lap2["sector_ticks"]
    excluded = interrupted_sector(t2) if is_partial else None
    pairs = []
    for s in range(min(N_SECTORS, len(t1), len(t2))):
        if s == START_SLOT and not with_start_slot:
            continue  # start slot: lap 1 also holds the pre-crossing ticks (see module doc)
        if s == excluded:
            continue  # the sector the partial died in: not a completed 2nd pass
        if t1[s] > 0 and t2[s] > 0:
            pairs.append((s, t1[s], t2[s]))
    return pairs


def sector_ratios(being: dict, with_start_slot: bool) -> list[float]:
    """r = t2/t1 for each sector driven on lap 1 AND on the 2nd pass."""
    return [t2 / t1 for _, t1, t2 in sector_pairs(being, with_start_slot)]


def gen_metrics(beings: list[dict], with_start_slot: bool) -> dict:
    """(a) laps, (b) primary per-sector metric, (c) lap2/lap1 in ticks."""
    n = len(beings)
    # A lap is a CLOSED lap: the partial record every being now ends with is not one.
    laps_n = np.array([len(closed_laps(b)) for b in beings], dtype=float)

    per_being_r, sectors_used, lap21_clean = [], [], []
    n_r_2v, n_r_partial = 0, 0
    for being in beings:
        pairs = sector_pairs(being, with_start_slot)
        if not pairs:
            continue
        partial = second_pass(being)[2]
        per_being_r.append(float(np.median([t2 / t1 for _, t1, t2 in pairs])))
        sectors_used.append(len(pairs))
        n_r_partial += int(partial)
        n_r_2v += int(not partial)
        # secondary, clean: whole lap restricted to the sectors the primary metric
        # uses (normally 1..22), so it cannot import the dirty start. Only beings
        # with a CLOSED lap 2 enter it: over a partial 2nd pass this sum is a
        # fragment against a whole lap, not a lap ratio.
        if not partial:
            sum1 = sum(t1 for _, t1, _ in pairs)
            lap21_clean.append(sum(t2 for _, _, t2 in pairs) / sum1)

    lap21_raw = []
    for being in beings:
        done = closed_laps(being)
        if len(done) >= 2:
            d1 = done[0]["tick_end"] - done[0]["tick_start"]
            d2 = done[1]["tick_end"] - done[1]["tick_start"]
            if d1 > 0:
                lap21_raw.append(d2 / d1)

    return {
        "n_beings": n,
        "median_laps": float(np.median(laps_n)) if n else 0.0,
        "frac_1v": float(np.mean(laps_n >= 1)) if n else 0.0,
        "frac_2v": float(np.mean(laps_n >= 2)) if n else 0.0,
        "n_2v": int((laps_n >= 2).sum()) if n else 0,
        # primary
        "r_median": float(np.median(per_being_r)) if per_being_r else None,
        "n_beings_r": len(per_being_r),
        # coverage split: closed 2nd lap vs partial 2nd pass (0 on old ledgers)
        "n_r_2v": n_r_2v,
        "n_r_partial": n_r_partial,
        "sectors_per_being": float(np.median(sectors_used)) if sectors_used else 0.0,
        # secondary: clean (22 sectors) and raw (tick_end - tick_start)
        "lap2_lap1_ticks": float(np.median(lap21_clean)) if lap21_clean else None,
        "lap2_lap1_ticks_raw": float(np.median(lap21_raw)) if lap21_raw else None,
        "n_lap2_lap1": len(lap21_raw),
    }


def analyse_run(run_dir: str, gens: list[int] | None, with_start_slot: bool) -> dict:
    available = run_gens(run_dir)
    wanted = available if gens is None else [g for g in gens if g in available]
    return {
        "path": run_dir,
        "gens_available": available,
        "gens": {g: gen_metrics(load_gen(run_dir, g), with_start_slot) for g in wanted},
        "exists": bool(available),
    }


# ------------------------------------------------------------ statistics ----


def select_gens(available: list[int], spec) -> list[int]:
    """`--gens last N` | `all` | explicit list."""
    if spec is None:
        return list(available)
    if spec[0] == "last":
        return list(available[-spec[1]:])
    if spec[0] == "all":
        return list(available)
    wanted = set(spec[1])
    return [g for g in available if g in wanted]


def paired_t(deltas: np.ndarray) -> dict:
    """One-sample t over the per-seed paired deltas (df = n - 1)."""
    n = len(deltas)
    mean = float(np.mean(deltas)) if n else float("nan")
    sd = float(np.std(deltas, ddof=1)) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        t = mean / (sd / math.sqrt(n))
        p = float(2 * _st.t.sf(abs(t), n - 1)) if _st is not None else None
    else:
        # sd == 0 with a non-zero mean is a degenerate "infinite t", not a result.
        t = float("inf") if (n > 1 and sd == 0 and mean != 0) else None
        p = None
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": p, "df": n - 1}


def mde(sd: float, n: int, alpha: float = ALPHA, power: float = POWER) -> dict:
    """Smallest delta detectable with `power`, n pairs and the observed sd.

    Approximation (the formula printed in the table):
        MDE = (t_{1-alpha/2, n-1} + t_{power, n-1}) * sd / sqrt(n)
    For n = 3, alpha = 0.05, power = 0.80: (4.3027 + 1.0607)*sd/sqrt(3) = 3.0965*sd.
    The exact one solves the NONCENTRAL t power (ncp = delta*sqrt(n)/sd); with df
    this small the approximation is optimistic, so both are reported.
    """
    if n <= 1 or _st is None or not np.isfinite(sd) or sd <= 0:
        return {"approx": None, "exact": None, "formula": MDE_FORMULA}
    df = n - 1
    t_crit = float(_st.t.ppf(1 - alpha / 2, df))
    approx = (t_crit + float(_st.t.ppf(power, df))) * sd / math.sqrt(n)

    def power_at(delta: float) -> float:
        ncp = delta * math.sqrt(n) / sd
        return float(_st.nct.sf(t_crit, df, ncp) + _st.nct.cdf(-t_crit, df, ncp))

    exact = None
    lo, hi = 0.0, max(approx * 4, sd * 20)
    if power_at(hi) >= power:
        for _ in range(80):  # bisection: power is monotone in delta
            mid = 0.5 * (lo + hi)
            if power_at(mid) < power:
                lo = mid
            else:
                hi = mid
        exact = 0.5 * (lo + hi)
    return {"approx": approx, "exact": exact, "formula": MDE_FORMULA}


# -------------------------------------------------------------- reporting ----


def fmt(x, places=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        if math.isnan(x):
            return "n/a"  # sd of n = 1, p without df
        return "inf" if x > 0 else "-inf"
    return f"{x:.{places}f}"


def print_run(name: str, res: dict) -> None:
    print(f"\n=== {name}  ({res['path']})")
    if not res["exists"]:
        print("    NO laps_gen*.jsonl — run has not started yet (or the ledger is off).")
        return
    print(
        "  gen |beings | med laps   | >=1V | >=2V | r=t2/t1 (pop med) |beings r |"
        " r 2V | r par | sectors/bng | lap2/lap1 ticks |   raw"
    )
    for g, m in sorted(res["gens"].items()):
        print(
            f"  {g:4d}|{m['n_beings']:6d} |{m['median_laps']:11.2f} |"
            f"{m['frac_1v']*100:5.0f}%|{m['frac_2v']*100:5.0f}%|"
            f"{fmt(m['r_median']):>18}|{m['n_beings_r']:8d} |"
            f"{m['n_r_2v']:5d} |{m['n_r_partial']:6d} |"
            f"{m['sectors_per_being']:12.0f} |{fmt(m['lap2_lap1_ticks']):>16} |"
            f"{fmt(m['lap2_lap1_ticks_raw']):>7}"
        )
    print("    r 2V = beings with lap 2 CLOSED | r par = partial 2nd pass only")


def print_pair(pair: dict) -> None:
    print(f"\n=== PAIR {pair['ctrl']} x {pair['hip']}")
    if not pair["gens"]:
        print("    no generation with a ledger in common — pair skipped.")
        return
    print("  gen | r ctrl | r hip  |  delta (hip-ctrl) | >=2V ctrl/hip | r par ctrl/hip")
    for row in pair["per_gen"]:
        print(
            f"  {row['gen']:4d}|{fmt(row['r_ctrl']):>8}|{fmt(row['r_hip']):>8}|"
            f"{fmt(row['delta']):>19}|{row['n2v_ctrl']:>7}/{row['n2v_hip']:<7}|"
            f"{row['npar_ctrl']:>7}/{row['npar_hip']:<7}"
        )
    print(
        f"  -> mean delta {fmt(pair['delta_mean'])} | median "
        f"{fmt(pair['delta_median'])} | gens {pair['gens'][0]}..{pair['gens'][-1]}"
    )


# ------------------------------------------------------------------ pairs ----


def compare_pair(ctrl_name, hip_name, base, gens_spec, with_start_slot) -> dict:
    """Per-generation delta of the primary metric, paired on the same generation."""
    ctrl_dir, hip_dir = resolve(ctrl_name, base), resolve(hip_name, base)
    common = sorted(set(run_gens(ctrl_dir)) & set(run_gens(hip_dir)))
    gens = select_gens(common, gens_spec)

    rows, deltas = [], []
    for g in gens:
        mc = gen_metrics(load_gen(ctrl_dir, g), with_start_slot)
        mh = gen_metrics(load_gen(hip_dir, g), with_start_slot)
        d = (
            mh["r_median"] - mc["r_median"]
            if (mc["r_median"] is not None and mh["r_median"] is not None)
            else None
        )
        if d is not None:
            deltas.append(d)
        rows.append(
            {
                "gen": g,
                "r_ctrl": mc["r_median"],
                "r_hip": mh["r_median"],
                "delta": d,
                "n2v_ctrl": mc["n_2v"],
                "n2v_hip": mh["n_2v"],
                # coverage of the PRIMARY metric, split: closed lap 2 vs partial pass
                "nr2v_ctrl": mc["n_r_2v"],
                "nr2v_hip": mh["n_r_2v"],
                "npar_ctrl": mc["n_r_partial"],
                "npar_hip": mh["n_r_partial"],
                "lap21_ctrl": mc["lap2_lap1_ticks"],
                "lap21_hip": mh["lap2_lap1_ticks"],
                "lap21_raw_ctrl": mc["lap2_lap1_ticks_raw"],
                "lap21_raw_hip": mh["lap2_lap1_ticks_raw"],
                "laps_ctrl": mc["median_laps"],
                "laps_hip": mh["median_laps"],
            }
        )
    return {
        "ctrl": ctrl_name,
        "hip": hip_name,
        "gens": gens,
        "per_gen": rows,
        "delta_mean": float(np.mean(deltas)) if deltas else None,
        "delta_median": float(np.median(deltas)) if deltas else None,
        "n_gens_with_delta": len(deltas),
    }


def write_png(pairs: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for pair in pairs:
        gens = [r["gen"] for r in pair["per_gen"] if r["delta"] is not None]
        ds = [r["delta"] for r in pair["per_gen"] if r["delta"] is not None]
        if gens:
            ax.plot(gens, ds, marker="o", label=f"{pair['hip']} - {pair['ctrl']}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("generation")
    ax.set_ylabel("delta r = t2/t1 (hip - ctrl)")
    ax.set_title("Sector hippocampus: 2nd pass vs 1st (negative = hippocampus faster)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"\nPNG: {path}")


# -------------------------------------------------------------- self-test ----


def _write_synthetic(dirpath: str, gen: int, beings: list[dict]) -> None:
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, f"laps_gen{gen:04d}.jsonl"), "w") as fh:
        for being in beings:
            fh.write(json.dumps(being) + "\n")


def _synthetic_being(i: int, factor: float | None) -> dict:
    """A being with 2 laps: 200 ticks per sector on lap 1, `factor`*200 on lap 2.

    Reproduces both real artefacts: lap 1 slot 0 inflated by the start, and lap 1
    slot 23 empty (the lap closes when ENTERING sector 23). Written WITHOUT any
    partial key, exactly like the ledgers the oldest binary left on disk — so every
    check built on it is also a check that those files still read as before.
    """
    if factor is None:  # a being that never closed a lap
        return {"being": i, "seed": 900 + i, "laps": []}
    t1 = [200] * N_SECTORS
    t2 = [int(round(200 * factor))] * N_SECTORS
    t1[0] = 1000  # dirty start
    t1[23] = 0  # lap closes on entering 23
    t2[23] = 170
    def lap(n, ticks, start):
        return {
            "lap": n,
            "tick_start": start,
            "tick_end": start + sum(ticks),
            "dist": 1600.0,
            "sector_ticks": ticks,
            "grass_ticks": 0,
            "damage": 0.0,
            "dry_events": 0,
        }

    return {"being": i, "seed": 900 + i, "laps": [lap(1, t1, 0), lap(2, t2, sum(t1))]}


def _synthetic_being_partial(i: int, factor: float, reached: int) -> dict:
    """1 closed lap + a lap 2 that DIED after `reached` sectors (`partial`).

    Lap 2 holds `factor`*200 ticks in slots 0..reached-1 and 0 everywhere else,
    which is what `LapTracker::finish` writes for a being that dies mid lap: the
    sectors it drove, and zeros for the sectors it never saw.
    """
    t1 = [200] * N_SECTORS
    t1[0] = 1000  # dirty start
    t1[23] = 0  # lap closes on entering 23
    t2 = [0] * N_SECTORS
    for sec in range(reached):
        t2[sec] = int(round(200 * factor))
    return {
        "being": i,
        "seed": 700 + i,
        "laps": [
            {
                "lap": 1,
                "tick_start": 0,
                "tick_end": sum(t1),
                "dist": 1600.0,
                "sector_ticks": t1,
                "grass_ticks": 0,
                "damage": 0.0,
                "dry_events": 0,
                "partial": False,
            },
            {
                "lap": 2,
                "tick_start": sum(t1),
                "tick_end": sum(t1) + sum(t2),
                "dist": 400.0,  # a fragment, not a lap distance
                "sector_ticks": t2,
                "grass_ticks": 0,
                "damage": 0.0,
                "dry_events": 0,
                "partial": True,
            },
        ],
    }


def _being_with_explicit_ticks(
    i: int, t1_full: int, t2_ticks: dict[int, int], lap2_tick_start: int, lap2_dist: float
) -> dict:
    """1 closed lap (uniform `t1_full` ticks/sector, sector 23 empty) + a partial lap 2
    with an explicit, arbitrary `sector_ticks` map (unset sectors default to 0).

    Used for the interrupted-sector regression cases (T7B): unlike
    `_synthetic_being_partial`, this lets a "reached" sector carry FEWER ticks than a
    full pass — the actual shape `LapTracker::finish` writes for a being that dies mid
    sector, which `_synthetic_being_partial` cannot express (it only knows "full" or "0").
    """
    t1 = [t1_full] * N_SECTORS
    t1[CLOSE_SLOT] = 0  # lap closes on entering CLOSE_SLOT
    t2 = [0] * N_SECTORS
    for sector, ticks in t2_ticks.items():
        t2[sector] = ticks
    return {
        "being": i,
        "seed": 800 + i,
        "laps": [
            {
                "lap": 1,
                "tick_start": 0,
                "tick_end": sum(t1),
                "dist": 1600.0,
                "sector_ticks": t1,
                "grass_ticks": 0,
                "damage": 0.0,
                "dry_events": 0,
                "partial": False,
            },
            {
                "lap": 2,
                "tick_start": lap2_tick_start,
                "tick_end": lap2_tick_start + sum(t2),
                "dist": lap2_dist,
                "sector_ticks": t2,
                "grass_ticks": 0,
                "damage": 0.0,
                "dry_events": 0,
                "partial": True,
            },
        ],
    }


def _with_empty_partial(being: dict) -> dict:
    """The same being as the new binary would write it: one empty partial appended.

    A being that closes a lap and is never ticked again still files the open lap
    (`laps.rs::finish`). Nothing in this tool may move because of it.
    """
    laps = [dict(lap, partial=False) for lap in being["laps"]]
    last = laps[-1]["lap"] if laps else 0
    laps.append(
        {
            "lap": last + 1,
            "tick_start": 0,
            "tick_end": 0,
            "dist": 0.0,
            "sector_ticks": [0] * N_SECTORS,
            "grass_ticks": 0,
            "damage": 0.0,
            "dry_events": 0,
            "partial": True,
        }
    )
    return dict(being, laps=laps)


def _with_legacy_partial_key(being: dict) -> dict:
    """The same being as the binary of 2026-09-12 wrote it, before the English pass:
    the flag on the wire spelled `parcial` instead of `partial`.

    Both spellings exist on disk under /media/llm-data/universe-runs, so reading only
    the new one would silently reclassify every partial lap of that day's ledgers as a
    CLOSED lap — inflating the lap counts and letting fragments into `lap2/lap1`.
    """
    laps = []
    for lap in being["laps"]:
        renamed = {k: v for k, v in lap.items() if k != "partial"}
        renamed["parcial"] = lap["partial"]
        laps.append(renamed)
    return dict(being, laps=laps)


def self_test() -> int:
    root = tempfile.mkdtemp(prefix="laps_self_test_", dir=os.path.dirname(DEFAULT_OUT))
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'ok  ' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            failures.append(name)

    # T1 - 2nd pass 10 % faster => r = 0.9 exactly, over 22 sectors.
    beings = [_synthetic_being(i, 0.9) for i in range(4)] + [_synthetic_being(9, None)]
    _write_synthetic(os.path.join(root, "sint_09"), 0, beings)
    m = gen_metrics(beings, with_start_slot=False)
    check("r = 0.9 when the 2nd pass is 10% faster", m["r_median"] == 0.9, f"r={m['r_median']}")
    check("22 sectors per being (0 and 23 out)", m["sectors_per_being"] == 22,
          f"n={m['sectors_per_being']}")
    check("4 beings enter the metric", m["n_beings_r"] == 4)
    check("median laps = 2 with 4/5 beings at 2 laps", m["median_laps"] == 2.0)
    check("frac >=2V = 0.8", abs(m["frac_2v"] - 0.8) < 1e-12)
    check(
        "clean lap2/lap1 = 3960/4400 = 0.9 (sectors 1..22)",
        m["lap2_lap1_ticks"] == 0.9,
        f"={m['lap2_lap1_ticks']}",
    )
    check(
        "raw lap2/lap1 = (180*23+170)/5400 (the dirty start drags it down)",
        abs(m["lap2_lap1_ticks_raw"] - (180 * 23 + 170) / 5400) < 1e-12,
        f"={m['lap2_lap1_ticks_raw']:.4f}",
    )

    # T2 - with the start slot the dirty 180/1000 enters and the count goes to 23.
    m0 = gen_metrics(beings, with_start_slot=True)
    check("--with-start-sector uses 23 sectors", m0["sectors_per_being"] == 23)

    # T3 - an arm 5 % slower on the 2nd pass => delta = +0.05 exactly.
    slower = [_synthetic_being(i, 0.95) for i in range(4)] + [_synthetic_being(9, None)]
    _write_synthetic(os.path.join(root, "sint_095"), 0, slower)
    pair = compare_pair("sint_09", "sint_095", root, ("all",), False)
    check("synthetic delta = +0.05", abs(pair["delta_mean"] - 0.05) < 1e-12,
          f"d={pair['delta_mean']}")

    # T4 - a run against an identical copy => delta 0 exactly, sd 0.
    _write_synthetic(os.path.join(root, "sint_09b"), 0, beings)
    same = compare_pair("sint_09", "sint_09b", root, ("all",), False)
    check("an identical pair gives delta exactly 0", same["delta_mean"] == 0.0,
          f"d={same['delta_mean']}")
    est = paired_t(np.array([0.0, 0.0, 0.0]))
    check("sd = 0 and t undefined on the identical pair", est["sd"] == 0.0 and est["t"] is None)

    # T5 - a PARTIAL 2nd pass enters the primary metric, minus its interrupted sector.
    # 4 beings, lap 2 died after 6 sectors at 10 % faster: slots 0..5 hold ticks, but slot
    # 0 is the dirty start (excluded as always) AND slot 5 is where the being died — this
    # fixture cannot tell "drove sector 5 fully then died at the boundary" from "died mid
    # sector 5", so `interrupted_sector` must drop 5 too. Only 1..4 pair up.
    partials = [_synthetic_being_partial(i, 0.9, 6) for i in range(4)]
    mp = gen_metrics(partials, with_start_slot=False)
    check("a partial 2nd pass gives r = 0.9", mp["r_median"] == 0.9, f"r={mp['r_median']}")
    check("4 sectors per being (1..4; 0 is the start, 5 the interrupted one)",
          mp["sectors_per_being"] == 4.0, f"n={mp['sectors_per_being']}")
    check("partial coverage counted apart", (mp["n_r_partial"], mp["n_r_2v"]) == (4, 0),
          f"par={mp['n_r_partial']} 2v={mp['n_r_2v']}")
    check("a partial lap is NOT counted as a lap", mp["median_laps"] == 1.0,
          f"med={mp['median_laps']}")
    check("frac >=2V = 0 when lap 2 is partial", mp["frac_2v"] == 0.0 and mp["n_2v"] == 0)
    check("clean lap2/lap1 stays n/a (a fragment is not a lap)",
          mp["lap2_lap1_ticks"] is None and mp["lap2_lap1_ticks_raw"] is None)

    # T6 - a partial that reached ONLY slot 0 gives no pair, with or without
    # --with-start-sector: slot 0 is both the start slot AND the only sector reached, so
    # it is also the interrupted one — the two exclusions overlap, neither flag brings it
    # back. A being that died before the 1st crossing of lap 2 stays visible in the
    # ledger (it is not dropped from the file) but never enters the metric.
    start_only = [_synthetic_being_partial(i, 0.9, 1) for i in range(4)]
    ms = gen_metrics(start_only, with_start_slot=False)
    check("a partial that reached only sector 0 does not enter the metric",
          ms["n_beings_r"] == 0 and ms["r_median"] is None)
    check("interrupted sector holds even under --with-start-sector (slot 0 is also it)",
          gen_metrics(start_only, with_start_slot=True)["n_beings_r"] == 0)

    # T7 - mixed population: 2 closed lap 2 at 0.9 and 2 partial at 0.8 => median 0.85
    # over 4 beings, with the coverage split 2/2.
    mixture = [_synthetic_being(i, 0.9) for i in range(2)] + [
        _synthetic_being_partial(10 + i, 0.8, 6) for i in range(2)
    ]
    mm = gen_metrics(mixture, with_start_slot=False)
    check("mixture: median r = 0.85", abs(mm["r_median"] - 0.85) < 1e-12, f"r={mm['r_median']}")
    check("mixture: coverage 2 closed / 2 partial",
          (mm["n_r_2v"], mm["n_r_partial"]) == (2, 2))
    check("mixture: clean lap2/lap1 comes from the closed ones only",
          mm["lap2_lap1_ticks"] == 0.9, f"={mm['lap2_lap1_ticks']}")

    # T7B - the external reviewer's exact cases for the interrupted-sector bug
    # (2026-09-12): a partial lap's LAST sector is where the being died, not a real pass.

    # Case 1: first pass closes at 200 ticks/sector; the partial second pass enters ONE
    # sector, survives 1 tick, dies. Before the fix this sector paired as r = 1/200 =
    # 0.005 (a fake 99.5 % speed-up) and the being counted; now it must count NOTHING.
    one_tick_death = [
        _being_with_explicit_ticks(
            i, t1_full=200, t2_ticks={5: 1}, lap2_tick_start=200 * 23, lap2_dist=0.1
        )
        for i in range(4)
    ]
    r1 = gen_metrics(one_tick_death, with_start_slot=False)
    check("1-tick death in a sector does not count as a 2nd pass",
          r1["n_beings_r"] == 0 and r1["r_median"] is None,
          f"n_beings_r={r1['n_beings_r']} r={r1['r_median']}")

    # Case 2: sectors 0..5 fully driven, sector 6 interrupted (fewer ticks) then dies.
    # Sectors 1..5 must count (0 is the start slot, as always); 6 must not.
    six_interrupted = [
        _being_with_explicit_ticks(
            i,
            t1_full=200,
            t2_ticks={0: 200, 1: 200, 2: 200, 3: 200, 4: 200, 5: 200, 6: 30},
            lap2_tick_start=200 * 23,
            lap2_dist=900.0,
        )
        for i in range(4)
    ]
    r2 = gen_metrics(six_interrupted, with_start_slot=False)
    check("sectors 1..5 count, interrupted sector 6 does not",
          r2["sectors_per_being"] == 5.0 and r2["r_median"] == 1.0,
          f"sectors={r2['sectors_per_being']} r={r2['r_median']}")

    # T8 - THE BACK-COMPATIBILITY CHECK: the same beings with the empty partial the new
    # binary appends must give a byte-for-byte identical metrics dict. If this fails,
    # every u1 number already published moved.
    new_ledger = [_with_empty_partial(b) for b in beings]
    old_metrics = gen_metrics(beings, with_start_slot=False)
    check("an empty partial moves NO number",
          gen_metrics(new_ledger, with_start_slot=False) == old_metrics,
          f"new={gen_metrics(new_ledger, with_start_slot=False)}")
    check("an old ledger (no partial key at all) reads as closed",
          old_metrics["n_r_2v"] == 4 and old_metrics["n_r_partial"] == 0)

    # T8B - the SAME beings with the key spelled `parcial`, the wire name of
    # 2026-09-12 before the English pass. Both spellings must read identically, or
    # the ledgers written that day stop being comparable with everything else.
    legacy = [_with_legacy_partial_key(b) for b in new_ledger]
    check("the `parcial` spelling reads exactly like `partial`",
          gen_metrics(legacy, with_start_slot=False)
          == gen_metrics(new_ledger, with_start_slot=False))

    # T9 - the printed MDE formula is the one the code applies.
    d = mde(0.01, 3)
    check("MDE approx = 3.0965*sd (n=3)", abs(d["approx"] - 3.0965 * 0.01) < 1e-5,
          f"={d['approx']:.6f}")
    check("MDE exact > approx (noncentral t)", d["exact"] > d["approx"],
          f"exact={d['exact']:.6f}")

    if failures:
        print(f"\n  temporaries kept for inspection: {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    print("SELF-TEST: " + ("GREEN" if not failures else f"RED ({len(failures)} failures)"))
    return 0 if not failures else 1


# ------------------------------------------------------------------- main ----


def parse_gens(values):
    """`last N` | `all` | an explicit list of generation numbers.

    The Portuguese spellings (`ultimas`, `todas`) are still accepted: they are what
    every recorded command in docs/ and in the lab's request files says.
    """
    if not values:
        return None
    head = values[0].lower()
    if head in ("last", "ultimas", "últimas"):
        return ("last", int(values[1]))
    if head in ("all", "todas"):
        return ("all",)
    return ("list", [int(v) for v in values])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="u1 battery analysis (PRD C5).")
    ap.add_argument("runs", nargs="*", help="run directories (name under --base, or absolute)")
    ap.add_argument("--pair", "--par", nargs="+", default=[], metavar="CTRL:HIP",
                    dest="pair", help="control:hippocampus pairs, one per seed")
    ap.add_argument("--gens", nargs="+", default=None,
                    help="'last N' | 'all' | list of generations")
    ap.add_argument("--base", default=DEFAULT_BASE, help="root of the runs")
    ap.add_argument("--out", "--saida", default=DEFAULT_OUT, dest="out", help="output JSON")
    ap.add_argument("--png", default=None, help="delta-per-generation curve (optional)")
    ap.add_argument("--with-start-sector", "--com-setor-inicial", action="store_true",
                    dest="with_start_sector",
                    help="include sector 0 (the contaminated start slot) in the metric")
    ap.add_argument("--self-test", "--teste", action="store_true", dest="self_test",
                    help="self-test on synthetic data")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    spec = parse_gens(args.gens)
    with_start_slot = args.with_start_sector
    summary = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "base": args.base,
        "start_sector_included": with_start_slot,
        "primary_metric": "pop. median of the per-being median of r = t2/t1 per sector (ticks)",
        "runs": {},
        "pairs": [],
        "paired_summary": None,
    }

    names = list(args.runs)
    for p in args.pair:
        if ":" not in p:
            ap.error(f"--pair expects CTRL:HIP, got {p!r}")
        names += p.split(":", 1)

    seen = []
    for name in names:
        if name in seen:
            continue
        seen.append(name)
        run_dir = resolve(name, args.base)
        available = run_gens(run_dir)
        res = analyse_run(run_dir, select_gens(available, spec) if available else None,
                          with_start_slot)
        summary["runs"][name] = res
        print_run(name, res)

    if args.pair:
        pairs = []
        for p in args.pair:
            ctrl, hip = p.split(":", 1)
            pair = compare_pair(ctrl, hip, args.base, spec, with_start_slot)
            pairs.append(pair)
            print_pair(pair)
        summary["pairs"] = pairs

        deltas = np.array([q["delta_mean"] for q in pairs if q["delta_mean"] is not None])
        if len(deltas):
            est = paired_t(deltas)
            d = mde(est["sd"], est["n"])
            summary["paired_summary"] = {
                **est,
                "mde_80": d,
                "deltas": deltas.tolist(),
                "delta_medians": [q["delta_median"] for q in pairs
                                  if q["delta_mean"] is not None],
                "t_basis": "mean delta per seed over the selected generations",
            }
            medians = [q["delta_median"] for q in pairs if q["delta_mean"] is not None]
            print("\n=== PAIRED BY SEED (primary metric, hip - ctrl)")
            for q in pairs:
                if q["delta_mean"] is None:
                    continue
                print(f"  {q['hip']:<18} mean delta {fmt(q['delta_mean'])} | "
                      f"median {fmt(q['delta_median'])}  ({q['n_gens_with_delta']} gens)")
            print(f"  n pairs            : {est['n']}")
            print(f"  mean deltas        : {[round(float(x), 4) for x in deltas]}")
            print(f"  median deltas      : {[round(float(x), 4) for x in medians]}")
            print(f"  mean delta         : {fmt(est['mean'])}"
                  f"   (the t and MDE below use the MEAN delta per seed)")
            print(f"  sd                 : {fmt(est['sd'])}")
            print(f"  t (df={est['df']})         : {fmt(est['t'], 3)}"
                  + (f"   p = {fmt(est['p'], 4)}" if est["p"] is not None else "   p = n/a"))
            print(f"  MDE 80% (approx)   : {fmt(d['approx'])}")
            print(f"  MDE 80% (exact)    : {fmt(d['exact'])}")
            print(f"  formula            : {MDE_FORMULA}")
            print("  reading: r < 1 = faster 2nd pass; delta < 0 = the hippocampus wins.")
        else:
            print("\n=== PAIRED: no pair with a metric (runs without a ledger?).")

        if args.png:
            write_png(pairs, args.png)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"\nJSON: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
