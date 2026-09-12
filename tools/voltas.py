#!/usr/bin/env python3
"""u1 battery analysis — hippocampus per sector against control (PRD C5).

Run it with this machine's python:
    /home/leonardo/projects/LLM/.venv-rocm/bin/python tools/voltas.py <run> [<run> ...]
    /home/leonardo/projects/LLM/.venv-rocm/bin/python tools/voltas.py \
        --par u1_s031_ctrl:u1_s031_hip u1_s077_ctrl:u1_s077_hip u1_s113_ctrl:u1_s113_hip \
        --gens ultimas 10
    /home/leonardo/projects/LLM/.venv-rocm/bin/python tools/voltas.py --teste

INPUT, READ ONLY: `<run>/laps_gen<NNNN>.jsonl`, one line per being:
`{"being":i,"seed":s,"laps":[{lap,tick_start,tick_end,dist,sector_ticks[24],
grass_ticks,damage,dry_events,parcial}, ...]}` (`src/bin/gpu_evolve/laps.rs`).

`parcial` (since 2026-09-12) marks the lap that was still OPEN when the
generation ended — always a being's last record, one per being. A ledger written
before that flag has no `parcial` key and every record in it is a closed lap;
this tool reads `parcial` as False when the key is absent, so the old u1 files
give exactly the numbers they always gave.

A partial lap has NO comparable time or distance (its `tick_end - tick_start`
and `dist` cover a fragment), so it is excluded from every lap COUNT (med
voltas, >=1V, >=2V) and from `lap2/lap1`. What it does carry honestly is
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
    fake r ~ 0.06. It is EXCLUDED by default; `--com-setor-inicial` puts it back
    for inspection. 22 sectors per being remain.

COVERAGE, TWO NUMBERS: `seres r` is how many beings entered the primary metric,
split into `r 2V` (a CLOSED lap 2) and `r par` (only a partial 2nd pass). The
second column is the one the partial lap bought: on a ledger written before
2026-09-12 it is always 0, and `u0c` gen 0 measured 27 of 60 beings while 54 of
60 closed lap 1. The partial pass is a WEAKER measurement — fewer sectors per
being, and the being died in it — so the two are reported apart instead of
averaged into one coverage figure.

Secondary: `lap2/lap1` in TICKS (not metres — per-lap distance moves with the
racing line, U6 amendment), computed over the SAME clean sectors as the primary
metric: sum(lap2.sector_ticks[clean]) / sum(lap1.sector_ticks[clean]), normally
sectors 1..22. The raw `tick_end - tick_start` version is kept beside it as
`bruto`: lap 1 starts at tick 0, so its raw duration carries the same start-up
that contaminates sector 0, and `bruto` is optimistic for lap 2 for that reason
(u0c gen 0: 0.83 raw against 1.01 clean).

NEVER writes into /media/llm-data or into a run directory: the only outputs are
the `--saida` JSON and the optional `--png`.
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
    "a39cc9a1-b1be-48b6-9c7c-89e12aefeb1c/scratchpad/voltas.json"
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
    "MDE = (t_{1-alfa/2, n-1} + t_{poder, n-1}) * sd / sqrt(n); "
    "n=3, alfa=0.05, poder=0.80 => (4.3027+1.0607)*sd/sqrt(3) = 3.0965*sd. "
    "A 'exata' resolve poder = P(|T'(df=n-1, ncp=Delta*sqrt(n)/sd)| > t_crit) = 0.80."
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
                print(f"  aviso: ultima linha incompleta em {path} (run em andamento)",
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

    A ledger written before the flag existed has no `parcial` key, and every
    record in it is a closed lap — hence the `False` default, which is what keeps
    the old u1 numbers identical.
    """
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
    n_r_2v, n_r_parcial = 0, 0
    for being in beings:
        pairs = sector_pairs(being, with_start_slot)
        if not pairs:
            continue
        parcial = second_pass(being)[2]
        per_being_r.append(float(np.median([t2 / t1 for _, t1, t2 in pairs])))
        sectors_used.append(len(pairs))
        n_r_parcial += int(parcial)
        n_r_2v += int(not parcial)
        # secondary, clean: whole lap restricted to the sectors the primary metric
        # uses (normally 1..22), so it cannot import the dirty start. Only beings
        # with a CLOSED lap 2 enter it: over a partial 2nd pass this sum is a
        # fragment against a whole lap, not a lap ratio.
        if not parcial:
            soma1 = sum(t1 for _, t1, _ in pairs)
            lap21_clean.append(sum(t2 for _, _, t2 in pairs) / soma1)

    lap21_raw = []
    for being in beings:
        done = closed_laps(being)
        if len(done) >= 2:
            d1 = done[0]["tick_end"] - done[0]["tick_start"]
            d2 = done[1]["tick_end"] - done[1]["tick_start"]
            if d1 > 0:
                lap21_raw.append(d2 / d1)

    return {
        "n_seres": n,
        "voltas_medianas": float(np.median(laps_n)) if n else 0.0,
        "frac_1v": float(np.mean(laps_n >= 1)) if n else 0.0,
        "frac_2v": float(np.mean(laps_n >= 2)) if n else 0.0,
        "n_2v": int((laps_n >= 2).sum()) if n else 0,
        # primary
        "r_mediana": float(np.median(per_being_r)) if per_being_r else None,
        "n_seres_r": len(per_being_r),
        # coverage split: closed 2nd lap vs partial 2nd pass (0 on old ledgers)
        "n_r_2v": n_r_2v,
        "n_r_parcial": n_r_parcial,
        "setores_por_ser": float(np.median(sectors_used)) if sectors_used else 0.0,
        # secondary: clean (22 sectors) and raw (tick_end - tick_start)
        "lap2_lap1_ticks": float(np.median(lap21_clean)) if lap21_clean else None,
        "lap2_lap1_ticks_bruto": float(np.median(lap21_raw)) if lap21_raw else None,
        "n_lap2_lap1": len(lap21_raw),
    }


def analyse_run(run_dir: str, gens: list[int] | None, with_start_slot: bool) -> dict:
    available = run_gens(run_dir)
    wanted = available if gens is None else [g for g in gens if g in available]
    return {
        "caminho": run_dir,
        "gens_disponiveis": available,
        "gens": {g: gen_metrics(load_gen(run_dir, g), with_start_slot) for g in wanted},
        "existe": bool(available),
    }


# ------------------------------------------------------------ statistics ----


def select_gens(available: list[int], spec) -> list[int]:
    """`--gens ultimas N` | `todas` | explicit list."""
    if spec is None:
        return list(available)
    if spec[0] == "ultimas":
        return list(available[-spec[1]:])
    if spec[0] == "todas":
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
    return {"n": n, "media": mean, "sd": sd, "t": t, "p": p, "df": n - 1}


def mde(sd: float, n: int, alpha: float = ALPHA, power: float = POWER) -> dict:
    """Smallest delta detectable with `power`, n pairs and the observed sd.

    Approximation (the formula printed in the table):
        MDE = (t_{1-alpha/2, n-1} + t_{power, n-1}) * sd / sqrt(n)
    For n = 3, alpha = 0.05, power = 0.80: (4.3027 + 1.0607)*sd/sqrt(3) = 3.0965*sd.
    The exact one solves the NONCENTRAL t power (ncp = delta*sqrt(n)/sd); with df
    this small the approximation is optimistic, so both are reported.
    """
    if n <= 1 or _st is None or not np.isfinite(sd) or sd <= 0:
        return {"aprox": None, "exata": None, "formula": MDE_FORMULA}
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
    return {"aprox": approx, "exata": exact, "formula": MDE_FORMULA}


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
    print(f"\n=== {name}  ({res['caminho']})")
    if not res["existe"]:
        print("    SEM laps_gen*.jsonl — run ainda nao rodou (ou ledger desligado).")
        return
    print(
        "  gen | seres | med voltas | >=1V | >=2V | r=t2/t1 (med pop) | seres r |"
        " r 2V | r par | setores/ser | lap2/lap1 ticks | bruto"
    )
    for g, m in sorted(res["gens"].items()):
        print(
            f"  {g:4d}|{m['n_seres']:6d} |{m['voltas_medianas']:11.2f} |"
            f"{m['frac_1v']*100:5.0f}%|{m['frac_2v']*100:5.0f}%|"
            f"{fmt(m['r_mediana']):>18}|{m['n_seres_r']:8d} |"
            f"{m['n_r_2v']:5d} |{m['n_r_parcial']:6d} |"
            f"{m['setores_por_ser']:12.0f} |{fmt(m['lap2_lap1_ticks']):>16} |"
            f"{fmt(m['lap2_lap1_ticks_bruto']):>7}"
        )
    print("    r 2V = seres com volta 2 FECHADA | r par = so 2a passagem parcial")


def print_pair(pair: dict) -> None:
    print(f"\n=== PAR {pair['ctrl']} x {pair['hip']}")
    if not pair["gens"]:
        print("    sem geracoes em comum com ledger — par ignorado.")
        return
    print("  gen | r ctrl | r hip  |  delta (hip-ctrl) | >=2V ctrl/hip | r par ctrl/hip")
    for row in pair["por_gen"]:
        print(
            f"  {row['gen']:4d}|{fmt(row['r_ctrl']):>8}|{fmt(row['r_hip']):>8}|"
            f"{fmt(row['delta']):>19}|{row['n2v_ctrl']:>7}/{row['n2v_hip']:<7}|"
            f"{row['npar_ctrl']:>7}/{row['npar_hip']:<7}"
        )
    print(
        f"  -> delta medio {fmt(pair['delta_medio'])} | mediano "
        f"{fmt(pair['delta_mediano'])} | gens {pair['gens'][0]}..{pair['gens'][-1]}"
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
            mh["r_mediana"] - mc["r_mediana"]
            if (mc["r_mediana"] is not None and mh["r_mediana"] is not None)
            else None
        )
        if d is not None:
            deltas.append(d)
        rows.append(
            {
                "gen": g,
                "r_ctrl": mc["r_mediana"],
                "r_hip": mh["r_mediana"],
                "delta": d,
                "n2v_ctrl": mc["n_2v"],
                "n2v_hip": mh["n_2v"],
                # coverage of the PRIMARY metric, split: closed lap 2 vs partial pass
                "nr2v_ctrl": mc["n_r_2v"],
                "nr2v_hip": mh["n_r_2v"],
                "npar_ctrl": mc["n_r_parcial"],
                "npar_hip": mh["n_r_parcial"],
                "lap21_ctrl": mc["lap2_lap1_ticks"],
                "lap21_hip": mh["lap2_lap1_ticks"],
                "lap21_bruto_ctrl": mc["lap2_lap1_ticks_bruto"],
                "lap21_bruto_hip": mh["lap2_lap1_ticks_bruto"],
                "voltas_ctrl": mc["voltas_medianas"],
                "voltas_hip": mh["voltas_medianas"],
            }
        )
    return {
        "ctrl": ctrl_name,
        "hip": hip_name,
        "gens": gens,
        "por_gen": rows,
        "delta_medio": float(np.mean(deltas)) if deltas else None,
        "delta_mediano": float(np.median(deltas)) if deltas else None,
        "n_gens_com_delta": len(deltas),
    }


def write_png(pairs: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for pair in pairs:
        gens = [r["gen"] for r in pair["por_gen"] if r["delta"] is not None]
        ds = [r["delta"] for r in pair["por_gen"] if r["delta"] is not None]
        if gens:
            ax.plot(gens, ds, marker="o", label=f"{pair['hip']} - {pair['ctrl']}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("geracao")
    ax.set_ylabel("delta r = t2/t1 (hip - ctrl)")
    ax.set_title("Hipocampo por setor: 2a passagem vs 1a (negativo = hip mais rapido)")
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
    slot 23 empty (the lap closes when ENTERING sector 23). Written WITHOUT the
    `parcial` key, exactly like the ledgers the old binary left on disk — so every
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


def _synthetic_being_parcial(i: int, factor: float, reached: int) -> dict:
    """1 closed lap + a lap 2 that DIED after `reached` sectors (`parcial`).

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
                "parcial": False,
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
                "parcial": True,
            },
        ],
    }


def _being_with_explicit_ticks(
    i: int, t1_full: int, t2_ticks: dict[int, int], lap2_tick_start: int, lap2_dist: float
) -> dict:
    """1 closed lap (uniform `t1_full` ticks/sector, sector 23 empty) + a partial lap 2
    with an explicit, arbitrary `sector_ticks` map (unset sectors default to 0).

    Used for the interrupted-sector regression cases (T7B): unlike
    `_synthetic_being_parcial`, this lets a "reached" sector carry FEWER ticks than a
    full pass — the actual shape `LapTracker::finish` writes for a being that dies mid
    sector, which `_synthetic_being_parcial` cannot express (it only knows "full" or "0").
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
                "parcial": False,
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
                "parcial": True,
            },
        ],
    }


def _with_empty_parcial(being: dict) -> dict:
    """The same being as the new binary would write it: one empty partial appended.

    A being that closes a lap and is never ticked again still files the open lap
    (`laps.rs::finish`). Nothing in this tool may move because of it.
    """
    laps = [dict(lap, parcial=False) for lap in being["laps"]]
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
            "parcial": True,
        }
    )
    return dict(being, laps=laps)


def self_test() -> int:
    root = tempfile.mkdtemp(prefix="voltas_teste_", dir=os.path.dirname(DEFAULT_OUT))
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'ok ' if cond else 'FALHA'}] {name} {detail}")
        if not cond:
            failures.append(name)

    # T1 - 2nd pass 10 % faster => r = 0.9 exactly, over 22 sectors.
    beings = [_synthetic_being(i, 0.9) for i in range(4)] + [_synthetic_being(9, None)]
    _write_synthetic(os.path.join(root, "sint_09"), 0, beings)
    m = gen_metrics(beings, with_start_slot=False)
    check("r = 0.9 com 2a passagem 10% mais rapida", m["r_mediana"] == 0.9, f"r={m['r_mediana']}")
    check("22 setores por ser (fora 0 e 23)", m["setores_por_ser"] == 22, f"n={m['setores_por_ser']}")
    check("4 seres entram na metrica", m["n_seres_r"] == 4)
    check("mediana de voltas = 2 com 4/5 seres a 2V", m["voltas_medianas"] == 2.0)
    check("frac >=2V = 0.8", abs(m["frac_2v"] - 0.8) < 1e-12)
    check(
        "lap2/lap1 limpo = 3960/4400 = 0.9 (setores 1..22)",
        m["lap2_lap1_ticks"] == 0.9,
        f"={m['lap2_lap1_ticks']}",
    )
    check(
        "lap2/lap1 bruto = (180*23+170)/5400 (largada suja puxa p/ baixo)",
        abs(m["lap2_lap1_ticks_bruto"] - (180 * 23 + 170) / 5400) < 1e-12,
        f"={m['lap2_lap1_ticks_bruto']:.4f}",
    )

    # T2 - with the start slot the dirty 180/1000 enters and the count goes to 23.
    m0 = gen_metrics(beings, with_start_slot=True)
    check("com --com-setor-inicial usa 23 setores", m0["setores_por_ser"] == 23)

    # T3 - an arm 5 % slower on the 2nd pass => delta = +0.05 exactly.
    slower = [_synthetic_being(i, 0.95) for i in range(4)] + [_synthetic_being(9, None)]
    _write_synthetic(os.path.join(root, "sint_095"), 0, slower)
    pair = compare_pair("sint_09", "sint_095", root, ("todas",), False)
    check("delta sintetico = +0.05", abs(pair["delta_medio"] - 0.05) < 1e-12, f"d={pair['delta_medio']}")

    # T4 - a run against an identical copy => delta 0 exactly, sd 0.
    _write_synthetic(os.path.join(root, "sint_09b"), 0, beings)
    same = compare_pair("sint_09", "sint_09b", root, ("todas",), False)
    check("par identico da delta 0 exato", same["delta_medio"] == 0.0, f"d={same['delta_medio']}")
    est = paired_t(np.array([0.0, 0.0, 0.0]))
    check("sd = 0 e t indefinido no par identico", est["sd"] == 0.0 and est["t"] is None)

    # T5 - a PARTIAL 2nd pass enters the primary metric, minus its interrupted sector.
    # 4 beings, lap 2 died after 6 sectors at 10 % faster: slots 0..5 hold ticks, but slot
    # 0 is the dirty start (excluded as always) AND slot 5 is where the being died — this
    # fixture cannot tell "drove sector 5 fully then died at the boundary" from "died mid
    # sector 5", so `interrupted_sector` must drop 5 too. Only 1..4 pair up.
    parciais = [_synthetic_being_parcial(i, 0.9, 6) for i in range(4)]
    mp = gen_metrics(parciais, with_start_slot=False)
    check("2a passagem parcial da r = 0.9", mp["r_mediana"] == 0.9, f"r={mp['r_mediana']}")
    check("4 setores por ser (1..4; 0 e largada, 5 e o interrompido)",
          mp["setores_por_ser"] == 4.0, f"n={mp['setores_por_ser']}")
    check("cobertura parcial contada em separado", (mp["n_r_parcial"], mp["n_r_2v"]) == (4, 0),
          f"par={mp['n_r_parcial']} 2v={mp['n_r_2v']}")
    check("volta parcial NAO conta como volta", mp["voltas_medianas"] == 1.0,
          f"med={mp['voltas_medianas']}")
    check("frac >=2V = 0 com lap 2 parcial", mp["frac_2v"] == 0.0 and mp["n_2v"] == 0)
    check("lap2/lap1 limpo fica n/a (fragmento nao e volta)",
          mp["lap2_lap1_ticks"] is None and mp["lap2_lap1_ticks_bruto"] is None)

    # T6 - a partial that reached ONLY slot 0 gives no pair, with or without
    # --com-setor-inicial: slot 0 is both the start slot AND the only sector reached, so
    # it is also the interrupted one — the two exclusions overlap, neither flag brings it
    # back. A being that died before the 1st crossing of lap 2 stays visible in the
    # ledger (it is not dropped from the file) but never enters the metric.
    so_largada = [_synthetic_being_parcial(i, 0.9, 1) for i in range(4)]
    ms = gen_metrics(so_largada, with_start_slot=False)
    check("parcial so no setor 0 nao entra na metrica",
          ms["n_seres_r"] == 0 and ms["r_mediana"] is None)
    check("interrupted sector holds even under --com-setor-inicial (slot 0 is also it)",
          gen_metrics(so_largada, with_start_slot=True)["n_seres_r"] == 0)

    # T7 - mixed population: 2 closed lap 2 at 0.9 and 2 partial at 0.8 => median 0.85
    # over 4 beings, with the coverage split 2/2.
    mistura = [_synthetic_being(i, 0.9) for i in range(2)] + [
        _synthetic_being_parcial(10 + i, 0.8, 6) for i in range(2)
    ]
    mm = gen_metrics(mistura, with_start_slot=False)
    check("mistura: r mediana = 0.85", abs(mm["r_mediana"] - 0.85) < 1e-12, f"r={mm['r_mediana']}")
    check("mistura: cobertura 2 fechadas / 2 parciais",
          (mm["n_r_2v"], mm["n_r_parcial"]) == (2, 2))
    check("mistura: lap2/lap1 limpo vem so das fechadas", mm["lap2_lap1_ticks"] == 0.9,
          f"={mm['lap2_lap1_ticks']}")

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
          r1["n_seres_r"] == 0 and r1["r_mediana"] is None,
          f"n_seres_r={r1['n_seres_r']} r={r1['r_mediana']}")

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
          r2["setores_por_ser"] == 5.0 and r2["r_mediana"] == 1.0,
          f"setores={r2['setores_por_ser']} r={r2['r_mediana']}")

    # T8 - THE BACK-COMPATIBILITY CHECK: the same beings with the empty partial the new
    # binary appends must give a byte-for-byte identical metrics dict. If this fails,
    # every u1 number already published moved.
    novo = [_with_empty_parcial(b) for b in beings]
    velho_m = gen_metrics(beings, with_start_slot=False)
    check("parcial vazio nao muda NENHUM numero",
          gen_metrics(novo, with_start_slot=False) == velho_m,
          f"novo={gen_metrics(novo, with_start_slot=False)}")
    check("ledger antigo (sem a chave parcial) le como fechado",
          velho_m["n_r_2v"] == 4 and velho_m["n_r_parcial"] == 0)

    # T9 - the printed MDE formula is the one the code applies.
    d = mde(0.01, 3)
    check("MDE aprox = 3.0965*sd (n=3)", abs(d["aprox"] - 3.0965 * 0.01) < 1e-5, f"={d['aprox']:.6f}")
    check("MDE exata > aproximada (t nao-central)", d["exata"] > d["aprox"], f"exata={d['exata']:.6f}")

    if failures:
        print(f"\n  temporarios mantidos para inspecao: {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    print("AUTO-TESTE: " + ("VERDE" if not failures else f"VERMELHO ({len(failures)} falhas)"))
    return 0 if not failures else 1


# ------------------------------------------------------------------- main ----


def parse_gens(values):
    if not values:
        return None
    head = values[0].lower()
    if head in ("ultimas", "últimas", "last"):
        return ("ultimas", int(values[1]))
    if head in ("todas", "all"):
        return ("todas",)
    return ("lista", [int(v) for v in values])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analise da bateria u1 (PRD C5).")
    ap.add_argument("runs", nargs="*", help="diretorios de run (nome sob --base ou absoluto)")
    ap.add_argument("--par", nargs="+", default=[], metavar="CTRL:HIP",
                    help="pares controle:hipocampo, um por semente")
    ap.add_argument("--gens", nargs="+", default=None,
                    help="'ultimas N' | 'todas' | lista de geracoes")
    ap.add_argument("--base", default=DEFAULT_BASE, help="raiz dos runs")
    ap.add_argument("--saida", default=DEFAULT_OUT, help="JSON de saida")
    ap.add_argument("--png", default=None, help="curva de delta por geracao (opcional)")
    ap.add_argument("--com-setor-inicial", action="store_true",
                    help="inclui o setor 0 (slot da largada, contaminado) na metrica")
    ap.add_argument("--teste", action="store_true", help="auto-teste com dados sinteticos")
    args = ap.parse_args(argv)

    if args.teste:
        return self_test()

    spec = parse_gens(args.gens)
    with_start_slot = args.com_setor_inicial
    summary = {
        "gerado_em": _dt.datetime.now().isoformat(timespec="seconds"),
        "base": args.base,
        "setor_inicial_incluido": with_start_slot,
        "metrica_primaria": "mediana pop. da mediana por ser de r = t2/t1 por setor (ticks)",
        "runs": {},
        "pares": [],
        "resumo_pareado": None,
    }

    names = list(args.runs)
    for p in args.par:
        if ":" not in p:
            ap.error(f"--par espera CTRL:HIP, recebi {p!r}")
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

    if args.par:
        pairs = []
        for p in args.par:
            ctrl, hip = p.split(":", 1)
            pair = compare_pair(ctrl, hip, args.base, spec, with_start_slot)
            pairs.append(pair)
            print_pair(pair)
        summary["pares"] = pairs

        deltas = np.array([q["delta_medio"] for q in pairs if q["delta_medio"] is not None])
        if len(deltas):
            est = paired_t(deltas)
            d = mde(est["sd"], est["n"])
            summary["resumo_pareado"] = {
                **est,
                "mde_80": d,
                "deltas": deltas.tolist(),
                "deltas_medianos": [q["delta_mediano"] for q in pairs
                                    if q["delta_medio"] is not None],
                "base_do_t": "delta medio por semente sobre as gens selecionadas",
            }
            medians = [q["delta_mediano"] for q in pairs if q["delta_medio"] is not None]
            print("\n=== PAREADO POR SEMENTE (metrica primaria, hip - ctrl)")
            for q in pairs:
                if q["delta_medio"] is None:
                    continue
                print(f"  {q['hip']:<18} delta medio {fmt(q['delta_medio'])} | "
                      f"mediano {fmt(q['delta_mediano'])}  ({q['n_gens_com_delta']} gens)")
            print(f"  n pares            : {est['n']}")
            print(f"  deltas medios      : {[round(float(x), 4) for x in deltas]}")
            print(f"  deltas medianos    : {[round(float(x), 4) for x in medians]}")
            print(f"  delta medio        : {fmt(est['media'])}"
                  f"   (t e MDE abaixo usam o delta MEDIO por semente)")
            print(f"  sd                 : {fmt(est['sd'])}")
            print(f"  t (df={est['df']})         : {fmt(est['t'], 3)}"
                  + (f"   p = {fmt(est['p'], 4)}" if est["p"] is not None else "   p = n/a"))
            print(f"  MDE 80% (aprox)    : {fmt(d['aprox'])}")
            print(f"  MDE 80% (exata)    : {fmt(d['exata'])}")
            print(f"  formula            : {MDE_FORMULA}")
            print("  leitura: r < 1 = 2a passagem mais rapida; delta < 0 = hipocampo ganha.")
        else:
            print("\n=== PAREADO: nenhum par com metrica (runs sem ledger?).")

        if args.png:
            write_png(pairs, args.png)

    os.makedirs(os.path.dirname(os.path.abspath(args.saida)), exist_ok=True)
    with open(args.saida, "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"\nJSON: {args.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
