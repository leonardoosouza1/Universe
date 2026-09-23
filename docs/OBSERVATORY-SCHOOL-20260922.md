# Observatory Stage 5 — school and fitness in the panel (2026-09-22)

Item 5 of the 2026-09-22 program, panel half. City evolution (regions, alternative
paths, distributed services) is **not** part of this record; see §6.

The panel is the real artifact the owner asked to be validated: `lab-ui` served by
`lab_api` on `http://127.0.0.1:8131/`, reading `/media/llm-data/lab.db` and the run
directories under `/media/llm-data/universe-runs/`. Everything below was checked
against the Stage 4 training runs (`school-comparison-20260922/train-*`).

## 1. Files touched (not committed — see §5)

| Repo | File | Change |
|---|---|---|
| neuromod-lab | `src/lab_api/routes/school_observatory.py` | **new** — bounded, read-only projection of `school_result_genNNNN.json` (latest 200 generations, 8 MB cap per file) plus the `[curriculum]` lines of `run.log` (64 MB cap) |
| neuromod-lab | `src/lab_api/routes/db.py` | route `GET /api/db/school-observatory/{name}` (404 when the run is not registered; empty `generations` when it has no school) |
| neuromod-lab | `src/lab_api/routes/taxi_observatory.py` | `summarize()` also aggregates `money`, `task_progress_score` and the eight `fitness_components` as flat `metrics.fitness_<component>`; new module-level `aggregate()` |
| neuromod-lab | `tests/api/test_school_observatory_projection.py` | **new** — 4 transport tests (summary/beings/curriculum state, empty vs unknown run, oversized or malformed log, flat fitness aggregation) |
| lab-ui | `src/views/TaxiObservatoryView.vue` | school competence trajectory, per-generation school block, fitness decomposition block, observer-privileged labels (PT) |
| lab-ui | `dist/` | rebuilt (`vue-tsc -b && vite build`) |

`taxi_observatory.py` and `TaxiObservatoryView.vue` are Astra's untracked files; both
patches are additive (new keys, new sections) and leave the existing catalog, summary
and trace-index behaviour unchanged.

## 2. Endpoint shapes

`GET /api/db/school-observatory/{name}` →

```json
{
 "name": "train-uniform-s7",
 "generations": [
  {
   "lesson": "turn_far",
   "mode": "bench",
   "curriculum": "uniform",
   "curriculum_index": 3,
   "start_fold": "train",
   "horizon_ticks": 1280,
   "attempts": 16,
   "successes": 0,
   "failures": 14,
   "truncated": 2,
   "running": 0,
   "deaths": 0,
   "competence_p_hat": 0.05555555555555555,
   "failures_by_code": {
    "contact": 8,
    "drove_through_junction": 5,
    "wrong_exit": 1
   },
   "progress_mean": 0.6399469971656799,
   "progress_median": 0.6761655211448669,
   "score_median": 198.14144897460938,
   "score_max": 418.53216552734375,
   "energy_spent_mean_j": 289508.84375,
   "contact_ticks_total": 8.0,
   "elite_status": "failed",
   "elite_progress": 0.46658289432525635,
   "elite_same_genome": true,
   "generation": 15,
   "curriculum_state": {
    "mode": "uniform",
    "lesson": "turn_far",
    "index": 3,
    "count": 4,
    "streak": 0,
    "advance": false,
    "threshold_p": 0.5,
    "threshold_gens": 2
   },
   "beings": [
    {
     "slot": 0,
     "status": "failed",
     "fail": "contact",
     "progress": 0.46658289432525635,
     "score": 33.204586029052734,
     "entry_speed_mps": 0.0,
     "max_speed_mps": 4.867164134979248,
     "distance_m": 27.55072021484375,
     "energy_spent_j": 178021.5,
     "outcome_tick": 550.0,
     "contact_ticks": 1.0,
     "integrity": 0.9991313815116882,
     "refusal_code": 0.0,
     "...": "..."
    }
   ]
  }
 ],
 "scope": "judge results per lesson and generation; observer-privileged, never seen by the being; latest 200 generations"
}
```

`GET /api/db/taxi-observatory/{name}` (existing) gained, per generation,
`metrics.fitness_total|events|task_progress|money|age|alive_bonus|integrity_penalty|death_penalty`
and `metrics.money`, `metrics.task_progress_score`, each `{n, mean, min, max, sum}` with
`mean` null when no finite value exists.

Verified on the running service (21:00, 2026-09-22):

```
curl -s http://127.0.0.1:8131/api/db/school-observatory/train-uniform-s7
→ 16 generations; gen 15 lesson=turn_far successes=0/16 curriculum_state.index=3/4; 16 beings
curl -s http://127.0.0.1:8131/api/db/taxi-observatory/train-uniform-s7
→ gen 15 fitness_total mean 33.54 (min 31.48, max 35.02, n 16); fitness_money mean 23.69; fitness_task_progress 0
```

## 3. What the panel shows, and who can see it

The organism panel (`/`, "Evolução em uma cidade") now has three new blocks per run:

1. **Competência na escola** — one bar per generation, height p̂ = (S+1)/(T+2), colour
   = lesson (move, stop, turn_near, turn_far). Labelled *informação do observador: o ser
   nunca vê o juiz, a lição ou a nota*.
2. **Escola** (per generation) — lesson and its meaning in words, mode (bench: passengers
   and taxi–taxi contacts off, fitness = lesson score; life: taxi fitness), curriculum,
   start fold, horizon; successes/attempts, p̂, failures, truncated, deaths, median and
   max score, median progress; the curriculum sentence ("lição 4 de 4, sequência 0 de 2
   gerações com p̂ ≥ 0.5 · permanece na lição"); failure-code bars with the code
   explained in words (bateu, atravessou o cruzamento sem virar, saiu pela via errada,
   ultrapassou a área, saiu do corredor, não chegou com velocidade); a folded table
   per being (status, code, progress, score, entry/max speed, distance, energy, tick).
3. **Fitness decomposto** (per generation) — mean/min/max/coverage of each fitness
   component with a "Como é calculada" column in words (1000 per delivery; 0–100 per
   approach, only on a new best route distance, +100 with a passenger on board; money;
   0.001 per second alive; 10 if alive; −100 × damage; −100 if dead). In bench mode
   the block states that selection used the lesson score, not this total.

Being-available vs observer-privileged:

| Reaches the brain | Observer only |
|---|---|
| the 192 sensory channels (retina, navigation observation v3, body signals routed by `perception/channels.rs`) | lesson, fold, curriculum state, judge verdict, failure code, p̂ |
| in life mode, one scalar per tick from the feedback adapter (delivered − 2·damage + energy/money/progress terms, per `--feedback-mode`) | the fitness decomposition (selection happens on the host after the life, `organism_scenario.rs`) |
| nothing of the above in bench mode (feedback adapter off, plasticity off) | the 3D brain activity view and the lifetime P/E buffers |

A 3D activity trace shows what the network did, not that it learned: the Stage 3
measurement (`SCHOOL-INCENTIVES-20260922.md` §4.3) found the acquired weights P at
1.6e-4 with residual 1e-5 on the motor, so any behavioural difference between
generations in these runs is inherited, not acquired. The panel therefore never
labels activity as learning; the learning claim lives in the school trajectory and the
held-out evaluation (`SCHOOL-COMPARISON-20260922.md`).

## 4. Commands

```bash
# projection tests (transport only; no organism step on CPU)
cd ~/projects/LLM/neuromod-lab
OMP_NUM_THREADS=1 ~/projects/LLM/.venv-rocm/bin/python -m pytest -q \
  tests/api/test_school_observatory_projection.py tests/api/test_taxi_observatory_projection.py
# → 6 passed

# UI build (one job, low priority)
cd ~/projects/LLM/lab-ui && nice -n 10 npm run build          # vue-tsc -b && vite build → dist/

# register the runs (manifest.json required per run dir)
cd ~/projects/LLM/neuromod-lab
~/projects/LLM/.venv-rocm/bin/python src/lab_db.py ingest /media/llm-data/universe-runs/school-comparison-20260922 \
  --match 'train-*' --db /media/llm-data/lab.db

# serve UI + API on one origin (loopback)
cd ~/projects/LLM && PYTHONPATH=neuromod-lab/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nohup ~/projects/LLM/.venv-rocm/bin/python -m lab_api --port 8131 --bind 127.0.0.1 \
  --runs-root /media/llm-data/universe-runs --ui-dist ~/projects/LLM/lab-ui/dist \
  > /media/llm-data/universe-runs/school-comparison-20260922/lab-api-8131.log 2>&1 &

# screenshot of the real panel
google-chrome --headless=new --no-sandbox --disable-gpu --user-data-dir=<scratch>/chrome-profile \
  --window-size=3440,5200 --hide-scrollbars --virtual-time-budget=10000 \
  --screenshot=/media/llm-data/universe-runs/school-comparison-20260922/panel-train-uniform-s7-tall.png \
  http://127.0.0.1:8131/
```

Evidence: `panel-train-uniform-s7-3440x1240.png` (first screen: catalog, trajectory,
gen 15 summary) and `panel-train-uniform-s7-tall.png` (3440×2500: gen 15 with the
school block — turn_far 0/16, p̂ 0.0556, contact 8/16, drove_through_junction 5/16,
wrong_exit 1/16, "lição 4 de 4 · permanece na lição" — and the fitness block headed
"bancada: a seleção desta geração usou a nota da lição, não este total") in
`/media/llm-data/universe-runs/school-comparison-20260922/`. Both were opened and
read after the final build (`dist/assets/index-Dr_ci5vX.js`); the numbers match the
endpoint output of §2. A first build had silently missed the bench note because the
patch command's stdin was redirected; the served bundle hash was compared with
`dist/index.html` before re-capturing.

The six training runs needed hand-written `manifest.json` files (the runs were
launched without `--driving-diagnostics`, so `gpu_evolve` wrote no `command.txt`); each
manifest carries the exact argv from `battery.sh`, the exit code from `battery.log` and
a `reconstructed_from` note with the binary sha. The 96 evaluation runs are not
registered (their `organism_gen0000.json` is one generation of clones; the comparison
document reads them from the logs instead).

## 5. Limitations

- Not committed: the neuromod-lab and lab-ui working trees carry Astra's uncommitted
  work in the same files (`routes/db.py`, `lab_db.py`, and the two untracked files);
  committing my slice alone would either split their change set or sweep it in. The
  patches are listed in §1 for whoever commits that tree.
- `TestTraceRoutes` in `tests/test_lab_api.py` has pre-existing 404 failures unrelated
  to this change; not touched.
- The command chain raw → head → modulation → applied is **not** surfaced (the
  observation.jsonl of `--driving-diagnostics` carries it; the training runs were made
  without that flag to keep them light).
- The catalog only lists runs with a `city` block in `organism_gen*.json`; evaluation
  runs are excluded on purpose (see §4).
- Screenshot verification is visual (two PNGs read back); no DOM-level assertion.

## 6. Next test

City evolution for item 5 (`src/organism_city.rs`: regions, alternative paths between
the five street centres, distributed stations) and a directed route distance for
task progress (Stage 3 §4.4), each behind a flag and with a paired A/B on the school
lessons before any panel work.
