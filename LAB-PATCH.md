# LAB-PATCH — what to change in `neuromod-lab` after the English rename pass

Written 2026-09-12 from the `english-names` branch of Universe. **Nothing in
`~/projects/LLM/neuromod-lab` was edited by that branch**; this file is the whole
diff, to be applied by hand when the owner chooses.

## TL;DR — nothing is urgent

Every Portuguese flag and `--tau-bank` value still works. `gpu_evolve` resolves them
through `src/bin/gpu_evolve/cli_flags.rs` and prints one warning per alias per run:

```
flag `--pistas` is deprecated, use `--tracks`
```

So the queue keeps running untouched. The only thing that CHANGED behaviourally for
the lab is the two renamed **binaries/tools** — and the lab does not invoke either
(checked: `grep -rn 'sensibilidade\|voltas\.py' neuromod-lab/src neuromod-lab/scripts`
finds nothing outside a dated report).

## 1. `src/universe_run.py` — required only if you want the warnings gone

The wrapper has one hardcoded Portuguese flag of its own.

**Line 11**, the `--pistas` passthrough:

```python
ap.add_argument('--pistas', nargs='*', default=None)
```
```python
# after
ap.add_argument('--tracks', '--pistas', dest='tracks', nargs='*', default=None)
```

**Line 16**, where it is emitted:

```python
if a.pistas: args += ['--pistas', ','.join(a.pistas)]
```
```python
# after
if a.tracks: args += ['--tracks', ','.join(a.tracks)]
```

Keeping `--pistas` as an argparse alias matters: 35 request files under
`_nmd/rascunhos/` carry `"pistas": [...]`, and the wrapper turns a request key into
`--<key>`, so the key reaches this wrapper, not the binary.

**`--saida` stays exactly as it is.** Its default value is the string `'--out'`,
which is the flag `gpu_evolve` reads, and `--out` in `gpu_evolve` was already English
and is unchanged. Renaming the wrapper's own `--saida` option to `--out` would
collide with the wrapper's existing required `--out` (the run directory). Leave it.

Also worth updating for accuracy, not function: the module docstring's example on
lines 4-7 spells the flags `--retomar`, `--pistas`, which now print deprecation
warnings. The canonical spelling of that same command is:

```
gpu_evolve --gens 1 --ticks 20000 --resume --tracks 8005,8200,9001 --seed 77 --out <dir>
```

## 2. `_nmd/rascunhos/*.json` — OPTIONAL, and I recommend leaving them

35 request files under `/media/llm-data/universe-runs/neuromod-lab-taxi/_nmd/rascunhos/`
(including `abortado/` and `proxima_ligada/`) use these `args` keys, all of which are
deprecated aliases that still resolve:

| key, `n` files | canonical |
|---|---:|
| `seres` (35) | `beings` |
| `pistas` (35) | `tracks` |
| `retomar` (35) | `resume` |
| `mascara-regioes` (35) | `region-mask` |
| `tau-semeado` (35) | `tau-seeded` |
| `replay-cada` (35) | `replay-every` |
| `hipocampo-setor` (21) | `hippocampus-sector` |
| `sem-hipocampo` (14) | `no-hippocampus` |
| `hipocampo-apagar` (3) | `hippocampus-erase` |

Unchanged keys: `bin`, `gens`, `ticks`, `seed`, `width`, `hip-p`, `hip-sigma`.

**Why leave them:** a request file is the RECORD of what was asked for, and several
of these runs are already on disk with a `manifest.json` beside them. Rewriting the
request after the fact makes the record disagree with the run. Write NEW requests with
the canonical keys; migrate an old one only if you are re-running it.

Note that `pistas` is consumed by `universe_run.py` itself (see §1), so a request
written with `"tracks"` needs the §1 patch applied first, or the key falls through to
`gpu_evolve` as `--tracks 8005` — which works for a single track but not for a list,
because the wrapper is what joins a list with commas.

## 3. Nothing to change in `scripts/nmd.py`

Checked for the renamed log line. `nmd.py` has exactly two output regexes:

```python
RE_GER  = re.compile(r'^\s*ger\s+(\d+)\s*\|\s*(.*?)\s*\(melhor')
RE_ARM  = re.compile(r'^\s{2}(\S.*?)\s{2,}NP\s+(\d+)')
```

Neither matches `gpu_evolve` output: `gpu_evolve` prints no `ger N | … (melhor` line
at all (its per-generation line is `[setores] mediana …`), and `NP` is the lab's own
arm table. The line this pass renamed, `Evolução:` → `Evolution:`, is a run-header
line in `main.rs` that nothing parses — its `check-language: allow` comment claimed it
was "parsed by tools", and that claim was wrong; the comment is gone with the rename.

## 4. `tools/laps.py` (was `tools/voltas.py`) — for whoever reads the ledgers

It lives in Universe, not in the lab, but the lab's reports cite it. New spelling:

```
/home/leonardo/projects/LLM/.venv-rocm/bin/python tools/laps.py \
    --pair u1v_s031_ctrl:u1v_s031_hip u1v_s077_ctrl:u1v_s077_hip u1v_s113_ctrl:u1v_s113_hip \
    --gens last 5
```

`--par`, `--gens ultimas N`, `--gens todas`, `--teste`, `--saida` and
`--com-setor-inicial` all still work as aliases. What did change and has no alias:
the **JSON keys** of the `--out` file (`r_mediana` → `r_median`, `n_seres` →
`n_beings`, `por_gen` → `per_gen`, `delta_medio` → `delta_mean`, and so on — the
commit message lists all of them). Any lab notebook keyed on the old names needs
updating; nothing in the lab reads that JSON today.

The ledger key `parcial` became `partial` in `laps.rs`. `laps.py` reads BOTH, and a
record with neither as closed, so the u1/u1v/u3/u10b ledgers already on disk report
exactly the numbers they reported before (verified: the u1v three-pair command prints
identical numbers before and after).
