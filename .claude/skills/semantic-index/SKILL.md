---
name: semantic-index
description: Ask the codebase a question in plain language and get file:line back. Use BEFORE grepping when you do not already know the identifier — for "where is X decided", "why is Y like this", "what already exists for Z" — and before proposing any new mechanism, to check whether it is already implemented and merely switched off.
---

# Semantic index — ask the repository, do not guess

This repository is ~55k lines of Rust plus WGSL shaders, and its most valuable
knowledge is in doc-comments that record *measured* findings. `grep` only finds
those when you already know the exact word. This index answers plain-language
questions and returns `file:line`.

## Commands

```bash
PY=/home/leonardo/projects/LLM/.venv-rocm/bin/python

$PY tools/semantic_index.py "how is the fuel tank sized"     # ask a question
$PY tools/semantic_index.py --symbol frente_do_waypoint      # exact identifier
$PY tools/semantic_index.py --stats                          # index health
$PY tools/semantic_index.py --build                          # rebuild (~2 min, CPU)
```

Rebuild after a large refactor or translation wave, otherwise the index still
answers with the old text. The index lives in `.semantic-index/` and is
git-ignored — it is derived data, every contributor builds their own.

## When to use it

1. **Before proposing a new mechanism.** The single most expensive recurring
   mistake in this project is building something that already exists and is
   merely disabled (`cortex_l3` had 1152 mutated weights and was not in the
   forward pass; the sensory KAN was frozen at identity). Ask first.
2. **When you need the *why*, not the *what*.** "why does the pilot override run
   before the physics state is assembled" is answerable here; grep is not.
3. **Before an audit.** Get the candidate `file:line` list from the index, then
   open each one — the index points, it does not prove.

## How to read the output

```
0.567  src/bin/gpu_evolve/main.rs:167  [fn] frente_do_waypoint
        fn frente_do_waypoint(wps: &[bevy::math::Vec2], wi: usize) -> [f32; 3] {
```

Score is cosine similarity plus a lexical-overlap bonus; it is a ranking, not a
probability. Anything above ~0.45 is usually worth opening.

## Limits — read these before trusting a hit

- **A hit is a pointer, never evidence.** This project's rule stands: open the
  `file:line` and read it. Comments in this repository have been wrong before.
- **The embedding model is English-centric** and part of the repository is still
  being migrated from Portuguese (see `docs/MIGRATION.md`). For a not-yet-migrated
  file, the lexical half of the score carries most of the weight — so query with
  identifiers you expect to see, not only prose.
- **Archived documents are excluded** (`docs/arquivo`, `docs/superpowers`): they
  are superseded reports and used to bury live code in the results.
- **Empty result is information**: if a mechanism does not surface, prefer
  "not found in the index" over asserting that it does not exist — check with a
  targeted grep before concluding.
