---
name: Bug report
about: Something crashes, hangs, or produces a result that is provably wrong
title: ''
labels: bug
assignees: ''
---

## The exact command

<!-- Copy-paste it, with every flag and the seed. A bug in this project without the
     command is not reproducible: results change with --seed, --gens, --ticks, the
     track list and every behaviour flag. -->

```bash

```

## GPU and driver

<!-- The tick runs on wgpu/WGSL compute; behaviour differs by backend and driver
     version. On Linux, `wgpu-info` or the block below is enough. -->

- GPU:
- Driver / runtime version:
- OS:
- Backend, if you know it (Vulkan / GL):
- `rustc --version`:

## The log

<!-- As text, not a screenshot. Include the lines BEFORE the failure — the crash line
     alone usually says less than what led to it. Trim the middle of a long run if you
     must, and mark where you trimmed. -->

```

```

## What you expected, and what you got

<!-- If this is a wrong-number bug rather than a crash: give the number you got, the
     number you expected, and how you know what to expect (the metric's physical
     ceiling, a previous run, another seed). A hard 0% or a hard 100% is usually a
     metric hitting its limit rather than the model failing. -->

## Does it reproduce?

- [ ] Every run
- [ ] Only with certain seeds (which:            )
- [ ] Saw it once and cannot reproduce it

## Anything else

<!-- The champion `.dna` you loaded, the run directory under
     /media/llm-data/universe-runs/, the last commit where it worked. -->
