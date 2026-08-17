---
name: Feature request
about: Propose a change to the simulation, the brain, the tooling or the docs
title: ''
labels: enhancement
assignees: ''
---

## What you want

<!-- One paragraph. What would exist afterwards that does not exist now? -->

## Why — and what makes you think it would help

<!-- The evidence, if you have it: a measurement, a run that stalled, a limit you hit.
     "It would probably be faster / smarter" is a hypothesis; say so plainly rather
     than dressing it as a result. -->

## Does it change behaviour?

<!-- Physics, perception, fitness, or the shape of the brain's input all invalidate
     every trained champion in the repository, silently. Those changes ship behind a
     CLI flag that defaults to off, with a paired A/B (>= 3 seeds). See
     CONTRIBUTING.md § "The golden rule". -->

- [ ] No — tooling, docs, refactor, or a pure translation
- [ ] Yes, and it would go behind a flag named `--`
- [ ] Not sure

## How would we know it worked?

<!-- Name the number that would move, and the command that would show it. State the
     metric's ceiling under the config you would run: a result that hits a hard 0% or
     a hard 100% is usually the metric at its limit, not the change succeeding. -->

## Did you check whether it already exists?

<!-- Several mechanisms in this tree are implemented but not wired up: state that is
     computed and never reaches the brain, weights that are mutated and never read.
     A grep before proposing is cheaper than an architecture. Paste what you found,
     even if the answer was "nothing". -->

## Are you offering to implement it?

- [ ] Yes
- [ ] No, proposing only
