# Universe: mandatory workflow for agents

Before editing, diagnosing behavior, running a battery, or publishing a replay,
read [the validation workflow](docs/VALIDATION-WORKFLOW.md). It is the operational
entry point for this repository and its trace/UI consumers, alongside the
scientific handbooks referenced by the parent `AGENTS.md`.

1. Open one evidence record using
   [the template](docs/templates/BEHAVIOR-INVESTIGATION.md). State the user's
   observable acceptance, exact failing run, and current unresolved failure.
2. Verify the executed path and trace the **same being, generation and tick** from
   input through brain, applied action, body, physics, task and recorder. Confirm
   active shader includes; a similarly named legacy file is not evidence.
3. Find the first demonstrated discrepancy before changing behavior. Use a
   bounded, one-variable test to distinguish a suspected cause. Missing evidence
   stays `NOT MEASURED`; do not fill it with a plausible story.
4. Validate numerical brain/body/physics changes on the **real GPU only**, in
   small sequential runs. No CPU simulation, CPU oracle, software Vulkan fallback
   or unbounded battery. Host file/metric inspection is allowed. Compile only the
   required target, with one job and low priority.
5. Recheck the original failure, adjacent contracts and the actual replay in the
   browser. An exit code, changed weight, fitness increase, speed peak, moving
   animation or scripted pilot does not prove learned useful behavior.
6. If acceptance fails, record why, revise the hypothesis and continue within the
   execution budget. Do not add unrelated layers/features or publish a diagnostic
   as a successful result. Do not conceal failures by changing the map, horizon,
   controller, initial passenger or acceptance after seeing the result.
7. Every completion claim links its command, artifact and measured acceptance.
   Review delegated work directly; an agent's summary is not validation.

The research runner enforces the source/binary-bound GPU mechanism receipt from
`scripts/check_native_flow_gpu.py`. Behavioral and visual gates remain review
gates, not a claim that CI enforces them. The responsible agent must enforce
them and report any missing gate explicitly. User instructions take
precedence. Preserve unrelated edits and existing scientific evidence.
