# Organism GPU verification

Run the bounded Radeon/Vulkan continuation check with an unused output path:

```bash
scripts/verify_organism_gpu.sh /media/llm-data/universe-runs/organism-gpu-check-$(date +%s)
```

The script runs focused ignored GPU fixtures, builds the debug `gpu_evolve`, and
runs 192 ticks for two beings, checkpoints at
tick 96 (inside a telemetry window), resumes into a different directory, and compares the entire final observation (except its relocated genome path) and organism result. It is a mechanical continuation check only; it does not measure learning,
fitness, or evolutionary progress. The script rejects an existing output path,
uses one build job and one Rayon thread, and requires the configured Radeon
Vulkan ICD. A software adapter is never accepted as a fallback.
