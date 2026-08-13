import numpy as np, matplotlib.pyplot as plt
from replay_reader import load_run
import glob

plt.figure(figsize=(20, 8))
for path in sorted(glob.glob('telemetry/*/generations.ndjson')):
    run_id = path.split('/')[1]
    frames, gens = load_run(run_id)

    gen_means = {}
    for gen in sorted(np.unique(frames['generation'])):
        g = frames[(frames['generation'] == gen) & (frames['rank'] == 0)]
        if len(g): gen_means[gen] = g['sensory_wave'].mean(axis=0)

    gens_list = sorted(gen_means.keys())
    diffs = np.array([gen_means[g] for g in gens_list])

    plt.imshow(diffs.T, aspect='auto', cmap='RdBu_r', vmin=-64, vmax=64)
    plt.xlabel('Generation'); plt.ylabel('Sensory Channel (0-191)')
    plt.colorbar(label='Mean Activation'); plt.title(f'Neural Activation Evolution: {run_id}')
    plt.tight_layout(); plt.savefig(f'neural_activation_{run_id}.png', dpi=150)
print("✓ neural_activation_*.png")
