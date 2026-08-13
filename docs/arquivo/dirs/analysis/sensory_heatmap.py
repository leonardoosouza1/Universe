import numpy as np, matplotlib.pyplot as plt
from replay_reader import load_run
import glob

plt.figure(figsize=(20, 8))
for path in sorted(glob.glob('telemetry/*/generations.ndjson')):
    run_id = path.split('/')[1]
    frames, _ = load_run(run_id)
    ch = frames[frames['rank'] == 0]

    bins = np.linspace(ch['pos_x'].min(), ch['pos_x'].max(), 50)
    heatmap = np.zeros((192, len(bins)-1))
    for i in range(len(bins)-1):
        mask = (ch['pos_x'] >= bins[i]) & (ch['pos_x'] < bins[i+1])
        if mask.sum() > 0:
            heatmap[:, i] = ch[mask]['sensory_wave'].mean(axis=0)

    plt.imshow(heatmap, aspect='auto', cmap='RdBu_r', vmin=-64, vmax=64)
    plt.xlabel('Track X Position'); plt.ylabel('Sensory Channel (0-191)')
    plt.colorbar(label='Mean Activation'); plt.title(f'Sensory Heatmap: {run_id}')
    plt.tight_layout(); plt.savefig(f'sensory_heatmap_{run_id}.png', dpi=150)
print("✓ sensory_heatmap_*.png")
