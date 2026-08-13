import glob, matplotlib.pyplot as plt, matplotlib.cm as cm
from replay_reader import load_run, get_champion

plt.figure(figsize=(16, 10))
for path in sorted(glob.glob('telemetry/*/generations.ndjson')):
    run_id = path.split('/')[1]
    frames, gens = load_run(run_id)
    max_gen = gens['gen'].max()
    for gen in list(range(1, max_gen+1, max(1, max_gen//10))) + [max_gen]:
        ch = get_champion(frames, gen)
        if len(ch) > 0:
            color = cm.viridis(gen / max_gen)
            plt.plot(ch['pos_x'], ch['pos_z'], color=color, alpha=0.6, lw=1, label=f'{run_id} gen{gen}')

plt.axis('equal'); plt.xlabel('X (m)'); plt.ylabel('Z (m)')
plt.title('Champion Trajectory Evolution'); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig('trajectory_map.png', dpi=150)
print("✓ trajectory_map.png")
