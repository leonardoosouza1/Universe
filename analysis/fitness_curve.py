import pandas as pd, glob, matplotlib.pyplot as plt
from replay_reader import load_run

plt.figure(figsize=(14, 6))
for path in sorted(glob.glob('telemetry/*/generations.ndjson')):
    run_id = path.split('/')[1]
    _, gens = load_run(run_id)
    plt.plot(gens['gen'], gens['champion_dist_m'], label=run_id, marker='o', markersize=3)
    plt.fill_between(gens['gen'], gens['p50_dist_m'], gens['p90_dist_m'], alpha=0.1)

plt.xlabel('Generation'); plt.ylabel('Champion Distance (m)')
plt.title('Fitness Curve: Champion Distance by Generation'); plt.legend()
plt.grid(); plt.tight_layout(); plt.savefig('fitness_curve.png', dpi=150)
print("✓ fitness_curve.png")
