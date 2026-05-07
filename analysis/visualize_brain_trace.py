#!/usr/bin/env python3
"""Visualize brain trace: retina + sensory channels + neural outputs."""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def visualize_brain_snapshot(snapshot, idx=0, save_path=None):
    """Render a single brain snapshot: retina (heatmap) + sensory (line) + outputs (line)."""

    fig = plt.figure(figsize=(16, 10))

    # Extract data
    t = snapshot['t']
    speed = snapshot['speed_kmh']
    retina_fov = np.array(snapshot['retina_fov'], dtype=np.int8).reshape(14, 8)
    sensory = np.array(snapshot['sensory_compact'], dtype=np.int8)
    throttle = snapshot['raw_throttle']
    steer = snapshot['raw_steer']
    brake = snapshot['raw_brake']

    # --- Panel 1: Retina FOV heatmap (8 bins × 14 rows) ---
    ax1 = plt.subplot(2, 3, 1)
    im = ax1.imshow(retina_fov, cmap='RdBu_r', aspect='auto', vmin=-128, vmax=127)
    ax1.set_xlabel('FOV Bin (0=left, 7=right)')
    ax1.set_ylabel('Retina Row (0=horizon, 13=ground)')
    ax1.set_title(f'Retina FOV @ t={t:.1f}s | Speed={speed:.0f}km/h')
    plt.colorbar(im, ax=ax1, label='Intensity (-128=road, 0=dark, 127=wall)')

    # --- Panel 2: Sensory channels bar plot ---
    ax2 = plt.subplot(2, 3, 2)
    labels = ['Speed', 'FlowAlign', 'Curv40m', 'Curv70m', 'Curv120m', 'WallL', 'WallR', 'TTC', 'LateralVel', 'HP']
    colors = ['blue' if s > 0 else 'red' for s in sensory[:10]]
    ax2.bar(range(10), sensory[:10], color=colors, alpha=0.7)
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Channel Value')
    ax2.set_ylim(-128, 127)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_title('Sensory Channels (Compact)')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Neural outputs (throttle, steer, brake) ---
    ax3 = plt.subplot(2, 3, 3)
    outputs = [throttle, steer, brake]
    output_names = ['Throttle', 'Steering', 'Brake']
    colors_out = ['green', 'blue', 'red']
    x_pos = np.arange(len(outputs))
    bars = ax3.bar(x_pos, outputs, color=colors_out, alpha=0.7, width=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(output_names)
    ax3.set_ylabel('Output Intent')
    ax3.set_ylim(-1.2, 1.2)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Neural Outputs (Final Intent)')
    for i, (bar, val) in enumerate(zip(bars, outputs)):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

    # --- Panels 4-6: Time series of sensory + outputs (if multiple snapshots available) ---
    ax4 = plt.subplot(2, 3, 4)
    ax4.text(0.5, 0.5, f't={t:.1f}s\nSnapshot {idx}', ha='center', va='center', fontsize=12, transform=ax4.transAxes)
    ax4.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_brain_trace_sequence(snapshots, output_dir='./vision_replay'):
    """Render all snapshots as a sequence + summary."""

    Path(output_dir).mkdir(exist_ok=True)

    # Individual frames
    for i, snap in enumerate(snapshots):
        path = Path(output_dir) / f'snapshot_{i:02d}.png'
        visualize_brain_snapshot(snap, idx=i, save_path=str(path))

    # Summary: all time series on one plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    times = [s['t'] for s in snapshots]
    throttles = [s['raw_throttle'] for s in snapshots]
    steers = [s['raw_steer'] for s in snapshots]
    brakes = [s['raw_brake'] for s in snapshots]
    speeds = [s['speed_kmh'] for s in snapshots]
    wall_ls = [s['sensory_compact'][5] for s in snapshots]
    wall_rs = [s['sensory_compact'][6] for s in snapshots]

    # Throttle over time
    axes[0].plot(times, throttles, 'g-', linewidth=2, label='Throttle')
    axes[0].fill_between(times, throttles, alpha=0.3, color='green')
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Throttle Intent')
    axes[0].set_ylim(-1.2, 1.2)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Neural Outputs Over Time')

    # Steering + walls over time
    axes[1].plot(times, steers, 'b-', linewidth=2, label='Steering')
    axes[1].plot(times, np.array(wall_ls)/127, 'r--', alpha=0.7, label='Wall Left')
    axes[1].plot(times, np.array(wall_rs)/127, 'orange', linestyle='--', alpha=0.7, label='Wall Right')
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Steering / Wall Proximity')
    axes[1].set_ylim(-1.2, 1.2)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Brake + speed
    axes[2].bar(times, brakes, width=0.01, alpha=0.7, color='red', label='Brake')
    axes[2].plot(times, np.array(speeds)/100, 'k-', linewidth=2, label='Speed (km/h / 100)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Brake / Speed')
    axes[2].set_ylim(-0.1, 1.2)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    summary_path = Path(output_dir) / 'summary.png'
    plt.savefig(str(summary_path), dpi=100)
    print(f"Saved summary: {summary_path}")
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_brain_trace.py <report.md|brain_trace.json> [output_dir]")
        print("\nExample:")
        print("  python visualize_brain_trace.py generations/gen_0001.md ./vision_replay")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './vision_replay'

    # Try to parse JSON directly
    if input_path.suffix == '.json':
        with open(input_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                snapshots = data
            else:
                snapshots = data.get('brain_trace', [])
        visualize_brain_trace_sequence(snapshots, output_dir)
    else:
        print(f"Format not yet supported: {input_path.suffix}")
        print("Save brain_trace as JSON for visualization.")
