import numpy as np, pathlib, pandas as pd

DTYPE = np.dtype([
    ('frame_tick', '<u4'), ('generation', '<u2'), ('rank', 'u1'), ('flags', 'u1'),
    ('pos_x', '<f4'), ('pos_z', '<f4'), ('heading', '<f4'), ('speed_ms', '<f4'),
    ('throttle', '<f4'), ('steer', '<f4'), ('brake', '<f4'), ('hp', '<f4'),
    ('distance_m', '<f4'), ('track_progress', '<f4'),
    ('total_laps', '<u2'), ('racer_id', '<u2'), ('joy_tick', '<f4'),
    ('sensory_wave', 'i1', (192,)), ('_reserved', 'u1', (8,)),
])

def load_run(run_id):
    base = pathlib.Path(f'telemetry/{run_id}')
    frames = np.fromfile(base / 'replay.bin', dtype=DTYPE)
    gens = pd.read_json(base / 'generations.ndjson', lines=True)
    return frames, gens

def get_champion(frames, gen):
    return frames[(frames['generation'] == gen) & (frames['rank'] == 0)]
