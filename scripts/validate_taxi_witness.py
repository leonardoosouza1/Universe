#!/usr/bin/env python3
"""Validate recorded physical feasibility; never simulates a model on the host."""
import argparse
import json
import math
from pathlib import Path


def validate(root: Path):
    scenario = json.loads((root / 'organism_gen0000.json').read_text())
    assert scenario.get('controller') == 'organism-taxi-reference-witness-v1', 'reference identity missing'
    assert scenario.get('learning_evidence') is False, 'must not be labeled learned behavior'
    results = json.loads((root / 'organism_result_gen0000.json').read_text())
    assert len(results) == 1, 'witness requires one being'
    outcome = results[0]
    assert outcome['deliveries'] == 1 and outcome['alive'] == 1, 'one completed delivery and surviving body required'
    assert outcome['collision_energy_j'] == 0, 'collision during witness'
    assert outcome['traffic'][8] == 0, 'red light violation during witness'
    assert outcome['energy_j'] > 0 and outcome['water_kg'] > 0, 'depleted resources'
    picked_up = False
    delivered = None
    max_post_delivery_drift = 0.0
    samples = 0
    with (root / 'observation.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row['generation'] == 0 and row['being'] == 0
            b, t = row['body'], row['taxi']
            assert all(math.isfinite(b[k]) for k in ['speed', 'x', 'z'])
            samples += 1
            if t['onboard'] > .5:
                picked_up = True
            if t['deliveries'] == 1 and delivered is None:
                assert picked_up, 'no recorded pickup before delivery'
                distance = math.hypot(b['x'] - t['destination_x_m'], b['z'] - t['destination_z_m'])
                assert distance <= 4 and b['speed'] < .5, 'no observed legal delivery endpoint'
                delivered = {'tick': row['tick'], 'seconds': row['tick'] / scenario['tick_rate'], 'x_m': b['x'], 'z_m': b['z'], 'speed_mps': b['speed'], 'destination_distance_m': distance}
            if delivered is not None:
                max_post_delivery_drift = max(max_post_delivery_drift, math.hypot(b['x'] - delivered['x_m'], b['z'] - delivered['z_m']))
    assert delivered is not None
    report = {'controller': scenario['controller'], 'learning_evidence': False, 'samples': samples, 'delivery': delivered, 'outcome': outcome, 'max_post_delivery_drift_m': max_post_delivery_drift, 'rest_tolerance_m': .05}
    assert max_post_delivery_drift <= .05, f'brake-held creep {max_post_delivery_drift:.6f}m exceeds .05m'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    report = validate(args.run)
    print(json.dumps(report, indent=2))
