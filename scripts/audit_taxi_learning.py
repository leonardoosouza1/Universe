#!/usr/bin/env python3
"""Read existing GPU evidence; never simulate or reconstruct a neural forward pass.

Exit 1 when the recorded school battery has no complete contact-free delivery.
Passing is limited to the recorded attempts, not city coverage or generalization.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def audit(directory, require_all_starts=False):
    generations = []
    controllers = set()
    command_file = directory / 'command.json'
    command = json.loads(command_file.read_text()) if command_file.exists() else None
    if command is None:
        # Batteries launched from a shell script have no command.json; since
        # 2026-09-24 every per-generation manifest carries the argv the binary
        # itself ran under, which is the same declaration from a better witness.
        first = directory / 'organism_gen0000.json'
        if first.exists():
            command = json.loads(first.read_text()).get('provenance', {}).get('argv')
            if not isinstance(command, list):
                command = None
    expected_generations = None
    if command is not None and '--gens' in command:
        expected_generations = int(command[command.index('--gens') + 1])
    files = sorted(directory.glob('school_result_gen*.json'))
    if expected_generations is None:
        raise ValueError('missing declared generation count (no command.json and no provenance.argv in organism_gen0000.json); cannot certify a complete battery')
    if len(files) != expected_generations:
        raise ValueError(f'incomplete battery: {len(files)} / {expected_generations} generations')
    previous_generation = None
    for file in files:
        record = json.loads(file.read_text())
        generation = record['generation']
        if previous_generation is not None and generation != previous_generation + 1:
            raise ValueError('noncontiguous generations')
        previous_generation = generation
        manifest = json.loads((directory / f'organism_gen{generation:04}.json').read_text())
        controller = manifest.get('controller')
        controllers.add(controller)
        if controller not in ('evolved-native', 'evolved-linear-wave-v2') or manifest.get('learning_evidence') is not True:
            raise ValueError(f'{file}: scripted/unknown controller cannot certify learned behavior: {controller}')
        if '--taxi-reference-witness' in (command or []):
            raise ValueError('command selects a scripted witness; conflicting learning claim')
        if manifest.get('generation') != generation:
            raise ValueError('generation identity mismatch')
        beings = record['beings']
        selection = json.loads((directory / f'selection_gen{generation:04}.json').read_text())
        body = json.loads((directory / f'organism_result_gen{generation:04}.json').read_text())
        if len(beings) != record['attempts'] or len(body) != len(beings):
            raise ValueError(f'{file}: population/attempt mismatch')
        if not isinstance(selection.get('replicas_per_genome'), int) or selection['replicas_per_genome'] <= 0:
            raise ValueError('replica count must be a positive integer')
        if any(b['genome'] != b['slot'] // selection['replicas_per_genome'] for b in beings):
            raise ValueError('attempt-to-genotype mapping violates replica contract')
        if {b['slot'] for b in beings} != set(range(len(beings))) or {b['slot'] for b in body} != set(range(len(body))):
            raise ValueError('duplicate or missing body/attempt identity')
        replicas = selection['replicas_per_genome']
        if any(type(b.get('replica_start')) is not int
               or b['replica_start'] != b['slot'] % replicas for b in beings):
            raise ValueError('attempt replica identity violates slot mapping')
        starts_by_genome = {}
        for b in beings:
            start = b.get('start_index')
            if type(start) is not int or start < 0:
                raise ValueError('missing or invalid distinct start identity')
            starts = starts_by_genome.setdefault(b['genome'], set())
            if start in starts:
                raise ValueError('duplicate distinct start within one genotype')
            starts.add(start)
        # K=1 population trials legitimately assign a different start per genome.
        # Replicated K>1 selection, however, compares a common start batch.
        if replicas > 1 and len({tuple(sorted(starts)) for starts in starts_by_genome.values()}) != 1:
            raise ValueError('genotypes were evaluated on different distinct start sets')
        if (len(selection['scores']) != record['genome_count']
                or len(selection['scores']) != selection['genome_count']
                or selection['replicas_per_genome'] * selection['genome_count'] != len(beings)
                or not all(math.isfinite(s) for s in selection['scores'])):
            raise ValueError('invalid selection vector or replica population')
        if record.get('running', 0) or any(b['tick'] != record['horizon_ticks'] for b in body):
            raise ValueError('unfinished generation or inconsistent final body horizon')
        if sum(int(b['success']) for b in beings) != record['successes']:
            raise ValueError('success aggregate does not match per-being evidence')
        if record.get('event_span') != 'judged_attempt_frozen_at_outcome_v1':
            raise ValueError('unknown event phase; cannot mix final-life and judged events')
        body_by_slot = {b['slot']: b for b in body}
        for b in beings:
            for field in ('score', 'distance_m', 'collision_energy_j', 'contact_ticks', 'delivery_delta', 'outcome_tick'):
                if not isinstance(b[field], (int, float)) or not math.isfinite(b[field]):
                    raise ValueError(f'invalid {field} at slot {b["slot"]}')
            if not 0 <= b['outcome_tick'] <= record['horizon_ticks']:
                raise ValueError('outcome outside recorded horizon')
            for field in ('accepted_pickup_tick', 'accepted_dropoff_tick'):
                t = b[field]
                if t is not None and (not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 <= t <= b['outcome_tick']):
                    raise ValueError('service event outside judged span')
            if b['delivery_delta'] > body_by_slot[b['slot']]['deliveries']:
                raise ValueError('school delivery not supported by authoritative body counter')
        complete = [b for b in beings if b['success']
                    and b['accepted_pickup_tick'] is not None
                    and b['accepted_dropoff_tick'] is not None
                    and b['accepted_pickup_tick'] < b['accepted_dropoff_tick'] <= b['outcome_tick']
                    and b['delivery_delta'] >= 1 and b['distance_m'] > 0]
        clean = [b for b in complete if b['alive'] and b['contact_ticks'] == 0
                 and b['collision_energy_j'] == 0]
        body_contract = json.loads((directory / f'body_gen{generation:04}.json').read_text())
        metabolic = [row['organism_genes'] for row in body_contract['population']]
        if len(metabolic) != len(beings) or any(len(g) != 16 or not all(math.isfinite(v) for v in g) for g in metabolic):
            raise ValueError('missing or invalid metabolic genome evidence')
        neutralized = body_contract['organism_metabolism_neutralized']
        if neutralized and any(any(v != 0 for v in g) for g in metabolic):
            raise ValueError('neutral metabolism claim contradicts recorded genes')
        clean_per_genome = {}
        for b in clean:
            clean_per_genome[b['genome']] = clean_per_genome.get(b['genome'], 0) + 1
        robust = sum(count == selection['replicas_per_genome'] for count in clean_per_genome.values())
        generations.append(dict(generation=generation, attempts=len(beings),
            all_start_clean_genotypes=robust,
            vehicle_capacity_evolution=body_contract['body_enabled'],
            metabolism_neutralized=neutralized,
            distinct_metabolic_genomes=len({tuple(g) for g in metabolic}),
            accepted_pickups=sum(b['accepted_pickup_tick'] is not None for b in beings),
            complete_deliveries=len(complete), contact_free_deliveries=len(clean),
            selection_objective=selection['objective'], best_selection=max(selection['scores']),
            median_selection=statistics.median(selection['scores']),
            failures=record['failures_by_code'], max_distance_m=max(b['distance_m'] for b in beings),
            alive=sum(b['alive'] > .5 for b in body),
            energy_j_range=[min(b['energy_j'] for b in body), max(b['energy_j'] for b in body)],
            water_kg_range=[min(b['water_kg'] for b in body), max(b['water_kg'] for b in body)],
            max_engine_temp_c=max(b['engine_temp_c'] for b in body),
            max_engine_wear=max(b['engine_wear'] for b in body)))
    if not generations:
        raise ValueError('no school results; missing evidence is not a passing result')
    metric = 'all_start_clean_genotypes' if require_all_starts else 'contact_free_deliveries'
    passed = any(g[metric] > 0 for g in generations)
    return dict(schema_version=1, status='passed_recorded_attempt' if passed else 'failed',
        scope='recorded learned full-trip school attempts only; not city coverage or generalization',
        complete_generation_count=len(generations), controllers=sorted(controllers),
        acceptance='one genotype completes every evaluated start cleanly' if require_all_starts else 'one clean recorded full trip',
        generations=generations, total_complete_deliveries=sum(g['complete_deliveries'] for g in generations),
        total_contact_free_deliveries=sum(g['contact_free_deliveries'] for g in generations),
        total_accepted_pickups=sum(g['accepted_pickups'] for g in generations),
        body_evolution_causal_benefit='not measured; changed genes/capacities do not establish benefit',
        city_coverage='not measured; odometer is not spatial coverage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--require-all-starts', action='store_true', help='require one same genotype to pass all its evaluated starts without contact')
    args = parser.parse_args()
    try:
        report = audit(args.run, args.require_all_starts)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    text = json.dumps(report, indent=2)
    if args.out:
        if not args.out.resolve().is_relative_to(Path('/media/llm-data')):
            parser.error('--out must be below /media/llm-data (storage policy)')
        if args.out.exists():
            parser.error('refusing to overwrite prior evidence')
        args.out.write_text(text + '\n')
    print(text)
    raise SystemExit(0 if report['status'] == 'passed_recorded_attempt' else 1)


if __name__ == '__main__':
    main()
