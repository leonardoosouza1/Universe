"""Evidence-contract tests only: JSON records, no CPU simulation or neural oracle."""
import json
import tempfile
import unittest
from pathlib import Path
from audit_taxi_learning import audit

class EvidenceContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.write('command.json', ['gpu_evolve', '--gens', '1'])
        self.write('organism_gen0000.json', dict(generation=0, controller='evolved-native', learning_evidence=True))
        self.write('body_gen0000.json', dict(body_enabled=False, organism_metabolism_neutralized=True, population=[dict(organism_genes=[0]*16)]))
        being = dict(slot=0, genome=0, replica_start=0, start_index=0, success=True, alive=True, score=1000, accepted_pickup_tick=10,
                     accepted_dropoff_tick=30, outcome_tick=30, delivery_delta=1, distance_m=18,
                     contact_ticks=0, collision_energy_j=0)
        self.write('school_result_gen0000.json', dict(generation=0, beings=[being], attempts=1,
            genome_count=1, successes=1, horizon_ticks=64, running=0, failures_by_code={},
            event_span='judged_attempt_frozen_at_outcome_v1'))
        self.write('selection_gen0000.json', dict(objective='school_bench', scores=[1000],
            genome_count=1, replicas_per_genome=1))
        self.write('organism_result_gen0000.json', [dict(slot=0, tick=64, alive=1, deliveries=1,
            energy_j=4900000, water_kg=10, engine_temp_c=25, engine_wear=0)])

    def write(self, name, value):
        (self.path / name).write_text(json.dumps(value))

    def mutate(self, name, change):
        value = json.loads((self.path / name).read_text()); change(value); self.write(name, value)

    def test_complete_learned_attempt(self):
        self.assertEqual(audit(self.path)['status'], 'passed_recorded_attempt')

    def test_all_starts_cannot_pool_different_genotypes(self):
        school = json.loads((self.path / 'school_result_gen0000.json').read_text())
        template = school['beings'][0]
        beings = [dict(template, slot=i, genome=i//2, replica_start=i%2, start_index=i%2, success=i%2==0,
                       delivery_delta=int(i%2==0), accepted_dropoff_tick=30 if i%2==0 else None)
                  for i in range(4)]
        self.write('body_gen0000.json', dict(body_enabled=False, organism_metabolism_neutralized=True, population=[dict(organism_genes=[0]*16) for _ in range(4)]))
        school.update(beings=beings, attempts=4, genome_count=2, successes=2)
        self.write('school_result_gen0000.json', school)
        body = json.loads((self.path / 'organism_result_gen0000.json').read_text())[0]
        self.write('organism_result_gen0000.json', [dict(body, slot=i) for i in range(4)])
        self.write('selection_gen0000.json', dict(objective='school_bench', scores=[1,1], genome_count=2, replicas_per_genome=2))
        self.assertEqual(audit(self.path)['status'], 'passed_recorded_attempt')
        self.assertEqual(audit(self.path, require_all_starts=True)['status'], 'failed')

    def test_duplicate_start_cannot_certify_distinct_coverage(self):
        school = json.loads((self.path / 'school_result_gen0000.json').read_text())
        school.update(beings=[dict(school['beings'][0], slot=i, replica_start=i) for i in range(2)],
                      attempts=2, successes=2)
        self.write('school_result_gen0000.json', school)
        self.write('body_gen0000.json', dict(body_enabled=False, organism_metabolism_neutralized=True,
                   population=[dict(organism_genes=[0]*16) for _ in range(2)]))
        body = json.loads((self.path / 'organism_result_gen0000.json').read_text())[0]
        self.write('organism_result_gen0000.json', [dict(body, slot=i) for i in range(2)])
        self.write('selection_gen0000.json', dict(objective='school_bench', scores=[1000],
                   genome_count=1, replicas_per_genome=2))
        with self.assertRaisesRegex(ValueError, 'distinct start'):
            audit(self.path, require_all_starts=True)

    def test_wrong_replica_identity_rejected(self):
        self.mutate('school_result_gen0000.json', lambda r: r['beings'][0].update(replica_start=1))
        with self.assertRaisesRegex(ValueError, 'replica identity'):
            audit(self.path, require_all_starts=True)

    def test_single_replica_population_can_use_different_starts(self):
        school = json.loads((self.path / 'school_result_gen0000.json').read_text())
        school.update(beings=[dict(school['beings'][0], slot=i, genome=i, start_index=i)
                              for i in range(2)], attempts=2, successes=2, genome_count=2)
        self.write('school_result_gen0000.json', school)
        self.write('body_gen0000.json', dict(body_enabled=False, organism_metabolism_neutralized=True,
                   population=[dict(organism_genes=[0]*16) for _ in range(2)]))
        body = json.loads((self.path / 'organism_result_gen0000.json').read_text())[0]
        self.write('organism_result_gen0000.json', [dict(body, slot=i) for i in range(2)])
        self.write('selection_gen0000.json', dict(objective='school_bench', scores=[1000,1000],
                   genome_count=2, replicas_per_genome=1))
        self.assertEqual(audit(self.path)['status'], 'passed_recorded_attempt')

    def test_witness_cannot_pass_as_learning(self):
        self.mutate('organism_gen0000.json', lambda r: r.update(controller='organism-taxi-reference-witness-v2', learning_evidence=False))
        with self.assertRaisesRegex(ValueError, 'scripted/unknown'):
            audit(self.path)

    def test_conflicting_command_rejected(self):
        self.write('command.json', ['gpu_evolve', '--gens', '1', '--taxi-reference-witness'])
        with self.assertRaisesRegex(ValueError, 'scripted witness'):
            audit(self.path)

    def test_incomplete_battery(self):
        self.write('command.json', ['gpu_evolve', '--gens', '2'])
        with self.assertRaisesRegex(ValueError, 'incomplete battery'):
            audit(self.path)

    def test_truncated_body(self):
        self.mutate('organism_result_gen0000.json', lambda r: r[0].update(tick=32))
        with self.assertRaisesRegex(ValueError, 'unfinished generation'):
            audit(self.path)

    def test_event_after_frozen_outcome(self):
        self.mutate('school_result_gen0000.json', lambda r: r['beings'][0].update(accepted_dropoff_tick=31))
        with self.assertRaisesRegex(ValueError, 'outside judged span'):
            audit(self.path)

    def test_unbacked_delivery(self):
        self.mutate('organism_result_gen0000.json', lambda r: r[0].update(deliveries=0))
        with self.assertRaisesRegex(ValueError, 'authoritative body'):
            audit(self.path)

    def test_invalid_selection(self):
        self.mutate('selection_gen0000.json', lambda r: r.update(scores=[float('nan')]))
        with self.assertRaisesRegex(ValueError, 'selection vector'):
            audit(self.path)

    def test_false_metabolic_control(self):
        self.mutate('body_gen0000.json', lambda r: r['population'][0].update(organism_genes=[1]*16))
        with self.assertRaisesRegex(ValueError, 'neutral metabolism claim'):
            audit(self.path)

    def test_wrong_slot(self):
        self.mutate('organism_result_gen0000.json', lambda r: r[0].update(slot=1))
        with self.assertRaisesRegex(ValueError, 'identity'):
            audit(self.path)

    def test_no_delivery_fails_behavior(self):
        self.mutate('school_result_gen0000.json', lambda r: (r.update(successes=0), r['beings'][0].update(success=False, delivery_delta=0, accepted_dropoff_tick=None)))
        self.assertEqual(audit(self.path)['status'], 'failed')

if __name__ == '__main__':
    unittest.main()
