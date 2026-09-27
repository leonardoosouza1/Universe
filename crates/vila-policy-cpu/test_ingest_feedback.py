"""Closed Paper feedback must retain identity and reject incomplete episodes."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from ingest_feedback import ingest


BIN = Path(__file__).parent / "target/debug/vila-policy-cpu"


class FeedbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BIN.is_file():
            raise unittest.SkipTest("build vila-policy-cpu before integration tests")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.control = root / "control"
        self.control.mkdir()
        self.episode = root / "A-test"
        (self.episode / "day_0000").mkdir(parents=True)
        self.state = self.control / "state.json"
        self.receipt = self.control / "feedback.json"
        self.decision_path = self.control / "decision.json"
        self.pilot_path = self.control / "pilot.json"
        state = dict(schema="vila-goal-policy/v2", features=["bias", "negative_distance",
                     "negative_path_length", "negative_crowding", "night"],
                     weights=[0.0] * 5, updates=0, candidate_stats={})
        decision = dict(schema="vila-goal-policy/v2", goal="return_home",
                        action="walk_to", episode="A-test", villager_uuid="uuid",
                        tick=100, candidate_id="home_route", x=1.0, y=80.0, z=2.0,
                        features=[1.0, -0.5, -0.2, 0.0, 1.0],
                        expected_reward=0.0, policy_updates=0)
        self.decision_path.write_text(json.dumps(dict(schema="vila-pilot-dispatch/v1",
                                                    seq=1, live_tick=110,
                                                    state=state, decision=decision)))
        self.pilot_path.write_text(json.dumps(dict(seq=1, episode="A-test",
                                                 villager_uuid="uuid", issued_tick=110,
                                                 expires_tick=130, action="walk_to",
                                                 x=1.0, y=80.0, z=2.0, speed=0.6)))
        (self.episode / "episode.json").write_text(json.dumps(dict(
            id="A-test", start_tick=100, day_ticks=24000,
            ai_pilot_enabled=True, recorder_enabled=True)))
        (self.episode / "episode_end.json").write_text(json.dumps(dict(
            end_tick=340, lines_dropped=0)))
        self.action = dict(tick=120, type="pilot_action", v=9, seq=1,
                           episode="A-test", villager_uuid="uuid", action="walk_to",
                           path_accepted=True, target_x=1.0, target_y=80.0,
                           target_z=2.0, start_distance=10.0)
        self.outcome = dict(tick=320, type="pilot_outcome", v=9, seq=1,
                            episode="A-test", villager_uuid="uuid", action_tick=120,
                            reason="deadline", start_distance=10.0,
                            remaining_distance=9.0, progress=1.0, arrived=False,
                            villager_available=True, had_home_before=True,
                            has_home_after=True, slept_during=False,
                            health_before=5.0, health_after=5.0)

    def write_events(self, events):
        with gzip.open(self.episode / "day_0000/events.jsonl.gz", "wt") as output:
            for event in events:
                output.write(json.dumps(event) + "\n")

    def run_ingest(self):
        return ingest(self.decision_path, self.pilot_path, self.episode, BIN,
                      self.state, self.receipt)

    def test_real_rust_learn_updates_once_and_matches_rescue(self):
        self.write_events([self.action, dict(tick=200, type="rescue", v=9), self.outcome])
        receipt = self.run_ingest()
        self.assertTrue(receipt["outcome"]["rescued"])
        self.assertFalse(receipt["outcome"]["slept"])
        self.assertEqual(json.loads(self.state.read_text())["updates"], 1)
        self.assertLess(receipt["reward"], 0)
        with self.assertRaises(ValueError):
            self.run_ingest()

    def test_missing_sleep_contract_fails_without_state(self):
        self.outcome.pop("slept_during")
        self.write_events([self.action, self.outcome])
        with self.assertRaisesRegex(ValueError, "sleep outcome"):
            self.run_ingest()
        self.assertFalse(self.state.exists())

    def test_dropped_lines_fail_closed(self):
        (self.episode / "episode_end.json").write_text(json.dumps(dict(
            end_tick=340, lines_dropped=1)))
        self.write_events([self.action, self.outcome])
        with self.assertRaisesRegex(ValueError, "dropped lines"):
            self.run_ingest()
        self.assertFalse(self.state.exists())


if __name__ == "__main__":
    unittest.main()
