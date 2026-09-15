import unittest
from types import SimpleNamespace

from checkup_report import motor_summary, summarize
from run_champion import control_arguments


def sample(raw, applied, alive=True, being=0, speed=15):
    wave = [0] * 192
    wave[104], wave[118], wave[123] = 127, 5, 120
    return {"being": being, "wave": wave, "body": {
        "alive": alive, "speed": speed,
        "raw_motor": [.4, .2, raw], "applied_motor": [.3, .1, applied]}}


class CheckupReportTests(unittest.TestCase):
    def test_negative_policy_and_suppressed_positive_are_separate(self):
        result = motor_summary([sample(-.4, 0), sample(.4, 0), sample(.2, .15)])
        self.assertEqual(result["raw_brake_positive_count"], 2)
        self.assertEqual(result["applied_brake_over_0_05_count"], 1)
        self.assertEqual(result["positive_raw_brake_but_zero_applied_count"], 1)
        self.assertIn("Requires motor_trace", result["reason_attribution"])

    def test_dead_samples_do_not_inflate_braking_and_coverage_is_explicit(self):
        report = {"run": "fixture", "generation": 0, "input_sha256": {},
                  "meta": {"selected_being": 0, "sample_every": 16},
                  "observations": [sample(-.2, 0), sample(1, 1, alive=False)],
                  "windows": [], "beings": {"0": [], "1": []}}
        result = summarize(report)
        self.assertEqual(result["population"]["recorded_beings"], 2)
        self.assertEqual(result["neural_coverage"]["observed_beings"], [0])
        self.assertFalse(result["neural_coverage"]["selection_is_best"])
        selected = result["sampled_beings"]["0"]
        self.assertEqual(selected["alive"]["samples"], 1)
        self.assertEqual(selected["alive"]["raw_brake_positive_count"], 0)
        self.assertEqual(selected["channels"]["118"]["name"], "forward_distance_normalized")
        self.assertIn("lateral", selected["channels"]["123"]["name"])

    def test_missing_or_nonfinite_measurements_do_not_become_zero_intent(self):
        broken = sample(float("nan"), 0)
        result = motor_summary([broken, {"body": {}}])
        self.assertEqual(result["usable_motor_samples"], 0)
        self.assertIsNone(result["raw_brake_mean"])

    def test_experiment_flags_do_not_enable_other_interventions(self):
        self.assertEqual(control_arguments(SimpleNamespace()), [])
        self.assertEqual(control_arguments(SimpleNamespace(throttle_arbitration="signed")),
                         ["--throttle-arbitration", "signed"])
        self.assertEqual(control_arguments(SimpleNamespace(wall_probes="continuous")),
                         ["--wall-probes", "continuous"])

    def test_gate_reason_comes_only_from_recorded_trace(self):
        observation = sample(.5, 0, speed=20)
        observation["body"]["motor_trace"] = {
            "pre_physics_speed_mps": 1.9, "brake_suppression": "low_speed_gate",
            "override_cause": "none"}
        result = motor_summary([observation])
        self.assertEqual(result["recorded_brake_suppression"], {"low_speed_gate": 1})


if __name__ == "__main__":
    unittest.main()
