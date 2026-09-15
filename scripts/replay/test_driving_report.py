import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("driving_report", Path(__file__).with_name("driving_report.py"))
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

CSV_FIELDS = "generation,first_tick,last_tick,being,ticks,alive_ticks,contact_ticks,impact_ticks,contact_events,brake_ticks,overlap_ticks,stationary_ticks,reverse_ticks,near_wall_ticks,speed_sum,distance_m,collision_energy_j,impulse_ns,impact_peak_mps,slip_peak,slip_angle_peak_rad,throttle_sum,brake_sum,x,z,speed,throttle,steer,brake,wall_left,wall_front,wall_right,life,fuel".split(",")


class DrivingReportTests(unittest.TestCase):
    def test_exports_aggregates_and_observation_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            run, out = Path(td) / "run", Path(td) / "out"
            run.mkdir()
            (run / "driving.meta.json").write_text(json.dumps({"schema_version": 1, "selected_being": 0, "retina_rows": 14, "retina_cols": 126, "sample_phase": "pre/body"}))
            row = dict.fromkeys(CSV_FIELDS, "0")
            row.update(ticks="64", alive_ticks="32", contact_ticks="8", impact_ticks="2", speed_sum="64", throttle_sum="16", brake_sum="8", fuel="0.75")
            (run / "driving.csv").write_text(",".join(CSV_FIELDS) + "\n" + ",".join(row[x] for x in CSV_FIELDS) + "\n")
            obs = {"generation": 0, "tick": 64, "being": 0, "retina": [513] * (14 * 126), "wave": [1] * 192, "fast_state": [0.1] * 96}
            (run / "observation.jsonl").write_text(json.dumps(obs) + "\n")
            (run / "observation.jsonl").write_text(json.dumps(obs) + "\n" + json.dumps(obs) + "\n")
            with self.assertRaises(ValueError):
                MOD.make_report(run)
            (run / "observation.jsonl").write_text(json.dumps(obs) + "\n")
            report = MOD.make_report(run)
            self.assertEqual(report["beings"]["0"][0]["alive_fraction"], 0.5)
            self.assertEqual(report["beings"]["0"][0]["contact_fraction_alive"], 0.25)
            self.assertEqual(len(report["observations"][0]["retina"]), 1764)
            MOD.write_report(report, out)
            self.assertTrue((out / "driving_data.js").exists())
            self.assertIn("window.DRIVING_DATA=", (out / "driving_data.js").read_text())
            self.assertEqual(json.loads((out / "report.json").read_text())["schema_version"], 1)

    def test_rejects_nonfinite_and_duplicate_temporal_records(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "run"
            run.mkdir()
            (run / "driving.meta.json").write_text(json.dumps({"schema_version": 1}))
            row = dict.fromkeys(CSV_FIELDS, "0")
            row.update(ticks="1", alive_ticks="1", speed_sum="nan")
            (run / "driving.csv").write_text(",".join(CSV_FIELDS) + "\n" + ",".join(row[x] for x in CSV_FIELDS) + "\n")
            with self.assertRaises(ValueError):
                MOD.make_report(run)


if __name__ == "__main__":
    unittest.main()
