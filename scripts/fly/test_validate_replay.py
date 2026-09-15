import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("validate_replay", Path(__file__).with_name("validate_replay.py"))
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def square_frames(points, laps=1):
    track = points + [points[0]]
    out = []
    for lap in range(laps):
        for (x0, z0), (x1, z1) in zip(track, track[1:]):
            for step in range(5):
                fraction = step / 5
                out.append({"t": len(out) * 64,
                            "x": x0 + (x1 - x0) * fraction,
                            "z": z0 + (z1 - z0) * fraction})
    out.append({"t": len(out) * 64, "x": points[0][0], "z": points[0][1]})
    return out


class ReplayValidationTests(unittest.TestCase):
    def setUp(self):
        self.track = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]

    def check(self, frames, expected, **kwargs):
        result = MOD.validate({"track": self.track, "beings": [{"frames": frames}]}, min_fraction=1.0, **kwargs)
        self.assertEqual(result["valid"], expected, result)

    def test_complete_square(self):
        frames = square_frames(self.track[:-1], 1)
        result = MOD.validate({"track": self.track, "beings": [{"id": "b<&", "frames": frames}]}, min_fraction=1.0)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["beings"][0]["id"], "b<&")
        self.assertEqual(len(result["beings"][0]["lap_completion_ticks"]), 1)

    def test_annotation_is_escaped_and_keeps_failed_gate(self):
        html = "<html><head></head><body>ok</body></html>"
        html += "<script>window.REPLAY_TRACK=" + json.dumps(self.track) + ";</script>"
        html += "<script>window.REPLAY_BEINGS=" + json.dumps([{"id": "<b", "frames": [{"t": 0, "x": 2, "z": 2}, {"t": 64, "x": 2, "z": 2}]}]) + ";</script>"
        html += "<script>window.REPLAY_META={};</script>"
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "in.html", Path(directory) / "out.html"
            source.write_text(html, encoding="utf-8")
            status = MOD.main([str(source), "--min-fraction", "1", "--annotate-html", str(output)])
            self.assertEqual(status, 1)
            annotated = output.read_text(encoding="utf-8")
            self.assertIn("window.REPLAY_VALIDATION=", annotated)
            self.assertNotIn("<b", annotated.split("window.REPLAY_VALIDATION=", 1)[1].split(";</script>", 1)[0])
            self.assertIn("input_sha256", annotated)

    def test_stationary_fails(self):
        self.check([{"t": i * 64, "x": 2, "z": 2} for i in range(4)], False)

    def test_back_and_forth_across_seam_is_not_lap(self):
        frames = [{"t": i * 64, "x": x, "z": z} for i, (x, z) in enumerate([(1, 0), (0, 0), (1, 0), (0, 0), (1, 0)])]
        self.check(frames, False)

    def test_sector_missing_or_jump_fails(self):
        self.check([{"t": i * 64, "x": x, "z": z} for i, (x, z) in enumerate([(0, 0), (10, 0), (0, 10), (0, 0)])], False)
        self.check([{"t": 0, "x": 0, "z": 0}, {"t": 64, "x": 1000, "z": 1000}], False)

    def test_corridor_euclidean_and_tick_order_checks(self):
        meta = {"track_width": 4, "tick_rate": 64, "sample_every": 64}
        outside = [{"t": 0, "x": 0, "z": 20}, {"t": 64, "x": 1, "z": 20}]
        result = MOD.validate({"track": self.track, "beings": [{"frames": outside}], "meta": meta})
        self.assertEqual(result["beings"][0]["reason"], "out_of_corridor")
        sideways = [{"t": 0, "x": 0, "z": 0}, {"t": 64, "x": 0, "z": 1000}]
        result = MOD.validate({"track": self.track, "beings": [{"frames": sideways}]}, min_fraction=1)
        self.assertEqual(result["beings"][0]["reason"], "implausible_jump")
        unordered = [{"t": 64, "x": 0, "z": 0}, {"t": 0, "x": 1, "z": 0}]
        result = MOD.validate({"track": self.track, "beings": [{"frames": unordered}]}, min_fraction=1)
        self.assertEqual(result["beings"][0]["reason"], "non_increasing_ticks")

    def test_metadata_sampling_and_truncated_series(self):
        frames = square_frames(self.track[:-1], 1)
        frames = [{**frame, "t": i * 5} for i, frame in enumerate(frames)]
        meta = {"tick_rate": 50, "sample_every": 5, "track_width": 4}
        result = MOD.validate({"track": self.track, "beings": [{"frames": frames}], "meta": meta}, min_fraction=1)
        self.assertTrue(result["valid"], result)
        partial = frames[:5]
        result = MOD.validate({"track": self.track, "beings": [{"frames": frames}, {"frames": partial}]}, min_fraction=1)
        self.assertEqual(result["beings"][1]["reason"], "truncated_series")

    def test_corrupt_series_cannot_hide_behind_a_passing_majority(self):
        frames = square_frames(self.track[:-1], 1)
        beings = [{"frames": frames} for _ in range(9)] + [{"frames": frames[:4]}]
        result = MOD.validate({"track": self.track, "beings": beings}, min_fraction=0.8)
        self.assertFalse(result["valid"])
        self.assertFalse(result["integrity_valid"])

    def test_whole_recording_truncated_before_planned_horizon(self):
        frames = square_frames(self.track[:-1], 1)
        meta = {"tick_rate": 64, "sample_every": 64, "planned_ticks": 6400}
        result = MOD.validate({"track": self.track, "beings": [{"frames": frames}], "meta": meta})
        self.assertEqual(result["beings"][0]["reason"], "truncated_series")
        self.assertFalse(result["valid"])

    def test_invalid_width_and_fractional_sampling_fail(self):
        for meta in [{"track_width": -1}, {"sample_every": 0.5}]:
            with self.assertRaises(ValueError):
                MOD.validate({"track": self.track, "beings": [{"frames": square_frames(self.track[:-1])}], "meta": meta})

    def test_completed_lap_is_retained_after_reversing(self):
        frames = square_frames(self.track[:-1], 1)
        frames.append({"t": frames[-1]["t"] + 64, "x": 0, "z": 0.1})
        result = MOD.validate({"track": self.track, "beings": [{"frames": frames}]})
        self.assertEqual(result["beings"][0]["laps"], 1)
        self.assertEqual(len(result["beings"][0]["lap_completion_ticks"]), 1)

    def test_duplicate_ids_cannot_alias_another_beings_laps(self):
        frames = square_frames(self.track[:-1], 1)
        for ids in [(7, 7), (7, "7")]:
            with self.assertRaisesRegex(ValueError, "unique"):
                MOD.validate({"track": self.track, "beings": [{"id": value, "frames": frames} for value in ids]})


if __name__ == "__main__":
    unittest.main()
