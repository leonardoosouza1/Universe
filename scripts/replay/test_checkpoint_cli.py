"""Opt-in GPU/CLI regression: a corrupt checkpoint must never start a random run.

Run after building gpu_evolve, with exclusive GPU access:
python3 -m unittest scripts/replay/test_checkpoint_cli.py -v
"""

from pathlib import Path
import subprocess
import tempfile
import unittest


class CheckpointContractTests(unittest.TestCase):
    def test_corrupt_resume_aborts_without_creating_a_replay(self):
        root = Path(__file__).resolve().parents[2]
        binary = root / "target/release/gpu_evolve"
        self.assertTrue(binary.is_file(), "build gpu_evolve before running this GPU test")
        with tempfile.TemporaryDirectory(prefix="universe-checkpoint-test-") as directory:
            output = Path(directory)
            (output / "evolve_best_gen0000.bin").write_bytes(b"invalid checkpoint")
            result = subprocess.run([
                str(binary), "--gens", "1", "--ticks", "8", "--beings", "1",
                "--resume", "--out", str(output),
            ], cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("refusing to substitute a random population", result.stderr)
            self.assertFalse((output / "gen0000.replay").exists())


if __name__ == "__main__":
    unittest.main()
