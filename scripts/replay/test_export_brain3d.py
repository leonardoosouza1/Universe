import importlib.util, tempfile, unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("export_brain3d", Path(__file__).with_name("export_brain3d.py"))
MOD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)

class ExportTests(unittest.TestCase):
    def test_binary_payload_is_little_endian_and_round_trips(self):
        import numpy as np
        values = np.array([1, 0xFFFFFFFF], dtype="<u4")
        self.assertEqual(np.frombuffer(__import__('base64').b64decode(MOD.b64(values)), dtype="<u4").tolist(), values.tolist())

    def test_annotation_fallback_uses_position_when_soma_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.tsv"; p.write_text("root_id\tsoma_x\tsoma_y\tsoma_z\tpos_x\tpos_y\tpos_z\n7\t\t\t\t1\t2\t3\n")
            self.assertEqual(MOD.load_annotations(p)[7]["z"], 3.0)

    def test_partial_soma_uses_complete_position_triplet(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.tsv"
            p.write_text("root_id\tsoma_x\tsoma_y\tsoma_z\tpos_x\tpos_y\tpos_z\n7\t999\t\t555\t1\t2\t3\n")
            got=MOD.load_annotations(p)[7]
            self.assertEqual([got[k] for k in ("x","y","z")],[1.,2.,3.])

if __name__ == "__main__": unittest.main()
