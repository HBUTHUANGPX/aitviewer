import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from examples.local_smpl_viewer import ModelSpec, first_existing_path
from examples.load_local_smpl_npz import parse_smpl_npz


class FakeBodyModel:
    NUM_BODY_JOINTS = 23


class FakeSmplLayer:
    model_type = "smpl"
    num_betas = 10
    bm = FakeBodyModel()


class LocalSmplViewerScriptTest(unittest.TestCase):
    def test_first_existing_path_returns_first_available_candidate(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            missing = root / "missing"
            existing = root / "models"
            existing.mkdir()

            self.assertEqual(first_existing_path((missing, existing)), existing)

    def test_model_spec_exposes_existing_candidate_from_user_paths(self):
        spec = ModelSpec(
            model_type="smplh",
            gender="neutral",
            candidates=(
                Path("/home/hpx/HPX_Loco/loco-mujoco/datasets/smplh"),
                Path("/home/hpx/2025_5_24/loco-mujoco/datasets/smpl"),
            ),
        )

        self.assertIsNotNone(first_existing_path(spec.candidates))


class LocalSmplNpzLoaderTest(unittest.TestCase):
    def test_parse_generic_smpl_poses_npz(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.npz"
            np.savez(
                path,
                poses=np.zeros((5, 72), dtype=np.float32),
                transl=np.ones((5, 3), dtype=np.float32),
                betas=np.zeros(10, dtype=np.float32),
            )

            params = parse_smpl_npz(path, FakeSmplLayer())

            self.assertEqual(params["poses_root"].shape, (5, 3))
            self.assertEqual(params["poses_body"].shape, (5, 69))
            self.assertEqual(params["trans"].shape, (5, 3))

    def test_parse_aitviewer_export_npz(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "exported.npz"
            np.savez(
                path,
                poses_body=np.zeros((4, 69), dtype=np.float32),
                poses_root=np.zeros((4, 3), dtype=np.float32),
                trans=np.zeros((4, 3), dtype=np.float32),
                betas=np.zeros((1, 10), dtype=np.float32),
            )

            params = parse_smpl_npz(path, FakeSmplLayer(), start_frame=1, stride=2)

            self.assertEqual(params["poses_root"].shape, (2, 3))
            self.assertEqual(params["poses_body"].shape, (2, 69))

    def test_parse_split_smpl_keys_npz(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "split_keys.npz"
            np.savez(
                path,
                smpl_global_orient=np.zeros((6, 3), dtype=np.float32),
                smpl_body_pose=np.zeros((6, 69), dtype=np.float32),
                smpl_transl=np.ones((6, 3), dtype=np.float32),
                smpl_betas=np.zeros((6, 16), dtype=np.float32),
            )

            params = parse_smpl_npz(path, FakeSmplLayer(), start_frame=1, end_frame=5, stride=2)

            self.assertEqual(params["poses_root"].shape, (2, 3))
            self.assertEqual(params["poses_body"].shape, (2, 69))
            self.assertEqual(params["trans"].shape, (2, 3))
            self.assertEqual(params["betas"].shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
