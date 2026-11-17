import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from flow_objective import timestamp_objective, to_pixel_displacement  # noqa: E402
from dsec_data import SequencePaths  # noqa: E402
from short_data import center_crop, pack_sample, prepare, validate_config  # noqa: E402
from short_train import load_cache, ordered_index  # noqa: E402


class ShortTrainingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "configs/short_train.json").read_text())

    def test_split_excludes_other_segments_from_same_recording(self):
        validate_config(self.config)
        bad = copy.deepcopy(self.config)
        bad["splits"]["validation"].append("zurich_city_02_b")
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_config(bad)

    def test_crop_preserves_displacement_and_removes_outside_events(self):
        sample = {
            "resolution": (8, 10),
            "x": np.array([0.0, 4.5, 9.0]),
            "y": np.array([0.0, 4.0, 7.0]),
            "t": np.array([100, 200, 300]),
            "p": np.array([1, 0, 1]),
            "flow": np.ones((8, 10, 2), dtype=np.float32) * 3,
            "valid": np.ones((8, 10), dtype=bool),
            "start_us": 0,
            "end_us": 400,
        }
        cropped = center_crop(sample, (4, 6))
        np.testing.assert_equal(cropped["x"], [2.5])
        np.testing.assert_equal(cropped["y"], [2.0])
        np.testing.assert_equal(cropped["flow"], np.full((4, 6, 2), 3))
        self.assertEqual(sample["x"][1], 4.5)

    def test_training_representation_needs_no_gt_or_mask(self):
        config = {**self.config, "crop_hw": [32, 32]}
        sample = {
            "resolution": (32, 32),
            "x": np.array([10.25]),
            "y": np.array([15.5]),
            "p": np.array([1]),
            "t": np.array([123456789123], dtype=np.int64),
            "start_us": 123456789000,
            "end_us": 123456890000,
        }
        encoded, count = pack_sample(sample, config)
        self.assertEqual(count, 1)
        self.assertNotIn("flow", encoded)
        self.assertNotIn("valid", encoded)
        self.assertAlmostEqual(float(encoded["counts"].sum()), 1)

    def test_shared_flow_units_do_not_depend_on_snn_steps(self):
        raw = torch.ones(1, 2, 32, 48) / 48
        torch.testing.assert_close(to_pixel_displacement(raw), torch.ones_like(raw))

    def test_known_translation_has_lower_timestamp_loss_than_zero(self):
        times = torch.linspace(0.05, 0.95, 19)
        events = torch.stack((8 + 4 * times, torch.ones_like(times) * 10, times, torch.ones_like(times)), dim=1)
        wrong = torch.zeros(1, 2, 32, 32)
        right = wrong.clone()
        right[:, 0] = 4 / 32
        loss_wrong, _ = timestamp_objective(wrong, events, 0, 0)
        loss_right, _ = timestamp_objective(right, events, 0, 0)
        self.assertLess(loss_right.item(), loss_wrong.item())
        parameter = torch.tensor(2 / 32, requires_grad=True)
        predicted = torch.stack((parameter.expand(32, 32), torch.zeros(32, 32)))[None]
        loss, _ = timestamp_objective(predicted, events, 0, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad).item())
        self.assertNotEqual(parameter.grad.item(), 0)

    def test_outside_warps_have_a_recovery_gradient(self):
        events = torch.tensor([[10.0, 10.0, 0.25, 1.0], [11.0, 10.0, 0.75, 1.0]])
        flow = torch.ones(1, 2, 32, 32, requires_grad=True) * 10
        flow.retain_grad()
        loss, components = timestamp_objective(flow, events)
        loss.backward()
        self.assertGreater(components["boundary"], 0)
        self.assertEqual(components["warped_in_frame_fraction"], 0)
        self.assertGreater(flow.grad.abs().sum().item(), 0)

    def test_resume_sample_order_is_step_addressable(self):
        full = [ordered_index(s, 5, 7) for s in range(13)]
        resumed = [ordered_index(s, 5, 7) for s in range(6)] + [ordered_index(s, 5, 7) for s in range(6, 13)]
        self.assertEqual(full, resumed)
        self.assertEqual(sorted(full[:5]), list(range(5)))

    def test_preparation_reads_labels_only_for_validation_and_reserves_test(self):
        config = {
            **self.config,
            "crop_hw": [32, 32],
            "validation_windows_per_sequence": 1,
            "splits": {"train": ["drive01_a"], "validation": ["drive02_a"], "test": ["drive03_a"]},
        }
        reads = []

        class FakeSequence:
            def __init__(self, paths):
                self.name = paths.root.name
                if self.name == "drive03_a":
                    raise AssertionError("Test data opened")

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def __len__(self):
                return 2

            def read_sample(self, index, include_gt):
                reads.append((self.name, include_gt))
                sample = {
                    "resolution": (32, 32),
                    "x": np.array([12.25]),
                    "y": np.array([15.0]),
                    "t": np.array([100001 + index * 100000]),
                    "p": np.array([1]),
                    "start_us": 100000 + index * 100000,
                    "end_us": 200000 + index * 100000,
                }
                if include_gt:
                    sample.update(flow=np.ones((32, 32, 2), dtype=np.float32), valid=np.ones((32, 32), dtype=bool))
                return sample

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for names in config["splits"].values():
                directory = root / names[0]
                directory.mkdir()
                (directory / "events.h5").write_bytes(b"fixture")
                (directory / "timestamps.txt").write_text("100000,200000\n200000,300000\n")
                paths.append(
                    SequencePaths(
                        directory,
                        directory / "events.h5",
                        directory / "rectify_map.h5",
                        directory / "flow",
                        directory / "timestamps.txt",
                    )
                )
            with patch("short_data.discover_sequences", return_value=paths), patch(
                "short_data.DsecSequence", FakeSequence
            ):
                manifest_path = prepare(root, config, root / "cache")
            manifest = json.loads(manifest_path.read_text())
            cache = load_cache(manifest_path, manifest)
            self.assertEqual(reads, [("drive01_a", False), ("drive01_a", False), ("drive02_a", True)])
            self.assertEqual(len(cache["train"]), 2)
            self.assertNotIn("flow", cache["train"][0])
            self.assertIn("flow", cache["validation"][0])
            self.assertEqual(manifest["test_reserved"], ["drive03_a"])
            self.assertFalse(manifest["test_labels_read"])


if __name__ == "__main__":
    unittest.main()
