import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_probe_data import prepare_probe  # noqa: E402
from fit_probe_smoke import synthetic_labels  # noqa: E402
from gt_control import contract_for, prepare_control, supervised_epe  # noqa: E402
from short_data import digest  # noqa: E402
from short_train import load_cache  # noqa: E402
from training_smoke import synthetic_manifest  # noqa: E402


class SupervisedControlTests(unittest.TestCase):
    def test_masked_loss_preserves_zero_motion_and_ignores_invalid_nan(self):
        raw = torch.zeros(1, 2, 2, 2, requires_grad=True)
        target = torch.tensor([[[3.0, 4.0], [0.0, 0.0]], [[float("nan"), float("nan")], [100.0, 100.0]]])
        valid = torch.tensor([[True, True], [False, False]])
        loss = supervised_epe(raw, target, valid)
        self.assertAlmostEqual(loss.item(), (25.000001**0.5 + 0.001) / 2, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertLess(raw.grad[0, 0, 0, 0], 0)
        self.assertTrue((raw.grad[0, :, 1, :] == 0).all())
        self.assertTrue((raw.grad[0, :, 0, 1] == 0).all())

    def test_pixel_scaling_and_gradient_direction(self):
        for desired in (-3.0, 3.0):
            raw = torch.zeros(1, 2, 4, 8, requires_grad=True)
            target = torch.zeros(4, 8, 2)
            target[..., 0] = desired
            loss = supervised_epe(raw, target, torch.ones(4, 8, dtype=torch.bool))
            loss.backward()
            self.assertLess(raw.grad[0, 0].mean().item() * desired, 0)
            correct = raw.detach().clone()
            correct[:, 0] = desired / 8
            self.assertAlmostEqual(supervised_epe(correct, target, torch.ones(4, 8, dtype=torch.bool)).item(), 0.001)

    def test_empty_or_nonfinite_valid_gt_is_rejected(self):
        flow = torch.zeros(1, 2, 2, 2)
        target = torch.zeros(2, 2, 2)
        with self.assertRaisesRegex(ValueError, "No valid"):
            supervised_epe(flow, target, torch.zeros(2, 2, dtype=torch.bool))
        target[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            supervised_epe(flow, target, torch.ones(2, 2, dtype=torch.bool))

    def test_supervised_copy_preserves_source_and_inputs_with_a_distinct_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source/data").mkdir(parents=True)
            synthetic_manifest(root / "source/data")
            probe = {
                "protocol": "selfsup_fit_probe_v1",
                "train_record_indices": [0, 1],
                "updates": 4,
                "monitor_every": 2,
            }
            with patch("fit_probe_data.read_monitor_labels", side_effect=synthetic_labels):
                original = prepare_probe(root / "source", probe, root / "fit/data")
            checksum = digest(original)
            control = {
                "protocol": "supervised_fit_control_v1",
                "updates": 4,
                "monitor_every": 2,
                "epe_epsilon_px": 0.001,
            }
            path = prepare_control(root / "fit", control, root / "gt/data")
            saved = json.loads(path.read_text())
            self.assertEqual(digest(original), checksum)
            self.assertTrue(saved["training_gt_labels_used_for_gradients"])
            self.assertFalse(json.loads(original.read_text())["training_gt_labels_used_for_gradients"])
            self.assertEqual(saved["train"], json.loads(original.read_text())["train"])
            self.assertNotIn("flow", load_cache(path, saved)["train"][0])
            self.assertTrue(contract_for(path).startswith("supervised:"))
            with self.assertRaises(FileExistsError):
                prepare_control(root / "fit", control, root / "gt/data")


if __name__ == "__main__":
    unittest.main()
