import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_short_run import (  # noqa: E402
    checkpoint_contract,
    load_validation,
    load_weights,
    prediction_diagnostics,
    time_inference,
    weighted_metrics,
)
from short_data import digest  # noqa: E402
from training_smoke import synthetic_manifest  # noqa: E402


class DiagnosticTests(unittest.TestCase):
    def test_zero_reference_uses_valid_mask_including_zero_motion(self):
        target = np.array([[[3, 4], [0, 0], [np.nan, np.nan]]], dtype=np.float32)
        valid = np.array([[True, True, False]])
        predicted = np.zeros_like(target)
        result = prediction_diagnostics(predicted, target, valid)
        self.assertEqual(result["metrics"]["epe_px"], 2.5)
        self.assertEqual(result["zero_metrics"]["valid_pixels"], 2)
        self.assertEqual(result["gt_valid_pixels"]["magnitude_mean_px"], 2.5)
        predicted[0, 2, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            prediction_diagnostics(predicted, target, valid)

    def test_pixel_weighting_is_not_a_window_average(self):
        rows = [
            {"metrics": {"valid_pixels": n, "epe_px": e, **{f"{k}pe_percent": 10 * e for k in (1, 2, 3)}}}
            for n, e in ((1, 1), (3, 3))
        ]
        result = weighted_metrics(rows, "metrics")
        self.assertEqual(result["epe_px"], 2.5)
        self.assertEqual(result["1pe_percent"], 25)

    def test_validation_loader_never_opens_training_or_test_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = synthetic_manifest(Path(tmp))
            manifest = json.loads(path.read_text())
            for record in manifest["train"]:
                (path.parent / record["file"]).unlink()
            self.assertEqual(len(load_validation(path, manifest)), 1)
            record = manifest["validation"][0]
            record["sequence"] = manifest["config"]["splits"]["test"][0]
            with self.assertRaisesRegex(ValueError, "validation split"):
                load_validation(path, manifest)
            record["sequence"] = manifest["config"]["splits"]["validation"][0]
            (path.parent / record["file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_validation(path, manifest)

    def test_checkpoint_contract_rejects_changed_manifest_and_wrong_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text("{}")
            model = torch.nn.Linear(2, 2)
            state = {
                "contract": checkpoint_contract(manifest),
                "model_kind": "snn",
                "model": model.state_dict(),
                "completed_updates": 20,
            }
            checkpoint = Path(tmp) / "checkpoint.pt"
            torch.save(state, checkpoint)
            before = digest(checkpoint)
            self.assertEqual(load_weights(model, checkpoint, manifest, "snn"), 20)
            with self.assertRaisesRegex(ValueError, "contract"):
                load_weights(model, checkpoint, manifest, "segmentation")
            manifest.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "contract"):
                load_weights(model, checkpoint, manifest, "snn")
            self.assertEqual(before, digest(checkpoint))

    def test_timing_runs_whole_snn_window_and_resets_between_calls(self):
        class FakeSnn:
            def __init__(self):
                self.steps = []
                self.state = 0

            def reset_states(self):
                if self.state:
                    self.steps.append(self.state)
                self.state = 0

            def __call__(self, voxel, counts):
                self.state += 1
                return {"flow": [torch.ones_like(counts) * self.state / 32]}

        model = FakeSnn()
        encoded = {"voxels": torch.zeros(5, 32, 32), "snn_counts": torch.zeros(5, 2, 32, 32)}
        result = time_inference(model, "snn", encoded, "cpu", warmup=2, repeats=3)
        self.assertEqual(model.steps, [5, 5, 5, 5, 5])
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(model.state, 0)
        self.assertGreater(result["median_ms"], 0)


if __name__ == "__main__":
    unittest.main()
