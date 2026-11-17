import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_probe_data import load_monitor_labels, prepare_probe, read_monitor_labels, validate_probe  # noqa: E402
from fit_probe_smoke import synthetic_labels  # noqa: E402
from short_data import digest  # noqa: E402
from short_train import load_cache  # noqa: E402
from training_smoke import synthetic_manifest  # noqa: E402
import test_dsec_data  # noqa: E402


class FitProbeTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "protocol": "selfsup_fit_probe_v1",
            "train_record_indices": [0, 1],
            "updates": 4,
            "monitor_every": 2,
        }

    def test_selection_must_be_explicit_unique_training_records(self):
        validate_probe(self.config, 2)
        for indices in ([0, 0], [0, 2], [-1], [], [True]):
            with self.assertRaisesRegex(ValueError, "training record"):
                validate_probe({**self.config, "train_record_indices": indices}, 2)

    def test_probe_keeps_inputs_and_gt_separate_and_does_not_read_test_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source/data").mkdir(parents=True)
            source_path = synthetic_manifest(root / "source/data")
            source_hash = digest(source_path)
            with patch("fit_probe_data.read_monitor_labels", side_effect=synthetic_labels) as reader:
                path = prepare_probe(root / "source", self.config, root / "probe/data")
            self.assertEqual(reader.call_count, 2)
            source = json.loads(source_path.read_text())
            self.assertTrue(all(call.args[1] in source["train"] for call in reader.call_args_list))
            manifest = json.loads(path.read_text())
            self.assertEqual(digest(source_path), source_hash)
            self.assertEqual([r["sha256"] for r in manifest["train"]], [r["sha256"] for r in source["train"]])
            cache = load_cache(path, manifest)
            self.assertNotIn("flow", cache["train"][0])
            labels = load_monitor_labels(path, manifest)
            self.assertEqual(labels[0]["flow"][0, 0, 0], 4)
            monitor = path.parent / manifest["monitor_train"][0]["file"]
            monitor.unlink()
            self.assertEqual(len(load_cache(path, manifest)["train"]), 2)
            with self.assertRaises(FileNotFoundError):
                load_monitor_labels(path, manifest)

    def test_monitor_label_hash_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source/data").mkdir(parents=True)
            synthetic_manifest(root / "source/data")
            with patch("fit_probe_data.read_monitor_labels", side_effect=synthetic_labels):
                path = prepare_probe(root / "source", self.config, root / "probe/data")
            manifest = json.loads(path.read_text())
            label = path.parent / manifest["monitor_train"][0]["file"]
            label.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Monitor labels"):
                load_monitor_labels(path, manifest)

    def test_raw_monitor_gt_uses_original_timestamp_png_pair_and_crop_units(self):
        # Reuse the independent raw DSEC fixture, including valid zero motion.
        fixture = test_dsec_data.DsecTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        paths = fixture.paths()
        source = {"root": str(fixture.root), "config": {"crop_hw": [2, 2]}}
        record = {
            "sequence": paths.root.name,
            "index": 0,
            "start_us": fixture.offset + 1000,
            "end_us": fixture.offset + 2000,
            "timestamps_sha256": digest(paths.timestamps),
            "source_events_bytes": paths.events.stat().st_size,
        }
        with patch("dsec_data.DsecSequence.read_events", side_effect=AssertionError("Events unnecessarily reread")):
            labels, provenance = read_monitor_labels(source, record)
        self.assertTrue(labels["valid"][0, 0])
        np.testing.assert_equal(labels["flow"][0, 0], [0, 0])
        np.testing.assert_equal(labels["flow"][0, 1], [2.5, -0.5])
        self.assertTrue(provenance["source_png"].endswith("000002.png"))
        record["start_us"] += 1
        with self.assertRaisesRegex(ValueError, "interval"):
            read_monitor_labels(source, record)


if __name__ == "__main__":
    unittest.main()
