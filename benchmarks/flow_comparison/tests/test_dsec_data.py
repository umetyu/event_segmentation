"""Independent fixtures for DSEC's channel, time, mask and interpolation contracts."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dsec_data import DsecSequence, discover_sequences, flow_metrics, load_flow_png, representations  # noqa: E402


class DsecTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seq = self.root / "train/example"
        event_dir = self.seq / "events/left"
        flow_dir = self.seq / "flow/forward"
        event_dir.mkdir(parents=True)
        flow_dir.mkdir(parents=True)
        self.offset = 50_000_000_000
        self.times = np.array([0, 999, 1000, 1000, 1999, 2000, 2500, 3999, 4000, 6000], dtype=np.int64)
        with h5py.File(event_dir / "events.h5", "w") as f:
            f["events/t"] = self.times
            f["events/x"] = np.array([1, 1, 1, 1, 3, 1, 1, 1, 1, 1], dtype=np.uint16)
            f["events/y"] = np.ones(10, dtype=np.uint16)
            f["events/p"] = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
            f["ms_to_idx"] = np.searchsorted(self.times, np.arange(7) * 1000)
            f["t_offset"] = self.offset
        yy, xx = np.mgrid[:4, :4].astype(np.float32)
        rect = np.stack([xx, yy], axis=-1)
        rect[1, 1] = [1.25, 1.5]
        rect[1, 3] = [4.25, 1]
        with h5py.File(event_dir / "rectify_map.h5", "w") as f:
            f["rectify_map"] = rect
        self.rgb = np.zeros((4, 4, 3), dtype=np.uint16)
        self.rgb[1, 1] = [32768, 32768, 1]  # valid zero motion must remain valid
        self.rgb[1, 2] = [33088, 32704, 1]  # u=2.5, v=-0.5
        for name in ("000002.png", "000100.png", "000300.png"):
            self.assertTrue(cv2.imwrite(str(flow_dir / name), self.rgb[..., ::-1]))
        np.savetxt(
            self.seq / "flow/forward_timestamps.txt",
            np.array([[1000, 2000], [3000, 4000], [4000, 5000]]) + self.offset,
            fmt="%d",
            delimiter=",",
            header="from_timestamp_us, to_timestamp_us",
        )

    def paths(self):
        paths = discover_sequences(self.root)
        self.assertEqual(len(paths), 1)
        return paths[0]

    def test_png_channel_order_valid_zero_and_signed_displacement(self):
        flow, valid = load_flow_png(self.seq / "flow/forward/000002.png")
        self.assertEqual(int(valid.sum()), 2)
        self.assertTrue(valid[1, 1])
        np.testing.assert_equal(flow[1, 1], [0, 0])
        np.testing.assert_equal(flow[1, 2], [2.5, -0.5])
        self.assertFalse(valid[0, 0])

    def test_selfsupervised_sample_does_not_read_gt_png(self):
        with DsecSequence(self.paths()) as seq, patch("dsec_data.load_flow_png", side_effect=AssertionError("GT read")):
            sample = seq.read_sample(0, include_gt=False)
            self.assertNotIn("flow", sample)
            self.assertNotIn("valid", sample)
            self.assertEqual(representations(sample)["voxels"].shape, (5, 4, 4))

    def test_timestamp_offset_duplicate_boundaries_and_empty_window(self):
        with DsecSequence(self.paths()) as seq:
            events = seq.read_events(self.offset + 1000, self.offset + 2000)
            np.testing.assert_equal(events["t"] - self.offset, [1000, 1000, 1999])
            self.assertEqual(len(seq.read_events(self.offset + 2600, self.offset + 3000)["t"]), 0)
            # ceil(end/1000) is outside the coarse index: exact HDF5 search still works.
            np.testing.assert_equal(seq.read_events(self.offset + 6000, self.offset + 6001)["t"], [self.offset + 6000])
            with self.assertRaises(ValueError):
                seq.read_events(self.offset - 1, self.offset + 10)
            with self.assertRaises(ValueError):
                seq.read_events(self.offset + 6000, self.offset + 6002)

    def test_numeric_png_alignment_rectification_and_out_of_frame_filter(self):
        with DsecSequence(self.paths()) as seq:
            sample = seq[0]
            self.assertEqual(Path(sample["flow_file"]).name, "000002.png")
            self.assertEqual(Path(seq[1]["flow_file"]).name, "000100.png")
            self.assertEqual(sample["raw_event_count"], 3)
            np.testing.assert_equal(sample["x"], [1.25, 1.25])
            np.testing.assert_equal(sample["y"], [1.5, 1.5])
            # Opposite polarities cancel in voxels, but count channels retain both.
            encoded = representations(sample)
            self.assertAlmostEqual(float(encoded["counts"].sum()), 2)
            self.assertEqual(float(encoded["voxels"].sum()), 0)
            np.testing.assert_allclose(encoded["counts"][0, 1:3, 1:3], [[0.375, 0.125], [0.375, 0.125]])

    def test_empty_events_and_exact_gt_interval_binning(self):
        sample = {
            "x": np.array([1.0, 1.0]),
            "y": np.array([1.0, 1.0]),
            "p": np.array([1, 0]),
            "t": np.array([self.offset, self.offset + 999]),
            "start_us": self.offset,
            "end_us": self.offset + 1000,
            "valid": np.ones((4, 4), dtype=bool),
        }
        encoded = representations(sample, bins=5, steps=5)
        self.assertEqual(float(encoded["snn_counts"][0, 0, 1, 1]), 1)
        self.assertEqual(float(encoded["snn_counts"][4, 1, 1, 1]), 1)
        self.assertAlmostEqual(float(encoded["voxels"][4, 1, 1]), -0.996, places=5)
        for k in ("x", "y", "p", "t"):
            sample[k] = sample[k][:0]
        self.assertTrue(all(np.all(v == 0) for v in representations(sample).values()))

    def test_masked_metrics_strict_thresholds_and_nan_handling(self):
        target = np.zeros((1, 5, 2), dtype=np.float32)
        pred = np.array([[[0, 0], [1, 0], [2, 0], [3, 0], [np.nan, np.nan]]])
        valid = np.array([[True, True, True, True, False]])
        metrics = flow_metrics(pred, target, valid)
        self.assertEqual(
            metrics, {"valid_pixels": 4, "epe_px": 1.5, "1pe_percent": 50, "2pe_percent": 25, "3pe_percent": 0}
        )
        with self.assertRaises(ValueError):
            flow_metrics(pred, target, np.ones_like(valid))
        with self.assertRaises(ValueError):
            flow_metrics(pred, target, np.zeros_like(valid))

    def test_missing_gt_rows_are_not_silently_repaired(self):
        (self.seq / "flow/forward/000100.png").unlink()
        with self.assertRaisesRegex(ValueError, "count differs"):
            DsecSequence(self.paths())

    def test_converted_h5_is_not_mistaken_for_raw_dsec(self):
        with h5py.File(self.root / "converted.h5", "w") as f:
            f["events/ts"] = [1.0, 2.0]
            f["flow/flowmap000000000"] = np.zeros((4, 4, 2))
        from dsec_preflight import inventory

        item = next(x for x in inventory(self.root)["hdf5_files"] if x["path"] == "converted.h5")
        self.assertTrue(item["converted_events_schema"])
        self.assertEqual(len(discover_sequences(self.root)), 1)

    def test_flat_events_directory_keeps_sequence_root(self):
        # Regression for the reported Izar layout: <sequence>/events/events.h5.
        for path in (self.seq / "events/left").iterdir():
            path.rename(self.seq / "events" / path.name)
        (self.seq / "events/left").rmdir()
        paths = self.paths()
        self.assertEqual(paths.root, self.seq)
        self.assertEqual(paths.events, self.seq / "events/events.h5")
        with DsecSequence(paths) as sequence:
            self.assertEqual(sequence[0]["start_us"], self.offset + 1000)
            self.assertEqual(sequence[0]["valid"].sum(), 2)

    def test_flat_flow_pngs_and_sequence_timestamp_file(self):
        for path in (self.seq / "flow/forward").iterdir():
            path.rename(self.seq / "flow" / path.name)
        (self.seq / "flow/forward").rmdir()
        (self.seq / "flow/forward_timestamps.txt").rename(self.seq / "flow_timestamps.txt")
        with DsecSequence(self.paths()) as sequence:
            self.assertEqual(Path(sequence[1]["flow_file"]).name, "000100.png")
            np.testing.assert_equal(sequence[0]["flow"][1, 2], [2.5, -0.5])
        # A generic flow directory must still reject backward time intervals.
        timestamp_path = self.seq / "flow_timestamps.txt"
        timestamps = np.loadtxt(timestamp_path, delimiter=",", dtype=np.int64)
        np.savetxt(timestamp_path, timestamps[:, ::-1], delimiter=",", fmt="%d")
        with self.assertRaisesRegex(ValueError, "Forward flow intervals"):
            DsecSequence(self.paths())

    def test_unmatched_layout_reports_locations_without_selecting_backward_flow(self):
        (self.seq / "flow/forward").rename(self.seq / "flow/backward")
        (self.seq / "flow/forward_timestamps.txt").rename(self.seq / "flow/backward_timestamps.txt")
        diagnostics = []
        self.assertEqual(discover_sequences(self.root, diagnostics), [])
        self.assertEqual(len(diagnostics), 1)
        self.assertFalse(diagnostics[0]["matched"])
        layout = {entry["directory"]: entry for entry in diagnostics[0]["layout"]}
        self.assertEqual(layout["flow/backward"]["file_count"], 3)
        self.assertIn("backward_timestamps.txt", layout["flow"]["timestamp_files"])


if __name__ == "__main__":
    unittest.main()
