"""Read original DSEC forward-flow data without modifying the dataset.

Format reference: https://dsec.ifi.uzh.ch/data-format/
Times remain integer microseconds; flow is pixel displacement, not pixels/s.
"""

from dataclasses import dataclass
from pathlib import Path
import os

import cv2
import h5py
import hdf5plugin  # noqa: F401 -- registers the compression filters used by DSEC
import numpy as np


@dataclass(frozen=True)
class SequencePaths:
    root: Path
    events: Path
    rectify: Path
    flow: Path
    timestamps: Path


def first_file(paths):
    return next((p for p in paths if p.is_file()), None)


def layout_summary(seq):
    """Show bounded file-layout evidence, including PNG and timestamp locations."""
    result = []
    for directory, dirs, files in os.walk(seq):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if len(Path(directory).relative_to(seq).parts) >= 2:
            dirs.clear()
        names = sorted(files)
        result.append(
            {
                "directory": str(Path(directory).relative_to(seq)),
                "file_count": len(names),
                "file_examples": names[:5],
                "timestamp_files": [n for n in names if "timestamp" in n.lower()][:8],
            }
        )
        if len(result) >= 12:
            break
    return result


def discover_sequences(root, diagnostics=None):
    """Support the official combined archive and common per-sequence layouts."""
    root = Path(root).resolve()
    found = []
    for directory, dirs, files in os.walk(root):
        # Avoid walking PNG directories or unrelated deeply nested experiment data.
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if len(Path(directory).relative_to(root).parts) >= 5:
            dirs.clear()
        if "events.h5" not in files:
            continue
        event_dir = Path(directory)
        if event_dir.name == "right" or event_dir.name.endswith("events_right"):
            continue
        if event_dir.name == "left" and event_dir.parent.name == "events":
            seq = event_dir.parent.parent
        elif event_dir.name == "events" or event_dir.name.endswith("events_left"):
            seq = event_dir.parent
        else:
            seq = event_dir
        rectify = first_file([event_dir / "rectify_map.h5", event_dir / "rectify_maps.h5"])
        flow_dirs = [
            seq / "flow/forward",
            seq / "optical_flow_forward",
            seq / "optical_flow_forward_event",
            seq / f"{seq.name}_optical_flow_forward_event",
            seq / "flow",
        ]
        flow_found = False
        matched = None
        for flow in flow_dirs:
            if not flow.is_dir() or next(flow.glob("*.png"), None) is None:
                continue
            flow_found = True
            timestamps = first_file(
                [
                    flow / "timestamps.txt",
                    seq / "flow/forward_timestamps.txt",
                    seq / "optical_flow_forward_timestamps.txt",
                    seq / f"{seq.name}_optical_flow_forward_timestamps.txt",
                    seq / "flow_timestamps.txt",
                    seq / "forward_timestamps.txt",
                ]
            )
            if timestamps is not None and rectify is not None:
                matched = SequencePaths(seq, event_dir / "events.h5", rectify, flow, timestamps)
                found.append(matched)
                break
        if diagnostics is not None and len(diagnostics) < 4:
            diagnostics.append(
                {
                    "sequence": str(seq),
                    "events": str(event_dir / "events.h5"),
                    "rectification_found": rectify is not None,
                    "flow_png_directory_found": flow_found,
                    "matched": matched is not None,
                    "layout": layout_summary(seq),
                }
            )
    return sorted(found, key=lambda p: str(p.root))


def load_flow_png(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.uint16 or bgr.ndim != 3 or bgr.shape[-1] != 3:
        raise ValueError(f"Expected a three-channel uint16 flow PNG: {path}")
    rgb = bgr[..., ::-1]  # DSEC encodes (u, v, valid) in RGB, OpenCV reads BGR.
    if not np.isin(rgb[..., 2], [0, 1]).all():
        raise ValueError(f"Invalid GT mask (expected 0/1): {path}")
    valid = rgb[..., 2] == 1
    flow = (rgb[..., :2].astype(np.float32) - 32768) / 128
    flow[~valid] = 0
    return flow, valid


class DsecSequence:
    def __init__(self, paths):
        self.paths = paths
        self.timestamps = np.loadtxt(paths.timestamps, delimiter=",", dtype=np.int64, ndmin=2)
        if self.timestamps.shape[1] != 2 or not len(self.timestamps):
            raise ValueError("Training forward timestamps must contain exactly two columns")
        if np.any(self.timestamps[:, 1] <= self.timestamps[:, 0]) or np.any(np.diff(self.timestamps[:, 0]) <= 0):
            raise ValueError("Forward flow intervals must be positive and ordered")
        # PNG indices need not be contiguous or start at zero. Rows correspond to sorted PNGs.
        self.flows = sorted(paths.flow.glob("*.png"), key=lambda p: int(p.stem))
        if len(self.flows) != len(self.timestamps):
            raise ValueError("Flow PNG count differs from timestamp row count; refusing to guess alignment")
        with h5py.File(paths.rectify, "r") as f:
            self.rectify = f["rectify_map"][()].astype(np.float32)
        if self.rectify.ndim != 3 or self.rectify.shape[-1] != 2:
            raise ValueError("Expected H x W x 2 rectification map")
        self.height, self.width = self.rectify.shape[:2]
        self.h5 = h5py.File(paths.events, "r")
        try:
            self.events = {key: self.h5[f"events/{key}"] for key in ("x", "y", "p", "t")}
            lengths = {a.shape for a in self.events.values()}
            if len(lengths) != 1 or self.events["t"].ndim != 1 or len(self.events["t"]) == 0:
                raise ValueError("Event arrays must be nonempty, one-dimensional, and equal in length")
            if self.events["t"].dtype.kind not in "iu":
                raise ValueError("Expected original DSEC integer-microsecond timestamps")
            self.offset = int(self.h5["t_offset"][()])
            self.ms_to_idx = np.asarray(self.h5["ms_to_idx"], dtype=np.int64)
            if (
                self.ms_to_idx.ndim != 1
                or not self.ms_to_idx.size
                or np.any(np.diff(self.ms_to_idx) < 0)
                or self.ms_to_idx.min() < 0
                or self.ms_to_idx.max() > len(self.events["t"])
            ):
                raise ValueError("Invalid millisecond-to-event index")
            self.last_time = int(self.events["t"][-1]) + self.offset
        except Exception:
            self.h5.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.h5.close()

    def __len__(self):
        return len(self.flows)

    def _lower_bound(self, timestamp):
        lo, hi = 0, len(self.events["t"])
        while lo < hi:
            mid = (lo + hi) // 2
            if int(self.events["t"][mid]) < timestamp:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def read_events(self, start, end):
        """Return [start, end), including duplicate timestamps at start only."""
        if start < self.offset or end > self.last_time + 1 or end <= start:
            raise ValueError(f"Event stream does not cover the complete GT interval: {start}, {end}")
        rel_start, rel_end = start - self.offset, end - self.offset
        ms_start, ms_end = rel_start // 1000, (rel_end + 999) // 1000
        if ms_end < len(self.ms_to_idx):
            lo, hi = int(self.ms_to_idx[ms_start]), int(self.ms_to_idx[ms_end])
        else:
            lo, hi = self._lower_bound(rel_start), self._lower_bound(rel_end)
        times = np.asarray(self.events["t"][lo:hi], dtype=np.int64)
        if np.any(np.diff(times) < 0):
            raise ValueError("Event timestamps are not sorted")
        first, last = np.searchsorted(times, [rel_start, rel_end], side="left")
        events = {k: np.asarray(v[lo + first : lo + last]) for k, v in self.events.items() if k != "t"}
        events["t"] = times[first:last] + self.offset
        return events

    def __getitem__(self, index):
        start, end = (int(t) for t in self.timestamps[index])
        events = self.read_events(start, end)
        x, y, p = (events[k] for k in ("x", "y", "p"))
        if (
            x.dtype.kind not in "iu"
            or y.dtype.kind not in "iu"
            or np.any((x < 0) | (x >= self.width) | (y < 0) | (y >= self.height))
            or not np.isin(p, [0, 1]).all()
        ):
            raise ValueError("Invalid raw event coordinates or polarities")
        xy = self.rectify[y, x]
        keep = (
            np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] <= self.width - 1)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] <= self.height - 1)
        )
        flow, valid = load_flow_png(self.flows[index])
        if flow.shape[:2] != (self.height, self.width):
            raise ValueError("GT and rectified event resolutions differ")
        return {
            "x": xy[keep, 0],
            "y": xy[keep, 1],
            "p": p[keep],
            "t": events["t"][keep],
            "start_us": start,
            "end_us": end,
            "raw_event_count": int(len(x)),
            "flow": flow,
            "valid": valid,
            "flow_file": str(self.flows[index]),
        }


def representations(sample, bins=5, steps=5):
    """Bilinear polarity counts and signed trilinear voxels from identical events.

    SNN count bins partition the complete GT interval with no preceding history.
    Channel 0 is positive, channel 1 negative. No normalization or GT masking.
    """
    if bins < 2 or steps < 1:
        raise ValueError("Need at least two voxel bins and one SNN step")
    h, w = sample["valid"].shape
    x, y, p = (sample[k] for k in ("x", "y", "p"))
    tau = (sample["t"] - sample["start_us"]).astype(np.float64) / (sample["end_us"] - sample["start_us"])
    if np.any((tau < 0) | (tau >= 1)):
        raise ValueError("Events must lie in the half-open GT time interval")
    voxels = np.zeros((bins, h, w), dtype=np.float32)
    snn_counts = np.zeros((steps, 2, h, w), dtype=np.float32)
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    time_bin = tau * (bins - 1)
    it = np.floor(time_bin).astype(np.int64)
    step = np.floor(tau * steps).astype(np.int64)
    channel = 1 - p.astype(np.int64)
    sign = 2 * p.astype(np.float32) - 1
    for dx in (0, 1):
        for dy in (0, 1):
            xx, yy = ix + dx, iy + dy
            weight = (1 - np.abs(x - xx)) * (1 - np.abs(y - yy))
            keep = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h) & (weight > 0)
            np.add.at(snn_counts, (step[keep], channel[keep], yy[keep], xx[keep]), weight[keep])
            for dt in (0, 1):
                tt = it + dt
                valid = keep & (tt < bins)
                values = weight[valid] * sign[valid] * (1 - np.abs(time_bin[valid] - tt[valid]))
                np.add.at(voxels, (tt[valid], yy[valid], xx[valid]), values)
    return {"voxels": voxels, "counts": snn_counts.sum(axis=0), "snn_counts": snn_counts}


def flow_metrics(predicted, target, valid):
    """Pixel-weighted EPE and strict >N-pixel error percentages, GT mask only."""
    if predicted.shape != target.shape or target.shape != (*valid.shape, 2):
        raise ValueError("Flow/mask shapes do not match")
    valid = valid.astype(bool)
    if not valid.any():
        raise ValueError("No valid ground-truth pixels")
    if not np.isfinite(predicted[valid]).all() or not np.isfinite(target[valid]).all():
        raise ValueError("Non-finite flow on a valid GT pixel")
    error = np.linalg.norm(predicted[valid].astype(np.float64) - target[valid], axis=-1)
    return {
        "valid_pixels": int(error.size),
        "epe_px": float(error.mean()),
        **{f"{n}pe_percent": float(100 * np.mean(error > n)) for n in (1, 2, 3)},
    }
