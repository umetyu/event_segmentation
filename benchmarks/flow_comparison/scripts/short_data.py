"""Freeze a small shared input cache and recording-disjoint split manifest."""

import hashlib
import json
from pathlib import Path

import numpy as np

from dsec_data import DsecSequence, discover_sequences, representations


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    names, groups = {}, {}
    if config["protocol"] != "selfsup_short_v1":
        raise ValueError("Unknown training protocol")
    if set(config["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Specify train, validation and test splits")
    for split, sequences in config["splits"].items():
        if not sequences:
            raise ValueError(f"Empty {split} split")
        for name in sequences:
            group = name.rsplit("_", 1)[0]
            if name in names or group in groups and groups[group] != split:
                raise ValueError(f"Sequence/recording leakage: {name}")
            names[name], groups[group] = split, split
    if len(config["crop_hw"]) != 2 or any(n < 32 or n % 16 for n in config["crop_hw"]):
        raise ValueError("Crop dimensions must be multiples of 16 and at least 32")
    for key in (
        "updates",
        "snn_steps",
        "loss_event_limit",
        "train_windows_per_sequence",
        "validation_windows_per_sequence",
    ):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["voxel_bins"] != 5 or config["learning_rate"] <= 0 or config["gradient_clip"] <= 0:
        raise ValueError("Invalid encoder/optimizer configuration")


def center_crop(sample, shape):
    h, w = shape
    sh, sw = sample["resolution"]
    if h > sh or w > sw:
        raise ValueError("Crop exceeds sensor resolution")
    y0, x0 = (sh - h) // 2, (sw - w) // 2
    keep = (sample["x"] >= x0) & (sample["x"] <= x0 + w - 1) & (sample["y"] >= y0) & (sample["y"] <= y0 + h - 1)
    cropped = dict(sample)
    for key in ("x", "y", "p", "t"):
        cropped[key] = sample[key][keep].copy()
    cropped["x"] -= x0
    cropped["y"] -= y0
    cropped["resolution"] = (h, w)
    if "flow" in sample:
        # Cropping changes coordinates, not flow displacement units.
        cropped["flow"] = sample["flow"][y0 : y0 + h, x0 : x0 + w].copy()
        cropped["valid"] = sample["valid"][y0 : y0 + h, x0 : x0 + w].copy()
    return cropped


def pack_sample(sample, config):
    sample = center_crop(sample, config["crop_hw"])
    count = len(sample["t"])
    if not count:
        raise ValueError("Selected crop has no events; choose an explicit different protocol")
    encoded = representations(sample, config["voxel_bins"], config["snn_steps"])
    # All events feed both networks. Only the expensive warping loss is subsampled,
    # once before either model sees the data, with identical deterministic indices.
    idx = np.linspace(0, count - 1, min(count, config["loss_event_limit"]), dtype=np.int64)
    tau = (sample["t"][idx] - sample["start_us"]).astype(np.float64) / (sample["end_us"] - sample["start_us"])
    encoded["loss_events"] = np.stack([sample["x"][idx], sample["y"][idx], tau, sample["p"][idx]], axis=1).astype(
        np.float32
    )
    if "flow" in sample:
        encoded.update(flow=sample["flow"], valid=sample["valid"])
    return encoded, count


def prepare(root, config, output):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sequences = discover_sequences(root)
    by_name = {}
    for seq in sequences:
        if seq.root.name in by_name:
            raise ValueError(f"Ambiguous duplicate sequence name: {seq.root.name}")
        by_name[seq.root.name] = seq
    required = {n for names in config["splits"].values() for n in names}
    missing = sorted(required - set(by_name))
    if missing:
        raise ValueError(f"Split is frozen; missing sequences: {missing}. No automatic reassignment.")
    manifest = {
        "protocol": config["protocol"],
        "root": str(Path(root).resolve()),
        "config": config,
        "train": [],
        "validation": [],
        "test_reserved": config["splits"]["test"],
        "test_labels_read": False,
        "training_gt_labels_read": False,
    }
    for split in ("train", "validation"):
        per_sequence = config[f"{split}_windows_per_sequence"]
        for name in config["splits"][split]:
            paths = by_name[name]
            with DsecSequence(paths) as seq:
                indices = np.linspace(0, len(seq) - 1, min(per_sequence, len(seq)), dtype=int)
                for index in indices:
                    sample = seq.read_sample(int(index), include_gt=split == "validation")
                    encoded, count = pack_sample(sample, config)
                    filename = f"{split}-{name}-{index}.npz"
                    np.savez(output / filename, **encoded)
                    manifest[split].append(
                        {
                            "file": filename,
                            "sha256": digest(output / filename),
                            "sequence": name,
                            "index": int(index),
                            "start_us": sample["start_us"],
                            "end_us": sample["end_us"],
                            "crop_events": count,
                            "loss_events": len(encoded["loss_events"]),
                            "timestamps_sha256": digest(paths.timestamps),
                            "source_events_bytes": paths.events.stat().st_size,
                        }
                    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output / "manifest.json"
