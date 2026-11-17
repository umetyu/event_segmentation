"""Prepare two fixed training inputs and isolated GT sidecars for monitoring only."""

import copy
import json
from pathlib import Path
import shutil

import numpy as np

from dsec_data import DsecSequence, discover_sequences, load_flow_png
from short_data import digest, validate_config
from short_train import atomic_json, load_cache


def validate_probe(config, training_count):
    if config["protocol"] != "selfsup_fit_probe_v1":
        raise ValueError("Unknown fit probe protocol")
    indices = config["train_record_indices"]
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(type(i) is not int or not 0 <= i < training_count for i in indices)
    ):
        raise ValueError("Select unique existing training record indices")
    if any(type(config[k]) is not int or config[k] < 1 for k in ("updates", "monitor_every")):
        raise ValueError("Positive integer update and monitoring budgets required")


def read_monitor_labels(source, record):
    matches = [p for p in discover_sequences(source["root"]) if p.root.name == record["sequence"]]
    if len(matches) != 1:
        raise ValueError("Cannot uniquely locate selected training sequence")
    paths = matches[0]
    if (
        digest(paths.timestamps) != record["timestamps_sha256"]
        or paths.events.stat().st_size != record["source_events_bytes"]
    ):
        raise ValueError("Source data metadata differs from the frozen input cache")
    with DsecSequence(paths) as sequence:
        index = record["index"]
        if tuple(sequence.timestamps[index]) != (record["start_us"], record["end_us"]):
            raise ValueError("GT interval differs from cached input")
        png = sequence.flows[index]
        flow, valid = load_flow_png(png)
        if flow.shape[:2] != (sequence.height, sequence.width):
            raise ValueError("GT resolution differs from the event camera")
    h, w = source["config"]["crop_hw"]
    if h > flow.shape[0] or w > flow.shape[1]:
        raise ValueError("Crop exceeds GT dimensions")
    y, x = (flow.shape[0] - h) // 2, (flow.shape[1] - w) // 2
    return {"flow": flow[y : y + h, x : x + w].copy(), "valid": valid[y : y + h, x : x + w].copy()}, {
        "source_png": str(png),
        "source_png_sha256": digest(png),
    }


def prepare_probe(source_run, config, output):
    source_path = (Path(source_run) / "data/manifest.json").resolve()
    source = json.loads(source_path.read_text())
    validate_config(source["config"])
    validate_probe(config, len(source["train"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol": source["config"]["protocol"],
        "experiment": "small_sample_selfsupervised_fit_probe",
        "config": copy.deepcopy(source["config"]),
        "probe": config,
        "source_manifest": str(source_path),
        "source_manifest_sha256": digest(source_path),
        "train": [],
        "validation": [],
        "monitor_train": [],
        "test_reserved": source["config"]["splits"]["test"],
        "training_gt_labels_used_for_gradients": False,
        "training_gt_labels_read_for_monitoring": True,
        "test_labels_read": False,
    }
    manifest["config"]["updates"] = config["updates"]
    for split in ("train", "validation"):
        selected = [source[split][i] for i in config["train_record_indices"]] if split == "train" else source[split]
        for index, record in enumerate(selected):
            if record["sequence"] not in source["config"]["splits"][split]:
                raise ValueError("Selected record violates the frozen recording split")
            path = (source_path.parent / record["file"]).resolve()
            if path.parent != source_path.parent or digest(path) != record["sha256"]:
                raise ValueError("Source cache path/hash mismatch")
            filename = f"{split}-{index}.npz"
            shutil.copyfile(path, output / filename)
            manifest[split].append({**record, "file": filename})
            if split == "train":
                labels, provenance = read_monitor_labels(source, record)
                label_file = f"monitor-train-{index}.npz"
                np.savez(output / label_file, **labels)
                manifest["monitor_train"].append(
                    {
                        "input_sha256": record["sha256"],
                        "file": label_file,
                        "sha256": digest(output / label_file),
                        **provenance,
                    }
                )
    manifest_path = output / "manifest.json"
    atomic_json(manifest_path, manifest)
    # Reuse the trainer's checks, including rejecting any training labels in input caches.
    load_cache(manifest_path, manifest)
    return manifest_path


def load_monitor_labels(manifest_path, manifest):
    if len(manifest["train"]) != len(manifest["monitor_train"]):
        raise ValueError("Missing training monitor labels")
    result = []
    h, w = manifest["config"]["crop_hw"]
    for record, monitor in zip(manifest["train"], manifest["monitor_train"]):
        path = (manifest_path.parent / monitor["file"]).resolve()
        if (
            path.parent != manifest_path.parent.resolve()
            or digest(path) != monitor["sha256"]
            or monitor["input_sha256"] != record["sha256"]
        ):
            raise ValueError("Monitor labels do not match the frozen training input")
        with np.load(path, allow_pickle=False) as data:
            flow, valid = data["flow"].copy(), data["valid"].copy()
        if flow.shape != (h, w, 2) or valid.shape != (h, w) or valid.dtype != np.bool_:
            raise ValueError("Invalid monitor label shape or mask")
        if not valid.any() or not np.isfinite(flow[valid]).all():
            raise ValueError("Invalid or empty monitoring ground truth")
        result.append({"flow": flow, "valid": valid})
    return result
