"""Check DSEC timestamps, rectification, GT decoding, and optional model inputs.

Untrained networks are checked only for finite outputs. No model accuracy or
latency claim is made. The zero-flow baseline checks the metric implementation.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np

from dsec_data import DsecSequence, discover_sequences, flow_metrics, representations


def inventory(root):
    """Bounded schema evidence for converted datasets; never read whole arrays."""
    result = {"entries": sorted(p.name for p in root.iterdir())[:40], "hdf5_files": []}
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if len(Path(directory).relative_to(root).parts) >= 4:
            dirs.clear()
        for name in sorted(files):
            if not name.endswith((".h5", ".hdf5")):
                continue
            path = Path(directory) / name
            item = {"path": str(path.relative_to(root))}
            try:
                with h5py.File(path, "r") as f:
                    item["root_keys"] = list(f.keys())[:20]
                    item["events"] = {
                        k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                        for k, v in f.get("events", {}).items()
                        if isinstance(v, h5py.Dataset)
                    }
                    item["converted_events_schema"] = "events/ts" in f
                    # In the existing converter, invalid pixels become (0, 0)
                    # and the original validity mask is discarded.
                    if "flow" in f and isinstance(f["flow"], h5py.Group) and len(f["flow"]):
                        key = next(iter(f["flow"]))
                        obj = f["flow"][key]
                        if isinstance(obj, h5py.Dataset):
                            item["first_flow"] = {
                                "name": key,
                                "shape": list(obj.shape),
                                "attribute_names": list(obj.attrs),
                            }
            except OSError as exc:
                item["error"] = str(exc)
            result["hdf5_files"].append(item)
            if len(result["hdf5_files"]) >= 8:
                return result
    return result


def model_worker(args):
    import torch
    from model_preflight import build_model

    torch.set_num_threads(2)
    torch.manual_seed(0)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this allocation")
    model, kind, _, _ = build_model(args.model_worker + "-matched")
    model.to(args.device).eval()
    with np.load(args.model_input) as data:
        tensors = {k: torch.from_numpy(data[k]).to(args.device) for k in data.files}
    voxels = tensors["voxels"].unsqueeze(0)
    counts = tensors["counts"].unsqueeze(0)
    inputs = tensors["snn_counts"] if kind == "snn" else tensors["counts"].unsqueeze(0)
    model.reset_states()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    shapes = []
    with torch.inference_mode():
        for step_counts in inputs:
            result = model(voxels, step_counts.unsqueeze(0))
            flows = result["flow"]
            if isinstance(flows, torch.Tensor):
                flows = [flows]
            for flow in flows:
                if flow.shape != counts.shape or not torch.isfinite(flow).all().item():
                    raise ValueError(f"Invalid {kind} flow output: {tuple(flow.shape)}")
            shapes = [list(flow.shape) for flow in flows]
    report = {
        "model": kind,
        "status": "real_event_input_forward_passed",
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "steps": len(inputs),
        "state_reset_before_window": True,
        "output_shapes": shapes,
        "weights": "random_initialization_not_trained",
        "torch": torch.__version__,
    }
    if args.device == "cuda":
        torch.cuda.synchronize()
        report.update(gpu=torch.cuda.get_device_name(0), peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return report


def check(args, report):
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DSEC directory does not exist: {root}")
    report["root"] = str(root)
    report["inventory"] = inventory(root)
    sequences = discover_sequences(root)
    report["available_sequences"] = [str(p.root.relative_to(root)) or "." for p in sequences]
    if args.sequence:
        sequences = [
            p for p in sequences if p.root.name == args.sequence or str(p.root.relative_to(root)) == args.sequence
        ]
    if not sequences:
        raise ValueError(
            "No original DSEC forward-flow sequence found. Need events.h5, rectify_map.h5, "
            "16-bit forward-flow PNGs including their validity channel, and two-column timestamps. "
            "See inventory in this report. Converted HDF5 data needs a verified adapter and its original "
            "validity masks; zero motion cannot be used to infer invalid pixels."
        )
    if args.device != "none":
        from fetch_sources import ensure_sources

        ensure_sources(download=False)
    with DsecSequence(sequences[0]) as seq:
        report.update(sequence=str(seq.paths.root), total_gt_windows=len(seq), event_time_offset_us=seq.offset)
        if (seq.height, seq.width) != (480, 640):
            raise ValueError("Original DSEC must have 480x640 resolution; resized data requires an explicit adapter")
        indices = np.linspace(0, len(seq) - 1, min(args.samples, len(seq)), dtype=int)
        report["samples"] = []
        for index in indices:
            sample = seq[int(index)]
            encoded = representations(sample, bins=5, steps=args.snn_steps)
            if len(sample["t"]) == 0:
                raise ValueError(f"No in-frame rectified events for GT window {index}")
            np.testing.assert_allclose(encoded["counts"].sum(dtype=np.float64), len(sample["t"]), rtol=1e-5)
            if not all(np.isfinite(v).all() for v in encoded.values()):
                raise ValueError("Non-finite event representation")
            oracle = flow_metrics(sample["flow"], sample["flow"], sample["valid"])
            if oracle["epe_px"] != 0:
                raise ValueError("GT-versus-GT EPE must be zero")
            report["samples"].append(
                {
                    "index": int(index),
                    "flow_file": sample["flow_file"],
                    "start_us": sample["start_us"],
                    "end_us": sample["end_us"],
                    "duration_us": sample["end_us"] - sample["start_us"],
                    "raw_events": sample["raw_event_count"],
                    "rectified_in_frame_events": len(sample["t"]),
                    "dropped_rectified_events": sample["raw_event_count"] - len(sample["t"]),
                    "representation_shapes": {k: list(v.shape) for k, v in encoded.items()},
                    "gt_identity_epe_px": oracle["epe_px"],
                    "zero_flow_baseline": flow_metrics(np.zeros_like(sample["flow"]), sample["flow"], sample["valid"]),
                }
            )
            if index == indices[0] and args.device != "none":
                with tempfile.TemporaryDirectory(prefix="dsec-preflight-") as tmp:
                    input_path = Path(tmp) / "representations.npz"
                    np.savez(input_path, **encoded)
                    report["model_input_sample_index"] = int(index)
                    report["models"] = []
                    for model in ("segmentation", "snn"):
                        proc = subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--model-worker",
                                model,
                                "--model-input",
                                str(input_path),
                                "--device",
                                args.device,
                            ],
                            capture_output=True,
                            text=True,
                        )
                        if proc.returncode:
                            raise RuntimeError(f"{model} worker failed:\n{proc.stdout}\n{proc.stderr}")
                        report["models"].append(json.loads(proc.stdout))
    report["status"] = "dsec_preflight_passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--sequence", help="Optional sequence basename or path relative to root")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--snn-steps", type=int, default=5)
    parser.add_argument("--device", choices=("none", "cpu", "cuda"), default="none")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-worker", choices=("segmentation", "snn"), help=argparse.SUPPRESS)
    parser.add_argument("--model-input", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.model_worker:
        print(json.dumps(model_worker(args)))
        return
    if args.root is None or args.samples < 1 or args.snn_steps < 1:
        parser.error("--root is required; samples and snn-steps must be positive")
    report = {
        "status": "failed",
        "purpose": "data_and_input_validation_only",
        "flow_direction": "forward",
        "flow_units": "pixels_over_exact_gt_interval",
        "event_window": "[from_timestamp_us, to_timestamp_us)",
        "gt_used_for_training": False,
        "training_or_model_accuracy_evaluation_performed": False,
    }
    try:
        check(args, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if report["status"] != "dsec_preflight_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
