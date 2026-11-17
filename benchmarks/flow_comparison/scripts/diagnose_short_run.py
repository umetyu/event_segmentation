"""Evaluate existing cropped checkpoints against zero flow; time model inference only."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from dsec_data import flow_metrics
from fetch_sources import ROOT, ensure_sources
from flow_objective import final_flow, timestamp_objective, to_pixel_displacement
from model_preflight import build_model
from short_data import digest, validate_config
from short_train import atomic_json


def checkpoint_contract(manifest_path):
    # Keep the exact short_train.py contract without editing its hashed source:
    # checkpoints already produced on Izar must remain usable.
    paths = [ROOT / "scripts" / name for name in ("short_train.py", "flow_objective.py", "model_preflight.py")]
    paths.append(ROOT / "sources.lock.json")
    return hashlib.sha256((digest(manifest_path) + "".join(digest(p) for p in paths)).encode()).hexdigest()


def load_validation(manifest_path, manifest):
    validate_config(manifest["config"])
    samples = []
    for record in manifest["validation"]:
        if record["sequence"] not in manifest["config"]["splits"]["validation"]:
            raise ValueError("Sample is outside the frozen validation split")
        path = (manifest_path.parent / record["file"]).resolve()
        if path.parent != manifest_path.parent.resolve() or digest(path) != record["sha256"]:
            raise ValueError("Validation cache path/hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            samples.append({k: torch.from_numpy(data[k].copy()) for k in data.files})
    if not samples:
        raise ValueError("Empty validation cache")
    return samples


def load_weights(model, checkpoint, manifest_path, kind):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["contract"] != checkpoint_contract(manifest_path) or state["model_kind"] != kind:
        raise ValueError("Checkpoint model/data/config/code contract does not match")
    model.load_state_dict(state["model"], strict=True)
    return state["completed_updates"]


def vector_statistics(vectors):
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 2)
    if not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Non-finite or empty flow vectors")
    magnitude = np.linalg.norm(vectors, axis=-1)
    return {
        "mean_xy_px": vectors.mean(axis=0).tolist(),
        "std_xy_px": vectors.std(axis=0).tolist(),
        "magnitude_mean_px": float(magnitude.mean()),
        "magnitude_median_px": float(np.median(magnitude)),
        "magnitude_p95_px": float(np.percentile(magnitude, 95)),
        "magnitude_max_px": float(magnitude.max()),
    }


def prediction_diagnostics(pixels, target, valid):
    return {
        "metrics": flow_metrics(pixels, target, valid),
        "zero_metrics": flow_metrics(np.zeros_like(pixels), target, valid),
        "prediction_all_pixels": vector_statistics(pixels),
        "prediction_valid_pixels": vector_statistics(pixels[valid]),
        "gt_valid_pixels": vector_statistics(target[valid]),
    }


def weighted_metrics(samples, key):
    count = sum(s[key]["valid_pixels"] for s in samples)
    return {
        "valid_pixels": count,
        **{
            k: sum(s[key][k] * s[key]["valid_pixels"] for s in samples) / count
            for k in ("epe_px", "1pe_percent", "2pe_percent", "3pe_percent")
        },
    }


def latency_statistics(milliseconds):
    return {
        "iterations": len(milliseconds),
        "median_ms": float(np.median(milliseconds)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "mean_ms": float(np.mean(milliseconds)),
    }


def time_inference(model, kind, encoded, device, warmup, repeats):
    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(warmup):
            output = to_pixel_displacement(final_flow(model, kind, encoded))
            del output
        model.reset_states()
        synchronize()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
        milliseconds = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            output = to_pixel_displacement(final_flow(model, kind, encoded))
            synchronize()
            milliseconds.append((time.perf_counter() - start) * 1000)
            del output
        result = {**latency_statistics(milliseconds), "samples_ms": milliseconds}
        if device == "cuda":
            result.update(
                resident_allocated_bytes=baseline,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated() - baseline,
            )
        model.reset_states()
    return result


def worker(args):
    sources = ensure_sources(download=False)
    manifest_path = (args.run / "data/manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    cache = load_validation(manifest_path, manifest)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, kind, _, _ = build_model(args.worker + "-matched")
    checkpoint = args.run / kind / "last.pt"
    updates = load_weights(model, checkpoint, manifest_path, kind)
    model.to(args.device).eval()
    rows = []
    for record, sample in zip(manifest["validation"], cache):
        # Neither GT nor the loss-event subset is an inference input.
        encoded = {k: sample[k].to(args.device) for k in ("voxels", "counts", "snn_counts")}
        with torch.inference_mode():
            flow = final_flow(model, kind, encoded)
            loss, components = timestamp_objective(
                flow, sample["loss_events"].to(args.device), config["smoothness_weight"], config["boundary_weight"]
            )
            zero_loss, zero_components = timestamp_objective(
                torch.zeros_like(flow),
                sample["loss_events"].to(args.device),
                config["smoothness_weight"],
                config["boundary_weight"],
            )
            if not torch.isfinite(loss).item() or not torch.isfinite(zero_loss).item():
                raise ValueError("Non-finite diagnostic loss")
            pixels = to_pixel_displacement(flow)[0].permute(1, 2, 0).cpu().numpy()
            row = {
                "sequence": record["sequence"],
                "index": record.get("index"),
                "selfsupervised_loss": loss.item(),
                "loss_components": components,
                "zero_selfsupervised_loss": zero_loss.item(),
                "zero_loss_components": zero_components,
                **prediction_diagnostics(pixels, sample["flow"].numpy(), sample["valid"].numpy()),
            }
            del flow, loss, zero_loss
            model.reset_states()
        # No loss, GT metric, disk IO, transfer, or encoding is timed.
        row["model_latency"] = time_inference(model, kind, encoded, args.device, args.warmup, args.repeats)
        rows.append(row)
        del encoded
    metrics, zero = weighted_metrics(rows, "metrics"), weighted_metrics(rows, "zero_metrics")
    report = {
        "model": kind,
        "completed_updates": updates,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": digest(manifest_path),
        "source_commits": {k: v["commit"] for k, v in sources.items()},
        "metrics": metrics,
        "zero_metrics": zero,
        "epe_minus_zero_px": metrics["epe_px"] - zero["epe_px"],
        "beats_zero_epe": metrics["epe_px"] < zero["epe_px"],
        "mean_selfsupervised_loss": float(np.mean([r["selfsupervised_loss"] for r in rows])),
        "mean_zero_selfsupervised_loss": float(np.mean([r["zero_selfsupervised_loss"] for r in rows])),
        "mean_warped_in_frame_fraction": float(
            np.mean([r["loss_components"]["warped_in_frame_fraction"] for r in rows])
        ),
        "model_latency": latency_statistics([v for r in rows for v in r["model_latency"]["samples_ms"]]),
        "samples": rows,
        "device": args.device,
        "torch": str(torch.__version__),
        "input_resolution_hw": config["crop_hw"],
        "steps_per_window": config["snn_steps"] if kind == "snn" else 1,
        "test_evaluated": False,
        "timing": {
            "scope": "reset_full_window_forward_finite_check_and_pixel_conversion_only",
            "excludes": ["disk_io", "event_encoding", "host_device_transfer", "loss", "gt_metrics", "model_loading"],
            "clock": "perf_counter_with_cuda_synchronize_before_and_after_if_cuda",
            "warmup_per_sample": args.warmup,
            "repeats_per_sample": args.repeats,
            "batch_size": 1,
            "dtype": "float32",
            "tf32": False,
            "cudnn_benchmark": False,
            "cpu_threads": 2,
        },
    }
    if args.device == "cuda":
        report.update(gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda)
    atomic_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing short-train-JOB_ID directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory (worker: JSON file)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--worker", choices=("segmentation", "snn"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("Warmup and repeat counts must be positive")
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "failed",
        "models": [],
        "test_evaluated": False,
        "scope": "cropped_short_run_diagnostics_not_final_accuracy_or_end_to_end_speed",
    }
    try:
        for kind in ("segmentation", "snn"):
            path = args.output / f"{kind}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--run",
                    str(args.run.resolve()),
                    "--output",
                    str(path.resolve()),
                    "--worker",
                    kind,
                    "--device",
                    args.device,
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                ],
                check=True,
            )
            result = json.loads(path.read_text())
            report["models"].append({k: v for k, v in result.items() if k != "samples"})
        a, b = report["models"]
        if a["manifest_sha256"] != b["manifest_sha256"] or a["zero_metrics"] != b["zero_metrics"]:
            raise ValueError("Models were not evaluated on the same validation data")
        report["status"] = "short_run_diagnostics_completed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic_json(args.output / "summary.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
