"""Supervised diagnostic on the same frozen inputs; distinct from self-supervised experiments."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from diagnose_short_run import weighted_metrics
from dsec_data import flow_metrics
from fetch_sources import ROOT, ensure_sources
from fit_probe_data import load_monitor_labels
from flow_objective import final_flow, to_pixel_displacement
from model_preflight import build_model
from short_data import digest, validate_config
from short_train import atomic_json, evaluate, load_cache, on_device, ordered_index, weight_digest


def validate_control(config):
    if config["protocol"] != "supervised_fit_control_v1":
        raise ValueError("Unknown supervised control protocol")
    if any(type(config[k]) is not int or config[k] < 1 for k in ("updates", "monitor_every")):
        raise ValueError("Positive integer update and monitoring budgets required")
    if not 0 < config["epe_epsilon_px"] <= 0.01:
        raise ValueError("Invalid smoothed-EPE epsilon")


def prepare_control(source_run, config, output):
    validate_control(config)
    source_path = (source_run / "data/manifest.json").resolve()
    source = json.loads(source_path.read_text())
    if source.get("experiment") != "small_sample_selfsupervised_fit_probe":
        raise ValueError("Supply a self-supervised fit-probe run")
    load_cache(source_path, source)
    load_monitor_labels(source_path, source)
    output.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(source)
    manifest.update(
        experiment="gt_supervised_fit_control",
        control=config,
        source_fit_manifest=str(source_path),
        source_fit_manifest_sha256=digest(source_path),
        training_gt_labels_used_for_gradients=True,
    )
    manifest["source_probe"] = manifest.pop("probe")
    manifest["config"]["updates"] = config["updates"]
    for group in ("train", "validation", "monitor_train"):
        for record in manifest[group]:
            path = (source_path.parent / record["file"]).resolve()
            if path.parent != source_path.parent or digest(path) != record["sha256"]:
                raise ValueError("Source cache path/hash mismatch")
            shutil.copyfile(path, output / record["file"])
    atomic_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def contract_for(manifest_path):
    paths = [Path(__file__), ROOT / "sources.lock.json"] + [
        ROOT / "scripts" / name
        for name in (
            "short_train.py",
            "flow_objective.py",
            "model_preflight.py",
            "fit_probe_data.py",
            "dsec_data.py",
        )
    ]
    return (
        "supervised:" + hashlib.sha256((digest(manifest_path) + "".join(digest(p) for p in paths)).encode()).hexdigest()
    )


def supervised_epe(flow, target, valid, epsilon=0.001):
    """Mean smoothed EPE in pixels, training GT mask only; valid zero motion is retained."""
    if flow.ndim != 4 or flow.shape[:2] != (1, 2):
        raise ValueError("Expected one dense two-component flow map")
    pixels = to_pixel_displacement(flow)[0].permute(1, 2, 0)
    if pixels.shape != target.shape or valid.shape != target.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("GT shape or mask mismatch")
    if not valid.any().item():
        raise ValueError("No valid training GT pixels")
    # Select BEFORE subtraction so NaNs in invalid GT cannot poison gradients.
    error = pixels[valid] - target[valid]
    if not torch.isfinite(error).all().item():
        raise ValueError("Non-finite valid prediction/GT")
    return (error.square().sum(dim=-1) + epsilon**2).sqrt().mean()


def score(model, kind, cache, labels, config, device, step):
    train_labeled = [{**sample, **label} for sample, label in zip(cache["train"], labels)]
    result = {"step": step}
    for split, samples in (("train", train_labeled), ("validation", cache["validation"])):
        evaluated = evaluate(model, kind, {"train": cache["train"], "validation": samples}, config, device)
        zero = weighted_metrics(
            [
                {"metrics": flow_metrics(np.zeros_like(s["flow"].numpy()), s["flow"].numpy(), s["valid"].numpy())}
                for s in samples
            ],
            "metrics",
        )
        result[split] = {
            "metrics": evaluated["pixel_weighted_metrics"],
            "zero_metrics": zero,
            "selfsupervised_loss": evaluated["selfsupervised_loss"],
            "samples": evaluated["samples"],
        }
    return result


def train_worker(args):
    sources = ensure_sources(download=False)
    path = args.output / "data/manifest.json"
    manifest = json.loads(path.read_text())
    config, control = manifest["config"], manifest["control"]
    validate_config(config)
    validate_control(control)
    if manifest["experiment"] != "gt_supervised_fit_control" or not manifest["training_gt_labels_used_for_gradients"]:
        raise ValueError("Supervised worker requires a distinct supervised manifest")
    until = args.until if args.until is not None else control["updates"]
    if not 0 < until <= control["updates"]:
        raise ValueError("Target outside frozen supervised update budget")
    cache = load_cache(path, manifest)
    labels = [{k: torch.from_numpy(v) for k, v in label.items()} for label in load_monitor_labels(path, manifest)]
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, kind, _, _ = build_model(args.worker + "-matched")
    model.to(args.device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    directory = args.output / kind
    directory.mkdir(exist_ok=True)
    checkpoint = directory / "last.pt"
    contract = contract_for(path)
    start, history, curves = 0, [], []
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["contract"] != contract or state["model_kind"] != kind or state["device_type"] != args.device:
            raise ValueError("Supervised checkpoint/data/config/code/device mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start, history, curves = state["completed_updates"], state["history"], state["curves"]
        initial_digest = state["initial_weight_digest"]
        del state
        if start > until:
            raise ValueError("Cannot rewind supervised checkpoint")
    else:
        if checkpoint.exists() or (directory / "summary.json").exists():
            raise FileExistsError("Existing supervised run; use --resume")
        initial_digest = weight_digest(model)
        curves.append(score(model, kind, cache, labels, config, args.device, 0))
    for step in range(start, until):
        index = ordered_index(step, len(cache["train"]), config["seed"])
        inputs = on_device(cache["train"][index], args.device)
        target = on_device(labels[index], args.device)
        optimizer.zero_grad(set_to_none=True)
        flow = final_flow(model, kind, inputs)
        loss = supervised_epe(flow, target["flow"], target["valid"], control["epe_epsilon_px"])
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise ValueError(f"Disconnected supervised gradients: {missing}")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "sample_index": index,
                "supervised_loss_px": loss.item(),
                "gradient_norm_before_clip": norm.item(),
            }
        )
        model.reset_states()
        del inputs, target, flow, loss
        if (step + 1) % control["monitor_every"] == 0 or step + 1 == until:
            row = score(model, kind, cache, labels, config, args.device, step + 1)
            curves.append(row)
            state = {
                "contract": contract,
                "model_kind": kind,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "completed_updates": step + 1,
                "history": history,
                "curves": curves,
                "initial_weight_digest": initial_digest,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                "device_type": args.device,
                "training_gt_labels_used_for_gradients": True,
            }
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(state, temporary)
            os.replace(temporary, checkpoint)
            del state
            atomic_json(
                directory / "progress.json",
                {
                    "model": kind,
                    "completed_updates": step + 1,
                    "curves": curves,
                    "training_gt_labels_used_for_gradients": True,
                },
            )
            print(
                json.dumps(
                    {
                        "model": kind,
                        "step": step + 1,
                        "train_epe": row["train"]["metrics"]["epe_px"],
                        "validation_epe": row["validation"]["metrics"]["epe_px"],
                    }
                ),
                flush=True,
            )
    report = {
        "status": "supervised_control_completed" if until == control["updates"] else "supervised_stage_completed",
        "model": kind,
        "completed_updates": until,
        "curves": curves,
        "history": history,
        "initial_weight_sha256": initial_digest,
        "final_weight_sha256": weight_digest(model),
        "contract": contract,
        "source_commits": {k: v["commit"] for k, v in sources.items()},
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_gt_labels_used_for_gradients": True,
        "test_evaluated": False,
        "device": args.device,
        "torch": str(torch.__version__),
    }
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
    atomic_json(directory / "summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gt_control.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--until", type=int)
    parser.add_argument("--worker", choices=("segmentation", "snn"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        train_worker(args)
        return
    if args.source_run is None:
        parser.error("--source-run is required")
    control = json.loads(args.config.read_text())
    validate_control(control)
    if args.resume:
        saved = json.loads((args.output / "data/manifest.json").read_text())
        if saved["control"] != control or saved["source_fit_manifest_sha256"] != digest(
            args.source_run / "data/manifest.json"
        ):
            raise ValueError("Supervised resume source/config mismatch")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        prepare_control(args.source_run, control, args.output / "data")
    report = {
        "status": "running",
        "models": [],
        "training_gt_labels_used_for_gradients": True,
        "test_evaluated": False,
        "scope": "supervised_diagnostic_not_selfsupervised_benchmark",
    }
    try:
        for kind in ("segmentation", "snn"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(args.output),
                "--worker",
                kind,
                "--device",
                args.device,
            ]
            if args.until is not None:
                command += ["--until", str(args.until)]
            if args.resume and (args.output / kind / "last.pt").exists():
                command.append("--resume")
            subprocess.run(command, check=True)
            result = json.loads((args.output / kind / "summary.json").read_text())
            report["models"].append({k: v for k, v in result.items() if k != "history"})
            atomic_json(args.output / "summary.json", report)
        report["status"] = (
            "supervised_control_completed" if args.until in (None, control["updates"]) else "supervised_stage_completed"
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Recover the latest saved curve even if a worker was interrupted after a checkpoint.
        report["models"] = []
        for kind in ("segmentation", "snn"):
            candidates = [
                json.loads(p.read_text())
                for name in ("summary.json", "progress.json")
                if (p := args.output / kind / name).exists()
            ]
            if candidates:
                result = max(candidates, key=lambda r: r["completed_updates"])
                report["models"].append({k: v for k, v in result.items() if k != "history"})
        atomic_json(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
