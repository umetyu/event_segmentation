"""Score training/validation EPE in a separate process; never update weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_short_run import checkpoint_contract, load_weights, weighted_metrics
from dsec_data import flow_metrics
from fetch_sources import ensure_sources
from fit_probe_data import load_monitor_labels
from model_preflight import build_model
from short_data import digest
from short_train import atomic_json, evaluate, load_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", choices=("segmentation", "snn"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial", action="store_true")
    args = parser.parse_args()
    ensure_sources(download=False)
    manifest_path = (args.run / "data/manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    cache = load_cache(manifest_path, manifest)
    labels = load_monitor_labels(manifest_path, manifest)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, kind, _, _ = build_model(args.model + "-matched")
    step, checksum = 0, None
    if not args.initial:
        checkpoint = args.run / kind / "last.pt"
        step = load_weights(model, checkpoint, manifest_path, kind)
        checksum = digest(checkpoint)
    model.to(args.device)
    train_labeled = [
        {**sample, **{k: torch.from_numpy(v) for k, v in label.items()}}
        for sample, label in zip(cache["train"], labels)
    ]
    report = {
        "model": kind,
        "step": step,
        "manifest_sha256": digest(manifest_path),
        "code_contract": checkpoint_contract(manifest_path),
        "checkpoint_sha256": checksum,
        "test_evaluated": False,
    }
    for split, samples in (("train", train_labeled), ("validation", cache["validation"])):
        result = evaluate(model, kind, {"train": cache["train"], "validation": samples}, config, args.device)
        zero = weighted_metrics(
            [
                {"metrics": flow_metrics(np.zeros_like(s["flow"].numpy()), s["flow"].numpy(), s["valid"].numpy())}
                for s in samples
            ],
            "metrics",
        )
        report[split] = {
            "selfsupervised_loss": result["selfsupervised_loss"],
            "metrics": result["pixel_weighted_metrics"],
            "zero_metrics": zero,
            "warped_in_frame_fraction": float(np.mean([row["warped_in_frame_fraction"] for row in result["samples"]])),
            "samples": result["samples"],
        }
    if not args.initial and checksum != digest(checkpoint):
        raise ValueError("Checkpoint changed during read-only monitoring")
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
