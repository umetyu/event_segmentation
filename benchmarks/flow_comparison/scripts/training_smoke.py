"""CPU-only synthetic optimizer/checkpoint regression; not a DSEC benchmark."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from fetch_sources import ROOT
from short_data import digest, pack_sample


def synthetic_manifest(root):
    config = json.loads((ROOT / "configs/short_train.json").read_text())
    config.update(
        crop_hw=[64, 80],
        updates=4,
        loss_event_limit=5000,
        splits={"train": ["synthetic_train_a"], "validation": ["synthetic_validation_a"], "test": ["synthetic_test_a"]},
    )
    manifest = {"config": config, "train": [], "validation": [], "fixture": "synthetic_only"}
    time = np.linspace(0.02, 0.98, 25)
    xx, yy, tt = np.meshgrid(np.arange(8, 68, 4), np.arange(8, 56, 4), time, indexing="ij")
    for split in ("train", "validation"):
        for index in range(2 if split == "train" else 1):
            sample = {
                "resolution": (64, 80),
                "x": (xx + 4 * tt + index * 0.25).ravel().astype(np.float32),
                "y": yy.ravel().astype(np.float32),
                "t": (tt.ravel() * 100000).astype(np.int64),
                "p": ((xx.ravel() // 4 + yy.ravel() // 4) % 2).astype(np.uint8),
                "start_us": 0,
                "end_us": 100000,
            }
            if split == "validation":
                flow = np.zeros((64, 80, 2), dtype=np.float32)
                flow[..., 0] = 4
                sample.update(flow=flow, valid=np.ones((64, 80), dtype=bool))
            encoded, _ = pack_sample(sample, config)
            filename = f"{split}-{index}.npz"
            np.savez(root / filename, **encoded)
            manifest[split].append(
                {"file": filename, "sha256": digest(root / filename), "sequence": config["splits"][split][0]}
            )
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/training-smoke.json")
    args = parser.parse_args()
    report = {"purpose": "synthetic_training_resume_check_only", "models": []}
    with tempfile.TemporaryDirectory(prefix="training-smoke-") as tmp:
        root = Path(tmp)
        (root / "data").mkdir()
        manifest = synthetic_manifest(root / "data")
        for kind in ("segmentation", "snn"):
            common = [
                sys.executable,
                str(ROOT / "scripts/short_train.py"),
                "--manifest",
                str(manifest),
                "--model",
                kind,
                "--device",
                "cpu",
            ]
            for directory, steps, resume in (("continuous", 4, False), ("resumed", 2, False), ("resumed", 4, True)):
                output = root / "continuous" / kind if directory == "continuous" else root / kind
                subprocess.run(
                    common + ["--output", str(output), "--until", str(steps)] + (["--resume"] if resume else []),
                    check=True,
                )
            continuous = torch.load(root / "continuous" / kind / "last.pt", map_location="cpu", weights_only=True)
            resumed = torch.load(root / kind / "last.pt", map_location="cpu", weights_only=True)
            for key in continuous["model"]:
                torch.testing.assert_close(continuous["model"][key], resumed["model"][key], rtol=0, atol=0)
            if continuous["history"] != resumed["history"]:
                raise ValueError("Resume changed losses or sample order")
            summary = json.loads((root / kind / "summary.json").read_text())
            report["models"].append(
                {
                    "model": kind,
                    "checkpoint_resume_exact_on_cpu": True,
                    "initial_probe_loss": summary["initial_validation"]["train_probe_loss"],
                    "final_probe_loss": summary["final_validation"]["train_probe_loss"],
                    "parameters_changed": summary["parameters_changed"],
                }
            )
            del continuous, resumed
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/diagnose_short_run.py"),
                "--run",
                str(root),
                "--output",
                str(root / "diagnostics"),
                "--device",
                "cpu",
                "--warmup",
                "1",
                "--repeats",
                "2",
            ],
            check=True,
        )
        diagnostics = json.loads((root / "diagnostics/summary.json").read_text())
        if diagnostics["status"] != "short_run_diagnostics_completed" or diagnostics["test_evaluated"]:
            raise ValueError("Unexpected diagnostic status")
        for result in diagnostics["models"]:
            saved = json.loads((root / result["model"] / "summary.json").read_text())
            np.testing.assert_allclose(
                result["metrics"]["epe_px"],
                saved["final_validation"]["pixel_weighted_metrics"]["epe_px"],
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(result["zero_metrics"]["epe_px"], 4.0)
            if result["model_latency"]["iterations"] != 2:
                raise ValueError("Missing timed inference repetitions")
        report["saved_checkpoint_diagnostics_passed"] = True
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
