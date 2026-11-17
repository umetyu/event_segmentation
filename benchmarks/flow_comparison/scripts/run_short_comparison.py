"""Prepare shared data, train each model, and exercise checkpoint resume."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from fetch_sources import ROOT
from short_data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/short_train.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["updates"] < 2:
        parser.error("At least two updates are needed for the checkpoint/resume check")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "models": [], "test_evaluated": False}
    try:
        manifest = prepare(args.root, config, args.output / "data")
        for kind in ("segmentation", "snn"):
            command = [
                sys.executable,
                str(ROOT / "scripts/short_train.py"),
                "--manifest",
                str(manifest),
                "--model",
                kind,
                "--device",
                args.device,
                "--output",
                str(args.output / kind),
            ]
            subprocess.run(command + ["--until", str(config["updates"] // 2)], check=True)
            subprocess.run(command + ["--until", str(config["updates"]), "--resume"], check=True)
            result = json.loads((args.output / kind / "summary.json").read_text())
            report["models"].append(
                {
                    **{
                        k: result[k]
                        for k in (
                            "model",
                            "completed_updates",
                            "resume_count",
                            "parameters_changed",
                            "train_probe_loss_decreased",
                            "validation_loss_decreased",
                            "trainable_parameters",
                            "checkpoint",
                        )
                    },
                    "initial_train_probe_loss": result["initial_validation"]["train_probe_loss"],
                    "final_train_probe_loss": result["final_validation"]["train_probe_loss"],
                    "initial_validation_loss": result["initial_validation"]["selfsupervised_loss"],
                    "final_validation_loss": result["final_validation"]["selfsupervised_loss"],
                    "initial_validation_metrics": result["initial_validation"]["pixel_weighted_metrics"],
                    "final_validation_metrics": result["final_validation"]["pixel_weighted_metrics"],
                }
            )
        seg, snn = [r["trainable_parameters"] for r in report["models"]]
        if abs(seg - snn) / seg > 0.01:
            raise ValueError("Parameter mismatch exceeds 1%")
        report.update(
            status="short_comparison_completed",
            config=config,
            manifest=str(manifest),
            scope="training_wiring_check_not_final_accuracy_or_speed_comparison",
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
