"""CPU real-model checks: supervised updates, exact resume and validation-label isolation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np

from fetch_sources import ROOT
from fit_probe_data import prepare_probe
from fit_probe_smoke import synthetic_labels
from gt_control import prepare_control
from short_data import digest
from short_train import atomic_json
from training_smoke import synthetic_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/gt-control-smoke.json")
    args = parser.parse_args()
    report = {"status": "passed", "synthetic_only": True, "models": []}
    with tempfile.TemporaryDirectory(prefix="gt-control-smoke-") as tmp:
        root = Path(tmp)
        (root / "source/data").mkdir(parents=True)
        synthetic_manifest(root / "source/data")
        probe = {"protocol": "selfsup_fit_probe_v1", "train_record_indices": [0, 1], "updates": 4, "monitor_every": 2}
        with patch("fit_probe_data.read_monitor_labels", side_effect=synthetic_labels):
            prepare_probe(root / "source", probe, root / "fit/data")
        control = {"protocol": "supervised_fit_control_v1", "updates": 4, "monitor_every": 2, "epe_epsilon_px": 0.001}
        for name in ("continuous", "resumed", "changed-validation"):
            prepare_control(root / "fit", control, root / name / "data")
        # Change only validation labels, including their cache hash. Training must be identical.
        manifest_path = root / "changed-validation/data/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["validation"]:
            path = manifest_path.parent / record["file"]
            with np.load(path, allow_pickle=False) as data:
                values = {k: data[k].copy() for k in data.files}
            values["flow"] *= -7
            np.savez(path, **values)
            record["sha256"] = digest(path)
        atomic_json(manifest_path, manifest)
        for kind in ("segmentation", "snn"):
            for name, steps, resume in (
                ("continuous", 4, False),
                ("resumed", 2, False),
                ("resumed", 4, True),
                ("changed-validation", 4, False),
            ):
                command = [
                    sys.executable,
                    str(ROOT / "scripts/gt_control.py"),
                    "--output",
                    str(root / name),
                    "--worker",
                    kind,
                    "--device",
                    "cpu",
                    "--until",
                    str(steps),
                ]
                subprocess.run(command + (["--resume"] if resume else []), check=True)
            results = [
                json.loads((root / name / kind / "summary.json").read_text())
                for name in ("continuous", "resumed", "changed-validation")
            ]
            a, b, c = results
            if a["final_weight_sha256"] != b["final_weight_sha256"] or a["history"] != b["history"]:
                raise ValueError("Supervised resume changed CPU weights or training history")
            if a["final_weight_sha256"] != c["final_weight_sha256"] or a["history"] != c["history"]:
                raise ValueError("Validation labels affected supervised training")
            if a["initial_weight_sha256"] == a["final_weight_sha256"]:
                raise ValueError("No supervised parameter updates")
            if a["curves"][-1]["train"]["metrics"]["epe_px"] >= a["curves"][0]["train"]["metrics"]["epe_px"]:
                raise ValueError("Synthetic training EPE did not decrease")
            report["models"].append(
                {
                    "model": kind,
                    "cpu_resume_exact": True,
                    "validation_labels_isolated": True,
                    "initial_train_epe": a["curves"][0]["train"]["metrics"]["epe_px"],
                    "final_train_epe": a["curves"][-1]["train"]["metrics"]["epe_px"],
                }
            )
        # Exercise the public coordinator on a completed resumed run without retraining.
        atomic_json(root / "control.json", control)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/gt_control.py"),
                "--source-run",
                str(root / "fit"),
                "--config",
                str(root / "control.json"),
                "--output",
                str(root / "resumed"),
                "--device",
                "cpu",
                "--resume",
            ],
            check=True,
        )
        subprocess.run([sys.executable, str(ROOT / "scripts/gt_control_report.py"), str(root / "resumed")], check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
