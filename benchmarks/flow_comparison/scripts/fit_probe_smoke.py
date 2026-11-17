"""Exercise the probe with real models on synthetic data, including missing GT during training."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np

from fetch_sources import ROOT
from fit_probe import run_comparison
from fit_probe_data import prepare_probe
from short_train import atomic_json
from training_smoke import synthetic_manifest


def synthetic_labels(source, record):
    h, w = source["config"]["crop_hw"]
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = 4
    return {"flow": flow, "valid": np.ones((h, w), dtype=bool)}, {"fixture": "synthetic_constant_translation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/fit-probe-smoke.json")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="fit-probe-smoke-") as tmp:
        root = Path(tmp)
        (root / "source/data").mkdir(parents=True)
        synthetic_manifest(root / "source/data")
        output = root / "probe"
        config = {"protocol": "selfsup_fit_probe_v1", "train_record_indices": [0, 1], "updates": 4, "monitor_every": 2}
        with patch("fit_probe_data.read_monitor_labels", side_effect=synthetic_labels):
            manifest = prepare_probe(root / "source", config, output / "data")
        # If the trainer tries to use monitoring labels, these runs must fail.
        labels = list((output / "data").glob("monitor-train-*.npz"))
        for path in labels:
            path.rename(path.with_suffix(".hidden"))
        for kind in ("segmentation", "snn"):
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/short_train.py"),
                    "--manifest",
                    str(manifest),
                    "--model",
                    kind,
                    "--device",
                    "cpu",
                    "--output",
                    str(output / kind),
                    "--until",
                    "2",
                ],
                check=True,
            )
        for path in labels:
            path.with_suffix(".hidden").rename(path)
        # Recover missing monitoring results at step 2, then resume to step 4.
        run_comparison(manifest, output, "cpu", until=2)
        run_comparison(manifest, output, "cpu", until=4)
        report = json.loads((output / "summary.json").read_text())
        if report["status"] != "fit_probe_completed" or report["test_evaluated"]:
            raise ValueError("Unexpected probe status")
        for model in report["models"]:
            if [r["step"] for r in model["curve"]] != [0, 2, 4] or model["completed_updates"] != 4:
                raise ValueError("Monitoring/resume lost a step")
            for row in model["curve"]:
                for split in ("train", "validation"):
                    np.testing.assert_allclose(row[split]["zero_metrics"]["epe_px"], 4.0)
                    if not np.isfinite(row[split]["metrics"]["epe_px"]):
                        raise ValueError("Non-finite monitored EPE")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output,
            {
                "status": "passed",
                "synthetic_only": True,
                "training_succeeded_with_monitor_gt_files_absent": True,
                "recovered_monitoring_and_resumed": True,
                "models": report["models"],
            },
        )
        subprocess.run([sys.executable, str(ROOT / "scripts/fit_probe_report.py"), str(output)], check=True)


if __name__ == "__main__":
    main()
