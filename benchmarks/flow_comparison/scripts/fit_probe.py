"""Repeat a frozen small training subset, scoring GT only in separate monitor workers."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from diagnose_short_run import checkpoint_contract
from fetch_sources import ROOT
from fit_probe_data import load_monitor_labels, prepare_probe, validate_probe
from short_data import digest
from short_train import atomic_json, load_cache


def completed_step(path, manifest, kind):
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["contract"] != checkpoint_contract(manifest) or state["model_kind"] != kind:
        raise ValueError("Existing checkpoint contract mismatch")
    return state["completed_updates"]


def run_comparison(manifest_path, output, device, until=None):
    manifest = json.loads(manifest_path.read_text())
    load_cache(manifest_path, manifest)
    load_monitor_labels(manifest_path, manifest)
    probe = manifest["probe"]
    until = probe["updates"] if until is None else until
    if not 0 < until <= probe["updates"] or until % probe["monitor_every"] and until != probe["updates"]:
        raise ValueError("Target must be a monitoring boundary within the frozen update budget")
    targets = list(range(probe["monitor_every"], until + 1, probe["monitor_every"]))
    if not targets or targets[-1] != until:
        targets.append(until)
    report = {
        "status": "running",
        "models": [],
        "probe": probe,
        "test_evaluated": False,
        "training_gt_labels_used_for_gradients": False,
        "scope": "small_sample_learning_diagnostic_not_convergence_or_final_benchmark",
    }
    try:
        for kind in ("segmentation", "snn"):
            directory = output / kind
            directory.mkdir(exist_ok=True)
            completed = completed_step(directory / "last.pt", manifest_path, kind)
            if completed > until:
                raise ValueError("Cannot rewind a completed checkpoint")
            curve = []
            report["models"].append({"model": kind, "completed_updates": completed, "curve": curve})
            for target in [0] + targets:
                monitor = directory / f"monitor-{target:06d}.json"
                if target and completed < target:
                    command = [
                        sys.executable,
                        str(ROOT / "scripts/short_train.py"),
                        "--manifest",
                        str(manifest_path),
                        "--model",
                        kind,
                        "--device",
                        device,
                        "--output",
                        str(directory),
                        "--until",
                        str(target),
                    ]
                    subprocess.run(command + (["--resume"] if completed else []), check=True)
                    completed = target
                if not monitor.exists():
                    if target and target != completed:
                        raise ValueError("Historical monitoring result missing; cannot recreate from later weights")
                    command = [
                        sys.executable,
                        str(ROOT / "scripts/fit_probe_monitor.py"),
                        "--run",
                        str(output),
                        "--model",
                        kind,
                        "--device",
                        device,
                        "--output",
                        str(monitor),
                    ]
                    subprocess.run(command + (["--initial"] if target == 0 else []), check=True)
                result = json.loads(monitor.read_text())
                if (
                    result["step"] != target
                    or result["model"] != kind
                    or result["manifest_sha256"] != digest(manifest_path)
                    or result["code_contract"] != checkpoint_contract(manifest_path)
                ):
                    raise ValueError("Monitoring result does not match model/step/manifest")
                curve.append(
                    {
                        "step": target,
                        **{
                            split: {k: v for k, v in result[split].items() if k != "samples"}
                            for split in ("train", "validation")
                        },
                    }
                )
                report["models"][-1]["completed_updates"] = completed
                atomic_json(output / "summary.json", report)
            report["models"][-1]["train_epe_decreased"] = (
                curve[-1]["train"]["metrics"]["epe_px"] < curve[0]["train"]["metrics"]["epe_px"]
            )
        report["status"] = "fit_probe_completed" if until == probe["updates"] else "fit_probe_stage_completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(output / "summary.json", report)
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fit_probe.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--until", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    config = json.loads(args.config.read_text())
    source = (args.source_run / "data/manifest.json").resolve()
    if args.resume:
        manifest_path = output / "data/manifest.json"
        saved = json.loads(manifest_path.read_text())
        validate_probe(config, len(json.loads(source.read_text())["train"]))
        if saved["probe"] != config or saved["source_manifest_sha256"] != digest(source):
            raise ValueError("Resume configuration/source manifest mismatch")
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest_path = prepare_probe(args.source_run, config, output / "data")
    run_comparison(manifest_path, output, args.device, args.until)


if __name__ == "__main__":
    main()
