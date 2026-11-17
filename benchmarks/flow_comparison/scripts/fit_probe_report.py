"""Print compact learning curves from the full saved diagnostic report."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "summary.json").read_text())
    print("Status:", report["status"])
    if "error" in report:
        print("Error:", report["error"])
    print("EPE is in pixels. Smaller is better. GT is used for monitoring only; test is not evaluated.")
    for model in report["models"]:
        print(f"\n{model['model']}: {model['completed_updates']} updates")
        print("step  train_loss train_EPE train_zero val_loss   val_EPE   val_zero train_keep val_keep")
        for row in model["curve"]:
            train, val = row["train"], row["validation"]
            print(
                f"{row['step']:4d}  {train['selfsupervised_loss']:10.6f} {train['metrics']['epe_px']:9.4f} "
                f"{train['zero_metrics']['epe_px']:10.4f} {val['selfsupervised_loss']:9.6f} "
                f"{val['metrics']['epe_px']:9.4f} {val['zero_metrics']['epe_px']:10.4f} "
                f"{train['warped_in_frame_fraction']:10.4f} {val['warped_in_frame_fraction']:8.4f}"
            )
    print("\nCompletion is not evidence of convergence. Full per-window results are in monitor-*.json.")


if __name__ == "__main__":
    main()
