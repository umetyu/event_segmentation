"""Print a supervised diagnostic without conflating it with self-supervised results."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "summary.json").read_text())
    print("Status:", report["status"])
    print("SUPERVISED DIAGNOSTIC: training GT is used for gradients. Validation/test GT is not used for gradients.")
    print("Objective: mean smoothed GT EPE in pixels. Not a self-supervised benchmark result.")
    if "error" in report:
        print("Error:", report["error"])
    for model in report["models"]:
        print(f"\n{model['model']}: {model['completed_updates']} updates")
        print("step  train_EPE train_zero   val_EPE  val_zero")
        for row in model["curves"]:
            train, val = row["train"], row["validation"]
            print(
                f"{row['step']:4d}  {train['metrics']['epe_px']:9.4f} {train['zero_metrics']['epe_px']:10.4f} "
                f"{val['metrics']['epe_px']:9.4f} {val['zero_metrics']['epe_px']:9.4f}"
            )
    print(
        "\nCompletion does not establish convergence; validation improvement is not required for this fitting diagnostic."
    )


if __name__ == "__main__":
    main()
