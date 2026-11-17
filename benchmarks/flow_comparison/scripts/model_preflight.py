"""Count matched upstream models and check synthetic forward/backward execution.

This is not a DSEC trainer, an accuracy evaluation, or a speed benchmark.
Each model is imported in a separate process because both upstream repositories
use the top-level Python package name 'models'.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from fetch_sources import ROOT, ensure_sources

CASES = (
    "segmentation-reference",
    "snn-reference",
    "segmentation-matched",
    "snn-matched",
)


def build_model(case):
    import torch

    kind = case.split("-")[0]
    matched = case.endswith("matched")
    if "models" in sys.modules:
        raise RuntimeError("Load each upstream model in a fresh Python process")
    sys.path.insert(0, str(ROOT / ".sources" / kind))
    from models import model as upstream

    width = 38 if kind == "snn" and matched else 32
    config = {
        "base_num_channels": width,
        "kernel_size": 3,
        "encoding": "voxel" if kind == "segmentation" else "cnt",
        "num_bins": 5 if kind == "segmentation" else 2,
        "norm_input": False,
        "mask_output": False,
        "activations": ["relu", None] if kind == "segmentation" else ["arctanspike", "arctanspike"],
        "spiking_neuron": (
            None
            if kind == "segmentation"
            else {
                "leak": [-4.0, 0.1],
                "thresh": [0.8, 0.1],
                "learn_leak": True,
                "learn_thresh": True,
                "hard_reset": True,
            }
        ),
    }
    cls = upstream.EVFlowNet_Segmentation if kind == "segmentation" else upstream.SpikingRecEVFlowNet
    model = cls(config)
    adjustment = None
    if kind == "segmentation" and matched:
        layers = model.multires_unet.encoders_optical_flow_module
        first, second = layers[0].conv2d, layers[1].conv2d
        if (first.in_channels, first.out_channels, second.in_channels, second.out_channels) != (512, 512, 512, 512):
            raise RuntimeError("Unexpected upstream affine-flow branch shape")

        def resized(old, inputs, outputs):
            return torch.nn.Conv2d(
                inputs,
                outputs,
                old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                dilation=old.dilation,
                groups=old.groups,
                bias=old.bias is not None,
                padding_mode=old.padding_mode,
            )

        # Every added weight participates in the optical-flow branch.
        # Preserve all activations, spatial strides, outputs, and skip connections.
        layers[0].conv2d = resized(first, 512, 548)
        layers[1].conv2d = resized(second, 548, 512)
        adjustment = "affine branch first intermediate width: 512 -> 548"
    return model, kind, width, adjustment


def run_worker(args):
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(0)
    model, kind, width, adjustment = build_model(args.worker)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "case": args.worker,
        "base_num_channels": width,
        "adjustment": adjustment,
        "trainable_parameters": trainable,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
    }
    if args.count_only or args.worker.endswith("reference"):
        return result

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected environment/allocation")
    model = model.to(args.device).train()
    if args.device == "cuda":
        device = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu": device.name,
                "gpu_memory_gib": device.total_memory / 2**30,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "cuda_runtime": torch.version.cuda,
                "compiled_cuda_architectures": torch.cuda.get_arch_list(),
            }
        )
        torch.cuda.reset_peak_memory_stats()

    model.reset_states()
    loss = None
    observed_shapes = []
    steps = 2 if kind == "snn" else 1
    for _ in range(steps):
        counts = torch.randint(0, 3, (1, 2, args.height, args.width), device=args.device).float()
        voxels = torch.randn(1, 5, args.height, args.width, device=args.device)
        outputs = model(voxels, counts)
        flows = outputs["flow"]
        if isinstance(flows, torch.Tensor):
            flows = [flows]
        for flow in flows:
            if tuple(flow.shape) != (1, 2, args.height, args.width):
                raise RuntimeError(f"Unexpected flow shape: {tuple(flow.shape)}")
            if not bool(torch.isfinite(flow).all()):
                raise RuntimeError("Non-finite synthetic flow output")
        observed_shapes = [list(flow.shape) for flow in flows]
        step_loss = sum(flow.square().mean() for flow in flows)
        loss = step_loss if loss is None else loss + step_loss
    loss.backward()
    missing_gradients = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite_gradients = [
        n for n, p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())
    ]
    if missing_gradients or nonfinite_gradients:
        raise RuntimeError(f"Gradient check failed: missing={missing_gradients}, nonfinite={nonfinite_gradients}")
    result.update(
        {
            "status": "synthetic_forward_backward_passed",
            "input_resolution_hw": [args.height, args.width],
            "synthetic_steps": steps,
            "output_shapes": observed_shapes,
            "loss": float(loss.detach()),
            "all_trainable_parameters_connected_to_loss": True,
            "purpose": "shape_and_gradient_check_only_not_accuracy_or_speed",
        }
    )
    if kind == "segmentation":
        result["resized_branch_has_nonzero_gradient"] = all(
            bool(layer.conv2d.weight.grad.abs().sum() > 0)
            for layer in model.multires_unet.encoders_optical_flow_module[:2]
        )
        if not result["resized_branch_has_nonzero_gradient"]:
            raise RuntimeError("Resized affine-flow branch received no gradient")
    if args.device == "cuda":
        torch.cuda.synchronize()
        result["synthetic_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.height, args.width) < 16 or args.height % 16 or args.width % 16:
        parser.error("Height and width must be positive multiples of 16 (at least 16)")
    manifest = ensure_sources(download=False)
    if args.worker:
        print(json.dumps(run_worker(args)))
        return

    results = []
    for case in CASES:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            case,
            "--device",
            args.device,
            "--height",
            str(args.height),
            "--width",
            str(args.width),
        ]
        if args.count_only:
            command.append("--count-only")
        proc = subprocess.run(command, text=True, capture_output=True)
        if proc.returncode:
            print(proc.stdout, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"Preflight failed for {case}: exit {proc.returncode}")
        results.append(json.loads(proc.stdout))
    seg, snn = (r["trainable_parameters"] for r in results if r["case"].endswith("matched"))
    relative_difference = abs(seg - snn) / seg
    report = {
        "source_commits": {name: repo["commit"] for name, repo in manifest.items()},
        "models": results,
        "matched_absolute_parameter_difference": abs(seg - snn),
        "matched_relative_parameter_difference": relative_difference,
        "parameter_tolerance": 0.01,
        "parameter_matching_passed": relative_difference <= 0.01,
        "dsec_training_or_evaluation_performed": False,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if not report["parameter_matching_passed"]:
        raise SystemExit("Model sizes exceed the 1% matching tolerance")


if __name__ == "__main__":
    main()
