"""Train one matched model with the shared self-supervised short-run protocol."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from dsec_data import flow_metrics
from fetch_sources import ROOT, ensure_sources
from flow_objective import final_flow, timestamp_objective, to_pixel_displacement
from model_preflight import build_model
from short_data import digest, validate_config


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def weight_digest(model):
    checksum = hashlib.sha256()
    for parameter in model.parameters():
        checksum.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def ordered_index(step, count, seed):
    epoch, offset = divmod(step, count)
    return int(np.random.default_rng(seed + epoch).permutation(count)[offset])


def loss_for(flow, data, config):
    return timestamp_objective(flow, data["loss_events"], config["smoothness_weight"], config["boundary_weight"])


def load_cache(manifest_path, manifest):
    cache = {}
    for split in ("train", "validation"):
        cache[split] = []
        for record in manifest[split]:
            if record["sequence"] not in manifest["config"]["splits"][split]:
                raise ValueError(f"Cached sample is outside the frozen {split} split")
            path = manifest_path.parent / record["file"]
            if digest(path) != record["sha256"]:
                raise ValueError(f"Cached input changed: {path}")
            with np.load(path, allow_pickle=False) as data:
                tensors = {k: torch.from_numpy(data[k].copy()) for k in data.files}
            if split == "train" and ({"flow", "valid"} & set(tensors)):
                raise ValueError("Training cache must not contain GT labels or masks")
            cache[split].append(tensors)
    if not cache["train"] or not cache["validation"]:
        raise ValueError("Need train and validation input caches")
    return cache


def on_device(sample, device):
    return {k: v.to(device) for k, v in sample.items()}


def evaluate(model, kind, cache, config, device):
    metrics = []
    # Validation must not perturb the RNG state used by resumed training.
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        model.eval()
        for sample in cache["validation"]:
            data = on_device(sample, device)
            flow = final_flow(model, kind, data)
            loss, components = loss_for(flow, data, config)
            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite validation loss")
            pixels = to_pixel_displacement(flow)[0].permute(1, 2, 0).cpu().numpy()
            metrics.append(
                {
                    "loss": loss.item(),
                    **components,
                    **flow_metrics(pixels, sample["flow"].numpy(), sample["valid"].numpy()),
                }
            )
        probe = on_device(cache["train"][0], device)
        probe_loss, _ = loss_for(final_flow(model, kind, probe), probe, config)
    model.reset_states()
    model.train()
    valid_pixels = sum(m["valid_pixels"] for m in metrics)
    return {
        "selfsupervised_loss": float(np.mean([m["loss"] for m in metrics])),
        "train_probe_loss": probe_loss.item(),
        "valid_pixels": valid_pixels,
        "pixel_weighted_metrics": {
            key: sum(m[key] * m["valid_pixels"] for m in metrics) / valid_pixels
            for key in ("epe_px", "1pe_percent", "2pe_percent", "3pe_percent")
        },
        "samples": metrics,
    }


def train(args):
    sources = ensure_sources(download=False)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    validate_config(config)
    if not 0 < args.until <= config["updates"]:
        raise ValueError("--until must be within the frozen update budget")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    if args.device == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    model, kind, _, _ = build_model(args.model + "-matched")
    model.to(args.device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    cache = load_cache(manifest_path, manifest)
    code_paths = [
        Path(__file__),
        ROOT / "scripts/flow_objective.py",
        ROOT / "scripts/model_preflight.py",
        ROOT / "sources.lock.json",
    ]
    contract = hashlib.sha256((digest(manifest_path) + "".join(digest(p) for p in code_paths)).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "last.pt"
    history, start, resume_count = [], 0, 0
    if args.resume:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["contract"] != contract or state["model_kind"] != kind:
            raise ValueError("Checkpoint model/data/config/code contract does not match")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if args.device == "cuda":
            if not state["cuda_rng"]:
                raise ValueError("Resume on the same device type used to create the checkpoint")
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        elif state["cuda_rng"]:
            raise ValueError("Resume on the same device type used to create the checkpoint")
        start, history = state["completed_updates"], state["history"]
        initial, original_digest = state["initial_validation"], state["initial_weight_digest"]
        resume_count = state["resume_count"] + 1
        if start >= args.until:
            raise ValueError("Resume target must exceed completed updates")
        del state
    else:
        if checkpoint_path.exists() or (args.output / "summary.json").exists():
            raise FileExistsError("Existing run: use --resume or a new output directory")
        initial = evaluate(model, kind, cache, config, args.device)
        original_digest = weight_digest(model)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(start, args.until):
        index = ordered_index(step, len(cache["train"]), config["seed"])
        data = on_device(cache["train"][index], args.device)
        optimizer.zero_grad(set_to_none=True)
        flow = final_flow(model, kind, data)
        loss, components = loss_for(flow, data, config)
        if not torch.isfinite(loss).item():
            raise ValueError(f"Non-finite training loss at step {step}")
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise ValueError(f"Parameters disconnected from common final-flow loss: {missing}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        if grad_norm.item() == 0:
            raise ValueError("All gradients are zero")
        optimizer.step()
        item = {
            "step": step + 1,
            "sample_index": index,
            "loss": loss.item(),
            "gradient_norm_before_clip": grad_norm.item(),
            **components,
        }
        history.append(item)
        print(json.dumps({"model": kind, **item}), flush=True)
        model.reset_states()
        del flow, loss, data
    final = evaluate(model, kind, cache, config, args.device)
    final_digest = weight_digest(model)
    if final_digest == original_digest:
        raise ValueError("Training did not change model parameters")
    state = {
        "contract": contract,
        "model_kind": kind,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "completed_updates": args.until,
        "history": history,
        "initial_validation": initial,
        "initial_weight_digest": original_digest,
        "resume_count": resume_count,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, checkpoint_path)
    summary = {
        "status": "short_training_completed",
        "model": kind,
        "completed_updates": args.until,
        "resume_count": resume_count,
        "resumed_from_update": start,
        "parameters_changed": True,
        "train_probe_loss_decreased": final["train_probe_loss"] < initial["train_probe_loss"],
        "validation_loss_decreased": final["selfsupervised_loss"] < initial["selfsupervised_loss"],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_validation": initial,
        "final_validation": final,
        "history": history,
        "training_gt_labels_used": False,
        "test_evaluated": False,
        "source_commits": {k: v["commit"] for k, v in sources.items()},
        "contract_sha256": contract,
        "final_weight_sha256": final_digest,
        "torch": str(torch.__version__),
        "config": config,
        "checkpoint": str(checkpoint_path),
        "scope": "short_cropped_training_check_not_final_benchmark",
        "reproducibility": "seeded_order_and_rng_saved_cuda_scatter_add_may_be_nondeterministic",
    }
    if args.device == "cuda":
        summary.update(
            gpu=torch.cuda.get_device_name(0), training_peak_allocated_bytes=torch.cuda.max_memory_allocated()
        )
    atomic_json(args.output / "summary.json", summary)
    print(
        json.dumps({"model": kind, "status": summary["status"], "summary": str(args.output / "summary.json")}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=("segmentation", "snn"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--until", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
