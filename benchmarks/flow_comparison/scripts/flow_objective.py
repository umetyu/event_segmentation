"""Shared self-supervised timestamp-warping objective, batch size one.

Both models' final normalized flow is interpreted as displacement over the
ENTIRE input interval and multiplied by max(H, W), following the upstream
flow_scaling convention. The SNN output is not multiplied by its step count.
This controlled retraining objective is not an exact reproduction of either paper.
"""

import torch
import torch.nn.functional as F


def final_flow(model, kind, encoded):
    model.reset_states()
    voxel = encoded["voxels"].unsqueeze(0)
    counts = encoded["snn_counts"] if kind == "snn" else encoded["counts"].unsqueeze(0)
    output = None
    for current in counts:
        output = model(voxel, current.unsqueeze(0))["flow"]
    # Earlier SNN prediction heads also feed subsequent decoders via skip connections.
    flow = output[-1] if isinstance(output, (list, tuple)) else output
    if flow.shape != (1, 2, *voxel.shape[-2:]) or not torch.isfinite(flow).all().item():
        raise ValueError("Invalid final flow")
    return flow


def to_pixel_displacement(flow):
    return flow * max(flow.shape[-2:])


def timestamp_objective(flow, events, smoothness_weight=0.001, boundary_weight=0.1):
    """events: N x 4 float tensor (x, y, normalized time, polarity 0/1).

    Bilinear flow sampling and event splatting, two time references and separate
    polarities. Smoothness operates on normalized flow. A differentiable boundary
    penalty discourages reducing contrast loss by moving events off the image.
    GT labels and masks are never inputs to this function.
    """
    if events.ndim != 2 or events.shape[1] != 4 or not len(events):
        raise ValueError("Need nonempty N x 4 events")
    h, w = flow.shape[-2:]
    xy, time, polarity = events[:, :2], events[:, 2], events[:, 3].long()
    grid = torch.stack((2 * xy[:, 0] / (w - 1) - 1, 2 * xy[:, 1] / (h - 1) - 1), dim=-1)
    sampled = F.grid_sample(to_pixel_displacement(flow), grid[None, None], align_corners=True)
    sampled = sampled[0, :, 0].T
    temporal_terms, boundary_terms, retention = [], [], []
    for reference in (0.0, 1.0):
        warped = xy + (reference - time[:, None]) * sampled
        distances = F.relu(-warped) + F.relu(warped - warped.new_tensor([w - 1, h - 1]))
        boundary_terms.append((distances / max(h, w)).square().mean())
        retention.append(((warped >= 0) & (warped <= warped.new_tensor([w - 1, h - 1]))).all(dim=1).float().mean())
        base = torch.floor(warped).detach()
        count = flow.new_zeros(2 * h * w)
        weighted_time = flow.new_zeros(2 * h * w)
        stamp = time if reference == 1 else 1 - time
        for dx in (0, 1):
            for dy in (0, 1):
                neighbor = base + base.new_tensor([dx, dy])
                weight = (1 - (warped - neighbor).abs()).prod(dim=1)
                keep = (neighbor[:, 0] >= 0) & (neighbor[:, 0] < w) & (neighbor[:, 1] >= 0) & (neighbor[:, 1] < h)
                idx = polarity[keep] * h * w + neighbor[keep, 1].long() * w + neighbor[keep, 0].long()
                count = count.scatter_add(0, idx, weight[keep])
                weighted_time = weighted_time.scatter_add(0, idx, weight[keep] * stamp[keep])
        average = weighted_time / (count + 1e-6)
        occupied = (count.reshape(2, h, w).sum(dim=0) > 1e-6).sum().clamp_min(1)
        temporal_terms.append(average.square().sum() / occupied)
    temporal = torch.stack(temporal_terms).mean()
    dx = flow[..., 1:] - flow[..., :-1]
    dy = flow[..., 1:, :] - flow[..., :-1, :]
    smoothness = ((dx.square() + 1e-6).sqrt().mean() + (dy.square() + 1e-6).sqrt().mean()) / 2
    boundary = torch.stack(boundary_terms).mean()
    loss = temporal + smoothness_weight * smoothness + boundary_weight * boundary
    return loss, {
        "timestamp": temporal.detach().item(),
        "smoothness": smoothness.detach().item(),
        "boundary": boundary.detach().item(),
        "warped_in_frame_fraction": torch.stack(retention).mean().detach().item(),
    }
