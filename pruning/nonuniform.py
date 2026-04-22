from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from evaluation.metrics import evaluate_model
from pruning.masking import apply_masks, compute_mask


def compute_layer_sensitivity(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    scores_dict: dict[str, torch.Tensor],
    layer_names: list[str],
) -> dict[str, float]:
    del criterion
    baseline_results = evaluate_model(model, val_loader, device)
    baseline_balanced_acc = baseline_results["balanced_accuracy"]
    sensitivity: dict[str, float] = {}

    for layer_name in layer_names:
        copied_model = deepcopy(model).to(device)
        mask = compute_mask(scores_dict[layer_name], sparsity=0.5)
        apply_masks(copied_model, {layer_name: mask})
        pruned_results = evaluate_model(copied_model, val_loader, device)
        sensitivity[layer_name] = baseline_balanced_acc - pruned_results["balanced_accuracy"]

    return sensitivity


def allocate_sparsity(
    sensitivity: dict[str, float],
    target_avg: float = 0.5,
    bins: dict | None = None,
    layer_param_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    bins = bins or {"low_sensitivity": 0.7, "medium_sensitivity": 0.5, "high_sensitivity": 0.3}
    ordered = sorted(sensitivity.items(), key=lambda item: item[1], reverse=True)
    num_layers = len(ordered)
    high_cut = max(1, num_layers // 3)
    medium_cut = max(high_cut + 1, (2 * num_layers) // 3)

    allocation: dict[str, float] = {}
    for index, (layer_name, _) in enumerate(ordered):
        if index < high_cut:
            allocation[layer_name] = float(bins["high_sensitivity"])
        elif index < medium_cut:
            allocation[layer_name] = float(bins["medium_sensitivity"])
        else:
            allocation[layer_name] = float(bins["low_sensitivity"])

    if not allocation:
        return allocation

    if layer_param_counts:
        current_avg = sum(allocation[name] * layer_param_counts.get(name, 1) for name in allocation) / max(
            1, sum(layer_param_counts.get(name, 1) for name in allocation)
        )
    else:
        current_avg = sum(allocation.values()) / len(allocation)

    scale = target_avg / max(current_avg, 1e-8)
    return {name: float(min(1.0, max(0.0, sparsity * scale))) for name, sparsity in allocation.items()}


def apply_nonuniform_pruning(
    model: nn.Module,
    scores_dict: dict[str, torch.Tensor],
    allocation: dict[str, float],
) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    for layer_name, sparsity in allocation.items():
        mask = compute_mask(scores_dict[layer_name], sparsity)
        masks[layer_name] = mask
    apply_masks(model, masks)
    return masks
