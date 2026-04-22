from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from models.load_models import get_layer_by_name


def compute_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    if sparsity <= 0:
        return torch.ones_like(score, dtype=score.dtype)
    if sparsity >= 1:
        return torch.zeros_like(score, dtype=score.dtype)

    flat = score.reshape(-1)
    keep_count = max(0, min(flat.numel(), int(round((1.0 - sparsity) * flat.numel()))))
    mask_flat = torch.zeros_like(flat, dtype=score.dtype)
    if keep_count > 0:
        topk_indices = torch.argsort(flat, descending=True)[:keep_count]
        mask_flat[topk_indices] = 1.0
    return mask_flat.reshape_as(score)


def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for layer_name, mask in masks.items():
            module = get_layer_by_name(model, layer_name)
            module.weight.mul_(mask.to(module.weight.device, dtype=module.weight.dtype))


def compute_global_masks(model: nn.Module, scores: dict[str, torch.Tensor], sparsity: float) -> dict[str, torch.Tensor]:
    if not scores:
        return {}
    flat_scores = torch.cat([score.reshape(-1) for score in scores.values()])
    if sparsity <= 0:
        return {name: torch.ones_like(score, dtype=score.dtype) for name, score in scores.items()}
    if sparsity >= 1:
        return {name: torch.zeros_like(score, dtype=score.dtype) for name, score in scores.items()}

    keep_count = max(0, min(flat_scores.numel(), int(round((1.0 - sparsity) * flat_scores.numel()))))
    global_mask = torch.zeros_like(flat_scores, dtype=flat_scores.dtype)
    if keep_count > 0:
        topk_indices = torch.argsort(flat_scores, descending=True)[:keep_count]
        global_mask[topk_indices] = 1.0

    masks: dict[str, torch.Tensor] = {}
    cursor = 0
    for layer_name, score in scores.items():
        count = score.numel()
        masks[layer_name] = global_mask[cursor : cursor + count].reshape_as(score)
        cursor += count
    return masks


def get_sparsity_stats(model: nn.Module, masks: dict[str, torch.Tensor]) -> dict:
    per_layer: dict[str, dict[str, float | int]] = {}
    total_params = 0
    nonzero_params = 0

    for layer_name, mask in masks.items():
        total = int(mask.numel())
        kept = int(mask.sum().item())
        pruned = total - kept
        per_layer[layer_name] = {
            "sparsity": pruned / max(1, total),
            "total": total,
            "pruned": pruned,
        }
        total_params += total
        nonzero_params += kept

    return {
        "global_sparsity": 1.0 - (nonzero_params / max(1, total_params)),
        "per_layer": per_layer,
        "total_params": total_params,
        "nonzero_params": nonzero_params,
    }
