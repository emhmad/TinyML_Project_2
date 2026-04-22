from __future__ import annotations

import torch.nn as nn

from models.load_models import classify_layer_type, get_linear_layer_names
from pruning.masking import apply_masks, compute_global_masks


def get_layer_groups(model: nn.Module) -> dict[str, list[str]]:
    groups = {"qkv": [], "attn_out": [], "mlp": [], "patch_embed": []}
    for layer_name, _ in get_linear_layer_names(model, exclude_keywords={"head"}):
        layer_type = classify_layer_type(layer_name)
        if layer_type in groups:
            groups[layer_type].append(layer_name)
    return groups


def prune_only_group(
    model: nn.Module,
    scores: dict[str, torch.Tensor],
    sparsity: float,
    group_name: str,
    layer_groups: dict[str, list[str]],
) -> dict[str, torch.Tensor]:
    target_layers = set(layer_groups.get(group_name, []))
    group_scores = {name: score for name, score in scores.items() if name in target_layers}
    masks = compute_global_masks(model, group_scores, sparsity)
    apply_masks(model, masks)
    return masks
