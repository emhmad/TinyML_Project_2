from __future__ import annotations

from typing import Iterable

import timm
import torch.nn as nn


def load_deit_model(model_name: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def get_linear_layer_names(
    model: nn.Module,
    exclude_keywords: Iterable[str] | None = None,
) -> list[tuple[str, nn.Linear]]:
    excluded = set(exclude_keywords or {"head"})
    layers: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(keyword in name for keyword in excluded):
            continue
        layers.append((name, module))
    return layers


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    module = model
    for part in layer_name.split("."):
        module = getattr(module, part)
    return module


def classify_layer_type(layer_name: str) -> str:
    if "attn.qkv" in layer_name:
        return "qkv"
    if "attn.proj" in layer_name:
        return "attn_out"
    if "mlp.fc1" in layer_name or "mlp.fc2" in layer_name:
        return "mlp"
    if "patch_embed" in layer_name:
        return "patch_embed"
    if layer_name.startswith("head") or ".head" in layer_name:
        return "head"
    return "other"
