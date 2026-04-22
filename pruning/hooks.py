from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class ActivationCollector:
    def __init__(self, model: nn.Module, target_layers: Iterable[tuple[str, nn.Module]]) -> None:
        self.model = model
        self.target_layers = dict(target_layers)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.sum_sq: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs, output) -> None:
            if not inputs:
                return
            activations = inputs[0]
            if activations is None:
                return
            features = activations.detach()
            flattened = features.reshape(-1, features.shape[-1]).float()
            self.sum_sq.setdefault(layer_name, torch.zeros(flattened.shape[-1], dtype=torch.float32))
            self.counts.setdefault(layer_name, 0)
            self.sum_sq[layer_name] += flattened.pow(2).sum(dim=0).cpu()
            self.counts[layer_name] += int(flattened.shape[0])

        return hook

    def register_hooks(self) -> None:
        for layer_name, module in self.target_layers.items():
            self.handles.append(module.register_forward_hook(self._make_hook(layer_name)))

    def get_activation_norms(self) -> dict[str, torch.Tensor]:
        norms: dict[str, torch.Tensor] = {}
        for layer_name, module in self.target_layers.items():
            if layer_name not in self.sum_sq:
                raise RuntimeError(f"No activation statistics were collected for layer '{layer_name}'.")
            norm = torch.sqrt(self.sum_sq[layer_name] / max(1, self.counts[layer_name]))
            if norm.shape[0] != module.in_features:
                raise RuntimeError(
                    f"Activation shape mismatch for {layer_name}: expected {module.in_features}, got {norm.shape[0]}. "
                    "This usually means output activations were captured instead of inputs."
                )
            norms[layer_name] = norm
        return norms

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class GradientCollector:
    def __init__(self, model: nn.Module, target_layers: Iterable[tuple[str, nn.Module]]) -> None:
        self.model = model
        self.target_layers = dict(target_layers)

    def get_gradients(self) -> dict[str, torch.Tensor]:
        gradients: dict[str, torch.Tensor] = {}
        for layer_name, module in self.target_layers.items():
            if module.weight.grad is None:
                raise RuntimeError(f"Missing gradient for layer '{layer_name}'.")
            gradients[layer_name] = module.weight.grad.detach().clone()
        return gradients
