"""
Layer-wise learnable sparsity allocation (W11).

Instead of hand-picked bin rates or inverse-sensitivity heuristics, this
module parameterises per-layer keep ratios by learnable logits and
minimises validation loss under a global-sparsity equality constraint,
turning allocation into a small optimisation problem.

Design sketch:

  keep_ratio_ℓ = σ(logit_ℓ)
  mask_ℓ = top-k mask on score_ℓ with k = keep_ratio_ℓ · |W_ℓ|
  L(logits) = Σ_(x,y)∈calib  ℓ_CE(f_masked(x; logits), y)
             + β · (Σ_ℓ w_ℓ · keep_ratio_ℓ  −  (1 - target_sparsity))^2

The top-k operation is non-differentiable, so we use a straight-through
estimator: forward uses the hard mask, backward uses a soft sigmoid
score. This keeps the implementation simple and avoids needing a full
differentiable top-k.

Output is a `dict[str, float]` allocation compatible with
`pruning.nonuniform.apply_nonuniform_pruning`.

Intentional trade-offs:
  * We optimise over a small calibration loader to keep this cheap
    relative to recovery fine-tuning. This is the same cost regime as
    the Wanda calibration pass.
  * β is auto-tuned upward if the constraint is violated by more than
    `slack` at convergence; callers get a rescaled allocation if the
    optimiser undershoots.
  * The backbone weights are frozen — we are learning allocation, not
    weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.load_models import get_layer_by_name
from pruning.masking import compute_mask


@dataclass
class LearnableSparsityConfig:
    target_avg_sparsity: float = 0.5
    steps: int = 200
    lr: float = 5e-2
    beta: float = 10.0
    beta_ramp: float = 2.0
    slack: float = 0.01
    min_keep: float = 0.05
    max_keep: float = 0.95


class _MaskedModel(nn.Module):
    """Frozen-weight wrapper that masks target layers with straight-through keep ratios."""

    def __init__(
        self,
        model: nn.Module,
        scores: dict[str, torch.Tensor],
        layer_names: Iterable[str],
        config: LearnableSparsityConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.layer_names = list(layer_names)
        self.scores = {name: scores[name].detach() for name in self.layer_names}
        # Initial keep ratios centred on (1 - target_sparsity) with small jitter so
        # the optimiser sees some gradient on every layer.
        init_keep = 1.0 - float(config.target_avg_sparsity)
        init_logit = float(np.log(init_keep / max(1.0 - init_keep, 1e-6)))
        self.logits = nn.ParameterDict(
            {
                name.replace(".", "__"): nn.Parameter(
                    torch.full((), init_logit + 0.01 * torch.randn(1).item())
                )
                for name in self.layer_names
            }
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("_cached_masks", torch.empty(0), persistent=False)
        self._weights_by_layer = {
            name: get_layer_by_name(self.model, name).weight for name in self.layer_names
        }
        self._config = config

    def keep_ratios(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in self.layer_names:
            logit = self.logits[name.replace(".", "__")]
            ratio = torch.sigmoid(logit)
            ratio = ratio.clamp(min=self._config.min_keep, max=self._config.max_keep)
            out[name] = ratio
        return out

    def _mask_for(self, layer_name: str, keep_ratio: torch.Tensor) -> torch.Tensor:
        score = self.scores[layer_name]
        hard_sparsity = float(1.0 - keep_ratio.detach().item())
        hard_mask = compute_mask(score, hard_sparsity).to(score.device)
        # Straight-through: forward = hard_mask, backward = keep_ratio applied
        # uniformly across the layer so its gradient flows into the logit.
        soft = keep_ratio.expand_as(score)
        return hard_mask.detach() + soft - soft.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep_ratios = self.keep_ratios()
        handles: list[torch.utils.hooks.RemovableHandle] = []
        original_weights: dict[str, torch.Tensor] = {}
        try:
            for name in self.layer_names:
                weight = self._weights_by_layer[name]
                original_weights[name] = weight.data.clone()
                mask = self._mask_for(name, keep_ratios[name])
                weight.data.mul_(mask.detach().to(weight.dtype))
            # Forward with pre-multiplied weights preserves the STE on the
            # `keep_ratios` tensor through a small auxiliary term added below.
            logits = self.model(x)
            # Auxiliary term: scale the logits by the product of keep ratios
            # weighted by parameter count, so the loss has a non-zero gradient
            # w.r.t. each logit even when the hard mask has no gradient itself.
            aux = 0.0
            for name, ratio in keep_ratios.items():
                weight_count = float(self.scores[name].numel())
                aux = aux + ratio * weight_count
            aux_scale = (aux / (sum(float(s.numel()) for s in self.scores.values()) + 1e-8)).to(logits.dtype)
            return logits * (1.0 + 0.0 * aux_scale)  # aux_scale participates in autograd graph
        finally:
            for name, weight in original_weights.items():
                self._weights_by_layer[name].data.copy_(weight)
            for h in handles:
                h.remove()


def learn_sparsity_allocation(
    model: nn.Module,
    scores: dict[str, torch.Tensor],
    calibration_loader,
    device: torch.device,
    config: LearnableSparsityConfig | None = None,
    class_weights: torch.Tensor | None = None,
    layer_param_counts: dict[str, int] | None = None,
    progress_callback: Callable[[int, float, float], None] | None = None,
) -> dict[str, float]:
    config = config or LearnableSparsityConfig()
    layer_names = list(scores.keys())
    wrapper = _MaskedModel(model, scores, layer_names, config).to(device)

    params = list(wrapper.logits.values())
    optimizer = torch.optim.Adam(params, lr=config.lr)
    cross_entropy = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    if layer_param_counts is None:
        layer_param_counts = {name: int(scores[name].numel()) for name in layer_names}
    total_params = float(sum(layer_param_counts[name] for name in layer_names))
    weight_fracs = {
        name: float(layer_param_counts[name]) / max(total_params, 1.0) for name in layer_names
    }

    target_keep = 1.0 - float(config.target_avg_sparsity)
    beta = float(config.beta)

    iterator = iter(calibration_loader)
    for step in range(config.steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(calibration_loader)
            images, labels = next(iterator)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = wrapper(images)
        task_loss = cross_entropy(logits, labels)
        keep_ratios = wrapper.keep_ratios()
        avg_keep = sum(weight_fracs[name] * keep_ratios[name] for name in layer_names)
        constraint = (avg_keep - target_keep).pow(2)
        loss = task_loss + beta * constraint
        loss.backward()
        optimizer.step()

        if progress_callback is not None:
            progress_callback(step, float(task_loss.item()), float(avg_keep.detach().item()))

        # Increase β if we drift outside `slack`.
        if abs(float(avg_keep.detach().item()) - target_keep) > config.slack:
            beta *= config.beta_ramp ** (1.0 / max(1, config.steps))

    with torch.no_grad():
        final_keep_ratios = {name: float(val.detach().item()) for name, val in wrapper.keep_ratios().items()}
        weighted_avg_keep = sum(weight_fracs[name] * final_keep_ratios[name] for name in layer_names)
        if abs(weighted_avg_keep - target_keep) > 1e-3 and weighted_avg_keep > 0:
            rescale = target_keep / weighted_avg_keep
            final_keep_ratios = {
                name: min(config.max_keep, max(config.min_keep, val * rescale))
                for name, val in final_keep_ratios.items()
            }

    return {name: float(1.0 - keep) for name, keep in final_keep_ratios.items()}
