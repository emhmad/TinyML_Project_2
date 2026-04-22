from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import numpy as np
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
    probe_sparsity: float = 0.5,
) -> dict[str, float]:
    """
    For each layer, prune only that layer at `probe_sparsity` and measure
    the balanced-accuracy drop. Larger drop => layer is more sensitive.
    """
    del criterion
    baseline_results = evaluate_model(model, val_loader, device, return_probs=False)
    baseline_balanced_acc = baseline_results["balanced_accuracy"]
    sensitivity: dict[str, float] = {}

    for layer_name in layer_names:
        copied_model = deepcopy(model).to(device)
        mask = compute_mask(scores_dict[layer_name], sparsity=probe_sparsity)
        apply_masks(copied_model, {layer_name: mask})
        pruned_results = evaluate_model(copied_model, val_loader, device, return_probs=False)
        sensitivity[layer_name] = baseline_balanced_acc - pruned_results["balanced_accuracy"]
        del copied_model

    return sensitivity


def allocate_sparsity(
    sensitivity: dict[str, float],
    target_avg: float = 0.5,
    bins: dict | None = None,
    layer_param_counts: dict[str, int] | None = None,
    num_bins: int = 3,
) -> dict[str, float]:
    """
    Binned allocation (the original 3-bin policy, now configurable via
    `num_bins` and `bins`). Kept as the default W11-ablation baseline.

    * `bins` maps a coarse sensitivity label to a sparsity target. For
      `num_bins=3` the expected keys are `low_sensitivity`,
      `medium_sensitivity`, `high_sensitivity`. For arbitrary bin counts,
      pass a list under `bins['rates']` instead, high-sensitivity first.
    """
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive; got {num_bins}")

    if bins is not None and "rates" in bins:
        rates = list(bins["rates"])
        if len(rates) != num_bins:
            raise ValueError(
                f"bins['rates'] has {len(rates)} entries but num_bins={num_bins}"
            )
    elif num_bins == 3:
        bins = bins or {"low_sensitivity": 0.7, "medium_sensitivity": 0.5, "high_sensitivity": 0.3}
        rates = [
            float(bins["high_sensitivity"]),
            float(bins["medium_sensitivity"]),
            float(bins["low_sensitivity"]),
        ]
    else:
        # Default: linearly interpolate sparsity from 0.3 (high-sens) to 0.7 (low-sens).
        rates = list(np.linspace(0.3, 0.7, num_bins))

    ordered = sorted(sensitivity.items(), key=lambda item: item[1], reverse=True)
    num_layers = len(ordered)
    if num_layers == 0:
        return {}

    # Distribute layers into `num_bins` contiguous chunks, high-sensitivity first.
    bin_indices = np.array_split(np.arange(num_layers), num_bins)
    allocation: dict[str, float] = {}
    for bin_id, idxs in enumerate(bin_indices):
        rate = rates[bin_id]
        for i in idxs:
            layer_name, _ = ordered[i]
            allocation[layer_name] = float(rate)

    return _rescale_to_target_average(allocation, target_avg, layer_param_counts)


def allocate_sparsity_continuous(
    sensitivity: dict[str, float],
    target_avg: float = 0.5,
    temperature: float = 1.0,
    layer_param_counts: dict[str, int] | None = None,
    min_sparsity: float = 0.1,
    max_sparsity: float = 0.9,
) -> dict[str, float]:
    """
    Continuous, sensitivity-weighted allocation (W11 alternative).
    Sparsity ∝ inverse-softened-sensitivity, rescaled so the weighted
    average matches `target_avg`. Gives every layer its own rate, not
    just a coarse bin.

    `temperature` controls how aggressively sensitive layers are protected:
    higher temperature => flatter allocation (closer to uniform), lower
    temperature => stronger protection of sensitive layers.
    """
    if not sensitivity:
        return {}
    names = list(sensitivity.keys())
    sens = np.array([sensitivity[n] for n in names], dtype=float)

    # Map sensitivity to a rank in [0, 1]: rank=0 most sensitive, rank=1 least.
    order = np.argsort(-sens)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(order))

    raw = min_sparsity + (max_sparsity - min_sparsity) * (ranks ** max(1e-6, temperature))
    allocation = {name: float(raw[i]) for i, name in enumerate(names)}
    return _rescale_to_target_average(
        allocation,
        target_avg,
        layer_param_counts,
        clip_min=min_sparsity,
        clip_max=max_sparsity,
    )


def allocate_sparsity_obs_like(
    hessian_diag: dict[str, torch.Tensor],
    weight_by_layer: dict[str, torch.Tensor],
    target_avg: float = 0.5,
    layer_param_counts: dict[str, int] | None = None,
    min_sparsity: float = 0.1,
    max_sparsity: float = 0.9,
) -> dict[str, float]:
    """
    OBS-style per-layer sparsity allocation (W11 alternative).

    Uses per-layer pruning "cost" E_ell = sum_j W[j]^2 / (H_ii + eps),
    where H_ii is the diagonal Hessian approximation (we reuse the
    Wanda-style activation second moment for H_ii when called from
    `e7_e10_nonuniform.py`). Layers with higher cost are protected
    (lower sparsity). The mapping is monotone in rank, then rescaled
    to hit `target_avg`.
    """
    eps = 1e-6
    costs: dict[str, float] = {}
    for name, weight in weight_by_layer.items():
        h = hessian_diag.get(name)
        if h is None:
            continue
        cost = (weight.detach().float().pow(2) / (h.to(weight.device).float().unsqueeze(0) + eps)).sum()
        costs[name] = float(cost.item())
    if not costs:
        return {}
    items = sorted(costs.items(), key=lambda kv: kv[1], reverse=True)
    ranks = {name: i / max(1, len(items) - 1) for i, (name, _) in enumerate(items)}
    allocation = {
        name: float(min_sparsity + (max_sparsity - min_sparsity) * rank)
        for name, rank in ranks.items()
    }
    return _rescale_to_target_average(
        allocation,
        target_avg,
        layer_param_counts,
        clip_min=min_sparsity,
        clip_max=max_sparsity,
    )


def _rescale_to_target_average(
    allocation: dict[str, float],
    target_avg: float,
    layer_param_counts: dict[str, int] | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> dict[str, float]:
    if not allocation:
        return {}
    names = list(allocation.keys())
    rates = np.array([allocation[n] for n in names], dtype=float)
    if layer_param_counts:
        weights = np.array([float(layer_param_counts.get(n, 1)) for n in names])
    else:
        weights = np.ones(len(names))
    weights = weights / max(1e-12, weights.sum())

    # Find a scalar c such that sum_i w_i * clip(r_i + c, [clip_min, clip_max]) = target_avg.
    def weighted_mean(c: float) -> float:
        return float(np.sum(weights * np.clip(rates + c, clip_min, clip_max)))

    lo = clip_min - rates.max() - 1.0
    hi = clip_max - rates.min() + 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if weighted_mean(mid) < target_avg:
            lo = mid
        else:
            hi = mid
    c_star = 0.5 * (lo + hi)
    final = np.clip(rates + c_star, clip_min, clip_max)
    return {name: float(final[i]) for i, name in enumerate(names)}


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


def sweep_bin_counts(
    sensitivity: dict[str, float],
    target_avg: float,
    bin_counts: Iterable[int] = (2, 3, 5),
    layer_param_counts: dict[str, int] | None = None,
) -> dict[int, dict[str, float]]:
    """
    Convenience helper for the bin-count ablation requested in W11.
    Returns an allocation per bin-count choice.
    """
    return {
        k: allocate_sparsity(
            sensitivity=sensitivity,
            target_avg=target_avg,
            layer_param_counts=layer_param_counts,
            num_bins=int(k),
        )
        for k in bin_counts
    }
