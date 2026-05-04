"""
Per-class gradient-flow analysis (P1.1).

Mechanism probe for the DCR inversion. The activation-outlier hypothesis
(W4, e_activation_stats.py) was falsified — kurtosis / top-5% / outlier
ratio at the layer level showed |r| < 0.11 with Wanda damage. That probe
was *unconditional*. The DCR effect lives on a *conditional* variable:
sensitivity drop on dangerous classes vs safe classes. So this script
asks the right conditional question.

For each (model, criterion, sparsity) configuration, it:

1. Loads the dense fine-tuned model and the corresponding pruning mask.
2. For each class c in {dangerous, safe}, computes
       g_c[layer] = mean_{x in val | y == c}  ‖∂L_c(x) / ∂W_layer‖
   on a fixed val batch — i.e. how much "useful gradient" each layer
   carries for class c on average.
3. For each pruning criterion, derives the score that would have been
   used to choose which weights to prune, and measures whether weights
   in high-g_dangerous layers were *over-pruned* relative to weights in
   high-g_safe layers.

Output CSV columns:
    model, seed, criterion, sparsity, layer, class_group,
    grad_norm, weights_pruned_in_layer, weights_total_in_layer,
    layer_pruning_rate, dangerous_safe_ratio,
    pearson_grad_vs_pruning_rate, spearman_grad_vs_pruning_rate

A positive Pearson/Spearman value with `class_group="dangerous"` and
the magnitude criterion means: layers carrying more dangerous-class
gradient mass are precisely the ones magnitude prunes most. That's the
mechanism for the DCR inversion.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from data.dataset import CLASS_NAMES
from evaluation.metrics import DANGEROUS_CLASSES, SAFE_CLASSES
from experiments.common import (
    build_dataloaders,
    load_trained_model,
    model_alias,
    resolve_calibration_path,
    resolve_checkpoint_path,
)
from models.load_models import get_linear_layer_names
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.io import ensure_dir
from utils.seed import set_seed


def _class_indices(class_names: list[str], group: Iterable[str]) -> list[int]:
    return [class_names.index(c) for c in group if c in class_names]


def _per_class_gradient_norms(
    model: nn.Module,
    target_layers: list[tuple[str, nn.Module]],
    batch: tuple[torch.Tensor, torch.Tensor],
    class_indices: list[int],
    device: torch.device,
) -> dict[str, float]:
    """
    Compute mean ‖∂L_c/∂W_layer‖_F for each layer, where L_c is the
    cross-entropy loss restricted to samples whose label is in
    `class_indices`. Returns a dict layer_name -> grad norm.

    If no samples in the batch belong to `class_indices`, returns NaNs.
    """
    images, labels = batch
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in class_indices:
        mask |= labels == idx
    if not mask.any():
        return {name: float("nan") for name, _ in target_layers}

    images = images[mask].to(device, non_blocking=True)
    labels = labels[mask].to(device, non_blocking=True)

    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss = F.cross_entropy(logits.float(), labels)
    loss.backward()

    out: dict[str, float] = {}
    for name, layer in target_layers:
        if layer.weight.grad is None:
            out[name] = float("nan")
        else:
            out[name] = float(layer.weight.grad.detach().norm().item())
    model.zero_grad(set_to_none=True)
    return out


def _layer_pruning_rate_from_mask(
    mask_state: dict[str, torch.Tensor],
    target_layers: list[tuple[str, nn.Module]],
) -> dict[str, tuple[int, int]]:
    """
    Returns dict layer_name -> (pruned_count, total_count) from a saved
    mask file. Mask values are 1 for kept, 0 for pruned (matching the
    convention in pruning/masking.py).
    """
    out: dict[str, tuple[int, int]] = {}
    for name, layer in target_layers:
        # save_masks stores per-layer .weight masks keyed by layer name.
        if name in mask_state:
            mask = mask_state[name]
        elif f"{name}.weight" in mask_state:
            mask = mask_state[f"{name}.weight"]
        else:
            out[name] = (0, layer.weight.numel())
            continue
        pruned = int((mask == 0).sum().item())
        out[name] = (pruned, int(mask.numel()))
    return out


def _correlate(values_x: list[float], values_y: list[float]) -> tuple[float, float, float, float, int]:
    """Pearson r, Pearson p, Spearman rho, Spearman p, n."""
    arr_x = np.asarray(values_x, dtype=np.float64)
    arr_y = np.asarray(values_y, dtype=np.float64)
    keep = np.isfinite(arr_x) & np.isfinite(arr_y)
    arr_x = arr_x[keep]
    arr_y = arr_y[keep]
    if len(arr_x) < 3 or arr_x.std() == 0 or arr_y.std() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), int(len(arr_x))
    pr, pp = pearsonr(arr_x, arr_y)
    sr, sp = spearmanr(arr_x, arr_y)
    return float(pr), float(pp), float(sr), float(sp), int(len(arr_x))


def run(
    config_path: str,
    model_names: list[str] | None = None,
    seed_override: int | None = None,
    sparsities: list[float] | None = None,
    criteria: list[str] | None = None,
) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    set_seed(seed)
    device = get_device()

    pruning_cfg = config["pruning"]
    sparsities = sparsities or list(pruning_cfg.get("sparsities", [0.5]))
    criteria = criteria or list(pruning_cfg.get("release_criteria") or pruning_cfg.get("criteria", ["magnitude", "wanda", "taylor"]))
    model_names = model_names or [config["models"]["teacher"], config["models"]["student"]]

    _, val_loader, _, _ = build_dataloaders(config, include_train=False)
    # Use a fixed first batch as the per-class gradient probe — keeps
    # this experiment deterministic across reruns and across criteria.
    fixed_batch = next(iter(val_loader))

    dangerous_idx = _class_indices(CLASS_NAMES, DANGEROUS_CLASSES)
    safe_idx = _class_indices(CLASS_NAMES, SAFE_CLASSES)

    results_dir = ensure_dir(config["logging"]["results_dir"])
    output_path = Path(results_dir) / "perclass_gradient_flow.csv"
    correlation_path = Path(results_dir) / "perclass_gradient_flow_correlation.csv"

    detail_rows: list[dict] = []
    correlation_rows: list[dict] = []

    for model_name in model_names:
        alias = model_alias(model_name)

        # Per-class gradient norms on the dense fine-tuned model — the
        # quantity each criterion *would* see if it were class-aware.
        model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
        target_layers = get_linear_layer_names(
            model, exclude_keywords=pruning_cfg["exclude_layers"]
        )

        # Make sure linear layer weights track grads even after the eval
        # state — load_trained_model puts the model in train-eval-mixed
        # mode, but requires_grad should be True by default.
        for _, layer in target_layers:
            layer.weight.requires_grad_(True)

        grad_dangerous = _per_class_gradient_norms(model, target_layers, fixed_batch, dangerous_idx, device)
        grad_safe = _per_class_gradient_norms(model, target_layers, fixed_batch, safe_idx, device)

        # Now, for each criterion × sparsity, look up the saved mask and
        # compute per-layer pruning rate; correlate against grad norms.
        for criterion in criteria:
            for sparsity in sparsities:
                mask_path = resolve_checkpoint_path(
                    config, f"masks/{alias}_{criterion}_s{sparsity:.1f}.pt"
                )
                if not mask_path.exists():
                    # Not all configs save masks for all criteria; skip cleanly.
                    continue
                mask_state = torch.load(mask_path, map_location="cpu")
                pruning_counts = _layer_pruning_rate_from_mask(mask_state, target_layers)

                grad_d_vals: list[float] = []
                grad_s_vals: list[float] = []
                rate_vals: list[float] = []
                for name, _ in target_layers:
                    pruned, total = pruning_counts.get(name, (0, 0))
                    rate = (pruned / total) if total > 0 else float("nan")
                    g_d = grad_dangerous.get(name, float("nan"))
                    g_s = grad_safe.get(name, float("nan"))
                    detail_rows.append({
                        "model": alias, "seed": seed, "criterion": criterion,
                        "sparsity": sparsity, "layer": name,
                        "grad_norm_dangerous": g_d,
                        "grad_norm_safe": g_s,
                        "dangerous_safe_ratio": (g_d / g_s) if (g_s and np.isfinite(g_s) and g_s > 0) else float("nan"),
                        "weights_pruned": pruned,
                        "weights_total": total,
                        "layer_pruning_rate": rate,
                    })
                    grad_d_vals.append(g_d)
                    grad_s_vals.append(g_s)
                    rate_vals.append(rate)

                # Correlation: does this criterion preferentially prune
                # layers carrying high dangerous-class gradient?
                pr_d, pp_d, sr_d, sp_d, n_d = _correlate(grad_d_vals, rate_vals)
                pr_s, pp_s, sr_s, sp_s, n_s = _correlate(grad_s_vals, rate_vals)
                correlation_rows.append({
                    "model": alias, "seed": seed, "criterion": criterion,
                    "sparsity": sparsity,
                    "pearson_grad_dangerous_vs_pruning_rate": pr_d,
                    "pearson_p_dangerous": pp_d,
                    "spearman_grad_dangerous_vs_pruning_rate": sr_d,
                    "spearman_p_dangerous": sp_d,
                    "pearson_grad_safe_vs_pruning_rate": pr_s,
                    "pearson_p_safe": pp_s,
                    "spearman_grad_safe_vs_pruning_rate": sr_s,
                    "spearman_p_safe": sp_s,
                    "n_layers": n_d,
                })

    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(output_path, index=False)
    if correlation_rows:
        pd.DataFrame(correlation_rows).to_csv(correlation_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-class gradient-flow mechanism probe (P1.1).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sparsities", type=float, nargs="*", default=None)
    parser.add_argument("--criteria", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.config,
        model_names=args.models,
        seed_override=args.seed,
        sparsities=args.sparsities,
        criteria=args.criteria,
    )


if __name__ == "__main__":
    main()
