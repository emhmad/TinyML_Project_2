"""
Activation-statistics experiment (W4).

Computes per-layer kurtosis, top-k activation concentration, and outlier
ratio for every target linear layer of the distilled and direct models.
Then, for Wanda specifically, measures single-layer pruning damage (the
balanced-accuracy drop when only that layer is pruned at 50% sparsity)
and correlates the two.

This turns the paper's hypothesis ("distillation concentrates information
into outlier activations that Wanda over-protects") into evidence: a
positive correlation between layer-wise activation concentration and
Wanda-specific damage supports the claim; no correlation or a negative
one falsifies it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from evaluation.metrics import CLASS_NAMES, evaluate_model
from experiments.common import (
    build_dataloaders,
    load_trained_model,
    model_alias,
    resolve_calibration_path,
)
from models.load_models import get_linear_layer_names
from pruning.activation_stats import (
    compute_activation_stats,
    correlate_stats_with_damage,
    layerwise_wanda_damage,
    stats_to_frame,
)
from pruning.scoring import magnitude_score, wanda_score
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.io import ensure_dir
from utils.seed import set_seed


def _load_calibration_tensors(
    config: dict, alias: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    activation_path = resolve_calibration_path(config, f"{alias}_activation_norms.pt")
    if not activation_path.exists():
        raise FileNotFoundError(
            f"Missing calibration tensor {activation_path}. "
            "Run experiments.e3_calibration first for this seed (or the canonical seed if share_finetune_across_seeds is on)."
        )
    activation_norms = torch.load(activation_path, map_location="cpu")
    gradient_path = resolve_calibration_path(config, f"{alias}_gradients.pt")
    gradients = torch.load(gradient_path, map_location="cpu") if gradient_path.exists() else None
    return activation_norms, gradients


def run(
    config_path: str,
    model_names: list[str] | None = None,
    seed_override: int | None = None,
    probe_sparsity: float = 0.5,
) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    set_seed(seed)
    device = get_device()

    _, val_loader, _, _ = build_dataloaders(config, include_train=False)
    model_names = model_names or [config["models"]["teacher"], config["models"]["student"]]

    results_dir = ensure_dir(config["logging"]["results_dir"])
    stats_path = Path(results_dir) / "activation_stats.csv"
    correlation_path = Path(results_dir) / "activation_stats_correlation.csv"
    per_layer_damage_path = Path(results_dir) / "activation_stats_layerwise_damage.csv"

    all_stats_frames: list[pd.DataFrame] = []
    all_correlations: list[pd.DataFrame] = []
    all_damage_frames: list[pd.DataFrame] = []

    for model_name in model_names:
        alias = model_alias(model_name)
        model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
        target_layers = get_linear_layer_names(model, exclude_keywords=config["pruning"]["exclude_layers"])
        activation_norms, _ = _load_calibration_tensors(config, alias)

        stats = compute_activation_stats(
            activation_norms,
            layer_names=[name for name, _ in target_layers],
        )
        stats_frame = stats_to_frame(stats)
        stats_frame.insert(0, "model", alias)
        stats_frame.insert(1, "seed", seed)
        all_stats_frames.append(stats_frame)

        baseline = evaluate_model(model, val_loader, device, class_names=CLASS_NAMES, return_probs=False)
        baseline_balanced_acc = float(baseline["balanced_accuracy"])

        # Per-layer damage for Wanda and magnitude so the correlation can be
        # compared (W4 asks specifically about Wanda, but having magnitude
        # as a contrast makes the claim falsifiable).
        for criterion_name, score_fn in (("wanda", wanda_score), ("magnitude", None)):
            scores: dict[str, torch.Tensor] = {}
            for layer_name, layer in target_layers:
                weight = layer.weight.detach()
                if criterion_name == "wanda":
                    scores[layer_name] = wanda_score(weight, activation_norms[layer_name])
                else:
                    scores[layer_name] = magnitude_score(weight)

            damage = layerwise_wanda_damage(
                model=model,
                target_layers=target_layers,
                scores=scores,
                val_loader=val_loader,
                device=device,
                sparsity=probe_sparsity,
                baseline_balanced_acc=baseline_balanced_acc,
            )
            damage_frame = pd.DataFrame(
                [{"model": alias, "seed": seed, "criterion": criterion_name, "layer": layer, "damage": value}
                 for layer, value in damage.items()]
            )
            all_damage_frames.append(damage_frame)

            correlations = correlate_stats_with_damage(stats_frame, damage)
            correlations.insert(0, "model", alias)
            correlations.insert(1, "seed", seed)
            correlations.insert(2, "criterion", criterion_name)
            correlations.insert(3, "probe_sparsity", probe_sparsity)
            all_correlations.append(correlations)

    if all_stats_frames:
        pd.concat(all_stats_frames, ignore_index=True).to_csv(stats_path, index=False)
    if all_correlations:
        pd.concat(all_correlations, ignore_index=True).to_csv(correlation_path, index=False)
    if all_damage_frames:
        pd.concat(all_damage_frames, ignore_index=True).to_csv(per_layer_damage_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-layer activation statistics and Wanda-damage correlation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--probe-sparsity", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.models, seed_override=args.seed, probe_sparsity=args.probe_sparsity)


if __name__ == "__main__":
    main()
