from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluation.metrics import CLASS_NAMES, evaluate_model
from experiments.common import build_dataloaders, load_trained_model, model_alias
from models.load_models import get_linear_layer_names
from pruning.masking import apply_masks, compute_global_masks
from pruning.nonuniform import allocate_sparsity, apply_nonuniform_pruning, compute_layer_sensitivity
from pruning.scoring import magnitude_score, wanda_score
from utils.config import get_device, load_config
from utils.io import append_csv_row, ensure_dir, save_masks


def _build_scores(model, activation_norms, exclude_layers):
    scores_mag: dict[str, torch.Tensor] = {}
    scores_wanda: dict[str, torch.Tensor] = {}
    target_layers = get_linear_layer_names(model, exclude_keywords=exclude_layers)
    for layer_name, layer in target_layers:
        weight = layer.weight.detach()
        scores_mag[layer_name] = magnitude_score(weight)
        scores_wanda[layer_name] = wanda_score(weight, activation_norms[layer_name])
    return target_layers, scores_mag, scores_wanda


def run(config_path: str) -> None:
    config = load_config(config_path)
    device = get_device()
    _, val_loader, _, _ = build_dataloaders(config, include_train=False)
    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    results_dir = ensure_dir(config["logging"]["results_dir"])
    model_name = config["models"]["teacher"]
    alias = model_alias(model_name)
    model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
    baseline = evaluate_model(model, val_loader, device, class_names=CLASS_NAMES)

    activation_norms = torch.load(checkpoint_dir / "calibration" / f"{alias}_activation_norms.pt", map_location="cpu")
    target_layers, scores_mag, scores_wanda = _build_scores(model, activation_norms, config["pruning"]["exclude_layers"])
    layer_names = [name for name, _ in target_layers]
    sensitivity = compute_layer_sensitivity(
        model,
        val_loader,
        criterion=None,
        device=device,
        scores_dict=scores_mag,
        layer_names=layer_names,
    )

    sensitivity_path = Path(results_dir) / "layer_sensitivity_scores.csv"
    allocation_path = Path(results_dir) / "nonuniform_allocation.csv"
    masks_dir = ensure_dir(checkpoint_dir / "masks" / "nonuniform")
    layer_param_counts = {name: score.numel() for name, score in scores_mag.items()}
    allocation = allocate_sparsity(
        sensitivity,
        target_avg=float(config["nonuniform"].get("target_avg_sparsity", 0.5)),
        bins=config["nonuniform"].get("bins", {}),
        layer_param_counts=layer_param_counts,
    )

    for layer_name in layer_names:
        append_csv_row(
            sensitivity_path,
            {
                "model": alias,
                "layer_name": layer_name,
                "sensitivity_drop": sensitivity[layer_name],
                "allocated_sparsity": allocation[layer_name],
                "param_count": layer_param_counts[layer_name],
            },
        )

    for criterion_name, scores in (("magnitude", scores_mag), ("wanda", scores_wanda)):
        uniform_model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
        uniform_masks = compute_global_masks(
            uniform_model,
            scores,
            float(config["nonuniform"].get("target_avg_sparsity", 0.5)),
        )
        apply_masks(uniform_model, uniform_masks)
        uniform_metrics = evaluate_model(uniform_model, val_loader, device, class_names=CLASS_NAMES)
        save_masks(masks_dir / f"{alias}_{criterion_name}_uniform.pt", uniform_masks)

        nonuniform_model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
        nonuniform_masks = apply_nonuniform_pruning(nonuniform_model, scores, allocation)
        nonuniform_metrics = evaluate_model(nonuniform_model, val_loader, device, class_names=CLASS_NAMES)
        save_masks(masks_dir / f"{alias}_{criterion_name}_nonuniform.pt", nonuniform_masks)

        for condition, metrics in (("dense", baseline), ("uniform", uniform_metrics), ("nonuniform", nonuniform_metrics)):
            append_csv_row(
                allocation_path,
                {
                    "model": alias,
                    "criterion": criterion_name,
                    "condition": condition,
                    "target_avg_sparsity": float(config["nonuniform"].get("target_avg_sparsity", 0.5)),
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
                    "bcc_sensitivity": metrics["per_class_sensitivity"]["bcc"],
                    "akiec_sensitivity": metrics["per_class_sensitivity"]["akiec"],
                    "nv_sensitivity": metrics["per_class_sensitivity"]["nv"],
                    "bkl_sensitivity": metrics["per_class_sensitivity"]["bkl"],
                    "df_sensitivity": metrics["per_class_sensitivity"]["df"],
                    "vasc_sensitivity": metrics["per_class_sensitivity"]["vasc"],
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-uniform pruning allocation experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
