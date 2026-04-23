from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluation.metrics import CLASS_NAMES, dangerous_class_degradation_ratio, evaluate_model
from experiments.common import (
    build_dataloaders,
    load_trained_model,
    model_alias,
    resolve_calibration_path,
)
from models.load_models import get_linear_layer_names
from pruning.masking import apply_masks, compute_global_masks
from pruning.learnable_sparsity import LearnableSparsityConfig, learn_sparsity_allocation
from pruning.nonuniform import (
    allocate_sparsity,
    allocate_sparsity_continuous,
    allocate_sparsity_obs_like,
    apply_nonuniform_pruning,
    compute_layer_sensitivity,
    sweep_bin_counts,
)
from pruning.scoring import magnitude_score, wanda_score
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.io import append_csv_row, ensure_dir, save_masks
from utils.seed import set_seed


def _build_scores(model, activation_norms, exclude_layers):
    scores_mag: dict[str, torch.Tensor] = {}
    scores_wanda: dict[str, torch.Tensor] = {}
    target_layers = get_linear_layer_names(model, exclude_keywords=exclude_layers)
    for layer_name, layer in target_layers:
        weight = layer.weight.detach()
        scores_mag[layer_name] = magnitude_score(weight)
        scores_wanda[layer_name] = wanda_score(weight, activation_norms[layer_name])
    return target_layers, scores_mag, scores_wanda


def _allocate_all_policies(
    sensitivity: dict[str, float],
    target_avg: float,
    layer_param_counts: dict[str, int],
    config: dict,
    weights: dict[str, torch.Tensor] | None = None,
    activation_norms: dict[str, torch.Tensor] | None = None,
    learnable_allocation: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Build every allocation policy in one place so downstream loops stay
    compact. Policies are keyed by a stable string name that also shows
    up in the output CSV.
    """
    nonuniform_cfg = config.get("nonuniform", {})
    policies: dict[str, dict[str, float]] = {}
    bin_counts = nonuniform_cfg.get("bin_counts", [3])
    for k, alloc in sweep_bin_counts(
        sensitivity=sensitivity,
        target_avg=target_avg,
        bin_counts=bin_counts,
        layer_param_counts=layer_param_counts,
    ).items():
        policies[f"binned_k{k}"] = alloc

    for temperature in nonuniform_cfg.get("continuous_temperatures", [1.0]):
        name = f"continuous_t{temperature:g}"
        policies[name] = allocate_sparsity_continuous(
            sensitivity=sensitivity,
            target_avg=target_avg,
            temperature=float(temperature),
            layer_param_counts=layer_param_counts,
        )

    if nonuniform_cfg.get("obs_like", False) and weights is not None and activation_norms is not None:
        policies["obs_like"] = allocate_sparsity_obs_like(
            hessian_diag={k: v.pow(2) for k, v in activation_norms.items()},
            weight_by_layer=weights,
            target_avg=target_avg,
            layer_param_counts=layer_param_counts,
        )

    # Always include the original 3-bin default as a reference.
    policies["binned_default"] = allocate_sparsity(
        sensitivity=sensitivity,
        target_avg=target_avg,
        bins=nonuniform_cfg.get("bins", {}),
        layer_param_counts=layer_param_counts,
        num_bins=3,
    )

    if learnable_allocation is not None:
        policies["learnable"] = learnable_allocation

    return policies


def run(config_path: str, seed_override: int | None = None) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    set_seed(seed)
    device = get_device()

    _, val_loader, calibration_loader, class_weights = build_dataloaders(
        config,
        include_train=False,
        calibration_size=int(config["pruning"].get("calibration_size", 128)),
    )
    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    results_dir = ensure_dir(config["logging"]["results_dir"])
    model_name = config["models"]["teacher"]
    alias = model_alias(model_name)
    model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
    baseline = evaluate_model(model, val_loader, device, class_names=CLASS_NAMES)
    baseline_sensitivity = dict(baseline["per_class_sensitivity"])

    activation_norms = torch.load(
        resolve_calibration_path(config, f"{alias}_activation_norms.pt"),
        map_location="cpu",
    )
    target_layers, scores_mag, scores_wanda = _build_scores(
        model, activation_norms, config["pruning"]["exclude_layers"]
    )
    layer_names = [name for name, _ in target_layers]
    sensitivity = compute_layer_sensitivity(
        model=model,
        val_loader=val_loader,
        criterion=None,
        device=device,
        scores_dict=scores_mag,
        layer_names=layer_names,
    )

    sensitivity_path = Path(results_dir) / "layer_sensitivity_scores.csv"
    allocation_path = Path(results_dir) / "nonuniform_allocation.csv"
    summary_path = Path(results_dir) / "nonuniform_pruning.csv"
    masks_dir = ensure_dir(checkpoint_dir / "masks" / "nonuniform")
    layer_param_counts = {name: score.numel() for name, score in scores_mag.items()}

    weights_by_layer = {name: layer.weight.detach() for name, layer in target_layers}

    learnable_allocation: dict[str, float] | None = None
    if bool(config.get("nonuniform", {}).get("learnable", False)) and calibration_loader is not None:
        learnable_cfg = LearnableSparsityConfig(
            target_avg_sparsity=float(config["nonuniform"].get("target_avg_sparsity", 0.5)),
            steps=int(config["nonuniform"].get("learnable_steps", 200)),
            lr=float(config["nonuniform"].get("learnable_lr", 5e-2)),
            beta=float(config["nonuniform"].get("learnable_beta", 10.0)),
        )
        learnable_allocation = learn_sparsity_allocation(
            model=model,
            scores=scores_mag,
            calibration_loader=calibration_loader,
            device=device,
            config=learnable_cfg,
            class_weights=class_weights,
            layer_param_counts=layer_param_counts,
        )

    policies = _allocate_all_policies(
        sensitivity=sensitivity,
        target_avg=float(config["nonuniform"].get("target_avg_sparsity", 0.5)),
        layer_param_counts=layer_param_counts,
        config=config,
        weights=weights_by_layer,
        activation_norms=activation_norms,
        learnable_allocation=learnable_allocation,
    )

    for layer_name in layer_names:
        row = {
            "seed": seed,
            "model": alias,
            "layer_name": layer_name,
            "sensitivity_drop": sensitivity[layer_name],
            "param_count": layer_param_counts[layer_name],
        }
        for policy_name, alloc in policies.items():
            row[f"allocation_{policy_name}"] = alloc.get(layer_name, 0.0)
        append_csv_row(sensitivity_path, row)

    target_avg = float(config["nonuniform"].get("target_avg_sparsity", 0.5))
    for criterion_name, scores in (("magnitude", scores_mag), ("wanda", scores_wanda)):
        uniform_model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
        uniform_masks = compute_global_masks(uniform_model, scores, target_avg)
        apply_masks(uniform_model, uniform_masks)
        uniform_metrics = evaluate_model(uniform_model, val_loader, device, class_names=CLASS_NAMES)
        save_masks(masks_dir / f"{alias}_{criterion_name}_uniform.pt", uniform_masks)

        for condition, metrics in (("dense", baseline), ("uniform", uniform_metrics)):
            append_csv_row(
                allocation_path,
                {
                    "seed": seed,
                    "model": alias,
                    "criterion": criterion_name,
                    "policy": condition,
                    "target_avg_sparsity": target_avg,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_auroc": metrics.get("macro_auroc"),
                    "melanoma_auroc": metrics.get("melanoma_auroc"),
                    "ece_top_label": metrics.get("ece_top_label"),
                    "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
                    "bcc_sensitivity": metrics["per_class_sensitivity"]["bcc"],
                    "akiec_sensitivity": metrics["per_class_sensitivity"]["akiec"],
                    "nv_sensitivity": metrics["per_class_sensitivity"]["nv"],
                    "bkl_sensitivity": metrics["per_class_sensitivity"]["bkl"],
                    "df_sensitivity": metrics["per_class_sensitivity"]["df"],
                    "vasc_sensitivity": metrics["per_class_sensitivity"]["vasc"],
                    "dangerous_class_degradation_ratio": (
                        dangerous_class_degradation_ratio(baseline_sensitivity, metrics["per_class_sensitivity"])
                        if condition != "dense" else 0.0
                    ),
                },
            )
            append_csv_row(
                summary_path,
                {
                    "seed": seed,
                    "model": alias,
                    "criterion": criterion_name,
                    "policy": condition,
                    "target_avg_sparsity": target_avg,
                    "balanced_acc": metrics["balanced_accuracy"],
                    "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
                    "macro_auroc": metrics.get("macro_auroc"),
                    "melanoma_auroc": metrics.get("melanoma_auroc"),
                    "ece_top_label": metrics.get("ece_top_label"),
                    "dangerous_class_degradation_ratio": (
                        dangerous_class_degradation_ratio(baseline_sensitivity, metrics["per_class_sensitivity"])
                        if condition != "dense" else 0.0
                    ),
                },
            )

        for policy_name, allocation in policies.items():
            pruned_model = load_trained_model(
                config, model_name, device, checkpoint_name=f"{alias}_ham10000"
            )
            nonuniform_masks = apply_nonuniform_pruning(pruned_model, scores, allocation)
            metrics = evaluate_model(pruned_model, val_loader, device, class_names=CLASS_NAMES)
            save_masks(masks_dir / f"{alias}_{criterion_name}_{policy_name}.pt", nonuniform_masks)
            dcr = dangerous_class_degradation_ratio(
                baseline_sensitivity, metrics["per_class_sensitivity"]
            )
            append_csv_row(
                allocation_path,
                {
                    "seed": seed,
                    "model": alias,
                    "criterion": criterion_name,
                    "policy": policy_name,
                    "target_avg_sparsity": target_avg,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_auroc": metrics.get("macro_auroc"),
                    "melanoma_auroc": metrics.get("melanoma_auroc"),
                    "ece_top_label": metrics.get("ece_top_label"),
                    "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
                    "bcc_sensitivity": metrics["per_class_sensitivity"]["bcc"],
                    "akiec_sensitivity": metrics["per_class_sensitivity"]["akiec"],
                    "nv_sensitivity": metrics["per_class_sensitivity"]["nv"],
                    "bkl_sensitivity": metrics["per_class_sensitivity"]["bkl"],
                    "df_sensitivity": metrics["per_class_sensitivity"]["df"],
                    "vasc_sensitivity": metrics["per_class_sensitivity"]["vasc"],
                    "dangerous_class_degradation_ratio": dcr,
                },
            )
            append_csv_row(
                summary_path,
                {
                    "seed": seed,
                    "model": alias,
                    "criterion": criterion_name,
                    "policy": policy_name,
                    "target_avg_sparsity": target_avg,
                    "balanced_acc": metrics["balanced_accuracy"],
                    "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
                    "macro_auroc": metrics.get("macro_auroc"),
                    "melanoma_auroc": metrics.get("melanoma_auroc"),
                    "ece_top_label": metrics.get("ece_top_label"),
                    "dangerous_class_degradation_ratio": dcr,
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-uniform pruning allocation + W11 policy sweep.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, seed_override=args.seed)


if __name__ == "__main__":
    main()
