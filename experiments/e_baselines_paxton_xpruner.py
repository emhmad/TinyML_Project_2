"""
External pruning baselines (W6).

Runs Paxton-style skewness-guided pruning and X-Pruner-style
explainability pruning on the same (DeiT-Tiny, DeiT-Small) checkpoints
and validation split as the main pruning matrix, so the non-uniform
+ sensitivity method in the paper has a real comparison point.

Output: results/logs/.../baselines_paxton_xpruner.csv, with the same
row schema as the pruning matrix (plus baseline_sensitivity-derived
dangerous-class degradation ratio).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from evaluation.metrics import CLASS_NAMES, dangerous_class_degradation_ratio, evaluate_model
from evaluation.model_size import get_model_size_kb
from experiments.common import (
    build_dataloaders,
    model_alias,
    resolve_calibration_path,
    resolve_checkpoint_path,
)
from experiments.e4_pruning_matrix import _row_from_metrics, _score_layers
from models.load_models import get_linear_layer_names, load_deit_model
from pruning.masking import apply_masks, compute_global_masks, get_sparsity_stats
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.io import append_csv_row, ensure_dir, load_checkpoint_state, save_masks
from utils.seed import set_seed


BASELINE_CRITERIA = ("skewness", "xpruner", "sparsegpt_pseudo")


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

    _, val_loader, _, _ = build_dataloaders(config, include_train=False)

    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    results_dir = ensure_dir(config["logging"]["results_dir"])
    masks_dir = ensure_dir(checkpoint_dir / "masks" / "baselines")
    log_path = results_dir / "baselines_paxton_xpruner.csv"

    model_names = model_names or [config["models"]["student"], config["models"]["teacher"]]
    sparsities = sparsities or config["pruning"]["sparsities"]
    criteria = list(criteria or BASELINE_CRITERIA)

    for model_name in model_names:
        alias = model_alias(model_name)
        base_model = load_deit_model(
            model_name=model_name,
            num_classes=int(config["models"].get("num_classes", 7)),
            pretrained=False,
        ).to(device)
        state_dict = load_checkpoint_state(
            resolve_checkpoint_path(config, f"{alias}_ham10000.pth"),
            map_location=device,
        )
        base_model.load_state_dict(state_dict)
        target_layers = get_linear_layer_names(base_model, exclude_keywords=config["pruning"]["exclude_layers"])

        activation_norms = torch.load(
            resolve_calibration_path(config, f"{alias}_activation_norms.pt"),
            map_location="cpu",
        )
        gradients = torch.load(
            resolve_calibration_path(config, f"{alias}_gradients.pt"),
            map_location="cpu",
        )

        baseline_results = evaluate_model(
            base_model, val_loader, device, class_names=CLASS_NAMES,
            progress_desc=f"{alias} dense"
        )
        baseline_sensitivity = dict(baseline_results["per_class_sensitivity"])

        for criterion_name in tqdm(criteria, desc=f"{alias} baselines", leave=False):
            scores = _score_layers(
                target_layers,
                criterion_name,
                activation_norms=activation_norms,
                gradients=gradients,
                seed=seed,
            )
            for sparsity in tqdm(sparsities, desc=f"{alias} {criterion_name}", leave=False):
                model = load_deit_model(
                    model_name=model_name,
                    num_classes=int(config["models"].get("num_classes", 7)),
                    pretrained=False,
                ).to(device)
                model.load_state_dict(state_dict)
                masks = compute_global_masks(model, scores, float(sparsity))
                apply_masks(model, masks)

                metrics = evaluate_model(
                    model,
                    val_loader,
                    device,
                    class_names=CLASS_NAMES,
                    progress_desc=f"{alias} {criterion_name} s={sparsity}",
                )
                stats = get_sparsity_stats(model, masks)
                sizes = get_model_size_kb(model, sparse=True)
                sizes["total_params"] = stats["total_params"]
                sizes["nonzero_params"] = stats["nonzero_params"]
                append_csv_row(
                    log_path,
                    _row_from_metrics(
                        seed=seed,
                        model=alias,
                        criterion=criterion_name,
                        sparsity=float(sparsity),
                        metrics=metrics,
                        sizes=sizes,
                        baseline_sensitivity=baseline_sensitivity,
                    ),
                )
                save_masks(masks_dir / f"{alias}_{criterion_name}_s{sparsity:.1f}.pt", masks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paxton + X-Pruner baselines (W6).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sparsities", nargs="*", type=float, default=None)
    parser.add_argument("--criteria", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.config,
        args.models,
        seed_override=args.seed,
        sparsities=args.sparsities,
        criteria=args.criteria,
    )


if __name__ == "__main__":
    main()
