from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from evaluation.metrics import (
    CLASS_NAMES,
    dangerous_class_degradation_ratio,
    evaluate_model,
)
from evaluation.model_size import get_model_size_kb
from experiments.common import (
    build_dataloaders,
    model_alias,
    resolve_calibration_path,
    resolve_checkpoint_path,
)
from models.load_models import get_linear_layer_names, load_deit_model
from pruning.masking import apply_masks, compute_global_masks, get_sparsity_stats
from pruning.scoring import (
    magnitude_score,
    random_score,
    skewness_score,
    sparsegpt_pseudo_score,
    taylor_score,
    wanda_score,
    xpruner_score,
)
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.io import append_csv_row, ensure_dir, load_checkpoint_state, save_masks
from utils.seed import set_seed


def _score_layers(
    target_layers,
    criterion_name,
    *,
    activation_norms=None,
    gradients=None,
    seed: int = 42,
):
    """
    Build per-layer importance scores for the requested criterion.
    Extended vs. the original implementation to cover Paxton skewness,
    X-Pruner, and a SparseGPT-flavoured diagonal-OBS baseline (W6).
    """
    scores: dict[str, torch.Tensor] = {}
    for index, (layer_name, layer) in enumerate(target_layers):
        weight = layer.weight.detach()
        if criterion_name == "magnitude":
            scores[layer_name] = magnitude_score(weight)
        elif criterion_name == "wanda":
            scores[layer_name] = wanda_score(weight, activation_norms[layer_name])
        elif criterion_name == "taylor":
            scores[layer_name] = taylor_score(weight, gradients[layer_name].to(weight.device))
        elif criterion_name == "random":
            scores[layer_name] = random_score(weight, seed=seed + index)
        elif criterion_name == "skewness":
            scores[layer_name] = skewness_score(weight)
        elif criterion_name == "xpruner":
            grad = gradients[layer_name].to(weight.device)
            output_sensitivity = grad.abs().sum(dim=1)
            scores[layer_name] = xpruner_score(
                weight,
                activation_norms[layer_name],
                output_sensitivity,
            )
        elif criterion_name == "sparsegpt_pseudo":
            scores[layer_name] = sparsegpt_pseudo_score(weight, activation_norms[layer_name])
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")
    return scores


def _row_from_metrics(
    *,
    seed: int,
    model: str,
    criterion: str,
    sparsity: float,
    metrics: dict,
    sizes: dict,
    baseline_sensitivity: dict[str, float] | None,
) -> dict:
    row = {
        "seed": seed,
        "model": model,
        "criterion": criterion,
        "sparsity": float(sparsity),
        "overall_acc": metrics["overall_accuracy"],
        "balanced_acc": metrics["balanced_accuracy"],
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
        "mel_precision": metrics["per_class_precision"]["mel"],
        "bcc_precision": metrics["per_class_precision"]["bcc"],
        "akiec_precision": metrics["per_class_precision"]["akiec"],
        "mel_specificity_at_90_sens": (
            metrics.get("melanoma_operating_points", {})
            .get("mel_sensitivity@0.9", {})
            .get("specificity")
        ),
        "mel_sensitivity_at_90_spec": (
            metrics.get("melanoma_operating_points", {})
            .get("mel_specificity@0.9", {})
            .get("sensitivity")
        ),
        "total_params": sizes["total_params"],
        "nonzero_params": sizes["nonzero_params"],
        "size_kb": sizes["size_kb"],  # Legacy column, now == effective_sparse_size_kb when pruned.
        "disk_size_kb": sizes["disk_size_kb"],
        "effective_sparse_size_kb": sizes["effective_sparse_size_kb"],
        "dense_size_kb": sizes["dense_size_kb"],
    }
    if baseline_sensitivity is not None:
        row["dangerous_class_degradation_ratio"] = dangerous_class_degradation_ratio(
            baseline_sensitivity=baseline_sensitivity,
            pruned_sensitivity=metrics["per_class_sensitivity"],
        )
    else:
        row["dangerous_class_degradation_ratio"] = 0.0
    return row


def run(
    config_path: str,
    model_names: list[str] | None = None,
    seed_override: int | None = None,
    criteria_override: list[str] | None = None,
) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    set_seed(seed)
    device = get_device()

    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    results_dir = ensure_dir(config["logging"]["results_dir"])
    masks_dir = ensure_dir(checkpoint_dir / "masks")
    log_path = results_dir / "pruning_matrix.csv"
    _, val_loader, _, _ = build_dataloaders(config, include_train=False)
    model_names = model_names or [config["models"]["student"], config["models"]["teacher"]]
    criteria = criteria_override or config["pruning"]["criteria"]

    for model_name in model_names:
        alias = model_alias(model_name)
        base_model = load_deit_model(
            model_name=model_name,
            num_classes=int(config["models"].get("num_classes", 7)),
            pretrained=False,
        ).to(device)
        checkpoint_path = resolve_checkpoint_path(config, f"{alias}_ham10000.pth")
        state_dict = load_checkpoint_state(checkpoint_path, map_location=device)
        base_model.load_state_dict(state_dict)
        target_layers = get_linear_layer_names(
            base_model, exclude_keywords=config["pruning"]["exclude_layers"]
        )

        activation_norms = torch.load(
            resolve_calibration_path(config, f"{alias}_activation_norms.pt"),
            map_location="cpu",
        )
        gradients = torch.load(
            resolve_calibration_path(config, f"{alias}_gradients.pt"),
            map_location="cpu",
        )

        baseline_results = evaluate_model(
            base_model,
            val_loader,
            device,
            class_names=CLASS_NAMES,
            progress_desc=f"{alias} dense eval",
        )
        baseline_sensitivity = dict(baseline_results["per_class_sensitivity"])
        baseline_sizes = get_model_size_kb(base_model)
        append_csv_row(
            log_path,
            _row_from_metrics(
                seed=seed,
                model=alias,
                criterion="dense",
                sparsity=0.0,
                metrics=baseline_results,
                sizes=baseline_sizes,
                baseline_sensitivity=None,
            ),
        )

        for criterion_name in tqdm(criteria, desc=f"{alias} criteria", leave=False):
            scores = _score_layers(
                target_layers,
                criterion_name,
                activation_norms=activation_norms,
                gradients=gradients,
                seed=seed,
            )
            for sparsity in tqdm(
                config["pruning"]["sparsities"], desc=f"{alias} {criterion_name}", leave=False
            ):
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
                stats = get_sparsity_stats(model, masks)  # noqa: F841
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
                save_masks(
                    masks_dir / f"{alias}_{criterion_name}_s{sparsity:.1f}.pt", masks
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pruning matrix experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--criteria",
        nargs="*",
        default=None,
        help="Override pruning criteria list (default: config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.models, seed_override=args.seed, criteria_override=args.criteria)


if __name__ == "__main__":
    main()
