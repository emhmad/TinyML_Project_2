"""
2:4 (and general N:M) structured sparsity experiment (W8 follow-through).

Unstructured pruning in dense storage is honest *only* about parameter
count, not about real file size or inference throughput. N:M sparsity
is hardware-addressable on Ampere and later GPUs and compresses on disk
with any sparse-tensor-aware serialiser. Running the core criterion
comparison under a 2:4 mask gives a like-for-like, actually deployable
variant to report alongside the unstructured tables.

Output schema: matches `pruning_matrix.csv` plus columns
`pattern_n`, `pattern_m`, and `verified` (the post-mask pattern check).
"""
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
from experiments.common import build_dataloaders, model_alias
from experiments.e4_pruning_matrix import _score_layers
from models.load_models import get_linear_layer_names, load_deit_model
from pruning.masking import apply_masks, get_sparsity_stats
from pruning.structured import NMPattern, compute_nm_masks, verify_nm_pattern
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.io import append_csv_row, ensure_dir, load_checkpoint_state, save_masks
from utils.seed import set_seed


def _default_patterns() -> list[NMPattern]:
    # 2:4 is the Ampere-native pattern; 1:4 (25% dense) is an aggressive
    # comparison point; 4:8 is accepted by some newer kernels as an
    # equivalent layout. Ordered from least to most aggressive.
    return [NMPattern(2, 4), NMPattern(4, 8), NMPattern(1, 4)]


def _row(
    *,
    seed: int,
    model: str,
    criterion: str,
    pattern: NMPattern,
    metrics: dict,
    sizes: dict,
    baseline_sensitivity: dict[str, float] | None,
    verified: bool,
) -> dict:
    row = {
        "seed": seed,
        "model": model,
        "criterion": criterion,
        "pattern_n": pattern.n_keep,
        "pattern_m": pattern.m_group,
        "sparsity": pattern.effective_sparsity,
        "verified": int(verified),
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
        "total_params": sizes["total_params"],
        "nonzero_params": sizes["nonzero_params"],
        "size_kb": sizes["size_kb"],
        "disk_size_kb": sizes["disk_size_kb"],
        "effective_sparse_size_kb": sizes["effective_sparse_size_kb"],
    }
    if baseline_sensitivity is not None:
        row["dangerous_class_degradation_ratio"] = dangerous_class_degradation_ratio(
            baseline_sensitivity, metrics["per_class_sensitivity"]
        )
    return row


def run(
    config_path: str,
    model_names: list[str] | None = None,
    seed_override: int | None = None,
    patterns: list[NMPattern] | None = None,
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
    masks_dir = ensure_dir(checkpoint_dir / "masks" / "structured")
    calibration_dir = checkpoint_dir / "calibration"
    log_path = results_dir / "structured_sparsity.csv"

    model_names = model_names or [config["models"]["student"], config["models"]["teacher"]]
    patterns = patterns or _default_patterns()
    criteria = criteria or list(config["pruning"].get("structured_criteria", ["magnitude", "wanda", "taylor"]))

    for model_name in model_names:
        alias = model_alias(model_name)
        base_model = load_deit_model(
            model_name=model_name,
            num_classes=int(config["models"].get("num_classes", 7)),
            pretrained=False,
        ).to(device)
        state_dict = load_checkpoint_state(checkpoint_dir / f"{alias}_ham10000.pth", map_location=device)
        base_model.load_state_dict(state_dict)
        target_layers = get_linear_layer_names(base_model, exclude_keywords=config["pruning"]["exclude_layers"])

        activation_norms = torch.load(calibration_dir / f"{alias}_activation_norms.pt", map_location="cpu")
        gradients = torch.load(calibration_dir / f"{alias}_gradients.pt", map_location="cpu")

        baseline = evaluate_model(base_model, val_loader, device, class_names=CLASS_NAMES)
        baseline_sensitivity = dict(baseline["per_class_sensitivity"])

        for criterion_name in tqdm(criteria, desc=f"{alias} structured", leave=False):
            scores = _score_layers(
                target_layers,
                criterion_name,
                activation_norms=activation_norms,
                gradients=gradients,
                seed=seed,
            )
            for pattern in patterns:
                model = load_deit_model(
                    model_name=model_name,
                    num_classes=int(config["models"].get("num_classes", 7)),
                    pretrained=False,
                ).to(device)
                model.load_state_dict(state_dict)
                masks = compute_nm_masks(model, scores, pattern)
                apply_masks(model, masks)

                verified = all(verify_nm_pattern(mask, pattern) for mask in masks.values())
                metrics = evaluate_model(model, val_loader, device, class_names=CLASS_NAMES)
                stats = get_sparsity_stats(model, masks)
                sizes = get_model_size_kb(model, sparse=True)
                sizes["total_params"] = stats["total_params"]
                sizes["nonzero_params"] = stats["nonzero_params"]

                append_csv_row(
                    log_path,
                    _row(
                        seed=seed,
                        model=alias,
                        criterion=criterion_name,
                        pattern=pattern,
                        metrics=metrics,
                        sizes=sizes,
                        baseline_sensitivity=baseline_sensitivity,
                        verified=verified,
                    ),
                )
                save_masks(
                    masks_dir / f"{alias}_{criterion_name}_nm{pattern.n_keep}x{pattern.m_group}.pt",
                    masks,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured N:M sparsity sweep (W8).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--criteria", nargs="*", default=None)
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        help="Patterns as N:M strings, e.g. '2:4 1:4 4:8'.",
    )
    return parser.parse_args()


def _parse_patterns(raw: list[str] | None) -> list[NMPattern] | None:
    if not raw:
        return None
    out: list[NMPattern] = []
    for entry in raw:
        n_str, m_str = entry.split(":")
        out.append(NMPattern(int(n_str), int(m_str)))
    return out


def main() -> None:
    args = parse_args()
    run(
        args.config,
        model_names=args.models,
        seed_override=args.seed,
        patterns=_parse_patterns(args.patterns),
        criteria=args.criteria,
    )


if __name__ == "__main__":
    main()
