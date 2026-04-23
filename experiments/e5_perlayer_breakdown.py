from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from evaluation.metrics import CLASS_NAMES, evaluate_model
from experiments.common import (
    build_dataloaders,
    load_trained_model,
    model_alias,
    resolve_calibration_path,
)
from pruning.layer_groups import get_layer_groups, prune_only_group
from pruning.scoring import magnitude_score, wanda_score
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.io import append_csv_row
from utils.seed import set_seed


def run(config_path: str, seed_override: int | None = None) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    set_seed(resolve_seed(config))
    device = get_device()
    _, val_loader, _, _ = build_dataloaders(config, include_train=False)
    log_path = Path(config["logging"]["results_dir"]) / "perlayer_breakdown.csv"

    model_name = config["models"]["teacher"]
    alias = model_alias(model_name)
    base_model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
    baseline = evaluate_model(
        base_model,
        val_loader,
        device,
        class_names=CLASS_NAMES,
        progress_desc=f"{alias} per-layer baseline",
    )
    activation_norms = torch.load(
        resolve_calibration_path(config, f"{alias}_activation_norms.pt"),
        map_location="cpu",
    )
    layer_groups = get_layer_groups(base_model)

    criteria = config.get("smoke", {}).get("e5_criteria", ["wanda", "magnitude"])
    groups = config.get("smoke", {}).get("e5_groups", ["qkv", "attn_out", "mlp", "patch_embed"])

    for criterion_name in tqdm(criteria, desc=f"{alias} per-layer criteria", leave=False):
        scores: dict[str, torch.Tensor] = {}
        for layer_name, layer in base_model.named_modules():
            if not isinstance(layer, torch.nn.Linear):
                continue
            if layer_name not in sum(layer_groups.values(), []):
                continue
            weight = layer.weight.detach()
            if criterion_name == "wanda":
                scores[layer_name] = wanda_score(weight, activation_norms[layer_name])
            else:
                scores[layer_name] = magnitude_score(weight)

        for group_name in tqdm(groups, desc=f"{alias} {criterion_name} groups", leave=False):
            model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
            masks = prune_only_group(model, scores, sparsity=0.5, group_name=group_name, layer_groups=layer_groups)
            metrics = evaluate_model(
                model,
                val_loader,
                device,
                class_names=CLASS_NAMES,
                progress_desc=f"{alias} {criterion_name} {group_name}",
            )

            append_csv_row(
                log_path,
                {
                    "model": alias,
                    "criterion": criterion_name,
                    "layer_type": group_name,
                    "sparsity": 0.5,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "balanced_accuracy_drop": baseline["balanced_accuracy"] - metrics["balanced_accuracy"],
                    "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
                    "bcc_sensitivity": metrics["per_class_sensitivity"]["bcc"],
                    "akiec_sensitivity": metrics["per_class_sensitivity"]["akiec"],
                    "nv_sensitivity": metrics["per_class_sensitivity"]["nv"],
                    "bkl_sensitivity": metrics["per_class_sensitivity"]["bkl"],
                    "df_sensitivity": metrics["per_class_sensitivity"]["df"],
                    "vasc_sensitivity": metrics["per_class_sensitivity"]["vasc"],
                    "masked_params": sum(mask.numel() for mask in masks.values()),
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-layer-type pruning breakdown.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, seed_override=args.seed)


if __name__ == "__main__":
    main()
