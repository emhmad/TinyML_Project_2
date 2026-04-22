from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import HAM10000Dataset, get_train_val_splits, get_transforms
from evaluation.metrics import CLASS_NAMES, evaluate_model
from evaluation.model_size import get_model_size_kb
from models.load_models import get_linear_layer_names, load_deit_model
from pruning.masking import apply_masks, compute_global_masks, get_sparsity_stats
from pruning.scoring import magnitude_score, random_score, taylor_score, wanda_score
from utils.config import get_device, load_config, should_pin_memory
from utils.io import append_csv_row, ensure_dir, load_checkpoint_state, save_masks


def _build_loaders(config):
    dataset_cfg = config["dataset"]
    pin_memory = should_pin_memory()
    metadata_csv = dataset_cfg.get("metadata_csv") or str(Path(dataset_cfg["root"]) / "processed_metadata.csv")
    train_indices, val_indices = get_train_val_splits(
        metadata_csv,
        train_ratio=float(dataset_cfg.get("train_split", 0.8)),
        seed=int(dataset_cfg.get("seed", 42)),
    )
    max_train_samples = dataset_cfg.get("max_train_samples")
    max_val_samples = dataset_cfg.get("max_val_samples")
    if max_train_samples is not None:
        train_indices = train_indices[: int(max_train_samples)]
    if max_val_samples is not None:
        val_indices = val_indices[: int(max_val_samples)]

    train_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        transform=get_transforms(
            split="val",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(config["augmentation"].get("resize_size", 256)),
            augmentation_cfg=config["augmentation"],
        ),
        indices=train_indices,
    )
    val_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        transform=get_transforms(
            split="val",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(config["augmentation"].get("resize_size", 256)),
            augmentation_cfg=config["augmentation"],
        ),
        indices=val_indices,
    )

    calibration_size = int(config["pruning"].get("calibration_size", 128))
    calibration_dataset = Subset(train_dataset, list(range(min(calibration_size, len(train_dataset)))))
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=min(32, calibration_size),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 4)),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["evaluation"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 4)),
        pin_memory=pin_memory,
    )
    return calibration_loader, val_loader


def _score_layers(target_layers, criterion_name, activation_norms=None, gradients=None):
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
            scores[layer_name] = random_score(weight, seed=42 + index)
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")
    return scores


def run(config_path: str, model_names: list[str] | None = None) -> None:
    config = load_config(config_path)
    device = get_device()
    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    results_dir = ensure_dir(config["logging"]["results_dir"])
    masks_dir = ensure_dir(checkpoint_dir / "masks")
    calibration_dir = checkpoint_dir / "calibration"
    log_path = results_dir / "pruning_matrix.csv"
    _, val_loader = _build_loaders(config)
    model_names = model_names or [config["models"]["student"], config["models"]["teacher"]]

    for model_name in model_names:
        alias = model_name.replace("_patch16_224", "")
        base_model = load_deit_model(
            model_name=model_name,
            num_classes=int(config["models"].get("num_classes", 7)),
            pretrained=False,
        ).to(device)
        checkpoint_path = checkpoint_dir / f"{alias}_ham10000.pth"
        state_dict = load_checkpoint_state(checkpoint_path, map_location=device)
        base_model.load_state_dict(state_dict)
        target_layers = get_linear_layer_names(base_model, exclude_keywords=config["pruning"]["exclude_layers"])

        activation_norms = torch.load(calibration_dir / f"{alias}_activation_norms.pt", map_location="cpu")
        gradients = torch.load(calibration_dir / f"{alias}_gradients.pt", map_location="cpu")
        baseline_results = evaluate_model(
            base_model,
            val_loader,
            device,
            class_names=CLASS_NAMES,
            progress_desc=f"{alias} dense eval",
        )
        append_csv_row(
            log_path,
            {
                "model": alias,
                "criterion": "dense",
                "sparsity": 0.0,
                "overall_acc": baseline_results["overall_accuracy"],
                "balanced_acc": baseline_results["balanced_accuracy"],
                "mel_sensitivity": baseline_results["per_class_sensitivity"]["mel"],
                "bcc_sensitivity": baseline_results["per_class_sensitivity"]["bcc"],
                "akiec_sensitivity": baseline_results["per_class_sensitivity"]["akiec"],
                "nv_sensitivity": baseline_results["per_class_sensitivity"]["nv"],
                "bkl_sensitivity": baseline_results["per_class_sensitivity"]["bkl"],
                "df_sensitivity": baseline_results["per_class_sensitivity"]["df"],
                "vasc_sensitivity": baseline_results["per_class_sensitivity"]["vasc"],
                "nonzero_params": get_model_size_kb(base_model)["nonzero_params"],
                "size_kb": get_model_size_kb(base_model)["size_kb"],
            },
        )

        for criterion_name in tqdm(config["pruning"]["criteria"], desc=f"{alias} criteria", leave=False):
            scores = _score_layers(target_layers, criterion_name, activation_norms=activation_norms, gradients=gradients)
            for sparsity in tqdm(config["pruning"]["sparsities"], desc=f"{alias} {criterion_name}", leave=False):
                model = load_deit_model(
                    model_name=model_name,
                    num_classes=int(config["models"].get("num_classes", 7)),
                    pretrained=False,
                ).to(device)
                model.load_state_dict(state_dict)
                masks = compute_global_masks(model, scores, float(sparsity))
                apply_masks(model, masks)

                stats = get_sparsity_stats(model, masks)
                metrics = evaluate_model(
                    model,
                    val_loader,
                    device,
                    class_names=CLASS_NAMES,
                    progress_desc=f"{alias} {criterion_name} s={sparsity}",
                )
                size_info = get_model_size_kb(model, sparse=True)

                append_csv_row(
                    log_path,
                    {
                        "model": alias,
                        "criterion": criterion_name,
                        "sparsity": float(sparsity),
                        "overall_acc": metrics["overall_accuracy"],
                        "balanced_acc": metrics["balanced_accuracy"],
                        "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
                        "bcc_sensitivity": metrics["per_class_sensitivity"]["bcc"],
                        "akiec_sensitivity": metrics["per_class_sensitivity"]["akiec"],
                        "nv_sensitivity": metrics["per_class_sensitivity"]["nv"],
                        "bkl_sensitivity": metrics["per_class_sensitivity"]["bkl"],
                        "df_sensitivity": metrics["per_class_sensitivity"]["df"],
                        "vasc_sensitivity": metrics["per_class_sensitivity"]["vasc"],
                        "nonzero_params": stats["nonzero_params"],
                        "size_kb": size_info["size_kb"],
                    },
                )
                save_masks(masks_dir / f"{alias}_{criterion_name}_s{sparsity:.1f}.pt", masks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pruning matrix experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.models)


if __name__ == "__main__":
    main()
