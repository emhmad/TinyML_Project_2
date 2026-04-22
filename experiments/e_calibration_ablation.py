from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import HAM10000Dataset, get_transforms
from evaluation.metrics import CLASS_NAMES, evaluate_model
from experiments.common import build_splits, collect_activation_norms, load_trained_model, metadata_csv_path
from models.load_models import get_linear_layer_names
from pruning.masking import apply_masks, compute_global_masks
from pruning.scoring import wanda_score
from utils.config import get_device, load_config, should_pin_memory
from utils.io import ensure_dir

CALIBRATION_SIZES = [16, 32, 64, 128, 256, 512]


def run(config_path: str) -> None:
    config = load_config(config_path)
    device = get_device()
    dataset_cfg = config["dataset"]
    metadata_csv = metadata_csv_path(config)
    train_indices, val_indices = build_splits(config)
    dataset_root = Path(dataset_cfg["root"])
    pin_memory = should_pin_memory(device)
    num_workers = int(dataset_cfg.get("num_workers", 4))

    train_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        image_dir=dataset_root,
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
        image_dir=dataset_root,
        transform=get_transforms(
            split="val",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(config["augmentation"].get("resize_size", 256)),
            augmentation_cfg=config["augmentation"],
        ),
        indices=val_indices,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["evaluation"].get("batch_size", 128)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    generator = torch.Generator()
    generator.manual_seed(42)
    ordered_indices = torch.randperm(len(train_dataset), generator=generator).tolist()

    model_name = config["models"]["teacher"]
    alias = model_name.replace("_patch16_224", "")
    results_dir = ensure_dir(config["logging"]["results_dir"])
    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    output_path = Path(results_dir) / "calibration_ablation.csv"

    base_model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
    target_layers = get_linear_layer_names(base_model, exclude_keywords=config["pruning"]["exclude_layers"])
    base_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}

    rows: list[dict[str, float | int]] = []
    for calibration_size in CALIBRATION_SIZES:
        subset = Subset(train_dataset, ordered_indices[:calibration_size])
        calibration_loader = DataLoader(
            subset,
            batch_size=min(32, calibration_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
        model.load_state_dict(base_state)
        activation_norms, _ = collect_activation_norms(
            model,
            calibration_loader,
            config["pruning"]["exclude_layers"],
            device,
        )
        scores = {
            layer_name: wanda_score(layer.weight.detach(), activation_norms[layer_name])
            for layer_name, layer in target_layers
        }
        masks = compute_global_masks(model, scores, sparsity=0.5)
        apply_masks(model, masks)
        metrics = evaluate_model(
            model,
            val_loader,
            device,
            class_names=CLASS_NAMES,
            progress_desc=f"wanda calibration {calibration_size}",
        )
        rows.append(
            {
                "calibration_size": calibration_size,
                "overall_acc": metrics["overall_accuracy"],
                "balanced_acc": metrics["balanced_accuracy"],
                "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
                "bcc_sensitivity": metrics["per_class_sensitivity"]["bcc"],
                "akiec_sensitivity": metrics["per_class_sensitivity"]["akiec"],
                "nv_sensitivity": metrics["per_class_sensitivity"]["nv"],
                "bkl_sensitivity": metrics["per_class_sensitivity"]["bkl"],
                "df_sensitivity": metrics["per_class_sensitivity"]["df"],
                "vasc_sensitivity": metrics["per_class_sensitivity"]["vasc"],
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wanda calibration-size ablation.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
