from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import HAM10000Dataset, compute_class_weights, get_train_val_splits, get_transforms
from models.load_models import get_linear_layer_names, load_deit_model
from pruning.hooks import ActivationCollector
from utils.config import should_pin_memory
from utils.io import load_checkpoint_state


def model_alias(model_name: str) -> str:
    return model_name.replace("_patch16_224", "")


def metadata_csv_path(config: dict[str, Any]) -> str:
    dataset_cfg = config["dataset"]
    return dataset_cfg.get("metadata_csv") or str(Path(dataset_cfg["root"]) / "processed_metadata.csv")


def build_splits(config: dict[str, Any]) -> tuple[list[int], list[int]]:
    dataset_cfg = config["dataset"]
    return get_train_val_splits(
        metadata_csv_path(config),
        train_ratio=float(dataset_cfg.get("train_split", 0.8)),
        seed=int(dataset_cfg.get("seed", 42)),
    )


def build_dataloaders(
    config: dict[str, Any],
    *,
    include_train: bool = True,
    calibration_size: int | None = None,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
) -> tuple[DataLoader | None, DataLoader, DataLoader | None, torch.Tensor]:
    dataset_cfg = config["dataset"]
    augmentation_cfg = config["augmentation"]
    pin_memory = should_pin_memory()
    metadata_csv = metadata_csv_path(config)
    train_indices, val_indices = build_splits(config)

    train_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        transform=get_transforms(
            split="train",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(augmentation_cfg.get("resize_size", 256)),
            augmentation_cfg=augmentation_cfg,
        ),
        indices=train_indices,
    )
    val_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        transform=get_transforms(
            split="val",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(augmentation_cfg.get("resize_size", 256)),
            augmentation_cfg=augmentation_cfg,
        ),
        indices=val_indices,
    )
    calibration_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        transform=get_transforms(
            split="val",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(augmentation_cfg.get("resize_size", 256)),
            augmentation_cfg=augmentation_cfg,
        ),
        indices=train_indices,
    )

    num_workers = int(dataset_cfg.get("num_workers", 4))
    train_loader = None
    if include_train:
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(train_batch_size or config["finetune"].get("batch_size", 64)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(eval_batch_size or config["evaluation"].get("batch_size", 128)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    calibration_loader = None
    if calibration_size is not None:
        subset = Subset(calibration_dataset, list(range(min(calibration_size, len(calibration_dataset)))))
        calibration_loader = DataLoader(
            subset,
            batch_size=min(32, calibration_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    class_weights = compute_class_weights(metadata_csv, train_indices)
    return train_loader, val_loader, calibration_loader, class_weights


def load_trained_model(
    config: dict[str, Any],
    model_name: str,
    device: torch.device,
    *,
    checkpoint_name: str | None = None,
    pretrained: bool = False,
) -> torch.nn.Module:
    model = load_deit_model(
        model_name=model_name,
        num_classes=int(config["models"].get("num_classes", 7)),
        pretrained=pretrained,
    ).to(device)
    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    alias = checkpoint_name or model_alias(model_name)
    state_dict = load_checkpoint_state(checkpoint_dir / f"{alias}.pth", map_location=device)
    model.load_state_dict(state_dict)
    return model


def collect_activation_norms(
    model: torch.nn.Module,
    calibration_loader: DataLoader,
    exclude_layers: list[str],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, torch.nn.Module]]]:
    target_layers = get_linear_layer_names(model, exclude_keywords=exclude_layers)
    collector = ActivationCollector(model, target_layers)
    collector.register_hooks()
    model.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            _ = model(images.to(device, non_blocking=True))
    norms = collector.get_activation_norms()
    collector.remove_hooks()
    return norms, target_layers
