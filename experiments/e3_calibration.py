from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.dataset import HAM10000Dataset, compute_class_weights, get_train_val_splits, get_transforms
from models.load_models import get_linear_layer_names, load_deit_model
from pruning.hooks import ActivationCollector, GradientCollector
from utils.config import get_device, load_config, should_pin_memory
from utils.io import ensure_dir, load_checkpoint_state


def _build_calibration_loader(config):
    dataset_cfg = config["dataset"]
    pin_memory = should_pin_memory()
    metadata_csv = dataset_cfg.get("metadata_csv") or str(Path(dataset_cfg["root"]) / "processed_metadata.csv")
    train_indices, _ = get_train_val_splits(
        metadata_csv,
        train_ratio=float(dataset_cfg.get("train_split", 0.8)),
        seed=int(dataset_cfg.get("seed", 42)),
    )
    max_train_samples = dataset_cfg.get("max_train_samples")
    if max_train_samples is not None:
        train_indices = train_indices[: int(max_train_samples)]
    calibration_size = int(config["pruning"].get("calibration_size", 128))
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
    calibration_dataset = Subset(train_dataset, list(range(min(calibration_size, len(train_dataset)))))
    loader = DataLoader(
        calibration_dataset,
        batch_size=min(32, calibration_size),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 4)),
        pin_memory=pin_memory,
    )
    class_weights = compute_class_weights(metadata_csv, train_indices)
    return loader, class_weights


def run(config_path: str, model_names: list[str] | None = None) -> None:
    config = load_config(config_path)
    device = get_device()
    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    output_dir = ensure_dir(checkpoint_dir / "calibration")
    calibration_loader, class_weights = _build_calibration_loader(config)
    model_names = model_names or [config["models"]["student"], config["models"]["teacher"]]

    for model_name in model_names:
        model = load_deit_model(
            model_name=model_name,
            num_classes=int(config["models"].get("num_classes", 7)),
            pretrained=False,
        ).to(device)
        alias = model_name.replace("_patch16_224", "")
        checkpoint_path = checkpoint_dir / f"{alias}_ham10000.pth"
        model.load_state_dict(load_checkpoint_state(checkpoint_path, map_location=device))
        target_layers = get_linear_layer_names(model, exclude_keywords=config["pruning"]["exclude_layers"])

        activation_collector = ActivationCollector(model, target_layers)
        activation_collector.register_hooks()
        model.eval()
        with torch.no_grad():
            for images, _ in calibration_loader:
                _ = model(images.to(device, non_blocking=True))
        activation_norms = activation_collector.get_activation_norms()
        activation_collector.remove_hooks()
        torch.save(activation_norms, output_dir / f"{alias}_activation_norms.pt")

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        model.zero_grad(set_to_none=True)
        for images, labels in calibration_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
        gradients = GradientCollector(model, target_layers).get_gradients()
        torch.save(gradients, output_dir / f"{alias}_gradients.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect pruning calibration tensors.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.models)


if __name__ == "__main__":
    main()
