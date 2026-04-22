from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import HAM10000Dataset, compute_class_weights, get_train_val_splits, get_transforms
from evaluation.metrics import evaluate_model
from models.load_models import load_deit_model
from utils.config import get_device, load_config, should_pin_memory
from utils.io import append_csv_row, ensure_dir, save_checkpoint


def _make_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    dataset_cfg = config["dataset"]
    finetune_cfg = config["finetune"]
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
    class_weights = compute_class_weights(metadata_csv, train_indices)

    train_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        transform=get_transforms(
            split="train",
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(finetune_cfg.get("batch_size", 64)),
        shuffle=True,
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
    return train_loader, val_loader, class_weights


def _train_one_model(
    model_name: str,
    config: dict[str, Any],
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
) -> None:
    model = load_deit_model(
        model_name=model_name,
        num_classes=int(config["models"].get("num_classes", 7)),
        pretrained=bool(config["models"].get("pretrained", True)),
    ).to(device)

    finetune_cfg = config["finetune"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(finetune_cfg.get("lr", 1e-4)),
        weight_decay=float(finetune_cfg.get("weight_decay", 0.05)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(finetune_cfg.get("epochs", 20))))
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if finetune_cfg.get("use_weighted_loss", True) else None
    )

    checkpoint_dir = ensure_dir(config["logging"]["checkpoints_dir"])
    results_dir = ensure_dir(config["logging"]["results_dir"])
    alias = model_name.replace("_patch16_224", "")
    checkpoint_path = checkpoint_dir / f"{alias}_ham10000.pth"
    history_path = results_dir / "finetune_history.csv"

    best_metric = float("-inf")
    best_state = model.state_dict()

    for epoch in range(int(finetune_cfg.get("epochs", 20))):
        model.train()
        running_loss = 0.0
        running_examples = 0
        running_correct = 0

        for images, labels in tqdm(train_loader, desc=f"{alias} epoch {epoch + 1}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_examples += batch_size
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())

        scheduler.step()
        val_results = evaluate_model(model, val_loader, device)
        row = {
            "model": alias,
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, running_examples),
            "train_accuracy": running_correct / max(1, running_examples),
            "val_balanced_accuracy": val_results["balanced_accuracy"],
            "val_melanoma_sensitivity": val_results["melanoma_sensitivity"],
        }
        append_csv_row(history_path, row)

        if val_results["balanced_accuracy"] > best_metric:
            best_metric = val_results["balanced_accuracy"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            save_checkpoint(
                checkpoint_path,
                model,
                model_name=model_name,
                best_balanced_accuracy=best_metric,
            )

    model.load_state_dict(best_state)
    save_checkpoint(
        checkpoint_path,
        model,
        model_name=model_name,
        best_balanced_accuracy=best_metric,
    )


def run(config_path: str, model_names: list[str] | None = None) -> None:
    config = load_config(config_path)
    device = get_device()
    train_loader, val_loader, class_weights = _make_loaders(config)
    model_names = model_names or [config["models"]["student"], config["models"]["teacher"]]
    for model_name in model_names:
        _train_one_model(model_name, config, device, train_loader, val_loader, class_weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DeiT models on HAM10000.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.models)


if __name__ == "__main__":
    main()
