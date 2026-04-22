from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm import tqdm

from data.dataset import HAM10000Dataset, compute_class_weights, get_train_val_splits, get_transforms
from evaluation.metrics import CLASS_NAMES, evaluate_model
from evaluation.model_size import get_model_size_kb
from pruning.masking import apply_masks, compute_global_masks
from utils.config import get_device, load_config, should_pin_memory
from utils.io import ensure_dir, load_checkpoint_state, save_checkpoint


def _load_mobilenet(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def _build_loaders(config):
    dataset_cfg = config["dataset"]
    pin_memory = should_pin_memory()
    metadata_csv = dataset_cfg.get("metadata_csv") or str(Path(dataset_cfg["root"]) / "processed_metadata.csv")
    train_indices, val_indices = get_train_val_splits(
        metadata_csv,
        train_ratio=float(dataset_cfg.get("train_split", 0.8)),
        seed=int(dataset_cfg.get("seed", 42)),
    )

    train_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        image_dir=dataset_cfg["root"],
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
        image_dir=dataset_cfg["root"],
        transform=get_transforms(
            split="val",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(config["augmentation"].get("resize_size", 256)),
            augmentation_cfg=config["augmentation"],
        ),
        indices=val_indices,
    )
    calibration_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        image_dir=dataset_cfg["root"],
        transform=get_transforms(
            split="val",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(config["augmentation"].get("resize_size", 256)),
            augmentation_cfg=config["augmentation"],
        ),
        indices=train_indices,
    )

    num_workers = int(dataset_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["finetune"].get("batch_size", 64)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["evaluation"].get("batch_size", 128)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    calibration_size = int(config["pruning"].get("calibration_size", 128))
    calibration_subset = Subset(calibration_dataset, list(range(min(calibration_size, len(calibration_dataset)))))
    calibration_loader = DataLoader(
        calibration_subset,
        batch_size=min(32, calibration_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    class_weights = compute_class_weights(metadata_csv, train_indices)
    return train_loader, val_loader, calibration_loader, class_weights


def _target_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    layers: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if name.startswith("classifier"):
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((name, module))
    return layers


class MobilenetActivationCollector:
    def __init__(self, target_layers: list[tuple[str, nn.Module]]) -> None:
        self.target_layers = dict(target_layers)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.sum_sq: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}

    def _hook(self, layer_name: str):
        def capture(module: nn.Module, inputs, output) -> None:
            if not inputs:
                return
            activations = inputs[0].detach().float().cpu()
            if activations.ndim == 4:
                flattened = activations.permute(0, 2, 3, 1).reshape(-1, activations.shape[1])
                feature_dim = activations.shape[1]
            elif activations.ndim == 2:
                flattened = activations.reshape(-1, activations.shape[-1])
                feature_dim = activations.shape[-1]
            else:
                raise RuntimeError(f"Unsupported activation shape for layer '{layer_name}': {tuple(activations.shape)}")

            self.sum_sq.setdefault(layer_name, torch.zeros(feature_dim, dtype=torch.float32))
            self.counts.setdefault(layer_name, 0)
            self.sum_sq[layer_name] += flattened.pow(2).sum(dim=0)
            self.counts[layer_name] += int(flattened.shape[0])

        return capture

    def register(self) -> None:
        for layer_name, module in self.target_layers.items():
            self.handles.append(module.register_forward_hook(self._hook(layer_name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def norms(self) -> dict[str, torch.Tensor]:
        return {
            layer_name: torch.sqrt(self.sum_sq[layer_name] / max(1, self.counts[layer_name]))
            for layer_name in self.target_layers
        }


def _collect_activation_norms(model: nn.Module, calibration_loader: DataLoader, device: torch.device):
    target_layers = _target_layers(model)
    collector = MobilenetActivationCollector(target_layers)
    collector.register()
    model.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            _ = model(images.to(device, non_blocking=True))
    norms = collector.norms()
    collector.close()
    return norms, target_layers


def _score_layers(
    target_layers: list[tuple[str, nn.Module]],
    criterion_name: str,
    activation_norms: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    scores: dict[str, torch.Tensor] = {}
    for layer_name, layer in target_layers:
        weight = layer.weight.detach()
        if criterion_name == "magnitude":
            scores[layer_name] = weight.abs()
        elif criterion_name == "wanda":
            norms = activation_norms[layer_name].to(weight.device)
            if weight.ndim == 4:
                if not isinstance(layer, nn.Conv2d):
                    raise RuntimeError(f"Expected Conv2d for 4D weight tensor in layer '{layer_name}'.")
                if layer.groups == 1:
                    scores[layer_name] = weight.abs() * norms.view(1, -1, 1, 1)
                else:
                    score = weight.abs().clone()
                    in_per_group = layer.in_channels // layer.groups
                    out_per_group = layer.out_channels // layer.groups
                    for group_idx in range(layer.groups):
                        in_start = group_idx * in_per_group
                        in_end = in_start + in_per_group
                        out_start = group_idx * out_per_group
                        out_end = out_start + out_per_group
                        score[out_start:out_end] *= norms[in_start:in_end].view(1, in_per_group, 1, 1)
                    scores[layer_name] = score
            else:
                scores[layer_name] = weight.abs() * norms.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")
    return scores


def _upsert_rows(csv_path: Path, rows: list[dict[str, float | int | str]]) -> None:
    frame = pd.DataFrame(rows)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = frame.drop_duplicates(subset=["model", "criterion", "sparsity"], keep="last")
    frame.to_csv(csv_path, index=False)


def run(config_path: str) -> None:
    config = load_config(config_path)
    device = get_device()
    train_loader, val_loader, calibration_loader, class_weights = _build_loaders(config)

    checkpoint_dir = ensure_dir(config["logging"]["checkpoints_dir"])
    results_dir = ensure_dir(config["logging"]["results_dir"])
    checkpoint_path = Path(checkpoint_dir) / "mobilenetv2_ham10000.pth"
    log_path = Path(results_dir) / "mobilenet_results.csv"

    model = _load_mobilenet(num_classes=int(config["models"].get("num_classes", 7)), pretrained=True).to(device)
    if checkpoint_path.exists():
        model.load_state_dict(load_checkpoint_state(checkpoint_path, map_location=device))
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=float(config["finetune"].get("lr", 1e-4)),
            weight_decay=float(config["finetune"].get("weight_decay", 0.05)),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(config["finetune"].get("epochs", 20))))
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        best_metric = float("-inf")
        best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        for epoch in range(int(config["finetune"].get("epochs", 20))):
            model.train()
            for images, labels in tqdm(train_loader, desc=f"mobilenetv2 epoch {epoch + 1}", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()
            val_metrics = evaluate_model(model, val_loader, device, class_names=CLASS_NAMES)
            if val_metrics["balanced_accuracy"] > best_metric:
                best_metric = float(val_metrics["balanced_accuracy"])
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                save_checkpoint(
                    checkpoint_path,
                    model,
                    model_name="mobilenetv2",
                    best_balanced_accuracy=best_metric,
                )

        model.load_state_dict(best_state)
        save_checkpoint(
            checkpoint_path,
            model,
            model_name="mobilenetv2",
            best_balanced_accuracy=best_metric,
        )

    dense_metrics = evaluate_model(model, val_loader, device, class_names=CLASS_NAMES)
    rows: list[dict[str, float | int | str]] = [
        {
            "model": "mobilenetv2",
            "criterion": "none",
            "sparsity": 0.0,
            "overall_acc": dense_metrics["overall_accuracy"],
            "balanced_acc": dense_metrics["balanced_accuracy"],
            "mel_sensitivity": dense_metrics["per_class_sensitivity"]["mel"],
            "bcc_sensitivity": dense_metrics["per_class_sensitivity"]["bcc"],
            "akiec_sensitivity": dense_metrics["per_class_sensitivity"]["akiec"],
            "nv_sensitivity": dense_metrics["per_class_sensitivity"]["nv"],
            "bkl_sensitivity": dense_metrics["per_class_sensitivity"]["bkl"],
            "df_sensitivity": dense_metrics["per_class_sensitivity"]["df"],
            "vasc_sensitivity": dense_metrics["per_class_sensitivity"]["vasc"],
            "size_kb": get_model_size_kb(model)["size_kb"],
        }
    ]

    activation_norms, target_layers = _collect_activation_norms(model, calibration_loader, device)
    base_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    for criterion_name in ("magnitude", "wanda"):
        pruned_model = _load_mobilenet(num_classes=int(config["models"].get("num_classes", 7)), pretrained=False).to(device)
        pruned_model.load_state_dict(base_state)
        scores = _score_layers(target_layers, criterion_name, activation_norms)
        masks = compute_global_masks(pruned_model, scores, sparsity=0.5)
        apply_masks(pruned_model, masks)
        metrics = evaluate_model(pruned_model, val_loader, device, class_names=CLASS_NAMES)
        rows.append(
            {
                "model": "mobilenetv2",
                "criterion": criterion_name,
                "sparsity": 0.5,
                "overall_acc": metrics["overall_accuracy"],
                "balanced_acc": metrics["balanced_accuracy"],
                "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
                "bcc_sensitivity": metrics["per_class_sensitivity"]["bcc"],
                "akiec_sensitivity": metrics["per_class_sensitivity"]["akiec"],
                "nv_sensitivity": metrics["per_class_sensitivity"]["nv"],
                "bkl_sensitivity": metrics["per_class_sensitivity"]["bkl"],
                "df_sensitivity": metrics["per_class_sensitivity"]["df"],
                "vasc_sensitivity": metrics["per_class_sensitivity"]["vasc"],
                "size_kb": get_model_size_kb(pruned_model, sparse=True)["size_kb"],
            }
        )

    _upsert_rows(log_path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the MobileNetV2 HAM10000 baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
