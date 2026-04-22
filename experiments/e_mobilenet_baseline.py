from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm import tqdm

from data.dataset import HAM10000Dataset, compute_class_weights, get_transforms
from evaluation.metrics import CLASS_NAMES, dangerous_class_degradation_ratio, evaluate_model
from evaluation.model_size import get_model_size_kb
from experiments.common import build_splits, metadata_csv_path
from models.load_models import get_layer_by_name
from pruning.masking import apply_masks, compute_global_masks
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed, should_pin_memory
from utils.distributed import (
    barrier,
    init_distributed,
    is_main_process,
    maybe_wrap_ddp,
    shutdown_distributed,
    unwrap_model,
)
from utils.io import ensure_dir, load_checkpoint_state, save_checkpoint
from utils.seed import set_seed, worker_init_fn


def _load_mobilenet(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def _build_loaders(config):
    dataset_cfg = config["dataset"]
    pin_memory = should_pin_memory()
    metadata_csv = metadata_csv_path(config)
    train_indices, val_indices = build_splits(config)

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
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["evaluation"].get("batch_size", 128)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    calibration_size = int(config["pruning"].get("calibration_size", 128))
    calibration_subset = Subset(calibration_dataset, list(range(min(calibration_size, len(calibration_dataset)))))
    calibration_loader = DataLoader(
        calibration_subset,
        batch_size=min(32, calibration_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
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


class _MobilenetActivationCollector:
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
    collector = _MobilenetActivationCollector(target_layers)
    collector.register()
    model.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            _ = model(images.to(device, non_blocking=True))
    norms = collector.norms()
    collector.close()
    return norms, target_layers


def _collect_cnn_gradients(
    model: nn.Module,
    target_layers: list[tuple[str, nn.Module]],
    calibration_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    model.zero_grad(set_to_none=True)
    for images, labels in calibration_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
    gradients: dict[str, torch.Tensor] = {}
    for layer_name, module in target_layers:
        if module.weight.grad is None:
            continue
        gradients[layer_name] = module.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    return gradients


def _score_layers(
    target_layers: list[tuple[str, nn.Module]],
    criterion_name: str,
    activation_norms: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor] | None = None,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    scores: dict[str, torch.Tensor] = {}
    for index, (layer_name, layer) in enumerate(target_layers):
        weight = layer.weight.detach()
        if criterion_name == "magnitude":
            scores[layer_name] = weight.abs()
        elif criterion_name == "wanda":
            norms = activation_norms[layer_name].to(weight.device)
            if weight.ndim == 4:
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
        elif criterion_name == "taylor":
            if gradients is None or layer_name not in gradients:
                raise ValueError(f"Taylor scoring requires gradients for {layer_name}.")
            scores[layer_name] = (weight * gradients[layer_name].to(weight.device)).abs()
        elif criterion_name == "random":
            generator = torch.Generator(device=weight.device)
            generator.manual_seed(seed + index)
            scores[layer_name] = torch.rand(weight.shape, generator=generator, device=weight.device, dtype=weight.dtype)
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")
    return scores


def _freeze_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    for layer_name, mask in masks.items():
        module = get_layer_by_name(model, layer_name)
        torch_prune.custom_from_mask(module, "weight", mask.to(module.weight.device, dtype=module.weight.dtype))


def _remove_reparam(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for layer_name, mask in masks.items():
            module = get_layer_by_name(model, layer_name)
            if hasattr(module, "weight_orig"):
                torch_prune.remove(module, "weight")
            module.weight.mul_(mask.to(module.weight.device, dtype=module.weight.dtype))


def _row(
    *,
    seed: int,
    model: str,
    criterion: str,
    sparsity: float,
    metrics: dict,
    sizes: dict,
    baseline_sensitivity: dict[str, float] | None,
    recovery_epochs: int = 0,
) -> dict:
    row = {
        "seed": seed,
        "model": model,
        "criterion": criterion,
        "sparsity": float(sparsity),
        "recovery_epochs": int(recovery_epochs),
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
        "size_kb": sizes["size_kb"],
        "disk_size_kb": sizes["disk_size_kb"],
        "effective_sparse_size_kb": sizes["effective_sparse_size_kb"],
    }
    if baseline_sensitivity is not None:
        row["dangerous_class_degradation_ratio"] = dangerous_class_degradation_ratio(
            baseline_sensitivity=baseline_sensitivity,
            pruned_sensitivity=metrics["per_class_sensitivity"],
        )
    return row


def _upsert_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = frame.drop_duplicates(
        subset=["seed", "model", "criterion", "sparsity", "recovery_epochs"],
        keep="last",
    )
    frame.to_csv(csv_path, index=False)


def _train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    config: dict[str, Any],
    checkpoint_path: Path,
    seed: int,
) -> nn.Module:
    """Shared training loop for MobileNetV2. DDP-aware."""
    ddp_model = maybe_wrap_ddp(model, device)
    optimizer = AdamW(
        ddp_model.parameters(),
        lr=float(config["finetune"].get("lr", 1e-4)),
        weight_decay=float(config["finetune"].get("weight_decay", 0.05)),
    )
    epochs = max(1, int(config["finetune"].get("epochs", 20)))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_metric = float("-inf")
    best_state = {name: t.detach().cpu().clone() for name, t in unwrap_model(ddp_model).state_dict().items()}

    for epoch in range(epochs):
        ddp_model.train()
        iterator = train_loader
        if is_main_process():
            iterator = tqdm(train_loader, desc=f"mobilenetv2 epoch {epoch + 1}", leave=False)
        for images, labels in iterator:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        barrier()

        if is_main_process():
            val_metrics = evaluate_model(unwrap_model(ddp_model), val_loader, device, class_names=CLASS_NAMES)
            if val_metrics["balanced_accuracy"] > best_metric:
                best_metric = float(val_metrics["balanced_accuracy"])
                best_state = {name: t.detach().cpu().clone() for name, t in unwrap_model(ddp_model).state_dict().items()}
                save_checkpoint(
                    checkpoint_path,
                    unwrap_model(ddp_model),
                    model_name="mobilenetv2",
                    seed=seed,
                    best_balanced_accuracy=best_metric,
                )
        barrier()

    unwrap_model(ddp_model).load_state_dict(best_state)
    if is_main_process():
        save_checkpoint(
            checkpoint_path,
            unwrap_model(ddp_model),
            model_name="mobilenetv2",
            seed=seed,
            best_balanced_accuracy=best_metric,
        )
    return unwrap_model(ddp_model)


def _recovery(
    base_state: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    config: dict[str, Any],
    recovery_epochs: int,
    lr: float,
) -> nn.Module:
    model = _load_mobilenet(
        num_classes=int(config["models"].get("num_classes", 7)), pretrained=False
    ).to(device)
    model.load_state_dict(base_state)
    _freeze_masks(model, masks)
    ddp_model = maybe_wrap_ddp(model, device)
    optimizer = AdamW(ddp_model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, recovery_epochs))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    for _ in range(recovery_epochs):
        ddp_model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        barrier()
    _remove_reparam(unwrap_model(ddp_model), masks)
    return unwrap_model(ddp_model)


def run(config_path: str, seed_override: int | None = None) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    set_seed(seed)
    init_distributed()
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
        model = _train(
            model,
            train_loader,
            val_loader,
            class_weights,
            device,
            config,
            checkpoint_path,
            seed,
        )

    dense_metrics = evaluate_model(model, val_loader, device, class_names=CLASS_NAMES)
    baseline_sensitivity = dict(dense_metrics["per_class_sensitivity"])
    dense_sizes = get_model_size_kb(model)

    rows: list[dict[str, Any]] = [
        _row(
            seed=seed,
            model="mobilenetv2",
            criterion="none",
            sparsity=0.0,
            metrics=dense_metrics,
            sizes=dense_sizes,
            baseline_sensitivity=None,
        )
    ]

    activation_norms, target_layers = _collect_activation_norms(model, calibration_loader, device)
    gradients = _collect_cnn_gradients(model, target_layers, calibration_loader, class_weights, device)
    base_state = {name: t.detach().cpu().clone() for name, t in model.state_dict().items()}

    # Expanded criterion x sparsity sweep (W7). Recovery is optional per config.
    criteria = list(config["pruning"].get("cnn_criteria", ["magnitude", "wanda", "taylor", "random"]))
    sparsities = list(config["pruning"].get("cnn_sparsities", config["pruning"]["sparsities"]))
    recovery_cfg = config.get("recovery", {})
    run_recovery = bool(recovery_cfg.get("cnn_enabled", True))
    recovery_epochs = int(recovery_cfg.get("epochs", 5))
    recovery_lr = float(recovery_cfg.get("lr", 1e-5))

    for criterion_name in criteria:
        scores = _score_layers(
            target_layers,
            criterion_name,
            activation_norms,
            gradients=gradients,
            seed=seed,
        )
        for sparsity in sparsities:
            pruned_model = _load_mobilenet(
                num_classes=int(config["models"].get("num_classes", 7)), pretrained=False
            ).to(device)
            pruned_model.load_state_dict(base_state)
            masks = compute_global_masks(pruned_model, scores, float(sparsity))
            apply_masks(pruned_model, masks)
            metrics = evaluate_model(pruned_model, val_loader, device, class_names=CLASS_NAMES)
            sizes = get_model_size_kb(pruned_model, sparse=True)
            rows.append(
                _row(
                    seed=seed,
                    model="mobilenetv2",
                    criterion=criterion_name,
                    sparsity=float(sparsity),
                    metrics=metrics,
                    sizes=sizes,
                    baseline_sensitivity=baseline_sensitivity,
                )
            )
            if run_recovery:
                recovered = _recovery(
                    base_state=base_state,
                    masks=masks,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    class_weights=class_weights,
                    device=device,
                    config=config,
                    recovery_epochs=recovery_epochs,
                    lr=recovery_lr,
                )
                rec_metrics = evaluate_model(recovered, val_loader, device, class_names=CLASS_NAMES)
                rec_sizes = get_model_size_kb(recovered, sparse=True)
                rows.append(
                    _row(
                        seed=seed,
                        model="mobilenetv2",
                        criterion=f"{criterion_name}+recovery",
                        sparsity=float(sparsity),
                        metrics=rec_metrics,
                        sizes=rec_sizes,
                        baseline_sensitivity=baseline_sensitivity,
                        recovery_epochs=recovery_epochs,
                    )
                )

    if is_main_process():
        _upsert_rows(log_path, rows)
    shutdown_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the MobileNetV2 HAM10000 baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, seed_override=args.seed)


if __name__ == "__main__":
    main()
