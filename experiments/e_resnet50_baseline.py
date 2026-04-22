"""
ResNet-50 baseline (W7 option a).

Stronger CNN than MobileNetV2 and not dominated by depthwise-separable
convolutions, so it avoids the argument that "pruning instability is a
MobileNet-specific artefact". Runs the full criterion × sparsity sweep
with optional recovery fine-tuning, mirroring the ViT pipeline so the
cross-architecture claim can be defended with a like-for-like comparison.

Launchable single-GPU or with torchrun. Reuses the activation-collection
and scoring helpers from the MobileNet baseline so we keep one code path
for CNNs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from evaluation.metrics import CLASS_NAMES, dangerous_class_degradation_ratio, evaluate_model
from evaluation.model_size import get_model_size_kb
from experiments.e_mobilenet_baseline import (
    _build_loaders,
    _collect_activation_norms,
    _collect_cnn_gradients,
    _freeze_masks,
    _recovery,
    _remove_reparam,
    _row,
    _score_layers,
    _upsert_rows,
)
from pruning.masking import apply_masks, compute_global_masks
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.distributed import (
    barrier,
    init_distributed,
    is_main_process,
    maybe_wrap_ddp,
    shutdown_distributed,
    unwrap_model,
)
from utils.io import ensure_dir, load_checkpoint_state, save_checkpoint
from utils.seed import set_seed


def _load_resnet50(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

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
            iterator = tqdm(train_loader, desc=f"resnet50 epoch {epoch + 1}", leave=False)
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
                    model_name="resnet50",
                    seed=seed,
                    best_balanced_accuracy=best_metric,
                )
        barrier()
    unwrap_model(ddp_model).load_state_dict(best_state)
    return unwrap_model(ddp_model)


def _recovery_resnet(
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
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    model = _load_resnet50(
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
    checkpoint_path = Path(checkpoint_dir) / "resnet50_ham10000.pth"
    log_path = Path(results_dir) / "resnet50_results.csv"

    model = _load_resnet50(num_classes=int(config["models"].get("num_classes", 7)), pretrained=True).to(device)
    if checkpoint_path.exists():
        model.load_state_dict(load_checkpoint_state(checkpoint_path, map_location=device))
    else:
        model = _train(
            model, train_loader, val_loader, class_weights, device, config, checkpoint_path, seed
        )

    dense_metrics = evaluate_model(model, val_loader, device, class_names=CLASS_NAMES)
    baseline_sensitivity = dict(dense_metrics["per_class_sensitivity"])
    dense_sizes = get_model_size_kb(model)

    rows: list[dict[str, Any]] = [
        _row(
            seed=seed,
            model="resnet50",
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

    criteria = list(config["pruning"].get("cnn_criteria", ["magnitude", "wanda", "taylor", "random"]))
    sparsities = list(config["pruning"].get("cnn_sparsities", config["pruning"]["sparsities"]))
    recovery_cfg = config.get("recovery", {})
    run_recovery = bool(recovery_cfg.get("cnn_enabled", True))
    recovery_epochs = int(recovery_cfg.get("epochs", 5))
    recovery_lr = float(recovery_cfg.get("lr", 1e-5))

    for criterion_name in criteria:
        scores = _score_layers(target_layers, criterion_name, activation_norms, gradients, seed=seed)
        for sparsity in sparsities:
            pruned = _load_resnet50(
                num_classes=int(config["models"].get("num_classes", 7)), pretrained=False
            ).to(device)
            pruned.load_state_dict(base_state)
            masks = compute_global_masks(pruned, scores, float(sparsity))
            apply_masks(pruned, masks)
            metrics = evaluate_model(pruned, val_loader, device, class_names=CLASS_NAMES)
            sizes = get_model_size_kb(pruned, sparse=True)
            rows.append(
                _row(
                    seed=seed,
                    model="resnet50",
                    criterion=criterion_name,
                    sparsity=float(sparsity),
                    metrics=metrics,
                    sizes=sizes,
                    baseline_sensitivity=baseline_sensitivity,
                )
            )
            if run_recovery:
                recovered = _recovery_resnet(
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
                        model="resnet50",
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
    parser = argparse.ArgumentParser(description="Train/prune ResNet-50 on HAM10000 (W7 strengthened CNN baseline).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, seed_override=args.seed)


if __name__ == "__main__":
    main()
