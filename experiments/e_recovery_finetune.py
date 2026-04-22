from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from evaluation.metrics import CLASS_NAMES, dangerous_class_degradation_ratio, evaluate_model
from evaluation.model_size import get_model_size_kb
from experiments.common import build_dataloaders, load_trained_model, model_alias
from models.load_models import get_layer_by_name, get_linear_layer_names
from pruning.hooks import ActivationCollector, GradientCollector
from pruning.masking import compute_global_masks
from pruning.scoring import (
    magnitude_score,
    skewness_score,
    sparsegpt_pseudo_score,
    taylor_score,
    wanda_score,
    xpruner_score,
)
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.distributed import (
    barrier,
    init_distributed,
    is_main_process,
    maybe_wrap_ddp,
    shutdown_distributed,
    unwrap_model,
)
from utils.io import ensure_dir, save_checkpoint, save_masks
from utils.seed import set_seed


RECOVERY_CRITERIA = ("magnitude", "wanda", "taylor", "skewness", "xpruner", "sparsegpt_pseudo")


def _collect_calibration_tensors(
    config: dict[str, Any],
    model_name: str,
    model: torch.nn.Module,
    calibration_loader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    alias = model_alias(model_name)
    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    calibration_dir = ensure_dir(checkpoint_dir / "calibration")
    activation_path = calibration_dir / f"{alias}_activation_norms.pt"
    gradient_path = calibration_dir / f"{alias}_gradients.pt"

    if activation_path.exists() and gradient_path.exists():
        return (
            torch.load(activation_path, map_location="cpu"),
            torch.load(gradient_path, map_location="cpu"),
        )

    target_layers = get_linear_layer_names(model, exclude_keywords=config["pruning"]["exclude_layers"])
    activation_collector = ActivationCollector(model, target_layers)
    activation_collector.register_hooks()
    model.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            _ = model(images.to(device, non_blocking=True))
    activation_norms = activation_collector.get_activation_norms()
    activation_collector.remove_hooks()
    torch.save(activation_norms, activation_path)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    model.zero_grad(set_to_none=True)
    for images, labels in calibration_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
    gradients = GradientCollector(model, target_layers).get_gradients()
    torch.save(gradients, gradient_path)
    model.zero_grad(set_to_none=True)
    return activation_norms, gradients


def _score_layers(
    target_layers: list[tuple[str, torch.nn.Module]],
    criterion_name: str,
    activation_norms: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    scores: dict[str, torch.Tensor] = {}
    for layer_name, layer in target_layers:
        weight = layer.weight.detach()
        if criterion_name == "magnitude":
            scores[layer_name] = magnitude_score(weight)
        elif criterion_name == "wanda":
            scores[layer_name] = wanda_score(weight, activation_norms[layer_name])
        elif criterion_name == "taylor":
            scores[layer_name] = taylor_score(weight, gradients[layer_name].to(weight.device))
        elif criterion_name == "skewness":
            scores[layer_name] = skewness_score(weight)
        elif criterion_name == "xpruner":
            out_sens = gradients[layer_name].to(weight.device).abs().sum(dim=1)
            scores[layer_name] = xpruner_score(weight, activation_norms[layer_name], out_sens)
        elif criterion_name == "sparsegpt_pseudo":
            scores[layer_name] = sparsegpt_pseudo_score(weight, activation_norms[layer_name])
        else:
            raise ValueError(f"Unsupported recovery criterion: {criterion_name}")
    return scores


def _freeze_masks(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> None:
    for layer_name, mask in masks.items():
        module = get_layer_by_name(model, layer_name)
        torch_prune.custom_from_mask(module, "weight", mask.to(module.weight.device, dtype=module.weight.dtype))


def _remove_pruning_reparametrization(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for layer_name, mask in masks.items():
            module = get_layer_by_name(model, layer_name)
            if hasattr(module, "weight_orig"):
                torch_prune.remove(module, "weight")
            module.weight.mul_(mask.to(module.weight.device, dtype=module.weight.dtype))


def _verify_mask_persistence(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> float:
    total = 0
    nonzero = 0
    for layer_name, mask in masks.items():
        module = get_layer_by_name(model, layer_name)
        weight = module.weight.detach()
        masked_weights = weight[mask.to(weight.device, dtype=torch.bool) == 0]
        if int(torch.count_nonzero(masked_weights).item()) != 0:
            raise RuntimeError(f"Pruned weights revived in layer '{layer_name}' during recovery finetuning.")
        total += int(mask.numel())
        nonzero += int(mask.sum().item())
    return 1.0 - (nonzero / max(1, total))


def _result_row(
    *,
    seed: int,
    model_name: str,
    criterion_name: str,
    sparsity: float,
    recovery_epochs: int,
    lr: float,
    metrics: dict[str, Any],
    baseline_sensitivity: dict[str, float] | None,
    size_kb: float,
    disk_size_kb: float,
    effective_sparse_size_kb: float,
    best_epoch: int,
) -> dict[str, Any]:
    row = {
        "seed": seed,
        "model": model_alias(model_name),
        "criterion": criterion_name,
        "sparsity": float(sparsity),
        "recovery_epochs": int(recovery_epochs),
        "recovery_lr": float(lr),
        "best_epoch": int(best_epoch),
        "overall_acc": metrics["overall_accuracy"],
        "balanced_acc": metrics["balanced_accuracy"],
        "macro_auroc": metrics.get("macro_auroc"),
        "melanoma_auroc": metrics.get("melanoma_auroc"),
        "ece_top_label": metrics.get("ece_top_label"),
        "akiec_sensitivity": metrics["per_class_sensitivity"]["akiec"],
        "bcc_sensitivity": metrics["per_class_sensitivity"]["bcc"],
        "bkl_sensitivity": metrics["per_class_sensitivity"]["bkl"],
        "df_sensitivity": metrics["per_class_sensitivity"]["df"],
        "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
        "nv_sensitivity": metrics["per_class_sensitivity"]["nv"],
        "vasc_sensitivity": metrics["per_class_sensitivity"]["vasc"],
        "size_kb": float(size_kb),
        "disk_size_kb": float(disk_size_kb),
        "effective_sparse_size_kb": float(effective_sparse_size_kb),
    }
    if baseline_sensitivity is not None:
        row["dangerous_class_degradation_ratio"] = dangerous_class_degradation_ratio(
            baseline_sensitivity=baseline_sensitivity,
            pruned_sensitivity=metrics["per_class_sensitivity"],
        )
    return row


def _resolve_sweep(recovery_cfg: dict[str, Any]) -> tuple[list[float], list[int], list[float], list[str]]:
    sweep_cfg = recovery_cfg.get("sweep") or {}
    sparsities = sweep_cfg.get("sparsities") or [float(recovery_cfg.get("sparsity", 0.5))]
    epochs_sweep = sweep_cfg.get("epochs") or [int(recovery_cfg.get("epochs", 5))]
    lrs = sweep_cfg.get("learning_rates") or [float(recovery_cfg.get("lr", 1e-5))]
    criteria = list(recovery_cfg.get("criteria", list(RECOVERY_CRITERIA[:3])))
    return (
        [float(s) for s in sparsities],
        [int(e) for e in epochs_sweep],
        [float(lr) for lr in lrs],
        criteria,
    )


def _dedupe_and_persist(log_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["seed", "model", "criterion", "sparsity", "recovery_epochs", "recovery_lr"],
        keep="last",
    )
    frame = frame.sort_values(
        ["model", "criterion", "sparsity", "recovery_epochs", "recovery_lr"]
    ).reset_index(drop=True)
    if is_main_process():
        frame.to_csv(log_path, index=False)
    return frame.to_dict(orient="records")


def _existing_key(rows: list[dict[str, Any]], key: dict[str, Any]) -> bool:
    for row in rows:
        if all(
            round(float(row[k]), 6) == round(float(v), 6) if isinstance(v, float)
            else str(row[k]) == str(v)
            for k, v in key.items()
            if k in row
        ):
            return True
    return False


def run(
    config_path: str,
    model_names: list[str] | None = None,
    seed_override: int | None = None,
) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    set_seed(seed)
    init_distributed()
    device = get_device()

    recovery_cfg = config.get("recovery", {})
    sparsities, epochs_sweep, lrs, criteria = _resolve_sweep(recovery_cfg)
    recovery_batch_size = int(recovery_cfg.get("batch_size", config["finetune"].get("batch_size", 64)))

    train_loader, val_loader, calibration_loader, class_weights = build_dataloaders(
        config,
        include_train=True,
        calibration_size=int(config["pruning"].get("calibration_size", 128)),
        train_batch_size=recovery_batch_size,
    )
    if train_loader is None or calibration_loader is None:
        raise RuntimeError("Recovery finetuning requires both training and calibration loaders.")

    model_names = model_names or [config["models"]["teacher"], config["models"]["student"]]
    checkpoint_dir = ensure_dir(config["logging"]["checkpoints_dir"])
    results_dir = ensure_dir(config["logging"]["results_dir"])
    masks_dir = ensure_dir(Path(checkpoint_dir) / "masks" / "recovery")
    log_path = Path(results_dir) / "recovery_finetune.csv"
    rows: list[dict[str, Any]] = []
    if log_path.exists():
        rows = pd.read_csv(log_path).to_dict(orient="records")

    for model_name in model_names:
        alias = model_alias(model_name)
        dense_model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
        dense_state = {name: tensor.detach().cpu().clone() for name, tensor in dense_model.state_dict().items()}

        baseline_metrics = evaluate_model(
            dense_model, val_loader, device, class_names=CLASS_NAMES, progress_desc=f"{alias} dense"
        )
        baseline_sensitivity = dict(baseline_metrics["per_class_sensitivity"])

        activation_norms, gradients = _collect_calibration_tensors(
            config, model_name, dense_model, calibration_loader, class_weights, device
        )
        target_layers = get_linear_layer_names(dense_model, exclude_keywords=config["pruning"]["exclude_layers"])
        del dense_model

        for criterion_name, sparsity, rec_epochs, lr in product(criteria, sparsities, epochs_sweep, lrs):
            row_key = {
                "seed": seed,
                "model": alias,
                "criterion": criterion_name,
                "sparsity": float(sparsity),
                "recovery_epochs": int(rec_epochs),
                "recovery_lr": float(lr),
            }
            if _existing_key(rows, row_key):
                continue

            checkpoint_path = (
                Path(checkpoint_dir)
                / f"recovery_{alias}_{criterion_name}_s{sparsity:.2f}_e{rec_epochs}_lr{lr:.0e}.pth"
            )

            model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
            model.load_state_dict(dense_state)
            scores = _score_layers(target_layers, criterion_name, activation_norms, gradients)
            masks = compute_global_masks(model, scores, float(sparsity))
            _freeze_masks(model, masks)
            if is_main_process():
                save_masks(
                    masks_dir / f"{alias}_{criterion_name}_s{sparsity:.2f}.pt", masks
                )

            ddp_model = maybe_wrap_ddp(model, device)
            optimizer = AdamW(
                ddp_model.parameters(),
                lr=float(lr),
                weight_decay=float(recovery_cfg.get("weight_decay", 0.05)),
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(rec_epochs)))
            criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device) if recovery_cfg.get("use_weighted_loss", True) else None
            )

            best_metric = float(baseline_metrics["balanced_accuracy"])
            best_epoch = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in unwrap_model(ddp_model).state_dict().items()}

            for epoch in range(int(rec_epochs)):
                if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)
                ddp_model.train()
                iterator = train_loader
                if is_main_process():
                    iterator = tqdm(
                        train_loader,
                        desc=f"{alias} {criterion_name} s={sparsity} ep{epoch + 1}/{rec_epochs} lr={lr}",
                        leave=False,
                    )
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
                    val_metrics = evaluate_model(
                        unwrap_model(ddp_model), val_loader, device, class_names=CLASS_NAMES,
                        progress_desc=f"{alias} {criterion_name} val",
                    )
                    if val_metrics["balanced_accuracy"] > best_metric:
                        best_metric = float(val_metrics["balanced_accuracy"])
                        best_epoch = epoch + 1
                        best_state = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in unwrap_model(ddp_model).state_dict().items()
                        }
                barrier()

            unwrap_model(ddp_model).load_state_dict(best_state)
            _remove_pruning_reparametrization(unwrap_model(ddp_model), masks)
            final_sparsity = _verify_mask_persistence(unwrap_model(ddp_model), masks)
            if abs(final_sparsity - sparsity) > 1e-4:
                raise RuntimeError(
                    f"Recovered model {alias}/{criterion_name} lost sparsity: expected {sparsity}, got {final_sparsity}."
                )

            if is_main_process():
                final_metrics = evaluate_model(
                    unwrap_model(ddp_model),
                    val_loader,
                    device,
                    class_names=CLASS_NAMES,
                    progress_desc=f"{alias} {criterion_name} recovered",
                )
                sizes = get_model_size_kb(unwrap_model(ddp_model), sparse=True)
                rows.append(
                    _result_row(
                        seed=seed,
                        model_name=model_name,
                        criterion_name=criterion_name,
                        sparsity=sparsity,
                        recovery_epochs=rec_epochs,
                        lr=lr,
                        metrics=final_metrics,
                        baseline_sensitivity=baseline_sensitivity,
                        size_kb=sizes["size_kb"],
                        disk_size_kb=sizes["disk_size_kb"],
                        effective_sparse_size_kb=sizes["effective_sparse_size_kb"],
                        best_epoch=best_epoch,
                    )
                )
                save_checkpoint(
                    checkpoint_path,
                    unwrap_model(ddp_model),
                    model_name=model_name,
                    seed=seed,
                    criterion=criterion_name,
                    sparsity=sparsity,
                    recovery_epochs=rec_epochs,
                    recovery_lr=lr,
                    best_epoch=best_epoch,
                    best_balanced_accuracy=float(final_metrics["balanced_accuracy"]),
                )
                rows = _dedupe_and_persist(log_path, rows)
            barrier()

    _dedupe_and_persist(log_path, rows)
    shutdown_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-pruning recovery finetuning sweep.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.models, seed_override=args.seed)


if __name__ == "__main__":
    main()
