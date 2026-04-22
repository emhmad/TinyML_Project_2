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
from tqdm import tqdm

from evaluation.metrics import CLASS_NAMES, evaluate_model
from evaluation.model_size import get_model_size_kb
from experiments.common import build_dataloaders, load_trained_model, model_alias
from models.load_models import get_layer_by_name, get_linear_layer_names
from pruning.hooks import ActivationCollector, GradientCollector
from pruning.masking import compute_global_masks
from pruning.scoring import magnitude_score, taylor_score, wanda_score
from utils.config import get_device, load_config
from utils.io import ensure_dir, save_checkpoint, save_masks


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
    model_name: str,
    criterion_name: str,
    sparsity: float,
    recovery_epochs: int,
    metrics: dict[str, Any],
    *,
    size_kb: float,
    best_epoch: int,
) -> dict[str, Any]:
    return {
        "model": model_alias(model_name),
        "criterion": criterion_name,
        "sparsity": float(sparsity),
        "recovery_epochs": int(recovery_epochs),
        "best_epoch": int(best_epoch),
        "overall_acc": metrics["overall_accuracy"],
        "balanced_acc": metrics["balanced_accuracy"],
        "akiec_sensitivity": metrics["per_class_sensitivity"]["akiec"],
        "bcc_sensitivity": metrics["per_class_sensitivity"]["bcc"],
        "bkl_sensitivity": metrics["per_class_sensitivity"]["bkl"],
        "df_sensitivity": metrics["per_class_sensitivity"]["df"],
        "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
        "nv_sensitivity": metrics["per_class_sensitivity"]["nv"],
        "vasc_sensitivity": metrics["per_class_sensitivity"]["vasc"],
        "size_kb": float(size_kb),
    }


def _load_one_shot_rows(results_dir: Path, model_names: list[str], sparsity: float) -> list[dict[str, Any]]:
    pruning_path = results_dir / "pruning_matrix.csv"
    if not pruning_path.exists():
        raise FileNotFoundError(
            f"Expected one-shot pruning results at {pruning_path}. Run E4 pruning matrix before recovery."
        )

    frame = pd.read_csv(pruning_path)
    aliases = {model_alias(name) for name in model_names}
    filtered = frame[
        frame["model"].isin(aliases)
        & frame["criterion"].isin(["magnitude", "wanda", "taylor"])
        & (frame["sparsity"].round(3) == round(float(sparsity), 3))
    ].copy()
    filtered = filtered.drop_duplicates(subset=["model", "criterion", "sparsity"], keep="last")
    filtered["recovery_epochs"] = 0
    filtered["best_epoch"] = 0
    rename_map = {
        "overall_acc": "overall_acc",
        "balanced_acc": "balanced_acc",
        "mel_sensitivity": "mel_sensitivity",
        "bcc_sensitivity": "bcc_sensitivity",
        "akiec_sensitivity": "akiec_sensitivity",
        "nv_sensitivity": "nv_sensitivity",
        "bkl_sensitivity": "bkl_sensitivity",
        "df_sensitivity": "df_sensitivity",
        "vasc_sensitivity": "vasc_sensitivity",
        "size_kb": "size_kb",
    }
    required = [
        "model",
        "criterion",
        "sparsity",
        "recovery_epochs",
        "best_epoch",
        "overall_acc",
        "balanced_acc",
        "akiec_sensitivity",
        "bcc_sensitivity",
        "bkl_sensitivity",
        "df_sensitivity",
        "mel_sensitivity",
        "nv_sensitivity",
        "vasc_sensitivity",
        "size_kb",
    ]
    return filtered.rename(columns=rename_map)[required].to_dict(orient="records")


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["model", "criterion", "sparsity", "recovery_epochs"], keep="last")
    return frame.sort_values(["model", "criterion", "recovery_epochs"]).reset_index(drop=True).to_dict(orient="records")


def _write_rows(log_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = _dedupe_rows(rows)
    pd.DataFrame(deduped).to_csv(log_path, index=False)
    return deduped


def _existing_recovery_row(
    rows: list[dict[str, Any]],
    alias: str,
    criterion_name: str,
    sparsity: float,
    recovery_epochs: int,
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row["model"] == alias
            and row["criterion"] == criterion_name
            and round(float(row["sparsity"]), 3) == round(float(sparsity), 3)
            and int(row["recovery_epochs"]) == int(recovery_epochs)
        ):
            return row
    return None


def run(config_path: str, model_names: list[str] | None = None) -> None:
    config = load_config(config_path)
    device = get_device()
    recovery_cfg = config.get("recovery", {})
    sparsity = float(recovery_cfg.get("sparsity", 0.5))
    recovery_epochs = int(recovery_cfg.get("epochs", 5))
    recovery_batch_size = int(recovery_cfg.get("batch_size", config["finetune"].get("batch_size", 64)))
    criteria = list(recovery_cfg.get("criteria", ["magnitude", "wanda", "taylor"]))

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

    log_path = Path(log_path)
    if log_path.exists():
        rows = pd.read_csv(log_path).to_dict(orient="records")
    else:
        rows = _load_one_shot_rows(Path(results_dir), model_names, sparsity)
        rows = _write_rows(log_path, rows)

    for model_name in model_names:
        alias = model_alias(model_name)
        dense_model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
        dense_state = {name: tensor.detach().cpu().clone() for name, tensor in dense_model.state_dict().items()}
        activation_norms, gradients = _collect_calibration_tensors(
            config,
            model_name,
            dense_model,
            calibration_loader,
            class_weights,
            device,
        )
        target_layers = get_linear_layer_names(dense_model, exclude_keywords=config["pruning"]["exclude_layers"])
        dense_model = None
        one_shot_lookup = {
            row["criterion"]: row
            for row in rows
            if row["model"] == alias and round(float(row["sparsity"]), 3) == round(sparsity, 3)
        }

        for criterion_name in criteria:
            checkpoint_path = Path(checkpoint_dir) / f"recovery_{alias}_{criterion_name}_50pct.pth"
            existing_row = _existing_recovery_row(rows, alias, criterion_name, sparsity, recovery_epochs)
            if existing_row is not None:
                continue

            if checkpoint_path.exists():
                recovered_model = load_trained_model(
                    config,
                    model_name,
                    device,
                    checkpoint_name=checkpoint_path.stem,
                )
                recovered_metrics = evaluate_model(
                    recovered_model,
                    val_loader,
                    device,
                    class_names=CLASS_NAMES,
                    progress_desc=f"{alias} {criterion_name} recovered",
                )
                rows.append(
                    _result_row(
                        model_name,
                        criterion_name,
                        sparsity,
                        recovery_epochs,
                        recovered_metrics,
                        size_kb=get_model_size_kb(recovered_model, sparse=True)["size_kb"],
                        best_epoch=recovery_epochs,
                    )
                )
                rows = _write_rows(log_path, rows)
                continue

            model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
            model.load_state_dict(dense_state)
            scores = _score_layers(target_layers, criterion_name, activation_norms, gradients)
            masks = compute_global_masks(model, scores, sparsity)
            _freeze_masks(model, masks)
            save_masks(masks_dir / f"{alias}_{criterion_name}_s{sparsity:.1f}.pt", masks)

            optimizer = AdamW(
                model.parameters(),
                lr=float(recovery_cfg.get("lr", 1e-5)),
                weight_decay=float(recovery_cfg.get("weight_decay", 0.05)),
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, recovery_epochs))
            criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device) if recovery_cfg.get("use_weighted_loss", True) else None
            )

            baseline_metrics = evaluate_model(
                model,
                val_loader,
                device,
                class_names=CLASS_NAMES,
                progress_desc=f"{alias} {criterion_name} one-shot",
            )
            best_metric = float(baseline_metrics["balanced_accuracy"])
            best_epoch = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

            for epoch in range(recovery_epochs):
                model.train()
                for images, labels in tqdm(
                    train_loader,
                    desc=f"{alias} {criterion_name} recovery {epoch + 1}/{recovery_epochs}",
                    leave=False,
                ):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                val_metrics = evaluate_model(
                    model,
                    val_loader,
                    device,
                    class_names=CLASS_NAMES,
                    progress_desc=f"{alias} {criterion_name} val",
                )
                if val_metrics["balanced_accuracy"] > best_metric:
                    best_metric = float(val_metrics["balanced_accuracy"])
                    best_epoch = epoch + 1
                    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

            model.load_state_dict(best_state)
            _remove_pruning_reparametrization(model, masks)
            final_sparsity = _verify_mask_persistence(model, masks)
            if abs(final_sparsity - sparsity) > 1e-4:
                raise RuntimeError(
                    f"Recovered model {alias}/{criterion_name} lost sparsity: expected {sparsity}, got {final_sparsity}."
                )

            final_metrics = evaluate_model(
                model,
                val_loader,
                device,
                class_names=CLASS_NAMES,
                progress_desc=f"{alias} {criterion_name} recovered",
            )
            final_size = get_model_size_kb(model, sparse=True)["size_kb"]
            rows.append(
                _result_row(
                    model_name,
                    criterion_name,
                    sparsity,
                    recovery_epochs,
                    final_metrics,
                    size_kb=final_size,
                    best_epoch=best_epoch,
                )
            )

            save_checkpoint(
                checkpoint_path,
                model,
                model_name=model_name,
                criterion=criterion_name,
                sparsity=sparsity,
                recovery_epochs=recovery_epochs,
                best_epoch=best_epoch,
                best_balanced_accuracy=float(final_metrics["balanced_accuracy"]),
            )
            rows = _write_rows(log_path, rows)

            one_shot_row = one_shot_lookup.get(criterion_name)
            if one_shot_row is not None and final_metrics["balanced_accuracy"] + 1e-6 < float(one_shot_row["balanced_acc"]):
                raise RuntimeError(
                    f"Recovery underperformed one-shot for {alias}/{criterion_name}. "
                    "This suggests the frozen-mask training path needs inspection."
                )

    _write_rows(log_path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-pruning recovery finetuning experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.models)


if __name__ == "__main__":
    main()
