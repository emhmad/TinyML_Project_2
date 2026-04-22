from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from evaluation.metrics import evaluate_model
from experiments.common import build_dataloaders, is_writer_process, model_alias
from models.load_models import load_deit_model
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.distributed import (
    barrier,
    init_distributed,
    is_main_process,
    maybe_wrap_ddp,
    shutdown_distributed,
    unwrap_model,
    world_size,
)
from utils.io import append_csv_row, ensure_dir, save_checkpoint
from utils.seed import set_seed


def _train_one_model(
    model_name: str,
    config: dict[str, Any],
    device: torch.device,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
) -> None:
    base_model = load_deit_model(
        model_name=model_name,
        num_classes=int(config["models"].get("num_classes", 7)),
        pretrained=bool(config["models"].get("pretrained", True)),
    )
    model = maybe_wrap_ddp(base_model, device)

    finetune_cfg = config["finetune"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(finetune_cfg.get("lr", 1e-4)),
        weight_decay=float(finetune_cfg.get("weight_decay", 0.05)),
    )
    epochs = max(1, int(finetune_cfg.get("epochs", 20)))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if finetune_cfg.get("use_weighted_loss", True) else None
    )

    checkpoint_dir = ensure_dir(config["logging"]["checkpoints_dir"])
    results_dir = ensure_dir(config["logging"]["results_dir"])
    alias = model_alias(model_name)
    checkpoint_path = checkpoint_dir / f"{alias}_ham10000.pth"
    history_path = results_dir / "finetune_history.csv"
    seed = resolve_seed(config)

    best_metric = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        running_examples = 0
        running_correct = 0

        iterator = train_loader
        if is_main_process():
            iterator = tqdm(train_loader, desc=f"{alias} epoch {epoch + 1}", leave=False)

        for images, labels in iterator:
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
        barrier()

        if is_main_process():
            val_results = evaluate_model(unwrap_model(model), val_loader, device)
            append_csv_row(
                history_path,
                {
                    "seed": seed,
                    "model": alias,
                    "epoch": epoch + 1,
                    "train_loss": running_loss / max(1, running_examples),
                    "train_accuracy": running_correct / max(1, running_examples),
                    "val_balanced_accuracy": val_results["balanced_accuracy"],
                    "val_melanoma_sensitivity": val_results["melanoma_sensitivity"],
                    "val_melanoma_auroc": val_results.get("melanoma_auroc"),
                    "val_macro_auroc": val_results.get("macro_auroc"),
                    "val_ece": val_results.get("ece_top_label"),
                },
            )

            if val_results["balanced_accuracy"] > best_metric:
                best_metric = float(val_results["balanced_accuracy"])
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in unwrap_model(model).state_dict().items()
                }
                save_checkpoint(
                    checkpoint_path,
                    unwrap_model(model),
                    model_name=model_name,
                    seed=seed,
                    best_balanced_accuracy=best_metric,
                )
        barrier()

    if is_main_process() and best_state is not None:
        unwrap_model(model).load_state_dict(best_state)
        save_checkpoint(
            checkpoint_path,
            unwrap_model(model),
            model_name=model_name,
            seed=seed,
            best_balanced_accuracy=best_metric,
        )
    barrier()


def run(config_path: str, model_names: list[str] | None = None, seed_override: int | None = None) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    set_seed(seed)
    init_distributed()
    device = get_device()

    train_loader, val_loader, _, class_weights = build_dataloaders(config, include_train=True)
    model_names = model_names or [config["models"]["student"], config["models"]["teacher"]]
    for model_name in model_names:
        _train_one_model(model_name, config, device, train_loader, val_loader, class_weights)

    shutdown_distributed()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DeiT models on HAM10000.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override config seed for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.models, seed_override=args.seed)


if __name__ == "__main__":
    main()
