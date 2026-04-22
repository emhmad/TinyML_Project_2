from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from evaluation.metrics import evaluate_model
from utils.distributed import (
    barrier,
    is_main_process,
    maybe_wrap_ddp,
    unwrap_model,
)
from utils.io import append_csv_row, save_checkpoint


class DistillationLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.class_weights = class_weights

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        temperature = self.temperature
        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature * temperature)
        hard_loss = F.cross_entropy(student_logits, labels, weight=self.class_weights)
        return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss


def train_distillation(
    teacher: nn.Module,
    student: nn.Module,
    train_loader,
    val_loader,
    config: dict[str, Any],
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    checkpoint_path: str | Path | None = None,
    log_path: str | Path | None = None,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    teacher = teacher.to(device)
    student = student.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    student = maybe_wrap_ddp(student, device)

    epochs = int(config.get("epochs", 20))
    optimizer = AdamW(
        student.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.05)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = DistillationLoss(
        temperature=float(config.get("temperature", 4.0)),
        alpha=float(config.get("alpha", 0.7)),
        class_weights=class_weights.to(device) if class_weights is not None else None,
    )

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_state = deepcopy(unwrap_model(student).state_dict())

    for epoch in range(epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        student.train()
        running_loss = 0.0
        running_examples = 0

        iterator = train_loader
        if is_main_process():
            iterator = tqdm(train_loader, desc=f"KD epoch {epoch + 1}/{epochs}", leave=False)
        for images, labels in iterator:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_examples += batch_size

        scheduler.step()
        barrier()

        if is_main_process():
            val_results = evaluate_model(unwrap_model(student), val_loader, device)
            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(1, running_examples),
                "balanced_accuracy": val_results["balanced_accuracy"],
                "melanoma_sensitivity": val_results["melanoma_sensitivity"],
                "melanoma_auroc": val_results.get("melanoma_auroc"),
                "macro_auroc": val_results.get("macro_auroc"),
                "ece_top_label": val_results.get("ece_top_label"),
            }
            history.append(epoch_result)
            if log_path is not None:
                append_csv_row(log_path, epoch_result)

            if val_results["balanced_accuracy"] > best_metric:
                best_metric = val_results["balanced_accuracy"]
                best_state = deepcopy(unwrap_model(student).state_dict())
                if checkpoint_path is not None:
                    save_checkpoint(
                        checkpoint_path,
                        unwrap_model(student),
                        best_balanced_accuracy=best_metric,
                        history=history,
                    )
        barrier()

    unwrap_model(student).load_state_dict(best_state)
    return unwrap_model(student), history
