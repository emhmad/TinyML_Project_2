from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DANGEROUS_CLASSES = ["mel", "bcc", "akiec"]


def _compute_specificity(conf_mat: np.ndarray, class_index: int) -> float:
    tp = conf_mat[class_index, class_index]
    fp = conf_mat[:, class_index].sum() - tp
    fn = conf_mat[class_index, :].sum() - tp
    tn = conf_mat.sum() - tp - fp - fn
    return float(tn / max(1, tn + fp))


def evaluate_model(model, data_loader, device, class_names=None, progress_desc: str | None = None):
    class_names = class_names or CLASS_NAMES
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    iterator = data_loader
    if progress_desc:
        iterator = tqdm(data_loader, desc=progress_desc, leave=False)

    with torch.no_grad():
        for images, labels in iterator:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            predictions.extend(preds.tolist())
            targets.extend(labels.numpy().tolist())

    predictions_np = np.asarray(predictions)
    targets_np = np.asarray(targets)
    conf_mat = confusion_matrix(targets_np, predictions_np, labels=list(range(len(class_names))))
    report = classification_report(
        targets_np,
        predictions_np,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_sensitivity = {name: float(report[name]["recall"]) for name in class_names}
    per_class_precision = {name: float(report[name]["precision"]) for name in class_names}
    per_class_f1 = {name: float(report[name]["f1-score"]) for name in class_names}
    per_class_specificity = {
        name: _compute_specificity(conf_mat, index) for index, name in enumerate(class_names)
    }

    return {
        "overall_accuracy": float(accuracy_score(targets_np, predictions_np)),
        "balanced_accuracy": float(balanced_accuracy_score(targets_np, predictions_np)),
        "per_class_sensitivity": per_class_sensitivity,
        "per_class_specificity": per_class_specificity,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "confusion_matrix": conf_mat,
        "melanoma_sensitivity": per_class_sensitivity.get("mel", 0.0),
        "bcc_sensitivity": per_class_sensitivity.get("bcc", 0.0),
        "predictions": predictions_np,
        "ground_truth": targets_np,
    }
