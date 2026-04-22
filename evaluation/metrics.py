from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DANGEROUS_CLASSES = ["mel", "bcc", "akiec"]
SAFE_CLASSES = ["nv", "bkl", "df", "vasc"]


def _compute_specificity(conf_mat: np.ndarray, class_index: int) -> float:
    tp = conf_mat[class_index, class_index]
    fp = conf_mat[:, class_index].sum() - tp
    fn = conf_mat[class_index, :].sum() - tp
    tn = conf_mat.sum() - tp - fp - fn
    return float(tn / max(1, tn + fp))


def _operating_point_threshold(
    scores: np.ndarray,
    targets_binary: np.ndarray,
    fix: str,
    value: float,
) -> tuple[float, float, float]:
    """
    Sweep thresholds over `scores`, find the operating point matching the
    requested constraint, and return (threshold, sensitivity, specificity).

    `fix='sensitivity'` returns the threshold with the highest specificity
    whose sensitivity is >= `value`.
    `fix='specificity'` returns the threshold with the highest sensitivity
    whose specificity is >= `value`.
    """
    pos = targets_binary == 1
    neg = targets_binary == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan"), float("nan"), float("nan")

    # Sweep unique score values as candidate thresholds.
    candidates = np.unique(scores)
    candidates = np.concatenate([[-np.inf], candidates, [np.inf]])

    best = (float("nan"), -1.0, -1.0)
    for threshold in candidates:
        predicted_pos = scores >= threshold
        tp = int((predicted_pos & pos).sum())
        fn = int((~predicted_pos & pos).sum())
        fp = int((predicted_pos & neg).sum())
        tn = int((~predicted_pos & neg).sum())
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        if fix == "sensitivity":
            if sens >= value and spec > best[2]:
                best = (float(threshold), sens, spec)
        elif fix == "specificity":
            if spec >= value and sens > best[1]:
                best = (float(threshold), sens, spec)
        else:
            raise ValueError(f"Unknown operating-point constraint: {fix}")
    return best


def expected_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Top-label Expected Calibration Error.
    Equal-width binning over predicted confidence. Returns ECE in [0, 1].
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == targets).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(targets)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _safe_auroc(targets_binary: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(targets_binary)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(targets_binary, scores))
    except ValueError:
        return float("nan")


def _per_class_auroc(
    probs: np.ndarray, targets: np.ndarray, class_names: Iterable[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, name in enumerate(class_names):
        binary = (targets == idx).astype(np.int64)
        out[name] = _safe_auroc(binary, probs[:, idx])
    return out


def dangerous_class_degradation_ratio(
    baseline_sensitivity: dict[str, float],
    pruned_sensitivity: dict[str, float],
    dangerous: Iterable[str] = DANGEROUS_CLASSES,
    safe: Iterable[str] = SAFE_CLASSES,
) -> float:
    """
    W13 single-number safety metric:
        (mean sensitivity drop on {mel, akiec, bcc})
      / (mean sensitivity drop on {nv, bkl, df, vasc})

    A value > 1.0 means compression preferentially forgets dangerous
    classes. Returns inf if the safe-class drop is zero but the dangerous
    drop is positive (i.e., all damage is concentrated on dangerous
    classes), and nan if both drops are zero.
    """
    dangerous_drops = [
        baseline_sensitivity[c] - pruned_sensitivity[c]
        for c in dangerous
        if c in baseline_sensitivity and c in pruned_sensitivity
    ]
    safe_drops = [
        baseline_sensitivity[c] - pruned_sensitivity[c]
        for c in safe
        if c in baseline_sensitivity and c in pruned_sensitivity
    ]
    if not dangerous_drops or not safe_drops:
        return float("nan")
    mean_dangerous = float(np.mean(dangerous_drops))
    mean_safe = float(np.mean(safe_drops))
    if mean_safe == 0 and mean_dangerous == 0:
        return float("nan")
    if mean_safe == 0:
        return float("inf")
    return mean_dangerous / mean_safe


def evaluate_model(
    model,
    data_loader,
    device,
    class_names: list[str] | None = None,
    progress_desc: str | None = None,
    operating_points: Iterable[tuple[str, float]] | None = None,
    return_probs: bool = True,
) -> dict:
    """
    Evaluate a classifier and return both hard-label metrics and
    probability-based medical metrics (AUROC, ECE, specificity@90% sens,
    sensitivity@90% spec). Backwards-compatible with callers that only
    read overall_accuracy / balanced_accuracy / per_class_sensitivity.

    `operating_points`: iterable of (fix, value) pairs where `fix` is
    'sensitivity' or 'specificity'. Default: [('sensitivity', 0.9),
    ('specificity', 0.9)].
    """
    class_names = class_names or CLASS_NAMES
    operating_points = list(operating_points or [("sensitivity", 0.9), ("specificity", 0.9)])

    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    probs_chunks: list[np.ndarray] = []
    iterator = data_loader
    if progress_desc:
        iterator = tqdm(data_loader, desc=progress_desc, leave=False)

    with torch.no_grad():
        for images, labels in iterator:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits.float(), dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            predictions.extend(preds.tolist())
            targets.extend(labels.numpy().tolist())
            if return_probs:
                probs_chunks.append(probs)

    predictions_np = np.asarray(predictions)
    targets_np = np.asarray(targets)
    probs_np = np.concatenate(probs_chunks, axis=0) if probs_chunks else None
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

    results: dict = {
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

    if probs_np is not None:
        results["probabilities"] = probs_np

        per_class_auroc = _per_class_auroc(probs_np, targets_np, class_names)
        results["per_class_auroc"] = per_class_auroc
        finite = [v for v in per_class_auroc.values() if np.isfinite(v)]
        results["macro_auroc"] = float(np.mean(finite)) if finite else float("nan")
        results["melanoma_auroc"] = per_class_auroc.get("mel", float("nan"))

        # Operating-point analysis for the melanoma detector (clinical focus).
        mel_idx = class_names.index("mel") if "mel" in class_names else None
        op_points: dict[str, dict[str, float]] = {}
        if mel_idx is not None:
            mel_scores = probs_np[:, mel_idx]
            mel_binary = (targets_np == mel_idx).astype(np.int64)
            for fix, value in operating_points:
                threshold, sens, spec = _operating_point_threshold(
                    mel_scores, mel_binary, fix=fix, value=value
                )
                op_points[f"mel_{fix}@{value:g}"] = {
                    "threshold": float(threshold),
                    "sensitivity": float(sens),
                    "specificity": float(spec),
                }
        results["melanoma_operating_points"] = op_points

        results["ece_top_label"] = expected_calibration_error(probs_np, targets_np)

    return results
