"""
Attention-lesion overlap metrics (W12).

Given an attention heatmap (from `models.attention_rollout.AttentionRollout`)
and a binary lesion segmentation mask (e.g. ISIC 2017/2018 ground truth
masks at the original image resolution), compute:

  * IoU between the thresholded attention heatmap and the mask.
  * Pointing-game accuracy: does the heatmap argmax land inside the
    lesion mask?
  * Mass-in-mask: fraction of the total attention mass that falls
    inside the mask (thresholded-free, robust to binarisation choice).

Thresholds are configurable. Reporting both a single threshold (0.5) and
a per-image adaptive threshold (attention mean + std) matches the
evaluation used in attention-map grounding papers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OverlapResult:
    iou: float
    pointing_game_hit: bool
    mass_in_mask: float
    threshold: float
    attention_area: float
    mask_area: float

    def as_row(self) -> dict:
        return {
            "iou": float(self.iou),
            "pointing_game_hit": int(self.pointing_game_hit),
            "mass_in_mask": float(self.mass_in_mask),
            "threshold": float(self.threshold),
            "attention_area": float(self.attention_area),
            "mask_area": float(self.mask_area),
        }


def _resize_to_match(heatmap: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    h, w = target_shape
    if heatmap.shape == (h, w):
        return heatmap
    pil = Image.fromarray((heatmap * 255.0).clip(0, 255).astype(np.uint8))
    resized = pil.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def attention_overlap(
    heatmap: np.ndarray,
    mask: np.ndarray,
    threshold: float | str = 0.5,
) -> OverlapResult:
    """
    `heatmap`: 2D array in [0, 1].
    `mask`: 2D array with values {0, 1} — the lesion segmentation.
    `threshold`: scalar in [0, 1], or 'otsu' for Otsu thresholding, or
        'adaptive' for `mean + std` of the heatmap.

    Both arrays are aligned to the mask's shape before computing the
    overlap, so callers don't have to upsample/downsample themselves.
    """
    mask = mask.astype(np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.max() > 1.0:
        mask = (mask > 0).astype(np.float32)

    heatmap = _resize_to_match(heatmap, mask.shape)

    if threshold == "adaptive":
        t_val = float(heatmap.mean() + heatmap.std())
    elif threshold == "otsu":
        t_val = _otsu(heatmap)
    else:
        t_val = float(threshold)

    attn_binary = (heatmap >= t_val).astype(np.float32)
    mask_binary = (mask >= 0.5).astype(np.float32)

    intersection = float((attn_binary * mask_binary).sum())
    union = float(np.maximum(attn_binary, mask_binary).sum())
    iou = intersection / union if union > 0 else float("nan")

    peak_idx = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    pointing_hit = bool(mask_binary[peak_idx] > 0.5)

    total_mass = float(heatmap.sum())
    inside_mass = float((heatmap * mask_binary).sum())
    mass_in_mask = inside_mass / total_mass if total_mass > 0 else float("nan")

    return OverlapResult(
        iou=iou,
        pointing_game_hit=pointing_hit,
        mass_in_mask=mass_in_mask,
        threshold=t_val,
        attention_area=float(attn_binary.mean()),
        mask_area=float(mask_binary.mean()),
    )


def _otsu(values: np.ndarray) -> float:
    """Otsu threshold on a normalised heatmap in [0, 1]."""
    flat = values.reshape(-1)
    hist, bin_edges = np.histogram(flat, bins=256, range=(0.0, 1.0))
    if hist.sum() == 0:
        return 0.5
    probabilities = hist / hist.sum()
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    weights = np.cumsum(probabilities)
    means = np.cumsum(probabilities * bin_centers)
    global_mean = means[-1]
    between = (global_mean * weights - means) ** 2 / np.maximum(weights * (1 - weights), 1e-12)
    return float(bin_centers[int(np.argmax(between))])


def summarise_overlap(results: Iterable[OverlapResult]) -> dict:
    """
    Summarise a list of per-image OverlapResult rows: mean and median
    IoU, pointing-game accuracy, mean mass-in-mask, and counts.
    """
    rows = list(results)
    if not rows:
        return {
            "n": 0,
            "mean_iou": float("nan"),
            "median_iou": float("nan"),
            "pointing_game_accuracy": float("nan"),
            "mean_mass_in_mask": float("nan"),
        }
    ious = np.array([r.iou for r in rows if np.isfinite(r.iou)], dtype=np.float64)
    hits = np.array([int(r.pointing_game_hit) for r in rows], dtype=np.float64)
    masses = np.array([r.mass_in_mask for r in rows if np.isfinite(r.mass_in_mask)], dtype=np.float64)
    return {
        "n": len(rows),
        "mean_iou": float(ious.mean()) if ious.size else float("nan"),
        "median_iou": float(np.median(ious)) if ious.size else float("nan"),
        "pointing_game_accuracy": float(hits.mean()),
        "mean_mass_in_mask": float(masses.mean()) if masses.size else float("nan"),
    }
