"""
Per-layer activation-distribution statistics used to test the Wanda
over-protection hypothesis (W4).

The hypothesis in the paper: distilled models concentrate information
into a few high-magnitude "outlier" activation channels, which Wanda
then refuses to prune — sacrificing smaller channels that were actually
doing useful work for rare/dangerous classes.

This module gives the evidence. For each target linear layer we collect:
  - kurtosis of the input activation magnitudes (per-channel RMS)
  - top-k concentration: fraction of total activation mass in the top-k
    channels (default k=5% of channels)
  - outlier ratio: fraction of channels whose RMS exceeds median + 3·MAD

Correlating those per-layer stats with Wanda's per-layer damage gives
the scientific evidence the reviewer is asking for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class LayerActivationStats:
    layer_name: str
    n_channels: int
    rms_mean: float
    rms_std: float
    kurtosis: float
    top1_concentration: float
    top5pct_concentration: float
    outlier_ratio: float

    def as_row(self) -> dict:
        return {
            "layer": self.layer_name,
            "n_channels": int(self.n_channels),
            "rms_mean": float(self.rms_mean),
            "rms_std": float(self.rms_std),
            "kurtosis": float(self.kurtosis),
            "top1_concentration": float(self.top1_concentration),
            "top5pct_concentration": float(self.top5pct_concentration),
            "outlier_ratio": float(self.outlier_ratio),
        }


def _kurtosis(values: torch.Tensor, eps: float = 1e-12) -> float:
    """Excess kurtosis (Fisher definition, so Gaussian => 0)."""
    mean = values.mean()
    centred = values - mean
    var = centred.pow(2).mean().clamp(min=eps)
    m4 = centred.pow(4).mean()
    return float((m4 / var.pow(2)).item() - 3.0)


def _outlier_ratio(values: torch.Tensor, z: float = 3.0, eps: float = 1e-12) -> float:
    median = values.median()
    mad = (values - median).abs().median().clamp(min=eps)
    threshold = median + z * mad
    return float((values > threshold).float().mean().item())


def compute_activation_stats(
    activation_norms: dict[str, torch.Tensor],
    layer_names: Iterable[str] | None = None,
    top_fraction: float = 0.05,
) -> list[LayerActivationStats]:
    """
    `activation_norms[layer]` is the per-input-channel RMS activation
    vector produced by `pruning.hooks.ActivationCollector`. From that we
    compute the layer-level summary stats needed for the Wanda hypothesis.
    """
    layer_names = list(layer_names) if layer_names is not None else list(activation_norms.keys())
    stats: list[LayerActivationStats] = []
    for name in layer_names:
        if name not in activation_norms:
            continue
        norms = activation_norms[name].detach().float().cpu()
        n_channels = int(norms.numel())
        if n_channels == 0:
            continue
        sorted_desc, _ = torch.sort(norms, descending=True)
        total = sorted_desc.sum().clamp(min=1e-12)
        k = max(1, int(round(top_fraction * n_channels)))
        top_k_mass = sorted_desc[:k].sum() / total
        top1_mass = sorted_desc[:1].sum() / total
        stats.append(
            LayerActivationStats(
                layer_name=name,
                n_channels=n_channels,
                rms_mean=float(norms.mean().item()),
                rms_std=float(norms.std(unbiased=False).item()),
                kurtosis=_kurtosis(norms),
                top1_concentration=float(top1_mass.item()),
                top5pct_concentration=float(top_k_mass.item()),
                outlier_ratio=_outlier_ratio(norms),
            )
        )
    return stats


def stats_to_frame(stats: Iterable[LayerActivationStats]) -> pd.DataFrame:
    return pd.DataFrame([s.as_row() for s in stats])


def correlate_stats_with_damage(
    stats_frame: pd.DataFrame,
    damage_by_layer: dict[str, float],
    stat_columns: Iterable[str] = ("kurtosis", "top5pct_concentration", "outlier_ratio"),
) -> pd.DataFrame:
    """
    `damage_by_layer[layer]` must be a scalar "damage" measurement from
    the Wanda-specific ablation (for example: balanced-accuracy drop when
    pruning only that layer at 50% sparsity using Wanda scoring).
    Returns a tidy DataFrame with Pearson and Spearman correlations
    between each selected activation statistic and per-layer damage.
    """
    merged = stats_frame.copy()
    merged["damage"] = merged["layer"].map(damage_by_layer)
    merged = merged.dropna(subset=["damage"])
    rows: list[dict] = []
    for col in stat_columns:
        if col not in merged.columns or len(merged) < 3:
            continue
        x = merged[col].to_numpy(dtype=float)
        y = merged["damage"].to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            pearson = float("nan")
        else:
            pearson = float(np.corrcoef(x, y)[0, 1])
        # Spearman via rank transform to avoid pulling in scipy here.
        x_rank = pd.Series(x).rank().to_numpy()
        y_rank = pd.Series(y).rank().to_numpy()
        if np.std(x_rank) == 0 or np.std(y_rank) == 0:
            spearman = float("nan")
        else:
            spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
        rows.append({
            "statistic": col,
            "n_layers": int(len(merged)),
            "pearson_r": pearson,
            "spearman_rho": spearman,
        })
    return pd.DataFrame(rows)


def layerwise_wanda_damage(
    model: nn.Module,
    target_layers: list[tuple[str, nn.Module]],
    scores: dict[str, torch.Tensor],
    val_loader,
    device: torch.device,
    sparsity: float,
    baseline_balanced_acc: float,
) -> dict[str, float]:
    """
    Per-layer single-layer pruning damage: for each layer we reset the
    model, prune only that layer with the supplied scoring, and measure
    the resulting balanced-accuracy drop against `baseline_balanced_acc`.
    The mapping returned is what gets fed into
    `correlate_stats_with_damage`.

    Heavy operation — budget roughly len(target_layers) full validation
    passes. Intended to run once per model, not per seed.
    """
    from copy import deepcopy

    from evaluation.metrics import evaluate_model
    from pruning.masking import apply_masks, compute_mask

    damage: dict[str, float] = {}
    for layer_name, _module in target_layers:
        copied = deepcopy(model).to(device)
        mask = compute_mask(scores[layer_name], sparsity=sparsity)
        apply_masks(copied, {layer_name: mask})
        metrics = evaluate_model(copied, val_loader, device, return_probs=False)
        damage[layer_name] = float(baseline_balanced_acc - metrics["balanced_accuracy"])
        del copied
    return damage
