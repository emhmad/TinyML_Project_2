"""
Structured N:M sparsity masks (W8 follow-through).

Unstructured masks don't actually shrink the file a NumPy-serialised
dense tensor takes on disk, and they don't light up the Ampere sparse
tensor cores. 2:4 (and general N:M) sparsity does both. This module
generates N:M masks from any scalar score tensor and applies them using
the same `apply_masks` / `compute_mask` contract as the unstructured
pipeline, so callers don't need two code paths.

Mask shape contract:
  - For Linear weights [out, in]: groups are contiguous along the input
    dimension. Inside each group of M elements, we keep the top N by
    score. This matches the layout that sparse tensor-core kernels
    expect (input-major grouping).
  - For Conv2d weights [out, in, kH, kW] with groups == 1: we flatten
    the per-output row to [in * kH * kW] and apply N:M along that axis.
    Depthwise convs (groups == in_channels) keep their weights dense —
    N:M on a single-element group is trivially 1:1 and we skip.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.load_models import get_layer_by_name
from pruning.masking import apply_masks


@dataclass(frozen=True)
class NMPattern:
    n_keep: int
    m_group: int

    def __post_init__(self) -> None:
        if self.n_keep <= 0 or self.m_group <= 0:
            raise ValueError(f"n_keep/m_group must be positive, got {self}")
        if self.n_keep >= self.m_group:
            raise ValueError(f"n_keep ({self.n_keep}) must be < m_group ({self.m_group})")

    @property
    def effective_sparsity(self) -> float:
        return 1.0 - self.n_keep / self.m_group


def _linear_nm_mask(score: torch.Tensor, pattern: NMPattern) -> torch.Tensor:
    out_dim, in_dim = score.shape
    m = pattern.m_group
    if in_dim % m != 0:
        # Zero-pad the score to a multiple of m_group, mask on the padded
        # shape, then slice back. Keeps one code path for ragged dims.
        pad = m - (in_dim % m)
        padded = torch.nn.functional.pad(score, (0, pad), value=float("-inf"))
        padded_mask = _linear_nm_mask(padded, pattern)
        return padded_mask[:, :in_dim].contiguous()
    groups = score.view(out_dim, in_dim // m, m)
    keep_count = pattern.n_keep
    topk_idx = groups.abs().topk(keep_count, dim=-1).indices
    mask = torch.zeros_like(groups)
    mask.scatter_(-1, topk_idx, 1.0)
    return mask.view(out_dim, in_dim)


def _conv_nm_mask(score: torch.Tensor, pattern: NMPattern) -> torch.Tensor:
    out_c, in_c, kh, kw = score.shape
    flat = score.reshape(out_c, in_c * kh * kw)
    mask_flat = _linear_nm_mask(flat, pattern)
    return mask_flat.reshape(out_c, in_c, kh, kw)


def compute_nm_mask(
    score: torch.Tensor,
    pattern: NMPattern,
    layer: nn.Module | None = None,
) -> torch.Tensor:
    """
    Returns a 0/1 mask with the same dtype/shape as `score` enforcing
    the N:M pattern. `layer` is optional — only consulted to skip
    depthwise convs where N:M is degenerate.
    """
    if isinstance(layer, nn.Conv2d) and layer.groups == layer.in_channels and layer.groups > 1:
        return torch.ones_like(score)
    if score.dim() == 2:
        return _linear_nm_mask(score, pattern)
    if score.dim() == 4:
        return _conv_nm_mask(score, pattern)
    raise ValueError(f"Unsupported score rank {score.dim()} for N:M masking.")


def compute_nm_masks(
    model: nn.Module,
    scores: dict[str, torch.Tensor],
    pattern: NMPattern,
) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    for layer_name, score in scores.items():
        module = get_layer_by_name(model, layer_name)
        masks[layer_name] = compute_nm_mask(score, pattern, layer=module)
    return masks


def apply_nm_pruning(
    model: nn.Module,
    scores: dict[str, torch.Tensor],
    pattern: NMPattern,
) -> dict[str, torch.Tensor]:
    masks = compute_nm_masks(model, scores, pattern)
    apply_masks(model, masks)
    return masks


def verify_nm_pattern(mask: torch.Tensor, pattern: NMPattern) -> bool:
    """
    Sanity-check that every group of M along the last axis has exactly
    `n_keep` nonzeros. Used by the structured-sparsity experiment to
    verify the mask before it's written to disk.
    """
    last = mask.shape[-1]
    if last % pattern.m_group != 0:
        return False
    reshaped = mask.view(*mask.shape[:-1], last // pattern.m_group, pattern.m_group)
    return bool((reshaped.sum(dim=-1) == pattern.n_keep).all().item())
