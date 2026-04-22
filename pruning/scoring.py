"""
Weight-importance scoring functions for one-shot unstructured pruning.

The four original criteria (magnitude, Wanda, Taylor, random) are kept
unchanged; this module adds two external baselines so the paper has
something real to compare `nonuniform + sensitivity` against:

  * skewness-guided (Paxton et al.) — layer-wise rescaling of magnitude
    scores by the skewness of the weight distribution. Layers whose
    weights are heavily skewed (long tails of large values) are considered
    more important per unit magnitude.
  * X-Pruner-style explainability — uses per-channel activation-vs-logit
    sensitivity as a proxy for the "explanatory contribution" of each
    output channel. For a Linear layer with input activation ||a_i|| and
    output sensitivity s_j = ||∂logit/∂out_j||, the score for weight W[j,i]
    is |W[j,i]| * ||a_i|| * s_j. Reduces to Wanda when s_j = 1 and to
    Taylor (up to a factor) when ||a_i|| = 1, so it sits between the two
    in the biased-estimator/activation-aware axis.

Both take whatever layer-level statistics they need via keyword arguments
so callers can reuse the calibration tensors already cached by
experiments/e3_calibration.py.
"""
from __future__ import annotations

import torch
from torch import Tensor


def magnitude_score(weight: Tensor) -> Tensor:
    return weight.abs()


def wanda_score(weight: Tensor, activation_norm: Tensor) -> Tensor:
    return weight.abs() * activation_norm.to(weight.device).unsqueeze(0)


def taylor_score(weight: Tensor, gradient: Tensor) -> Tensor:
    return (weight * gradient).abs()


def random_score(weight: Tensor, seed: int = 42) -> Tensor:
    generator = torch.Generator(device=weight.device)
    generator.manual_seed(seed)
    return torch.rand(weight.shape, generator=generator, device=weight.device, dtype=weight.dtype)


def _weight_skewness(weight: Tensor, eps: float = 1e-8) -> float:
    """
    Population skewness of |W|. Larger => longer right tail => more
    concentration of magnitude in a few weights. Paxton et al. argue
    that heavier-tailed layers are more informative per magnitude unit
    and should be protected during pruning.
    """
    flat = weight.detach().abs().flatten().float()
    mean = flat.mean()
    centred = flat - mean
    var = centred.pow(2).mean().clamp(min=eps)
    std = var.sqrt()
    skew = (centred / std).pow(3).mean()
    return float(skew.item())


def skewness_score(weight: Tensor, scale_mode: str = "exp") -> Tensor:
    """
    Paxton-style skewness-guided magnitude. The score per weight is
    |W| * g(skew(|W|)), where g controls how strongly a heavy-tailed
    layer is boosted:
      - `scale_mode='exp'` (default): g(s) = exp(s)
      - `scale_mode='softplus'`: g(s) = log1p(exp(s))
      - `scale_mode='identity'`: g(s) = 1 + max(0, s)

    When several layers are compared via a global threshold, this
    lifts high-skew layers' scores uniformly, so fewer of their weights
    are pruned — which is the behaviour Paxton et al. report for skin
    lesion CNNs.
    """
    skew = _weight_skewness(weight)
    if scale_mode == "exp":
        scale = float(torch.tensor(skew).exp().item())
    elif scale_mode == "softplus":
        scale = float(torch.nn.functional.softplus(torch.tensor(skew)).item())
    elif scale_mode == "identity":
        scale = 1.0 + max(0.0, skew)
    else:
        raise ValueError(f"Unknown scale_mode={scale_mode}")
    return weight.abs() * scale


def xpruner_score(
    weight: Tensor,
    activation_norm: Tensor,
    output_sensitivity: Tensor,
) -> Tensor:
    """
    X-Pruner-style explainability score for a Linear layer.
    weight: [out, in]
    activation_norm: [in]  — RMS of the pre-layer activation
    output_sensitivity: [out] — a non-negative per-output saliency
        (e.g., ||∂class_logit / ∂out_j||). Callers that do not have
        class-specific gradients can pass the absolute row-wise sum of
        the weight's gradient as a stand-in.

    Score = |W[j, i]| * activation_norm[i] * output_sensitivity[j].
    """
    if weight.dim() != 2:
        raise ValueError(
            f"xpruner_score expects a 2D weight tensor, got shape {tuple(weight.shape)}"
        )
    in_term = activation_norm.to(weight.device).unsqueeze(0)   # [1, in]
    out_term = output_sensitivity.to(weight.device).unsqueeze(1)  # [out, 1]
    return weight.abs() * in_term * out_term


def sparsegpt_pseudo_score(
    weight: Tensor,
    activation_norm: Tensor,
    damping: float = 1e-4,
) -> Tensor:
    """
    A lightweight approximation to SparseGPT's OBS-style criterion:
        score[j, i] = W[j, i]^2 / (H_ii + damping)
    where H is the Hessian approximation. We substitute the Wanda-style
    activation second moment as a cheap proxy for diag(H), so this
    function can run without the full Hessian machinery.

    Not a drop-in replacement for SparseGPT, but gives reviewers a
    diagonal-OBS-flavoured baseline on the same calibration data.
    """
    diag_h = activation_norm.to(weight.device).pow(2) + damping
    return weight.pow(2) / diag_h.unsqueeze(0)
