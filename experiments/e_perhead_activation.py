"""
Per-head class-conditional activation analysis (P1.3).

Depth-2 of the W4 falsified hypothesis. The original e_activation_stats
probe operated at the linear-layer level (n=48 layers on DeiT-Small) and
found |r| < 0.11 between activation outliers and Wanda damage. This
script re-runs the same statistics at finer granularity:

  * Per attention head (DeiT-Small: 12 blocks * 6 heads = 72 heads;
    DeiT-Tiny: 12 blocks * 3 heads = 36 heads), separately for the
    Q, K, V projections.
  * Conditioned on dangerous-only and safe-only sample batches, so we
    can detect class-conditional outliers — exactly the hypothesis
    flagged but untested at tex:430-432.

Outputs `perhead_activation_stats.csv`:

    model, seed, layer, layer_idx, head, head_dim, projection {q,k,v},
    class_group {dangerous, safe, all},
    n_samples,
    kurtosis_pre_softmax, top5pct_concentration, outlier_ratio,
    rms_norm

A class-conditional spike (e.g., dangerous-class kurtosis >> safe-class
on heads in the early layers) would be the missing mechanism for the
DCR inversion. A null result is also publishable — it constrains the
search space.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.dataset import CLASS_NAMES
from evaluation.metrics import DANGEROUS_CLASSES, SAFE_CLASSES
from experiments.common import build_dataloaders, load_trained_model, model_alias
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.io import ensure_dir
from utils.seed import set_seed


def _kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return float("nan")
    mu = values.mean()
    sigma = values.std()
    if sigma == 0:
        return float("nan")
    z = (values - mu) / sigma
    return float((z**4).mean() - 3.0)  # Fisher-Pearson excess kurtosis


def _top5pct_concentration(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    abs_values = np.abs(values)
    n_top = max(1, int(0.05 * abs_values.size))
    top_sum = np.partition(abs_values, -n_top)[-n_top:].sum()
    total = abs_values.sum()
    return float(top_sum / total) if total > 0 else float("nan")


def _outlier_ratio(values: np.ndarray, threshold_sigmas: float = 3.0) -> float:
    if values.size == 0:
        return float("nan")
    abs_values = np.abs(values)
    threshold = abs_values.mean() + threshold_sigmas * abs_values.std()
    return float((abs_values > threshold).mean()) if threshold > 0 else float("nan")


def _stats_for_tensor(tensor: torch.Tensor) -> dict[str, float]:
    arr = tensor.detach().float().cpu().numpy().ravel()
    return {
        "kurtosis": _kurtosis(arr),
        "top5pct_concentration": _top5pct_concentration(arr),
        "outlier_ratio": _outlier_ratio(arr),
        "rms_norm": float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan"),
    }


def _find_attention_blocks(model: nn.Module) -> list[tuple[int, str, nn.Module]]:
    """
    Walk a timm ViT/DeiT model and return [(block_idx, block_name, attn_module), ...].
    Looks for modules whose name ends with `.attn` and that expose a `qkv` Linear
    plus `num_heads` and `head_dim` attributes (the standard timm attention block).
    """
    blocks: list[tuple[int, str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name.endswith(".attn"):
            continue
        qkv = getattr(module, "qkv", None)
        if not isinstance(qkv, nn.Linear):
            continue
        if not hasattr(module, "num_heads") or not hasattr(module, "head_dim"):
            continue
        # Block index from the name like "blocks.0.attn"
        parts = name.split(".")
        block_idx = -1
        for token, nxt in zip(parts, parts[1:]):
            if token == "blocks":
                try:
                    block_idx = int(nxt)
                except ValueError:
                    block_idx = -1
                break
        blocks.append((block_idx, name, module))
    blocks.sort(key=lambda x: x[0])
    return blocks


class _PerHeadHook:
    """
    Captures the qkv pre-softmax activations and splits them per head and
    per Q/K/V projection. Produces a list of dicts (one per head x proj
    x batch) so the caller can aggregate stats either over class-mixed
    or class-conditional subsets without re-running the model.

    Storage is on CPU to keep GPU memory low. We only keep the raw
    head-projection slabs so post-hoc class filtering is possible.
    """

    def __init__(self, attn_module: nn.Module):
        self.attn = attn_module
        self.handle: torch.utils.hooks.RemovableHandle | None = None
        self.captures: list[torch.Tensor] = []

    def _hook(self, module, inputs, output):
        # qkv output shape: [B, N, 3 * num_heads * head_dim]
        # We split into Q, K, V then per-head.
        out = output.detach().to("cpu", non_blocking=True)
        self.captures.append(out)

    def attach(self) -> None:
        self.handle = self.attn.qkv.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _collect_class_conditional(
    model: nn.Module,
    val_loader,
    blocks: list[tuple[int, str, nn.Module]],
    device: torch.device,
    max_batches_per_group: int = 8,
) -> dict[str, list[tuple[int, str, nn.Module, torch.Tensor]]]:
    """
    Run inference and capture qkv outputs for each attention block,
    accumulated separately for dangerous-class samples, safe-class
    samples, and all samples. Returns:
        { class_group: [(block_idx, block_name, attn_module, qkv_outputs), ...] }
    """
    dangerous_idx = {CLASS_NAMES.index(c) for c in DANGEROUS_CLASSES if c in CLASS_NAMES}
    safe_idx = {CLASS_NAMES.index(c) for c in SAFE_CLASSES if c in CLASS_NAMES}

    hooks = [(_PerHeadHook(attn), block_idx, block_name, attn) for block_idx, block_name, attn in blocks]
    for h, *_ in hooks:
        h.attach()

    captures_per_group: dict[str, dict[int, torch.Tensor]] = {
        "all": {idx: torch.zeros(0) for idx, _, _ in blocks},
        "dangerous": {idx: torch.zeros(0) for idx, _, _ in blocks},
        "safe": {idx: torch.zeros(0) for idx, _, _ in blocks},
    }

    model.eval()
    with torch.no_grad():
        seen = 0
        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(device, non_blocking=True)
            for h, *_ in hooks:
                h.captures.clear()
            _ = model(batch_images)
            labels_np = batch_labels.numpy()
            mask_dangerous = np.isin(labels_np, list(dangerous_idx))
            mask_safe = np.isin(labels_np, list(safe_idx))

            for h, block_idx, _, _ in hooks:
                if not h.captures:
                    continue
                qkv = h.captures[0]  # [B, N, 3 * H * D]
                # Pool the spatial dim N to keep memory bounded; we only need
                # per-sample per-head distributions for the stats.
                # Mean over tokens preserves outlier structure across heads.
                pooled = qkv  # keep full [B, N, F]; we'll reshape below
                # Slice by class group
                d_slab = pooled[mask_dangerous] if mask_dangerous.any() else None
                s_slab = pooled[mask_safe] if mask_safe.any() else None

                def _append(prev: torch.Tensor, addition: torch.Tensor | None) -> torch.Tensor:
                    if addition is None or addition.numel() == 0:
                        return prev
                    if prev.numel() == 0:
                        return addition.clone()
                    return torch.cat([prev, addition], dim=0)

                captures_per_group["all"][block_idx] = _append(captures_per_group["all"][block_idx], pooled)
                captures_per_group["dangerous"][block_idx] = _append(captures_per_group["dangerous"][block_idx], d_slab)
                captures_per_group["safe"][block_idx] = _append(captures_per_group["safe"][block_idx], s_slab)

            seen += 1
            if seen >= max_batches_per_group:
                break

    for h, *_ in hooks:
        h.detach()

    # Build the return structure with module references for downstream slicing.
    structured: dict[str, list[tuple[int, str, nn.Module, torch.Tensor]]] = {
        group: [
            (block_idx, block_name, attn, captures_per_group[group][block_idx])
            for block_idx, block_name, attn in blocks
        ]
        for group in captures_per_group
    }
    return structured


def _stats_per_head_from_qkv(
    qkv: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> list[dict]:
    """
    qkv: [B, N, 3 * num_heads * head_dim]
    Returns one dict per (projection, head) with kurtosis / top5% / outlier / rms.
    """
    if qkv.numel() == 0:
        return []
    # Reshape to expose Q/K/V and head dimensions:
    #   [B, N, 3, num_heads, head_dim]
    bsz, ntok, feat = qkv.shape
    expected = 3 * num_heads * head_dim
    if feat != expected:
        return []
    reshaped = qkv.reshape(bsz, ntok, 3, num_heads, head_dim)
    rows: list[dict] = []
    for proj_idx, proj_name in enumerate(("q", "k", "v")):
        for head in range(num_heads):
            slab = reshaped[:, :, proj_idx, head, :]  # [B, N, head_dim]
            row = _stats_for_tensor(slab)
            row.update({
                "projection": proj_name,
                "head": head,
                "head_dim": head_dim,
                "n_samples": int(bsz),
            })
            rows.append(row)
    return rows


def run(
    config_path: str,
    model_names: list[str] | None = None,
    seed_override: int | None = None,
    max_batches_per_group: int = 8,
) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    set_seed(seed)
    device = get_device()

    _, val_loader, _, _ = build_dataloaders(config, include_train=False)
    model_names = model_names or [config["models"]["teacher"], config["models"]["student"]]

    results_dir = ensure_dir(config["logging"]["results_dir"])
    output_path = Path(results_dir) / "perhead_activation_stats.csv"

    all_rows: list[dict] = []
    for model_name in model_names:
        alias = model_alias(model_name)
        model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
        blocks = _find_attention_blocks(model)
        if not blocks:
            continue
        captures = _collect_class_conditional(
            model, val_loader, blocks, device, max_batches_per_group=max_batches_per_group
        )
        for group, slabs in captures.items():
            for block_idx, block_name, attn, qkv in slabs:
                rows = _stats_per_head_from_qkv(qkv, int(attn.num_heads), int(attn.head_dim))
                for r in rows:
                    r.update({
                        "model": alias, "seed": seed,
                        "layer": block_name, "layer_idx": block_idx,
                        "class_group": group,
                    })
                    all_rows.append(r)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-attention-head class-conditional activation statistics (P1.3)."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-batches-per-group",
        type=int,
        default=8,
        help="Cap on val batches collected (memory guard).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.config,
        model_names=args.models,
        seed_override=args.seed,
        max_batches_per_group=args.max_batches_per_group,
    )


if __name__ == "__main__":
    main()
