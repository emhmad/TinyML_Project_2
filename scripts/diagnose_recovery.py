"""
Recovery-sweep diagnostic (P0.1).

The revision_analysis.md flagged that recovery fine-tuning may have
been a no-op: identical means across recovery-epoch sweeps suggested
either an MPS gradient no-op or a learning-rate too low under MPS.
The original diagnostic (compare dense vs recovered) is incomplete —
that diff is always non-zero because pruning alone zeros 50% of weights.

This script does the right comparison: it loads the e5/e10/e20 recovery
checkpoints for each criterion and verifies (a) the sparsity mask is
preserved across epoch points, (b) the unmasked weights actually
*moved* between e5 and e20. If they are identical, recovery did not
train.

Run on the cluster login node — pure CPU work, no GPU needed.

Usage:
    python -m scripts.diagnose_recovery \
        --checkpoints-dir results/checkpoints_personal/seed_0 \
        --models deit_small deit_tiny \
        --criteria magnitude wanda taylor \
        --sparsity 0.5 \
        --epochs 5 10 20 \
        --lr 1.0e-04 \
        --output results/logs_personal/seed_0/recovery_diagnostic.csv
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
import torch


def _checkpoint_path(
    base: Path,
    model_alias: str,
    criterion: str,
    sparsity: float,
    epochs: int,
    lr: float,
) -> Path:
    return base / (
        f"recovery_{model_alias}_{criterion}_s{sparsity:.2f}_e{epochs}_lr{lr:.0e}.pth"
    )


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"]
    return blob


def _layer_diagnostics(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> tuple[float, float, float, int]:
    """
    Returns (mean_sparsity, weight_diff_norm_unmasked, mask_jaccard, n_pruned_layers).

    mask_jaccard: how much of the zero-positions overlap between the two
    states. ~1.0 means the same mask was preserved; <1.0 means the mask
    drifted (unexpected — recovery should freeze masks).
    """
    diff_sq_sum = 0.0
    sparsity_sum = 0.0
    intersection = 0
    union = 0
    n_layers = 0
    for key, tensor_a in state_a.items():
        if key not in state_b:
            continue
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape:
            continue
        if tensor_a.dim() < 2:
            # skip biases, LayerNorm, position embeddings
            continue
        zero_a = tensor_a == 0
        zero_b = tensor_b == 0
        sparsity = float(zero_b.float().mean().item())
        if sparsity < 0.05:
            # not a pruned layer
            continue
        n_layers += 1
        sparsity_sum += sparsity
        # diff at positions that are unmasked in BOTH states
        unmasked = (~zero_a) & (~zero_b)
        if unmasked.any():
            d = (tensor_a - tensor_b)[unmasked]
            diff_sq_sum += float(d.pow(2).sum().item())
        intersection += int((zero_a & zero_b).sum().item())
        union += int((zero_a | zero_b).sum().item())
    if n_layers == 0:
        return float("nan"), float("nan"), float("nan"), 0
    avg_sparsity = sparsity_sum / n_layers
    weight_diff_norm = diff_sq_sum**0.5
    mask_jaccard = (intersection / union) if union > 0 else float("nan")
    return avg_sparsity, weight_diff_norm, mask_jaccard, n_layers


def diagnose(
    checkpoints_dir: Path,
    models: list[str],
    criteria: list[str],
    sparsity: float,
    epochs: list[int],
    lr: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for model_alias, criterion in product(models, criteria):
        states: dict[int, dict[str, torch.Tensor]] = {}
        for e in epochs:
            path = _checkpoint_path(checkpoints_dir, model_alias, criterion, sparsity, e, lr)
            if not path.exists():
                rows.append({
                    "model": model_alias, "criterion": criterion, "sparsity": sparsity,
                    "comparison": f"e{e}_missing",
                    "checkpoint_path": str(path),
                    "weight_diff_norm_unmasked": float("nan"),
                    "mean_sparsity": float("nan"),
                    "mask_jaccard": float("nan"),
                    "n_pruned_layers": 0,
                    "verdict": "missing_checkpoint",
                })
                continue
            states[e] = _load_state(path)

        # Pairwise comparisons across the epoch sweep.
        sorted_epochs = sorted(states.keys())
        for i in range(len(sorted_epochs) - 1):
            e_lo, e_hi = sorted_epochs[i], sorted_epochs[i + 1]
            sparsity_avg, diff_norm, jaccard, n_layers = _layer_diagnostics(states[e_lo], states[e_hi])
            verdict = (
                "no_training_detected" if diff_norm < 1e-6
                else "weak_training" if diff_norm < 1e-2
                else "training_observed"
            )
            rows.append({
                "model": model_alias, "criterion": criterion, "sparsity": sparsity,
                "comparison": f"e{e_lo}_vs_e{e_hi}",
                "checkpoint_path": "",
                "weight_diff_norm_unmasked": diff_norm,
                "mean_sparsity": sparsity_avg,
                "mask_jaccard": jaccard,
                "n_pruned_layers": n_layers,
                "verdict": verdict,
            })

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether recovery fine-tuning actually trained the model."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("results/checkpoints_personal/seed_0"),
        help="Directory containing recovery_*.pth files for one seed.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deit_small", "deit_tiny"],
        help="Model aliases (without _patch16_224 suffix).",
    )
    parser.add_argument(
        "--criteria",
        nargs="+",
        default=["magnitude", "wanda", "taylor"],
    )
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--epochs", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-4,
        help="Learning rate suffix used in the checkpoint filenames (e.g., 1e-04).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/logs_personal/seed_0/recovery_diagnostic.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = diagnose(
        args.checkpoints_dir, args.models, args.criteria, args.sparsity, args.epochs, args.lr
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    # Compact stdout summary — the verdict per (model, criterion).
    print(f"[diagnose_recovery] wrote {args.output}")
    print(frame.to_string(index=False))
    bad = frame[frame["verdict"] == "no_training_detected"]
    if len(bad) > 0:
        print(
            f"\n[diagnose_recovery] WARNING: {len(bad)} comparisons show no training "
            "(weight diff < 1e-6 between consecutive epoch checkpoints). "
            "Rerun recovery with higher LR or on CPU device — see recovery.lr in your config."
        )
    else:
        print("\n[diagnose_recovery] OK: training observed across all comparisons.")


if __name__ == "__main__":
    main()
