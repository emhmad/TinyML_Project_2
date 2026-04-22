"""
Fig 9 — Wanda sensitivity to calibration set size.

Redraw (W15). The original figure set a base-2 log x-axis but kept the
raw integer tick labels {16, 32, 64, ...} so they collided at low
zoom. This version:
  - formats the x-axis as a base-2 log scale with explicit tick labels
    every power of two between the min and max measured size
  - shows the magnitude 50% reference as a horizontal band (mean ± std
    across seeds when available) rather than a single line
  - draws melanoma sensitivity and balanced accuracy as twin panels
    sharing a log x-axis, with error bars on each measured point when
    per-seed variance is available
"""
from __future__ import annotations

import argparse
from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatter

from plotting.style import STYLE, apply_style, save_figure
from utils.config import load_config


def _band(ax, x_min: float, x_max: float, centre: float, err: float | None, label: str) -> None:
    ax.axhline(centre, linestyle="--", color="black", linewidth=1.2, label=label)
    if err is not None and np.isfinite(err) and err > 0:
        ax.fill_between(
            [x_min, x_max],
            centre - err,
            centre + err,
            color="black",
            alpha=0.10,
            label=None,
        )


def _seed_aware_column(frame: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray | None]:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col in frame.columns:
        mean = frame[mean_col].to_numpy(dtype=float)
        std = frame[std_col].to_numpy(dtype=float) if std_col in frame.columns else None
        return mean, std
    return frame[metric].to_numpy(dtype=float), None


def run(config_path: str) -> None:
    apply_style()
    config = load_config(config_path)
    results_dir = Path(config["logging"]["results_dir"])
    figures_dir = Path(config["logging"]["figures_dir"])
    aggregated_dir = results_dir / "aggregated"

    if (aggregated_dir / "agg_calibration_ablation.csv").exists():
        ablation = pd.read_csv(aggregated_dir / "agg_calibration_ablation.csv")
    else:
        ablation = pd.read_csv(results_dir / "calibration_ablation.csv")
    ablation = ablation.sort_values("calibration_size")

    if (aggregated_dir / "agg_pruning_matrix.csv").exists():
        pruning = pd.read_csv(aggregated_dir / "agg_pruning_matrix.csv")
        reference = pruning[
            (pruning["model"] == "deit_small")
            & (pruning["criterion"] == "magnitude")
            & (pruning["sparsity"].round(3) == 0.5)
        ].tail(1)
        baseline_mel, baseline_mel_err = _seed_aware_column(reference, "mel_sensitivity")
        baseline_bal, baseline_bal_err = _seed_aware_column(reference, "balanced_acc")
        baseline_mel = float(baseline_mel[0]) if baseline_mel.size else float("nan")
        baseline_bal = float(baseline_bal[0]) if baseline_bal.size else float("nan")
        baseline_mel_err = float(baseline_mel_err[0]) if baseline_mel_err is not None else None
        baseline_bal_err = float(baseline_bal_err[0]) if baseline_bal_err is not None else None
    else:
        pruning = pd.read_csv(results_dir / "pruning_matrix.csv")
        reference = pruning[
            (pruning["model"] == "deit_small")
            & (pruning["criterion"] == "magnitude")
            & (pruning["sparsity"].round(3) == 0.5)
        ].tail(1)
        if reference.empty:
            raise RuntimeError("Could not find DeiT-Small magnitude 50% reference row in pruning_matrix.csv.")
        baseline_mel = float(reference["mel_sensitivity"].iloc[0])
        baseline_bal = float(reference["balanced_acc"].iloc[0])
        baseline_mel_err = None
        baseline_bal_err = None

    fig, axes = plt.subplots(1, 2, figsize=STYLE["figure_size_wide"], sharex=True)
    sizes = ablation["calibration_size"].to_numpy(dtype=float)
    mel_mean, mel_err = _seed_aware_column(ablation, "mel_sensitivity")
    bal_mean, bal_err = _seed_aware_column(ablation, "balanced_acc")

    axes[0].errorbar(
        sizes, mel_mean, yerr=mel_err, marker="o", linewidth=2.0, capsize=3.5, label="Wanda (observed)"
    )
    _band(axes[0], sizes.min(), sizes.max(), baseline_mel, baseline_mel_err, "Magnitude 50%")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Calibration Set Size (log₂ scale)")
    axes[0].set_ylabel("Melanoma Sensitivity")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Melanoma Sensitivity")
    axes[0].set_xticks(sizes)
    axes[0].set_xticklabels([str(int(s)) for s in sizes])
    axes[0].xaxis.set_major_formatter(LogFormatter(base=2))
    axes[0].grid(True, which="both", linestyle=":", alpha=0.35)
    axes[0].legend()

    axes[1].errorbar(
        sizes, bal_mean, yerr=bal_err, marker="o", linewidth=2.0, capsize=3.5, label="Wanda (observed)"
    )
    _band(axes[1], sizes.min(), sizes.max(), baseline_bal, baseline_bal_err, "Magnitude 50%")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Calibration Set Size (log₂ scale)")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Balanced Accuracy")
    axes[1].set_xticks(sizes)
    axes[1].set_xticklabels([str(int(s)) for s in sizes])
    axes[1].xaxis.set_major_formatter(LogFormatter(base=2))
    axes[1].grid(True, which="both", linestyle=":", alpha=0.35)
    axes[1].legend()

    fig.suptitle("Wanda Sensitivity to Calibration Set Size")
    save_figure(fig, figures_dir / "fig9_calibration_ablation.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Wanda calibration-size ablation (W15).")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
