"""
Fig 7 — does recovery finetuning change the criterion ranking?

Redraw (W15): the original figure packed four quantities into a narrow
two-panel grid at the default size, with bars that were legible only
at 150%+ zoom. This version:
  - uses the wide canvas from `STYLE["figure_size_wide"]`
  - adds error bars from per-seed variance when the columns are
    available (falls back silently on single-seed runs)
  - plots all three criteria as grouped bars with gap annotations so
    the relative move from one-shot -> recovered is explicit
  - separates melanoma-sensitivity and balanced-accuracy panels into
    two PDFs, both at the same width, so the paper can place them
    side-by-side or stacked without regenerating
"""
from __future__ import annotations

import argparse
from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.style import CRITERION_COLORS, STYLE, apply_style, save_figure
from utils.config import load_config


MODEL_TITLES = {
    "deit_small": "DeiT-Small",
    "deit_tiny": "DeiT-Tiny",
}


def _bar_with_error(ax, x, value, err, color, hatch=None, label=None):
    bar = ax.bar(
        x,
        value,
        width=0.38,
        color=color,
        edgecolor="black",
        linewidth=0.6,
        hatch=hatch,
        label=label,
    )
    if err is not None and np.isfinite(err):
        ax.errorbar(x, value, yerr=err, fmt="none", ecolor="black", capsize=3.5, linewidth=1.0)
    return bar


def _get_value(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col in df.columns:
        mean = float(df[mean_col].iloc[0])
        std = float(df[std_col].iloc[0]) if std_col in df.columns else float("nan")
    else:
        mean = float(df[metric].iloc[0])
        std = float("nan")
    return mean, std


def _plot_metric(
    frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    metric: str,
    output_path: Path,
    ylabel: str,
) -> None:
    apply_style()
    criteria = ["magnitude", "wanda", "taylor"]
    models = ["deit_small", "deit_tiny"]
    x = np.arange(len(criteria), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=STYLE["figure_size_wide"])
    for ax, model_name in zip(axes, models):
        model_df = frame[frame["model"] == model_name]
        baseline_row = baseline_frame.loc[baseline_frame["model"] == model_name]
        # baseline_eval uses `balanced_accuracy`, recovery_finetune uses `balanced_acc` —
        # try both naming conventions so this figure works against either source CSV.
        candidates = [f"{metric}_mean", metric, f"{metric.replace('balanced_acc','balanced_accuracy')}_mean", metric.replace("balanced_acc", "balanced_accuracy")]
        dense_baseline = float("nan")
        for col in candidates:
            if col in baseline_row.columns:
                dense_baseline = float(baseline_row[col].iloc[0])
                break

        for idx, criterion in enumerate(criteria):
            criterion_df = model_df[model_df["criterion"] == criterion]
            one_shot_df = criterion_df[criterion_df["recovery_epochs"] == 0]
            recovered_df = criterion_df[criterion_df["recovery_epochs"] > 0]
            if one_shot_df.empty or recovered_df.empty:
                continue
            one_shot_val, one_shot_err = _get_value(one_shot_df, metric)
            recovered_val, recovered_err = _get_value(recovered_df, metric)
            colour = CRITERION_COLORS.get(criterion, "#333333")
            label_one = "One-shot" if idx == 0 else None
            label_rec = "Recovered" if idx == 0 else None
            _bar_with_error(ax, x[idx] - 0.2, one_shot_val, one_shot_err, colour, hatch="//", label=label_one)
            _bar_with_error(ax, x[idx] + 0.2, recovered_val, recovered_err, colour, label=label_rec)

            # Annotate the gain
            if np.isfinite(one_shot_val) and np.isfinite(recovered_val):
                gain = recovered_val - one_shot_val
                ax.annotate(
                    f"{gain:+.2f}",
                    xy=(x[idx], max(one_shot_val, recovered_val) + 0.03),
                    ha="center",
                    fontsize=10,
                )

        ax.axhline(dense_baseline, linestyle="--", color="black", linewidth=1.2, label="Dense baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in criteria])
        ax.set_ylim(0.0, 1.10)
        ax.set_ylabel(ylabel)
        ax.set_title(MODEL_TITLES.get(model_name, model_name))
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].legend(loc="lower right")
    fig.suptitle("Does Recovery Fine-Tuning Change the Criterion Ranking?")
    save_figure(fig, output_path)
    plt.close(fig)


def run(config_path: str) -> None:
    config = load_config(config_path)
    results_dir = Path(config["logging"]["results_dir"])
    figures_dir = Path(config["logging"]["figures_dir"])

    # Prefer aggregated per-seed summaries when they exist; fall back to
    # the raw single-seed CSV so this still works on a quick sanity run.
    aggregated_dir = results_dir / "aggregated"
    if (aggregated_dir / "agg_recovery_finetune.csv").exists():
        frame = pd.read_csv(aggregated_dir / "agg_recovery_finetune.csv")
        baseline_frame = pd.read_csv(aggregated_dir / "agg_baseline_eval.csv")
    else:
        frame = pd.read_csv(results_dir / "recovery_finetune.csv")
        baseline_frame = pd.read_csv(results_dir / "baseline_eval.csv")

    _plot_metric(
        frame,
        baseline_frame,
        metric="mel_sensitivity",
        output_path=figures_dir / "fig7_recovery_mel.pdf",
        ylabel="Melanoma Sensitivity",
    )
    _plot_metric(
        frame,
        baseline_frame,
        metric="balanced_acc",
        output_path=figures_dir / "fig7_recovery_balacc.pdf",
        ylabel="Balanced Accuracy",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot recovery finetuning comparisons (W15).")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
