from __future__ import annotations

import argparse
from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.style import STYLE, apply_style, save_figure
from utils.config import load_config

RECOVERY_COLORS = {
    ("magnitude", 0): "#a6cee3",
    ("magnitude", 5): "#1f77b4",
    ("wanda", 0): "#fb9a99",
    ("wanda", 5): "#d62728",
    ("taylor", 0): "#b2df8a",
    ("taylor", 5): "#2ca02c",
}

MODEL_TITLES = {
    "deit_small": "DeiT-Small",
    "deit_tiny": "DeiT-Tiny",
}


def _plot_metric(frame: pd.DataFrame, baseline_frame: pd.DataFrame, metric: str, output_path: Path, ylabel: str) -> None:
    apply_style()
    criteria = ["magnitude", "wanda", "taylor"]
    models = ["deit_small", "deit_tiny"]
    x = np.arange(len(criteria))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(STYLE["figure_size"][0] * 2.0, STYLE["figure_size"][1]))
    for ax, model_name in zip(axes, models):
        model_df = frame[frame["model"] == model_name]
        dense_baseline = float(baseline_frame.loc[baseline_frame["model"] == model_name, metric].iloc[0])

        oneshot_values = []
        recovered_values = []
        for criterion in criteria:
            criterion_df = model_df[model_df["criterion"] == criterion]
            oneshot = float(criterion_df.loc[criterion_df["recovery_epochs"] == 0, metric].iloc[0])
            recovered = float(criterion_df.loc[criterion_df["recovery_epochs"] > 0, metric].iloc[0])
            oneshot_values.append(oneshot)
            recovered_values.append(recovered)

        for idx, criterion in enumerate(criteria):
            ax.bar(
                x[idx] - width / 2,
                oneshot_values[idx],
                width,
                color=RECOVERY_COLORS[(criterion, 0)],
                label="One-shot" if idx == 0 else None,
            )
            ax.bar(
                x[idx] + width / 2,
                recovered_values[idx],
                width,
                color=RECOVERY_COLORS[(criterion, 5)],
                label="Recovered" if idx == 0 else None,
            )

        ax.axhline(dense_baseline, linestyle="--", color="black", linewidth=1.2, label="Dense baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([name.capitalize() for name in criteria])
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(MODEL_TITLES.get(model_name, model_name))

    axes[0].legend()
    fig.suptitle("Does Recovery Finetuning Change the Criterion Ranking?")
    save_figure(fig, output_path)


def run(config_path: str) -> None:
    config = load_config(config_path)
    results_dir = Path(config["logging"]["results_dir"])
    figures_dir = Path(config["logging"]["figures_dir"])
    recovery_path = results_dir / "recovery_finetune.csv"
    baseline_path = results_dir / "baseline_eval.csv"

    frame = pd.read_csv(recovery_path)
    baseline_frame = pd.read_csv(baseline_path)
    baseline_frame = baseline_frame.rename(
        columns={
            "balanced_accuracy": "balanced_acc",
            "mel_sensitivity": "mel_sensitivity",
        }
    )
    baseline_frame = baseline_frame.drop_duplicates(subset=["model"], keep="last")

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
    parser = argparse.ArgumentParser(description="Plot recovery finetuning comparisons.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
