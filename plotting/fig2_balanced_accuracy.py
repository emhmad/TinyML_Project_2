from __future__ import annotations

import argparse
from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt
import pandas as pd

from plotting.style import CRITERION_COLORS, CRITERION_MARKERS, STYLE, apply_style, save_figure
from utils.config import load_config


def run(config_path: str) -> None:
    config = load_config(config_path)
    results_path = Path(config["logging"]["results_dir"]) / "pruning_matrix.csv"
    output_path = Path(config["logging"]["figures_dir"]) / "fig2_balanced_accuracy.pdf"
    frame = pd.read_csv(results_path)
    dense = frame[frame["criterion"] == "dense"][["model", "balanced_acc"]].rename(columns={"balanced_acc": "baseline_bal"})
    plot_df = frame[frame["criterion"] != "dense"].merge(dense, on="model", how="left")
    apply_style()

    models = sorted(plot_df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(STYLE["figure_size"][0] * len(models), STYLE["figure_size"][1]))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        model_df = plot_df[plot_df["model"] == model_name]
        for criterion in ("magnitude", "wanda", "taylor", "random"):
            criterion_df = model_df[model_df["criterion"] == criterion].sort_values("sparsity")
            if criterion_df.empty:
                continue
            ax.plot(
                criterion_df["sparsity"],
                criterion_df["balanced_acc"],
                color=CRITERION_COLORS[criterion],
                marker=CRITERION_MARKERS[criterion],
                linewidth=STYLE["line_width"],
                markersize=STYLE["marker_size"],
                label=criterion,
            )
        ax.axhline(float(model_df["baseline_bal"].iloc[0]), linestyle="--", color="black", linewidth=1.2)
        ax.set_title(model_name)
        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_ylim(0.0, 1.05)

    axes[0].legend()
    fig.suptitle("Balanced Accuracy Under Pruning")
    save_figure(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot balanced accuracy vs sparsity.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
