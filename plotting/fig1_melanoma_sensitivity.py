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
    output_path = Path(config["logging"]["figures_dir"]) / "fig1_melanoma_sensitivity.pdf"
    frame = pd.read_csv(results_path)
    apply_style()

    dense = frame[frame["criterion"] == "dense"][["model", "mel_sensitivity"]].rename(
        columns={"mel_sensitivity": "baseline_mel"}
    )
    plot_df = frame[frame["criterion"] != "dense"].merge(dense, on="model", how="left")
    models = sorted(plot_df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(STYLE["figure_size"][0] * len(models), STYLE["figure_size"][1]))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        model_df = plot_df[plot_df["model"] == model_name]
        ax.axhspan(0.0, 0.5, color="#f5d0d0", alpha=0.35)
        for criterion in ("magnitude", "wanda", "taylor", "random"):
            criterion_df = model_df[model_df["criterion"] == criterion].sort_values("sparsity")
            if criterion_df.empty:
                continue
            ax.plot(
                criterion_df["sparsity"],
                criterion_df["mel_sensitivity"],
                color=CRITERION_COLORS[criterion],
                marker=CRITERION_MARKERS[criterion],
                linewidth=STYLE["line_width"],
                markersize=STYLE["marker_size"],
                label=criterion,
            )
        baseline_value = float(model_df["baseline_mel"].iloc[0])
        ax.axhline(baseline_value, linestyle="--", color="black", linewidth=1.2)
        ax.set_title(model_name)
        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Melanoma Sensitivity")
        ax.set_ylim(0.0, 1.05)

    axes[0].legend()
    fig.suptitle("Melanoma Detection Rate Under Pruning")
    save_figure(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot melanoma sensitivity vs sparsity.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
