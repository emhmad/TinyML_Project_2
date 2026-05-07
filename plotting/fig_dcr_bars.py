"""
DCR (Dangerous-Class Degradation Ratio) bar chart per pruning criterion,
side-by-side for both DeiT backbones at 50% sparsity.

The visual makes the central revision finding obvious: magnitude pruning
sits well above DCR=1 (forgets dangerous classes preferentially),
Wanda sits well below (over-attends to dangerous classes). The
horizontal reference line at DCR=1 splits the plot into the two
failure modes the paper now reports as criterion-specific.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from plotting import mpl_setup as _mpl_setup  # noqa: F401  (configures cache)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.style import CRITERION_COLORS, STYLE, apply_style, save_figure


CRITERIA = ["magnitude", "wanda", "taylor", "random"]
MODEL_TITLES = {"deit_small": "DeiT-Small", "deit_tiny": "DeiT-Tiny"}


def run(aggregated_csv: Path, output_path: Path, sparsity: float = 0.5) -> None:
    apply_style()
    df = pd.read_csv(aggregated_csv)
    sub = df[(df["sparsity"].round(3) == round(sparsity, 3)) & (df["criterion"].isin(CRITERIA))]
    if sub.empty:
        raise RuntimeError(f"No rows matched sparsity={sparsity} for criteria {CRITERIA}.")

    models = sorted(sub["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=STYLE["figure_size_wide"], sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        rows = sub[sub["model"] == model].set_index("criterion").reindex(CRITERIA).reset_index()
        x = np.arange(len(CRITERIA))
        means = rows["dangerous_class_degradation_ratio_mean"].astype(float).to_numpy()
        stds = rows.get("dangerous_class_degradation_ratio_std", pd.Series(np.zeros(len(CRITERIA)))).fillna(0).to_numpy()
        colors = [CRITERION_COLORS.get(c, "#555555") for c in CRITERIA]

        bars = ax.bar(
            x, means, yerr=stds, capsize=3.5, color=colors,
            edgecolor="black", linewidth=0.6,
        )

        # Annotate each bar with its DCR value
        for xi, mean, std in zip(x, means, stds):
            offset = (std if np.isfinite(std) else 0.0) + 0.08
            ax.annotate(
                f"{mean:.2f}",
                xy=(xi, mean + offset if np.isfinite(mean) else 0.05),
                ha="center", fontsize=11, fontweight="bold",
            )

        # The DCR=1 reference line — splits "forgets dangerous" from "forgets safe"
        ax.axhline(1.0, linestyle="--", color="black", linewidth=1.4, alpha=0.7)
        ax.text(
            len(CRITERIA) - 0.5, 1.05, "DCR = 1\n(equal degradation)",
            ha="right", va="bottom", fontsize=10, alpha=0.75,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in CRITERIA])
        ax.set_title(MODEL_TITLES.get(model, model))
        ax.set_ylabel("Dangerous-Class Degradation Ratio (DCR)")
        ax.set_ylim(0, max(means.max() * 1.25 if np.isfinite(means).any() else 2.5, 2.5))
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Twin annotations on the leftmost panel — what each side of DCR=1 means
    axes[0].annotate(
        "↑ forgets dangerous classes first",
        xy=(0.02, 0.97), xycoords="axes fraction",
        fontsize=10, alpha=0.7, ha="left", va="top",
    )
    axes[0].annotate(
        "↓ over-attends to dangerous classes",
        xy=(0.02, 0.03), xycoords="axes fraction",
        fontsize=10, alpha=0.7, ha="left", va="bottom",
    )

    fig.suptitle(f"Dangerous-Class Degradation Ratio at {int(sparsity*100)}% Sparsity")
    save_figure(fig, output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DCR bar chart per criterion (talk visual A).")
    parser.add_argument(
        "--aggregated-csv",
        default="results/logs_personal/aggregated/agg_pruning_matrix.csv",
    )
    parser.add_argument(
        "--output", default="results/figures_personal/fig_dcr_bars.pdf",
    )
    parser.add_argument("--sparsity", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.aggregated_csv), Path(args.output), sparsity=args.sparsity)


if __name__ == "__main__":
    main()
