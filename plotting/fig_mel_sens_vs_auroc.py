"""
Mel sensitivity vs Mel AUROC scatter at 50% sparsity, both backbones.

The visual disambiguates "high sensitivity" (which can be inflated by
collapsing predictions toward the melanoma class) from real
discrimination ability (AUROC). Wanda on DeiT-Tiny lands in the
high-sensitivity / low-AUROC quadrant — the visible outlier — and that
single dot is the talk's "sensitivity inflation" demonstration.

Includes the dense baseline as a reference anchor and a diagonal-band
hint where collapsed classifiers tend to live.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from plotting import mpl_setup as _mpl_setup  # noqa: F401

import matplotlib.pyplot as plt
import pandas as pd

from plotting.style import CRITERION_COLORS, CRITERION_MARKERS, STYLE, apply_style, save_figure


CRITERIA = ["dense", "magnitude", "wanda", "taylor", "random"]
MODEL_MARKERS = {"deit_small": "o", "deit_tiny": "s"}
MODEL_TITLES = {"deit_small": "DeiT-Small", "deit_tiny": "DeiT-Tiny"}


def run(aggregated_csv: Path, output_path: Path, sparsity: float = 0.5) -> None:
    apply_style()
    df = pd.read_csv(aggregated_csv)
    sub = df[
        ((df["sparsity"].round(3) == round(sparsity, 3)) & (df["criterion"] != "dense"))
        | (df["criterion"] == "dense")
    ]
    sub = sub[sub["criterion"].isin(CRITERIA)]
    if sub.empty:
        raise RuntimeError("No matching rows in aggregated CSV.")

    fig, ax = plt.subplots(figsize=STYLE["figure_size"])

    # Shaded "collapse zone": low AUROC + high sensitivity = predicting mel everywhere.
    ax.fill_betweenx(y=[0.4, 0.82], x1=0.6, x2=1.05, color="#fdd", alpha=0.45, zorder=0)
    ax.text(
        0.99, 0.42, "classifier collapse\n(high sens, low AUROC)",
        ha="right", va="bottom", color="#a04040", fontsize=10, alpha=0.85,
    )

    for model, marker in MODEL_MARKERS.items():
        msub = sub[sub["model"] == model]
        for _, row in msub.iterrows():
            crit = str(row["criterion"])
            color = CRITERION_COLORS.get(crit, "#444444")
            x = float(row["mel_sensitivity_mean"])
            y = float(row["melanoma_auroc_mean"])
            xerr = float(row.get("mel_sensitivity_std", 0) or 0)
            yerr = float(row.get("melanoma_auroc_std", 0) or 0)

            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr, fmt=marker, color=color,
                markersize=11, markeredgecolor="black", markeredgewidth=0.6,
                ecolor="black", elinewidth=0.8, capsize=2.5, zorder=3,
            )
            ax.annotate(
                f"{crit}\n{MODEL_TITLES.get(model, model)}",
                xy=(x, y), xytext=(8, 5), textcoords="offset points",
                fontsize=9, alpha=0.85,
            )

    ax.set_xlabel("Melanoma sensitivity (mean)")
    ax.set_ylabel("Melanoma AUROC (mean)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Manual legend by model marker (criterion is in the per-point label)
    handles = [
        plt.Line2D([0], [0], marker=mk, color="black", linestyle="None",
                   markersize=10, label=MODEL_TITLES[m])
        for m, mk in MODEL_MARKERS.items()
    ]
    ax.legend(handles=handles, loc="lower left", title="Backbone")

    fig.suptitle("Mel Sensitivity vs Mel AUROC at 50% Sparsity")
    save_figure(fig, output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mel sens vs Mel AUROC scatter (talk visual B).")
    parser.add_argument(
        "--aggregated-csv",
        default="results/logs_personal/aggregated/agg_pruning_matrix.csv",
    )
    parser.add_argument(
        "--output", default="results/figures_personal/fig_mel_sens_vs_auroc.pdf",
    )
    parser.add_argument("--sparsity", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.aggregated_csv), Path(args.output), sparsity=args.sparsity)


if __name__ == "__main__":
    main()
