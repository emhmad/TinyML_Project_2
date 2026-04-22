from __future__ import annotations

import argparse
from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt
import pandas as pd

from plotting.style import CLASS_COLORS, apply_style, save_figure
from utils.config import load_config

CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def run(config_path: str, criterion: str = "wanda") -> None:
    config = load_config(config_path)
    input_path = Path(config["logging"]["results_dir"]) / "nonuniform_allocation.csv"
    output_path = Path(config["logging"]["figures_dir"]) / "fig4_nonuniform_vs_uniform.pdf"
    frame = pd.read_csv(input_path)
    frame = frame[frame["criterion"] == criterion]
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 5))
    conditions = ["dense", "uniform", "nonuniform"]
    width = 0.24
    x = range(len(CLASS_ORDER))

    for offset, condition in enumerate(conditions):
        row = frame[frame["condition"] == condition].iloc[-1]
        values = [row[f"{class_name}_sensitivity"] for class_name in CLASS_ORDER]
        ax.bar(
            [idx + (offset - 1) * width for idx in x],
            values,
            width=width,
            label=condition.replace("_", " "),
            alpha=0.9,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Per-Class Sensitivity")
    ax.set_title("Sensitivity-Guided Allocation vs Uniform Pruning")
    ax.legend()
    save_figure(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot non-uniform vs uniform pruning.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--criterion", default="wanda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.criterion)


if __name__ == "__main__":
    main()
