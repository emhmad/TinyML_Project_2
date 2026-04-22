from __future__ import annotations

import argparse
from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt
import pandas as pd

from plotting.style import apply_style, save_figure
from utils.config import load_config

CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def run(config_path: str, criterion: str = "wanda") -> None:
    config = load_config(config_path)
    input_path = Path(config["logging"]["results_dir"]) / "quantization_stacking.csv"
    output_path = Path(config["logging"]["figures_dir"]) / "fig5_stacking.pdf"
    frame = pd.read_csv(input_path)
    apply_style()

    selection = pd.concat(
        [
            frame[frame["model_config"] == "dense"].tail(1),
            frame[(frame["model_config"] == "pruned_only") & (frame["criterion"] == criterion)].tail(1),
            frame[(frame["model_config"] == "pruned_plus_quantized") & (frame["criterion"] == criterion)].tail(1),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.24
    x = range(len(CLASS_ORDER))
    labels = ["dense", "pruned_only", "pruned_plus_quantized"]

    for offset, label in enumerate(labels):
        row = selection[selection["model_config"] == label].iloc[-1]
        values = [row[f"{class_name}_sensitivity"] for class_name in CLASS_ORDER]
        ax.bar([idx + (offset - 1) * width for idx in x], values, width=width, label=label.replace("_", " "))

    ax.set_xticks(list(x))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Per-Class Sensitivity")
    ax.set_title("Does Stacking Quantization on Pruning Break Diagnostic Safety?")
    ax.legend()
    save_figure(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quantization stacking results.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--criterion", default="wanda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.criterion)


if __name__ == "__main__":
    main()
