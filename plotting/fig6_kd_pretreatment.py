from __future__ import annotations

import argparse
from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt
import pandas as pd

from plotting.style import apply_style, save_figure
from utils.config import load_config

CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def run(config_path: str) -> None:
    config = load_config(config_path)
    input_path = Path(config["logging"]["results_dir"]) / "kd_pretreatment.csv"
    output_path = Path(config["logging"]["figures_dir"]) / "fig6_kd_pretreatment.pdf"
    frame = pd.read_csv(input_path)
    frame = frame[frame["pruned"] == "yes"]
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.24
    x = range(len(CLASS_ORDER))
    variants = ["direct", "distilled", "imagenet_only"]

    for offset, variant in enumerate(variants):
        row = frame[frame["variant"] == variant].iloc[-1]
        values = [row[f"{class_name}_sensitivity"] for class_name in CLASS_ORDER]
        ax.bar([idx + (offset - 1) * width for idx in x], values, width=width, label=variant.replace("_", " "))

    ax.set_xticks(list(x))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Per-Class Sensitivity")
    ax.set_title("Does Knowledge Distillation Produce Pruning-Robust Features?")
    ax.legend()
    save_figure(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KD pre-treatment results.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
