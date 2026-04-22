from __future__ import annotations

import argparse
from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt
import pandas as pd

from plotting.style import CRITERION_COLORS, apply_style, save_figure
from utils.config import load_config


def run(config_path: str) -> None:
    config = load_config(config_path)
    input_path = Path(config["logging"]["results_dir"]) / "perlayer_breakdown.csv"
    output_path = Path(config["logging"]["figures_dir"]) / "fig3_perlayer_bars.pdf"
    frame = pd.read_csv(input_path)
    apply_style()

    pivot = frame.pivot_table(
        index="layer_type",
        columns="criterion",
        values="balanced_accuracy_drop",
        aggfunc="mean",
    ).fillna(0.0)
    order = ["qkv", "attn_out", "mlp", "patch_embed"]
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(pivot.index))
    width = 0.35
    ax.bar([idx - width / 2 for idx in x], pivot.get("wanda", pd.Series([0] * len(pivot.index))).tolist(), width=width, color=CRITERION_COLORS["wanda"], label="Wanda")
    ax.bar([idx + width / 2 for idx in x], pivot.get("magnitude", pd.Series([0] * len(pivot.index))).tolist(), width=width, color=CRITERION_COLORS["magnitude"], label="Magnitude")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["QKV", "Attn Output", "MLP", "Patch Embed"])
    ax.set_ylabel("Balanced Accuracy Drop")
    ax.set_title("Which ViT Components Encode Diagnostic Features?")
    ax.legend()
    save_figure(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-layer breakdown.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
