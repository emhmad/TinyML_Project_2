from __future__ import annotations

import argparse
from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt
import pandas as pd

from plotting.style import STYLE, apply_style, save_figure
from utils.config import load_config


def run(config_path: str) -> None:
    config = load_config(config_path)
    results_dir = Path(config["logging"]["results_dir"])
    figures_dir = Path(config["logging"]["figures_dir"])

    ablation = pd.read_csv(results_dir / "calibration_ablation.csv").sort_values("calibration_size")
    pruning = pd.read_csv(results_dir / "pruning_matrix.csv")
    reference = pruning[
        (pruning["model"] == "deit_small")
        & (pruning["criterion"] == "magnitude")
        & (pruning["sparsity"].round(3) == 0.5)
    ].tail(1)
    if reference.empty:
        raise RuntimeError("Could not find DeiT-Small magnitude 50% reference row in pruning_matrix.csv.")

    baseline_mel = float(reference["mel_sensitivity"].iloc[0])
    baseline_bal = float(reference["balanced_acc"].iloc[0])

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(STYLE["figure_size"][0] * 2, STYLE["figure_size"][1]))

    axes[0].plot(ablation["calibration_size"], ablation["mel_sensitivity"], marker="o", linewidth=2.0)
    axes[0].axhline(baseline_mel, linestyle="--", color="black", linewidth=1.2, label="Magnitude 50%")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Calibration Set Size")
    axes[0].set_ylabel("Melanoma Sensitivity")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Melanoma Sensitivity")
    axes[0].legend()

    axes[1].plot(ablation["calibration_size"], ablation["balanced_acc"], marker="o", linewidth=2.0)
    axes[1].axhline(baseline_bal, linestyle="--", color="black", linewidth=1.2, label="Magnitude 50%")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Calibration Set Size")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Balanced Accuracy")
    axes[1].legend()

    fig.suptitle("Wanda Sensitivity to Calibration Set Size")
    save_figure(fig, figures_dir / "fig9_calibration_ablation.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Wanda calibration-size ablation.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
