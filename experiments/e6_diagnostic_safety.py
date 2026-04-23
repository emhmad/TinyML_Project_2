from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.config import apply_seed_to_paths, load_config
from utils.io import ensure_dir


def run(config_path: str, seed_override: int | None = None) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    results_dir = ensure_dir(config["logging"]["results_dir"])
    source_path = Path(results_dir) / "pruning_matrix.csv"
    output_path = Path(results_dir) / "diagnostic_safety.csv"

    frame = pd.read_csv(source_path)
    dense = frame[frame["criterion"] == "dense"][["model", "overall_acc", "balanced_acc", "mel_sensitivity"]].copy()
    dense = dense.rename(
        columns={
            "overall_acc": "dense_overall_acc",
            "balanced_acc": "dense_balanced_acc",
            "mel_sensitivity": "dense_mel_sensitivity",
        }
    )

    comparison = frame[frame["criterion"] != "dense"].merge(dense, on="model", how="left")
    comparison["melanoma_sensitivity_drop"] = comparison["dense_mel_sensitivity"] - comparison["mel_sensitivity"]
    comparison["balanced_accuracy_drop"] = comparison["dense_balanced_acc"] - comparison["balanced_acc"]
    comparison.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build diagnostic safety summary from pruning results.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, seed_override=args.seed)


if __name__ == "__main__":
    main()
