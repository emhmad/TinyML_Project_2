"""
Second-dataset replication harness (W5).

The reviewer asked for at least one additional medical dataset to
defend the "compression in medical AI" claim. This script reuses the
DeiT pruning pipeline against an alternative processed-metadata CSV
(ISIC 2019 dermoscopy, CheXpert multi-label chest X-rays, PatchCamelyon
histopathology, etc.), with the constraint that the dataset has:

  * a CSV at `dataset.metadata_csv` with columns `image_path`, `label_idx`
  * a grouping column (usually `lesion_id` or `patient_id`) for the split
  * the same 7-class label map, OR a per-dataset label map supplied via
    `dataset.label_map` in the config

Data preparation is out of scope for this script — it expects the CSV
to already exist. The config path is kept at the top of the runner so
the sweep driver can iterate `configs/second_dataset_*.yaml` without
touching any Python.

If the second dataset is not available yet, `run()` exits early with a
clear message rather than raising — this keeps the pipeline green on
the main dataset while you stage the secondary data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments import e3_calibration, e4_pruning_matrix, e_baselines_paxton_xpruner, e_recovery_finetune
from utils.config import load_config


def run(config_path: str, seed_override: int | None = None) -> None:
    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {}) or {}
    metadata_csv = dataset_cfg.get("metadata_csv")
    if not metadata_csv or not Path(metadata_csv).exists():
        print(
            f"[e_second_dataset] metadata CSV not found at {metadata_csv!r}. "
            "Skipping: prepare the secondary dataset CSV and rerun."
        )
        return

    # Calibrate, run pruning matrix, baseline pruners, and recovery
    # fine-tuning against the alternative dataset. The evaluation
    # metrics auto-adapt — they key off the class_names list, which is
    # read from the dataset module (HAM10000 default). For non-skin
    # datasets, patch evaluation.metrics.CLASS_NAMES or pass the
    # `class_names` override via the config to keep per-class columns
    # sensible.
    e3_calibration.run(config_path, seed_override=seed_override)
    e4_pruning_matrix.run(config_path, seed_override=seed_override)
    e_baselines_paxton_xpruner.run(config_path, seed_override=seed_override)
    e_recovery_finetune.run(config_path, seed_override=seed_override)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate core pruning comparison on a second dataset.")
    parser.add_argument("--config", required=True, help="YAML config pointing at the alternative dataset.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, seed_override=args.seed)


if __name__ == "__main__":
    main()
