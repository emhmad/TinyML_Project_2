"""
Annotate aggregated result CSVs with clinical deployment regimes (W16).

Reads every aggregated result CSV under `<results_root>/aggregated/` and
emits a parallel `..._clinical.csv` with a `clinical_regime` column
(`triage_screen`, `specialist_referral`, `primary_diagnosis`,
`academic_only`). Also emits a top-level `clinical_thresholds.csv`
documenting which thresholds were used.

Usage:
    python -m scripts.clinical_gating --root results/logs_ciai
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation.clinical_thresholds import DEFAULT_THRESHOLDS, annotate_frame, threshold_table


_DEFAULT_METRIC_COLUMNS = {
    "mel_sensitivity_column": "mel_sensitivity_mean",
    "mel_specificity_column": "mel_specificity_at_90_sens_mean",
    "balanced_accuracy_column": "balanced_acc_mean",
}


def _pick_columns(frame: pd.DataFrame) -> dict:
    """
    Aggregated CSVs use `_mean` suffixed columns; per-seed CSVs use the
    raw names. Pick whichever schema matches.
    """
    cols = set(frame.columns)
    if "mel_sensitivity_mean" in cols:
        return dict(_DEFAULT_METRIC_COLUMNS)
    return {
        "mel_sensitivity_column": "mel_sensitivity",
        "mel_specificity_column": "mel_specificity_at_90_sens",
        "balanced_accuracy_column": "balanced_acc",
    }


def run(root: str | Path) -> None:
    root = Path(root)
    out_dir = root / "aggregated"
    if not out_dir.exists():
        raise FileNotFoundError(f"Expected aggregated results under {out_dir}.")

    threshold_table(DEFAULT_THRESHOLDS).to_csv(
        out_dir / "clinical_thresholds.csv", index=False
    )

    for csv_path in sorted(out_dir.glob("agg_*.csv")):
        frame = pd.read_csv(csv_path)
        if "mel_sensitivity_mean" not in frame.columns and "mel_sensitivity" not in frame.columns:
            continue
        column_kwargs = _pick_columns(frame)
        annotated = annotate_frame(frame, thresholds=DEFAULT_THRESHOLDS, **column_kwargs)
        annotated.to_csv(csv_path.with_name(csv_path.stem + "_clinical.csv"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag aggregated results with clinical regimes (W16).")
    parser.add_argument("--root", required=True, help="Logs directory (contains aggregated/)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
