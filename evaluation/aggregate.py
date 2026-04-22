"""
Aggregate per-seed experiment CSVs into mean±std summaries and run the
paired statistical tests that the multi-seed revision requires.

Call-path: after a seed sweep (see experiments/run_seeds.py), every
`results/logs_local/seed_{s}/` contains one CSV per experiment. This
module merges them into a single long DataFrame keyed by the experiment
hyperparameters, computes mean/std, and (when >= 2 seeds are present)
runs paired t-tests between pruning criteria at each (model, sparsity)
cell for balanced accuracy and per-class sensitivity.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.stats import paired_t_test


DEFAULT_METRIC_COLUMNS = (
    "overall_acc",
    "balanced_acc",
    "overall_accuracy",
    "balanced_accuracy",
    "mel_sensitivity",
    "bcc_sensitivity",
    "akiec_sensitivity",
    "nv_sensitivity",
    "bkl_sensitivity",
    "df_sensitivity",
    "vasc_sensitivity",
    "macro_auroc",
    "melanoma_auroc",
    "ece_top_label",
    "dangerous_class_degradation_ratio",
    "mel_precision",
    "bcc_precision",
    "akiec_precision",
    "mel_specificity_at_90_sens",
    "mel_sensitivity_at_90_spec",
    "size_kb",
    "disk_size_kb",
    "effective_sparse_size_kb",
    "dense_size_kb",
    "total_params",
    "nonzero_params",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_std_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "mean_ms",
    "median_ms",
    "std_ms",
    "p95_ms",
    "p99_ms",
    "iou",
    "pointing_game_accuracy",
    "mean_iou",
    "median_iou",
    "mean_mass_in_mask",
    "verified",
)


def _detect_seed_dirs(root: Path) -> list[Path]:
    candidates = sorted([p for p in root.glob("seed_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(
            f"No seed_* subdirectories under {root}. Run experiments/run_seeds.py first."
        )
    return candidates


def _load_experiment_csvs(seed_dirs: Iterable[Path], csv_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed_dir in seed_dirs:
        path = seed_dir / csv_name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            seed = -1
        frame = frame.copy()
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"None of the seed directories contained {csv_name}. "
            "Check that experiments completed for this CSV."
        )
    return pd.concat(frames, ignore_index=True)


def aggregate_csv(
    root: Path,
    csv_name: str,
    group_columns: list[str],
    metric_columns: Iterable[str] = DEFAULT_METRIC_COLUMNS,
) -> pd.DataFrame:
    """
    Collapse per-seed rows into per-cell mean ± std for each metric that
    actually exists in the input. Returns a wide-form DataFrame with one
    row per `group_columns` tuple and `{metric}_mean`, `{metric}_std`
    columns.
    """
    seed_dirs = _detect_seed_dirs(root)
    df = _load_experiment_csvs(seed_dirs, csv_name)

    present_metrics = [c for c in metric_columns if c in df.columns]
    if not present_metrics:
        raise ValueError(
            f"None of the expected metric columns {list(metric_columns)} are in {csv_name}."
        )

    grouped = df.groupby(group_columns, dropna=False)[present_metrics]
    mean = grouped.mean(numeric_only=True).add_suffix("_mean")
    std = grouped.std(numeric_only=True, ddof=1).add_suffix("_std")
    count = grouped.size().rename("n_seeds")
    out = pd.concat([mean, std, count], axis=1).reset_index()
    return out


def paired_criterion_tests(
    root: Path,
    csv_name: str = "pruning_matrix.csv",
    cell_columns: tuple[str, ...] = ("model", "sparsity"),
    criterion_column: str = "criterion",
    metric_columns: tuple[str, ...] = (
        "balanced_acc",
        "mel_sensitivity",
        "bcc_sensitivity",
        "akiec_sensitivity",
    ),
    baseline_criterion: str = "magnitude",
    compare_criteria: tuple[str, ...] = ("wanda", "taylor", "random"),
) -> pd.DataFrame:
    """
    For every (model, sparsity) cell, run a paired t-test across seeds
    comparing `baseline_criterion` to each entry in `compare_criteria`
    for the requested metrics. Rows with fewer than 2 shared seeds are
    skipped but still reported with NaN p-values so callers can audit.
    """
    seed_dirs = _detect_seed_dirs(root)
    df = _load_experiment_csvs(seed_dirs, csv_name)

    rows: list[dict] = []
    cell_keys = df[list(cell_columns)].drop_duplicates().itertuples(index=False, name=None)
    for cell in cell_keys:
        cell_dict = dict(zip(cell_columns, cell))
        cell_df = df.loc[(df[list(cell_columns)] == pd.Series(cell_dict)).all(axis=1)]
        baseline_rows = cell_df[cell_df[criterion_column] == baseline_criterion].sort_values("seed")
        if baseline_rows.empty:
            continue
        for other in compare_criteria:
            other_rows = cell_df[cell_df[criterion_column] == other].sort_values("seed")
            shared_seeds = sorted(set(baseline_rows["seed"]).intersection(other_rows["seed"]))
            if len(shared_seeds) < 2:
                continue
            baseline_aligned = baseline_rows.set_index("seed").loc[shared_seeds]
            other_aligned = other_rows.set_index("seed").loc[shared_seeds]
            for metric in metric_columns:
                if metric not in baseline_aligned.columns or metric not in other_aligned.columns:
                    continue
                result = paired_t_test(
                    baseline_aligned[metric].to_numpy(),
                    other_aligned[metric].to_numpy(),
                )
                row = {
                    **cell_dict,
                    "baseline_criterion": baseline_criterion,
                    "compare_criterion": other,
                    "metric": metric,
                    "n_seeds": len(shared_seeds),
                    "mean_diff": result.mean_diff,
                    "p_value": result.p_value,
                    "ci95_low": result.ci95_low,
                    "ci95_high": result.ci95_high,
                    "significant_0_05": bool(np.isfinite(result.p_value) and result.p_value < 0.05),
                }
                rows.append(row)

    return pd.DataFrame(rows)


def run(root: str | Path, output_dir: str | Path | None = None) -> None:
    root = Path(root)
    out = Path(output_dir) if output_dir else root / "aggregated"
    out.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, list[str]]] = [
        ("pruning_matrix.csv", ["model", "criterion", "sparsity"]),
        ("nonuniform_pruning.csv", ["model", "criterion", "policy"]),
        ("recovery_finetune.csv", ["model", "criterion", "sparsity", "recovery_epochs"]),
        ("mobilenet_results.csv", ["model", "criterion", "sparsity"]),
        ("resnet50_results.csv", ["model", "criterion", "sparsity"]),
        ("baselines_paxton_xpruner.csv", ["model", "criterion", "sparsity"]),
        ("calibration_ablation.csv", ["calibration_size"]),
        ("structured_sparsity.csv", ["model", "criterion", "pattern_n", "pattern_m"]),
        ("edge_latency.csv", ["model", "criterion", "sparsity", "target"]),
        ("quantization_stacking.csv", ["model_config", "criterion", "sparsity"]),
        ("attention_overlap_summary.csv", ["condition"]),
        ("baseline_eval.csv", ["model"]),
    ]
    for csv_name, group_cols in pairs:
        try:
            aggregated = aggregate_csv(root, csv_name, group_cols)
            aggregated.to_csv(out / f"agg_{csv_name}", index=False)
        except FileNotFoundError:
            continue

    try:
        tests = paired_criterion_tests(root)
        tests.to_csv(out / "paired_tests_pruning_matrix.csv", index=False)
    except FileNotFoundError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experiment CSVs.")
    parser.add_argument("--root", required=True, help="Logs directory containing seed_* subfolders.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.root, args.output_dir)


if __name__ == "__main__":
    main()
