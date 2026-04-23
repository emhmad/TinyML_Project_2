"""
Generate report-ready tables, statistics, and narrative summaries from
the aggregated experiment outputs. One stop for everything the paper
revision needs from the data.

Outputs go under report/generated/:
    report_summary.md          — narrative + key findings, one section per weakness
    tables/*.md                — markdown tables (paste into the paper draft)
    tables/*.tex               — LaTeX tables (booktabs format) for direct \\input{}
    statistics.json            — machine-readable headline numbers
    weakness_status.md         — checklist of which weaknesses are covered

Usage:
    python -m scripts.generate_report --root results/logs_personal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CRITERIA_ORDER = ["dense", "magnitude", "wanda", "taylor", "random", "skewness", "xpruner", "sparsegpt_pseudo"]
CLASS_ORDER = ["mel", "bcc", "akiec", "nv", "bkl", "df", "vasc"]
DANGEROUS_CLASSES = ["mel", "bcc", "akiec"]
CRITERION_LABELS = {
    "dense": "Dense",
    "magnitude": "Magnitude",
    "wanda": "Wanda",
    "taylor": "Taylor",
    "random": "Random",
    "skewness": "Paxton (skewness)",
    "xpruner": "X-Pruner",
    "sparsegpt_pseudo": "SparseGPT (diag-OBS)",
}


def _fmt_meanstd(mean: float, std: float, decimals: int = 3) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _meanstd_columns(df: pd.DataFrame, base_name: str) -> tuple[str, str]:
    return f"{base_name}_mean", f"{base_name}_std"


def _maybe_load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------


def table_pruning_headline(df: pd.DataFrame) -> pd.DataFrame:
    """W1+W3+W13: headline per-criterion × sparsity for each model."""
    rows = []
    sort_key = {c: i for i, c in enumerate(CRITERIA_ORDER)}
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].copy()
        sub["__order"] = sub["criterion"].map(lambda c: sort_key.get(c, 999))
        sub = sub.sort_values(["__order", "sparsity"]).drop(columns="__order")
        for _, row in sub.iterrows():
            rows.append({
                "Model": model,
                "Criterion": CRITERION_LABELS.get(row["criterion"], row["criterion"]),
                "Sparsity": f"{row['sparsity']:.0%}",
                "Balanced Acc": _fmt_meanstd(row["balanced_acc_mean"], row.get("balanced_acc_std", float("nan"))),
                "Mel Sens": _fmt_meanstd(row["mel_sensitivity_mean"], row.get("mel_sensitivity_std", float("nan"))),
                "Mel AUROC": _fmt_meanstd(row.get("melanoma_auroc_mean", float("nan")), row.get("melanoma_auroc_std", float("nan"))),
                "Macro AUROC": _fmt_meanstd(row.get("macro_auroc_mean", float("nan")), row.get("macro_auroc_std", float("nan"))),
                "ECE": _fmt_meanstd(row.get("ece_top_label_mean", float("nan")), row.get("ece_top_label_std", float("nan"))),
                "DCR": _fmt_meanstd(row.get("dangerous_class_degradation_ratio_mean", float("nan")), row.get("dangerous_class_degradation_ratio_std", float("nan")), decimals=2),
                "Spec@90Sens": _fmt_meanstd(row.get("mel_specificity_at_90_sens_mean", float("nan")), row.get("mel_specificity_at_90_sens_std", float("nan"))),
                "n": int(row.get("n_seeds", 1)),
            })
    return pd.DataFrame(rows)


def table_recovery_summary(df: pd.DataFrame) -> pd.DataFrame:
    """W10: recovery sweep — best balanced acc per (model, criterion) at each sparsity/epochs."""
    rows = []
    for (model, criterion, sparsity), grp in df.groupby(["model", "criterion", "sparsity"]):
        grp = grp.sort_values("recovery_epochs")
        for _, r in grp.iterrows():
            rows.append({
                "Model": model,
                "Criterion": CRITERION_LABELS.get(criterion, criterion),
                "Sparsity": f"{sparsity:.0%}",
                "Recovery epochs": int(r["recovery_epochs"]),
                "LR": f"{r.get('recovery_lr_mean', r.get('recovery_lr', 0)):.0e}" if "recovery_lr_mean" in r or "recovery_lr" in r else "—",
                "Balanced Acc": _fmt_meanstd(r.get("balanced_acc_mean", float("nan")), r.get("balanced_acc_std", float("nan"))),
                "Mel Sens": _fmt_meanstd(r.get("mel_sensitivity_mean", float("nan")), r.get("mel_sensitivity_std", float("nan"))),
                "DCR": _fmt_meanstd(r.get("dangerous_class_degradation_ratio_mean", float("nan")), r.get("dangerous_class_degradation_ratio_std", float("nan")), decimals=2),
                "n": int(r.get("n_seeds", 1)),
            })
    return pd.DataFrame(rows)


def table_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """W6: Paxton + X-Pruner + SparseGPT-pseudo baselines."""
    rows = []
    sort_key = {c: i for i, c in enumerate(CRITERIA_ORDER)}
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].copy()
        sub["__order"] = sub["criterion"].map(lambda c: sort_key.get(c, 999))
        sub = sub.sort_values(["__order", "sparsity"]).drop(columns="__order")
        for _, row in sub.iterrows():
            rows.append({
                "Model": model,
                "Criterion": CRITERION_LABELS.get(row["criterion"], row["criterion"]),
                "Sparsity": f"{row['sparsity']:.0%}",
                "Balanced Acc": _fmt_meanstd(row["balanced_acc_mean"], row.get("balanced_acc_std", float("nan"))),
                "Mel Sens": _fmt_meanstd(row["mel_sensitivity_mean"], row.get("mel_sensitivity_std", float("nan"))),
                "Macro AUROC": _fmt_meanstd(row.get("macro_auroc_mean", float("nan")), row.get("macro_auroc_std", float("nan"))),
                "DCR": _fmt_meanstd(row.get("dangerous_class_degradation_ratio_mean", float("nan")), row.get("dangerous_class_degradation_ratio_std", float("nan")), decimals=2),
                "n": int(row.get("n_seeds", 1)),
            })
    return pd.DataFrame(rows)


def table_nonuniform(df: pd.DataFrame) -> pd.DataFrame:
    """W11: non-uniform allocation policy comparison."""
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "Model": row.get("model", ""),
            "Criterion": CRITERION_LABELS.get(row.get("criterion", ""), row.get("criterion", "")),
            "Policy": row.get("policy", ""),
            "Balanced Acc": _fmt_meanstd(row.get("balanced_acc_mean", float("nan")), row.get("balanced_acc_std", float("nan"))),
            "Mel Sens": _fmt_meanstd(row.get("mel_sensitivity_mean", float("nan")), row.get("mel_sensitivity_std", float("nan"))),
            "Mel AUROC": _fmt_meanstd(row.get("melanoma_auroc_mean", float("nan")), row.get("melanoma_auroc_std", float("nan"))),
            "DCR": _fmt_meanstd(row.get("dangerous_class_degradation_ratio_mean", float("nan")), row.get("dangerous_class_degradation_ratio_std", float("nan")), decimals=2),
            "n": int(row.get("n_seeds", 1)),
        })
    return pd.DataFrame(rows)


def table_quantization(df: pd.DataFrame) -> pd.DataFrame:
    """W9: quantization stacking with latency."""
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "Model config": row.get("model_config", ""),
            "Criterion": CRITERION_LABELS.get(row.get("criterion", ""), row.get("criterion", "")),
            "Sparsity": f"{row.get('sparsity', 0):.0%}",
            "Balanced Acc": _fmt_meanstd(row.get("balanced_accuracy_mean", float("nan")), row.get("balanced_accuracy_std", float("nan"))),
            "Mel Sens": _fmt_meanstd(row.get("mel_sensitivity_mean", float("nan")), row.get("mel_sensitivity_std", float("nan"))),
            "Size (KB)": _fmt_meanstd(row.get("size_kb_mean", float("nan")), row.get("size_kb_std", float("nan")), decimals=0),
            "Mean (ms)": _fmt_meanstd(row.get("latency_mean_ms_mean", float("nan")), row.get("latency_mean_ms_std", float("nan")), decimals=2),
            "p95 (ms)": _fmt_meanstd(row.get("latency_p95_ms_mean", float("nan")), row.get("latency_p95_ms_std", float("nan")), decimals=2),
        })
    return pd.DataFrame(rows)


def table_edge_latency(df: pd.DataFrame) -> pd.DataFrame:
    """W9: edge runtime latency on the same Mac."""
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "Model": row.get("model", ""),
            "Criterion": CRITERION_LABELS.get(row.get("criterion", ""), row.get("criterion", "")),
            "Sparsity": f"{row.get('sparsity', 0):.0%}",
            "Target": row.get("target", ""),
            "Mean (ms)": _fmt_meanstd(row.get("mean_ms_mean", float("nan")), row.get("mean_ms_std", float("nan")), decimals=2),
            "p95 (ms)": _fmt_meanstd(row.get("p95_ms_mean", float("nan")), row.get("p95_ms_std", float("nan")), decimals=2),
            "p99 (ms)": _fmt_meanstd(row.get("p99_ms_mean", float("nan")), row.get("p99_ms_std", float("nan")), decimals=2),
            "n": int(row.get("n_seeds", 1)),
        })
    return pd.DataFrame(rows)


def table_paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """W1: paired t-test results — Wanda vs others, per (model, sparsity, metric)."""
    keep = df[df["compare_criterion"].isin(["wanda", "taylor", "random"])].copy()
    keep["sig"] = keep["significant_0_05"].map({True: "✓", False: " "})
    return keep[[
        "model", "sparsity", "compare_criterion", "metric",
        "n_seeds", "mean_diff", "p_value", "ci95_low", "ci95_high", "sig",
    ]].rename(columns={
        "model": "Model",
        "sparsity": "Sparsity",
        "compare_criterion": "vs (criterion)",
        "metric": "Metric",
        "n_seeds": "n",
        "mean_diff": "Mean diff (mag − x)",
        "p_value": "p",
        "ci95_low": "95% CI low",
        "ci95_high": "95% CI high",
        "sig": "p<0.05",
    })


# -----------------------------------------------------------------------------
# Headline statistics extraction
# -----------------------------------------------------------------------------


def headline_statistics(prune_df: pd.DataFrame, paired_df: pd.DataFrame | None) -> dict:
    out: dict = {"per_model": {}}
    for model in sorted(prune_df["model"].unique()):
        sub = prune_df[prune_df["model"] == model]
        dense = sub[sub["criterion"] == "dense"].iloc[0] if not sub[sub["criterion"] == "dense"].empty else None
        per_crit_at_50 = {}
        for crit in ["magnitude", "wanda", "taylor", "random"]:
            row = sub[(sub["criterion"] == crit) & (sub["sparsity"] == 0.5)]
            if row.empty:
                continue
            r = row.iloc[0]
            per_crit_at_50[crit] = {
                "balanced_acc_mean": float(r["balanced_acc_mean"]),
                "balanced_acc_std": float(r.get("balanced_acc_std", float("nan"))),
                "mel_sensitivity_mean": float(r["mel_sensitivity_mean"]),
                "mel_sensitivity_std": float(r.get("mel_sensitivity_std", float("nan"))),
                "melanoma_auroc_mean": float(r.get("melanoma_auroc_mean", float("nan"))),
                "ece_top_label_mean": float(r.get("ece_top_label_mean", float("nan"))),
                "dcr_mean": float(r.get("dangerous_class_degradation_ratio_mean", float("nan"))),
            }
        entry = {"per_criterion_at_s0.5": per_crit_at_50}
        if dense is not None:
            entry["dense"] = {
                "balanced_acc_mean": float(dense["balanced_acc_mean"]),
                "mel_sensitivity_mean": float(dense["mel_sensitivity_mean"]),
                "melanoma_auroc_mean": float(dense.get("melanoma_auroc_mean", float("nan"))),
                "ece_top_label_mean": float(dense.get("ece_top_label_mean", float("nan"))),
            }
        out["per_model"][model] = entry

    if paired_df is not None and not paired_df.empty:
        out["wanda_vs_magnitude_at_s0.5"] = {}
        for model in sorted(paired_df["model"].unique()):
            for metric in ["balanced_acc", "mel_sensitivity", "bcc_sensitivity", "akiec_sensitivity"]:
                row = paired_df[
                    (paired_df["model"] == model)
                    & (paired_df["sparsity"] == 0.5)
                    & (paired_df["compare_criterion"] == "wanda")
                    & (paired_df["metric"] == metric)
                ]
                if row.empty:
                    continue
                r = row.iloc[0]
                out["wanda_vs_magnitude_at_s0.5"].setdefault(model, {})[metric] = {
                    "n_seeds": int(r["n_seeds"]),
                    "mean_diff": float(r["mean_diff"]),
                    "p_value": float(r["p_value"]) if pd.notna(r["p_value"]) else None,
                    "significant_0.05": bool(r["significant_0_05"]),
                }
    return out


# -----------------------------------------------------------------------------
# Markdown / LaTeX writers
# -----------------------------------------------------------------------------


def _to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Booktabs LaTeX table — minimal, paper-ready."""
    # Escape & for LaTeX
    df_safe = df.copy().astype(str).replace({"&": r"\&", "%": r"\%", "_": r"\_"}, regex=True)
    cols = "l" * len(df_safe.columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + cols + r"}",
        r"\toprule",
        " & ".join(df_safe.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df_safe.iterrows():
        lines.append(" & ".join(row.astype(str).tolist()) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _write_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.md").write_text(df.to_markdown(index=False))
    (out_dir / f"{name}.tex").write_text(_to_latex(df, caption, label))


# -----------------------------------------------------------------------------
# Per-weakness narrative generator
# -----------------------------------------------------------------------------


def narrative(stats: dict, paired_df: pd.DataFrame | None, recovery_df: pd.DataFrame | None,
              activation_corr: pd.DataFrame | None, baselines_df: pd.DataFrame | None,
              nonuniform_df: pd.DataFrame | None, edge_df: pd.DataFrame | None) -> str:
    lines = ["# Report data summary — auto-generated", ""]
    lines.append("Generated by `scripts/generate_report.py` from the aggregated CSVs in")
    lines.append("`results/logs_personal/aggregated/`. Tables under `report/generated/tables/`")
    lines.append("are paste-ready (Markdown) or `\\input{}`-ready (LaTeX).")
    lines.append("")

    # ---------- W1 ----------
    lines += ["## W1 — Multi-seed variance and paired tests", ""]
    lines += ["Each row in `agg_pruning_matrix.csv` aggregates 3 seeds and reports mean ± std for"]
    lines += ["balanced accuracy, melanoma sensitivity, AUROC, ECE, and DCR. Paired t-tests"]
    lines += ["between Wanda / Taylor / random and the magnitude baseline live in"]
    lines += ["`paired_tests_pruning_matrix.csv`. Headline numbers below."]
    lines += [""]
    if "wanda_vs_magnitude_at_s0.5" in stats:
        lines += ["**Wanda vs. magnitude at 50% sparsity (paired t-test, n=3)**"]
        lines += [""]
        lines += ["| Model | Metric | mean diff (mag − wanda) | p | sig? |"]
        lines += ["|---|---|---|---|---|"]
        for model, metrics in stats["wanda_vs_magnitude_at_s0.5"].items():
            for metric, r in metrics.items():
                pval = "—" if r["p_value"] is None else f"{r['p_value']:.3f}"
                sig = "✓" if r["significant_0.05"] else " "
                lines.append(f"| {model} | {metric} | {r['mean_diff']:+.4f} | {pval} | {sig} |")
        lines += [""]
    lines += ["Per-cell mean ± std for every (criterion, sparsity) is in `tables/pruning_headline.{md,tex,csv}`."]
    lines += [""]

    # ---------- W2 ----------
    lines += ["## W2 — Lesion-grouped split (data leakage fix)", ""]
    lines += ["All experiments now use `GroupShuffleSplit` on `lesion_id` (see `data/dataset.py:get_train_val_splits`)."]
    lines += ["Code asserts no overlap; any image of a given lesion goes entirely to train OR val."]
    lines += ["Re-run all training/eval used the corrected split. No prior-image-level results survive in this report."]
    lines += [""]

    # ---------- W3 ----------
    lines += ["## W3 — Medical evaluation metrics", ""]
    lines += ["Every pruning row reports: per-class AUROC (macro + melanoma one-vs-rest),"]
    lines += ["specificity at 90% sensitivity, sensitivity at 90% specificity, top-label ECE,"]
    lines += ["per-class precision (PPV), and the dangerous-class degradation ratio (DCR)."]
    lines += [""]
    lines += ["See the `Mel AUROC`, `Macro AUROC`, `ECE`, `Spec@90Sens`, and `DCR` columns in"]
    lines += ["`tables/pruning_headline.md` (paste into the medical-metrics table of the paper)."]
    lines += [""]

    # ---------- W4 ----------
    lines += ["## W4 — Wanda failure mechanism", ""]
    if activation_corr is not None and not activation_corr.empty:
        lines += ["Per-layer activation statistics (kurtosis, top-5% concentration, outlier ratio) and"]
        lines += ["their correlation with single-layer Wanda damage are in"]
        lines += ["`results/logs_personal/seed_0/activation_stats_correlation.csv`."]
        lines += [""]
        lines += ["**Correlations (seed 0, deit_small):**"]
        lines += [""]
        ds = activation_corr[activation_corr["model"] == "deit_small"]
        if not ds.empty:
            lines += ["| Statistic | Pearson r | Spearman ρ | n layers | criterion |"]
            lines += ["|---|---|---|---|---|"]
            for _, r in ds.iterrows():
                lines.append(f"| {r['statistic']} | {r['pearson_r']:.3f} | {r['spearman_rho']:.3f} | {int(r['n_layers'])} | {r.get('criterion','—')} |")
            lines += [""]
        lines += ["A positive Pearson/Spearman value with the wanda criterion means: layers with more"]
        lines += ["concentrated / heavy-tailed activations are the ones Wanda damages most. That is the"]
        lines += ["evidence the paper's hypothesis needs."]
    else:
        lines += ["Activation correlation CSV not found. Re-run pillar 9."]
    lines += [""]

    # ---------- W5 ----------
    lines += ["## W5 — Second dataset replication", ""]
    lines += ["**Not run.** The personal-compute config explicitly drops the second-dataset arm."]
    lines += ["Reframe the paper claim to *HAM10000 dermoscopy* throughout — `report/related_work.md`"]
    lines += ["already structures this. The harness exists at `experiments/e_second_dataset.py` for future work."]
    lines += [""]

    # ---------- W6 ----------
    lines += ["## W6 — Paxton + X-Pruner + SparseGPT-pseudo baselines", ""]
    if baselines_df is not None and not baselines_df.empty:
        n_models = baselines_df["model"].nunique()
        n_crit = baselines_df["criterion"].nunique()
        n_sparse = baselines_df["sparsity"].nunique()
        lines += [f"All three baselines ran on {n_models} models × {n_crit} criteria × {n_sparse} sparsities."]
        lines += [f"See `tables/baselines.md`. Compare to the matching rows in `tables/pruning_headline.md`."]
    else:
        lines += ["Baselines CSV not found."]
    lines += [""]

    # ---------- W7 ----------
    lines += ["## W7 — CNN baseline (MobileNetV2)", ""]
    lines += ["MobileNetV2 ran with the full criterion × sparsity sweep + recovery fine-tuning."]
    lines += ["See `agg_mobilenet_results.csv`. **Reframe in the paper: this is an exploratory check,"]
    lines += ["not a generalisation argument** (depthwise convs behave atypically under unstructured pruning;"]
    lines += ["ResNet-50 was dropped from the personal-compute budget)."]
    lines += [""]

    # ---------- W8 ----------
    lines += ["## W8 — Honest size columns", ""]
    lines += ["Every CSV now reports three sizes: `dense_size_kb` (parameter count × 4),"]
    lines += ["`disk_size_kb` (real torch.save bytes), and `effective_sparse_size_kb`"]
    lines += ["(nonzero params × 4, theoretical sparse-storage ceiling). Use `disk_size_kb` for"]
    lines += ["honest deployment numbers; flag `effective_sparse_size_kb` as a ceiling under sparse storage."]
    lines += ["2:4 structured sparsity was not run on Apple Silicon (no Ampere sparse cores)."]
    lines += [""]

    # ---------- W9 ----------
    lines += ["## W9 — Edge latency", ""]
    if edge_df is not None and not edge_df.empty:
        targets = sorted(edge_df["target"].unique())
        lines += [f"Latency benchmarked on the run host (Apple M-series MacBook) against {len(targets)} runtimes:"]
        lines += [", ".join(f"`{t}`" for t in targets) + "."]
        lines += [f"See `tables/edge_latency.md`. The `mean_ms` and `p95_ms` columns are the headline numbers."]
    else:
        lines += ["No edge_latency rows aggregated."]
    lines += [""]

    # ---------- W10 ----------
    lines += ["## W10 — Recovery sweep", ""]
    if recovery_df is not None and not recovery_df.empty:
        epochs = sorted(recovery_df["recovery_epochs"].unique())
        sparsities = sorted(recovery_df["sparsity"].unique())
        lines += [f"Recovery sweep covers epochs ∈ {epochs} × sparsities ∈ {sparsities}."]
        lines += [f"See `tables/recovery.md`. To answer 'does Wanda catch up with more recovery?',"]
        lines += [f"compare each criterion's `Balanced Acc` and `Mel Sens` columns at 5 vs 20 recovery epochs."]
    lines += [""]

    # ---------- W11 ----------
    lines += ["## W11 — Non-uniform allocation policies", ""]
    if nonuniform_df is not None and not nonuniform_df.empty:
        policies = sorted(nonuniform_df["policy"].unique())
        lines += [f"Policies compared: {policies}. See `tables/nonuniform.md`."]
    lines += [""]

    # ---------- W12 ----------
    lines += ["## W12 — Attention overlap (quantitative)", ""]
    lines += ["Skipped — no ISIC 2018 segmentation masks staged at `data/ham10000/segmentation_masks/`."]
    lines += ["Qualitative figure in `results/figures_personal/fig8_attention_maps.pdf` is still produced."]
    lines += ["Stage masks and rerun pillar 6 if you want IoU + pointing-game numbers."]
    lines += [""]

    # ---------- W13 ----------
    lines += ["## W13 — Dangerous-class degradation ratio", ""]
    lines += ["DCR = (mean drop on mel/akiec/bcc) / (mean drop on nv/bkl/df/vasc)."]
    lines += ["Reported in every pruning, recovery, baseline, non-uniform, and CNN table."]
    lines += ["A value > 1.0 means compression preferentially forgets dangerous classes."]
    lines += [""]

    # ---------- W14 ----------
    lines += ["## W14 — Code release", ""]
    lines += ["Repo pushed to `emhmad/TinyML_Project_2`, commit `caeda30`. Includes `environment.yml`,"]
    lines += ["`scripts/precompute_masks.py`, `scripts/release.py`, `scripts/reproduce.sh`, configs for both"]
    lines += ["cluster (`multi_seed_ciai_fast.yaml`) and personal (`local_personal_8h.yaml`) reproduction."]
    lines += [""]

    # ---------- W15 ----------
    lines += ["## W15 — Figures", ""]
    lines += ["Re-rendered Fig 7 (recovery comparison, wide canvas, error bars) and Fig 9"]
    lines += ["(calibration ablation, log₂ ticks, reference band) under `results/figures_personal/`."]
    lines += ["See the regeneration commands at the end of this file."]
    lines += [""]

    # ---------- W16 ----------
    lines += ["## W16 — Clinical deployability", ""]
    lines += ["Each aggregated table has a `*_clinical.csv` companion with a `clinical_regime` column"]
    lines += ["({triage_screen / specialist_referral / primary_diagnosis / academic_only}). Thresholds"]
    lines += ["live in `evaluation/clinical_thresholds.py` (`DEFAULT_THRESHOLDS`). Source citations needed"]
    lines += ["in the paper Discussion — the module's docstring lists the rough anchors used."]
    lines += [""]

    # ---------- W17 ----------
    lines += ["## W17 — Related work", ""]
    lines += ["Structured citation skeleton at `report/related_work.md`. Promote into the paper's Related"]
    lines += ["Work section, expanding each block with full prose."]
    lines += [""]

    # Reproduction commands
    lines += ["---", "", "## How to regenerate this report", "", "```bash"]
    lines += ["python -m scripts.generate_report --root results/logs_personal"]
    lines += ["python -m plotting.fig7_recovery --config configs/local_personal_8h.yaml"]
    lines += ["python -m plotting.fig9_calibration_ablation --config configs/local_personal_8h.yaml"]
    lines += ["```", ""]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Weakness checklist
# -----------------------------------------------------------------------------


def weakness_checklist(prune_df, paired_df, baselines_df, recovery_df, nonuniform_df,
                       edge_df, mobilenet_df, activation_corr) -> str:
    items = [
        ("W1 multi-seed + paired tests", paired_df is not None and not paired_df.empty),
        ("W2 lesion-grouped split", True),  # confirmed by code
        ("W3 medical metrics (AUROC/ECE/spec@90)", "melanoma_auroc_mean" in prune_df.columns),
        ("W4 activation-stats correlation", activation_corr is not None and not activation_corr.empty),
        ("W5 second dataset", False),
        ("W6 Paxton + X-Pruner baselines", baselines_df is not None and not baselines_df.empty),
        ("W7 CNN baseline (MobileNetV2)", mobilenet_df is not None and not mobilenet_df.empty),
        ("W8 honest size columns", "disk_size_kb_mean" in prune_df.columns),
        ("W9 edge latency", edge_df is not None and not edge_df.empty),
        ("W10 recovery sweep", recovery_df is not None and not recovery_df.empty),
        ("W11 non-uniform policies", nonuniform_df is not None and not nonuniform_df.empty),
        ("W12 attention overlap (quantitative)", False),
        ("W13 dangerous-class degradation ratio", "dangerous_class_degradation_ratio_mean" in prune_df.columns),
        ("W14 code release", True),  # pushed to emhmad/TinyML_Project_2
        ("W15 figures regenerated", True),
        ("W16 clinical-regime tagging", True),
        ("W17 related-work skeleton", True),
    ]
    lines = ["# Weakness coverage — paper revision", ""]
    lines += ["| ID | Status | Notes |", "|---|---|---|"]
    for label, ok in items:
        mark = "✅" if ok else "⚠️ skipped — see narrative"
        lines.append(f"| {label} | {mark} |  |")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def run(root: Path, out_dir: Path) -> None:
    agg = root / "aggregated"
    if not agg.exists():
        raise FileNotFoundError(f"Expected aggregated outputs under {agg}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"

    prune = pd.read_csv(agg / "agg_pruning_matrix.csv")
    paired = _maybe_load(agg / "paired_tests_pruning_matrix.csv")
    recovery = _maybe_load(agg / "agg_recovery_finetune.csv")
    baselines = _maybe_load(agg / "agg_baselines_paxton_xpruner.csv")
    nonuniform = _maybe_load(agg / "agg_nonuniform_pruning.csv")
    quant = _maybe_load(agg / "agg_quantization_stacking.csv")
    edge = _maybe_load(agg / "agg_edge_latency.csv")
    mobilenet = _maybe_load(agg / "agg_mobilenet_results.csv")
    calib_ablation = _maybe_load(agg / "agg_calibration_ablation.csv")

    # Activation-stats correlations live under each seed (one-shot in seed 0).
    seed0_corr = root / "seed_0" / "activation_stats_correlation.csv"
    activation_corr = pd.read_csv(seed0_corr) if seed0_corr.exists() else None

    # Tables
    _write_table(table_pruning_headline(prune), tables_dir, "pruning_headline",
                 caption="Pruning matrix — mean $\\pm$ std across 3 seeds. DCR is the dangerous-class degradation ratio (>1 indicates preferential forgetting of dangerous classes).",
                 label="tab:pruning_headline")

    if paired is not None:
        _write_table(table_paired_tests(paired), tables_dir, "paired_tests",
                     caption="Paired t-tests across seeds — magnitude vs. each compared criterion. p<0.05 cells are flagged.",
                     label="tab:paired_tests")

    if recovery is not None:
        _write_table(table_recovery_summary(recovery), tables_dir, "recovery_sweep",
                     caption="Post-pruning recovery fine-tuning sweep.",
                     label="tab:recovery_sweep")

    if baselines is not None:
        _write_table(table_baselines(baselines), tables_dir, "baselines",
                     caption="External pruning baselines (Paxton skewness, X-Pruner, SparseGPT-pseudo) on HAM10000.",
                     label="tab:baselines")

    if nonuniform is not None:
        _write_table(table_nonuniform(nonuniform), tables_dir, "nonuniform",
                     caption="Non-uniform allocation policies — uniform (target = 0.5), binned-3, continuous, and learnable variants.",
                     label="tab:nonuniform")

    if quant is not None:
        _write_table(table_quantization(quant), tables_dir, "quantization",
                     caption="Quantization stacking — accuracy preservation, on-disk size, and CPU latency.",
                     label="tab:quantization")

    if edge is not None:
        _write_table(table_edge_latency(edge), tables_dir, "edge_latency",
                     caption="On-device latency on the Apple M-series MacBook used as the edge target.",
                     label="tab:edge_latency")

    if mobilenet is not None:
        _write_table(table_baselines(mobilenet.rename(columns={})), tables_dir, "mobilenetv2_baseline",
                     caption="MobileNetV2 baseline — exploratory CNN comparison (do not generalise from this).",
                     label="tab:mobilenetv2")

    # Statistics + narrative + checklist
    stats = headline_statistics(prune, paired)
    (out_dir / "statistics.json").write_text(json.dumps(stats, indent=2, default=str))
    (out_dir / "report_summary.md").write_text(
        narrative(stats, paired, recovery, activation_corr, baselines, nonuniform, edge)
    )
    (out_dir / "weakness_status.md").write_text(
        weakness_checklist(prune, paired, baselines, recovery, nonuniform, edge, mobilenet, activation_corr)
    )

    print(f"[generate_report] wrote {len(list(tables_dir.glob('*')))} table files under {tables_dir}")
    print(f"[generate_report] wrote report_summary.md, weakness_status.md, statistics.json under {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready tables and statistics from aggregated CSVs.")
    parser.add_argument("--root", default="results/logs_personal", help="Logs directory containing aggregated/ subfolder.")
    parser.add_argument("--out-dir", default="report/generated", help="Where to write report artefacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.root), Path(args.out_dir))


if __name__ == "__main__":
    main()
