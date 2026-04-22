from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from evaluation.latency import measure_latency
from evaluation.metrics import CLASS_NAMES, dangerous_class_degradation_ratio, evaluate_model
from evaluation.model_size import get_model_size_kb
from experiments.common import build_dataloaders, load_trained_model, model_alias
from pruning.masking import apply_masks
from quantization.ptq import get_quantized_model_size, quantize_model_dynamic
from utils.config import apply_seed_to_paths, load_config, resolve_seed
from utils.io import append_csv_row, ensure_dir
from utils.seed import set_seed


def _cpu_eval(model, val_loader):
    return evaluate_model(model, val_loader, torch.device("cpu"), class_names=CLASS_NAMES)


def _latency(model, warmup: int, timed_runs: int) -> dict[str, float]:
    result = measure_latency(
        model,
        device="cpu",
        warmup=warmup,
        timed_runs=timed_runs,
        warmup_seconds=1.0,
    )
    return {
        "latency_median_ms": result["median_ms"],
        "latency_mean_ms": result["mean_ms"],
        "latency_std_ms": result["std_ms"],
        "latency_p95_ms": result["p95_ms"],
        "latency_p99_ms": result["p99_ms"],
    }


def _row(
    *,
    seed: int,
    model_config: str,
    criterion: str,
    sparsity: float,
    metrics: dict,
    size_kb: float,
    latency: dict[str, float],
    baseline_sensitivity: dict[str, float] | None,
) -> dict[str, Any]:
    row = {
        "seed": seed,
        "model_config": model_config,
        "criterion": criterion,
        "sparsity": float(sparsity),
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_auroc": metrics.get("macro_auroc"),
        "melanoma_auroc": metrics.get("melanoma_auroc"),
        "ece_top_label": metrics.get("ece_top_label"),
        "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
        "bcc_sensitivity": metrics["per_class_sensitivity"]["bcc"],
        "akiec_sensitivity": metrics["per_class_sensitivity"]["akiec"],
        "nv_sensitivity": metrics["per_class_sensitivity"]["nv"],
        "bkl_sensitivity": metrics["per_class_sensitivity"]["bkl"],
        "df_sensitivity": metrics["per_class_sensitivity"]["df"],
        "vasc_sensitivity": metrics["per_class_sensitivity"]["vasc"],
        "size_kb": float(size_kb),
        **latency,
    }
    if baseline_sensitivity is not None and model_config != "dense":
        row["dangerous_class_degradation_ratio"] = dangerous_class_degradation_ratio(
            baseline_sensitivity, metrics["per_class_sensitivity"]
        )
    return row


def run(config_path: str, seed_override: int | None = None) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    set_seed(seed)

    _, val_loader, _, _ = build_dataloaders(config, include_train=False)
    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    masks_dir = checkpoint_dir / "masks"
    log_path = Path(config["logging"]["results_dir"]) / "quantization_stacking.csv"
    ensure_dir(log_path.parent)

    model_name = config["models"]["teacher"]
    alias = model_alias(model_name)
    warmup = int(config["evaluation"].get("latency_warmup_runs", 10))
    timed = int(config["evaluation"].get("latency_timed_runs", 100))

    # Dense + quantised-only pair.
    dense_model = load_trained_model(config, model_name, torch.device("cpu"), checkpoint_name=f"{alias}_ham10000")
    dense_metrics = _cpu_eval(dense_model, val_loader)
    baseline_sensitivity = dict(dense_metrics["per_class_sensitivity"])
    append_csv_row(
        log_path,
        _row(
            seed=seed,
            model_config="dense",
            criterion="none",
            sparsity=0.0,
            metrics=dense_metrics,
            size_kb=get_model_size_kb(dense_model)["size_kb"],
            latency=_latency(dense_model, warmup, timed),
            baseline_sensitivity=None,
        ),
    )

    quant_only = quantize_model_dynamic(dense_model)
    quant_only_metrics = _cpu_eval(quant_only, val_loader)
    append_csv_row(
        log_path,
        _row(
            seed=seed,
            model_config="quantized_only",
            criterion="none",
            sparsity=0.0,
            metrics=quant_only_metrics,
            size_kb=get_quantized_model_size(quant_only),
            latency=_latency(quant_only, warmup, timed),
            baseline_sensitivity=baseline_sensitivity,
        ),
    )

    # Cross product over criteria and sparsities that actually have masks
    # on disk. Previously this was hardcoded to wanda+magnitude at 0.5 —
    # now we just ingest every mask the pruning matrix produced.
    criteria = config["pruning"].get("quant_stack_criteria") or ["magnitude", "wanda", "taylor"]
    sparsities = config["pruning"].get("quant_stack_sparsities") or [0.5]
    for criterion_name in criteria:
        for sparsity in sparsities:
            mask_path = masks_dir / f"{alias}_{criterion_name}_s{float(sparsity):.1f}.pt"
            if not mask_path.exists():
                continue
            pruned = load_trained_model(
                config, model_name, torch.device("cpu"), checkpoint_name=f"{alias}_ham10000"
            )
            masks = torch.load(mask_path, map_location="cpu")
            apply_masks(pruned, masks)
            pruned_metrics = _cpu_eval(pruned, val_loader)
            append_csv_row(
                log_path,
                _row(
                    seed=seed,
                    model_config="pruned_only",
                    criterion=criterion_name,
                    sparsity=float(sparsity),
                    metrics=pruned_metrics,
                    size_kb=get_model_size_kb(pruned, sparse=True)["size_kb"],
                    latency=_latency(pruned, warmup, timed),
                    baseline_sensitivity=baseline_sensitivity,
                ),
            )

            stacked = quantize_model_dynamic(pruned)
            stacked_metrics = _cpu_eval(stacked, val_loader)
            append_csv_row(
                log_path,
                _row(
                    seed=seed,
                    model_config="pruned_plus_quantized",
                    criterion=criterion_name,
                    sparsity=float(sparsity),
                    metrics=stacked_metrics,
                    size_kb=get_quantized_model_size(stacked),
                    latency=_latency(stacked, warmup, timed),
                    baseline_sensitivity=baseline_sensitivity,
                ),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantization stacking experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, seed_override=args.seed)


if __name__ == "__main__":
    main()
