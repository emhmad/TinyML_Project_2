from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluation.latency import measure_latency
from evaluation.metrics import CLASS_NAMES, evaluate_model
from evaluation.model_size import get_model_size_kb
from experiments.common import build_dataloaders, load_trained_model, model_alias
from pruning.masking import apply_masks
from quantization.ptq import get_quantized_model_size, quantize_model_dynamic
from utils.config import load_config
from utils.io import append_csv_row


def _cpu_eval(model, val_loader):
    return evaluate_model(model, val_loader, torch.device("cpu"), class_names=CLASS_NAMES)


def run(config_path: str) -> None:
    config = load_config(config_path)
    _, val_loader, _, _ = build_dataloaders(config, include_train=False)
    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    masks_dir = checkpoint_dir / "masks"
    log_path = Path(config["logging"]["results_dir"]) / "quantization_stacking.csv"
    model_name = config["models"]["teacher"]
    alias = model_alias(model_name)

    dense_model = load_trained_model(config, model_name, torch.device("cpu"), checkpoint_name=f"{alias}_ham10000")
    dense_metrics = _cpu_eval(dense_model, val_loader)
    dense_latency = measure_latency(
        dense_model,
        device="cpu",
        warmup=int(config["evaluation"].get("latency_warmup_runs", 10)),
        timed_runs=int(config["evaluation"].get("latency_timed_runs", 100)),
    )
    quant_only_model = quantize_model_dynamic(dense_model)
    quant_only_metrics = _cpu_eval(quant_only_model, val_loader)
    quant_only_latency = measure_latency(
        quant_only_model,
        device="cpu",
        warmup=int(config["evaluation"].get("latency_warmup_runs", 10)),
        timed_runs=int(config["evaluation"].get("latency_timed_runs", 100)),
    )

    rows: list[dict[str, object]] = [
        {
            "model_config": "dense",
            "criterion": "none",
            "balanced_accuracy": dense_metrics["balanced_accuracy"],
            "mel_sensitivity": dense_metrics["per_class_sensitivity"]["mel"],
            "bcc_sensitivity": dense_metrics["per_class_sensitivity"]["bcc"],
            "akiec_sensitivity": dense_metrics["per_class_sensitivity"]["akiec"],
            "nv_sensitivity": dense_metrics["per_class_sensitivity"]["nv"],
            "bkl_sensitivity": dense_metrics["per_class_sensitivity"]["bkl"],
            "df_sensitivity": dense_metrics["per_class_sensitivity"]["df"],
            "vasc_sensitivity": dense_metrics["per_class_sensitivity"]["vasc"],
            "size_kb": get_model_size_kb(dense_model)["size_kb"],
            "latency_ms": dense_latency["median_ms"],
        },
        {
            "model_config": "quantized_only",
            "criterion": "none",
            "balanced_accuracy": quant_only_metrics["balanced_accuracy"],
            "mel_sensitivity": quant_only_metrics["per_class_sensitivity"]["mel"],
            "bcc_sensitivity": quant_only_metrics["per_class_sensitivity"]["bcc"],
            "akiec_sensitivity": quant_only_metrics["per_class_sensitivity"]["akiec"],
            "nv_sensitivity": quant_only_metrics["per_class_sensitivity"]["nv"],
            "bkl_sensitivity": quant_only_metrics["per_class_sensitivity"]["bkl"],
            "df_sensitivity": quant_only_metrics["per_class_sensitivity"]["df"],
            "vasc_sensitivity": quant_only_metrics["per_class_sensitivity"]["vasc"],
            "size_kb": get_quantized_model_size(quant_only_model),
            "latency_ms": quant_only_latency["median_ms"],
        },
    ]

    for criterion_name in ("wanda", "magnitude"):
        pruned_model = load_trained_model(config, model_name, torch.device("cpu"), checkpoint_name=f"{alias}_ham10000")
        mask_path = masks_dir / f"{alias}_{criterion_name}_s0.5.pt"
        masks = torch.load(mask_path, map_location="cpu")
        apply_masks(pruned_model, masks)
        pruned_metrics = _cpu_eval(pruned_model, val_loader)
        pruned_latency = measure_latency(
            pruned_model,
            device="cpu",
            warmup=int(config["evaluation"].get("latency_warmup_runs", 10)),
            timed_runs=int(config["evaluation"].get("latency_timed_runs", 100)),
        )

        quantized_pruned = quantize_model_dynamic(pruned_model)
        quantized_metrics = _cpu_eval(quantized_pruned, val_loader)
        quantized_latency = measure_latency(
            quantized_pruned,
            device="cpu",
            warmup=int(config["evaluation"].get("latency_warmup_runs", 10)),
            timed_runs=int(config["evaluation"].get("latency_timed_runs", 100)),
        )

        rows.extend(
            [
                {
                    "model_config": "pruned_only",
                    "criterion": criterion_name,
                    "balanced_accuracy": pruned_metrics["balanced_accuracy"],
                    "mel_sensitivity": pruned_metrics["per_class_sensitivity"]["mel"],
                    "bcc_sensitivity": pruned_metrics["per_class_sensitivity"]["bcc"],
                    "akiec_sensitivity": pruned_metrics["per_class_sensitivity"]["akiec"],
                    "nv_sensitivity": pruned_metrics["per_class_sensitivity"]["nv"],
                    "bkl_sensitivity": pruned_metrics["per_class_sensitivity"]["bkl"],
                    "df_sensitivity": pruned_metrics["per_class_sensitivity"]["df"],
                    "vasc_sensitivity": pruned_metrics["per_class_sensitivity"]["vasc"],
                    "size_kb": get_model_size_kb(pruned_model, sparse=True)["size_kb"],
                    "latency_ms": pruned_latency["median_ms"],
                },
                {
                    "model_config": "pruned_plus_quantized",
                    "criterion": criterion_name,
                    "balanced_accuracy": quantized_metrics["balanced_accuracy"],
                    "mel_sensitivity": quantized_metrics["per_class_sensitivity"]["mel"],
                    "bcc_sensitivity": quantized_metrics["per_class_sensitivity"]["bcc"],
                    "akiec_sensitivity": quantized_metrics["per_class_sensitivity"]["akiec"],
                    "nv_sensitivity": quantized_metrics["per_class_sensitivity"]["nv"],
                    "bkl_sensitivity": quantized_metrics["per_class_sensitivity"]["bkl"],
                    "df_sensitivity": quantized_metrics["per_class_sensitivity"]["df"],
                    "vasc_sensitivity": quantized_metrics["per_class_sensitivity"]["vasc"],
                    "size_kb": get_quantized_model_size(quantized_pruned),
                    "latency_ms": quantized_latency["median_ms"],
                },
            ]
        )

    for row in rows:
        append_csv_row(log_path, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantization stacking experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
