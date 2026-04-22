"""
Edge-target latency benchmark (W9).

Enumerates every deployed configuration in the results directory —
dense, pruned (per criterion × sparsity), recovered, quantized, and
structured — then benchmarks each one across the runtime targets
listed in `evaluation.edge.targets`. Default targets:

  torch_cpu:   PyTorch, CPU, 4 threads (matches a typical RPi-4 /
               mid-tier phone core count)
  onnx_cpu:    ONNX Runtime, CPUExecutionProvider, 4 threads
  onnx_coreml: ONNX Runtime, CoreMLExecutionProvider (Mac / iOS), skipped
               silently if the provider isn't installed
  tflite_cpu:  TFLite via XNNPACK, 4 threads (requires tflite-runtime on
               the benchmark host; skipped with a clear warning otherwise)

Output: `edge_latency.csv` with columns {seed, model, criterion,
sparsity, recovery_epochs, runtime, provider, mean_ms, median_ms,
p95_ms, p99_ms, std_ms, n, warmup_seconds, num_threads}. One row per
(config, target).

Intentional non-goals: this script does not reproduce accuracy metrics.
Those come from the pruning / recovery CSVs and are joined later by
the aggregator. Keeping accuracy and latency in separate CSVs lets
latency be re-run on a new host without re-scoring accuracy.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
import torch

from evaluation.latency import (
    export_onnx,
    measure_latency,
    measure_latency_onnx,
    measure_latency_tflite,
)
from experiments.common import load_trained_model, model_alias
from models.load_models import load_deit_model
from pruning.masking import apply_masks
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.io import append_csv_row, ensure_dir, load_checkpoint_state
from utils.seed import set_seed


DEFAULT_TARGETS = [
    {"name": "torch_cpu", "runtime": "torch", "device": "cpu", "num_threads": 4},
    {"name": "onnx_cpu", "runtime": "onnx", "providers": ["CPUExecutionProvider"], "num_threads": 4},
    {"name": "onnx_coreml", "runtime": "onnx", "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"], "num_threads": 4},
    {"name": "tflite_cpu", "runtime": "tflite", "num_threads": 4},
]


def _input_shape(config: dict) -> tuple[int, int, int, int]:
    size = int(config["dataset"].get("image_size", 224))
    return (1, 3, size, size)


def _bench_one(
    model: torch.nn.Module,
    target: dict,
    input_shape: tuple[int, int, int, int],
    warmup_seconds: float,
    onnx_cache: Path,
    config_key: str,
    timed_runs: int,
    warmup_runs: int,
) -> dict | None:
    runtime = target.get("runtime", "torch")
    num_threads = target.get("num_threads")

    if runtime == "torch":
        summary = measure_latency(
            model,
            input_shape=input_shape,
            device=target.get("device", "cpu"),
            warmup=warmup_runs,
            timed_runs=timed_runs,
            warmup_seconds=warmup_seconds,
            num_threads=num_threads,
        )
        return {
            "runtime": summary["runtime"],
            "provider": summary["provider"],
            "mean_ms": summary["mean_ms"],
            "median_ms": summary["median_ms"],
            "std_ms": summary["std_ms"],
            "p95_ms": summary["p95_ms"],
            "p99_ms": summary["p99_ms"],
            "n": summary["n"],
        }

    if runtime == "onnx":
        onnx_path = onnx_cache / f"{config_key}.onnx"
        if not onnx_path.exists():
            export_onnx(model, onnx_path, input_shape=input_shape)
        try:
            result = measure_latency_onnx(
                onnx_path,
                input_shape=input_shape,
                providers=target.get("providers"),
                warmup=warmup_runs,
                timed_runs=timed_runs,
                warmup_seconds=warmup_seconds,
                num_threads=num_threads,
            )
        except Exception as exc:  # noqa: BLE001 — skip whole target on any runtime error
            warnings.warn(f"Skipping ONNX target {target.get('name')}: {exc}")
            return None
        return result.as_row()

    if runtime == "tflite":
        tflite_path = onnx_cache.parent / "tflite" / f"{config_key}.tflite"
        if not tflite_path.exists():
            warnings.warn(
                f"Skipping TFLite target: {tflite_path} not found. "
                "Run scripts/export_tflite.py to produce it first."
            )
            return None
        try:
            result = measure_latency_tflite(
                tflite_path,
                input_shape=input_shape,
                warmup=warmup_runs,
                timed_runs=timed_runs,
                warmup_seconds=warmup_seconds,
                num_threads=num_threads,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Skipping TFLite target: {exc}")
            return None
        return result.as_row()

    raise ValueError(f"Unknown runtime: {runtime}")


def _load_dense(model_name: str, config: dict, device: torch.device) -> torch.nn.Module:
    alias = model_alias(model_name)
    return load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")


def _load_recovery(
    model_name: str,
    checkpoint_path: Path,
    config: dict,
    device: torch.device,
) -> torch.nn.Module:
    model = load_deit_model(
        model_name=model_name,
        num_classes=int(config["models"].get("num_classes", 7)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(load_checkpoint_state(checkpoint_path, map_location=device))
    return model


def _iter_pruned_configs(masks_dir: Path, aliases: list[str]) -> list[tuple[str, str, float, Path]]:
    """
    Yield tuples (alias, criterion, sparsity, mask_path) for every
    unstructured mask checkpoint that exists in `masks_dir`.
    """
    results: list[tuple[str, str, float, Path]] = []
    for mask_path in sorted(masks_dir.glob("*.pt")):
        stem = mask_path.stem  # e.g. deit_small_wanda_s0.5
        parts = stem.split("_")
        if len(parts) < 3 or not parts[-1].startswith("s"):
            continue
        try:
            sparsity = float(parts[-1][1:])
        except ValueError:
            continue
        criterion = parts[-2]
        alias = "_".join(parts[:-2])
        if alias not in aliases:
            continue
        results.append((alias, criterion, sparsity, mask_path))
    return results


def run(
    config_path: str,
    seed_override: int | None = None,
    targets: list[dict] | None = None,
) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    set_seed(seed)

    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    results_dir = ensure_dir(config["logging"]["results_dir"])
    masks_dir = checkpoint_dir / "masks"
    log_path = results_dir / "edge_latency.csv"
    onnx_cache = ensure_dir(checkpoint_dir / "onnx_cache")

    edge_cfg = (config.get("evaluation") or {}).get("edge") or {}
    warmup_seconds = float(edge_cfg.get("warmup_seconds", 1.0))
    timed_runs = int(config["evaluation"].get("latency_timed_runs", 100))
    warmup_runs = int(config["evaluation"].get("latency_warmup_runs", 10))
    targets = targets or edge_cfg.get("targets") or DEFAULT_TARGETS

    input_shape = _input_shape(config)

    # Latency is a CPU-bound measurement for meaningful edge targets.
    # Force CPU evaluation to avoid accidentally benchmarking a 3060 Ti
    # here; the CUDA latency column for GPU sanity stays in the main
    # pruning/quantization CSVs.
    device = torch.device("cpu")

    # Dense + recovery models aren't in masks_dir — iterate checkpoints
    # directly and classify by filename.
    model_names = [config["models"]["student"], config["models"]["teacher"]]
    aliases = [model_alias(m) for m in model_names]

    # 1. Dense
    for model_name in model_names:
        alias = model_alias(model_name)
        model = _load_dense(model_name, config, device)
        config_key = f"{alias}_dense_seed{seed}"
        for target in targets:
            row = _bench_one(
                model, target, input_shape, warmup_seconds, onnx_cache,
                config_key, timed_runs, warmup_runs,
            )
            if row is None:
                continue
            append_csv_row(
                log_path,
                {
                    "seed": seed,
                    "model": alias,
                    "criterion": "dense",
                    "sparsity": 0.0,
                    "recovery_epochs": 0,
                    "recovery_lr": 0.0,
                    "target": target.get("name", target.get("runtime")),
                    "warmup_seconds": warmup_seconds,
                    "num_threads": target.get("num_threads"),
                    **row,
                },
            )

    # 2. Pruned (every mask we have on disk)
    pruned_configs = _iter_pruned_configs(masks_dir, aliases)
    for alias, criterion, sparsity, mask_path in pruned_configs:
        model_name = next(m for m in model_names if model_alias(m) == alias)
        model = _load_dense(model_name, config, device)
        masks = torch.load(mask_path, map_location="cpu")
        apply_masks(model, masks)
        config_key = f"{alias}_{criterion}_s{sparsity:.2f}_seed{seed}"
        for target in targets:
            row = _bench_one(
                model, target, input_shape, warmup_seconds, onnx_cache,
                config_key, timed_runs, warmup_runs,
            )
            if row is None:
                continue
            append_csv_row(
                log_path,
                {
                    "seed": seed,
                    "model": alias,
                    "criterion": criterion,
                    "sparsity": float(sparsity),
                    "recovery_epochs": 0,
                    "recovery_lr": 0.0,
                    "target": target.get("name", target.get("runtime")),
                    "warmup_seconds": warmup_seconds,
                    "num_threads": target.get("num_threads"),
                    **row,
                },
            )

    # 3. Recovery checkpoints
    recovery_files = sorted(checkpoint_dir.glob("recovery_*.pth"))
    for checkpoint_path in recovery_files:
        stem = checkpoint_path.stem  # recovery_{alias}_{crit}_s{..}_e{..}_lr{..}
        parts = stem.split("_")
        if len(parts) < 5:
            continue
        alias_candidates = [a for a in aliases if "_".join(parts[1 : 1 + a.count("_") + 1]) == a]
        if not alias_candidates:
            continue
        alias = alias_candidates[0]
        rest = parts[1 + alias.count("_") + 1 :]
        if not rest:
            continue
        criterion = rest[0]
        sparsity = float(rest[1][1:]) if len(rest) > 1 and rest[1].startswith("s") else 0.0
        epochs = int(rest[2][1:]) if len(rest) > 2 and rest[2].startswith("e") else 0
        lr_token = rest[3][2:] if len(rest) > 3 and rest[3].startswith("lr") else "0"
        try:
            recovery_lr = float(lr_token)
        except ValueError:
            recovery_lr = 0.0
        model_name = next(m for m in model_names if model_alias(m) == alias)
        model = _load_recovery(model_name, checkpoint_path, config, device)
        config_key = f"{alias}_recovery_{criterion}_s{sparsity:.2f}_e{epochs}_seed{seed}"
        for target in targets:
            row = _bench_one(
                model, target, input_shape, warmup_seconds, onnx_cache,
                config_key, timed_runs, warmup_runs,
            )
            if row is None:
                continue
            append_csv_row(
                log_path,
                {
                    "seed": seed,
                    "model": alias,
                    "criterion": f"{criterion}+recovery",
                    "sparsity": float(sparsity),
                    "recovery_epochs": int(epochs),
                    "recovery_lr": float(recovery_lr),
                    "target": target.get("name", target.get("runtime")),
                    "warmup_seconds": warmup_seconds,
                    "num_threads": target.get("num_threads"),
                    **row,
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge-target latency benchmark (W9).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, seed_override=args.seed)


if __name__ == "__main__":
    main()
