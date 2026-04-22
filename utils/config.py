from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from utils.distributed import get_device as distributed_get_device
from utils.distributed import is_distributed


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Preserve the single-process semantics for experiments that never
    touch DDP, while still returning the correct `cuda:<local_rank>` when
    the caller is running under torchrun. Honours `prefer_cuda=False`
    for CPU-only runs (e.g., quantization latency benchmarks).

    When CUDA is unavailable, fall back to Apple Silicon MPS before CPU
    — matters for local runs on a MacBook where MPS is the real
    accelerator.
    """
    if not prefer_cuda:
        return torch.device("cpu")
    if is_distributed():
        return distributed_get_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def should_pin_memory(device: torch.device | None = None) -> bool:
    if device is None:
        return torch.cuda.is_available()
    return device.type == "cuda"  # MPS and CPU do not benefit from pinned host memory


def apply_seed_to_paths(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """
    Return a deep-copied config whose logging directories are suffixed
    with `/seed_{seed}`. Lets every experiment write its per-seed outputs
    under a clean path so aggregation can pick them up later.
    """
    merged = deepcopy(config)
    logging = merged.setdefault("logging", {})
    for key in ("results_dir", "checkpoints_dir", "figures_dir"):
        if key in logging and logging[key] is not None:
            logging[key] = str(Path(logging[key]) / f"seed_{seed}")
    merged.setdefault("experiment", {})["seed"] = int(seed)
    dataset_cfg = merged.setdefault("dataset", {})
    dataset_cfg["seed"] = int(seed)
    return merged


def resolve_seed(config: dict[str, Any], default: int = 42) -> int:
    """Pick the seed the current run should use (CLI > config > default)."""
    experiment_cfg = config.get("experiment", {}) or {}
    dataset_cfg = config.get("dataset", {}) or {}
    return int(experiment_cfg.get("seed", dataset_cfg.get("seed", default)))
