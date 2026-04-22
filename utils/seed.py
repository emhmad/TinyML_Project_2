from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed every RNG we touch so the multi-seed comparison is reproducible
    within a seed.

    `deterministic=True` enables the stricter cuDNN + torch deterministic
    path. Comes with a throughput cost, so it is off by default and
    switched on only when the caller cares about bit-identical reruns
    (e.g., the activation-statistics analysis in e_activation_stats.py).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch versions may not support warn_only; fall back silently.
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker seeder derived from the base torch seed. Pass this
    as `worker_init_fn=worker_init_fn` when constructing DataLoaders
    inside seed-controlled experiments.
    """
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
