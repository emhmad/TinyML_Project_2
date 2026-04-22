from __future__ import annotations

import torch

from utils.io import get_serialized_model_size_kb


def _count_dense_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total_params = 0
    nonzero_params = 0
    for tensor in model.state_dict().values():
        if not isinstance(tensor, torch.Tensor):
            continue
        total_params += tensor.numel()
        nonzero_params += int(torch.count_nonzero(tensor).item())
    return total_params, nonzero_params


def get_model_size_kb(model: torch.nn.Module, sparse: bool = False) -> dict[str, float | int]:
    total_params, nonzero_params = _count_dense_parameters(model)
    dense_size_kb = (total_params * 4) / 1024.0
    if sparse:
        size_kb = (nonzero_params * 4) / 1024.0
    else:
        size_kb = get_serialized_model_size_kb(model)

    return {
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "size_kb": float(size_kb),
        "dense_size_kb": float(dense_size_kb),
        "compression_ratio": float(dense_size_kb / max(size_kb, 1e-8)),
    }
