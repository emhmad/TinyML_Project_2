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
    """
    Report three distinct notions of size so downstream tables can be
    honest (W8). All values are kilobytes.

    * `dense_size_kb`: total_params * 4 bytes. Always reported; matches the
      file size of a dense fp32 checkpoint.
    * `disk_size_kb`: actual on-disk size after torch.save of the model's
      state_dict. This is the "real" file size — unstructured pruning in
      dense format does NOT reduce this.
    * `effective_sparse_size_kb`: nonzero_params * 4 bytes. What a pruned
      model WOULD take if stored in an idealised sparse format. Useful
      for ceiling analysis but misleading when reported as "size" without
      the caveat — so it is exposed under an explicit name.

    `size_kb` (kept for backwards compatibility):
      - when `sparse=True`, equals `effective_sparse_size_kb`
      - when `sparse=False`, equals `disk_size_kb`
    """
    total_params, nonzero_params = _count_dense_parameters(model)
    dense_size_kb = (total_params * 4) / 1024.0
    effective_sparse_size_kb = (nonzero_params * 4) / 1024.0
    disk_size_kb = get_serialized_model_size_kb(model)

    size_kb = effective_sparse_size_kb if sparse else disk_size_kb

    return {
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "size_kb": float(size_kb),
        "dense_size_kb": float(dense_size_kb),
        "disk_size_kb": float(disk_size_kb),
        "effective_sparse_size_kb": float(effective_sparse_size_kb),
        "compression_ratio": float(dense_size_kb / max(effective_sparse_size_kb, 1e-8)),
    }
