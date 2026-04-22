"""
Lightweight DDP helpers so the heavy training scripts (fine-tuning,
recovery fine-tuning, distillation) can run under `torchrun` on 4 GPUs
without special-casing every experiment.

Design contract:
- When launched without torchrun / env vars, `init_distributed()` is a
  no-op and `is_distributed()` returns False. All scripts keep working
  on a single GPU or CPU unchanged.
- When launched under torchrun (RANK, LOCAL_RANK, WORLD_SIZE in env),
  the process group is initialised, the current CUDA device is set,
  and downstream helpers return the right rank/world_size.
- `is_main_process()` is the "only rank 0 should write files" gate.

Typical usage inside an experiment:

    from utils.distributed import (
        init_distributed, is_main_process, barrier, maybe_wrap_ddp,
        get_device, shutdown_distributed, build_distributed_sampler,
    )

    init_distributed()
    device = get_device()
    model = model.to(device)
    model = maybe_wrap_ddp(model, device)
    sampler = build_distributed_sampler(train_dataset, shuffle=True, seed=seed)
    ...
    shutdown_distributed()
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def _env_flag(name: str) -> bool:
    return os.environ.get(name) not in (None, "")


def is_distributed_launch() -> bool:
    return _env_flag("RANK") and _env_flag("WORLD_SIZE") and _env_flag("LOCAL_RANK")


def init_distributed(backend: str | None = None) -> bool:
    """Return True if this process actually joined a DDP group."""
    if not is_distributed_launch():
        return False
    if dist.is_available() and dist.is_initialized():
        return True
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
    return True


def shutdown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def local_rank() -> int:
    if is_distributed():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def is_main_process() -> bool:
    return rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        idx = local_rank() if is_distributed() else 0
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def maybe_wrap_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """
    Wrap `model` in DistributedDataParallel when multiple processes are
    active; otherwise return it unchanged.
    """
    model = model.to(device)
    if is_distributed() and world_size() > 1:
        device_ids = [device.index] if device.type == "cuda" else None
        return DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=find_unused_parameters,
        )
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def build_distributed_sampler(
    dataset: Dataset,
    shuffle: bool,
    seed: int,
) -> DistributedSampler | None:
    if not is_distributed() or world_size() <= 1:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=world_size(),
        rank=rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )


def wrap_loader_for_ddp(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    worker_init_fn=None,
) -> tuple[DataLoader, DistributedSampler | None]:
    """
    Construct a DataLoader that works under either single-process or DDP
    launch. When DDP is active, `shuffle` is handled by
    DistributedSampler and must be False on the DataLoader itself — this
    helper takes care of that.
    """
    sampler = build_distributed_sampler(dataset, shuffle=shuffle, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


@contextmanager
def main_process_only():
    """Context manager: body runs only on rank 0, other ranks wait."""
    try:
        if is_main_process():
            yield True
        else:
            yield False
    finally:
        barrier()
