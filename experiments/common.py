from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import HAM10000Dataset, compute_class_weights, get_train_val_splits, get_transforms
from models.load_models import get_linear_layer_names, load_deit_model
from pruning.hooks import ActivationCollector
from utils.config import resolve_seed, should_pin_memory
from utils.distributed import build_distributed_sampler, is_distributed, is_main_process, world_size
from utils.io import load_checkpoint_state
from utils.seed import worker_init_fn


def model_alias(model_name: str) -> str:
    return model_name.replace("_patch16_224", "")


def metadata_csv_path(config: dict[str, Any]) -> str:
    dataset_cfg = config["dataset"]
    return dataset_cfg.get("metadata_csv") or str(Path(dataset_cfg["root"]) / "processed_metadata.csv")


def build_splits(config: dict[str, Any]) -> tuple[list[int], list[int]]:
    dataset_cfg = config["dataset"]
    seed = resolve_seed(config)
    group_by_lesion = bool(dataset_cfg.get("group_by_lesion", True))
    return get_train_val_splits(
        metadata_csv_path(config),
        train_ratio=float(dataset_cfg.get("train_split", 0.8)),
        seed=seed,
        group_by_lesion=group_by_lesion,
    )


def build_dataloaders(
    config: dict[str, Any],
    *,
    include_train: bool = True,
    calibration_size: int | None = None,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    shuffle_train: bool = True,
) -> tuple[DataLoader | None, DataLoader, DataLoader | None, torch.Tensor]:
    """
    Construct train/val/calibration DataLoaders, DDP-aware when the
    process group is initialised. Returns the same tuple shape as before
    so existing callers keep working.
    """
    dataset_cfg = config["dataset"]
    augmentation_cfg = config["augmentation"]
    pin_memory = should_pin_memory()
    metadata_csv = metadata_csv_path(config)
    train_indices, val_indices = build_splits(config)
    seed = resolve_seed(config)

    train_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        transform=get_transforms(
            split="train",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(augmentation_cfg.get("resize_size", 256)),
            augmentation_cfg=augmentation_cfg,
        ),
        indices=train_indices,
    )
    val_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        transform=get_transforms(
            split="val",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(augmentation_cfg.get("resize_size", 256)),
            augmentation_cfg=augmentation_cfg,
        ),
        indices=val_indices,
    )
    calibration_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        transform=get_transforms(
            split="val",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(augmentation_cfg.get("resize_size", 256)),
            augmentation_cfg=augmentation_cfg,
        ),
        indices=train_indices,
    )

    num_workers = int(dataset_cfg.get("num_workers", 4))
    train_loader: DataLoader | None = None
    if include_train:
        train_batch = int(train_batch_size or config["finetune"].get("batch_size", 64))
        train_sampler = build_distributed_sampler(train_dataset, shuffle=shuffle_train, seed=seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch,
            shuffle=(shuffle_train and train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=num_workers > 0,
        )

    # Evaluation stays in a single process — we aggregate predictions on rank 0.
    # Running evaluation under a DistributedSampler risks dropping/duplicating
    # samples and changes the metric definition between single- and multi-GPU
    # runs, so we keep it deterministic instead.
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(eval_batch_size or config["evaluation"].get("batch_size", 128)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )

    calibration_loader: DataLoader | None = None
    if calibration_size is not None:
        subset = Subset(calibration_dataset, list(range(min(calibration_size, len(calibration_dataset)))))
        calibration_loader = DataLoader(
            subset,
            batch_size=min(32, calibration_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )

    class_weights = compute_class_weights(metadata_csv, train_indices)
    return train_loader, val_loader, calibration_loader, class_weights


def _shared_checkpoint_dir(config: dict[str, Any]) -> Path | None:
    """
    When experiment.share_finetune_across_seeds is true, the fine-tune
    was run only under the first seed's directory. Later seeds resolve
    their checkpoint lookups against that shared location.
    """
    experiment_cfg = config.get("experiment", {}) or {}
    if not bool(experiment_cfg.get("share_finetune_across_seeds", False)):
        return None
    seeds = experiment_cfg.get("seeds") or [experiment_cfg.get("seed", 42)]
    if not seeds:
        return None
    # logging.checkpoints_dir has already had `/seed_<current_seed>` appended
    # by apply_seed_to_paths; swap that for the first seed's directory.
    current = Path(config["logging"]["checkpoints_dir"])
    if current.name.startswith("seed_"):
        return current.parent / f"seed_{int(seeds[0])}"
    return current


def load_trained_model(
    config: dict[str, Any],
    model_name: str,
    device: torch.device,
    *,
    checkpoint_name: str | None = None,
    pretrained: bool = False,
) -> torch.nn.Module:
    model = load_deit_model(
        model_name=model_name,
        num_classes=int(config["models"].get("num_classes", 7)),
        pretrained=pretrained,
    ).to(device)
    alias = checkpoint_name or model_alias(model_name)
    primary = Path(config["logging"]["checkpoints_dir"]) / f"{alias}.pth"
    shared_dir = _shared_checkpoint_dir(config)
    candidate = shared_dir / f"{alias}.pth" if shared_dir else None
    source = primary if primary.exists() else candidate
    if source is None or not source.exists():
        raise FileNotFoundError(
            f"Checkpoint {alias}.pth not found at {primary}"
            + (f" or {candidate}" if candidate else "")
        )
    state_dict = load_checkpoint_state(source, map_location=device)
    model.load_state_dict(state_dict)
    return model


def collect_activation_norms(
    model: torch.nn.Module,
    calibration_loader: DataLoader,
    exclude_layers: list[str],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, torch.nn.Module]]]:
    target_layers = get_linear_layer_names(model, exclude_keywords=exclude_layers)
    collector = ActivationCollector(model, target_layers)
    collector.register_hooks()
    model.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            _ = model(images.to(device, non_blocking=True))
    norms = collector.get_activation_norms()
    collector.remove_hooks()
    return norms, target_layers


def resolve_checkpoint_path(config: dict[str, Any], filename: str) -> Path:
    """
    Returns the first existing path for `filename` — checking the
    current seed's checkpoints_dir first, then (if share_finetune is
    enabled) the canonical first-seed dir. Use this anywhere code
    currently hand-joins `Path(config["logging"]["checkpoints_dir"]) / filename`.
    """
    primary = Path(config["logging"]["checkpoints_dir"]) / filename
    if primary.exists():
        return primary
    shared = _shared_checkpoint_dir(config)
    if shared is not None:
        candidate = shared / filename
        if candidate.exists():
            return candidate
    return primary  # nonexistent, let caller raise where it matters


def resolve_calibration_path(config: dict[str, Any], filename: str) -> Path:
    return resolve_checkpoint_path(config, str(Path("calibration") / filename))


def is_writer_process() -> bool:
    """
    Alias for `is_main_process()` — clearer in contexts where callers are
    only using it to gate CSV / checkpoint writes.
    """
    return is_main_process()


def has_multi_gpu() -> bool:
    return is_distributed() and world_size() > 1
