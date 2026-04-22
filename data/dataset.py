from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_MAP = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Columns that must exist in processed_metadata.csv for lesion-grouped splitting.
REQUIRED_COLUMNS = {"image_path", "label_idx"}
LESION_GROUP_COLUMN = "lesion_id"


class HAM10000Dataset(Dataset):
    """HAM10000 dataset backed by a processed metadata CSV."""

    def __init__(
        self,
        metadata_csv: str | Path,
        image_dir: str | Path | None = None,
        transform: transforms.Compose | None = None,
        indices: list[int] | None = None,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.image_dir = Path(image_dir) if image_dir else None
        self.transform = transform
        self.frame = pd.read_csv(self.metadata_csv)
        missing = REQUIRED_COLUMNS - set(self.frame.columns)
        if missing:
            raise ValueError(f"Metadata CSV missing required columns: {sorted(missing)}")
        if indices is not None:
            self.frame = self.frame.iloc[indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image_path = Path(row["image_path"])
        if image_path.is_absolute():
            if not image_path.exists():
                fallback_root = self.image_dir or self.metadata_csv.parent
                image_path = fallback_root / image_path.name
        elif self.image_dir is not None:
            image_path = self.image_dir / image_path
        else:
            image_path = self.metadata_csv.parent / image_path
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label_idx"])
        return image, label


def _stratified_image_split(
    frame: pd.DataFrame, train_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    labels = frame["label_idx"].to_numpy()
    train_idx, val_idx = next(splitter.split(np.arange(len(frame)), labels))
    return train_idx.tolist(), val_idx.tolist()


def _lesion_grouped_split(
    frame: pd.DataFrame, train_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """
    Patient/lesion-grouped split: every image sharing a lesion_id goes
    entirely to train or to val. This is the W2 fix — prevents the
    per-lesion leakage that inflates every metric when HAM10000 is split
    image-wise (same lesion can contribute duplicate images to both sides).
    """
    groups = frame[LESION_GROUP_COLUMN].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.arange(len(frame)), groups=groups))

    # Sanity-check: no lesion_id appears on both sides.
    train_groups = set(groups[train_idx].tolist())
    val_groups = set(groups[val_idx].tolist())
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(
            f"Lesion-grouped split produced {len(overlap)} overlapping lesion_ids; "
            "this is a bug in the splitter."
        )
    return train_idx.tolist(), val_idx.tolist()


def get_train_val_splits(
    metadata_csv: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    group_by_lesion: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Returns (train_indices, val_indices) into the processed metadata CSV.

    When `group_by_lesion=True` (default, and the correct setting for any
    claim-bearing experiment), uses GroupShuffleSplit on `lesion_id`.
    Falls back to stratified-by-class only if `lesion_id` is unavailable
    and the caller explicitly opts in via `group_by_lesion=False`.
    """
    frame = pd.read_csv(metadata_csv)
    if group_by_lesion:
        if LESION_GROUP_COLUMN not in frame.columns:
            raise ValueError(
                f"Processed metadata is missing '{LESION_GROUP_COLUMN}'. "
                "Re-run data/download_ham10000.py to regenerate it — patient-level "
                "splits require this column and image-level splits leak patient identity."
            )
        return _lesion_grouped_split(frame, train_ratio=train_ratio, seed=seed)
    return _stratified_image_split(frame, train_ratio=train_ratio, seed=seed)


def compute_class_weights(metadata_csv: str | Path, train_indices: list[int]) -> torch.Tensor:
    frame = pd.read_csv(metadata_csv).iloc[train_indices]
    counts = frame["label_idx"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    inverse = 1.0 / counts.clip(lower=1).to_numpy(dtype="float32")
    weights = torch.tensor(inverse, dtype=torch.float32)
    weights = weights / weights.sum() * len(CLASS_NAMES)
    return weights


def get_transforms(
    split: str = "train",
    image_size: int = 224,
    resize_size: int = 256,
    augmentation_cfg: dict[str, Any] | None = None,
) -> transforms.Compose:
    augmentation_cfg = augmentation_cfg or {}
    normalize_cfg = augmentation_cfg.get(
        "normalize",
        {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    )
    normalize = transforms.Normalize(mean=normalize_cfg["mean"], std=normalize_cfg["std"])

    if split == "train":
        jitter_cfg = augmentation_cfg.get("color_jitter", {})
        transform_steps: list[Any] = [transforms.Resize(resize_size), transforms.RandomCrop(image_size)]
        if augmentation_cfg.get("horizontal_flip", True):
            transform_steps.append(transforms.RandomHorizontalFlip())
        if augmentation_cfg.get("vertical_flip", True):
            transform_steps.append(transforms.RandomVerticalFlip())
        transform_steps.extend(
            [
                transforms.ColorJitter(
                    brightness=jitter_cfg.get("brightness", 0.2),
                    contrast=jitter_cfg.get("contrast", 0.2),
                    saturation=jitter_cfg.get("saturation", 0.2),
                    hue=jitter_cfg.get("hue", 0.1),
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return transforms.Compose(transform_steps)

    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
