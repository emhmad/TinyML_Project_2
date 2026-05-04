"""
ISIC 2019 dataset loader (L2-full).

ISIC 2019 has 8 classes; HAM10000 has 7. Mapping used here:

    ISIC 2019  ->  HAM10000
    MEL        ->  mel
    NV         ->  nv
    BCC        ->  bcc
    AK         ->  akiec    (AK == actinic keratosis)
    BKL        ->  bkl      (benign keratosis-like, includes solar
                             lentigo, seborrheic keratosis, lichen
                             planus-like keratoses)
    DF         ->  df
    VASC       ->  vasc
    SCC        ->  (dropped — no analogue in HAM10000)
    UNK        ->  (dropped — unknown / noise label)

This drops ~1% of ISIC 2019 (SCC) so the 7-class evaluation harness
keeps working without per-dataset class remaps. Train/val splits are
patient-grouped on `lesion_id` when available, otherwise stratified by
class label.

Usage (mirrors data.dataset.HAM10000Dataset):
    >>> ds = ISIC2019Dataset("data/isic2019/processed_metadata.csv")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

from data.dataset import LABEL_MAP, REQUIRED_COLUMNS, get_transforms

# Mapping from ISIC 2019 raw class strings to HAM10000 label indices.
ISIC2019_TO_HAM_INDEX: dict[str, int] = {
    "MEL": LABEL_MAP["mel"],
    "NV": LABEL_MAP["nv"],
    "BCC": LABEL_MAP["bcc"],
    "AK": LABEL_MAP["akiec"],
    "BKL": LABEL_MAP["bkl"],
    "DF": LABEL_MAP["df"],
    "VASC": LABEL_MAP["vasc"],
    # SCC and UNK are intentionally absent — rows containing them get
    # filtered out at metadata build time.
}

DROPPED_CLASSES: set[str] = {"SCC", "UNK"}


class ISIC2019Dataset(Dataset):
    """
    ISIC 2019 dermoscopy backed by a processed metadata CSV with the
    same columns as HAM10000 (`image_path`, `label_idx`, `lesion_id`,
    `image_id`). All HAM10000 callers (`HAM10000Dataset`,
    `compute_class_weights`, `get_train_val_splits`) work unchanged
    against an ISIC-2019 metadata CSV that has been remapped to the
    same 7-class label space.
    """

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
            raise ValueError(
                f"ISIC 2019 metadata CSV missing required columns: {sorted(missing)}. "
                "Run scripts.build_isic2019_metadata first."
            )
        if indices is not None:
            self.frame = self.frame.iloc[indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
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


def get_isic2019_splits(
    metadata_csv: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    group_by_lesion: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Returns (train_indices, val_indices). When `group_by_lesion=True`
    and the metadata has a `lesion_id` column, uses GroupShuffleSplit;
    otherwise falls back to stratified-by-class. ISIC 2019's 'lesion_id'
    is sometimes absent (image-level annotations only) — the build
    script populates it with the image_id when grouping data is missing,
    which preserves the API but degrades to image-level splits.
    """
    frame = pd.read_csv(metadata_csv)
    if group_by_lesion and "lesion_id" in frame.columns and frame["lesion_id"].notna().all():
        groups = frame["lesion_id"].to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.arange(len(frame)), groups=groups))
    else:
        labels = frame["label_idx"].to_numpy()
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.arange(len(frame)), labels))
    return train_idx.tolist(), val_idx.tolist()


__all__ = [
    "ISIC2019Dataset",
    "ISIC2019_TO_HAM_INDEX",
    "DROPPED_CLASSES",
    "get_isic2019_splits",
    "get_transforms",  # re-exported for convenience
]
