"""
Build a HAM10000-compatible processed metadata CSV from raw ISIC 2019.

Expected raw layout (anywhere under --source-dir):
  ISIC_2019_Training_GroundTruth.csv      - one-hot per-class labels
  ISIC_2019_Training_Metadata.csv         - patient/lesion/anatom_site
  ISIC_2019_Training_Input/*.jpg          - 25,331 JPEG images
  (or ISIC_2019_Training_Input.zip extracted in-place)

Produces:
  --output-dir/processed_metadata.csv with columns
    image_id, image_path, label, label_idx, lesion_id, anatom_site_general,
    age, sex, dataset_source

Class remap: SCC and UNK rows are dropped (no analogue in HAM10000's
7-class space). Remaining 7 classes map to HAM10000 indices via
data.isic2019.ISIC2019_TO_HAM_INDEX.

Usage:
  python -m scripts.build_isic2019_metadata \
      --source-dir data/isic2019 \
      --output-dir data/isic2019
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset import LABEL_MAP
from data.isic2019 import DROPPED_CLASSES, ISIC2019_TO_HAM_INDEX
from utils.io import ensure_dir


def _find_csv(source_dir: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        for path in source_dir.rglob(candidate):
            return path
    raise FileNotFoundError(
        f"Could not find any of {candidates} under {source_dir}. "
        "Download ISIC 2019 from the ISIC archive (Training set) and extract first."
    )


def _discover_image_files(source_dir: Path) -> dict[str, Path]:
    image_files: dict[str, Path] = {}
    for image_path in source_dir.rglob("*.jpg"):
        image_files[image_path.stem] = image_path.resolve()
    for image_path in source_dir.rglob("*.JPG"):
        image_files[image_path.stem] = image_path.resolve()
    return image_files


def build(source_dir: Path, output_dir: Path) -> Path:
    output_dir = ensure_dir(output_dir)

    gt_path = _find_csv(source_dir, ["ISIC_2019_Training_GroundTruth.csv"])
    meta_path: Path | None = None
    try:
        meta_path = _find_csv(source_dir, ["ISIC_2019_Training_Metadata.csv"])
    except FileNotFoundError:
        # Patient/lesion metadata is optional; we'll fall back to image-level grouping.
        meta_path = None

    image_lookup = _discover_image_files(source_dir)
    if not image_lookup:
        raise FileNotFoundError(
            f"No JPG files found under {source_dir}. Extract ISIC_2019_Training_Input.zip first."
        )

    gt = pd.read_csv(gt_path)
    if "image" not in gt.columns:
        raise ValueError(
            f"{gt_path} missing 'image' column — got {list(gt.columns)}. "
            "Did the ISIC archive change format?"
        )
    class_columns = [c for c in gt.columns if c != "image"]

    # Find the dominant (one-hot) class per row.
    label_strs = gt[class_columns].idxmax(axis=1).str.upper()
    gt["label"] = label_strs.values
    gt["image_id"] = gt["image"].astype(str)

    before = len(gt)
    gt = gt[~gt["label"].isin(DROPPED_CLASSES)].reset_index(drop=True)
    dropped = before - len(gt)
    print(f"[build_isic2019_metadata] dropped {dropped} rows in {sorted(DROPPED_CLASSES)} ({before} -> {len(gt)})")

    gt["label_idx"] = gt["label"].map(ISIC2019_TO_HAM_INDEX)
    if gt["label_idx"].isna().any():
        unmapped = gt[gt["label_idx"].isna()]["label"].unique().tolist()
        raise RuntimeError(f"Unmapped ISIC 2019 classes: {unmapped}")
    gt["label_idx"] = gt["label_idx"].astype(int)

    # Verify HAM10000 label-index space alignment.
    expected = set(LABEL_MAP.values())
    actual = set(gt["label_idx"].unique().tolist())
    if not actual.issubset(expected):
        raise RuntimeError(
            f"Label remap produced indices {actual} outside HAM10000 space {expected}."
        )

    # Resolve image_path; drop rows whose image we don't have on disk.
    gt["image_path"] = gt["image_id"].map(lambda iid: str(image_lookup.get(iid, "")))
    missing = gt[gt["image_path"] == ""]
    if len(missing) > 0:
        print(f"[build_isic2019_metadata] WARNING: {len(missing)} rows have no matching image file; dropping.")
        gt = gt[gt["image_path"] != ""].reset_index(drop=True)

    # Merge patient/lesion metadata when available.
    if meta_path is not None:
        meta = pd.read_csv(meta_path)
        if "image" not in meta.columns:
            raise ValueError(f"{meta_path} missing 'image' column — got {list(meta.columns)}.")
        meta["image_id"] = meta["image"].astype(str)
        keep_cols = ["image_id"]
        for opt in ("lesion_id", "anatom_site_general", "age_approx", "sex"):
            if opt in meta.columns:
                keep_cols.append(opt)
        gt = gt.merge(meta[keep_cols], on="image_id", how="left")
        if "age_approx" in gt.columns and "age" not in gt.columns:
            gt = gt.rename(columns={"age_approx": "age"})
    else:
        # Synthesize grouping fields so downstream callers don't need to special-case.
        gt["lesion_id"] = gt["image_id"]
        gt["anatom_site_general"] = np.nan
        gt["age"] = np.nan
        gt["sex"] = np.nan

    if "lesion_id" not in gt.columns:
        gt["lesion_id"] = gt["image_id"]
    gt["lesion_id"] = gt["lesion_id"].fillna(gt["image_id"])

    gt["dataset_source"] = "isic2019"

    columns = [
        "image_id", "image_path", "label", "label_idx", "lesion_id",
        "anatom_site_general", "age", "sex", "dataset_source",
    ]
    columns = [c for c in columns if c in gt.columns]
    out_path = output_dir / "processed_metadata.csv"
    gt[columns].to_csv(out_path, index=False)

    print(f"[build_isic2019_metadata] wrote {out_path}  rows={len(gt)}  classes={sorted(gt['label'].unique())}")
    print(gt["label"].value_counts().to_string())
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw ISIC 2019 to HAM10000-compatible processed metadata."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
