"""
Stage ISIC 2018 Task 1 lesion segmentation masks into the layout that
experiments.e_attention_viz expects (W12 quantitative attention overlap).

Input: a directory containing the ISIC 2018 Task 1 ground-truth masks.
The official archive is named "ISIC2018_Task1_Training_GroundTruth.zip"
and contains files of the form
    ISIC_<image_id>_segmentation.png

Output: the configured `dataset.segmentation_mask_dir` (default
`data/ham10000/segmentation_masks/`) populated with `<image_id>.png`
files keyed by HAM10000 image_id. ISIC 2018 Task 1 and HAM10000 share
image ids for ~10k images — this script joins them.

Usage:
  python -m scripts.build_isic2018_segmasks \
      --source-dir data/isic2018_task1_groundtruth \
      --metadata-csv data/ham10000/processed_metadata.csv \
      --output-dir data/ham10000/segmentation_masks

Notes:
  * Mask values are kept as-is (0 / 255). The downstream IoU pass
    (`evaluation.attention_overlap.attention_overlap`) binarises them.
  * If the ISIC 2018 file uses "_segmentation.png" suffix, this script
    drops the suffix when copying so e_attention_viz._find_mask matches.
  * Missing masks for some HAM10000 images is normal — log them but
    don't fail; the IoU pass simply skips those samples.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from utils.io import ensure_dir


def _normalize_mask_filename(stem: str) -> str:
    """Strip ISIC 2018 suffixes so the destination key is just `<image_id>.png`."""
    for suffix in ("_segmentation", "_mask"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def stage(source_dir: Path, metadata_csv: Path, output_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"ISIC 2018 mask source directory not found: {source_dir}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"HAM10000 metadata CSV not found: {metadata_csv}")
    output_dir = ensure_dir(output_dir)

    frame = pd.read_csv(metadata_csv)
    image_id_col = "image_id" if "image_id" in frame.columns else None
    if image_id_col is None:
        # Fall back to deriving image_id from image_path basename.
        frame["image_id"] = frame["image_path"].apply(lambda p: Path(p).stem)
        image_id_col = "image_id"
    valid_ids = set(frame[image_id_col].astype(str).tolist())

    # Find candidate mask files in the source.
    candidates: dict[str, Path] = {}
    for mask_path in source_dir.rglob("*.png"):
        stem = _normalize_mask_filename(mask_path.stem)
        candidates[stem] = mask_path
    print(
        f"[build_isic2018_segmasks] discovered {len(candidates)} candidate masks "
        f"under {source_dir}; HAM10000 has {len(valid_ids)} image_ids."
    )

    matched = 0
    unmatched_dst = 0
    for image_id in valid_ids:
        src = candidates.get(image_id)
        if src is None:
            unmatched_dst += 1
            continue
        dst = output_dir / f"{image_id}.png"
        if dst.exists() and dst.stat().st_size > 0:
            matched += 1
            continue
        shutil.copyfile(src, dst)
        matched += 1

    print(
        f"[build_isic2018_segmasks] staged {matched} masks into {output_dir} "
        f"({unmatched_dst} HAM10000 image_ids had no matching ISIC 2018 mask)."
    )

    if matched == 0:
        raise RuntimeError(
            "No masks were staged. Verify that --source-dir contains ISIC 2018 Task 1 "
            "ground-truth PNGs and that --metadata-csv points at the HAM10000 processed CSV."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage ISIC 2018 segmentation masks for HAM10000 attention-overlap analysis."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/ham10000/processed_metadata.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ham10000/segmentation_masks"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage(args.source_dir, args.metadata_csv, args.output_dir)


if __name__ == "__main__":
    main()
