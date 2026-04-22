from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.dataset import LABEL_MAP
from utils.io import ensure_dir


def discover_image_files(source_dir: str | Path) -> dict[str, Path]:
    source_dir = Path(source_dir)
    image_roots = [
        source_dir,
        source_dir / "HAM10000_images",
        source_dir / "HAM10000_images_part_1",
        source_dir / "HAM10000_images_part_2",
        source_dir / "images",
    ]
    image_files: dict[str, Path] = {}
    for root in image_roots:
        if not root.exists():
            continue
        for image_path in root.glob("*.jpg"):
            image_files[image_path.stem] = image_path.resolve()
    return image_files


def build_processed_metadata(source_dir: str | Path, output_dir: str | Path) -> Path:
    source_dir = Path(source_dir)
    output_dir = ensure_dir(output_dir)
    metadata_candidates = [
        source_dir / "HAM10000_metadata.csv",
        source_dir / "raw" / "HAM10000_metadata.csv",
        source_dir / "HAM10000_metadata.tab",
        source_dir / "raw" / "HAM10000_metadata.tab",
    ]
    metadata_path = next((path for path in metadata_candidates if path.exists()), None)
    if metadata_path is None:
        raise FileNotFoundError(
            "Missing raw metadata in the expected HAM10000 source locations. "
            "Download and extract HAM10000 first, then rerun this script."
        )

    image_lookup = discover_image_files(source_dir)
    if not image_lookup:
        raise FileNotFoundError(
            "No HAM10000 image files were found. Expected extracted JPG files "
            "under HAM10000_images, HAM10000_images_part_1, or HAM10000_images_part_2."
        )

    frame = pd.read_csv(metadata_path)
    frame = frame[frame["dx"].isin(LABEL_MAP)].copy()
    frame["image_path"] = frame["image_id"].map(lambda image_id: str(image_lookup.get(image_id, "")))
    missing = frame["image_path"] == ""
    if missing.any():
        missing_ids = frame.loc[missing, "image_id"].head(5).tolist()
        raise FileNotFoundError(f"Missing image files for sample IDs: {missing_ids}")

    frame["label_idx"] = frame["dx"].map(LABEL_MAP)
    frame["label_name"] = frame["dx"]
    processed = frame[["image_path", "label_idx", "label_name"]].copy()

    output_path = output_dir / "processed_metadata.csv"
    processed.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HAM10000 metadata for training.")
    parser.add_argument("--source-dir", required=True, help="Directory containing raw HAM10000 files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write processed metadata into.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_processed_metadata(args.source_dir, args.output_dir)
    print(f"Processed metadata written to {output_path}")


if __name__ == "__main__":
    main()
