"""
Attention visualisation + quantitative overlap (W12).

Produces two artefacts:

1. The original three-panel qualitative figure (kept so the paper's
   visual argument still works).
2. `attention_overlap.csv` — per-sample IoU / pointing-game / mass-in-
   mask for every melanoma validation image that has a lesion
   segmentation mask available, across every pruning condition in
   `CONDITIONS`. Plus `attention_overlap_summary.csv` with the per-
   condition aggregates so downstream tables can compare Wanda vs.
   magnitude attention diffusion with real numbers, not eyeballing.

Mask lookup: expects a `dataset.segmentation_mask_dir` path in the
config; each `<image_id>.png` / `<image_id>_segmentation.png` is looked
up by stem against the metadata CSV. Samples without a mask are counted
and reported but skip the quantitative pass.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from data.dataset import CLASS_NAMES, get_transforms
from evaluation.attention_overlap import attention_overlap, summarise_overlap
from experiments.common import build_splits, load_trained_model, metadata_csv_path
from models.attention_rollout import AttentionRollout
from pruning.masking import apply_masks
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed
from utils.io import ensure_dir


MEL_INDEX = CLASS_NAMES.index("mel")
CONDITIONS = [
    ("dense", "Dense", None),
    ("magnitude", "Magnitude 50%", "magnitude_s0.5.pt"),
    ("wanda", "Wanda 50%", "wanda_s0.5.pt"),
    ("taylor", "Taylor 50%", "taylor_s0.5.pt"),
]


@dataclass
class SampleRecord:
    val_position: int
    image_id: str
    image_path: Path
    mask_path: Path | None
    melanoma_confidence: float


def _find_mask(image_id: str, mask_dir: Path | None) -> Path | None:
    if mask_dir is None or not mask_dir.exists():
        return None
    for candidate in (
        mask_dir / f"{image_id}.png",
        mask_dir / f"{image_id}_segmentation.png",
        mask_dir / f"{image_id}_mask.png",
    ):
        if candidate.exists():
            return candidate
    return None


def _resolve_image_path(raw_path: str, metadata_csv: Path, image_root: Path) -> Path:
    image_path = Path(raw_path)
    preferred = image_root / image_path.name
    if preferred.exists():
        return preferred
    if image_path.is_absolute():
        return image_path
    fallback = image_root / image_path
    if fallback.exists():
        return fallback
    return metadata_csv.parent / image_path


def _denormalize_image(tensor: torch.Tensor, config: dict) -> np.ndarray:
    normalize_cfg = config["augmentation"]["normalize"]
    mean = torch.tensor(normalize_cfg["mean"], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(normalize_cfg["std"], dtype=tensor.dtype).view(3, 1, 1)
    image = tensor.detach().cpu() * std + mean
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def _predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> tuple[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device, non_blocking=True))
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    pred_idx = int(probs.argmax().item())
    return CLASS_NAMES[pred_idx], float(probs[pred_idx].item())


def _collect_melanoma_samples(
    config: dict, device: torch.device, mask_dir: Path | None
) -> tuple[list[SampleRecord], object]:
    model_name = config["models"]["teacher"]
    alias = model_name.replace("_patch16_224", "")
    model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
    model.eval()

    metadata_csv = Path(metadata_csv_path(config))
    dataset_root = Path(config["dataset"]["root"])
    frame = pd.read_csv(metadata_csv)
    _, val_indices = build_splits(config)
    val_frame = frame.iloc[val_indices].reset_index(drop=True)
    val_transform = get_transforms(
        split="val",
        image_size=int(config["dataset"].get("image_size", 224)),
        resize_size=int(config["augmentation"].get("resize_size", 256)),
        augmentation_cfg=config["augmentation"],
    )

    samples: list[SampleRecord] = []
    for position, row in val_frame[val_frame["label_idx"] == MEL_INDEX].iterrows():
        image_path = _resolve_image_path(str(row["image_path"]), metadata_csv, dataset_root)
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            continue
        tensor = val_transform(image)
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0).to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        mel_conf = float(probs[MEL_INDEX].item())
        samples.append(
            SampleRecord(
                val_position=int(position),
                image_id=str(row.get("image_id", Path(row["image_path"]).stem)),
                image_path=image_path,
                mask_path=_find_mask(str(row.get("image_id", Path(row["image_path"]).stem)), mask_dir),
                melanoma_confidence=mel_conf,
            )
        )
    return samples, val_transform


def _load_condition_model(
    config: dict,
    device: torch.device,
    criterion_name: str,
    alias: str,
) -> torch.nn.Module:
    model = load_trained_model(config, config["models"]["teacher"], device, checkpoint_name=f"{alias}_ham10000")
    if criterion_name != "dense":
        mask_path = Path(config["logging"]["checkpoints_dir"]) / "masks" / f"{alias}_{criterion_name}_s0.5.pt"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for attention viz condition {criterion_name}: {mask_path}"
            )
        masks = torch.load(mask_path, map_location="cpu")
        apply_masks(model, masks)
    return model


def _overlay_axis(ax, image_np: np.ndarray, rollout: np.ndarray, title: str) -> None:
    ax.imshow(image_np)
    ax.imshow(rollout, cmap="jet", alpha=0.5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _qualitative_figure(
    samples: list[SampleRecord],
    models_by_condition: dict[str, tuple[str, torch.nn.Module, AttentionRollout]],
    val_transform,
    config: dict,
    device: torch.device,
    figures_dir: Path,
    attention_dir: Path,
) -> None:
    """Reproduce the original 3 melanoma × 4 condition figure."""
    picks = samples[: min(3, len(samples))]
    if len(picks) < 3:
        return
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for row_index, sample in enumerate(picks):
        raw_image = Image.open(sample.image_path).convert("RGB")
        tensor = val_transform(raw_image)
        image_np = _denormalize_image(tensor, config)
        for col_index, (criterion_name, label, _) in enumerate(CONDITIONS):
            _, model, rollout = models_by_condition[criterion_name]
            pred_label, pred_conf = _predict(model, tensor, device)
            rollout_map = rollout.get_rollout(tensor.unsqueeze(0), device)
            title = f"{label}\n{pred_label} {pred_conf:.2f}"
            ax = axes[row_index, col_index]
            _overlay_axis(ax, image_np, rollout_map, title)
            if col_index == 0:
                ax.set_ylabel(sample.image_id, rotation=90, fontsize=11)

            panel_fig, panel_ax = plt.subplots(figsize=(4, 4))
            _overlay_axis(panel_ax, image_np, rollout_map, title)
            panel_fig.savefig(
                attention_dir / f"img{row_index + 1}_{criterion_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(panel_fig)
    fig.suptitle("Where Does the Model Look? Attention Maps Under Pruning", fontsize=16)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig8_attention_maps.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _quantitative_overlap(
    samples: list[SampleRecord],
    models_by_condition: dict[str, tuple[str, torch.nn.Module, AttentionRollout]],
    val_transform,
    device: torch.device,
    overlap_path: Path,
    summary_path: Path,
    seed: int,
    threshold: float | str,
) -> None:
    samples_with_masks = [s for s in samples if s.mask_path is not None]
    if not samples_with_masks:
        return

    per_sample_rows: list[dict] = []
    for sample in samples_with_masks:
        raw_image = Image.open(sample.image_path).convert("RGB")
        mask = np.asarray(Image.open(sample.mask_path).convert("L"))
        tensor = val_transform(raw_image)
        for criterion_name, _, _ in CONDITIONS:
            if criterion_name not in models_by_condition:
                continue
            _, model, rollout = models_by_condition[criterion_name]
            rollout_map = rollout.get_rollout(tensor.unsqueeze(0), device)
            result = attention_overlap(rollout_map, mask, threshold=threshold)
            row = {
                "seed": seed,
                "image_id": sample.image_id,
                "val_position": sample.val_position,
                "condition": criterion_name,
                **result.as_row(),
            }
            per_sample_rows.append(row)

    pd.DataFrame(per_sample_rows).to_csv(overlap_path, index=False)

    summary_rows = []
    frame = pd.DataFrame(per_sample_rows)
    for condition, sub in frame.groupby("condition"):
        summary = summarise_overlap(
            [_row_to_overlap(row) for row in sub.to_dict(orient="records")]
        )
        summary_rows.append({"seed": seed, "condition": condition, **summary})
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)


def _row_to_overlap(row: dict):
    from evaluation.attention_overlap import OverlapResult

    return OverlapResult(
        iou=float(row["iou"]),
        pointing_game_hit=bool(int(row["pointing_game_hit"])),
        mass_in_mask=float(row["mass_in_mask"]),
        threshold=float(row["threshold"]),
        attention_area=float(row["attention_area"]),
        mask_area=float(row["mask_area"]),
    )


def run(config_path: str, seed_override: int | None = None) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    device = get_device()

    figures_dir = ensure_dir(Path(config["logging"]["figures_dir"]))
    attention_dir = ensure_dir(figures_dir / "attention")
    logs_dir = ensure_dir(Path(config["logging"]["results_dir"]))

    mask_dir_cfg = config.get("dataset", {}).get("segmentation_mask_dir")
    mask_dir = Path(mask_dir_cfg) if mask_dir_cfg else None
    samples, val_transform = _collect_melanoma_samples(config, device, mask_dir)

    selection_rows = [
        {
            "seed": seed,
            "image_id": sample.image_id,
            "val_position": sample.val_position,
            "image_path": str(sample.image_path),
            "dense_melanoma_confidence": sample.melanoma_confidence,
            "has_mask": int(sample.mask_path is not None),
        }
        for sample in samples
    ]
    pd.DataFrame(selection_rows).to_csv(logs_dir / "attention_viz_samples.csv", index=False)

    alias = config["models"]["teacher"].replace("_patch16_224", "")
    models_by_condition: dict[str, tuple[str, torch.nn.Module, AttentionRollout]] = {}
    try:
        for criterion_name, label, _ in CONDITIONS:
            model = _load_condition_model(config, device, criterion_name, alias)
            rollout = AttentionRollout(model)
            models_by_condition[criterion_name] = (label, model, rollout)

        samples_for_quality = sorted(samples, key=lambda s: s.melanoma_confidence, reverse=True)
        _qualitative_figure(
            samples_for_quality,
            models_by_condition,
            val_transform,
            config,
            device,
            figures_dir,
            attention_dir,
        )

        threshold = config.get("evaluation", {}).get("attention_overlap_threshold", "adaptive")
        _quantitative_overlap(
            samples=samples,
            models_by_condition=models_by_condition,
            val_transform=val_transform,
            device=device,
            overlap_path=logs_dir / "attention_overlap.csv",
            summary_path=logs_dir / "attention_overlap_summary.csv",
            seed=seed,
            threshold=threshold,
        )
    finally:
        for _, _, rollout in models_by_condition.values():
            rollout.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeiT attention rollout + W12 overlap analysis.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, seed_override=args.seed)


if __name__ == "__main__":
    main()
