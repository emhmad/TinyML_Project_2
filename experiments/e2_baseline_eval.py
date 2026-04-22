from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from data.dataset import HAM10000Dataset, get_transforms
from evaluation.metrics import CLASS_NAMES, evaluate_model
from experiments.common import build_splits, metadata_csv_path
from models.load_models import load_deit_model
from utils.config import apply_seed_to_paths, get_device, load_config, resolve_seed, should_pin_memory
from utils.io import append_csv_row, load_checkpoint_state
from utils.seed import set_seed


def _build_val_loader(config):
    dataset_cfg = config["dataset"]
    pin_memory = should_pin_memory()
    metadata_csv = metadata_csv_path(config)
    _, val_indices = build_splits(config)
    max_val_samples = dataset_cfg.get("max_val_samples")
    if max_val_samples is not None:
        val_indices = val_indices[: int(max_val_samples)]
    val_dataset = HAM10000Dataset(
        metadata_csv=metadata_csv,
        transform=get_transforms(
            split="val",
            image_size=int(dataset_cfg.get("image_size", 224)),
            resize_size=int(config["augmentation"].get("resize_size", 256)),
            augmentation_cfg=config["augmentation"],
        ),
        indices=val_indices,
    )
    return DataLoader(
        val_dataset,
        batch_size=int(config["evaluation"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 4)),
        pin_memory=pin_memory,
    )


def run(config_path: str, model_names: list[str] | None = None, seed_override: int | None = None) -> None:
    config = load_config(config_path)
    if seed_override is not None:
        config = apply_seed_to_paths(config, int(seed_override))
    seed = resolve_seed(config)
    set_seed(seed)
    device = get_device()
    val_loader = _build_val_loader(config)
    model_names = model_names or [config["models"]["student"], config["models"]["teacher"]]
    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    log_path = Path(config["logging"]["results_dir"]) / "baseline_eval.csv"

    for model_name in model_names:
        model = load_deit_model(
            model_name=model_name,
            num_classes=int(config["models"].get("num_classes", 7)),
            pretrained=False,
        ).to(device)
        alias = model_name.replace("_patch16_224", "")
        checkpoint_path = checkpoint_dir / f"{alias}_ham10000.pth"
        model.load_state_dict(load_checkpoint_state(checkpoint_path, map_location=device))
        results = evaluate_model(model, val_loader, device, class_names=CLASS_NAMES)

        row = {
            "seed": seed,
            "model": alias,
            "overall_accuracy": results["overall_accuracy"],
            "balanced_accuracy": results["balanced_accuracy"],
            "macro_auroc": results.get("macro_auroc"),
            "melanoma_auroc": results.get("melanoma_auroc"),
            "ece_top_label": results.get("ece_top_label"),
            "mel_sensitivity": results["per_class_sensitivity"]["mel"],
            "bcc_sensitivity": results["per_class_sensitivity"]["bcc"],
            "akiec_sensitivity": results["per_class_sensitivity"]["akiec"],
            "nv_sensitivity": results["per_class_sensitivity"]["nv"],
            "bkl_sensitivity": results["per_class_sensitivity"]["bkl"],
            "df_sensitivity": results["per_class_sensitivity"]["df"],
            "vasc_sensitivity": results["per_class_sensitivity"]["vasc"],
        }
        append_csv_row(log_path, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense baseline checkpoints.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.models, seed_override=args.seed)


if __name__ == "__main__":
    main()
