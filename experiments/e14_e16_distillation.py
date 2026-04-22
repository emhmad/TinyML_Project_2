from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.metrics import CLASS_NAMES, evaluate_model
from experiments.common import build_dataloaders, collect_activation_norms, load_trained_model, model_alias
from models.distillation import train_distillation
from models.load_models import load_deit_model
from pruning.masking import apply_masks, compute_global_masks
from pruning.scoring import wanda_score
from utils.config import get_device, load_config
from utils.io import append_csv_row, ensure_dir, save_masks
from utils.seed import set_seed


def _evaluate_and_log(log_path, variant, pruned, metrics):
    append_csv_row(
        log_path,
        {
            "variant": variant,
            "pruned": "yes" if pruned else "no",
            "balanced_accuracy": metrics["balanced_accuracy"],
            "mel_sensitivity": metrics["per_class_sensitivity"]["mel"],
            "bcc_sensitivity": metrics["per_class_sensitivity"]["bcc"],
            "akiec_sensitivity": metrics["per_class_sensitivity"]["akiec"],
            "nv_sensitivity": metrics["per_class_sensitivity"]["nv"],
            "bkl_sensitivity": metrics["per_class_sensitivity"]["bkl"],
            "df_sensitivity": metrics["per_class_sensitivity"]["df"],
            "vasc_sensitivity": metrics["per_class_sensitivity"]["vasc"],
        },
    )


def _prune_with_wanda(config, model, calibration_loader, device):
    activation_norms, target_layers = collect_activation_norms(
        model,
        calibration_loader,
        config["pruning"]["exclude_layers"],
        device,
    )
    scores = {name: wanda_score(layer.weight.detach(), activation_norms[name]) for name, layer in target_layers}
    masks = compute_global_masks(model, scores, sparsity=0.5)
    apply_masks(model, masks)
    return masks


def run(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(int(config["dataset"].get("seed", 42)))
    device = get_device()
    train_loader, val_loader, calibration_loader, class_weights = build_dataloaders(
        config,
        include_train=True,
        calibration_size=int(config["pruning"].get("calibration_size", 128)),
    )

    checkpoint_dir = ensure_dir(config["logging"]["checkpoints_dir"])
    masks_dir = ensure_dir(Path(checkpoint_dir) / "masks" / "kd")
    log_path = Path(config["logging"]["results_dir"]) / "kd_pretreatment.csv"

    teacher_name = config["models"]["teacher"]
    student_name = config["models"]["student"]
    teacher_alias = model_alias(teacher_name)
    student_alias = model_alias(student_name)

    teacher = load_trained_model(config, teacher_name, device, checkpoint_name=f"{teacher_alias}_ham10000")
    direct_student = load_trained_model(config, student_name, device, checkpoint_name=f"{student_alias}_ham10000")
    imagenet_student = load_deit_model(
        student_name,
        num_classes=int(config["models"].get("num_classes", 7)),
        pretrained=bool(config["models"].get("pretrained", True)),
    ).to(device)
    distilled_student = load_deit_model(
        student_name,
        num_classes=int(config["models"].get("num_classes", 7)),
        pretrained=bool(config["models"].get("pretrained", True)),
    ).to(device)

    distilled_checkpoint = Path(checkpoint_dir) / f"{student_alias}_distilled.pth"
    distilled_log = Path(config["logging"]["results_dir"]) / "distillation_history.csv"
    distilled_student, _ = train_distillation(
        teacher=teacher,
        student=distilled_student,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config["distillation"],
        device=device,
        class_weights=class_weights,
        checkpoint_path=distilled_checkpoint,
        log_path=distilled_log,
    )

    variants = {
        "direct": direct_student,
        "distilled": distilled_student,
        "imagenet_only": imagenet_student,
    }

    for variant_name, model in variants.items():
        dense_metrics = evaluate_model(model, val_loader, device, class_names=CLASS_NAMES)
        _evaluate_and_log(log_path, variant_name, False, dense_metrics)

        pruned_model = model
        masks = _prune_with_wanda(config, pruned_model, calibration_loader, device)
        save_masks(masks_dir / f"{variant_name}_wanda_50.pt", masks)
        pruned_metrics = evaluate_model(pruned_model, val_loader, device, class_names=CLASS_NAMES)
        _evaluate_and_log(log_path, variant_name, True, pruned_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distillation pre-treatment experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
