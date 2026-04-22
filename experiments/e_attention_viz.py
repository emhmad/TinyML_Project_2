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
from experiments.common import build_splits, load_trained_model, metadata_csv_path
from pruning.masking import apply_masks
from utils.config import get_device, load_config
from utils.io import ensure_dir

MEL_INDEX = CLASS_NAMES.index("mel")
CONDITIONS = [
    ("dense", "Dense", None),
    ("magnitude", "Magnitude 50%", "deit_small_magnitude_s0.5.pt"),
    ("wanda", "Wanda 50%", "deit_small_wanda_s0.5.pt"),
    ("taylor", "Taylor 50%", "deit_small_taylor_s0.5.pt"),
]


@dataclass
class SampleRecord:
    val_position: int
    image_id: str
    image_path: Path
    melanoma_confidence: float


class AttentionRollout:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.attentions: list[torch.Tensor] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._orig_fused_flags: dict[int, bool] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for block in self.model.blocks:
            attn = block.attn
            if hasattr(attn, "fused_attn"):
                self._orig_fused_flags[id(attn)] = bool(attn.fused_attn)
                attn.fused_attn = False

            if hasattr(attn, "attn_drop"):
                self.handles.append(attn.attn_drop.register_forward_hook(self._hook_fn))
            else:
                raise RuntimeError("Attention module does not expose attn_drop; rollout hook cannot be attached.")

    def _hook_fn(self, module, inputs, output) -> None:
        self.attentions.append(output.detach().cpu())

    def clear(self) -> None:
        self.attentions.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for block in self.model.blocks:
            attn = block.attn
            if hasattr(attn, "fused_attn") and id(attn) in self._orig_fused_flags:
                attn.fused_attn = self._orig_fused_flags[id(attn)]

    def get_rollout(self, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
        self.clear()
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image_tensor.to(device, non_blocking=True))

        if not self.attentions:
            raise RuntimeError("No attention tensors were collected during rollout.")

        joint_attention = None
        for attention in self.attentions:
            attn = attention.mean(dim=1)[0]
            identity = torch.eye(attn.size(-1), dtype=attn.dtype)
            attn = 0.5 * attn + 0.5 * identity
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            joint_attention = attn if joint_attention is None else attn @ joint_attention

        prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        spatial_tokens = joint_attention[0, prefix_tokens:]
        grid_h, grid_w = self.model.patch_embed.grid_size
        rollout = spatial_tokens.reshape(grid_h, grid_w)
        rollout = F.interpolate(
            rollout.unsqueeze(0).unsqueeze(0),
            size=(int(image_tensor.shape[-2]), int(image_tensor.shape[-1])),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        rollout = rollout - rollout.min()
        rollout = rollout / rollout.max().clamp_min(1e-8)
        return rollout.numpy()


def _resolve_image_path(raw_path: str, metadata_csv: Path, image_root: Path) -> Path:
    image_path = Path(raw_path)
    preferred = image_root / image_path.name
    if preferred.exists():
        return preferred
    if image_path.is_absolute():
        if image_path.exists():
            return image_path
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


def _select_samples(config: dict, device: torch.device) -> tuple[list[SampleRecord], torch.Tensor]:
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

    melanoma_candidates: list[SampleRecord] = []
    for position, row in val_frame[val_frame["label_idx"] == MEL_INDEX].iterrows():
        image_path = _resolve_image_path(str(row["image_path"]), metadata_csv, dataset_root)
        image = Image.open(image_path).convert("RGB")
        tensor = val_transform(image)
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0).to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        pred_idx = int(probs.argmax().item())
        if pred_idx != MEL_INDEX:
            continue
        melanoma_candidates.append(
            SampleRecord(
                val_position=int(position),
                image_id=image_path.stem,
                image_path=image_path,
                melanoma_confidence=float(probs[MEL_INDEX].item()),
            )
        )

    if len(melanoma_candidates) < 3:
        raise RuntimeError("Could not find at least 3 correctly classified melanoma images for attention visualization.")

    melanoma_candidates.sort(key=lambda sample: sample.melanoma_confidence, reverse=True)
    shortlist = melanoma_candidates[: min(20, len(melanoma_candidates))]
    pick_positions = np.linspace(0, len(shortlist) - 1, 3, dtype=int)
    selected = [shortlist[index] for index in pick_positions]
    return selected, val_transform


def _load_condition_model(config: dict, device: torch.device, criterion_name: str) -> torch.nn.Module:
    model_name = config["models"]["teacher"]
    alias = model_name.replace("_patch16_224", "")
    model = load_trained_model(config, model_name, device, checkpoint_name=f"{alias}_ham10000")
    if criterion_name != "dense":
        mask_path = Path(config["logging"]["checkpoints_dir"]) / "masks" / f"{alias}_{criterion_name}_s0.5.pt"
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


def run(config_path: str) -> None:
    config = load_config(config_path)
    device = get_device()
    figures_dir = ensure_dir(Path(config["logging"]["figures_dir"]))
    attention_dir = ensure_dir(figures_dir / "attention")
    logs_dir = ensure_dir(Path(config["logging"]["results_dir"]))

    samples, val_transform = _select_samples(config, device)

    selection_rows = [
        {
            "image_id": sample.image_id,
            "val_position": sample.val_position,
            "image_path": str(sample.image_path),
            "dense_melanoma_confidence": sample.melanoma_confidence,
        }
        for sample in samples
    ]
    pd.DataFrame(selection_rows).to_csv(logs_dir / "attention_viz_samples.csv", index=False)

    models: dict[str, tuple[str, torch.nn.Module, AttentionRollout]] = {}
    try:
        for criterion_name, label, _ in CONDITIONS:
            model = _load_condition_model(config, device, criterion_name)
            rollout = AttentionRollout(model)
            models[criterion_name] = (label, model, rollout)

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for row_index, sample in enumerate(samples):
            raw_image = Image.open(sample.image_path).convert("RGB")
            tensor = val_transform(raw_image)
            image_np = _denormalize_image(tensor, config)

            for col_index, (criterion_name, label, _) in enumerate(CONDITIONS):
                _, model, rollout = models[criterion_name]
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
    finally:
        for _, model, rollout in models.values():
            rollout.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DeiT attention rollout visualizations.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
