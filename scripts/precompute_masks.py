"""
Pre-compute and export pruning masks for the release bundle (W14).

Regenerates every mask the pruning matrix produced — magnitude / Wanda /
Taylor / random / Paxton-skewness / X-Pruner / SparseGPT-pseudo — for
every sparsity in the config, and saves them into a release-friendly
directory layout:

    release/
      masks/
        deit_small/
          magnitude/
            s0.5.pt
            s0.7.pt
          wanda/
            ...
        deit_tiny/
          ...
      MANIFEST.json

MANIFEST.json records the config path, the git commit (if available),
each mask's SHA256, and its effective sparsity. Reviewers can
independently verify the mask is correct before loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from experiments.common import model_alias
from experiments.e4_pruning_matrix import _score_layers
from models.load_models import get_linear_layer_names, load_deit_model
from pruning.masking import compute_global_masks
from utils.config import load_config
from utils.io import ensure_dir, load_checkpoint_state, save_masks


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run(config_path: str, output_dir: str | Path) -> None:
    config = load_config(config_path)
    release_dir = ensure_dir(Path(output_dir))
    masks_root = ensure_dir(release_dir / "masks")
    manifest: dict = {
        "config_path": str(Path(config_path).resolve()),
        "git_commit": _git_commit(),
        "models": [],
    }

    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    calibration_dir = checkpoint_dir / "calibration"

    model_names = [config["models"]["student"], config["models"]["teacher"]]
    criteria = config["pruning"].get(
        "release_criteria",
        ["magnitude", "wanda", "taylor", "random", "skewness", "xpruner", "sparsegpt_pseudo"],
    )
    sparsities = list(config["pruning"]["sparsities"])
    device = torch.device("cpu")

    for model_name in model_names:
        alias = model_alias(model_name)
        base = load_deit_model(
            model_name=model_name,
            num_classes=int(config["models"].get("num_classes", 7)),
            pretrained=False,
        ).to(device)
        state_dict = load_checkpoint_state(checkpoint_dir / f"{alias}_ham10000.pth", map_location=device)
        base.load_state_dict(state_dict)
        target_layers = get_linear_layer_names(base, exclude_keywords=config["pruning"]["exclude_layers"])
        activation_norms = torch.load(
            calibration_dir / f"{alias}_activation_norms.pt", map_location="cpu"
        )
        gradients = torch.load(calibration_dir / f"{alias}_gradients.pt", map_location="cpu")

        model_entry: dict = {"model": alias, "criteria": []}
        for criterion in criteria:
            scores = _score_layers(
                target_layers,
                criterion,
                activation_norms=activation_norms,
                gradients=gradients,
                seed=42,
            )
            criterion_dir = ensure_dir(masks_root / alias / criterion)
            sparsity_entries = []
            for sparsity in sparsities:
                mask_path = criterion_dir / f"s{float(sparsity):.2f}.pt"
                masks = compute_global_masks(base, scores, float(sparsity))
                save_masks(mask_path, masks)
                sparsity_entries.append(
                    {
                        "sparsity": float(sparsity),
                        "path": str(mask_path.relative_to(release_dir)),
                        "sha256": _sha256(mask_path),
                    }
                )
            model_entry["criteria"].append({"name": criterion, "sparsities": sparsity_entries})
        manifest["models"].append(model_entry)

    with (release_dir / "MANIFEST.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute pruning masks for the release bundle.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="release")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
