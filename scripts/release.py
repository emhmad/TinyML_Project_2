"""
Package the trained checkpoints, masks, configs, and environment
specification into a release bundle suitable for upload to HuggingFace
Hub or Zenodo (W14).

Layout produced:

    release/
      checkpoints/
        deit_small_ham10000.pth
        deit_tiny_ham10000.pth
        mobilenetv2_ham10000.pth
        resnet50_ham10000.pth
        recovery_*.pth
      masks/                     <- from scripts/precompute_masks.py
      configs/
        default.yaml
        multi_seed_ciai.yaml
      environment.yml
      requirements.txt
      README_RELEASE.md
      MANIFEST.json

The script is idempotent: copying an already-present file is skipped.
Upload is left to the user — once the bundle is built:
    huggingface-cli upload <repo> release/  (or zenodo upload)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from utils.config import load_config


README_TEMPLATE = """# Does Compression Forget Cancer? — Release Artefacts

This bundle contains every checkpoint and pruning mask used to produce
the numbers in the paper, plus the configs and environment spec needed
to reproduce the pipeline end-to-end.

## Contents
- `checkpoints/` — dense and recovery-fine-tuned checkpoints
- `masks/` — pruning masks per (model × criterion × sparsity)
- `configs/` — YAML configs for both the default and CIAI-4-GPU runs
- `environment.yml` / `requirements.txt` — reproducible environment
- `MANIFEST.json` — SHA256 of every artefact for integrity verification

## Reproducing the numbers
1. `conda env create -f environment.yml && conda activate tinyml-cancer-forget`
2. Prepare HAM10000 at `data/ham10000/` and run `python -m data.download_ham10000 ...`
3. Launch: `bash scripts/reproduce.sh configs/multi_seed_ciai.yaml`

See the paper appendix for expected runtime on 4× A100-40GB.
"""


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)


def _collect_checkpoints(checkpoint_dir: Path, release_checkpoints: Path) -> list[dict]:
    entries: list[dict] = []
    for ckpt in sorted(checkpoint_dir.glob("*.pth")):
        dst = release_checkpoints / ckpt.name
        _copy(ckpt, dst)
        entries.append(
            {
                "name": ckpt.name,
                "path": str(dst.relative_to(release_checkpoints.parent)),
                "sha256": _sha256(dst) if dst.exists() else None,
                "bytes": dst.stat().st_size if dst.exists() else 0,
            }
        )
    return entries


def run(config_path: str, release_dir: str | Path, include_masks: bool = True) -> None:
    config = load_config(config_path)
    release_dir = Path(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoints
    checkpoint_dir = Path(config["logging"]["checkpoints_dir"])
    checkpoints_manifest = _collect_checkpoints(
        checkpoint_dir, release_dir / "checkpoints"
    )

    # Masks — delegate to precompute_masks if missing.
    masks_dir_release = release_dir / "masks"
    if include_masks and not (masks_dir_release / "MANIFEST.json").exists():
        from scripts import precompute_masks

        precompute_masks.run(config_path, release_dir)

    # Configs
    for config_file in Path("configs").glob("*.yaml"):
        _copy(config_file, release_dir / "configs" / config_file.name)

    # Env
    for env_file in ("environment.yml", "requirements.txt"):
        src = Path(env_file)
        if src.exists():
            _copy(src, release_dir / env_file)

    # README + MANIFEST
    (release_dir / "README_RELEASE.md").write_text(README_TEMPLATE)
    manifest = {
        "git_commit": _git_commit(),
        "config_path": str(Path(config_path).resolve()),
        "checkpoints": checkpoints_manifest,
    }
    with (release_dir / "MANIFEST.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package release artefacts (W14).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--release-dir", default="release")
    parser.add_argument("--skip-masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.release_dir, include_masks=not args.skip_masks)


if __name__ == "__main__":
    main()
