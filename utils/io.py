from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from tempfile import mkstemp
from typing import Any

import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_csv_row(csv_path: str | Path, row: dict[str, Any]) -> None:
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    payload = dict(row)
    payload.setdefault("timestamp", timestamp_utc())
    frame = pd.DataFrame([payload])
    frame.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def save_checkpoint(path: str | Path, model: torch.nn.Module, **metadata: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {"state_dict": model.state_dict(), **metadata}
    torch.save(payload, path)


def load_checkpoint_state(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def save_masks(path: str | Path, masks: dict[str, torch.Tensor]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    serializable = {name: mask.detach().cpu() for name, mask in masks.items()}
    torch.save(serializable, path)


def get_serialized_model_size_kb(model: torch.nn.Module) -> float:
    fd, temp_path = mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), temp_path)
        return Path(temp_path).stat().st_size / 1024.0
    finally:
        Path(temp_path).unlink(missing_ok=True)
