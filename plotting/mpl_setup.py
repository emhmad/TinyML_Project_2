from __future__ import annotations

import os
from pathlib import Path


def configure_matplotlib_cache() -> None:
    cache_dir = Path(__file__).resolve().parents[1] / "results" / "mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


configure_matplotlib_cache()
