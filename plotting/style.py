from __future__ import annotations

from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt

STYLE = {
    "figure_size": (8, 5),
    "dpi": 300,
    "font_family": "serif",
    "font_size": 12,
    "title_size": 14,
    "legend_size": 10,
    "line_width": 2.0,
    "marker_size": 8,
    "grid_alpha": 0.3,
    "save_format": "pdf",
    "save_format_pres": "png",
}

CRITERION_COLORS = {
    "magnitude": "#1f77b4",
    "wanda": "#d62728",
    "taylor": "#2ca02c",
    "random": "#7f7f7f",
}

CRITERION_MARKERS = {
    "magnitude": "o",
    "wanda": "s",
    "taylor": "^",
    "random": "x",
}

CLASS_COLORS = {
    "mel": "#d62728",
    "bcc": "#ff7f0e",
    "akiec": "#e377c2",
    "bkl": "#2ca02c",
    "nv": "#1f77b4",
    "df": "#9467bd",
    "vasc": "#8c564b",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": STYLE["dpi"],
            "font.family": STYLE["font_family"],
            "font.size": STYLE["font_size"],
            "axes.titlesize": STYLE["title_size"],
            "legend.fontsize": STYLE["legend_size"],
            "axes.grid": True,
            "grid.alpha": STYLE["grid_alpha"],
        }
    )


def save_figure(fig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
