from __future__ import annotations

from pathlib import Path

from plotting import mpl_setup as _mpl_setup

import matplotlib.pyplot as plt

# Print-safe defaults (W15). The previous configuration produced
# cramped PDFs at the reviewer's request, so we:
#   - widen the default canvas so single-column figures still breathe
#   - push fonts up to 12pt base / 14pt axis / 16pt title so they stay
#     legible after conference-pipeline shrink-to-fit
#   - lock a consistent colour map between criteria, sparsities, and
#     the CNN/ViT arms so cross-figure comparison is visual, not
#     label-matching
#   - default to PDF for submission and add a twin PNG saver for slides
STYLE = {
    "figure_size": (9, 5.5),
    "figure_size_wide": (14, 5.5),
    "figure_size_square": (7, 7),
    "dpi": 300,
    "font_family": "serif",
    "font_size": 12,
    "axes_label_size": 13,
    "title_size": 14,
    "suptitle_size": 16,
    "legend_size": 11,
    "tick_label_size": 11,
    "line_width": 2.2,
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
    "skewness": "#9467bd",
    "xpruner": "#17becf",
    "sparsegpt_pseudo": "#bcbd22",
}

CRITERION_MARKERS = {
    "magnitude": "o",
    "wanda": "s",
    "taylor": "^",
    "random": "x",
    "skewness": "D",
    "xpruner": "P",
    "sparsegpt_pseudo": "v",
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
            "savefig.dpi": STYLE["dpi"],
            "font.family": STYLE["font_family"],
            "font.size": STYLE["font_size"],
            "axes.labelsize": STYLE["axes_label_size"],
            "axes.titlesize": STYLE["title_size"],
            "figure.titlesize": STYLE["suptitle_size"],
            "legend.fontsize": STYLE["legend_size"],
            "xtick.labelsize": STYLE["tick_label_size"],
            "ytick.labelsize": STYLE["tick_label_size"],
            "axes.grid": True,
            "grid.alpha": STYLE["grid_alpha"],
            "lines.linewidth": STYLE["line_width"],
            "lines.markersize": STYLE["marker_size"],
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "pdf.fonttype": 42,  # embed fonts so reviewers see them correctly
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, output_path: str | Path, also_png: bool = True) -> None:
    """
    Save a figure with tight layout and an optional companion PNG for
    presentations. W15 asked us to make every figure print-ready and
    reusable outside the paper, so we produce both by default.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if also_png and output_path.suffix.lower() != ".png":
        fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=STYLE["dpi"])
