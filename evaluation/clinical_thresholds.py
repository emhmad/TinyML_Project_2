"""
Clinical-threshold gating (W16 code artefact).

The paper's own recovered-best melanoma sensitivity is ~0.82, i.e. it
still misses about one in five melanomas. "Safe deployment path" is
too strong for that number. This module expresses the clinical bar as
named thresholds so the paper can honestly report which compressed
configurations meet which deployment regime, and which are purely
academic.

The thresholds below are conservative rough anchors from published
dermatology-AI literature and dermatologist performance studies; they
are not a regulatory standard. The paper should cite the source it
relies on when quoting specific numbers — the point of this module is
to make the classification reproducible from a CSV, not to adjudicate
the literature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ClinicalThreshold:
    name: str
    description: str
    min_melanoma_sensitivity: float
    min_melanoma_specificity: float | None = None
    min_balanced_accuracy: float | None = None


DEFAULT_THRESHOLDS: tuple[ClinicalThreshold, ...] = (
    ClinicalThreshold(
        name="triage_screen",
        description=(
            "Primary-care triage screener. Low specificity is tolerable because "
            "every positive goes to a dermatologist for confirmation, but "
            "missing melanomas at screening is not acceptable."
        ),
        min_melanoma_sensitivity=0.90,
        min_melanoma_specificity=0.50,
    ),
    ClinicalThreshold(
        name="specialist_referral",
        description=(
            "Specialist-referral assistant. Sensitivity floor matches the "
            "lower end of published dermatologist cohorts (~0.85)."
        ),
        min_melanoma_sensitivity=0.85,
        min_melanoma_specificity=0.75,
    ),
    ClinicalThreshold(
        name="primary_diagnosis",
        description=(
            "Primary diagnostic aid. Sensitivity and specificity both need to "
            "be at the upper end of expert dermatologist performance."
        ),
        min_melanoma_sensitivity=0.92,
        min_melanoma_specificity=0.85,
        min_balanced_accuracy=0.80,
    ),
    ClinicalThreshold(
        name="academic_only",
        description=(
            "Sensitivity is below the level we would recommend for any "
            "unsupervised clinical touchpoint; report as research only."
        ),
        min_melanoma_sensitivity=0.0,
    ),
)


def classify_row(
    row: dict,
    thresholds: Iterable[ClinicalThreshold] = DEFAULT_THRESHOLDS,
    mel_sensitivity_column: str = "mel_sensitivity",
    mel_specificity_column: str = "mel_specificity_at_90_sens",
    balanced_accuracy_column: str = "balanced_acc",
) -> str:
    """
    Return the strongest deployment regime a given result row meets.
    The thresholds are evaluated in declaration order; the highest one
    that passes wins.

    Missing columns are treated as "constraint auto-passes" — so you
    can call this on an evaluation CSV that only has sensitivity and
    still get a useful triage classification.
    """
    sens = float(row.get(mel_sensitivity_column, float("nan")))
    spec = row.get(mel_specificity_column)
    spec_value = float(spec) if spec is not None and pd.notna(spec) else None
    bal = row.get(balanced_accuracy_column)
    bal_value = float(bal) if bal is not None and pd.notna(bal) else None

    winner = "academic_only"
    winner_rank = -1
    for rank, threshold in enumerate(thresholds):
        if sens < threshold.min_melanoma_sensitivity:
            continue
        if (
            threshold.min_melanoma_specificity is not None
            and spec_value is not None
            and spec_value < threshold.min_melanoma_specificity
        ):
            continue
        if (
            threshold.min_balanced_accuracy is not None
            and bal_value is not None
            and bal_value < threshold.min_balanced_accuracy
        ):
            continue
        if rank > winner_rank:
            winner = threshold.name
            winner_rank = rank
    return winner


def annotate_frame(
    frame: pd.DataFrame,
    thresholds: Iterable[ClinicalThreshold] = DEFAULT_THRESHOLDS,
    **column_kwargs,
) -> pd.DataFrame:
    """
    Add a `clinical_regime` column to any results DataFrame (one call
    per CSV). Returns a copy; does not mutate the input.
    """
    thresholds = tuple(thresholds)
    annotated = frame.copy()
    annotated["clinical_regime"] = annotated.to_dict(orient="records")
    annotated["clinical_regime"] = annotated["clinical_regime"].map(
        lambda row: classify_row(row, thresholds=thresholds, **column_kwargs)
    )
    return annotated


def threshold_table(thresholds: Iterable[ClinicalThreshold] = DEFAULT_THRESHOLDS) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": t.name,
                "description": t.description,
                "min_melanoma_sensitivity": t.min_melanoma_sensitivity,
                "min_melanoma_specificity": t.min_melanoma_specificity,
                "min_balanced_accuracy": t.min_balanced_accuracy,
            }
            for t in thresholds
        ]
    )
