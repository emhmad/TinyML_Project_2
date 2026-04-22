"""
Paired statistical tests used to make seed-level comparisons honest.

Nothing heavyweight: McNemar's exact test for per-class sensitivity
differences (paired binary outcomes on the same validation set) and a
paired t-test wrapper for balanced-accuracy comparisons across seeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class McNemarResult:
    b: int  # A correct, B wrong
    c: int  # A wrong, B correct
    statistic: float
    p_value: float
    method: str

    def as_dict(self) -> dict:
        return {
            "b": self.b,
            "c": self.c,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "method": self.method,
        }


def mcnemar_test(
    correct_a: Iterable[bool] | np.ndarray,
    correct_b: Iterable[bool] | np.ndarray,
    *,
    exact_threshold: int = 25,
) -> McNemarResult:
    """
    McNemar's paired test on two boolean per-sample correctness vectors.

    Uses the exact binomial tail when `b + c <= exact_threshold`, otherwise
    the continuity-corrected chi-square approximation. Returns a frozen
    dataclass with b, c, test statistic, p-value, and method name.

    Recommended usage here: per-class "correctly predicted this dangerous
    class" booleans computed on the *same* validation set by method A vs.
    method B (e.g., Wanda vs. magnitude), which is the only comparison
    where single-run gaps are interpretable — per-seed comparison is the
    right scope for the paired t-test below.
    """
    a = np.asarray(list(correct_a), dtype=bool)
    b = np.asarray(list(correct_b), dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    disc_b = int(((a) & (~b)).sum())  # A right, B wrong
    disc_c = int(((~a) & (b)).sum())  # A wrong, B right
    n = disc_b + disc_c

    if n == 0:
        return McNemarResult(b=disc_b, c=disc_c, statistic=0.0, p_value=1.0, method="exact-binomial")

    if n <= exact_threshold:
        # Two-sided exact binomial on min(b, c) ~ Binomial(n, 0.5).
        k = min(disc_b, disc_c)
        p_value = float(stats.binom.cdf(k, n, 0.5) * 2)
        p_value = min(1.0, p_value)
        return McNemarResult(b=disc_b, c=disc_c, statistic=float(k), p_value=p_value, method="exact-binomial")

    # Continuity-corrected chi-square approximation.
    chi2 = (abs(disc_b - disc_c) - 1) ** 2 / n
    p_value = float(stats.chi2.sf(chi2, df=1))
    return McNemarResult(b=disc_b, c=disc_c, statistic=float(chi2), p_value=p_value, method="chi2-cc")


@dataclass(frozen=True)
class PairedTTestResult:
    n: int
    mean_diff: float
    std_diff: float
    statistic: float
    p_value: float
    ci95_low: float
    ci95_high: float

    def as_dict(self) -> dict:
        return {
            "n": int(self.n),
            "mean_diff": float(self.mean_diff),
            "std_diff": float(self.std_diff),
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "ci95_low": float(self.ci95_low),
            "ci95_high": float(self.ci95_high),
        }


def paired_t_test(
    values_a: Iterable[float] | np.ndarray,
    values_b: Iterable[float] | np.ndarray,
) -> PairedTTestResult:
    """
    Paired t-test over matched observations (e.g., same seed → two metrics).
    Returns the mean difference, its std, t-statistic, two-sided p-value,
    and 95% CI on the mean difference. If n < 2, returns NaNs with a large
    p-value (so callers can filter without crashing).
    """
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    n = int(diff.size)
    if n < 2:
        return PairedTTestResult(
            n=n,
            mean_diff=float(diff.mean()) if n == 1 else float("nan"),
            std_diff=float("nan"),
            statistic=float("nan"),
            p_value=float("nan"),
            ci95_low=float("nan"),
            ci95_high=float("nan"),
        )
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    sem = std_diff / np.sqrt(n)
    if sem == 0:
        # All differences identical → deterministic result; treat as not significant if zero.
        return PairedTTestResult(
            n=n,
            mean_diff=mean_diff,
            std_diff=std_diff,
            statistic=float("inf") if mean_diff != 0 else 0.0,
            p_value=0.0 if mean_diff != 0 else 1.0,
            ci95_low=mean_diff,
            ci95_high=mean_diff,
        )
    t_stat = mean_diff / sem
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    crit = stats.t.ppf(0.975, df=n - 1)
    half_width = crit * sem
    return PairedTTestResult(
        n=n,
        mean_diff=mean_diff,
        std_diff=std_diff,
        statistic=float(t_stat),
        p_value=p_value,
        ci95_low=float(mean_diff - half_width),
        ci95_high=float(mean_diff + half_width),
    )
