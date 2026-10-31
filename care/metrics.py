"""Metrics: validity (coverage), usefulness (width) and adaptiveness (width versus tumour size)."""
import numpy as np
from scipy.stats import spearmanr


def coverage(lo, hi, r):
    """Fraction of patients whose true ratio lies inside [lo, hi]."""
    return float(np.mean((r >= lo) & (r <= hi)))


def mean_width(lo, hi):
    return float(np.mean(hi - lo))


def width_size_corr(lo, hi, size):
    """Spearman correlation between interval width and tumour size (negative = adaptive).

    Returns 0.0 when the width is constant (correlation undefined).
    """
    w = hi - lo
    if np.ptp(w) == 0:
        return 0.0
    return float(spearmanr(w, size).statistic)


def width_by_size_tertile(lo, hi, size):
    """Mean width for small / medium / large tumours (tertiles of `size`)."""
    q1, q2 = np.quantile(size, [1 / 3, 2 / 3])
    w = hi - lo
    return {"small": float(w[size <= q1].mean()),
            "medium": float(w[(size > q1) & (size <= q2)].mean()),
            "large": float(w[size > q2].mean())}
