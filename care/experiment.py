"""Repeated random calibration/test splits, as used in notebook 2.

With only ~100 held-out patients, a single split gives a noisy coverage. Repeating the split
many times and averaging gives a stable estimate (and a standard deviation).
"""
import numpy as np
import pandas as pd

from .stats import n_cases, subset, true_ratio
from . import intervals, baselines, metrics

CARE_NAMES = {"vbias": "CARE (V-Bias)", "ece": "CARE (ECE)"}


def all_methods(cal, test, level, rng):
    """Fit every method on `cal`, predict on `test`. Returns {name: (lo, hi)}."""
    out = {}
    for kind, name in CARE_NAMES.items():
        out[name] = intervals.predict(test, intervals.fit_care(cal, level, kind))
    out["Conformal"] = baselines.conformal_predict(test, baselines.conformal_fit(cal, level))
    out["Bootstrap"] = baselines.bootstrap_predict(test, level, rng)
    return out


def run_repeated_splits(stats, n_repeats=100, level=0.68, cal_fraction=0.5, seed=0):
    """One row per (repeat, method): coverage, mean width, width-size correlation."""
    rng = np.random.default_rng(seed)
    P = n_cases(stats)
    n_cal = int(round(cal_fraction * P))
    rows = []
    for rep in range(n_repeats):
        perm = rng.permutation(P)
        cal, test = subset(stats, perm[:n_cal]), subset(stats, perm[n_cal:])
        r = true_ratio(test)
        for name, (lo, hi) in all_methods(cal, test, level, rng).items():
            rows.append({"repeat": rep, "method": name,
                         "coverage": metrics.coverage(lo, hi, r),
                         "width": metrics.mean_width(lo, hi),
                         "corr": metrics.width_size_corr(lo, hi, test["syB"])})
    return pd.DataFrame(rows)


def summarize_runs(df):
    """Mean and standard deviation over the repeated splits, per method."""
    g = df.groupby("method")[["coverage", "width", "corr"]]
    out = g.mean().join(g.std(), lsuffix="_mean", rsuffix="_std")
    return out.loc[["Bootstrap", "Conformal", "CARE (V-Bias)", "CARE (ECE)"]]


def coverage_vs_level(stats, levels, n_repeats=30, cal_fraction=0.5, seed=0):
    """Tunability: coverage as a function of the requested confidence level."""
    frames = []
    for lv in levels:
        d = run_repeated_splits(stats, n_repeats, lv, cal_fraction, seed)
        d["level"] = lv
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
