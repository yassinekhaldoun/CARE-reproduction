"""The CARE interval for a ratio of two predicted volumes.

Two uncertainty sources are added (union bound):
  1. estimation: a ratio of two means is noisy. Delta-method variance + Markov bound,
        beta = sqrt(SE / alpha)           (holds with probability >= 1 - alpha)
  2. calibration: the network is biased. Quantiles q_A, q_B of per-patient |V-Bias| (or ECE)
     on a calibration set shift the numerator and the denominator,
        lower_c = (mean_gA - q_A) / (mean_gB + q_B)
        upper_c = (mean_gA + q_A) / (mean_gB - q_B)      (probability >= 1 - delta)
Final interval: [lower_c - beta, upper_c + beta], probability >= 1 - alpha - delta.

Assumption: region A is inside region B, so the true ratio lies in [0, 1] and the interval is
clipped to [0, 1]. This also keeps widths finite when q_B >= mean_gB (denominator guard).
"""
import numpy as np

from .stats import vbias, ece

EPS = 1e-12


def _moments(s):
    n = s["n"].astype(np.float64)
    mA, mB = s["sgA"] / n, s["sgB"] / n
    vA = (s["sgA2"] - s["sgA"] ** 2 / n) / (n - 1)
    vB = (s["sgB2"] - s["sgB"] ** 2 / n) / (n - 1)
    cAB = (s["sgAgB"] - s["sgA"] * s["sgB"] / n) / (n - 1)
    return n, mA, mB, vA, vB, cAB


def estimation_variance(s):
    """Plug-in delta-method variance of r_hat = mean(g_A) / mean(g_B) (y = g_A, x = g_B)."""
    n, my, mx, vy, vx, cxy = _moments(s)
    var = (vy / mx ** 2 + vx * my ** 2 / mx ** 4 - 2 * cxy * my / mx ** 3) / n
    return np.maximum(var, 0.0)


def estimation_halfwidth(s, alpha):
    return np.sqrt(estimation_variance(s) / alpha)


def calibration_bounds(s, qA, qB):
    """(lower_c, upper_c), unclipped upper may be inf when q_B >= mean_gB."""
    n = s["n"].astype(np.float64)
    mA, mB = s["sgA"] / n, s["sgB"] / n
    lower = np.maximum(mA - qA, 0.0) / (mB + qB)
    denom = mB - qB
    safe = np.where(denom > EPS, denom, 1.0)
    upper = np.where(denom > EPS, (mA + qA) / safe, np.inf)
    return lower, upper


def predict(s, params):
    """Interval (lo, hi) for each patient in `s`, clipped to [0, 1]."""
    lower, upper = calibration_bounds(s, params["qA"], params["qB"])
    beta = estimation_halfwidth(s, params["alpha"])
    lo = np.clip(lower - beta, 0.0, 1.0)
    hi = np.clip(upper + beta, 0.0, 1.0)
    return lo, hi


def parts(s, params):
    """Widths of the two components, for the decomposition figure (before clipping)."""
    lower, upper = calibration_bounds(s, params["qA"], params["qB"])
    beta = estimation_halfwidth(s, params["alpha"])
    upper = np.minimum(upper, 1.0)
    return {"calibration": np.maximum(upper - lower, 0.0), "estimation": 2 * beta}


def _errors(cal, kind):
    if kind == "vbias":
        return np.abs(vbias(cal, "A")), np.abs(vbias(cal, "B"))
    if kind == "ece":
        return ece(cal, "A"), ece(cal, "B")
    raise ValueError(kind)


def fit_care(cal, level=0.68, kind="vbias", n_grid=30):
    """Choose q_A, q_B, alpha, delta on the calibration set.

    The budget alpha + delta = 1 - level is split by grid search, keeping the split with the
    smallest mean interval width on the calibration set (as in the paper).
    """
    errA, errB = _errors(cal, kind)
    budget = 1.0 - level
    best = None
    for alpha in np.linspace(0.01, budget - 0.01, n_grid):
        delta = budget - alpha
        qA = np.quantile(errA, 1 - delta / 2, method="higher")
        qB = np.quantile(errB, 1 - delta / 2, method="higher")
        params = {"alpha": float(alpha), "delta": float(delta), "qA": float(qA), "qB": float(qB),
                  "kind": kind, "level": level}
        lo, hi = predict(cal, params)
        w = float(np.mean(hi - lo))
        if best is None or w < best[0]:
            best = (w, params)
    return best[1]
