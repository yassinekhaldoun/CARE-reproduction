"""Baselines compared with CARE.

conformal   split conformal prediction: r_hat +/- q, q = the `level` quantile of |r - r_hat| on the
            calibration set. Valid on average, but the width is the same for every patient.
bootstrap   PARAMETRIC stand-in for the paper's voxel resampling: it only needs the stored sums.
            It resamples the two voxel means from a bivariate normal with the plug-in covariance
            and takes percentiles of the ratio. Like the real bootstrap, it captures the random
            noise of the ratio and ignores the network's miscalibration.
"""
import numpy as np

from .stats import pred_ratio, true_ratio
from .intervals import _moments, EPS


def conformal_fit(cal, level=0.68):
    err = np.abs(true_ratio(cal) - pred_ratio(cal))
    return float(np.quantile(err, level, method="higher"))


def conformal_predict(s, q):
    r = pred_ratio(s)
    return np.clip(r - q, 0.0, 1.0), np.clip(r + q, 0.0, 1.0)


def bootstrap_predict(s, level=0.68, rng=None, reps=200):
    rng = rng or np.random.default_rng(0)
    n, mA, mB, vA, vB, cAB = _moments(s)
    P = len(n)
    rho = np.clip(cAB / np.sqrt(np.maximum(vA * vB, EPS)), -1.0, 1.0)
    e1 = rng.standard_normal((P, reps))
    e2 = rng.standard_normal((P, reps))
    dA = np.sqrt(vA / n)[:, None] * e1
    dB = np.sqrt(vB / n)[:, None] * (rho[:, None] * e1 + np.sqrt(1 - rho[:, None] ** 2) * e2)
    num, den = mA[:, None] + dA, mB[:, None] + dB
    ratios = np.where(den > EPS, num / np.where(den > EPS, den, 1.0), np.nan)
    tail = (1 - level) / 2
    lo = np.nanquantile(ratios, tail, axis=1)
    hi = np.nanquantile(ratios, 1 - tail, axis=1)
    return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)
