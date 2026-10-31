"""Per-patient sufficient statistics.

A "stats" object is a dict of NumPy arrays with one row per patient:

    n                       number of voxels counted
    sgA, sgB                sums of the predicted probabilities g_A, g_B
    sgA2, sgB2, sgAgB       sums of g_A^2, g_B^2, g_A * g_B
    syA, syB                sums of the ground-truth labels y_A, y_B
    binsA, binsB            shape (P, N_BINS, 3): per probability bin, the voxel count,
                            the sum of probabilities and the sum of labels

Everything CARE needs can be computed from these numbers, so a 3D volume is compressed into
about 100 floats. `summarize_case` is used both by the synthetic generator (laptop) and by the
Kaggle extraction script, so the two sides cannot disagree on the format.
"""
import numpy as np

N_BINS = 15
SCALAR_FIELDS = ["n", "sgA", "sgB", "sgA2", "sgB2", "sgAgB", "syA", "syB"]
BIN_FIELDS = ["binsA", "binsB"]


def _bins(g, y):
    idx = np.minimum((g * N_BINS).astype(np.int64), N_BINS - 1)
    count = np.bincount(idx, minlength=N_BINS).astype(np.float64)
    conf = np.bincount(idx, weights=g, minlength=N_BINS)
    label = np.bincount(idx, weights=y, minlength=N_BINS)
    return np.stack([count, conf, label], axis=1)


def summarize_case(gA, gB, yA, yB):
    """Compress the voxel-level predictions and labels of ONE patient into sufficient statistics."""
    gA = np.asarray(gA, dtype=np.float64).ravel()
    gB = np.asarray(gB, dtype=np.float64).ravel()
    yA = np.asarray(yA, dtype=np.float64).ravel()
    yB = np.asarray(yB, dtype=np.float64).ravel()
    return {
        "n": gA.size, "sgA": gA.sum(), "sgB": gB.sum(),
        "sgA2": (gA * gA).sum(), "sgB2": (gB * gB).sum(), "sgAgB": (gA * gB).sum(),
        "syA": yA.sum(), "syB": yB.sum(),
        "binsA": _bins(gA, yA), "binsB": _bins(gB, yB),
    }


def stack_cases(cases):
    """List of per-patient dicts -> one dict of arrays (one row per patient)."""
    return {k: np.stack([np.asarray(c[k]) for c in cases]) for k in cases[0]}


def save_stats(path, stats):
    np.savez_compressed(path, **stats)


def load_stats(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def n_cases(stats):
    return len(stats["n"])


def subset(stats, idx):
    return {k: v[idx] for k, v in stats.items()}


def true_ratio(s):
    return s["syA"] / s["syB"]


def pred_ratio(s):
    return s["sgA"] / s["sgB"]


def vbias(s, region):
    """Signed volume bias per patient: mean over voxels of (g - y)."""
    return (s["sg" + region] - s["sy" + region]) / s["n"]


def ece(s, region):
    """Binned expected calibration error per patient: sum_b |sum_conf_b - sum_label_b| / n."""
    b = s["bins" + region]
    return np.abs(b[:, :, 1] - b[:, :, 2]).sum(axis=1) / s["n"]


def check_stats(stats):
    """Raise ValueError if the statistics are malformed. Call this on every file you load."""
    missing = [k for k in SCALAR_FIELDS + BIN_FIELDS if k not in stats]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    P = len(stats["n"])
    for k in SCALAR_FIELDS:
        if stats[k].shape != (P,):
            raise ValueError(f"{k}: expected shape ({P},), got {stats[k].shape}")
        if not np.all(np.isfinite(stats[k])):
            raise ValueError(f"{k}: contains NaN or infinite values")
    for k in BIN_FIELDS:
        if stats[k].shape != (P, N_BINS, 3):
            raise ValueError(f"{k}: expected shape ({P}, {N_BINS}, 3), got {stats[k].shape}")
        if not np.all(np.isfinite(stats[k])):
            raise ValueError(f"{k}: contains NaN or infinite values")
        if not np.allclose(stats[k][:, :, 0].sum(axis=1), stats["n"]):
            raise ValueError(f"{k}: bin counts do not sum to n")
    if np.any(stats["syB"] <= 0) or np.any(stats["sgB"] <= 0):
        raise ValueError("every patient needs syB > 0 and sgB > 0 (a tumour must exist)")
    r = true_ratio(stats)
    if np.any(r < 0) or np.any(r > 1 + 1e-9):
        raise ValueError("true ratio outside [0, 1]: is region A really inside region B?")
    return True
