"""Fake per-patient statistics with a KNOWN true ratio and a controlled miscalibration.

Each patient has n voxels. Labels: region B (tumour) occupies a random fraction of the voxels,
region A (necrosis) is a random fraction of B. The "network" sees a noisy score per voxel
(Gaussian, centred at +a for positives and -a for negatives) and converts it to a probability with
the Bayes posterior for a GLOBAL prior. Miscalibration knobs:

    temp     > 1 makes the network over-confident, < 1 under-confident
    shift_*  adds a constant to the logit (systematic over- or under-estimation of the volume)
    tau      patient-to-patient random shift (heterogeneous calibration, as in real data)

Defaults describe a moderately accurate, moderately miscalibrated network (a = 2.25 gives a
per-patient volume error of roughly 10% of the predicted volume). Pass temp=1, shift_A=0,
shift_B=0, tau=0 to get a nearly calibrated one.

Simplifications to keep in mind: voxels are independent (real tumours are blobs), and region A
is not forced to satisfy g_A <= g_B.
"""
import numpy as np

from .stats import summarize_case, stack_cases


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _prob(z, prior, a, s, temp, shift):
    logit = np.log(prior / (1 - prior)) + 2 * a * z / s ** 2
    return _sigmoid(temp * logit + shift)


def make_case(rng, n=50_000, size_range=(0.004, 0.08), necrosis_range=(0.1, 0.6),
              a=2.25, s=1.0, prior_A=0.01, prior_B=0.03,
              temp=1.2, shift_A=0.2, shift_B=0.1, tau=0.3):
    lo, hi = np.log(size_range[0]), np.log(size_range[1])
    fB = np.exp(rng.uniform(lo, hi))
    q = rng.uniform(*necrosis_range)
    yB = rng.random(n) < fB
    yA = yB & (rng.random(n) < q)
    zB = np.where(yB, a, -a) + s * rng.standard_normal(n)
    zA = np.where(yA, a, -a) + s * rng.standard_normal(n)
    dA = shift_A + tau * rng.standard_normal()
    dB = shift_B + tau * rng.standard_normal()
    gB = _prob(zB, prior_B, a, s, temp, dB)
    gA = _prob(zA, prior_A, a, s, temp, dA)
    return summarize_case(gA, gB, yA, yB)


def generate(num_cases, seed=0, **kwargs):
    """Stats for `num_cases` fake patients. kwargs are passed to make_case."""
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < num_cases:
        c = make_case(rng, **kwargs)
        if c["syB"] > 0 and c["syA"] > 0:     # need a non-empty tumour with some necrosis
            cases.append(c)
    stats = stack_cases(cases)
    stats["case_id"] = np.array([f"synth{i:04d}" for i in range(num_cases)])
    return stats
