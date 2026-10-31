"""Tests of the CARE code on synthetic data with a known true ratio.

Run from the project root:   python -m pytest tests/ -q
Each test states a property that the theory (or the paper) predicts.
"""
import numpy as np
import pytest

from care import stats as S, synthetic, intervals as I, baselines as B, metrics as M, experiment as E


@pytest.fixture(scope="module")
def data():
    st = synthetic.generate(300, seed=1)
    S.check_stats(st)
    return st


@pytest.fixture(scope="module")
def runs(data):
    return E.run_repeated_splits(data, n_repeats=30, level=0.68, seed=0)


def mean_of(runs, method, col):
    return runs[runs.method == method][col].mean()


# ---- the data format ------------------------------------------------------------------------
def test_ece_is_at_least_abs_vbias(data):
    """|V-Bias| <= binned ECE for every patient (triangle inequality), the inequality CARE uses."""
    for region in "AB":
        assert np.all(S.ece(data, region) + 1e-15 >= np.abs(S.vbias(data, region)))


def test_check_stats_rejects_bad_files(data):
    bad = {k: v.copy() for k, v in data.items()}
    bad["sgA"][0] = np.nan
    with pytest.raises(ValueError):
        S.check_stats(bad)
    bad = {k: v for k, v in data.items() if k != "binsA"}
    with pytest.raises(ValueError):
        S.check_stats(bad)


def test_save_load_roundtrip(data, tmp_path):
    path = tmp_path / "stats.npz"
    S.save_stats(path, data)
    back = S.load_stats(path)
    assert set(back) == set(data)
    for k in data:
        assert np.array_equal(back[k], data[k])


# ---- the estimation term --------------------------------------------------------------------
def test_plugin_variance_matches_simulation():
    """The plug-in delta-method variance equals the variance of r_hat over repeated draws."""
    rng = np.random.default_rng(0)
    n, reps = 1000, 2000
    cases = []
    for _ in range(reps):
        gB = rng.beta(2, 5, n)
        gA = gB * rng.uniform(0, 1, n)
        cases.append(S.summarize_case(gA, gB, np.ones(n), np.ones(n)))
    st = S.stack_cases(cases)
    empirical = np.var(S.pred_ratio(st), ddof=1)
    predicted = I.estimation_variance(st).mean()
    assert predicted == pytest.approx(empirical, rel=0.10)


# ---- validity -------------------------------------------------------------------------------
def test_conformal_coverage_close_to_level(runs):
    assert mean_of(runs, "Conformal", "coverage") == pytest.approx(0.68, abs=0.06)


@pytest.mark.parametrize("method", ["CARE (V-Bias)", "CARE (ECE)"])
def test_care_coverage_at_least_level(runs, method):
    assert mean_of(runs, method, "coverage") >= 0.68 - 0.02


def test_bootstrap_undercovers(runs):
    """Resampling captures only random noise, so it must cover less than conformal (paper: 5-9%)."""
    assert mean_of(runs, "Bootstrap", "coverage") < mean_of(runs, "Conformal", "coverage") - 0.2


def test_calibration_quantile_bound_holds(data):
    """On new patients, |V-Bias| <= q at about the requested rate (>= 1 - delta/2)."""
    rng = np.random.default_rng(3)
    perm = rng.permutation(S.n_cases(data))
    cal, test = S.subset(data, perm[:150]), S.subset(data, perm[150:])
    p = I.fit_care(cal, 0.68, "vbias")
    share = np.mean(np.abs(S.vbias(test, "A")) <= p["qA"])
    assert share >= 1 - p["delta"] / 2 - 0.08     # 0.08 = sampling slack with 150 test patients


# ---- width ----------------------------------------------------------------------------------
def test_ece_variant_is_wider_than_vbias(runs):
    """ECE >= |V-Bias| patient by patient, so its quantiles and intervals are at least as wide."""
    assert mean_of(runs, "CARE (ECE)", "width") >= mean_of(runs, "CARE (V-Bias)", "width") - 1e-12


def test_care_is_adaptive_and_conformal_is_not(runs):
    """CARE widths shrink as tumours grow (negative correlation); conformal widths barely depend on size."""
    assert mean_of(runs, "CARE (V-Bias)", "corr") < -0.5
    assert abs(mean_of(runs, "Conformal", "corr")) < 0.3


@pytest.mark.parametrize("fit_predict", ["care", "conformal"])
def test_width_grows_with_level(data, fit_predict):
    rng = np.random.default_rng(5)
    perm = rng.permutation(S.n_cases(data))
    cal, test = S.subset(data, perm[:150]), S.subset(data, perm[150:])
    widths = []
    for lv in (0.3, 0.5, 0.7, 0.9):
        if fit_predict == "care":
            lo, hi = I.predict(test, I.fit_care(cal, lv, "vbias"))
        else:
            lo, hi = B.conformal_predict(test, B.conformal_fit(cal, lv))
        widths.append(M.mean_width(lo, hi))
    assert all(b >= a - 1e-9 for a, b in zip(widths, widths[1:]))


# ---- edge cases -----------------------------------------------------------------------------
def test_denominator_guard_keeps_intervals_finite(data):
    """A huge q_B makes the denominator mean_gB - q_B negative: the interval must stay in [0, 1]."""
    params = {"alpha": 0.1, "delta": 0.2, "qA": 10.0, "qB": 10.0}
    lo, hi = I.predict(data, params)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(lo >= 0) and np.all(hi <= 1) and np.all(lo <= hi)


def test_intervals_are_ordered_and_in_unit_range(data):
    rng = np.random.default_rng(7)
    cal, test = S.subset(data, np.arange(150)), S.subset(data, np.arange(150, 300))
    for name, (lo, hi) in E.all_methods(cal, test, 0.68, rng).items():
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), name
        assert np.all(lo <= hi) and np.all(lo >= 0) and np.all(hi <= 1), name
