"""Tests of care.extract: from softmax probabilities and labels to sufficient statistics."""
import numpy as np
import pytest

from care import extract, stats as S


def fake_case(seed=0, shape=(12, 14, 10)):
    """Tiny 'patient': random labels 0..3 and a soft prediction that mostly agrees with them."""
    rng = np.random.default_rng(seed)
    seg = rng.choice(4, size=shape, p=[0.7, 0.1, 0.1, 0.1])
    logits = rng.normal(size=(4,) + shape) + 3.0 * np.eye(4)[seg].transpose(3, 0, 1, 2)
    e = np.exp(logits - logits.max(axis=0))
    return e / e.sum(axis=0), seg


def test_region_probabilities_use_the_right_classes():
    probs, seg = fake_case()
    st = extract.case_stats(probs, seg)
    assert st["sgA"] == pytest.approx(probs[2].sum())                        # necrosis = class 2
    assert st["sgB"] == pytest.approx((1.0 - probs[0]).sum())                # whole tumour = 1 - background
    assert st["syA"] == (seg == 2).sum()
    assert st["syB"] == (seg > 0).sum()
    assert st["n"] == seg.size


def test_region_a_is_inside_region_b():
    probs, seg = fake_case(seed=1)
    st = extract.case_stats(probs, seg)
    assert st["syA"] <= st["syB"] and st["sgA"] <= st["sgB"] + 1e-9


def test_label_array_with_leading_axis_is_accepted():
    probs, seg = fake_case(seed=2)
    a = extract.case_stats(probs, seg)
    b = extract.case_stats(probs, seg[None])          # SimpleITK read_seg returns shape (1, X, Y, Z)
    assert a["syA"] == b["syA"] and a["sgA"] == b["sgA"]


def test_output_matches_the_stats_format():
    cases = [extract.case_stats(*fake_case(seed=s)) for s in range(5)]
    st = S.stack_cases(cases)
    assert S.check_stats(st)
    assert S.n_cases(st) == 5


def test_bad_inputs_are_rejected():
    probs, seg = fake_case()
    with pytest.raises(ValueError):
        extract.case_stats(probs, seg[:-1])                              # different voxel grid
    with pytest.raises(ValueError):
        extract.case_stats(np.log(probs + 1e-9), seg)                    # logits, not probabilities
    with pytest.raises(ValueError):
        extract.case_stats(probs * 0.5, seg)                             # does not sum to 1
    with pytest.raises(ValueError):
        extract.case_stats(probs[:3], seg)                               # class 3 missing
