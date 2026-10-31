"""From ONE patient's nnU-Net output to the sufficient statistics of care.stats.

Pure NumPy, so it is tested on the laptop (tests/test_extract.py) and imported unchanged by the
Kaggle script scripts/extract_stats.py.

Region definitions for MSD Task01 BrainTumour (labels: 0 background, 1 edema,
2 non-enhancing tumour, 3 enhancing tumour):
    region A (numerator)   = class 2                    "necrosis"        [VERIFY against the paper]
    region B (denominator) = classes 1 + 2 + 3          "whole tumour"    [VERIFY against the paper]
The soft probabilities of a region are the sum of its class probabilities, g_B = 1 - p(background).
"""
import numpy as np

from .stats import summarize_case

NECROSIS_CLASSES = (2,)
TUMOUR_CLASSES = (1, 2, 3)


def case_stats(probs, seg, a_classes=NECROSIS_CLASSES, b_classes=TUMOUR_CLASSES):
    """Sufficient statistics for one patient.

    probs: softmax probabilities, shape (n_classes, X, Y, Z), summing to 1 over axis 0
    seg:   ground-truth integer labels, shape (X, Y, Z) or (1, X, Y, Z)
    """
    probs = np.asarray(probs)
    seg = np.asarray(seg)
    if seg.ndim == probs.ndim:
        seg = seg[0]
    if probs.shape[1:] != seg.shape:
        raise ValueError(f"probabilities {probs.shape} and labels {seg.shape} do not have the same voxel grid")
    if probs.min() < -1e-4 or probs.max() > 1 + 1e-4:
        raise ValueError("probabilities outside [0, 1]: did you pass logits instead of softmax output?")
    if not np.allclose(probs.sum(axis=0).mean(), 1.0, atol=1e-3):
        raise ValueError("probabilities do not sum to 1 over the class axis")
    if max(a_classes + b_classes) >= probs.shape[0]:
        raise ValueError("a region uses a class index that the network does not have")

    gA = probs[list(a_classes)].sum(axis=0)
    gB = probs[list(b_classes)].sum(axis=0)
    gA = np.clip(gA, 0.0, 1.0)
    gB = np.clip(gB, 0.0, 1.0)
    yA = np.isin(seg, a_classes)
    yB = np.isin(seg, b_classes)
    return summarize_case(gA, gB, yA, yB)
