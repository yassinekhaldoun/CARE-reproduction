"""CARE (Confidence-Aware Ratio Estimation) partial reproduction.

Modules:
    stats        per-patient sufficient statistics: the file format shared by Kaggle and the laptop
    synthetic    fake statistics with a known true ratio and a controlled miscalibration
    intervals    the CARE interval (estimation term + calibration term)
    baselines    conformal prediction and a parametric bootstrap
    metrics      coverage, width, adaptiveness
    experiment   repeated random calibration/test splits
"""
