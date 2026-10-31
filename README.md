# CARE reproduction

Partial reproduction of *CARE: Confidence-Aware Ratio Estimation for Medical Biomarkers*
(Li et al., MIDL 2026): confidence intervals for the necrosis-to-tumour ratio computed from a
segmentation network (nnU-Net 3D, MSD Task01 BrainTumour).

Working guide: `docs/CARE_reproduction_guide.pdf`.

## Layout
- `care/`: the method (statistics format, intervals, baselines, metrics, experiment loop)
- `tests/`: checks on synthetic data with a known true ratio
- `scripts/`: `make_synthetic.py` (laptop), `explore_ratio_variance.py` (formula check)
- `notebooks/02_laptop_analysis.ipynb`: the analysis, runs on the laptop
- `data/`: `synthetic_stats.npz` now, `stats.npz` (from Kaggle) later

## Run on the laptop
```
conda env create -f environment.yml      # once
conda activate care
python -m pytest tests/ -q
python scripts/make_synthetic.py
```
Then open `notebooks/02_laptop_analysis.ipynb` in VS Code and select the `care` kernel.

## Status
Laptop side done on synthetic data. Kaggle side (training, statistics extraction) not started.
Deviations from the paper are logged in the guide (Part 8).
