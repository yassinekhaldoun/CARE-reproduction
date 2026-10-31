"""Create data/synthetic_stats.npz: fake per-patient statistics with the same format as the real file.

Usage (from the project root):   python scripts/make_synthetic.py
Notebook 2 uses this file until the real data/stats.npz arrives from Kaggle.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from care import synthetic, stats as S  # noqa: E402

NUM_CASES = 100          # about the number of held-out patients of one nnU-Net fold
SEED = 0

if __name__ == "__main__":
    st = synthetic.generate(NUM_CASES, seed=SEED)
    S.check_stats(st)
    out = ROOT / "data" / "synthetic_stats.npz"
    S.save_stats(out, st)
    print(f"saved {S.n_cases(st)} patients to {out}")
