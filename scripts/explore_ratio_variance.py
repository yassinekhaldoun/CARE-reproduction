"""Task 2a: check the delta-method variance of a ratio of two means by simulation.

Setting: n voxels, x = g_B and y = g_A are per-voxel probabilities (A is a sub-region of B).
The predicted ratio is r_hat = mean(y) / mean(x).  If we could redraw the n voxels many times,
r_hat would vary; we measure that variance directly and compare it with two candidate formulas:

  correct  : (1/n) * [ var(y)/mx^2 + var(x)*my^2/mx^4 - 2*cov(x,y)*my/mx^3 ]
  summary  : (1/n) * [ var(y)/mx   + var(x)*my^2/mx^4 - 2*cov(x,y)*my/mx^3 ]   (first term as in the summary I read)

We also test the plug-in estimator (moments estimated from ONE sample), because that is what
the real code can compute for a patient.
"""
import numpy as np

rng = np.random.default_rng(0)
A_SHAPE, B_SHAPE = 2.0, 5.0          # g_B ~ Beta(2, 5): mean about 0.29, clearly not 1


def draw(shape):
    gb = rng.beta(A_SHAPE, B_SHAPE, size=shape)
    ga = gb * rng.uniform(0.0, 1.0, size=shape)   # g_A <= g_B: A is inside B
    return ga, gb


# True moments, from a very large sample
ga, gb = draw(2_000_000)
mx, my = gb.mean(), ga.mean()
vx, vy = gb.var(), ga.var()
cxy = np.mean((gb - mx) * (ga - my))
print(f"true moments: mx={mx:.4f} my={my:.4f} ratio={my/mx:.4f} vx={vx:.5f} vy={vy:.5f} cxy={cxy:.5f}\n")

REPS, CHUNK = 5000, 250
print(f"{'n':>6} | {'empirical':>11} | {'correct':>11} | {'summary':>11} | {'plug-in mean':>12}")
for n in (100, 1000, 10000):
    ratios, plugin = [], []
    for _ in range(REPS // CHUNK):
        y, x = draw((CHUNK, n))
        sx, sy = x.mean(1), y.mean(1)
        ratios.append(sy / sx)
        # plug-in SE from each simulated sample (ddof=1), as the real code will do
        v_x = x.var(1, ddof=1)
        v_y = y.var(1, ddof=1)
        c_xy = ((x - sx[:, None]) * (y - sy[:, None])).sum(1) / (n - 1)
        plugin.append((v_y / sx**2 + v_x * sy**2 / sx**4 - 2 * c_xy * sy / sx**3) / n)
    emp = np.var(np.concatenate(ratios), ddof=1)
    f_correct = (vy / mx**2 + vx * my**2 / mx**4 - 2 * cxy * my / mx**3) / n
    f_summary = (vy / mx + vx * my**2 / mx**4 - 2 * cxy * my / mx**3) / n
    print(f"{n:>6} | {emp:>11.3e} | {f_correct:>11.3e} | {f_summary:>11.3e} | {np.concatenate(plugin).mean():>12.3e}")
