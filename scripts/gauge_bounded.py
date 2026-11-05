r"""
How cheap is it to erase the diagnostics? The gauge against the fit residual.
=============================================================================
The closure of the EV3 abundances is approximate: the twelve relative abundances sum to
between 0.9998 and 1.0002 rather than exactly one. A per-row shift of size M therefore is not
exactly free: it changes each predicted growth rate by about M times the closure error, so
2e-4 per unit of M. Meanwhile the diagnostics fall roughly as 1/M. The right question is
therefore quantitative: how far can the diagnostics of the fitted matrix be pushed down before
the predictions degrade by a noticeable fraction of the fit residual?
"""
import os, sys, argparse
import numpy as np
from scipy.optimize import minimize

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=".")
ARGS = ap.parse_args()
sys.path.insert(0, os.path.join(ARGS.repo, "scripts")); os.chdir(ARGS.repo)
from constrained_vs_free_glv import load_communities, transitions, fit_free, S
from symmetrizability_invariants import invariants
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")

comms = load_communities(); tr = transitions(comms)
rf, Jf = fit_free(tr)
off = ~np.eye(S, dtype=bool); scale = float(np.median(np.abs(Jf[off])))

res = []
for a, g, m in tr:
    p = rf + Jf @ a
    res.extend((g[m] - p[m]).tolist())
res = np.array(res); sd = float(res.std())
gsd = float(np.std([g[m] for a, g, m in tr for g in [g]][0])) if False else None
allg = np.array([v for a, g, m in tr for v in g[m]])
print(f"fit residual standard deviation      : {sd:.4f}")
print(f"spread of the observed growth rates  : {allg.std():.4f}")
print(f"median |off-diagonal| of the free fit: {scale:.4f}")

def shift(J, c): return J - c[:, None]
def pred_change(c):
    w = 0.0
    for a, g, m in tr:
        w = max(w, float(np.max(np.abs((shift(Jf, c) - Jf) @ a + c))))
    return w
def rms_change(c):
    v = []
    for a, g, m in tr:
        v.extend((((shift(Jf, c) - Jf) @ a + c)[m]).tolist())
    return float(np.sqrt(np.mean(np.array(v) ** 2))) if v else 0.0

print(f"\nuniform shifts: diagnostics against the cost in prediction")
print(f"  {'M':>8}{'M/scale':>9}{'f_sign':>8}{'r_cycle':>9}"
      f"{'rms pred change':>17}{'as a share of the residual sd':>31}")
for M in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
    c = -M * np.ones(S)
    f1, fs, rc, E = invariants(shift(Jf, c))
    rmsc = rms_change(c)
    print(f"  {M:>8.1f}{M/scale:>9.1f}{fs:>8.3f}{rc:>9.4f}{rmsc:>17.5f}{rmsc/sd:>31.4f}")

print("\nbest gauge subject to a bound on the prediction change")
for frac in (0.01, 0.05, 0.10):
    lim = frac * sd
    def obj(c):
        pc = rms_change(c)
        f1, fs, rc, E = invariants(shift(Jf, c))
        pen = 0.0 if pc <= lim else 1e3 * (pc - lim)
        if E < 3 or not np.isfinite(rc): return 10.0 + pen
        return fs + rc + pen
    best = None
    for k in range(10):
        c0 = np.zeros(S) if k == 0 else np.random.default_rng(k).normal(0, 2 * scale, S)
        r_ = minimize(obj, c0, method="Powell",
                      options=dict(maxiter=30000, xtol=1e-7, ftol=1e-11))
        if best is None or r_.fun < best[0]: best = (r_.fun, r_.x.copy())
    c = best[1]; f1, fs, rc, E = invariants(shift(Jf, c))
    print(f"  prediction change limited to {frac:.0%} of the residual sd "
          f"({lim:.5f}):")
    print(f"    f_oneway {f1:.3f}  f_sign {fs:.3f}  r_cycle {rc:.4f}  concordant {E}  "
          f"achieved change {rms_change(c):.5f}")
    print(f"    shift sizes / coupling scale: "
          f"{np.array2string(np.abs(c)/scale, precision=1, max_line_width=120)}")

print("\nfor contrast, the same sweep on the PUBLISHED matrix, which was fitted to absolute")
print("measurements and therefore has no gauge freedom at all; the shift is only a")
print("robustness probe there, not a reparameterisation")
V = np.loadtxt("data/venturelli2018/interaction_matrix_averaged.csv", delimiter=",", skiprows=1)
sv = float(np.median(np.abs(V[off][V[off] != 0])))
print(f"  {'M/scale':>9}{'f_sign':>8}{'r_cycle':>9}")
for M in (0.0, 0.5, 1.0, 2.0, 5.0, 20.0):
    f1, fs, rc, E = invariants(V - (-M * sv) * np.ones(S)[:, None] * -1)
    f1, fs, rc, E = invariants(V + M * sv * np.ones((S, S)))
    print(f"  {M:>9.1f}{fs:>8.3f}{rc:>9.4f}")
