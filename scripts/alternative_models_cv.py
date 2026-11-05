"""
The designed-consortium trajectories under alternative dynamical models (BIB revision, R1-3)
===========================================================================================

Reviewer 1 asks for a comparison with alternative dynamical models rather than only fitted
versus scrambled structure. The descent test itself is specific to the gLV potential, so the
comparison is made on what every model can be scored on: leave-one-community-out one-step
prediction of the per-capita growth proxy, with the same transitions, the same equation mask and
the same off-diagonal ridge as the published free/constrained comparison. Models, in order of
structure:

  persistence      g = 0 (composition carried forward)
  growth only      g_i = r_i (independent species, no interactions)
  log-linear MAR   g_i = r_i + sum_j B_ij log(x_j + eps_pc)  (multivariate autoregressive /
                   Gompertz model of community time series, interactions in log abundance),
                   at the gLV penalty and with its penalty chosen by nested leave-one-community-
                   out CV inside each training set (the regressors live on a log scale, so the
                   gLV value is not a fair default)
  free gLV         g_i = r_i + sum_j J_ij x_j
  symmetrizable    the free gLV restricted to J = D^{-1} K, K symmetric (the weighted-gradient
  gLV              family the descent test probes)

Reported: held-out MSE per model (mean over the 13 held-out communities), the paired difference
to the symmetrizable gLV with a bootstrap interval over communities and a paired sign-flip
permutation p, and the number of communities each model predicts better.

Run:  OMP_NUM_THREADS=2 python3 scripts/alternative_models_cv.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import (load_communities, transitions, fit_free, fit_constrained,  # noqa: E402
                                     pred_mse, S, EPS, LAM)

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

NBOOT = 10000
NPERM = 20000


def fit_growth_only(trans):
    r = np.zeros(S)
    for i in range(S):
        ys = [g[i] for a, g, m in trans if m[i]]
        r[i] = np.mean(ys) if ys else 0.0
    return r, np.zeros((S, S))


def fit_mar(trans, lam=LAM):
    """Log-linear MAR: g_i = r_i + sum_j B_ij log(x_j + eps); ridge on off-diagonal B."""
    r = np.zeros(S); B = np.zeros((S, S))
    for i in range(S):
        rows, ys = [], []
        for a, g, m in trans:
            if m[i]:
                rows.append(np.concatenate([[1.0], np.log(a + EPS)])); ys.append(g[i])
        if len(rows) < S:
            continue
        Z = np.array(rows); y = np.array(ys)
        P = np.full(S + 1, lam); P[0] = 0.0; P[1 + i] = 0.0
        beta = np.linalg.solve(Z.T @ Z + np.diag(P) + 1e-6 * np.eye(S + 1), Z.T @ y)
        r[i] = beta[0]; B[i] = beta[1:]
    return r, B


def mse_mar(r, B, trans):
    se, n = 0.0, 0
    for a, g, m in trans:
        pred = r + B @ np.log(a + EPS)
        se += np.sum((g[m] - pred[m]) ** 2); n += m.sum()
    return se / max(n, 1)


MAR_GRID = (0.5, 5.0, 50.0, 500.0, 5000.0)


def mar_nested(train_comms):
    """Choose the MAR penalty by leave-one-community-out CV inside the training communities only."""
    best, best_lam = np.inf, None
    for lam in MAR_GRID:
        e = []
        for h in range(len(train_comms)):
            tr = transitions([c for k, c in enumerate(train_comms) if k != h])
            te = transitions([train_comms[h]])
            r, B = fit_mar(tr, lam)
            e.append(mse_mar(r, B, te))
        if np.mean(e) < best:
            best, best_lam = np.mean(e), lam
    return best_lam


def main():
    comms = load_communities()
    names = ["persistence", "growth only", "log-linear MAR", "MAR, nested penalty",
             "free gLV", "symmetrizable gLV"]
    npar = [0, S, S + S * S, S + S * S, S + S * S, S + S * (S + 1) // 2 + (S - 1)]
    chosen = []
    err = {k: [] for k in names}
    for h in range(len(comms)):
        train = transitions([c for k, c in enumerate(comms) if k != h])
        test = transitions([comms[h]])
        err["persistence"].append(pred_mse(np.zeros(S), np.zeros((S, S)), test))
        r, J = fit_growth_only(train); err["growth only"].append(pred_mse(r, J, test))
        r, B = fit_mar(train); err["log-linear MAR"].append(mse_mar(r, B, test))
        lam = mar_nested([c for k, c in enumerate(comms) if k != h]); chosen.append(lam)
        r, B = fit_mar(train, lam); err["MAR, nested penalty"].append(mse_mar(r, B, test))
        r, J = fit_free(train); err["free gLV"].append(pred_mse(r, J, test))
        r, J = fit_constrained(train); err["symmetrizable gLV"].append(pred_mse(r, J, test))
    err = {k: np.array(v) for k, v in err.items()}
    ref = err["symmetrizable gLV"]
    rng = np.random.default_rng(3)
    print(f"{'model':<20}{'params':>7}{'CV MSE':>9}{'minus symm.':>13}{'95% bootstrap':>20}"
          f"{'perm p':>9}{'better in':>11}")
    for k, q in zip(names, npar):
        e = err[k]
        diff = e - ref
        if k == "symmetrizable gLV":
            print(f"{k:<20}{q:>7}{e.mean():>9.3f}{'':>13}{'':>20}{'':>9}{'':>11}")
            continue
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(NBOOT)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        flips = rng.choice([-1, 1], size=(NPERM, len(diff)))
        pp = (np.sum(np.abs((flips * diff).mean(1)) >= abs(diff.mean())) + 1) / (NPERM + 1)
        print(f"{k:<20}{q:>7}{e.mean():>9.3f}{diff.mean():>+13.4f}   [{lo:+.4f}, {hi:+.4f}]"
              f"{pp:>9.4f}{int(np.sum(diff < 0)):>8}/13")
    print(f"\nMAR penalty chosen by nested CV per outer fold: {chosen}")
    np.savez("figures/alternative_models_cv.npz", mar_lambda=np.array(chosen), names=np.array(names), npar=np.array(npar),
             **{k.replace(" ", "_"): v for k, v in err.items()})
    print("\nwrote figures/alternative_models_cv.npz")


if __name__ == "__main__":
    main()
