"""
C10 check -- does the keystone ranking depend on the fabricated r=1 growth rates?
=================================================================================

The keystone knockouts (human_keystones.py, keystones_from_data.py) discard the
regression intercept and simulate with r=np.ones(S). The intercept coef[0] IS the
per-taxon growth rate of the fitted gLV, so r=1 is a fabricated input. Here we
extract the inferred r (same ridge fit, same normalization as fit_matrix) and
re-run the knockout with r_inferred, then compare the ranking to the r=1 ranking
(top-K overlap + Spearman). If they disagree, the ranking is an artefact of r=1.

Run:  python3 check_keystone_r.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

from infer_and_place import RIDGE
from human_keystones import load, design, bc, S_TAXA, LIT

NMAX, EPS = 50.0, 1e-3


def relax(J, n0, r, T=120):
    """Robust relaxation: looser tolerances + capped step so that a stiff
    inferred-r system still integrates in seconds (the ranking only needs the
    attractor, not high-accuracy transients)."""
    def rhs(t, n):
        n = np.clip(n, 0, NMAX)
        return n * (r - n + J @ n) + EPS
    s = solve_ivp(rhs, (0, T), np.clip(np.nan_to_num(n0), 0, NMAX), method="LSODA",
                  t_eval=[T], rtol=1e-4, atol=1e-6, max_step=2.0)
    return np.clip(np.nan_to_num(s.y[:, -1]), 0, NMAX)


def knockout_impact(J, r):
    S = J.shape[0]
    nb = relax(J, np.random.default_rng(0).uniform(0.1, 1, S), r)
    imp = np.zeros(S)
    for i in range(S):
        Jk = J.copy(); Jk[i, :] = 0; Jk[:, i] = 0
        n0 = nb.copy(); n0[i] = 0
        nf = relax(Jk, n0, r)
        oth = np.delete(np.arange(S), i)
        imp[i] = bc(nf[oth], nb[oth])
    return imp / (imp.max() + 1e-12)


def fit_J_and_r(Y, Z):
    """Ridge fit returning BOTH the normalized interaction matrix J and the
    normalized growth-rate vector r (the intercept), consistent with
    infer_and_place.fit_matrix's self-scaling."""
    S = Y.shape[1]
    A = Z.T @ Z + RIDGE * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ Y)          # (1+S) x S
    J = coef[1:1 + S].T
    r = coef[0].copy()                          # intercept = per-taxon growth rate
    diag = np.diag(J)
    self_scale = np.median(-diag[diag < 0]) if np.any(diag < 0) else 1.0
    if self_scale <= 0:
        self_scale = 1.0
    return J / self_scale, r / self_scale


def main():
    X, day, pid, names = load()
    prop = X / X.sum(1, keepdims=True)
    top = np.argsort(prop.mean(0))[::-1][:S_TAXA]
    labels = names[top]
    Y, Z, _ = design(X, day, pid, top)

    J, r_raw = fit_J_and_r(Y, Z)
    # bound r to a physiological range so the stiff inferred system integrates;
    # sign and rank of r are preserved, only extreme magnitudes are capped.
    r_inf = np.clip(r_raw, -2.0, 2.0)
    print(f"human QMP: {Y.shape[0]} transitions, {S_TAXA} genera", flush=True)
    print(f"inferred r (raw): mean={r_raw.mean():+.2f}, sd={r_raw.std():.2f}, "
          f"range=[{r_raw.min():+.2f}, {r_raw.max():+.2f}], "
          f"n_negative={(r_raw < 0).sum()}/{S_TAXA}  (paper assumes all r=+1)", flush=True)

    imp_1 = knockout_impact(J, np.ones(J.shape[0]))   # r = ones (as in the paper)
    print("  [r=1 knockouts done]", flush=True)
    imp_r = knockout_impact(J, r_inf)                 # r = inferred (clipped)
    print("  [r=inferred knockouts done]", flush=True)

    order1 = np.argsort(imp_1)[::-1]
    orderr = np.argsort(imp_r)[::-1]
    for K in (6, 12):
        ov = len(set(order1[:K]) & set(orderr[:K]))
        print(f"  top-{K} overlap (r=1 vs r_inferred): {ov}/{K}")
    rho = spearmanr(imp_1, imp_r).correlation
    print(f"  Spearman rank correlation over all {S_TAXA} genera: rho={rho:.2f}")

    print("\n  rank  r=1 genus            | r_inferred genus")
    print("  " + "-" * 48)
    for j in range(8):
        g1 = labels[order1[j]] + (" *" if labels[order1[j]] in LIT else "")
        gr = labels[orderr[j]] + (" *" if labels[orderr[j]] in LIT else "")
        print(f"  {j+1:>4}  {g1:20s} | {gr:20s}")

    verdict = ("ROBUST: ranking largely preserved" if rho >= 0.6 else
               "FRAGILE: ranking depends materially on the fabricated r=1")
    print(f"\nverdict: {verdict}  (* = classic literature keystone)")


if __name__ == "__main__":
    main()
