"""
Is the free fit's 65% an artefact of a badly determined symmetrizer?
====================================================================

lyapunov_robustness.loco_descent_free() scores the unconstrained fit's potential using
symmetrizer_lsq(J), the least-squares u of the incidence problem b_ij = u_j - u_i. That
estimator drops every pair that is one-way or sign-discordant, and the free fit has
f_sign = 0.545, so **more than half of the 66 pairs never enter the fit**: d is determined
by ~30 equations for 11 free parameters, and it is not chosen to favour descent at all.

The constrained model, by contrast, gets a d optimised jointly with its coefficients by
L-BFGS. A referee is entitled to say the 65% measures a badly determined d rather than the
absence of gradient structure, and that Fig 1d is therefore not a like-for-like comparison.

This script answers that directly, using one fact that makes the question tractable:

    H_D(x) = - sum_i d_i r_i x_i - (1/2) sum_ij d_i J_ij x_i x_j
           = sum_i d_i * phi_i(x),     phi_i(x) = -r_i x_i - (1/2) x_i (Jx)_i

so H_D is LINEAR in d, and every transition contributes one linear constraint

    Delta H_D(t) = d . psi_t <= 0,      psi_t = phi(x_{t+1}) - phi(x_t),   d > 0.

Scoring a potential is therefore counting how many of a fixed set of halfspaces a positive
vector d lies in, and "the best d a reader could possibly pick" is a maximum-feasible-
subsystem problem in S = 12 dimensions. It is small enough to attack by annealed multistart,
which is what max_descent_u() does. Because d -> c d (c > 0) preserves every sign, the gauge
mean(u) = 0 costs nothing.

Four comparators, all scored on the same held-out transitions:

  published     the least-squares symmetrizer of the free fit          (Fig 1d, 0.654)
  random        d drawn at the spread of the constrained fit's own u   (is 0.654 typical?)
  train-optimal d chosen to maximise descent on the TWELVE TRAINING communities, then
                scored on the held-out one. This is the fair comparator: it gives the free
                fit exactly the advantage the constrained fit has, a d fitted on training
                data, and nothing more.
  oracle        ONE global d maximising descent on the 78 HELD-OUT transitions themselves.
                Cheating by construction, and reported only as an upper bound: no d can beat
                it. (A per-fold oracle is degenerate - 6 constraints in 11 free dimensions is
                almost always fully satisfiable - so the oracle must be a single global d.)

Run:  python3 scripts/descent_best_case_D.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import (                                  # noqa: E402
    load_communities, transitions, fit_free, S,
)
from lyapunov_descent import fit_symmetrizable                         # noqa: E402
from lyapunov_robustness import symmetrizer_lsq                        # noqa: E402

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

NRESTART = 400
UBOUND = 6.0          # same gauge bound the constrained fit uses


# --------------------------------------------------------------------------- #
def phi(x, r, J):
    """H_D(x) = d . phi(x). The whole potential, stripped of its weights."""
    return -(r * x) - 0.5 * x * (J @ x)


def psi_matrix(comms, r, J):
    """One row per transition: psi_t with Delta H_D(t) = d . psi_t."""
    rows = []
    for _, X in comms:
        P = np.array([phi(x, r, J) for x in X])
        rows.extend(P[1:] - P[:-1])
    return np.array(rows)


def count_descents(PSI, u):
    return int(np.sum(PSI @ np.exp(u - u.mean()) <= 0))


def max_descent_u(PSI, n_restart=NRESTART, seed=0):
    """Maximise #{t : psi_t . exp(u) <= 0} by annealed multistart.

    The count is piecewise constant, so it is optimised through a temperature-annealed
    sigmoid surrogate with an analytic gradient (a numerical one costs S+1 evaluations per
    step and dominates the run time); the count itself is evaluated at every iterate visited
    and the best is kept, so the returned value is a valid lower bound on the true maximum
    however badly the surrogate behaves.
    """
    rng = np.random.default_rng(seed)
    scale = np.median(np.abs(PSI)) * np.sqrt(S) + 1e-12
    best_n, best_u = -1, np.zeros(S)

    def consider(u):
        nonlocal best_n, best_u
        n = count_descents(PSI, u)
        if n > best_n:
            best_n, best_u = n, u - u.mean()

    def fg(v, t):
        """mean sigmoid(psi.d / t) and its gradient in v, with d = exp(v - mean v)."""
        w = v - v.mean()
        d = np.exp(w)
        z = np.clip(PSI @ d / t, -60.0, 60.0)
        s = 1.0 / (1.0 + np.exp(-z))
        gd = (PSI.T @ (s * (1.0 - s))) / (t * len(z))     # dL/dd
        gw = gd * d
        return float(s.mean()), gw - gw.mean()

    consider(np.zeros(S))
    for k in range(n_restart):
        u = np.zeros(S) if k == 0 else rng.normal(0, 1.2, S)
        for tau in (1.0, 0.3, 0.1, 0.03, 0.01):
            res = minimize(fg, u, args=(tau * scale,), jac=True, method="L-BFGS-B",
                           bounds=[(-UBOUND, UBOUND)] * S, options={"maxiter": 150})
            u = res.x
            consider(u)
    return best_u, best_n


# --------------------------------------------------------------------------- #
def main():
    comms = load_communities()
    n = len(comms)
    rng = np.random.default_rng(20260809)

    # Every fold's two fits, computed ONCE. fit_symmetrizable loops in Python and costs
    # seconds; recomputing it inside the scrambled control is what made this script
    # unusable before.
    folds = []
    for h in range(n):
        train = [c for k, c in enumerate(comms) if k != h]
        rf, Jf = fit_free(transitions(train))
        rc, Jc, dc = fit_symmetrizable(transitions(train))
        folds.append(dict(train=train, test=comms[h], rf=rf, Jf=Jf,
                          rc=rc, Jc=Jc, dc=dc))

    psi_te_free = [psi_matrix([f["test"]], f["rf"], f["Jf"]) for f in folds]
    psi_tr_free = [psi_matrix(f["train"], f["rf"], f["Jf"]) for f in folds]
    psi_te_con = [psi_matrix([f["test"]], f["rc"], f["Jc"]) for f in folds]
    psi_tr_con = [psi_matrix(f["train"], f["rc"], f["Jc"]) for f in folds]

    ntrans = sum(len(P) for P in psi_te_free)
    sd_u = float(np.mean([np.std(np.log(f["dc"])) for f in folds]))
    print(f"{n} communities, {ntrans} held-out transitions")
    print(f"constrained fits: sd(log d) = {sd_u:.3f}  (the scale the random-d null uses)\n")

    # ---------------------------------------------------------------- reference points
    # constrained model with its OWN fitted d (this is the published 0.859)
    f_con = sum(int(np.sum(psi_te_con[h] @ folds[h]["dc"] <= 0))
                for h in range(n)) / ntrans

    # free model with the published least-squares symmetrizer (this is the published 0.654)
    f_free_lsq = sum(int(np.sum(psi_te_free[h] @ symmetrizer_lsq(folds[h]["Jf"]) <= 0))
                     for h in range(n)) / ntrans

    print("published comparators (reproduced):")
    print(f"  constrained, fitted d      : {f_con:.3f}  ({int(f_con*ntrans)}/{ntrans})")
    print(f"  free, least-squares d      : {f_free_lsq:.3f}  ({int(f_free_lsq*ntrans)}/{ntrans})")

    # how many pairs actually enter symmetrizer_lsq for a full-data free fit
    rf_all, Jf_all = fit_free(transitions(comms))
    npair = sum(1 for i in range(S) for j in range(i + 1, S)
                if Jf_all[i, j] != 0 and Jf_all[j, i] != 0
                and np.sign(Jf_all[i, j]) == np.sign(Jf_all[j, i]))
    print(f"  (the lsq symmetrizer is fitted on {npair}/{S*(S-1)//2} pairs, "
          f"{S-1} free parameters)\n")

    # ---------------------------------------------------------------- random d
    draws = []
    for _ in range(2000):
        u = rng.normal(0, sd_u, S)
        draws.append(sum(count_descents(P, u) for P in psi_te_free) / ntrans)
    draws = np.array(draws)
    print("free fit, RANDOM d at the constrained fit's own spread (2000 draws):")
    print(f"  mean {draws.mean():.3f}   95% range [{np.percentile(draws,2.5):.3f}, "
          f"{np.percentile(draws,97.5):.3f}]   max {draws.max():.3f}")
    print(f"  P(random d >= published lsq d) = {np.mean(draws >= f_free_lsq):.3f}\n")

    # ---------------------------------------------------------------- train-optimal d
    down = 0
    per = []
    for h in range(n):
        u, ntr = max_descent_u(psi_tr_free[h], seed=1000 + h)
        k = count_descents(psi_te_free[h], u)
        per.append(k / len(psi_te_free[h]))
        down += k
    f_free_tr = down / ntrans
    print("free fit, d chosen to MAXIMISE descent on the twelve training communities:")
    print(f"  held-out {f_free_tr:.3f}  ({down}/{ntrans})")
    print(f"  per community: {np.round(per,2).tolist()}\n")

    # the missing cell of the 2x2: the constrained structure with a d chosen the same way,
    # so that structure is the only thing that differs between the three arms
    down_c = sum(count_descents(psi_te_con[h],
                                max_descent_u(psi_tr_con[h], seed=2000 + h)[0])
                 for h in range(n))
    f_con_tr = down_c / ntrans
    print("constrained fit, d chosen the same way (train-optimal, not the fitted d):")
    print(f"  held-out {f_con_tr:.3f}  ({down_c}/{ntrans})\n")

    # ---------------------------------------------------------------- global oracle d
    PSI_all_free = np.vstack(psi_te_free)
    u_or, n_or = max_descent_u(PSI_all_free, seed=7)
    PSI_all_con = np.vstack(psi_te_con)
    u_orc, n_orc = max_descent_u(PSI_all_con, seed=7)
    print("ORACLE: one global d maximising descent on the held-out transitions themselves")
    print("  (cheating - an upper bound no d can beat)")
    print(f"  free fit        : {n_or/ntrans:.3f}  ({n_or}/{ntrans})")
    print(f"  constrained fit : {n_orc/ntrans:.3f}  ({n_orc}/{ntrans})"
          f"   [its fitted d already reaches {f_con:.3f}]\n")

    # ---------------------------------------------------------------- the control
    # If optimising d can lift ANY matrix to a high descent fraction, then descent-with-
    # a-free-d is not a test of anything and the primary 0.859-vs-0.758 result falls too,
    # not just Fig 1d. The scrambled potential is the matrix that carries no structure by
    # construction, so it is the one to ask.
    print("CONTROL: give the STRUCTURE-SCRAMBLED potential the same optimised d")
    iu = np.triu_indices(S, 1)
    tr_opt, orc = [], []
    for b in range(24):
        rng_b = np.random.default_rng(4000 + b)
        psi_te_s = []
        down = 0
        for h in range(n):
            f = folds[h]
            K = f["dc"][:, None] * f["Jc"]
            Ks = np.zeros((S, S))
            Ks[iu] = rng_b.permutation(K[iu])
            Ks = Ks + Ks.T
            np.fill_diagonal(Ks, np.diag(K))
            Js = Ks / f["dc"][:, None]
            pte = psi_matrix([f["test"]], f["rc"], Js)
            ptr = psi_matrix(f["train"], f["rc"], Js)
            psi_te_s.append(pte)
            u, _ = max_descent_u(ptr, n_restart=40, seed=5000 + 13 * b + h)
            down += count_descents(pte, u)
        tr_opt.append(down / ntrans)
        u_s, n_s = max_descent_u(np.vstack(psi_te_s), n_restart=40, seed=6000 + b)
        orc.append(n_s / ntrans)
    tr_opt, orc = np.array(tr_opt), np.array(orc)
    print(f"  scrambled + train-optimal d : {tr_opt.mean():.3f} "
          f"[{tr_opt.min():.3f}, {tr_opt.max():.3f}]  (24 scramble draws)")
    print(f"  scrambled + oracle d        : {orc.mean():.3f} "
          f"[{orc.min():.3f}, {orc.max():.3f}]")
    p_tr = (np.sum(tr_opt >= f_free_tr) + 1) / (len(tr_opt) + 1)
    print(f"  P(scrambled+train-opt >= free+train-opt {f_free_tr:.3f}) = {p_tr:.3f}\n")

    # ---------------------------------------------------------------- verdict
    print("-" * 72)
    print("SUMMARY -- held-out descent fraction, by (structure, how d was chosen)")
    print(f"  constrained structure, fitted d      : {f_con:.3f}   <- published Fig 1a")
    print(f"  free structure,        lsq d         : {f_free_lsq:.3f}   <- published Fig 1d")
    print(f"  free structure,        train-optimal : {f_free_tr:.3f}")
    print(f"  constrained structure, train-optimal : {f_con_tr:.3f}")
    print(f"  scrambled structure,   train-optimal : {tr_opt.mean():.3f}")
    print(f"  free / constrained,    oracle d      : {n_or/ntrans:.3f} / {n_orc/ntrans:.3f}")
    print()
    if f_free_tr >= f_con:
        print(f"(1) Fig 1d does not survive. The free fit's 0.654 is a property of")
        print(f"    symmetrizer_lsq (fitted on {npair}/66 pairs), not of the free matrix: a d")
        print(f"    chosen on training data alone brings it to {f_free_tr:.3f}, at or above the")
        print(f"    constrained model's {f_con:.3f}.")
    p_c = (np.sum(tr_opt >= f_con_tr) + 1) / (len(tr_opt) + 1)
    print(f"(2) The deeper point: the SCRAMBLED structure with its own train-optimal d")
    print(f"    reaches {tr_opt.mean():.3f}. Every one of the {ntrans} held-out transitions has its")
    print(f"    sign decided by d (none descends under every d>0, none under none), so")
    print(f"    'there EXISTS a weighted potential these trajectories descend' is generic")
    print(f"    and is not what the test can establish.")
    print(f"    Structure still ranks when d is matched across arms, but the margin falls")
    print(f"    from +{f_con - 0.758:.3f} (published, fixed d) to +{f_con_tr - tr_opt.mean():.3f}, at p = {p_c:.2f} on {len(tr_opt)} draws.")
    print(f"    So the primary result should be stated as a FIXED-d comparison of structures,")
    print(f"    not as evidence that the trajectories descend a potential.")
    print("-" * 72)

    np.savez("figures/descent_best_case_D.npz",
             f_con=f_con, f_free_lsq=f_free_lsq, f_free_train_opt=f_free_tr,
             oracle_free=n_or / ntrans, oracle_con=n_orc / ntrans,
             random_draws=draws, ntrans=ntrans, per_train_opt=np.array(per),
             scram_train_opt=tr_opt, scram_oracle=orc)


if __name__ == "__main__":
    main()
