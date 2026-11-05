"""
The two central analyses under lasso and elastic-net estimators (BIB revision: R1-2, R4-4)
=========================================================================================

Reviewer 1 asks that the main conclusions be tested across gLV inference methods; Reviewer 4
that the claim "the observed asymmetry is manufactured by inference" be restricted to the
estimators actually calibrated. Every published analysis uses ridge. This script repeats

  (1) the held-out descent test (lyapunov_robustness.loco_descent: 13 leave-one-community-out
      folds, fitted-vs-scrambled potentials at the fitted weights d, 5000 scrambles per fold,
      same seeds), with the symmetrizable model fitted under an L1 (lasso) and an elastic-net
      penalty instead of ridge; and
  (2) the inference-aware null of Fig 3 (inference_aware_null.py: simulate from a fitted
      symmetrizable gLV with matched process noise, re-fit the FREE gLV through the identical
      pipeline, recompute the invariant diagnostics), with the free model re-fitted by lasso
      and by elastic net. Under ridge the free fit has no exact zeros, so the one-way-support
      diagnostic f_one is untestable; under L1 it becomes testable and is reported.

Penalties (all on OFF-diagonal interaction coefficients only, as in the published ridge; the
growth rates and the self-interactions are unpenalised)
-------------------------------------------------------------------------------------------
  objective   (1/2) ||y - A beta||^2 + lam1 sum_k w1_k |beta_k| + (lam2/2) sum_k w2_k beta_k^2
  ridge       lam1 = 0, lam2 = 0.5 (published)
  lasso       lam2 = 0
  elastic net glmnet-style mixing fixed at rho = 0.5: lam1 = rho*lam, lam2 = (1 - rho)*lam
  free fit         : w1 = w2 = 1 on J_ij, i != j (per-species regressions, as fit_free)
  symmetrizable fit: K = D J symmetric; the penalty is on the off-diagonal J it implies,
                     |J_ij| + |J_ji| = |K_ij| (1/d_i + 1/d_j) and J_ij^2 + J_ji^2 =
                     K_ij^2 (1/d_i^2 + 1/d_j^2), so w1 = 1/d_i + 1/d_j and w2 = 1/d_i^2 + 1/d_j^2
                     for K_ij (i < j), exactly the published ridge weights for w2. The weights d
                     are optimised as in the published fit: L-BFGS-B on the unpenalised
                     training MSE from u = 0, maxiter 60.

Penalty strength: a rule fixed before any scoring
-------------------------------------------------
Matched effective degrees of freedom. On the full EV3 data, at the published ridge fit's
weights (symmetrizable model) or per species (free model), lam is set so that the estimator's
degrees of freedom equal the ridge fit's: ridge df = trace of the hat matrix; lasso df = number
of nonzero coefficients (Zou, Hastie & Tibshirani 2007); elastic-net df = trace of the ridge
hat matrix on the active set (Zou & Hastie 2005). The calibrated lam is then held fixed for
every fold, every simulated data set and the observed fit alike. The descent test is also
repeated at 0.5x and 2x the calibrated lam, mirroring the published ridge sweep.

Run:  OMP_NUM_THREADS=1 python3 scripts/alternative_estimators.py [--workers 4] [--nnull 1000]
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np                                                       # noqa: E402
from scipy.optimize import minimize                                      # noqa: E402
from sklearn.linear_model import Lasso                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import (                                    # noqa: E402
    load_communities, transitions, fit_free, fit_constrained, _constrained_inner,
    S, EPS, LAM,
)
from lyapunov_descent import H_D                                         # noqa: E402
from lyapunov_robustness import scramble, NPERM                          # noqa: E402
from inference_aware_null import resid_std, simulate                     # noqa: E402
from symmetrizability_invariants import invariants as inv3               # noqa: E402
from descent_best_case_D import UBOUND                                   # noqa: E402

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

RHO = 0.5                                     # elastic-net mixing, fixed a priori
PAIRS = [(i, j) for i in range(S) for j in range(i, S)]
PIDX = {p: k for k, p in enumerate(PAIRS)}
OFFP = np.array([i != j for (i, j) in PAIRS])
PI = np.array([i for (i, j) in PAIRS])
PJ = np.array([j for (i, j) in PAIRS])
PHASE = {0: "growth", 1: "plateau", 2: "transfer", 3: "plateau", 4: "transfer", 5: "plateau"}


# --------------------------------------------------------------------------- #
#  generic penalised least squares
# --------------------------------------------------------------------------- #
def penalized_ls(A, y, w1, w2, lam1, lam2):
    """argmin (1/2)||y - A b||^2 + lam1 sum w1|b| + (lam2/2) sum w2 b^2.

    Columns with w1 = w2 = 0 are unpenalised and profiled out (Frisch-Waugh-Lovell); the L2
    part is absorbed by data augmentation; the L1 part by column rescaling, then sklearn's
    coordinate-descent Lasso. Returns (beta, active mask).
    """
    pen = (w1 > 0) | (w2 > 0)
    n, p = A.shape
    if lam1 == 0.0:
        P = np.diag(lam2 * w2) + 1e-6 * np.eye(p)
        beta = np.linalg.solve(A.T @ A + P, A.T @ y)
        return beta, np.ones(p, bool)
    npen = int(pen.sum())
    U, Pm = A[:, ~pen], A[:, pen]
    if lam2 > 0:
        Ua = np.vstack([U, np.zeros((npen, U.shape[1]))])
        Pa = np.vstack([Pm, np.diag(np.sqrt(lam2 * w2[pen]))])
        ya = np.concatenate([y, np.zeros(npen)])
    else:
        Ua, Pa, ya = U, Pm, y
    Up = np.linalg.pinv(Ua)
    yr = ya - Ua @ (Up @ ya)
    Pr = Pa - Ua @ (Up @ Pa)
    sc = w1[pen]
    X = Pr / sc[None, :]
    m = Lasso(alpha=lam1 / X.shape[0], fit_intercept=False, max_iter=100000, tol=1e-10,
              selection="cyclic")
    m.fit(X, yr)
    bp = m.coef_ / sc
    bu = Up @ (ya - Pa @ bp)
    beta = np.zeros(p)
    beta[pen] = bp
    beta[~pen] = bu
    active = np.ones(p, bool)
    active[pen] = bp != 0
    return beta, active


def df_of(A, w2, lam1, lam2, active):
    """Effective degrees of freedom: ridge hat trace on the active set."""
    As = A[:, active]
    if As.shape[1] == 0:
        return 0.0
    P = np.diag(lam2 * w2[active]) + 1e-6 * np.eye(As.shape[1])
    H = As @ np.linalg.solve(As.T @ As + P, As.T)
    if lam1 > 0 and lam2 == 0:
        return float(active.sum())              # lasso: number of nonzeros
    return float(np.trace(H))


def lams(kind, lam):
    if kind == "ridge":
        return 0.0, lam
    if kind == "lasso":
        return lam, 0.0
    if kind == "enet":
        return RHO * lam, (1 - RHO) * lam
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
#  free gLV (per species), any penalty
# --------------------------------------------------------------------------- #
def fit_free_pen(trans, kind, lam):
    lam1, lam2 = lams(kind, lam)
    r = np.zeros(S)
    J = np.zeros((S, S))
    df = 0.0
    for i in range(S):
        rows, ys = [], []
        for a, g, m in trans:
            if m[i]:
                rows.append(np.concatenate([[1.0], a]))
                ys.append(g[i])
        if len(rows) < S:                      # as fit_free: too few equations
            continue
        Z = np.array(rows)
        y = np.array(ys)
        w = np.ones(S + 1)
        w[0] = 0.0
        w[1 + i] = 0.0
        beta, act = penalized_ls(Z, y, w, w, lam1, lam2)
        r[i] = beta[0]
        J[i] = beta[1:]
        df += df_of(Z, w, lam1, lam2, act)
    return r, J, df


# --------------------------------------------------------------------------- #
#  symmetrizable gLV, any penalty
# --------------------------------------------------------------------------- #
def sym_design(trans):
    """Base design: row (t, i) for every masked (transition, species); columns r (S) then K
    pairs; K columns hold a_j and are divided by d_i at fit time."""
    rows_sp, ys, base = [], [], []
    for a, g, m in trans:
        for i in range(S):
            if not m[i]:
                continue
            row = np.zeros(S + len(PAIRS))
            row[i] = 1.0
            for j in range(S):
                key = (i, j) if i <= j else (j, i)
                row[S + PIDX[key]] += a[j]
            base.append(row)
            rows_sp.append(i)
            ys.append(g[i])
    return np.array(base), np.array(ys), np.array(rows_sp)


def sym_inner(u, des, kind, lam):
    base, y, sp = des
    d = np.exp(u - u.mean())
    A = base.copy()
    A[:, S:] /= d[sp][:, None]
    lam1, lam2 = lams(kind, lam)
    w1 = np.zeros(S + len(PAIRS))
    w2 = np.zeros(S + len(PAIRS))
    w1[S:][OFFP] = 1.0 / d[PI[OFFP]] + 1.0 / d[PJ[OFFP]]
    w2[S:][OFFP] = 1.0 / d[PI[OFFP]] ** 2 + 1.0 / d[PJ[OFFP]] ** 2
    beta, act = penalized_ls(A, y, w1, w2, lam1, lam2)
    r = beta[:S]
    K = np.zeros((S, S))
    K[PI, PJ] = beta[S:]
    K[PJ, PI] = beta[S:]
    J = K / d[:, None]
    mse = float(np.mean((A @ beta - y) ** 2))
    return r, J, mse, d, (A, w2, lam1, lam2, act)


def fit_sym_pen(trans, kind, lam, des=None, bound=None, return_u=False):
    """bound: optional box |u_i| <= bound on the log-weights. Unbounded, the lasso and
    elastic-net fits run away to weight ratios of e^40 (the outer objective is the
    UNPENALISED training MSE while the penalty weights depend on d, so extreme d switches the
    penalty off for some coefficients); the box is the repository's own gauge bound for this
    parametrisation (descent_best_case_D.UBOUND = 6) and leaves the published ridge fit inside."""
    des = des if des is not None else sym_design(trans)
    kw = {} if bound is None else {"bounds": [(-bound, bound)] * S}
    res = minimize(lambda u: sym_inner(u, des, kind, lam)[2], np.zeros(S),
                   method="L-BFGS-B", options={"maxiter": 60}, **kw)
    r, J, mse, d, _ = sym_inner(res.x, des, kind, lam)
    if return_u:
        return r, J, d, res.x
    return r, J, d


# --------------------------------------------------------------------------- #
#  matched degrees of freedom
# --------------------------------------------------------------------------- #
def calibrate(df_target, sig_df_fun, lo=1e-3, hi=10.0, ngrid=600):
    """Breakpoint-robust matched-df rule.

    Bisection on df lands exactly on a breakpoint of the L1 path (df jumps there), where a
    1e-5 change in lam changes the support. Instead: scan lam on a fine log grid, group
    consecutive grid points with an IDENTICAL active set into plateaus, evaluate df at each
    plateau's geometric centre, and return the centre of the plateau whose df is closest to
    the target. sig_df_fun(lam) -> (active-set signature as bytes, df)."""
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), ngrid))
    sigs = [sig_df_fun(x)[0] for x in grid]
    runs, start = [], 0
    for k in range(1, ngrid + 1):
        if k == ngrid or sigs[k] != sigs[start]:
            runs.append((start, k - 1))
            start = k
    best = None
    for a, b in runs:
        if b - a < 1:                       # single-point plateau: too narrow to centre
            continue
        centre = float(np.sqrt(grid[a] * grid[b]))
        sig_c, df_c = sig_df_fun(centre)
        if sig_c != sigs[a]:
            continue
        key = abs(df_c - df_target)
        if best is None or key < best[0]:
            best = (key, centre, df_c, grid[a], grid[b])
    if best is None:
        raise RuntimeError("no plateau found")
    return best[1], best[2], best[3], best[4]


# --------------------------------------------------------------------------- #
#  descent test, any estimator
# --------------------------------------------------------------------------- #
_COMMS = None


def _init(comms):
    global _COMMS
    _COMMS = comms


def _fold(job):
    kind, lam, h, nperm, bound = job
    comms = _COMMS
    train = transitions([c for k, c in enumerate(comms) if k != h])
    r, J, d, u = fit_sym_pen(train, kind, lam, bound=bound, return_u=True)
    name, X = comms[h]
    obs = np.diff([H_D(x, r, J, d) for x in X])
    K = d[:, None] * J
    nd = np.empty((nperm, len(obs)))
    for i in range(nperm):
        rng = np.random.default_rng(80000 + 997 * h + i)
        Js = scramble(K, d, rng)
        nd[i] = np.diff([H_D(x, r, Js, d) for x in X])
    off = ~np.eye(S, dtype=bool)
    nz = int(np.sum(J[off] != 0))
    return kind, lam, h, obs, nd, nz, d, float(np.max(np.abs(u))), bound


def descent_summary(obs_list, null_list, sel=None):
    o = np.concatenate(obs_list) <= 0
    nd = np.concatenate(null_list, axis=1) <= 0           # (nperm, ntrans)
    if sel is not None:
        o, nd = o[sel], nd[:, sel]
    f_obs = o.mean()
    f_null = nd.mean(axis=1)
    b = int(np.sum(f_null >= f_obs))
    return dict(n=len(o), down=int(o.sum()), f_obs=f_obs, f_null=f_null.mean(),
                lo=np.percentile(f_null, 2.5), hi=np.percentile(f_null, 97.5),
                margin=f_obs - f_null.mean(), b=b, p=(b + 1) / (len(f_null) + 1),
                B=len(f_null))


# --------------------------------------------------------------------------- #
#  inference-aware null, any free-fit estimator
# --------------------------------------------------------------------------- #
_NULLCFG = None


def _init_null(cfg):
    global _NULLCFG
    _NULLCFG = cfg


def pair_counts(J):
    """(interacting pairs, reciprocal pairs, discordant pairs, cyclomatic number of the
    reciprocal sign-concordant graph). r_cycle is identically 0 when the last is 0."""
    iu = np.triu_indices(S, 1)
    a, b = J[iu], J.T[iu]
    both = (a != 0) & (b != 0)
    either = (a != 0) | (b != 0)
    disc = both & (np.sign(a) != np.sign(b))
    conc = both & ~disc
    ii, jj = iu[0][conc], iu[1][conc]
    # cyclomatic number E - V + C over the nodes touched by concordant edges
    parent = list(range(S))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(ii, jj):
        parent[find(x)] = find(y)
    nodes = set(ii.tolist()) | set(jj.tolist())
    comps = len({find(x) for x in nodes})
    cyc = len(ii) - len(nodes) + comps
    return int(either.sum()), int(both.sum()), int(disc.sum()), int(cyc)


def _null_rep(i):
    comms, gts, ests = _NULLCFG
    out = []
    for gname, (rc, Jc, sproc) in gts.items():
        sim = simulate(comms, rc, Jc, sproc, np.random.default_rng(1000 + i))
        Y = transitions(sim)
        for ename, (kind, lam) in ests.items():
            if len(Y) < 10:
                out.append((gname, ename, np.nan, np.nan, np.nan, -1, -1, -1, -1))
                continue
            _, Jn, _ = fit_free_pen(Y, kind, lam)
            f1, fs, rcyc, _ = inv3(Jn)
            out.append((gname, ename, f1, fs, rcyc) + pair_counts(Jn))
    return i, out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nnull", type=int, default=1000)
    ap.add_argument("--nperm", type=int, default=NPERM)
    ap.add_argument("--skip-descent", action="store_true")
    ap.add_argument("--skip-null", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw from the saved results only")
    ap.add_argument("--unbounded-too", action="store_true",
                    help="also report the runaway unbounded lasso/enet fits, as a diagnostic")
    args = ap.parse_args()
    if args.replot:
        make_figure(np.load("figures/alternative_estimators.npy", allow_pickle=True).item())
        return
    from multiprocessing import Pool

    T0 = time.time()
    comms = load_communities()
    tr = transitions(comms)
    out = {}

    # ------------------------------------------------------------ 0. checks
    rf, Jf = fit_free(tr)
    rf2, Jf2, df_free_ridge = fit_free_pen(tr, "ridge", LAM)
    print(f"check free ridge reimplementation: max|dJ| = {np.max(np.abs(Jf - Jf2)):.2e}")
    des = sym_design(tr)
    for u in (np.zeros(S), np.random.default_rng(3).normal(0, 0.8, S)):
        r0, J0, m0 = _constrained_inner(u, tr, LAM)
        r1, J1, m1, _, _ = sym_inner(u, des, "ridge", LAM)
        print(f"check symmetrizable ridge reimplementation: max|dJ| = "
              f"{np.max(np.abs(J0 - J1)):.2e}, dMSE = {abs(m0 - m1):.2e}")
    f1o, fso, rco, _ = inv3(Jf)
    print(f"published free ridge fit: f_one={f1o:.3f} f_sign={fso:.3f} r_cycle={rco:.3f}\n")

    # ------------------------------------------------------------ 1. calibration
    rc_pub, Jc_pub = fit_constrained(tr)
    # the published fit's weights: recover u from the full-data fit
    res = minimize(lambda u: _constrained_inner(u, tr, LAM)[2], np.zeros(S),
                   method="L-BFGS-B", options={"maxiter": 60})
    u_pub = res.x
    _, _, _, _, (A0, w20, l10, l20, act0) = sym_inner(u_pub, des, "ridge", LAM)
    df_sym_ridge = df_of(A0, w20, 0.0, LAM, act0)

    def sig_sym(kind):
        def f(lam):
            r_, J_, _, _, (A, w2, l1, l2, act) = sym_inner(u_pub, des, kind, lam)
            return act.tobytes(), df_of(A, w2, l1, l2, act)
        return f

    def sig_free(kind):
        def f(lam):
            r_, J_, df = fit_free_pen(tr, kind, lam)
            return (J_ != 0).tobytes(), df
        return f

    cal, cal_info = {}, {}
    for kind in ("lasso", "enet"):
        for tag, target, fun in (("sym", df_sym_ridge, sig_sym(kind)),
                                 ("free", df_free_ridge, sig_free(kind))):
            lam, dfc, a, b = calibrate(target, fun)
            cal[(tag, kind)] = lam
            cal_info[(tag, kind)] = (dfc, a, b)
    print("matched effective degrees of freedom (full EV3 data; centre of the active-set "
          "plateau whose df is closest to ridge):")
    print(f"  symmetrizable: ridge df = {df_sym_ridge:.2f} (of {S + len(PAIRS)} parameters)")
    for kind in ("lasso", "enet"):
        dfc, a, b = cal_info[("sym", kind)]
        print(f"     {kind:<6} lam = {cal[('sym', kind)]:.4g}  (plateau [{a:.4g}, {b:.4g}])"
              f"  -> df = {dfc:.2f}")
    print(f"  free         : ridge df = {df_free_ridge:.2f} (of {S * (S + 1)} parameters)")
    for kind in ("lasso", "enet"):
        dfc, a, b = cal_info[("free", kind)]
        print(f"     {kind:<6} lam = {cal[('free', kind)]:.4g}  (plateau [{a:.4g}, {b:.4g}])"
              f"  -> df = {dfc:.2f}")
    print()
    out["cal"] = {f"{a}_{b}": v for (a, b), v in cal.items()}
    out["df_sym_ridge"] = df_sym_ridge
    out["df_free_ridge"] = df_free_ridge

    # ------------------------------------------------------------ 2. descent test
    if not args.skip_descent:
        UB = UBOUND
        settings = [("ridge", LAM, "published", None), ("ridge", LAM, "bounded", UB)]
        for kind in ("lasso", "enet"):
            lam = cal[("sym", kind)]
            settings += [(kind, lam, "matched df", UB), (kind, 0.5 * lam, "0.5x", UB),
                         (kind, 2.0 * lam, "2x", UB)]
        if args.unbounded_too:
            for kind in ("lasso", "enet"):
                settings.append((kind, cal[("sym", kind)], "unbounded", None))
        jobs = [(kind, lam, h, args.nperm, bd) for kind, lam, _, bd in settings
                for h in range(len(comms))]
        t0 = time.time()
        with Pool(args.workers, initializer=_init, initargs=(comms,)) as pool:
            res = pool.map(_fold, jobs, chunksize=1)
        print(f"descent folds done [{time.time() - t0:.0f} s]")
        phase = np.array([PHASE[k] for _ in comms for k in range(6)])
        print("\nHELD-OUT DESCENT TEST by estimator (13 folds, 78 transitions, "
              f"{args.nperm} scrambles per fold, fitted d in both arms)")
        print(f"{'estimator':<10}{'lam':>9}{'rule':>12}{'nz offdiag J':>14}{'max|u|':>8}"
              f"{'at bound':>9}{'fitted':>8}"
              f"{'scrambled':>11}{'null 95%':>17}{'margin':>9}{'p':>10}"
              f"{'plateau margin':>16}{'p plateau':>11}")
        rows = []
        for kind, lam, rule, bd in settings:
            rr = sorted([x for x in res if x[0] == kind and x[1] == lam and x[8] == bd],
                        key=lambda x: x[2])
            umax = max(x[7] for x in rr)
            atb = sum(1 for x in rr if bd is not None and x[7] >= bd - 1e-6)
            obs = [x[3] for x in rr]
            nul = [x[4] for x in rr]
            nz = np.mean([x[5] for x in rr])
            sa = descent_summary(obs, nul)
            sp = descent_summary(obs, nul, phase == "plateau")
            pa = f"<{1/(sa['B']+1):.1e}" if sa["b"] == 0 else f"{sa['p']:.4f}"
            pp = f"<{1/(sp['B']+1):.1e}" if sp["b"] == 0 else f"{sp['p']:.4f}"
            print(f"{kind:<10}{lam:>9.4g}{rule:>12}{nz:>14.1f}{umax:>8.2f}{atb:>6}/13"
                  f"{sa['f_obs']:>8.3f}"
                  f"{sa['f_null']:>11.3f}   [{sa['lo']:.3f},{sa['hi']:.3f}]"
                  f"{sa['margin']:>+9.3f}{pa:>10}{sp['margin']:>+16.3f}{pp:>11}")
            rows.append((kind, lam, rule, nz, sa, sp, umax, atb))
        out["descent"] = rows
        print()

    # ------------------------------------------------------------ 3. inference-aware null
    if not args.skip_null:
        ests = {"ridge": ("ridge", LAM),
                "lasso": ("lasso", cal[("free", "lasso")]),
                "enet": ("enet", cal[("free", "enet")])}
        obs, obs_cnt = {}, {}
        for ename, (kind, lam) in ests.items():
            _, Jo, _ = fit_free_pen(tr, kind, lam)
            obs[ename] = inv3(Jo)[:3]
            obs_cnt[ename] = pair_counts(Jo)
            print(f"observed free {ename:<6}: interacting {obs_cnt[ename][0]}, reciprocal "
                  f"{obs_cnt[ename][1]}, discordant {obs_cnt[ename][2]}, cycles in the "
                  f"concordant graph {obs_cnt[ename][3]}")
        # ground truths: the published ridge symmetrizable fit (same simulated data as the
        # published null for i < 300), and each penalty's own symmetrizable fit
        gts = {"GT ridge (published)": (rc_pub, Jc_pub, resid_std(comms, rc_pub, Jc_pub))}
        for kind in ("lasso", "enet"):
            r_, J_, d_, u_ = fit_sym_pen(tr, kind, cal[("sym", kind)], des, bound=UBOUND,
                                         return_u=True)
            print(f"GT {kind}: max|u| = {np.max(np.abs(u_)):.2f} (bound {UBOUND})")
            gts[f"GT {kind} (own)"] = (r_, J_, resid_std(comms, r_, J_))
        t0 = time.time()
        with Pool(args.workers, initializer=_init_null, initargs=((comms, gts, ests),)) as pool:
            nres = pool.map(_null_rep, range(args.nnull), chunksize=10)
        print(f"null replicates done [{time.time() - t0:.0f} s]")
        nres.sort()
        print(f"\nINFERENCE-AWARE NULL by free-fit estimator (N = {args.nnull}; p = (b+1)/(B+1), "
              "b = #null >= observed)")
        print(f"{'ground truth':<22}{'estimator':<8}{'diag':<9}{'observed':>9}{'null median':>13}"
              f"{'null 95th':>11}{'p':>9}{'p (first 300)':>15}{'p | same #pairs':>17}  note")
        table = []
        for gname in gts:
            for ename in ests:
                vals = np.array([[v[2], v[3], v[4], v[5], v[6], v[7], v[8]] for _, o in nres
                                 for v in o if v[0] == gname and v[1] == ename])
                nboth, ncyc = vals[:, 4], vals[:, 6]
                ob_both, ob_cyc = obs_cnt[ename][1], obs_cnt[ename][3]
                for k, dname in enumerate(("f_one", "f_sign", "r_cycle")):
                    nv = vals[:, k].copy()
                    if dname == "f_sign":
                        nv[nboth < 1] = np.nan          # undefined without reciprocal pairs
                    ok = np.isfinite(nv)
                    nv_all = nv[ok]
                    ob = obs[ename][k]
                    b = int(np.sum(nv_all >= ob))
                    p = (b + 1) / (len(nv_all) + 1)
                    nv300 = nv[:300][np.isfinite(nv[:300])]
                    p300 = (int(np.sum(nv300 >= ob)) + 1) / (len(nv300) + 1)
                    pc, note = np.nan, ""
                    if dname == "f_sign":
                        same = ok & (nboth == ob_both)
                        if same.sum() >= 20:
                            pc = (int(np.sum(nv[same] >= ob)) + 1) / (same.sum() + 1)
                        note = (f"reciprocal pairs obs {ob_both}, null median "
                                f"{np.median(nboth[nboth >= 0]):.0f}; same-#pairs n={int(same.sum())}")
                    if dname == "r_cycle":
                        note = (f"cycles obs {ob_cyc}, null with >=1 cycle "
                                f"{np.mean(ncyc >= 1) * 100:.0f}%")
                        if ob_cyc == 0:
                            note += "  (UNDEFINED: no cycle, r_cycle = 0 identically)"
                    if dname == "f_one" and np.nanmax(vals[:, 0]) == 0 and ob == 0:
                        note = "UNTESTABLE: no exact zeros under ridge"
                    pcs = f"{pc:.3f}" if np.isfinite(pc) else "  -"
                    print(f"{gname:<22}{ename:<8}{dname:<9}{ob:>9.3f}{np.median(nv_all):>13.3f}"
                          f"{np.percentile(nv_all, 95):>11.3f}{p:>9.3f}{p300:>15.3f}{pcs:>17}  {note}")
                    table.append((gname, ename, dname, ob, float(np.median(nv_all)),
                                  float(np.percentile(nv_all, 95)), p, p300, len(nv_all), pc, note))
        out["null"] = table
        out["null_raw"] = nres
        out["obs"] = obs
        out["obs_cnt"] = obs_cnt

    np.save("figures/alternative_estimators.npy", out, allow_pickle=True)

    # ------------------------------------------------------------ figure
    try:
        make_figure(out)
    except Exception as e:                                   # never lose the numbers
        print(f"[fig] failed: {e}")
    print(f"\ntotal wall time {time.time() - T0:.0f} s")


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14.0, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    if "descent" in out:
        rows = [r for r in out["descent"] if r[2] != "unbounded"]
        labs = [f"{r[0]}\n{r[2]}" for r in rows]
        fo = np.array([r[4]["f_obs"] for r in rows])
        fn = np.array([r[4]["f_null"] for r in rows])
        lo = np.array([r[4]["lo"] for r in rows])
        hi = np.array([r[4]["hi"] for r in rows])
        pv = [r[4]["p"] for r in rows]
        x = np.arange(len(rows))
        a.errorbar(x, fn, yerr=[fn - lo, hi - fn], fmt="o", color="#888888", capsize=3,
                   label="scrambled structure, mean and 95%")
        a.scatter(x, fo, color="#D55E00", zorder=5, s=45, label="fitted structure")
        for xi, (f, p_) in enumerate(zip(fo, pv)):
            a.text(xi, max(f, hi[xi]) + 0.012, f"p={p_:.3f}" if p_ >= 2e-4 else "p<2e-4",
                   ha="center", fontsize=8.5, rotation=90, va="bottom")
        a.set_xticks(x)
        a.set_xticklabels(labs, fontsize=9.5, rotation=45, ha="right")
        a.set_ylabel("held-out descent fraction, fitted d", fontsize=11)
        a.set_ylim(min(lo.min(), fo.min()) - 0.04, 1.02)
        a.set_title("(a) descent test by estimator (lasso, enet: |u| <= 6)", loc="left",
                    fontsize=11)
        a.tick_params(axis="y", labelsize=10)
        a.legend(frameon=False, fontsize=9.5, loc="lower left")
    a = ax[1]
    if "null" in out:
        rows = [t for t in out["null"] if t[0].startswith("GT ridge")]
        x = np.arange(len(rows))
        for xi, t in enumerate(rows):
            ename, dname, p_, pc, note = t[1], t[2], t[6], t[9], t[10]
            if "UNTESTABLE" in note or "UNDEFINED" in note:
                a.text(xi, 0.03, "untestable" if "UNTESTABLE" in note else "undefined",
                       rotation=90, ha="center", va="bottom", fontsize=9, color="#555555")
                continue
            a.bar(xi - 0.18 if np.isfinite(pc) else xi, p_, width=0.36 if np.isfinite(pc) else 0.6,
                  color="#4C72B0")
            if np.isfinite(pc) and dname == "f_sign" and ename != "ridge":
                a.bar(xi + 0.18, pc, width=0.36, color="#9ecae1")
        a.axhline(0.05, color="k", lw=1, ls="--")
        a.set_xticks(x)
        a.set_xticklabels([f"{t[1]}\n{t[2]}" for t in rows], fontsize=9.5)
        a.set_ylabel("p, observed vs inference-aware null", fontsize=11)
        a.set_ylim(0, 1.05)
        a.set_xlim(-0.7, len(rows) - 0.3)
        a.set_title("(b) inference-aware null by free-fit estimator", loc="left", fontsize=11)
        a.tick_params(axis="y", labelsize=10)
        from matplotlib.patches import Patch
        a.legend(handles=[Patch(color="#4C72B0", label="p"),
                          Patch(color="#9ecae1",
                                label="p, conditional on the number of reciprocal pairs")],
                 frameon=False, fontsize=9.5, loc="upper center")
    for x_ in ax:
        for sp in ("top", "right"):
            x_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/fig_alternative_estimators.png", dpi=170, bbox_inches="tight")
    print("[fig] figures/fig_alternative_estimators.png")


if __name__ == "__main__":
    main()
