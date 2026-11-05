"""
Specificity of the descent test over many ways of leaving the symmetrizable manifold
====================================================================================

BIB-26-2181 revision, node C (Reviewer 2 major 3, Reviewer 3 points 3-4, Reviewer 4 point 3)
and the comparator part of node B (Reviewer 4 point 2).

descent_specificity.py measured specificity along ONE fixed antisymmetric direction, from ONE
ground truth, at a process-noise scale chosen to reproduce the observed descent fraction. The
reviewers ask whether that generalises. This script replaces the single curve by an ensemble:

  ground truths, all starting from J0 = the symmetrizable fit to the real EV3 data
    a  antisym   J0 + alpha P, P a random antisymmetric direction (8 seeds; seed 0 is the
                 published direction)
    b  oneway    condition (i) only: a growing random set of reciprocal pairs made one-way
    c  sign      condition (ii) only: a growing random set of pairs given discordant signs
    d  cycle     condition (iii) only: J_ij -> J_ij e^{k c_ij/2}, J_ji -> J_ji e^{-k c_ij/2},
                 c antisymmetric and projected onto the cycle space of the pair graph, so
                 support and signs are untouched and only the log-ratios acquire curl
    e  free      the FREE fit to the real data and its extrapolations J0 + t (J_free - J0):
                 non-symmetrizable (f_sign 0.55, r_cycle 1.5) yet tied with the symmetrizable
                 model in leave-one-community-out prediction. The matched-performance
                 alternative Reviewer 4 asks for, and the direction in which the estimator's
                 own asymmetry lies.

  common scale  delta_M = distance of the ground truth's vector field from the NEAREST weighted
                gradient field, measured along the observed EV3 states:
                    delta_M = min_{r_s, J_s symmetrizable} RMS[(r + J x) - (r_s + J_s x)]
                              / RMS[J0 x]
                over the (community, time, species) equations the fit itself uses. It is
                invariant under the species-wise rescaling (x_j -> s_j x_j, J_ij -> J_ij/s_j
                leaves every per-capita growth rate unchanged), it is zero exactly on the
                manifold as seen by the data, and it measures what a trajectory-based test
                could possibly detect. Unlike a Frobenius residual on coefficients it cannot be
                driven to zero by an unbounded symmetrizer (the paper's objection to R_sym):
                the weights are bounded exactly as in the fitter, |u| <= 6.
                Also reported: delta_VF (vector-field displacement from J0 itself) and
                delta_F (Frobenius displacement / ||offdiag J0||).

  noise         process noise sigma_p (on the latent log-abundances, propagates) and
                observation noise sigma_o (multiplicative lognormal on each reported
                composition, renormalised, does not propagate), estimated INDEPENDENTLY of
                the descent statistic from the one-step residuals of the real fit: their
                lag-1 autocovariance is -sigma_o^2 under an MA(1) error, the rest of their
                variance bounds sigma_p from above. A 3x3 grid around those values replaces
                the calibration to the observed descent fraction; the published calibrated
                setting is rerun for continuity.

  protocol      unchanged: on every replicate the symmetrizable model is fitted on 12
                simulated communities (descent_specificity.fit_constrained_fast_d) and scored
                on the 13th, 78 transitions pooled, 400 structure-scrambled potentials,
                reject at alpha = 0.05. Scoring is vectorised but uses the published seeds and
                permutations, and is asserted equal to the published loop on the real data.
                The predictive comparison (detection_power.py) is computed on the SAME
                replicates from the same constrained fits, so the two tests are compared on
                identical data.

Run:  OMP_NUM_THREADS=1 python3 scripts/descent_specificity_ensemble.py --scan   (minutes)
      OMP_NUM_THREADS=1 python3 scripts/descent_specificity_ensemble.py          (full run)
      OMP_NUM_THREADS=1 python3 scripts/descent_specificity_ensemble.py --check-published
      python3 scripts/descent_specificity_ensemble.py --analyse     (tables + figure)

Runtime on this machine, shared with three other jobs: --scan 36 min, --run 17 min (4 workers),
--check-published 4 min. Outputs: figures/descent_specificity_ensemble.npz,
figures/descent_specificity_ensemble_pubcheck.npz, figures/fig_descent_specificity_ensemble.png.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys                                                                    # noqa: E402
import time                                                                   # noqa: E402
from concurrent.futures import ProcessPoolExecutor                           # noqa: E402
from pathlib import Path                                                      # noqa: E402

import numpy as np                                                            # noqa: E402
from scipy.optimize import minimize                                           # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
os.chdir(REPO)

from constrained_vs_free_glv import (load_communities, transitions, fit_free,  # noqa: E402
                                     pred_mse, S, EPS, ABMIN, LAM)
from inference_aware_null import CAP                                          # noqa: E402
from inference_aware_null_compositional import simulate_compositional         # noqa: E402
from power_analysis import antisym, base_design, _PAIRS, _PK                  # noqa: E402
from descent_specificity import (fit_constrained_fast_d,                      # noqa: E402
                                 loco_descent_stat)
from symmetrizability_invariants import invariants as inv3                    # noqa: E402
from detection_power import perm_p                                            # noqa: E402

import warnings                                                               # noqa: E402
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = "figures"
NPZ = os.path.join(OUT, "descent_specificity_ensemble.npz")
FIG = os.path.join(OUT, "fig_descent_specificity_ensemble.png")
ALPHA = 0.05
NPERM = 400
WORKERS = 6
IU = np.triu_indices(S, 1)
PLATEAU = np.array([False, True, False, True, False, True])   # k = 1, 3, 5 of six transitions


# =========================================================================== geometry
def real_states(comms):
    tr = transitions(comms)
    A = np.array([a for a, g, m in tr])
    Mk = np.array([m for a, g, m in tr])
    return A, Mk


def _fit_sym(A, G, Mk, lam=1e-4, maxiter=300, u0=None):
    """Least-squares weighted-gradient (symmetrizable) fit to a given growth field G on the
    states A, equations masked by Mk. Same parametrisation and bounds as the fitter."""
    trans = [(A[t], G[t], Mk[t]) for t in range(len(A))]
    R, K0, y, esp = base_design(trans)
    npar = S + len(_PAIRS)

    def solve(u):
        d = np.exp(np.clip(u - u.mean(), -6.0, 6.0))
        Mm = np.hstack([R, K0 / d[esp][:, None]])
        reg = np.zeros(npar)
        for (i, j) in _PAIRS:
            if i != j:
                reg[S + _PK[(i, j)]] = lam * (1.0 / d[i] ** 2 + 1.0 / d[j] ** 2)
        beta = np.linalg.solve(Mm.T @ Mm + np.diag(reg) + 1e-9 * np.eye(npar), Mm.T @ y)
        return beta, d, Mm

    def loss(u):
        beta, _, Mm = solve(u)
        return float(np.mean((y - Mm @ beta) ** 2))

    best = None
    for start in ([np.zeros(S)] + ([u0] if u0 is not None else [])):
        res = minimize(loss, start, method="L-BFGS-B", bounds=[(-6, 6)] * S,
                       options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    beta, d, _ = solve(best.x)
    K = np.zeros((S, S))
    for (i, j) in _PAIRS:
        K[i, j] = K[j, i] = beta[S + _PK[(i, j)]]
    return beta[:S], K / d[:, None], d


class Geometry:
    """Distances of a ground truth (r, J) from J0 and from the symmetrizable manifold."""

    def __init__(self, comms, r0, J0, d0):
        self.A, self.Mk = real_states(comms)
        self.r0, self.J0 = r0, J0
        self.u0 = np.log(d0) - np.log(d0).mean()
        self.scale = np.sqrt(np.mean((self.A @ J0.T)[self.Mk] ** 2))
        off = ~np.eye(S, dtype=bool)
        self.off = off
        self.nJ0 = np.linalg.norm(J0[off])

    def field(self, r, J):
        return r[None, :] + self.A @ J.T

    def d_vf(self, r, J):
        dif = (self.field(r, J) - self.field(self.r0, self.J0))[self.Mk]
        return float(np.sqrt(np.mean(dif ** 2)) / self.scale)

    def d_M(self, r, J, maxiter=300):
        G = self.field(r, J)
        rs, Js, _ = _fit_sym(self.A, G, self.Mk, maxiter=maxiter, u0=self.u0)
        dif = (G - self.field(rs, Js))[self.Mk]
        return float(np.sqrt(np.mean(dif ** 2)) / self.scale)

    def d_F(self, J):
        return float(np.linalg.norm((J - self.J0)[self.off]) / self.nJ0)

    def describe(self, r, J):
        f1, fs, rc, _ = inv3(J)
        return dict(dM=self.d_M(r, J), dVF=self.d_vf(r, J), dF=self.d_F(J),
                    f_one=f1, f_sign=fs, r_cycle=rc)


# =========================================================================== ground truths
def gt_antisym(J0, seed, alpha):
    P = antisym(J0, np.random.default_rng(seed))
    return J0 + alpha * P


def _pair_order(seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(IU[0]))
    side = rng.integers(0, 2, len(order))        # which entry of the pair is altered
    return order, side


def gt_oneway(J0, seed, L):
    J = J0.copy()
    order, side = _pair_order(1000 + seed)
    for k in order[:L]:
        i, j = IU[0][k], IU[1][k]
        if side[k]:
            J[i, j] = 0.0
        else:
            J[j, i] = 0.0
    return J


def gt_sign(J0, seed, L):
    J = J0.copy()
    order, side = _pair_order(2000 + seed)
    for k in order[:L]:
        i, j = IU[0][k], IU[1][k]
        if side[k]:
            J[i, j] = -J[i, j]
        else:
            J[j, i] = -J[j, i]
    return J


def cycle_field(seed):
    """Antisymmetric c on all pairs with its gradient part removed: pure curl."""
    rng = np.random.default_rng(3000 + seed)
    z = rng.standard_normal(len(IU[0]))
    B = np.zeros((len(z), S))
    B[np.arange(len(z)), IU[1]] = 1.0
    B[np.arange(len(z)), IU[0]] = -1.0
    u, *_ = np.linalg.lstsq(B, z, rcond=None)
    c = z - B @ u
    return c / np.sqrt(np.mean(c ** 2))


def gt_cycle(J0, seed, kappa):
    c = cycle_field(seed)
    J = J0.copy()
    J[IU] = J0[IU] * np.exp(+kappa * c / 2.0)
    J[IU[1], IU[0]] = J0[IU[1], IU[0]] * np.exp(-kappa * c / 2.0)
    return J


def gt_free(r0, J0, rf, Jf, t):
    return r0 + t * (rf - r0), J0 + t * (Jf - J0)


def bisect_param(fn, target, lo, hi, it=22):
    """Smallest parameter in [lo, hi] at which the monotone-ish fn reaches target."""
    flo, fhi = fn(lo), fn(hi)
    if fhi < target:
        return hi, fhi
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if fm < target:
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return (hi, fhi) if abs(fhi - target) <= abs(flo - target) else (lo, flo)


# =========================================================================== simulation
def simulate_sep(comms, r, J, sp, so, seed):
    """simulate_compositional with observation noise separated from process noise.

    The latent state follows the published compositional simulator exactly (same rng stream,
    so sigma_o = 0 reproduces it bit for bit). The REPORTED composition is the latent one
    times exp(sigma_o * N(0,1)) per species, renormalised: sequencing noise that does not
    propagate. Dropped species are left at their initial trace value, as published.
    """
    rng = np.random.default_rng(seed)
    out = []
    lat = []
    for name, X in comms:
        present = X.mean(0) > 1e-3
        T = len(X)
        traj = np.zeros((T, S))
        x = X[0].copy()
        traj[0] = x
        for t in range(1, T):
            drift = r + J @ x
            x = np.clip(np.exp(np.log(x + EPS) + drift + sp * rng.standard_normal(S)), 0, CAP)
            x[~present] = X[0][~present]
            s = x.sum()
            if s > 0:
                x = x / s
            traj[t] = x
        lat.append((name, traj, present))
    if so > 0:
        rngo = np.random.default_rng(seed + 10_000_019)
        for name, traj, present in lat:
            Y = traj * np.exp(so * rngo.standard_normal(traj.shape))
            Y[:, ~present] = traj[:, ~present]
            Y = Y / Y.sum(1, keepdims=True)
            out.append((name, Y))
    else:
        out = [(name, traj) for name, traj, _ in lat]
    return out


# =========================================================================== scoring
def score_fold(X, r, J, d, nperm, seed, h, exact=False):
    """Descent booleans for the fitted potential and its nperm scrambles on one community.

    Identical to descent_fraction + lyapunov_robustness.scramble with
    rng = default_rng(seed + 7919 h + i), vectorised over the scrambles.
    """
    K = d[:, None] * J
    ku = K[IU]
    kd = np.diag(K)
    lin = X @ (d * r)
    qd = (X ** 2) @ kd
    Q = X[:, IU[0]] * X[:, IU[1]]
    H_obs = -lin - 0.5 * (qd + 2.0 * (Q @ ku))
    if exact:
        Kp = np.empty((nperm, len(ku)))
        for i in range(nperm):
            Kp[i] = np.random.default_rng(seed + 7919 * h + i).permutation(ku)
    else:
        rng = np.random.default_rng([seed, h, 7919])
        Kp = ku[np.argsort(rng.random((nperm, len(ku))), axis=1)]
    H_null = -lin[None, :] - 0.5 * (qd[None, :] + 2.0 * (Kp @ Q.T))
    return np.diff(H_obs) <= 0, np.diff(H_null, axis=1) <= 0


def protocol(comms, nperm=NPERM, seed=0, predictive=True, exact=False):
    """Held-out descent test and predictive comparison on one (simulated) data set."""
    dob, dnu, adv = [], [], []
    for h in range(len(comms)):
        train = transitions([c for k, c in enumerate(comms) if k != h])
        r, J, d = fit_constrained_fast_d(train)
        X = comms[h][1]
        o, n = score_fold(X, r, J, d, nperm, seed, h, exact)
        dob.append(o)
        dnu.append(n)
        if predictive:
            te = transitions([comms[h]])
            rf, Jf = fit_free(train)
            adv.append(pred_mse(r, J, te) - pred_mse(rf, Jf, te))
    dob = np.concatenate(dob)                      # (78,)
    dnu = np.concatenate(dnu, axis=1)              # (nperm, 78)
    plat = np.tile(PLATEAU, len(comms))

    def stat(sel):
        f = dob[sel].mean()
        nl = dnu[:, sel].mean(axis=1)
        return f, float(nl.mean()), float((np.sum(nl >= f) + 1) / (nperm + 1))

    f_all, n_all, p_all = stat(np.ones(len(dob), bool))
    f_pl, n_pl, p_pl = stat(plat)
    adv = np.asarray(adv)
    out = [f_all, n_all, p_all, f_pl, n_pl, p_pl]
    if predictive:
        out += [float(adv.mean()), perm_p(adv)]
    else:
        out += [np.nan, np.nan]
    return np.array(out)


# --------------------------------------------------------------------------- workers
_COMMS = None


def _init(comms):
    global _COMMS
    _COMMS = comms
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")


def _job(args):
    r, J, sp, so, seed = args
    sim = simulate_sep(_COMMS, r, J, sp, so, seed)
    return protocol(sim, seed=seed)


def run_cell(pool, r, J, sp, so, base, n):
    jobs = [(r, J, sp, so, base + 31 * k) for k in range(n)]
    return np.array(list(pool.map(_job, jobs, chunksize=2)))


def roughness(comms):
    tr = transitions(comms)
    g = np.array([g for a, g, m in tr])
    m = np.array([m for a, g, m in tr])
    return float(np.sqrt(np.mean(g[m] ** 2)))


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


# =========================================================================== noise estimate
def noise_estimate(comms, r0, J0):
    """sigma_o from the lag-1 residual autocovariance (MA(1)), sigma_p from the remainder."""
    series = {}
    for ci, (_, X) in enumerate(comms):
        for t, (a, b) in enumerate(zip(X[:-1], X[1:])):
            g = np.log(b + EPS) - np.log(a + EPS)
            m = (a > ABMIN) & (b > ABMIN)
            e = g - (r0 + J0 @ a)
            for i in np.where(m)[0]:
                series.setdefault((ci, i), {})[t] = e[i]
    e = np.array([v for s in series.values() for v in s.values()])
    pr = np.array([s[t] * s[t + 1] for s in series.values() for t in s if t + 1 in s])
    var, c1 = float(np.mean(e ** 2)), float(np.mean(pr))
    so = float(np.sqrt(max(0.0, -c1)))
    sp = float(np.sqrt(max(0.0, var - 2 * so ** 2)))
    return dict(resid_sd=float(np.sqrt(var)), lag1=c1, sigma_o=so, sigma_p=sp, npairs=len(pr))


# =========================================================================== scan
def setup():
    comms = load_communities()
    tr = transitions(comms)
    r0, J0, d0 = fit_constrained_fast_d(tr)
    rf, Jf = fit_free(tr)
    return comms, r0, J0, d0, rf, Jf


def scan():
    t0 = time.time()
    comms, r0, J0, d0, rf, Jf = setup()
    geo = Geometry(comms, r0, J0, d0)

    # ---- anchors: vectorised scoring == published loop; simulator == published at sigma_o=0
    pub = loco_descent_stat(comms, fit_constrained_fast_d, nperm=NPERM, seed=11)
    mine = protocol(comms, seed=11, predictive=False, exact=True)
    print(f"anchor, real data: published loop {pub[0]:.4f} / {pub[1]:.4f} / p={pub[2]:.4f}"
          f"   vectorised {mine[0]:.4f} / {mine[1]:.4f} / p={mine[2]:.4f}")
    assert abs(pub[0] - mine[0]) < 1e-12 and abs(pub[1] - mine[1]) < 1e-12
    a = simulate_compositional(comms, r0, J0, 0.15, np.random.default_rng(5))
    b = simulate_sep(comms, r0, J0, 0.15, 0.0, 5)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    print("anchor: simulate_sep(sigma_o=0) reproduces simulate_compositional exactly\n")

    # ---- geometry
    print(f"J0 on the manifold: {geo.describe(r0, J0)}")
    for dl in [1, 2, 4, 8]:
        J = gt_antisym(J0, 0, dl)
        print(f"published P, delta={dl}: {fmt(geo.describe(r0, J))}")
    for t in [0.5, 1, 2, 4, 8]:
        r, J = gt_free(r0, J0, rf, Jf, t)
        print(f"free path t={t}: {fmt(geo.describe(r, J))}")
    for L in [5, 10, 20, 40, 66]:
        print(f"oneway L={L}: {fmt(geo.describe(r0, gt_oneway(J0, 0, L)))}")
    for L in [5, 10, 20, 40, 66]:
        print(f"sign   L={L}: {fmt(geo.describe(r0, gt_sign(J0, 0, L)))}")
    for k in [0.25, 0.5, 1, 2, 4]:
        print(f"cycle  k={k}: {fmt(geo.describe(r0, gt_cycle(J0, 0, k)))}")

    # ---- independent noise estimate
    ne = noise_estimate(comms, r0, J0)
    print(f"\nnoise from residuals: {ne}")
    print(f"real roughness RMS(g) = {roughness(comms):.3f}\n")

    # ---- quick grid at delta = 0: roughness of simulated data, descent, margin, rejection
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init,
                             initargs=(comms,)) as pool:
        print("published calibration grid (sigma_o = 0), 60 reps each:")
        for sp in [0.05, 0.10, 0.15, 0.20, 0.30]:
            R = run_cell(pool, r0, J0, sp, 0.0, 900000, 60)
            rough = np.mean([roughness(simulate_sep(comms, r0, J0, sp, 0.0, 900000 + 31 * k))
                             for k in range(20)])
            print(f"   sp={sp:.2f}: rough={rough:.3f} descent={R[:,0].mean():.3f} "
                  f"null={R[:,1].mean():.3f} margin={np.mean(R[:,0]-R[:,1]):+.3f} "
                  f"reject={np.mean(R[:,2] < ALPHA):.2f}")
        print("3x3 grid, 40 reps each:")
        for sp in [0.10, 0.20, 0.42]:
            for so in [0.0, 0.12, 0.25]:
                R = run_cell(pool, r0, J0, sp, so, 950000, 40)
                rough = np.mean([roughness(simulate_sep(comms, r0, J0, sp, so,
                                                        950000 + 31 * k)) for k in range(20)])
                print(f"   sp={sp:.2f} so={so:.2f}: rough={rough:.3f} "
                      f"descent={R[:,0].mean():.3f} null={R[:,1].mean():.3f} "
                      f"margin={np.mean(R[:,0]-R[:,1]):+.3f} reject={np.mean(R[:,2]<ALPHA):.2f}"
                      f"  plateau margin={np.mean(R[:,3]-R[:,4]):+.3f}"
                      f"  pred size={np.mean(R[:,7] <= ALPHA):.2f}")
    print(f"\nscan done in {time.time()-t0:.0f} s")


def fmt(dct):
    return " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in dct.items())


if __name__ == "__main__" and "--scan" in sys.argv:
    scan()


# =========================================================================== full run
TARGETS = [0.04, 0.08, 0.16, 0.32]          # delta_VF; = published delta 1, 2, 4, 8 on seed 0
FREE_T = [1.0, 3.0, 6.0, 12.0]
SETTINGS = [(0.10, 0.12), (0.20, 0.12), (0.20, 0.00)]   # (sigma_p, sigma_o); last = published
GRID_P = [0.10, 0.20, 0.42]
GRID_O = [0.00, 0.12, 0.25]
FAMS = ["antisym", "oneway", "sign", "cycle", "free"]
NSEED = {"antisym": 8, "oneway": 6, "sign": 6, "cycle": 6}
REP_GT = 12          # replicates per (ground truth, setting)
REP_FREE = 60        # per free-path point and setting
REP_BASE = 150       # J0 itself, per setting
REP_GRID = 60        # added per 3x3 cell (J0 and free fit); the scan's 40 are not reused


def greedy_subset(geo, r0, J0, maker_one, seed, target):
    """Add altered pairs in a random order, skipping any that overshoot, until delta_VF
    reaches the target. maker_one(J, i, j, side) alters one pair in place."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(IU[0]))
    side = rng.integers(0, 2, len(order))
    J = J0.copy()
    cur = 0.0
    for k in order:
        Jt = J.copy()
        maker_one(Jt, IU[0][k], IU[1][k], side[k])
        v = geo.d_vf(r0, Jt)
        if v <= 1.15 * target:
            J, cur = Jt, v
        if cur >= 0.9 * target:
            break
    return J, cur


def _oneway_one(J, i, j, side):
    if side:
        J[i, j] = 0.0
    else:
        J[j, i] = 0.0


def _sign_one(J, i, j, side):
    if side:
        J[i, j] = -J[i, j]
    else:
        J[j, i] = -J[j, i]


def build_truths(geo, r0, J0, rf, Jf):
    """List of dicts: family, seed, level index, r, J, and the achieved delta_VF."""
    out = [dict(fam="base", seed=0, lev=-1, r=r0, J=J0.copy(), dVF=0.0)]
    for s in range(NSEED["antisym"]):
        P = antisym(J0, np.random.default_rng(s))
        unit = geo.d_vf(r0, J0 + P)
        for li, tg in enumerate(TARGETS):
            J = J0 + (tg / unit) * P
            out.append(dict(fam="antisym", seed=s, lev=li, r=r0, J=J, dVF=geo.d_vf(r0, J)))
    for fam, one, off in (("oneway", _oneway_one, 1000), ("sign", _sign_one, 2000)):
        for s in range(NSEED[fam]):
            for li, tg in enumerate(TARGETS):
                J, v = greedy_subset(geo, r0, J0, one, off + 37 * s + li, tg)
                out.append(dict(fam=fam, seed=s, lev=li, r=r0, J=J, dVF=v))
    for s in range(NSEED["cycle"]):
        for li, tg in enumerate(TARGETS):
            kap, v = bisect_param(lambda k: geo.d_vf(r0, gt_cycle(J0, s, k)), tg, 0.0, 6.0)
            out.append(dict(fam="cycle", seed=s, lev=li, r=r0, J=gt_cycle(J0, s, kap), dVF=v,
                            kappa=kap))
    for li, t in enumerate(FREE_T):
        r, J = gt_free(r0, J0, rf, Jf, t)
        out.append(dict(fam="free", seed=0, lev=li, r=r, J=J, dVF=geo.d_vf(r, J), t=t))
    return out


_GEO = None


def _init_geo(comms, r0, J0, d0):
    global _GEO
    _init(comms)
    globals()["_GEO"] = Geometry(comms, r0, J0, d0)


def _describe_job(args):
    r, J = args
    f1, fs, rc, _ = inv3(J)
    return (_GEO.d_M(r, J, maxiter=120), _GEO.d_F(J), f1, fs, rc)


def cell_seed(si, fam, seed, lev):
    return 1_000_000 * (FAMS.index(fam) + 1 if fam in FAMS else 0) + 10_000 * seed + \
        1_000 * (lev + 1)


def full_run(workers=4):
    t0 = time.time()
    comms, r0, J0, d0, rf, Jf = setup()
    geo = Geometry(comms, r0, J0, d0)
    truths = build_truths(geo, r0, J0, rf, Jf)
    print(f"{len(truths)} ground truths built in {time.time()-t0:.0f} s", flush=True)

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_geo,
                             initargs=(comms, r0, J0, d0)) as pool:
        desc = list(pool.map(_describe_job, [(g["r"], g["J"]) for g in truths]))
        for g, (dm, dF, f1, fs, rc) in zip(truths, desc):
            g.update(dM=dm, dF=dF, f_one=f1, f_sign=fs, r_cycle=rc)
        print(f"geometry done at {time.time()-t0:.0f} s", flush=True)

        # ---- family design at three noise settings (common random numbers across settings)
        rows = []          # (setting index, truth index, replicate results)
        for si, (sp, so) in enumerate(SETTINGS):
            jobs, idx = [], []
            for ti, g in enumerate(truths):
                n = REP_BASE if g["fam"] == "base" else (REP_FREE if g["fam"] == "free"
                                                         else REP_GT)
                base = cell_seed(si, g["fam"], g["seed"], g["lev"])
                for k in range(n):
                    jobs.append((g["r"], g["J"], sp, so, base + 31 * k))
                    idx.append(ti)
            res = list(pool.map(_job, jobs, chunksize=4))
            for ti, rr in zip(idx, res):
                rows.append((si, ti, rr))
            print(f"setting {si} (sp={sp}, so={so}): {len(jobs)} replicates, "
                  f"{time.time()-t0:.0f} s", flush=True)

        # ---- 3x3 noise grid at J0 and at the free fit (t = 1)
        free1 = [ti for ti, g in enumerate(truths) if g["fam"] == "free" and g["t"] == 1.0][0]
        grid = []
        for sp in GRID_P:
            for so in GRID_O:
                for ti in (0, free1):
                    g = truths[ti]
                    base = 50_000_000 + cell_seed(0, g["fam"], g["seed"], g["lev"])
                    jobs = [(g["r"], g["J"], sp, so, base + 31 * k) for k in range(REP_GRID)]
                    res = np.array(list(pool.map(_job, jobs, chunksize=4)))
                    grid.append((sp, so, ti, res))
            print(f"grid row sp={sp} done, {time.time()-t0:.0f} s", flush=True)

    # ---- save
    meta = np.array([[FAMS.index(g["fam"]) if g["fam"] in FAMS else -1, g["seed"], g["lev"],
                      g["dVF"], g["dM"], g["dF"], g["f_one"], g["f_sign"],
                      g["r_cycle"] if np.isfinite(g["r_cycle"]) else np.nan]
                     for g in truths])
    R = np.array([np.concatenate([[si, ti], rr]) for si, ti, rr in rows])
    G = np.array([np.concatenate([[sp, so, ti], r_]) for sp, so, ti, res in grid for r_ in res])
    np.savez(NPZ, meta=meta, rows=R, grid=G, settings=np.array(SETTINGS),
             targets=np.array(TARGETS), free_t=np.array(FREE_T),
             fams=np.array(FAMS), noise=np.array(list(noise_estimate(comms, r0, J0).values())))
    print(f"saved {NPZ} at {time.time()-t0:.0f} s; run --analyse for tables and figure",
          flush=True)


if __name__ == "__main__" and "--run" in sys.argv:
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    full_run(w)


# =========================================================================== analysis
FCOL = {"antisym": "#0072B2", "oneway": "#E69F00", "sign": "#009E73", "cycle": "#CC79A7",
        "free": "#D55E00"}
FLAB = {"antisym": "a  random antisymmetric (8 directions)",
        "oneway": "b  one-way pairs, condition (i)",
        "sign": "c  discordant signs, condition (ii)",
        "cycle": "d  cycle inconsistency, condition (iii)",
        "free": "e  toward the free fit (t = 1, 3, 6, 12)"}
OBS_MARGIN, OBS_MARGIN_PL = 0.100, 0.149


def _rate(rej):
    k, n = int(np.sum(rej)), len(rej)
    lo, hi = wilson(k, n)
    return k / max(n, 1), lo, hi, n


def _calib(pv):
    """Largest p threshold whose rejection rate on the delta = 0 replicates is <= alpha."""
    c = [x for x in np.sort(np.unique(pv)) if np.mean(pv <= x) <= ALPHA]
    return max(c) if c else 0.0


def analyse():
    D = np.load(NPZ, allow_pickle=True)
    meta, R, G = D["meta"], D["rows"], D["grid"]
    settings = [tuple(s) for s in D["settings"]]
    fams = list(D["fams"])
    noise = D["noise"]
    print(f"noise from residuals: resid sd {noise[0]:.3f}, lag-1 cov {noise[1]:+.4f}, "
          f"sigma_o {noise[2]:.3f}, sigma_p <= {noise[3]:.3f} ({int(noise[4])} pairs)\n")
    fam_of = {ti: (fams[int(m[0])] if m[0] >= 0 else "base") for ti, m in enumerate(meta)}
    out = {"settings": settings, "fam": {}, "base": {}}

    # ------------------------------------------------------------ table A: delta = 0
    print("A. symmetrizable ground truth (J0): sensitivity, simulated margin, predictive size")
    print(f"{'sigma_p':>7} {'sigma_o':>7} | {'descent reject':>22} | {'margin mean+-sd':>16} "
          f"{'max':>6} {'P>=.100':>7} | {'plateau':>7} {'P>=.149':>7} | {'pred size':>20} {'p*':>7}")
    for si, (sp, so) in enumerate(settings):
        X = R[(R[:, 0] == si) & (R[:, 1] == 0)][:, 2:]
        # pool the 3x3-grid replicates of the same cell for the predictive calibration
        Gx = G[(np.isclose(G[:, 0], sp)) & (np.isclose(G[:, 1], so)) & (G[:, 2] == 0)][:, 3:]
        pool_p = np.concatenate([X[:, 7], Gx[:, 7]]) if len(Gx) else X[:, 7]
        rej = _rate(X[:, 2] < ALPHA)
        m, mp = X[:, 0] - X[:, 1], X[:, 3] - X[:, 4]
        sz = _rate(pool_p <= ALPHA)
        ps = _calib(pool_p)
        out["base"][si] = dict(rej=rej, m=m, mp=mp, size=sz, p_star=ps)
        print(f"{sp:>7.2f} {so:>7.2f} | {rej[0]:>6.2f} [{rej[1]:.2f},{rej[2]:.2f}] n={rej[3]:<4}| "
              f"{m.mean():+.3f}+-{m.std():.3f} {m.max():+.3f} {np.mean(m >= OBS_MARGIN):>7.3f} | "
              f"{mp.mean():+.3f} {np.mean(mp >= OBS_MARGIN_PL):>7.3f} | "
              f"{sz[0]:>6.2f} [{sz[1]:.2f},{sz[2]:.2f}] {ps:>7.4f}")

    # ------------------------------------------------------------ table B: families
    print("\nB. departures from the manifold: descent-test rejection (supports a gradient) and "
          "size-calibrated predictive rejection (detects asymmetry)")
    for si, (sp, so) in enumerate(settings):
        print(f"\n  setting sigma_p={sp:.2f} sigma_o={so:.2f}; "
              f"delta=0 sensitivity {out['base'][si]['rej'][0]:.2f}")
        print(f"  {'family':<8} {'lev':>3} {'dVF':>6} {'dM':>6} {'dF':>6} {'f_one':>6} "
              f"{'f_sign':>6} {'r_cyc':>6} | {'descent reject':>24} | {'pred. calibrated':>22}")
        ps = out["base"][si]["p_star"]
        for fam in fams:
            tis = [ti for ti in fam_of if fam_of[ti] == fam]
            levs = sorted({int(meta[ti][2]) for ti in tis})
            for lv in levs:
                tl = [ti for ti in tis if int(meta[ti][2]) == lv]
                X = R[(R[:, 0] == si) & np.isin(R[:, 1], tl)][:, 2:]
                mm = meta[tl]
                rej = _rate(X[:, 2] < ALPHA)
                prj = _rate(X[:, 7] <= ps)
                out["fam"][(si, fam, lv)] = dict(dVF=mm[:, 3].mean(), dM=mm[:, 4].mean(),
                                                 dF=mm[:, 5].mean(), rej=rej, pred=prj,
                                                 dVF_all=mm[:, 3], dM_all=mm[:, 4])
                print(f"  {fam:<8} {lv:>3} {mm[:,3].mean():>6.3f} {mm[:,4].mean():>6.3f} "
                      f"{mm[:,5].mean():>6.3f} {np.nanmean(mm[:,6]):>6.2f} "
                      f"{np.nanmean(mm[:,7]):>6.2f} {np.nanmean(mm[:,8]):>6.2f} | "
                      f"{rej[0]:>6.2f} [{rej[1]:.2f},{rej[2]:.2f}] n={rej[3]:<4} | "
                      f"{prj[0]:>6.2f} [{prj[1]:.2f},{prj[2]:.2f}]")

    # ------------------------------------------------------------ table C: 3x3 grid
    print("\nC. noise grid (60 replicates per cell): descent rejection at J0 (sensitivity) and "
          "at the free fit; predictive size at J0")
    sp_l, so_l = sorted(set(G[:, 0])), sorted(set(G[:, 1]))
    free1 = [ti for ti in fam_of if fam_of[ti] == "free" and int(meta[ti][2]) == 0][0]
    gsens = np.zeros((len(sp_l), len(so_l), 3))
    gfree = np.zeros((len(sp_l), len(so_l), 3))
    gmarg = np.zeros((len(sp_l), len(so_l)))
    for a, sp in enumerate(sp_l):
        for b, so in enumerate(so_l):
            X0 = G[(G[:, 0] == sp) & (G[:, 1] == so) & (G[:, 2] == 0)][:, 3:]
            X1 = G[(G[:, 0] == sp) & (G[:, 1] == so) & (G[:, 2] == free1)][:, 3:]
            r0_, r1_, sz = _rate(X0[:, 2] < ALPHA), _rate(X1[:, 2] < ALPHA), \
                _rate(X0[:, 7] <= ALPHA)
            gsens[a, b] = r0_[:3]
            gfree[a, b] = r1_[:3]
            gmarg[a, b] = np.mean(X0[:, 0] - X0[:, 1])
            print(f"  sp={sp:.2f} so={so:.2f}: J0 {r0_[0]:.2f} [{r0_[1]:.2f},{r0_[2]:.2f}]   "
                  f"free fit {r1_[0]:.2f} [{r1_[1]:.2f},{r1_[2]:.2f}]   margin@J0 "
                  f"{gmarg[a,b]:+.3f}   pred size {sz[0]:.2f} [{sz[1]:.2f},{sz[2]:.2f}]")
    out.update(gsens=gsens, gfree=gfree, gmarg=gmarg, sp_l=sp_l, so_l=so_l)

    # ------------------------------------------------------------ table D: what predicts it
    print("\nD. which distance organises the rejection? Spearman rho between per-truth descent "
          "rejection and each measure, over all departure truths")
    from scipy.stats import spearmanr
    for si, (sp, so) in enumerate(settings):
        rows = []
        for ti in fam_of:
            if fam_of[ti] == "base":
                continue
            X = R[(R[:, 0] == si) & (R[:, 1] == ti)][:, 2:]
            rows.append([np.mean(X[:, 2] < ALPHA)] + list(meta[ti][3:9]))
        rows = np.array(rows)
        txt = []
        for j, nm in enumerate(["dVF", "dM", "dF", "f_one", "f_sign", "r_cycle"]):
            ok = np.isfinite(rows[:, j + 1])
            rho = spearmanr(rows[ok, 0], rows[ok, j + 1]).correlation
            txt.append(f"{nm} {rho:+.2f}")
        print(f"  sp={sp:.2f} so={so:.2f}: " + "   ".join(txt))
    make_figure(out)
    return out


def _load_pubcheck():
    f = NPZ.replace(".npz", "_pubcheck.npz")
    return np.load(f) if os.path.exists(f) else None


def _overlay_pubcheck(ax):
    """Published direction reproduced (delta 0, 4, 8) and the 8 directions at delta_VF 0.314."""
    pc = _load_pubcheck()
    if pc is None:
        return []
    xs = [0.02, 0.157, 0.314]
    ys = [np.mean(R[:, 2] < ALPHA) for R in pc["pub"]]
    ax.plot(xs, ys, "k:", lw=1.5)
    h1, = ax.plot(xs, ys, "ko", mfc="white", ms=8, mew=1.6, zorder=6)
    rng = np.random.default_rng(3)
    h2 = None
    for k, R in enumerate(pc["dirs"]):
        n = len(R)
        kk = int(np.sum(R[:, 2] < ALPHA))
        lo, hi = wilson(kk, n)
        x = 0.314 * np.exp(rng.uniform(0.07, 0.22))
        h2 = ax.errorbar([x], [kk / n], yerr=[[kk / n - lo], [hi - kk / n]], fmt="D",
                         color="#0072B2", mfc="#56B4E9", ms=6, capsize=2, zorder=6)
    ax.legend([h1, h2], ["published direction, reproduced\n(published seeds, 100 replicates)",
                         "8 random directions at the same\ndisplacement, 60 replicates each"],
              frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0.0, 0.07))
    return [h1, h2]


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings = out["settings"]
    fig = plt.figure(figsize=(16.5, 10.6))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.27, top=0.88, bottom=0.07,
                          left=0.06, right=0.98)
    titles = ["$\\sigma_p=0.10$, $\\sigma_o=0.12$", "$\\sigma_p=0.20$, $\\sigma_o=0.12$",
              "$\\sigma_p=0.20$, $\\sigma_o=0$ (published setting)"]
    x0 = 0.02
    ticks = [0.02, 0.04, 0.08, 0.16, 0.32]
    fam_handles = []
    for si in range(len(settings)):
        ax = fig.add_subplot(gs[0, si])
        b = out["base"][si]
        hb = ax.errorbar([x0], [b["rej"][0]], yerr=[[b["rej"][0] - b["rej"][1]],
                                                    [b["rej"][2] - b["rej"][0]]],
                         fmt="ks", ms=8, capsize=3, zorder=5)
        for fam in FCOL:
            keys = sorted(k for k in out["fam"] if k[0] == si and k[1] == fam)
            xs = [out["fam"][k]["dVF"] for k in keys]
            ys = np.array([out["fam"][k]["rej"][:3] for k in keys])
            h = ax.errorbar(xs, ys[:, 0], yerr=[ys[:, 0] - ys[:, 1], ys[:, 2] - ys[:, 0]],
                            fmt="o-", color=FCOL[fam], lw=1.8, ms=5.5, capsize=2.5,
                            alpha=0.95)
            if si == 0:
                fam_handles.append((h, FLAB[fam]))
        if si == 0:
            fam_handles.insert(0, (hb, "symmetrizable truth $J_0$ (150 replicates)"))
        if si == len(settings) - 1:
            _overlay_pubcheck(ax)
        ax.axhline(ALPHA, color="grey", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(0.016, 0.48)
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0", "0.04", "0.08", "0.16", "0.32"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.04)
        ax.set_xlabel("displacement of the truth's vector field, $\\delta_{\\rm VF}$",
                      fontsize=12)
        if si == 0:
            ax.set_ylabel("P(descent test rejects the scrambled null)", fontsize=12)
        ax.set_title(f"{'abc'[si]}   {titles[si]}", loc="left", fontsize=12)
        ax.tick_params(labelsize=10.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.legend([h for h, _ in fam_handles], [l for _, l in fam_handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False, fontsize=10.5)

    # ---- grid heatmaps
    for j, (key, ttl) in enumerate([("gsens", "d   rejection, symmetrizable truth $J_0$"),
                                    ("gfree", "e   rejection, free-fit truth ($t=1$)")]):
        ax = fig.add_subplot(gs[1, j])
        M = out[key][:, :, 0]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues", origin="lower")
        for a in range(M.shape[0]):
            for c in range(M.shape[1]):
                ax.text(c, a, f"{M[a,c]:.2f}\n[{out[key][a,c,1]:.2f}, {out[key][a,c,2]:.2f}]",
                        ha="center", va="center", fontsize=10,
                        color="white" if M[a, c] > 0.55 else "black")
        ax.set_xticks(range(len(out["so_l"])))
        ax.set_xticklabels([f"{v:.2f}" for v in out["so_l"]], fontsize=10.5)
        ax.set_yticks(range(len(out["sp_l"])))
        ax.set_yticklabels([f"{v:.2f}" for v in out["sp_l"]], fontsize=10.5)
        ax.set_xlabel("observation noise $\\sigma_o$", fontsize=12)
        ax.set_ylabel("process noise $\\sigma_p$", fontsize=12)
        ax.set_title(ttl, loc="left", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=10)

    # ---- simulated margins vs observed
    ax = fig.add_subplot(gs[1, 2])
    bins = np.linspace(-0.12, 0.22, 35)
    cols = ["#0072B2", "#009E73", "#777777"]
    short = ["$\\sigma_p$ 0.10, $\\sigma_o$ 0.12", "$\\sigma_p$ 0.20, $\\sigma_o$ 0.12",
             "$\\sigma_p$ 0.20, $\\sigma_o$ 0"]
    hs = []
    for si in range(len(settings)):
        b = out["base"][si]
        ax.hist(b["m"], bins=bins, histtype="step", lw=2, color=cols[si])
        hs.append(plt.Line2D([], [], color=cols[si], lw=2))
    ymax = ax.get_ylim()[1] * 2.0
    ax.set_ylim(0, ymax)
    pc = _load_pubcheck()
    if pc is not None:
        mdir = [float(np.mean(R[:, 0] - R[:, 1])) for R in pc["dirs"]]
        hv, = ax.plot(mdir, [ymax * 0.965] * len(mdir), "v", color="#CC79A7", ms=9)
    ax.axvline(OBS_MARGIN, color="#D55E00", lw=2)
    ax.axvline(OBS_MARGIN_PL, color="#D55E00", lw=2, ls="--")
    ax.text(OBS_MARGIN + 0.003, ymax * 0.55, "observed,\nall: +0.100", color="#D55E00",
            fontsize=10, va="top")
    ax.text(OBS_MARGIN_PL + 0.003, ymax * 0.32, "observed,\nplateau:\n+0.149",
            color="#D55E00", fontsize=10, va="top")
    labels = [f"$J_0$, {s}" for s in short]
    if pc is not None:
        hs.append(hv)
        labels.append("non-symmetrizable truths,\n8 directions at $\\delta_{\\rm VF}=0.31$")
    ax.legend(hs, labels, frameon=False, fontsize=9.5, loc="upper left",
              bbox_to_anchor=(0.0, 0.91))
    ax.set_xlim(-0.12, 0.22)
    ax.set_xlabel("margin: fitted minus scrambled descent fraction", fontsize=12)
    ax.set_ylabel("replicates", fontsize=12)
    ax.set_title("f   simulated margins and the observed one", loc="left", fontsize=12)
    ax.tick_params(labelsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(FIG, dpi=200)
    print(f"\n[fig] {FIG}")


if __name__ == "__main__" and ("--analyse" in sys.argv or "--replot" in sys.argv):
    analyse()


# =========================================================================== published check
def _job_exact(args):
    """The published replicate exactly: compositional simulator, published scramble seeds."""
    r, J, sp, so, seed = args
    sim = simulate_sep(_COMMS, r, J, sp, so, seed)
    return protocol(sim, seed=seed, exact=True, predictive=False)


def check_published(workers=4, n_pub=100, n_dir=60):
    """(1) Reproduce the published specificity points (direction seed 0, sigma = 0.20, the
    published replicate seeds 500000 + 1000 delta + 31 k) with the published scrambles.
    (2) Score each of the 8 antisymmetric directions at the published largest displacement
    (delta_VF = 0.314, i.e. delta = 8 on direction 0) with n_dir replicates each."""
    comms, r0, J0, d0, rf, Jf = setup()
    geo = Geometry(comms, r0, J0, d0)
    res_pub, res_dir = {}, {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(comms,)) as pool:
        P0 = antisym(J0, np.random.default_rng(0))
        for dl in [0.0, 4.0, 8.0]:
            jobs = [(r0, J0 + dl * P0, 0.20, 0.0, 500000 + int(dl * 1000) + 31 * k)
                    for k in range(n_pub)]
            R = np.array(list(pool.map(_job_exact, jobs, chunksize=2)))
            res_pub[dl] = R
            k = int(np.sum(R[:, 2] < ALPHA))
            lo, hi = wilson(k, n_pub)
            print(f"published direction, delta={dl:g} (dVF {geo.d_vf(r0, J0 + dl*P0):.3f}): "
                  f"reject {k/n_pub:.2f} [{lo:.2f},{hi:.2f}]  descent {R[:,0].mean():.3f} "
                  f"scrambled {R[:,1].mean():.3f}", flush=True)
        for s in range(8):
            Ps = antisym(J0, np.random.default_rng(s))
            J = J0 + (0.314 / geo.d_vf(r0, J0 + Ps)) * Ps
            jobs = [(r0, J, 0.20, 0.0, 600000 + 1000 * s + 31 * k) for k in range(n_dir)]
            R = np.array(list(pool.map(_job_exact, jobs, chunksize=2)))
            res_dir[s] = R
            k = int(np.sum(R[:, 2] < ALPHA))
            lo, hi = wilson(k, n_dir)
            f1, fs, rc, _ = inv3(J)
            print(f"direction {s}: dVF {geo.d_vf(r0, J):.3f} dM {geo.d_M(r0, J, 120):.3f} "
                  f"f_sign {fs:.2f} r_cycle {rc:.2f}: reject {k/n_dir:.2f} [{lo:.2f},{hi:.2f}]"
                  f"  descent {R[:,0].mean():.3f} scrambled {R[:,1].mean():.3f}", flush=True)
    np.savez(NPZ.replace(".npz", "_pubcheck.npz"),
             pub=np.array([res_pub[d] for d in [0.0, 4.0, 8.0]]),
             dirs=np.array([res_dir[s] for s in range(8)]))


if __name__ == "__main__" and "--check-published" in sys.argv:
    check_published()
