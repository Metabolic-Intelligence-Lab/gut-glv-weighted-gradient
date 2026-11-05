"""
The d-optimized descent comparison with an exact solver and enough draws (BIB revision, node B)
==============================================================================================

descent_best_case_D.py asks whether the fitted structure still beats a scrambled one when each
arm is given its OWN weights d, chosen to maximise descent on the twelve training communities.
It answered on 24 scramble draws (p = 0.12, Monte Carlo resolution 1/25), with a heuristic
optimiser (annealed sigmoid multistart) that was run with 400 restarts for the fitted arm and
40 for the scrambled arm. Both limits are removed here.

The optimisation problem
------------------------
H_D is linear in d, so every transition t contributes one homogeneous halfspace
psi_t . d <= 0, and "the d maximising training descent" is a maximum feasible subsystem
problem over the positive orthant. Two facts make it exactly solvable as a small MILP:

  * scale: d -> c d (c > 0) changes no sign, so normalise sum_i d_i = 1;
  * the box u in [-6, 6]^S used by the heuristic (d = exp(u - mean u)) is, up to scale,
    exactly the set {d > 0 : max_i log d_i - min_j log d_j <= 12}, i.e. the linear
    constraints d_i <= e^12 d_j for all i != j.

On the simplex, a_t . d <= max_i a_ti for a_t = psi_t / ||psi_t||, which gives a tight
per-row big-M. With binaries z_t (z_t = 1 lets transition t ascend) and a margin s >= 0:

    minimise    sum_t z_t  -  0.5 s
    subject to  a_t . d + s <= (max(0, max_i a_ti) + 1) z_t      for every mixed-sign row t
                d_i <= e^12 d_j  (i != j),   sum_i d_i = 1,   d >= 0,   0 <= s <= 1.

Since s <= 1 the margin weight 0.5 can never buy a violation: the solution maximises the
descent count EXACTLY over the same feasible set the heuristic searches, and among the
maximisers returns the most robust one (largest uniform normalised slack). This tie-break
is deterministic and fixed before any scoring. Rows that descend for every d > 0 (all
entries <= 0) or ascend for every d > 0 (all >= 0, one > 0) are counted directly.

The comparison (all arms scored on the same 78 held-out transitions)
-------------------------------------------------------------------
  fitted      : constrained (symmetrizable) fit of the fold, d = MILP optimum on its training
                transitions, scored on the held-out community.
  scrambled b : the fold's K = D J with its off-diagonal entries permuted (the multiset of
                off-diagonal values and the diagonal preserved, then symmetrised), mapped back
                with the fitted d; then d re-chosen by the SAME MILP on the training
                transitions and scored on the held-out community. Draw b uses the same
                permutation seed as descent_best_case_D.py (4000 + b), so its first 24 draws
                are the published ones.
  p           : one-sided Monte Carlo permutation p = (b + 1)/(B + 1), b = #{draws whose
                pooled held-out fraction >= the fitted arm's}. The unit of resampling is the
                structure permutation; within a draw every fold uses its own permutation of
                its own K, and the 78 held-out transitions are pooled across folds.

Run:  OMP_NUM_THREADS=1 python3 scripts/descent_best_case_D_fast.py [--draws 1000] [--workers 4]
      --validate-only runs the solver validation and stops.
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
from scipy.optimize import milp, LinearConstraint, Bounds                # noqa: E402
from scipy.sparse import csr_matrix                                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import load_communities, transitions, fit_free, S  # noqa: E402
from lyapunov_descent import fit_symmetrizable                                  # noqa: E402
from descent_best_case_D import psi_matrix, count_descents, max_descent_u, UBOUND  # noqa: E402

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

LOGR = 2.0 * UBOUND           # max log-ratio between any two weights (= 12)
TIME_LIMIT = 120.0            # seconds per MILP; status is recorded


# --------------------------------------------------------------------------- #
def _ratio_rows(S_):
    rows = []
    R = np.exp(LOGR)
    for i in range(S_):
        for j in range(S_):
            if i != j:
                r = np.zeros(S_)
                r[i] = 1.0
                r[j] = -R
                rows.append(r)
    return np.array(rows)


_RATIO = None


def milp_max_descent(PSI):
    """Exact maximum of #{t : psi_t . d <= 0} over the heuristic's feasible set.

    Returns (d normalised to sum 1, count by the published criterion PSI @ d <= 0,
    MILP objective count, margin s, solver status, wall time).
    """
    global _RATIO
    t0 = time.time()
    PSI = np.asarray(PSI, float)
    S_ = PSI.shape[1]
    if _RATIO is None or _RATIO.shape[1] != S_:
        _RATIO = _ratio_rows(S_)
    nrm = np.linalg.norm(PSI, axis=1)
    A = PSI / np.where(nrm > 0, nrm, 1.0)[:, None]
    mx, mn = A.max(axis=1), A.min(axis=1)
    always_down = (nrm == 0) | (mx <= 0)
    always_up = (~always_down) & (mn >= 0) & (mx > 0)
    mixed = ~(always_down | always_up)
    Am = A[mixed]
    m = len(Am)
    # variables: d (S_), s (1), z (m)
    nv = S_ + 1 + m
    c = np.zeros(nv)
    c[S_] = -0.5
    c[S_ + 1:] = 1.0
    integrality = np.zeros(nv)
    integrality[S_ + 1:] = 1
    lb = np.zeros(nv)
    ub = np.ones(nv)
    cons = []
    # descent rows
    if m:
        M = np.maximum(Am.max(axis=1), 0.0) + 1.0
        Ad = np.zeros((m, nv))
        Ad[:, :S_] = Am
        Ad[:, S_] = 1.0
        Ad[np.arange(m), S_ + 1 + np.arange(m)] = -M
        cons.append(LinearConstraint(csr_matrix(Ad), -np.inf, 0.0))
    # ratio rows
    Ar = np.zeros((len(_RATIO), nv))
    Ar[:, :S_] = _RATIO
    cons.append(LinearConstraint(csr_matrix(Ar), -np.inf, 0.0))
    # normalisation
    An = np.zeros((1, nv))
    An[0, :S_] = 1.0
    cons.append(LinearConstraint(An, 1.0, 1.0))
    res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub),
               options={"time_limit": TIME_LIMIT, "disp": False, "mip_rel_gap": 0.0})
    if res.x is None:
        return None, -1, -1, np.nan, res.status, time.time() - t0
    d = np.clip(res.x[:S_], 0.0, None)
    z = np.round(res.x[S_ + 1:])
    s = float(res.x[S_])
    n_obj = int(always_down.sum() + (m - z.sum()))
    # Polish: HiGHS accepts MILP points within a 1e-7 feasibility tolerance, which at weight
    # ratios near e^12 can leave a "satisfied" row slightly ascending. Re-solve the maximum
    # margin LP on the selected subsystem at a 1e-10 tolerance and keep the polished d.
    d_pol, s_pol = _polish(Am[z == 0], S_)
    if d_pol is not None and int(np.sum(PSI @ d_pol <= 0)) >= int(np.sum(PSI @ d <= 0)):
        d, s = d_pol, s_pol
    n_pub = int(np.sum(PSI @ d <= 0))
    return d, n_pub, n_obj, s, res.status, time.time() - t0


def _polish(Asat, S_):
    from scipy.optimize import linprog
    nv = S_ + 1
    c = np.zeros(nv)
    c[S_] = -1.0
    A_ub = [np.hstack([Asat, np.ones((len(Asat), 1))])] if len(Asat) else []
    A_ub.append(np.hstack([_RATIO, np.zeros((len(_RATIO), 1))]))
    A_ub = np.vstack(A_ub)
    b_ub = np.zeros(len(A_ub))
    A_eq = np.zeros((1, nv))
    A_eq[0, :S_] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, 1)] * S_ + [(0, 1)], method="highs",
                  options={"primal_feasibility_tolerance": 1e-10,
                           "dual_feasibility_tolerance": 1e-10})
    if res.status != 0 or res.x is None:
        return None, np.nan
    return np.clip(res.x[:S_], 0.0, None), float(res.x[S_])


def d_to_u(d):
    u = np.log(np.maximum(d, 1e-300))
    return u - u.mean()


# --------------------------------------------------------------------------- #
#  fold fits (computed once) and scrambling
# --------------------------------------------------------------------------- #
def build_folds(comms):
    folds = []
    for h in range(len(comms)):
        train = [c for k, c in enumerate(comms) if k != h]
        rf, Jf = fit_free(transitions(train))
        rc, Jc, dc = fit_symmetrizable(transitions(train))
        folds.append(dict(train=train, test=comms[h], rf=rf, Jf=Jf, rc=rc, Jc=Jc, dc=dc))
    return folds


def scrambled_J(f, rng_b):
    iu = np.triu_indices(S, 1)
    K = f["dc"][:, None] * f["Jc"]
    Ks = np.zeros((S, S))
    Ks[iu] = rng_b.permutation(K[iu])
    Ks = Ks + Ks.T
    np.fill_diagonal(Ks, np.diag(K))
    return Ks / f["dc"][:, None]


_FOLDS = None


def _init(folds):
    global _FOLDS
    _FOLDS = folds


def draw_milp(b):
    """Pooled held-out descent of scrambled draw b with MILP train-optimal d per fold."""
    rng_b = np.random.default_rng(4000 + b)
    down = tot = 0
    worst_status, tmax, smin = 0, 0.0, np.inf
    for f in _FOLDS:
        Js = scrambled_J(f, rng_b)
        pte = psi_matrix([f["test"]], f["rc"], Js)
        ptr = psi_matrix(f["train"], f["rc"], Js)
        d, n_pub, n_obj, s, st, dt = milp_max_descent(ptr)
        worst_status = max(worst_status, st)
        tmax = max(tmax, dt)
        smin = min(smin, s)
        down += int(np.sum(pte @ d <= 0))
        tot += len(pte)
    return b, down / tot, worst_status, tmax, smin


def draw_heuristic_equal(b):
    """The published heuristic with the FITTED arm's effort (400 restarts) on the scrambled
    arm, same permutation seeds: isolates the effect of unequal optimisation effort."""
    rng_b = np.random.default_rng(4000 + b)
    down = tot = 0
    for h, f in enumerate(_FOLDS):
        Js = scrambled_J(f, rng_b)
        pte = psi_matrix([f["test"]], f["rc"], Js)
        ptr = psi_matrix(f["train"], f["rc"], Js)
        u, _ = max_descent_u(ptr, n_restart=400, seed=5000 + 13 * b + h)
        down += count_descents(pte, u)
        tot += len(pte)
    return b, down / tot


def draw_heuristic(b):
    """The published scrambled control for draw b (n_restart = 40, published seeds), plus the
    MILP on the same training matrices: returns per-fold training counts of both."""
    rng_b = np.random.default_rng(4000 + b)
    out = []
    down_h = down_m = tot = 0
    for h, f in enumerate(_FOLDS):
        Js = scrambled_J(f, rng_b)
        pte = psi_matrix([f["test"]], f["rc"], Js)
        ptr = psi_matrix(f["train"], f["rc"], Js)
        u, n_h = max_descent_u(ptr, n_restart=40, seed=5000 + 13 * b + h)
        d, n_m, n_obj, s, st, dt = milp_max_descent(ptr)
        out.append((n_h, n_m, n_obj))
        down_h += count_descents(pte, u)
        down_m += int(np.sum(pte @ d <= 0))
        tot += len(pte)
    return b, out, down_h / tot, down_m / tot


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--validate-draws", type=int, default=24)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--heur-draws", type=int, default=300)
    ap.add_argument("--replot", action="store_true", help="redraw from the saved npz only")
    args = ap.parse_args()
    if args.replot:
        make_figure()
        return

    t_start = time.time()
    comms = load_communities()
    n = len(comms)
    folds = build_folds(comms)
    psi_te_con = [psi_matrix([f["test"]], f["rc"], f["Jc"]) for f in folds]
    psi_tr_con = [psi_matrix(f["train"], f["rc"], f["Jc"]) for f in folds]
    psi_te_free = [psi_matrix([f["test"]], f["rf"], f["Jf"]) for f in folds]
    psi_tr_free = [psi_matrix(f["train"], f["rf"], f["Jf"]) for f in folds]
    ntrans = sum(len(P) for P in psi_te_con)
    print(f"{n} communities, {ntrans} held-out transitions; fits done in "
          f"{time.time() - t_start:.0f} s\n")

    # ------------------------------------------------------------ fixed-d reference
    f_con_fixed = sum(int(np.sum(psi_te_con[h] @ folds[h]["dc"] <= 0)) for h in range(n)) / ntrans
    print(f"fixed-d reference (published Fig 1a): constrained fit, fitted d = {f_con_fixed:.3f}\n")

    # ------------------------------------------------------------ (1) validation, observed
    print("(1) SOLVER VALIDATION on the observed folds: training descent count")
    print("    heuristic = descent_best_case_D.max_descent_u, published settings "
          "(400 restarts, seeds 1000+h / 2000+h)")
    val = []
    held = {"free": [0, 0], "con": [0, 0]}
    for arm, PTR, PTE, seed0 in [("free", psi_tr_free, psi_te_free, 1000),
                                  ("con", psi_tr_con, psi_te_con, 2000)]:
        for h in range(n):
            t0 = time.time()
            u, n_h = max_descent_u(PTR[h], seed=seed0 + h)
            th = time.time() - t0
            d, n_m, n_obj, s, st, tm = milp_max_descent(PTR[h])
            val.append((arm, h, len(PTR[h]), n_h, n_m, n_obj, s, st, th, tm))
            held[arm][0] += count_descents(PTE[h], u)
            held[arm][1] += int(np.sum(PTE[h] @ d <= 0))
    print(f"    {'arm':<5}{'fold':>5}{'m':>5}{'heur':>6}{'MILP':>6}{'obj':>6}{'margin s':>11}"
          f"{'status':>8}{'t_heur':>9}{'t_MILP':>9}")
    for arm, h, m, n_h, n_m, n_obj, s, st, th, tm in val:
        flag = "" if n_m >= n_h else "   <-- MILP BELOW HEURISTIC"
        print(f"    {arm:<5}{h:>5}{m:>5}{n_h:>6}{n_m:>6}{n_obj:>6}{s:>11.2e}{st:>8}"
              f"{th:>8.1f}s{tm:>8.2f}s{flag}")
    worse = sum(1 for v in val if v[4] < v[3])
    better = sum(1 for v in val if v[4] > v[3])
    print(f"    MILP < heuristic on {worse}/{len(val)} folds; MILP > heuristic on "
          f"{better}/{len(val)}; equal on {len(val) - worse - better}")
    print(f"    held-out descent with heuristic d: free {held['free'][0]/ntrans:.3f}, "
          f"constrained {held['con'][0]/ntrans:.3f}  (published 0.910 for constrained)")
    print(f"    held-out descent with MILP d     : free {held['free'][1]/ntrans:.3f}, "
          f"constrained {held['con'][1]/ntrans:.3f}\n")
    f_con_opt = held["con"][1] / ntrans
    f_free_opt = held["free"][1] / ntrans

    # ------------------------------------------------------------ (2) validation, scrambled
    from multiprocessing import Pool
    print(f"(2) SOLVER VALIDATION on the {args.validate_draws} published scramble draws "
          "(heuristic with the published 40 restarts and seeds)")
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(folds,)) as pool:
        vres = pool.map(draw_heuristic, range(args.validate_draws))
    vres.sort()
    pairs = [(nh, nm, no) for _, out, _, _ in vres for (nh, nm, no) in out]
    w = sum(1 for nh, nm, _ in pairs if nm < nh)
    bt = sum(1 for nh, nm, _ in pairs if nm > nh)
    gain = np.array([nm - nh for nh, nm, _ in pairs])
    fh = np.array([r[2] for r in vres])
    fm = np.array([r[3] for r in vres])
    print(f"    {len(pairs)} fold-draws: MILP < heuristic on {w}, > on {bt}, equal on "
          f"{len(pairs) - w - bt}; mean training gain {gain.mean():+.2f} transitions "
          f"(max {gain.max():+d})")
    print(f"    pooled held-out, heuristic d: mean {fh.mean():.3f} [{fh.min():.3f}, {fh.max():.3f}] "
          f"(published 0.867)")
    print(f"    pooled held-out, MILP d     : mean {fm.mean():.3f} [{fm.min():.3f}, {fm.max():.3f}]")
    p_h24 = (np.sum(fh >= held["con"][0] / ntrans) + 1) / (len(fh) + 1)
    print(f"    published-style p on these draws (heuristic both arms): {p_h24:.3f}   "
          f"[{time.time() - t0:.0f} s]\n")

    np.savez("figures/descent_best_case_D_fast_validation.npz",
             val=np.array([(0 if a == 'free' else 1, h, m, nh, nm, no, s, st, th, tm)
                           for a, h, m, nh, nm, no, s, st, th, tm in val]),
             val_scr_pairs=np.array(pairs), val_scr_fh=fh, val_scr_fm=fm,
             f_con_heur=held["con"][0] / ntrans, f_free_heur=held["free"][0] / ntrans,
             f_con_milp=f_con_opt, f_free_milp=f_free_opt, f_con_fixed=f_con_fixed,
             ntrans=ntrans)
    if args.validate_only:
        return

    # ------------------------------------------------------------ (3) the comparison
    print(f"(3) d-OPTIMISED COMPARISON, exact solver on both arms, B = {args.draws} draws")
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(folds,)) as pool:
        res = pool.map(draw_milp, range(args.draws), chunksize=4)
    res.sort()
    fs = np.array([r[1] for r in res])
    status = np.array([r[2] for r in res])
    smin = np.array([r[4] for r in res])
    b = int(np.sum(fs >= f_con_opt))
    p = (b + 1) / (len(fs) + 1)
    margin = f_con_opt - fs.mean()
    print(f"    fitted structure, MILP train-optimal d : {f_con_opt:.3f} "
          f"({int(round(f_con_opt * ntrans))}/{ntrans})")
    print(f"    scrambled, MILP train-optimal d        : mean {fs.mean():.3f}  "
          f"95% [{np.percentile(fs, 2.5):.3f}, {np.percentile(fs, 97.5):.3f}]  "
          f"max {fs.max():.3f}")
    print(f"    margin {margin:+.3f};  draws >= fitted: {b}/{len(fs)};  p = {p:.4f}")
    print(f"    MILP status: {np.sum(status == 0)}/{len(status)} draws all folds optimal; "
          f"draws with a zero-margin optimum in some fold: {np.sum(smin <= 1e-12)}")
    print(f"    [{time.time() - t0:.0f} s]")
    # free structure for the record
    print(f"\n    free structure, MILP train-optimal d (record): {f_free_opt:.3f}")

    # sensitivity of the p to the first-24 subset (the published draws)
    p24 = (np.sum(fs[:24] >= f_con_opt) + 1) / 25
    print(f"    on the first 24 draws only (the published permutations): p = {p24:.3f}")

    # ------------------------------------------------------------ (4) equal-effort heuristic
    fe = np.array([])
    if args.heur_draws > 0:
        print(f"\n(4) EQUAL-EFFORT HEURISTIC: published optimiser with 400 restarts on BOTH arms, "
              f"B = {args.heur_draws} draws")
        t0 = time.time()
        with Pool(args.workers, initializer=_init, initargs=(folds,)) as pool:
            hres = pool.map(draw_heuristic_equal, range(args.heur_draws), chunksize=2)
        hres.sort()
        fe = np.array([r[1] for r in hres])
        f_h = held["con"][0] / ntrans
        bh = int(np.sum(fe >= f_h))
        ph = (bh + 1) / (len(fe) + 1)
        print(f"    fitted structure, heuristic d (400 restarts): {f_h:.3f}")
        print(f"    scrambled, heuristic d (400 restarts)       : mean {fe.mean():.3f}  "
              f"95% [{np.percentile(fe, 2.5):.3f}, {np.percentile(fe, 97.5):.3f}]")
        print(f"    margin {f_h - fe.mean():+.3f};  draws >= fitted: {bh}/{len(fe)};  p = {ph:.4f}"
              f"   [{time.time() - t0:.0f} s]")

    np.savez("figures/descent_best_case_D_fast.npz",
             f_con_fixed=f_con_fixed, f_con_opt=f_con_opt, f_free_opt=f_free_opt,
             scram_opt=fs, status=status, smin=smin, p=p, b=b, ntrans=ntrans,
             f_con_heur=held["con"][0] / ntrans, val_scr_fh=fh, val_scr_fm=fm,
             scram_heur_equal=fe)

    make_figure()
    print(f"total wall time {time.time() - t_start:.0f} s")


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load("figures/descent_best_case_D_fast.npz")
    v = np.load("figures/descent_best_case_D_fast_validation.npz")
    ntrans = int(z["ntrans"])
    val, pairs = v["val"], v["val_scr_pairs"]
    d_obs = val[:, 4] - val[:, 3]                  # observed folds: MILP recount - heuristic
    d_scr = pairs[:, 1] - pairs[:, 0]              # scrambled folds: MILP recount - heuristic
    obj_below = int(np.sum(val[:, 5] < val[:, 3]) + np.sum(pairs[:, 2] < pairs[:, 0]))
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
    a = ax[0]
    allv = np.concatenate([d_obs, d_scr]).astype(int)
    vals, cnts = np.unique(allv, return_counts=True)
    a.bar(vals, cnts, color=["#D55E00" if x < 0 else "#4C72B0" for x in vals], width=0.8)
    a.set_xticks(range(int(vals.min()), int(vals.max()) + 1, 2))
    a.set_xlabel("training descents: exact solver minus heuristic", fontsize=11)
    a.set_ylabel(f"fold fits (n = {len(allv)})", fontsize=11)
    a.set_title("(a) solver check", loc="left", fontsize=11)
    a.text(0.97, 0.95, f"MILP objective below heuristic: {obj_below}/{len(allv)}\n"
           f"exact recount below heuristic: {int(np.sum(allv < 0))}/{len(allv)}\n"
           "(solver-tolerance cases, orange)", transform=a.transAxes, ha="right", va="top",
           fontsize=9.5)
    a.tick_params(labelsize=10)
    bins = np.arange(0.70, 1.0001, 1 / ntrans) - 0.5 / ntrans
    a = ax[1]
    fs, fco = z["scram_opt"], float(z["f_con_opt"])
    a.hist(fs, bins=bins, color="#bbbbbb", label=f"scrambled, own optimal d (B = {len(fs)})")
    a.axvline(fco, color="#D55E00", lw=2.2, label=f"fitted, own optimal d = {fco:.3f}")
    pa = (np.sum(fs >= fco) + 1) / (len(fs) + 1)
    a.set_xlabel("held-out descent fraction (78 transitions)", fontsize=11)
    a.set_ylabel("draws", fontsize=11)
    a.set_title(f"(b) exact solver, max-margin tie-break: p = {pa:.2f}", loc="left", fontsize=11)
    a.legend(frameon=False, fontsize=9.5, loc="upper left")
    a.set_ylim(0, a.get_ylim()[1] * 1.3)
    a.tick_params(labelsize=10)
    a = ax[2]
    fe = z["scram_heur_equal"]
    fh = float(z["f_con_heur"])
    if len(fe):
        a.hist(fe, bins=bins, color="#bbbbbb", label=f"scrambled, 400 restarts (B = {len(fe)})")
        a.axvline(fh, color="#D55E00", lw=2.2, label=f"fitted, 400 restarts = {fh:.3f}")
        ph = (np.sum(fe >= fh) + 1) / (len(fe) + 1)
        a.set_title(f"(c) published heuristic, equal effort: p = {ph:.2f}", loc="left",
                    fontsize=11)
        a.legend(frameon=False, fontsize=9.5, loc="upper left")
        a.set_ylim(0, a.get_ylim()[1] * 1.3)
    a.set_xlabel("held-out descent fraction (78 transitions)", fontsize=11)
    a.set_ylabel("draws", fontsize=11)
    a.tick_params(labelsize=10)
    for x in ax:
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/fig_descent_best_case_D_fast.png", dpi=170, bbox_inches="tight")
    print("[fig] figures/fig_descent_best_case_D_fast.png")


if __name__ == "__main__":
    main()
