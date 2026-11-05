r"""
Constructive demonstration that a high descent fraction carries no information about
gradient structure.
====================================================================================
Lyapunov's theorem: every asymptotically stable fixed point admits a function that decreases
strictly along all nearby trajectories, whether or not the flow is a gradient. Stability is
shared by both hypotheses of the manuscript, so the existence of a decreasing function
cannot separate them, and no amount of data or any better reference distribution can repair
that: the defect is in the statistic, not in the null.

The demonstration is explicit. For each NON-symmetrizable ground truth of the specificity
ensemble, take the Jacobian A = diag(x*) J at the coexisting stationary state and solve the
Lyapunov equation A^T P + P A = -I for P positive definite. The quadratic
V(x) = (x - x*)^T P (x - x*) then decreases along the flow of a system that is provably not a
weighted-gradient system. We measure its descent fraction on simulated trajectories of the
real design and compare it with the descent fraction of the fitted symmetrizable potential,
the statistic the manuscript uses.

If the constructed V descends as often as, or more often than, the fitted potential on a
non-symmetrizable truth, the descent fraction is demonstrably not a test of gradient
structure.
"""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
repo = sys.argv[1] if len(sys.argv) > 1 else "."
sys.path.insert(0, os.path.join(repo, "scripts")); os.chdir(repo)
from constrained_vs_free_glv import load_communities, transitions, fit_free, S
from descent_specificity import fit_constrained_fast_d
import descent_specificity_ensemble as ENS
from power_analysis import antisym
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")

comms = load_communities(); tr = transitions(comms)
r0, J0, d0 = fit_constrained_fast_d(tr); rf, Jf = fit_free(tr)
geo = ENS.Geometry(comms, r0, J0, d0)
CAP, FLOOR, TAU = 8.0, 1e-12, 1e-2
RNG = np.random.default_rng(0)

truths = [("base", r0, J0.copy(), 0.0)]
P0 = antisym(J0, RNG); unit = geo.d_vf(r0, J0+P0)
for li in (1,3):
    J = J0 + (ENS.TARGETS[li]/unit)*P0; truths.append((f"antisym{li}", r0, J, geo.d_vf(r0,J)))
for fam, one, off in (("oneway",ENS._oneway_one,1000),("sign",ENS._sign_one,2000)):
    for li in (1,3):
        J,v = ENS.greedy_subset(geo, r0, J0, one, off+3, ENS.TARGETS[li]); truths.append((f"{fam}{li}",r0,J,v))
for li in (1,3):
    kap,v = ENS.bisect_param(lambda k: geo.d_vf(r0, ENS.gt_cycle(J0,0,k)), ENS.TARGETS[li], 0.0, 6.0)
    truths.append((f"cycle{li}", r0, ENS.gt_cycle(J0,0,kap), v))
for li,t in enumerate((1.0,3.0)):
    r,J = ENS.gt_free(r0,J0,rf,Jf,t); truths.append((f"free{li}", r, J, geo.d_vf(r,J)))

def sym_residual(J, d):
    A = np.diag(d) @ J
    return float(np.linalg.norm(A - A.T) / np.linalg.norm(A))

def stationary(r, J, x0, pres, n=400):
    x = x0.copy(); x[~pres] = 0.0
    for _ in range(n):
        x = np.clip(np.exp(np.log(np.maximum(x, FLOOR)) + r + J @ x), 0, CAP); x[~pres] = 0.0
    return x

def trajectories(r, J, pres, sp, nstart=8, T=12, seed=1):
    """Trajectories of the real design's shape: several inocula, short relaxation."""
    rng = np.random.default_rng(seed)
    base = comms[0][1][0].copy()
    out = []
    for k in range(nstart):
        x = base.copy()
        if k:                                   # perturbed inocula spanning directions
            w = rng.uniform(0.2, 2.5, S); x = x * w
        x[~pres] = 0.0
        traj = [x.copy()]
        for _ in range(T - 1):
            x = np.clip(np.exp(np.log(np.maximum(x, FLOOR)) + r + J @ x
                               + sp * rng.standard_normal(S)), 0, CAP)
            x[~pres] = 0.0; traj.append(x.copy())
        out.append(np.array(traj))
    return out

def frac_descent(trajs, Vfun, sv):
    down = tot = 0
    for Y in trajs:
        v = np.array([Vfun(y[sv]) for y in Y])
        d = np.diff(v)
        ok = np.isfinite(d)
        down += int(np.sum(d[ok] < 0)); tot += int(ok.sum())
    return down / max(tot, 1)

def H_fitted(xs, xstar, d):
    """The manuscript's weighted potential: sum_i d_i (x_i - x_i* ln x_i)."""
    return float(np.sum(d * (xs - xstar * np.log(np.maximum(xs, FLOOR)))))

pres = comms[0][1].mean(0) > 1e-3
print(f"host {comms[0][0]}; trajectories: 8 inocula x 12 steps, process noise 0.10\n")
print(f"{'truth':<10}{'dVF':>7}{'||DJ-(DJ)^T||/||DJ||':>22}{'descent, fitted H':>19}"
      f"{'descent, Lyapunov V':>21}{'P>0':>6}")
for name, r, J, v in truths:
    x = stationary(r, J, comms[0][1][0].copy(), pres)
    sv = np.where(x > TAU)[0]
    if len(sv) < 3:
        print(f"{name:<10}{v:>7.3f}  no coexistence"); continue
    Js = J[np.ix_(sv, sv)]; xs = x[sv]
    A = np.diag(xs) @ Js
    # the manuscript's own symmetrizer, refitted for this truth's survivor block
    res = sym_residual(Js, np.array([d0[i] for i in sv]))
    try:
        Pm = solve_continuous_lyapunov(A.T, -np.eye(len(sv)))
        Pm = 0.5 * (Pm + Pm.T)
        pos = bool(np.all(np.linalg.eigvalsh(Pm) > 0))
    except Exception:
        Pm, pos = None, False
    trajs = trajectories(r, J, pres, 0.10, seed=7)
    fh = frac_descent(trajs, lambda y: H_fitted(y, xs, np.array([d0[i] for i in sv])), sv)
    fv = frac_descent(trajs, lambda y: float((y - xs) @ Pm @ (y - xs)), sv) if pos else np.nan
    print(f"{name:<10}{v:>7.3f}{res:>22.3f}{fh:>19.3f}{fv:>21.3f}{str(pos):>6}")

print("\nreading: the first numeric column is the exact symmetrizability violation of the")
print("truth on its own survivor block. The last two are how often the two candidate")
print("potentials decrease along the same trajectories.")
