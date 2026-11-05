"""
Is the descent margin robust to which near-optimal weights d the optimiser returns? (R4-7)
=========================================================================================

optimizer_diagnostics.py shows that every published constrained fit converges (14 to 27
L-BFGS-B iterations) and sits within 3.3e-4 (relative) of the best objective found from 20
random starts, but that the weights are weakly identified: starts that reach the same objective
differ by up to 8.4 in u = log d, and d spans four orders of magnitude within a fit. The fixed-d
test is valid at any given d, so the question is whether its result depends on which
near-optimal d is used.

For every random start k = 1..NSTART (identical seeds and call order to optimizer_diagnostics),
this script assembles a complete leave-one-community-out set from start k's solution in every
fold, and scores held-out descent against the structure-scrambled null at that fold's own d
(B = NPERM_K draws per fold). It reports the distribution over k of the descent fraction, the
scrambled mean, the margin and p, pooled and on the biomass-plateau transitions, alongside the
published (u0 = 0) fits. It also reports which species carry the weakly identified directions,
and the range of the effective weights d_i <x_i>, which is what actually enters H_D.

Run:  OMP_NUM_THREADS=1 python3 scripts/optimizer_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import load_communities, transitions, _constrained_inner, S, LAM  # noqa: E402
from lyapunov_descent import H_D                                                           # noqa: E402
from lyapunov_robustness import scramble                                                   # noqa: E402

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

NSTART = 20
NPERM_K = 1000
OUT = "figures/optimizer_sensitivity.npz"
SPECIES = ["BH", "CA", "BU", "PC", "BO", "BV", "BT", "EL", "FP", "CH", "DP", "ER"]
PHASE = {0: "growth", 1: "plateau", 2: "transfer", 3: "plateau", 4: "transfer", 5: "plateau"}


def fit(trans, u0, maxiter):
    res = minimize(lambda u: _constrained_inner(u, trans, LAM)[2], u0,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    r, J, obj = _constrained_inner(res.x, trans, LAM)
    u = res.x - res.x.mean()
    return dict(r=r, J=J, d=np.exp(u), u=u, obj=float(obj))


def deltas(X, r, J, d):
    h = [H_D(x, r, J, d) for x in X]
    return np.diff(h)


def score(comms, fits, seed0):
    """Held-out descent for one complete LOCO set; returns per-transition obs and null."""
    obs, null, phase = [], [], []
    for h, (name, X) in enumerate(comms):
        f = fits[h]
        dl = deltas(X, f["r"], f["J"], f["d"])
        K = f["d"][:, None] * f["J"]
        nd = np.empty((NPERM_K, len(dl)))
        for i in range(NPERM_K):
            rng = np.random.default_rng(seed0 + 997 * h + i)
            nd[i] = deltas(X, f["r"], scramble(K, f["d"], rng), f["d"])
        obs.extend(dl.tolist())
        null.extend(nd.T.tolist())
        phase.extend(PHASE[k] for k in range(len(dl)))
    return np.array(obs), np.array(null), np.array(phase)


def summarise(obs, null, sel):
    f_obs = np.mean(obs[sel] <= 0)
    f_null = np.mean(null[sel] <= 0, axis=0)
    b = int(np.sum(f_null >= f_obs))
    return f_obs, f_null.mean(), f_obs - f_null.mean(), (b + 1) / (len(f_null) + 1)


def main():
    comms = load_communities()
    rng = np.random.default_rng(2026)                    # same seeds as optimizer_diagnostics
    sets = [("full", transitions(comms))] + [
        (comms[h][0], transitions([c for k, c in enumerate(comms) if k != h]))
        for h in range(len(comms))]
    published, starts = [], []
    for name, trans in sets:
        published.append(fit(trans, np.zeros(S), 60))
        _ = fit(trans, np.zeros(S), 2000)                # keeps the RNG call order identical
        starts.append([fit(trans, rng.normal(0, 1, S), 2000) for _ in range(NSTART)])
        print(f"  fitted {name}", flush=True)

    fold_pub = published[1:]
    fold_starts = starts[1:]

    # weakly identified directions: per-species spread of u across near-optimal solutions
    spread = np.array([[np.ptp([s["u"][i] for s in fs]) for i in range(S)] for fs in fold_starts])
    mean_x = np.mean(np.vstack([X for _, X in comms]), axis=0)
    print("\nper-species range of u across the 20 starts (median over the 13 folds), and <x>:")
    for i in np.argsort(-np.median(spread, axis=0)):
        print(f"    {SPECIES[i]:<3} u-range {np.median(spread[:, i]):6.2f}   <x> {mean_x[i]:.4f}")
    eff = np.array([p["d"] * mean_x for p in fold_pub])
    big = mean_x > 5e-3
    print(f"effective weight d_i<x_i>, max/min over species with <x> > 5e-3: "
          f"[{min((e[big].max() / e[big].min()) for e in eff):.1f}, "
          f"{max((e[big].max() / e[big].min()) for e in eff):.1f}] across folds")

    # descent under the published fits and under each start's complete LOCO set
    res = {}
    obs, null, ph = score(comms, fold_pub, 80000)
    res["published"] = (summarise(obs, null, np.ones(len(obs), bool)),
                        summarise(obs, null, ph == "plateau"))
    for k in range(NSTART):
        fits = [fs[k] for fs in fold_starts]
        obs, null, ph = score(comms, fits, 80000)
        res[f"start{k}"] = (summarise(obs, null, np.ones(len(obs), bool)),
                            summarise(obs, null, ph == "plateau"))

    print(f"\n{'set':<11}{'all: fit':>9}{'scr':>7}{'margin':>8}{'p':>8}"
          f"{'plateau: fit':>14}{'scr':>7}{'margin':>8}{'p':>8}")
    for key, (a, p) in res.items():
        print(f"{key:<11}{a[0]:>9.3f}{a[1]:>7.3f}{a[2]:>+8.3f}{a[3]:>8.4f}"
              f"{p[0]:>14.3f}{p[1]:>7.3f}{p[2]:>+8.3f}{p[3]:>8.4f}")
    ks = [res[f"start{k}"] for k in range(NSTART)]
    for lab, j in (("all", 0), ("plateau", 1)):
        m = np.array([r[j][2] for r in ks])
        pv = np.array([r[j][3] for r in ks])
        print(f"\n{lab}: margin over {NSTART} near-optimal LOCO sets: min {m.min():+.3f}, "
              f"median {np.median(m):+.3f}, max {m.max():+.3f};  p < 0.05 in "
              f"{int(np.sum(pv < 0.05))}/{NSTART};  max p {pv.max():.4f}")
    np.savez(OUT, spread=spread, mean_x=mean_x,
             summary=np.array([[*a, *p] for a, p in res.values()]),
             keys=np.array(list(res.keys())))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
