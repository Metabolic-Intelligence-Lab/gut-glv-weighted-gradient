"""
Optimiser diagnostics for the constrained (diagonally symmetrizable) fit (BIB revision, R4-7)
=============================================================================================

Reviewer 4 asks for the optimisation bounds, the convergence criteria, the range of the fitted
diagonal weights and the sensitivity to multiple initialisations, because the published fit
starts L-BFGS-B from u = 0. This script reports, for the full fit and for each of the 13
leave-one-community-out folds used by the descent test:

  * the published fit (lyapunov_descent.fit_symmetrizable: L-BFGS-B, u0 = 0, maxiter 60):
    objective, iterations, scipy's convergence flag and message;
  * the same fit run to convergence (maxiter 2000, default tolerances);
  * NSTART random initialisations u0 ~ N(0, 1): best objective, spread of the solutions in the
    gauge-fixed coordinates (mean u = 0), and whether any start beats the published objective;
  * the range of the fitted weights d = exp(u - mean u);
  * the held-out descent fraction under the published and under the best multistart solution.

The parametrisation is unconstrained (d = exp(u) > 0 by construction), so there are no bounds;
the only gauge freedom, a common shift of u, leaves DJ symmetric and every descent sign
unchanged, and is fixed by mean(u) = 0.

Run:  OMP_NUM_THREADS=2 python3 scripts/optimizer_diagnostics.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import load_communities, transitions, _constrained_inner, S, LAM  # noqa: E402
from lyapunov_descent import descent_fraction                                              # noqa: E402

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

NSTART = 20


def fit(trans, u0, maxiter):
    res = minimize(lambda u: _constrained_inner(u, trans, LAM)[2], u0,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    r, J, obj = _constrained_inner(res.x, trans, LAM)
    u = res.x - res.x.mean()
    return dict(r=r, J=J, d=np.exp(u), u=u, obj=float(obj), nit=int(res.nit),
                success=bool(res.success), msg=str(res.message))


def main():
    comms = load_communities()
    rng = np.random.default_rng(2026)
    rows = []
    folds = [("full fit", transitions(comms), None)] + [
        (f"fold {comms[h][0]}", transitions([c for k, c in enumerate(comms) if k != h]), h)
        for h in range(len(comms))]
    pub_down = best_down = tot = 0
    for name, trans, h in folds:
        pub = fit(trans, np.zeros(S), 60)
        conv = fit(trans, np.zeros(S), 2000)
        starts = [fit(trans, rng.normal(0, 1, S), 2000) for _ in range(NSTART)]
        best = min(starts + [conv], key=lambda s: s["obj"])
        spread = max(np.max(np.abs(s["u"] - best["u"])) for s in starts)
        rows.append((name, pub, conv, best, spread, starts))
        if h is not None:
            fp, n, _ = descent_fraction([comms[h]], pub["r"], pub["J"], pub["d"])
            fb, _, _ = descent_fraction([comms[h]], best["r"], best["J"], best["d"])
            pub_down += fp * n
            best_down += fb * n
            tot += n

    print(f"{'fit':<12}{'pub obj':>11}{'nit':>5}{'conv':>6}{'conv obj':>11}{'best of ' + str(NSTART):>12}"
          f"{'rel gap':>10}{'max|du|':>9}{'d min':>8}{'d max':>8}{'max/min':>9}")
    gaps, ratios, spreads, unconv = [], [], [], 0
    for name, pub, conv, best, spread, starts in rows:
        gap = (pub["obj"] - best["obj"]) / abs(best["obj"])
        gaps.append(gap)
        ratios.append(pub["d"].max() / pub["d"].min())
        spreads.append(spread)
        unconv += (not pub["success"])
        print(f"{name:<12}{pub['obj']:>11.4f}{pub['nit']:>5}{'yes' if pub['success'] else 'NO':>6}"
              f"{conv['obj']:>11.4f}{best['obj']:>12.4f}{gap:>10.2e}{spread:>9.3f}"
              f"{pub['d'].min():>8.3f}{pub['d'].max():>8.3f}{ratios[-1]:>9.2f}")
    print(f"\npublished fits not flagged converged by scipy at maxiter=60: {unconv}/{len(rows)}")
    print(f"messages of unconverged published fits: "
          f"{sorted({r[1]['msg'] for r in rows if not r[1]['success']})}")
    print(f"relative objective gap published vs best multistart: max {max(gaps):.2e}, "
          f"median {np.median(gaps):.2e}")
    print(f"max |u - u_best| across {NSTART} random starts: max {max(spreads):.3f}, "
          f"median {np.median(spreads):.3f}")
    print(f"fitted weight ratio d_max/d_min (published): range [{min(ratios):.2f}, {max(ratios):.2f}]")
    print(f"held-out descent: published fits {pub_down/tot:.3f}, best-multistart fits "
          f"{best_down/tot:.3f} ({int(tot)} transitions)")


if __name__ == "__main__":
    main()
