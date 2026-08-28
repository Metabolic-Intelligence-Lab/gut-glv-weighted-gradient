"""
Do the measured trajectories descend a weighted potential?
==========================================================

The predictive test (constrained_vs_free_glv.py) compares the one-step accuracy of a free
and a diagonally symmetrizable gLV and finds a tie, but detection_power.py shows that test
never exceeds power 0.28 on these trajectory lengths: it asks two nearly identical fits to
out-predict each other. The manuscript says as much, and notes that one-step prediction is
insensitive to exactly what separates the two models.

This is the sharper instrument. Diagonal symmetrizability is not a statement about accuracy,
it is the statement that the dynamics is a weighted gradient flow, and that carries a
qualitative, directional prediction on every single transition.

For dx_i/dt = x_i (r_i + sum_j J_ij x_j) with a positive diagonal D=diag(d) making DJ
symmetric (d_i J_ij = d_j J_ji), define

    H_D(x) = - sum_i d_i r_i x_i  -  (1/2) sum_ij d_i J_ij x_i x_j .

Then dH_D/dx_k = -d_k [ r_k + (Jx)_k ], so dx_k/dt = -(x_k/d_k) dH_D/dx_k and

    dH_D/dt = - sum_k (x_k / d_k) (dH_D/dx_k)^2  <=  0      (x_k >= 0, d_k > 0),

i.e. H_D can never increase. So the test statistic is simply the fraction of observed
consecutive transitions along which H_D goes down. With 13 EV3 communities at ~7 time points
that is ~70 transitions, and it is a sign test with a hard one-sided prediction rather than a
magnitude comparison between two fits.

Three numbers make it readable:

PRIMARY TEST -- is the FITTED potential descended better than a generic one?
    Scored on the identical real trajectories, so relaxation onto a steady composition, the
    compositional constraint and the clustering of transitions within communities all cancel:
    only the structure of the potential differs. The reference is the fitted K with its
    off-diagonal entries shuffled, which preserves the magnitude spectrum and destroys the
    structure. This is the comparison the result rests on.

DIAGNOSTIC, NOT A TEST -- the end-to-end null (simulate symmetrizable, re-fit, score).
    Run and reported, but it CANNOT be used for this statistic, and the reason is worth
    recording. inference_aware_null.simulate() propagates log-abundances with process noise
    and never restores the compositional constraint, so where the real EV3 samples sum to
    1.000 exactly, the simulated ones range over 0.44 to 2.13. H_D is linear and quadratic in
    x, so total abundance moves it strongly, and the simulation explores H_D in a way the data
    cannot. That, and not equilibration or noise scale, is why an earlier version of this
    script found the observed descent ABOVE the null (0.83 vs 0.67) and read it as a finding:
    it is an artefact. Heteroscedasticity was checked and excluded (sigma on all residuals
    0.450, on settled transitions 0.452).

    NOTE for inference_aware_null.py itself: it uses the same off-simplex simulator to build
    the null for f_sign and r_cycle. The consequence there is a null that is probably too
    wide, which would make its "observed asymmetry is not distinguishable from the null"
    conclusion conservative rather than wrong, but it should be re-derived on a
    composition-preserving simulator before being relied on.

CHANCE -- the same statistic with the time order shuffled within each community.

Caveat kept in view: EV3 is relative abundance, so the fitted gLV lives in compositional
coordinates. The observed statistic and both references are computed through the identical
pipeline, so the comparison is fair, but the absolute descent fraction is a property of the
fitted representation, not of the organism.

Run:  python3 scripts/lyapunov_descent.py     (writes figures/fig_lyapunov_descent.png)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import (          # noqa: E402
    load_communities, transitions, _constrained_inner, S, EPS, LAM,
)
from inference_aware_null import resid_std, simulate   # noqa: E402

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = "figures"
NDRAW = 60           # end-to-end null draws (diagnostic only)
NSHUF = 300          # time-shuffle draws
NPERM = 400          # structure-shuffled potentials (the primary test)


def fit_symmetrizable(trans, lam=LAM):
    """As constrained_vs_free_glv.fit_constrained, but also return the weights d.

    d is what makes DJ symmetric, and H_D cannot be written without it.
    """
    res = minimize(lambda u: _constrained_inner(u, trans, lam)[2], np.zeros(S),
                   method="L-BFGS-B", options={"maxiter": 60})
    r, J, _ = _constrained_inner(res.x, trans, lam)
    d = np.exp(res.x - res.x.mean())
    return r, J, d


def H_D(x, r, J, d):
    """The weighted potential. Non-increasing along the flow iff DJ is symmetric."""
    return float(-np.sum(d * r * x) - 0.5 * np.sum(np.outer(d * x, x) * J))


def descent_fraction(comms, r, J, d):
    """Fraction of consecutive observed transitions along which H_D does not increase."""
    down = tot = 0
    deltas = []
    for _, X in comms:
        h = [H_D(x, r, J, d) for x in X]
        for a, b in zip(h[:-1], h[1:]):
            deltas.append(b - a)
            tot += 1
            down += (b <= a)
    return down / max(tot, 1), tot, np.array(deltas)


def settled_resid_std(comms, r, J):
    """Process noise estimated on the SETTLED half of each trajectory.

    resid_std() uses every transition, so it is dominated by model error during the initial
    relaxation; injecting that as persistent noise leaves the simulation jittering at
    equilibrium when the real data have gone quiet. The descent statistic lives in the quiet
    regime, so that is where the noise must be matched.
    """
    res = []
    for _, X in comms:
        k = max(1, (len(X) - 1) // 2)
        for a, b in list(zip(X[:-1], X[1:]))[k:]:
            g = np.log(b + EPS) - np.log(a + EPS)
            m = (a > 5e-3) & (b > 5e-3)
            res.extend((g[m] - (r + J @ a)[m]).tolist())
    return float(np.std(res)) if res else 0.0


def shuffled_potentials(comms, r, J, d, n=NPERM):
    """Descent fraction under potentials with the fitted magnitudes but scrambled structure."""
    K = d[:, None] * J
    iu = np.triu_indices(S, 1)
    out = []
    for i in range(n):
        rng = np.random.default_rng(40000 + i)
        Ks = np.zeros((S, S))
        Ks[iu] = rng.permutation(K[iu])
        Ks = Ks + Ks.T
        np.fill_diagonal(Ks, np.diag(K))
        f, _, _ = descent_fraction(comms, r, Ks / d[:, None], d)
        out.append(f)
    return np.array(out)


def symmetry_residual(J, d):
    """Relative departure of DJ from symmetry; ~0 confirms the fit really is symmetrizable."""
    M = d[:, None] * J
    return float(np.linalg.norm(M - M.T) / max(np.linalg.norm(M), 1e-12))


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    comms = load_communities()
    trans = transitions(comms)
    print(f"{len(comms)} EV3 communities, {len(trans)} one-step transitions\n")

    # ---- the symmetrizable ground truth fitted to the real trajectories
    r, J, d = fit_symmetrizable(trans)
    print(f"fitted symmetrizable model: ||DJ-(DJ)^T||/||DJ|| = {symmetry_residual(J, d):.2e} "
          f"(0 by construction)")
    obs, ntrans, dobs = descent_fraction(comms, r, J, d)
    print(f"OBSERVED descent fraction : {obs:.3f}   ({int(round(obs*ntrans))}/{ntrans} "
          f"transitions with dH_D <= 0)")

    # ---- PRIMARY: is this potential special, or would any of the same magnitude do?
    shuf = shuffled_potentials(comms, r, J, d)
    p_shuf = (np.sum(shuf >= obs) + 1) / (len(shuf) + 1)
    print(f"\nPRIMARY -- structure-shuffled potentials (same magnitudes, N={len(shuf)}):")
    print(f"  descent fraction        : {shuf.mean():.3f}  "
          f"[{np.percentile(shuf,2.5):.3f}, {np.percentile(shuf,97.5):.3f}]")
    print(f"  p(shuffled >= observed) : {p_shuf:.3f}")

    # ---- SECONDARY: end-to-end null, with noise calibrated where the statistic lives
    sproc_all = resid_std(comms, r, J)
    sproc = settled_resid_std(comms, r, J)
    print(f"\nDIAGNOSTIC -- end-to-end null (NOT usable: the simulator leaves the simplex,")
    print(f"  see docstring). sigma all={sproc_all:.3f}, settled={sproc:.3f}")
    null = []
    for i in range(NDRAW):
        rng = np.random.default_rng(20000 + i)
        sim = simulate(comms, r, J, sproc, rng)
        try:
            rs, Js, ds = fit_symmetrizable(transitions(sim))
            f, _, _ = descent_fraction(sim, rs, Js, ds)
        except Exception:
            continue
        null.append(f)
    null = np.array(null)

    # ---- chance baseline: shuffle the time order within each community
    chance = []
    for i in range(NSHUF):
        rng = np.random.default_rng(30000 + i)
        reordered = [(n, X[rng.permutation(len(X))]) for n, X in comms]
        f, _, _ = descent_fraction(reordered, r, J, d)
        chance.append(f)
    chance = np.array(chance)
    p_null = (np.sum(null <= obs) + 1) / (len(null) + 1)
    print(f"  null descent fraction   : {null.mean():.3f}  "
          f"[{np.percentile(null,2.5):.3f}, {np.percentile(null,97.5):.3f}]  (N={len(null)})")
    print(f"  time-shuffled chance    : {chance.mean():.3f}  "
          f"[{np.percentile(chance,2.5):.3f}, {np.percentile(chance,97.5):.3f}]")
    print(f"  one-sided p (obs <= null): {p_null:.3f}")

    print("\nRESULT")
    print(f"  The trajectories descend the fitted potential on {obs:.0%} of transitions,")
    print(f"  against {chance.mean():.0%} for shuffled time order and {shuf.mean():.0%} for")
    print(f"  potentials of the same magnitudes with the structure destroyed (p={p_shuf:.3f}).")
    if p_shuf < 0.05:
        print("  Most of the descent is generic relaxation onto a steady composition; the")
        print(f"  structure-specific increment is {obs-shuf.mean():+.0%}, and it is significant.")
        print("  This is positive evidence for a weighted-gradient description, on the one")
        print("  designed community with raw trajectories, at a resolution the one-step")
        print("  predictive comparison did not have.")
    else:
        print("  The fitted potential is not descended better than a generic one: the descent")
        print("  reflects relaxation, and the test is uninformative about symmetrizability.")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.3))

    ax[0].hist(shuf, bins=24, color="#4C72B0", alpha=.8,
               label=f"same magnitudes,\nstructure shuffled (N={len(shuf)})")
    ax[0].hist(chance, bins=24, color="#999999", alpha=.55,
               label="time order shuffled")
    ax[0].axvline(obs, color="#D55E00", lw=2.5, label=f"observed = {obs:.2f}")
    ax[0].set_xlabel("fraction of transitions with $\\Delta H_D \\leq 0$")
    ax[0].set_ylabel("count")
    ax[0].set_title("a  The fitted potential is descended more often than a\n"
                    f"generic one of the same magnitudes ($p$ = {p_shuf:.3f}, $n$ = {ntrans})",
                    loc="left", fontsize=10.5)
    ax[0].legend(fontsize=8, frameon=False, loc="upper left")

    order = np.argsort(dobs)
    ax[1].bar(range(len(dobs)), dobs[order],
              color=["#4C72B0" if v <= 0 else "#D55E00" for v in dobs[order]])
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_yscale("symlog", linthresh=1e-2)
    ax[1].set_xlabel(f"observed transition (sorted), $n$ = {ntrans}")
    ax[1].set_ylabel("$\\Delta H_D$ per transition (symlog)")
    ax[1].set_title("b  Per-transition change in the potential\n"
                    "blue descends as a gradient flow requires, orange climbs",
                    loc="left", fontsize=10.5)

    fig.tight_layout()
    out = f"{OUT}/fig_lyapunov_descent.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    np.savez(f"{OUT}/lyapunov_descent.npz", observed=obs, null=null, chance=chance,
             shuffled=shuf, deltas=dobs, p_shuffled=p_shuf, p_null=p_null,
             n_transitions=ntrans, sigma_settled=sproc, sigma_all=sproc_all)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
