"""
Two controls the H_D descent test needs before it can be believed
=================================================================

lyapunov_descent.py finds the EV3 trajectories descending the fitted weighted potential on
65/78 transitions (0.83) against a time-shuffled chance level of 0.50, but ALSO above the
end-to-end symmetrizable null (0.68, 95% [0.58, 0.81]). Data cannot be more symmetrizable
than a symmetrizable ground truth, so one of two things is wrong, and both are checked here.

CONTROL A -- is the FITTED potential special, or would any potential do?
    Trajectories relaxing onto a steady composition will descend a great many functions that
    happen to have a minimum near that steady state, and EV3 is relative abundance living on
    a simplex, so this is a live worry. We score the identical real trajectories under
    potentials built from (i) a random symmetric K of matched scale with random positive
    weights, and (ii) the fitted K with its entries shuffled, keeping the magnitude spectrum.
    If those descend just as often, the result is an artefact of relaxation and the fitted
    structure carries nothing.

CONTROL B -- is the null's noise too big?
    The null injects process noise at sigma = std of the fit's one-step residuals. Those
    residuals contain model misspecification as well as process noise, so simulating all of
    it makes the synthetic trajectories wander more than the real ones, which would depress
    the null's descent fraction and manufacture exactly the anomaly observed. We compare the
    one-step dispersion of the real and simulated trajectories directly.

Run:  python3 scripts/lyapunov_controls.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import load_communities, transitions, S, EPS   # noqa: E402
from inference_aware_null import resid_std, simulate                        # noqa: E402
from lyapunov_descent import fit_symmetrizable, descent_fraction, H_D       # noqa: E402

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

NRAND = 400


def dispersion(comms):
    """Mean absolute one-step change in log-composition: how far the state moves per step."""
    v = []
    for _, X in comms:
        for a, b in zip(X[:-1], X[1:]):
            v.append(np.mean(np.abs(np.log(b + EPS) - np.log(a + EPS))))
    return float(np.mean(v))


def main():
    comms = load_communities()
    trans = transitions(comms)
    r, J, d = fit_symmetrizable(trans)
    obs, ntrans, _ = descent_fraction(comms, r, J, d)
    print(f"observed descent fraction: {obs:.3f}  ({ntrans} transitions)\n")

    # ---------------------------------------------------------------- CONTROL A
    K = d[:, None] * J                       # the symmetric matrix behind the fit
    off = K[~np.eye(S, dtype=bool)]
    scale = off.std()
    iu = np.triu_indices(S, 1)

    rand_frac, shuf_frac = [], []
    for i in range(NRAND):
        rng = np.random.default_rng(40000 + i)

        # (i) random symmetric K of matched scale, random positive weights of matched spread
        Kr = np.zeros((S, S))
        Kr[iu] = rng.normal(0, scale, size=len(iu[0]))
        Kr = Kr + Kr.T
        np.fill_diagonal(Kr, np.diag(K))     # keep the self-limitation, which sets the basin
        dr = np.exp(rng.normal(0, np.std(np.log(d)), size=S))
        f, _, _ = descent_fraction(comms, r, Kr / dr[:, None], dr)
        rand_frac.append(f)

        # (ii) the fitted K with its off-diagonal entries shuffled (magnitudes preserved)
        Ks = np.zeros((S, S))
        Ks[iu] = rng.permutation(K[iu])
        Ks = Ks + Ks.T
        np.fill_diagonal(Ks, np.diag(K))
        f, _, _ = descent_fraction(comms, r, Ks / d[:, None], d)
        shuf_frac.append(f)

    rand_frac = np.array(rand_frac)
    shuf_frac = np.array(shuf_frac)
    p_rand = (np.sum(rand_frac >= obs) + 1) / (len(rand_frac) + 1)
    p_shuf = (np.sum(shuf_frac >= obs) + 1) / (len(shuf_frac) + 1)

    print("CONTROL A -- is the fitted potential special?")
    print(f"  random symmetric K, random weights : {rand_frac.mean():.3f} "
          f"[{np.percentile(rand_frac,2.5):.3f}, {np.percentile(rand_frac,97.5):.3f}]"
          f"   p(random >= observed) = {p_rand:.3f}")
    print(f"  fitted K, off-diagonals shuffled   : {shuf_frac.mean():.3f} "
          f"[{np.percentile(shuf_frac,2.5):.3f}, {np.percentile(shuf_frac,97.5):.3f}]"
          f"   p(shuffled >= observed) = {p_shuf:.3f}")
    if p_rand > 0.05 or p_shuf > 0.05:
        print("  => NOT specific: a generic potential descends about as often. The descent\n"
              "     result reflects relaxation onto a steady composition, not symmetrizability.")
    else:
        print("  => specific: the fitted potential descends far better than generic ones.")

    # ---------------------------------------------------------------- CONTROL B
    sproc = resid_std(comms, r, J)
    disp_real = dispersion(comms)
    disp_sim = np.mean([dispersion(simulate(comms, r, J, sproc,
                                            np.random.default_rng(50000 + i)))
                        for i in range(40)])
    print(f"\nCONTROL B -- is the null's noise calibrated?")
    print(f"  sigma injected into the null        : {sproc:.3f}")
    print(f"  one-step dispersion, real data      : {disp_real:.3f}")
    print(f"  one-step dispersion, simulated null : {disp_sim:.3f}"
          f"   (ratio {disp_sim/disp_real:.2f}x)")
    if disp_sim > 1.15 * disp_real:
        print("  => the null trajectories move MORE than the real ones, so the null under-\n"
              "     states descent and 'observed above null' is a noise artefact, not a finding.")
    elif disp_sim < 0.85 * disp_real:
        print("  => the null trajectories move LESS than the real ones.")
    else:
        print("  => dispersion matches; the null noise is not the explanation.")

    np.savez("figures/lyapunov_controls.npz", observed=obs, rand=rand_frac, shuf=shuf_frac,
             disp_real=disp_real, disp_sim=disp_sim, sigma=sproc)
    print("\nsaved figures/lyapunov_controls.npz")


if __name__ == "__main__":
    main()
