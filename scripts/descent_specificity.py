"""
Does the descent test have specificity? The end-to-end null the paper declines to run
=====================================================================================

The descent result is the paper's headline, and it rests on one comparator: a potential of
the same magnitudes with its structure scrambled. The Methods decline the obvious second
one, the null that Fig 3 runs for the matrix diagnostics, on this ground:

    "An end-to-end null of the kind used for the invariants is *not* usable for this
     statistic: the simulator does not preserve the compositional constraint."

That was true of inference_aware_null.simulate(). It has not been true since
inference_aware_null_compositional.simulate_compositional() was written to fix exactly that
defect for Fig A of S1 Text. The stated obstacle is gone, and the missing null is the one a
referee will ask for first, because Fig 5 has already shown that the *predictive* comparison
has no power at any injected asymmetry. The manuscript asserts the trajectory test does not
share that limitation. This measures whether it does.

The design mirrors detection_power.py exactly, with the scored statistic swapped:

    ground truth   J_delta = J0 + delta * P,  J0 the symmetrizable fit to the real EV3 data,
                   P the same fixed antisymmetric direction detection_power.py uses (seed 0),
                   so the x-axis is comparable point for point with Fig 5.
    data           simulate_compositional(): matched process noise, every observation
                   renormalised to unit total, as sequencing does.
    statistic      the paper's own held-out descent protocol: fit the symmetrizable model on
                   twelve simulated communities, score Delta H_D <= 0 on the thirteenth, pool
                   to 78 transitions, and test against the structure-scrambled null.

Read the two ends differently, because the ground truth changes meaning along the axis:

    delta = 0    the truth IS symmetrizable, so the rejection rate is SENSITIVITY. The paper
                 needs this to be high: it is the test recovering a gradient flow it should
                 recover, and it also says whether the observed 0.86 is what a symmetrizable
                 system yields through this pipeline.
    delta > 0    the truth is NOT symmetrizable, so the rejection rate is a FALSE POSITIVE
                 rate. If it stays near the delta = 0 value, the statistic cannot tell a
                 gradient flow from a non-gradient one and the headline claim has no
                 specificity, whatever its p-value on the real data.

Run:  OMP_NUM_THREADS=1 python3 scripts/descent_specificity.py
      ~15 min on 14 workers; each replicate costs 13 constrained fits.
      --replot redraws from figures/descent_specificity.npz without recomputing.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as betadist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import (                                   # noqa: E402
    load_communities, transitions, S, LAM,
)
from inference_aware_null import resid_std                              # noqa: E402
from inference_aware_null_compositional import simulate_compositional   # noqa: E402
from power_analysis import antisym, base_design, _PAIRS, _PK            # noqa: E402
from lyapunov_descent import fit_symmetrizable, descent_fraction        # noqa: E402

import warnings  # noqa: E402
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = "figures"
ALPHA = 0.05
DELTAS = [0.0, 1.0, 2.0, 4.0, 8.0]
N_REP = 400          # the fast fitter costs 0.03 s, so this is minutes, not hours
NPERM = 400          # scrambled draws per replicate; p resolution 1/401


# --------------------------------------------------------------------------- #
def fit_constrained_fast_d(trans, lam=LAM):
    """power_analysis.fit_constrained_fast, returning the weights d as well.

    d is what makes DJ symmetric and H_D cannot be written without it; the published
    fit_symmetrizable() returns it but loops in Python, which is too slow for 7800 fits.
    Agreement between the two on the real data is asserted in main() before any replicate
    is run, so the power curve is anchored to the number the manuscript quotes.
    """
    R, K0, y, esp = base_design(trans)
    npar = S + len(_PAIRS)

    def solve(u):
        d = np.exp(np.clip(u - u.mean(), -6.0, 6.0))
        M = np.hstack([R, K0 / d[esp][:, None]])
        reg = np.zeros(npar)
        for (i, j) in _PAIRS:
            if i != j:
                reg[S + _PK[(i, j)]] = lam * (1.0 / d[i] ** 2 + 1.0 / d[j] ** 2)
        beta = np.linalg.solve(M.T @ M + np.diag(reg) + 1e-6 * np.eye(npar), M.T @ y)
        return beta, d, M

    def loss(u):
        beta, _, M = solve(u)
        return float(np.mean((y - M @ beta) ** 2))

    res = minimize(loss, np.zeros(S), method="L-BFGS-B", bounds=[(-6, 6)] * S,
                   options={"maxiter": 25})
    beta, d, _ = solve(res.x)
    K = np.zeros((S, S))
    for (i, j) in _PAIRS:
        K[i, j] = K[j, i] = beta[S + _PK[(i, j)]]
    return beta[:S], K / d[:, None], d


def scramble(K, d, rng):
    """lyapunov_robustness.scramble: same magnitudes, structure destroyed."""
    iu = np.triu_indices(S, 1)
    Ks = np.zeros((S, S))
    Ks[iu] = rng.permutation(K[iu])
    Ks = Ks + Ks.T
    np.fill_diagonal(Ks, np.diag(K))
    return Ks / d[:, None]


def loco_descent_stat(comms, fitter, nperm=NPERM, seed=0):
    """The manuscript's held-out descent protocol, returning (fraction, null mean, p)."""
    down = tot = 0
    null = np.zeros(nperm)
    for h in range(len(comms)):
        train = transitions([c for k, c in enumerate(comms) if k != h])
        if not train:
            continue
        r, J, d = fitter(train)
        f, n, _ = descent_fraction([comms[h]], r, J, d)
        down += f * n
        tot += n
        K = d[:, None] * J
        for i in range(nperm):
            rng = np.random.default_rng(seed + 7919 * h + i)
            fs, ns, _ = descent_fraction([comms[h]], r, scramble(K, d, rng), d)
            null[i] += fs * ns
    if tot == 0:
        return np.nan, np.nan, 1.0
    obs = down / tot
    null = null / tot
    return obs, float(null.mean()), float((np.sum(null >= obs) + 1) / (nperm + 1))


def one_replicate(args):
    comms, r0, Jd, sproc, seed = args
    sim = simulate_compositional(comms, r0, Jd, sproc, np.random.default_rng(seed))
    return loco_descent_stat(sim, fit_constrained_fast_d, seed=seed)


def ci(k, n):
    return (betadist.ppf(0.025, k + 0.5, n - k + 0.5),
            betadist.ppf(0.975, k + 0.5, n - k + 0.5))


# --------------------------------------------------------------------------- #
def make_figure(deltas, frac, fracsd, nullm, rej, rlo, rhi, obs_real, null_real):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))

    ax = axes[0]
    frac, fracsd, nullm = map(np.asarray, (frac, fracsd, nullm))
    ax.errorbar(deltas, frac, yerr=fracsd, fmt="o-", color="#D55E00", lw=2, ms=7,
                capsize=4, label="held-out descent, simulated")
    ax.plot(deltas, nullm, "s--", color="#4C72B0", lw=1.6, ms=5,
            label="structure-scrambled null")
    ax.axhline(obs_real, color="#333333", ls=":", lw=1.4,
               label=f"real data = {obs_real:.2f}")
    ax.set_xlabel("injected antisymmetric perturbation $\\delta$")
    ax.set_ylabel("fraction with $\\Delta H_D \\leq 0$")
    ax.set_title("Descent falls as the ground truth leaves\nthe symmetrizable manifold",
                 fontsize=10)
    ax.set_ylim(0.4, 1.0)

    ax = axes[1]
    rej, rlo, rhi = map(np.asarray, (rej, rlo, rhi))
    ax.errorbar(deltas, rej, yerr=[rej - rlo, rhi - rej], fmt="o-", color="#0072B2",
                lw=2, ms=7, capsize=4)
    ax.axhline(ALPHA, color="grey", ls=":", lw=1, label="nominal level")
    ax.axhline(0.8, color="#009E73", ls="-.", lw=1, label="conventional 0.8")
    ax.set_xlabel("injected antisymmetric perturbation $\\delta$\n"
                  "($\\delta=0$: sensitivity.  $\\delta>0$: false-positive rate)")
    ax.set_ylabel(f"P(reject scrambled null at $\\alpha={ALPHA}$)")
    ax.set_title("Sensitivity at $\\delta=0$, specificity beyond it", fontsize=10)
    ax.set_ylim(0, 1)

    for a in axes:
        a.legend(frameon=False, fontsize=8, loc="best")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_descent_specificity.png")
    fig.savefig(p, dpi=200)
    print(f"[fig] {p} written")


def main():
    os.makedirs(OUT, exist_ok=True)
    npz = os.path.join(OUT, "descent_specificity.npz")
    if "--replot" in sys.argv:
        D = np.load(npz)
        make_figure(D["deltas"], D["frac"], D["fracsd"], D["nullm"], D["rej"],
                    D["rlo"], D["rhi"], float(D["obs_real"]), float(D["null_real"]))
        return

    comms = load_communities()
    tr = transitions(comms)

    # ---- anchor: does the fast fitter reproduce the manuscript's held-out numbers?
    obs_s, null_s, p_s = loco_descent_stat(comms, fit_symmetrizable, seed=11)
    obs_f, null_f, p_f = loco_descent_stat(comms, fit_constrained_fast_d, seed=11)
    print("real data, held-out descent protocol:")
    print(f"  published fitter  : {obs_s:.3f}  (scrambled {null_s:.3f}, p={p_s:.3f})")
    print(f"  fast fitter       : {obs_f:.3f}  (scrambled {null_f:.3f}, p={p_f:.3f})")
    if abs(obs_s - obs_f) > 0.04:
        print("  !! the two fitters disagree; the power curve would not be anchored. STOP.")
        return
    print("  -> the fast fitter is an acceptable stand-in.\n")

    r0, J0, _ = fit_constrained_fast_d(tr)
    Pdir = antisym(J0, np.random.default_rng(0))     # same direction Fig 5 uses

    # ---- calibrate the process noise, because the residual scale does not survive contact
    # with this statistic. resid_std() is the one-step residual spread, 0.450, and it mixes
    # observation noise with model error; injected as persistent process noise it makes the
    # simulated trajectories far rougher than the real EV3 series, which are means over
    # replicate cultures. Simulating a PERFECTLY symmetrizable ground truth at 0.450 gives a
    # held-out descent of 0.687 against the observed 0.859: the null is then measuring the
    # simulator, not the statistic. So we first find the sigma at which the symmetrizable
    # ground truth reproduces what the data show, and run the delta sweep there.
    s_resid = resid_std(comms, r0, J0)
    print(f"one-step residual sigma = {s_resid:.3f}  (NOT usable directly, see below)")
    print("calibrating the process noise against the observed held-out descent "
          f"{obs_s:.3f}:")
    grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, s_resid]
    cal = []
    nw = max(1, min(14, (os.cpu_count() or 4) - 2))
    for sg in grid:
        jobs = [(comms, r0, J0, sg, 900000 + 31 * k) for k in range(150)]
        with ProcessPoolExecutor(max_workers=nw) as ex:
            res = list(ex.map(one_replicate, jobs))
        o = np.array([a for a, b, c in res]); nl = np.array([b for a, b, c in res])
        cal.append((sg, o.mean(), nl.mean()))
        print(f"    sigma={sg:.3f}  descent={o.mean():.3f}  scrambled={nl.mean():.3f}  "
              f"margin={o.mean()-nl.mean():+.3f}")
    sproc = min(cal, key=lambda c: abs(c[1] - obs_s))[0]
    print(f"  -> sigma = {sproc:.3f} reproduces the observed descent; the delta sweep "
          f"uses it.\n")
    print(f"{N_REP} replicates per delta, {NPERM} scrambles each\n")

    frac, fracsd, nullm, rej, rlo, rhi = [], [], [], [], [], []
    nw = max(1, min(14, (os.cpu_count() or 4) - 2))
    for dl in DELTAS:
        Jd = J0 + dl * Pdir
        jobs = [(comms, r0, Jd, sproc, 500000 + int(dl * 1000) + 31 * k)
                for k in range(N_REP)]
        with ProcessPoolExecutor(max_workers=nw) as ex:
            res = list(ex.map(one_replicate, jobs))
        o = np.array([a for a, b, c in res])
        nl = np.array([b for a, b, c in res])
        pv = np.array([c for a, b, c in res])
        ok = np.isfinite(o)
        k = int(np.sum(pv[ok] < ALPHA))
        lo, hi = ci(k, int(ok.sum()))
        frac.append(o[ok].mean()); fracsd.append(o[ok].std())
        nullm.append(nl[ok].mean()); rej.append(k / ok.sum())
        rlo.append(lo); rhi.append(hi)
        print(f"  delta={dl:<5} descent={o[ok].mean():.3f}+/-{o[ok].std():.3f}  "
              f"scrambled={nl[ok].mean():.3f}  margin={o[ok].mean()-nl[ok].mean():+.3f}  "
              f"reject@0.05={k}/{int(ok.sum())}={k/ok.sum():.2f} [{lo:.2f},{hi:.2f}]")

    np.savez(npz, deltas=np.array(DELTAS), frac=frac, fracsd=fracsd, nullm=nullm,
             rej=rej, rlo=rlo, rhi=rhi, obs_real=obs_s, null_real=null_s)
    make_figure(DELTAS, frac, fracsd, nullm, rej, rlo, rhi, obs_s, null_s)

    print("\n" + "-" * 74)
    print(f"sensitivity  (delta=0, symmetrizable truth) : {rej[0]:.2f}")
    print(f"false positives at the largest asymmetry    : {rej[-1]:.2f}")
    if rej[0] > 0.5 and rej[-1] < rej[0] / 2:
        print("=> the statistic separates gradient from non-gradient ground truths: the")
        print("   headline claim has the specificity the predictive test lacked.")
    else:
        print("=> the statistic does NOT separate them: it rejects the scrambled null at a")
        print("   similar rate whether or not the truth is symmetrizable, so the real-data")
        print("   p = 0.002 cannot be read as evidence FOR a weighted gradient flow.")
    print("-" * 74)


if __name__ == "__main__":
    main()
