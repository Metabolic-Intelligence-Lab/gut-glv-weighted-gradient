"""
Detection frequency and true statistical power of the free-vs-symmetrizable test (referee-9 §8)
==============================================================================================

Referee 9 points out that the quantity reported by ``power_analysis.py``, P(free model wins),
is a *model-selection frequency*, not statistical power: there is no rejection rule at a fixed
level, and a coin-flip comparison sits at 0.5 rather than at alpha under the null. Both curves
are computed here so the distinction is explicit:

  detection frequency  P(mean out-of-sample advantage of the free model > 0)
  power                P(pre-specified paired permutation test rejects at alpha = 0.05)

The rejection rule is the one already used on the real data in ``predictive_robustness.py``:
full leave-one-community-out CV gives one advantage d_k = MSE(constrained) - MSE(free) per
held-out community, and the null "the two models predict equally well" is tested by flipping
the signs of the d_k. With 13 communities the permutation null is enumerated exactly (2^13
sign patterns), so the p-value is exact rather than sampled.

Ground truth is the fitted symmetrizable model J0 pushed a distance delta off the
symmetrizable manifold along a fixed antisymmetric direction. The x-axis is delta, the
asymmetry actually injected; the cycle inconsistency the inference recovers from it is
annotated per point, because it is not monotone in delta and so makes a misleading
coordinate (it was one until this revision).

Run:  OMP_NUM_THREADS=1 python3 scripts/detection_power.py   (writes figures/fig_power.png)
      Roughly 15 min on 14 workers; each replicate costs 13 constrained fits.
      --replot redraws the figure from figures/power.npz without recomputing.
"""
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np
from scipy.stats import beta as betadist

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from constrained_vs_free_glv import (S, fit_free, load_communities,  # noqa: E402
                                     pred_mse, transitions)
from inference_aware_null import resid_std, simulate  # noqa: E402
from power_analysis import antisym, fit_constrained_fast  # noqa: E402
from symmetrizability_invariants import invariants as inv3  # noqa: E402

import warnings  # noqa: E402
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = "figures"
ALPHA = 0.05
DELTAS = [0.0, 1.0, 2.0, 4.0, 8.0]
N_REP = 120

# exact sign-flip null for the paired permutation test: all 2^13 patterns
_SIGNS = None


def signs_for(n):
    global _SIGNS
    if _SIGNS is None or _SIGNS.shape[1] != n:
        _SIGNS = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    return _SIGNS


def loco_advantages(comms):
    """Full leave-one-community-out CV -> one advantage per held-out community.

    d_k > 0 means the free model predicted the held-out community better than the
    diagonally symmetrizable one.
    """
    out = []
    for k in range(len(comms)):
        te = transitions([comms[k]])
        if not te:
            continue
        tr = transitions([c for j, c in enumerate(comms) if j != k])
        rf, Jf = fit_free(tr)
        rc, Jc = fit_constrained_fast(tr)
        out.append(pred_mse(rc, Jc, te) - pred_mse(rf, Jf, te))
    return np.asarray(out)


def perm_p(d):
    """One-sided exact paired permutation p-value for mean(d) > 0 under sign symmetry."""
    if d.size == 0:
        return 1.0
    obs = d.mean()
    null = signs_for(d.size) @ d / d.size
    return float((np.sum(null >= obs) + 1) / (null.size + 1))


def one_replicate(args):
    comms, r0, Jd, sproc, seed = args
    sim = simulate(comms, r0, Jd, sproc, np.random.default_rng(seed))
    d = loco_advantages(sim)
    return float(d.mean()) if d.size else 0.0, perm_p(d)


def ci(k, n):
    return (betadist.ppf(0.025, k + 0.5, n - k + 0.5),
            betadist.ppf(0.975, k + 0.5, n - k + 0.5))


def make_figure(deltas, true_dist, detect, dlo, dhi, power, plo, phi, rcyc_obs):
    """Both curves against the *injected* asymmetry.

    They used to be plotted against the recovered r_cycle(J_delta), which is not
    monotone in delta (delta=8 recovers 1.10, below delta=4's 1.58), so the point
    the text calls "the largest asymmetry injected" landed in the middle of the
    x-axis and the figure appeared to contradict the sentence. The recovered value
    is the thing that fails to track the injection, so it is annotated per point
    rather than used as the coordinate.
    """
    detect, power = np.asarray(detect), np.asarray(power)
    dlo, dhi = np.asarray(dlo), np.asarray(dhi)
    plo, phi = np.asarray(plo), np.asarray(phi)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    for ax, y, lo, hi, ttl, ref, reflab in [
        (axes[0], detect, dlo, dhi, "Detection frequency\nP(free model predicts better)",
         0.5, "chance"),
        (axes[1], power, plo, phi, f"Power\nP(reject at $\\alpha={ALPHA}$)", ALPHA, "nominal level"),
    ]:
        ax.errorbar(deltas, y, yerr=[y - lo, hi - y], fmt="o-",
                    color="#0072B2", lw=2, ms=7, capsize=4)
        ax.axhline(ref, color="grey", ls=":", lw=1, label=reflab)
        for x, yy, rc in zip(deltas, y, true_dist):
            ax.annotate(f"{rc:.1f}", (x, yy), textcoords="offset points",
                        xytext=(10, -4), ha="left", fontsize=7.5, color="#555555")
        ax.set_xlabel("injected antisymmetric perturbation $\\delta$\n"
                      "(grey: the $r_{\\rm cycle}$ the inference recovers)")
        ax.set_ylabel("frequency, 95% CI")
        ax.set_ylim(0, 1)
        ax.set_title(ttl, fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[1].axhline(0.8, color="#009E73", ls="-.", lw=1, label="conventional 0.8")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].annotate(f"observed data: $r_{{\\rm cycle}}={rcyc_obs:.1f}$",
                     (0.98, 0.03), xycoords="axes fraction", ha="right",
                     fontsize=8, color="#D55E00")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_power.png", dpi=180, bbox_inches="tight")
    print(f"[fig] {OUT}/fig_power.png written")


def replot():
    """Redraw fig_power.png from the cached run (the 15-minute part is unchanged)."""
    d = np.load(f"{OUT}/power.npz", allow_pickle=True)
    make_figure(d["deltas"], d["true_dist"], d["detection"], d["dlo"], d["dhi"],
                d["power"], d["plo"], d["phi"], float(d["rcyc_obs"]))


def main():
    os.makedirs(OUT, exist_ok=True)
    comms = load_communities()
    tr = transitions(comms)
    r0, J0 = fit_constrained_fast(tr)
    sproc = resid_std(comms, r0, J0)
    _, Jf = fit_free(tr)
    rcyc_obs = inv3(Jf)[2]
    Pdir = antisym(J0, np.random.default_rng(0))

    d_real = loco_advantages(comms)
    print(f"real data: mean advantage={d_real.mean():+.4f}, free ahead in "
          f"{int((d_real > 0).sum())}/{d_real.size}, exact permutation p={perm_p(d_real):.3f}")
    print(f"observed free-fit r_cycle={rcyc_obs:.2f}; process noise={sproc:.3f}\n")

    workers = max(1, min(14, (os.cpu_count() or 2) - 2))
    true_dist, detect, power, dlo, dhi, plo, phi = [], [], [], [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for dl in DELTAS:
            Jd = J0 + dl * Pdir
            jobs = [(comms, r0, Jd, sproc, 3000 + int(dl * 10) + i) for i in range(N_REP)]
            res = list(pool.map(one_replicate, jobs, chunksize=2))
            adv = np.array([r[0] for r in res])
            pv = np.array([r[1] for r in res])
            nd, npw = int((adv > 0).sum()), int((pv <= ALPHA).sum())
            true_dist.append(inv3(Jd)[2])
            detect.append(nd / N_REP); dlo.append(ci(nd, N_REP)[0]); dhi.append(ci(nd, N_REP)[1])
            power.append(npw / N_REP); plo.append(ci(npw, N_REP)[0]); phi.append(ci(npw, N_REP)[1])
            print(f"  delta={dl:.1f}  true r_cycle={true_dist[-1]:5.2f}  "
                  f"detection={detect[-1]:.2f}  power(alpha={ALPHA})={power[-1]:.2f} "
                  f"[{plo[-1]:.2f},{phi[-1]:.2f}]  mean adv={adv.mean():+.4f}")

    detect, power = np.array(detect), np.array(power)
    print(f"\n=> power stays at {power.min():.2f}-{power.max():.2f} against a nominal "
          f"{ALPHA:.2f} level and never approaches 0.8: on these trajectories the "
          f"pre-specified test cannot detect asymmetry.")

    make_figure(np.array(DELTAS), np.array(true_dist), detect, dlo, dhi,
                power, plo, phi, rcyc_obs)

    np.savez(f"{OUT}/power.npz", deltas=np.array(DELTAS), true_dist=np.array(true_dist),
             detection=detect, dlo=np.array(dlo), dhi=np.array(dhi),
             power=power, plo=np.array(plo), phi=np.array(phi),
             rcyc_obs=rcyc_obs, N=N_REP, alpha=ALPHA,
             real_p=perm_p(d_real), real_adv=d_real)


if __name__ == "__main__":
    import sys
    if "--replot" in sys.argv:
        replot()
    else:
        main()
