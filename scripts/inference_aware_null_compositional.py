"""
The inference-aware null, with the compositional constraint restored
====================================================================

inference_aware_null.py builds the null for f_sign and r_cycle by simulating from a fitted
symmetrizable gLV and re-fitting a free model. Its simulator propagates log-abundances with
process noise and never restores the constraint that a sequencing measurement imposes: the
real EV3 samples sum to 1.000 exactly, its simulated samples range over 0.44 to 2.13. The
simulated data are therefore not the kind of data the estimator was ever shown, and the null
is built on the wrong sample space.

The fix is to apply, at each simulated observation, the operation the measurement applies:
renormalise to unit total. Sequencing returns proportions; the null should too.

This matters because the paper's headline rests on that null. If it is too wide, the finding
that the observed asymmetry sits INSIDE it is conservative rather than wrong, and tightening
it can only sharpen the conclusion, possibly reversing it. This script runs both simulators
side by side so the size of the effect is visible rather than argued about.

Run:  python3 scripts/inference_aware_null_compositional.py
      (writes figures/fig_infnull_compositional.png)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import (          # noqa: E402
    load_communities, transitions, fit_free, fit_constrained, invariants, S, EPS,
)
from inference_aware_null import resid_std, simulate as simulate_free, CAP   # noqa: E402

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = "figures"
N = 300


def simulate_compositional(comms, r, J, sproc, rng):
    """As inference_aware_null.simulate, but each observation is renormalised to unit total.

    That is the measurement operation the real data went through, and without it the null
    lives off the simplex the estimator was calibrated on.
    """
    out = []
    for name, X in comms:
        present = X.mean(0) > 1e-3
        T = len(X)
        traj = np.zeros((T, S))
        x = X[0].copy()
        traj[0] = x
        for t in range(1, T):
            drift = r + J @ x
            x = np.clip(np.exp(np.log(x + EPS) + drift + sproc * rng.standard_normal(S)),
                        0, CAP)
            x[~present] = X[0][~present]
            s = x.sum()
            if s > 0:
                x = x / s                      # <- the measurement
            traj[t] = x
        out.append((name, traj))
    return out


def run_null(comms, rc, Jc, sproc, sim_fn, seed0):
    fs, rr, tot = [], [], []
    for i in range(N):
        sim = sim_fn(comms, rc, Jc, sproc, np.random.default_rng(seed0 + i))
        tot.extend([X.sum(1).mean() for _, X in sim])
        Y = transitions(sim)
        if len(Y) < 10:
            continue
        _, Jn = fit_free(Y)
        f, r = invariants(Jn)
        fs.append(f)
        rr.append(r)
    fs = np.array(fs)
    rr = np.array(rr)
    return fs, rr[np.isfinite(rr)], float(np.mean(tot))


def report(name, obs, null):
    b = int(np.sum(null >= obs))
    p = (b + 1) / (len(null) + 1)
    print(f"    {name:8}: null median={np.median(null):.3f}  95th={np.percentile(null,95):.3f}"
          f"   obs={obs:.3f}   {b}/{len(null)} exceed   p={p:.4f}")
    return p


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    comms = load_communities()
    tr = transitions(comms)
    rf, Jf = fit_free(tr)
    fsign_obs, rcyc_obs = invariants(Jf)
    rc, Jc = fit_constrained(tr)
    sproc = resid_std(comms, rc, Jc)
    real_total = float(np.mean([X.sum(1).mean() for _, X in comms]))
    print(f"observed free fit: f_sign={fsign_obs:.3f}  r_cycle={rcyc_obs:.3f}")
    print(f"real samples sum to {real_total:.3f} on average; process-noise sigma={sproc:.3f}\n")

    print("A. published null -- simulator does NOT preserve the composition")
    fs_a, rc_a, tot_a = run_null(comms, rc, Jc, sproc, simulate_free, 1000)
    print(f"    simulated samples sum to {tot_a:.3f} on average")
    p_fs_a = report("f_sign", fsign_obs, fs_a)
    p_rc_a = report("r_cycle", rcyc_obs, rc_a)

    print("\nB. corrected null -- each observation renormalised, as sequencing does")
    fs_b, rc_b, tot_b = run_null(comms, rc, Jc, sproc, simulate_compositional, 2000)
    print(f"    simulated samples sum to {tot_b:.3f} on average")
    p_fs_b = report("f_sign", fsign_obs, fs_b)
    p_rc_b = report("r_cycle", rcyc_obs, rc_b)

    print("\nCONSEQUENCE")
    flip = ((p_fs_a >= 0.05) != (p_fs_b >= 0.05)) or ((p_rc_a >= 0.05) != (p_rc_b >= 0.05))
    for nm, pa, pb in (("f_sign", p_fs_a, p_fs_b), ("r_cycle", p_rc_a, p_rc_b)):
        print(f"    {nm:8}: p {pa:.3f} -> {pb:.3f}")
    if flip:
        print("    The correction CHANGES the verdict at alpha=0.05. The published null was")
        print("    too wide, and the observed asymmetry is not inside the corrected one.")
    else:
        print("    The verdict is unchanged: the observed asymmetry still sits inside the")
        print("    null once the simulator is corrected, so the published conclusion holds")
        print("    and is now resting on the right sample space.")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for k, (nm, obs, a, b) in enumerate((("$f_{\\rm sign}$", fsign_obs, fs_a, fs_b),
                                         ("$r_{\\rm cycle}$", rcyc_obs, rc_a, rc_b))):
        ax[k].hist(a, bins=26, color="#999999", alpha=.65,
                   label="published null\n(leaves the simplex)")
        ax[k].hist(b, bins=26, color="#4C72B0", alpha=.75,
                   label="corrected null\n(renormalised)")
        ax[k].axvline(obs, color="#D55E00", lw=2.5, label=f"observed = {obs:.2f}")
        ax[k].set_xlabel(nm)
        ax[k].set_ylabel("count")
        ax[k].legend(fontsize=8, frameon=False)
    ax[0].set_title(f"a  Sign discordance   $p$: {p_fs_a:.2f} $\\rightarrow$ {p_fs_b:.2f}",
                    loc="left", fontsize=11)
    ax[1].set_title(f"b  Cycle inconsistency   $p$: {p_rc_a:.2f} $\\rightarrow$ {p_rc_b:.2f}",
                    loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_infnull_compositional.png", dpi=150)
    plt.close(fig)
    np.savez(f"{OUT}/infnull_compositional.npz",
             fsign_obs=fsign_obs, rcyc_obs=rcyc_obs,
             fs_published=fs_a, rc_published=rc_a, fs_corrected=fs_b, rc_corrected=rc_b,
             p=(p_fs_a, p_rc_a, p_fs_b, p_rc_b))
    print(f"\nsaved {OUT}/fig_infnull_compositional.png")


if __name__ == "__main__":
    main()
