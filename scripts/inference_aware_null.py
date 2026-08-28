"""
Inference-aware (end-to-end) null for the symmetrizability diagnostics  (referee-6 P0.2)
========================================================================================

The referee's central statistical demand: the earlier null (a symmetric matrix + additive
coefficient noise) is a noise model on PARAMETERS, not a parametric bootstrap of the
EXPERIMENT. The decisive test regenerates data under a symmetrizable model and pushes it
through the whole inference:

  1. fit a diagonally symmetrizable gLV (J = D^{-1} K, K=K^T, d>0) to the RAW Venturelli EV3
     trajectories;
  2. estimate the process-noise scale from that fit's one-step residuals;
  3. simulate new trajectories from the symmetrizable model with matched noise, keeping the
     same 13 leave-one-species-out community design and initial conditions;
  4. RE-FIT the FREE gLV with the identical pipeline (same ridge, same design);
  5. compute f_sign and r_cycle on the re-fitted matrix.

If inference noise on a truly symmetrizable system reproduces the observed sign-discordance
and cycle-inconsistency, the point-estimate asymmetry is an inference artefact; if the
observed values sit in the extreme tail of this end-to-end null, the data genuinely resist a
symmetrizable description. This is the null the earlier coefficient-noise version approximated.

Run:  python3 inference_aware_null.py     (writes figures/fig_inference_null.png)
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import constrained_vs_free_glv as CF
from constrained_vs_free_glv import (load_communities, transitions, fit_free, fit_constrained,
                                     invariants, pred_mse, S, EPS, ABMIN, LAM)
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"
CAP = 8.0


def resid_std(comms, r, J):
    """Std of one-step log-growth residuals of a fitted model over the real transitions."""
    res = []
    for a, g, m in transitions(comms):
        pred = r + J @ a
        res.extend((g[m] - pred[m]).tolist())
    return float(np.std(res))


def simulate(comms, r, J, sproc, rng):
    """Regenerate each community's trajectory from the fitted gLV with matched process noise,
    from the observed initial condition, keeping absent (dropped) species near zero."""
    out = []
    for name, X in comms:
        present = X.mean(0) > 1e-3
        T = len(X); traj = np.zeros((T, S)); x = X[0].copy(); traj[0] = x
        for t in range(1, T):
            drift = r + J @ x
            x = np.clip(np.exp(np.log(x + EPS) + drift + sproc * rng.standard_normal(S)), 0, CAP)
            x[~present] = X[0][~present]
            traj[t] = x
        out.append((name, traj))
    return out


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    comms = load_communities()
    tr = transitions(comms)
    # observed FREE fit (the point estimate whose asymmetry is in question)
    rf, Jf = fit_free(tr); fsign_obs, rcyc_obs = invariants(Jf)
    # symmetrizable ground truth from the real data + its residual noise scale
    rc, Jc = fit_constrained(tr); sproc = resid_std(comms, rc, Jc)
    print(f"observed free fit:  f_sign={fsign_obs:.3f}  r_cycle={rcyc_obs:.3f}")
    print(f"symmetrizable fit:  f_sign={invariants(Jc)[0]:.3f}  r_cycle={invariants(Jc)[1]:.3f} "
          f"(=0 by construction);  process-noise sigma={sproc:.3f}\n")

    # end-to-end null: simulate from the symmetrizable model, re-fit free, recompute invariants
    N = 300
    fs_null, rc_null = [], []
    for i in range(N):
        sim = simulate(comms, rc, Jc, sproc, np.random.default_rng(1000 + i))
        Y = transitions(sim)
        if len(Y) < 10:
            continue
        _, Jn = fit_free(Y)
        f, r = invariants(Jn)
        fs_null.append(f); rc_null.append(r)
    fs_null = np.array(fs_null); rc_null = np.array(rc_null)
    rc_null = rc_null[np.isfinite(rc_null)]

    def report(name, obs, null):
        b = int(np.sum(null >= obs)); B = len(null)
        p = (b + 1) / (B + 1)
        print(f"  {name:8}: null median={np.median(null):.3f} [95%: {np.percentile(null,95):.3f}]  "
              f"obs={obs:.3f}  ->  {b}/{B} exceed, p_MC={p:.4f}")
        return p
    print(f"end-to-end null (simulate symmetrizable -> re-fit free, N={len(fs_null)}):")
    p_fs = report("f_sign", fsign_obs, fs_null)
    p_rc = report("r_cycle", rcyc_obs, rc_null)
    verdict = ("observed asymmetry EXCEEDS the inference-aware null -> data resist a "
               "symmetrizable description" if (p_fs < 0.05 or p_rc < 0.05) else
               "observed asymmetry is within the inference-aware null -> the point-estimate "
               "asymmetry is consistent with inference of a symmetrizable system")
    print(f"\nverdict: {verdict}")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for a, obs, null, lab, pv in [(ax[0], fsign_obs, fs_null, r"$f_{\rm sign}$", p_fs),
                                  (ax[1], rcyc_obs, rc_null, r"$r_{\rm cycle}$", p_rc)]:
        a.hist(null, bins=30, color="#bbbbbb", alpha=0.7, density=True,
               label="symmetrizable\n+ full inference")
        a.axvline(obs, color="#D55E00", lw=2, label=f"observed = {obs:.2f}")
        a.set_xlabel(lab); a.set_ylabel("density")
        a.set_title(f"{lab}: end-to-end null ($p={pv:.3f}$)", loc="left", fontsize=11)
        a.legend(frameon=False, fontsize=8.5)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.suptitle("Inference-aware null: regenerate under a symmetrizable gLV, re-fit free",
                 fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_inference_null.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_inference_null.png written")
    np.savez(f"{OUT}/inference_null.npz", fsign_obs=fsign_obs, rcyc_obs=rcyc_obs,
             fs_null=fs_null, rc_null=rc_null, p_fs=p_fs, p_rc=p_rc, sproc=sproc)


if __name__ == "__main__":
    main()
