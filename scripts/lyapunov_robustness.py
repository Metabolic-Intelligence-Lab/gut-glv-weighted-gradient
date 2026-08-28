"""
Robustness of the H_D descent result
====================================

lyapunov_descent.py finds the real EV3 trajectories descending the fitted weighted potential
on 65/78 transitions (0.83) against 0.73 for potentials of the same magnitudes with the
structure scrambled (p = 0.025). Three things could be wrong with that, and the first is
serious.

(1) IN-SAMPLE ADVANTAGE.  The fitted potential was fitted to these very trajectories, and the
    scrambled comparators were not. A model that predicts a trajectory well will tend to have
    that trajectory descend its potential, so the contrast may be measuring fit quality rather
    than gradient structure. The fix is out-of-sample: for each community, fit the
    symmetrizable model on the OTHER twelve and score descent on the held-out one. Nothing is
    then fitted to what it is scored on. This is the check the claim lives or dies by.

(2) CLUSTERING.  The 78 transitions sit inside 13 communities and are not independent. We
    therefore also resample whole communities, so the effective n is 13, and report the
    per-community descent fractions so it is visible whether one community carries the result.

(3) REGULARISATION.  The ridge penalty sets how much structure the fit is allowed; the
    predictive test already reports lambda in {0.25, 0.5, 1} and this should too.

Run:  python3 scripts/lyapunov_robustness.py   (writes figures/fig_lyapunov_robustness.png)
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import (                                   # noqa: E402
    load_communities, transitions, fit_free, S,
)
from lyapunov_descent import fit_symmetrizable, descent_fraction        # noqa: E402

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = "figures"
NPERM = 5000     # was 400, at which zero exceedances pinned every p at the 1/401 floor
NBOOT = 2000


def fmt_p(k, n):
    """Monte-Carlo p as text. With zero exceedances the estimate is a resolution floor,
    not a value, and printing '0.002' for 0/400 invites it to be read as one."""
    return f"< {1.0/(n+1):.4f}" if k == 0 else f"= {(k+1)/(n+1):.4f}"


def scramble(K, d, rng):
    """A potential with the same off-diagonal magnitudes and the structure destroyed."""
    iu = np.triu_indices(S, 1)
    Ks = np.zeros((S, S))
    Ks[iu] = rng.permutation(K[iu])
    Ks = Ks + Ks.T
    np.fill_diagonal(Ks, np.diag(K))
    return Ks / d[:, None]


def symmetrizer_lsq(J):
    """The positive diagonal that comes closest to symmetrizing J.

    DJ is symmetric iff d_i J_ij = d_j J_ji, i.e. iff b_ij = log|J_ij/J_ji| equals
    u_j - u_i with u = log d. For a J that is not exactly symmetrizable there is no
    such u, and the least-squares u is the best a reader could do if they took the
    unconstrained fit at face value and wrote down its potential anyway. (Its RMS
    residual is r_cycle, the diagnostic of the main text.)
    """
    rows, b = [], []
    for i in range(S):
        for j in range(i + 1, S):
            a, c = J[i, j], J[j, i]
            if a == 0 or c == 0 or np.sign(a) != np.sign(c):
                continue
            e = np.zeros(S); e[j] = 1.0; e[i] = -1.0
            rows.append(e); b.append(np.log(abs(a) / abs(c)))
    if not rows:
        return np.ones(S)
    u, *_ = np.linalg.lstsq(np.array(rows), np.array(b), rcond=None)
    return np.exp(u - u.mean())


def loco_descent_free(comms, lam=0.5):
    """The same held-out protocol, scoring the potential of the UNCONSTRAINED fit.

    The scrambled control destroys structure but also destroys fit quality, so it
    cannot by itself separate "these trajectories descend a potential" from "the
    constrained model fits well". The free fit is the missing comparator: it
    predicts held-out communities as well as the constrained one (CV MSE 0.240 vs
    0.241) but was never restricted to the symmetrizable manifold, so its potential
    is the one a reader gets by ignoring the constraint entirely.
    """
    down = tot = 0
    per = []
    for h in range(len(comms)):
        train = transitions([c for k, c in enumerate(comms) if k != h])
        r, J = fit_free(train, lam)
        d = symmetrizer_lsq(J)
        f, n, _ = descent_fraction([comms[h]], r, J, d)
        per.append(f)
        down += f * n
        tot += n
    return down / tot, np.array(per)


def loco_descent(comms, lam=0.5, nperm=NPERM):
    """Out-of-sample: fit on the other communities, score descent on the held-out one.

    Returns the pooled out-of-sample descent fraction, the per-community fractions, and the
    null distribution obtained by scrambling each fold's potential.
    """
    down = tot = 0
    per_comm = []
    null_pooled = np.zeros(nperm)
    null_tot = 0
    for h in range(len(comms)):
        train = transitions([c for k, c in enumerate(comms) if k != h])
        r, J, d = fit_symmetrizable(train, lam)
        f, n, _ = descent_fraction([comms[h]], r, J, d)
        per_comm.append((comms[h][0], f, n))
        down += f * n
        tot += n
        K = d[:, None] * J
        for i in range(nperm):
            rng = np.random.default_rng(80000 + 997 * h + i)
            fs, ns, _ = descent_fraction([comms[h]], r, scramble(K, d, rng), d)
            null_pooled[i] += fs * ns
        null_tot += n
    return down / tot, np.array(per_comm, dtype=object), null_pooled / null_tot, tot


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    comms = load_communities()

    # ---------------------------------------------------------------- in-sample reference
    r, J, d = fit_symmetrizable(transitions(comms))
    obs_in, ntrans, _ = descent_fraction(comms, r, J, d)
    print(f"in-sample descent fraction (for reference): {obs_in:.3f}  ({ntrans} transitions)\n")

    # ---------------------------------------------------------------- (1) out-of-sample
    print("(1) OUT-OF-SAMPLE -- fit on 12 communities, score the 13th")
    oos, per_comm, oos_null, n_oos = loco_descent(comms)
    k_oos = int(np.sum(oos_null >= oos))
    p_oos = (k_oos + 1) / (len(oos_null) + 1)
    print(f"    held-out descent fraction : {oos:.3f}   ({n_oos} transitions)")
    print(f"    scrambled-structure null  : {oos_null.mean():.3f} "
          f"[{np.percentile(oos_null,2.5):.3f}, {np.percentile(oos_null,97.5):.3f}]")
    print(f"    margin                    : {oos - oos_null.mean():+.3f}")
    print(f"    p(scrambled >= observed)  : {fmt_p(k_oos, len(oos_null))}  "
          f"({k_oos}/{len(oos_null)} exceedances)")
    if p_oos < 0.05:
        print("    => survives out of sample: the contrast is not an in-sample artefact.")
    else:
        print("    => DOES NOT survive out of sample: the in-sample contrast was fit quality,")
        print("       not gradient structure. The affirmative claim cannot be made.")

    # ---------------------------------------------------------------- (2) clustering
    print("\n(2) CLUSTERING -- per-community held-out descent, and a community-level bootstrap")
    for name, f, n in per_comm:
        print(f"    {str(name):>6} : {f:.2f}  ({int(n)} transitions)")
    fr = np.array([f for _, f, _ in per_comm], dtype=float)
    wt = np.array([n for _, _, n in per_comm], dtype=float)
    rng = np.random.default_rng(90210)
    boot = []
    for _ in range(NBOOT):
        idx = rng.integers(0, len(fr), len(fr))
        boot.append(np.sum(fr[idx] * wt[idx]) / np.sum(wt[idx]))
    boot = np.array(boot)
    print(f"    community-level bootstrap 95% CI on the held-out fraction: "
          f"[{np.percentile(boot,2.5):.3f}, {np.percentile(boot,97.5):.3f}]")
    print(f"    communities above the scrambled-null mean: "
          f"{int(np.sum(fr > oos_null.mean()))}/{len(fr)}")

    # -------------------------------------------------- (2b) the free-fit comparator
    print("\n(2b) FREE-FIT POTENTIAL -- does the symmetrizability constraint buy the descent?")
    oos_free, per_free = loco_descent_free(comms)
    diff = fr - per_free                       # per community, constrained minus free
    # exact two-sided sign-flip null over all 2^13 patterns; the tolerance keeps the
    # observed pattern itself inside the tail, where rounding would otherwise drop it
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=len(diff))))
    p_free = float(np.mean(np.abs(signs @ diff) / len(diff) >= abs(diff.mean()) - 1e-12))
    print(f"    held-out descent, constrained fit : {oos:.3f}")
    print(f"    held-out descent, free fit        : {oos_free:.3f}")
    print(f"    scrambled-structure null          : {oos_null.mean():.3f}")
    print(f"    paired over the 13 communities    : mean gap {diff.mean():+.3f}, "
          f"constrained ahead in {int((diff > 0).sum())}/{len(diff)}, "
          f"exact sign-flip p={p_free:.4f}")
    print("    => RETAINED AS A DIAGNOSTIC ONLY. This contrast was panel (d) and was read as")
    print("       'the constraint, not the fit quality, is what the trajectories track'. It")
    print("       does not support that. descent_best_case_D.py shows the free fit's low")
    print("       figure is a property of symmetrizer_lsq, which is fitted on 30 of 66 pairs")
    print("       because it must drop every sign-discordant one: give the free fit a d")
    print("       chosen on training data alone and it reaches 0.885, above the constrained")
    print("       0.859. Every one of the 78 transitions has its sign decided by d, so no")
    print("       comparison that lets the two arms use different symmetrizers is meaningful.")
    print("       Panel (d) now carries the end-to-end specificity curve instead.")

    # ---------------------------------------------------------------- (3) ridge
    print("\n(3) REGULARISATION -- out-of-sample result across the ridge penalty")
    ridge = {}
    for lam in (0.25, 0.5, 1.0):
        o, _, nl, _ = loco_descent(comms, lam=lam, nperm=2000)
        k = int(np.sum(nl >= o))
        ridge[lam] = (o, nl.mean(), (k + 1) / (len(nl) + 1))
        print(f"    lambda={lam:<5}: held-out {o:.3f}   scrambled {nl.mean():.3f}   "
              f"margin {o-nl.mean():+.3f}   p {fmt_p(k, len(nl))}")
    marg = [ridge[l][0] - ridge[l][1] for l in sorted(ridge)]
    print(f"    -> the MARGIN, which is the claim, runs {marg[0]:+.3f} {marg[1]:+.3f} "
          f"{marg[2]:+.3f}: it shrinks {marg[0]/marg[2]:.1f}x across the sweep, so the")
    print("       result is directionally stable but the effect size is not.")

    # ------------------------------------------------------------------ figure
    # 2x2 rather than 1x4: at journal column width four panels in a row shrink the
    # tick labels past legibility.
    fig, axes2 = plt.subplots(2, 2, figsize=(11.4, 8.4))
    ax = axes2.ravel()

    ax[0].hist(oos_null, bins=26, color="#4C72B0", alpha=.8,
               label=f"structure scrambled\n(N={len(oos_null)})")
    ax[0].axvline(oos, color="#D55E00", lw=2.5, label=f"held-out = {oos:.2f}")
    ax[0].set_xlabel("out-of-sample fraction with $\\Delta H_D \\leq 0$")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"a  Out of sample, nothing is fitted to what it scores\n"
                    f"margin +{oos - oos_null.mean():.3f}, $p$ {fmt_p(k_oos, len(oos_null))}, "
                    f"$n$ = {n_oos} held-out transitions", loc="left", fontsize=10.5)
    ax[0].legend(fontsize=8, frameon=False)

    o = np.argsort(fr)
    ax[1].barh(range(len(fr)), fr[o], color="#4C72B0")
    ax[1].axvline(oos_null.mean(), color="#888", ls="--", lw=1.2, label="scrambled null")
    ax[1].axvline(oos, color="#D55E00", lw=2, label="pooled held-out")
    ax[1].set_yticks(range(len(fr)))
    ax[1].set_yticklabels([str(per_comm[i][0]) for i in o], fontsize=7)
    ax[1].set_xlabel("held-out descent fraction")
    ax[1].set_title("b  Per community: is one carrying it?", loc="left", fontsize=10.5)
    ax[1].legend(fontsize=8, frameon=False, loc="lower right")

    # The claim is the MARGIN, not the observed fraction, so annotate the margin: it is
    # 0.139, 0.101 and 0.047 across the sweep, and quoting only the observed 0.87/0.86/0.81
    # (or only the p-values, which sit at the Monte-Carlo floor at every lambda) would let a
    # reader take a threefold shrinkage for stability.
    lams = sorted(ridge)
    ax[2].plot(lams, [ridge[l][0] for l in lams], "o-", color="#D55E00", label="held-out")
    ax[2].plot(lams, [ridge[l][1] for l in lams], "s--", color="#4C72B0", label="scrambled")
    ax[2].fill_between(lams, [ridge[l][1] for l in lams], [ridge[l][0] for l in lams],
                       color="#D55E00", alpha=.13)
    for l in lams:
        ax[2].annotate(f"+{ridge[l][0]-ridge[l][1]:.3f}", (l, ridge[l][0]),
                       textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax[2].set_xlabel("ridge penalty $\\lambda$")
    ax[2].set_ylabel("descent fraction")
    ax[2].set_ylim(0.4, 1.0)
    ax[2].set_title("c  Sensitivity to regularisation\nthe margin shrinks threefold "
                    "across the sweep", loc="left", fontsize=10.5)
    ax[2].legend(fontsize=8, frameon=False)

    # (d) used to contrast the constrained fit against the free fit's least-squares
    # symmetrizer. That comparison does not survive descent_best_case_D.py: the free fit's
    # 0.65 is a property of symmetrizer_lsq, which is fitted on 30 of 66 pairs, and a d
    # chosen on training data alone brings the free fit to 0.885, above the constrained
    # 0.859. The panel now carries the end-to-end specificity curve instead, which is what
    # the claim actually needs: does the statistic track the ground truth?
    spec = os.path.join(OUT, "descent_specificity.npz")
    if os.path.exists(spec):
        D = np.load(spec)
        dl, rej, lo, hi = D["deltas"], D["rej"], D["rlo"], D["rhi"]
        ax[3].errorbar(dl, rej, yerr=[rej - lo, hi - rej], fmt="o-", color="#0072B2",
                       lw=2, ms=7, capsize=4)
        ax[3].axhline(0.05, color="grey", ls=":", lw=1, label="nominal level")
        ax[3].axhline(0.8, color="#009E73", ls="-.", lw=1, label="conventional 0.8")
        ax[3].set_xlabel("injected antisymmetric perturbation $\\delta$\n"
                         "($\\delta=0$: symmetrizable truth)")
        ax[3].set_ylabel("P(reject scrambled null)")
        ax[3].set_ylim(0, 1)
        ax[3].set_title("d  Specificity: the statistic tracks the ground truth\n"
                        f"{rej[0]:.2f} under a symmetrizable truth, {rej[-1]:.2f} at the "
                        "largest asymmetry", loc="left", fontsize=10.5)
        ax[3].legend(fontsize=8, frameon=False, loc="upper right")
    else:
        ax[3].text(.5, .5, "run descent_specificity.py first", ha="center",
                   transform=ax[3].transAxes)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_lyapunov_robustness.png", dpi=150)
    plt.close(fig)
    np.savez(f"{OUT}/lyapunov_robustness.npz", in_sample=obs_in, oos=oos, oos_null=oos_null,
             p_oos=p_oos, per_comm_frac=fr, per_comm_n=wt, boot=boot,
             oos_free=oos_free, per_comm_free=per_free, p_free=p_free,
             ridge=np.array([[l, *ridge[l]] for l in lams]))
    print(f"\nsaved {OUT}/fig_lyapunov_robustness.png")


if __name__ == "__main__":
    main()
