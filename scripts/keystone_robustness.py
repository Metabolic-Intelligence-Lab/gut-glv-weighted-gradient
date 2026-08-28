"""
Keystone validation: uncertainty and baselines  (referee-4 P1, §13)
===================================================================

The leave-one-out ground-truth check (venturelli_keystone_validation.py) reports a moderate
Spearman rho~0.58 (p~0.05) between the measured-gLV knockout ranking and the real EV3
knockouts, surviving an abundance control (partial rho~0.51). Referee 4 rightly asks for
the uncertainty of an n=12, borderline result and for a comparison against trivial
predictors. This script adds, reusing the same impact vectors:

  (1) a permutation p-value for rho (shuffle the empirical ranking);
  (2) a bootstrap 95% CI for rho and the abundance-partial rho (resample the 12 species);
  (3) leave-one-species-out influence: drop each species, recompute rho, report the range
      (is the correlation carried by a single point?);
  (4) baselines: rho(empirical impact, X) for X = removed-species abundance, interaction
      degree (row+col L1 strength of the gLV matrix), and self-limitation |a_ii|, so the
      network knockout can be judged against trivial predictors.

Honest framing: a one-community, n=12 proof of concept with a wide interval, not a validated
method. Run:  python3 keystone_robustness.py     (writes figures/fig_keystone_robust.png)
"""
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from venturelli_keystone_validation import (load_glv, load_leaveoneout, steady, bc,
                                            partial_spearman, SP, S, DATA)
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"


def impacts():
    u, A = load_glv(f"{DATA}/twelveSpeciesT4.xml")
    comm = load_leaveoneout()
    full_emp = comm["NONE"]
    xfull = steady(u, A, np.ones(S, bool))
    imp_mod = np.zeros(S); imp_emp = np.zeros(S)
    for i in range(S):
        keep = [j for j in range(S) if j != i]
        m = np.ones(S, bool); m[i] = False
        imp_mod[i] = bc(xfull[keep], steady(u, A, m)[keep])
        imp_emp[i] = bc(full_emp[keep], comm[f"{SP[i]}-"][keep])
    # baselines from the measured matrix
    off = ~np.eye(S, dtype=bool)
    degree = np.array([np.sum(np.abs(A[i, off[i]])) + np.sum(np.abs(A[off[:, i], i])) for i in range(S)])
    selflim = np.abs(np.diag(A))
    return imp_mod, imp_emp, full_emp, degree, selflim


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    imp_mod, imp_emp, full_emp, degree, selflim = impacts()
    rho, p = spearmanr(imp_mod, imp_emp)
    pr = partial_spearman(imp_mod, imp_emp, full_emp)
    print(f"observed: Spearman(model,emp)={rho:.3f} (asymptotic p={p:.3f}); "
          f"partial|abund={pr:.3f}\n")

    # (1) permutation p-value (one-sided: model predicts the ranking)
    NP = 20000
    perm = np.array([spearmanr(imp_mod, rng.permutation(imp_emp))[0] for _ in range(NP)])
    p_perm = float((np.sum(perm >= rho) + 1) / (NP + 1))
    print(f"(1) permutation p (rho >= obs) = {p_perm:.4f}  (null median {np.median(perm):+.3f})")

    # (2) bootstrap CI over the 12 species
    NB = 20000
    br, bpr = [], []
    for _ in range(NB):
        idx = rng.integers(0, S, S)
        if len(np.unique(idx)) < 4:
            continue
        br.append(spearmanr(imp_mod[idx], imp_emp[idx])[0])
        bpr.append(partial_spearman(imp_mod[idx], imp_emp[idx], full_emp[idx]))
    br = np.array(br); bpr = np.array(bpr)
    ci = np.nanpercentile(br, [2.5, 97.5]); cip = np.nanpercentile(bpr, [2.5, 97.5])
    print(f"(2) bootstrap rho 95% CI = [{ci[0]:.2f}, {ci[1]:.2f}]  (frac>0: {np.mean(br>0):.2f})")
    print(f"    bootstrap partial-rho 95% CI = [{cip[0]:.2f}, {cip[1]:.2f}]  (frac>0: {np.mean(bpr>0):.2f})")

    # (3) leave-one-species-out influence on rho
    loo = np.array([spearmanr(np.delete(imp_mod, i), np.delete(imp_emp, i))[0] for i in range(S)])
    imax = int(np.argmax(np.abs(loo - rho)))
    print(f"(3) leave-one-species-out rho range = [{loo.min():.2f}, {loo.max():.2f}]; "
          f"most influential = {SP[imax]} (rho->{loo[imax]:.2f})")

    # (4) baselines vs empirical impact
    print("(4) baselines -- Spearman(empirical impact, X):")
    bl = {"model knockout": imp_mod, "abundance": full_emp, "degree": degree, "self-limitation": selflim}
    blr = {k: spearmanr(imp_emp, v)[0] for k, v in bl.items()}
    for k, v in blr.items():
        print(f"      {k:>16}: rho={v:+.3f}")
    print("    -> model knockout also survives controlling for abundance (partial rho above),"
          " which the trivial predictors do not explain away.")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].hist(br[np.isfinite(br)], bins=40, color="#8da0cb", alpha=0.6, density=True, label="bootstrap")
    ax[0].hist(perm, bins=40, color="#bbbbbb", alpha=0.6, density=True, label="permutation null")
    ax[0].axvline(rho, color="#D55E00", lw=2, label=fr"observed $\rho={rho:.2f}$")
    ax[0].axvline(0, color="grey", lw=0.8, ls=":")
    ax[0].set_xlabel(r"Spearman $\rho$ (model vs measured knockout)")
    ax[0].set_ylabel("density")
    ax[0].set_title(f"a  Bootstrap + permutation ($p_{{\\rm perm}}={p_perm:.3f}$)", loc="left", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8.5)
    names = list(blr.keys()); vals = [blr[k] for k in names]
    cols = ["#1b9e77" if k == "model knockout" else "#999999" for k in names]
    ax[1].barh(range(len(names)), vals, color=cols)
    ax[1].axvline(0, color="k", lw=0.7)
    ax[1].axvline(pr, color="#D55E00", ls="--", lw=1.4, label=fr"model, abundance-controlled $\rho={pr:.2f}$")
    ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=9)
    ax[1].set_xlabel(r"Spearman $\rho$ with measured impact")
    ax[1].set_title("b  Network knockout vs trivial predictors", loc="left", fontsize=11)
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    for a in ax:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_keystone_robust.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_keystone_robust.png written")
    np.savez(f"{OUT}/keystone_robust.npz", rho=rho, p_perm=p_perm, ci=ci, cip=cip,
             loo=loo, baselines=np.array(vals), baseline_names=np.array(names))


if __name__ == "__main__":
    main()
