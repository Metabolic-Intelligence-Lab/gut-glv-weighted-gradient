r"""
Does restricting the descent test to the biomass plateau restore specificity?
=============================================================================
On the plateau the compositional criterion of the Proposition is exact and the observed margin
is largest, +0.149 against +0.100 pooled, so it is the natural place to look for the
discrimination the pooled calibration does not provide. The ensemble already scores every
replicate twice, pooled and plateau-restricted, so the question is answered from the cached
run with no re-simulation.

Statistic: the margin, fitted minus scrambled, which is continuous, rather than the Monte
Carlo p, which piles up at its floor and cannot be calibrated by a quantile. Critical value:
the 95th percentile of that margin under the symmetrizable truth in the same noise setting,
so every number below is a size-0.05 power.

Result: it does not. Power stays between 0.04 and 0.19 for every family under both
restrictions, no ordering by distance from the manifold appears, and the predictively tied
free fit is rejected no more often than a symmetrizable truth.

Run:  python3 scripts/plateau_specificity.py
"""
import numpy as np
from scipy.stats import spearmanr

NPZ = "figures/descent_specificity_ensemble.npz"
F_ALL, N_ALL, F_PL, N_PL = 2, 3, 5, 6


def main():
    z = np.load(NPZ, allow_pickle=True)
    R, meta, fams, settings = z["rows"], z["meta"], z["fams"], z["settings"]
    famname = {i: f for i, f in enumerate(fams)}
    fam = lambda ti: famname.get(int(meta[ti][0]), "base") if meta[ti][0] >= 0 else "base"
    base_ti = [ti for ti in range(len(meta)) if fam(ti) == "base"][0]
    print("size-0.05 power of the descent test, pooled against plateau-restricted\n")
    for si, (sp, so) in enumerate(settings):
        sel = R[:, 0] == si
        b = R[sel & (R[:, 1] == base_ti)]
        if len(b) < 20:
            continue
        c_all = np.quantile(b[:, F_ALL] - b[:, N_ALL], 0.95)
        c_pl = np.quantile(b[:, F_PL] - b[:, N_PL], 0.95)
        print(f"-- sigma_p={sp:.2f} sigma_o={so:.2f}  n_base={len(b)}  "
              f"critical margin pooled {c_all:+.3f}, plateau {c_pl:+.3f} "
              f"(observed +0.100 and +0.149)")
        agg = {}
        for ti in range(len(meta)):
            X = R[sel & (R[:, 1] == ti)]
            if len(X) < 5:
                continue
            agg.setdefault(fam(ti), [[], [], []])
            agg[fam(ti)][0].append(np.mean((X[:, F_ALL] - X[:, N_ALL]) > c_all))
            agg[fam(ti)][1].append(np.mean((X[:, F_PL] - X[:, N_PL]) > c_pl))
            agg[fam(ti)][2].append(meta[ti][3])
        print(f"   {'family':<10}{'truths':>7}{'pooled':>9}{'plateau':>9}")
        for k in ["base"] + [f for f in fams if f != "base" and f in agg]:
            if k in agg:
                a, p, _ = agg[k]
                print(f"   {k:<10}{len(a):>7}{np.mean(a):>9.2f}{np.mean(p):>9.2f}")
        da, pa, pp = [], [], []
        for k, (a, p, d) in agg.items():
            if k != "base":
                da += d; pa += a; pp += p
        da, pa, pp = np.array(da), np.array(pa), np.array(pp)
        ok = np.isfinite(da)
        print(f"   Spearman(power, d_VF) over {int(ok.sum())} non-symmetrizable truths: "
              f"pooled {spearmanr(pa[ok], da[ok]).correlation:+.2f}, "
              f"plateau {spearmanr(pp[ok], da[ok]).correlation:+.2f}\n")


if __name__ == "__main__":
    main()
