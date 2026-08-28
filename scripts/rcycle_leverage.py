"""
Edge leverage of Venturelli's cycle inconsistency r_cycle  (referee-6 P1, §4.3/§12)
===================================================================================

r_cycle uses b_ij = log|J_ij/J_ji|, which explodes when either coefficient is near zero.
An unweighted RMS residual then treats a precise edge and an almost-unidentified one alike.
This script quantifies the referee's worry for Venturelli (whose point-estimate r_cycle=2.2):
edge-magnitude distribution, correlation of |b_ij| with 1/min-coefficient, leave-one-edge-out
sensitivity, and r_cycle after dropping the weakest edges. (Venturelli's averaged matrix has
no per-edge posterior, so an uncertainty-weighted r_cycle would need the underlying fits;
the min-coefficient is used as the identifiability proxy.)

Run:  python3 rcycle_leverage.py
"""
import numpy as np
np.seterr(all="ignore")


def main():
    V = np.loadtxt("data/venturelli2018/interaction_matrix_averaged.csv", delimiter=",", skiprows=1)
    S = len(V); iu = np.triu_indices(S, 1); a, b = V[iu], V.T[iu]
    conc = (a != 0) & (b != 0) & (np.sign(a) == np.sign(b))
    ii, jj = iu[0][conc], iu[1][conc]; E = len(ii)
    bij = np.log(np.abs(a[conc] / b[conc])); mincoef = np.minimum(np.abs(a[conc]), np.abs(b[conc]))

    def rcyc(mask):
        idx = np.where(mask)[0]; Ei = len(idx)
        if Ei < 2:
            return 0.0
        B = np.zeros((Ei, S)); B[np.arange(Ei), jj[idx]] = 1; B[np.arange(Ei), ii[idx]] = -1
        u, *_ = np.linalg.lstsq(B, bij[idx], rcond=None)
        return float(np.sqrt(np.mean((bij[idx] - B @ u) ** 2)))

    full = rcyc(np.ones(E, bool))
    loo = np.array([rcyc(np.arange(E) != k) for k in range(E)])
    thr = np.percentile(mincoef, 20); keep = mincoef > thr
    print(f"Venturelli: E={E} concordant reciprocal edges, r_cycle={full:.3f}")
    print(f"  |b_ij| in [{np.abs(bij).min():.2f}, {np.abs(bij).max():.2f}] (median {np.median(np.abs(bij)):.2f})")
    print(f"  min-coefficient per edge: median {np.median(mincoef):.3f}, min {mincoef.min():.4f}")
    print(f"  corr(|b_ij|, 1/min-coef) = {np.corrcoef(np.abs(bij), 1/mincoef)[0,1]:.2f} "
          "(near 1 => the large log-ratios are the near-zero-coefficient edges)")
    print(f"  leave-one-edge-out r_cycle in [{loo.min():.2f}, {loo.max():.2f}]: one near-zero edge "
          f"pulls it from {full:.2f} to {loo.min():.2f}")
    print(f"  dropping the weakest 20% of edges (min-coef<{thr:.3f}): r_cycle={rcyc(keep):.3f} "
          f"(vs {full:.3f} full) -> r_cycle is not robust to small-coefficient edges")
    np.savez("figures/rcycle_leverage.npz", r_full=full, loo=loo, r_trim=rcyc(keep), mincoef=mincoef)


if __name__ == "__main__":
    main()
