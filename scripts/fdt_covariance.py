"""
The irreversibility index depends on the noise covariance  (referee-4 P1, §10)
==============================================================================

The FDT/irreversibility index I = ||Omega||_F/(T sqrt S), Omega = AC + T*1 with C solving
the Lyapunov equation AC + C A^T = -2 T*1, assumes ISOTROPIC noise Q = T*1. Detailed balance
and the entropy-production rate are properties of the PAIR (A, Q), not of the drift A alone.
Under a species rescaling x = S y the drift and the noise co-transform,
    A -> S^{-1} A S,   Q -> S^{-1} Q S^{-1},
and the coordinate-invariant entropy-production rate is unchanged. But if one re-imposes
isotropic noise Q = T*1 in EVERY coordinate (as the plug-in index does), the model changes and
I moves. So I is invariant only to a UNIFORM rescaling, not to the species-specific rescaling
that motivated the whole paper; I_data computed from the inferred matrix under assumed
isotropic noise is therefore a noise-model-dependent diagnostic, not a coordinate-invariant
physical measurement.

This script demonstrates both facts on a stable synthetic community matrix:
  (a) re-imposed isotropic noise after A -> S^{-1} A S  ->  I drifts (coordinate-dependent);
  (b) co-transformed noise Q -> S^{-1} Q S^{-1}          ->  the entropy-production rate is
      invariant (to numerical precision).

Run:  python3 fdt_covariance.py     (writes figures/fig_fdt_covariance.png)
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"


def stable_community(S, gamma, rng, margin=0.4):
    """Dense random community matrix with corr(A_ij,A_ji)=gamma, shifted Hurwitz."""
    a = rng.normal(size=(S, S)); b = rng.normal(size=(S, S))
    A = np.zeros((S, S)); iu = np.triu_indices(S, 1)
    s = 1.0 / np.sqrt(S)
    A[iu] = s * a[iu]
    A[(iu[1], iu[0])] = s * (gamma * a[iu] + np.sqrt(max(1 - gamma**2, 0)) * b[iu])
    np.fill_diagonal(A, 0.0)
    A -= (np.max(np.linalg.eigvals(A).real) + margin) * np.eye(S)   # make Hurwitz
    return A


def lyap_C(A, Q):
    """Stationary covariance: A C + C A^T = -2 Q."""
    return solve_continuous_lyapunov(A, -2.0 * Q)


def I_index(A, Q):
    """Irreversibility index ||A C + Q||_F / (scale * sqrt S), Omega=AC+Q antisymmetric."""
    S = len(A); C = lyap_C(A, Q)
    Om = A @ C + Q
    scale = np.trace(Q) / S
    return float(np.linalg.norm(Om, "fro") / (scale * np.sqrt(S)))


def entropy_production(A, Q):
    """Coordinate-invariant steady-state entropy production rate of the linear diffusion
    dx = A x dt + sqrt(2Q) dW:  sigma = -Tr(A_irr) with A_irr = (AC - C A^T)/2 . C^{-1}...
    We use the standard form sigma = Tr(Q^{-1} Omega C^{-1} Omega^T)/2 is not it; the clean
    invariant is sigma = Tr( (Omega C^{-1})^2 ) with Omega=AC+Q the antisymmetric current
    matrix (Omega C^{-1} is the irreversible drift), invariant under x->Ty since Omega->
    T Omega T^T, C->T C T^T => Omega C^{-1} -> T (Omega C^{-1}) T^{-1} (a similarity)."""
    S = len(A); C = lyap_C(A, Q)
    Om = A @ C + Q                      # antisymmetric probability-current matrix
    M = Om @ np.linalg.inv(C)           # irreversible drift; similarity-covariant
    return float(-np.trace(M @ M) / S)  # Tr of a similarity invariant (>=0)


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    S, T = 40, 1.0
    A = stable_community(S, 0.1, rng)
    Q0 = T * np.eye(S)
    I0 = I_index(A, Q0); EP0 = entropy_production(A, Q0)
    print(f"baseline (isotropic Q): I={I0:.3f}  entropy_production={EP0:.4f}\n")

    spreads = [0.0, 0.5, 1.0, 1.5, 2.0]
    print(f"{'log-s SD':>9} | {'I (re-imposed isotropic)':>24} | {'EP (co-transformed Q)':>22}")
    print("-" * 62)
    I_iso, EP_cov = [], []
    for sd in spreads:
        Iv, Ev = [], []
        for k in range(20):
            s = np.exp(np.random.default_rng(100 + k).normal(0, sd, S))
            Sm = np.diag(s); Sinv = np.diag(1.0 / s)
            Ap = Sinv @ A @ Sm                       # A' = S^{-1} A S
            # (a) re-impose isotropic noise in the new coordinates
            Iv.append(I_index(Ap, T * np.eye(S)))
            # (b) co-transform the noise Q' = S^{-1} Q S^{-1}
            Qp = Sinv @ Q0 @ Sinv
            Ev.append(entropy_production(Ap, Qp))
        I_iso.append((np.median(Iv), np.percentile(Iv, 10), np.percentile(Iv, 90)))
        EP_cov.append(np.median(Ev))
        print(f"{sd:>9.1f} | {np.median(Iv):>10.3f} [{np.percentile(Iv,10):.2f},{np.percentile(Iv,90):.2f}] | "
              f"{np.median(Ev):>22.4f}")
    print(f"\n(a) re-imposed isotropic: I ranges {I_iso[0][0]:.2f}->{I_iso[-1][0]:.2f} "
          f"({I_iso[-1][0]/I0:.1f}x) purely from the species rescaling -> coordinate-DEPENDENT.")
    print(f"(b) co-transformed noise: entropy production stays {EP_cov[0]:.3f}+/-"
          f"{np.std(EP_cov):.1e} -> the invariant physics is unchanged.")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    med = [x[0] for x in I_iso]; lo = [x[1] for x in I_iso]; hi = [x[2] for x in I_iso]
    ax[0].plot(spreads, med, "o-", color="#D55E00", lw=2, ms=8)
    ax[0].fill_between(spreads, lo, hi, color="#D55E00", alpha=0.15)
    ax[0].axhline(I0, color="grey", ls=":", lw=1, label=f"baseline $I={I0:.2f}$")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("species-rescaling spread  SD$(\\log s)$")
    ax[0].set_ylabel(r"$I$ with re-imposed isotropic noise")
    ax[0].set_title("a  Re-imposing isotropic noise: $I$ is\ngauge-dependent", loc="left", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(spreads, EP_cov, "s-", color="#1b9e77", lw=2, ms=8)
    ax[1].axhline(EP0, color="grey", ls=":", lw=1, label=f"baseline $={EP0:.3f}$")
    ax[1].set_xlabel("species-rescaling spread  SD$(\\log s)$")
    ax[1].set_ylabel("entropy-production rate (co-transformed $Q$)")
    ax[1].set_title("b  With the noise co-transformed, the\nphysics is invariant", loc="left", fontsize=11)
    ax[1].legend(frameon=False, fontsize=9)
    ymid = EP0
    ax[1].set_ylim(ymid * 0.5, ymid * 1.5)
    for a in ax:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_fdt_covariance.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_fdt_covariance.png written")
    np.savez(f"{OUT}/fdt_covariance.npz", spreads=spreads, I_iso=med, EP_cov=EP_cov, I0=I0, EP0=EP0)


if __name__ == "__main__":
    main()
