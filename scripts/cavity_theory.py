"""
Fig. 12 -- Analytic phase boundaries from the cavity / DMFT method
==================================================================

Dynamical mean-field (cavity) theory for the disordered Generalized
Lotka-Volterra model

    dn_i/dt = n_i ( 1 - n_i + sum_{j!=i} a_ij n_j ),
    <a_ij> = -mu/S,  Var(a_ij) = sigma^2/S,  corr(a_ij, a_ji) = gamma.

The cavity reduction to a single effective species gives, at a fixed point,
    n = w / g  for w>0  (else 0),     w = 1 - mu M + sigma sqrt(q) z,  z~N(0,1)
with response  nu = phi/g  and  g = 1 - gamma sigma^2 nu, hence

    (1)  g = [1 + sqrt(1 - 4 gamma sigma^2 Phi(D))] / 2          (susceptibility)
    (2)  g^2 = sigma^2 [ (D^2+1) Phi(D) + D phi(D) ]            (second moment)
    phi = Phi(D),   D = (1 - mu M)/(sigma sqrt q)               (survivor fraction)

The resilient (unique-equilibrium) phase is bounded by the elliptic/replicon
instability edge, sigma sqrt(phi) (1 + gamma) = 1. On this locus the closure forces
Delta=0, hence phi=1/2, so the boundary has the closed form

    sigma_c(gamma) = sqrt(2) / (1 + gamma).

Beyond it the unstable phase is multiple-attractor (marginal) near gamma=1 and
persistent/chaotic toward gamma<0. The response discriminant 1-4 gamma sigma^2 phi=0
is NOT a separate boundary: it is the square-root branch point of the
replica-symmetric solution, which coincides with the stability edge only at gamma=1
(the de Almeida-Thouless point) and for gamma<1 lies at HIGHER sigma, inside the
already-unstable region. We therefore plot sigma_c(gamma) as the phase boundary,
overlay the closed form as a check, and show the branch point only as a faint
reference that makes the gamma=1 coincidence visible.

Scale-free correction (discussed, not exact): a degree-k node sums k couplings,
so it feels an effective disorder sigma_eff ~ sigma sqrt(k/<k>); the hubs of a
scale-free network therefore cross the boundary first, lowering the effective
threshold by ~ sqrt(<k>/k_max). This is the analytic root of the
keystone/Barabasi fragility.

Run:  python3 cavity_theory.py
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from asymmetric_and_phases import classify

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"
Phi = norm.cdf
pdf = norm.pdf


def _H(D, sigma, gamma):
    disc = 1 - 4 * gamma * sigma**2 * Phi(D)
    if disc < 0:
        return None
    g = (1 + np.sqrt(disc)) / 2
    return g**2 - sigma**2 * ((D**2 + 1) * Phi(D) + D * pdf(D))


def solve_phi(sigma, gamma):
    """Survivor fraction phi at the DMFT fixed point; None if it has vanished
    (glassy / multiple-attractor regime). The root Delta ~ 1/sigma for small
    sigma, so the scan must extend to large positive Delta."""
    Ds = np.linspace(-10, 120, 4000)
    prev = None
    for D in Ds:
        h = _H(D, sigma, gamma)
        if h is None:
            return None
        if prev is not None and prev[1] * h < 0:
            root = brentq(lambda x: _H(x, sigma, gamma), prev[0], D)
            return float(Phi(root))
        prev = (D, h)
    return None


def sigma_glass(gamma, smax=5.0):
    """Boundary where the fixed point disappears (gamma>0)."""
    if gamma <= 0:
        return np.nan
    lo, hi = 0.05, smax
    if solve_phi(lo, gamma) is None:
        return lo
    if solve_phi(hi, gamma) is not None:
        return np.nan
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if solve_phi(mid, gamma) is None:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def sigma_may(gamma, smax=5.0):
    """Elliptic-law marginal-stability boundary  sigma sqrt(phi)(1+gamma)=1."""
    if 1 + gamma <= 1e-6:
        return np.nan

    def f(s):
        ph = solve_phi(s, gamma)
        if ph is None:
            return None
        return s * np.sqrt(ph) * (1 + gamma) - 1.0
    lo, hi = 0.05, smax
    flo = f(lo)
    prev = (lo, flo)
    for s in np.linspace(0.1, smax, 120):
        fv = f(s)
        if fv is None:
            return np.nan
        if prev[1] is not None and prev[1] * fv < 0:
            return brentq(lambda x: f(x), prev[0], s)
        prev = (s, fv)
    return np.nan


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    gammas = np.linspace(-1.0, 1.0, 41)
    s_glass = np.array([sigma_glass(g) for g in gammas])
    s_may = np.array([sigma_may(g) for g in gammas])
    for g, sg, sm in zip(gammas[::5], s_glass[::5], s_may[::5]):
        print(f"  gamma={g:+.2f}  sigma_glass={sg:.2f}  sigma_may={sm:.2f}")

    # numerical phase grid (dense mean-field, mu=2 to match the analytics) -- cached
    cache = f"{OUT}/phase_grid_mu2.npz"
    nsig = np.linspace(0.3, 5.0, 14)
    ngam = np.linspace(-1.0, 1.0, 13)
    if os.path.exists(cache):
        grid = np.load(cache)["grid"]
        print("  [num] grid loaded from cache")
    else:
        S = 80
        r = np.ones(S)
        grid = np.zeros((len(ngam), len(nsig)), dtype=int)
        for i, g in enumerate(ngam):
            for j, s in enumerate(nsig):
                grid[i, j] = classify(S, r, s, g, seed=300 + j, mu=2.0)
            print(f"  [num] gamma={g:+.2f} done")
        np.savez(cache, grid=grid)

    cmap = ListedColormap(["#1b9e77", "#7570b3", "#d95f02"])
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=2,
              extent=[nsig[0], nsig[-1], ngam[0], ngam[-1]], alpha=0.55)

    closed = np.sqrt(2.0) / (1.0 + gammas)      # sigma_c = sqrt2/(1+gamma) with phi=1/2 on the edge
    closed[(1.0 + gammas) <= 1e-6] = np.nan
    ax.plot(s_may, gammas, "k-", lw=2.8,
            label=r"resilient-phase boundary  $\sigma\sqrt{\phi}\,(1+\gamma)=1$")
    ax.plot(closed, gammas, color="#e6194B", ls="--", lw=1.8,
            label=r"closed form  $\sigma_c=\sqrt{2}/(1+\gamma)$  ($\phi=1/2$)")
    ax.plot(s_glass, gammas, color="grey", ls=":", lw=1.6,
            label=r"RS branch point  $1-4\gamma\sigma^2\phi=0$  (meets edge only at $\gamma=1$)")
    ax.axhline(0.0, color="grey", lw=0.6, ls=":")
    ax.axhline(0.31, color="grey", lw=0.6, ls=":")
    ax.text(4.7, 0.15, "empirical\n$\\gamma$ band", ha="right", fontsize=8, color="black")
    ax.set_xlim(nsig[0], nsig[-1]); ax.set_ylim(ngam[0], ngam[-1])
    ax.set_xlabel(r"disorder strength  $\sigma$")
    ax.set_ylabel(r"interaction symmetry  $\gamma$")
    ax.set_title("Analytic cavity/DMFT phase boundary vs numerical phases\n"
                 "(green: resilient, purple: multiple attractors, orange: chaos; boundary = elliptic/replicon\n"
                 "edge $\\sigma_c=\\sqrt{2}/(1+\\gamma)$; S=80 numerics developed at slightly higher $\\sigma$)", fontsize=9.0)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig12_cavity_phase_boundary.png", dpi=150)
    plt.close(fig)
    print("[fig12] cavity phase boundary written")


if __name__ == "__main__":
    main()
