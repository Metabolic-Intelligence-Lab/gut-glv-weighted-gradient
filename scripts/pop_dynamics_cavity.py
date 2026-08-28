"""
Fig. 14 -- The full distributional cavity: spectral density for an arbitrary
degree distribution (population dynamics)
==========================================================================

The hub-scaling result (Fig. 13) used only the largest eigenvalue. The complete
cavity treatment for a sparse random matrix on a locally-tree-like graph yields
the WHOLE spectral density rho(lambda) for any degree distribution P(k), via the
population-dynamics algorithm (Rogers, Perez-Castillo, Kuhn, Takeda 2008):

  cavity resolvent:   G = 1 / ( z - sum_{l in cavity neighbours} J_l^2 G_l ),
  z = lambda + i*eta,  rho(lambda) = -(1/pi) < Im G >.

Cavity neighbours are drawn from the EXCESS-degree distribution (edge-biased
degree minus one); the per-edge weight variance is sigma^2/<k> (sigma=1 here).

We (i) run the population dynamics on the Barabasi-Albert degree sequence to get
the bulk spectral density, (ii) overlay the histogram from DIRECT diagonalisation
of a large weighted instance, and (iii) mark the isolated hub eigenvalues that
sit OUTSIDE the bulk edge -- the localised keystone modes that set stability.
The distributional cavity thus predicts the bulk for any P(k); the scale-free
hubs add discrete outliers beyond it.

Run:  python3 pop_dynamics_cavity.py
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"


def population_dynamics(deg, lambdas, eta=0.03, P=2000, sweeps=50,
                        M=6000, seed=1):
    """Spectral density rho(lambda) for a symmetric sparse weighted matrix whose
    graph has degree sequence `deg` and per-edge weight variance 1/<k>."""
    rng = np.random.default_rng(seed)
    cbar = deg.mean()
    w_std = np.sqrt(1.0 / cbar)
    stubs = np.repeat(np.arange(len(deg)), deg.astype(int))   # edge-biased pool

    pop = np.full(P, -1j, dtype=complex)
    rho = np.zeros(len(lambdas))
    # sweep lambda downward, warm-starting the population
    order = np.argsort(lambdas)[::-1]
    for oi in order:
        z = lambdas[oi] + 1j * eta
        for _ in range(sweeps):
            # excess degrees for a batch
            nb = deg[stubs[rng.integers(0, stubs.size, size=P)]].astype(int) - 1
            nb = np.clip(nb, 0, None)
            for p in range(P):
                k = nb[p]
                if k:
                    idx = rng.integers(0, P, size=k)
                    J2 = rng.normal(0, w_std, size=k) ** 2
                    s = np.sum(J2 * pop[idx])
                else:
                    s = 0.0
                pop[rng.integers(P)] = 1.0 / (z - s)
        # sample full marginals (full degree = node-biased = uniform over nodes)
        im = 0.0
        ks = deg[rng.integers(0, len(deg), size=M)].astype(int)
        for k in ks:
            if k:
                idx = rng.integers(0, P, size=k)
                J2 = rng.normal(0, w_std, size=k) ** 2
                s = np.sum(J2 * pop[idx])
            else:
                s = 0.0
            im += (1.0 / (z - s)).imag
        rho[oi] = -im / M / np.pi
    return rho


def direct_spectrum(S, m, seed=3):
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(S, m, seed=seed)
    deg = np.array([d for _, d in G.degree()], dtype=float)
    cbar = deg.mean()
    A = nx.to_numpy_array(G)
    iu = np.triu_indices(S, 1)
    w = np.zeros((S, S))
    mask = A[iu] > 0
    vals = rng.normal(0, np.sqrt(1.0 / cbar), size=int(mask.sum()))
    wi = np.array(iu[0])[mask]; wj = np.array(iu[1])[mask]
    w[wi, wj] = vals; w[wj, wi] = vals
    eig = np.linalg.eigvalsh(w)
    return eig, deg


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    # direct diagonalisation (truth)
    eig, degD = direct_spectrum(2000, 2, seed=3)
    bulk_edge = 2.0 * np.sqrt(1.0)        # ~2 sqrt(<k>*var)=2 sqrt(1) for var=1/<k>
    outliers = eig[eig > bulk_edge + 0.15]

    # population-dynamics cavity on the SAME degree distribution
    lambdas = np.linspace(-3.2, 3.2, 49)
    rho = population_dynamics(degD, lambdas, seed=1)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.hist(eig, bins=120, density=True, color="#9ecae1", alpha=0.8,
            label="direct diagonalisation (S=2000)")
    ax.plot(lambdas, rho, "-", color="#d95f02", lw=2.4,
            label="distributional cavity  $\\rho(\\lambda)$ (population dynamics)")
    for j, lo in enumerate(np.sort(outliers)[::-1][:8]):
        ax.axvline(lo, color="#1b9e77", lw=1.3, alpha=0.8)
    ax.axvline(bulk_edge, color="grey", ls="--", lw=1.2)
    ax.text(bulk_edge + 0.03, ax.get_ylim()[1] * 0.6, "bulk edge", rotation=90,
            fontsize=8, color="grey")
    if len(outliers):
        ax.text(outliers.max(), 0.05, "hub-localised\noutliers\n(keystone modes)",
                ha="right", fontsize=8, color="#1b9e77")
    ax.set_xlabel("eigenvalue  $\\lambda$")
    ax.set_ylabel("spectral density  $\\rho(\\lambda)$")
    ax.set_title("Full distributional cavity vs direct spectrum (scale-free)\n"
                 "population dynamics predicts the bulk for any P(k); scale-free hubs add isolated outliers")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xlim(-3.2, max(3.2, (outliers.max() + 0.3) if len(outliers) else 3.2))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig14_distributional_cavity.png", dpi=150)
    plt.close(fig)
    print(f"[fig14] written | bulk edge~{bulk_edge:.2f}, "
          f"n_outliers={len(outliers)}, largest eigenvalue={eig.max():.2f}, "
          f"rho_max={rho.max():.2f}")


if __name__ == "__main__":
    main()
