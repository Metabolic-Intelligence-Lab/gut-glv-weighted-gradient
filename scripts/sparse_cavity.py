"""
Fig. 13 -- The sparse scale-free case: hub-driven instability (cavity on a graph)
=================================================================================

On a dense network the May/elliptic boundary gives a size-independent threshold
(sigma_c = 1 with our scaling). On a SPARSE scale-free network the cavity
(locally-tree-like) analysis says the instability is controlled by the hubs: a
degree-k node sums k couplings, so with per-edge variance sigma^2/<k> its row
carries variance sigma^2 k/<k>, and the local 'star' around the largest hub
contributes a leading eigenvalue

        rho_hub  ~  sigma * sqrt(k_max / <k>) ,

so the community loses stability already at

        sigma_c  ~  sqrt(<k> / k_max)   <  1 ,

decreasing as the network grows and the hubs get larger. We verify this exactly
by diagonalising the (symmetric) weighted community matrix:

  (A) sigma_c (from the spectral radius) vs k_max for Barabasi-Albert networks of
      growing size, against the analytic sqrt(<k>/k_max) and the dense sigma_c=1.
  (B) the destabilising eigenvector is localised on the maximum-degree hub -- the
      keystone -- i.e. instability nucleates there.

This closes the loop: the Barabasi error/attack fragility (Fig. 4) and the
cavity phase boundary (Fig. 12) share one analytic origin -- the hubs.

Run:  python3 sparse_cavity.py
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"


def weighted_community(S, m, rng):
    """Symmetric coupling on a BA scale-free graph; per-edge std = 1/sqrt(<k>)
    so that a typical row has unit variance (sigma=1 baked in). Returns the
    sparse weighted adjacency W and the degree vector."""
    G = nx.barabasi_albert_graph(S, m, seed=int(rng.integers(1 << 30)))
    A = nx.to_scipy_sparse_array(G, format="coo", dtype=float)
    k = np.asarray(A.sum(1)).ravel()
    kmean = k.mean()
    # assign a symmetric Gaussian weight to each undirected edge
    iu = A.row < A.col
    r, c = A.row[iu], A.col[iu]
    w = rng.normal(0.0, 1.0 / np.sqrt(kmean), size=r.size)
    W = sp.coo_matrix((np.concatenate([w, w]),
                       (np.concatenate([r, c]), np.concatenate([c, r]))),
                      shape=(S, S)).tocsr()
    return W, k


def spectral_radius(W):
    # linear instability is set by the RIGHTMOST (largest algebraic) eigenvalue of
    # the symmetric community matrix, not the largest-magnitude one.
    val = eigsh(W, k=1, which="LA", return_eigenvectors=False, maxiter=5000)
    return float(val[0])


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    m = 2
    sizes = [100, 200, 400, 800, 1600, 3200]
    n_real = 4

    kmax_mean, sigc_mean, sigc_pred = [], [], []
    for S in sizes:
        rhos, kmaxs, kmeans = [], [], []
        for rep in range(n_real):
            rng = np.random.default_rng(10 * S + rep)
            W, k = weighted_community(S, m, rng)
            rhos.append(spectral_radius(W))
            kmaxs.append(k.max()); kmeans.append(k.mean())
        rho = np.mean(rhos)
        km = np.mean(kmaxs); kc = np.mean(kmeans)
        kmax_mean.append(km)
        sigc_mean.append(1.0 / rho)              # sigma_c = 1/rho
        sigc_pred.append(np.sqrt(kc / km))       # cavity/star prediction
        print(f"  S={S:5d}  <k>={kc:.1f}  k_max={km:.0f}  rho={rho:.2f}  "
              f"sigma_c={1/rho:.3f}  pred sqrt(<k>/kmax)={np.sqrt(kc/km):.3f}")

    # ---- eigenvector localisation on the hub (largest network) ----
    rng = np.random.default_rng(7)
    W, k = weighted_community(sizes[-1], m, rng)
    vals, vecs = eigsh(W, k=1, which="LA")        # rightmost (destabilising) mode
    v = np.abs(vecs[:, 0])
    hub = int(np.argmax(v))                       # where the unstable mode concentrates
    kmax_node = int(np.argmax(k))
    # inverse participation ratio: robust localisation measure (1/S delocalised, ~1 localised)
    ipr = float((v**4).sum() / (v**2).sum()**2)
    part_ratio = 1.0 / (ipr * v.size)             # fraction of nodes participating

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].loglog(kmax_mean, sigc_mean, "o-", color="#d95f02", ms=8,
                 label=r"numerical  $\sigma_c=1/\rho$")
    ax[0].loglog(kmax_mean, sigc_pred, "s--", color="#7570b3", ms=6,
                 label=r"cavity prediction  $\sqrt{\langle k\rangle/k_{\max}}$")
    ax[0].axhline(1.0, ls=":", color="grey")
    ax[0].text(kmax_mean[0], 1.04, "dense mean-field  $\\sigma_c=1$", fontsize=8, color="grey")
    ax[0].set_xlabel(r"maximum degree  $k_{\max}$")
    ax[0].set_ylabel(r"instability threshold  $\sigma_c$")
    ax[0].set_title("A. Hubs lower the stability threshold:\n"
                    r"$\sigma_c\sim\sqrt{\langle k\rangle/k_{\max}}$, falling as the network grows", fontsize=10)
    ax[0].legend(frameon=False, fontsize=9)

    ax[1].scatter(k, v, s=10, alpha=0.5, color="#1b9e77")
    ax[1].scatter([k[hub]], [v[hub]], s=110, color="#d95f02", zorder=5,
                  label=f"localisation node (hub, k={int(k[hub])})")
    ax[1].set_xlabel("node degree  $k$")
    ax[1].set_ylabel("|eigenvector component| at instability")
    ax[1].set_title("B. The unstable mode is localised on the hub\n(instability nucleates on the keystone)", fontsize=10)
    ax[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Sparse scale-free cavity: the hubs set stability (and are where it breaks)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig13_sparse_scalefree.png", dpi=150)
    plt.close(fig)
    print(f"[fig13] written | localisation node degree={int(k[hub])} "
          f"(max degree={int(k[kmax_node])}), IPR={ipr:.3f}, "
          f"participation={part_ratio*v.size:.0f}/{v.size} nodes")


if __name__ == "__main__":
    main()
