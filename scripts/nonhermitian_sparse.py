"""
Fig. 15 -- Unifying topology and asymmetry: the non-Hermitian sparse case
=========================================================================

This closes the programme by combining the two analytic axes:
  * asymmetry gamma  (dense elliptic law -- Fig. 12), and
  * scale-free topology / hubs  (sparse cavity -- Figs. 13-14).

For a Barabasi-Albert network we place gamma-correlated couplings on each edge,
    J_ij = s z1,  J_ji = s (gamma z1 + sqrt(1-gamma^2) z2),  s = 1/sqrt(<k>),
and read the stability-setting quantity directly from the spectrum: the spectral
ABSCISSA  a(gamma) = max Re(eigenvalue),  so  sigma_c = 1/a.

The full non-Hermitian cavity (Rogers & Perez-Castillo 2009) would obtain this
from Hermitization with 2x2 matrix cavity precisions; here we compute the
spectrum exactly, which both gives the answer and would validate that cavity.

Results:
  (A) sigma_c(gamma): the sparse scale-free threshold lies BELOW the dense
      elliptic prediction 1/(1+gamma) for every gamma -- hubs add fragility at
      all asymmetries -- and both diverge as gamma -> -1 (antisymmetric = stable).
  (B) complex eigenvalue clouds: the elliptic bulk plus hub outliers, which are
      REAL (destabilising) for gamma>0 but turn imaginary (oscillatory, harmless
      to Re) for gamma<0 -- the hub-level reason antisymmetry is stabilising.

Run:  python3 nonhermitian_sparse.py
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"


def build_sparse(S, m, gamma, seed):
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(S, m, seed=seed)
    A = nx.to_scipy_sparse_array(G, format="coo", dtype=float)
    k = np.asarray(A.sum(1)).ravel()
    s = 1.0 / np.sqrt(k.mean())
    iu = A.row < A.col
    ri, ci = A.row[iu], A.col[iu]
    z1 = rng.normal(size=ri.size); z2 = rng.normal(size=ri.size)
    Jij = s * z1
    Jji = s * (gamma * z1 + np.sqrt(max(1 - gamma**2, 0.0)) * z2)
    rows = np.concatenate([ri, ci]); cols = np.concatenate([ci, ri])
    vals = np.concatenate([Jij, Jji])
    return sp.coo_matrix((vals, (rows, cols)), shape=(S, S)).tocsr(), k


def abscissa(S, m, gamma, seed):
    """Spectral abscissa max Re(lambda). Try sparse Arnoldi; fall back to a dense
    diagonalisation (robust for the strongly non-normal / near-antisymmetric
    cases where Arnoldi fails to converge)."""
    J, k = build_sparse(S, m, gamma, seed)
    try:
        ev = eigs(J, k=8, which="LR", return_eigenvectors=False, maxiter=4000)
        return float(np.max(ev.real))
    except Exception:
        ev = np.linalg.eigvals(J.toarray())
        return float(np.max(ev.real))


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    gammas = np.linspace(-1.0, 1.0, 13)
    sc_sparse, sc_dense = [], []
    for g in gammas:
        a = np.mean([max(abscissa(1500, 2, g, 100 + r), 1e-3) for r in range(3)])
        sc_sparse.append(1.0 / a)
        sc_dense.append(1.0 / (1.0 + g) if g > -0.999 else np.inf)
        print(f"  gamma={g:+.2f}  abscissa={a:.3f}  sigma_c(sparse)={1/a:.3f}  "
              f"sigma_c(dense)={sc_dense[-1]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

    ax[0].plot(gammas, sc_sparse, "o-", color="#d95f02", ms=7,
               label="sparse scale-free  (exact abscissa)")
    ax[0].plot(gammas, sc_dense, "s--", color="#7570b3", ms=6,
               label=r"dense elliptic law  $1/(1+\gamma)$")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"interaction symmetry  $\gamma$")
    ax[0].set_ylabel(r"stability threshold  $\sigma_c=1/a$")
    ax[0].set_title("A. Topology $\\times$ asymmetry: scale-free hubs lower\n"
                    "$\\sigma_c$ below the dense law at every $\\gamma$", fontsize=10)
    ax[0].legend(frameon=False, fontsize=9)

    # complex eigenvalue clouds (full dense spectrum of a sparse instance)
    for g, axi, col in [(-0.5, ax[1], "#1b9e77"), (0.5, ax[1], "#d95f02")]:
        J, k = build_sparse(700, 2, g, 7)
        ev = np.linalg.eigvals(J.toarray())
        axi.scatter(ev.real, ev.imag, s=6, alpha=0.45, color=col,
                    label=f"$\\gamma={g:+.1f}$")
        absc = ev.real.max()
        axi.scatter([absc], [ev.imag[np.argmax(ev.real)]], s=70,
                    edgecolor="black", facecolor=col, zorder=5)
    ax[1].axvline(1.0, ls=":", color="grey")
    ax[1].text(1.02, ax[1].get_ylim()[1]*0.8, "stability\nlimit", fontsize=8, color="grey")
    ax[1].set_xlabel(r"Re $\lambda$"); ax[1].set_ylabel(r"Im $\lambda$")
    ax[1].set_title("B. Complex spectrum: elliptic bulk + hub outliers\n"
                    "(real & destabilising for $\\gamma>0$; imaginary for $\\gamma<0$)", fontsize=10)
    ax[1].legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("Non-Hermitian sparse case: one picture for real topology and real asymmetry",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig15_nonhermitian_sparse.png", dpi=150)
    plt.close(fig)
    print("[fig15] written")


if __name__ == "__main__":
    main()
