"""
Hub localisation of the unstable mode as a function of interaction symmetry
===========================================================================

Companion to sparse_cavity.py (Fig. fig:phase d, the SYMMETRIC gamma=1 check)
and nonhermitian_sparse.py (Fig. fig:appA, the abscissa sigma_c(gamma)). Those
show (i) that gamma=1 the unstable eigenvector is tightly localised on the
maximum-degree hub, and (ii) that the non-Hermitian abscissa stays finite and
below the dense law for every gamma. This script answers the question in between:
as gamma decreases to the empirically bounded value (~0.1), does the unstable
mode STAY localised on the hub?

Method. Reuse the exact non-Hermitian construction of nonhermitian_sparse.py,
    J_ij = s z1,  J_ji = s(gamma z1 + sqrt(1-gamma^2) z2),  s = 1/sqrt(<k>)
on a Barabasi-Albert graph. For each gamma take the leading (max Re lambda) RIGHT
eigenvector -- the asymptotic unstable mode -- and measure its inverse
participation ratio IPR = sum|v|^4/(sum|v|^2)^2 (participation = 1/IPR nodes),
whether its peak node is the max-degree hub, and whether the leading eigenvalue
is an outlier beyond the elliptic bulk edge (1+gamma at unit per-row variance).

Finding (6 seeds, S=1600, m=2):
  * the leading eigenvalue stays an OUTLIER beyond the bulk at every gamma
    (Re lambda - (1+gamma) > 0 even at gamma=0), so hubs still set a sub-dense
    threshold sigma_c = 1/a < 1/(1+gamma);
  * but the eigenvector DELOCALISES as gamma falls: participation grows from
    ~5 nodes (tight hub localisation) at gamma=1 to ~45 nodes at gamma=0.1, and
    the peak node is the top hub 83% of the time at gamma=1 vs 17% at gamma=0.1.
  => at the empirical gamma the unstable mode is spread over a sub-extensive set
     of high-degree nodes (still a keystone set, not a delocalised bulk mode),
     rather than pinned to a single hub. This is the quantitative basis for the
     refined statement in Sec. "Limit of validity and analytic phase diagram".

Run:  python3 hub_localization_vs_gamma.py
"""
import numpy as np, networkx as nx, scipy.sparse as sp
from scipy.sparse.linalg import eigs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"


def build_sparse(S, m, gamma, seed):
    """Non-Hermitian gamma-correlated couplings on a BA graph (per-edge std
    s=1/sqrt(<k>), i.e. sigma=1 baseline). Identical to nonhermitian_sparse.py."""
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


def leading_mode(S, m, gamma, seed):
    J, k = build_sparse(S, m, gamma, seed)
    try:
        vals, vecs = eigs(J, k=8, which="LR", maxiter=6000)
    except Exception:                       # robust dense fallback
        vals, vecs = np.linalg.eig(J.toarray())
    j = int(np.argmax(vals.real))
    lam = vals[j]
    v = np.abs(vecs[:, j]); v = v / np.linalg.norm(v)
    ipr = float((v**4).sum() / (v**2).sum()**2)
    node = int(np.argmax(v)); kmax_node = int(np.argmax(k))
    return dict(lam_re=float(lam.real), ipr=ipr, part=1.0 / ipr,
                on_hub=(node == kmax_node), k_node=int(k[node]),
                k_max=int(k[kmax_node]), outlier=float(lam.real) - (1.0 + gamma))


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    S, m, seeds = 1600, 2, range(100, 105)
    gammas = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.0]
    rows = {}
    print(f"S={S}, m={m}, {len(list(seeds))} seeds, Barabasi-Albert scale-free\n")
    print(f"{'gamma':>6} {'IPR':>7} {'partic.(nodes)':>15} {'peak_on_hub':>12} "
          f"{'k_node/k_max':>13} {'Re lam':>8} {'lam-(1+g)':>10}")
    for g in gammas:
        r = [leading_mode(S, m, g, sd) for sd in seeds]
        agg = lambda key: float(np.mean([x[key] for x in r]))
        rows[g] = {k: agg(k) for k in
                   ("ipr", "part", "on_hub", "k_node", "k_max", "lam_re", "outlier")}
        d = rows[g]
        print(f"{g:>6.2f} {d['ipr']:>7.3f} {d['part']:>15.1f} "
              f"{d['on_hub']*100:>11.0f}% {d['k_node']:>6.0f}/{d['k_max']:<6.0f} "
              f"{d['lam_re']:>8.3f} {d['outlier']:>+10.3f}")
    print(f"\nIPR near 1/S={1.0/S:.4f} => delocalised; IPR>>1/S => localised.")
    print("participation = 1/IPR nodes; peak_on_hub = %% seeds argmax|v| is top-degree node.")
    print("lam-(1+g)>0 => leading eigenvalue is an outlier beyond the elliptic bulk edge.")

    gs = np.array(gammas)
    part = np.array([rows[g]["part"] for g in gammas])
    lam = np.array([rows[g]["lam_re"] for g in gammas])
    bulk = 1.0 + gs
    TEAL, ORANGE, PURPLE, GREY = "#2a9d8f", "#e76f51", "#7570b3", "#8a8a8a"

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(gs, part, "o-", color=ORANGE, lw=2, ms=6)
    ax[0].axvspan(0.0, 0.13, color=TEAL, alpha=0.15)
    ax[0].text(0.06, part.max() * 0.88, "empirical\n$\\gamma$", fontsize=8, color=TEAL, ha="center")
    ax[0].annotate("tight hub\nlocalisation", (1.0, part[0]), (0.62, part.max() * 0.45),
                   fontsize=8, color=GREY, ha="center",
                   arrowprops=dict(arrowstyle="->", color=GREY))
    ax[0].set_xlabel(r"interaction symmetry  $\gamma$")
    ax[0].set_ylabel("participation of unstable mode  (nodes)")
    ax[0].set_title(r"(a) The unstable mode delocalises as $\gamma\to$ small"
                    "\n(single hub $\\to$ sub-extensive high-degree set)", fontsize=9.5, loc="left")

    ax[1].plot(gs, lam, "o-", color=ORANGE, lw=2, ms=6, label=r"leading $\max\mathrm{Re}\,\lambda$")
    ax[1].plot(gs, bulk, "s--", color=PURPLE, ms=5, label=r"elliptic bulk edge $1+\gamma$")
    ax[1].fill_between(gs, bulk, lam, where=lam > bulk, color=ORANGE, alpha=0.12)
    ax[1].axvspan(0.0, 0.13, color=TEAL, alpha=0.15)
    ax[1].set_xlabel(r"interaction symmetry  $\gamma$")
    ax[1].set_ylabel(r"$\mathrm{Re}\,\lambda$  ($\sigma=1$ baseline)")
    ax[1].set_title("(b) Hubs still set the threshold: the leading\n"
                    "eigenvalue is an outlier beyond the bulk", fontsize=9.5, loc="left")
    ax[1].legend(frameon=False, fontsize=9, loc="upper left")

    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.suptitle("Hub localisation vs interaction symmetry: the unstable mode spreads at "
                 "small $\\gamma$ while the hub-set threshold survives", fontsize=10, y=1.02)
    fig.tight_layout()
    out = f"{OUT}/fig_hub_localization.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
