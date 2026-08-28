"""
Finite-size scaling and non-normality of the hub mode  (referee-3 P0.5)
======================================================================

Two questions the single-S localization plot cannot answer:
  (1) Is the least-stable mode's localization SUB-EXTENSIVE? Sweep S and fit the
      participation P(S) = 1/IPR of the leading right eigenvector to P ~ S^alpha:
      alpha~0 (O(1), single-hub), 0<alpha<1 (sub-extensive keystone set), alpha~1
      (delocalized bulk).
  (2) Is the linear-stability reading robust to NON-NORMALITY? For a non-Hermitian
      matrix, an eigenvalue's sensitivity is its condition number
      kappa = 1/|<v_L|v_R>| (v_L,v_R left/right eigvecs, unit norm); large kappa means
      pseudospectrum / transient amplification can dominate the naive spectral abscissa.

Sparse Barabasi-Albert community matrix, per-edge variance 1/<k>, corr(J_ij,J_ji)=gamma.
Run:  python3 hub_finite_size.py     (writes figures/fig_hub_finite_size.png)
"""
import numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"

def community(S, m, gamma, rng):
    G = nx.barabasi_albert_graph(S, m, seed=int(rng.integers(1 << 30)))
    A = nx.to_numpy_array(G); k = A.sum(1); s = 1.0 / np.sqrt(k.mean())
    iu = np.triu_indices(S, 1)
    a = rng.normal(size=len(iu[0])); b = rng.normal(size=len(iu[0]))
    J = np.zeros((S, S))
    J[iu] = s * a
    J[(iu[1], iu[0])] = s * (gamma * a + np.sqrt(max(1 - gamma**2, 0)) * b)
    J *= A
    return J, k

def leading(J):
    w, V = np.linalg.eig(J)
    i = int(np.argmax(w.real))
    vR = V[:, i] / (np.linalg.norm(V[:, i]) + 1e-30)
    part = 1.0 / np.sum(np.abs(vR)**4)
    Vn = V / (np.linalg.norm(V, axis=0) + 1e-30)
    try:
        kappa = np.linalg.norm(np.linalg.inv(Vn)[i, :])   # cond. number of leading eigenvalue
    except np.linalg.LinAlgError:
        kappa = np.nan
    # right-eigenvector participation on the top-degree nodes (hub weight)
    return float(part), float(kappa), float(w[i].real)

def main():
    import os; os.makedirs(OUT, exist_ok=True)
    Ss = [200, 400, 800, 1600]
    gammas = [1.0, 0.1]
    res = {g: {"S": [], "P": [], "Perr": [], "kap": []} for g in gammas}
    for g in gammas:
        print(f"gamma={g}:  {'S':>5} {'partic.':>8} {'kappa':>8}")
        for S in Ss:
            ps, ks = [], []
            for rep in range(5):
                J, k = community(S, 2, g, np.random.default_rng(100 + S + rep))
                p, kap, _ = leading(J)
                ps.append(p); ks.append(kap)
            res[g]["S"].append(S); res[g]["P"].append(np.mean(ps))
            res[g]["Perr"].append(np.std(ps)); res[g]["kap"].append(np.median(ks))
            print(f"        {S:>5} {np.mean(ps):>8.1f} {np.median(ks):>8.1f}")
    fits = {}
    for g in gammas:
        a, _ = np.polyfit(np.log(res[g]["S"]), np.log(res[g]["P"]), 1)
        fits[g] = a
        print(f"gamma={g}: participation exponent alpha = {a:.2f}  "
              f"({'O(1)/hub' if a<0.15 else 'sub-extensive' if a<0.85 else 'delocalised'})")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    col = {1.0: "#1b9e77", 0.1: "#0072B2"}
    for g in gammas:
        ax[0].errorbar(res[g]["S"], res[g]["P"], yerr=res[g]["Perr"], fmt="o-",
                       color=col[g], capsize=3, label=fr"$\gamma={g}$ ($\alpha={fits[g]:.2f}$)")
        ax[1].plot(res[g]["S"], res[g]["kap"], "s-", color=col[g], label=fr"$\gamma={g}$")
    ax[0].plot(Ss, [Ss[i]/Ss[0]*res[0.1]['P'][0] for i in range(len(Ss))], "k:", lw=1, label=r"$\propto S$ (delocalised)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("system size $S$"); ax[0].set_ylabel("leading-mode participation $1/\\mathrm{IPR}$")
    ax[0].set_title("a  Sub-extensive localization: $P\\sim S^{\\alpha}$, $\\alpha<1$", loc="left", fontsize=12)
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("system size $S$"); ax[1].set_ylabel("leading-eigenvalue condition number $\\kappa$")
    ax[1].set_title("b  Non-normality of the least-stable mode", loc="left", fontsize=12)
    ax[1].legend(frameon=False, fontsize=9)
    for a in ax:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_hub_finite_size.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_hub_finite_size.png written")

if __name__ == "__main__":
    main()
