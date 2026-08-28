"""
Hub fragility on the ECOLOGICAL Jacobian  (referee P0-1)
========================================================

The sparse hub analysis (check_sigmac_vs_gamma.py, nonhermitian_sparse.py,
hub_localization_vs_gamma.py) diagonalises the interaction/community matrix. The
true linear-stability operator of a GLV coexistence equilibrium n* is, however, the
abundance-weighted ecological Jacobian

    M = D(n*) (J - I),      D(n*) = diag(n1*, ..., nS*),

since  d n_i/dt = n_i (r_i - n_i + sum_j J_ij n_j)  gives, at a feasible fixed point,
Jacobian_ij = n_i* (J_ij - delta_ij). A topological hub with LOW abundance is
down-weighted, so hub control of stability is not automatic.

This script integrates the GLV to a feasible equilibrium on a sparse Barabasi-Albert
graph with gamma-correlated couplings, then compares the leading (rightmost)
right-eigenvector of J, J-I and D(n*)(J-I) on the surviving subcommunity:
participation 1/IPR, and Spearman correlation with node DEGREE and with ABUNDANCE.
It also reports corr(degree, abundance), the quantity that decides whether weighting
helps or hurts the hubs.

Conclusion (see verdict): in the feasible (competitive, low-gamma) regime relevant to
the microbiome, J-I and the ecological Jacobian give the SAME leading-mode degree
correlation, so hub-associated fragility is robust to abundance weighting -- but it is
a degree-correlated candidate mechanism, not strict single-hub control.

Run:  python3 hub_jacobian.py      (writes figures/fig_hub_jacobian.png)
"""
import numpy as np, networkx as nx
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
np.seterr(all="ignore")
OUT = "figures"

def build(S, m, gamma, sigma0, mu, seed):
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(S, m, seed=int(rng.integers(1 << 30)))
    A = nx.to_numpy_array(G); k = A.sum(1); kmean = k.mean()
    s = sigma0 / np.sqrt(kmean); mm = mu / kmean
    iu = np.triu_indices(S, 1)
    a = rng.normal(size=len(iu[0])); b = rng.normal(size=len(iu[0]))
    J = np.zeros((S, S))
    J[iu]            = -mm + s * a
    J[(iu[1], iu[0])] = -mm + s * (gamma * a + np.sqrt(max(1 - gamma**2, 0)) * b)
    J *= A
    return J, k

def equilibrium(J, r, eps=1e-6, T=350):
    def rhs(t, n):
        n = np.clip(n, 0, None); return n * (r - n + J @ n) + eps
    sol = solve_ivp(rhs, (0, T), 0.4 * np.ones(len(r)), method="LSODA",
                    rtol=1e-7, atol=1e-9, t_eval=[T])
    return np.clip(sol.y[:, -1], 0, None)

def lead_evec(M):
    w, V = np.linalg.eig(M); i = int(np.argmax(w.real))
    v = np.abs(V[:, i]); v = v / (np.linalg.norm(v) + 1e-30)
    return w[i].real, v, 1.0 / np.sum(v**4)

def collect(gamma, sigma0, mu, S=350, m=2, seeds=range(1, 8)):
    rows = {n: {"part": [], "rdeg": [], "rab": []} for n in ["J", "J-I", "D(n)(J-I)"]}
    ka, nsurv, scat = [], [], None
    for sd in seeds:
        J, k = build(S, m, gamma, sigma0, mu, sd)
        n = equilibrium(J, np.ones(S))
        surv = n > 1e-3; idx = np.where(surv)[0]
        if len(idx) < 20:
            continue
        Js = J[np.ix_(idx, idx)]; ns = n[idx]; ks = k[idx]; I = np.eye(len(idx))
        nsurv.append(len(idx)); ka.append(spearmanr(ks, ns).correlation)
        if scat is None:
            scat = (ks.copy(), ns.copy())
        for name, M in [("J", Js), ("J-I", Js - I), ("D(n)(J-I)", ns[:, None] * (Js - I))]:
            lam, v, part = lead_evec(M)
            rows[name]["part"].append(part)
            rows[name]["rdeg"].append(spearmanr(v, ks).correlation)
            rows[name]["rab"].append(spearmanr(v, ns).correlation)
    m_ = lambda x: float(np.nanmean(x)) if len(x) else np.nan
    summ = {n: {kk: m_(vv) for kk, vv in d.items()} for n, d in rows.items()}
    return dict(nsurv=m_(nsurv), ka=m_(ka), scat=scat, summ=summ)

def main():
    import os; os.makedirs(OUT, exist_ok=True)
    # microbiome-relevant: net competition mu>0, empirical low gamma; plus gamma=1 ref
    configs = [("competitive, $\\gamma$=0.1", 0.1, 0.8, 2.0),
               ("competitive, $\\gamma$=1",   1.0, 0.8, 2.0),
               ("zero-mean, $\\gamma$=0.1",   0.1, 0.8, 0.0)]
    res = {}
    print(f"{'config':>22} | {'surv':>4} {'r(k,n)':>7} | "
          f"{'operator':>10} {'part':>5} {'r(v,k)':>7} {'r(v,n)':>7}")
    for label, g, s0, mu in configs:
        r = collect(g, s0, mu); res[label] = r
        for op in ["J", "J-I", "D(n)(J-I)"]:
            d = r["summ"][op]
            print(f"{label:>22} | {r['nsurv']:>4.0f} {r['ka']:>+7.2f} | "
                  f"{op:>10} {d['part']:>5.1f} {d['rdeg']:>+7.2f} {d['rab']:>+7.2f}")

    # ---- figure: (a) degree-abundance; (b) leading-mode degree corr per operator ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ks, ns = res["competitive, $\\gamma$=0.1"]["scat"]
    ax[0].scatter(ks, ns, s=14, alpha=0.5, color="#7570b3")
    rho = res["competitive, $\\gamma$=0.1"]["ka"]
    ax[0].set_xlabel("node degree $k$"); ax[0].set_ylabel("equilibrium abundance $n^*$")
    ax[0].set_title(f"A. Hubs are low-abundance under competition\nSpearman(k,$n^*$)={rho:+.2f} "
                    "($\\gamma$=0.1), so $D(n^*)$ down-weights hubs")
    ops = ["J", "J-I", "D(n)(J-I)"]
    x = np.arange(len(ops)); w = 0.35
    for j, (label, g, s0, mu) in enumerate(configs[:2]):
        rdeg = [res[label]["summ"][o]["rdeg"] for o in ops]
        ax[1].bar(x + (j - 0.5) * w, rdeg, w, label=label)
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].set_xticks(x); ax[1].set_xticklabels(ops)
    ax[1].set_ylabel("Spearman(leading eigenvector, degree)")
    ax[1].set_title("B. Leading-mode degree correlation survives\nabundance weighting: $J-I \\approx D(n^*)(J-I)$")
    ax[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Hub-associated fragility on the ecological Jacobian $D(n^*)(J-I)$", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_hub_jacobian.png", dpi=150)
    plt.close(fig)
    print(f"\n[fig] {OUT}/fig_hub_jacobian.png written")
    verdict = "ROBUST" if all(
        res[c[0]]["summ"]["D(n)(J-I)"]["rdeg"] > 0.2 for c in configs[:2]) else "WEAKENED"
    print(f"verdict: hub-degree correlation of the ecological-Jacobian leading mode: {verdict}")

if __name__ == "__main__":
    main()
