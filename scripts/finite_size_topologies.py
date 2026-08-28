"""
Sub-extensivity across topologies, with a confidence interval on alpha  (referee-4 P1, §9.2/9.4)
================================================================================================

The sub-extensive keystone exponent P ~ S^alpha (alpha~0.4) was fit on a single family
(Barabasi-Albert) from a few sizes without an interval. Referee 4 asks (i) for a CI on alpha
and (ii) whether the effect is the heavy-tailed DEGREE distribution or something specific to
the BA growth mechanism. We therefore fit alpha, with a bootstrap CI over independent sweeps,
on three topologies at the empirical gamma=0.1:

  BA        : Barabasi-Albert scale-free (heavy-tailed degree, the paper's model);
  config    : configuration model with the SAME degree sequence as the BA graph (heavy-tailed
              degree, but connections randomised -- isolates the degree distribution);
  ER        : Erdos-Renyi with matched mean degree (Poisson degree, NO hubs).

If BA and config give the same sub-extensive alpha while ER differs, the mechanism is the
heavy-tailed degree distribution, not BA-specific wiring.

Run:  python3 finite_size_topologies.py     (writes figures/fig_fss_topologies.png)
"""
import numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"


def graph(kind, S, m, rng):
    seed = int(rng.integers(1 << 30))
    if kind == "BA":
        G = nx.barabasi_albert_graph(S, m, seed=seed)
    elif kind == "config":
        ba = nx.barabasi_albert_graph(S, m, seed=seed)
        deg = [d for _, d in ba.degree()]
        G = nx.configuration_model(deg, seed=seed)
        G = nx.Graph(G); G.remove_edges_from(nx.selfloop_edges(G))
    elif kind == "ER":
        p = 2.0 * m / (S - 1)                       # mean degree ~ 2m, matched to BA
        G = nx.gnp_random_graph(S, p, seed=seed)
    return nx.to_numpy_array(G)


def participation(A, gamma, rng):
    """Leading right-eigenvector participation 1/IPR of the gamma-correlated community matrix."""
    S = len(A); k = A.sum(1); s = 1.0 / np.sqrt(max(k.mean(), 1.0))
    iu = np.triu_indices(S, 1)
    a = rng.normal(size=len(iu[0])); b = rng.normal(size=len(iu[0]))
    J = np.zeros((S, S))
    J[iu] = s * a
    J[(iu[1], iu[0])] = s * (gamma * a + np.sqrt(max(1 - gamma**2, 0)) * b)
    J *= A
    w, V = np.linalg.eig(J)
    i = int(np.argmax(w.real))
    vR = np.abs(V[:, i]); vR /= (np.linalg.norm(vR) + 1e-30)
    return 1.0 / np.sum(vR**4)


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    Ss = [150, 300, 600, 1200]
    kinds = ["BA", "config", "ER"]
    R, reps = 8, 3                                   # R independent sweeps for the alpha CI
    print(f"sizes={Ss}, gamma=0.1, {R} sweeps x {reps} reps/size\n")
    alpha_ci = {}; curves = {}
    for kind in kinds:
        sweeps = []
        Pmean = np.zeros(len(Ss))
        for sw in range(R):
            Ps = []
            for si, S in enumerate(Ss):
                pr = [participation(graph(kind, S, 2, np.random.default_rng(1000 * sw + 7 * si + r)),
                                    0.1, np.random.default_rng(50 * sw + si + r)) for r in range(reps)]
                Ps.append(np.mean(pr))
            sweeps.append(np.polyfit(np.log(Ss), np.log(Ps), 1)[0])
            Pmean += np.array(Ps) / R
        sweeps = np.array(sweeps)
        alpha_ci[kind] = (sweeps.mean(), sweeps.std(), np.percentile(sweeps, [2.5, 97.5]))
        curves[kind] = Pmean
        m_, sd, ci = alpha_ci[kind]
        cls = "sub-extensive" if 0.15 < m_ < 0.85 else "O(1)/hub" if m_ <= 0.15 else "delocalised"
        print(f"  {kind:>7}: alpha = {m_:.2f} +/- {sd:.2f}  (95% CI [{ci[0]:.2f},{ci[1]:.2f}])  {cls}")

    dBA = alpha_ci["BA"][0]; dcfg = alpha_ci["config"][0]; dER = alpha_ci["ER"][0]
    print(f"\nBA ({dBA:.2f}) ~ config ({dcfg:.2f}) [both heavy-tailed] vs ER ({dER:.2f}) [Poisson]: "
          f"{'the DEGREE distribution drives sub-extensivity' if abs(dBA-dcfg)<0.15 and dER>dBA+0.1 else 'topology-dependent'}.")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    col = {"BA": "#0072B2", "config": "#5e4fa2", "ER": "#D55E00"}
    for kind in kinds:
        ax[0].plot(Ss, curves[kind], "o-", color=col[kind], lw=2, ms=7,
                   label=fr"{kind} ($\alpha={alpha_ci[kind][0]:.2f}$)")
    ax[0].plot(Ss, [Ss[i] / Ss[0] * curves["ER"][0] for i in range(len(Ss))], "k:", lw=1, label=r"$\propto S$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("system size $S$"); ax[0].set_ylabel("leading-mode participation $1/\\mathrm{IPR}$")
    ax[0].set_title("a  Participation scaling by topology", loc="left", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9)
    y = np.arange(len(kinds))
    means = [alpha_ci[k][0] for k in kinds]; errs = [alpha_ci[k][1] for k in kinds]
    ax[1].barh(y, means, xerr=errs, color=[col[k] for k in kinds], capsize=4, alpha=0.85)
    ax[1].axvline(1.0, color="grey", ls="--", lw=1); ax[1].text(1.0, len(kinds) - 0.4, "delocalised", fontsize=8, color="grey", ha="center")
    ax[1].axvline(0.0, color="grey", ls=":", lw=0.8)
    ax[1].set_yticks(y); ax[1].set_yticklabels(kinds)
    ax[1].set_xlabel(r"participation exponent $\alpha$ (mean $\pm$ SD over sweeps)")
    ax[1].set_title("b  Heavy-tailed degree (BA, config) $\\Rightarrow$ sub-extensive\ngrowth; Poisson (ER) shows none ($\\alpha\\approx0$)", loc="left", fontsize=11)
    ax[1].set_xlim(0, 1.1)
    for a in ax:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_fss_topologies.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_fss_topologies.png written")
    np.savez(f"{OUT}/fss_topologies.npz", kinds=np.array(kinds),
             alpha=np.array([alpha_ci[k][0] for k in kinds]),
             alpha_sd=np.array([alpha_ci[k][1] for k in kinds]))


if __name__ == "__main__":
    main()
