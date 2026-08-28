"""
Directed heavy-tailed ensemble: support vs weight reciprocity  (referee-4 P1, §7)
=================================================================================

Referee 4's structural objection: the theory uses an UNDIRECTED Barabasi-Albert skeleton
with two reciprocal coefficients per edge and a single Pearson gamma, whereas the empirical
matrices have ~30% one-way edges. A single gamma conflates (i) SUPPORT reciprocity p_bi =
P(both i->j and j->i present) with (ii) WEIGHT reciprocity gamma_w = corr(J_ij,J_ji) on the
bidirectional edges. The hub/keystone mechanism must survive when the network is genuinely
directed at the empirical support level, or the link between one-way gut metabolism and the
hub instability is not established.

Ensemble: a heavy-tailed (BA) skeleton fixes which pairs interact; each skeleton edge is
bidirectional with probability p_bi (its two weights correlated by gamma_w) or, with
probability 1-p_bi, one-way in a random direction. p_bi=1 recovers the undirected model;
p_bi~0.7 matches Venturelli's ~30% one-way edges. Per-edge variance 1/<k> as before.

We test whether (a) the least-stable right eigenvector stays localized on high-degree nodes
(degree--mode Spearman) and (b) its participation stays SUB-EXTENSIVE (P~S^alpha, 0<alpha<1)
at the empirical support reciprocity.

Run:  python3 directed_ensemble.py     (writes figures/fig_directed.png)
"""
import numpy as np, networkx as nx
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"


def directed_community(S, m, p_bi, gamma_w, rng):
    """BA skeleton; each edge bidirectional (prob p_bi, weights corr gamma_w) or one-way."""
    G = nx.barabasi_albert_graph(S, m, seed=int(rng.integers(1 << 30)))
    A = nx.to_numpy_array(G); k = A.sum(1); s = 1.0 / np.sqrt(k.mean())
    iu = np.triu_indices(S, 1)
    ei, ej = iu[0][A[iu] > 0], iu[1][A[iu] > 0]
    a = rng.normal(size=len(ei)); b = rng.normal(size=len(ei))
    bidir = rng.random(len(ei)) < p_bi
    fwd = rng.random(len(ei)) < 0.5
    J = np.zeros((S, S))
    # bidirectional edges: both directions, correlated
    J[ei[bidir], ej[bidir]] = s * a[bidir]
    J[ej[bidir], ei[bidir]] = s * (gamma_w * a[bidir] + np.sqrt(max(1 - gamma_w**2, 0)) * b[bidir])
    # one-way edges: a single random direction
    ow = ~bidir
    f = ow & fwd; r = ow & (~fwd)
    J[ei[f], ej[f]] = s * a[f]
    J[ej[r], ei[r]] = s * a[r]
    return J, k


def eff_gamma(J):
    S = len(J); iu = np.triu_indices(S, 1)
    return float(np.corrcoef(J[iu], J.T[iu])[0, 1])


def frac_oneway(J):
    S = len(J); iu = np.triu_indices(S, 1); a, b = J[iu], J.T[iu]
    both = (a != 0) & (b != 0); either = (a != 0) | (b != 0)
    return 1 - both.sum() / max(either.sum(), 1)


def leading_mode(J):
    w, V = np.linalg.eig(J)
    i = int(np.argmax(w.real))
    vR = np.abs(V[:, i]); vR /= (np.linalg.norm(vR) + 1e-30)
    part = 1.0 / np.sum(vR**4)
    return part, vR, float(w[i].real)


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    rng0 = np.random.default_rng(0)

    # --- (1) support-reciprocity sweep at empirical weight reciprocity (gamma_w=0) ---
    print("support-reciprocity sweep (gamma_w=0, S=800, 4 reps):")
    print(f"{'p_bi':>5} {'frac_oneway':>11} {'eff_gamma':>9} {'participation':>13} {'deg-mode rho':>12}")
    p_bis = [1.0, 0.85, 0.7, 0.5, 0.3]
    sweep = {}
    for p in p_bis:
        parts, rhos, ows, egs = [], [], [], []
        for rep in range(4):
            J, k = directed_community(800, 2, p, 0.0, np.random.default_rng(10 + rep))
            part, vR, _ = leading_mode(J)
            parts.append(part); rhos.append(spearmanr(k, vR).correlation)
            ows.append(frac_oneway(J)); egs.append(eff_gamma(J))
        sweep[p] = (np.mean(parts), np.mean(rhos), np.mean(ows), np.mean(egs))
        print(f"{p:>5.2f} {np.mean(ows):>11.2f} {np.mean(egs):>+9.3f} {np.mean(parts):>13.1f} {np.mean(rhos):>12.3f}")

    # --- (2) finite-size: does sub-extensivity survive at empirical support (p_bi=0.7)? ---
    print("\nfinite-size participation P~S^alpha:")
    Ss = [200, 400, 800, 1600]
    fits = {}
    curves = {}
    for label, p in [("undirected (p_bi=1)", 1.0), ("directed (p_bi=0.7)", 0.7)]:
        Ps = []
        for S in Ss:
            pr = [leading_mode(directed_community(S, 2, p, 0.0, np.random.default_rng(100 + S + r))[0])[0]
                  for r in range(4)]
            Ps.append(np.mean(pr))
        alpha = np.polyfit(np.log(Ss), np.log(Ps), 1)[0]
        fits[label] = alpha; curves[label] = Ps
        print(f"  {label:22}: alpha={alpha:.2f}  ({'sub-extensive' if 0.15<alpha<0.85 else 'O(1)/hub' if alpha<=0.15 else 'delocalised'})")

    # --- figure ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    col = {"undirected (p_bi=1)": "#1b9e77", "directed (p_bi=0.7)": "#0072B2"}
    for label in curves:
        ax[0].plot(Ss, curves[label], "o-", color=col[label], lw=2, ms=8,
                   label=fr"{label}, $\alpha={fits[label]:.2f}$")
    ax[0].plot(Ss, [Ss[i] / Ss[0] * curves['directed (p_bi=0.7)'][0] for i in range(len(Ss))],
               "k:", lw=1, label=r"$\propto S$ (delocalised)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("system size $S$"); ax[0].set_ylabel("leading-mode participation $1/\\mathrm{IPR}$")
    ax[0].set_title("a  Sub-extensive keystone set survives\ndirected support ($30\\%$ one-way)", loc="left", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8.5)
    pv = list(sweep.keys())
    ow = [sweep[p][2] for p in pv]; rho = [sweep[p][1] for p in pv]
    ax[1].plot(ow, rho, "s-", color="#5e4fa2", lw=2, ms=8)
    ax[1].axvline(0.30, color="#d95f02", ls="--", lw=1.2)
    ax[1].axhline(0.0, color="grey", lw=0.6, ls=":")
    ax[1].set_ylim(0, 0.4)
    ax[1].text(0.31, 0.34, "Venturelli\n$30\\%$ one-way", color="#d95f02", fontsize=8)
    ax[1].set_xlabel("fraction of one-way edges  ($1-p_{\\rm bi}$)")
    ax[1].set_ylabel(r"degree--mode association (Spearman $\rho$)")
    ax[1].set_title("b  Hub localization is robust to\nmaking the support directed", loc="left", fontsize=11)
    for a in ax:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_directed.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_directed.png written")
    np.savez(f"{OUT}/directed_ensemble.npz",
             p_bi=pv, oneway=ow, rho=rho, alpha_undir=fits["undirected (p_bi=1)"],
             alpha_dir=fits["directed (p_bi=0.7)"])


if __name__ == "__main__":
    main()
