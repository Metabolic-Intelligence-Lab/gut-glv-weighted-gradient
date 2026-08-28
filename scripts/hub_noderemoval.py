"""
Node-removal sensitivity of the least-stable mode, and G_max scaling  (referee-6 P1, §9.3)
==========================================================================================

The right eigenvector alone does not prove that the leading eigenvalue is sensitive to the
hubs. The referee asks for the biorthogonal sensitivity |l_i r_i| and the ACTUAL change in the
spectral abscissa when a node is deleted, and their correlation with degree. We also scale the
transient gain G_max with S to check the O(1) reading is not a single-size artefact.

Barabasi-Albert community matrix, gamma-correlated couplings (as in hub_finite_size).
Run:  python3 hub_noderemoval.py     (writes figures/fig_noderemoval.png)
"""
import numpy as np
from scipy.linalg import expm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hub_finite_size import community
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"


def leading_lr(M):
    """Leading (rightmost) eigenvalue and its unit right/left eigenvectors."""
    w, VR = np.linalg.eig(M); i = int(np.argmax(w.real))
    wl, VL = np.linalg.eig(M.T); j = int(np.argmin(np.abs(wl - w[i])))
    r = VR[:, i] / np.linalg.norm(VR[:, i]); l = VL[:, j] / np.linalg.norm(VL[:, j])
    return w[i], l, r


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    gamma = 0.1

    # --- node-removal sensitivity at fixed S ---
    S = 200
    deg_all, dalpha_all, sens_all = [], [], []
    for rep in range(4):
        J, k = community(S, 2, gamma, np.random.default_rng(30 + rep))
        w0, l, r = leading_lr(J)
        sens = np.abs(l) * np.abs(r)                      # biorthogonal node weight |l_i r_i|
        dalpha = np.zeros(S)
        for i in range(S):
            keep = np.arange(S) != i
            wi = np.linalg.eigvals(J[np.ix_(keep, keep)])
            dalpha[i] = w0.real - wi.real.max()           # drop in spectral abscissa on removing i
        deg_all.append(k); dalpha_all.append(dalpha); sens_all.append(sens)
    deg = np.concatenate(deg_all); dalpha = np.concatenate(dalpha_all); sens = np.concatenate(sens_all)
    from scipy.stats import spearmanr
    print(f"node-removal at S={S}, gamma={gamma}, 4 reps:")
    print(f"  Spearman(degree, spectral-abscissa drop on removal) = {spearmanr(deg, dalpha).correlation:+.2f}")
    print(f"  Spearman(degree, biorthogonal sensitivity |l r|)     = {spearmanr(deg, sens).correlation:+.2f}")
    print(f"  Spearman(|l r|, spectral-abscissa drop)              = {spearmanr(sens, dalpha).correlation:+.2f}")

    # --- G_max scaling with S ---
    Ss = [100, 200, 400, 800]; tg = np.linspace(0, 15, 40)
    Gs = []
    for Sz in Ss:
        g = []
        for rep in range(3):
            J, _ = community(Sz, 2, gamma, np.random.default_rng(60 + Sz + rep))
            a = np.max(np.linalg.eigvals(J).real); M = J - (a + 0.1) * np.eye(Sz)  # abscissa -0.1
            g.append(max(np.linalg.norm(expm(M * t), 2) for t in tg))
        Gs.append(np.mean(g))
    slope = np.polyfit(np.log(Ss), np.log(Gs), 1)[0]
    print(f"\nG_max vs S: {[round(x,2) for x in Gs]} at S={Ss}; log-log slope={slope:+.2f} "
          f"({'grows with S' if slope>0.15 else 'roughly O(1)'})")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    ax[0].scatter(deg, dalpha, s=14, alpha=0.5, color="#0072B2", edgecolor="none")
    ax[0].axhline(0, color="grey", lw=0.6, ls=":")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("node degree $k_i$"); ax[0].set_ylabel(r"spectral-abscissa drop on removing $i$")
    ax[0].set_title(fr"a  Degree does NOT predict the abscissa drop on removal ($\rho={spearmanr(deg,dalpha).correlation:.2f}$):" "\nlocalization $\\neq$ stability control", loc="left", fontsize=9.5)
    ax[1].plot(Ss, Gs, "o-", color="#D55E00", lw=2, ms=8)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("system size $S$"); ax[1].set_ylabel(r"transient gain $G_{\max}$")
    ax[1].set_title(fr"b  $G_{{\max}}$ grows modestly with $S$ (slope ${slope:+.2f}$):" "\nnot asymptotically $O(1)$", loc="left", fontsize=9.5)
    for a in ax:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_noderemoval.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_noderemoval.png written")
    np.savez(f"{OUT}/noderemoval.npz", deg=deg, dalpha=dalpha, sens=sens, Gs=np.array(Gs), Gslope=slope)


if __name__ == "__main__":
    main()
