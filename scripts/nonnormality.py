"""
Full non-normality diagnostics of the least-stable mode  (referee-4 P1, §9.3)
=============================================================================

The manuscript claimed the spectral-abscissa (linear-stability) reading "survives an O(1)
pseudospectral correction" from the leading-eigenvalue condition number kappa~3-4. Referee 4
is right that kappa alone bounds neither the transient gain nor the pseudospectrum. We
therefore compute the quantities that actually settle it, at matched asymptotic stability
(shift each operator so its spectral abscissa is -0.1), for gamma=1 (normal) vs gamma=0.1
(empirical), on the bare community matrix J and the abundance-weighted ecological Jacobian
D(n*)(J-1):

  alpha(M)  spectral abscissa  = max Re lambda           (asymptotic growth; fixed to -0.1)
  omega(M)  numerical abscissa = lambda_max[(M+M^T)/2]   (initial growth of ||x||)
  Gmax      = max_t ||exp(Mt)||_2                        (peak transient amplification)
  K         Kreiss constant (lower bound on Gmax)        sup_{eta>0} eta ||(eta I - M_0)^{-1}||
  kappa, s  leading-eigenvalue condition number 1/|<vL|vR>| and biorthogonal sensitivity

If omega<=0 there is no transient growth at all; if Gmax=O(1) the linear reading is
qualitatively safe; a large Gmax would demand softening. Reported honestly either way.

Run:  python3 nonnormality.py     (writes figures/fig_nonnormality.png)
"""
import numpy as np
from scipy.linalg import expm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hub_finite_size import community
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"


def shift_to_margin(M, margin=0.1):
    """Shift M by a multiple of I so its spectral abscissa is -margin."""
    a = np.max(np.linalg.eigvals(M).real)
    return M - (a + margin) * np.eye(len(M))


def diagnostics(M, tgrid):
    w = np.linalg.eigvals(M)
    alpha = float(w.real.max())
    Sym = 0.5 * (M + M.T)
    omega = float(np.linalg.eigvalsh(Sym).max())
    # transient envelope G(t)=||exp(Mt)||_2
    G = np.array([np.linalg.norm(expm(M * t), 2) for t in tgrid])
    Gmax = float(G.max()); tmax = float(tgrid[int(G.argmax())])
    # Kreiss lower bound: shift so abscissa=0, sup_{eta>0} eta ||(eta I - M0)^{-1}||
    M0 = M - alpha * np.eye(len(M))
    K = 0.0
    for eta in np.linspace(0.05, 3.0, 20):
        r = eta * np.linalg.norm(np.linalg.inv(eta * np.eye(len(M)) - M0), 2)
        K = max(K, r)
    # leading-eigenvalue condition number and biorthogonal sensitivity: right eigvec of M
    # and left eigvec of M (= right eigvec of M^T), matched by eigenvalue.
    wR, VR = np.linalg.eig(M); i = int(np.argmax(wR.real))
    wL, VL = np.linalg.eig(M.T)
    jl = int(np.argmin(np.abs(wL - wR[i])))
    vR = VR[:, i] / np.linalg.norm(VR[:, i])
    vL = VL[:, jl] / np.linalg.norm(VL[:, jl])
    overlap = abs(np.vdot(vL, vR))
    kappa = 1.0 / (overlap + 1e-30)
    return dict(alpha=alpha, omega=omega, Gmax=Gmax, tmax=tmax, K=K, kappa=float(kappa),
                overlap=float(overlap), G=G)


def eco_jacobian(J, rng):
    """Abundance-weighted ecological Jacobian D(n*)(J-I). We construct a guaranteed-feasible
    coexistence equilibrium by choosing a heterogeneous positive n* (lognormal) and setting
    the growth rates r=n*-Jn* so that n* solves the GLV fixed point exactly; the Jacobian
    there is D(n*)(J-1) regardless of r."""
    S = len(J)
    nstar = np.exp(rng.normal(0.0, 0.5, S))          # positive, heterogeneous abundances
    return np.diag(nstar) @ (J - np.eye(S))


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    S, m, reps = 200, 2, 4
    tgrid = np.concatenate([np.linspace(0, 5, 45), np.linspace(5.2, 40, 25)])
    gammas = [1.0, 0.1]
    res = {g: {"comm": [], "eco": []} for g in gammas}
    Gcurves = {}
    for g in gammas:
        for rep in range(reps):
            rng = np.random.default_rng(200 + rep)
            J, k = community(S, m, g, rng)
            Mc = shift_to_margin(J)
            dc = diagnostics(Mc, tgrid); res[g]["comm"].append(dc)
            if rep == 0:
                Gcurves[g] = (tgrid, dc["G"])
            Me = eco_jacobian(J, rng)
            if Me is not None:
                res[g]["eco"].append(diagnostics(shift_to_margin(Me), tgrid))
            print(f"  done gamma={g} rep={rep+1}/{reps} (comm Gmax={dc['Gmax']:.2f})", flush=True)

    def agg(lst, key):
        v = np.array([d[key] for d in lst]); return v.mean(), v.std()
    print(f"S={S}, BA m={m}, {reps} reps, operators shifted to spectral abscissa = -0.1\n")
    print(f"{'':>16} | {'omega (num.abs)':>16} | {'Gmax (transient)':>17} | {'Kreiss':>8} | {'kappa':>7}")
    print("-" * 78)
    for g in gammas:
        for which in ("comm", "eco"):
            L = res[g][which]
            if not L:
                print(f"gamma={g} {which:>4}: (no feasible equilibrium)"); continue
            om, oe = agg(L, "omega"); gm, ge = agg(L, "Gmax"); kr, _ = agg(L, "K"); ka, _ = agg(L, "kappa")
            tag = "community J" if which == "comm" else "eco Jacobian"
            print(f"gamma={g:<3} {tag:>11} | {om:>7.2f}+/-{oe:<5.2f} | {gm:>7.2f}+/-{ge:<5.2f}   | {kr:>8.2f} | {ka:>7.2f}")

    gm01 = agg(res[0.1]["comm"], "Gmax")[0]; gm1 = agg(res[1.0]["comm"], "Gmax")[0]
    verdict = ("Gmax is O(1) -> transient amplification bounded; linear-stability reading is "
               "qualitatively safe but Gmax must be stated" if gm01 < 10 else
               "Gmax is large -> transient/pseudospectral effects are NOT O(1); soften the claim")
    print(f"\nGmax: gamma=1 -> {gm1:.2f} (near-normal), gamma=0.1 -> {gm01:.2f}. {verdict}")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    col = {1.0: "#1b9e77", 0.1: "#0072B2"}
    for g in gammas:
        t, G = Gcurves[g]
        ax[0].plot(t, G, color=col[g], lw=2, label=fr"$\gamma={g}$ ($G_{{\max}}={G.max():.1f}$)")
    ax[0].axhline(1.0, color="grey", ls=":", lw=1)
    ax[0].set_xlabel("time $t$"); ax[0].set_ylabel(r"$\|e^{Mt}\|_2$ (transient amplification)")
    ax[0].set_xlim(0, 25)
    ax[0].set_title("a  Transient growth at matched stability\n(normal $\\Rightarrow$ no growth)", loc="left", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9)
    labels = [r"$\alpha$", r"$\omega$", r"$G_{\max}$", r"$\kappa$"]
    x = np.arange(len(labels)); w = 0.35
    for gi, g in enumerate(gammas):
        L = res[g]["comm"]
        vals = [agg(L, "alpha")[0], agg(L, "omega")[0], agg(L, "Gmax")[0], agg(L, "kappa")[0]]
        ax[1].bar(x + (gi - 0.5) * w, vals, w, color=col[g], label=fr"$\gamma={g}$")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_title("b  Stability vs non-normality diagnostics\n(community matrix)", loc="left", fontsize=11)
    ax[1].legend(frameon=False, fontsize=9)
    for a in ax:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_nonnormality.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_nonnormality.png written")
    eco01 = res[0.1]["eco"]; eco1 = res[1.0]["eco"]
    np.savez(f"{OUT}/nonnormality.npz",
             gamma01_Gmax=gm01, gamma1_Gmax=gm1,
             omega01=agg(res[0.1]['comm'], 'omega')[0], kappa01=agg(res[0.1]['comm'], 'kappa')[0],
             eco01_Gmax=(agg(eco01, 'Gmax')[0] if eco01 else np.nan),
             eco01_omega=(agg(eco01, 'omega')[0] if eco01 else np.nan),
             eco1_Gmax=(agg(eco1, 'Gmax')[0] if eco1 else np.nan))
    print(f"ECO Jacobian: gamma=1 Gmax={agg(eco1,'Gmax')[0]:.2f}, gamma=0.1 Gmax={agg(eco01,'Gmax')[0]:.2f} omega={agg(eco01,'omega')[0]:.2f}")


if __name__ == "__main__":
    main()
