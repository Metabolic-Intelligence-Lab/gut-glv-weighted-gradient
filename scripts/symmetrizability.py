"""DEPRECATED (not used by either manuscript).
Computes the Frobenius-ratio residual R_sym, which is NOT invariant under positive
species-wise rescaling: a diagonal congruence reweights the two norms unequally and
collapses it by ~1e3 on a fixed non-symmetrizable matrix. It was replaced by the
invariant triple (f_one, f_sign, r_cycle) in symmetrizability_invariants.py, which is
what the manuscript reports. Kept only so the retraction in the text can be checked.


Coordinate-invariant test of non-potentiality  (referee-3 P0.1)
===============================================================

The Pearson reciprocity gamma = corr(J_ij, J_ji) is NOT coordinate-invariant: a
per-species rescaling x_i = s_i y_i sends a symmetric A to J' = A diag(s), which is
not symmetric (gamma drops) yet is still diagonally symmetrizable (diag(s) J' is
symmetric) -- a weighted gradient flow / potential still exists. So low gamma alone
does not prove non-potentiality.

The invariant test is the symmetrization residual
    R_sym(J) = min_{d>0} || diag(d) J - J^T diag(d) ||_F / || diag(d) J ||_F,
which is 0 iff J is diagonally symmetrizable (a weighted quadratic potential exists).

Result: for exactly-symmetrizable matrices R_sym=0 at any gamma (down to the empirical
band); but the real experimentally-constrained matrices are genuinely non-symmetrizable
-- Venturelli R_sym~0.49, Clark R_sym~0.38 (vs ~1 for fully asymmetric, ~0 for
symmetrizable). Their low reciprocity therefore reflects real non-potentiality, not a
scaling artefact. Venturelli additionally has ~30% unidirectional edges (support
non-reciprocity), which no diagonal rescaling can symmetrize.

Run:  python3 symmetrizability.py      (writes figures/fig_symmetrizability.png)
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "figures"

def gamma_pearson(J, bidir_only=True):
    S = len(J); iu = np.triu_indices(S, 1); a, b = J[iu], J.T[iu]
    m = ((a != 0) & (b != 0)) if bidir_only else np.ones(len(a), bool)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan

def R_sym(J, restarts=6, maxit=800):
    S = len(J)
    def obj(u):
        d = np.exp(np.clip(u - u.mean(), -12, 12))
        M = d[:, None] * J - J.T * d[None, :]
        return np.sum(M**2) / (np.sum((d[:, None] * J)**2) + 1e-12)
    best = np.inf
    for s in range(restarts):
        r = minimize(obj, np.random.default_rng(s).normal(0, 1.0, S), method="L-BFGS-B",
                     bounds=[(-14, 14)] * S, options={"maxiter": maxit})
        best = min(best, r.fun)
    return float(np.sqrt(max(best, 0.0)))

def sym_gauss(S, rng):
    A = rng.normal(size=(S, S)); A = np.triu(A, 1) + np.triu(A, 1).T
    np.fill_diagonal(A, -abs(rng.normal(1, 0.2, S))); return A

def frac_unidir(J):
    S = len(J); iu = np.triu_indices(S, 1); a, b = J[iu], J.T[iu]
    return 1 - ((a != 0) & (b != 0)).sum() / max(((a != 0) | (b != 0)).sum(), 1)

def main():
    import os; os.makedirs(OUT, exist_ok=True)
    V = np.loadtxt(f"data/venturelli2018/interaction_matrix_averaged.csv", delimiter=",", skiprows=1)
    C = np.loadtxt(f"data/clark2021/interaction_matrix_RLC8_posteriormean.csv", delimiter=",")

    # symmetrizable locus: symmetric A, rescaled by lognormal(s); gamma falls, R_sym stays 0
    print("symmetrizable locus (symmetric A -> J'=A diag(s)):")
    loc_g, loc_r = [], []
    for sd in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        gs, rs = [], []
        for k in range(12):
            rng = np.random.default_rng(k); A = sym_gauss(20, rng)
            Jp = A * np.exp(rng.normal(0, sd, 20))[None, :]
            gs.append(gamma_pearson(Jp, False)); rs.append(R_sym(Jp, restarts=3, maxit=500))
        loc_g.append(np.median(gs)); loc_r.append(np.median(rs))
        print(f"  sd={sd:.1f}: gamma={np.median(gs):+.3f}  R_sym={np.median(rs):.4f}")

    rng = np.random.default_rng(0)
    pts = [("symmetric", sym_gauss(12, rng), "#1b9e77"),
           ("asymmetric", np.where(np.eye(12) == 1, -1.0, rng.normal(size=(12, 12))), "#7f7f7f"),
           ("Venturelli", V, "#0072B2"), ("Clark", C, "#0072B2")]
    print("\nmatrices:  name  gamma  R_sym  frac_unidir")
    P = []
    for n, J, col in pts:
        g, r, u = gamma_pearson(J), R_sym(J, restarts=8, maxit=1200), frac_unidir(J)
        P.append((n, g, r, col)); print(f"  {n:>12}  {g:+.3f}  {r:.4f}  {u:.2f}")

    # ---- figure: R_sym vs Pearson gamma ----
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axvspan(-0.05, 0.13, color="#0072B2", alpha=0.10)
    ax.text(0.04, 0.9, "empirical\n$\\gamma$ band", color="#0a4c7a", fontsize=9, ha="center")
    ax.plot(loc_g, loc_r, "o-", color="#1b9e77", lw=2, ms=7,
            label="exactly symmetrizable (symmetric $\\times$ rescaling)")
    ax.annotate("a potential system\ncan reach the empirical $\\gamma$\nwith $R_{\\rm sym}\\approx0$",
                xy=(loc_g[-1], loc_r[-1]), xytext=(0.35, 0.12), fontsize=9, color="#0a6b4e",
                arrowprops=dict(arrowstyle="->", color="#0a6b4e"))
    for n, g, r, col in P:
        mk = "*" if n in ("Venturelli", "Clark") else "s"
        ms = 22 if mk == "*" else 11
        ax.scatter(g, r, marker=mk, s=ms**2 if mk == "*" else 130, color=col,
                   edgecolor="white", linewidth=1.2, zorder=5)
        ax.annotate(n, (g, r), (g + 0.03, r + 0.03), fontsize=10,
                    fontweight="bold" if n in ("Venturelli", "Clark") else "normal")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(r"Pearson reciprocity  $\gamma=\mathrm{corr}(J_{ij},J_{ji})$")
    ax.set_ylabel(r"symmetrization residual  $R_{\rm sym}$  (0 = potential exists)")
    ax.set_title("Low $\\gamma$ is not enough; $R_{\\rm sym}$ decides potentiality.\n"
                 "The real matrices are genuinely non-symmetrizable.", fontsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_symmetrizability.png", dpi=170, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_symmetrizability.png written")

if __name__ == "__main__":
    main()
