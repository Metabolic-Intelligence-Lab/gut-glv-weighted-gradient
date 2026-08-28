"""
Does a SPECTRAL quantity predict the measured knockouts, or only the nonlinear model? (referee-9 §9)
====================================================================================================

Referee 9 objects, correctly, that the manuscript validates a *nonlinear in-silico knockout*
ranking against the Venturelli leave-one-out experiments, and then discusses a *spectral*
localization mechanism, without ever showing that the spectral quantity itself predicts the
experiment. Eigenvector localization, eigenvalue sensitivity, node-removal effect on the
spectral abscissa, nonlinear knockout impact and experimental keystone-ness are five different
things.

This script tests them side by side on the one system with ground truth. For the designed
12-species gLV of Venturelli et al., at its feasible coexistence equilibrium n*, it computes
per species:

  spectral    |v_i|          leading right-eigenvector weight of the ecological Jacobian
  spectral    |l_i v_i|      biorthogonal weight = sensitivity of the leading eigenvalue to a
                             diagonal perturbation at node i
  spectral    -d(abscissa)   change in the spectral abscissa when node i is deleted
  topological k_i            weighted degree of |A|
  nonlinear   BC impact      the manuscript's in-silico knockout (relaxation to a new equilibrium)
  confound    n*_i           equilibrium relative abundance

and correlates each with the measured leave-one-out impact (Spearman, with a permutation
p-value and a partial correlation controlling for abundance).

Run:  OMP_NUM_THREADS=4 PYTHONPATH=scripts python3 scripts/spectral_vs_knockout.py
"""
import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from venturelli_keystone_validation import (DATA, SP, S, bc, load_glv,  # noqa: E402
                                            load_leaveoneout, partial_spearman, steady)

import warnings  # noqa: E402
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = "figures"
NPERM = 20000


def eco_jacobian(u, A, x):
    """Ecological Jacobian of the gLV at the equilibrium x: D(x)(A - 1) on the survivors."""
    keep = np.where(x > 1e-9)[0]
    M = np.diag(x[keep]) @ (A[np.ix_(keep, keep)] - np.eye(len(keep)))
    return M, keep


def leading(M):
    w, vr = np.linalg.eig(M)
    k = int(np.argmax(w.real))
    _, vl = np.linalg.eig(M.T)
    kl = int(np.argmax(np.linalg.eigvals(M.T).real))
    return w[k].real, vr[:, k], vl[:, kl]


def perm_p(x, y, rng, nperm=NPERM):
    obs = abs(spearmanr(x, y).statistic)
    y = np.asarray(y)
    hits = sum(abs(spearmanr(x, rng.permutation(y)).statistic) >= obs for _ in range(nperm))
    return (hits + 1) / (nperm + 1)


def main():
    rng = np.random.default_rng(0)
    u, A = load_glv(f"{DATA}/twelveSpeciesT4.xml")
    comm = load_leaveoneout()
    full_emp = comm["NONE"]
    xfull = steady(u, A, np.ones(S, bool))

    M, keep = eco_jacobian(u, A, xfull)
    abscissa, vr, vl = leading(M)
    w_eig = np.zeros(S); w_bio = np.zeros(S)
    w_eig[keep] = np.abs(vr) / np.abs(vr).sum()
    bio = np.abs(vl * vr)
    w_bio[keep] = bio / bio.sum()

    # node deletion: how much does removing i move the spectral abscissa?
    d_abs = np.zeros(S)
    for pos, i in enumerate(keep):
        sub = [k for k in range(len(keep)) if k != pos]
        d_abs[i] = abscissa - np.max(np.linalg.eigvals(M[np.ix_(sub, sub)]).real)

    degree = np.abs(A - np.diag(np.diag(A))).sum(1)

    imp_mod = np.zeros(S); imp_emp = np.zeros(S)
    for i in range(S):
        others = [j for j in range(S) if j != i]
        m = np.ones(S, bool); m[i] = False
        imp_mod[i] = bc(xfull[others], steady(u, A, m)[others])
        imp_emp[i] = bc(full_emp[others], comm[f"{SP[i]}-"][others])

    predictors = [
        ("eigenvector weight |v_i|", w_eig, "spectral"),
        ("biorthogonal |l_i v_i|", w_bio, "spectral"),
        ("node-removal $\\Delta$abscissa", d_abs, "spectral"),
        ("weighted degree", degree, "topological"),
        ("nonlinear knockout (model)", imp_mod, "nonlinear"),
        ("equilibrium abundance", full_emp, "confound"),
    ]

    print(f"leading eigenvalue of the ecological Jacobian: {abscissa:+.4f}  "
          f"({len(keep)}/{S} survivors)\n")
    print(f"{'predictor':<30} {'kind':<12} {'rho':>7} {'p_perm':>8} {'rho|abund':>10}")
    rows = []
    for name, x, kind in predictors:
        rho = spearmanr(x, imp_emp).statistic
        pp = perm_p(x, imp_emp, rng)
        pr = np.nan if kind == "confound" else partial_spearman(x, imp_emp, full_emp)
        rows.append((name, kind, rho, pp, pr))
        print(f"{name:<30} {kind:<12} {rho:>7.3f} {pp:>8.4f} {pr:>10.3f}")

    spec = [r for r in rows if r[1] == "spectral"]
    nl = [r for r in rows if r[1] == "nonlinear"][0]
    print(f"\n=> best spectral predictor: rho={max(abs(r[2]) for r in spec):.3f}; "
          f"nonlinear knockout: rho={abs(nl[2]):.3f}. "
          f"{'The spectral quantities do NOT reproduce the nonlinear ranking' if max(abs(r[2]) for r in spec) < abs(nl[2]) - 0.15 else 'Comparable.'}")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    names = [r[0] for r in rows][::-1]
    vals = [r[2] for r in rows][::-1]
    cols = {"spectral": "#0072B2", "topological": "#999999",
            "nonlinear": "#D55E00", "confound": "#009E73"}
    ax.barh(names, vals, color=[cols[r[1]] for r in rows][::-1])
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel(r"Spearman $\rho$ with the measured leave-one-out impact ($n=12$)")
    ax.set_xlim(-1, 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Spectral localization and experimental keystone-ness are different things",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_spectral_vs_knockout.png", dpi=180, bbox_inches="tight")
    print(f"[fig] {OUT}/fig_spectral_vs_knockout.png written")
    np.savez(f"{OUT}/spectral_vs_knockout.npz",
             names=np.array([r[0] for r in rows]), kind=np.array([r[1] for r in rows]),
             rho=np.array([r[2] for r in rows]), p=np.array([r[3] for r in rows]),
             partial=np.array([r[4] for r in rows]), imp_emp=imp_emp, imp_mod=imp_mod,
             w_eig=w_eig, w_bio=w_bio, d_abs=d_abs, abundance=full_emp)


if __name__ == "__main__":
    main()
