"""
Interaction symmetry gamma from bottom-up designed-community gut matrices
=========================================================================

The central empirical claim, gut interactions are weakly reciprocal (gamma small,
far from the symmetric-glass gamma=1), is strongest when read from interaction
matrices that were *measured* bottom-up rather than inferred from observational
series. This figure collects the measured communities on one gamma axis.

Convention (single, used everywhere here)
-----------------------------------------
gamma = corr(a_ij, a_ji) over ALL off-diagonal ordered pairs (i != j, every
pair, not just the upper triangle), so the value is invariant under relabeling
of the species order; scale-invariant too, so the diagonal normalisation used
for mu/sigma does not affect it (`gamma_offdiag`). Structural zeros (unmodeled
interactions, exactly a_ij=0) are INCLUDED, matching the deposited posterior
draws. The magnitude is sensitive to this choice: for the sparse Clark
posterior-mean matrix 380/600 pairs are exactly (0,0), and restricting to the
220 modeled pairs would give gamma=-0.52 instead of -0.26 (`gamma_modeled`,
printed for transparency, not plotted).

  - Venturelli 2018: 12-species synthetic human-gut community, gLV fit to mono-,
    pairwise- and multi-species designed co-cultures. Central value = plug-in of
    the averaged matrix (gamma=+0.06). Uncertainty = spread across the measured
    designed-community fits themselves (twelveSpeciesT{1..4}.xml): T1 is a
    monoculture-only fit with no off-diagonal structure (gamma undefined, skipped);
    the three informative fits give +0.04, +0.06, +0.09, i.e. strictly positive.
    This replicate spread is the honest measured uncertainty; a single-matrix pair
    bootstrap of the averaged matrix straddles zero ([-0.04,+0.17]) but that band
    is an artefact of one averaged matrix, not of the three measured fits.
  - Clark 2021: 25-species synthetic human-gut community (same lab/convention),
    Bayesian gLV. Central value = median over the RLC8 posterior draws (the
    principled Bayesian summary, gamma=-0.06, 95% CI [-0.25,0.10]). The plug-in of
    the sparse posterior-mean matrix gives gamma=-0.26 (recomputed from the
    deposited 25x25 matrix over all ordered off-diagonal pairs); it differs from
    the ensemble median because the mean matrix is 63% exact zeros (Bayesian
    shrinkage) while individual draws are continuous. See
    data/clark2021/PROVENANCE.txt.

NOTE on comparability: the two plotted central values are different kinds of
estimator (Venturelli = plug-in of the averaged matrix; Clark = posterior median),
because Clark provides a full Bayesian posterior while Venturelli provides a small
set of point fits. Both are weakly reciprocal and far from gamma=1: Venturelli is
weakly positive (all measured fits > 0), Clark is marginally negative.

Run:  python3 measured_gamma_figure.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"

SPECIES12 = ["BH", "BO", "BT", "BU", "BV", "CA", "CH", "DP", "EL", "ER", "FP", "PC"]
SBML_NS = "{http://www.sbml.org/sbml/level3/version1/core}"


def _offdiag_pairs(A):
    """All off-diagonal ordered pairs (a_ij, a_ji), i != j, of A.

    Uses every ordered pair (not just the upper triangle) so the correlation
    is invariant under relabeling of the species order: a_ij and a_ji each
    appear once as x and once as y, making corr(x, y) symmetric in the two
    vectors and independent of the arbitrary node ordering.
    """
    m = ~np.eye(A.shape[0], dtype=bool)
    return A[m], A.T[m]


def gamma_offdiag(A):
    """gamma = corr(a_ij, a_ji) over all off-diagonal pairs (structural zeros in)."""
    x, y = _offdiag_pairs(A)
    return float(np.corrcoef(x, y)[0, 1])


def gamma_modeled(A):
    """Same correlation restricted to pairs with at least one nonzero coefficient."""
    x, y = _offdiag_pairs(A)
    m = (x != 0) | (y != 0)
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum()), int(len(x))


def _load_sbml_matrix(path):
    """Reconstruct the 12x12 interaction matrix from a Venturelli SBML fit file."""
    import xml.etree.ElementTree as ET
    P = {p.get("id"): float(p.get("value"))
         for p in ET.parse(path).getroot().iter(SBML_NS + "parameter")
         if p.get("value") is not None}
    A = np.zeros((len(SPECIES12), len(SPECIES12)))
    for i, a in enumerate(SPECIES12):
        for j, b in enumerate(SPECIES12):
            A[i, j] = P.get(f"a{a}{b}", 0.0)   # a_{ab} = effect of b on a
    return A


def venturelli():
    A = np.loadtxt("data/venturelli2018/interaction_matrix_averaged.csv",
                   delimiter=",", skiprows=1)
    g = gamma_offdiag(A)                                  # plug-in of averaged matrix
    reps = []
    for T in (1, 2, 3, 4):
        p = f"data/venturelli2018/twelveSpeciesT{T}.xml"
        if not os.path.exists(p):
            continue
        M = _load_sbml_matrix(p)
        if np.any(M[~np.eye(12, dtype=bool)] != 0):       # skip monoculture-only fit
            reps.append(gamma_offdiag(M))
    reps = np.array(sorted(reps))
    # bootstrap over off-diagonal pairs of the averaged matrix (secondary, printed)
    x, y = _offdiag_pairs(A)
    rng = np.random.default_rng(0)
    bs = [np.corrcoef(x[s], y[s])[0, 1]
          for s in (rng.integers(0, len(x), len(x)) for _ in range(5000))]
    boot_ci = np.percentile(bs, [2.5, 97.5])
    return g, reps, boot_ci


def clark():
    gs = np.loadtxt("data/clark2021/gamma_posterior_samples.csv", delimiter=",")
    ndistinct = len(np.unique(gs))                        # rows 1 and 57 coincide
    med = float(np.median(gs))
    ci = np.percentile(gs, [2.5, 97.5])
    A = np.loadtxt("data/clark2021/interaction_matrix_RLC8_posteriormean.csv",
                   delimiter=",")
    plug = gamma_offdiag(A)
    plug_mod, nmod, ntot = gamma_modeled(A)
    return dict(median=med, ci=ci, plug=plug, plug_mod=plug_mod,
                nmod=nmod, ntot=ntot, n=len(gs), ndistinct=ndistinct)


def main():
    gv, reps, boot_ci = venturelli()
    c = clark()

    print(f"Venturelli 2018 (12 sp, human gut): plug-in gamma = {gv:+.3f}")
    if len(reps):
        sign = "all strictly positive" if np.all(reps > 0) else "mixed sign"
        print(f"    replicate fits (informative): {np.round(reps, 3)}  "
              f"range [{reps.min():+.3f},{reps.max():+.3f}]  ({sign})")
    else:
        print("    replicate fits (informative): none found "
              "(no informative SBML fit); plotted bar falls back to bootstrap CI")
    print(f"    single-matrix bootstrap CI (secondary): "
          f"[{boot_ci[0]:+.3f},{boot_ci[1]:+.3f}]")
    print(f"Clark 2021      (25 sp, human gut): posterior median gamma = {c['median']:+.3f}  "
          f"95% CI [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]  "
          f"(n={c['n']} draws, {c['ndistinct']} distinct)")
    print(f"    plug-in posterior-mean matrix (all pairs): {c['plug']:+.3f}")
    print(f"    plug-in restricted to modeled pairs "
          f"({c['nmod']}/{c['ntot']} nonzero): {c['plug_mod']:+.3f}")

    # plotted central value + error bar for each community
    if len(reps):                            # Venturelli: replicate spread,
        v_ci = (min(reps.min(), gv), max(reps.max(), gv))  # widened to contain gv
    else:                                    # no informative fit: fall back to bootstrap CI
        v_ci = (min(boot_ci[0], gv), max(boot_ci[1], gv))
    rows = [("Clark 2021\n(25 sp, human gut)", c["median"], c["ci"]),
            ("Venturelli 2018\n(12 sp, human gut)", gv, v_ci)]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.axvspan(-0.3, 0.3, color="#2c6fbb", alpha=.07)
    ax.axvline(0, color="0.6", lw=.9)
    ax.axvline(1, color="#c0392b", lw=1.4, ls="--")
    ax.text(1, 1.55, "symmetric\nglass ($\\gamma=1$)", color="#c0392b",
            ha="center", va="bottom", fontsize=8)
    ax.text(0, 1.55, "weakly reciprocal", color="#2c6fbb", ha="center",
            va="bottom", fontsize=8)
    for k, (lab, g, ci) in enumerate(rows):
        ax.errorbar(g, k, xerr=[[g - ci[0]], [ci[1] - g]], fmt="o", ms=9,
                    color="#2c6fbb", ecolor="#2c6fbb", capsize=4, lw=1.6)
        ax.annotate(f"{g:+.2f}", (g, k), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylim(-0.6, 1.5)
    ax.set_xlim(-0.55, 1.15)
    ax.set_xlabel(r"interaction symmetry $\gamma=\mathrm{corr}(a_{ij},a_{ji})$"
                  "   (bottom-up designed-community estimates)")
    fig.tight_layout()
    out = f"{OUT}/fig25_measured_gamma.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
