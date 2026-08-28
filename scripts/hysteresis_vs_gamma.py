"""
Fig. 21 -- Where does the hysteresis come from? Structured niche bistability vs
the disorder-induced glassy phase (resolving the phase-placement tension)
============================================================================

Referee tension (critique point 4): the inferred community sits at large sigma
and gamma~0, formally the *chaotic / non-potential* corner of the random-GLV
phase diagram, where the disorder-induced multiple-attractor (glassy) phase does
NOT exist. Yet the clinic shows two-state hysteresis. If hysteresis required the
glassy phase, the picture would be inconsistent.

Resolution tested here: the two-state (enterotype) hysteresis is driven by
MACROSCOPIC niche structure -- two guilds in mutual competition, a low-rank,
mean-field bistability -- NOT by the microscopic disorder. Competition between
guilds is symmetric by nature (mutual), but the idiosyncratic species-level
disorder need not be. We therefore split

    J = J_structure(guild means, symmetric, low-rank)  +  J_disorder(gamma_d)

and sweep the symmetry gamma_d of the DISORDER from 1 (spin-glass) to 0 (fully
non-reciprocal). If the hysteresis loop area is ~invariant, the loop lives in the
structured sector and survives at gamma~0: the observed hysteresis is compatible
with a non-potential microscopic regime, and does not require the glassy phase.

Run:  python3 hysteresis_vs_gamma.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spin_glass_microbiome import scale_free_graph, integrate, relax, RNG

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"


def enterotype_couplings_gamma(A, guild, w_in, w_out, sigma, gamma_d, rng):
    """Guild structure (symmetric, low-rank) + disorder with tunable symmetry.

    structure : same-guild = +w_in (mutualistic), cross-guild = -w_out
                (competitive). Depends only on guild identity -> symmetric.
    disorder  : per-edge bivariate normal with corr(D_ij, D_ji) = gamma_d.
    """
    S = A.shape[0]
    kmean = max(A.sum(1).mean(), 1.0)
    same = (guild[:, None] == guild[None, :])
    base = np.where(same, w_in, -w_out)           # symmetric structure

    a = rng.normal(size=(S, S)); b = rng.normal(size=(S, S))
    iu = np.triu_indices(S, 1)
    D = np.zeros((S, S))
    D[iu] = sigma * a[iu]
    D.T[iu] = sigma * (gamma_d * a[iu] + np.sqrt(max(1 - gamma_d**2, 0)) * b[iu])

    J = (base + D) * A
    J /= np.sqrt(kmean)
    np.fill_diagonal(J, 0.0)
    return J


def loop_area(J, guild, r, S):
    fields = np.concatenate([np.linspace(-1.5, 1.5, 60),
                             np.linspace(1.5, -1.5, 60)])
    n = relax(RNG.uniform(0.1, 1.0, S), r, J)
    order = []
    for h in fields:
        def hf(t, H=h):
            f = np.zeros(S); f[guild == 1] = H; f[guild == 0] = -H; return f
        _, y = integrate(n.copy(), r, J, hf, 120, t_eval=np.linspace(0, 120, 4))
        n = y[:, -1]
        order.append(n[guild == 1].sum() / max(n.sum(), 1e-9))
    order = np.array(order)
    up, dn = order[:60], order[60:]
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy>=2 removed trapz
    area = abs(trap(up, fields[:60]) - trap(dn[::-1], fields[:60]))
    return fields, up, dn, area


def measured_gamma(J):
    S = J.shape[0]
    iu = np.triu_indices(S, 1)
    m = (J != 0)[iu] & (J.T != 0)[iu]
    if m.sum() < 5:
        return np.nan
    return float(np.corrcoef(J[iu][m], J.T[iu][m])[0, 1])


def loop_seed_avg(A, guild, sigma, gamma_d, n_seed=8, seed0=20):
    """Loop area averaged over n_seed independent disorder realizations
    (each with its own quenched r), so the result is not a single-seed artefact."""
    S = A.shape[0]
    areas = []
    for sd in range(seed0, seed0 + n_seed):
        rng = np.random.default_rng(sd)
        J = enterotype_couplings_gamma(A, guild, 0.4, 1.2, sigma, gamma_d, rng)
        r = np.random.default_rng(1000 + sd).uniform(0.9, 1.1, S)
        _, _, _, area = loop_area(J, guild, r, S)
        areas.append(area)
    return np.array(areas)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    S = 120
    A, k, _ = scale_free_graph(S, m=2, seed=3)
    guild = (np.arange(S) % 2).astype(int)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # (left) representative loops at weak disorder (robust across seeds): the loop
    # is present at every symmetry gamma_d, showing it is structural.
    sig_rep = 0.15
    cols = {1.0: "#1b9e77", 0.5: "#7570b3", 0.0: "#d95f02"}
    for gd in [1.0, 0.5, 0.0]:
        rng = np.random.default_rng(23)
        J = enterotype_couplings_gamma(A, guild, 0.4, 1.2, sig_rep, gd, rng)
        r = np.random.default_rng(1023).uniform(0.9, 1.1, S)
        fields, up, dn, area = loop_area(J, guild, r, S)
        axL.plot(fields[:60], up, "-", lw=1.6, color=cols[gd],
                 label=fr"$\gamma_d={gd:.1f}$ (full $\gamma={measured_gamma(J):.2f}$)")
        axL.plot(fields[60:], dn, "--", lw=1.6, color=cols[gd])
    axL.set_xlabel("environmental field $h$")
    axL.set_ylabel("order parameter (guild-B fraction)")
    axL.set_title(f"Representative loops at weak disorder ($\\sigma={sig_rep}$):\n"
                  "an open loop at every symmetry $\\gamma_d$", fontsize=9.5)
    axL.legend(frameon=False, fontsize=8, loc="upper left")

    # (right) seed-averaged loop area vs gamma_d, for several disorder strengths:
    # the loop is robust (present) at every gamma_d; only the symmetric strong-
    # disorder corner (gamma_d->1, large sigma) collapses/fragments.
    gds = np.array([0.0, 0.3, 0.6, 1.0])
    print(f"{'sigma':>6} {'gamma_d':>8} | {'loop area mean [min,max] over seeds':>36}")
    print("-" * 56)
    for sig, mk in [(0.15, "o"), (0.30, "s"), (0.50, "^")]:
        means, lo, hi = [], [], []
        for gd in gds:
            a = loop_seed_avg(A, guild, sig, gd)
            means.append(a.mean()); lo.append(a.min()); hi.append(a.max())
            print(f"{sig:>6.2f} {gd:>8.1f} | {a.mean():>8.3f}  [{a.min():.3f}, {a.max():.3f}]")
        means, lo, hi = np.array(means), np.array(lo), np.array(hi)
        axR.errorbar(gds, means, yerr=[means - lo, hi - means], fmt=mk + "-",
                     capsize=4, ms=7, label=fr"$\sigma={sig}$")
    axR.set_xlabel(r"disorder symmetry  $\gamma_d$")
    axR.set_ylabel("loop area (mean $\\pm$ range, 8 seeds)")
    axR.set_title("Loop is robust at every $\\gamma_d$;\nonly the symmetric strong-disorder "
                  "corner collapses", fontsize=9.5)
    axR.legend(frameon=False, fontsize=8)
    axR.axhline(0, ls=":", color="grey", lw=0.8)

    fig.suptitle("Two-state hysteresis is a structural competitive fold, robust to interaction "
                 "symmetry:\nan open loop at every $\\gamma_d$ (left); seed-averaged, only the "
                 "symmetric strong-disorder corner fragments it (right)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f"{OUT}/fig21_hysteresis_vs_gamma.png", dpi=150)
    plt.close(fig)
    print("[fig21] written")


if __name__ == "__main__":
    main()
