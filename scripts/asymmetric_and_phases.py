"""
Asymmetric interactions: the glass -> chaos boundary, and the phase diagram
===========================================================================

Extends the symmetric spin-glass picture (spin_glass_microbiome.py) to the
biologically realistic case J_ij != J_ji. Following the disordered-GLV
literature (Bunin 2017; Roy et al. 2019; Galla 2018):

  - the symmetry/correlation of the coupling matrix is tuned by gamma in [-1,1]
        gamma = +1  fully symmetric   -> Lyapunov function exists -> fixed points
        gamma =  0  uncorrelated
        gamma = -1  antisymmetric (predator-prey) -> persistent dynamics
  - as disorder sigma grows, the system leaves the unique-fixed-point phase;
    with enough asymmetry it enters a PERSISTENT (limit-cycle / chaotic) phase
    where NO equilibrium is reached. There the energy-landscape language breaks
    down -- the honest caveat of the theory, made quantitative.

Outputs (./figures):
  fig5_glass_to_chaos.png   late-time fluctuation amplitude vs sigma, per gamma
  fig6_phase_diagram.png    (sigma, gamma) plane classified into 3 phases

Run:  python3 asymmetric_and_phases.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from spin_glass_microbiome import scale_free_graph, NMAX, EPS

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"


# ---------------------------------------------------------------------------
# Couplings with tunable asymmetry
# ---------------------------------------------------------------------------
def dense_couplings(S, mu, sigma, gamma, rng):
    """Fully-connected random couplings (mean-field disordered-GLV; Bunin 2017,
    Galla 2018) with prescribed correlation gamma between J_ij and J_ji.

        J_ij = -mu/S + (sigma/sqrt(S)) * z1
        J_ji = -mu/S + (sigma/sqrt(S)) * (gamma*z1 + sqrt(1-gamma^2)*z2)

    gamma=+1 symmetric (Lyapunov function -> fixed points),
    gamma=-1 antisymmetric (predator-prey -> persistent dynamics).
    The 1/sqrt(S) scaling is the standard thermodynamic-limit normalisation that
    makes the phase boundaries sigma-controlled and self-averaging.
    """
    z1 = rng.normal(size=(S, S))
    z2 = rng.normal(size=(S, S))
    Jij = -mu / S + (sigma / np.sqrt(S)) * z1
    Jji = -mu / S + (sigma / np.sqrt(S)) * (gamma * z1 + np.sqrt(max(1 - gamma**2, 0.0)) * z2)
    J = np.zeros((S, S))
    iu = np.triu_indices(S, 1)
    J[iu] = Jij[iu]
    J[(iu[1], iu[0])] = Jji[iu]
    np.fill_diagonal(J, 0.0)
    return J


def rhs(t, n, r, J, eps=EPS):
    n = np.clip(n, 0.0, NMAX)
    return n * (r - n + J @ n) + eps


def run_traj(n0, r, J, t_end, n_eval=400):
    teval = np.linspace(0, t_end, n_eval)
    sol = solve_ivp(rhs, (0, t_end), np.clip(np.nan_to_num(n0), 0, NMAX),
                    args=(r, J), t_eval=teval, method="LSODA",
                    rtol=1e-6, atol=1e-8, max_step=2.0)
    return sol.t, np.clip(np.nan_to_num(sol.y), 0.0, NMAX)


def late_fluctuation(y, frac=0.3):
    """Mean (over species) temporal coefficient of variation in the last `frac`
    of the trajectory. ~0 => fixed point reached; >0 => persistent dynamics."""
    w = int(y.shape[1] * (1 - frac))
    tail = y[:, w:]
    mu = tail.mean(1)
    sd = tail.std(1)
    keep = mu > 1e-3
    if keep.sum() == 0:
        return 0.0
    return float(np.mean(sd[keep] / mu[keep]))


# ---------------------------------------------------------------------------
# Figure 5 -- glass -> chaos as disorder grows, for several asymmetries
# ---------------------------------------------------------------------------
def _mean_fluct(S, r, mu, sigma, gamma, n_real=6, t_end=800):
    vals = []
    for rep in range(n_real):
        J = dense_couplings(S, mu=mu, sigma=sigma, gamma=gamma,
                            rng=np.random.default_rng(10 + rep))
        _, y = run_traj(np.random.default_rng(2 + rep).uniform(0.1, 1, S),
                        r, J, t_end)
        vals.append(late_fluctuation(y))
    return float(np.mean(vals))


def fig_glass_to_chaos():
    S = 150
    r = np.ones(S)
    mu = 2.0

    # Panel A: fluctuation vs disorder, symmetric vs uncorrelated couplings
    sigmas = np.linspace(0.3, 4.0, 16)
    fA_sym = [_mean_fluct(S, r, mu, s, +1.0) for s in sigmas]
    fA_unc = [_mean_fluct(S, r, mu, s, 0.0) for s in sigmas]

    # Panel B: fluctuation vs symmetry at fixed strong disorder
    gammas = np.linspace(1.0, -1.0, 11)
    fB = [_mean_fluct(S, r, mu, 3.2, g) for g in gammas]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(sigmas, fA_sym, "-o", ms=4, color="#1b9e77", label="gamma=+1 (symmetric)")
    ax[0].plot(sigmas, fA_unc, "-o", ms=4, color="#d95f02", label="gamma=0 (uncorrelated)")
    ax[0].axhline(0.02, ls="--", color="grey", lw=1)
    ax[0].set_xlabel("disorder strength  sigma")
    ax[0].set_ylabel("late-time fluctuation (mean temporal CV)")
    ax[0].set_title("A. Symmetric couplings stay at fixed points;\nbreaking reciprocity opens persistent dynamics")
    ax[0].legend(frameon=False, fontsize=9)

    ax[1].plot(gammas, fB, "-o", ms=4, color="#7570b3")
    ax[1].axhline(0.02, ls="--", color="grey", lw=1)
    ax[1].axvline(1.0, ls=":", color="#1b9e77")
    ax[1].set_xlabel("interaction symmetry  gamma  (+1 sym  ->  -1 antisym)")
    ax[1].set_ylabel("late-time fluctuation (mean temporal CV)")
    ax[1].set_title("B. At strong disorder the energy-landscape (glass)\nlimit gamma=+1 is the special, always-relaxing case")
    ax[1].invert_xaxis()

    fig.suptitle("Glass -> chaos: the spin-glass description requires (near-)symmetric interactions",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_glass_to_chaos.png", dpi=150)
    plt.close(fig)
    print(f"[fig5] glass->chaos written (sym max CV={max(fA_sym):.3f}, unc max CV={max(fA_unc):.3f})")


# ---------------------------------------------------------------------------
# Figure 6 -- phase diagram in the (sigma, gamma) plane
# ---------------------------------------------------------------------------
def classify(S, r, sigma, gamma, n_ic=4, seed=0, mu=2.0):
    """Return 0=unique fixed point, 1=multiple attractors (glassy),
    2=persistent (chaotic/limit cycle)."""
    rng = np.random.default_rng(seed)
    J = dense_couplings(S, mu=mu, sigma=sigma, gamma=gamma, rng=rng)
    finals, persistent = [], False
    for _ in range(n_ic):
        _, y = run_traj(rng.uniform(0.05, 1.0, S), r, J, 600)
        if late_fluctuation(y) > 0.05:
            persistent = True
            break
        f = y[:, -1]
        finals.append(f / max(np.linalg.norm(f), 1e-9))
    if persistent:
        return 2
    # compare basins reached from different initial conditions
    qs = [finals[a] @ finals[b]
          for a in range(len(finals)) for b in range(a + 1, len(finals))]
    return 0 if (len(qs) and min(qs) > 0.98) else 1


def fig_phase_diagram(n_sigma=40, n_gamma=34):
    S = 120
    r = np.ones(S)
    sigmas = np.linspace(0.3, 3.5, n_sigma)
    gammas = np.linspace(-1.0, 1.0, n_gamma)
    grid = np.zeros((len(gammas), len(sigmas)), dtype=int)
    for i, g in enumerate(gammas):
        for j, s in enumerate(sigmas):
            grid[i, j] = classify(S, r, s, g, seed=100 + j)
        if i % 5 == 0:
            print(f"  phase row {i+1}/{n_gamma} (gamma={g:+.2f}) done")

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap(["#1b9e77", "#7570b3", "#d95f02"])
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    # pcolormesh on cell edges -> crisp, correctly registered cells
    ds = (sigmas[1] - sigmas[0]) / 2; dg = (gammas[1] - gammas[0]) / 2
    se = np.linspace(sigmas[0] - ds, sigmas[-1] + ds, len(sigmas) + 1)
    ge = np.linspace(gammas[0] - dg, gammas[-1] + dg, len(gammas) + 1)
    ax.pcolormesh(se, ge, grid, cmap=cmap, vmin=0, vmax=2, shading="flat")
    ax.set_xlabel(r"disorder strength  $\sigma$")
    ax.set_ylabel(r"interaction symmetry  $\gamma$")
    ax.set_title("Phase diagram of the disordered microbiome")
    ax.legend(handles=[Patch(fc="#1b9e77", label="unique fixed point (resilient)"),
                       Patch(fc="#7570b3", label="multiple attractors (glassy / hysteretic)"),
                       Patch(fc="#d95f02", label="persistent (chaotic / cycles)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=1,
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_phase_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    counts = np.bincount(grid.ravel(), minlength=3)
    print(f"[fig6] phase diagram written ({n_sigma}x{n_gamma}; "
          f"UFP={counts[0]}, glassy={counts[1]}, chaos={counts[2]})")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_glass_to_chaos()
    fig_phase_diagram()
    print("\nDone.")
