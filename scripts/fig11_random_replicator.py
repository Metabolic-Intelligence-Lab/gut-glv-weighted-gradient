"""
Fig. 11 -- The random-replicator model across symmetry, with the empirical band.

The disordered GLV with tunable coupling correlation gamma IS the random
replicator (Opper-Diederich; Galla); the symmetric spin glass is its gamma=1
special case. Here, at fixed strong disorder, we sweep gamma from antisymmetric
(-1) to symmetric (+1) and measure the two observables that matter for the
microbiome story:

  - multistability   M(gamma): fraction of disorder realisations that admit
                     alternative stable states (initial-condition-dependent
                     equilibria)  -> substrate of hysteresis
  - persistence      P(gamma): late-time fluctuation amplitude
                     -> non-equilibrium (chaotic / cyclic) dynamics

The grey band marks the empirically inferred range gamma ~ 0..0.3 (Fig. 10):
the real microbiome sits at the low-gamma edge, where multistability is already
present while the energy-landscape (gamma=1) idealisation does not hold.

Run:  python3 fig11_random_replicator.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from asymmetric_and_phases import dense_couplings, run_traj, late_fluctuation

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"


def observables(sigma, gamma, S=60, n_real=6, n_ic=4):
    mult, pers = [], []
    for rep in range(n_real):
        J = dense_couplings(S, mu=2.0, sigma=sigma, gamma=gamma,
                            rng=np.random.default_rng(100 + rep))
        r = np.ones(S)
        finals, persistent = [], False
        rng = np.random.default_rng(200 + rep)
        for _ in range(n_ic):
            _, y = run_traj(rng.uniform(0.05, 1.0, S), r, J, 500)
            pers.append(late_fluctuation(y))
            if late_fluctuation(y) > 0.05:
                persistent = True
            f = y[:, -1]
            finals.append(f / max(np.linalg.norm(f), 1e-9))
        if not persistent and len(finals) > 1:
            qmin = min(finals[a] @ finals[b]
                       for a in range(len(finals)) for b in range(a + 1, len(finals)))
            mult.append(1.0 if qmin < 0.95 else 0.0)
        else:
            mult.append(1.0 if persistent else 0.0)   # persistent counts as non-unique
    return float(np.mean(mult)), float(np.mean(pers))


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    sigma = 2.5
    gammas = np.linspace(-1.0, 1.0, 13)
    M, P = [], []
    for g in gammas:
        m, p = observables(sigma, g)
        M.append(m); P.append(p)
        print(f"  gamma={g:+.2f}  multistability={m:.2f}  persistence={p:.3f}")
    M, P = np.array(M), np.array(P)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axvspan(0.0, 0.31, color="#d95f02", alpha=0.12)
    ax.text(0.155, 0.93, "empirical\nrange\n(Fig. 10)", ha="center", fontsize=8, color="#d95f02")
    ax.plot(gammas, M, "-o", color="#7570b3", lw=2, label="multistability (alternative stable states)")
    ax.plot(gammas, P / max(P.max(), 1e-9), "-s", color="#1b9e77", lw=2,
            label="persistence (non-equilibrium dynamics, normalised)")
    ax.axvline(1.0, ls=":", color="grey")
    ax.text(1.0, -0.07, "spin glass\n(gamma=1)", ha="center", fontsize=8, color="grey")
    ax.set_xlabel("interaction symmetry  gamma")
    ax.set_ylabel("propensity (fraction / normalised amplitude)")
    ax.set_title("The random replicator across symmetry (sigma fixed, strong disorder):\n"
                 "the empirical low-gamma microbiome already shows multistability -> hysteresis substrate")
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax.set_ylim(-0.12, 1.12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig11_random_replicator.png", dpi=150)
    plt.close(fig)
    print("[fig11] random replicator written")


if __name__ == "__main__":
    main()
