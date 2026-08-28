"""
Spin-glass model of gut-microbiota resistance, resilience and hysteresis
========================================================================

Toy model that reproduces, from a single disordered Generalized Lotka-Volterra
(GLV) dynamics on a scale-free interaction network, the three response regimes
described in:

    Tap, Ruppe, Derrien (2022) "The Human Gut Microbiota in all its States:
    From Disturbance to Resilience" (Comprehensive Gut Microbiota, ch. 2.14),
    Fig. 3 -- resistance / resilience / hysteresis.

The three theoretical layers:
  (A) Topology      -> Barabasi-Albert scale-free graph (hubs = keystone species)
  (B) Disorder      -> symmetric, mixed-sign couplings J_ij (frustration)
  (C) Dynamics      -> GLV + environmental field h(t) + demographic noise (T)

Outputs (saved in ./figures):
  fig1_three_regimes.png   reproduction of book Fig. 3 (state vs time)
  fig2_hysteresis_loop.png order parameter vs swept field -> catastrophic bifurcation
  fig3_overlap_Pq.png      replica-overlap distribution P(q): resilient vs glassy
  fig4_error_vs_attack.png Barabasi error- vs attack-tolerance (random vs keystone removal)

Run:  python3 spin_glass_microbiome.py
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

RNG = np.random.default_rng(7)
OUT = "figures"


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------
def scale_free_graph(S, m=2, seed=7):
    """Barabasi-Albert scale-free topology; returns adjacency + degree vector."""
    G = nx.barabasi_albert_graph(S, m, seed=seed)
    A = nx.to_numpy_array(G)
    k = A.sum(1)
    return A, k, G


def enterotype_couplings(A, guild, w_in, w_out, sigma, rng):
    """Two competing guilds on top of the network topology (enterotype model).

    Within-guild interactions are mutualistic (+w_in), cross-guild interactions
    are competitive (-w_out); quenched disorder Normal(0,sigma) is added on each
    edge. This is the canonical bistable construction: it produces two alternative
    stable states separated by a barrier -- the Bacteroides/Prevotella
    'barrier between states' of Tap et al. (2022).
    """
    S = A.shape[0]
    kmean = max(A.sum(1).mean(), 1.0)
    same = (guild[:, None] == guild[None, :])
    base = np.where(same, w_in, -w_out)
    U = base + rng.normal(0.0, sigma, size=(S, S))
    U = np.triu(U, 1)
    J = U + U.T
    J *= A
    J /= np.sqrt(kmean)
    np.fill_diagonal(J, 0.0)
    return J


def coupling_matrix(A, mu, sigma, rng):
    """Symmetric mixed-sign couplings on the edges of A.

    J_ij = J_ji ~ Normal(mean=-mu, std=sigma) on existing edges, 0 otherwise.
    Negative mean  -> net competition (drives multistability).
    Symmetry       -> a Lyapunov function exists (spin-glass energy landscape).
    Couplings are scaled by 1/sqrt(<k>) so the interaction term stays O(1)
    against the self-limitation -n_i and the attractor remains bounded
    (standard disordered-GLV scaling, cf. Bunin 2017).
    """
    S = A.shape[0]
    kmean = max(A.sum(1).mean(), 1.0)
    U = rng.normal(-mu, sigma, size=(S, S))
    U = np.triu(U, 1)
    J = U + U.T            # symmetric
    J *= A                 # keep only network edges
    J /= np.sqrt(kmean)    # disorder scaling -> bounded dynamics
    np.fill_diagonal(J, 0.0)
    return J


NMAX = 50.0    # hard abundance cap -> guarantees a bounded phase space
EPS = 1e-3     # immigration / reservoir influx -> allows recolonisation
               # (without it n=0 is absorbing and 'recovery' is impossible)


def glv_rhs(t, n, r, J, h_func, eps=EPS):
    """Generalized Lotka-Volterra with self-limitation, external field h(t),
    and a small immigration floor eps (dispersal from a reservoir)."""
    n = np.clip(n, 0.0, NMAX)
    h = h_func(t)
    return n * (r + h - n + J @ n) + eps


def integrate(n0, r, J, h_func, t_end, t_eval=None, eps=EPS):
    n0 = np.clip(np.nan_to_num(n0), 0.0, NMAX)
    sol = solve_ivp(glv_rhs, (0, t_end), n0, args=(r, J, h_func, eps),
                    t_eval=t_eval, method="LSODA", rtol=1e-6, atol=1e-8,
                    max_step=2.0)
    y = np.clip(np.nan_to_num(sol.y), 0.0, NMAX)
    return sol.t, y


def relax(n0, r, J, t_end=400):
    """Relax to a fixed point under zero field; return final abundance vector."""
    _, y = integrate(n0, r, J, lambda t: 0.0, t_end,
                     t_eval=np.linspace(0, t_end, 5))
    return y[:, -1]


# ---------------------------------------------------------------------------
# Figure 1 -- three response regimes (reproduce book Fig. 3)
# ---------------------------------------------------------------------------
def fig_three_regimes():
    """One bistable community, three disturbance doses.

    Faithful to Tap et al. (2022): 'when the disturbance level goes beyond the
    retention level, the ecosystem was unable to return to the previous stable
    state'. The SAME system shows resistance / resilience / hysteresis depending
    only on the dose of an antibiotic-like perturbation that depresses the
    incumbent guild A.
    """
    S = 120
    A, k, _ = scale_free_graph(S, m=2, seed=3)
    guild = (np.arange(S) % 2).astype(int)    # two enterotype guilds A / B
    r = np.random.default_rng(1).uniform(0.9, 1.1, S)
    J = enterotype_couplings(A, guild, w_in=0.4, w_out=1.2, sigma=0.12,
                             rng=np.random.default_rng(11))
    n_base = relax(np.where(guild == 0, 1.0, 0.1), r, J)   # baseline = enterotype A

    doses = {                                  # (antibiotic intensity, label)
        "Resistant  (low dose)":      0.30,
        "Resilient  (sub-threshold)": 0.70,
        "Hysteretic (over-threshold)": 0.90,
    }
    t_pulse = (60, 100)
    t_end = 400
    teval = np.linspace(0, t_end, 1600)

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    colors = ["#1b9e77", "#7570b3", "#d95f02"]

    for (name, P), c in zip(doses.items(), colors):
        def h_func(t, P=P):
            field = np.zeros(S)
            if t_pulse[0] <= t <= t_pulse[1]:
                field[guild == 0] = -P        # antibiotic depresses incumbent guild
            return field

        t, y = integrate(n_base.copy(), r, J, h_func, t_end, teval)
        num = np.abs(y - n_base[:, None]).sum(0)
        den = (y.sum(0) + n_base.sum())
        bc = num / np.where(den == 0, 1, den)
        ax.plot(t, bc, lw=2.3, color=c, label=name)

    ax.axvspan(*t_pulse, color="grey", alpha=0.15)
    ax.text(80, ax.get_ylim()[1]*0.02, "disturbance", ha="center", fontsize=8, color="grey")
    ax.set_xlabel("time")
    ax.set_ylabel("microbiome state\n(Bray-Curtis distance from baseline)")
    ax.set_title("Resistance / Resilience / Hysteresis from ONE bistable community\n"
                 "(dose crosses the retention threshold; cf. Tap et al. 2022, Fig. 3)")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_three_regimes.png", dpi=150)
    plt.close(fig)
    print("[fig1] three regimes written")


# ---------------------------------------------------------------------------
# Figure 2 -- hysteresis loop under a swept environmental field
# ---------------------------------------------------------------------------
def fig_hysteresis_loop():
    S = 120
    A, k, _ = scale_free_graph(S, m=2, seed=3)
    guild = (np.arange(S) % 2).astype(int)
    r = RNG.uniform(0.9, 1.1, S)
    rng = np.random.default_rng(11)
    # bistable enterotype regime at the edge of bistability -> the field can
    # cross the barrier, at different thresholds up vs down (genuine fold loop)
    J = enterotype_couplings(A, guild, w_in=0.4, w_out=1.2, sigma=0.15, rng=rng)

    # environmental control parameter tilts the balance between the two guilds
    # (e.g. fibre-rich vs protein/fat-rich diet, or an antibiotic favouring one)
    fields = np.concatenate([np.linspace(-1.5, 1.5, 60),
                             np.linspace(1.5, -1.5, 60)])
    n = relax(RNG.uniform(0.1, 1.0, S), r, J)
    order = []
    for h in fields:
        def hf(t, H=h):
            f = np.zeros(S); f[guild == 1] = H; f[guild == 0] = -H; return f
        _, y = integrate(n.copy(), r, J, hf, 120,
                         t_eval=np.linspace(0, 120, 4))
        n = y[:, -1]                          # adiabatic continuation
        # order parameter: abundance fraction held by guild B
        order.append(n[guild == 1].sum() / max(n.sum(), 1e-9))
    order = np.array(order)
    up, dn = order[:60], order[60:]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(fields[:60], up, "-o", ms=3, color="#d95f02", label="increasing field")
    ax.plot(fields[60:], dn, "-o", ms=3, color="#7570b3", label="decreasing field")
    ax.set_xlabel("environmental field  h  (diet / drug control parameter)")
    ax.set_ylabel("order parameter\n(guild-B / enterotype-B abundance fraction)")
    ax.set_title("Hysteresis loop & catastrophic bifurcation\n(open loop = irreversibility = trapped in alternative stable state)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_hysteresis_loop.png", dpi=150)
    plt.close(fig)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy>=2 removed trapz
    area = trap(up, fields[:60]) - trap(dn[::-1], fields[:60])
    print(f"[fig2] hysteresis loop written (loop area ~ {abs(area):.3f})")


# ---------------------------------------------------------------------------
# Figure 3 -- replica overlap distribution P(q): resilient vs glassy
# ---------------------------------------------------------------------------
def overlap_distribution(mu, sigma, S=100, n_rep=60, seed=0):
    A, _, _ = scale_free_graph(S, m=2, seed=seed)
    r = np.random.default_rng(seed).uniform(0.9, 1.1, S)
    J = coupling_matrix(A, mu, sigma, np.random.default_rng(seed + 1))
    rng = np.random.default_rng(seed + 2)
    states = []
    for _ in range(n_rep):                    # many random initial conditions = replicas
        nf = relax(rng.uniform(0.05, 1.0, S), r, J, t_end=300)
        s = nf / max(np.linalg.norm(nf), 1e-9)
        states.append(s)
    states = np.array(states)
    qs = []
    for a in range(n_rep):
        for b in range(a + 1, n_rep):
            qs.append(float(states[a] @ states[b]))
    return np.array(qs)


def fig_overlap():
    q_res = overlap_distribution(mu=0.10, sigma=0.25, seed=5)   # resilient
    q_gla = overlap_distribution(mu=0.30, sigma=1.10, seed=5)   # glassy

    bins = np.linspace(0.55, 1.02, 40)        # common range so both are comparable
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, sharex=True)
    ax[0].hist(np.clip(q_res, bins[0], bins[-1]), bins=bins, color="#1b9e77", alpha=0.85)
    ax[0].set_title("Resilient regime\n(single basin -> P(q) one sharp spike at q=1)")
    ax[1].hist(np.clip(q_gla, bins[0], bins[-1]), bins=bins, color="#d95f02", alpha=0.85)
    ax[1].set_title("Glassy / hysteretic regime\n(many basins -> broad, multi-peak P(q))")
    for a in ax:
        a.set_xlabel("replica overlap  q")
    ax[0].set_ylabel("count")
    fig.suptitle("Overlap distribution P(q): a measurable signature of multistability (RSB)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_overlap_Pq.png", dpi=150)
    plt.close(fig)
    print(f"[fig3] P(q) written (std resilient={q_res.std():.3f}, glassy={q_gla.std():.3f})")


# ---------------------------------------------------------------------------
# Figure 4 -- Barabasi error vs attack tolerance (random vs keystone removal)
# ---------------------------------------------------------------------------
def fig_error_attack():
    S = 150
    A, k, _ = scale_free_graph(S, m=2, seed=9)
    r = RNG.uniform(0.9, 1.1, S)
    J = coupling_matrix(A, mu=0.20, sigma=0.7, rng=np.random.default_rng(20))
    n_base = relax(RNG.uniform(0.1, 1.0, S), r, J)

    def state_shift_after_removal(order_idx, frac):
        n0 = n_base.copy()
        n_remove = int(frac * S)
        killed = order_idx[:n_remove]
        n0[killed] = 0.0
        Jk = J.copy()
        Jk[killed, :] = 0.0
        Jk[:, killed] = 0.0
        nf = relax(n0, r, Jk)
        num = np.abs(nf - n_base).sum()
        den = nf.sum() + n_base.sum()
        return num / max(den, 1e-9)

    fracs = np.linspace(0.0, 0.30, 13)
    rng = np.random.default_rng(99)
    rand_order = rng.permutation(S)            # random failure
    attack_order = np.argsort(k)[::-1]         # targeted: highest-degree first (keystones)

    shift_rand = [state_shift_after_removal(rand_order, f) for f in fracs]
    shift_att = [state_shift_after_removal(attack_order, f) for f in fracs]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(fracs * 100, shift_rand, "-o", color="#1b9e77",
            label="random removal (error tolerance)")
    ax.plot(fracs * 100, shift_att, "-s", color="#d95f02",
            label="keystone/hub removal (targeted attack)")
    ax.set_xlabel("% species removed")
    ax.set_ylabel("community state shift\n(Bray-Curtis from baseline)")
    ax.set_title("Error vs attack tolerance (Albert-Jeong-Barabasi)\nkeystone loss tips the community; random loss is absorbed")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_error_vs_attack.png", dpi=150)
    plt.close(fig)
    print(f"[fig4] error-vs-attack written "
          f"(shift@20%: random={shift_rand[8]:.3f}, attack={shift_att[8]:.3f})")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_three_regimes()
    fig_hysteresis_loop()
    fig_overlap()
    fig_error_attack()
    print("\nAll figures written to ./figures/")
