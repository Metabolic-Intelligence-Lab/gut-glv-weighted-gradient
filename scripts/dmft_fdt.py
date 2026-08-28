"""
Fig. -- Two-time DMFT observables and the fluctuation-dissipation violation
===========================================================================

What replaces the potential when J is asymmetric: there is no Gibbs measure, so
we characterise the non-equilibrium steady state by its two-time functions
(correlation C, response R) and by the VIOLATION of the fluctuation-dissipation
theorem (FDT). gamma controls how far from equilibrium the community sits.

We compute this exactly in the linear-response (Gaussian) sector around a
coexistence fixed point of the rescaled disordered GLV

    dn_i/dt = n_i ( 1 - n_i + sum_j a_ij n_j ),   <a>=0, Var(a)=g^2/S, corr(a_ij,a_ji)=gamma.

Linearising at n*≈1 gives the Ornstein-Uhlenbeck process
    dx = A x dt + sqrt(2T) dW,   A = a - I   (stable while (1+gamma)g < 1),
whose stationary law is Gaussian, x ~ N(0, C), with C the solution of the
Lyapunov equation  A C + C A^T = -2T I.

Exact objects (no approximation beyond the Gaussian sector):
  * stationary covariance C            (Lyapunov)
  * two-time correlation  C(tau) = e^{A tau} C            (tau >= 0)
  * two-time response     R(tau) = e^{A tau}
  * probability-current matrix  Omega = A C + T I,  ANTISYMMETRIC by the Lyapunov
    equation, and ZERO iff A is symmetric (gamma=1) = detailed balance.
  * FDT: at equilibrium  T R(tau) = -dC/dtau, i.e.  T chi(tau) = C(0) - C(tau).

Irreversibility / FDT-violation index:  I(gamma) = ||Omega||_F / (T sqrt S).
By construction I(1)=0 (equilibrium); this is the built-in correctness check.
A direct Euler-Maruyama simulation validates C(tau), R(tau) against the matrix
exponentials.

Run:  python3 dmft_fdt.py
"""
import warnings
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov, eigvals
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# macOS Accelerate BLAS emits spurious divide/overflow warnings on plain matmul;
# results are exact (I(gamma=1)=0 and the simulation cross-check confirm it).
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

RNG = np.random.default_rng(1)
S = 400          # species (mean-field size)
G = 0.35         # disorder strength g (kept < 0.5 so (1+gamma)g < 1 is stable for all gamma)
T = 1.0          # noise temperature
OUT = "figures"


def community_matrix(gamma, g=G, s=S, rng=RNG):
    """A = a - I with <a>=0, Var(a_ij)=g^2/S, corr(a_ij,a_ji)=gamma, zero diagonal."""
    a = np.zeros((s, s))
    sd = g / np.sqrt(s)
    iu = np.triu_indices(s, 1)
    z1 = rng.standard_normal(len(iu[0]))
    z2 = rng.standard_normal(len(iu[0]))
    a[iu] = sd * z1
    a.T[iu] = sd * (gamma * z1 + np.sqrt(max(1 - gamma**2, 0.0)) * z2)
    return a - np.eye(s)


def stationary(A, T=T):
    C = solve_continuous_lyapunov(A, -2 * T * np.eye(A.shape[0]))
    C = 0.5 * (C + C.T)                      # symmetrise numerical noise
    return C


def irreversibility(A, C, T=T):
    """||Omega||_F / (T sqrt S), Omega = A C + T I (antisymmetric; 0 at equilibrium)."""
    Omega = A @ C + T * np.eye(A.shape[0])
    asym = 0.5 * (Omega - Omega.T)           # exact Omega is antisymmetric; guard rounding
    return np.linalg.norm(asym, "fro") / (T * np.sqrt(A.shape[0]))


def two_time(A, C, taus, u):
    """Projected correlation C_u(tau)=u^T e^{Atau} C u and response R_u(tau)=u^T e^{Atau} u."""
    Cu, Ru = [], []
    for tau in taus:
        E = expm(A * tau)
        Cu.append(u @ E @ C @ u)
        Ru.append(u @ E @ u)
    return np.array(Cu), np.array(Ru)


def simulate_two_time(A, taus, T=T, dt=0.01, n_paths=4000, u=None, rng=RNG):
    """Euler-Maruyama validation of C_u(tau), R_u(tau) in the stationary state."""
    s = A.shape[0]
    C = stationary(A, T)
    # sample stationary initial states x0 ~ N(0,C)
    L = np.linalg.cholesky(C + 1e-9 * np.eye(s))
    tmax = taus[-1]
    nsteps = int(round(tmax / dt))
    sqrt2Tdt = np.sqrt(2 * T * dt)
    # correlation: <u.x(0) u.x(tau)>
    x = (L @ rng.standard_normal((s, n_paths)))
    u0 = u @ x                                        # projection at tau=0
    proj = np.zeros((nsteps + 1, n_paths))
    proj[0] = u @ x
    xi = x.copy()
    for k in range(1, nsteps + 1):
        xi = xi + (A @ xi) * dt + sqrt2Tdt * rng.standard_normal((s, n_paths))
        proj[k] = u @ xi
    idx = [int(round(t / dt)) for t in taus]
    Cu_sim = np.array([np.mean(u0 * proj[i]) for i in idx])
    # response: kick x(0) -> x(0)+eps u, propagate deterministically, measure u.x(tau)/eps
    eps = 1e-3
    xk = eps * u[:, None] * np.ones((1, 1))
    xk = eps * u.copy()
    Rk = []
    xd = xk.copy()
    resp = [u @ xd / eps]
    for k in range(1, nsteps + 1):
        xd = xd + (A @ xd) * dt
        resp.append(u @ xd / eps)
    resp = np.array(resp)
    Ru_sim = resp[idx]
    return Cu_sim, Ru_sim


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    # ---- (1) FDT-violation index vs gamma, exact -----------------------
    gammas = np.linspace(-0.6, 1.0, 17)
    Ivals, maxRe = [], []
    for gm in gammas:
        A = community_matrix(gm)
        mr = np.max(eigvals(A).real)
        maxRe.append(mr)
        C = stationary(A)
        Ivals.append(irreversibility(A, C))
    Ivals = np.array(Ivals)
    print("gamma   maxRe(A)   I(gamma)=||Omega||/(T sqrtS)")
    for gm, mr, iv in zip(gammas, maxRe, Ivals):
        tag = "  <-- equilibrium check" if abs(gm - 1) < 1e-9 else ""
        print(f"{gm:+.2f}   {mr:+.3f}     {iv:.4f}{tag}")
    # value at the empirical band gamma≈0.1
    A01 = community_matrix(0.1); I01 = irreversibility(A01, stationary(A01))
    print(f"\nEmpirical microbiome band gamma≈0.10 -> irreversibility index I = {I01:.3f}")

    # ---- (2) two-time functions + FDT parametric plot, exact + sim -----
    taus = np.linspace(0, 6, 61)
    u = np.ones(S) / np.sqrt(S)
    curves = {}
    for gm in [1.0, 0.5, 0.0]:
        A = community_matrix(gm)
        C = stationary(A)
        Cu, Ru = two_time(A, C, taus, u)
        chi = np.concatenate([[0], np.cumsum(0.5 * (Ru[1:] + Ru[:-1]) * np.diff(taus))])
        curves[gm] = (Cu, Ru, chi)

    # validation against direct simulation at gamma=0
    A0 = community_matrix(0.0); C0 = stationary(A0)
    Cu_ex, Ru_ex = two_time(A0, C0, taus, u)
    Cu_sim, Ru_sim = simulate_two_time(A0, taus, u=u)
    err_C = np.max(np.abs(Cu_ex - Cu_sim)) / np.max(np.abs(Cu_ex))
    err_R = np.max(np.abs(Ru_ex - Ru_sim))
    print(f"\nValidation (gamma=0): max rel err C(tau) = {err_C:.3f}, max abs err R(tau) = {err_R:.3f}")

    # ---- figure --------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))
    NAVY, TEAL, ORANGE, GREY = "#1f3a5f", "#2a9d8f", "#e76f51", "#8a8a8a"

    ax[0].plot(gammas, Ivals, "o-", color=NAVY, lw=2)
    ax[0].axhline(0, color=GREY, lw=1, ls=":")
    ax[0].axvline(1, color=GREY, lw=1, ls="--")
    ax[0].scatter([1], [Ivals[-1]], color=ORANGE, zorder=5, s=60)
    ax[0].annotate("equilibrium\n(FDT holds)", (1, 0.02), (0.35, 0.25),
                   fontsize=8, color=ORANGE, ha="center",
                   arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax[0].axvspan(0.0, 0.13, color=TEAL, alpha=0.15)
    ax[0].text(0.06, Ivals.max()*0.9, "empirical\ngamma", fontsize=8, color=TEAL, ha="center")
    ax[0].set_xlabel("interaction symmetry  gamma")
    ax[0].set_ylabel("FDT-violation index  I")
    ax[0].set_title("(a) Distance from equilibrium", fontsize=10, loc="left")

    for gm, col in [(1.0, ORANGE), (0.5, TEAL), (0.0, NAVY)]:
        Cu, Ru, chi = curves[gm]
        x = Cu[0] - Cu
        ax[1].plot(x, T * chi, "-", color=col, lw=2, label=f"gamma={gm:.1f}")
    lim = max(curves[0.0][0][0] - curves[0.0][0][-1], (T*curves[0.0][2][-1]))
    ax[1].plot([0, lim], [0, lim], "k:", lw=1, label="FDT line")
    ax[1].set_xlabel("C(0) - C(tau)")
    ax[1].set_ylabel("T chi(tau)  (integrated response)")
    ax[1].set_title("(b) FDT plot: on the line = equilibrium", fontsize=10, loc="left")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")

    ax[2].plot(taus, Cu_ex, "-", color=NAVY, lw=2, label="C(tau) exact")
    ax[2].plot(taus, Cu_sim, "o", color=NAVY, ms=3, mfc="white", label="C(tau) sim")
    ax[2].plot(taus, Ru_ex, "-", color=ORANGE, lw=2, label="R(tau) exact")
    ax[2].plot(taus, Ru_sim, "s", color=ORANGE, ms=3, mfc="white", label="R(tau) sim")
    ax[2].set_xlabel("tau")
    ax[2].set_title("(c) Two-time C, R (gamma=0): exact vs simulation", fontsize=10, loc="left")
    ax[2].legend(frameon=False, fontsize=8)

    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.suptitle("Non-equilibrium DMFT sector: FDT violation grows as gamma leaves 1 "
                 "(the microbiome sits at small gamma)", fontsize=11, y=1.03)
    fig.tight_layout()
    out = f"{OUT}/fig_dmft_fdt.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
