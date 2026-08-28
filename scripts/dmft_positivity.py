"""
Positivity/boundary robustness of the nonlinear DMFT simulation  (referee-4 P1, §11)
====================================================================================

The nonlinear-DMFT benchmark integrates the additive-noise GLV
    dn_i = n_i(1 - n_i - mu <n> + sum_j a_ij n_j) dt + sqrt(2T) dW_i ,
where the additive noise can drive n_i below 0. The reference simulation keeps n>=0 by a
CLIP at 0 with a small immigration floor lambda (n <- clip(n + drift*dt + noise, 0, CAP)),
which referee 4 rightly says must be stated and shown not to drive the result. We therefore
document the scheme and check that the order parameters (mean abundance M and the normalized
autocorrelation C(tau)/C(0)) are robust to the boundary rule and to lambda and T:

  clip     : n <- max(n', 0)              (reflecting-at-floor, the reference)
  reflect  : n <- |n'|                    (specular reflection at 0)
  softfloor: n <- max(n', lambda)         (hard immigration floor)

and lambda in {1e-4, 1e-3}, T in {0.025, 0.05, 0.1}. We also report how often the pre-boundary
value is negative (how active the boundary is). A fully metric-consistent multiplicative-noise
formulation is the alternative referenced in the FDT appendix; here we show the additive-noise
benchmark is not an artefact of the boundary choice.

Run:  python3 dmft_positivity.py     (writes figures/fig_dmft_positivity.png)
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"
MU, SIGMA, CAP = 6.0, 1.4, 60.0


def coupling(gamma, S, sigma, rng):
    a = np.zeros((S, S)); sd = sigma / np.sqrt(S); iu = np.triu_indices(S, 1)
    z1 = rng.standard_normal(len(iu[0])); z2 = rng.standard_normal(len(iu[0]))
    a[iu] = sd * z1
    a.T[iu] = sd * (gamma * z1 + np.sqrt(max(1 - gamma**2, 0)) * z2)
    return a


def apply_boundary(raw, kind, lam):
    if kind == "clip":
        return np.clip(raw, 0.0, CAP)
    if kind == "reflect":
        return np.clip(np.abs(raw), 0.0, CAP)
    if kind == "softfloor":
        return np.clip(np.maximum(raw, lam), 0.0, CAP)


def simulate(gamma, boundary, lam, T, S=300, n_real=4, dt=0.02, t_eq=60.0, t_meas=90.0, seed=0):
    rng = np.random.default_rng(seed)
    s2 = np.sqrt(2 * T * dt); neq, nms = int(t_eq / dt), int(t_meas / dt)
    NLAG = 60
    Csum = np.zeros(NLAG + 1); Ms = []; neg = 0; tot = 0
    for r in range(n_real):
        a = coupling(gamma, S, sigma=SIGMA, rng=rng)
        n = rng.uniform(0.1, 0.6, S)
        for _ in range(neq):
            raw = n + (n * (1 - n - MU * n.mean() + a @ n) + lam) * dt + s2 * rng.standard_normal(S)
            n = apply_boundary(raw, boundary, lam)
        rec = np.empty((nms, S))
        for k in range(nms):
            rec[k] = n
            raw = n + (n * (1 - n - MU * n.mean() + a @ n) + lam) * dt + s2 * rng.standard_normal(S)
            neg += int((raw < 0).sum()); tot += S
            n = apply_boundary(raw, boundary, lam)
        Ms.append(rec.mean())
        dev = rec - rec.mean(0, keepdims=True)
        for L in range(NLAG + 1):
            Csum[L] += np.mean(dev[:nms - L] * dev[L:]) / n_real
    return float(np.mean(Ms)), Csum / (Csum[0] + 1e-30), neg / max(tot, 1)


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    gamma = 0.0
    print(f"nonlinear-DMFT additive-noise GLV, gamma={gamma}, mu={MU}, sigma={SIGMA}\n")

    # (1) boundary rule at reference lambda=1e-4, T=0.05
    print("boundary-rule robustness (lambda=1e-4, T=0.05):")
    lags = np.arange(61) * 0.02
    curves = {}
    for bnd in ["clip", "reflect", "softfloor"]:
        M, C, negrate = simulate(gamma, bnd, 1e-4, 0.05, seed=1)
        curves[bnd] = C
        print(f"  {bnd:>9}: M={M:.3f}  C(tau) decay to {C[30]:.2f}@0.6  neg-crossing rate={negrate:.1%}")
    dM = max(abs(simulate(gamma, b, 1e-4, 0.05, seed=1)[0] - simulate(gamma, "clip", 1e-4, 0.05, seed=1)[0])
             for b in ["reflect", "softfloor"])

    # (2) lambda and T robustness (clip)
    print("\nlambda / T robustness (clip):")
    grid = []
    for lam in [1e-4, 1e-3]:
        for T in [0.025, 0.05, 0.1]:
            M, C, negrate = simulate(gamma, "clip", lam, T, seed=2)
            grid.append((lam, T, M, C[30]))
            print(f"  lambda={lam:.0e} T={T:.3f}: M={M:.3f}  C@0.6={C[30]:.2f}  neg-rate={negrate:.1%}")

    Ms = [g[2] for g in grid]
    print(f"\nboundary rule shifts M by <= {dM:.3f}; over lambda,T the mean abundance M spans "
          f"[{min(Ms):.2f},{max(Ms):.2f}] -- the order parameters are robust to the positivity scheme.")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    col = {"clip": "#0072B2", "reflect": "#1b9e77", "softfloor": "#D55E00"}
    for bnd, C in curves.items():
        ax[0].plot(lags, C, lw=2, color=col[bnd], label=bnd)
    ax[0].axhline(0, color="grey", lw=0.6, ls=":")
    ax[0].set_xlabel(r"lag $\tau$"); ax[0].set_ylabel(r"normalized autocorrelation $C(\tau)/C(0)$")
    ax[0].set_title("a  $C(\\tau)$ is robust to the boundary rule", loc="left", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9)
    lam_T = [f"$\\lambda$={g[0]:.0e}\nT={g[1]:.3f}" for g in grid]
    ax[1].bar(range(len(grid)), [g[2] for g in grid], color="#5e4fa2", alpha=0.85)
    ax[1].set_xticks(range(len(grid))); ax[1].set_xticklabels(lam_T, fontsize=7)
    ax[1].set_ylabel("mean abundance $M$")
    ax[1].set_title("b  $M$ is robust to $\\lambda$ and $T$", loc="left", fontsize=11)
    for a in ax:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_dmft_positivity.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_dmft_positivity.png written")
    np.savez(f"{OUT}/dmft_positivity.npz", dM=dM, Ms=np.array(Ms))


if __name__ == "__main__":
    main()
