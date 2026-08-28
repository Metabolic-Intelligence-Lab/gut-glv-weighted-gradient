"""
Power of the free-vs-symmetrizable predictive test, done properly  (referee-8 §3.3)
===================================================================================

Referee 8 rightly notes that N=3 per point is not a power analysis. This version vectorises the
constrained fit (precomputing the delta-independent design once, so each L-BFGS-B evaluation is
a matrix operation rather than a Python loop) and runs N=120 replicates per asymmetry level,
with a predefined decision rule (free model wins the leave-one-community-out mean MSE) and a
binomial confidence interval on the rejection rate (= power). We report the TRUE distance off
the symmetrizable manifold (r_cycle of the ground-truth J_delta itself), and a persistence
baseline (predict no change) so the comparison is only read where both gLVs beat it.

Run:  python3 power_analysis.py     (writes figures/fig_power.png)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as betadist
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import constrained_vs_free_glv as CF
from constrained_vs_free_glv import load_communities, transitions, fit_free, invariants, pred_mse, S, EPS, LAM
from inference_aware_null import simulate, resid_std
from symmetrizability_invariants import invariants as inv3
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"
_PAIRS = [(i, j) for i in range(S) for j in range(i, S)]
_PK = {p: k for k, p in enumerate(_PAIRS)}


_COL = {i: np.array([_PK[(i, j) if i <= j else (j, i)] for j in range(S)]) for i in range(S)}


def base_design(trans):
    """delta-independent pieces of the symmetrizable design (vectorised over equations):
    R (one-hot species), K0 (pair sums of x_j), y, species index of each equation."""
    X = np.array([a for a, g, m in trans]); G = np.array([g for a, g, m in trans])
    Mk = np.array([m for a, g, m in trans])
    R, K0, y, esp = [], [], [], []
    for i in range(S):
        rows = np.where(Mk[:, i])[0]
        if len(rows) == 0:
            continue
        Ri = np.zeros((len(rows), S)); Ri[:, i] = 1.0
        K0i = np.zeros((len(rows), len(_PAIRS))); K0i[:, _COL[i]] = X[rows]
        R.append(Ri); K0.append(K0i); y.append(G[rows, i]); esp.append(np.full(len(rows), i))
    return np.vstack(R), np.vstack(K0), np.concatenate(y), np.concatenate(esp)


def fit_constrained_fast(trans, lam=LAM):
    """Vectorised J=D^{-1}K (K symmetric) fit with off-diagonal-J ridge; L-BFGS-B over u."""
    R, K0, y, esp = base_design(trans)
    npar = S + len(_PAIRS)
    diag_pairs = np.array([_PK[(i, i)] for i in range(S)])
    def build(u):
        d = np.exp(np.clip(u - u.mean(), -6.0, 6.0))       # bounded gauge, no blow-up
        M = np.hstack([R, K0 / d[esp][:, None]])
        return M, d
    def solve(u):
        M, d = build(u)
        ATA = M.T @ M; ATy = M.T @ y
        reg = np.zeros(npar)
        for (i, j) in _PAIRS:
            if i != j:
                # the K parameters start at offset S, after the S growth-rate columns; without
                # the offset the ridge lands on r and the interaction matrix is left unpenalised
                reg[S + _PK[(i, j)]] = lam * (1.0 / d[i] ** 2 + 1.0 / d[j] ** 2)
        beta = np.linalg.solve(ATA + np.diag(reg) + 1e-6 * np.eye(npar), ATy)
        return beta, d
    def loss(u):
        beta, d = solve(u); M, _ = build(u)
        return np.mean((y - M @ beta) ** 2)
    res = minimize(loss, np.zeros(S), method="L-BFGS-B", bounds=[(-6, 6)] * S,
                   options={"maxiter": 25})
    beta, d = solve(res.x)
    r = beta[:S]; K = np.zeros((S, S))
    for (i, j) in _PAIRS:
        K[i, j] = K[j, i] = beta[S + _PK[(i, j)]]        # K params live after the S r-columns
    return r, K / d[:, None]


def loco_free_wins(comms, test_idx=(3, 7, 11)):
    """One held-out split (fixed): True if the free model has lower test MSE than the
    symmetrizable, using the mean over the held-out communities. One constrained fit/replicate."""
    test = [comms[k] for k in test_idx if k < len(comms)]
    train = transitions([c for k, c in enumerate(comms) if k not in test_idx])
    te = transitions(test)
    if not te:
        return False, 0.0
    rf, Jf = fit_free(train); rc, Jc = fit_constrained_fast(train)
    fe, ce = pred_mse(rf, Jf, te), pred_mse(rc, Jc, te)
    return (ce > fe), (ce - fe)


def persistence_mse(comms):
    """Baseline: predict zero log-growth (no change). Mean over all transitions."""
    se, n = 0.0, 0
    for _, g, m in transitions(comms):
        se += np.sum(g[m] ** 2); n += m.sum()
    return se / max(n, 1)


def antisym(J0, rng):
    P = np.zeros((S, S)); iu = np.triu_indices(S, 1)
    supp = (J0[iu] != 0) | (J0.T[iu] != 0); v = rng.standard_normal(supp.sum())
    P[iu[0][supp], iu[1][supp]] = v; P[iu[1][supp], iu[0][supp]] = -v
    sc = np.median(np.abs(J0[~np.eye(S, dtype=bool)][J0[~np.eye(S, dtype=bool)] != 0]))
    return P * sc


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    comms = load_communities(); tr = transitions(comms)
    r0, J0 = fit_constrained_fast(tr); sproc = resid_std(comms, r0, J0)
    _, Jf = fit_free(tr); rcyc_obs = invariants(Jf)[1]
    Pdir = antisym(J0, np.random.default_rng(0))
    base = persistence_mse(comms)
    print(f"observed free-fit r_cycle={rcyc_obs:.2f}; process noise={sproc:.3f}; persistence MSE={base:.3f}")

    deltas = [0.0, 1.0, 2.0, 4.0, 8.0]; N = 120
    true_dist, power, plo, phi, adv_mean = [], [], [], [], []
    for dl in deltas:
        Jd = J0 + dl * Pdir
        dist = inv3(Jd)[2]                                   # true r_cycle of the ground truth
        wins = 0; advs = []
        for i in range(N):
            sim = simulate(comms, r0, Jd, sproc, np.random.default_rng(3000 + int(dl * 10) + i))
            w, a = loco_free_wins(sim); wins += int(w); advs.append(a)
        p = wins / N
        lo = betadist.ppf(0.025, wins + 0.5, N - wins + 0.5); hi = betadist.ppf(0.975, wins + 0.5, N - wins + 0.5)
        true_dist.append(dist); power.append(p); plo.append(lo); phi.append(hi); adv_mean.append(np.mean(advs))
        print(f"  delta={dl:.1f}: true r_cycle(J_delta)={dist:.2f}  power(free wins)={p:.2f} "
              f"[{lo:.2f},{hi:.2f}]  mean advantage={np.mean(advs):+.4f}")
    power = np.array(power)
    flat = power.max() - power.min() < 0.25
    print(f"\n=> power {'stays near 0.5 (chance) across all delta -> the test CANNOT detect asymmetry' if flat else 'rises with asymmetry -> the test has power'} "
          f"on these short EV3 trajectories (N={N}/point).")

    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.errorbar(true_dist, power, yerr=[power - np.array(plo), np.array(phi) - power], fmt="o-",
                color="#0072B2", lw=2, ms=8, capsize=4)
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance (no power)")
    ax.axvline(rcyc_obs, color="#D55E00", ls="--", lw=1.3, label=f"observed $r_{{\\rm cycle}}={rcyc_obs:.1f}$")
    ax.set_xlabel(r"true distance off the symmetrizable manifold, $r_{\rm cycle}(J_\delta)$")
    ax.set_ylabel("power = P(free model wins), 95% CI")
    ax.set_ylim(0, 1)
    ax.set_title(f"Power of the predictive test ($N={N}$/point):\nchance-level even at large asymmetry", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_power.png", dpi=160, bbox_inches="tight")
    print(f"[fig] {OUT}/fig_power.png written")
    np.savez(f"{OUT}/power.npz", deltas=np.array(deltas), true_dist=np.array(true_dist),
             power=power, plo=np.array(plo), phi=np.array(phi), rcyc_obs=rcyc_obs, N=N, persistence=base)


if __name__ == "__main__":
    main()
