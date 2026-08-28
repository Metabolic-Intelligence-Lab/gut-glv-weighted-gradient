"""
C7 check -- does the central formula's sqrt(gamma) dependence hold numerically?
===============================================================================

The manuscript's hub formula is  sigma_c ~ sqrt(<k> / (gamma * k_max)),  i.e. the
destabilising spectral abscissa a(gamma) ~ sqrt(gamma * k_max / <k>) grows like
sqrt(gamma). sparse_cavity.py only ever tested gamma=1 (symmetry baked in), so the
sqrt(gamma) factor was never verified. Here we build the sparse, partially
symmetric community matrix on a Barabasi-Albert graph (per-edge variance
1/<k>, corr(J_ij,J_ji)=gamma) and measure the spectral abscissa
a(gamma)=max Re lambda directly (dense eig, reliable), for gamma from ~0 to 1.

We report:
  * a(gamma) and sigma_c=1/a(gamma), vs the mean-field star prediction
    a_pred = sqrt(gamma * k_max / <k>);
  * the log-log slope of a vs gamma over the mean-field regime (gamma>=0.3);
    the claim predicts slope ~0.5. At small gamma the mean-field estimate
    diverges (sigma_c->inf) while the true abscissa stays finite -- the crossover
    the manuscript asserts.

Run:  python3 check_sigmac_vs_gamma.py
"""

import numpy as np
import networkx as nx

np.seterr(over="ignore", invalid="ignore", divide="ignore")


def community_matrix(S, m, gamma, rng):
    """Sparse partially-symmetric community matrix on a BA graph. Per-edge std
    1/sqrt(<k>) (sigma=1 baked in), corr(J_ij, J_ji)=gamma, zero mean, zero
    diagonal (self-limitation is the separate -1)."""
    G = nx.barabasi_albert_graph(S, m, seed=int(rng.integers(1 << 30)))
    A = nx.to_numpy_array(G)
    k = A.sum(1)
    kmean = k.mean()
    s = 1.0 / np.sqrt(kmean)
    J = np.zeros((S, S))
    iu = np.triu_indices(S, 1)
    a = rng.normal(size=len(iu[0]))
    b = rng.normal(size=len(iu[0]))
    upper = s * a
    lower = s * (gamma * a + np.sqrt(max(1 - gamma**2, 0.0)) * b)
    J[iu] = upper
    J[(iu[1], iu[0])] = lower
    J *= A                      # keep only graph edges
    return J, k


def abscissa(J):
    """Spectral abscissa max Re(lambda) via dense eig (reliable for these sizes)."""
    lam = np.linalg.eigvals(J)
    return float(np.max(lam.real))


def main():
    S, m, n_real = 600, 2, 4
    gammas = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    print(f"design: S={S}, m={m}, realizations={n_real}  (dense spectral abscissa)")
    print(f"{'gamma':>6} | {'a=maxRe(lam)':>12} | {'sigma_c=1/a':>11} | "
          f"{'star a_pred':>11} | {'sigma_c_pred':>12}")
    print("-" * 66)
    a_meas, a_pred_l, kmax_l, kmean_l = [], [], [], []
    for g in gammas:
        aa, kmaxs, kmeans = [], [], []
        for rep in range(n_real):
            rng = np.random.default_rng(4000 + int(g * 100) * 13 + rep)
            J, k = community_matrix(S, m, g, rng)
            aa.append(abscissa(J)); kmaxs.append(k.max()); kmeans.append(k.mean())
        a = np.mean(aa); km = np.mean(kmaxs); kc = np.mean(kmeans)
        a_pred = np.sqrt(g * km / kc)
        a_meas.append(a); a_pred_l.append(a_pred); kmax_l.append(km); kmean_l.append(kc)
        print(f"{g:>6.2f} | {a:>12.3f} | {1/a:>11.3f} | {a_pred:>11.3f} | "
              f"{(1/a_pred if a_pred>0 else np.inf):>12.3f}")

    a_meas = np.array(a_meas)
    # log-log slope of a(gamma) over the mean-field regime gamma>=0.3
    reg = gammas >= 0.3
    slope, intercept = np.polyfit(np.log(gammas[reg]), np.log(a_meas[reg]), 1)
    print(f"\nlog-log slope of a(gamma) over gamma>=0.3: {slope:.3f} "
          f"(sqrt-law predicts 0.5)")
    # crossover: mean-field sigma_c_pred diverges as gamma->0, measured stays finite
    print(f"at gamma={gammas[0]:.2f}: measured a={a_meas[0]:.3f} (finite, "
          f"sigma_c={1/a_meas[0]:.2f}) while star a_pred={a_pred_l[0]:.3f} "
          f"-> mean-field sigma_c_pred={1/a_pred_l[0]:.1f} (diverging)")
    verdict = ("CONFIRMED: a(gamma) ~ gamma^0.5 in the mean-field regime and the "
               "abscissa stays finite at small gamma"
               if 0.35 <= slope <= 0.65 else
               "PARTIAL: slope deviates from 0.5 -- report measured exponent")
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
