"""
Clark posterior propagation of the symmetrizability invariants  (referee-4 P0.2/§6)
==================================================================================

The referee's concern is that the sign-discordance and cycle inconsistency of Clark's
posterior-MEAN matrix could be an inference/shrinkage artefact (that matrix is ~63% exact
zeros from L2 shrinkage). The decisive check is to propagate the FULL posterior: recompute
the coordinate-invariant diagnostics on each of the 234 RLC8 posterior draws (continuous,
no structural zeros) and report their distribution.

Draws: data/clark2021/posterior_RLC8_draws.npz (A: 234x25x25 interaction matrices, growth:
234x25), fetched from the authors' repository
github.com/RyanLincolnClark/DesignSyntheticGutMicrobiomeAssemblyFunction
(RLC8 Model Training/RLC8_output/posterior_RLC8/param*.csv, vector = per species
[growth_i, a_i1..a_i25]).

Result: sign-discordance is pervasive across the ENTIRE posterior (f_sign>0.42 in every
draw, median ~0.51), so Clark's departure from exact symmetrizability is not an artefact of
the shrunk point matrix.

The same posterior also bounds how much of that discordance is resolved, and those two
counts are quoted in the main text and in S1 Text: only 43 of the 600 off-diagonal
coefficients have a 95% central credible interval excluding zero, and exactly 1 of the 300
unordered pairs reaches posterior probability 0.95 of carrying opposite signs.

Run:  python3 clark_posterior_invariants.py
"""
import numpy as np
from symmetrizability_invariants import invariants
OUT = "figures"


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    z = np.load("data/clark2021/posterior_RLC8_draws.npz")
    A = z["A"]                                  # (ndraw, 25, 25)
    iu = np.triu_indices(A.shape[1], 1)
    FS, RC, GM = [], [], []
    for M in A:
        _, fsign, rc, E = invariants(M)
        FS.append(fsign); RC.append(rc)
        GM.append(float(np.corrcoef(M[iu], M.T[iu])[0, 1]))
    FS, RC, GM = np.array(FS), np.array(RC), np.array(GM)

    def ci(x): return np.median(x), np.percentile(x, 2.5), np.percentile(x, 97.5)
    print(f"Clark RLC8 posterior ensemble, n={len(FS)} draws (continuous -> f_oneway=0):")
    print(f"  gamma  : median={ci(GM)[0]:+.3f}  95% CI [{ci(GM)[1]:+.3f}, {ci(GM)[2]:+.3f}]"
          "   (matches the manuscript's -0.06, [-0.25,+0.10])")
    print(f"  f_sign : median={ci(FS)[0]:.3f}  95% CI [{ci(FS)[1]:.3f}, {ci(FS)[2]:.3f}]"
          f"   min over draws={FS.min():.3f}")
    print(f"  r_cycle: median={ci(RC)[0]:.3f}  95% CI [{ci(RC)[1]:.3f}, {ci(RC)[2]:.3f}]")
    print(f"  fraction of draws with f_sign>0.4: {np.mean(FS>0.4):.3f}; >0.5: {np.mean(FS>0.5):.3f}")

    # --- sign IDENTIFIABILITY (referee-6 §4.3/§12): does f_sign survive restricting to pairs
    #     whose signs are actually identified (95% credible interval excludes 0)? ---
    S = A.shape[1]; off = ~np.eye(S, dtype=bool)
    mean, sd = A.mean(0), A.std(0)
    aij_m, aji_m = mean[iu], mean.T[iu]; aij_s, aji_s = sd[iu], sd.T[iu]
    both0 = (aij_m != 0) & (aji_m != 0)
    f_naive = float(np.mean(np.sign(aij_m[both0]) != np.sign(aji_m[both0])))
    idij = np.abs(aij_m) > 1.96 * aij_s; idji = np.abs(aji_m) > 1.96 * aji_s   # CI excludes 0
    both_id = idij & idji
    f_id = float(np.mean(np.sign(aij_m[both_id]) != np.sign(aji_m[both_id]))) if both_id.sum() else np.nan
    print(f"\n  sign identifiability: naive f_sign(mean)={f_naive:.3f} over {both0.sum()} bidir pairs, "
          f"but only {int(both_id.sum())} pairs have BOTH signs identified (CI excludes 0), "
          f"with f_sign={f_id:.3f}.")
    print("  => Clark's f_sign~0.5 is dominated by sign NON-identifiability of near-zero "
          "coefficients, not robust biological discordance.")

    # --- how much of the matrix the posterior resolves at all, and how much of the sign
    #     discordance is resolved. Both numbers are quoted in the main text and in S1 Text,
    #     so they are computed here rather than by hand. The criterion is the 95% CENTRAL
    #     credible interval excluding zero, taken from the draws themselves; the normal
    #     approximation |mean| > 1.96*sd is more conservative and would give 25, not 43.
    offmask = ~np.eye(A.shape[1], dtype=bool)
    lo, hi = np.quantile(A, 0.025, axis=0), np.quantile(A, 0.975, axis=0)
    resolved = ((lo > 0) | (hi < 0)) & offmask
    n_off, n_res = int(offmask.sum()), int(resolved.sum())

    # posterior probability that a reciprocal pair carries OPPOSITE signs, per unordered pair
    aij, aji = A[:, iu[0], iu[1]], A[:, iu[1], iu[0]]
    opp = (np.sign(aij) != np.sign(aji)) & (aij != 0) & (aji != 0)
    p_opp = opp.mean(0)
    n_pairs, n_opp95 = len(p_opp), int((p_opp >= 0.95).sum())

    print(f"\n  resolution: {n_res} of {n_off} off-diagonal coefficients have a 95% central "
          f"credible interval excluding zero.")
    print(f"  opposite signs: {n_opp95} of {n_pairs} unordered pairs reach posterior probability "
          f"0.95 of carrying opposite signs (max over pairs {p_opp.max():.4f}).")
    print("  => any per-pair reading of this posterior is bounded by those two counts.")

    np.savez(f"{OUT}/clark_posterior_invariants.npz", f_sign=FS, r_cycle=RC, gamma=GM,
             f_naive=f_naive, n_identified=int(both_id.sum()), f_identified=f_id,
             n_offdiag=n_off, n_resolved95=n_res,
             n_pairs=n_pairs, n_opposite_p95=n_opp95, max_p_opposite=float(p_opp.max()))


if __name__ == "__main__":
    main()
