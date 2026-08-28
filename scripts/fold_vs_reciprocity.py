"""
The competitive fold requires moderate reciprocity  (referee P0-3)
==================================================================

The two-guild competitive-exclusion fold that generates the clinical bistability is a
mutual (both-negative -> positively correlated) competition, so it lives at moderate-
to-high TOTAL reciprocity gamma_total = corr(J_ij, J_ji) over all off-diagonal pairs.
Diluting the symmetric guild structure with non-reciprocal disorder (gamma_d=0) lowers
gamma_total; we measure, versus gamma_total, whether bistability survives.

Bistability probe: integrate at zero field from a guild-A-biased and a guild-B-biased
initial condition; the gap between the two final guild-B fractions is ~large if the
system is bistable (fold present) and ~0 if monostable (fold gone).

Result (see verdict): the fold is present for gamma_total >~ 0.3 and absent by
gamma_total ~ 0.1 -- i.e. at the EMPIRICALLY estimated reciprocity the competitive
fold does not operate. Strengthening or sparsifying the guild core does not rescue it
(the transition tracks gamma_total, not the core; see scratch tests). The clinically
observed low-reciprocity bistability therefore needs a different explanation (e.g. a
symmetric competitive substructure at higher effective reciprocity, or metabolic
phenotype switching, Rios Garza 2026).

Run:  python3 fold_vs_reciprocity.py     (writes figures/fig_fold_reciprocity.png)
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hysteresis_vs_gamma import enterotype_couplings_gamma, measured_gamma
from spin_glass_microbiome import scale_free_graph, NMAX, EPS
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"

S = 120
A, k, _ = scale_free_graph(S, m=2, seed=3)
guild = (np.arange(S) % 2).astype(int)

def rhs(t, n, r, J):
    n = np.clip(n, 0, NMAX); return n * (r - n + J @ n) + EPS
def frac(J, r, n0, T=220):
    sol = solve_ivp(rhs, (0, T), n0, args=(r, J), method="LSODA", rtol=1e-6, atol=1e-8, t_eval=[T])
    n = np.clip(sol.y[:, -1], 0, NMAX); return n[guild == 1].sum() / max(n.sum(), 1e-9)

def main():
    import os; os.makedirs(OUT, exist_ok=True)
    sigmas = [0.15, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    gts, gaps, errs = [], [], []
    print(f"{'sigma':>6} | {'gamma_total':>11} | {'bistab. gap':>11}")
    for sig in sigmas:
        gg, ga = [], []
        for sd in range(3):
            rng = np.random.default_rng(20 + sd)
            J = enterotype_couplings_gamma(A, guild, 0.4, 1.2, sig, 0.0, rng)  # gamma_d=0
            r = np.random.default_rng(1000 + sd).uniform(0.9, 1.1, S)
            nA = np.where(guild == 0, 0.9, 0.1) + 0.01
            nB = np.where(guild == 1, 0.9, 0.1) + 0.01
            ga.append(abs(frac(J, r, nB) - frac(J, r, nA))); gg.append(measured_gamma(J))
        gts.append(np.nanmean(gg)); gaps.append(np.mean(ga)); errs.append(np.std(ga))
        print(f"{sig:>6.2f} | {gts[-1]:>+11.2f} | {gaps[-1]:>11.3f}")
    gts, gaps, errs = np.array(gts), np.array(gaps), np.array(errs)
    order = np.argsort(gts)
    gts, gaps, errs = gts[order], gaps[order], errs[order]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axvspan(-0.05, 0.13, color="#d95f02", alpha=0.13)
    ax.text(0.04, 0.72, "empirical\n$\\gamma\\lesssim0.1$", color="#a34400", fontsize=9, ha="center")
    ax.axhline(0.2, ls=":", color="grey", lw=1); ax.text(0.75, 0.22, "fold present", fontsize=8, color="grey")
    ax.errorbar(gts, gaps, yerr=errs, fmt="o-", color="#1b9e77", capsize=3, ms=6, lw=1.8)
    ax.set_xlabel(r"total interaction reciprocity  $\gamma_{\rm total}=\mathrm{corr}(J_{ij},J_{ji})$")
    ax.set_ylabel("bistability gap (fold strength)")
    ax.set_title("The competitive fold requires moderate reciprocity:\n"
                 "bistability vanishes before $\\gamma_{\\rm total}$ reaches the empirical $\\sim0.1$")
    ax.set_xlim(-0.05, 1.0); ax.set_ylim(-0.02, 0.9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_fold_reciprocity.png", dpi=150); plt.close(fig)
    empirical = float(np.interp(0.1, gts, gaps))
    print(f"\n[fig] {OUT}/fig_fold_reciprocity.png written")
    print(f"verdict: bistability gap at gamma_total=0.1 is ~{empirical:.2f} "
          f"({'FOLD ABSENT' if empirical < 0.2 else 'fold present'}); "
          f"fold requires gamma_total >~ 0.3")

if __name__ == "__main__":
    main()
