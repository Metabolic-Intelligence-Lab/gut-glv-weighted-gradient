"""
Knockout validation: per-fit provenance, differences between predictors, BU- contamination
==========================================================================================
(BIB-26-2181 revision, Reviewer 4 point 6.)

The manuscript validates an in-silico knockout ranking of Venturelli's gLV against the twelve
single-species leave-one-out communities of Dataset EV3 (venturelli_keystone_validation.py,
spectral_vs_knockout.py, keystone_robustness.py), using the fit trained on T4. Reviewer 4
asks for three things this script supplies.

1. TRAINING PROVENANCE, from the original study (Venturelli et al. 2018, Appendix Figs
   S25-S28 and Fig S15 caption):
       T1  monospecies (M) only                       -> no inter-species interactions
       T2  M + PW1 (pairwise, ~1:1 inoculum)
       T3  M + PW1 + PW2 (pairwise, 95:5 and 5:95)    -> the original paper's main model
       T4  M + PW1 + PW2 + the FULL 12-member community
   "The validation data set for T1-T3 included 12 single-species dropouts (11-member
   communities) and the full community (12-member) and T4 was validated on the set of
   11-member consortia." The impact of a knockout is measured AGAINST the full community, so
   only T2 and T3 are out of sample with respect to both the dropout outcomes and the
   full-community reference. T4 has seen the reference. T1 has no interactions, so its
   in-silico knockout leaves the composition of the remaining species unchanged and its
   impact is identically zero (reported, rho undefined).

2. DIFFERENCES between predictor correlations. For each competing predictor X (abundance,
   two weighted-degree definitions used in the code base, self-limitation, leading
   right-eigenvector weight, biorthogonal weight, node-removal abscissa shift), the
   difference rho(knockout, measured) - rho(X, measured), with
     - a paired bootstrap over the 12 species (same resample for both terms), percentile 95%;
     - a paired swap-permutation test: within each species the RANKS of the knockout and of
       X are exchanged with probability 1/2, which is exact under the null that the two
       rankings are exchangeable in their association with the measured impact.

3. BU- CONTAMINATION. The original study reports BU detected in the BU- consortium at 47.5 h
   and present for the final three time points; in EV3 the BU fraction in BU- is 0.0042,
   0.0158, 0.0224 at 48, 60, 72 h. The measured impact uses the mean composition at 60 and
   72 h, both contaminated. Sensitivities: (a) drop the BU- experiment (n = 11); (b) measure
   every community, not only BU-, on the latest pre-contamination window, 24 and 36 h; (c) the
   strict background window 12 and 24 h. Note that the impact metric is Bray-Curtis on the 11
   shared species, so for BU- the contaminating BU itself is already excluded from the metric;
   only its effect on the other species enters.

Immigration floor eps in {1e-4, 1e-3, 1e-2} is swept per fit, as in the manuscript's
"0.43-0.65 across fits and eps".

Run:  OMP_NUM_THREADS=1 python3 scripts/knockout_robustness_rev.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from venturelli_keystone_validation import load_glv, bc, partial_spearman, SP, S, DATA   # noqa: E402
from spectral_vs_knockout import eco_jacobian, leading                                     # noqa: E402
from constrained_vs_free_glv import load_communities                                       # noqa: E402

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = "figures"
FITS = ["T1", "T2", "T3", "T4"]
PROVENANCE = {
    "T1": "M only (no interactions)",
    "T2": "M + PW1",
    "T3": "M + PW1 + PW2 (paper's main model)",
    "T4": "M + PW1 + PW2 + full community",
}
OUT_OF_SAMPLE = {"T1": "yes (no interactions)", "T2": "yes", "T3": "yes",
                 "T4": "dropouts only; full-community reference in training"}
EPSS = [1e-4, 1e-3, 1e-2]
TIMES = [0, 12, 24, 36, 48, 60, 72]          # h, Venturelli Methods: samples every ~12 h
WINDOWS = {"published 60+72 h": (5, 6), "pre-contamination 24+36 h": (2, 3),
           "background 12+24 h": (1, 2)}
NPERM = 20000
NBOOT = 10000
BU = SP.index("BU")


def steady(u, A, mask, eps, tmax=800):
    """As venturelli_keystone_validation.steady, with the immigration floor as a parameter."""
    idx = np.where(mask)[0]

    def rhs(t, x):
        xi = np.clip(x, 0, None); xi[~mask] = 0
        dx = np.zeros(S); dx[idx] = xi[idx] * (u[idx] + A[idx] @ xi) + eps
        return dx
    sol = solve_ivp(rhs, [0, tmax], np.where(mask, 0.1, 0.0),
                    method="LSODA", rtol=1e-8, atol=1e-10)
    xf = np.clip(sol.y[:, -1], 0, None); xf[~mask] = 0
    return xf


def compositions(window):
    """Mean composition over the chosen EV3 time points, renormalised, per community."""
    out = {}
    for name, X in load_communities():
        m = X[list(window)].mean(0)
        out[name] = m / m.sum()
    return out


def empirical_impacts(comp):
    full = comp["NONE"]
    imp = np.zeros(S)
    for i in range(S):
        keep = [j for j in range(S) if j != i]
        imp[i] = bc(full[keep], comp[f"{SP[i]}-"][keep])
    return imp, full


def model_quantities(fit, eps):
    u, A = load_glv(f"{DATA}/twelveSpecies{fit}.xml")
    xfull = steady(u, A, np.ones(S, bool), eps)
    imp = np.zeros(S)
    for i in range(S):
        keep = [j for j in range(S) if j != i]
        m = np.ones(S, bool); m[i] = False
        imp[i] = bc(xfull[keep], steady(u, A, m, eps)[keep])
    off = ~np.eye(S, dtype=bool)
    q = {"knockout": imp,
         "self-limitation |a_ii|": np.abs(np.diag(A)),
         "total strength (row+col)": np.array([np.abs(A[i, off[i]]).sum() + np.abs(A[off[:, i], i]).sum()
                                               for i in range(S)]),
         "in-strength (row)": np.abs(A - np.diag(np.diag(A))).sum(1)}
    M, keep = eco_jacobian(u, A, xfull)
    absc, vr, vl = leading(M)
    w_eig = np.zeros(S); w_bio = np.zeros(S); d_abs = np.zeros(S)
    w_eig[keep] = np.abs(vr) / np.abs(vr).sum()
    bio = np.abs(vl * vr); w_bio[keep] = bio / bio.sum()
    for pos, i in enumerate(keep):
        sub = [k for k in range(len(keep)) if k != pos]
        d_abs[i] = absc - np.max(np.linalg.eigvals(M[np.ix_(sub, sub)]).real)
    q["eigenvector weight |v_i|"] = w_eig
    q["biorthogonal |l_i v_i|"] = w_bio
    q["node-removal d(abscissa)"] = d_abs
    n_off = int(np.sum(A[off] != 0))
    return q, n_off


def rho(x, y):
    return float(spearmanr(x, y).statistic)


def perm_p_one_sided(x, y, rng, nperm=NPERM):
    obs = rho(x, y)
    null = np.array([rho(x, rng.permutation(y)) for _ in range(nperm)])
    return (np.sum(null >= obs) + 1) / (nperm + 1)


def boot_ci(x, y, z, rng, nboot=NBOOT):
    n = len(x); br, bp = [], []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 4:
            continue
        br.append(rho(x[idx], y[idx]))
        bp.append(partial_spearman(x[idx], y[idx], z[idx]))
    br, bp = np.array(br), np.array(bp)
    return np.nanpercentile(br, [2.5, 97.5]), np.nanpercentile(bp, [2.5, 97.5])


def delta_rho(ko, xp, emp, rng):
    """rho(ko, emp) - rho(xp, emp): paired bootstrap CI and paired rank-swap permutation p."""
    obs = rho(ko, emp) - rho(xp, emp)
    n = len(ko); bd = []
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 4:
            continue
        bd.append(rho(ko[idx], emp[idx]) - rho(xp[idx], emp[idx]))
    lo, hi = np.nanpercentile(bd, [2.5, 97.5])
    rk, rx = rankdata(ko), rankdata(xp)
    hits = 0
    for _ in range(NPERM):
        sw = rng.random(n) < 0.5
        a = np.where(sw, rx, rk); b = np.where(sw, rk, rx)
        hits += abs(rho(a, emp) - rho(b, emp)) >= abs(obs) - 1e-12
    return obs, lo, hi, (hits + 1) / (NPERM + 1)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)
    comp_pub = compositions(WINDOWS["published 60+72 h"])
    emp, abund = empirical_impacts(comp_pub)

    # ------------------------------------------------------------------ model quantities
    Q, NOFF = {}, {}
    for f in FITS:
        for e in EPSS:
            Q[(f, e)], NOFF[f] = model_quantities(f, e)

    # ------------------------------------------------------------------ (2) per-fit table
    print("=" * 100)
    print("(2) PER-FIT KNOCKOUT VALIDATION, published window (60+72 h), eps = 1e-3, n = 12")
    print("=" * 100)
    print(f"{'fit':<4}{'training set':<38}{'nonzero a_ij':>13}{'rho':>7}{'p_asym':>8}{'p_perm':>8}"
          f"{'rho 95% boot':>17}{'rho|abund':>10}{'partial 95% boot':>19}  out of sample?")
    per_fit = {}
    for f in FITS:
        ko = Q[(f, 1e-3)]["knockout"]
        if np.max(np.abs(ko)) < 1e-6:
            print(f"{f:<4}{PROVENANCE[f]:<38}{NOFF[f]:>13}   knockout impact identically zero "
                  f"(max {np.max(np.abs(ko)):.1e}): rho undefined")
            per_fit[f] = None
            continue
        r_, p_ = spearmanr(ko, emp)
        pp = perm_p_one_sided(ko, emp, rng)
        ci, cip = boot_ci(ko, emp, abund, rng)
        pr = partial_spearman(ko, emp, abund)
        per_fit[f] = dict(rho=r_, p=p_, pperm=pp, ci=ci, partial=pr, cip=cip)
        print(f"{f:<4}{PROVENANCE[f]:<38}{NOFF[f]:>13}{r_:>7.2f}{p_:>8.3f}{pp:>8.4f}"
              f"   [{ci[0]:+.2f},{ci[1]:+.2f}]{pr:>10.2f}    [{cip[0]:+.2f},{cip[1]:+.2f}]  {OUT_OF_SAMPLE[f]}")

    print("\n    eps sweep, rho(knockout, measured) by fit:")
    print(f"    {'fit':<4}" + "".join(f"{'eps=' + format(e, '.0e'):>11}" for e in EPSS))
    eps_tab = {}
    for f in FITS:
        vals = []
        for e in EPSS:
            ko = Q[(f, e)]["knockout"]
            vals.append(np.nan if np.max(np.abs(ko)) < 1e-6 else rho(ko, emp))
        eps_tab[f] = vals
        print(f"    {f:<4}" + "".join(f"{v:>11.2f}" if np.isfinite(v) else f"{'undef':>11}" for v in vals))
    rng_all = [v for f in ("T2", "T3", "T4") for v in eps_tab[f] if np.isfinite(v)]
    rng_oos = [v for f in ("T2", "T3") for v in eps_tab[f] if np.isfinite(v)]
    print(f"    range over T2-T4 x eps: [{min(rng_all):.2f}, {max(rng_all):.2f}];  "
          f"out-of-sample fits T2-T3 only: [{min(rng_oos):.2f}, {max(rng_oos):.2f}]")

    # ------------------------------------------------------------------ (1) delta-rho
    print("\n" + "=" * 100)
    print("(1) DIFFERENCE IN SPEARMAN RHO WITH THE MEASURED IMPACT: knockout minus predictor X")
    print("    paired bootstrap over species (95% percentile) and paired rank-swap permutation p")
    print("=" * 100)
    dtab = {}
    for f in ("T3", "T4"):
        q = Q[(f, 1e-3)]
        ko = q["knockout"]
        preds = {"abundance (measured, 60+72 h)": abund}
        preds.update({k: v for k, v in q.items() if k != "knockout"})
        print(f"\n  fit {f}  (rho knockout = {rho(ko, emp):+.2f})")
        print(f"  {'predictor X':<34}{'rho X':>7}{'delta rho':>11}{'95% boot':>18}{'p_swap':>9}")
        for name, xp in preds.items():
            d, lo, hi, ps = delta_rho(ko, xp, emp, rng)
            dtab[(f, name)] = (rho(xp, emp), d, lo, hi, ps)
            print(f"  {name:<34}{rho(xp, emp):>7.2f}{d:>+11.2f}   [{lo:+.2f},{hi:+.2f}]{ps:>9.3f}")

    # ------------------------------------------------------------------ (3) BU- contamination
    print("\n" + "=" * 100)
    print("(3) BU- CONTAMINATION SENSITIVITY, eps = 1e-3")
    print("=" * 100)
    bu_series = dict(load_communities())["BU-"][:, BU]
    print("    BU fraction in the BU- consortium at " +
          ", ".join(f"{t} h: {v:.4f}" for t, v in zip(TIMES, bu_series)))
    keep11 = np.array([i for i in range(S) if i != BU])
    sens = {}
    print(f"\n    {'variant':<44}{'fit':<5}{'n':>3}{'rho':>7}{'p_perm':>9}{'rho 95% boot':>17}{'rho|abund':>10}")
    for f in ("T2", "T3", "T4"):
        ko = Q[(f, 1e-3)]["knockout"]
        variants = [("published, all 12", ko, emp, abund),
                    ("drop the BU- experiment", ko[keep11], emp[keep11], abund[keep11])]
        for wname in ("pre-contamination 24+36 h", "background 12+24 h"):
            e2, a2 = empirical_impacts(compositions(WINDOWS[wname]))
            variants.append((f"all 12 measured at {wname}", ko, e2, a2))
        for vname, k, e, a in variants:
            pp = perm_p_one_sided(k, e, rng)
            ci, _ = boot_ci(k, e, a, rng)
            pr = partial_spearman(k, e, a)
            sens[(f, vname)] = (len(k), rho(k, e), pp, ci, pr)
            print(f"    {vname:<44}{f:<5}{len(k):>3}{rho(k, e):>7.2f}{pp:>9.4f}"
                  f"   [{ci[0]:+.2f},{ci[1]:+.2f}]{pr:>10.2f}")
        print()
    # how much do the measured impacts move between windows?
    e_pre, _ = empirical_impacts(compositions(WINDOWS["pre-contamination 24+36 h"]))
    print(f"    rank agreement of measured impacts, 60+72 h vs 24+36 h: rho = {rho(emp, e_pre):.2f}")
    print(f"    BU- measured impact: 60+72 h = {emp[BU]:.3f}, 24+36 h = {e_pre[BU]:.3f}; "
          f"rank among 12: {int(rankdata(-emp)[BU])} vs {int(rankdata(-e_pre)[BU])}")

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10.5,
                         "ytick.labelsize": 10.5, "axes.titlesize": 12})
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    # a: per fit and variant
    variants = ["published, all 12", "drop the BU- experiment", "all 12 measured at pre-contamination 24+36 h"]
    labels = ["published (60+72 h)", "without BU- (n=11)", "pre-contamination (24+36 h)"]
    cols = ["#1a1a1a", "#0072B2", "#D55E00"]
    for j, (v, lab, c) in enumerate(zip(variants, labels, cols)):
        for k, f in enumerate(("T2", "T3", "T4")):
            n, r_, pp, ci, pr = sens[(f, v)]
            x = k + (j - 1) * 0.22
            ax[0].errorbar(x, r_, yerr=[[r_ - ci[0]], [ci[1] - r_]], fmt="o", color=c, ms=7,
                           capsize=3, label=lab if k == 0 else None)
    ax[0].axhline(0, color="#888888", lw=0.8)
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels(["T2\nM+PW1", "T3\nM+PW1+PW2", "T4\n+ full community"])
    ax[0].set_ylabel(r"Spearman $\rho$, in-silico vs measured knockout")
    ax[0].set_ylim(-1, 1)
    ax[0].set_title("a  Per training set, with 95% bootstrap intervals", loc="left")
    ax[0].legend(frameon=False, fontsize=9.5, loc="lower left")
    # b: delta rho forest, T3 and T4
    names = list(dict.fromkeys(n for (_, n) in dtab))
    yy = np.arange(len(names))[::-1]
    for f, off, c in (("T3", -0.15, "#0072B2"), ("T4", 0.15, "#1a1a1a")):
        for y, nm in zip(yy, names):
            _, d, lo, hi, ps = dtab[(f, nm)]
            ax[1].errorbar(d, y + off, xerr=[[d - lo], [hi - d]], fmt="o", color=c, ms=6,
                           capsize=3, label=f if y == yy[0] else None)
    ax[1].axvline(0, color="#888888", lw=0.8)
    ax[1].set_yticks(yy); ax[1].set_yticklabels(names)
    ax[1].set_xlabel(r"$\rho_{\rm knockout}-\rho_X$ (paired bootstrap 95%)")
    ax[1].set_title("b  Knockout minus each competing predictor", loc="left")
    ax[1].legend(frameon=False, fontsize=9.5, title="fit", loc="lower right")
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_knockout_robustness_rev.png", dpi=200, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_knockout_robustness_rev.png")

    np.savez(f"{OUT}/knockout_robustness_rev.npz",
             emp=emp, abund=abund, fits=np.array(FITS), epss=np.array(EPSS),
             eps_tab=np.array([eps_tab[f] for f in FITS]),
             ko_T2=Q[("T2", 1e-3)]["knockout"], ko_T3=Q[("T3", 1e-3)]["knockout"],
             ko_T4=Q[("T4", 1e-3)]["knockout"], e_pre=e_pre, bu_series=bu_series,
             dtab_keys=np.array([f"{f}|{n}" for (f, n) in dtab]),
             dtab_vals=np.array([dtab[k] for k in dtab]),
             sens_keys=np.array([f"{f}|{v}" for (f, v) in sens]),
             sens_vals=np.array([[sens[k][0], sens[k][1], sens[k][2], sens[k][3][0], sens[k][3][1],
                                  sens[k][4]] for k in sens]))
    print(f"[npz] {OUT}/knockout_robustness_rev.npz")


if __name__ == "__main__":
    main()
