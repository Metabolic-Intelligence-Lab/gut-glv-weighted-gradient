"""
Coordinate-invariant tests of non-potentiality  (referee-4 P0, replaces R_sym)
==============================================================================

Referee 4 is correct: the Frobenius-ratio residual
    R_sym(J) = inf_{d>0} ||diag(d)J - J^T diag(d)||_F / ||diag(d)J||_F
is NOT coordinate-invariant. Under a species rescaling x=S y the couplings become
J'=J diag(s) and a symmetrizer maps D->S D, but the two Frobenius norms are reweighted
by the diagonal congruence S(.)S differently, so the ratio changes; re-optimising over D
does not restore invariance. Concretely, a genuinely non-symmetrizable 3x3 has R_sym
driven from 0.21 to 5e-5 by column rescaling alone (test_Rsym_noninvariance below),
and on a reducible matrix the infimum -> 0 without any positive symmetrizer, so the
"zero iff a potential exists" claim fails too.

What IS invariant is the *exact* diagonal-symmetrizability obstruction, separated into
three coordinate-free diagnostics (a positive diagonal D with DJ=J^T D exists iff all
three vanish):
  f_oneway : fraction of interacting pairs present in only one direction (support asymmetry);
  f_sign   : fraction of bidirectional pairs with discordant signs  (no d>0 can match a
             +/- reciprocal pair);
  r_cycle  : on the reciprocal, sign-concordant subgraph, the RMS inconsistency of the
             log-ratios b_ij=log|J_ij/J_ji| around cycles,
                r_cycle = min_u { mean_(i,j) [ b_ij - (u_j-u_i) ]^2 }^{1/2},
             which is 0 iff b is a gradient (a symmetrizer exists on that component).
Under J->J diag(s), b_ij -> b_ij + (log s_j - log s_i): a pure gradient that cancels
around every cycle, so r_cycle is exactly invariant (verified numerically). f_oneway and
f_sign are invariant because support and sign are invariant under positive rescaling.

Because Venturelli and Clark are *inferred* gLV estimates, an exact violation could in
principle be inference noise. We therefore calibrate a null: a symmetric (hence
symmetrizable) ground truth of matched support/scale, corrupted by inference noise whose
level is anchored, for Clark, to the spread of its published posterior gamma draws. The
observed invariants are compared to this symmetrizable-plus-noise null.

Run:  python3 symmetrizability_invariants.py   (writes figures/fig_invariants.png)
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from symmetrizability import R_sym          # the refuted (non-invariant) metric, for the contrast panel
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"


# --------------------------------------------------------------------------- #
#  invariant diagnostics
# --------------------------------------------------------------------------- #
def invariants(J, thresh=0.0):
    """(f_oneway, f_sign, r_cycle, n_concordant_edges). thresh: |J|<=thresh*scale
    treated as structural zero (for sensitivity analysis). scale = median |offdiag!=0|."""
    S = len(J); iu = np.triu_indices(S, 1)
    a, b = J[iu].copy(), J.T[iu].copy()
    if thresh > 0:
        sc = np.median(np.abs(J[~np.eye(S, dtype=bool)][J[~np.eye(S, dtype=bool)] != 0]))
        a[np.abs(a) <= thresh * sc] = 0.0
        b[np.abs(b) <= thresh * sc] = 0.0
    both = (a != 0) & (b != 0)
    either = (a != 0) | (b != 0)
    f_oneway = 1 - both.sum() / max(either.sum(), 1)
    disc = both & (np.sign(a) != np.sign(b))
    f_sign = disc.sum() / max(both.sum(), 1)
    conc = both & (np.sign(a) == np.sign(b))
    ii, jj = iu[0][conc], iu[1][conc]
    E = len(ii)
    if E == 0:
        return f_oneway, f_sign, np.nan, 0
    bij = np.log(np.abs(a[conc] / b[conc]))
    B = np.zeros((E, S)); B[np.arange(E), jj] = 1; B[np.arange(E), ii] = -1
    u, *_ = np.linalg.lstsq(B, bij, rcond=None)
    r_cycle = float(np.sqrt(np.mean((bij - B @ u) ** 2)))
    return float(f_oneway), float(f_sign), r_cycle, E


# --------------------------------------------------------------------------- #
#  null: symmetric (symmetrizable) ground truth of matched support/scale + noise
# --------------------------------------------------------------------------- #
def symmetric_null(J, rng):
    """Symmetric matrix matched to J: same off-diagonal nonzero density, magnitudes
    drawn from J's empirical |nonzero| distribution, random signs, made symmetric
    (=> f_sign=0, r_cycle=0, f_oneway=0 exactly). Diagonal copied from J."""
    S = len(J); off = ~np.eye(S, dtype=bool)
    mags = np.abs(J[off]); mags = mags[mags > 0]
    dens = len(mags) / off.sum()
    A = np.zeros((S, S)); iu = np.triu_indices(S, 1)
    on = rng.random(len(iu[0])) < dens
    vals = rng.choice(mags, on.sum()) * rng.choice([-1.0, 1.0], on.sum())
    A[iu[0][on], iu[1][on]] = vals
    A = A + A.T
    np.fill_diagonal(A, np.diag(J))
    return A


def corrupt(A, nu, rng):
    """Independent additive inference noise per directed coefficient (breaks symmetry),
    scaled by the nonzero-coefficient scale. Noise perturbs only the modeled (nonzero)
    coefficients, so the symmetric support is preserved; nu is the relative noise level."""
    S = len(A); off = ~np.eye(S, dtype=bool)
    sc = np.median(np.abs(A[off][A[off] != 0])) if np.any(A[off] != 0) else 1.0
    J = A.copy(); mask = off & (np.abs(A) > 0)
    J[mask] = A[mask] + nu * sc * rng.standard_normal(mask.sum())
    return J


def gamma_pearson(J):
    S = len(J); iu = np.triu_indices(S, 1)
    return float(np.corrcoef(J[iu], J.T[iu])[0, 1])


def null_distribution(J, nu, rng, n=400):
    """f_sign, r_cycle, gamma over n noisy draws of a symmetrizable ground truth."""
    fs, rc, gm = [], [], []
    for _ in range(n):
        A = symmetric_null(J, rng)
        Jn = corrupt(A, nu, rng)
        _, f, r, E = invariants(Jn)
        fs.append(f); rc.append(r if E > 0 else np.nan); gm.append(gamma_pearson(Jn))
    return np.array(fs), np.array(rc), np.array(gm)


def calibrate_nu(J, target_gamma_std, rng, grid=np.linspace(0.1, 1.5, 15)):
    """Pick the noise level whose null gamma-spread matches an external posterior
    gamma spread (anchors the null to the real inference uncertainty)."""
    best, bnu = np.inf, grid[0]
    for nu in grid:
        _, _, gm = null_distribution(J, nu, rng, n=120)
        d = abs(np.nanstd(gm) - target_gamma_std)
        if d < best:
            best, bnu = d, nu
    return float(bnu)


# --------------------------------------------------------------------------- #
def main():
    import os; os.makedirs(OUT, exist_ok=True)
    V = np.loadtxt("data/venturelli2018/interaction_matrix_averaged.csv", delimiter=",", skiprows=1)
    C = np.loadtxt("data/clark2021/interaction_matrix_RLC8_posteriormean.csv", delimiter=",")
    gpost = np.loadtxt("data/clark2021/gamma_posterior_samples.csv")
    gstd = float(np.std(gpost))
    print(f"Clark posterior gamma: median={np.median(gpost):+.3f} std={gstd:.3f} (n={len(gpost)})\n")

    # --- (0) confirm R_sym is NOT invariant; r_cycle IS  (the reason for the switch) ---
    Jcx = np.array([[0., 1, 1], [1, 0, 1], [2, 1, 0]])   # genuinely non-symmetrizable
    print("R_sym vs r_cycle under column rescaling of a non-symmetrizable 3x3:")
    scales = [(1, 1, 1), (10, 1, 1), (1, 10, .1), (100, .01, 1)]
    rs_collapse, rc_flat = [], []
    for s in scales:
        Js = Jcx * np.array(s, float)[None, :]
        rs = R_sym(Js, restarts=6, maxit=1200); _, _, rc, _ = invariants(Js)
        rs_collapse.append(rs); rc_flat.append(rc)
        print(f"   S={str(s):<14}: R_sym={rs:.5f}   r_cycle={rc:.5f}")
    print(f"   -> R_sym collapses {rs_collapse[0]/rs_collapse[-1]:.0f}x; r_cycle invariant.\n")

    # --- (1) observed invariants on the real matrices, with threshold sensitivity ---
    print("observed invariants (f_oneway, f_sign, r_cycle):")
    obs = {}
    for name, J in [("Venturelli", V), ("Clark", C)]:
        f1, fs, rc, E = invariants(J)
        obs[name] = dict(f1=f1, fs=fs, rc=rc, E=E, J=J)
        thr = [(t, invariants(J, thresh=t)[:2]) for t in (0.1, 0.25, 0.5)]
        tt = "  ".join(f"@{t}:f1={v[0]:.2f},fsign={v[1]:.2f}" for t, v in thr)
        print(f"   {name:11}: f_oneway={f1:.3f} f_sign={fs:.3f} r_cycle={rc:.3f} (E={E}) | sens {tt}")
        # invariance of the whole triple under random rescale
        s = np.exp(np.random.default_rng(7).normal(0, 1.5, len(J)))
        f1b, fsb, rcb, _ = invariants(J * s[None, :])
        print(f"   {name:11}: after random rescale -> f_oneway={f1b:.3f} f_sign={fsb:.3f} r_cycle={rcb:.3f} (invariant)")

    # --- (2) symmetrizable + inference-noise null, swept over the noise level nu ---
    print("\nnull: symmetrizable ground truth + inference noise; p-value vs noise level nu")
    nus = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
    P = {(n, k): [] for n in ("Venturelli", "Clark") for k in ("fs", "rc")}
    gstd_nu = []
    for nu in nus:
        fsV, rcV, _   = null_distribution(V, nu, np.random.default_rng(11), n=500)
        fsC, rcC, gmC = null_distribution(C, nu, np.random.default_rng(12), n=500)
        gstd_nu.append(float(np.nanstd(gmC)))
        arrs = {("Venturelli", "fs"): fsV, ("Clark", "fs"): fsC,
                ("Venturelli", "rc"): rcV[np.isfinite(rcV)], ("Clark", "rc"): rcC[np.isfinite(rcC)]}
        for (n, k), arr in arrs.items():
            P[(n, k)].append(float(np.mean(arr >= obs[n]["fs" if k == "fs" else "rc"])))
    P = {kk: np.array(vv) for kk, vv in P.items()}
    ic = int(np.argmin(np.abs(np.array(gstd_nu) - gstd)))
    nu_cal = float(nus[ic])
    print(f"   calibrated noise nu={nu_cal:.2f} (null gamma std {gstd_nu[ic]:.3f} ~ posterior {gstd:.3f})")
    for (n, k), p in P.items():
        pc = float(np.interp(nu_cal, nus, p))
        tag = "ROBUST (p<0.05 all nu)" if np.all(p < 0.05) else ("marginal" if pc < 0.05 else "within noise")
        print(f"   {n:11} {k:2}: p(nu_cal)={pc:.3f}   [{tag}]")

    # --- high-precision Monte-Carlo p at the calibrated noise for the two robust channels,
    #     with the (b+1)/(B+1) correction and a 95% binomial upper bound (referee-6 §5,§12) ---
    B = 5000
    print(f"\nhigh-precision MC p at nu={nu_cal:.2f} (B={B} draws, (b+1)/(B+1) corrected):")
    from scipy.stats import binom
    pmc = {}
    for name, J, key in [("Venturelli", V, "rc"), ("Clark", C, "fs")]:
        fs, rc, _ = null_distribution(J, nu_cal, np.random.default_rng(99), n=B)
        arr = (rc[np.isfinite(rc)] if key == "rc" else fs)
        b = int(np.sum(arr >= obs[name]["rc" if key == "rc" else "fs"]))
        p_corr = (b + 1) / (len(arr) + 1)
        hi = binom.ppf(0.95, len(arr), max(b, 1) / len(arr)) / len(arr) if b > 0 else 3.0 / len(arr)
        pmc[(name, key)] = (b, len(arr), p_corr, hi)
        print(f"   {name:11} {key}: {b}/{len(arr)} exceedances -> p_MC={p_corr:.4f} (95% upper bound {hi:.4f})")

    # --- figure ---
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.7))
    # panel a: R_sym collapses, r_cycle flat  (why we replaced the metric)
    xs = np.arange(len(scales))
    ax[0].plot(xs, rs_collapse, "o-", color="#D55E00", lw=2, ms=8, label=r"$R_{\rm sym}$ (Frobenius ratio) -- gauge-dependent")
    ax[0].plot(xs, rc_flat, "s-", color="#1b9e77", lw=2, ms=8, label=r"$r_{\rm cycle}$ (log-ratio cycles) -- invariant")
    ax[0].set_yscale("log")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(["1:1:1", "10:1:1", "1:10:.1", "100:.01:1"], fontsize=8)
    ax[0].set_xlabel("species rescaling  $\\mathrm{diag}(s)$  (identical dynamics)")
    ax[0].set_ylabel("residual value")
    ax[0].set_title("a  A non-symmetrizable $3\\times3$: the Frobenius ratio is\ngauge-dependent, the cycle residual is not", loc="left", fontsize=11)
    ax[0].legend(fontsize=8, frameon=False, loc="lower left")
    # panel b: p-value vs inference-noise level -> robustness of each invariant to the null
    style = {("Venturelli", "fs"): ("#0072B2", "-", "o"), ("Clark", "fs"): ("#5e4fa2", "-", "o"),
             ("Venturelli", "rc"): ("#0072B2", "--", "s"), ("Clark", "rc"): ("#5e4fa2", "--", "s")}
    lab = {("Venturelli", "fs"): r"Venturelli $f_{\rm sign}$", ("Clark", "fs"): r"Clark $f_{\rm sign}$",
           ("Venturelli", "rc"): r"Venturelli $r_{\rm cycle}$", ("Clark", "rc"): r"Clark $r_{\rm cycle}$"}
    for kk, (c, ls, mk) in style.items():
        ax[1].plot(nus, P[kk], ls=ls, marker=mk, color=c, lw=1.8, ms=6, label=lab[kk])
    ax[1].axhline(0.05, color="k", ls=":", lw=1); ax[1].text(nus[-1], 0.065, "$p=0.05$", ha="right", fontsize=8)
    ax[1].axvline(nu_cal, color="#888", lw=1.2)
    ax[1].text(nu_cal, 0.92, "calibrated\nnoise", ha="center", fontsize=7.5, color="#666")
    ax[1].set_xlabel(r"inference-noise level $\nu$ (rel. to coefficient scale)")
    ax[1].set_ylabel(r"null exceedance $p$  (obs consistent with symmetrizable)")
    ax[1].set_title("b  Two channels exceed a coefficient-noise null; this null perturbs\nparameters only, and the inference-aware null reverses the reading", loc="left", fontsize=11)
    ax[1].legend(fontsize=8, frameon=False, loc="center right"); ax[1].set_ylim(0, 1.0)
    for a in ax:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_invariants.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_invariants.png written")
    np.savez(f"{OUT}/invariants.npz",
             venturelli=[obs["Venturelli"]["f1"], obs["Venturelli"]["fs"], obs["Venturelli"]["rc"]],
             clark=[obs["Clark"]["f1"], obs["Clark"]["fs"], obs["Clark"]["rc"]],
             nus=nus, p_Vfs=P[("Venturelli", "fs")], p_Cfs=P[("Clark", "fs")],
             p_Vrc=P[("Venturelli", "rc")], p_Crc=P[("Clark", "rc")],
             nu_cal=nu_cal, gstd=gstd)


if __name__ == "__main__":
    main()
