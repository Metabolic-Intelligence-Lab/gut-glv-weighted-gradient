"""
Honest uncertainty on the reciprocity gamma  (referee-3 P0.3)
=============================================================

Two objections to the way the small human gamma is defended:

  (A) The reported bootstrap CI resamples TRANSITIONS i.i.d. But the transitions
      are not independent: they are clustered within a handful of subjects and are
      temporally autocorrelated within each subject's series. An i.i.d.-transition
      bootstrap therefore UNDER-states the uncertainty (it treats 301 correlated
      rows as 301 independent draws). The honest interval comes from a two-level
      resample that respects both structures:
        - SUBJECT bootstrap: resample whole subjects with replacement (between-
          subject clustering; here only 5 clusters -> genuinely coarse);
        - BLOCK  bootstrap: moving blocks of consecutive transitions within a
          subject (within-subject temporal autocorrelation);
        - HIERARCHICAL: resample subjects, then moving-blocks within each -> the
          reported honest CI.

  (B) "Large true gamma is excluded" was argued from the mean recovery line only.
      Here we invert the FULL simulated distribution of gamma_hat: for a grid of
      gamma_true we estimate P(gamma_hat <= gamma_obs | gamma_true) and read off the
      one-sided upper confidence bound
          gamma_up(alpha) = sup { g : P_g(gamma_hat <= gamma_obs) >= alpha },
      a proper test-inversion confidence set (simulation-based inference), valid in
      the worst plausible nuisance case we sweep.

Run:  python3 gamma_confidence.py      (writes figures/fig_gamma_confidence.png)
Slow part is (B) -- the GLV recovery simulator. Run in background.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

import infer_and_place as M
from infer_and_place import (load, load_qpcr, fit_matrix, disorder_stats,
                             S_TAXA, MAX_GAP, PERTURBS, DATA)

OUT = "figures"
ALPHAS = (0.05, 0.10, 0.16)   # 95%, 90%, ~84% one-sided upper bounds
BLOCK_L = 7                   # moving-block length (~1 week of daily transitions)


# --------------------------------------------------------------------------- #
#  (A)  design assembly WITH subject tags + within-subject block structure
# --------------------------------------------------------------------------- #
def build_tagged(data_dir=DATA, subjects=(1, 2, 3, 4, 5)):
    """Exactly the transition design of infer_and_place.build_design, but also
    returns, per transition row, its subject id. Rows are grouped by subject and
    time-ordered within a subject, so a subject's rows form one contiguous block
    on which moving-block resampling is well defined."""
    rel, subj, time, sids = load(data_dir)
    qpcr = load_qpcr(data_dir)
    top = np.argsort(rel.mean(0))[::-1][:S_TAXA]
    load_vec = np.array([qpcr.get(s, np.nan) for s in sids])
    Xabs = rel[:, top] * load_vec[:, None]
    scale = np.nanmedian(np.where(Xabs > 0, Xabs, np.nan), axis=0)
    Xs = Xabs / scale
    pert = np.zeros((len(sids), len(PERTURBS)))
    for j, (_, s, e) in enumerate(PERTURBS):
        for i in range(len(sids)):
            if subj[i] in subjects and s <= time[i] <= e:
                pert[i, j] = 1.0
    rows_y, rows_Z, tag = [], [], []
    eps = 1e-3
    for sb in subjects:
        idx = np.where(subj == sb)[0]
        idx = idx[np.argsort(time[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            dt = time[b] - time[a]
            if dt <= 0 or dt > MAX_GAP:
                continue
            xa, xb = Xs[a], Xs[b]
            if np.any(np.isnan(xa)) or np.any(np.isnan(xb)):
                continue
            g = (np.log(xb + eps) - np.log(xa + eps)) / dt
            rows_y.append(g)
            rows_Z.append(np.concatenate([[1.0], xa, pert[a]]))
            tag.append(sb)
    return np.array(rows_y), np.array(rows_Z), np.array(tag)


def gamma_of(Y, Z):
    try:
        _, _, g = disorder_stats(fit_matrix(Y, Z))
        return g
    except Exception:
        return np.nan


def moving_block_rows(order_rows, rng, L=BLOCK_L):
    """Moving-block resample of a single subject's contiguous, time-ordered row
    indices. Returns a same-length resampled index list preserving local temporal
    order within each block."""
    n = len(order_rows)
    if n <= L:
        return order_rows.copy()
    out = []
    n_start = n - L + 1
    while len(out) < n:
        s = rng.integers(0, n_start)
        out.extend(order_rows[s:s + L])
    return np.array(out[:n])


def bootstraps_real(nboot=2000, seed=0):
    """Four resampling schemes for gamma on the real cohort:
       iid   : resample transitions i.i.d.            (the current, over-optimistic CI)
       block : moving-block within subject            (temporal autocorrelation only)
       subj  : resample whole subjects                (between-subject clustering only)
       hier  : subjects, then blocks within each      (honest, both structures)"""
    Y, Z, tag = build_tagged()
    subs = np.unique(tag)
    rows_by = {s: np.where(tag == s)[0] for s in subs}
    ghat = gamma_of(Y, Z)
    rng = np.random.default_rng(seed)
    nt = Y.shape[0]
    out = {k: [] for k in ("iid", "block", "subj", "hier")}

    for _ in range(nboot):
        # iid transitions
        s = rng.integers(0, nt, nt)
        out["iid"].append(gamma_of(Y[s], Z[s]))
        # moving-block within each subject (keep all subjects)
        idx = np.concatenate([moving_block_rows(rows_by[s], rng) for s in subs])
        out["block"].append(gamma_of(Y[idx], Z[idx]))
        # subject resample (whole clusters, with replacement)
        drawn = rng.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([rows_by[s] for s in drawn])
        out["subj"].append(gamma_of(Y[idx], Z[idx]))
        # hierarchical: subjects then blocks within each drawn subject
        idx = np.concatenate([moving_block_rows(rows_by[s], rng) for s in drawn])
        out["hier"].append(gamma_of(Y[idx], Z[idx]))

    res = {}
    for k, v in out.items():
        v = np.array(v); v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        res[k] = dict(lo=float(lo), hi=float(hi), width=float(hi - lo),
                      med=float(np.median(v)), samples=v)
    return ghat, res


# --------------------------------------------------------------------------- #
#  (B)  simulator-based confidence set: invert P(ghat <= gobs | gamma_true)
# --------------------------------------------------------------------------- #
def recovery_distribution(grid, n_rep=40, cache=f"{OUT}/gamma_confidence_dist.npz"):
    """For each gamma_true in grid, return the array of recovered gamma_hat over
    n_rep independent synthetic communities pushed through the SAME pipeline.
    Deterministic (fixed per-rep seeds) -> cached so figure tweaks don't re-simulate."""
    import os
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(np.round(z["grid"], 3)) == list(np.round(grid, 3)) and int(z["n_rep"]) == n_rep:
            print("   (loaded cached recovery distribution)")
            return {float(g): z[f"d{i}"] for i, g in enumerate(grid)}
    from recovery_gamma import recover_once
    dist = {}
    for gt in grid:
        vals = []
        for rep in range(n_rep):
            gh, sg, bad, _ = recover_once(gt, seed=7000 + int(round(gt * 100)) * 31 + rep)
            if not bad and np.isfinite(gh):
                vals.append(gh)
        dist[gt] = np.array(vals)
        print(f"   gamma_true={gt:.2f}: n={len(vals):2d}  "
              f"ghat med={np.median(vals):+.3f}  "
              f"[{np.percentile(vals,5):+.3f},{np.percentile(vals,95):+.3f}]", flush=True)
    np.savez(cache, grid=np.array(grid), n_rep=n_rep,
             **{f"d{i}": dist[g] for i, g in enumerate(grid)})
    return dist


def upper_bounds(dist, g_obs, alphas=ALPHAS):
    """gamma_up(alpha) = largest gamma_true with P(ghat <= g_obs) >= alpha,
    linearly interpolated in gamma_true on the exceedance curve p(gamma_true)."""
    gs = np.array(sorted(dist))
    p = np.array([np.mean(dist[g] <= g_obs) for g in gs])   # monotone decreasing in g
    bounds = {}
    for al in alphas:
        if p[0] < al:                      # even gamma_true=0 already too unlikely
            bounds[al] = float(gs[0]); continue
        if p[-1] >= al:                    # grid does not reach the crossing
            bounds[al] = float(gs[-1]); continue
        k = np.where(p < al)[0][0]         # first grid point below alpha
        g0, g1, p0, p1 = gs[k-1], gs[k], p[k-1], p[k]
        bounds[al] = float(g0 + (al - p0) * (g1 - g0) / (p1 - p0))
    return gs, p, bounds


# --------------------------------------------------------------------------- #
def main():
    import os; os.makedirs(OUT, exist_ok=True)

    print("=== (A) honest bootstrap CIs on the real cohort (gamma) ===")
    ghat, res = bootstraps_real(nboot=2000)
    print(f"full-sample gamma_hat = {ghat:+.4f}   ({M.DATA}, {len(res['iid']['samples'])} resamples kept)\n")
    print(f"{'scheme':>34} | {'2.5%':>7} {'97.5%':>7} | {'width':>6} | vs i.i.d.")
    print("-" * 78)
    lbl = {"iid":  "i.i.d. transitions (current CI)",
           "block":"moving-block (temporal autocorr.)",
           "subj": "subject resample (5 clusters)",
           "hier": "HIERARCHICAL subject+block (honest)"}
    w_iid = res["iid"]["width"]
    for k in ("iid", "block", "subj", "hier"):
        r = res[k]
        print(f"{lbl[k]:>34} | {r['lo']:>+7.3f} {r['hi']:>+7.3f} | {r['width']:>6.3f} | "
              f"x{r['width']/w_iid:4.2f}")
    hier = res["hier"]
    print(f"\nHonest 95% CI for gamma:  [{hier['lo']:+.3f}, {hier['hi']:+.3f}]  "
          f"(i.i.d. CI understates width by x{hier['width']/w_iid:.1f})")

    print("\n=== (B) simulator-based confidence set for gamma_true ===")
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    dist = recovery_distribution(grid, n_rep=40)
    # observed gamma_hat to invert: the cohort estimate, and the conservative
    # upper edge of the cross-cohort empirical band (0.13).
    for g_obs, tag in [(max(ghat, 0.0), "cohort gamma_hat"), (0.13, "empirical band upper edge")]:
        gs, p, bounds = upper_bounds(dist, g_obs)
        txt = "  ".join(f"{int((1-a)*100)}%<={bounds[a]:.2f}" for a in ALPHAS)
        print(f"  g_obs={g_obs:.3f} ({tag}):  one-sided upper bounds  {txt}")

    # headline numbers for the figure/text use the conservative g_obs = 0.13
    gs, p, bounds = upper_bounds(dist, 0.13)
    print("  exceedance p(gamma_true)=P(ghat<=0.13):")
    print("   " + "  ".join(f"{g:.1f}:{pp:.2f}" for g, pp in zip(gs, p)))

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.6))
    # panel a: bootstrap CIs
    order = ["iid", "block", "subj", "hier"]
    col = {"iid": "#7f7f7f", "block": "#0072B2", "subj": "#D55E00", "hier": "#1b9e77"}
    for i, k in enumerate(order):
        r = res[k]
        ax[0].plot([r["lo"], r["hi"]], [i, i], color=col[k], lw=6, solid_capstyle="round",
                   alpha=0.85)
        ax[0].plot(r["med"], i, "o", color="white", ms=7, mec=col[k], mew=2, zorder=5)
    ax[0].axvline(ghat, color="k", ls="--", lw=1, label=f"full-sample $\\hat\\gamma={ghat:+.3f}$")
    ax[0].axvline(0, color="grey", ls=":", lw=0.8)
    ax[0].set_yticks(range(len(order)))
    ax[0].set_yticklabels(["i.i.d.\ntransitions", "moving\nblock", "subject\n(5 clusters)",
                           "hierarchical\n(honest)"], fontsize=9)
    ax[0].set_xlabel(r"reciprocity  $\gamma$  (95% bootstrap CI)")
    ax[0].set_title("a  Clustering widens the honest CI", loc="left", fontsize=12)
    ax[0].legend(loc="lower right", fontsize=8, frameon=False)
    ax[0].set_ylim(-0.6, len(order) - 0.4)

    # panel b: test-inversion exceedance curve p(gamma_true)=P(ghat<=g_obs).
    # gamma_true is rejected (from above) where p < alpha; the crossing is the bound.
    gf = np.linspace(gs[0], gs[-1], 200)
    pf = np.interp(gf, gs, p)
    ax[1].plot(gf, pf, "-", color="#5e4fa2", lw=2, zorder=4)
    ax[1].plot(gs, p, "o", color="#5e4fa2", ms=5, zorder=5)
    acol = {0.05: "#D55E00", 0.16: "#1b9e77"}
    for al in (0.16, 0.05):
        b = bounds[al]
        ax[1].axhline(al, color=acol[al], ls=":", lw=1)
        if gs[0] < b < gs[-1]:
            ax[1].plot([b, b], [0, al], color=acol[al], ls="--", lw=1.4)
            ax[1].plot(b, al, "v", color=acol[al], ms=9, zorder=6)
        lvl = int((1 - al) * 100)
        ax[1].text(0.815, al + 0.012, f"{lvl}%", color=acol[al], fontsize=8, ha="right")
    ax[1].axvspan(-0.02, bounds[0.16], color="#1b9e77", alpha=0.08)
    ax[1].set_xlabel(r"true reciprocity  $\gamma_{\rm true}$")
    ax[1].set_ylabel(r"$P(\hat\gamma\leq\gamma_{\rm obs}\mid\gamma_{\rm true})$,  $\gamma_{\rm obs}=0.13$")
    ax[1].set_title("b  Test-inversion confidence set for $\\gamma_{\\rm true}$", loc="left", fontsize=12)
    ax[1].annotate(fr"$\gamma_{{\rm true}}\!\lesssim\!{bounds[0.16]:.2f}$ (1$\sigma$);"
                   "\nweak at 95%: broad,\noverlapping recovery",
                   xy=(bounds[0.16], 0.16), xytext=(0.34, 0.55), fontsize=8, color="#0a6b4e",
                   arrowprops=dict(arrowstyle="->", color="#0a6b4e"))
    ax[1].set_xlim(-0.05, 0.85); ax[1].set_ylim(0, 1.02)
    for a in ax:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_gamma_confidence.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_gamma_confidence.png written")
    np.savez(f"{OUT}/gamma_confidence.npz",
             ghat=ghat, ci_iid=[res['iid']['lo'], res['iid']['hi']],
             ci_hier=[hier['lo'], hier['hi']],
             grid=gs, p_exceed=p, bound95=bounds[0.05])


if __name__ == "__main__":
    main()
