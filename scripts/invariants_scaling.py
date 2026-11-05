"""
What a perfectly symmetrizable community looks like after inference, by size and noise
======================================================================================
(BIB-26-2181 revision, Reviewer 1 point 6.)

The three rescaling-invariant diagnostics (f_one, f_sign, r_cycle) vanish exactly on a
diagonally symmetrizable matrix, but an ESTIMATED matrix is never exact. A reader applying the
protocol to their own community needs the values a symmetrizable truth produces after the
estimator, at their community size S and noise level. This script tabulates them.

Ground truth (EV3-templated, so S = 12 sits in the regime of the real data):
  * species profiles (r_i, a_ii, log d_i) are resampled with replacement from the diagonally
    symmetrizable fit to the Venturelli EV3 trajectories (lyapunov_descent.fit_symmetrizable);
  * the symmetrised interaction matrix M = D^{1/2} J D^{-1/2} gets a symmetric Gaussian
    off-diagonal on a random support of density c, scaled so its spectral scale
    sigma * sqrt(S c) equals the EV3 fit's (0.75); J_ij = M_ij sqrt(d_j / d_i), so DJ is
    symmetric and the truth is exactly symmetrizable (f_one = f_sign = r_cycle = 0);
  * the whole matrix is multiplied by S/12, which keeps sum_j J_ij x_j of the same order at
    every S when relative abundances are ~1/S (same per-species interaction regime);
  * a draw is kept only if the symmetric M is negative definite (stable truth).

Data, with the EV3 design generalised to S: the full community plus all S single-species
dropouts, 7 observations each, Dirichlet(1) initial composition over present species, absent
species held at a 1e-4 background. Dynamics: the project's composition-preserving simulator
(inference_aware_null_compositional.simulate_compositional, generalised to S): log-step
x <- x exp(r + J x + sigma_p xi), renormalised to unit total, dt = 1 per observation.
Observation noise: each RECORDED composition is multiplied by exp(sigma_o eta) and renormalised
(the state is not affected).

Inference: the identical pipeline (constrained_vs_free_glv.transitions and fit_free,
generalised to S): one-step log-ratio growth, pseudocount 1e-3, species masked below 5e-3,
ridge lambda = 0.5 on off-diagonal J, a species row left at zero if it has fewer than S
equations. Diagnostics: symmetrizability_invariants.invariants at support threshold 0 (primary)
and 0.1 x median |J| (sensitivity).

Grid: S in {8, 12, 25, 50}; support density c in {1.0, 0.3}; process noise sigma_p in
{0.1, 0.45} (0.45 = the one-step residual scale of the EV3 symmetrizable fit, used by the
manuscript's inference-aware null); observation noise sigma_o in {0, 0.1, 0.3}; R replicates per
cell. Anchor: S = 12 with the EV3 fit itself as truth and the real EV3 design reproduces the
manuscript's inference-aware null regime.

Run:  OMP_NUM_THREADS=1 python3 scripts/invariants_scaling.py [--R 100] [--workers 3]
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = "figures"
EPS = 1e-3       # pseudocount, as constrained_vs_free_glv.EPS
ABMIN = 5e-3     # as constrained_vs_free_glv.ABMIN
LAM = 0.5        # as constrained_vs_free_glv.LAM
CAP = 8.0        # as inference_aware_null.CAP
T_OBS = 7
BG = 1e-4
SIZES = [8, 12, 25, 50]
DENS = [1.0, 0.3]
SIG_P = [0.1, 0.45]
SIG_O = [0.0, 0.1, 0.3]
SPEC = 0.75      # spectral scale of the EV3 symmetrizable fit's off-diagonal (std * sqrt(12))


# ------------------------------------------------------------------ the pipeline, any S
def transitions(comms):
    T = []
    for X in comms:
        for a, b in zip(X[:-1], X[1:]):
            g = np.log(b + EPS) - np.log(a + EPS)
            T.append((a, g, (a > ABMIN) & (b > ABMIN)))
    return T


def fit_free(trans, S, lam=LAM):
    """constrained_vs_free_glv.fit_free with S as a parameter (same equations, same ridge)."""
    r = np.zeros(S); J = np.zeros((S, S)); skipped = 0
    for i in range(S):
        rows, ys = [], []
        for a, g, m in trans:
            if m[i]:
                rows.append(np.concatenate([[1.0], a])); ys.append(g[i])
        if len(rows) < S:
            skipped += 1
            continue
        Z = np.array(rows); y = np.array(ys)
        P = np.full(S + 1, lam); P[0] = 0.0; P[1 + i] = 0.0
        beta = np.linalg.solve(Z.T @ Z + np.diag(P) + 1e-6 * np.eye(S + 1), Z.T @ y)
        r[i] = beta[0]; J[i] = beta[1:]
    return r, J, skipped


def simulate(r, J, designs, x0s, sig_p, sig_o, rng, absolute=False):
    """simulate_compositional generalised to S, plus observation noise on the record.

    absolute=True is the contrast run: the same log-step gLV with NO renormalisation of the
    state or of the record, i.e. data in the coordinates the gLV is written in.
    """
    S = len(r); out = []
    for present, x0 in zip(designs, x0s):
        x = x0.copy(); rec = np.zeros((T_OBS, S))
        for t in range(T_OBS):
            if t > 0:
                x = np.clip(np.exp(np.log(x + EPS) + r + J @ x + sig_p * rng.standard_normal(S)), 0, CAP)
                x[~present] = BG
                s = x.sum()
                if s > 0 and not absolute:
                    x = x / s
            y = x * np.exp(sig_o * rng.standard_normal(S)) if sig_o > 0 else x.copy()
            y[~present] = BG
            rec[t] = y if absolute else y / y.sum()
        out.append(rec)
    return out


def design(S, rng):
    designs, x0s = [], []
    for k in range(S + 1):                         # k = S: the full community
        present = np.ones(S, bool)
        if k < S:
            present[k] = False
        x0 = np.full(S, BG)
        w = rng.dirichlet(np.ones(present.sum()))
        x0[present] = w * (1 - BG * (~present).sum())
        designs.append(present); x0s.append(x0)
    return designs, x0s


def template():
    import warnings
    warnings.filterwarnings("ignore"); np.seterr(all="ignore")
    from constrained_vs_free_glv import load_communities, transitions as tr12
    from lyapunov_descent import fit_symmetrizable
    r, J, d = fit_symmetrizable(tr12(load_communities()))
    return r, np.diag(J).copy(), np.log(d), (r, J, d)


def truth(S, c, prof, rng, tries=200, uscale=1.0):
    r0, a0, u0 = prof
    for _ in range(tries):
        k = rng.integers(0, len(r0), S)
        r, a, u = r0[k].copy(), a0[k].copy(), u0[k].copy() * uscale
        d = np.exp(u - u.mean())
        iu = np.triu_indices(S, 1)
        on = rng.random(len(iu[0])) < c
        Moff = np.zeros((S, S))
        Moff[iu[0][on], iu[1][on]] = rng.standard_normal(on.sum()) * SPEC / np.sqrt(S * c)
        Moff = Moff + Moff.T
        M = Moff + np.diag(a)
        if np.max(np.linalg.eigvalsh(M)) >= 0:
            continue
        sq = np.sqrt(d)
        J = (M / sq[:, None]) * sq[None, :]          # J_ij = M_ij sqrt(d_j/d_i)
        J = J * (S / 12.0)
        return r, J, d
    raise RuntimeError("no stable truth found")


def run_cell(args):
    S, c, sp, so, R, seed, prof = args[:7]
    absolute = args[7] if len(args) > 7 else False
    lam = args[8] if len(args) > 8 else LAM
    uscale = args[9] if len(args) > 9 else 1.0
    from symmetrizability_invariants import invariants
    import warnings
    warnings.filterwarnings("ignore"); np.seterr(all="ignore")
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(R):
        r, J, d = truth(S, c, prof, rng, uscale=uscale)
        # exactness check on the truth
        f1t, fst, rct, _ = invariants(J)
        designs, x0s = design(S, rng)
        sim = simulate(r, J, designs, x0s, sp, so, rng, absolute=absolute)
        _, Jh, skipped = fit_free(transitions(sim), S, lam=lam)
        f1, fs, rc, E = invariants(Jh)
        f1b, fsb, rcb, Eb = invariants(Jh, thresh=0.1)
        res.append((f1, fs, rc, f1b, fsb, rcb, skipped, f1t, fst, rct if np.isfinite(rct) else 0.0))
    return (S, c, sp, so), np.array(res, dtype=float)


def anchor(R, seed, fit, sig_p=0.45, sig_o=0.0):
    """S = 12, EV3 symmetrizable fit as truth, real EV3 design and initial conditions."""
    from symmetrizability_invariants import invariants
    from constrained_vs_free_glv import load_communities
    import warnings
    warnings.filterwarnings("ignore"); np.seterr(all="ignore")
    r, J, d = fit
    comms = load_communities()
    designs = [X.mean(0) > 1e-3 for _, X in comms]
    x0s = [X[0].copy() for _, X in comms]
    rng = np.random.default_rng(seed); out = []
    for _ in range(R):
        sim = simulate(r, J, designs, x0s, sig_p, sig_o, rng)
        _, Jh, sk = fit_free(transitions(sim), 12)
        f1, fs, rc, _ = invariants(Jh)
        out.append((f1, fs, rc))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--replot", action="store_true", help="redraw the figure from the caches")
    ap.add_argument("--anchor", action="store_true",
                    help="real EV3 design, EV3 fit as truth, across the noise grid")
    ap.add_argument("--spread", action="store_true",
                    help="sweep the spread of the true symmetrizer, std(log d)")
    ap.add_argument("--control", action="store_true",
                    help="positive control: well-conditioned design, invariants must vanish")
    ap.add_argument("--absolute", action="store_true",
                    help="contrast run without compositional closure (c=1, sig_o in {0, 0.3})")
    a = ap.parse_args()
    if a.absolute:
        return main_absolute(a)
    if a.control:
        positive_control(a.R)
        return
    if a.spread:
        return main_spread(a)
    if a.replot:
        return plot_figure()
    if a.anchor:
        return main_anchor_noise(a)
    import os
    os.makedirs(OUT, exist_ok=True)
    r0, a0, u0, fit = template()
    prof = (r0, a0, u0)
    print(f"template: EV3 symmetrizable fit, std(log d) = {u0.std():.2f}, "
          f"a_ii in [{a0.min():.1f}, {a0.max():.1f}], r in [{r0.min():.2f}, {r0.max():.2f}]")
    cells = list(itertools.product(SIZES, DENS, SIG_P, SIG_O))
    jobs = [(S, c, sp, so, a.R, 10_000 + 97 * k, prof) for k, (S, c, sp, so) in enumerate(cells)]
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        results = dict(pool.map(run_cell, jobs))

    q = lambda x: (np.nanmean(x), np.nanpercentile(x, 5), np.nanpercentile(x, 95))
    print("Invariant diagnostics of the FREE ridge fit when the TRUTH is exactly symmetrizable")
    print("(mean [5%, 95%] over replicates; support threshold 0; f_one > 0 only from skipped rows)")
    print(f"{'S':>3} {'c':>4} {'sig_p':>6} {'sig_o':>6} | {'f_one':^22} | {'f_sign':^22} | "
          f"{'r_cycle':^22} | {'rows skipped':>12} | truth max(f,r)")
    rows = []
    for (S, c, sp, so) in cells:
        X = results[(S, c, sp, so)]
        f1, fs, rc = q(X[:, 0]), q(X[:, 1]), q(X[:, 2])
        sk = X[:, 6].mean()
        tmax = np.nanmax(X[:, 7:10])
        rows.append((S, c, sp, so, *f1, *fs, *rc, sk))
        print(f"{S:>3} {c:>4.1f} {sp:>6.2f} {so:>6.2f} | {f1[0]:.2f} [{f1[1]:.2f},{f1[2]:.2f}]"
              f"       | {fs[0]:.2f} [{fs[1]:.2f},{fs[2]:.2f}]       | {rc[0]:.2f} [{rc[1]:.2f},{rc[2]:.2f}]"
              f"       | {sk:>12.2f} | {tmax:.1e}")
    print("\nSensitivity: support threshold 0.1 x median |J| (coefficients below treated as zero)")
    print(f"{'S':>3} {'c':>4} {'sig_p':>6} {'sig_o':>6} | {'f_one':^22} | {'f_sign':^22} | {'r_cycle':^22}")
    for (S, c, sp, so) in cells:
        X = results[(S, c, sp, so)]
        f1, fs, rc = q(X[:, 3]), q(X[:, 4]), q(X[:, 5])
        print(f"{S:>3} {c:>4.1f} {sp:>6.2f} {so:>6.2f} | {f1[0]:.2f} [{f1[1]:.2f},{f1[2]:.2f}]"
              f"       | {fs[0]:.2f} [{fs[1]:.2f},{fs[2]:.2f}]       | {rc[0]:.2f} [{rc[1]:.2f},{rc[2]:.2f}]")

    A = anchor(a.R, 777, fit)
    print(f"\nAnchor (S=12, EV3 symmetrizable fit as truth, real EV3 design, sig_p=0.45, sig_o=0): "
          f"f_sign {np.mean(A[:,1]):.2f} [{np.percentile(A[:,1],5):.2f},{np.percentile(A[:,1],95):.2f}], "
          f"r_cycle {np.nanmean(A[:,2]):.2f} [{np.nanpercentile(A[:,2],5):.2f},{np.nanpercentile(A[:,2],95):.2f}]")
    print("Observed, for reference: free ridge re-fit of EV3 (S=12) f_sign 0.55, r_cycle 1.5; "
          "published Venturelli (S=12) f_one 0.30, f_sign 0.44, r_cycle 2.2; Clark (S=25) "
          "f_one 0.00, f_sign 0.56, r_cycle 1.0")

    plot_figure()
    np.savez(f"{OUT}/invariants_scaling.npz",
             cells=np.array(cells), stats=np.array(rows),
             raw=np.array([results[k] for k in cells], dtype=object), anchor=A,
             sizes=np.array(SIZES), dens=np.array(DENS), sig_p=np.array(SIG_P), sig_o=np.array(SIG_O))
    print(f"[npz] {OUT}/invariants_scaling.npz")
    plot_figure()


def plot_figure():
    """Redraw from the caches: main grid, absolute-coordinates contrast, positive control."""
    import os
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, NullLocator
    M = np.load(f"{OUT}/invariants_scaling.npz", allow_pickle=True)
    cells = [tuple(c) for c in M["cells"]]
    raw = dict(zip(cells, M["raw"]))
    absf = f"{OUT}/invariants_scaling_absolute.npz"
    ctlf = f"{OUT}/invariants_scaling_control.npz"
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    colors = {0.0: "#0072B2", 0.1: "#E69F00", 0.3: "#D55E00"}
    c = 1.0
    for k, (ax, lab) in enumerate(zip(axes, [r"$f_{\rm one}$", r"$f_{\rm sign}$", r"$r_{\rm cycle}$"])):
        X = [raw[(S, c, 0.45, 0.0)][:, k] for S in SIZES]
        ax.fill_between(SIZES, [np.nanpercentile(x, 5) for x in X], [np.nanpercentile(x, 95) for x in X],
                        color="#0072B2", alpha=0.15, lw=0,
                        label=r"5-95%, compositional, $\sigma_p=0.45$, $\sigma_o=0$")
        for sp in SIG_P:
            for so in SIG_O:
                m = [np.nanmean(raw[(S, c, sp, so)][:, k]) for S in SIZES]
                ax.plot(SIZES, m, ls="-" if sp == 0.45 else "--", color=colors[so], marker="o", ms=3.5,
                        lw=1.3, label=rf"mean, $\sigma_p={sp}$, $\sigma_o={so}$")
        if os.path.exists(absf):
            A = np.load(absf, allow_pickle=True)
            acells = [tuple(x) for x in A["cells"]]
            araw = dict(zip(acells, A["raw"]))
            if all((S, 1.0, 0.0, 0.0) in araw for S in SIZES):
                ax.plot(SIZES, [np.nanmean(araw[(S, 1.0, 0.0, 0.0)][:, k]) for S in SIZES], color="k",
                        ls="-.", lw=1.6, marker="s", ms=4,
                        label="mean, absolute coordinates, no noise")
        if os.path.exists(ctlf) and k > 0:
            C = np.load(ctlf, allow_pickle=True)
            key = {kk: i for i, kk in enumerate(C["keys"])}
            arr = C["fsign"] if k == 1 else C["rcycle"]
            ax.plot(SIZES, [np.nanmean(arr[key[f"{S}|0.0"]]) for S in SIZES], color="#777777", ls=":",
                    lw=2, label="positive control: well-conditioned design, no noise")
        if k == 0:
            ax.plot(SIZES, [0, 0, 0, 0], color="#777777", ls=":", lw=2)
        # references: only the EV3 re-fit came from this pipeline
        ev3 = {0: 0.0, 1: 0.55, 2: 1.5}[k]
        ax.scatter([12], [ev3], marker="*", s=230, color="k", zorder=6,
                   label="observed, free ridge re-fit of EV3 (this pipeline)")
        pub = {0: [(12, 0.30), (25, 0.00)], 1: [(12, 0.44), (25, 0.56)], 2: [(12, 2.2), (25, 1.0)]}[k]
        ax.scatter([p[0] for p in pub], [p[1] for p in pub], marker="D", s=60, facecolor="none",
                   edgecolor="#444444", lw=1.4, zorder=6,
                   label="published Venturelli (S=12) and Clark (S=25), other estimators")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(SIZES))
        ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in SIZES]))
        ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("community size S")
        ax.set_ylabel(lab)
        ax.set_title(f"{'abc'[k]}  {lab}", loc="left")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.suptitle("An exactly symmetrizable truth after the identical inference pipeline, "
                 "EV3-like design (full community + all single dropouts, 7 points each)",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_invariants_scaling.png", dpi=200, bbox_inches="tight")
    print(f"[fig] {OUT}/fig_invariants_scaling.png")


def positive_control(R=100, seed=90_000):
    """Code check: same truths, same estimator and invariants, but a WELL-CONDITIONED design.

    Regressor states are drawn independently (uniform 0.01-0.5 per species), so they are not
    collinear, and the response is exactly g = r + J x (+ optional small noise), with no
    pseudocount and no masking. If the estimator and the invariant code are right, f_sign and
    r_cycle must go to ~0 here; anything left in the EV3-like runs is then the design.
    """
    from symmetrizability_invariants import invariants
    r0, a0, u0, _ = template()
    prof = (r0, a0, u0)
    rng = np.random.default_rng(seed)
    out = {}
    for S in SIZES:
        for noise in (0.0, 0.1):
            res = []
            for _ in range(R):
                r, J, d = truth(S, 1.0, prof, rng)
                n = 6 * (S + 1)                     # as many transitions as the EV3-like design
                Xa = rng.uniform(0.01, 0.5, size=(n, S))
                G = r[None, :] + Xa @ J.T + noise * rng.standard_normal((n, S))
                trans = [(Xa[k], G[k], np.ones(S, bool)) for k in range(n)]
                _, Jh, _ = fit_free(trans, S, lam=0.0005)
                f1, fs, rc, _ = invariants(Jh)
                res.append((fs, rc))
            out[(S, noise)] = np.array(res)
    np.savez(f"{OUT}/invariants_scaling_control.npz",
             keys=np.array([f"{S}|{n}" for (S, n) in out]),
             fsign=np.array([out[k][:, 0] for k in out]), rcycle=np.array([out[k][:, 1] for k in out]))
    print("POSITIVE CONTROL: well-conditioned design, exact gLV response, lambda = 5e-4")
    print(f"{'S':>3} {'noise':>6} | {'f_sign':^20} | {'r_cycle':^20}")
    for (S, noise), X in out.items():
        print(f"{S:>3} {noise:>6.2f} | {np.mean(X[:,0]):.2f} [{np.percentile(X[:,0],5):.2f},{np.percentile(X[:,0],95):.2f}]"
              f"     | {np.nanmean(X[:,1]):.3f} [{np.nanpercentile(X[:,1],5):.3f},{np.nanpercentile(X[:,1],95):.3f}]")
    return out


def main_anchor_noise(a):
    """The real EV3 design and the EV3 symmetrizable fit as truth, across the noise grid."""
    _, _, _, fit = template()
    print("ANCHOR, real EV3 design (S = 12), EV3 symmetrizable fit as truth, noise swept")
    print(f"{'sig_p':>6} {'sig_o':>6} | {'f_sign':^20} | {'r_cycle':^20}")
    rows = []
    for k, (sp, so) in enumerate(itertools.product([0.0] + SIG_P, SIG_O)):
        A = anchor(a.R, 600 + k, fit, sp, so)
        rows.append((sp, so, np.mean(A[:, 1]), np.percentile(A[:, 1], 5), np.percentile(A[:, 1], 95),
                     np.nanmean(A[:, 2]), np.nanpercentile(A[:, 2], 5), np.nanpercentile(A[:, 2], 95)))
        print(f"{sp:>6.2f} {so:>6.2f} | {rows[-1][2]:.2f} [{rows[-1][3]:.2f},{rows[-1][4]:.2f}]     | "
              f"{rows[-1][5]:.2f} [{rows[-1][6]:.2f},{rows[-1][7]:.2f}]")
    np.savez(f"{OUT}/invariants_scaling_anchor.npz", rows=np.array(rows))
    print(f"[npz] {OUT}/invariants_scaling_anchor.npz")


def main_spread(a):
    """EV3-like compositional design, sig_p = 0.45, sig_o = 0, c = 1: symmetrizer spread swept."""
    r0, a0, u0, _ = template()
    prof = (r0, a0, u0)
    targets = [0.0, 0.5, 1.0, 2.0, float(u0.std())]
    jobs = []
    for S in (12, 25):
        for k, t in enumerate(targets):
            jobs.append((S, 1.0, 0.45, 0.0, a.R, 80_000 + 31 * S + k, prof, False, LAM,
                         t / u0.std()))
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        res = pool.map(run_cell, jobs)
    print("SPREAD: EV3-like compositional design, sig_p = 0.45, sig_o = 0, c = 1; std(log d) swept")
    print(f"{'S':>3} {'std log d':>10} | {'f_sign':^20} | {'r_cycle':^20}")
    rows = []
    for job, (_, X) in zip(jobs, res):
        S, t = job[0], job[9] * u0.std()
        rows.append((S, t, np.mean(X[:, 1]), np.percentile(X[:, 1], 5), np.percentile(X[:, 1], 95),
                     np.nanmean(X[:, 2]), np.nanpercentile(X[:, 2], 5), np.nanpercentile(X[:, 2], 95)))
        print(f"{S:>3} {t:>10.2f} | {rows[-1][2]:.2f} [{rows[-1][3]:.2f},{rows[-1][4]:.2f}]     | "
              f"{rows[-1][5]:.2f} [{rows[-1][6]:.2f},{rows[-1][7]:.2f}]")
    np.savez(f"{OUT}/invariants_scaling_spread.npz", rows=np.array(rows))
    print(f"[npz] {OUT}/invariants_scaling_spread.npz")


def main_absolute(a):
    """Same truths, same estimator, no renormalisation: isolates the effect of closure."""
    import os
    os.makedirs(OUT, exist_ok=True)
    r0, a0, u0, _ = template()
    prof = (r0, a0, u0)
    cells = list(itertools.product(SIZES, [1.0], [0.0] + SIG_P, [0.0, 0.3]))
    jobs = [(S, c, sp, so, a.R, 50_000 + 97 * k, prof, True) for k, (S, c, sp, so) in enumerate(cells)]
    # mechanism check: noiseless absolute data at S = 12, ridge strength swept
    lams = [0.5, 0.05, 0.005, 0.0005]
    ljobs = [(12, 1.0, 0.0, 0.0, a.R, 70_000 + k, prof, True, lam) for k, lam in enumerate(lams)]
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        results = dict(pool.map(run_cell, jobs))
        lres = pool.map(run_cell, ljobs)
    q = lambda x: (np.nanmean(x), np.nanpercentile(x, 5), np.nanpercentile(x, 95))
    print("CONTRAST: absolute coordinates, no compositional closure (c = 1, threshold 0)")
    print(f"{'S':>3} {'sig_p':>6} {'sig_o':>6} | {'f_one':^20} | {'f_sign':^20} | {'r_cycle':^20} | skipped")
    rows = []
    for (S, c, sp, so) in cells:
        X = results[(S, c, sp, so)]
        f1, fs, rc = q(X[:, 0]), q(X[:, 1]), q(X[:, 2])
        rows.append((S, c, sp, so, *f1, *fs, *rc, X[:, 6].mean()))
        print(f"{S:>3} {sp:>6.2f} {so:>6.2f} | {f1[0]:.2f} [{f1[1]:.2f},{f1[2]:.2f}]     | "
              f"{fs[0]:.2f} [{fs[1]:.2f},{fs[2]:.2f}]     | {rc[0]:.2f} [{rc[1]:.2f},{rc[2]:.2f}]     | {X[:, 6].mean():.2f}")
    print("\nMECHANISM: S = 12, absolute coordinates, NO noise (sig_p = sig_o = 0), ridge lambda swept")
    print(f"{'lambda':>8} | {'f_sign':^20} | {'r_cycle':^20}")
    lstats = []
    for lam, (_, X) in zip(lams, lres):
        fs, rc = q(X[:, 1]), q(X[:, 2])
        lstats.append((lam, *fs, *rc))
        print(f"{lam:>8.4f} | {fs[0]:.2f} [{fs[1]:.2f},{fs[2]:.2f}]     | {rc[0]:.2f} [{rc[1]:.2f},{rc[2]:.2f}]")
    np.savez(f"{OUT}/invariants_scaling_absolute.npz", cells=np.array(cells), stats=np.array(rows),
             raw=np.array([results[k] for k in cells], dtype=object), lam_stats=np.array(lstats))
    print(f"[npz] {OUT}/invariants_scaling_absolute.npz")


if __name__ == "__main__":
    main()
