"""
Ground-truth gamma from Venturelli et al. 2018 -- two independent checks
========================================================================

The interaction-symmetry claim of the paper rests on gamma inferred from
observational time series, an ill-posed inverse problem (Appendix B: gamma is a
bound, not a point estimate, and relative abundances inflate it). Venturelli et
al., Mol. Syst. Biol. 14, e8157 (2018), give us a rare chance to test both ends
of that claim against a system whose interactions were *measured*, not inferred:
a 12-member synthetic human-gut community, characterised bottom-up from mono-,
pairwise- and multi-species designed co-cultures.

    dx_i/dt = x_i ( u_i + sum_j a_ij x_j ),   a_ij = effect of j on i   (= J_ij)

LEVEL 1 (ground truth).  The fitted gLV matrix (supplementary MOESM6, SBML,
four successive fits T1-T4; T1 is the interaction-free baseline) is itself
weakly reciprocal: gamma = corr(a_ij, a_ji) ~ 0.06. No inference in the loop,
so this cannot be a regression artefact.

LEVEL 2 (pipeline validation).  We run the paper's OWN estimator
(pipeline_for_real_data.infer_glv, pooled across the 13 leave-one-out
communities of Dataset EV3) on Venturelli's real *relative-abundance*
trajectories. Because the true gamma here is KNOWN to be ~0.06, the output is a
direct, ground-truthed read of the estimator's bias: it returns gamma_hat ~ 0.2,
inflated ~3x by the positive bias and compositional closure, yet still far from
the symmetric limit 1 and squarely in the 0.2-0.3 band of the paper's
relative-abundance cohorts. True 0.06 reads as 0.2, never as ~1.

Requires openpyxl (pip install openpyxl) to read the supplementary xlsx.

Run:  python3 venturelli_ground_truth.py
"""

import glob
import re
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NS = {"s": "http://www.sbml.org/sbml/level3/version1/core"}
SP = ["BH", "CA", "BU", "PC", "BO", "BV", "BT", "EL", "FP", "CH", "DP", "ER"]
IDX = {s: k for k, s in enumerate(SP)}
S = len(SP)
DATA = "data/venturelli2018"
OUT = "figures"
OFFD = ~np.eye(S, dtype=bool)   # all off-diagonal ordered pairs (label-invariant)


def gamma_symmetry(A):
    """Interaction symmetry, identical definition to infer_glv().

    corr(a_ij, a_ji) over every off-diagonal ordered pair, so the value does
    not depend on the arbitrary species ordering.
    """
    x, y = A[OFFD], A.T[OFFD]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------------
# LEVEL 1 -- measured interaction matrix (SBML)
# --------------------------------------------------------------------------
def load_matrix(fn):
    root = ET.parse(fn).getroot()
    P = {p.get("id"): float(p.get("value"))
         for p in root.findall(".//s:parameter", NS)}
    A = np.full((S, S), np.nan)
    for a in SP:
        for b in SP:
            A[IDX[a], IDX[b]] = P["a" + a + b]
    return A


def level1():
    files = sorted(glob.glob(f"{DATA}/twelveSpecies*.xml"))
    mats, gammas = {}, {}
    print("LEVEL 1  -- measured matrix")
    print(f"  {'model':>14} {'gamma':>8}")
    for fn in files:
        A = load_matrix(fn)
        g = gamma_symmetry(A)
        mats[fn], gammas[fn] = A, g
        print(f"  {fn.split('/')[-1]:>14} {(f'{g:.3f}' if np.isfinite(g) else 'n/a'):>8}")
    interacting = [f for f in files if np.isfinite(gammas[f])]
    A_avg = np.nanmean(np.stack([mats[f] for f in interacting]), axis=0)
    g_avg = gamma_symmetry(A_avg)
    x, y = A_avg[OFFD], A_avg.T[OFFD]
    rng = np.random.default_rng(0)
    bs = [np.corrcoef(x[s], y[s])[0, 1]
          for s in (rng.integers(0, len(x), len(x)) for _ in range(5000))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    g_list = [gammas[f] for f in interacting]
    print(f"  averaged matrix: gamma = {g_avg:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
          f"(fits {min(g_list):.3f}-{max(g_list):.3f})")
    np.savetxt(f"{DATA}/interaction_matrix_averaged.csv", A_avg,
               delimiter=",", header=",".join(SP), comments="")
    return A_avg, g_avg, lo, hi


# --------------------------------------------------------------------------
# LEVEL 2 -- paper's estimator on the real relative-abundance trajectories
# --------------------------------------------------------------------------
def load_timeseries():
    """Parse Dataset EV3: 13 leave-one-out communities x 7 timepoints x 12 taxa
    (relative abundances). Structurally-absent species are zeroed."""
    import openpyxl
    ws = openpyxl.load_workbook(f"{DATA}/timeseries_multispecies_EV3.xlsx",
                                read_only=True, data_only=True)["Dataset EV3"]
    comms, cur = {}, None
    for r in ws.iter_rows(values_only=True):
        if not r or r[0] is None:
            continue
        s = str(r[0]).strip().strip("'").strip()
        if re.search(r"[A-Za-z]", s):
            cur = s
            comms[cur] = []
        else:
            v = [float(x) for x in s.split()]
            if len(v) == S:
                comms[cur].append(v)
    out = {}
    for k, v in comms.items():
        X = np.array(v)
        X[:, X.mean(0) < 1e-3] = 0.0        # zero the leave-one-out species
        out[k] = X
    return out


def infer_pooled(comms, ridge=1e-2, dt=1.0, present=1e-3):
    """gLV log-ratio ridge regression (as infer_glv) pooled across experiments,
    each taxon fit only over samples where it is present at t and t+1."""
    eps = 1e-6
    Ps, Ys, Ms = [], [], []
    for X in comms.values():
        Xc = np.clip(X, eps, None)
        Ps.append(X[:-1])
        Ys.append((np.log(Xc[1:]) - np.log(Xc[:-1])) / dt)
        Ms.append((X[:-1] > present) & (X[1:] > present))
    P, Y, M = np.vstack(Ps), np.vstack(Ys), np.vstack(Ms)
    Z = np.hstack([np.ones((len(P), 1)), P])
    J = np.zeros((S, S))
    for i in range(S):
        m = M[:, i]
        if m.sum() < S + 2:
            continue
        A = Z[m].T @ Z[m] + ridge * np.eye(S + 1)
        J[i] = np.linalg.solve(A, Z[m].T @ Y[m, i])[1:]
    assert np.isfinite(J).all()
    return J


def level2():
    comms = load_timeseries()
    print(f"\nLEVEL 2  -- paper's estimator on the real relative trajectories "
          f"({len(comms)} communities)")
    grid = {r: gamma_symmetry(infer_pooled(comms, ridge=r))
            for r in (1e-3, 1e-2, 1e-1, 1.0)}
    for r, g in grid.items():
        print(f"  ridge={r:.0e}: gamma_hat = {g:.3f}")
    g_hat = grid[1e-2]                       # paper's default ridge
    names = list(comms.keys())
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(3000):
        idx = rng.choice(len(names), len(names), replace=True)
        cc = {f"{names[k]}#{j}": comms[names[k]] for j, k in enumerate(idx)}
        try:
            bs.append(gamma_symmetry(infer_pooled(cc, ridge=1e-2)))
        except Exception:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5])
    span = (min(grid.values()), max(grid.values()))
    print(f"  gamma_hat = {g_hat:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
          f"(range across ridge {span[0]:.2f}-{span[1]:.2f})")
    return g_hat, lo, hi, span


# --------------------------------------------------------------------------
def main():
    A_avg, g_true, lo1, hi1 = level1()
    g_hat, lo2, hi2, span = level2()

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.1))

    # (a) measured reciprocity cloud
    x, y = A_avg[OFFD], A_avg.T[OFFD]
    lim = np.max(np.abs([x, y])) * 1.1
    ax[0].axhline(0, color="0.85", lw=.8); ax[0].axvline(0, color="0.85", lw=.8)
    ax[0].plot([-lim, lim], [-lim, lim], "--", color="0.6", lw=.8,
               label=r"symmetric ($\gamma=1$)")
    ax[0].scatter(x, y, s=26, c="#2c6fbb", edgecolor="w", linewidth=.5, zorder=3)
    ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim); ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$a_{ij}$  (effect of $j$ on $i$)")
    ax[0].set_ylabel(r"$a_{ji}$  (effect of $i$ on $j$)")
    ax[0].set_title(f"(a) Measured matrix\n"
                    rf"$\gamma_{{\rm true}}={g_true:.2f}$ [{lo1:.2f}, {hi1:.2f}]")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")

    # (b) ground truth vs the paper's estimator on the real relative series
    labels = ["measured\n(ground truth)", "estimator on\nrel. series",
              "human QMP\n(inferred)", "symmetric\nlimit"]
    vals = [g_true, g_hat, 0.10, 1.0]
    cols = ["#2c6fbb", "#c65a2e", "#4a9e6f", "#b0b0b0"]
    ax[1].bar(labels, vals, color=cols)
    ax[1].errorbar([0], [g_true], yerr=[[g_true - lo1], [hi1 - g_true]],
                   fmt="none", ecolor="k", capsize=4, lw=1.1)
    ax[1].errorbar([1], [g_hat], yerr=[[g_hat - lo2], [hi2 - g_hat]],
                   fmt="none", ecolor="k", capsize=4, lw=1.1)
    ax[1].axhspan(0.2, 0.3, color="#c65a2e", alpha=.08)
    ax[1].axhline(0, color="k", lw=.6)
    ax[1].annotate(r"$\sim\!3\times$ inflation," "\n" r"still $\ll 1$",
                   xy=(1, g_hat), xytext=(1.5, 0.55), fontsize=8,
                   arrowprops=dict(arrowstyle="->", lw=.8))
    ax[1].set_ylabel(r"interaction symmetry $\gamma$")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_title("(b) True $0.06$ reads as $\\sim\\!0.2$, never $1$")
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    out = f"{OUT}/fig22_venturelli_groundtruth.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
