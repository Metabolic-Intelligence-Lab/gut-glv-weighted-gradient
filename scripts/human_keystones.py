"""
Fig. 19 -- Clinical-grade keystones from a HUMAN, absolute-abundance cohort
===========================================================================

The mouse keystone tool (Fig. 16) used a small, noisy, 30-taxon inferred matrix.
Here we redo it on the Vandeputte et al. (2021) human QMP cohort -- 20 subjects,
daily, with flow-cytometry ABSOLUTE abundances -- the cleanest data for the job.

For the top genera we infer the gLV interaction matrix, run the in-silico
knockout (remove a genus, measure how far the rest of the community moves), and
add bootstrap stability (resample transitions, re-infer, re-rank). We then mark
the genera independently established as gut keystones in the literature.

Run (after the Vandeputte .rda is present):  python3 human_keystones.py
"""

import numpy as np
import rdata
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infer_and_place import fit_matrix

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"
S_TAXA = 40
NMAX, EPS = 50.0, 1e-3
LIT = {"Bacteroides", "Blautia", "Akkermansia", "Faecalibacterium", "Prevotella",
       "Roseburia", "Ruminococcus", "Eubacterium", "Bifidobacterium",
       "Parabacteroides", "Alistipes"}


def load():
    o = rdata.read_rda("data/vandeputte/temporalMicrobiome20.rda")["temporalMicrobiome20"]
    C = np.asarray(o.assays.data.listData["counts"], dtype=float)
    names = np.array([str(x).replace("g_", "") for x in
                      np.asarray(o.rowRanges.partitioning.NAMES)])
    cd = o.colData.listData
    day = np.asarray(cd["day"], dtype=float).ravel()
    pid = np.array([str(x) for x in np.asarray(cd["person_id"]).ravel()])
    X = np.nan_to_num(C.T)
    day = np.where(day < -1e9, np.nan, day)
    ok = (~np.isnan(day)) & (X.sum(1) > 0)
    return X[ok], day[ok], pid[ok], names


def design(X, day, pid, top, max_gap=3.0):
    scale = np.median(np.where(X[:, top] > 0, X[:, top], np.nan), 0)
    scale = np.where(np.isnan(scale) | (scale <= 0), 1.0, scale)
    Xs = X[:, top] / scale
    Y, Z, eps = [], [], 1e-3
    for sb in np.unique(pid):
        idx = np.where(pid == sb)[0]; idx = idx[np.argsort(day[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            dt = day[b] - day[a]
            if 0 < dt <= max_gap:
                Y.append((np.log(Xs[b] + eps) - np.log(Xs[a] + eps)) / dt)
                Z.append(np.concatenate([[1.0], Xs[a]]))
    return np.array(Y), np.array(Z), scale


def relax(J, n0, r, T=300):
    def rhs(t, n):
        n = np.clip(n, 0, NMAX); return n * (r - n + J @ n) + EPS
    s = solve_ivp(rhs, (0, T), np.clip(np.nan_to_num(n0), 0, NMAX), method="LSODA",
                  t_eval=[T], rtol=1e-6, atol=1e-8, max_step=2.0)
    return np.clip(np.nan_to_num(s.y[:, -1]), 0, NMAX)


def bc(a, b):
    return np.abs(a - b).sum() / (a.sum() + b.sum() + 1e-9)


def knockout_impact(J):
    S = J.shape[0]; r = np.ones(S)
    nb = relax(J, np.random.default_rng(0).uniform(0.1, 1, S), r)
    imp = np.zeros(S)
    for i in range(S):
        Jk = J.copy(); Jk[i, :] = 0; Jk[:, i] = 0
        n0 = nb.copy(); n0[i] = 0
        nf = relax(Jk, n0, r)
        oth = np.delete(np.arange(S), i)
        imp[i] = bc(nf[oth], nb[oth])
    return imp / (imp.max() + 1e-12)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    X, day, pid, names = load()
    prop = X / X.sum(1, keepdims=True)
    top = np.argsort(prop.mean(0))[::-1][:S_TAXA]
    labels = names[top]
    Y, Z, _ = design(X, day, pid, top)
    print(f"human QMP: {X.shape[0]} samples, {len(np.unique(pid))} subjects, "
          f"{Y.shape[0]} transitions, top {S_TAXA} genera")

    imp = knockout_impact(fit_matrix(Y, Z))

    # bootstrap stability over transitions
    rng = np.random.default_rng(1); B = 25
    boot = np.zeros((B, S_TAXA))
    for b in range(B):
        s = rng.integers(0, Y.shape[0], Y.shape[0])
        boot[b] = knockout_impact(fit_matrix(Y[s], Z[s]))
    lo = np.percentile(boot, 25, 0); hi = np.percentile(boot, 75, 0)

    order = np.argsort(imp)[::-1]
    print("Top human keystone genera (knockout impact, IQR):")
    for i in order[:12]:
        tag = " [literature keystone]" if labels[i] in LIT else ""
        print(f"  {labels[i]:20s} {imp[i]:.2f} [{lo[i]:.2f},{hi[i]:.2f}]{tag}")

    topk = order[:12][::-1]
    err_lo = np.clip(imp[topk] - lo[topk], 0, None)
    err_hi = np.clip(hi[topk] - imp[topk], 0, None)
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    cols = ["#d95f02" if labels[i] in LIT else "#7570b3" for i in topk]
    ax.barh(range(len(topk)), imp[topk], color=cols,
            xerr=[err_lo, err_hi], error_kw=dict(ecolor="black", lw=1))
    ax.set_yticks(range(len(topk)))
    ax.set_yticklabels([labels[i] for i in topk], fontsize=9)
    ax.set_xlabel("keystone impact (in-silico knockout)  +/- bootstrap IQR")
    ax.set_title("Keystone candidate genera from a human absolute-abundance cohort\n"
                 "(Vandeputte 2021 QMP, 40 genera; orange = classic gut keystones)",
                 fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#d95f02", label="classic keystone genus"),
                       Patch(color="#7570b3", label="other high-impact candidate")],
              frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig19_human_keystones.png", dpi=150)
    plt.close(fig)
    n_lit = sum(labels[i] in LIT for i in order[:12])
    print(f"[fig19] written | {n_lit}/12 top keystones are classic literature keystones")


if __name__ == "__main__":
    main()
