"""
Fig. 18 -- gamma in a human cohort with absolute abundance, with and without load
=============================================================================

Vandeputte et al. (2021, Nat. Commun.): daily quantitative microbiome profiling
(flow-cytometry cell counts + 16S) of 20 women over six weeks -- the dense human
time series with absolute load that pins down the human gamma free of
compositional bias. We extract it from the miaTime `temporalMicrobiome20` object
and run the same gLV inference.

We infer gamma two ways on the SAME samples:
   absolute :  x_i = proportion_i * cell_count_per_gram   (QMP, the real load)
   relative :  x_i = proportion_i                          (compositional)

If the human absolute gamma is ~0, the human relative-abundance cohorts (Fig. 10)
were biased upward and the mouse-human gap was largely artefactual; if it stays
~0.3, the difference is genuine.

Run:  python3 infer_vandeputte.py
"""

import numpy as np
import rdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infer_and_place import fit_matrix, disorder_stats, bootstrap, bc_ci, S_TAXA

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"
NA_INT = -2147483648


def load_vandeputte():
    o = rdata.read_rda("data/vandeputte/temporalMicrobiome20.rda")["temporalMicrobiome20"]
    counts = np.asarray(o.assays.data.listData["counts"], dtype=float)   # taxa x samples
    cd = o.colData.listData
    day = np.asarray(cd["day"], dtype=float).ravel()
    pid = np.asarray([str(x) for x in np.asarray(cd["person_id"]).ravel()])
    X = np.nan_to_num(counts.T)                                          # samples x taxa (QMP, absolute)
    day = np.where(day < -1e9, np.nan, day)
    ok = (~np.isnan(day)) & (X.sum(1) > 0)
    return X[ok], day[ok], pid[ok]


def design(X, day, pid, use_load, top, max_gap=3.0):
    # X is the QMP (absolute) count matrix; 'absolute' uses it directly,
    # 'relative' uses proportions (load discarded).
    prop = X / X.sum(1, keepdims=True)
    Xa = X[:, top] if use_load else prop[:, top]
    scale = np.median(np.where(Xa > 0, Xa, np.nan), axis=0)
    scale = np.where(np.isnan(scale) | (scale <= 0), 1.0, scale)
    Xs = Xa / scale
    rows_y, rows_Z, eps = [], [], 1e-3
    for sb in np.unique(pid):
        idx = np.where(pid == sb)[0]
        idx = idx[np.argsort(day[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            dt = day[b] - day[a]
            if dt <= 0 or dt > max_gap:
                continue
            g = (np.log(Xs[b] + eps) - np.log(Xs[a] + eps)) / dt
            rows_y.append(g); rows_Z.append(np.concatenate([[1.0], Xs[a]]))
    return np.array(rows_y), np.array(rows_Z)


def gamma_ci(X, day, pid, use_load, top):
    Y, Z = design(X, day, pid, use_load, top)
    J = fit_matrix(Y, Z)
    _, _, gamma = disorder_stats(J)
    _, _, bg = bootstrap(Y, Z)
    lo, hi = bc_ci(gamma, bg)
    return gamma, lo, hi, Y.shape[0]


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    X, day, pid = load_vandeputte()
    print(f"loaded: {X.shape[0]} valid samples, {X.shape[1]} taxa, "
          f"{len(np.unique(pid))} subjects, days {np.nanmin(day):.0f}-{np.nanmax(day):.0f}")

    prop = X / X.sum(1, keepdims=True)
    top = np.argsort(prop.mean(0))[::-1][:S_TAXA]

    res = {}
    for use_load, lab in [(True, "absolute (QMP, flow load)"),
                          (False, "relative (compositional)")]:
        g, lo, hi, n = gamma_ci(X, day, pid, use_load, top)
        res[lab] = (g, lo, hi)
        print(f"  {lab:26s}: gamma={g:+.3f} [{lo:+.3f},{hi:+.3f}]  (n_trans={n})")

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    cols = {"absolute (QMP, flow load)": "#1b9e77", "relative (compositional)": "#d95f02"}
    for i, (lab, (g, lo, hi)) in enumerate(res.items()):
        ax.errorbar(i, g, yerr=[[max(g - lo, 0)], [max(hi - g, 0)]], fmt="o",
                    ms=13, color=cols[lab], capsize=6)
    ax.axhline(0, ls="--", color="grey", lw=1)
    ax.axhline(0.3, ls=":", color="#d95f02", lw=1)
    ax.text(1.4, 0.3, "human relative\ncohorts (Fig.10)", fontsize=8, color="#d95f02", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["absolute\n(QMP load)", "relative\n(compositional)"])
    ax.set_xlim(-0.5, 1.8); ax.set_ylim(-0.2, 0.6)
    ax.set_ylabel("inferred interaction symmetry  $\\gamma$")
    ax.set_title("Human cohort with absolute load (Vandeputte 2021):\n"
                 "$\\gamma$ with and without the measured load, on identical samples", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig18_vandeputte_human_absolute.png", dpi=150)
    plt.close(fig)
    print("[fig18] written")


if __name__ == "__main__":
    main()
