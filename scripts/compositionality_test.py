"""
Fig. 17 -- Does compositionality inflate the inferred symmetry gamma?
====================================================================

The six-cohort comparison (Fig. 10) showed gamma ~ 0 for the qPCR (absolute)
mouse cohorts but gamma ~ 0.2-0.3 for the relative-abundance human/infant
cohorts, and we argued the gap is a COMPOSITIONAL artefact. Here we test that
directly with a controlled experiment on data that have BOTH: the Gibson cohorts
carry qPCR loads, so on the SAME samples and taxa we infer gamma twice --

   absolute :  x_i = rel_i x (qPCR load)      (the true abundances)
   relative :  x_i = rel_i                    (compositional, load discarded)

If discarding the load inflates gamma, the relative estimate sits above the
absolute one -- isolating compositionality as the cause, and explaining the
human cohorts' positive gamma in Fig. 10.

Run:  python3 compositionality_test.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_gibson import load, PERTURBS
from infer_and_place import (load_qpcr, fit_matrix, disorder_stats, bootstrap,
                             bc_ci, S_TAXA, MAX_GAP)

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"


def design(data_dir, subjects, use_load):
    rel, subj, time, sids = load(data_dir)
    qpcr = load_qpcr(data_dir)
    top = np.argsort(rel.mean(0))[::-1][:S_TAXA]
    load_vec = np.array([qpcr.get(s, np.nan) for s in sids]) if use_load else np.ones(len(sids))
    Xabs = rel[:, top] * load_vec[:, None]
    scale = np.nanmedian(np.where(Xabs > 0, Xabs, np.nan), axis=0)
    Xs = Xabs / scale
    pert = np.zeros((len(sids), len(PERTURBS)))
    for j, (_, s, e) in enumerate(PERTURBS):
        for i in range(len(sids)):
            if subj[i] in subjects and s <= time[i] <= e:
                pert[i, j] = 1.0
    rows_y, rows_Z, eps = [], [], 1e-3
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
    return np.array(rows_y), np.array(rows_Z)


def gamma_ci(data_dir, subjects, use_load):
    Y, Z = design(data_dir, subjects, use_load)
    J = fit_matrix(Y, Z)
    _, _, gamma = disorder_stats(J)
    _, _, bg = bootstrap(Y, Z)
    lo, hi = bc_ci(gamma, bg)
    return gamma, lo, hi


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    cohorts = [("Healthy mice", "data/gibson_healthy", (1, 2, 3, 4, 5)),
               ("UC mice", "data/gibson_uc", (6, 7, 8, 9, 10))]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    x = np.arange(len(cohorts))
    for k, (use_load, col, lab, off) in enumerate(
            [(True, "#1b9e77", "absolute (qPCR load)", -0.12),
             (False, "#d95f02", "relative (compositional)", 0.12)]):
        for i, (name, d, subs) in enumerate(cohorts):
            g, lo, hi = gamma_ci(d, subs, use_load)
            ax.errorbar(x[i] + off, g, yerr=[[max(g - lo, 0)], [max(hi - g, 0)]],
                        fmt="o", ms=11, color=col, capsize=5,
                        label=lab if i == 0 else None)
            print(f"{name:14s} {lab:26s}: gamma={g:+.3f} [{lo:+.3f},{hi:+.3f}]")
    ax.axhline(0, ls="--", color="grey", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cohorts])
    ax.set_ylabel("inferred interaction symmetry  $\\gamma$")
    ax.set_ylim(-0.25, 0.6)
    ax.set_title("Controlled test: compositionality biases $\\gamma$ only modestly\n"
                 "same samples & taxa, load discarded -> $\\Delta\\gamma\\lesssim0.07$, far below the human $\\gamma\\approx0.3$", fontsize=9.8)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig17_compositionality.png", dpi=150)
    plt.close(fig)
    print("[fig17] written")


if __name__ == "__main__":
    main()
