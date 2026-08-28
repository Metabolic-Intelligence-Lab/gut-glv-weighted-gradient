"""DEPRECATED (result absent from the manuscript).
Cross-cohort healthy-vs-UC keystone reproducibility. As its own docstring notes, the
two Gibson cohorts share a benchmark and taxonomy, so this is a within-benchmark
consistency check and NOT independent external validation; it is therefore not cited.
Its output name (fig17_keystone_validation.png) also collides conceptually with
fig17_compositionality.png, which is the fig17 that ships.


Fig. 17 -- Keystone reproducibility across two Gibson cohorts (NOT external validation)
=======================================================================================

A keystone call is only useful if it is reproducible. We infer the interaction
matrix and run the in-silico knockout on two cohorts of the SAME Gibson/MDSINE2
benchmark -- healthy mice (subjects 1-5) and ulcerative-colitis mice (6-10) --
aggregate the knockout impact to genus level, and ask whether the ranking agrees.

CAVEAT: these two cohorts share the same benchmark, taxonomy and pipeline, so this
is a within-benchmark consistency check, NOT independent external validation. The
shared abundance structure inflates the cross-cohort agreement. Genuine validation
would require single-species dropout experiments (e.g. gnotobiotic), which we do
not have here; the keystone calls remain hypotheses.

Run:  python3 validate_keystones.py
"""

import csv
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_gibson import load
from infer_and_place import build_design, fit_matrix, S_TAXA

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"
TAX = "data/gibson_healthy/rdp_species.tsv"
NMAX, EPS = 50.0, 1e-3


def genus_map():
    g = {}
    for row in csv.DictReader(open(TAX), delimiter="\t"):
        asv = row[list(row)[0]]
        for level in ("Genus", "Family", "Order"):
            v = row.get(level, "")
            if v and v.lower() not in ("na", "", "unclassified"):
                g[asv] = v
                break
    return g


def asv_ids(data_dir):
    with open(f"{data_dir}/counts.tsv") as fh:
        r = csv.reader(fh, delimiter="\t"); next(r)
        return [row[0] for row in r]


def relax(J, n0, r, T=300):
    def rhs(t, n):
        n = np.clip(n, 0, NMAX)
        return n * (r - n + J @ n) + EPS
    s = solve_ivp(rhs, (0, T), np.clip(np.nan_to_num(n0), 0, NMAX),
                  method="LSODA", t_eval=[T], rtol=1e-6, atol=1e-8, max_step=2.0)
    return np.clip(np.nan_to_num(s.y[:, -1]), 0, NMAX)


def bc(a, b):
    return np.abs(a - b).sum() / (a.sum() + b.sum() + 1e-9)


def keystone_by_genus(data_dir, subjects, gmap):
    rel, subj, time, _ = load(data_dir)
    top = np.argsort(rel.mean(0))[::-1][:S_TAXA]
    ids = np.array(asv_ids(data_dir))[top]
    J = fit_matrix(*build_design(data_dir, subjects))
    S = J.shape[0]; r = np.ones(S)
    nb = relax(J, np.random.default_rng(0).uniform(0.1, 1, S), r)
    impact = np.zeros(S)
    for i in range(S):
        Jk = J.copy(); Jk[i, :] = 0; Jk[:, i] = 0
        n0 = nb.copy(); n0[i] = 0
        nf = relax(Jk, n0, r)
        others = np.delete(np.arange(S), i)
        impact[i] = bc(nf[others], nb[others])
    impact /= impact.max()
    # aggregate to genus: a genus's keystone score = its most impactful member
    by = {}
    for a, im in zip(ids, impact):
        g = gmap.get(a, a)
        by[g] = max(by.get(g, 0.0), im)
    return by


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    gmap = genus_map()
    H = keystone_by_genus("data/gibson_healthy", (1, 2, 3, 4, 5), gmap)
    U = keystone_by_genus("data/gibson_uc", (6, 7, 8, 9, 10), gmap)

    shared = sorted(set(H) & set(U))
    h = np.array([H[g] for g in shared])
    u = np.array([U[g] for g in shared])
    corr = np.corrcoef(h, u)[0, 1]
    K = 6
    topH = set(np.array(shared)[np.argsort(h)[::-1][:K]])
    topU = set(np.array(shared)[np.argsort(u)[::-1][:K]])
    both = topH & topU
    print(f"shared genera: {len(shared)}  | full Pearson r={corr:.2f}")
    print(f"top-{K} keystones shared across cohorts: {len(both)}/{K}  -> {sorted(both)}")

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    for i, g in enumerate(shared):
        leading = g in both
        ax.scatter(h[i], u[i], s=110 if leading else 30,
                   color="#d95f02" if leading else "#9ecae1",
                   edgecolor="black" if leading else "none",
                   zorder=4 if leading else 2, alpha=0.9 if leading else 0.7)
        if g in (topH | topU):
            ax.annotate(g, (h[i], u[i]), fontsize=8, xytext=(5, 2),
                        textcoords="offset points")
    lim = [0, 1.08]
    ax.plot(lim, lim, ls=":", color="grey")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("keystone impact -- healthy cohort")
    ax.set_ylabel("keystone impact -- ulcerative-colitis cohort")
    ax.set_title(f"Fig. 17 - Keystone ranking across two Gibson cohorts (same benchmark)\n"
                 f"{len(both)} of the top-{K} agree (orange) -- within-benchmark consistency,\n"
                 "NOT independent validation; calls remain hypotheses", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig17_keystone_validation.png", dpi=150)
    plt.close(fig)
    print(f"[fig17] written | top-{K} overlap={len(both)}/{K}, full r={corr:.2f}")


if __name__ == "__main__":
    main()
