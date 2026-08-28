"""DEPRECATED (figure dropped from the manuscript).
Produces figures/fig16_keystones.png, Gibson-inferred keystone candidates. The
manuscript now reports keystones only where there is knockout ground truth
(venturelli_keystone_validation.py) plus the human candidates from human_keystones.py.


Fig. 16 -- Translational tool 1: identify keystone taxa from real data
======================================================================

The theory's operational test for a keystone (Fig. 4) is an in-silico knockout:
remove a taxon and measure how far the REST of the community moves. We run that
test on the interaction matrix J inferred from the real Gibson (healthy) cohort.

  1. relax the inferred GLV to a baseline community;
  2. for each taxon, set it (and its interactions) to zero, re-relax, and measure
     the Bray-Curtis shift of the OTHER taxa = its keystone impact;
  3. rank and name the keystones via the RDP taxonomy;
  4. show keystone impact is NOT explained by abundance -- you need the network.

Honest note: the rightmost-eigenvector 'keystone score' (the linear proxy) does
NOT predict knockout impact for this small, noisy, regularised inferred matrix
(r ~ -0.2); the direct knockout -- the operational definition -- is what we use.

Run:  python3 keystones_from_data.py
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
DATA = "data/gibson_healthy"
NMAX, EPS = 50.0, 1e-3


def relax(J, n0, r, T=300):
    def rhs(t, n):
        n = np.clip(n, 0, NMAX)
        return n * (r - n + J @ n) + EPS
    s = solve_ivp(rhs, (0, T), np.clip(np.nan_to_num(n0), 0, NMAX),
                  method="LSODA", t_eval=[T], rtol=1e-6, atol=1e-8, max_step=2.0)
    return np.clip(np.nan_to_num(s.y[:, -1]), 0, NMAX)


def taxa_names():
    name = {}
    for row in csv.DictReader(open(f"{DATA}/rdp_species.tsv"), delimiter="\t"):
        asv = row[list(row)[0]]
        for level in ("Genus", "Family", "Order", "Class"):
            v = row.get(level, "")
            if v and v.lower() not in ("na", "", "unclassified"):
                name[asv] = v
                break
        else:
            name[asv] = asv
    return name


def asv_ids():
    with open(f"{DATA}/counts.tsv") as fh:
        r = csv.reader(fh, delimiter="\t"); next(r)
        return [row[0] for row in r]


def bc(a, b):
    return np.abs(a - b).sum() / (a.sum() + b.sum() + 1e-9)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    rel, subj, time, _ = load(DATA)
    top = np.argsort(rel.mean(0))[::-1][:S_TAXA]
    ids = np.array(asv_ids())[top]
    names = taxa_names()
    labels = [names.get(a, a) for a in ids]
    abund = rel.mean(0)[top]; abund = abund / abund.max()

    Y, Z = build_design(DATA, (1, 2, 3, 4, 5))
    J = fit_matrix(Y, Z)
    S = J.shape[0]
    r = np.ones(S)

    nb = relax(J, np.random.default_rng(0).uniform(0.1, 1, S), r)
    print(f"baseline survivors = {(nb>1e-2).sum()}/{S}")

    impact = np.zeros(S)
    for i in range(S):
        Jk = J.copy(); Jk[i, :] = 0; Jk[:, i] = 0
        n0 = nb.copy(); n0[i] = 0
        nf = relax(Jk, n0, r)
        others = np.delete(np.arange(S), i)
        impact[i] = bc(nf[others], nb[others])
    impact = impact / impact.max()

    order = np.argsort(impact)[::-1]
    print("Keystone ranking by in-silico knockout impact (real Gibson healthy):")
    for rnk, i in enumerate(order[:10], 1):
        print(f"  {rnk:2d}. {labels[i]:24s}  impact={impact[i]:.2f}  abundance={abund[i]:.2f}")
    corr = np.corrcoef(impact, abund)[0, 1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    topk = order[:10][::-1]
    ax[0].barh(range(len(topk)), impact[topk], color="#d95f02")
    ax[0].set_yticks(range(len(topk)))
    ax[0].set_yticklabels([labels[i] for i in topk], fontsize=8)
    ax[0].set_xlabel("keystone impact  (community shift when removed)")
    ax[0].set_title("A. Keystones from in-silico knockout on the inferred matrix\n(real Gibson healthy cohort)", fontsize=10)

    ax[1].scatter(abund, impact, s=30, color="#1b9e77", alpha=0.7)
    for i in order[:5]:
        ax[1].annotate(labels[i], (abund[i], impact[i]), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("mean relative abundance (normalised)")
    ax[1].set_ylabel("keystone impact")
    ax[1].set_title(f"B. Keystones are not just the abundant taxa\n(impact vs abundance: r = {corr:.2f})", fontsize=10)

    fig.suptitle("Fig. 16 - Translational tool: naming the keystone taxa from real data",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig16_keystones.png", dpi=150)
    plt.close(fig)
    print(f"[fig16] written | top keystone = {labels[order[0]]}, "
          f"impact-abundance r = {corr:.2f}")


if __name__ == "__main__":
    main()
