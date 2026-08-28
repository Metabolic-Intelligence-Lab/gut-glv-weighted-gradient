"""
Fig. 10 -- Inferred interaction symmetry across SIX real cohorts
================================================================

Same gLV inference applied to six real longitudinal cohorts spanning two species
and four studies:
    Healthy mice            Gibson/MDSINE2  (qPCR absolute)      dense (daily)
    Ulcerative-colitis mice Gibson/MDSINE2  (qPCR absolute)      dense (daily)
    Human diet swap         David 2014      (relative)           dense (daily)
    Human pregnancy         DiGiulio 2015   (relative)           weekly
    Infant gut              Bokulich 2016   (relative)           monthly (sparse)
    Infant gut              Karelia/DIABIMMUNE (relative)        ~quarterly (sparse)

The scale-invariant symmetry gamma is the comparable quantity. Honest finding:
gamma is small in every cohort -- far from the symmetric spin-glass limit 1 --
but not identically zero (mice ~0, humans ~0.2-0.3). Sparsely-sampled infant
cohorts are shown but flagged: their long inter-sample gaps make the growth-rate
approximation unreliable, so their gamma is only indicative.

Run:  python3 compare_cohorts.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infer_and_place import build_design as gibson_design, fit_matrix, disorder_stats, bootstrap, bc_ci
from infer_mitre import infer_cohort

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"


def gibson_stats(data_dir, subjects):
    Y, Z = gibson_design(data_dir, subjects)
    J = fit_matrix(Y, Z)
    mu, sigma, gamma = disorder_stats(J)
    bm, bs, bg = bootstrap(Y, Z)
    return dict(mu=mu, sigma=sigma, gamma=gamma, boot_mu=bm, boot_sigma=bs,
                boot_gamma=bg, n=Y.shape[0])


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    # (label, getter, colour, dense?)
    specs = [
        ("Healthy mice\n(qPCR, daily)",  lambda: gibson_stats("data/gibson_healthy", (1, 2, 3, 4, 5)), "#1b9e77", True),
        ("UC mice\n(qPCR, daily)",       lambda: gibson_stats("data/gibson_uc", (6, 7, 8, 9, 10)),       "#1b9e77", True),
        ("Human diet\n(David, daily)",   lambda: infer_cohort("data/david2014", 3, (0, 5)),               "#d95f02", True),
        ("Human pregnancy\n(DiGiulio, weekly)", lambda: infer_cohort("data/digiulio", 10),                "#d95f02", True),
        ("Infant gut\n(Bokulich, monthly)",     lambda: infer_cohort("data/bokulich", 45),                "#999999", False),
        ("Infant gut\n(Karelia, sparse)",       lambda: infer_cohort("data/karelia", 110),                "#999999", False),
    ]

    results = []
    for name, getter, col, dense in specs:
        r = getter()
        results.append((name, r, col, dense))
        glo, ghi = bc_ci(r["gamma"], r["boot_gamma"])
        print(f"{name.splitlines()[0]:18s}: n={r['n']:4d}  mu={r['mu']:+.2f}  "
              f"sigma={r['sigma']:6.2f}  gamma={r['gamma']:+.3f} [{glo:+.3f},{ghi:+.3f}]"
              f"{'' if dense else '   (sparse - indicative)'}")

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for i, (name, r, col, dense) in enumerate(results):
        val = r["gamma"]
        lo, hi = bc_ci(val, r["boot_gamma"])
        ax.errorbar(i, val, yerr=[[max(val - lo, 0)], [max(hi - val, 0)]],
                    fmt="o" if dense else "s", ms=12, color=col, capsize=5,
                    markerfacecolor=col if dense else "white",
                    markeredgecolor=col, mew=2)
    ax.axhline(0, ls="--", color="grey", lw=1)
    ax.axhline(1, ls=":", color="grey", lw=1.2)
    ax.text(-0.4, 1.02, "symmetric  (pure spin-glass, gamma=1)", fontsize=8, color="grey")
    ax.text(-0.4, 0.03, "asymmetric / uncorrelated  (gamma=0)", fontsize=8, color="grey")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([n for n, _, _, _ in results], fontsize=7.5)
    ax.set_ylabel("inferred interaction symmetry  gamma")
    ax.set_ylim(-0.4, 1.15)
    ax.set_xlim(-0.5, len(results) - 0.5)
    ax.set_title("Interaction symmetry across six longitudinal cohorts (the seventh,\n"
                 "Vandeputte, is shown separately): small everywhere, ~0 in mice and\n"
                 "~0.2-0.3 in humans (open squares = sparsely sampled, indicative only)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig10_cohort_comparison.png", dpi=150)
    plt.close(fig)
    print("[fig10] six-cohort comparison written")


if __name__ == "__main__":
    main()
