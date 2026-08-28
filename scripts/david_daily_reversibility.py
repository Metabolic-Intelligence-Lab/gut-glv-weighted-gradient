"""
Reversibility contrast in the David daily time series (subjects A and B)
=======================================================================

The Gibson/MDSINE2 evidence for hysteresis is one-directional: three antibiotic
doses, never reversed, so incomplete recovery there cannot be told apart from
slow monotonic relaxation (main text; Gonze 2017). The David daily data supply
the missing within-host contrast: the SAME estimator, the SAME diagnostic, two
perturbations of very different strength in two hosts of the same study.

    Subject A -- travel abroad, days 71-122   (reversible lifestyle perturbation)
    Subject B -- enteric infection, days 150-159              (strong shock)

David et al., "Host lifestyle affects human microbiota on daily timescales",
Genome Biol. 15, R89 (2014). OTU tables and collection-day metadata are the
seqtime redistribution (github.com/hallucigenia-sparsa/seqtime, .rda).

We compute the paper's own diagnostics (pipeline_for_real_data): Bray-Curtis
distance to the pre-perturbation baseline, and the return ratio (1 = returns to
baseline = resilience; 0 = stays displaced = persistent shift / basin change).

Finding: subject A recovers (return ratio ~0.9, resilience) while subject B does
not (return ratio ~0.6, a durable non-returning shift), reproducing David's
qualitative report and giving a real-data instance of a reversible vs a
near-irreversible perturbation, which uniform slow relaxation does not explain.

Requires rdata (pip install rdata).  Run:  python3 david_daily_reversibility.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rdata

from pipeline_for_real_data import (bray_curtis_to_baseline, return_ratio,
                                    early_warning)

DATA = "data/david_daily"
OUT = "figures"
# perturbation windows (days), from David et al. 2014
EVENTS = {"A": (71, 122, "travel abroad"),
          "B": (150, 159, "enteric infection")}


def load(tag):
    otu = rdata.read_rda(f"{DATA}/david_stool{tag}_otus.rda")[f"david_stool{tag}_otus"]
    met = rdata.read_rda(f"{DATA}/david_stool{tag}_metadata.rda")[f"david_stool{tag}_metadata"]
    import pandas as pd
    O, M = pd.DataFrame(otu), pd.DataFrame(met)
    days = M.loc["collection-day"].astype(float).values
    X = O.T.values.astype(float)                       # samples x OTUs (counts)
    o = np.argsort(days)
    days, X = days[o], X[o]
    X = X / np.clip(X.sum(1, keepdims=True), 1, None)  # relative abundance
    return days, X


def analyse(tag):
    p0, p1, name = EVENTS[tag]
    days, X = load(tag)
    base = np.where(days < p0)[0]
    recov = np.where(days > p1)[0]
    d = bray_curtis_to_baseline(X, base)
    rr = return_ratio(d, base, recov)
    var, _ = early_warning(d)
    return dict(tag=tag, name=name, days=days, d=d, p0=p0, p1=p1,
                base=base, recov=recov, rr=rr,
                base_lvl=d[base].mean(),
                recov_lvl=d[recov].mean())


def main():
    A, B = analyse("A"), analyse("B")
    for r in (A, B):
        verdict = "RESILIENT (recovers)" if r["rr"] > 0.7 else \
                  "PERSISTENT SHIFT (does not return)"
        print(f"Subject {r['tag']} ({r['name']}): "
              f"baseline BC={r['base_lvl']:.2f}, recovery BC={r['recov_lvl']:.2f}, "
              f"return ratio={r['rr']:.2f} -> {verdict}")

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for a, r, col in zip(ax, (A, B), ("#2c6fbb", "#c65a2e")):
        a.axvspan(r["p0"], r["p1"], color=col, alpha=.12,
                  label=f"{r['name']}\n(days {r['p0']}-{r['p1']})")
        a.axhline(r["base_lvl"], ls="--", color="0.5", lw=.9, label="baseline")
        a.plot(r["days"], r["d"], "-", color=col, lw=1.0, marker="o", ms=2.4)
        a.set_xlabel("day")
        a.set_title(f"Subject {r['tag']}: {r['name']}\n"
                    rf"return ratio $={r['rr']:.2f}$ "
                    f"({'resilience' if r['rr'] > 0.7 else 'incomplete recovery'})")
        a.legend(frameon=False, fontsize=8, loc="upper right")
    ax[0].set_ylabel("Bray-Curtis distance to baseline")
    fig.suptitle("Reversible vs incompletely-recovering perturbation, same study, "
                 "same pipeline", fontsize=10)
    fig.tight_layout()
    out = f"{OUT}/fig23_david_reversibility.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
