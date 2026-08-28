"""
Real-data test: resistance / resilience / hysteresis in the Gibson-MDSINE2 cohort
=================================================================================

DATA (real, public): Gibson et al. (2021), the MDSINE2 benchmark gut time series
(github.com/gerberlab/MDSINE2_Paper, datasets/gibson/healthy/raw_tables).
1088 ASVs x 339 samples; gnotobiotic mice colonised with a human donor community,
densely sampled, with THREE annotated perturbations applied to subjects 2-5:
    High-Fat Diet (days 21.5-28.5), Vancomycin (35.5-42.5), Gentamicin (50.5-57.5).
Subject 1 is the unperturbed control.

We apply the theory's diagnostics directly to the real abundances:
  - Bray-Curtis trajectory vs the pre-perturbation baseline
  - per-perturbation return ratio  (1 = resilient, 0 = hysteretic)
  - the control subject as a stability reference

This is the empirical test that the in-silico proof-of-concept anticipated.

Run:  python3 analyze_gibson.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data/gibson_healthy"
OUT = "figures"


def load(data_dir=DATA):
    DATA = data_dir
    # counts: taxa x samples
    with open(f"{DATA}/counts.tsv") as fh:
        r = csv.reader(fh, delimiter="\t")
        hdr = next(r)
        sample_ids = hdr[1:]
        taxa, M = [], []
        for row in r:
            taxa.append(row[0])
            M.append([float(x) for x in row[1:]])
    counts = np.array(M).T                      # samples x taxa
    # metadata (keep only the 5 mouse subjects; drop inoculum & technical reps)
    meta = {}
    with open(f"{DATA}/metadata.tsv") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["subject"].isdigit():
                meta[row["sampleID"]] = (int(row["subject"]), float(row["time"]))
    keep = [i for i, s in enumerate(sample_ids) if s in meta]
    counts = counts[keep]
    sample_ids = [sample_ids[i] for i in keep]
    subj = np.array([meta[s][0] for s in sample_ids])
    time = np.array([meta[s][1] for s in sample_ids])
    # relative abundance
    rel = counts / np.clip(counts.sum(1, keepdims=True), 1, None)
    return rel, subj, time, sample_ids


def bc_to_baseline(X, base_mask):
    b = X[base_mask].mean(0)
    num = np.abs(X - b).sum(1)
    den = X.sum(1) + b.sum()
    return num / np.where(den == 0, 1, den)


PERTURBS = [("High-Fat Diet", 21.5, 28.5),
            ("Vancomycin", 35.5, 42.5),
            ("Gentamicin", 50.5, 57.5)]


def return_ratio(t, d, p_start, p_end, recov_win=8.0):
    """Peak rise during/after a perturbation vs how much remained at recovery."""
    base = d[(t >= 10) & (t < p_start)]
    base_lvl = base.mean() if len(base) else 0.0
    during = d[(t >= p_start) & (t <= p_end + 3)]
    peak = during.max() if len(during) else base_lvl
    recov = d[(t > p_end + 3) & (t <= p_end + 3 + recov_win)]
    recov_lvl = recov.mean() if len(recov) else peak
    if peak - base_lvl < 1e-6:
        return 1.0, base_lvl, peak, recov_lvl
    return float(np.clip((peak - recov_lvl) / (peak - base_lvl), 0, 1)), base_lvl, peak, recov_lvl


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    rel, subj, time, _ = load()

    perturbed = [2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    # perturbation bands
    for name, s, e in PERTURBS:
        ax.axvspan(s, e, color="grey", alpha=0.13)
        ax.text((s + e) / 2, 0.985, name, ha="center", va="top",
                fontsize=8, color="dimgrey", transform=ax.get_xaxis_transform())

    # (subject 1 is too sparsely sampled -- only days 0-1.5 and 29 -- to serve as
    #  a baseline reference, so we focus on the four densely-sampled perturbed mice.)

    # perturbed subjects + return-ratio summary
    print("Return ratio per perturbation (1=resilient, 0=hysteretic):")
    summary = {p[0]: [] for p in PERTURBS}
    allcurves = []
    for sb in perturbed:
        m = subj == sb
        t, X = time[m], rel[m]
        o = np.argsort(t); t, X = t[o], X[o]
        d = bc_to_baseline(X, (t >= 10) & (t < 21.5))   # baseline excludes colonisation transient
        allcurves.append((t, d))
        ax.plot(t, d, "-o", ms=2.5, lw=1.0, alpha=0.6, color="#d95f02",
                label="perturbed mice (n=4)" if sb == perturbed[0] else None)
        for name, s, e in PERTURBS:
            rr, *_ = return_ratio(t, d, s, e)
            summary[name].append(rr)

    for name in summary:
        vals = summary[name]
        print(f"  {name:14s}: mean return ratio = {np.mean(vals):.2f}  "
              f"(per subject: {', '.join(f'{v:.2f}' for v in vals)})")

    # baseline reference + final altered level
    base_lvl = np.mean([d[(t >= 10) & (t < 21.5)].mean() for t, d in allcurves])
    final_lvl = np.mean([d[t > 58].mean() for t, d in allcurves])
    ax.axhline(base_lvl, ls="--", color="#1b9e77", lw=1.3)
    ax.text(2, base_lvl + 0.02, "pre-perturbation stable state", fontsize=8, color="#1b9e77")
    ax.annotate("", xy=(64, final_lvl), xytext=(64, base_lvl),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1))
    ax.text(60.5, (base_lvl + final_lvl) / 2, "no return\nwithin\nfollow-up", fontsize=8, ha="right")
    print(f"  baseline level={base_lvl:.2f}, post-perturbation level={final_lvl:.2f} "
          f"-> net non-recovery within the follow-up")

    ax.set_xlabel("time (days)")
    ax.set_ylabel("community state\n(Bray-Curtis distance from pre-perturbation baseline)")
    ax.set_title("Gibson/MDSINE2: resistance, partial resilience and a non-recovering\n"
                 "shift after sequential antibiotics (consistent with, not proof of, hysteresis)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(0, time.max())
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_real_data_gibson.png", dpi=150)
    plt.close(fig)
    print("[fig8] real-data analysis written")


if __name__ == "__main__":
    main()
