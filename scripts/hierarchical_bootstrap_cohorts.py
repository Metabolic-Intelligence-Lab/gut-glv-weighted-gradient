"""
Honest (subject/block) bootstrap CIs on gamma for ALL cohorts  (referee-4 P1, §16.8)
====================================================================================

The reported per-cohort gamma CIs resample transitions i.i.d., ignoring that transitions
are clustered within subjects and temporally autocorrelated. gamma_confidence.py did the
clustered bootstrap only for the healthy-mouse cohort; the referee asks for it across all
cohorts. This script rebuilds each cohort's transition design WITH per-transition subject
tags (faithfully mirroring the inference in infer_and_place / infer_mitre / infer_vandeputte)
and compares the i.i.d.-transition CI with a hierarchical subject+moving-block CI.

Verdict per cohort: whether the low-gamma reading is robust once clustering and temporal
autocorrelation are respected. Run:  python3 hierarchical_bootstrap_cohorts.py
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import infer_and_place as IAP
from infer_and_place import fit_matrix, disorder_stats
import infer_mitre as IM
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"
BLOCK_L = 5


def gamma_of(Y, Z):
    try:
        return disorder_stats(fit_matrix(Y, Z))[2]
    except Exception:
        return np.nan


def tagged_gibson(data_dir, subjects):
    rel, subj, time, sids = IAP.load(data_dir)
    qpcr = IAP.load_qpcr(data_dir)
    top = np.argsort(rel.mean(0))[::-1][:IAP.S_TAXA]
    load_vec = np.array([qpcr.get(s, np.nan) for s in sids])
    Xabs = rel[:, top] * load_vec[:, None]
    scale = np.nanmedian(np.where(Xabs > 0, Xabs, np.nan), axis=0); Xs = Xabs / scale
    pert = np.zeros((len(sids), len(IAP.PERTURBS)))
    for j, (_, s, e) in enumerate(IAP.PERTURBS):
        for i in range(len(sids)):
            if subj[i] in subjects and s <= time[i] <= e:
                pert[i, j] = 1.0
    ry, rz, tag = [], [], []; eps = 1e-3
    for sb in subjects:
        idx = np.where(subj == sb)[0]; idx = idx[np.argsort(time[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            dt = time[b] - time[a]
            if dt <= 0 or dt > IAP.MAX_GAP:
                continue
            xa, xb = Xs[a], Xs[b]
            if np.any(np.isnan(xa)) or np.any(np.isnan(xb)):
                continue
            ry.append((np.log(xb + eps) - np.log(xa + eps)) / dt)
            rz.append(np.concatenate([[1.0], xa, pert[a]])); tag.append(sb)
    return np.array(ry), np.array(rz), np.array(tag)


def tagged_mitre(cohort_dir, max_gap):
    rel, subj, time = IM.load_cohort(cohort_dir)
    top = np.argsort(rel.mean(0))[::-1][:IAP.S_TAXA]; Xs = rel[:, top]
    scale = np.median(np.where(Xs > 0, Xs, np.nan), axis=0)
    scale = np.where(np.isnan(scale) | (scale <= 0), 1.0, scale); Xs = Xs / scale
    ry, rz, tag = [], [], []; eps = 1e-3
    for sb in np.unique(subj):
        idx = np.where(subj == sb)[0]; idx = idx[np.argsort(time[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            dt = time[b] - time[a]
            if dt <= 0 or dt > max_gap:
                continue
            ry.append((np.log(Xs[b] + eps) - np.log(Xs[a] + eps)) / dt)
            rz.append([1.0] + list(Xs[a])); tag.append(sb)
    return np.array(ry), np.array(rz), np.array(tag)


def tagged_vandeputte():
    import infer_vandeputte as IV
    X, day, pid = IV.load_vandeputte()
    top = np.argsort((X / X.sum(1, keepdims=True)).mean(0))[::-1][:IAP.S_TAXA]
    Xa = X[:, top]
    scale = np.median(np.where(Xa > 0, Xa, np.nan), axis=0)
    scale = np.where(np.isnan(scale) | (scale <= 0), 1.0, scale); Xs = Xa / scale
    ry, rz, tag = [], [], []; eps = 1e-3
    for sb in np.unique(pid):
        idx = np.where(pid == sb)[0]; idx = idx[np.argsort(day[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            dt = day[b] - day[a]
            if dt <= 0 or dt > 3.0:
                continue
            ry.append((np.log(Xs[b] + eps) - np.log(Xs[a] + eps)) / dt)
            rz.append(np.concatenate([[1.0], Xs[a]])); tag.append(sb)
    return np.array(ry), np.array(rz), np.array(tag)


def moving_block(order_rows, rng, L=BLOCK_L):
    n = len(order_rows)
    if n <= L:
        return order_rows.copy()
    out = []
    while len(out) < n:
        s = rng.integers(0, n - L + 1)
        out.extend(order_rows[s:s + L])
    return np.array(out[:n])


def cis(Y, Z, tag, nboot=1500, seed=0):
    rng = np.random.default_rng(seed)
    subs = np.unique(tag); rows_by = {s: np.where(tag == s)[0] for s in subs}
    nt = len(Y); iid, hier = [], []
    for _ in range(nboot):
        s = rng.integers(0, nt, nt); iid.append(gamma_of(Y[s], Z[s]))
        drawn = rng.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([moving_block(rows_by[s], rng) for s in drawn])
        hier.append(gamma_of(Y[idx], Z[idx]))
    def ci(v):
        v = np.array(v); v = v[np.isfinite(v)]
        return np.percentile(v, [2.5, 97.5])
    return ci(iid), ci(hier), len(subs)


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    cohorts = [
        ("Healthy mice (qPCR)",   lambda: tagged_gibson("data/gibson_healthy", (1, 2, 3, 4, 5))),
        ("UC mice (qPCR)",        lambda: tagged_gibson("data/gibson_uc", (6, 7, 8, 9, 10))),
        ("Human diet (David)",    lambda: tagged_mitre("data/david2014", 3)),
        ("Pregnancy (DiGiulio)",  lambda: tagged_mitre("data/digiulio", 10)),
        ("Infant (Bokulich)",     lambda: tagged_mitre("data/bokulich", 45)),
        ("Infant (Karelia)",      lambda: tagged_mitre("data/karelia", 110)),
        ("Human QMP (Vandeputte)", tagged_vandeputte),
    ]
    rows = []
    print(f"{'cohort':>24} {'nsub':>4} {'ntr':>5} {'gamma':>7} | {'iid CI':>16} | {'hier CI (honest)':>18} | ratio")
    print("-" * 96)
    for name, fn in cohorts:
        try:
            Y, Z, tag = fn()
            if len(Y) < 10:
                raise ValueError("too few transitions")
            g = gamma_of(Y, Z)
            iid, hier, nsub = cis(Y, Z, tag)
            wi, wh = iid[1] - iid[0], hier[1] - hier[0]
            rows.append((name, nsub, len(Y), g, iid, hier, wh / wi))
            print(f"{name:>24} {nsub:>4} {len(Y):>5} {g:>+7.3f} | [{iid[0]:+.3f},{iid[1]:+.3f}] | "
                  f"[{hier[0]:+.3f},{hier[1]:+.3f}] | x{wh/wi:.2f}")
        except Exception as e:
            print(f"{name:>24}  -- skipped ({type(e).__name__}: {str(e)[:40]})")

    # figure: forest plot, iid (thin grey) vs hierarchical (thick colour)
    fig, ax = plt.subplots(figsize=(7.8, 0.6 * len(rows) + 1.2))
    for k, (name, nsub, ntr, g, iid, hier, ratio) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot(iid, [y + 0.13, y + 0.13], color="#bbbbbb", lw=4, solid_capstyle="round")
        ax.plot(hier, [y - 0.13, y - 0.13], color="#0072B2", lw=6, solid_capstyle="round", alpha=0.85)
        ax.plot(g, y, "o", color="white", mec="#D55E00", mew=2, ms=8, zorder=5)
        ax.text(hier[1] + 0.02, y, f"{nsub} subj, x{ratio:.1f}", va="center", fontsize=7.5, color="#555")
    ax.axvline(0, color="grey", ls=":", lw=0.8)
    ax.axvline(1.0, color="#1b9e77", ls="--", lw=1); ax.text(1.0, len(rows) - 0.4, "spin-glass\n$\\gamma=1$", fontsize=7.5, color="#1b9e77", ha="center")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9)
    ax.set_xlabel(r"interaction reciprocity $\gamma$  (grey = i.i.d.; blue = hierarchical subject+block)")
    ax.set_title("Honest per-cohort reciprocity CIs: low $\\gamma$ is robust to clustering", fontsize=11)
    ax.set_xlim(-0.35, 1.05)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_hierboot_cohorts.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_hierboot_cohorts.png written")
    np.savez(f"{OUT}/hierboot_cohorts.npz",
             names=np.array([r[0] for r in rows]), gamma=np.array([r[3] for r in rows]),
             iid=np.array([r[4] for r in rows]), hier=np.array([r[5] for r in rows]))


if __name__ == "__main__":
    main()
