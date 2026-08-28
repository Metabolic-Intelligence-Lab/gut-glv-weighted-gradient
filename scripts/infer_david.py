"""
Human cohort: infer disorder statistics from David et al. (2014, Nature)
========================================================================

Tests whether gamma ~ 0 (asymmetric interactions) holds in HUMANS, not only in
gnotobiotic mice. Data: the David et al. (2014) diet-swap study (~21 subjects on
a plant- or animal-based diet for 5 days, sampled days -5..+10), preprocessed in
the MITRE package (github.com/gerberlab/mitre, example_data/david).

No qPCR here, so we work in relative abundance; gamma is scale-invariant and
therefore the robust quantity to compare across studies. The controlled-diet
window is included as an external-field covariate. Reuses the inference core
from infer_and_place.py.

Run:  python3 infer_david.py
"""

import csv
import numpy as np

from infer_and_place import fit_matrix, disorder_stats, bootstrap, S_TAXA, MAX_GAP

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

DATA = "data/david2014"


def load_david():
    # abundance: rows = samples, cols = OTUs (first col = sampleID)
    with open(f"{DATA}/abundance.csv") as fh:
        r = csv.reader(fh)
        header = next(r)
        otus = header[1:]
        sids, M = [], []
        for row in r:
            sids.append(row[0])
            M.append([float(x) for x in row[1:]])
    counts = np.array(M)                                  # samples x taxa
    rel = counts / np.clip(counts.sum(1, keepdims=True), 1, None)

    # metadata (no header): sampleID, subjectID, time
    meta = {}
    for row in csv.reader(open(f"{DATA}/sample_metadata.csv")):
        if len(row) >= 3:
            try:
                meta[row[0]] = (row[1], float(row[2]))
            except ValueError:
                pass
    keep = [i for i, s in enumerate(sids) if s in meta]
    rel = rel[keep]
    sids = [sids[i] for i in keep]
    subj = np.array([meta[s][0] for s in sids])
    time = np.array([meta[s][1] for s in sids])
    return rel, subj, time, sids


def build_design():
    rel, subj, time, sids = load_david()
    top = np.argsort(rel.mean(0))[::-1][:S_TAXA]
    Xs = rel[:, top]
    scale = np.median(np.where(Xs > 0, Xs, np.nan), axis=0)
    scale = np.where(np.isnan(scale) | (scale <= 0), 1.0, scale)
    Xs = Xs / scale                                       # each taxon ~ O(1)

    # external field: controlled-diet window (days 0..5)
    on_diet = ((time >= 0) & (time <= 5)).astype(float)

    rows_y, rows_Z = [], []
    eps = 1e-3
    for sb in np.unique(subj):
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
            z = np.concatenate([[1.0], xa, [on_diet[a]]])
            rows_y.append(g)
            rows_Z.append(z)
    return np.array(rows_y), np.array(rows_Z)


def main():
    Y, Z = build_design()
    print(f"David 2014 (human): transitions={Y.shape[0]}, taxa={Y.shape[1]}, "
          f"subjects={len(np.unique(load_david()[1]))}")
    J = fit_matrix(Y, Z)
    mu, sigma, gamma = disorder_stats(J)
    bm, bs, bg = bootstrap(Y, Z)
    print(f"  mu    = {mu:6.2f}  [{np.percentile(bm,2.5):.2f}, {np.percentile(bm,97.5):.2f}]")
    print(f"  sigma = {sigma:6.2f}  [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]")
    print(f"  gamma = {gamma:6.3f}  [{np.percentile(bg,2.5):+.3f}, {np.percentile(bg,97.5):+.3f}]")
    off = J[~np.eye(J.shape[0], dtype=bool)]
    print(f"  fraction competitive (negative): {(off<0).mean():.2f}")
    np.savez("figures/david_stats.npz", mu=mu, sigma=sigma, gamma=gamma,
             boot_mu=bm, boot_sigma=bs, boot_gamma=bg)
    print("[saved] figures/david_stats.npz")


if __name__ == "__main__":
    main()
