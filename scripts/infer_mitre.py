"""
Generic gLV inference for MITRE-format cohorts (abundance.csv + sample_metadata.csv)
====================================================================================

MITRE (github.com/gerberlab/mitre) ships several real human/infant longitudinal
cohorts in a common CSV format:
    abundance.csv        rows = samples, cols = OTUs (first col = sampleID)
    sample_metadata.csv  no header: sampleID, subjectID, time
This module loads any of them and builds the gLV design matrix, reusing the
inference core from infer_and_place.py. The per-cohort `max_gap` (in the cohort's
own time units) controls which consecutive-sample transitions are usable for the
growth-rate approximation -- crucial because sampling density varies from daily
(David) to monthly (karelia).
"""

import csv
import numpy as np

from infer_and_place import fit_matrix, disorder_stats, bootstrap, S_TAXA

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")


def load_cohort(cohort_dir):
    with open(f"{cohort_dir}/abundance.csv") as fh:
        r = csv.reader(fh)
        next(r)                                          # header = OTU names
        sids, M = [], []
        for row in r:
            sids.append(row[0])
            M.append([float(x) if x else 0.0 for x in row[1:]])
    counts = np.array(M)
    rel = counts / np.clip(counts.sum(1, keepdims=True), 1, None)

    meta = {}
    for row in csv.reader(open(f"{cohort_dir}/sample_metadata.csv")):
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
    return rel, subj, time


def build_design(cohort_dir, max_gap, field_window=None):
    """field_window = (start, end) in cohort time units adds an external-field
    covariate active in that window (e.g. controlled-diet period)."""
    rel, subj, time = load_cohort(cohort_dir)
    top = np.argsort(rel.mean(0))[::-1][:S_TAXA]
    Xs = rel[:, top]
    scale = np.median(np.where(Xs > 0, Xs, np.nan), axis=0)
    scale = np.where(np.isnan(scale) | (scale <= 0), 1.0, scale)
    Xs = Xs / scale

    field = None
    if field_window is not None:
        s, e = field_window
        field = ((time >= s) & (time <= e)).astype(float)

    rows_y, rows_Z = [], []
    eps = 1e-3
    for sb in np.unique(subj):
        idx = np.where(subj == sb)[0]
        idx = idx[np.argsort(time[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            dt = time[b] - time[a]
            if dt <= 0 or dt > max_gap:
                continue
            xa, xb = Xs[a], Xs[b]
            g = (np.log(xb + eps) - np.log(xa + eps)) / dt
            z = [1.0] + list(xa)
            if field is not None:
                z.append(field[a])
            rows_y.append(g)
            rows_Z.append(z)
    return np.array(rows_y), np.array(rows_Z), len(np.unique(subj))


def infer_cohort(cohort_dir, max_gap, field_window=None):
    Y, Z, nsub = build_design(cohort_dir, max_gap, field_window)
    J = fit_matrix(Y, Z)
    mu, sigma, gamma = disorder_stats(J)
    bm, bs, bg = bootstrap(Y, Z)
    off = J[~np.eye(J.shape[0], dtype=bool)]
    return dict(mu=mu, sigma=sigma, gamma=gamma, boot_mu=bm, boot_sigma=bs,
                boot_gamma=bg, n=Y.shape[0], nsub=nsub,
                frac_neg=float((off < 0).mean()))
