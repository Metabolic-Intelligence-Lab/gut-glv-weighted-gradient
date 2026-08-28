"""
Infer the interaction matrix from the real Gibson/MDSINE2 data and place the
community on the theory's phase diagram
=======================================================================

Closes the loop theory <-> data. From the real absolute abundances we fit a
regularized Generalized Lotka-Volterra model (MDSINE-style), separating the
antibiotic/diet perturbations as external-field covariates, and read off the
disorder statistics that parametrize the spin-glass theory:

    mu     net competition          = -mean(offdiag J) * S
    sigma  disorder strength        =  std(offdiag J) * sqrt(S)
    gamma  interaction symmetry     =  corr(J_ij, J_ji)        [scale-invariant]

J is normalized by the typical self-limitation (diagonal -> -1), matching the
model convention dot n_i = n_i(r_i - n_i + sum_j J_ij n_j). We bootstrap over
transitions for uncertainty, then locate (sigma, gamma) on the phase diagram.

Run:  python3 infer_and_place.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_gibson import load, PERTURBS

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"
DATA = "data/gibson_healthy"
S_TAXA = 30          # number of taxa retained (by mean abundance)
RIDGE = 0.5
MAX_GAP = 3.0        # days; skip transitions across larger sampling gaps


def load_qpcr(data_dir=DATA):
    q = {}
    for row in csv.DictReader(open(f"{data_dir}/qpcr.tsv"), delimiter="\t"):
        vals = [float(row[k]) for k in row
                if k != "sampleID" and row[k] not in ("", "NA", None)]
        if vals:
            q[row["sampleID"]] = float(np.mean(vals))
    return q


def build_design(data_dir=DATA, subjects=(1, 2, 3, 4, 5)):
    rel, subj, time, sids = load(data_dir)
    qpcr = load_qpcr(data_dir)

    top = np.argsort(rel.mean(0))[::-1][:S_TAXA]
    load_vec = np.array([qpcr.get(s, np.nan) for s in sids])

    # absolute abundances of the retained taxa (cells/g, arbitrary scale)
    Xabs = rel[:, top] * load_vec[:, None]
    scale = np.nanmedian(np.where(Xabs > 0, Xabs, np.nan), axis=0)
    Xs = Xabs / scale                       # each taxon ~ O(1)

    # perturbation indicators per sample (subjects 2-5 only)
    pert = np.zeros((len(sids), len(PERTURBS)))
    for j, (_, s, e) in enumerate(PERTURBS):
        for i in range(len(sids)):
            if subj[i] in subjects and s <= time[i] <= e:
                pert[i, j] = 1.0

    # assemble transitions within each subject
    rows_y, rows_Z = [], []
    eps = 1e-3
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
            g = (np.log(xb + eps) - np.log(xa + eps)) / dt      # per-taxon growth proxy
            z = np.concatenate([[1.0], xa, pert[a]])            # intercept + abund + field
            rows_y.append(g)
            rows_Z.append(z)
    Y = np.array(rows_y)                     # n_trans x S
    Z = np.array(rows_Z)                     # n_trans x (1+S+P)
    return Y, Z


def fit_matrix(Y, Z):
    """Ridge regression; return interaction matrix J (S x S), normalized so the
    typical self-limitation is -1."""
    S = Y.shape[1]
    A = Z.T @ Z + RIDGE * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ Y)       # (1+S+P) x S
    J = coef[1:1 + S].T                       # S x S : row i = effects on taxon i
    diag = np.diag(J)
    self_scale = np.median(-diag[diag < 0]) if np.any(diag < 0) else 1.0
    if self_scale <= 0:
        self_scale = 1.0
    return J / self_scale


def disorder_stats(J):
    S = J.shape[0]
    off = J[~np.eye(S, dtype=bool)]
    iu = np.triu_indices(S, 1)
    a, b = J[iu], J.T[iu]
    gamma = float(np.corrcoef(a, b)[0, 1])
    mu = float(-off.mean() * S)
    sigma = float(off.std() * np.sqrt(S))
    return mu, sigma, gamma


def percentile_ci(theta_hat, boot, alpha=0.05):
    """Plain percentile bootstrap CI over resampled transitions (N resamples).

    The interval is the [alpha/2, 1-alpha/2] percentiles of the bootstrap
    distribution, reported as-is. We do NOT widen it to contain the full-sample
    estimate: when a skewed/biased resampling distribution of a correlation like
    gamma leaves theta_hat just outside its own bootstrap interval, that is left
    visible as an honest flag of an unstable estimate rather than hidden. The
    theta_hat argument is kept for signature compatibility and is unused. Returns
    (lo, hi)."""
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    if boot.size < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# Backward-compatible alias for the previous (widening) name; callers now get the
# honest percentile CI.
bc_ci = percentile_ci


def bootstrap(Y, Z, n=300):
    rng = np.random.default_rng(0)
    nt = Y.shape[0]
    mus, sigmas, gammas = [], [], []
    for _ in range(n):
        s = rng.integers(0, nt, nt)
        J = fit_matrix(Y[s], Z[s])
        m, sg, g = disorder_stats(J)
        mus.append(m); sigmas.append(sg); gammas.append(g)
    return (np.array(mus), np.array(sigmas), np.array(gammas))


def infer(data_dir=DATA, subjects=(1, 2, 3, 4, 5)):
    """Full pipeline for one cohort -> dict of disorder statistics + bootstraps."""
    Y, Z = build_design(data_dir, subjects)
    J = fit_matrix(Y, Z)
    mu, sigma, gamma = disorder_stats(J)
    bm, bs, bg = bootstrap(Y, Z)
    off = J[~np.eye(J.shape[0], dtype=bool)]
    return dict(J=J, n_trans=Y.shape[0], mu=mu, sigma=sigma, gamma=gamma,
                boot_mu=bm, boot_sigma=bs, boot_gamma=bg,
                frac_neg=float((off < 0).mean()))


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    Y, Z = build_design()
    print(f"transitions used: {Y.shape[0]}  taxa: {Y.shape[1]}")
    J = fit_matrix(Y, Z)
    mu, sigma, gamma = disorder_stats(J)
    bm, bs, bg = bootstrap(Y, Z)
    mlo, mhi = bc_ci(mu, bm); slo, shi = bc_ci(sigma, bs); glo, ghi = bc_ci(gamma, bg)
    print(f"  mu    = {mu:6.2f}   (competition; BC 95% CI [{mlo:.2f}, {mhi:.2f}])")
    print(f"  sigma = {sigma:6.2f}   (disorder;   BC 95% CI [{slo:.2f}, {shi:.2f}])")
    print(f"  gamma = {gamma:6.2f}   (symmetry;   BC 95% CI [{glo:.2f}, {ghi:.2f}])")
    frac_neg = float((J[~np.eye(J.shape[0], dtype=bool)] < 0).mean())
    print(f"  fraction of competitive (negative) interactions: {frac_neg:.2f}")

    np.savez(f"{OUT}/inferred_stats.npz", J=J, mu=mu, sigma=sigma, gamma=gamma,
             boot_sigma=bs, boot_gamma=bg)
    print("[saved] figures/inferred_stats.npz")


if __name__ == "__main__":
    main()
