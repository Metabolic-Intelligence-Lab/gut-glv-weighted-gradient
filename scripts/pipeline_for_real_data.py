"""
Analysis pipeline for REAL longitudinal microbiome data
=======================================================

Turns the theory's predictions into measurements on an actual abundance time
series (e.g. the Dethlefsen-Relman ciprofloxacin OTU table, or David et al.
2014). Point `--csv` at a table with:

    rows    = samples (ordered in time)
    columns = taxa/OTU relative abundances
    plus a 'day' column (numeric time) and an optional 'phase' column
          (e.g. 'pre', 'abx', 'post')

and it computes the same diagnostics used on the model:

  1. State trajectory:    Bray-Curtis distance from the pre-perturbation baseline
  2. Resilience vs hysteresis:  does the post-perturbation state return to the
                          baseline cloud, or settle in a new region? (return ratio)
  3. Early-warning signal:  rolling variance + lag-1 autocorrelation of the state
                          (critical slowing down; Scheffer 2009, Dakos 2012)
  4. (optional) interaction inference: a sparse ridge gLV fit to estimate J,
                          then test symmetry and locate the subject on the
                          phase diagram (mu, sigma, gamma).

This file is intentionally dependency-light (numpy + optional pandas). It is the
bridge from the in-silico proof-of-concept (proof_of_concept.py) to a direct
empirical test. No real table is bundled here.

Usage:
    python3 pipeline_for_real_data.py --csv abundances.csv \
        --time-col day --baseline-end 0 --perturb-end 30
"""

import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Core diagnostics (pure numpy; X is samples x taxa relative-abundance matrix)
# ---------------------------------------------------------------------------
def bray_curtis_to_baseline(X, baseline_idx):
    """Bray-Curtis distance of every sample to the mean baseline composition."""
    b = X[baseline_idx].mean(0)
    num = np.abs(X - b).sum(1)
    den = X.sum(1) + b.sum()
    return num / np.where(den == 0, 1, den)


def return_ratio(d, baseline_idx, recovery_idx):
    """1 = full return to baseline distance, 0 = stays as far as the peak.
    Distinguishes resilience (->1) from hysteresis (->0)."""
    base_level = d[baseline_idx].mean()
    peak = d.max()
    recov = d[recovery_idx].mean()
    if peak - base_level < 1e-9:
        return 1.0
    return float(np.clip((peak - recov) / (peak - base_level), 0, 1))


def early_warning(d, win=8):
    """Rolling variance and lag-1 autocorrelation of the state series."""
    var = np.full_like(d, np.nan, dtype=float)
    ac1 = np.full_like(d, np.nan, dtype=float)
    for i in range(win, len(d)):
        w = d[i - win:i]
        var[i] = np.var(w)
        if w[:-1].std() > 0 and w[1:].std() > 0:
            ac1[i] = np.corrcoef(w[:-1], w[1:])[0, 1]
    return var, ac1


def infer_glv(X, days, ridge=1e-2):
    """Crude discrete-time gLV inference (log-ratio regression) to estimate the
    interaction matrix J, then report disorder statistics and symmetry.

        log(x_{t+1}/x_t)/dt ~ r + (J - I) x_t
    Solved per taxon by ridge regression. Returns J and a dict of summaries.
    """
    eps = 1e-6
    X = np.clip(X, eps, None)
    dt = np.diff(days).astype(float)
    dt[dt == 0] = 1.0
    Y = (np.log(X[1:]) - np.log(X[:-1])) / dt[:, None]   # growth-rate proxy
    P = X[:-1]                                            # predictors
    S = X.shape[1]
    Z = np.hstack([np.ones((P.shape[0], 1)), P])         # intercept + abundances
    A = Z.T @ Z + ridge * np.eye(Z.shape[1])
    B = Z.T @ Y
    coef = np.linalg.solve(A, B)                         # (1+S) x S
    J = coef[1:].T                                       # S x S interaction matrix
    off = J[~np.eye(S, dtype=bool)]
    sym = float(np.corrcoef(J[np.triu_indices(S, 1)],
                            J.T[np.triu_indices(S, 1)])[0, 1])
    return J, dict(mu=float(-off.mean() * S),
                   sigma=float(off.std() * np.sqrt(S)),
                   gamma=sym)


def classify_subject(return_r, ew_rising, gamma):
    if return_r > 0.7:
        return "RESILIENT (returns to baseline)"
    if gamma is not None and gamma < 0.2:
        return "ASYMMETRIC regime (persistent dynamics likely; glass language weak)"
    return "HYSTERETIC (incomplete recovery; trapped in alternative stable state)"


def run(X, days, baseline_idx, recovery_idx, do_infer=False):
    d = bray_curtis_to_baseline(X, baseline_idx)
    rr = return_ratio(d, baseline_idx, recovery_idx)
    var, ac1 = early_warning(d)
    ew_rising = np.nanmax(var) > 3 * np.nanmedian(var[~np.isnan(var)] + 1e-12)
    gamma = None
    if do_infer:
        _, stats = infer_glv(X, days)
        gamma = stats["gamma"]
        print(f"  inferred disorder: mu={stats['mu']:.2f}, sigma={stats['sigma']:.2f}, "
              f"symmetry gamma={stats['gamma']:.2f}")
    print(f"  return ratio = {rr:.2f}  (1=resilient, 0=hysteretic)")
    print(f"  early-warning (rising variance) = {ew_rising}")
    print(f"  -> {classify_subject(rr, ew_rising, gamma)}")
    return dict(distance=d, return_ratio=rr, variance=var, autocorr=ac1)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--time-col", default="day")
    ap.add_argument("--baseline-end", type=float, default=0.0,
                    help="days <= this are pre-perturbation baseline")
    ap.add_argument("--perturb-end", type=float, default=30.0,
                    help="recovery window is days > this")
    ap.add_argument("--infer", action="store_true")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas required for CSV loading: pip install pandas")
    df = pd.read_csv(args.csv).sort_values(args.time_col)
    days = df[args.time_col].to_numpy()
    taxa = [c for c in df.columns if c not in (args.time_col, "phase", "subject")]
    X = df[taxa].to_numpy(dtype=float)

    base_idx = np.where(days <= args.baseline_end)[0]
    recov_idx = np.where(days > args.perturb_end)[0]
    if len(base_idx) == 0 or len(recov_idx) == 0:
        raise SystemExit("baseline or recovery window empty; check thresholds")
    print(f"Loaded {X.shape[0]} samples x {X.shape[1]} taxa")
    run(X, days, base_idx, recov_idx, do_infer=args.infer)


if __name__ == "__main__":
    main()
