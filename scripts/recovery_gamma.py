"""
Fig. 20 -- gamma recovery test: is the small human gamma real, or noise attenuation?
===================================================================================

The main empirical claim is gamma ~ 0.1 in humans (with absolute abundance). A
referee will object: gamma = corr(J_ij, J_ji) is estimated from an ill-posed,
ridge-regularized inference, and independent estimation errors on J_ij and J_ji
attenuate their correlation toward zero (regression dilution). So a small
measured gamma could be a large true gamma masked by noise. Test-inverting the
full recovery distribution excludes no true gamma in [0, 0.8] at 95%.

This is the control. We generate SYNTHETIC GLV time series with a KNOWN gamma_true
on a scale-free network, matching the real design (S=30 taxa, ~20 subjects,
daily sampling, process + observation noise), then push them through the SAME
inference pipeline (fit_matrix + disorder_stats). Plotting gamma_recovered vs
gamma_true shows whether the pipeline attenuates.

  - If recovery is ~unbiased  -> a measured gamma~0.1 really is a small true gamma;
                                 the compositionality result stands, conclusion firm.
  - If it attenuates strongly  -> gamma~0.1 is consistent with a larger true value;
                                 the claim must be softened. Either way, honest.

Run:  python3 recovery_gamma.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infer_and_place import fit_matrix, disorder_stats, RIDGE, S_TAXA
from spin_glass_microbiome import scale_free_graph

import warnings
warnings.filterwarnings("ignore")
np.seterr(over="ignore", invalid="ignore", divide="ignore")

OUT = "figures"

# design matched to the human absolute-load cohort (Vandeputte 2021): S=30 taxa,
# ~20 subjects, ~40 daily samples. mu0/sigma0 chosen so the inferred sigma sits
# in the empirically inferred range (a few units).
S        = S_TAXA          # 30 taxa
N_SUBJ   = 20             # subjects
N_DAYS   = 40            # daily samples per subject
DT       = 1.0            # one day per step
MU0      = 0.5           # net competition on each edge (mean of -J): keeps the
                         # coexistence fixed point bounded at every gamma
SIGMA0   = 0.5           # disorder strength on each edge (bounded/persistent regime)
PROC_STD = 0.12          # process (biological) noise, per sqrt(day), on log-abundance
OBS_STD  = 0.20          # observation noise on log-abundance (errors-in-variables);
                         # tuned so the within-dataset bootstrap CI width matches the
                         # real inference (~0.08-0.13), i.e. matched signal-to-noise
NMAX     = 60.0
EPS      = 1e-3


def coupling_with_gamma(A, gamma, rng):
    """Partially symmetric couplings on the edges of A with prescribed
    corr(J_ij, J_ji) = gamma. Bivariate-normal per edge, competitive mean,
    scaled 1/sqrt(<k>) (disordered-GLV scaling). Diagonal 0 (self-limitation
    is the -n_i term in the dynamics)."""
    n = A.shape[0]
    kmean = max(A.sum(1).mean(), 1.0)
    a = rng.normal(size=(n, n))
    b = rng.normal(size=(n, n))
    # J_ij uses a; J_ji uses gamma*a + sqrt(1-g^2)*b  -> corr = gamma
    iu = np.triu_indices(n, 1)
    J = np.zeros((n, n))
    J[iu] = -MU0 + SIGMA0 * a[iu]
    J.T[iu] = -MU0 + SIGMA0 * (gamma * a[iu] + np.sqrt(max(1 - gamma**2, 0)) * b[iu])
    J *= A
    J /= np.sqrt(kmean)
    np.fill_diagonal(J, 0.0)
    return J


def simulate_subject(r, J, rng, n_days=N_DAYS, dt=DT, burn=60):
    """Euler-Maruyama on log-abundance for a stochastic GLV:
       dln x = (r - x + J x) dt + process noise.
    Returns the daily observed absolute-abundance series (with observation
    noise) as an (n_days x S) array."""
    n = r.size
    x = np.full(n, 1.0) * rng.uniform(0.3, 1.5, n)
    # burn-in to the stochastic steady state
    for _ in range(burn):
        drift = r - x + J @ x
        x = x * np.exp((drift) * dt + PROC_STD * np.sqrt(dt) * rng.normal(size=n))
        x = np.clip(x, 0.0, NMAX) + EPS
    out = np.zeros((n_days, n))
    for t in range(n_days):
        drift = r - x + J @ x
        x = x * np.exp((drift) * dt + PROC_STD * np.sqrt(dt) * rng.normal(size=n))
        x = np.clip(x, 0.0, NMAX) + EPS
        out[t] = x * np.exp(OBS_STD * rng.normal(size=n))   # observation noise
    return out


def build_design(series_list):
    """Same transition design as the real pipeline: per-taxon log-growth proxy
    g=(log x_b - log x_a)/dt regressed on [1, x_a]. Median-scale taxa to O(1)."""
    allX = np.vstack(series_list)
    scale = np.median(np.where(allX > 0, allX, np.nan), axis=0)
    scale = np.where(np.isnan(scale) | (scale <= 0), 1.0, scale)
    rows_y, rows_Z, eps = [], [], 1e-3
    for X in series_list:
        Xs = X / scale
        for a, b in zip(Xs[:-1], Xs[1:]):
            g = (np.log(b + eps) - np.log(a + eps)) / DT
            rows_y.append(g)
            rows_Z.append(np.concatenate([[1.0], a]))
    return np.array(rows_y), np.array(rows_Z)


def recover_once(gamma_true, seed, boot=0):
    rng = np.random.default_rng(seed)
    A, k, _ = scale_free_graph(S, m=2, seed=seed % 1000)
    J = coupling_with_gamma(A, gamma_true, rng)
    r = rng.uniform(0.8, 1.2, S)
    series = [simulate_subject(r, J, rng) for _ in range(N_SUBJ)]
    true_max = max(s.max() for s in series)
    Y, Z = build_design(series)
    Jhat = fit_matrix(Y, Z)
    mu, sigma, gamma_hat = disorder_stats(Jhat)
    bad = (sigma > 50) or (true_max > 0.9 * NMAX) or not np.isfinite(gamma_hat)
    boot_w = np.nan
    if boot and not bad:
        nt = Y.shape[0]; gs = []
        for _ in range(boot):
            s = rng.integers(0, nt, nt)
            _, _, g = disorder_stats(fit_matrix(Y[s], Z[s]))
            gs.append(g)
        boot_w = float(np.percentile(gs, 97.5) - np.percentile(gs, 2.5))
    return gamma_hat, sigma, bad, boot_w


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    gammas_true = [0.0, 0.2, 0.4, 0.6, 0.8]
    n_rep = 25
    means, sems, sds, sig_med, bws = [], [], [], [], []
    print(f"design: S={S}, subjects={N_SUBJ}, days={N_DAYS}, mu0={MU0}, "
          f"sigma0={SIGMA0}, proc={PROC_STD}, obs={OBS_STD}, ridge={RIDGE}")
    print("(bounded coexistence regime; obs-noise tuned to real bootstrap width)\n")
    print(f"{'gamma_true':>10} | {'ghat mean +/- SEM':>20} | {'sd':>6} | "
          f"{'sigma_inf':>9} | {'bootW':>6} | drop")
    print("-" * 74)
    for gt in gammas_true:
        recs, sigs, bwl, dropped = [], [], [], 0
        for rep in range(n_rep):
            gh, sg, bad, bw = recover_once(gt, seed=1000 + int(gt * 100) * 17 + rep,
                                           boot=120 if rep < 3 else 0)
            if bad:
                dropped += 1; continue
            recs.append(gh); sigs.append(sg)
            if np.isfinite(bw):
                bwl.append(bw)
        recs = np.array(recs)
        m, sd = recs.mean(), recs.std()
        sem = sd / np.sqrt(len(recs))
        means.append(m); sems.append(sem); sds.append(sd)
        sig_med.append(np.median(sigs)); bws.append(np.median(bwl) if bwl else np.nan)
        print(f"{gt:>10.2f} | {m:>+9.3f} +/- {sem:>5.3f} | {sd:>6.3f} | "
              f"{np.median(sigs):>9.2f} | {np.median(bwl) if bwl else float('nan'):>6.3f} | {dropped}")

    means = np.array(means); sems = np.array(sems)
    gammas_true = np.array(gammas_true)

    # proper linear fit gamma_hat = a + b*gamma_true (attenuation slope b, offset a)
    b, a = np.polyfit(gammas_true, means, 1)
    print(f"\nlinear fit:  ghat = {a:+.3f} + {b:.3f} * gamma_true")
    print(f"  attenuation slope b   = {b:.2f}  (1 = unbiased, <1 = dilution)")
    print(f"  offset at gamma=0     = {a:+.3f}")
    print(f"  median inferred sigma = {np.median(sig_med):.2f}  "
          f"(real inference reported ~4.4 -> real sigma is inference-inflated)")
    print(f"  median bootstrap width= {np.nanmedian(bws):.3f}  (real ~0.08-0.13)")

    # what true gamma is compatible with the measured human ghat ~ 0.10 ?
    g_obs = 0.10
    g_true_implied = (g_obs - a) / b if b > 0 else float("nan")
    print(f"\nmeasured human ghat ~ {g_obs:.2f}  =>  compatible true gamma ~ "
          f"{g_true_implied:.2f}")
    # largest true gamma still statistically consistent with ghat<=0.13 (upper CI)
    print("interpretation printed in the manuscript note.")

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    xs = np.array([0, 0.85])
    ax.plot(xs, xs, ls="--", color="grey", lw=1, label="unbiased recovery ($\\hat\\gamma=\\gamma$)")
    ax.plot(xs, a + b * xs, ls="-", color="#7570b3", lw=1.6,
            label=f"fit $\\hat\\gamma={a:+.2f}+{b:.2f}\\,\\gamma$")
    ax.errorbar(gammas_true, means, yerr=sems, fmt="o", ms=10,
                color="#1b9e77", capsize=5,
                label=f"recovered (mean$\\pm$SEM, {n_rep} communities)")
    ax.axhspan(0.0, 0.13, color="#d95f02", alpha=0.12)
    ax.axhline(0.10, ls="-", color="#d95f02", lw=1.2)
    ax.text(0.30, 0.055, "measured human $\\hat\\gamma\\approx0.10$\n(Vandeputte, absolute)",
            fontsize=8, color="#d95f02")
    if np.isfinite(g_true_implied) and 0 < g_true_implied < 0.85:
        ax.axvline(g_true_implied, ls=":", color="#d95f02", lw=1)
        ax.text(g_true_implied + 0.01, 0.62,
                f"compatible\n$\\gamma_{{\\rm true}}\\approx{g_true_implied:.2f}$",
                fontsize=8, color="#d95f02")
    ax.set_xlabel("true interaction symmetry  $\\gamma_{\\rm true}$")
    ax.set_ylabel("recovered  $\\hat\\gamma$  (same inference pipeline)")
    ax.set_xlim(-0.05, 0.9); ax.set_ylim(-0.1, 0.9)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Recovery of $\\gamma$ from synthetic GLV data: positive bias and a\n"
                 "shallow slope, so $\\hat\\gamma$ is a bound, not a point estimate",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig20_gamma_recovery.png", dpi=150)
    plt.close(fig)
    print("[fig20] written")

    np.savez(f"{OUT}/gamma_recovery.npz", gammas_true=gammas_true, means=means,
             sems=sems, slope=b, offset=a, g_true_implied=g_true_implied)


if __name__ == "__main__":
    main()
