r"""
Clark's sign-discordance against a magnitude-preserving symmetrizable surrogate
===============================================================================
Correction (g) of the response letter, and the Results sentence that carries it, say that
Clark's f_sign "survives a magnitude-preserving symmetrizable surrogate but its p then ranges
from 2e-4 to 0.022 with the noise model". That range has no source in the repository. It was
produced by null_audit2.py, which is absent, and scripts/cycle_null_replicated.py refers to
that file by name four times, including a comment saying its surrogate is "reproduced here so
this script is self-contained". The surrogate was carried over; the Clark reading was not.

What the published null does, and why a second one was needed. scripts/symmetrizability_
invariants.py builds its null from symmetric_null(), which matches only the off-diagonal
density and draws magnitudes from the empirical distribution with random signs and random
support. A referee can object that a surrogate free to place its own support and magnitudes
is easier to exceed than the observed matrix. The severe surrogate removes that freedom:
full_symmetrizable() keeps the observed support, the observed signs up to concordance and the
observed magnitudes, and is exactly symmetrizable, so f_oneway = f_sign = r_cycle = 0.

Noise. Two models, as the letter says, because the answer depends on which one is assumed:
    additive       J = A + nu * scale * N(0,1) on the modelled coefficients
    proportional   J = A * (1 + nu * N(0,1))
Both are the project's own functions: corrupt() from symmetrizability_invariants.py and
corrupt_mult() from cycle_null_replicated.py, the latter copied in because that module runs
its whole analysis at import.

Calibration. As elsewhere in the paper, nu is chosen so that the null reproduces the spread
of Clark's published posterior coupling-correlation draws, data/clark2021/
gamma_posterior_samples.csv. The calibration is done separately for each noise model, on the
severe surrogate rather than on symmetric_null, because the two have different gamma spreads
at the same nu.

Statistic and p. f_sign is the fraction of bidirectional pairs whose two directions carry
discordant signs. The Monte-Carlo p is (k + 1) / (B + 1) with k the number of null draws
reaching the observed value, so the floor at B = 5000 is 2e-4, which is the lower end the
letter quotes.

Criterion, fixed before running.
  * The claim stands if the two noise models bracket p inside 2e-4 to 0.022, to the precision
    quoted, with the observed f_sign exceeding the null in both.
  * The claim needs correcting to the measured values, not withdrawing, if the exceedance
    holds in both models at a different p.
  * The claim must be withdrawn if either model gives p above 0.05, because the letter
    presents the Clark channel as surviving this surrogate.

Run from the repository root:
    PYTHONPATH=scripts python3 scripts/clark_fsign_severe_null.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from symmetrizability_invariants import invariants, corrupt, gamma_pearson  # noqa: E402

SEED = 20261105
B = 5000
NCAL = 300
GRID = np.concatenate([np.linspace(0.05, 2.0, 40), np.linspace(2.2, 8.0, 30)])
CLAIMED = (2e-4, 0.022)


def corrupt_mult(A, nu, rng):
    """Inference noise proportional to each coefficient. Copied from
    scripts/cycle_null_replicated.py, which cannot be imported: it runs at module level."""
    off = ~np.eye(len(A), dtype=bool)
    J = A.copy(); mask = off & (np.abs(A) > 0)
    J[mask] = A[mask] * (1.0 + nu * rng.standard_normal(mask.sum()))
    return J


def full_symmetrizable(J):
    """Exactly symmetrizable surrogate preserving support, signs up to concordance and
    magnitudes. Copied from scripts/cycle_null_replicated.py for the same reason."""
    n = len(J); iu = np.triu_indices(n, 1)
    a, b = J[iu].copy(), J.T[iu].copy()
    inter = (a != 0) | (b != 0)
    ii, jj = iu[0][inter], iu[1][inter]
    aa, bb = a[inter], b[inter]
    s = np.where(np.abs(aa) >= np.abs(bb), np.sign(aa), np.sign(bb)); s[s == 0] = 1.0
    am = np.where(aa != 0, np.abs(aa), np.abs(bb))
    bm = np.where(bb != 0, np.abs(bb), np.abs(aa))
    gm = np.sqrt(am * bm)
    E = len(ii)
    Bm = np.zeros((E, n)); Bm[np.arange(E), jj] = 1; Bm[np.arange(E), ii] = -1
    u, *_ = np.linalg.lstsq(Bm, np.log(am / bm), rcond=None)
    fit = Bm @ u
    A = np.zeros_like(J); np.fill_diagonal(A, np.diag(J))
    A[ii, jj] = s * gm * np.exp(fit / 2)
    A[jj, ii] = s * gm * np.exp(-fit / 2)
    return A


C = np.loadtxt(ROOT / "data/clark2021/interaction_matrix_RLC8_posteriormean.csv", delimiter=",")
gpost = np.loadtxt(ROOT / "data/clark2021/gamma_posterior_samples.csv")
GSTD = float(np.std(gpost))

f1, fs_obs, rc_obs, E = invariants(C)
print(f"Clark observed: f_oneway={f1:.3f} f_sign={fs_obs:.3f} r_cycle={rc_obs:.3f} "
      f"on {E} concordant edges")
print(f"Clark posterior coupling-correlation spread: {GSTD:.3f} (n={len(gpost)})\n")

A = full_symmetrizable(C)
ia = invariants(A)
print(f"severe surrogate: f_oneway={ia[0]:.3g} f_sign={ia[1]:.3g} r_cycle={ia[2]:.3e} "
      f"on {ia[3]} edges")
assert ia[1] < 1e-12 and ia[2] < 1e-10, "the surrogate is not exactly symmetrizable"
same_support = np.array_equal(C != 0, A != 0)
print(f"support preserved: {same_support}; magnitudes preserved as geometric means of "
      f"each reciprocal pair\n")

MODELS = {"additive": corrupt, "proportional": corrupt_mult}
rows = {}
for name, fn in MODELS.items():
    # calibrate nu so the null reproduces the posterior coupling-correlation spread
    rng = np.random.default_rng(SEED)
    best, bnu, bspread = np.inf, GRID[0], float("nan")
    for nu in GRID:
        gm = [gamma_pearson(fn(A, nu, rng)) for _ in range(NCAL)]
        sd = float(np.std(gm))
        d = abs(sd - GSTD)
        if d < best:
            best, bnu, bspread = d, float(nu), sd
    on_edge = bnu >= GRID[-1] - 1e-9
    print(f"  {name:>12}: calibrated nu={bnu:.3f}, null coupling-correlation spread "
          f"{bspread:.4f} against the posterior's {GSTD:.4f}"
          f"{'  [ON THE GRID EDGE: the calibration is a lower bound on the noise]' if on_edge else ''}")
    rng = np.random.default_rng(SEED + 1)
    draws = np.array([invariants(fn(A, bnu, rng))[1] for _ in range(B)])
    k = int(np.sum(draws >= fs_obs))
    p = (k + 1) / (B + 1)
    rows[name] = dict(nu=bnu, null_gamma_spread=bspread, nu_on_grid_edge=bool(on_edge), p=float(p), k=k, B=B,
                      null_median=float(np.median(draws)), null_q95=float(np.quantile(draws, .95)),
                      observed=float(fs_obs))
    print(f"  {name:>12}: nu={bnu:.3f}  null median f_sign {np.median(draws):.3f}  "
          f"q95 {np.quantile(draws,.95):.3f}  observed {fs_obs:.3f}  "
          f"p={p:.4g}  ({k}/{B} exceedances)")

ps = [r["p"] for r in rows.values()]
lo, hi = min(ps), max(ps)
print(f"\n  range over the two noise models: {lo:.4g} to {hi:.4g}")
print(f"  letter claims: {CLAIMED[0]:.4g} to {CLAIMED[1]:.4g}")
if abs(lo - CLAIMED[0]) < 1e-4 and abs(hi - CLAIMED[1]) < 2e-3:
    verdict = "claim stands as written"
elif hi < 0.05:
    verdict = f"exceedance holds in both models; the letter should quote {lo:.4g} to {hi:.4g}"
else:
    verdict = "claim must be withdrawn: one noise model leaves the channel non-significant"
print(f"  VERDICT: {verdict}")

out = ROOT / "manuscript/revision_BIB/finale/analisi" / (Path(__file__).stem + ".json")
out.write_text(json.dumps(dict(
    seed=SEED, B=B, ncal=NCAL, clark_gamma_std=GSTD,
    observed=dict(f_oneway=f1, f_sign=fs_obs, r_cycle=rc_obs, edges=E),
    surrogate=dict(f_oneway=ia[0], f_sign=ia[1], r_cycle=ia[2], edges=ia[3],
                   support_preserved=bool(same_support)),
    models=rows, measured_range=[lo, hi], claimed_range=list(CLAIMED), verdict=verdict,
), indent=2), encoding="utf-8")
print(f"\n  wrote {out.relative_to(ROOT)}")
