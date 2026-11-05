r"""
The saturation invariance of r_cycle, and the pair-specific case that breaks it
==============================================================================
The manuscript states two properties of r_cycle (Results, and the S1 Text section on the
compositional gauge):

  (i)  it is unchanged by a saturating response shared across species, "exactly to machine
       precision up to a 50% reduction in response", so the cycle channel does not assume
       mass-action kinetics;
  (ii) the invariance fails if each directed interaction saturates differently, which
       "manufactures an apparent r_cycle of 0.011 to 0.074 under mild saturation, and of
       0.24 to 0.66 once saturation exceeds about 10% and the constants spread by a
       factor e".

No script in the repository produced either number: the generating script was lost with a
session scratchpad, and a repository-wide search for "satur" in scripts/ returns nothing,
while Table 3 of the manuscript claims every analysis has one. This script supplies the
source, and it does not fully confirm (ii). See the verdict at the end.

Why (i) is exact rather than approximate. r_cycle is the RMS residual of the least-squares
fit of b_ij = log|J_ij / J_ji| to node differences u_j - u_i on the reciprocal
sign-concordant subgraph (scripts/symmetrizability_invariants.py, invariants()). A response
phi shared by all species enters the measured coefficient of species j on species i as a
factor f_j depending on j alone, so

    b'_ij = log|J_ij f_j / (J_ji f_i)| = b_ij + log f_j - log f_i,

itself a node difference. The design matrix of the fit spans every node difference, so the
perturbation lies exactly in its column space and the residual is untouched. Support and
signs are untouched too, every f_j being positive, so f_oneway and f_sign are invariant as
well. The invariance therefore holds for ANY positive per-species factor vector, which is
what this script tests: random factor vectors are a stronger test than any one saturating
response.

Why pair-specific saturation breaks it. With a constant K_ij per directed interaction the
factor is f_ij, and b'_ij = b_ij + log f_ij - log f_ji is no longer a node difference. What
is left over is what the fit reads as cycle inconsistency.

Model. Michaelis-Menten response, f = 1 / (1 + x/K). The reduction in response at the
operating abundance is rho = (x/K) / (1 + x/K), so fixing rho fixes x/K = rho/(1-rho).
Pair-specific constants are K_ij = K0 exp(z_ij), and the phrase "the constants spread by a
factor e" is ambiguous about z, which is why the published range could not be reproduced
blind. Both readings are computed:

    lognormal   z ~ Normal(0, 1):      one log standard deviation is a factor e
    uniform     z ~ Uniform(-0.5, 0.5): the full range is a factor e

The lognormal reading is the one the text will quote, because multiplicative constants
across pairs are naturally lognormal and because it is the reading that reproduces the
published order of magnitude. The uniform reading is reported as the sensitivity.

Baseline. The truth is the exactly symmetrizable surrogate of the published Venturelli
matrix: it preserves support, signs up to concordance and magnitudes, and has r_cycle = 0
exactly. Any r_cycle measured after perturbing it is manufactured, which is the claim under
test.

Criterion, fixed before running.
  * (i) passes if the relative change in r_cycle stays below 1e-10 at every reduction level
    up to 50% over all random factor vectors. A larger change falsifies the algebra and the
    sentence must go.
  * (ii) stands as written if the measured ranges contain the published 0.011 to 0.074 and
    0.24 to 0.66 under at least one stated reading of the spread.
  * If a reading reproduces the qualitative picture at different values, the text takes the
    measured values and names the reading, so the number stops being unreproducible.

Run from the repository root:
    PYTHONPATH=scripts python3 scripts/saturation_pairwise.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from symmetrizability_invariants import invariants  # noqa: E402


def full_symmetrizable(J):
    """Exactly symmetrizable surrogate preserving support, signs up to concordance and
    magnitudes: f_oneway = f_sign = r_cycle = 0.

    Copied from scripts/cycle_null_replicated.py, which cannot be imported because it runs
    its whole analysis at module level. That file carries the same function for the same
    reason, so this follows the project's own precedent rather than adding a duplication.
    """
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
    B = np.zeros((E, n)); B[np.arange(E), jj] = 1; B[np.arange(E), ii] = -1
    u, *_ = np.linalg.lstsq(B, np.log(am / bm), rcond=None)
    fit = B @ u
    A = np.zeros_like(J); np.fill_diagonal(A, np.diag(J))
    A[ii, jj] = s * gm * np.exp(fit / 2)
    A[jj, ii] = s * gm * np.exp(-fit / 2)
    return A


SEED = 20261105
NDRAW = 400
MILD = 0.05                          # "mild saturation": a 5% reduction in response
STRONG = (0.15, 0.25, 0.35, 0.50)    # "saturation exceeds about 10%"
SHARED_LEVELS = (0.10, 0.20, 0.30, 0.40, 0.50)
CLAIMED = {"mild": (0.011, 0.074), "strong": (0.24, 0.66)}

V = np.loadtxt(ROOT / "data/venturelli2018/interaction_matrix_averaged.csv",
               delimiter=",", skiprows=1)
A = full_symmetrizable(V)
S = len(A)
off = ~np.eye(S, dtype=bool)
base = invariants(A)
print(f"baseline symmetrizable truth: f_oneway={base[0]:.3g} f_sign={base[1]:.3g} "
      f"r_cycle={base[2]:.3e} on {base[3]} concordant edges\n")
assert base[2] < 1e-12, "the surrogate is not exactly symmetrizable; nothing below is valid"

rng = np.random.default_rng(SEED)


def draw_z(model):
    if model == "lognormal":
        return rng.normal(0.0, 1.0, (S, S))
    return rng.uniform(-0.5, 0.5, (S, S))


# ---------------------------------------------------------------- (i) shared response
print("(i) a saturating response shared across species")
shared, worst = [], 0.0
for rho in SHARED_LEVELS:
    xK = rho / (1.0 - rho)
    devs = []
    for _ in range(NDRAW):
        # each species sits at its own abundance, so one shared response still gives each
        # species its own factor: the case where node cancellation has to do the work
        f = 1.0 / (1.0 + rng.uniform(0.25, 4.0, S) * xK)
        M = A.copy(); M[off] = (A * f[None, :])[off]
        inv = invariants(M)
        devs.append(abs(inv[2] - base[2]))
        assert inv[0] == base[0] and inv[1] == base[1], "support or signs moved"
    m = float(np.max(devs))
    worst = max(worst, m)
    shared.append(dict(reduction=rho, max_abs_change=m))
    print(f"    reduction {rho:>4.0%}: largest change in r_cycle over {NDRAW} draws {m:.2e}")
inv_ok = worst < 1e-10
print(f"    -> largest change at any level: {worst:.2e}  "
      f"({'INVARIANT to machine precision' if inv_ok else 'NOT INVARIANT'})\n")


# ---------------------------------------------------------------- (ii) pair-specific
def pair_specific(rho, model, ndraw=NDRAW):
    xK0 = rho / (1.0 - rho)
    vals = []
    for _ in range(ndraw):
        F = 1.0 / (1.0 + xK0 * np.exp(-draw_z(model)))
        M = A.copy(); M[off] = (A * F)[off]
        vals.append(invariants(M)[2])
    return np.array(vals)


results = {}
for model, label in (("lognormal", "lognormal, one log sd = factor e"),
                     ("uniform", "uniform, full range = factor e")):
    print(f"(ii) pair-specific saturation, constants {label}")
    mild = pair_specific(MILD, model)
    print(f"    mild, {MILD:.0%} reduction: {mild.min():.3f} to {mild.max():.3f} "
          f"(median {np.median(mild):.3f})")
    rows, pool = [], []
    for rho in STRONG:
        v = pair_specific(rho, model)
        pool.append(v)
        rows.append(dict(reduction=rho, lo=float(v.min()), hi=float(v.max()),
                         median=float(np.median(v))))
        print(f"    strong, {rho:.0%} reduction: {v.min():.3f} to {v.max():.3f} "
              f"(median {np.median(v):.3f})")
    strong = np.concatenate(pool)
    print(f"    -> above a 10% reduction: {strong.min():.3f} to {strong.max():.3f}\n")
    results[model] = dict(
        label=label,
        mild=dict(reduction=MILD, lo=float(mild.min()), hi=float(mild.max()),
                  median=float(np.median(mild))),
        strong_rows=rows,
        strong=dict(lo=float(strong.min()), hi=float(strong.max())))


# ---------------------------------------------------------------- verdict
print("verdict against the text")
print(f"    invariance: claimed exact, measured {worst:.2e}  "
      f"{'CONFIRMED' if inv_ok else 'FALSIFIED'}")


def contains(lo, hi, clo, chi):
    return lo <= clo and hi >= chi


for model in results:
    m, s = results[model]["mild"], results[model]["strong"]
    mok = contains(m["lo"], m["hi"], *CLAIMED["mild"])
    sok = contains(s["lo"], s["hi"], *CLAIMED["strong"])
    print(f"    {model:>9}: mild {m['lo']:.3f}-{m['hi']:.3f} vs claimed "
          f"{CLAIMED['mild'][0]}-{CLAIMED['mild'][1]} "
          f"({'contains' if mok else 'does not contain'}); "
          f"strong {s['lo']:.3f}-{s['hi']:.3f} vs claimed "
          f"{CLAIMED['strong'][0]}-{CLAIMED['strong'][1]} "
          f"({'contains' if sok else 'does not contain'})")

out = ROOT / "manuscript/revision_BIB/finale/analisi" / (Path(__file__).stem + ".json")
out.write_text(json.dumps(dict(
    seed=SEED, ndraw=NDRAW,
    baseline=dict(f_oneway=base[0], f_sign=base[1], r_cycle=base[2], edges=base[3]),
    shared=shared, shared_invariant=bool(inv_ok), shared_max_abs_change=worst,
    pair_specific=results, claimed=CLAIMED,
    quoted_model="lognormal",
), indent=2), encoding="utf-8")
print(f"\n  wrote {out.relative_to(ROOT)}")
