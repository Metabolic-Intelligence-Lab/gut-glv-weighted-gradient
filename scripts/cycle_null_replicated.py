r"""
A calibrated null for the cycle obstructions on the replicated support
======================================================================
cycle_witnesses.py reports the four independent cycles of Venturelli's three fitted
parameter sets on their shared support as stability across fits plus robustness to the
compositional gauge, with no p-value. averaging_audit.py shows that the published
aggregate r_cycle of 2.169 is inflated by the averaging and falls to 0.214 on that shared
support. null_audit2.py builds an exactly symmetrizable surrogate of a published matrix
that preserves support, signs up to concordance and magnitudes, and reads the aggregate
diagnostics against it. The two have never been composed. This script composes them.

The question. Could a cycle residual of the observed size, stable in sign across the three
fits, arise from a truly symmetrizable system plus inference noise?

Design.
1. The edge set is FIXED to the 14 species pairs that are nonzero in both directions in
   every fit and sign-concordant in the published mean, the edge set of averaging_audit.py
   and cycle_witnesses.py. It is identical in both arms. This removes the one weakness of
   the aggregate null, whose surrogate fills one-way pairs symmetrically and therefore
   scores r_cycle on 34 to 53 concordant edges against the observed 25.
2. The null truth is full_symmetrizable(V): exactly symmetrizable, support, signs and
   magnitudes of the published matrix preserved. The random-support truth of
   symmetrizability_invariants.symmetric_null cannot be used here, because it does not keep
   the observed support and the restricted statistic would be undefined on it.
3. Three fits are simulated from the one truth, since the three published fits are three
   estimates of one system. Their noise is drawn in two regimes, because the real fits are
   nested and their errors are therefore correlated to an unknown degree:
     indep   three independent draws, the most favourable reading of "agreement across fits";
     shared  one draw used for all three, perfectly correlated, which makes the mean of the
             three as noisy as a single fit and is the conservative bound.
   The truth is between the two. A claim should survive the conservative regime.
4. Statistics are two-sided in the cycle residual, |sum b_ij| around the cycle, because the
   orientation of each cycle in Table 4 was chosen from the observed data. Exceedance is
   p = (b+1)/(B+1).
5. b_ij = log|J_ij / J_ji| is evaluated on the fixed edge set on every draw, whether or not
   a corrupted edge keeps its sign concordance. A lost concordance would in reality drop the
   edge; keeping it widens the null, so the reported p is conservative. The frequency is
   reported as a diagnostic.
6. The noise level nu is calibrated, as in null_audit2.py, so that the null reproduces the
   spread of Clark's published posterior coupling-correlation draws. Nothing is hardcoded.

Runs from the repository root with no arguments; writes figures/cycle_null_replicated.json.
"""
import os, re, json, itertools, argparse
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=".")
ap.add_argument("--out", default="figures"); ap.add_argument("--B", type=int, default=5000)
ARGS = ap.parse_args()
import sys; sys.path.insert(0, os.path.join(ARGS.repo, "scripts")); os.chdir(ARGS.repo)
from symmetrizability_invariants import corrupt, gamma_pearson
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")

SP = ["BH", "CA", "BU", "PC", "BO", "BV", "BT", "EL", "FP", "CH", "DP", "ER"]; S = 12
IDX = {s: k for k, s in enumerate(SP)}
OFF = ~np.eye(S, dtype=bool)


def parse(p):
    t = open(p).read(); M = np.zeros((S, S))
    for m in re.finditer(r'<parameter[^>]*id="a([A-Z]{2})([A-Z]{2})"[^>]*value="([^"]+)"', t):
        a, b, v = m.group(1), m.group(2), float(m.group(3))
        if a in IDX and b in IDX: M[IDX[a], IDX[b]] = v
    return M


T = {t: parse(f"data/venturelli2018/twelveSpecies{t}.xml") for t in ("T2", "T3", "T4")}
TL = list(T.values())
V = np.mean(TL, axis=0)
Vpub = np.loadtxt("data/venturelli2018/interaction_matrix_averaged.csv",
                  delimiter=",", skiprows=1)
assert np.abs(V - Vpub).max() < 1e-4, "published matrix is not the mean of T2..T4"
GSTD = float(np.std(np.loadtxt("data/clark2021/gamma_posterior_samples.csv")))


# --------------------------------------------------------------------------- #
#  the surrogate of null_audit2.py, reproduced here so this script is self-contained
# --------------------------------------------------------------------------- #
def full_symmetrizable(J):
    """Exactly symmetrizable surrogate preserving support, signs-up-to-concordance and
    magnitudes: f_oneway = f_sign = r_cycle = 0. Size taken from J."""
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


def corrupt_mult(A, nu, rng):
    """Inference noise proportional to each coefficient. Size taken from A."""
    off = ~np.eye(len(A), dtype=bool)
    J = A.copy(); mask = off & (np.abs(A) > 0)
    J[mask] = A[mask] * (1.0 + nu * rng.standard_normal(mask.sum()))
    return J


# --------------------------------------------------------------------------- #
#  1. the fixed edge set and the cycle basis, exactly as cycle_witnesses.py
# --------------------------------------------------------------------------- #
edges = [(i, j) for i, j in itertools.combinations(range(S), 2)
         if all(M[i, j] != 0 and M[j, i] != 0 for M in TL)
         and np.sign(V[i, j]) == np.sign(V[j, i])]
EI = np.array([e[0] for e in edges]); EJ = np.array([e[1] for e in edges])
NE = len(edges)
print(f"shared, sign-concordant pairs (fixed edge set): {NE}")

adj = {k: [] for k in range(S)}
for i, j in edges: adj[i].append(j); adj[j].append(i)
parent, tree = {}, set()
for root in range(S):
    if root in parent or not adj[root]: continue
    parent[root] = None; queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in parent:
                parent[v] = u; tree.add(frozenset((u, v))); queue.append(v)


def path_to_root(v):
    out = [v]
    while parent[v] is not None: v = parent[v]; out.append(v)
    return out


cycles = []
for i, j in edges:
    if frozenset((i, j)) in tree: continue
    pi, pj = path_to_root(i), path_to_root(j)
    common = next(x for x in pi if x in pj)
    cycles.append(pi[:pi.index(common) + 1] + pj[:pj.index(common)][::-1])
# orientation as in Table 4: mean residual over the three fits negative


def raw_resid(M, c):
    return sum(np.log(abs(M[a, b] / M[b, a])) for a, b in zip(c, c[1:] + c[:1]))


cycles = [c[::-1] if np.mean([raw_resid(M, c) for M in TL]) > 0 else c for c in cycles]
NAMES = ["-".join(SP[k] for k in c) for c in cycles]
print(f"independent cycles on that edge set: {len(cycles)}  {NAMES}")

# how well determined each cycle is: the smallest coefficient magnitude it rests on.
# Every cycle is reported with its own p, so no cycle is selected on the observed data and
# the multiplicity is covered by the worst-cycle statistic as well.
MINCOEF = [min(min(abs(V[a, b]), abs(V[b, a])) for a, b in zip(c, c[1:] + c[:1]))
           for c in cycles]
ORDER = sorted(range(len(cycles)), key=lambda k: -MINCOEF[k])


# --------------------------------------------------------------------------- #
#  2. statistics on the fixed edge set
# --------------------------------------------------------------------------- #
BMAT = np.zeros((NE, S)); BMAT[np.arange(NE), EJ] = 1; BMAT[np.arange(NE), EI] = -1
EPOS = {frozenset(e): k for k, e in enumerate(edges)}
CYC_ROWS = []
for c in cycles:
    row = np.zeros(NE)
    for a, b in zip(c, c[1:] + c[:1]):
        k = EPOS[frozenset((a, b))]
        row[k] += 1.0 if (EI[k], EJ[k]) == (a, b) else -1.0
    CYC_ROWS.append(row)
CYC_ROWS = np.array(CYC_ROWS)


def bvec(M):
    return np.log(np.abs(M[EI, EJ] / M[EJ, EI]))


def rcycle_fixed(b):
    u, *_ = np.linalg.lstsq(BMAT, b, rcond=None)
    return float(np.sqrt(np.mean((b - BMAT @ u) ** 2)))


def stats(mats):
    """mats: list of three matrices. Returns the statistics the null is read against."""
    bs = np.array([bvec(M) for M in mats])                  # (3, NE)
    res = bs @ CYC_ROWS.T                                   # (3, ncyc) cycle residuals
    rc = [rcycle_fixed(b) for b in bs]
    rc_mean = rcycle_fixed(bvec(np.mean(mats, axis=0)))
    mres = res.mean(0)
    conc = np.array([np.sign(M[EI, EJ]) == np.sign(M[EJ, EI]) for M in mats])
    out = dict(
        rc_per_fit_max=float(np.max(rc)), rc_per_fit_min=float(np.min(rc)),
        rc_mean_matrix=float(rc_mean),
        abs_mean_max=float(np.max(np.abs(mres))),
        n_sign_agree=int(np.sum([np.all(res[:, k] > 0) or np.all(res[:, k] < 0)
                                 for k in range(res.shape[1])])),
        all_edges_concordant=bool(conc.all()),
        res=res, rc=rc)
    for k in range(len(cycles)):
        out[f"cyc:{NAMES[k]}"] = float(abs(mres[k]))
    return out


OBS = stats(TL)
print("\nobserved, on the fixed edge set")
print(f"  r_cycle per fit                        : "
      f"{' '.join(f'{v:.4f}' for v in OBS['rc'])}")
print(f"  r_cycle of the mean matrix             : {OBS['rc_mean_matrix']:.4f}")
for k in ORDER:
    print(f"  {NAMES[k]:<16} per fit {np.array2string(OBS['res'][:, k], precision=3, floatmode='fixed')}"
          f"  mean {OBS['res'][:, k].mean():+.3f}  smallest coefficient {MINCOEF[k]:.3f}")
print(f"  cycles with the same residual sign in all three fits: "
      f"{OBS['n_sign_agree']} of {len(cycles)}")


# --------------------------------------------------------------------------- #
#  3. calibration of the noise level, as in null_audit2.py
# --------------------------------------------------------------------------- #
A0 = full_symmetrizable(V)
chk = bvec(A0)
print(f"\nsanity: the surrogate is exactly symmetrizable on the fixed edge set, "
      f"max |cycle residual| {np.max(np.abs(chk @ CYC_ROWS.T)):.2e}, "
      f"r_cycle {rcycle_fixed(chk):.2e}")

C = np.loadtxt("data/clark2021/interaction_matrix_RLC8_posteriormean.csv", delimiter=",")


def calibrate(corrupt_fn, grid, seed=101):
    """nu reproducing the spread of Clark's posterior coupling-correlation draws."""
    AC = full_symmetrizable(C)
    best, bnu, bg = np.inf, grid[0], np.nan
    for nu in grid:
        rng = np.random.default_rng(seed)
        g = [gamma_pearson(corrupt_fn(AC, nu, rng)) for _ in range(300)]
        d = abs(np.nanstd(g) - GSTD)
        if d < best: best, bnu, bg = d, float(nu), float(np.nanstd(g))
    return bnu, bg


print(f"\ncalibration A, to the Clark posterior reciprocity spread ({GSTD:.3f}):")
NU_HI = {}
for cn, cf, grid in (("add", corrupt, np.linspace(0.25, 5.0, 20)),
                     ("mult", corrupt_mult, np.linspace(0.25, 6.0, 24))):
    NU_HI[cn], g = calibrate(cf, grid)
    print(f"  noise={cn:4}: nu={NU_HI[cn]:.2f} (null gamma std {g:.3f})")

# ------------------------------------------------------------------ #
#  calibration B: to the dispersion the three fits actually show
# ------------------------------------------------------------------ #
# An aggregate statistic such as gamma has a small sampling spread even when each
# coefficient is very noisy, so calibration A leaves the per-coefficient noise almost
# unconstrained and sets it enormous. The three fits bound it directly: if the inference
# noise were that large the three would not agree on a cycle residual to within 0.04.
# Calibration B therefore fixes nu so that the null reproduces the observed fit-to-fit
# dispersion of the four cycle residuals. Because the fits are NESTED they share training
# data, so their agreement understates the true inference uncertainty: calibration B is a
# LOWER bound on the noise and the resulting p is optimistic, as calibration A is an upper
# bound and its p is pessimistic. The two bracket the answer.
OBS_SD = float(np.sqrt(np.mean(np.std(OBS["res"], axis=0, ddof=1) ** 2)))
print(f"\ncalibration B, to the fit-to-fit dispersion of the four cycle residuals "
      f"(pooled sd {OBS_SD:.4f}):")


def calibrate_interfit(corrupt_fn, grid, seed=77, n=200):
    best, bnu, bs = np.inf, grid[0], np.nan
    for nu in grid:
        rng = np.random.default_rng(seed); sds = []
        for _ in range(n):
            mats = [corrupt_fn(A0, nu, rng) for _ in range(3)]
            res = np.array([bvec(M) for M in mats]) @ CYC_ROWS.T
            sds.append(np.sqrt(np.mean(np.std(res, axis=0, ddof=1) ** 2)))
        s = float(np.median(sds)); d = abs(s - OBS_SD)
        if d < best: best, bnu, bs = d, float(nu), s
    return bnu, bs


NU_LO = {}
for cn, cf, grid in (("add", corrupt, np.linspace(0.002, 0.30, 40)),
                     ("mult", corrupt_mult, np.linspace(0.002, 0.30, 40))):
    NU_LO[cn], s = calibrate_interfit(cf, grid)
    print(f"  noise={cn:4}: nu={NU_LO[cn]:.4f} (null fit-to-fit pooled sd {s:.4f})")

CAL = {"A, Clark posterior": NU_HI, "B, fit-to-fit": NU_LO}


# --------------------------------------------------------------------------- #
#  4. the null
# --------------------------------------------------------------------------- #
KEYS = (["rc_per_fit_max", "rc_mean_matrix"]
        + [f"cyc:{NAMES[k]}" for k in ORDER] + ["abs_mean_max"])
LABEL = {"rc_per_fit_max": "r_cycle, worst of the three fits",
         "rc_mean_matrix": "r_cycle of the mean matrix",
         "abs_mean_max":   "|mean residual|, worst of the four cycles",
         "n_sign_agree":   "cycles agreeing in sign in all three fits"}
for k in range(len(cycles)):
    LABEL[f"cyc:{NAMES[k]}"] = f"|mean residual| {NAMES[k]} (min coef {MINCOEF[k]:.2f})"
rows = []
CELLS = [(ca, cn, rg) for ca in CAL for cn in ("add", "mult") for rg in ("indep", "shared")
         # calibration B is derived from the independent scatter across fits, so the
         # perfectly correlated regime is not defined for it: with one shared draw the
         # fit-to-fit dispersion is zero and cannot be matched to the observed one.
         if not (ca.startswith("B") and rg == "shared")]
CFN = {"add": corrupt, "mult": corrupt_mult}
print(f"\n==== exceedance against a symmetrizable truth, B={ARGS.B} ====")
print(f"  {'calibration':<20}{'noise':<6}{'fits':<8}{'statistic':<44}{'obs':>8}"
      f"{'null med':>10}{'null 95%':>10}{'p':>9}")
for calname, cn, regime in CELLS:
        cf = CFN[cn]; nu = CAL[calname][cn]
        rng = np.random.default_rng(2024)
        acc = {k: [] for k in KEYS}; agree = []; lost = 0
        for _ in range(ARGS.B):
            if regime == "indep":
                mats = [cf(A0, nu, rng) for _ in range(3)]
            else:
                one = cf(A0, nu, rng); mats = [one, one, one]
            s = stats(mats)
            for k in KEYS: acc[k].append(s[k])
            agree.append(s["n_sign_agree"]); lost += (not s["all_edges_concordant"])
        for k in KEYS + ["n_sign_agree"]:
            arr = np.array(acc[k] if k in acc else agree, float); o = OBS[k]
            p = (int(np.sum(arr >= o)) + 1) / (len(arr) + 1)
            print(f"  {calname:<20}{cn:<6}{regime:<8}{LABEL[k]:<44}{o:>8.3f}"
                  f"{np.nanmedian(arr):>10.3f}{np.nanquantile(arr, 0.95):>10.3f}{p:>9.4f}")
            rows.append(dict(calibration=calname, noise=cn, regime=regime, statistic=k,
                             label=LABEL[k], observed=float(o),
                             null_median=float(np.nanmedian(arr)),
                             null_q95=float(np.nanquantile(arr, 0.95)), p=float(p),
                             B=ARGS.B, nu=nu))
        print(f"  {'':<34}draws in which a fixed edge lost sign concordance: "
              f"{lost / ARGS.B:.3f} (kept in the null, which widens it)")

os.makedirs(ARGS.out, exist_ok=True)
json.dump(dict(fixed_edges=NE, cycles=NAMES, min_coefficient=MINCOEF,
               clark_gamma_std=GSTD, obs_interfit_pooled_sd=OBS_SD,
               nu={k: v for k, v in CAL.items()},
               observed=dict({k: float(OBS[k]) for k in KEYS},
                             n_sign_agree=int(OBS["n_sign_agree"])),
               observed_rc_per_fit=[float(v) for v in OBS["rc"]],
               observed_residuals={NAMES[k]: [float(v) for v in OBS["res"][:, k]]
                                   for k in range(len(cycles))},
               rows=rows),
          open(os.path.join(ARGS.out, "cycle_null_replicated.json"), "w"), indent=1)
print("\nsaved cycle_null_replicated.json")
