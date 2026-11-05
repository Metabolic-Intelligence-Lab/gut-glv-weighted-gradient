"""
The held-out descent test on the cohorts with a MEASURED microbial load (BIB revision, node D)
=============================================================================================

The submitted manuscript said the descent test "cannot be run on the nine longitudinal
datasets, because none supplies a bottom-up matrix to build a potential from". Reviewer 4
(point 5) noticed that the principal potential is not built from a bottom-up matrix at all: it
is the symmetrizable gLV fitted to trajectories, leave-one-community-out. So the test needs
only trajectories, and three cohorts have trajectories in ABSOLUTE abundance, because their
total load was measured:

    gibson_healthy   gnotobiotic mice, qPCR load, ~twice-daily sampling, 3 perturbations
    gibson_uc        the same protocol, ulcerative-colitis donor community
    vandeputte       20 women, daily flow-cytometry quantitative profiling (QMP)

On these data Eq. (5) applies as derived, with no passage to the simplex, which is the test
Reviewers 2 and 4 ask for (R2-1, R4-1), on systems assembled independently of the Venturelli
consortium (R3-1, R4-5).

Design, mirroring lyapunov_robustness.loco_descent except where stated:

  preprocessing   exactly the published cohort pipeline (infer_and_place.build_design,
                  infer_vandeputte.load_vandeputte/design): top 30 taxa by mean proportion,
                  absolute abundance = proportion x load (Gibson) or the QMP counts as
                  distributed (Vandeputte, already absolute), each taxon divided by its
                  nonzero median. That rescaling is constant in time and per taxon, so it
                  is absorbed by D and leaves H_D unchanged. Totals are never renormalised.
  growth rates    g = [log(x_b + eps) - log(x_a + eps)] / dt, eps = 1e-3, transitions with
                  0 < dt <= MAX_GAP = 3 d (the published cohort rule).
  equation mask   a taxon contributes an equation only where its RELATIVE abundance exceeds
                  5e-3 at both ends of the transition: the published EV3 rule, verbatim.
                  Sensitivity: no mask (the published cohort gamma-hat convention).
  forcing         Gibson perturbation indicators (high-fat diet, vancomycin, gentamicin,
                  analyze_gibson.PERTURBS) enter the regression as covariates at the start
                  sample, as infer_and_place.build_design does; they are unpenalised, like r.
                  H_D is built from (r, J, d) only: the potential of the UNFORCED flow.
  model           diagonally symmetrizable gLV, J = D^{-1} K, K = K^T, d = exp(u - mean u),
                  ridge lam*(1/d_i^2 + 1/d_j^2) on K_ij (i<j), i.e. on off-diagonal J only;
                  u by L-BFGS-B from u = 0, maxiter 60, minimising training MSE: the
                  published fit_symmetrizable, vectorised for general S (validated on EV3
                  against the published code, see --validate).
  held out        leave-one-subject-out; Delta H_D <= 0 counted on the held-out subject.
  null            per fold, B scrambles of the off-diagonal upper triangle of K at the same
                  d, diagonal kept (lyapunov_robustness.scramble), seeds 80000 + 997 h + i;
                  pooled across folds per scramble index; p = (b + 1)/(B + 1).
  uncertainty     subject-level bootstrap of the margin (2000 resamples).

Transition classes, pre-specified from existing code conventions (Gibson only):

  colonization    t_a < 10 d (analyze_gibson.return_ratio starts its baseline at day 10)
  forced          [t_a, t_b] overlaps a perturbation window
  recovery        an endpoint within RECOV days after a window ends; RECOV = 3 d primary
                  (the response window of analyze_gibson.return_ratio), 0 and 8 d reported
  unforced        everything else: the transitions on which Eq. (5) is guaranteed

Optimiser diagnostics (Reviewer 4 point 7): the published optimiser spreads d over about four
decades on EV3 (max/min 6.5e3 to 7.3e4 across folds) and wider here. Two sensitivities are
pre-specified, before any cohort result: d bounded to five decades (the widest EV3 fold, rounded
up) and L-BFGS-B run to 500 iterations.

Configurations: lam in {0.25, 0.5, 1, 2} (primary 0.5, B = 5000; sweep B = 2000); no mask;
Gibson thinned to a 1-day and a 2-day grid (Reviewer 1 point 2, sampling density); Vandeputte
with 40 taxa (the manuscript text says 40; the published gamma-hat script uses 30).

Run:  python3 scripts/descent_cohorts.py              (all configurations, 4 workers)
      python3 scripts/descent_cohorts.py --validate   (EV3 check against the published code)
      python3 scripts/descent_cohorts.py --replot     (figure and tables from the cache)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import warnings                                                             # noqa: E402
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = ROOT / "figures"
CACHE = OUT / "descent_cohorts.npz"
FIG = OUT / "fig_descent_cohorts.png"

EPS = 1e-3
ABMIN = 5e-3            # published EV3 rule, on relative abundance
MAX_GAP = 3.0           # published cohort rule, days
S_TAXA = 30
LAMS = [0.25, 0.5, 1.0, 2.0]
B_MAIN, B_SWEEP = 5000, 2000
NBOOT = 2000
RECOVS = [3.0, 0.0, 8.0]          # first is primary
COLONIZATION_END = 10.0
WORKERS = 4


# =========================================================================== data
def gibson_series(data_dir, subjects, grid=None, s_taxa=S_TAXA, relative=False):
    """Per-subject absolute-abundance series, exactly as infer_and_place.build_design.

    relative=True is a diagnostic only: the same taxa as proportions (load discarded), to tell
    load-measurement noise apart from steady-state fluctuation.
    """
    from analyze_gibson import load, PERTURBS
    from infer_and_place import load_qpcr
    rel, subj, tim, sids = load(str(ROOT / data_dir))
    qpcr = load_qpcr(str(ROOT / data_dir))
    top = np.argsort(rel.mean(0))[::-1][:s_taxa]
    load_vec = np.array([qpcr.get(s, np.nan) for s in sids])
    if relative:
        load_vec = np.where(np.isfinite(load_vec), 1.0, np.nan)   # same samples, no load
    Xabs = rel[:, top] * load_vec[:, None]
    scale = np.nanmedian(np.where(Xabs > 0, Xabs, np.nan), axis=0)
    Xs = Xabs / scale
    R = rel[:, top]
    pert = np.zeros((len(sids), len(PERTURBS)))
    for j, (_, s, e) in enumerate(PERTURBS):
        for i in range(len(sids)):
            if subj[i] in subjects and s <= tim[i] <= e:
                pert[i, j] = 1.0
    series = []
    for sb in subjects:
        idx = np.where(subj == sb)[0]
        idx = idx[np.argsort(tim[idx])]
        if grid is not None:
            idx = thin(idx, tim, grid)
        if len(idx) >= 2:
            series.append(dict(subject=int(sb), t=tim[idx], X=Xs[idx], R=R[idx], Z=pert[idx]))
    return series


def vandeputte_series(grid=None, s_taxa=S_TAXA, as_coded=False, relative=False):
    """As infer_vandeputte.design(use_load=True): QMP counts are already absolute.

    The Methods rescale each taxon "to O(1) by its nonzero median". infer_vandeputte.design
    computes that median with np.median over an array holding NaN for the zeros, which returns
    NaN for every taxon with a zero (all 30 here), and then falls back to a scale of 1: no taxon
    is rescaled and the regressors stay at ~5e8 cells, where the ridge is inert. The primary
    run applies the documented scaling (nanmedian, as infer_and_place.build_design does for
    Gibson); as_coded=True reproduces the published code.
    """
    from infer_vandeputte import load_vandeputte
    X, day, pid = load_vandeputte()
    prop = X / X.sum(1, keepdims=True)
    top = np.argsort(prop.mean(0))[::-1][:s_taxa]
    Xa = prop[:, top] if relative else X[:, top]
    med = np.median if as_coded else np.nanmedian
    scale = med(np.where(Xa > 0, Xa, np.nan), axis=0)
    scale = np.where(np.isnan(scale) | (scale <= 0), 1.0, scale)
    Xs = Xa / scale
    R = prop[:, top]
    series = []
    for k, sb in enumerate(np.unique(pid)):
        idx = np.where(pid == sb)[0]
        idx = idx[np.argsort(day[idx])]
        if grid is not None:
            idx = thin(idx, day, grid)
        if len(idx) >= 2:
            series.append(dict(subject=k, t=day[idx], X=Xs[idx], R=R[idx],
                               Z=np.zeros((len(idx), 0))))
    return series


def thin(idx, tim, grid):
    """Greedy thinning to a sampling grid: keep a sample once >= grid - 0.25 d after the last."""
    keep, last = [], -np.inf
    for i in idx:
        if tim[i] >= last + grid - 0.25:
            keep.append(i)
            last = tim[i]
    return np.array(keep, dtype=int)


def make_transitions(series, use_mask=True):
    """(xa, g, z, mask) per transition, plus (subject, t_a, t_b)."""
    trans, meta = [], []
    for s in series:
        t, X, R, Z = s["t"], s["X"], s["R"], s["Z"]
        for a in range(len(t) - 1):
            b = a + 1
            dt = t[b] - t[a]
            if dt <= 0 or dt > MAX_GAP:
                continue
            if not (np.all(np.isfinite(X[a])) and np.all(np.isfinite(X[b]))):
                continue                      # as build_design: no load, no transition
            g = (np.log(X[b] + EPS) - np.log(X[a] + EPS)) / dt
            m = ((R[a] > ABMIN) & (R[b] > ABMIN)) if use_mask else np.ones(X.shape[1], bool)
            trans.append((X[a], g, Z[a], m))
            meta.append((s["subject"], t[a], t[b], a, b))
    return trans, meta


# =========================================================================== model
class SymFit:
    """Vectorised constrained_vs_free_glv._constrained_inner for general S and P covariates.

    For fixed d the problem is linear in (r, K, B): species i's equations have design
    [1, x_a / d_i, z_a], mapped onto (r_i, K_{(i,j)} for all j, B_{i,p}). The per-species
    Gram matrices of [1, x_a, z_a] do not depend on d, so they are computed once and rescaled.
    """

    def __init__(self, trans, S, P, lam):
        self.S, self.P, self.lam = S, P, lam
        L = 1 + S + P
        self.G = np.zeros((S, L, L))
        self.h = np.zeros((S, L))
        self.yy = np.zeros(S)
        self.n = 0
        if trans:
            A = np.array([np.concatenate([[1.0], xa, z]) for xa, _, z, _ in trans])
            Y = np.array([g for _, g, _, _ in trans])
            M = np.array([m for _, _, _, m in trans])
            for i in range(S):
                Ai, yi = A[M[:, i]], Y[M[:, i], i]
                self.G[i] = Ai.T @ Ai
                self.h[i] = Ai.T @ yi
                self.yy[i] = yi @ yi
            self.n = int(M.sum())
        self.pairs = [(i, j) for i in range(S) for j in range(i, S)]
        pk = {pr: k for k, pr in enumerate(self.pairs)}
        self.npk = len(self.pairs)
        self.npar = S + self.npk + S * P
        self.gidx = []
        for i in range(S):
            gi = [i] + [S + pk[(min(i, j), max(i, j))] for j in range(S)] \
                 + [S + self.npk + i * P + p for p in range(P)]
            self.gidx.append(np.array(gi))
        off = [k for k, (i, j) in enumerate(self.pairs) if i != j]
        self.off_k = np.array(off)
        self.off_i = np.array([self.pairs[k][0] for k in off])
        self.off_j = np.array([self.pairs[k][1] for k in off])

    def solve(self, u):
        S, P = self.S, self.P
        d = np.exp(u - u.mean())
        ATA = np.zeros((self.npar, self.npar))
        ATy = np.zeros(self.npar)
        Gs, hs = [], []
        for i in range(S):
            sc = np.ones(1 + S + P)
            sc[1:1 + S] = 1.0 / d[i]
            Gi = self.G[i] * np.outer(sc, sc)
            hi = self.h[i] * sc
            gi = self.gidx[i]
            ATA[np.ix_(gi, gi)] += Gi
            ATy[gi] += hi
            Gs.append(Gi); hs.append(hi)
        reg = np.zeros(self.npar)
        reg[S + self.off_k] = self.lam * (1.0 / d[self.off_i] ** 2 + 1.0 / d[self.off_j] ** 2)
        beta = np.linalg.solve(ATA + np.diag(reg) + 1e-6 * np.eye(self.npar), ATy)
        sse = 0.0
        for i in range(S):
            bl = beta[self.gidx[i]]
            sse += self.yy[i] - 2.0 * bl @ hs[i] + bl @ Gs[i] @ bl
        return beta, d, sse / max(self.n, 1)

    def mse(self, u):
        return self.solve(u)[2]

    def unpack(self, beta, d):
        S, P = self.S, self.P
        r = beta[:S].copy()
        K = np.zeros((S, S))
        for k, (i, j) in enumerate(self.pairs):
            K[i, j] = K[j, i] = beta[S + k]
        Bc = beta[S + self.npk:].reshape(S, P) if P else np.zeros((S, 0))
        return r, K / d[:, None], K, Bc


def fit_symmetrizable_general(trans, S, P, lam, maxiter=60, dbound=None):
    """maxiter=60 from u=0, unbounded: the published optimiser. dbound (decades) caps
    max(d)/min(d) at 10**dbound through box bounds on u; a pre-specified sensitivity."""
    F = SymFit(trans, S, P, lam)
    bounds = None
    if dbound is not None:
        a = 0.5 * dbound * np.log(10.0)
        bounds = [(-a, a)] * S
    res = minimize(F.mse, np.zeros(S), method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter})
    beta, d, mse = F.solve(res.x)
    r, J, K, Bc = F.unpack(beta, d)
    diag = dict(nit=int(res.nit), success=bool(res.success), train_mse=float(mse),
                d_min=float(d.min()), d_max=float(d.max()),
                sym_resid=float(np.linalg.norm(K - K.T) / max(np.linalg.norm(K), 1e-12)))
    return r, J, K, d, Bc, diag


# =========================================================================== scoring
def potentials(X, r, K, d, Kvals_matrix=None):
    """H_D at each sample, for the fitted K and (optionally) a stack of scrambled off-diagonals."""
    S = X.shape[1]
    iu = np.triu_indices(S, 1)
    lin = X @ (d * r)
    dg = (X ** 2) @ np.diag(K)
    Pq = X[:, iu[0]] * X[:, iu[1]]
    H = -lin - 0.5 * (dg + 2.0 * Pq @ K[iu])
    if Kvals_matrix is None:
        return H
    Hs = -lin[:, None] - 0.5 * (dg[:, None] + 2.0 * Pq @ Kvals_matrix.T)
    return H, Hs


def run_fold(job):
    """One held-out subject under one configuration."""
    cfg, h = job
    series = build_series(cfg)
    S = series[0]["X"].shape[1]
    P = series[0]["Z"].shape[1]
    train = [s for k, s in enumerate(series) if k != h]
    test = series[h]
    tr, _ = make_transitions(train, cfg["mask"])
    te, meta = make_transitions([test], cfg["mask"])
    t0 = time.time()
    r, J, K, d, Bc, diag = fit_symmetrizable_general(tr, S, P, cfg["lam"],
                                                     cfg.get("maxiter", 60), cfg.get("dbound"))
    diag["fit_seconds"] = time.time() - t0
    diag["n_train"] = len(tr)
    if not te:
        return dict(cfg=cfg["name"], h=h, obs=np.zeros(0), down_null=np.zeros((cfg["B"], 0), bool),
                    meta=[], diag=diag)
    iu = np.triu_indices(S, 1)
    Kv = K[iu]
    V = np.stack([np.random.default_rng(80000 + 997 * h + i).permutation(Kv)
                  for i in range(cfg["B"])])
    H, Hs = potentials(test["X"], r, K, d, V)
    ia = np.array([m[3] for m in meta]); ib = np.array([m[4] for m in meta])
    obs = H[ib] - H[ia]
    dnull = (Hs[ib] - Hs[ia]) <= 0                       # (n_te, B)
    return dict(cfg=cfg["name"], h=h, obs=obs, down_null=dnull.T, meta=meta, diag=diag)


_SERIES_CACHE = {}


def build_series(cfg):
    key = (cfg["cohort"], cfg.get("grid"), cfg.get("s_taxa", S_TAXA))
    as_coded = cfg["cohort"] == "vandeputte_ascoded"
    relative = cfg["cohort"].endswith("_relative")
    if key not in _SERIES_CACHE:
        if cfg["cohort"].startswith("gibson_healthy"):
            s = gibson_series("data/gibson_healthy", (1, 2, 3, 4, 5), cfg.get("grid"), key[2],
                              relative)
        elif cfg["cohort"].startswith("gibson_uc"):
            s = gibson_series("data/gibson_uc", (6, 7, 8, 9, 10), cfg.get("grid"), key[2],
                              relative)
        else:
            s = vandeputte_series(cfg.get("grid"), key[2], as_coded, relative)
        # a subject with no usable transition (Gibson healthy subject 1 has no qPCR) is dropped
        s = [x for x in s if make_transitions([x], use_mask=False)[0]]
        _SERIES_CACHE[key] = s
    return _SERIES_CACHE[key]


def configs():
    out = []
    for c in ("gibson_healthy", "gibson_uc", "vandeputte", "vandeputte_ascoded"):
        for lam in LAMS:
            out.append(dict(name=f"{c}|lam={lam}", cohort=c, lam=lam, mask=True,
                            B=B_MAIN if lam == 0.5 else B_SWEEP))
        out.append(dict(name=f"{c}|nomask", cohort=c, lam=0.5, mask=False, B=B_SWEEP))
        out.append(dict(name=f"{c}|d<=1e5", cohort=c, lam=0.5, mask=True, B=B_SWEEP,
                        dbound=5.0))
        out.append(dict(name=f"{c}|maxiter=500", cohort=c, lam=0.5, mask=True, B=B_SWEEP,
                        maxiter=500))
        if c.startswith("gibson"):
            for grid in (1.0, 2.0):
                out.append(dict(name=f"{c}|grid={grid:g}d", cohort=c, lam=0.5, mask=True,
                                B=B_SWEEP, grid=grid))
        else:
            out.append(dict(name=f"{c}|S=40", cohort=c, lam=0.5, mask=True, B=B_SWEEP,
                            s_taxa=40))
    for c in ("gibson_healthy_relative", "gibson_uc_relative", "vandeputte_relative"):
        out.append(dict(name=f"{c}|lam=0.5", cohort=c, lam=0.5, mask=True, B=B_SWEEP))
    return out


# =========================================================================== summaries
def classify(ta, tb, recov):
    from analyze_gibson import PERTURBS
    if ta < COLONIZATION_END:
        return "colonization"
    for _, s, e in PERTURBS:
        if ta <= e and tb >= s:
            return "forced"
    if recov > 0:
        for _, s, e in PERTURBS:
            if (e < ta <= e + recov) or (e < tb <= e + recov):
                return "recovery"
    return "unforced"


def summarise(obs, dnull, subj, sel, rng):
    """Pooled held-out descent vs pooled scrambled null, subject bootstrap of the margin."""
    sel = np.asarray(sel, bool)
    n = int(sel.sum())
    if n == 0:
        return None
    o = obs[sel] <= 0
    dn = dnull[:, sel]                                   # (B, n)
    f_obs = o.mean()
    f_null = dn.mean(axis=1)
    b = int(np.sum(f_null >= f_obs))
    B = len(f_null)
    ids = np.unique(subj[sel])
    per_t = dnull.mean(axis=0)            # each transition's scrambled descent rate
    down_t = obs <= 0
    margins = []
    for _ in range(NBOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([np.where(sel & (subj == c))[0] for c in pick])
        margins.append(down_t[idx].mean() - per_t[idx].mean())
    lo, hi = np.percentile(margins, [2.5, 97.5])
    return dict(n=n, down=int(o.sum()), f_obs=float(f_obs), f_null=float(f_null.mean()),
                null_lo=float(np.percentile(f_null, 2.5)),
                null_hi=float(np.percentile(f_null, 97.5)),
                margin=float(f_obs - f_null.mean()), m_lo=float(lo), m_hi=float(hi),
                b=b, B=B, p=(b + 1) / (B + 1), n_subj=len(ids))


def fmt_p(s):
    return f"<{1 / (s['B'] + 1):.1e}" if s["b"] == 0 else f"{s['p']:.4f}"


def assemble(results, name):
    rs = sorted([r for r in results if r["cfg"] == name], key=lambda r: r["h"])
    obs = np.concatenate([r["obs"] for r in rs])
    dnull = np.concatenate([r["down_null"] for r in rs], axis=1)
    meta = [m for r in rs for m in r["meta"]]
    subj = np.array([m[0] for m in meta])
    ta = np.array([m[1] for m in meta]); tb = np.array([m[2] for m in meta])
    diags = [r["diag"] for r in rs]
    return obs, dnull, subj, ta, tb, diags


def table(results, out_lines):
    def emit(s=""):
        print(s); out_lines.append(s)
    rng = np.random.default_rng(7)
    hdr = (f"{'configuration':<34}{'subset':<16}{'n':>5}{'subj':>5}{'fitted':>8}{'scrambled':>10}"
           f"{'null 95%':>17}{'margin':>9}{'margin 95% (boot)':>21}{'p':>11}")
    rows = {}
    for cfg in configs():
        obs, dnull, subj, ta, tb, diags = assemble(results, cfg["name"])
        if len(obs) == 0:
            continue
        emit("")
        emit(hdr)
        subsets = [("all", np.ones(len(obs), bool))]
        if cfg["cohort"].startswith("gibson"):
            for k, rec in enumerate(RECOVS):
                cls = np.array([classify(a, b, rec) for a, b in zip(ta, tb)])
                tag = "" if k == 0 else f" R={rec:g}"
                if k == 0:
                    subsets += [("unforced", cls == "unforced"),
                                ("forced", cls == "forced"),
                                ("recovery", cls == "recovery"),
                                ("colonization", cls == "colonization"),
                                ("non-unforced", cls != "unforced")]
                else:
                    subsets += [("unforced" + tag, cls == "unforced")]
        for sname, sel in subsets:
            s = summarise(obs, dnull, subj, sel, rng)
            if s is None:
                continue
            rows[(cfg["name"], sname)] = s
            emit(f"{cfg['name']:<34}{sname:<16}{s['n']:>5}{s['n_subj']:>5}{s['f_obs']:>8.3f}"
                 f"{s['f_null']:>10.3f}   [{s['null_lo']:.3f},{s['null_hi']:.3f}]"
                 f"{s['margin']:>+9.3f}     [{s['m_lo']:+.3f},{s['m_hi']:+.3f}]{fmt_p(s):>11}")
        nit = [d_["nit"] for d_ in diags]; ok = [d_["success"] for d_ in diags]
        emit(f"    fits: {len(diags)} folds, L-BFGS nit {min(nit)}-{max(nit)}, converged "
             f"{sum(ok)}/{len(ok)}, d range [{min(d_['d_min'] for d_ in diags):.3g}, "
             f"{max(d_['d_max'] for d_ in diags):.3g}], DJ symmetry resid "
             f"{max(d_['sym_resid'] for d_ in diags):.1e}, train transitions "
             f"{min(d_['n_train'] for d_ in diags)}-{max(d_['n_train'] for d_ in diags)}, "
             f"fit time {np.mean([d_['fit_seconds'] for d_ in diags]):.0f} s/fold")
    return rows


# =========================================================================== figure
def ev3_reference(lam, B=B_SWEEP):
    """The published EV3 held-out protocol (relative abundance), as per-transition arrays.

    Uses the vectorised fit, which reproduces the published result exactly (--validate).
    """
    from constrained_vs_free_glv import load_communities, transitions, S as S12
    comms = load_communities()
    obs, dn, subj = [], [], []
    iu = np.triu_indices(S12, 1)
    for h in range(len(comms)):
        trh = [(a, g, np.zeros(0), m) for a, g, m in
               transitions([c for k, c in enumerate(comms) if k != h])]
        r, J, K, d, _, _ = fit_symmetrizable_general(trh, S12, 0, lam)
        V = np.stack([np.random.default_rng(80000 + 997 * h + i).permutation(K[iu])
                      for i in range(B)])
        H, Hs = potentials(comms[h][1], r, K, d, V)
        obs.append(np.diff(H)); dn.append(np.diff(Hs, axis=0) <= 0)
        subj += [h] * (len(H) - 1)
    return np.concatenate(obs), np.concatenate(dn, axis=0).T, np.array(subj)


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10})
    rng = np.random.default_rng(7)
    ev3 = {}
    for lam in LAMS:
        o, dn, sb = ev3_reference(lam, B_MAIN if lam == 0.5 else B_SWEEP)
        ev3[lam] = summarise(o, dn, sb, np.ones(len(o), bool), rng)
    groups = [("EV3 consortium\n(relative, batch)", [("ev3", "all", ev3[0.5])]),
              ("Mice, healthy\n(qPCR)", [("gibson_healthy", "all", None),
                                         ("gibson_healthy", "unforced", None)]),
              ("Mice, UC donor\n(qPCR)", [("gibson_uc", "all", None),
                                          ("gibson_uc", "unforced", None)]),
              ("Humans\n(flow-cytometry QMP)", [("vandeputte", "all", None)])]
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = ax[0]
    xt, xl, x = [], [], 0.0
    for glab, items in groups:
        x0 = x
        for c, sname, s in items:
            if s is None:
                s = rows.get((f"{c}|lam=0.5", sname))
            if s is None:
                continue
            a.plot([x, x], [s["null_lo"], s["null_hi"]], color="0.6", lw=7, solid_capstyle="butt")
            a.plot(x, s["f_null"], "_", color="0.2", ms=20, mew=2.2)
            a.plot(x, s["f_obs"], "o", color="#0072B2", ms=10, zorder=5)
            ptxt = "p" + fmt_p(s) if s["b"] == 0 else f"p = {s['p']:.2f}"
            a.text(x, max(s["f_obs"], s["null_hi"]) + 0.015, f"{s['margin']:+.3f}\n{ptxt}",
                   ha="center", va="bottom", fontsize=9.5)
            xt.append(x); xl.append(sname)
            x += 1.0
        a.text((x0 + x - 1.0) / 2, -0.17, glab, transform=a.get_xaxis_transform(),
               ha="center", va="top", fontsize=10.5)
        x += 0.7
    a.set_xticks(xt); a.set_xticklabels(xl)
    a.set_xlim(-0.6, x - 0.1)
    a.set_ylim(0.38, 1.0)
    a.axhline(0.5, color="0.5", ls=":", lw=1)
    a.set_ylabel("held-out fraction with $\\Delta H_D \\leq 0$")
    a.set_title("(a) Held-out descent, $\\lambda=0.5$")
    a.plot([], [], "o", color="#0072B2", label="fitted potential")
    a.plot([], [], color="0.6", lw=7, label="structure-scrambled, 95% range")
    a.legend(loc="upper right", frameon=False)
    b = ax[1]
    series = [("ev3", "EV3 consortium", "#000000"), ("gibson_healthy", "Mice, healthy", "#009E73"),
              ("gibson_uc", "Mice, UC donor", "#D55E00"), ("vandeputte", "Humans (QMP)", "#0072B2")]
    for k, (c, lab, col) in enumerate(series):
        xs, ms, lo, hi = [], [], [], []
        for lam in LAMS:
            s = ev3[lam] if c == "ev3" else rows.get((f"{c}|lam={lam}", "all"))
            if s is None:
                continue
            xs.append(lam * (1 + 0.05 * (k - 1.5))); ms.append(s["margin"])
            lo.append(s["margin"] - s["m_lo"]); hi.append(s["m_hi"] - s["margin"])
        b.errorbar(xs, ms, yerr=[lo, hi], fmt="o-", color=col, capsize=4, ms=7, label=lab)
    b.axhline(0, color="0.4", ls="--", lw=1)
    b.set_xscale("log"); b.set_xticks(LAMS); b.set_xticklabels([f"{v:g}" for v in LAMS])
    b.minorticks_off()
    b.set_xlabel("ridge penalty $\\lambda$")
    b.set_ylabel("margin, fitted minus scrambled")
    b.set_title("(b) Margin across regularisation, all transitions")
    b.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(FIG, dpi=200)
    plt.close(fig)
    print(f"[figure] {FIG}")
    for lam in LAMS:
        s = ev3[lam]
        print(f"EV3 reference lam={lam}: {s['f_obs']:.3f} vs {s['f_null']:.3f}, margin "
              f"{s['margin']:+.3f} [{s['m_lo']:+.3f},{s['m_hi']:+.3f}], p {fmt_p(s)}")


# =========================================================================== validation
def validate():
    """The vectorised fit must reproduce the published one, and the published EV3 result."""
    from constrained_vs_free_glv import load_communities, transitions, _constrained_inner, S as S12
    from lyapunov_descent import fit_symmetrizable
    os.chdir(ROOT)
    comms = load_communities()
    tr = transitions(comms)
    tr4 = [(a, g, np.zeros(0), m) for a, g, m in tr]
    rng = np.random.default_rng(3)
    u = rng.normal(0, 0.5, S12)
    r0, J0, mse0 = _constrained_inner(u, tr, 0.5)
    F = SymFit(tr4, S12, 0, 0.5)
    beta, d, mse1 = F.solve(u)
    r1, J1, _, _ = F.unpack(beta, d)
    print(f"inner solve at random u: max|dr|={np.abs(r1 - r0).max():.2e} "
          f"max|dJ|={np.abs(J1 - J0).max():.2e}  mse {mse0:.10f} vs {mse1:.10f}")
    rP, JP, dP = fit_symmetrizable(tr, 0.5)
    rV, JV, KV, dV, _, dg = fit_symmetrizable_general(tr4, S12, 0, 0.5)
    print(f"full fit: max|dr|={np.abs(rV - rP).max():.2e} max|dJ|={np.abs(JV - JP).max():.2e} "
          f"max|dd|={np.abs(dV - dP).max():.2e}  (nit {dg['nit']})")
    # published held-out protocol with the vectorised fit and vectorised scoring
    down = tot = 0
    null = np.zeros(5000)
    for h in range(len(comms)):
        trh = [(a, g, np.zeros(0), m) for a, g, m in
               transitions([c for k, c in enumerate(comms) if k != h])]
        r, J, K, dd, _, _ = fit_symmetrizable_general(trh, S12, 0, 0.5)
        X = comms[h][1]
        iu = np.triu_indices(S12, 1)
        V = np.stack([np.random.default_rng(80000 + 997 * h + i).permutation(K[iu])
                      for i in range(5000)])
        H, Hs = potentials(X, r, K, dd, V)
        dH = np.diff(H); dHs = np.diff(Hs, axis=0)
        down += int((dH <= 0).sum()); tot += len(dH)
        null += (dHs <= 0).sum(axis=0)
    f, fn = down / tot, null / tot
    print(f"EV3 held-out with the vectorised code: {down}/{tot} = {f:.3f} vs scrambled "
          f"{fn.mean():.3f}, margin {f - fn.mean():+.3f}, b = {int(np.sum(fn >= f))}/5000 "
          f"(published: 67/78 = 0.859 vs 0.759, margin +0.100, b = 0)")


# =========================================================================== main
def main():
    os.chdir(ROOT)
    if "--validate" in sys.argv:
        validate()
        return
    results = load_cache() if CACHE.exists() else []
    done = {r["cfg"] for r in results}
    if "--replot" not in sys.argv:
        jobs = []
        for cfg in configs():
            if cfg["name"] in done:
                continue
            nser = len(build_series(cfg))
            for h in range(nser):
                jobs.append((cfg, h))
        print(f"{len(jobs)} fold fits to run ({len(done)} configurations cached), {WORKERS} workers")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for k, res in enumerate(ex.map(run_fold, jobs, chunksize=1)):
                results.append(res)
                if (k + 1) % 10 == 0:
                    print(f"  {k + 1}/{len(jobs)} folds done, {time.time() - t0:.0f} s", flush=True)
        print(f"compute: {time.time() - t0:.0f} s")
        save_cache(results)
    lines = []
    rows = table(results, lines)
    figure(rows)


def save_cache(results):
    payload = {}
    for k, r in enumerate(results):
        payload[f"r{k}_cfg"] = np.array(r["cfg"])
        payload[f"r{k}_h"] = np.array(r["h"])
        payload[f"r{k}_obs"] = r["obs"]
        payload[f"r{k}_null_bits"] = np.packbits(r["down_null"], axis=0)
        payload[f"r{k}_nB"] = np.array(r["down_null"].shape[0])
        payload[f"r{k}_meta"] = np.array(r["meta"], dtype=float).reshape(-1, 5)
        payload[f"r{k}_diag"] = np.array(repr(r["diag"]))
    payload["n_results"] = np.array(len(results))
    np.savez_compressed(CACHE, **payload)
    print(f"[cache] {CACHE}")


def load_cache():
    z = np.load(CACHE, allow_pickle=False)
    out = []
    for k in range(int(z["n_results"])):
        nB = int(z[f"r{k}_nB"])
        meta = [tuple(row) for row in z[f"r{k}_meta"]]
        meta = [(int(m[0]), m[1], m[2], int(m[3]), int(m[4])) for m in meta]
        n = len(meta)
        bits = np.unpackbits(z[f"r{k}_null_bits"], axis=0, count=nB).astype(bool) if n else \
            np.zeros((nB, 0), bool)
        out.append(dict(cfg=str(z[f"r{k}_cfg"]), h=int(z[f"r{k}_h"]), obs=z[f"r{k}_obs"],
                        down_null=bits, meta=meta, diag=eval(str(z[f"r{k}_diag"]))))
    return out


if __name__ == "__main__":
    main()
