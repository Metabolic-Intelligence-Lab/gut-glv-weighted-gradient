"""
Does asymmetry earn its keep? Constrained vs free gLV on raw Venturelli data (referee-4 P0.2)
============================================================================================

Referee 4's central statistical demand: the designed-community matrices are inferred, so an
exact symmetrizability violation of a point estimate is not proof unless the data *require*
the asymmetry. The clean test is predictive: fit two gLV models to the same raw trajectories,

  free        :  dot(ln x_i) = r_i + sum_j J_ij x_j           (J unrestricted)
  constrained :  J = D^{-1} K,  K = K^T,  d_i>0               (diagonally symmetrizable,
                 i.e. an exact weighted-gradient / potential system by construction)

and compare their OUT-OF-SAMPLE one-step prediction error under leave-one-community-out
cross-validation. If the free model predicts held-out communities better, the data require
non-symmetrizability (the thesis is supported by prediction, not just by a point estimate);
if not, the apparent asymmetry buys no predictive power and should not be over-read.

Data: Venturelli 2018 EV3 -- the full 12-member consortium and each single-species dropout
(13 communities x ~7 timepoints, relative abundance; species order BH CA BU PC BO BV BT EL
FP CH DP ER). Relative abundances are compositional, but both models see the identical data,
so the *comparison* is fair; we report it as a predictive contrast, not an absolute fit.

Run:  python3 constrained_vs_free_glv.py     (writes figures/fig_constrained_free.png)
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
OUT = "figures"
SPECIES = ["BH","CA","BU","PC","BO","BV","BT","EL","FP","CH","DP","ER"]
S = 12
EPS = 1e-3
ABMIN = 5e-3          # min relative abundance for a reliable per-capita growth
LAM = 0.5             # ridge penalty on interaction coefficients (both models)


def load_communities():
    import openpyxl
    wb = openpyxl.load_workbook("data/venturelli2018/timeseries_multispecies_EV3.xlsx",
                                read_only=True, data_only=True)
    comms, name, block = [], None, []
    for r in wb["Dataset EV3"].iter_rows(values_only=True):
        s = r[0]
        if s is None:
            continue
        s = str(s).strip()
        if s.startswith("'") or (not s[0].isdigit() and "." not in s.split()[0]):
            if name is not None and len(block) >= 3:
                comms.append((name, np.array(block)))
            name = s.strip("'"); block = []
        else:
            vals = [float(v) for v in s.split()]
            if len(vals) == S:
                block.append(vals)
    if name is not None and len(block) >= 3:
        comms.append((name, np.array(block)))
    return comms


def transitions(comms):
    """List of (x_a[S], g[S], mask[S]) one-step transitions (dt=1)."""
    T = []
    for _, X in comms:
        for a, b in zip(X[:-1], X[1:]):
            g = (np.log(b + EPS) - np.log(a + EPS))
            mask = (a > ABMIN) & (b > ABMIN)
            T.append((a, g, mask))
    return T


# --------------------------------------------------------------------------- #
def fit_free(trans, lam=LAM):
    """Ridge fit of r_i, J (free) per species: g_i = r_i + sum_j J_ij x_j."""
    r = np.zeros(S); J = np.zeros((S, S))
    for i in range(S):
        rows, ys = [], []
        for a, g, m in trans:
            if m[i]:
                rows.append(np.concatenate([[1.0], a])); ys.append(g[i])
        if len(rows) < S:      # too few equations for this species
            continue
        Z = np.array(rows); y = np.array(ys)
        P = np.full(S + 1, lam); P[0] = 0.0; P[1 + i] = 0.0   # ridge on OFF-diagonal J only
        beta = np.linalg.solve(Z.T @ Z + np.diag(P) + 1e-6 * np.eye(S + 1), Z.T @ y)
        r[i] = beta[0]; J[i] = beta[1:]
    return r, J


def _constrained_inner(u, trans, lam):
    """Given log-weights u (d=exp(u-mean)), best symmetric K and r by weighted linear
    ridge; returns (r, J=D^{-1}K, training MSE)."""
    d = np.exp(u - u.mean())
    # params: r (S) then K upper-tri incl diag, index map
    pairs = [(i, j) for i in range(S) for j in range(i, S)]
    pk = {(i, j): S + k for k, (i, j) in enumerate(pairs)}
    npar = S + len(pairs)
    ATA = np.zeros((npar, npar)); ATy = np.zeros(npar)
    for a, g, m in trans:
        for i in range(S):
            if not m[i]:
                continue
            row = np.zeros(npar); row[i] = 1.0
            for j in range(S):
                key = (i, j) if i <= j else (j, i)
                row[pk[key]] += a[j] / d[i]        # J_ij = K_ij/d_i, K symmetric
            ATA += np.outer(row, row); ATy += row * g[i]
    # ridge on OFF-diagonal J only, expressed through K: |J_ij|^2 = K_ij^2/d_i^2 and
    # |J_ji|^2 = K_ij^2/d_j^2, so K_ij (i<j) is penalised by lam*(1/d_i^2+1/d_j^2);
    # diagonal K_ii (self-interaction) is not penalised. Same off-diagonal-J penalty as free.
    reg = np.zeros(npar)
    for (i, j) in pairs:
        if i != j:
            reg[pk[(i, j)]] = lam * (1.0 / d[i] ** 2 + 1.0 / d[j] ** 2)
    beta = np.linalg.solve(ATA + np.diag(reg) + 1e-6 * np.eye(npar), ATy)
    r = beta[:S]
    K = np.zeros((S, S))
    for (i, j) in pairs:
        K[i, j] = K[j, i] = beta[pk[(i, j)]]
    J = K / d[:, None]
    # training MSE
    se, n = 0.0, 0
    for a, g, m in trans:
        pred = r + J @ a
        se += np.sum((g[m] - pred[m]) ** 2); n += m.sum()
    return r, J, se / max(n, 1)


def fit_constrained(trans, lam=LAM):
    """Optimise the diagonal weights u; inner solve gives K,r. Start from symmetric (u=0)."""
    res = minimize(lambda u: _constrained_inner(u, trans, lam)[2], np.zeros(S),
                   method="L-BFGS-B", options={"maxiter": 60})
    r, J, _ = _constrained_inner(res.x, trans, lam)
    return r, J


def pred_mse(r, J, trans):
    se, n = 0.0, 0
    for a, g, m in trans:
        pred = r + J @ a
        se += np.sum((g[m] - pred[m]) ** 2); n += m.sum()
    return se / max(n, 1)


def loco_cv(comms, lam=LAM):
    """Leave-one-community-out: fit on the rest, predict the held-out community."""
    free_e, con_e = [], []
    for h in range(len(comms)):
        train = transitions([c for k, c in enumerate(comms) if k != h])
        test = transitions([comms[h]])
        if not test:
            continue
        rf, Jf = fit_free(train, lam); rc, Jc = fit_constrained(train, lam)
        free_e.append(pred_mse(rf, Jf, test)); con_e.append(pred_mse(rc, Jc, test))
    return np.array(free_e), np.array(con_e)


def invariants(J):
    iu = np.triu_indices(S, 1); a, b = J[iu], J.T[iu]
    both = (a != 0) & (b != 0)
    f_sign = (both & (np.sign(a) != np.sign(b))).sum() / max(both.sum(), 1)
    conc = both & (np.sign(a) == np.sign(b)); ii, jj = iu[0][conc], iu[1][conc]
    if len(ii) == 0:
        return f_sign, np.nan
    bij = np.log(np.abs(a[conc] / b[conc]))
    B = np.zeros((len(ii), S)); B[np.arange(len(ii)), jj] = 1; B[np.arange(len(ii)), ii] = -1
    u, *_ = np.linalg.lstsq(B, bij, rcond=None)
    return f_sign, float(np.sqrt(np.mean((bij - B @ u) ** 2)))


def main():
    import os; os.makedirs(OUT, exist_ok=True)
    comms = load_communities()
    print(f"communities: {len(comms)}  ({', '.join(n for n, _ in comms)})")
    print(f"total transitions: {len(transitions(comms))}\n")

    rf, Jf = fit_free(transitions(comms)); rc, Jc = fit_constrained(transitions(comms))
    fsf, rcf = invariants(Jf); fsc, rcc = invariants(Jc)
    print("full-data fits:")
    print(f"  free  : train MSE={pred_mse(rf,Jf,transitions(comms)):.4f}  f_sign={fsf:.3f}  r_cycle={rcf:.3f}")
    print(f"  constr: train MSE={pred_mse(rc,Jc,transitions(comms)):.4f}  f_sign={fsc:.3f}  r_cycle={rcc:.3f} (=0 by construction)")

    print("\nleave-one-community-out cross-validation (out-of-sample MSE):")
    fe, ce = loco_cv(comms)
    print(f"  free   CV MSE = {fe.mean():.4f} +/- {fe.std()/np.sqrt(len(fe)):.4f}")
    print(f"  constr CV MSE = {ce.mean():.4f} +/- {ce.std()/np.sqrt(len(ce)):.4f}")
    diff = ce - fe                                   # >0 means free predicts better
    from math import sqrt
    tstat = diff.mean() / (diff.std(ddof=1) / sqrt(len(diff)) + 1e-12)
    print(f"  paired diff (constr - free) = {diff.mean():+.4f} +/- {diff.std(ddof=1)/sqrt(len(diff)):.4f}  (t={tstat:+.2f})")
    win = (fe < ce).mean()
    verdict = ("free predicts better -> data REQUIRE asymmetry"
               if diff.mean() > 0 and tstat > 1.5 else
               "no robust predictive gain from asymmetry -> point-estimate violation not confirmed by prediction")
    print(f"  free wins in {win*100:.0f}% of folds. Verdict: {verdict}")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax[0].scatter(ce, fe, s=60, color="#0072B2", zorder=5, edgecolor="white")
    lim = [0, max(fe.max(), ce.max()) * 1.05]
    ax[0].plot(lim, lim, "--", color="grey", lw=1)
    ax[0].set_xlabel("constrained (symmetrizable) CV MSE")
    ax[0].set_ylabel("free CV MSE")
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_title("a  Held-out prediction, per community\n(below diagonal = free wins)", loc="left", fontsize=11)
    lab = [n for n, _ in comms]
    ax[1].bar(np.arange(len(diff)) - 0.0, diff, color=["#1b9e77" if d > 0 else "#D55E00" for d in diff])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(range(len(lab))); ax[1].set_xticklabels(lab, rotation=90, fontsize=7)
    ax[1].set_ylabel("CV MSE:  constrained $-$ free")
    ax[1].set_title(f"b  Per-community advantage of asymmetry\nmean {diff.mean():+.4f} (t={tstat:+.2f})", loc="left", fontsize=11)
    for a in ax:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_constrained_free.png", dpi=160, bbox_inches="tight")
    print(f"\n[fig] {OUT}/fig_constrained_free.png written")
    np.savez(f"{OUT}/constrained_free.npz", free_cv=fe, con_cv=ce,
             f_sign_free=fsf, r_cycle_free=rcf, tstat=tstat)


if __name__ == "__main__":
    main()
