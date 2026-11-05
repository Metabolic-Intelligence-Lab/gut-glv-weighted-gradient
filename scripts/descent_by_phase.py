"""
The held-out descent test, split by growth phase (BIB revision, node A)
=======================================================================

Reviewers 2 and 4 object that Eq. (5) is derived in absolute abundances while the descent test
is scored on the relative abundances of Dataset EV3. The revision answers with a proposition:
if the total biomass N is constant over a transition, n = N* x, the composition follows a gLV
with J' = N* J and the same symmetrizer, and H_D(n) = N* H'_D(x). Compositional descent is then
EQUIVALENT to descent in absolute abundance, and N* never needs to be known.

Whether N is constant is an empirical question, and the original study answers it.
Venturelli et al. 2018 (Mol Syst Biol 14:e8157), Methods: samples every ~12 h for 72 h (the
seven rows of each EV3 block), serial transfers every 24 h at 1:20 dilution. Appendix Fig S10a
plots OD600 for all 13 communities: within each 24 h stage the biomass rises to a plateau in
roughly 10 h and then stays approximately flat. So the six transitions of every community are
of three kinds:

    k = 0        0 -> 12 h                     growth from the inoculum, N rises steeply
    k = 1, 3, 5  12->24, 36->48, 60->72 h      biomass plateau, N approximately constant
    k = 2, 4     24->36, 48->60 h              span a 1:20 transfer and the regrowth after it

Only the plateau transitions satisfy the proposition. This script re-scores the published
held-out protocol (lyapunov_robustness.loco_descent: same fits, same scrambled potentials, same
seeds, so the pooled result reproduces 86% vs 76%) on each kind separately, with a
community-level bootstrap of the margin, and repeats everything without the BU- community,
which the original study reports as contaminated by BU from 47.5 h to 72 h.

Run:  OMP_NUM_THREADS=4 python3 scripts/descent_by_phase.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained_vs_free_glv import load_communities, transitions          # noqa: E402
from lyapunov_descent import fit_symmetrizable, H_D                         # noqa: E402
from lyapunov_robustness import scramble, NPERM                             # noqa: E402

import warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

PHASE = {0: "growth", 1: "plateau", 2: "transfer", 3: "plateau", 4: "transfer", 5: "plateau"}
NBOOT = 2000


def deltas(X, r, J, d):
    h = [H_D(x, r, J, d) for x in X]
    return np.diff(h)


def held_out(comms, lam=0.5, nperm=NPERM):
    """Per held-out transition: observed Delta H_D and its nperm scrambled counterparts.

    Seeds match lyapunov_robustness.loco_descent exactly, so scramble i of fold h is the same
    matrix there and here.
    """
    rows, null = [], []
    for h in range(len(comms)):
        train = transitions([c for i, c in enumerate(comms) if i != h])
        r, J, d = fit_symmetrizable(train, lam)
        name, X = comms[h]
        obs = deltas(X, r, J, d)
        K = d[:, None] * J
        nd = np.empty((nperm, len(obs)))
        for i in range(nperm):
            rng = np.random.default_rng(80000 + 997 * h + i)
            nd[i] = deltas(X, r, scramble(K, d, rng), d)
        for k in range(len(obs)):
            rows.append((h, name, k, PHASE.get(k, "?"), obs[k]))
            null.append(nd[:, k])
    return rows, np.array(null)          # null: (n_transitions, nperm)


def summary(rows, null, sel, rng):
    o = np.array([r[4] <= 0 for r in rows])
    comm = np.array([r[0] for r in rows])
    down_null = null <= 0
    f_obs = o[sel].mean()
    f_null = down_null[sel].mean(axis=0)
    k = int(np.sum(f_null >= f_obs))
    p = (k + 1) / (len(f_null) + 1)
    # community-level bootstrap of the margin: resample whole communities
    ids = np.unique(comm[sel])
    margins = []
    for _ in range(NBOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([np.where(sel & (comm == c))[0] for c in pick])
        margins.append(o[idx].mean() - down_null[idx].mean())
    lo, hi = np.percentile(margins, [2.5, 97.5])
    return dict(n=int(sel.sum()), down=int(o[sel].sum()), f_obs=f_obs, f_null=f_null.mean(),
                null_lo=np.percentile(f_null, 2.5), null_hi=np.percentile(f_null, 97.5),
                margin=f_obs - f_null.mean(), m_lo=lo, m_hi=hi, k=k, p=p, nperm=len(f_null))


def report(label, comms):
    rows, null = held_out(comms)
    phase = np.array([r[3] for r in rows])
    rng = np.random.default_rng(7)
    print(f"\n=== {label}: {len(comms)} communities, {len(rows)} held-out transitions ===")
    print(f"{'subset':<22}{'n':>4}{'down':>6}{'fitted':>8}{'scrambled':>11}"
          f"{'null 95%':>16}{'margin':>9}{'margin 95% (boot)':>20}{'p':>12}")
    for name, sel in [("all", np.ones(len(rows), bool)),
                      ("plateau (k=1,3,5)", phase == "plateau"),
                      ("transfer (k=2,4)", phase == "transfer"),
                      ("growth (k=0)", phase == "growth"),
                      ("non-plateau", phase != "plateau")]:
        s = summary(rows, null, sel, rng)
        p = f"<{1/(s['nperm']+1):.1e}" if s["k"] == 0 else f"{s['p']:.4f}"
        print(f"{name:<22}{s['n']:>4}{s['down']:>6}{s['f_obs']:>8.3f}{s['f_null']:>11.3f}"
              f"   [{s['null_lo']:.3f},{s['null_hi']:.3f}]{s['margin']:>+9.3f}"
              f"     [{s['m_lo']:+.3f},{s['m_hi']:+.3f}]{p:>12}")
    # where do the ascents sit?
    asc = [(r[1], r[2], r[3], r[4]) for r in rows if r[4] > 0]
    print(f"ascending held-out transitions: {len(asc)}")
    for nm, k, ph, dl in asc:
        print(f"    {nm:<6} k={k} {ph:<9} dH={dl:+.3f}")
    return rows, null


def main():
    comms = load_communities()
    names = [c[0] for c in comms]
    print("communities:", names)
    report("published protocol", comms)
    bu = [c for c in comms if c[0].strip().upper() != "BU-"]
    if len(bu) == len(comms):
        print("\n(no BU- community found by name; BU sensitivity skipped)")
    else:
        report("without the BU- community (contaminated from 47.5 h)", bu)


if __name__ == "__main__":
    main()
