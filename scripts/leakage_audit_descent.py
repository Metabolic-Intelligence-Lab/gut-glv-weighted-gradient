r"""
Control D2: is the held-out descent margin inflated by the choice of the ridge penalty?
=======================================================================================
The held-out margin is the one positive result that survives a multiplicity correction over
every p the paper reports, so it carries the paper. Its protocol holds the ridge penalty at
lambda = 0.5, "fixed in advance for every analysis", and refits the weights d inside each fold.
Reading the code confirms the fold is clean: the scored community is excluded from the fit, d
comes from the training folds only, and `transitions` computes no statistic over the full
dataset, so nothing is standardised across folds. One decision is left outside the fold, the
penalty itself, and a referee is entitled to ask what it bought.

This measures it rather than arguing about it, by running three protocols that differ only in
where lambda comes from:
  P1 PUBLISHED   lambda = 0.5 for every fold, as the paper states.
  P2 NESTED      lambda chosen inside each outer fold, by leave-one-community-out predictive
                 MSE over the twelve training communities alone. The held-out community touches
                 nothing, not the fit and not the selection. This is the protocol a referee
                 asking about leakage wants to see.
  P3 ORACLE      lambda chosen to maximise the pooled margin itself, with every community
                 visible. Not a candidate protocol: it bounds what tuning could have bought,
                 which is the quantity the question is really about.

Statistic: the pooled held-out margin, the fitted descent fraction minus the mean fraction of
magnitude-preserving structure-scrambled potentials scored on the identical held-out
transitions. The scramble is drawn inside each fold from that fold's own fitted potential.

MANDATORY CONTROL: P1 must reproduce the published 0.86 held-out fraction and +0.100 margin to
the precision the manuscript prints them. If it does not, this script has the wrong protocol
and none of its numbers mean anything.

CRITERION, fixed before execution:
  1. The claim stands as written if P2's margin falls inside the community-level bootstrap
     interval the paper already reports for the margin, [+0.050, +0.156]. A penalty chosen
     honestly inside the fold then lands where the fixed one did, and the fixed choice bought
     nothing.
  2. The claim is corrected to P2's value if P2 falls below that interval: the published margin
     would then depend on a penalty chosen with the held-out data in view.
  3. Whatever happens, P3 minus P1 is reported as the margin tuning could have added, since a
     reader cannot otherwise tell whether 0.5 was a lucky choice.

Run:  OMP_NUM_THREADS=1 python3 scripts/leakage_audit_descent.py
Writes: figures/leakage_audit_descent.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from constrained_vs_free_glv import load_communities, transitions, pred_mse   # noqa: E402
from lyapunov_robustness import fit_symmetrizable, scramble, descent_fraction  # noqa: E402

LAMBDAS = [0.25, 0.5, 1.0, 2.0]        # the sweep the manuscript already reports
NPERM = 400                             # for the MEAN of the scrambled fraction, not a p
PUBLISHED = dict(fraction=0.86, margin=0.100, boot_lo=0.050, boot_hi=0.156)
OUT = "figures/leakage_audit_descent.json"


def fold_margin(comms, h, lam, nperm=NPERM):
    """Fit on every community but h, score descent on h, and on scrambles of that same fit."""
    train = transitions([c for k, c in enumerate(comms) if k != h])
    r, J, d = fit_symmetrizable(train, lam)
    f, n, _ = descent_fraction([comms[h]], r, J, d)
    K = d[:, None] * J
    null = np.empty(nperm)
    for i in range(nperm):
        rng = np.random.default_rng(80000 + 997 * h + i)
        fs, _, _ = descent_fraction([comms[h]], r, scramble(K, d, rng), d)
        null[i] = fs
    return f, n, float(null.mean())


def pooled(comms, lam_of_fold, nperm=NPERM):
    down = tot = 0.0
    nulldown = 0.0
    per = []
    for h in range(len(comms)):
        lam = lam_of_fold(h)
        f, n, nm = fold_margin(comms, h, lam, nperm)
        down += f * n
        nulldown += nm * n
        tot += n
        per.append(dict(community=str(comms[h][0]), lam=lam, fraction=f, n=int(n),
                        scrambled=nm))
    return down / tot, nulldown / tot, int(tot), per


def pick_lambda_inside(comms, h):
    """Leave-one-community-out predictive MSE over the TRAINING folds only."""
    train_idx = [k for k in range(len(comms)) if k != h]
    best, best_mse = None, np.inf
    for lam in LAMBDAS:
        se = 0.0
        for inner in train_idx:
            fit_on = transitions([comms[k] for k in train_idx if k != inner])
            r, J, _ = fit_symmetrizable(fit_on, lam)
            se += pred_mse(r, J, transitions([comms[inner]]))
        if se < best_mse:
            best, best_mse = lam, se
    return best


def main():
    t0 = time.time()
    comms = load_communities()
    print(f"{len(comms)} communities, lambdas {LAMBDAS}, {NPERM} scrambles per fold\n")

    print("P1 PUBLISHED, lambda = 0.5 in every fold")
    f1, s1, n1, per1 = pooled(comms, lambda h: 0.5)
    m1 = f1 - s1
    print(f"   held-out fraction {f1:.3f} ({n1} transitions), scrambled {s1:.3f}, "
          f"margin {m1:+.3f}")

    ok_f = abs(f1 - PUBLISHED["fraction"]) < 0.005
    ok_m = abs(m1 - PUBLISHED["margin"]) < 0.005
    print(f"\n--- mandatory control ---\n   published {PUBLISHED['fraction']:.2f} / "
          f"{PUBLISHED['margin']:+.3f}   recomputed {f1:.3f} / {m1:+.3f}   "
          f"{'passed' if ok_f and ok_m else 'FAILED'}")
    if not (ok_f and ok_m):
        print("   the protocol here is not the published one; nothing below is usable")
        return

    print(f"\nP2 NESTED, lambda chosen inside each fold ({time.time()-t0:.0f} s so far, "
          "this is the slow part)")
    picks = {}
    for h in range(len(comms)):
        picks[h] = pick_lambda_inside(comms, h)
        print(f"   fold {h:>2} ({str(comms[h][0])[:18]:<18}) picks lambda = {picks[h]}",
              flush=True)
    f2, s2, n2, per2 = pooled(comms, lambda h: picks[h])
    m2 = f2 - s2
    print(f"   held-out fraction {f2:.3f}, scrambled {s2:.3f}, margin {m2:+.3f}")

    print("\nP3 ORACLE, one lambda for all folds, chosen to maximise the margin")
    oracle = {}
    for lam in LAMBDAS:
        f, s, _, _ = pooled(comms, lambda h, L=lam: L)
        oracle[lam] = f - s
        print(f"   lambda {lam:<5} margin {f - s:+.3f}")
    lam_star = max(oracle, key=oracle.get)
    m3 = oracle[lam_star]

    print("\n--- criterion ---")
    inside = PUBLISHED["boot_lo"] <= m2 <= PUBLISHED["boot_hi"]
    if inside:
        print(f"BRANCH 1: the nested margin {m2:+.3f} falls inside the interval the paper "
              f"already reports, [{PUBLISHED['boot_lo']:+.3f},{PUBLISHED['boot_hi']:+.3f}]. "
              "Choosing the penalty honestly inside each fold lands where the fixed choice "
              "did, so the fixed choice bought nothing. Nothing to change.")
    else:
        print(f"BRANCH 2, verified defect: the nested margin is {m2:+.3f}, outside "
              f"[{PUBLISHED['boot_lo']:+.3f},{PUBLISHED['boot_hi']:+.3f}]. The published "
              "margin depends on a penalty not chosen inside the fold.")
    print(f"\nBRANCH 3 report: the best single penalty is lambda = {lam_star} at {m3:+.3f}, "
          f"so tuning could have added {m3 - m1:+.3f} over the published {m1:+.3f}. "
          f"lambda = 0.5 is {'not' if m3 > m1 else ''} the favourable choice"
          f"{', which is the point' if m3 > m1 else ''}.")
    counts = {str(l): sum(1 for v in picks.values() if v == l) for l in LAMBDAS}
    print(f"\nwhich penalty the inner CV prefers, by fold: {counts}")

    with open(OUT, "w") as f:
        json.dump(dict(generated_by="scripts/leakage_audit_descent.py", lambdas=LAMBDAS,
                       nperm=NPERM, published=PUBLISHED,
                       P1=dict(fraction=f1, scrambled=s1, margin=m1, n=n1, per_fold=per1),
                       P2=dict(fraction=f2, scrambled=s2, margin=m2, n=n2, per_fold=per2,
                               picks={str(k): v for k, v in picks.items()}, counts=counts),
                       P3=dict(by_lambda={str(k): v for k, v in oracle.items()},
                               best_lambda=lam_star, margin=m3,
                               tuning_gain=m3 - m1),
                       verdict="no selection effect" if inside else "selection effect",
                       seconds=time.time() - t0), f, indent=1)
    print(f"\nwrote {OUT}  ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
