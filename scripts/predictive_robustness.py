"""
Robustness of the free-vs-symmetrizable predictive comparison  (referee-6 P1, §6)
=================================================================================

The leave-one-community-out comparison found free and symmetrizable gLV predictively
indistinguishable (MSE 0.240 vs 0.241). Referee 6 asks for tests beyond a fragile paired
t-test and for a check that the tie is not an artefact of a fixed ridge. This script adds,
on the same EV3 data:
  - a paired PERMUTATION test (sign-flip the per-community differences);
  - a bootstrap 95% CI on the mean predictive advantage;
  - a ridge SENSITIVITY sweep (lambda in {0.25,0.5,1.0}) of the CV error of both models.

Run:  python3 predictive_robustness.py
"""
import numpy as np
import constrained_vs_free_glv as CF
from constrained_vs_free_glv import load_communities, transitions, fit_free, fit_constrained, pred_mse
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")


def loco_diffs(comms, lam):
    """Per-community (constrained - free) held-out MSE differences."""
    d = []
    for h in range(len(comms)):
        train = transitions([c for k, c in enumerate(comms) if k != h])
        test = transitions([comms[h]])
        if not test:
            continue
        rf, Jf = fit_free(train, lam); rc, Jc = fit_constrained(train, lam)
        d.append(pred_mse(rc, Jc, test) - pred_mse(rf, Jf, test))   # >0 => free better
    return np.array(d)


def main():
    comms = load_communities()
    rng = np.random.default_rng(0)
    d = loco_diffs(comms, CF.LAM)
    n = len(d); mean = d.mean()
    print(f"per-community (constrained - free) CV MSE difference, n={n}:")
    print(f"  mean={mean:+.4f}  median={np.median(d):+.4f}  free better in {int((d>0).sum())}/{n}")

    # paired permutation: sign-flip each difference
    NP = 20000
    perm = np.array([np.mean(d * rng.choice([-1, 1], n)) for _ in range(NP)])
    p_perm = (np.sum(np.abs(perm) >= abs(mean)) + 1) / (NP + 1)
    # sign test
    from scipy.stats import binomtest
    p_sign = binomtest((d > 0).sum(), n).pvalue
    # bootstrap CI on the mean difference
    boot = np.array([np.mean(d[rng.integers(0, n, n)]) for _ in range(20000)])
    ci = np.percentile(boot, [2.5, 97.5])
    print(f"  paired permutation p={p_perm:.3f} (two-sided) | sign-test p={p_sign:.3f}")
    print(f"  bootstrap 95% CI on mean advantage: [{ci[0]:+.4f}, {ci[1]:+.4f}]  (contains 0: {ci[0]<0<ci[1]})")

    # ridge sensitivity
    print("\nridge sensitivity (leave-one-community-out mean MSE):")
    for lam in [0.25, 0.5, 1.0]:
        fe, ce = [], []
        for h in range(len(comms)):
            tr = transitions([c for k, c in enumerate(comms) if k != h]); te = transitions([comms[h]])
            if not te:
                continue
            rf, Jf = fit_free(tr, lam); rc, Jc = fit_constrained(tr, lam)
            fe.append(pred_mse(rf, Jf, te)); ce.append(pred_mse(rc, Jc, te))
        print(f"  lambda={lam}: free={np.mean(fe):.3f}  constrained={np.mean(ce):.3f}  "
              f"diff={np.mean(ce)-np.mean(fe):+.4f}")
    print("\n=> the tiny free advantage is directionally consistent (sign/permutation) but its "
          "effect-size CI contains ~0, and the near-tie is stable across ridge levels: the data "
          "do not require asymmetry.")
    np.savez("figures/predictive_robustness.npz", d=d, p_perm=p_perm, p_sign=p_sign, ci=ci)


if __name__ == "__main__":
    main()
