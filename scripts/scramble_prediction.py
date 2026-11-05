r"""
Does the fitted arrangement also predict better than the same scrambles, or only descend better?
================================================================================================
The submitted text claimed "No accuracy comparison measures that quantity", which was removed in
revision because it is not supported and because `lyapunov_robustness.py`'s own docstring says
the scramble "destroys structure but also destroys fit quality, so it cannot by itself separate"
the two hypotheses. Removing the sentence left the gap open rather than closing it. This closes
it, by asking the scramble comparison the predictive question as well as the descent question, on
the identical folds and the identical scrambles.

If the fitted arrangement orders held-out relaxation better than its own scrambles but does NOT
predict held-out transitions better than them, then the descent margin is not a restatement of
fit quality, and the removed sentence can be replaced by a measured one. If it predicts better
too, the margin does not isolate descent, and the paper must say so.

Statistic, per fold: the held-out one-step predictive MSE of the fitted (r, J), and of each
magnitude-preserving structure scramble of the same potential with the same weights d, scored on
the same held-out transitions. Pooled over folds by transition count, exactly as the descent
fraction is pooled. Reported as the advantage, scrambled minus fitted, so a positive value means
the fit predicts better. The Monte Carlo p is the fraction of scrambles whose pooled advantage
is at least the fitted one, which here is the fraction reaching zero or above.

MANDATORY CONTROL: the same run must reproduce the published held-out descent fraction 0.86 and
margin +0.100, since it is the same folds, the same scrambles and the same seeds. If it does
not, the comparison is not the published one and nothing here is usable.

CRITERION, fixed before execution, three branches:
  1. The fitted arrangement does not predict better than its scrambles (p > 0.05). The descent
     margin then separates the two hypotheses the accuracy comparison cannot, and the text may
     say so as a measured result.
  2. It predicts better (p <= 0.05) but by less than the descent margin separates. Then the
     margin is partly fit quality, and the text must state both numbers.
  3. It predicts better and the predictive separation is at least as strong as the descent one.
     Then the descent test adds nothing over accuracy and the paper's central instrument is
     redundant; that would be the most consequential outcome and would have to be reported.

Run:  OMP_NUM_THREADS=1 python3 scripts/scramble_prediction.py
Writes: figures/scramble_prediction.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constrained_vs_free_glv import load_communities, transitions, pred_mse       # noqa: E402
from lyapunov_robustness import fit_symmetrizable, scramble, descent_fraction     # noqa: E402

NPERM = 400
LAM = 0.5
OUT = "figures/scramble_prediction.json"
PUB = dict(fraction=0.86, margin=0.100)


def main():
    t0 = time.time()
    comms = load_communities()
    n_fit = n_tot = 0.0
    d_fit = d_scr = 0.0
    mse_fit_num = 0.0
    mse_scr_num = np.zeros(NPERM)
    per = []
    for h in range(len(comms)):
        train = transitions([c for k, c in enumerate(comms) if k != h])
        test = transitions([comms[h]])
        r, J, d = fit_symmetrizable(train, LAM)
        f, n, _ = descent_fraction([comms[h]], r, J, d)
        m_fit = pred_mse(r, J, test)
        K = d[:, None] * J
        ms, fs = np.empty(NPERM), np.empty(NPERM)
        for i in range(NPERM):
            rng = np.random.default_rng(80000 + 997 * h + i)   # the published seeds
            # scramble() already returns D^-1 K_s, i.e. the J of the scrambled potential.
            # Dividing by d again was the first version of this script and the mandatory
            # control caught it: the scrambled descent fraction came out 0.659 against the
            # published 0.758 and the scrambled MSE blew up to 1e9.
            Js = scramble(K, d, rng)
            ms[i] = pred_mse(r, Js, test)
            fs[i], _, _ = descent_fraction([comms[h]], r, Js, d)
        nt = len(test)
        d_fit += f * n
        d_scr += fs.mean() * n
        n_tot += n
        mse_fit_num += m_fit * nt
        mse_scr_num += ms * nt
        n_fit += nt
        per.append(dict(community=str(comms[h][0]), n_trans=int(nt),
                        descent_fitted=f, descent_scrambled=float(fs.mean()),
                        mse_fitted=m_fit, mse_scrambled=float(ms.mean())))
        print(f"  fold {h:>2} {str(comms[h][0])[:14]:<14} descent {f:.3f} vs {fs.mean():.3f}   "
              f"mse {m_fit:.4f} vs {ms.mean():.4f}", flush=True)

    desc_f, desc_s = d_fit / n_tot, d_scr / n_tot
    mse_f = mse_fit_num / n_fit
    mse_s = mse_scr_num / n_fit
    adv = mse_s - mse_f                      # positive = the fit predicts better
    p_pred = float(np.mean(adv <= 0.0))      # scrambles that match or beat the fit
    print(f"\ndescent: fitted {desc_f:.3f}, scrambled {desc_s:.3f}, margin {desc_f-desc_s:+.3f}")
    print(f"prediction: fitted MSE {mse_f:.4f}, scrambled mean {mse_s.mean():.4f} "
          f"[{np.percentile(mse_s,2.5):.4f},{np.percentile(mse_s,97.5):.4f}]")
    print(f"advantage {adv.mean():+.4f} ({100*adv.mean()/mse_s.mean():+.1f}% of the scrambled "
          f"MSE), p = {p_pred:.4f} over {NPERM} scrambles")

    print("\n--- mandatory control ---")
    ok = abs(desc_f - PUB["fraction"]) < 0.006 and abs((desc_f-desc_s) - PUB["margin"]) < 0.006
    print(f"   published {PUB['fraction']:.2f} / {PUB['margin']:+.3f}   here "
          f"{desc_f:.3f} / {desc_f-desc_s:+.3f}   {'passed' if ok else 'FAILED'}")
    if not ok:
        print("   not the published comparison; nothing above is usable")
        return

    print("\n--- criterion ---")
    if p_pred > 0.05:
        print(f"BRANCH 1: the fitted arrangement does NOT predict better than its own scrambles "
              f"(p = {p_pred:.3f}), while it descends better than all of them. The descent "
              "margin therefore separates what a one-step accuracy comparison does not.")
    else:
        rel = adv.mean() / mse_s.mean()
        print(f"BRANCH 2 or 3: the fit predicts better than its scrambles, p = {p_pred:.4f}, "
              f"by {100*rel:.1f}% of the scrambled MSE. The descent margin is then not a pure "
              "structure effect and the text must state both.")
    json.dump(dict(generated_by="scripts/scramble_prediction.py", nperm=NPERM, lam=LAM,
                   descent=dict(fitted=desc_f, scrambled=desc_s, margin=desc_f-desc_s),
                   prediction=dict(mse_fitted=mse_f, mse_scrambled_mean=float(mse_s.mean()),
                                   mse_scrambled_lo=float(np.percentile(mse_s, 2.5)),
                                   mse_scrambled_hi=float(np.percentile(mse_s, 97.5)),
                                   advantage_mean=float(adv.mean()),
                                   advantage_rel=float(adv.mean()/mse_s.mean()),
                                   p=p_pred),
                   control_passed=bool(ok), per_fold=per, seconds=time.time()-t0),
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}  ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
