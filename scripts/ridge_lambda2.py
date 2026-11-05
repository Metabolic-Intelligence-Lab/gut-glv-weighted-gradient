r"""
The ridge sweep at lambda = 2, which the response letter claims and no file carries
===================================================================================
All three response letters assert, under R1-2, R1-5 and R3-2, that the ridge sweep
"now extends to lambda = 2 (+0.035, p = 0.003; Fig F in S1)". That value appears in
no .tex file of any version, and figures/lyapunov_robustness.npz stops at lambda = 1:
its `ridge` array holds exactly three rows, 0.25, 0.5 and 1.0, because the grid in
scripts/lyapunov_robustness.py line 209 is the literal tuple (0.25, 0.5, 1.0).

So the claim has no source. This script supplies one.

Metric. The held-out descent margin, pooled over the 78 transitions: the fraction of
held-out transitions on which the fitted potential decreases, minus the mean of the
same fraction over nperm structure-scrambled potentials. The Monte-Carlo p is
(k + 1) / (nperm + 1) with k the number of scrambles reaching the observed value.
Identical definitions, functions and seeds as the published sweep.

Criterion, fixed before running.
  * Control: lambda in {0.25, 0.5, 1.0} must reproduce the cached margins
    +0.140, +0.101, +0.046 to three decimals. If they do not, this script is wrong
    and nothing else here is usable.
  * The claim stands as written if lambda = 2 gives a margin of +0.035 and p = 0.003,
    each to the precision the letter quotes.
  * The claim needs correcting, not withdrawing, if the margin keeps its sign and
    p stays below 0.05 at a different value: the letter then takes the true numbers.
  * The claim must be withdrawn if the margin loses its sign or p rises above 0.05,
    because the letter calls the whole sweep "all significant".

Run from the repository root:
    PYTHONPATH=scripts python3 scripts/ridge_lambda2.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from constrained_vs_free_glv import load_communities  # noqa: E402
from lyapunov_robustness import loco_descent  # noqa: E402

NPERM = 2000  # the published sweep's value, not the module's NPERM = 5000
CACHED = {0.25: 0.140, 0.5: 0.101, 1.0: 0.046}  # margins in figures/lyapunov_robustness.npz
CLAIMED = {"lambda": 2.0, "margin": 0.035, "p": 0.003}

comms = load_communities()
print(f"{len(comms)} communities loaded\n")

rows = []
for lam in (0.25, 0.5, 1.0, 2.0):
    obs, _, null, ntr = loco_descent(comms, lam=lam, nperm=NPERM)
    k = int(np.sum(null >= obs))
    p = (k + 1) / (len(null) + 1)
    margin = obs - null.mean()
    rows.append(dict(lam=lam, held_out=float(obs), scrambled=float(null.mean()),
                     margin=float(margin), p=float(p), k=k, nperm=len(null),
                     transitions=int(ntr)))
    print(f"  lambda={lam:<5} held-out {obs:.3f}  scrambled {null.mean():.3f}  "
          f"margin {margin:+.3f}  p {p:.4f}  ({k}/{len(null)} exceedances)")

print()
ok = True
for lam, want in CACHED.items():
    got = next(r["margin"] for r in rows if r["lam"] == lam)
    agree = abs(got - want) < 5e-4
    ok &= agree
    print(f"  control lambda={lam}: cached {want:+.3f}  recomputed {got:+.3f}  "
          f"{'OK' if agree else 'MISMATCH'}")
if not ok:
    print("\n  CONTROL FAILED: the recomputation does not reproduce the cache. Stop here.")

r2 = next(r for r in rows if r["lam"] == 2.0)
print(f"\n  lambda=2: margin {r2['margin']:+.3f} (letter claims "
      f"{CLAIMED['margin']:+.3f}), p {r2['p']:.4f} (letter claims {CLAIMED['p']})")
if abs(r2["margin"] - CLAIMED["margin"]) < 5e-4 and abs(r2["p"] - CLAIMED["p"]) < 5e-4:
    verdict = "claim stands as written"
elif r2["margin"] > 0 and r2["p"] < 0.05:
    verdict = "claim needs correcting to the true numbers, not withdrawing"
else:
    verdict = "claim must be withdrawn: the sweep is not significant at lambda=2"
print(f"  VERDICT: {verdict}")

out = ROOT / "manuscript/revision_BIB/finale/analisi" / (Path(__file__).stem + ".json")
out.write_text(json.dumps(
    dict(nperm=NPERM, cached_control=CACHED, claimed=CLAIMED, rows=rows,
         control_passed=bool(ok), verdict=verdict), indent=2), encoding="utf-8")
print(f"\n  wrote {out.relative_to(ROOT)}")
