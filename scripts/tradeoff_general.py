r"""
Is the relaxation-reciprocity trade-off general?
================================================
Scaling the diagonal of one published matrix gave an inverse relation between the slowest
relaxation rate of a community and the size of its steady-state cross-responses, which is
what sets the measurability of interaction reciprocity. This tests whether the relation is a
property of that matrix or of stable interacting communities in general, on the second
published matrix and on random ensembles.

The two quantities, for a community sitting at a positive equilibrium x* of
dx_i/dt = x_i (r + Jx)_i:
  relaxation rate  lambda = -max Re eig( diag(x*) J ),  settling time 3/lambda
  cross-response   median |chi_ij| / median |chi_ii| with chi = -J^-1
"""
import os, sys, argparse, itertools
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=".")
ap.add_argument("--draws", type=int, default=400)
ARGS = ap.parse_args()
sys.path.insert(0, os.path.join(ARGS.repo, "scripts")); os.chdir(ARGS.repo)
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")

def metrics(J, xstar):
    ev = np.linalg.eigvals(np.diag(xstar) @ J)
    lam = -float(np.max(ev.real))
    try:
        chi = -np.linalg.inv(J)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    off = ~np.eye(len(J), dtype=bool)
    rel = float(np.median(np.abs(chi[off])) / np.median(np.abs(np.diag(chi))))
    return lam, rel

def scale_diag(J, a):
    K = J.copy(); np.fill_diagonal(K, np.diag(J) * a); return K

print("=== published Clark matrix, 25 species ===")
C = np.loadtxt("data/clark2021/interaction_matrix_RLC8_posteriormean.csv", delimiter=",")
gC = np.loadtxt("data/clark2021/growth_RLC8_posteriormean.csv")
xs = np.maximum(-np.linalg.solve(C, gC), 1e-3); xs = xs / xs.sum()
print(f"  {'alpha':>7}{'relaxation rate':>17}{'settling time':>15}"
      f"{'cross-response':>16}{'precision':>11}")
for a in (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0):
    lam, rel = metrics(scale_diag(C, a), xs)
    if not np.isfinite(lam) or lam <= 0:
        print(f"  {a:>7.1f}   unstable"); continue
    print(f"  {a:>7.1f}{lam:>17.4f}{3/lam:>15.0f}{rel:>16.4f}{100*rel/2:>10.2f}%")

print("\n=== random ensembles: diagonal fixed at -1, off-diagonals N(0, sigma) ===")
print("  for each sigma, the draws that admit a positive stable equilibrium")
print(f"  {'S':>4}{'sigma':>8}{'stable frac':>13}{'median rate':>13}"
      f"{'median settling':>17}{'median cross-resp':>19}{'precision':>11}")
rng = np.random.default_rng(3)
for Sn in (12, 25):
    for sg in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        lams, rels, ok = [], [], 0
        for _ in range(ARGS.draws):
            J = rng.normal(0, sg, (Sn, Sn)); np.fill_diagonal(J, -1.0)
            r = rng.uniform(0.2, 1.0, Sn)
            try:
                x = -np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                continue
            if np.any(x <= 0): continue
            lam, rel = metrics(J, x)
            if not np.isfinite(lam) or lam <= 0: continue
            ok += 1; lams.append(lam); rels.append(rel)
        if ok < 10:
            print(f"  {Sn:>4}{sg:>8.2f}{ok/ARGS.draws:>13.3f}   too few stable draws"); continue
        print(f"  {Sn:>4}{sg:>8.2f}{ok/ARGS.draws:>13.3f}{np.median(lams):>13.4f}"
              f"{3/np.median(lams):>17.1f}{np.median(rels):>19.4f}"
              f"{100*np.median(rels)/2:>10.2f}%")

print("\n=== the relation itself: product of settling time and required precision ===")
print("  if the trade-off were an exact hyperbola this product would be constant")
rng = np.random.default_rng(11)
for Sn in (12, 25):
    prods = []
    for _ in range(2000):
        sg = 10 ** rng.uniform(-2, -0.3)
        J = rng.normal(0, sg, (Sn, Sn)); np.fill_diagonal(J, -1.0)
        r = rng.uniform(0.2, 1.0, Sn)
        try: x = -np.linalg.solve(J, r)
        except np.linalg.LinAlgError: continue
        if np.any(x <= 0): continue
        lam, rel = metrics(J, x)
        if not np.isfinite(lam) or lam <= 0: continue
        prods.append((3 / lam) * (100 * rel / 2))
    prods = np.array(prods)
    if len(prods) > 20:
        q = np.quantile(prods, [.1, .5, .9])
        print(f"  S={Sn}: settling time x precision in per cent, quantiles 10/50/90 "
              f"{q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f}   (n={len(prods)})")
